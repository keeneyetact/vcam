//! Per-frame kernels around RVM: input conversion, the model's fused
//! full-resolution guided-filter tail, single-person isolation and the
//! compositor's feather + blend.
//!
//! Each one is a single fused pass with no temporary full-size arrays, rows
//! split across a small thread pool. The pool is deliberately small (default 4,
//! `VCAM_NATIVE_THREADS` overrides): onnxruntime already runs RVM's stages on
//! their own threads, and the output thread runs these *during* the next
//! inference, so grabbing every core here would just slow the matting down.

use std::sync::OnceLock;

use rayon::prelude::*;

use crate::faces::Face;
use crate::imgproc::{reflect101, Frame, BYTES, FLOATS, U32S};

const INV255: f32 = 1.0 / 255.0;

pub fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("VCAM_NATIVE_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("vcam-kernel-{i}"))
            .build()
            .expect("failed to start the kernel thread pool")
    })
}

// --------------------------------------------------------------------------- //
// RVM input / output conversion
// --------------------------------------------------------------------------- //

/// BGR frame -> RVM's normalised RGB float32 (1,3,H,W), written into `src`.
pub fn fill_src(frame: &Frame, src: &mut [f32]) {
    let (h, w) = (frame.h, frame.w);
    assert_eq!(src.len(), 3 * h * w, "src must be (1, 3, H, W) matching the frame");
    let (r, gb) = src.split_at_mut(h * w);
    let (g, b) = gb.split_at_mut(h * w);
    pool().install(|| {
        r.par_chunks_mut(w)
            .zip(g.par_chunks_mut(w))
            .zip(b.par_chunks_mut(w))
            .zip(frame.data.par_chunks(w * 3))
            .for_each(|(((r, g), b), row)| {
                for (x, px) in row.chunks_exact(3).enumerate() {
                    r[x] = px[2] as f32 * INV255;
                    g[x] = px[1] as f32 * INV255;
                    b[x] = px[0] as f32 * INV255;
                }
            });
    });
}

/// Round half to even (like OpenCV's cvRound) for |x| < 2^22. Adding and
/// subtracting 1.5 * 2^23 makes the FPU's default rounding do the work; unlike
/// f32::round_ties_even it compiles to plain SSE2 adds, so loops vectorize.
#[inline(always)]
fn rte(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (x + MAGIC) - MAGIC
}

/// `rte(x) as u8` for x in [0, 255] (not NaN), but vectorizable: the float
/// `as` cast's saturation and NaN checks make LLVM scalarize the loop (~2.7x
/// slower per byte). x + 1.5 * 2^23 holds the rounded integer in the low
/// mantissa bits, so it's read straight out of the bit pattern.
#[inline(always)]
fn level_u8(x: f32) -> u8 {
    const MAGIC: f32 = 12_582_912.0;
    (x + MAGIC).to_bits() as u8
}

/// Saturating x*255 like cv2.convertScaleAbs (abs, round half to even).
#[inline(always)]
fn unit_to_u8(v: f32) -> u8 {
    level_u8((v * 255.0).abs().min(255.0)) // `min` also maps NaN to 255
}

/// RVM foreground float32 (1,3,H,W) RGB in [0,1] -> BGR frame.
pub fn fgr_to_bgr(fgr: &[f32], h: usize, w: usize) -> Frame {
    assert_eq!(fgr.len(), 3 * h * w, "fgr must be (1, 3, H, W)");
    let mut out = Frame::scratch(w, h);
    let (r, gb) = fgr.split_at(h * w);
    let (g, b) = gb.split_at(h * w);
    pool().install(|| {
        out.data
            .par_chunks_mut(w * 3)
            .zip(r.par_chunks(w))
            .zip(g.par_chunks(w))
            .zip(b.par_chunks(w))
            .for_each(|(((row, r), g), b)| {
                for (x, px) in row.chunks_exact_mut(3).enumerate() {
                    px[0] = unit_to_u8(b[x]);
                    px[1] = unit_to_u8(g[x]);
                    px[2] = unit_to_u8(r[x]);
                }
            });
    });
    out
}

/// One output coordinate of ONNX Resize (mode=linear,
/// coordinate_transformation_mode=pytorch_half_pixel): the two source indices
/// and the weight of the second, exactly as onnxruntime computes them.
#[inline]
fn resize_coord(o: usize, out_len: usize, in_len: usize) -> (usize, usize, f32) {
    let scale = out_len as f32 / in_len as f32;
    let x = if out_len > 1 { (o as f32 + 0.5) / scale - 0.5 } else { 0.0 };
    let x = x.clamp(0.0, (in_len - 1) as f32);
    let x0 = x as usize;
    (x0, (x0 + 1).min(in_len - 1), x - x0 as f32)
}

/// RVM's deep-guided-filter tail, fused: upsample the coarse coefficient maps
/// `a` and `b` (1,4,ch,cw) to the frame's size, apply them to the frame and
/// return (alpha HxW, foreground BGR):
///     x       = [R, G, B, mean(R,G,B)] of the frame in [0, 1]
///     out     = up(a) * x + up(b)
///     fgr     = clip(out[0:3] + x[0:3], 0, 1)       (-> BGR u8)
///     alpha   = clip(out[3], 0, 1)
/// This is the graph after tensors 777 (a) / 779 (b) of rvm_mobilenetv3: two
/// full-resolution Resize ops plus Mul/Add/Clip that onnxruntime would run as
/// ~6 separate full-size passes.
pub fn guided_upsample(a: &[f32], b: &[f32], ch: usize, cw: usize, frame: &Frame) -> (Vec<f32>, Frame) {
    let (h, w) = (frame.h, frame.w);
    assert!(ch > 0 && cw > 0 && a.len() == 4 * ch * cw && b.len() == a.len(),
            "a and b must be (1, 4, h, w)");
    let mut alpha = FLOATS.scratch(h * w);
    let mut fg = Frame::scratch(w, h);
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| resize_coord(x, w, cw)).collect();
    let plane = ch * cw;
    pool().install(|| {
        alpha
            .par_chunks_mut(w)
            .zip(fg.data.par_chunks_mut(w * 3))
            .zip(frame.data.par_chunks(w * 3))
            .enumerate()
            .for_each_init(
                || vec![[0f32; 8]; cw],
                |rows, (y, ((arow, frow_out), frow))| {
                    // Vertical pass: blend the two coarse rows of all 8
                    // coefficient planes once per output row, interleaved so a
                    // pixel's 8 coefficients are one contiguous read.
                    let (y0, y1, dy) = resize_coord(y, h, ch);
                    for c in 0..8 {
                        let src = if c < 4 { a } else { b };
                        let p = &src[(c % 4) * plane..];
                        let (r0, r1) = (&p[y0 * cw..][..cw], &p[y1 * cw..][..cw]);
                        for (d, (&u, &v)) in rows.iter_mut().zip(r0.iter().zip(r1)) {
                            d[c] = u * (1.0 - dy) + v * dy;
                        }
                    }
                    // Horizontal pass + guided-filter apply, per pixel.
                    let px = frow.chunks_exact(3).zip(frow_out.chunks_exact_mut(3)).zip(arow.iter_mut());
                    for (((px, out), al), &(x0, x1, dx)) in px.zip(&cols) {
                        let (r0, r1) = (&rows[x0], &rows[x1]);
                        let mut up = [0f32; 8];
                        for c in 0..8 {
                            up[c] = r0[c] * (1.0 - dx) + r1[c] * dx;
                        }
                        let rgb = [px[2] as f32 * INV255, px[1] as f32 * INV255, px[0] as f32 * INV255];
                        let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                        for c in 0..3 {
                            let v = (up[c] * rgb[c] + up[4 + c] + rgb[c]).clamp(0.0, 1.0);
                            out[2 - c] = unit_to_u8(v); // RGB -> BGR
                        }
                        *al = (up[3] * mean + up[7]).clamp(0.0, 1.0);
                    }
                },
            );
    });
    (alpha, fg)
}

// --------------------------------------------------------------------------- //
// Single-person isolation
// --------------------------------------------------------------------------- //

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let gp = parent[parent[x as usize] as usize];
        parent[x as usize] = gp; // path halving
        x = gp;
    }
    x
}

/// Half-widths of OpenCV's 11x11 MORPH_ELLIPSE, row by row (dy = -5..=5).
const ELLIPSE11: [usize; 11] = [0, 3, 4, 5, 5, 5, 5, 5, 4, 3, 0];

/// A component smaller than 1/MIN_SHARE of the largest is speckle or a stray
/// fragment, never the subject, however close to the centre it sits.
const MIN_SHARE: u32 = 10;

/// The label (index into the per-component tables, zero where not a root) of
/// the component whose centroid is nearest the frame centre, among those at
/// least 1/MIN_SHARE the size of the largest; the first in label order on ties.
fn pick_central(area: &[u32], sum2x: &[u64], sum2y: &[u64], h: usize, w: usize) -> usize {
    let largest = area.iter().copied().max().unwrap_or(0);
    let mut best = (0usize, f64::INFINITY);
    for (l, &n) in area.iter().enumerate() {
        if n == 0 || n * MIN_SHARE < largest {
            continue;
        }
        // Centroid and centre, both doubled: sum2 / n against the frame size.
        let dx = sum2x[l] as f64 / n as f64 - w as f64;
        let dy = sum2y[l] as f64 / n as f64 - h as f64;
        let d = dx * dx + dy * dy;
        if d < best.1 {
            best = (l, d);
        }
    }
    best.0
}

/// How much nearer the centre another face must be to take over from the one
/// kept last frame, as a fraction of the frame width (stops flip-flopping).
const STICKINESS: f32 = 0.08;
/// Vertical distances count this much (squared) when splitting a blob between
/// two faces: bodies hang below heads, so the cut leans towards vertical.
const CUT_VERTICAL: f32 = 0.1;
/// Width of the soft edge along a cut, as a fraction of the kept face width.
const CUT_SOFTNESS: f32 = 0.25;
/// Labelling-resolution px over which the matte fades out past the kept
/// blob's dilated edge: removing the rest along a hard line reads as a
/// scissor cut.
const FADE: usize = 6;
/// How far the keep weight may drop per frame, so whatever is removed
/// dissolves over a few frames instead of vanishing in one. It may rise at
/// once, so the kept person is never held back.
const FADE_OUT_STEP: f32 = 0.25;

/// A straight cut through a blob shared by two people: alpha is kept where
/// `(x - mx) * nx + (y - my) * ny` is negative, fading out over `soft` px.
struct Cut {
    mx: f32,
    my: f32,
    nx: f32,
    ny: f32,
    soft: f32,
}

impl Cut {
    /// Between the kept face `t` and another face `f`, nearer `t` the
    /// smaller its face is.
    fn between(t: &Face, f: &Face) -> Cut {
        let k = t.w / (t.w + f.w);
        let (mx, my) = (t.x + (f.x - t.x) * k, t.y + (f.y - t.y) * k);
        let (nx, ny) = (f.x - t.x, (f.y - t.y) * CUT_VERTICAL);
        let len = nx.hypot(ny).max(1e-6);
        Cut { mx, my, nx: nx / len, ny: ny / len, soft: (t.w * CUT_SOFTNESS).max(1.0) }
    }

    #[inline]
    fn weight(&self, x: f32, y: f32) -> f32 {
        (0.5 - ((x - self.mx) * self.nx + (y - self.my) * self.ny) / self.soft).clamp(0.0, 1.0)
    }
}

/// Keep only one person in an HxW matte, in place; fade out the rest while
/// preserving the kept subject's soft edges. Components are labelled at half
/// resolution above 640 px (blobs are large). `history` carries the keep
/// weights from frame to frame (empty to start afresh).
///
/// With `faces` (frame pixels), the person is the face on the matte nearest
/// the frame centre -- favouring `prev`, the one kept last time -- and if
/// other faces share its blob (people touching), the blob is cut between
/// them. Without, it's the blob nearest the centre. Returns the face kept, or
/// `prev` when no face was usable.
pub fn keep_central(
    a: &mut [f32],
    h: usize,
    w: usize,
    faces: &[Face],
    prev: Option<[f32; 2]>,
    history: &mut Vec<f32>,
) -> Option<[f32; 2]> {
    assert_eq!(a.len(), h * w, "alpha must be HxW");
    // An exact 2x INTER_LINEAR downscale is the mean of each 2x2 block.
    let half = h.max(w) > 640;
    let (sh, sw) = if half { (h / 2, w / 2) } else { (h, w) };
    if sh == 0 || sw == 0 {
        return prev;
    }
    let mut bin = BYTES.scratch(sh * sw);
    {
        let a: &[f32] = a;
        pool().install(|| {
            bin.par_chunks_mut(sw).enumerate().for_each(|(y, row)| {
                if half {
                    let (r0, r1) = (&a[2 * y * w..], &a[(2 * y + 1) * w..]);
                    for (x, v) in row.iter_mut().enumerate() {
                        let s = r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1];
                        *v = (s * 0.25 > 0.5) as u8;
                    }
                } else {
                    for (x, v) in row.iter_mut().enumerate() {
                        *v = (a[y * w + x] > 0.5) as u8;
                    }
                }
            });
        });
    }

    // Horizontal runs of foreground, in raster order; row y's runs are
    // runs[row_at[y]..row_at[y + 1]]. A person blob is a run or two per row, so
    // labelling runs instead of pixels makes the serial part tiny.
    let mut runs: Vec<(u32, u32)> = Vec::new(); // (start, end exclusive)
    let mut row_at = vec![0usize; sh + 1];
    for y in 0..sh {
        let row = &bin[y * sw..][..sw];
        let mut x = 0;
        while x < sw {
            match row[x..].iter().position(|&v| v != 0) {
                None => break,
                Some(d) => x += d,
            }
            let s = x;
            x += row[x..].iter().position(|&v| v == 0).unwrap_or(sw - x);
            runs.push((s as u32, x as u32));
        }
        row_at[y + 1] = runs.len();
    }
    BYTES.give(bin);
    if runs.is_empty() {
        history.clear();
        return prev; // nothing detected; leave the matte as-is
    }

    // 8-connected components over the runs, union-find: runs in adjacent rows
    // touch when their spans, widened by one pixel, overlap. The smaller run
    // index always wins a merge, so each root is the component's first pixel in
    // raster order -- the same order cv2.connectedComponents numbers them in,
    // which keeps the tie-break (first in label order) well defined.
    let mut parent: Vec<u32> = (0..runs.len() as u32).collect();
    for y in 1..sh {
        let (prev, cur) = (row_at[y - 1]..row_at[y], row_at[y]..row_at[y + 1]);
        let mut j = prev.start;
        for i in cur {
            let (s, e) = runs[i];
            // Skip runs above that end left of this one; they can't touch the
            // later runs in this row either.
            while j < prev.end && runs[j].1 < s {
                j += 1;
            }
            let mut k = j;
            while k < prev.end && runs[k].0 <= e {
                let (a, b) = (find(&mut parent, i as u32), find(&mut parent, k as u32));
                if a != b {
                    parent[a.max(b) as usize] = a.min(b);
                }
                k += 1;
            }
        }
    }
    // Per component: area and the sums of 2x+1 / 2y+1 over its pixels (twice
    // the pixel-centre coordinates, so a run [s, e) adds exactly e^2 - s^2).
    let mut area = vec![0u32; runs.len()];
    let mut sum2x = vec![0u64; runs.len()];
    let mut sum2y = vec![0u64; runs.len()];
    for y in 0..sh {
        for i in row_at[y]..row_at[y + 1] {
            let (s, e) = runs[i];
            let r = find(&mut parent, i as u32);
            parent[i] = r;
            let r = r as usize;
            area[r] += e - s;
            sum2x[r] += e as u64 * e as u64 - s as u64 * s as u64;
            sum2y[r] += (e - s) as u64 * (2 * y as u64 + 1);
        }
    }

    // Faces on the matte, with their blob: the run under the face centre, or
    // failing that under the chin. A face off the matte (a poster, a TV) or on
    // a speck doesn't count.
    let largest = area.iter().copied().max().unwrap_or(0);
    let blob_at = |x: f32, y: f32| -> Option<usize> {
        let (x, y) = (x * sw as f32 / w as f32, y * sh as f32 / h as f32);
        if !(0.0..sw as f32).contains(&x) || !(0.0..sh as f32).contains(&y) {
            return None;
        }
        let (x, y) = (x as u32, y as usize);
        let i = (row_at[y]..row_at[y + 1]).find(|&i| runs[i].0 <= x && x < runs[i].1)?;
        let l = parent[i] as usize;
        (area[l] * MIN_SHARE >= largest).then_some(l)
    };
    let on: Vec<(&Face, usize)> = faces
        .iter()
        .filter_map(|f| Some((f, blob_at(f.x, f.y).or_else(|| blob_at(f.x, f.y + 0.6 * f.h))?)))
        .collect();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let target = on.iter().copied().min_by(|(f, _), (g, _)| {
        let d = |f: &Face| {
            let sticky = prev.is_some_and(|[px, py]| (f.x - px).hypot(f.y - py) < f.w);
            (f.x - cx).hypot(f.y - cy) - if sticky { STICKINESS * w as f32 } else { 0.0 }
        };
        d(f).total_cmp(&d(g))
    });
    let (central, cuts, kept_face) = match target {
        Some((t, l)) => {
            let cuts: Vec<Cut> =
                on.iter().filter(|&&(f, m)| m == l && !std::ptr::eq(f, t)).map(|(f, _)| Cut::between(t, f)).collect();
            (l, cuts, Some([t.x, t.y]))
        }
        None => (pick_central(&area, &sum2x, &sum2y, sh, sw), Vec::new(), prev),
    };
    let kept = |i: usize| parent[i] as usize == central;

    // The blob's bounding box: nothing outside it (+5 px of dilation, + the
    // fade) is kept this frame.
    let (mut x0, mut x1, mut y0, mut y1) = (sw, 0, sh, 0);
    for y in 0..sh {
        for i in (row_at[y]..row_at[y + 1]).filter(|&i| kept(i)) {
            let (s, e) = runs[i];
            (x0, x1, y0, y1) = (x0.min(s as usize), x1.max(e as usize - 1), y0.min(y), y1.max(y));
        }
    }
    let m = 5 + FADE;
    let (x0, x1) = (x0.saturating_sub(m), (x1 + 1 + m).min(sw));
    let (y0, y1) = (y0.saturating_sub(m), (y1 + 1 + m).min(sh));

    // Per-row prefix counts of the kept blob, so the 11x11 elliptical dilation
    // is 11 O(1) span tests per pixel instead of ~100 neighbour reads.
    let pw = sw + 1;
    let mut pre = U32S.scratch(sh * pw);
    pool().install(|| {
        pre.par_chunks_mut(pw).enumerate().for_each(|(y, p)| {
            p[0] = 0;
            let (mut acc, mut x) = (0u32, 0usize);
            for i in (row_at[y]..row_at[y + 1]).filter(|&i| kept(i)) {
                let (s, e) = (runs[i].0 as usize, runs[i].1 as usize);
                p[x + 1..=s].fill(acc);
                for v in &mut p[s + 1..=e] {
                    acc += 1;
                    *v = acc;
                }
                x = e;
            }
            p[x + 1..].fill(acc);
        });
    });

    // The dilated blob at low resolution, one row per task.
    let mut keep = BYTES.zeroed(sh * sw);
    pool().install(|| {
        keep.par_chunks_mut(sw).enumerate().for_each(|(sy, krow)| {
            if sy < y0 || sy >= y1 {
                return;
            }
            for (x, k) in krow.iter_mut().enumerate().take(x1).skip(x0) {
                *k = ELLIPSE11.iter().enumerate().any(|(j, &hw)| {
                    let y = sy as isize + j as isize - 5;
                    if y < 0 || y >= sh as isize {
                        return false;
                    }
                    let p = &pre[y as usize * pw..];
                    p[(x + hw + 1).min(sw)] > p[x.saturating_sub(hw)]
                }) as u8;
            }
        });
    });
    U32S.give(pre);

    // Keep weights: 1 on the dilated blob, easing to 0 over FADE px past it
    // (a box blur of the blob, doubled so it's 1 at the blob's edge, then
    // smoothstepped), times the soft cuts between touching people.
    let side = 2 * FADE + 1;
    let mut hsum = U32S.zeroed(sh * sw);
    pool().install(|| {
        hsum.par_chunks_mut(sw).enumerate().for_each_init(
            || vec![0u32; sw + 1],
            |pre, (y, row)| {
                if y < y0 || y >= y1 {
                    return;
                }
                for (x, &k) in keep[y * sw..][..sw].iter().enumerate() {
                    pre[x + 1] = pre[x] + k as u32;
                }
                for (x, v) in row.iter_mut().enumerate().take(x1).skip(x0) {
                    *v = pre[(x + FADE + 1).min(sw)] - pre[x.saturating_sub(FADE)];
                }
            },
        );
    });
    let (fx, fy) = (w as f32 / sw as f32, h as f32 / sh as f32);
    let mut weight = FLOATS.zeroed(sh * sw);
    pool().install(|| {
        weight.par_chunks_mut(sw).enumerate().for_each(|(y, row)| {
            if y < y0 || y >= y1 {
                return;
            }
            let krow = &keep[y * sw..][..sw];
            for (x, v) in row.iter_mut().enumerate().take(x1).skip(x0) {
                *v = if krow[x] != 0 {
                    1.0
                } else {
                    let n: u32 = (y.saturating_sub(FADE)..(y + FADE + 1).min(sh)).map(|yy| hsum[yy * sw + x]).sum();
                    let t = (2.0 * n as f32 / (side * side) as f32).min(1.0);
                    t * t * (3.0 - 2.0 * t)
                };
                if *v > 0.0 {
                    for c in &cuts {
                        *v *= c.weight((x as f32 + 0.5) * fx, (y as f32 + 0.5) * fy);
                    }
                }
            }
        });
    });
    U32S.give(hsum);
    BYTES.give(keep);
    if history.len() == weight.len() {
        for (v, &p) in weight.iter_mut().zip(history.iter()) {
            *v = v.max(p - FADE_OUT_STEP);
        }
    }

    // Scale the matte by the weights, upsampled bilinearly to full resolution.
    let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| resize_coord(x, w, sw)).collect();
    {
        let weight: &[f32] = &weight;
        pool().install(|| {
            a.par_chunks_mut(w).enumerate().for_each(|(yy, row)| {
                let (r0, r1, dy) = resize_coord(yy, h, sh);
                let (r0, r1) = (&weight[r0 * sw..][..sw], &weight[r1 * sw..][..sw]);
                for (v, &(x0, x1, dx)) in row.iter_mut().zip(&cols) {
                    if *v > 0.0 {
                        let top = r0[x0] * (1.0 - dx) + r0[x1] * dx;
                        let bottom = r1[x0] * (1.0 - dx) + r1[x1] * dx;
                        *v *= top * (1.0 - dy) + bottom * dy;
                    }
                }
            });
        });
    }
    FLOATS.give(std::mem::replace(history, weight));
    kept_face
}

// --------------------------------------------------------------------------- //
// Compositing
// --------------------------------------------------------------------------- //

/// cv2.getGaussianKernel(k, 0): OpenCV's fixed tables for k <= 7, else the
/// sigma-from-ksize formula.
fn gaussian_kernel(k: usize) -> Vec<f32> {
    match k {
        1 => vec![1.0],
        3 => vec![0.25, 0.5, 0.25],
        5 => vec![0.0625, 0.25, 0.375, 0.25, 0.0625],
        7 => vec![0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125],
        _ => {
            let sigma = 0.3 * ((k as f64 - 1.0) * 0.5 - 1.0) + 0.8;
            let c = (k as f64 - 1.0) * 0.5;
            let w: Vec<f64> = (0..k)
                .map(|i| (-((i as f64 - c).powi(2)) / (2.0 * sigma * sigma)).exp())
                .collect();
            let sum: f64 = w.iter().sum();
            w.iter().map(|v| (v / sum) as f32).collect()
        }
    }
}

/// out = bg + (fg - bg) * feather(alpha) / 255, as a new frame: alpha -> u8
/// levels (truncating), Gaussian-feathered with a `ksize` kernel, then the
/// integer blend -- fused into two passes.
pub fn composite(fg: &Frame, bg: &Frame, alpha: &[f32], ksize: usize) -> Frame {
    composite_graded(fg, bg, alpha, ksize, None)
}

/// Matching applied to a picture background as it's blended (see `look`).
pub struct Grade<'a> {
    /// Per-channel tone curve for the background.
    pub lut: &'a [[u8; 256]; 3],
    /// Grain sigma, in levels, added to the background.
    pub grain: f32,
    /// Changes every frame, so the grain moves like sensor noise.
    pub seed: u32,
    pub wrap: Option<Wrap<'a>>,
}

/// Light wrap: background light spilling onto the person's edge.
pub struct Wrap<'a> {
    /// Background coverage around each point (blurred 1 - alpha), at the
    /// size of `light`.
    pub mask: &'a [f32],
    /// Blurred background light, small.
    pub light: &'a Frame,
    pub strength: f32,
}

type Taps = Vec<(usize, usize, f32)>;

/// Bilinear taps mapping a full-size axis onto a small grid's.
fn grid_taps(out_len: usize, in_len: usize) -> Taps {
    (0..out_len)
        .map(|o| {
            let s = ((o as f32 + 0.5) * in_len as f32 / out_len as f32 - 0.5).clamp(0.0, (in_len - 1) as f32);
            let i0 = s as usize;
            (i0, (i0 + 1).min(in_len - 1), s - i0 as f32)
        })
        .collect()
}

/// Two zero-mean, unit-variance (triangular) grain values from a pixel hash.
#[inline]
fn grain_pair(seed: u32, i: u32) -> (f32, f32) {
    let mut x = i.wrapping_mul(0x9E37_79B9) ^ seed.wrapping_mul(0x85EB_CA6B);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    // A difference of two uniform bytes has variance ~ 2 * 255^2 / 12.
    const K: f32 = 2.449_489_8 / 255.0; // sqrt(6) / 255
    let b = x.to_le_bytes();
    ((b[0] as f32 - b[1] as f32) * K, (b[2] as f32 - b[3] as f32) * K)
}

/// `composite`, with the background graded (tone curve + grain) and the
/// person's edge light-wrapped when `grade` is given.
pub fn composite_graded(fg: &Frame, bg: &Frame, alpha: &[f32], ksize: usize, grade: Option<&Grade>) -> Frame {
    let (h, w) = (fg.h, fg.w);
    assert!(bg.size() == fg.size(), "fg and bg must be the same size");
    assert_eq!(alpha.len(), h * w, "alpha must be HxW matching the image");
    assert!(ksize % 2 == 1, "ksize must be odd");
    let kern = gaussian_kernel(ksize);
    let r = (ksize / 2) as isize;
    let mut out = Frame::scratch(w, h);
    let mut tmp = FLOATS.scratch(h * w);
    let wrap_taps = grade.and_then(|g| g.wrap.as_ref()).map(|wr| {
        assert_eq!(wr.mask.len(), wr.light.w * wr.light.h, "wrap mask must match its light");
        (grid_taps(w, wr.light.w), grid_taps(h, wr.light.h))
    });

    pool().install(|| {
        // Pass 1: matte -> u8 levels, horizontal blur into a float buffer.
        // (The kernels are dyadic fractions, so these sums are exact.)
        tmp.par_chunks_mut(w).zip(alpha.par_chunks(w)).for_each_init(
            || vec![0f32; w],
            |u, (t, arow)| {
                for (d, &a) in u.iter_mut().zip(arow) {
                    *d = ((a * 255.0) as u8) as f32; // truncates, like numpy astype(uint8)
                }
                let ru = r as usize;
                for (x, t) in t.iter_mut().enumerate() {
                    *t = if x >= ru && x + ru < w {
                        // Interior: straight dot product, no border handling.
                        u[x - ru..=x + ru].iter().zip(&kern).map(|(v, k)| k * v).sum()
                    } else {
                        let x0 = x as isize - r;
                        kern.iter()
                            .enumerate()
                            .map(|(j, &k)| k * u[reflect101(x0 + j as isize, w)])
                            .sum()
                    };
                }
            },
        );

        // Pass 2: vertical blur -> feathered alpha, then blend the row.
        out.data
            .par_chunks_mut(w * 3)
            .zip(fg.data.par_chunks(w * 3))
            .zip(bg.data.par_chunks(w * 3))
            .enumerate()
            .for_each_init(
                || (vec![0f32; w], vec![0f32; w * 3]),
                |(acc, wide), (y, ((orow, frow), brow))| {
                    acc.fill(0.0);
                    for (j, &k) in kern.iter().enumerate() {
                        let src = reflect101(y as isize - r + j as isize, h);
                        for (v, &t) in acc.iter_mut().zip(&tmp[src * w..(src + 1) * w]) {
                            *v += k * t;
                        }
                    }
                    // The feathered alpha, repeated per channel so the blend is
                    // one flat (vectorizable) loop over the row's bytes.
                    for (e, &v) in wide.chunks_exact_mut(3).zip(acc.iter()) {
                        // Round half up, like OpenCV's fixed-point 8-bit blur.
                        // v >= 0, so truncation is floor (f32::floor is a libm
                        // call on baseline x86-64).
                        e.fill((((v + 0.5) as i32) as f32).min(255.0) * INV255);
                    }
                    if let Some(g) = grade {
                        graded_row(g, y, wrap_taps.as_ref(), orow, frow, brow, wide);
                        return;
                    }
                    // Everything but the scaled difference is a small integer,
                    // so this float math matches the integer blend exactly.
                    for (((o, &f), &b), &a) in orow.iter_mut().zip(frow).zip(brow).zip(wide.iter()) {
                        let (fv, bv) = (f as f32, b as f32);
                        *o = level_u8((bv + rte((fv - bv) * a)).clamp(0.0, 255.0));
                    }
                },
            );
    });
    FLOATS.give(tmp);
    out
}

/// One output row of `composite_graded` under a grade; `a` is the feathered
/// alpha, repeated per channel.
fn graded_row(g: &Grade, y: usize, taps: Option<&(Taps, Taps)>, orow: &mut [u8], frow: &[u8], brow: &[u8], a: &[f32]) {
    let w = frow.len() / 3;
    let wrap = g.wrap.as_ref().zip(taps).map(|(wr, (xt, yt))| {
        let (y0, y1, fy) = yt[y];
        (wr, xt, y0 * wr.light.w, y1 * wr.light.w, fy)
    });
    for x in 0..w {
        let i = x * 3;
        let av = a[i];
        let mut f = [frow[i] as f32, frow[i + 1] as f32, frow[i + 2] as f32];
        if let Some((wr, xt, r0, r1, fy)) = wrap.filter(|_| av > 0.0) {
            let (x0, x1, fx) = xt[x];
            let lerp2 = |v00: f32, v01: f32, v10: f32, v11: f32| {
                let top = v00 + (v01 - v00) * fx;
                top + (v10 + (v11 - v10) * fx - top) * fy
            };
            let k = wr.strength * lerp2(wr.mask[r0 + x0], wr.mask[r0 + x1], wr.mask[r1 + x0], wr.mask[r1 + x1]);
            if k > 0.004 {
                let l = &wr.light.data;
                for (c, fc) in f.iter_mut().enumerate() {
                    let lv = lerp2(l[(r0 + x0) * 3 + c] as f32, l[(r0 + x1) * 3 + c] as f32,
                                   l[(r1 + x0) * 3 + c] as f32, l[(r1 + x1) * 3 + c] as f32);
                    *fc += (lv - *fc) * k;
                }
            }
        }
        if av >= 1.0 {
            for c in 0..3 {
                orow[i + c] = round_u8(f[c]);
            }
            continue;
        }
        // Mostly luma grain with a little chroma, like a webcam sensor's.
        let (n0, n1) = if g.grain > 0.0 { grain_pair(g.seed, (y * w + x) as u32) } else { (0.0, 0.0) };
        let (luma, chroma) = (g.grain * n0, 0.35 * g.grain * n1);
        let noise = [luma + chroma, luma, luma - chroma];
        for c in 0..3 {
            let b = g.lut[c][brow[i + c] as usize] as f32 + noise[c];
            orow[i + c] = round_u8(b + (f[c] - b) * av);
        }
    }
}

/// `v.round().clamp(0.0, 255.0) as u8` without the libm `round` call (and so
/// vectorizable): once clamped, v >= 0 and truncating v + 0.5 rounds half away
/// from zero. (Only 0.49999997, the f32 just below 0.5, comes out a level high.)
#[inline(always)]
fn round_u8(v: f32) -> u8 {
    (v.clamp(0.0, 255.0) + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_central_drops_speck_near_centre() {
        let (h, w) = (40, 60);
        let mut a = vec![0f32; h * w];
        for y in 5..35 {
            for x in 5..30 {
                a[y * w + x] = 1.0; // big blob, left of centre
            }
        }
        for y in 10..14 {
            for x in 50..54 {
                a[y * w + x] = 1.0; // speck, > 5 px away
            }
        }
        keep_central(&mut a, h, w, &[], None, &mut Vec::new());
        assert_eq!(a[20 * w + 20], 1.0);
        assert_eq!(a[12 * w + 52], 0.0);
    }

    #[test]
    fn keep_central_prefers_centre_over_size() {
        let (h, w) = (60, 120);
        let mut a = vec![0f32; h * w];
        for y in 0..60 {
            for x in 0..30 {
                a[y * w + x] = 1.0; // large person at the left edge
            }
        }
        for y in 20..60 {
            for x in 50..70 {
                a[y * w + x] = 1.0; // smaller person in the middle
            }
        }
        keep_central(&mut a, h, w, &[], None, &mut Vec::new());
        assert_eq!(a[40 * w + 60], 1.0);
        assert_eq!(a[30 * w + 10], 0.0);
    }

    fn face(x: f32, y: f32, size: f32) -> Face {
        Face { x, y, w: size, h: size, score: 0.9 }
    }

    /// Two people shoulder to shoulder: one blob, two heads.
    fn two_touching(h: usize, w: usize) -> Vec<f32> {
        let mut a = vec![0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let head = |cx: usize| (x as isize - cx as isize).pow(2) + (y as isize - 25).pow(2) <= 15 * 15;
                if head(70) || head(140) || (y >= 40 && (30..180).contains(&x)) {
                    a[y * w + x] = 1.0;
                }
            }
        }
        a
    }

    #[test]
    fn keep_central_cuts_between_touching_people() {
        let (h, w) = (120, 200);
        let mut a = two_touching(h, w);
        let faces = [face(70.0, 25.0, 30.0), face(140.0, 25.0, 30.0)];
        // Face 70 is nearer the centre (100).
        let kept = keep_central(&mut a, h, w, &faces, None, &mut Vec::new());
        assert_eq!(kept, Some([70.0, 25.0]));
        assert_eq!(a[25 * w + 70], 1.0, "kept head");
        assert_eq!(a[100 * w + 40], 1.0, "kept body");
        assert_eq!(a[25 * w + 140], 0.0, "other head");
        assert_eq!(a[100 * w + 160], 0.0, "other body");
        // The cut sits halfway between equal faces, with a soft edge.
        assert!(a[100 * w + 105] > 0.0 && a[100 * w + 105] < 1.0);
    }

    #[test]
    fn keep_central_sticks_with_previous_face() {
        let (h, w) = (120, 200);
        let faces = [face(70.0, 25.0, 30.0), face(140.0, 25.0, 30.0)];
        // The right face is only a little nearer the centre than the left...
        let near = [face(62.0, 25.0, 30.0), face(132.0, 25.0, 30.0)];
        let mut a = two_touching(h, w);
        assert_eq!(keep_central(&mut a, h, w, &near, None, &mut Vec::new()), Some([132.0, 25.0]));
        // ...so if the left one was kept last time, it stays.
        let mut a = two_touching(h, w);
        assert_eq!(keep_central(&mut a, h, w, &near, Some([60.0, 25.0]), &mut Vec::new()), Some([62.0, 25.0]));
        assert_eq!(a[25 * w + 140], 0.0);
        // A face off the matte (a poster) is ignored.
        let mut a = two_touching(h, w);
        let kept = keep_central(&mut a, h, w, &[face(100.0, 5.0, 10.0), faces[0]], None, &mut Vec::new());
        assert_eq!(kept, Some([70.0, 25.0]));
    }

    /// Pixel-by-pixel reference: 8-connected flood fill in raster order, keep
    /// the component nearest the centre, dilate it with the 11x11 ellipse.
    fn keep_central_reference(a: &mut [f32], h: usize, w: usize) {
        let mut label = vec![0usize; h * w];
        let mut areas = vec![0usize];
        let (mut sx, mut sy) = (vec![0u64], vec![0u64]);
        for start in 0..h * w {
            if a[start] <= 0.5 || label[start] != 0 {
                continue;
            }
            let l = areas.len();
            let (mut stack, mut n, mut px, mut py) = (vec![start], 0, 0u64, 0u64);
            label[start] = l;
            while let Some(i) = stack.pop() {
                n += 1;
                let (y, x) = ((i / w) as isize, (i % w) as isize);
                px += 2 * x as u64 + 1;
                py += 2 * y as u64 + 1;
                for (dy, dx) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dy, dx))) {
                    let (ny, nx) = (y + dy, x + dx);
                    if ny < 0 || nx < 0 || ny >= h as isize || nx >= w as isize {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if a[j] > 0.5 && label[j] == 0 {
                        label[j] = l;
                        stack.push(j);
                    }
                }
            }
            areas.push(n);
            sx.push(px);
            sy.push(py);
        }
        if areas.len() == 1 {
            return;
        }
        let areas: Vec<u32> = areas.iter().map(|&n| n as u32).collect();
        let best = pick_central(&areas, &sx, &sy, h, w);
        for y in 0..h {
            for x in 0..w {
                let near = ELLIPSE11.iter().enumerate().any(|(j, &hw)| {
                    let yy = y as isize + j as isize - 5;
                    (0..h as isize).contains(&yy)
                        && (x.saturating_sub(hw)..=(x + hw).min(w - 1)).any(|xx| label[yy as usize * w + xx] == best)
                });
                if !near {
                    a[y * w + x] = 0.0;
                }
            }
        }
    }

    #[test]
    fn keep_central_matches_pixel_labelling() {
        let mut seed = 1u32;
        for (h, w, density) in [(37, 53, 0.45), (64, 64, 0.6), (40, 90, 0.3), (1, 30, 0.5), (25, 1, 0.5)] {
            let mut a = vec![0f32; h * w];
            for v in &mut a {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *v = if ((seed >> 8) as f32 / (1u32 << 24) as f32) < density { 0.9 } else { 0.1 };
            }
            let orig = a.clone();
            let mut expect = a.clone();
            keep_central_reference(&mut expect, h, w);
            keep_central(&mut a, h, w, &[], None, &mut Vec::new());
            // The reference's kept region is untouched; past it the matte
            // only fades, and is gone more than FADE px out.
            for i in 0..h * w {
                if expect[i] != 0.0 {
                    assert_eq!(a[i], orig[i], "{h}x{w} @ {density}: kept pixel {i} changed");
                    continue;
                }
                assert!(a[i] <= orig[i], "{h}x{w} @ {density}: pixel {i} grew");
                let (y, x) = (i / w, i % w);
                let near = (y.saturating_sub(FADE)..(y + FADE + 1).min(h))
                    .any(|yy| (x.saturating_sub(FADE)..(x + FADE + 1).min(w)).any(|xx| expect[yy * w + xx] != 0.0));
                if !near {
                    assert_eq!(a[i], 0.0, "{h}x{w} @ {density}: pixel {i} far from the kept blob survived");
                }
            }
        }
    }

    #[test]
    fn keep_central_fades_out_past_the_blob() {
        let (h, w) = (40, 80);
        let mut a = vec![0f32; h * w];
        for y in 0..h {
            for x in 30..50 {
                a[y * w + x] = 1.0; // the person, mid-frame
            }
            for x in 58..80 {
                a[y * w + x] = 0.4; // someone else's soft edge, just past the dilation
            }
        }
        keep_central(&mut a, h, w, &[], None, &mut Vec::new());
        let row = &a[20 * w..][..w];
        assert_eq!(row[40], 1.0);
        assert_eq!(row[54], 0.0, "background stays background");
        // Fades smoothly to nothing rather than stopping at a line.
        assert!(row[58] > 0.0 && row[58] < 0.4, "{}", row[58]);
        assert!(row[58..].windows(2).all(|p| p[1] <= p[0]));
        assert_eq!(row[70], 0.0);
    }

    #[test]
    fn keep_central_fades_a_removed_person_over_frames() {
        let (h, w) = (40, 120);
        let frame = |bridge: bool| {
            let mut a = vec![0f32; h * w];
            for y in 10..40 {
                for x in (45..75).chain(95..115) {
                    a[y * w + x] = 1.0;
                }
                if bridge && y >= 30 {
                    a[y * w + 75..y * w + 95].fill(1.0); // touching: one blob
                }
            }
            a
        };
        let mut history = Vec::new();
        let mut a = frame(true);
        keep_central(&mut a, h, w, &[], None, &mut history);
        assert_eq!(a[20 * w + 105], 1.0, "touching: kept with the blob");
        // They step apart: the other person dissolves over a few frames...
        let mut seen = Vec::new();
        for _ in 0..5 {
            let mut a = frame(false);
            keep_central(&mut a, h, w, &[], None, &mut history);
            assert_eq!(a[20 * w + 60], 1.0, "the kept person never fades");
            seen.push(a[20 * w + 105]);
        }
        assert_eq!(seen, [0.75, 0.5, 0.25, 0.0, 0.0]);
        // ...but whatever joins the kept blob shows at once.
        let mut a = frame(true);
        keep_central(&mut a, h, w, &[], None, &mut history);
        assert_eq!(a[20 * w + 105], 1.0);
    }

    #[test]
    fn level_u8_matches_rounding_cast() {
        // Every multiple of 1/64 in [0, 255] (all the exact halves included),
        // plus the f32s next to each half.
        for i in 0..=255 * 64 {
            let x = i as f32 / 64.0;
            for v in [x, f32::from_bits(x.to_bits().saturating_sub(1)), f32::from_bits(x.to_bits() + 1).min(255.0)] {
                assert_eq!(level_u8(v), rte(v) as u8, "{v}");
            }
        }
        assert_eq!((unit_to_u8(f32::NAN), unit_to_u8(-0.5), unit_to_u8(2.0)), (255, 128, 255));
    }

    #[test]
    fn composite_extremes() {
        let fg = Frame::filled(16, 8, [200, 100, 50]);
        let bg = Frame::filled(16, 8, [0, 0, 0]);
        assert_eq!(composite(&fg, &bg, &vec![1.0; 128], 3), fg);
        assert_eq!(composite(&fg, &bg, &vec![0.0; 128], 3), bg);
    }

    #[test]
    fn identity_grade_matches_plain_composite() {
        let (w, h) = (40, 24);
        let mut fg = Frame::new(w, h);
        let mut bg = Frame::new(w, h);
        for (i, v) in fg.data.iter_mut().enumerate() {
            *v = (i * 7 % 251) as u8;
        }
        for (i, v) in bg.data.iter_mut().enumerate() {
            *v = (i * 13 % 241) as u8;
        }
        let alpha: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / (w - 1) as f32).collect();
        let mut lut = [[0u8; 256]; 3];
        for c in lut.iter_mut() {
            for (v, o) in c.iter_mut().enumerate() {
                *o = v as u8;
            }
        }
        let g = Grade { lut: &lut, grain: 0.0, seed: 0, wrap: None };
        let plain = composite(&fg, &bg, &alpha, 3);
        let graded = composite_graded(&fg, &bg, &alpha, 3, Some(&g));
        // Only the rounding of exact halves may differ.
        let worst = plain.data.iter().zip(&graded.data).map(|(&p, &q)| (p as i32 - q as i32).abs()).max().unwrap();
        assert!(worst <= 1, "off by {worst}");
    }

    #[test]
    fn grain_is_zero_mean_unit_variance() {
        let n = 200_000u32;
        let (mut s, mut ss) = (0f64, 0f64);
        for i in 0..n {
            let (a, b) = grain_pair(7, i);
            s += (a + b) as f64;
            ss += (a * a + b * b) as f64;
        }
        let mean = s / (2 * n) as f64;
        let var = ss / (2 * n) as f64 - mean * mean;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.02, "mean {mean} var {var}");
    }
}

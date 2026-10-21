//! Rust per-frame kernels for vcam.
//!
//! Each function mirrors a numpy/OpenCV routine in vcam.py (which stays as the
//! fallback when this module isn't built) and produces the same result, but as
//! one fused pass: no temporary arrays, rows split across a small thread pool,
//! and the GIL released while it runs so the camera/output threads keep going.
//!
//! The pool is deliberately small (default 4, `VCAM_NATIVE_THREADS` overrides):
//! onnxruntime already runs RVM on 8 threads, and the compositor works on a
//! second Python thread *during* the next inference, so grabbing every core
//! here would just slow the matting down.

use std::sync::OnceLock;

use numpy::{
    PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

const INV255: f32 = 1.0 / 255.0;

fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("VCAM_NATIVE_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(4)
            .max(1);
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("vcam-native-{i}"))
            .build()
            .expect("failed to start the vcam_native thread pool")
    })
}

fn bad(msg: &str) -> PyErr {
    PyValueError::new_err(msg.to_string())
}

/// (H, W) of an HxWx3 image, or a ValueError.
fn hw3(shape: &[usize], what: &str) -> PyResult<(usize, usize)> {
    match shape {
        [h, w, 3] => Ok((*h, *w)),
        _ => Err(bad(&format!("{what} must be an HxWx3 uint8 image"))),
    }
}

// --------------------------------------------------------------------------- //
// RVM input / output conversion
// --------------------------------------------------------------------------- //

/// BGR uint8 HxWx3 -> RVM's normalised RGB float32 (1,3,H,W), written into `src`.
/// Same as `RvmEngine.process`'s channel copy + `np.multiply(src, 1/255)`.
#[pyfunction]
fn fill_src(
    py: Python<'_>,
    frame: PyReadonlyArray3<'_, u8>,
    mut src: PyReadwriteArray4<'_, f32>,
) -> PyResult<()> {
    let (h, w) = hw3(frame.shape(), "frame")?;
    if src.shape() != [1, 3, h, w] {
        return Err(bad("src must be float32 (1, 3, H, W) matching the frame"));
    }
    let f = frame.as_slice()?;
    let s = src.as_slice_mut()?;
    py.allow_threads(|| {
        let (r, gb) = s.split_at_mut(h * w);
        let (g, b) = gb.split_at_mut(h * w);
        pool().install(|| {
            r.par_chunks_mut(w)
                .zip(g.par_chunks_mut(w))
                .zip(b.par_chunks_mut(w))
                .zip(f.par_chunks(w * 3))
                .for_each(|(((r, g), b), row)| {
                    for (x, px) in row.chunks_exact(3).enumerate() {
                        r[x] = px[2] as f32 * INV255;
                        g[x] = px[1] as f32 * INV255;
                        b[x] = px[0] as f32 * INV255;
                    }
                });
        });
    });
    Ok(())
}

/// Round half to even (like OpenCV's cvRound) for |x| < 2^22. Adding and
/// subtracting 1.5 * 2^23 makes the FPU's default rounding do the work; unlike
/// f32::round_ties_even it compiles to plain SSE2 adds, so loops vectorize.
#[inline(always)]
fn rte(x: f32) -> f32 {
    const MAGIC: f32 = 12_582_912.0;
    (x + MAGIC) - MAGIC
}

/// Saturating x*255 like cv2.convertScaleAbs (abs, round half to even).
#[inline(always)]
fn unit_to_u8(v: f32) -> u8 {
    rte((v * 255.0).abs().min(255.0)) as u8 // `as` saturates (and maps NaN to 0)
}

/// RVM foreground float32 (1,3,H,W) RGB in [0,1] -> new BGR uint8 HxWx3.
/// Same as the `cv2.merge(convertScaleAbs(...))` in `RvmEngine.process`. Returns
/// a fresh array because the output thread may still hold the previous frame's.
#[pyfunction]
fn fgr_to_bgr<'py>(
    py: Python<'py>,
    fgr: PyReadonlyArray4<'py, f32>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let (h, w) = match fgr.shape() {
        [1, 3, h, w] => (*h, *w),
        _ => return Err(bad("fgr must be float32 (1, 3, H, W)")),
    };
    let f = fgr.as_slice()?;
    // SAFETY: every element is written below before the array is returned.
    let out = unsafe { PyArray3::<u8>::new(py, [h, w, 3], false) };
    {
        let mut rw = out.readwrite();
        let o = rw.as_slice_mut()?;
        let (r, gb) = f.split_at(h * w);
        let (g, b) = gb.split_at(h * w);
        py.allow_threads(|| {
            pool().install(|| {
                o.par_chunks_mut(w * 3)
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
        });
    }
    Ok(out)
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
/// `a` and `b` (1,4,h,w) to the frame's size, apply them to the frame and
/// return (alpha HxW float32, foreground BGR uint8 HxWx3) as new arrays:
///     x       = [R, G, B, mean(R,G,B)] of the frame in [0, 1]
///     out     = up(a) * x + up(b)
///     fgr     = clip(out[0:3] + x[0:3], 0, 1)       (-> BGR uint8)
///     alpha   = clip(out[3], 0, 1)
/// This is the graph after tensors 777 (a) / 779 (b) of rvm_mobilenetv3: two
/// full-resolution Resize ops plus Mul/Add/Clip that onnxruntime runs as ~6
/// separate full-size passes, and fgr_to_bgr on top.
#[pyfunction]
fn guided_upsample<'py>(
    py: Python<'py>,
    a: PyReadonlyArray4<'py, f32>,
    b: PyReadonlyArray4<'py, f32>,
    frame: PyReadonlyArray3<'py, u8>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray3<u8>>)> {
    let (h, w) = hw3(frame.shape(), "frame")?;
    let (ch, cw) = match a.shape() {
        [1, 4, ch, cw] if *ch > 0 && *cw > 0 => (*ch, *cw),
        _ => return Err(bad("a must be float32 (1, 4, h, w)")),
    };
    if b.shape() != a.shape() {
        return Err(bad("b must have the same shape as a"));
    }
    let (av, bv, f) = (a.as_slice()?, b.as_slice()?, frame.as_slice()?);
    // SAFETY: every element of both arrays is written below before returning.
    let alpha = unsafe { PyArray2::<f32>::new(py, [h, w], false) };
    let fg = unsafe { PyArray3::<u8>::new(py, [h, w, 3], false) };
    {
        let mut alpha_rw = alpha.readwrite();
        let mut fg_rw = fg.readwrite();
        let (ao, fo) = (alpha_rw.as_slice_mut()?, fg_rw.as_slice_mut()?);
        let cols: Vec<(usize, usize, f32)> = (0..w).map(|x| resize_coord(x, w, cw)).collect();
        let plane = ch * cw;
        py.allow_threads(|| {
            pool().install(|| {
                ao.par_chunks_mut(w)
                    .zip(fo.par_chunks_mut(w * 3))
                    .zip(f.par_chunks(w * 3))
                    .enumerate()
                    .for_each_init(
                        || vec![0f32; 8 * cw],
                        |rows, (y, ((arow, frow_out), frow))| {
                            // Vertical pass: blend the two coarse rows of all 8
                            // coefficient planes once per output row.
                            let (y0, y1, dy) = resize_coord(y, h, ch);
                            for c in 0..8 {
                                let src = if c < 4 { av } else { bv };
                                let p = &src[(c % 4) * plane..];
                                let (r0, r1) = (&p[y0 * cw..][..cw], &p[y1 * cw..][..cw]);
                                for (d, (&u, &v)) in rows[c * cw..][..cw].iter_mut().zip(r0.iter().zip(r1)) {
                                    *d = u * (1.0 - dy) + v * dy;
                                }
                            }
                            // Horizontal pass + guided-filter apply, per pixel.
                            for (x, &(x0, x1, dx)) in cols.iter().enumerate() {
                                let up = |c: usize| {
                                    let r = &rows[c * cw..];
                                    r[x0] * (1.0 - dx) + r[x1] * dx
                                };
                                let px = &frow[x * 3..x * 3 + 3];
                                let rgb = [px[2] as f32 * INV255, px[1] as f32 * INV255, px[0] as f32 * INV255];
                                let mean = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
                                let out = &mut frow_out[x * 3..x * 3 + 3];
                                for c in 0..3 {
                                    let v = (up(c) * rgb[c] + up(4 + c) + rgb[c]).clamp(0.0, 1.0);
                                    out[2 - c] = unit_to_u8(v); // RGB -> BGR
                                }
                                arow[x] = (up(3) * mean + up(7)).clamp(0.0, 1.0);
                            }
                        },
                    );
            });
        });
    }
    Ok((alpha, fg))
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

fn keep_largest_impl(a: &mut [f32], h: usize, w: usize) {
    // Label at half resolution above 640 px, like vcam._keep_largest (blobs are
    // large). An exact 2x INTER_LINEAR downscale is the mean of each 2x2 block.
    let half = h.max(w) > 640;
    let (sh, sw) = if half { (h / 2, w / 2) } else { (h, w) };
    if sh == 0 || sw == 0 {
        return;
    }
    let mut bin = vec![0u8; sh * sw];
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

    // 8-connected components, union-find. The smaller provisional label always
    // wins a merge, so each root is the component's first pixel in raster order
    // -- the same order cv2.connectedComponents numbers them in, which keeps the
    // tie-break ("first largest") identical.
    let mut labels = vec![0u32; sh * sw];
    let mut parent: Vec<u32> = vec![0];
    for y in 0..sh {
        for x in 0..sw {
            let i = y * sw + x;
            if bin[i] == 0 {
                continue;
            }
            let mut nb = [0u32; 4];
            let mut k = 0;
            let mut push = |l: u32| {
                if l != 0 {
                    nb[k] = l;
                    k += 1;
                }
            };
            if x > 0 {
                push(labels[i - 1]);
            }
            if y > 0 {
                let up = i - sw;
                if x > 0 {
                    push(labels[up - 1]);
                }
                push(labels[up]);
                if x + 1 < sw {
                    push(labels[up + 1]);
                }
            }
            if k == 0 {
                let l = parent.len() as u32;
                parent.push(l);
                labels[i] = l;
            } else {
                let mut root = find(&mut parent, nb[0]);
                for &n in &nb[1..k] {
                    let r = find(&mut parent, n);
                    if r != root {
                        let (lo, hi) = (root.min(r), root.max(r));
                        parent[hi as usize] = lo;
                        root = lo;
                    }
                }
                labels[i] = root;
            }
        }
    }
    if parent.len() == 1 {
        return; // nothing detected; leave the matte as-is
    }
    let mut area = vec![0u32; parent.len()];
    for l in labels.iter_mut().filter(|l| **l != 0) {
        *l = find(&mut parent, *l);
        area[*l as usize] += 1;
    }
    let mut biggest = 0usize;
    for (l, &n) in area.iter().enumerate().skip(1) {
        if n > area[biggest] {
            biggest = l;
        }
    }
    let biggest = biggest as u32;

    // Per-row prefix counts of the kept blob, so the 11x11 elliptical dilation
    // is 11 O(1) span tests per pixel instead of ~100 neighbour reads. Also note
    // the blob's bounding box: nothing outside it (+5 px of dilation) survives.
    let pw = sw + 1;
    let mut pre = vec![0u32; sh * pw];
    let (mut x0, mut x1, mut y0, mut y1) = (sw, 0, sh, 0);
    for y in 0..sh {
        let mut acc = 0;
        for x in 0..sw {
            let hit = labels[y * sw + x] == biggest;
            if hit {
                (x0, x1, y0, y1) = (x0.min(x), x1.max(x), y0.min(y), y1.max(y));
            }
            acc += hit as u32;
            pre[y * pw + x + 1] = acc;
        }
    }
    let (x0, x1) = (x0.saturating_sub(5), (x1 + 6).min(sw));
    let (y0, y1) = (y0.saturating_sub(5), (y1 + 6).min(sh));

    // The dilated blob at low resolution, one row per task.
    let mut keep = vec![0u8; sh * sw];
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

    // Zero the matte outside the dilated blob, mapping each full-res pixel to its
    // nearest low-res one (INTER_NEAREST upscale). Same as `alpha * keep`, but in
    // place instead of allocating a new full-size float array.
    pool().install(|| {
        a.par_chunks_mut(w).enumerate().for_each(|(yy, row)| {
            let krow = &keep[(yy * sh / h) * sw..][..sw];
            for (xx, v) in row.iter_mut().enumerate() {
                if krow[xx * sw / w] == 0 {
                    *v = 0.0;
                }
            }
        });
    });
}

/// Keep only the single largest person blob in a float32 HxW matte, in place.
/// Same as vcam._keep_largest.
#[pyfunction]
fn keep_largest(py: Python<'_>, mut alpha: PyReadwriteArray2<'_, f32>) -> PyResult<()> {
    let (h, w) = match alpha.shape() {
        [h, w] => (*h, *w),
        _ => return Err(bad("alpha must be float32 HxW")),
    };
    let a = alpha.as_slice_mut()?;
    py.allow_threads(|| keep_largest_impl(a, h, w));
    Ok(())
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

/// OpenCV's default border (BORDER_REFLECT_101: ...2 1 | 0 1 2 ... n-2 | n-1 n-2...).
#[inline]
fn reflect101(i: isize, n: usize) -> usize {
    let n = n as isize;
    if n == 1 {
        return 0;
    }
    let mut i = i;
    while i < 0 || i >= n {
        i = if i < 0 { -i } else { 2 * n - 2 - i };
    }
    i as usize
}

/// out = bg + (fg - bg) * feather(alpha) / 255, written into `out`.
/// Same as Compositor.render: alpha -> uint8 (truncating), Gaussian-feathered
/// with a `ksize` kernel, then the int16 blend -- fused into two passes.
#[pyfunction]
fn composite(
    py: Python<'_>,
    fg: PyReadonlyArray3<'_, u8>,
    bg: PyReadonlyArray3<'_, u8>,
    alpha: PyReadonlyArray2<'_, f32>,
    mut out: PyReadwriteArray3<'_, u8>,
    ksize: usize,
) -> PyResult<()> {
    let (h, w) = hw3(fg.shape(), "fg")?;
    if hw3(bg.shape(), "bg")? != (h, w) || hw3(out.shape(), "out")? != (h, w) {
        return Err(bad("fg, bg and out must all be the same HxWx3 size"));
    }
    if alpha.shape() != [h, w] {
        return Err(bad("alpha must be float32 HxW matching the image"));
    }
    if ksize % 2 == 0 {
        return Err(bad("ksize must be odd"));
    }
    let (f, b, al) = (fg.as_slice()?, bg.as_slice()?, alpha.as_slice()?);
    let o = out.as_slice_mut()?;
    let kern = gaussian_kernel(ksize);
    let r = (ksize / 2) as isize;

    py.allow_threads(|| {
        pool().install(|| {
            // Pass 1: matte -> uint8 levels, horizontal blur into a float buffer.
            // (The kernels are dyadic fractions, so these sums are exact.)
            let mut tmp = vec![0f32; h * w];
            tmp.par_chunks_mut(w).zip(al.par_chunks(w)).for_each_init(
                || vec![0f32; w],
                |u, (t, arow)| {
                    for (d, &a) in u.iter_mut().zip(arow) {
                        *d = ((a * 255.0) as u8) as f32; // numpy astype(uint8) truncates
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
            o.par_chunks_mut(w * 3)
                .zip(f.par_chunks(w * 3))
                .zip(b.par_chunks(w * 3))
                .enumerate()
                .for_each_init(
                    || vec![0f32; w],
                    |acc, (y, ((orow, frow), brow))| {
                        acc.iter_mut().for_each(|v| *v = 0.0);
                        for (j, &k) in kern.iter().enumerate() {
                            let src = reflect101(y as isize - r + j as isize, h);
                            for (v, &t) in acc.iter_mut().zip(&tmp[src * w..(src + 1) * w]) {
                                *v += k * t;
                            }
                        }
                        for x in 0..w {
                            // Round half up, like OpenCV's fixed-point 8-bit blur.
                            let a = (acc[x] + 0.5).floor().min(255.0) * INV255;
                            for c in 0..3 {
                                let i = x * 3 + c;
                                let (fv, bv) = (frow[i] as i32, brow[i] as i32);
                                let d = rte((fv - bv) as f32 * a) as i32;
                                orow[i] = (bv + d).clamp(0, 255) as u8;
                            }
                        }
                    },
                );
        });
    });
    Ok(())
}

#[pymodule]
fn vcam_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fill_src, m)?)?;
    m.add_function(wrap_pyfunction!(fgr_to_bgr, m)?)?;
    m.add_function(wrap_pyfunction!(guided_upsample, m)?)?;
    m.add_function(wrap_pyfunction!(keep_largest, m)?)?;
    m.add_function(wrap_pyfunction!(composite, m)?)?;
    Ok(())
}

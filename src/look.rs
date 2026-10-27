//! "Realistic" picture backgrounds: make the picture read as the room the
//! webcam is actually filming, rather than a photo pasted behind you.
//!
//! A pasted picture gives itself away by not matching the camera, more than by
//! the matte's edge, so four things are matched to the live frame:
//!   * light -- the picture's brightness, colour cast and contrast are pulled
//!     part-way towards the real room's (measured behind you every frame,
//!     smoothed), so it follows the room's light and the camera's exposure;
//!   * focus -- a slight lens softness, the way a webcam renders a room a
//!     metre or two behind the person it's focused on;
//!   * grain -- the webcam's own sensor noise, measured and added as moving
//!     grain, so the background isn't cleaner than the person;
//!   * light wrap -- a little of the background's light spills onto the
//!     person's outline, as it would from a real backdrop.

use std::sync::Mutex;

use rayon::prelude::*;

use crate::imgproc::{self, Frame};
use crate::kernels::{self, Grade, Wrap};

/// How far the picture is pulled towards the room's mean colour / contrast
/// (0 = untouched, 1 = fully the room's). Half keeps the picture's own look.
const MATCH_MEAN: f32 = 0.5;
const MATCH_CONTRAST: f32 = 0.5;
/// Cap on the colour-cast shift (levels), so a strongly coloured wall can't
/// tint the picture.
const MAX_CAST: f32 = 16.0;
/// Light-wrap strength at the very edge.
const WRAP: f32 = 0.4;
/// Per-frame smoothing of the measurements (~0.4 s at 30 fps).
const EMA: f32 = 0.08;
/// Sample every Nth pixel in x and y when measuring the frame.
const STRIDE: usize = 4;

const LUMA: [f32; 3] = [0.114, 0.587, 0.299]; // BGR

/// Mean BGR and luma standard deviation.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tone {
    mean: [f32; 3],
    ystd: f32,
}

struct State {
    room: Option<Tone>,
    /// Webcam noise sigma, in levels.
    noise: Option<f32>,
    frame: u32,
}

pub struct Look {
    /// The picture's own tone, measured once.
    tone: Tone,
    /// Quarter-size, heavily blurred picture: the light that wraps the edges.
    light: Frame,
    state: Mutex<State>,
}

impl Look {
    /// `bg` is the picture as it will be composited (sized to the frame).
    /// Returns it with the lens softness applied, plus the look state.
    pub fn new(bg: &Frame, soften: bool) -> (Frame, Self) {
        let bg = if soften {
            bg.clone()
        } else {
            imgproc::gaussian_blur(bg, 1.2 * bg.h as f64 / 720.0)
        };
        let tone = tone(&bg, None).unwrap_or(Tone { mean: [128.0; 3], ystd: 50.0 });
        let small = imgproc::resize(&bg, (bg.w / 4).max(1), (bg.h / 4).max(1));
        let light = imgproc::gaussian_blur(&small, 2.0);
        (bg, Self { tone, light, state: Mutex::new(State { room: None, noise: None, frame: 0 }) })
    }

    /// Composite `fg` over the (softened) picture `bg`, matched to `raw` --
    /// the camera frame the matte came from.
    pub fn composite(&self, fg: &Frame, bg: &Frame, raw: &Frame, alpha: &[f32], ksize: usize) -> Frame {
        let (room, noise, seed) = {
            let mut s = self.state.lock().unwrap();
            if let Some(t) = tone(raw, Some(alpha)) {
                s.room = Some(match s.room {
                    Some(r) => Tone {
                        mean: [0, 1, 2].map(|c| r.mean[c] + (t.mean[c] - r.mean[c]) * EMA),
                        ystd: r.ystd + (t.ystd - r.ystd) * EMA,
                    },
                    None => t,
                });
            }
            // The noise level drifts slowly (with exposure / gain): measure it
            // every 8th frame, at 8x the smoothing.
            if s.frame % 8 == 0 {
                let n = noise_sigma(raw);
                s.noise = Some(s.noise.map_or(n, |v| v + (n - v) * (EMA * 8.0).min(1.0)));
            }
            s.frame = s.frame.wrapping_add(1);
            (s.room, s.noise.unwrap_or(0.0), s.frame)
        };
        let lut = tone_curve(self.tone, room.unwrap_or(self.tone));

        // The wrap: how much background surrounds each point (blurred
        // inverse matte), and the graded light it brings, at quarter size.
        let (sw, sh) = self.light.size();
        let mut mask = coverage(alpha, fg.w, fg.h, sw, sh);
        blur_plane(&mut mask, sw, sh, 1.5);
        let mut light = self.light.clone();
        for px in light.data.chunks_exact_mut(3) {
            for c in 0..3 {
                px[c] = lut[c][px[c] as usize];
            }
        }
        let grade = Grade {
            lut: &lut,
            grain: noise.clamp(0.0, 5.0),
            seed,
            wrap: Some(Wrap { mask: &mask, light: &light, strength: WRAP }),
        };
        kernels::composite_graded(fg, bg, alpha, ksize, Some(&grade))
    }
}

/// Tone of `img` (sampled), restricted to the background (alpha < 0.05) when
/// a matte is given. None if too little of it is visible to measure.
fn tone(img: &Frame, alpha: Option<&[f32]>) -> Option<Tone> {
    let (mut sum, mut y, mut yy, mut n) = ([0f64; 3], 0f64, 0f64, 0usize);
    for row in (0..img.h).step_by(STRIDE) {
        for x in (0..img.w).step_by(STRIDE) {
            let i = row * img.w + x;
            if alpha.is_some_and(|a| a[i] >= 0.05) {
                continue;
            }
            let px = &img.data[i * 3..i * 3 + 3];
            let mut l = 0f64;
            for c in 0..3 {
                sum[c] += px[c] as f64;
                l += LUMA[c] as f64 * px[c] as f64;
            }
            y += l;
            yy += l * l;
            n += 1;
        }
    }
    let total = img.h.div_ceil(STRIDE) * img.w.div_ceil(STRIDE);
    if n < (total / 50).max(16) {
        return None; // the person fills the frame: keep the last reading
    }
    let nf = n as f64;
    let ym = y / nf;
    Some(Tone {
        mean: sum.map(|s| (s / nf) as f32),
        ystd: (yy / nf - ym * ym).max(0.0).sqrt() as f32,
    })
}

/// Per-channel tone curves taking the picture's tone part-way to the room's.
fn tone_curve(pic: Tone, room: Tone) -> [[u8; 256]; 3] {
    let luma = |m: [f32; 3]| (0..3).map(|c| LUMA[c] * m[c]).sum::<f32>();
    let dy = luma(room.mean) - luma(pic.mean);
    let k = if pic.ystd > 1.0 { (room.ystd / pic.ystd).powf(MATCH_CONTRAST).clamp(0.8, 1.25) } else { 1.0 };
    let mut lut = [[0u8; 256]; 3];
    for c in 0..3 {
        let cast = (room.mean[c] - pic.mean[c] - dy).clamp(-MAX_CAST, MAX_CAST);
        let target = pic.mean[c] + MATCH_MEAN * (dy + cast);
        for (v, o) in lut[c].iter_mut().enumerate() {
            *o = (target + k * (v as f32 - pic.mean[c])).round().clamp(0.0, 255.0) as u8;
        }
    }
    lut
}

/// The webcam's noise level (sigma, in levels), by Immerkær's method: the
/// 3x3 Laplacian-difference mask cancels image structure, and its median
/// response over the frame is ~6 sigma for Gaussian noise.
fn noise_sigma(img: &Frame) -> f32 {
    let (w, h) = (img.w, img.h);
    if w < 3 || h < 3 {
        return 0.0;
    }
    let y = |x: usize, r: usize| {
        let p = &img.data[(r * w + x) * 3..];
        LUMA[0] * p[0] as f32 + LUMA[1] * p[1] as f32 + LUMA[2] * p[2] as f32
    };
    let mut resp = Vec::with_capacity((w / STRIDE) * (h / STRIDE));
    for r in (1..h - 1).step_by(STRIDE) {
        for x in (1..w - 1).step_by(STRIDE) {
            let l = 4.0 * y(x, r) - 2.0 * (y(x - 1, r) + y(x + 1, r) + y(x, r - 1) + y(x, r + 1))
                + y(x - 1, r - 1) + y(x + 1, r - 1) + y(x - 1, r + 1) + y(x + 1, r + 1);
            resp.push(l.abs());
        }
    }
    let mid = resp.len() / 2;
    let (_, m, _) = resp.select_nth_unstable_by(mid, f32::total_cmp);
    *m * 1.4826 / 6.0
}

/// 1 - alpha, box-averaged down to (sw, sh): how much background each cell holds.
fn coverage(alpha: &[f32], w: usize, h: usize, sw: usize, sh: usize) -> Vec<f32> {
    let mut out = vec![0f32; sw * sh];
    let rows = out.par_chunks_exact_mut(sw).enumerate();
    kernels::pool().install(|| rows.for_each(|(sy, row)| {
        let (y0, y1) = (sy * h / sh, ((sy + 1) * h / sh).max(sy * h / sh + 1));
        for (sx, o) in row.iter_mut().enumerate() {
            let (x0, x1) = (sx * w / sw, ((sx + 1) * w / sw).max(sx * w / sw + 1));
            let mut s = 0f32;
            for y in y0..y1 {
                s += alpha[y * w + x0..y * w + x1].iter().sum::<f32>();
            }
            *o = 1.0 - s / ((y1 - y0) * (x1 - x0)) as f32;
        }
    }));
    out
}

/// In-place separable Gaussian blur of a small single-channel plane, rows
/// split across the kernel pool. Runs every frame, so the horizontal taps read
/// a reflect-padded row and the vertical pass accumulates whole rows (no
/// per-tap border handling); the sums are in the same order as the plain loop.
fn blur_plane(p: &mut [f32], w: usize, h: usize, sigma: f32) {
    let r = (sigma * 3.0).ceil() as isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let ks: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / ks).collect();
    let mut tmp = vec![0f32; w * h];
    kernels::pool().install(|| {
        tmp.par_chunks_mut(w).zip(p.par_chunks(w)).for_each_init(
            || vec![0f32; w + k.len() - 1],
            |pad, (t, s)| {
                for (i, v) in pad.iter_mut().enumerate() {
                    *v = s[imgproc::reflect101(i as isize - r, w)];
                }
                for (x, t) in t.iter_mut().enumerate() {
                    *t = pad[x..x + k.len()].iter().zip(&k).map(|(v, kv)| kv * v).sum();
                }
            },
        );
        p.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            row.fill(0.0);
            for (j, &kv) in k.iter().enumerate() {
                let src = &tmp[imgproc::reflect101(y as isize + j as isize - r, h) * w..][..w];
                for (o, &v) in row.iter_mut().zip(src) {
                    *o += kv * v;
                }
            }
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noisy(w: usize, h: usize, sigma: f32) -> Frame {
        let mut f = Frame::filled(w, h, [120, 120, 120]);
        let mut s = 12345u32;
        let mut gauss = || {
            // Sum of 12 uniforms - 6: unit-variance, near-Gaussian.
            (0..12).map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (s >> 8) as f32 / (1u32 << 24) as f32
            }).sum::<f32>() - 6.0
        };
        for px in f.data.chunks_exact_mut(3) {
            let v = (120.0 + sigma * gauss()).round().clamp(0.0, 255.0) as u8;
            px.fill(v);
        }
        f
    }

    #[test]
    fn noise_estimate_tracks_sigma() {
        assert!(noise_sigma(&Frame::filled(320, 180, [90, 100, 110])) < 0.01);
        for sigma in [2.0f32, 4.0, 8.0] {
            let est = noise_sigma(&noisy(640, 360, sigma));
            assert!((est - sigma).abs() < sigma * 0.2, "sigma {sigma}: estimated {est}");
        }
    }

    #[test]
    fn tone_curve_is_identity_for_matching_room() {
        let t = Tone { mean: [100.0, 120.0, 140.0], ystd: 40.0 };
        let lut = tone_curve(t, t);
        for c in 0..3 {
            for v in 0..256 {
                assert_eq!(lut[c][v] as usize, v);
            }
        }
    }

    #[test]
    fn tone_curve_moves_half_way_and_caps_cast() {
        let pic = Tone { mean: [128.0; 3], ystd: 40.0 };
        let dark = Tone { mean: [64.0; 3], ystd: 40.0 };
        assert_eq!(tone_curve(pic, dark)[1][128], 96);
        // A saturated blue wall: the cast (blue vs red spread) is capped.
        let blue = Tone { mean: [228.0, 128.0, 28.0], ystd: 40.0 };
        let lut = tone_curve(pic, blue);
        let spread = lut[0][128] as f32 - lut[2][128] as f32;
        assert!(spread > 0.0 && spread <= 2.0 * MAX_CAST * MATCH_MEAN + 1.0, "spread {spread}");
    }

    #[test]
    fn room_is_measured_behind_the_person_only() {
        let (w, h) = (64, 32);
        let mut img = Frame::filled(w, h, [10, 10, 10]);
        let mut alpha = vec![0f32; w * h];
        for y in 0..h {
            for x in w / 2..w {
                alpha[y * w + x] = 1.0;
                img.data[(y * w + x) * 3..][..3].copy_from_slice(&[250, 250, 250]);
            }
        }
        let t = tone(&img, Some(&alpha)).unwrap();
        assert_eq!(t.mean, [10.0; 3]);
        assert!(tone(&img, Some(&vec![1.0; w * h])).is_none());
    }
}

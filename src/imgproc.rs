//! BGR images and the handful of OpenCV-style operations vcam needs: area /
//! bilinear resize, Gaussian blur, flip, cover-fit, and image file loading.
//!
//! Everything is separable and split across rows with rayon; the kernels
//! follow OpenCV's conventions (half-pixel centres, BORDER_REFLECT_101, the
//! sigma -> kernel size rule) so results match what the Python version did.

use std::path::Path;

use anyhow::{Context, Result};
use image::ImageDecoder;
use rayon::prelude::*;

/// An 8-bit BGR image, rows packed (`data.len() == w * h * 3`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Frame {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h * 3] }
    }

    pub fn filled(w: usize, h: usize, bgr: [u8; 3]) -> Self {
        let mut data = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            data.extend_from_slice(&bgr);
        }
        Self { w, h, data }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.w, self.h)
    }

    /// Horizontally mirrored copy.
    pub fn flipped(&self) -> Frame {
        let mut out = Frame::new(self.w, self.h);
        let w = self.w;
        out.data.par_chunks_mut(w * 3).zip(self.data.par_chunks(w * 3)).for_each(|(o, s)| {
            for x in 0..w {
                o[x * 3..x * 3 + 3].copy_from_slice(&s[(w - 1 - x) * 3..(w - 1 - x) * 3 + 3]);
            }
        });
        out
    }

    /// RGB bytes (for display).
    pub fn to_rgb(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.data.len()];
        out.par_chunks_mut(3 * 1024).zip(self.data.par_chunks(3 * 1024)).for_each(|(o, s)| {
            for (o, s) in o.chunks_exact_mut(3).zip(s.chunks_exact(3)) {
                o[0] = s[2];
                o[1] = s[1];
                o[2] = s[0];
            }
        });
        out
    }
}

// --------------------------------------------------------------------------- //
// Resampling
// --------------------------------------------------------------------------- //

/// Source taps of one output coordinate: `weights[i]` applies to `start + i`.
struct Taps {
    start: usize,
    weights: Vec<f32>,
}

/// cv2.INTER_LINEAR taps (half-pixel centres, edge-clamped).
fn linear_taps(out_len: usize, in_len: usize) -> Vec<Taps> {
    let scale = in_len as f64 / out_len as f64;
    (0..out_len)
        .map(|o| {
            let x = ((o as f64 + 0.5) * scale - 0.5).max(0.0);
            let x0 = (x.floor() as usize).min(in_len - 1);
            let f = (x - x0 as f64) as f32;
            if x0 + 1 >= in_len {
                Taps { start: in_len - 1, weights: vec![1.0] }
            } else {
                Taps { start: x0, weights: vec![1.0 - f, f] }
            }
        })
        .collect()
}

/// cv2.INTER_AREA taps for downscaling: each output pixel averages the source
/// interval it covers, weighting partially covered pixels by overlap.
fn area_taps(out_len: usize, in_len: usize) -> Vec<Taps> {
    let scale = in_len as f64 / out_len as f64;
    (0..out_len)
        .map(|o| {
            let (a, b) = (o as f64 * scale, ((o + 1) as f64 * scale).min(in_len as f64));
            let start = a.floor() as usize;
            let end = (b.ceil() as usize).min(in_len);
            let weights = (start..end)
                .map(|i| ((b.min(i as f64 + 1.0) - a.max(i as f64)) / (b - a)) as f32)
                .collect();
            Taps { start, weights }
        })
        .collect()
}

/// Separable resample of an interleaved `c`-channel u8 image.
fn resample(src: &[u8], sw: usize, sh: usize, c: usize, dw: usize, dh: usize,
            xt: &[Taps], yt: &[Taps]) -> Vec<u8> {
    // Horizontal pass into f32 rows (only the source rows something reads).
    let mut tmp = vec![0f32; sh * dw * c];
    tmp.par_chunks_mut(dw * c).zip(src.par_chunks(sw * c)).for_each(|(t, s)| {
        for (x, tap) in xt.iter().enumerate() {
            for ch in 0..c {
                let mut acc = 0f32;
                for (i, &wt) in tap.weights.iter().enumerate() {
                    acc += wt * s[(tap.start + i) * c + ch] as f32;
                }
                t[x * c + ch] = acc;
            }
        }
    });
    let mut out = vec![0u8; dw * dh * c];
    out.par_chunks_mut(dw * c).zip(yt.par_iter()).for_each(|(o, tap)| {
        let mut acc = vec![0f32; dw * c];
        for (i, &wt) in tap.weights.iter().enumerate() {
            let row = &tmp[(tap.start + i) * dw * c..][..dw * c];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += wt * v;
            }
        }
        for (o, a) in o.iter_mut().zip(&acc) {
            *o = (a + 0.5).clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// Resize choosing INTER_AREA when shrinking an axis and INTER_LINEAR otherwise.
pub fn resize(src: &Frame, w: usize, h: usize) -> Frame {
    if (w, h) == (src.w, src.h) {
        return src.clone();
    }
    let xt = if w < src.w { area_taps(w, src.w) } else { linear_taps(w, src.w) };
    let yt = if h < src.h { area_taps(h, src.h) } else { linear_taps(h, src.h) };
    Frame { w, h, data: resample(&src.data, src.w, src.h, 3, w, h, &xt, &yt) }
}

/// Bilinear resize (cv2.INTER_LINEAR), e.g. for upscaling the output.
pub fn resize_linear(src: &Frame, w: usize, h: usize) -> Frame {
    if (w, h) == (src.w, src.h) {
        return src.clone();
    }
    let (xt, yt) = (linear_taps(w, src.w), linear_taps(h, src.h));
    Frame { w, h, data: resample(&src.data, src.w, src.h, 3, w, h, &xt, &yt) }
}

/// Resize `img` to fill (w, h), cropping the overflow (no distortion).
pub fn resize_cover(img: &Frame, w: usize, h: usize) -> Frame {
    let scale = (w as f64 / img.w as f64).max(h as f64 / img.h as f64);
    let nw = ((img.w as f64 * scale).round() as usize).max(w);
    let nh = ((img.h as f64 * scale).round() as usize).max(h);
    let resized = resize(img, nw, nh);
    let (x0, y0) = ((nw - w) / 2, (nh - h) / 2);
    let mut out = Frame::new(w, h);
    for y in 0..h {
        let s = &resized.data[((y0 + y) * nw + x0) * 3..][..w * 3];
        out.data[y * w * 3..][..w * 3].copy_from_slice(s);
    }
    out
}

// --------------------------------------------------------------------------- //
// Gaussian blur
// --------------------------------------------------------------------------- //

/// OpenCV's default border (BORDER_REFLECT_101: ...2 1 | 0 1 2 ... n-2 | n-1 n-2...).
#[inline]
pub fn reflect101(i: isize, n: usize) -> usize {
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

/// Normalised Gaussian taps for `sigma`, sized like cv2.GaussianBlur((0, 0), sigma)
/// on 8-bit images (ksize = round(sigma * 6 + 1) | 1).
fn gaussian_taps(sigma: f64) -> Vec<f32> {
    let k = (((sigma * 6.0 + 1.0).round() as usize) | 1).max(1);
    let c = (k as f64 - 1.0) * 0.5;
    let w: Vec<f64> = (0..k).map(|i| (-(i as f64 - c).powi(2) / (2.0 * sigma * sigma)).exp()).collect();
    let sum: f64 = w.iter().sum();
    w.iter().map(|v| (v / sum) as f32).collect()
}

/// Gaussian blur of a BGR image (cv2.GaussianBlur with ksize (0, 0)).
pub fn gaussian_blur(img: &Frame, sigma: f64) -> Frame {
    let kern = gaussian_taps(sigma.max(0.01));
    let r = (kern.len() / 2) as isize;
    let (w, h) = (img.w, img.h);
    let mut tmp = vec![0f32; w * h * 3];
    tmp.par_chunks_mut(w * 3).zip(img.data.par_chunks(w * 3)).for_each(|(t, s)| {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0f32;
                for (j, &k) in kern.iter().enumerate() {
                    acc += k * s[reflect101(x as isize - r + j as isize, w) * 3 + c] as f32;
                }
                t[x * 3 + c] = acc;
            }
        }
    });
    let mut out = Frame::new(w, h);
    out.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, o)| {
        let mut acc = vec![0f32; w * 3];
        for (j, &k) in kern.iter().enumerate() {
            let row = &tmp[reflect101(y as isize - r + j as isize, h) * w * 3..][..w * 3];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += k * v;
            }
        }
        for (o, a) in o.iter_mut().zip(&acc) {
            *o = (a + 0.5).clamp(0.0, 255.0) as u8;
        }
    });
    out
}

/// Soft, out-of-focus version of a chosen background picture (the Meet-style
/// "blurred office photo" look). Built once per background choice, so there's
/// no speed budget: downscale -> blur -> upscale gives a smoother, more
/// photographic result than one big-sigma blur on the full-res image.
/// `strength` scales the blur sigma; 1.0 is the original fixed amount.
pub fn blur_background_image(img: &Frame, strength: f32) -> Frame {
    let (sw, sh) = ((img.w / 8).max(1), (img.h / 8).max(1));
    let small = gaussian_blur(&resize(img, sw, sh), 6.0 * strength.max(0.01) as f64);
    resize_linear(&small, img.w, img.h)
}

/// BGR frame -> RGB bytes of (w, h), for the GUI preview.
pub fn make_preview(frame: &Frame, w: usize, h: usize) -> Vec<u8> {
    resize(frame, w, h).to_rgb()
}

// --------------------------------------------------------------------------- //
// Files
// --------------------------------------------------------------------------- //

/// Decode an image file to RGB8, honouring its EXIF orientation.
pub fn load_rgb(path: &Path) -> Result<image::RgbImage> {
    let mut decoder = image::ImageReader::open(path)?.with_guessed_format()?.into_decoder()?;
    let orientation = decoder.orientation()?;
    let mut img = image::DynamicImage::from_decoder(decoder)?;
    img.apply_orientation(orientation);
    Ok(img.to_rgb8())
}

pub fn rgb_to_frame(img: &image::RgbImage) -> Frame {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = img.as_raw().clone();
    for px in data.chunks_exact_mut(3) {
        px.swap(0, 2);
    }
    Frame { w, h, data }
}

/// A background picture, cover-fitted to (w, h).
pub fn load_background_image(path: &Path, w: usize, h: usize) -> Result<Frame> {
    let img = load_rgb(path).with_context(|| format!("Could not read background image: {}", path.display()))?;
    Ok(resize_cover(&rgb_to_frame(&img), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_exact_halving_is_block_mean() {
        let mut f = Frame::new(4, 2);
        for (i, v) in f.data.iter_mut().enumerate() {
            *v = (i * 10 % 250) as u8;
        }
        let r = resize(&f, 2, 1);
        for x in 0..2 {
            for c in 0..3 {
                let s: u32 = [(0, 2 * x), (0, 2 * x + 1), (1, 2 * x), (1, 2 * x + 1)]
                    .iter()
                    .map(|&(y, xx)| f.data[(y * 4 + xx) * 3 + c] as u32)
                    .sum();
                assert!((r.data[x * 3 + c] as i32 - (s as f32 / 4.0).round() as i32).abs() <= 1);
            }
        }
    }

    #[test]
    fn cover_keeps_size() {
        let f = Frame::filled(300, 100, [1, 2, 3]);
        let c = resize_cover(&f, 128, 72);
        assert_eq!((c.w, c.h), (128, 72));
        assert!(c.data.chunks(3).all(|p| p == [1, 2, 3]));
    }

    #[test]
    fn blur_preserves_flat() {
        let f = Frame::filled(40, 30, [10, 200, 90]);
        assert_eq!(gaussian_blur(&f, 3.0), f);
    }
}

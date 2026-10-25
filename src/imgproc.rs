//! BGR images and the handful of OpenCV-style operations vcam needs: area /
//! bilinear resize, Gaussian blur, flip, cover-fit, and image file loading.
//!
//! Everything is separable and split across rows with rayon; the kernels
//! follow OpenCV's conventions (half-pixel centres, BORDER_REFLECT_101, the
//! sigma -> kernel size rule) so results match what the Python version did.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, Result};
use image::ImageDecoder;
use rayon::prelude::*;

// --------------------------------------------------------------------------- //
// Buffer recycling
// --------------------------------------------------------------------------- //

/// A free list for frame-sized buffers. Each frame allocates and frees a dozen
/// of them (frames, mattes, filter temporaries), and on Windows blocks this
/// large bypass the heap's cache: every one is fresh VirtualAlloc memory,
/// page-faulted in on first touch (~0.5 ms per 720p frame) and handed back to
/// the OS when freed. Reusing them removes that cost.
pub struct Recycler<T> {
    free: Mutex<Vec<Vec<T>>>,
}

/// Smaller buffers are left to the heap, which caches them itself.
const MIN_POOLED_BYTES: usize = 256 << 10;
/// Bound on what one pool holds onto.
const MAX_POOLED_BYTES: usize = 64 << 20;

impl<T: Copy + Default> Recycler<T> {
    const fn new() -> Self {
        Self { free: Mutex::new(Vec::new()) }
    }

    fn pooled(cap: usize) -> bool {
        cap * std::mem::size_of::<T>() >= MIN_POOLED_BYTES
    }

    /// A buffer of `len` elements with unspecified (stale) contents: only for
    /// callers that overwrite every element.
    pub fn scratch(&self, len: usize) -> Vec<T> {
        if Self::pooled(len) {
            let mut free = self.free.lock().unwrap();
            // Best fit: the smallest buffer that holds `len`.
            let best = (0..free.len()).filter(|&i| free[i].capacity() >= len).min_by_key(|&i| free[i].capacity());
            if let Some(i) = best {
                let mut v = free.swap_remove(i);
                drop(free);
                v.resize(len, T::default()); // truncates, or zeroes only the extension
                return v;
            }
        }
        vec![T::default(); len]
    }

    /// A zero-filled buffer of `len` elements.
    pub fn zeroed(&self, len: usize) -> Vec<T> {
        let mut v = self.scratch(len);
        v.fill(T::default());
        v
    }

    /// Hand a buffer back for reuse.
    pub fn give(&self, v: Vec<T>) {
        if !Self::pooled(v.capacity()) {
            return;
        }
        let mut free = self.free.lock().unwrap();
        let held: usize = free.iter().map(Vec::capacity).sum();
        if (held + v.capacity()) * std::mem::size_of::<T>() <= MAX_POOLED_BYTES {
            free.push(v);
        }
    }
}

pub static BYTES: Recycler<u8> = Recycler::new();
pub static FLOATS: Recycler<f32> = Recycler::new();
pub static U32S: Recycler<u32> = Recycler::new();

/// An 8-bit BGR image, rows packed (`data.len() == w * h * 3`). Its buffer is
/// recycled when it's dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl Drop for Frame {
    fn drop(&mut self) {
        BYTES.give(std::mem::take(&mut self.data));
    }
}

impl Frame {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: BYTES.zeroed(w * h * 3) }
    }

    /// A frame with unspecified contents, for producers that write every pixel.
    pub fn scratch(w: usize, h: usize) -> Self {
        Self { w, h, data: BYTES.scratch(w * h * 3) }
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
        let mut out = Frame::scratch(self.w, self.h);
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
        let mut out = BYTES.scratch(self.data.len());
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

/// Source taps of every output coordinate along one axis, in flat tables:
/// output `o` reads `len` source pixels from `start`, weighted by
/// `weights[woff..woff + len]`.
struct Taps {
    /// (start, len, woff) per output coordinate.
    spans: Vec<(usize, usize, usize)>,
    weights: Vec<f32>,
}

impl Taps {
    fn new(out_len: usize, mut tap: impl FnMut(usize, &mut Vec<f32>) -> usize) -> Self {
        let mut t = Taps { spans: Vec::with_capacity(out_len), weights: Vec::with_capacity(out_len * 2) };
        for o in 0..out_len {
            let woff = t.weights.len();
            let start = tap(o, &mut t.weights);
            t.spans.push((start, t.weights.len() - woff, woff));
        }
        t
    }

    fn get(&self, o: usize) -> (usize, &[f32]) {
        let (start, len, woff) = self.spans[o];
        (start, &self.weights[woff..woff + len])
    }
}

/// cv2.INTER_LINEAR taps (half-pixel centres, edge-clamped).
fn linear_taps(out_len: usize, in_len: usize) -> Taps {
    let scale = in_len as f64 / out_len as f64;
    Taps::new(out_len, |o, wts| {
        let x = ((o as f64 + 0.5) * scale - 0.5).max(0.0);
        let x0 = (x.floor() as usize).min(in_len - 1);
        let f = (x - x0 as f64) as f32;
        if x0 + 1 >= in_len {
            wts.push(1.0);
            in_len - 1
        } else {
            wts.extend([1.0 - f, f]);
            x0
        }
    })
}

/// cv2.INTER_AREA taps for downscaling: each output pixel averages the source
/// interval it covers, weighting partially covered pixels by overlap.
fn area_taps(out_len: usize, in_len: usize) -> Taps {
    let scale = in_len as f64 / out_len as f64;
    Taps::new(out_len, |o, wts| {
        let (a, b) = (o as f64 * scale, ((o + 1) as f64 * scale).min(in_len as f64));
        let start = a.floor() as usize;
        let end = (b.ceil() as usize).min(in_len);
        wts.extend((start..end).map(|i| ((b.min(i as f64 + 1.0) - a.max(i as f64)) / (b - a)) as f32));
        start
    })
}

/// Separable resample of a BGR image.
fn resample(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize, xt: &Taps, yt: &Taps) -> Vec<u8> {
    // Horizontal pass into f32 rows, all three channels per tap.
    let mut tmp = FLOATS.scratch(sh * dw * 3);
    tmp.par_chunks_mut(dw * 3).zip(src.par_chunks(sw * 3)).for_each(|(t, s)| {
        for (x, t) in t.chunks_exact_mut(3).enumerate() {
            let (start, wts) = xt.get(x);
            let mut acc = [0f32; 3];
            for (px, &wt) in s[start * 3..(start + wts.len()) * 3].chunks_exact(3).zip(wts) {
                for c in 0..3 {
                    acc[c] += wt * px[c] as f32;
                }
            }
            t.copy_from_slice(&acc);
        }
    });
    let mut out = BYTES.scratch(dw * dh * 3);
    let n = dw * 3;
    out.par_chunks_mut(n).enumerate().for_each_init(
        || vec![0f32; n],
        |acc, (y, o)| {
            let (start, wts) = yt.get(y);
            acc.fill(0.0);
            for (i, &wt) in wts.iter().enumerate() {
                for (a, &v) in acc.iter_mut().zip(&tmp[(start + i) * n..][..n]) {
                    *a += wt * v;
                }
            }
            for (o, a) in o.iter_mut().zip(acc.iter()) {
                *o = (a + 0.5).clamp(0.0, 255.0) as u8;
            }
        },
    );
    FLOATS.give(tmp);
    out
}

/// Resize choosing INTER_AREA when shrinking an axis and INTER_LINEAR otherwise.
pub fn resize(src: &Frame, w: usize, h: usize) -> Frame {
    if (w, h) == (src.w, src.h) {
        return src.clone();
    }
    let xt = if w < src.w { area_taps(w, src.w) } else { linear_taps(w, src.w) };
    let yt = if h < src.h { area_taps(h, src.h) } else { linear_taps(h, src.h) };
    Frame { w, h, data: resample(&src.data, src.w, src.h, w, h, &xt, &yt) }
}

/// Bilinear resize (cv2.INTER_LINEAR), e.g. for upscaling the output.
pub fn resize_linear(src: &Frame, w: usize, h: usize) -> Frame {
    if (w, h) == (src.w, src.h) {
        return src.clone();
    }
    let (xt, yt) = (linear_taps(w, src.w), linear_taps(h, src.h));
    Frame { w, h, data: resample(&src.data, src.w, src.h, w, h, &xt, &yt) }
}

/// Resize `img` to fill (w, h), cropping the overflow (no distortion).
pub fn resize_cover(img: &Frame, w: usize, h: usize) -> Frame {
    let scale = (w as f64 / img.w as f64).max(h as f64 / img.h as f64);
    let nw = ((img.w as f64 * scale).round() as usize).max(w);
    let nh = ((img.h as f64 * scale).round() as usize).max(h);
    let resized = resize(img, nw, nh);
    if (nw, nh) == (w, h) {
        return resized; // same aspect ratio: nothing to crop
    }
    let (x0, y0) = ((nw - w) / 2, (nh - h) / 2);
    let mut out = Frame::scratch(w, h);
    out.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, o)| {
        o.copy_from_slice(&resized.data[((y0 + y) * nw + x0) * 3..][..w * 3]);
    });
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
    let kl = kern.len();
    // Horizontal pass on a row padded with its reflected border, so the taps
    // are a straight dot product (no per-tap border handling).
    let mut tmp = FLOATS.scratch(w * h * 3);
    tmp.par_chunks_mut(w * 3).zip(img.data.par_chunks(w * 3)).for_each_init(
        || vec![0f32; (w + kl - 1) * 3],
        |pad, (t, s)| {
            for (i, p) in pad.chunks_exact_mut(3).enumerate() {
                let sx = reflect101(i as isize - r, w);
                for c in 0..3 {
                    p[c] = s[sx * 3 + c] as f32;
                }
            }
            for (x, t) in t.chunks_exact_mut(3).enumerate() {
                let mut acc = [0f32; 3];
                for (p, &k) in pad[x * 3..(x + kl) * 3].chunks_exact(3).zip(&kern) {
                    for c in 0..3 {
                        acc[c] += k * p[c];
                    }
                }
                t.copy_from_slice(&acc);
            }
        },
    );
    let mut out = Frame::scratch(w, h);
    out.data.par_chunks_mut(w * 3).enumerate().for_each_init(
        || vec![0f32; w * 3],
        |acc, (y, o)| {
            acc.fill(0.0);
            for (j, &k) in kern.iter().enumerate() {
                let row = &tmp[reflect101(y as isize - r + j as isize, h) * w * 3..][..w * 3];
                for (a, &v) in acc.iter_mut().zip(row) {
                    *a += k * v;
                }
            }
            for (o, a) in o.iter_mut().zip(acc.iter()) {
                *o = (a + 0.5).clamp(0.0, 255.0) as u8;
            }
        },
    );
    FLOATS.give(tmp);
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
    let mut f = resize(frame, w, h);
    // Swap to RGB in place rather than copying.
    f.data.par_chunks_mut(3 * 1024).for_each(|c| c.chunks_exact_mut(3).for_each(|px| px.swap(0, 2)));
    std::mem::take(&mut f.data)
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

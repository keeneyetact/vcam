//! The background image library: built-in pictures plus copies of the ones the
//! user adds (in %APPDATA%\vcam\backgrounds, so they survive the original
//! being moved or deleted).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use image::imageops::FilterType;

use vcam::imgproc::{self, Frame, Framing};
use vcam::paths;

pub const IMAGE_EXTS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "webp"];
/// Imported photos are downscaled to this (the output is 720p).
const MAX_LIBRARY_SIDE: u32 = 2560;

fn is_image(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

fn mtime(p: &Path) -> Option<SystemTime> {
    std::fs::metadata(p).and_then(|m| m.modified()).ok()
}

/// Images in `dir`, oldest first (library order == order added).
fn list_images(dir: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut files: Vec<PathBuf> = rd.flatten().map(|e| e.path()).filter(|p| p.is_file() && is_image(p)).collect();
    files.sort_by_key(|p| (mtime(p), p.file_name().map(|n| n.to_os_string())));
    files
}

/// (path, removable): built-in backgrounds first, then the user's.
pub fn library() -> Vec<(PathBuf, bool)> {
    list_images(&paths::builtin_backgrounds_dir())
        .into_iter()
        .map(|p| (p, false))
        .chain(list_images(&paths::library_dir()).into_iter().map(|p| (p, true)))
        .collect()
}

pub fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

pub fn in_library(path: &Path) -> bool {
    let Some(parent) = path.canonicalize().ok().and_then(|p| p.parent().map(Path::to_path_buf)) else {
        return false;
    };
    [paths::builtin_backgrounds_dir(), paths::library_dir()]
        .iter()
        .any(|d| d.canonicalize().is_ok_and(|d| d == parent))
}

/// Copy an image into the library; returns the copy's path, or None if it
/// isn't a readable image. Huge photos are downscaled (with their EXIF
/// rotation baked in), so switching to them stays fast.
pub fn import_image(src: &Path) -> Option<PathBuf> {
    let img = imgproc::load_rgb(src).ok()?;
    let lib = paths::library_dir();
    std::fs::create_dir_all(&lib).ok()?;
    let stem = src.file_stem()?.to_string_lossy().into_owned();
    let ext = src.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_else(|| "png".into());
    let mut dest = lib.join(src.file_name()?);
    let mut n = 2;
    while dest.exists() {
        dest = lib.join(format!("{stem}-{n}.{ext}"));
        n += 1;
    }
    if img.width().max(img.height()) > MAX_LIBRARY_SIDE {
        let k = MAX_LIBRARY_SIDE as f64 / img.width().max(img.height()) as f64;
        let (w, h) = ((img.width() as f64 * k).round() as u32, (img.height() as f64 * k).round() as u32);
        let small = image::imageops::resize(&img, w.max(1), h.max(1), FilterType::Lanczos3);
        let lower = ext.to_ascii_lowercase();
        if lower == "jpg" || lower == "jpeg" {
            let file = std::fs::File::create(&dest).ok()?;
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92);
            enc.encode_image(&small).ok()?;
        } else {
            small.save(&dest).ok()?;
        }
    } else {
        std::fs::copy(src, &dest).ok()?;
    }
    if let Ok(f) = std::fs::File::options().write(true).open(&dest) {
        let _ = f.set_modified(SystemTime::now()); // library order == order added
    }
    Some(dest)
}

/// Small decoded copies of library pictures for thumbnails: each photo is
/// decoded once (a big photo takes ~100 ms) and every thumbnail size is cut
/// from that.
#[derive(Default)]
pub struct ThumbCache {
    bases: Vec<((PathBuf, Option<SystemTime>), Frame)>,
}

impl ThumbCache {
    /// (w, h) RGB thumbnail of `path`, cropped and fitted as `framing` says.
    pub fn fitted(&mut self, path: &Path, framing: &Framing, w: usize, h: usize) -> Option<Vec<u8>> {
        let key = (path.to_path_buf(), mtime(path));
        let base = match self.bases.iter().position(|(k, _)| *k == key) {
            Some(i) => &self.bases[i].1,
            None => {
                let img = imgproc::rgb_to_frame(&imgproc::load_rgb(path).ok()?);
                let k = (480.0 / img.w.max(img.h) as f64).min(1.0);
                let base = imgproc::resize(&img, ((img.w as f64 * k) as usize).max(1), ((img.h as f64 * k) as usize).max(1));
                self.bases.push((key, base));
                &self.bases.last().unwrap().1
            }
        };
        Some(imgproc::frame_background(base, framing, w, h).to_rgb())
    }
}

//! Where vcam reads and writes things.

use std::path::PathBuf;

pub const APP_NAME: &str = "vcam";

/// The built-in background picture and window icon, compiled into the binary.
pub const DEFAULT_BACKGROUND: &[u8] = include_bytes!("../assets/backgrounds/default.jpg");
pub const ICON_PNG: &[u8] = include_bytes!("../assets/icon-256.png");

pub fn exe_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn env_dir(var: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// %APPDATA%\vcam: settings and the user's background library.
pub fn config_dir() -> PathBuf {
    env_dir("APPDATA").join(APP_NAME)
}

/// %LOCALAPPDATA%\vcam: downloaded / generated models and other caches.
pub fn cache_dir() -> PathBuf {
    env_dir("LOCALAPPDATA").join(APP_NAME)
}

pub fn settings_path() -> PathBuf {
    config_dir().join("settings.json")
}

/// Where images the user adds are copied, so they survive the original being
/// moved or deleted.
pub fn library_dir() -> PathBuf {
    config_dir().join("backgrounds")
}

/// Folder holding the built-in backgrounds, written out from the binary on
/// first use (the gallery and the loader work on files).
pub fn builtin_backgrounds_dir() -> PathBuf {
    let dir = cache_dir().join("builtin-backgrounds");
    let file = dir.join("default.jpg");
    let current = std::fs::metadata(&file).map(|m| m.len() == DEFAULT_BACKGROUND.len() as u64);
    if !matches!(current, Ok(true)) {
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&file, DEFAULT_BACKGROUND);
    }
    dir
}

/// `VCAM_MODEL_DIR`, if set, is the only place models are read and written.
fn model_dir_override() -> Option<PathBuf> {
    std::env::var_os("VCAM_MODEL_DIR").map(PathBuf::from)
}

/// Model search path: next to the exe, the working directory, then the cache.
pub fn model_dirs() -> Vec<PathBuf> {
    if let Some(d) = model_dir_override() {
        return vec![d];
    }
    let mut dirs = vec![exe_dir().join("models")];
    if let Ok(cwd) = std::env::current_dir() {
        dirs.push(cwd.join("models"));
    }
    dirs.push(writable_model_dir());
    dirs
}

pub fn writable_model_dir() -> PathBuf {
    model_dir_override().unwrap_or_else(|| cache_dir().join("models"))
}

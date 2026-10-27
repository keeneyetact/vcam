//! The control panel's remembered settings (%APPDATA%\vcam\settings.json).
//!
//! Read field by field, so a damaged or older file (including the one the
//! Python version wrote) never stops the app: anything missing or of the wrong
//! type falls back to its default.

use std::path::PathBuf;

use serde_json::{json, Map, Value};

use vcam::paths;

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Camera device name (older files stored "Camera N").
    pub camera: String,
    pub mirror: bool,
    pub mode: String,
    pub bg_path: Option<PathBuf>,
    pub blur_image: bool,
    pub blur_strength: Option<f64>,
    pub realistic: bool,
    pub speed: String,
    pub quality: String,
    pub one_person: bool,
    pub fix_exposure: bool,
    /// Picture brightness, -1 (darker) to 1 (brighter); 0 is as captured.
    pub brightness: f64,
    pub library_migrated: bool,
    pub advanced_open: bool,
    pub sidebar_open: bool,
}

impl Settings {
    pub fn load() -> Self {
        let m: Map<String, Value> = std::fs::read_to_string(paths::settings_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let s = |k: &str| m.get(k).and_then(Value::as_str).unwrap_or_default().to_string();
        let b = |k: &str, d: bool| m.get(k).and_then(Value::as_bool).unwrap_or(d);
        Self {
            camera: s("camera"),
            mirror: b("mirror", false),
            mode: s("mode"),
            bg_path: m.get("bg_path").and_then(Value::as_str).map(PathBuf::from),
            blur_image: b("blur_image", false),
            blur_strength: m.get("blur_strength").and_then(Value::as_f64),
            realistic: b("realistic", false),
            speed: s("speed"),
            quality: s("quality"),
            one_person: b("one_person", true),
            fix_exposure: b("fix_exposure", false),
            brightness: m.get("brightness").and_then(Value::as_f64).unwrap_or(0.0),
            library_migrated: b("library_migrated", false),
            advanced_open: b("advanced_open", true),
            sidebar_open: b("sidebar_open", true),
        }
    }

    /// Settings are a convenience; failing to save never blocks the app.
    pub fn save(&self) {
        let data = json!({
            "camera": self.camera,
            "mirror": self.mirror,
            "mode": self.mode,
            "bg_path": self.bg_path.as_ref().map(|p| p.to_string_lossy().into_owned()),
            "blur_image": self.blur_image,
            "blur_strength": self.blur_strength.map(|v| (v * 1000.0).round() / 1000.0),
            "realistic": self.realistic,
            "speed": self.speed,
            "quality": self.quality,
            "one_person": self.one_person,
            "fix_exposure": self.fix_exposure,
            "brightness": (self.brightness * 1000.0).round() / 1000.0,
            "library_migrated": self.library_migrated,
            "advanced_open": self.advanced_open,
            "sidebar_open": self.sidebar_open,
        });
        let path = paths::settings_path();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(&data) {
            let _ = std::fs::write(path, text);
        }
    }
}

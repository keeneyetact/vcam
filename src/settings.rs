//! The control panel's remembered settings (%APPDATA%\vcam\settings.json).
//!
//! Read field by field, so a damaged or older file (including the one the
//! Python version wrote) never stops the app: anything missing or of the wrong
//! type falls back to its default.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::{json, Map, Value};

use vcam::imgproc::{FitMode, Framing};
use vcam::paths;

const FITS: [(FitMode, &str); 3] = [(FitMode::Fill, "fill"), (FitMode::Fit, "fit"), (FitMode::Stretch, "stretch")];

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// Camera device name (older files stored "Camera N").
    pub camera: String,
    pub mirror: bool,
    pub mode: String,
    pub bg_path: Option<PathBuf>,
    /// Each picture's crop and fit, by path.
    pub framings: HashMap<PathBuf, Framing>,
    /// The camera picture's crop and fit.
    pub camera_framing: Framing,
    pub blur_image: bool,
    pub blur_strength: Option<f64>,
    pub realistic: bool,
    /// Outline the camera picture in the app's preview only, never in the
    /// virtual camera (picture background).
    pub outline: bool,
    pub speed: String,
    pub quality: String,
    pub one_person: bool,
    /// The dim-light frame-rate fix (adaptive exposure).
    pub keep_frame_rate: bool,
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
            framings: m
                .get("framings")
                .and_then(Value::as_object)
                .map(|o| o.iter().filter_map(|(k, v)| Some((PathBuf::from(k), framing_from(v)?))).collect())
                .unwrap_or_default(),
            camera_framing: m.get("camera_framing").and_then(framing_from).unwrap_or_default(),
            blur_image: b("blur_image", false),
            blur_strength: m.get("blur_strength").and_then(Value::as_f64),
            realistic: b("realistic", false),
            outline: b("outline", false),
            speed: s("speed"),
            quality: s("quality"),
            one_person: b("one_person", true),
            // Off by default. A new key: the old "keep_frame_rate" defaulted
            // on and was saved that way, and the older "fix_exposure" pinned
            // the exposure always.
            keep_frame_rate: b("dim_light_fps", false),
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
            "framings": self
                .framings
                .iter()
                .map(|(p, f)| (p.to_string_lossy().into_owned(), framing_to(f)))
                .collect::<Map<String, Value>>(),
            "camera_framing": framing_to(&self.camera_framing),
            "blur_image": self.blur_image,
            "blur_strength": self.blur_strength.map(|v| (v * 1000.0).round() / 1000.0),
            "realistic": self.realistic,
            "outline": self.outline,
            "speed": self.speed,
            "quality": self.quality,
            "one_person": self.one_person,
            "dim_light_fps": self.keep_frame_rate,
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

fn framing_from(v: &Value) -> Option<Framing> {
    let fit = v.get("fit").and_then(Value::as_str).and_then(|n| FITS.iter().find(|f| f.1 == n))?.0;
    let rect = |key: &str| -> Option<[f32; 4]> {
        let a = v.get(key).and_then(Value::as_array).filter(|a| a.len() == 4)?;
        let mut r = [0f32; 4];
        for (o, v) in r.iter_mut().zip(a) {
            *o = v.as_f64()? as f32;
        }
        Some(Framing::clamped(r))
    };
    Some(Framing { fit, crop: rect("crop")?, place: rect("place").unwrap_or(Framing::WHOLE) })
}

fn framing_to(f: &Framing) -> Value {
    let fit = FITS.iter().find(|x| x.0 == f.fit).map_or("fill", |x| x.1);
    let r = |r: [f32; 4]| r.map(|v| (v as f64 * 10000.0).round() / 10000.0);
    json!({ "fit": fit, "crop": r(f.crop), "place": r(f.place) })
}

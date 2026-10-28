//! The control panel.
//!
//!   * Live preview front and centre; one big Start / Stop button.
//!   * One-time virtual-camera setup that only shows up when it's needed (so the
//!     camera appears in Google Meet, Zoom, Teams, ... as "OBS Virtual Camera").
//!   * Background picker: Original, Blur, your own image, or a green screen.
//!   * Advanced settings (performance, edge detail, driver) tucked away.
//!   * Follows the Windows light/dark theme, is DPI-aware, and remembers settings.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use eframe::egui::{
    self, pos2, vec2, Align, Align2, Color32, CursorIcon, FontFamily, FontId, Key, Layout, Margin, Modifiers, Pos2,
    Rect, Response, RichText, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Theme, Vec2,
};
use rfd::{MessageButtons, MessageDialog, MessageDialogResult, MessageLevel};

use vcam::capture::ExposureMode;
use vcam::imgproc::{FitMode, Framing};
use vcam::worker::{BgMode, Preview, Status, VCamWorker};
use vcam::{capture, imgproc, paths, system, virtualcam, Frame};

use crate::library::{self, ThumbCache};
use crate::settings::Settings;

const OBS_URL: &str = "https://obsproject.com/download";

/// label, capture (w, h), output (w, h). "Smooth" captures at a fast low-res
/// the webcam can deliver at 30 fps, then upscales the matted output. The
/// capture size is 16:9 (matches the 1280x720 output) so the upscale is
/// uniform, and 360p mattes ~20% faster than 480p.
const SPEED_MODES: [(&str, usize, usize, Option<usize>, Option<usize>); 2] = [
    ("Sharp · 720p", 1280, 720, None, None),
    ("Smooth · 30 fps", 640, 360, Some(1280), Some(720)),
];

/// label -> RVM downsample_ratio (the quality<->speed dial; resolution
/// unchanged). Higher = sharper hair/edges but slower. Above ~0.5 you leave
/// RVM's trained coarse-resolution range, so the matte barely gets sharper
/// while speed craters; the two slowest rows are there so you can judge for
/// yourself, "Fine" is the practical sharpest.
const QUALITY_PRESETS: [(&str, f32); 8] = [
    ("Finest (very slow)", 0.75),
    ("Extra fine (slow)", 0.60),
    ("Fine", 0.50),
    ("Sharper", 0.40),
    ("Balanced (recommended)", 0.375),
    ("Smoother", 0.30),
    ("Fast", 0.25),
    ("Fastest", 0.20),
];
const QUALITY_DEFAULT: usize = 4;

/// The "Keep full frame rate in dim light" switch: on, the exposure is fixed
/// only once dim light actually slows the camera.
fn exposure_mode(keep_frame_rate: bool) -> ExposureMode {
    if keep_frame_rate { ExposureMode::Adaptive } else { ExposureMode::Camera }
}

/// Blur sigma multiplier range for the "Blur amount" slider (both the live Blur
/// mode and "Soften image"); 1.0 is the original fixed amount.
const BLUR_MIN: f32 = 0.01;
const BLUR_MAX: f32 = 1.0;
const BLUR_DEFAULT: f32 = 0.25;

const MODES: [(BgMode, &str, &str); 4] = [
    (BgMode::None, "none", "Original"),
    (BgMode::Blur, "blur", "Blur"),
    (BgMode::Image, "image", "Image"),
    (BgMode::Color, "color", "Green"),
];

const FITS: [(FitMode, &str, &str); 3] = [
    (FitMode::Fill, "Fill", "Fill the frame; the crop keeps the frame's shape, nothing is squeezed."),
    (FitMode::Fit, "Fit", "Show the whole crop, with soft blurred edges filling the sides."),
    (FitMode::Stretch, "Stretch", "Stretch the crop to fill the frame exactly."),
];

/// The crop editor shows the picture at most this many px a side.
const EDITOR_PX: u32 = 1024;

/// Thumbnail pixel sizes (drawn at 2x for high-DPI screens).
const TILE_PX: (usize, usize) = (240, 132);
const CELL_PX: (usize, usize) = (168, 94);

struct Palette {
    bg: Color32,
    surface: Color32,
    field: Color32,
    border: Color32,
    text: Color32,
    muted: Color32,
    accent: Color32,
    accent_hover: Color32,
    on_accent: Color32,
    danger: Color32,
    danger_hover: Color32,
    success: Color32,
    warn: Color32,
    tile: Color32,
    tile_sel: Color32,
    notice: Color32,
    notice_border: Color32,
    preview: Color32,
    preview_text: Color32,
}

const fn hex(v: u32) -> Color32 {
    Color32::from_rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
}

const LIGHT: Palette = Palette {
    bg: hex(0xf6f7f9), surface: hex(0xffffff), field: hex(0xffffff), border: hex(0xdfe3e8),
    text: hex(0x1f2328), muted: hex(0x5f6b7a), accent: hex(0x2563eb), accent_hover: hex(0x1d4ed8),
    on_accent: hex(0xffffff), danger: hex(0xdc2626), danger_hover: hex(0xb91c1c),
    success: hex(0x15803d), warn: hex(0xb45309), tile: hex(0xffffff), tile_sel: hex(0xeaf1fe),
    notice: hex(0xfff7ed), notice_border: hex(0xfed7aa), preview: hex(0x111317), preview_text: hex(0x9aa2ad),
};

const DARK: Palette = Palette {
    bg: hex(0x17191e), surface: hex(0x1f2228), field: hex(0x272b33), border: hex(0x343943),
    text: hex(0xe7e9ec), muted: hex(0x9aa3af), accent: hex(0x3b82f6), accent_hover: hex(0x2f6fe0),
    on_accent: hex(0xffffff), danger: hex(0xef4444), danger_hover: hex(0xdc2626),
    success: hex(0x4ade80), warn: hex(0xfbbf24), tile: hex(0x272b33), tile_sel: hex(0x1d2b48),
    notice: hex(0x2a2216), notice_border: hex(0x5b4520), preview: hex(0x0b0c0f), preview_text: hex(0x8b939e),
};

fn visuals(p: &Palette, dark: bool) -> egui::Visuals {
    let mut v = if dark { egui::Visuals::dark() } else { egui::Visuals::light() };
    v.panel_fill = p.bg;
    v.window_fill = p.surface;
    v.window_stroke = Stroke::new(1.0, p.border);
    v.extreme_bg_color = p.field;
    v.faint_bg_color = p.field;
    v.override_text_color = Some(p.text);
    v.hyperlink_color = p.accent;
    v.selection.bg_fill = p.accent;
    v.selection.stroke = Stroke::new(1.0, p.on_accent);
    let w = &mut v.widgets;
    w.noninteractive.bg_fill = p.surface;
    w.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
    w.noninteractive.fg_stroke = Stroke::new(1.0, p.text);
    for (s, fill, stroke) in [
        (&mut w.inactive, p.field, p.border),
        (&mut w.hovered, p.tile_sel, p.accent),
        (&mut w.active, p.tile_sel, p.accent),
        (&mut w.open, p.field, p.accent),
    ] {
        s.bg_fill = fill;
        s.weak_bg_fill = fill;
        s.bg_stroke = Stroke::new(1.0, stroke);
        s.fg_stroke = Stroke::new(1.0, p.text);
        s.corner_radius = 6.into();
        s.expansion = 0.0;
    }
    v
}

fn semi(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("semi".into()))
}

/// Turn a worker error into a plain-language next step.
pub fn friendly_error(raw: &str) -> String {
    let m = raw.to_lowercase();
    if m.contains("could not open camera") {
        "Can't access your webcam. Close other apps that might be using it (Meet, Zoom, Teams, OBS), \
         or pick a different camera, then try again."
            .into()
    } else if m.contains("stopped delivering") {
        "Your webcam stopped sending video. Check it's connected, then press Start camera.".into()
    } else if m.contains("download") {
        "Couldn't download the background model. Check your internet connection and try again.".into()
    } else if m.contains("onnxruntime") || m.contains("engine") {
        "The background engine couldn't start. Try restarting vcam.".into()
    } else if m.contains("already in use") {
        "The virtual camera is already in use — stop OBS's own virtual camera (or another vcam), \
         then press Start camera."
            .into()
    } else if m.contains("obs") || m.contains("virtual camera") || m.contains("backend") {
        "The virtual camera isn't set up yet. Use “Set up virtual camera” on the right, then press Start camera.".into()
    } else {
        format!("Something went wrong: {raw}")
    }
}

fn confirm(title: &str, text: &str) -> bool {
    MessageDialog::new()
        .set_title(title)
        .set_description(text)
        .set_level(MessageLevel::Info)
        .set_buttons(MessageButtons::YesNo)
        .show()
        == MessageDialogResult::Yes
}

fn message(level: MessageLevel, title: &str, text: &str) {
    MessageDialog::new().set_title(title).set_description(text).set_level(level).show();
}

// ------------------------------------------------------------ thumbnails --

/// A tiny RGB drawing surface for the generated tile thumbnails.
struct Canvas {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize, c: [u8; 3]) -> Self {
        Self { w, h, rgb: c.iter().copied().cycle().take(w * h * 3).collect() }
    }

    fn put(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if x < self.w && y < self.h {
            self.rgb[(y * self.w + x) * 3..][..3].copy_from_slice(&c);
        }
    }

    fn rect(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: [u8; 3]) {
        for y in y0.max(0.0) as usize..(y1.ceil() as usize).min(self.h) {
            for x in x0.max(0.0) as usize..(x1.ceil() as usize).min(self.w) {
                self.put(x, y, c);
            }
        }
    }

    /// Filled ellipse inscribed in the box (like PIL's ImageDraw.ellipse).
    fn ellipse(&mut self, x0: f64, y0: f64, x1: f64, y1: f64, c: [u8; 3]) {
        let (cx, cy, rx, ry) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (x1 - x0) / 2.0, (y1 - y0) / 2.0);
        for y in y0.max(0.0) as usize..(y1.ceil() as usize).min(self.h) {
            for x in x0.max(0.0) as usize..(x1.ceil() as usize).min(self.w) {
                let (dx, dy) = ((x as f64 + 0.5 - cx) / rx, (y as f64 + 0.5 - cy) / ry);
                if dx * dx + dy * dy <= 1.0 {
                    self.put(x, y, c);
                }
            }
        }
    }

    /// Draw a simple head-and-shoulders silhouette.
    fn person(mut self) -> Self {
        let (w, h) = (self.w as f64, self.h as f64);
        let col = [38, 42, 50];
        let (hr, cx, cy) = (h * 0.17, w / 2.0, h * 0.42);
        self.ellipse(cx - hr, cy - hr, cx + hr, cy + hr, col);
        self.ellipse(cx - w * 0.24, h * 0.66, cx + w * 0.24, h * 1.35, col);
        self
    }

    fn blurred(self, sigma: f64) -> Self {
        let mut bgr = self.rgb;
        bgr.chunks_exact_mut(3).for_each(|p| p.swap(0, 2));
        let f = imgproc::gaussian_blur(&Frame { w: self.w, h: self.h, data: bgr }, sigma);
        Self { w: f.w, h: f.h, rgb: f.to_rgb() }
    }

    fn image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgb([self.w, self.h], &self.rgb)
    }
}

fn texture(ctx: &egui::Context, name: &str, img: egui::ColorImage) -> TextureHandle {
    ctx.load_texture(name, img, TextureOptions::LINEAR)
}

// ------------------------------------------------------------------- app --

/// A part of the crop rectangle to drag: -1 the left / top edge, 1 the right /
/// bottom one, 0 neither; (0, 0) moves the whole crop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Grip {
    x: i8,
    y: i8,
}

impl Grip {
    const MOVE: Grip = Grip { x: 0, y: 0 };

    fn cursor(self, dragging: bool) -> CursorIcon {
        match (self.x, self.y) {
            (0, 0) if dragging => CursorIcon::Grabbing,
            (0, 0) => CursorIcon::Grab,
            (0, _) => CursorIcon::ResizeVertical,
            (_, 0) => CursorIcon::ResizeHorizontal,
            (x, y) if x == y => CursorIcon::ResizeNwSe,
            _ => CursorIcon::ResizeNeSw,
        }
    }
}

/// The grip of crop rectangle `r` under `pos` (screen), or None outside it.
/// With the shape locked only the corners resize.
fn grip_at(r: Rect, pos: Pos2, locked: bool) -> Option<Grip> {
    const TOL: f32 = 9.0;
    if !r.expand(TOL).contains(pos) {
        return None;
    }
    let side = |p: f32, lo: f32, hi: f32| {
        let tol = TOL.min((hi - lo) / 3.0);
        if (p - lo).abs() <= tol { -1 } else if (p - hi).abs() <= tol { 1 } else { 0 }
    };
    let g = Grip { x: side(pos.x, r.left(), r.right()), y: side(pos.y, r.top(), r.bottom()) };
    Some(if locked && (g.x == 0 || g.y == 0) { Grip::MOVE } else { g })
}

/// Smallest crop sides for a crop of shape `lock` (w / h, in fractions of
/// the picture), or for a free one.
fn min_sides(lock: Option<f32>) -> (f32, f32) {
    let m = Framing::MIN_SIDE;
    match lock {
        Some(a) => (m.max(m * a), m.max(m * a) / a),
        None => (m, m),
    }
}

/// `c` dragged by `d` (fractions of the picture) at `grip`, kept inside the
/// picture; `lock` keeps its shape (w / h, in fractions).
fn dragged_crop(c: [f32; 4], grip: Grip, d: Vec2, lock: Option<f32>) -> [f32; 4] {
    let [x0, y0, x1, y1] = c;
    if grip == Grip::MOVE {
        let dx = d.x.clamp(-x0, 1.0 - x1);
        let dy = d.y.clamp(-y0, 1.0 - y1);
        return [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
    }
    let (mw, mh) = min_sides(lock);
    let Some(a) = lock else {
        let mut c = c;
        match grip.x {
            -1 => c[0] = (x0 + d.x).clamp(0.0, x1 - mw),
            1 => c[2] = (x1 + d.x).clamp(x0 + mw, 1.0),
            _ => {}
        }
        match grip.y {
            -1 => c[1] = (y0 + d.y).clamp(0.0, y1 - mh),
            1 => c[3] = (y1 + d.y).clamp(y0 + mh, 1.0),
            _ => {}
        }
        return c;
    };
    // Locked: resize about the opposite corner, following whichever of the
    // pointer's x / y asks for the bigger crop.
    let (gx, gy) = (grip.x as f32, grip.y as f32);
    let (ax, ay) = (if gx > 0.0 { x0 } else { x1 }, if gy > 0.0 { y0 } else { y1 });
    let (cx, cy) = (if gx > 0.0 { x1 } else { x0 } + d.x, if gy > 0.0 { y1 } else { y0 } + d.y);
    let want = ((cx - ax) * gx).max((cy - ay) * gy * a);
    let room = (if gx > 0.0 { 1.0 - ax } else { ax }).min((if gy > 0.0 { 1.0 - ay } else { ay }) * a);
    let w = want.max(mw).min(room);
    let (nx, ny) = (ax + w * gx, ay + w / a * gy);
    [ax.min(nx), ay.min(ny), ax.max(nx), ay.max(ny)]
}

/// `c` scaled by `k` about its centre (shape kept), kept inside the picture.
fn zoomed_crop(c: [f32; 4], k: f32, lock: Option<f32>) -> [f32; 4] {
    let (mut w, mut h) = ((c[2] - c[0]) * k, (c[3] - c[1]) * k);
    let over = w.max(h).max(1.0); // never bigger than the picture
    (w, h) = (w / over, h / over);
    let (mw, mh) = min_sides(lock);
    let under = (mw / w).max(mh / h).max(1.0);
    (w, h) = (w * under, h * under);
    let (mx, my) = ((c[0] + c[2]) / 2.0, (c[1] + c[3]) / 2.0);
    let (x0, y0) = ((mx - w / 2.0).clamp(0.0, 1.0 - w), (my - h / 2.0).clamp(0.0, 1.0 - h));
    [x0, y0, x0 + w, y0 + h]
}

/// `c` slid against an edge of the picture (-1 / 1 per axis) or centred (0);
/// None leaves that axis alone.
fn aligned_crop(c: [f32; 4], x: Option<i8>, y: Option<i8>) -> [f32; 4] {
    let place = |lo: f32, hi: f32, to: Option<i8>| {
        let len = hi - lo;
        let lo = match to {
            Some(-1) => 0.0,
            Some(1) => 1.0 - len,
            Some(_) => (1.0 - len) / 2.0,
            None => lo,
        };
        (lo, lo + len)
    };
    let ((x0, x1), (y0, y1)) = (place(c[0], c[2], x), place(c[1], c[3], y));
    [x0, y0, x1, y1]
}

/// What a crop editor crops.
#[derive(Clone, Debug, PartialEq)]
enum CropTarget {
    /// A background picture.
    Picture(PathBuf),
    /// The live camera picture.
    Camera,
}

/// The crop editor window.
struct CropEditor {
    target: CropTarget,
    /// The picture, downscaled: loaded off the UI thread, or the live camera.
    tex: Option<TextureHandle>,
    job: Option<mpsc::Receiver<Option<egui::ColorImage>>>,
    /// The camera frame `tex` shows.
    shown: Option<Arc<Frame>>,
    /// What's being dragged, the crop when the drag began, and where.
    drag: Option<(Grip, [f32; 4], Pos2)>,
}

impl CropEditor {
    fn camera() -> Self {
        Self { target: CropTarget::Camera, tex: None, job: None, shown: None, drag: None }
    }

    fn picture(ctx: &egui::Context, path: PathBuf) -> Self {
        let (tx, rx) = mpsc::channel();
        let (src, ctx) = (path.clone(), ctx.clone());
        std::thread::spawn(move || {
            let img = imgproc::load_rgb(&src).ok().map(|img| {
                let k = (EDITOR_PX as f64 / img.width().max(img.height()) as f64).min(1.0);
                let (w, h) = (((img.width() as f64 * k) as u32).max(1), ((img.height() as f64 * k) as u32).max(1));
                let img = image::imageops::resize(&img, w, h, image::imageops::FilterType::Triangle);
                egui::ColorImage::from_rgb([w as usize, h as usize], img.as_raw())
            });
            let _ = tx.send(img);
            ctx.request_repaint();
        });
        Self { target: CropTarget::Picture(path), tex: None, job: Some(rx), shown: None, drag: None }
    }
}

struct GalleryItem {
    path: PathBuf,
    removable: bool,
    tex: TextureHandle,
}

pub struct App {
    worker: Arc<VCamWorker>,
    dark: bool,
    cameras: Vec<String>,
    cam: usize,
    mirror: bool,
    mode: BgMode,
    bg_path: Option<PathBuf>,
    framings: HashMap<PathBuf, Framing>,
    cam_framing: Framing,
    crop_edit: Option<CropEditor>,
    /// The picture tile waits for a crop drag / zoom to finish.
    tile_stale: bool,
    blur_image: bool,
    blur: f32,
    realistic: bool,
    speed: usize,
    quality: usize,
    one_person: bool,
    keep_frame_rate: bool,
    brightness: f32,
    adv_open: bool,
    side_open: bool,

    vc: virtualcam::Status,
    busy: Option<String>,
    vc_job: Option<(bool, mpsc::Receiver<(bool, String)>)>,
    cams_job: Option<mpsc::Receiver<Vec<String>>>,
    preview_tex: Option<TextureHandle>,
    preview_shown: Option<Arc<Preview>>,
    tile_tex: Vec<TextureHandle>,
    gallery: Vec<GalleryItem>,
    thumbs: ThumbCache,
    logo: Option<TextureHandle>,
    save_at: Option<Instant>,
    was_focused: bool,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = &cc.egui_ctx;
        ctx.set_visuals_of(Theme::Light, visuals(&LIGHT, false));
        ctx.set_visuals_of(Theme::Dark, visuals(&DARK, true));
        for theme in [Theme::Light, Theme::Dark] {
            ctx.style_mut_of(theme, |s| {
                s.spacing.item_spacing = vec2(8.0, 6.0);
                s.spacing.button_padding = vec2(10.0, 5.0);
                s.spacing.interact_size.y = 26.0;
            });
        }

        let s = Settings::load();
        let mut bg_path = s.bg_path.clone();
        if !s.library_migrated {
            // One-time: bring the image picked before the library existed (and
            // a personal 2.png next to the app) into the background library.
            let personal = paths::exe_dir().join("2.png");
            for old in [bg_path.clone(), Some(personal)].into_iter().flatten() {
                if old.is_file() && !library::in_library(&old) {
                    if let Some(copy) = library::import_image(&old) {
                        if Some(&old) == s.bg_path.as_ref() {
                            bg_path = Some(copy);
                        }
                    }
                }
            }
        }
        let mut bg_path = bg_path.filter(|p| p.is_file());
        if bg_path.is_none() {
            bg_path = library::library().first().map(|(p, _)| p.clone());
        }
        let mut mode = MODES
            .iter()
            .find(|m| m.1 == s.mode)
            .map(|m| m.0)
            .unwrap_or(if bg_path.is_some() { BgMode::Image } else { BgMode::Blur });
        if mode == BgMode::Image && bg_path.is_none() {
            mode = BgMode::Blur;
        }
        let cameras = capture::list_cameras();
        let cam = cameras
            .iter()
            .position(|n| *n == s.camera)
            .or_else(|| s.camera.strip_prefix("Camera ").and_then(|n| n.parse::<usize>().ok()).map(|n| n.saturating_sub(1)))
            .unwrap_or(0);

        let worker = Arc::new(VCamWorker::default());
        let repaint = ctx.clone();
        worker.set_on_preview(move || repaint.request_repaint());

        let logo = image::load_from_memory(paths::ICON_PNG).ok().map(|img| {
            let img = img.resize(64, 64, image::imageops::FilterType::Lanczos3).to_rgba8();
            let ci = egui::ColorImage::from_rgba_unmultiplied([img.width() as usize, img.height() as usize], &img);
            texture(ctx, "logo", ci)
        });

        let mut app = Self {
            worker,
            dark: ctx.theme() == Theme::Dark,
            cameras,
            cam,
            mirror: s.mirror,
            mode,
            bg_path,
            framings: s.framings.clone(),
            cam_framing: s.camera_framing,
            crop_edit: None,
            tile_stale: false,
            blur_image: s.blur_image,
            blur: s.blur_strength.map_or(BLUR_DEFAULT, |v| (v as f32).clamp(BLUR_MIN, BLUR_MAX)),
            realistic: s.realistic,
            speed: SPEED_MODES.iter().position(|m| m.0 == s.speed).unwrap_or(0),
            quality: QUALITY_PRESETS.iter().position(|q| q.0 == s.quality).unwrap_or(QUALITY_DEFAULT),
            one_person: s.one_person,
            keep_frame_rate: s.keep_frame_rate,
            brightness: (s.brightness as f32).clamp(-1.0, 1.0),
            adv_open: s.advanced_open,
            side_open: s.sidebar_open,
            vc: virtualcam::status(),
            busy: None,
            vc_job: None,
            cams_job: None,
            preview_tex: None,
            preview_shown: None,
            tile_tex: Vec::new(),
            gallery: Vec::new(),
            thumbs: ThumbCache::default(),
            logo,
            save_at: None,
            was_focused: true,
        };
        app.render_thumbs(ctx);
        app.render_gallery(ctx);
        if !s.library_migrated {
            app.save_settings();
        }
        app
    }

    fn pal(&self) -> &'static Palette {
        if self.dark {
            &DARK
        } else {
            &LIGHT
        }
    }

    // --------------------------------------------------------- persistence --

    fn schedule_save(&mut self) {
        self.save_at = Some(Instant::now() + Duration::from_millis(500));
    }

    fn save_settings(&mut self) {
        self.save_at = None;
        Settings {
            camera: self.cameras.get(self.cam).cloned().unwrap_or_else(|| format!("Camera {}", self.cam + 1)),
            mirror: self.mirror,
            mode: MODES.iter().find(|m| m.0 == self.mode).map_or("blur", |m| m.1).into(),
            bg_path: self.bg_path.clone(),
            framings: self.framings.clone(),
            camera_framing: self.cam_framing,
            blur_image: self.blur_image,
            blur_strength: Some(self.blur as f64),
            realistic: self.realistic,
            speed: SPEED_MODES[self.speed].0.into(),
            quality: QUALITY_PRESETS[self.quality].0.into(),
            one_person: self.one_person,
            keep_frame_rate: self.keep_frame_rate,
            brightness: self.brightness as f64,
            library_migrated: true,
            advanced_open: self.adv_open,
            sidebar_open: self.side_open,
        }
        .save();
    }

    // ---------------------------------------------------------- thumbnails --

    fn render_thumbs(&mut self, ctx: &egui::Context) {
        let (tw, th) = TILE_PX;
        let (w, h) = (tw as f64, th as f64);
        // Original: a room in focus; Blur: the same room out of focus.
        let mut room = Canvas::new(tw, th, [196, 184, 164]);
        room.rect(0.0, 0.0, w * 0.45, h, [120, 146, 176]);
        room.rect(w * 0.62, h * 0.15, w * 0.9, h * 0.7, [232, 220, 196]);
        room.rect(0.0, h * 0.78, w, h, [140, 112, 90]);
        let blur = Canvas { w: tw, h: th, rgb: room.rgb.clone() }.blurred(10.0).person();
        let room = room.person();
        // Image: the chosen picture (or a "+" placeholder).
        let pic = self
            .bg_path
            .clone()
            .and_then(|p| self.thumbs.fitted(&p, &self.framing(), tw, th))
            .map(|rgb| Canvas { w: tw, h: th, rgb }.person())
            .unwrap_or_else(|| {
                let mut c = Canvas::new(tw, th, if self.dark { [60, 66, 76] } else { [210, 214, 220] });
                let (mx, my, a, t) = (w / 2.0, h / 2.0, 18.0, 2.0);
                c.rect(mx - a, my - t, mx + a, my + t, [110, 118, 130]);
                c.rect(mx - t, my - a, mx + t, my + a, [110, 118, 130]);
                c
            });
        let green = Canvas::new(tw, th, [0, 177, 64]).person();
        self.tile_tex = [room, blur, pic, green]
            .iter()
            .enumerate()
            .map(|(i, c)| texture(ctx, &format!("tile{i}"), c.image()))
            .collect();
    }

    fn render_gallery(&mut self, ctx: &egui::Context) {
        let (cw, ch) = CELL_PX;
        self.gallery = library::library()
            .into_iter()
            .filter_map(|(path, removable)| {
                let rgb = self.thumbs.fitted(&path, &Framing::default(), cw, ch)?;
                let tex = texture(ctx, &path.to_string_lossy(), egui::ColorImage::from_rgb([cw, ch], &rgb));
                Some(GalleryItem { path, removable, tex })
            })
            .collect();
    }

    // ------------------------------------------------------- image library --

    fn select_image(&mut self, ctx: &egui::Context, path: PathBuf) {
        // Tell the worker first (it loads the picture on its own thread), then
        // just move the highlight -- no gallery rebuild, no re-decoding.
        self.bg_path = Some(path.clone());
        self.mode = BgMode::Image;
        self.worker.set_background(BgMode::Image, Some(path), self.framing(), Some(self.blur_image));
        self.render_thumbs(ctx);
        self.schedule_save();
    }

    fn add_images(&mut self, ctx: &egui::Context) -> bool {
        let Some(picked) = rfd::FileDialog::new()
            .set_title("Add background images")
            .add_filter("Images", &library::IMAGE_EXTS)
            .add_filter("All files", &["*"])
            .pick_files()
        else {
            return false;
        };
        self.import_images(ctx, &picked)
    }

    /// Copy pictures (picked or dropped onto the window) into the library and
    /// use the last one.
    fn import_images(&mut self, ctx: &egui::Context, picked: &[PathBuf]) -> bool {
        let added: Vec<PathBuf> = picked.iter().filter_map(|p| library::import_image(p)).collect();
        if added.len() < picked.len() {
            message(
                MessageLevel::Warning,
                "Add images",
                &format!("{} file(s) couldn't be opened as an image and were skipped.", picked.len() - added.len()),
            );
        }
        let Some(last) = added.last().cloned() else { return false };
        self.render_gallery(ctx);
        self.select_image(ctx, last);
        true
    }

    fn remove_image(&mut self, ctx: &egui::Context, path: &Path) {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        if !confirm("Remove background", &format!("Remove “{name}” from your backgrounds?\n\nYour original file isn't affected.")) {
            return;
        }
        if let Err(e) = std::fs::remove_file(path) {
            message(MessageLevel::Error, "Remove background", &format!("Couldn't remove it: {e}"));
            return;
        }
        if self.bg_path.as_deref().is_some_and(|p| library::same_file(p, path) || !p.exists()) {
            self.bg_path = library::library().first().map(|(p, _)| p.clone());
            if self.bg_path.is_none() && self.mode == BgMode::Image {
                self.mode = BgMode::Blur;
            }
            self.worker.set_background(self.mode, self.bg_path.clone(), self.framing(), Some(self.blur_image));
        }
        self.framings.remove(path);
        self.render_gallery(ctx);
        self.render_thumbs(ctx);
        self.schedule_save();
    }

    /// The chosen picture's crop and fit.
    fn framing(&self) -> Framing {
        self.bg_path.as_ref().and_then(|p| self.framings.get(p)).copied().unwrap_or_default()
    }

    /// Change the chosen picture's crop / fit; it applies live. `dragging`:
    /// more changes follow at once, so the thumbnail waits for the last.
    fn set_framing(&mut self, ctx: &egui::Context, f: Framing, dragging: bool) {
        let Some(path) = self.bg_path.clone() else { return };
        let f = Framing { crop: Framing::clamped(f.crop), ..f };
        if f == Framing::default() {
            self.framings.remove(&path);
        } else {
            self.framings.insert(path, f);
        }
        self.worker.set_framing(f);
        if dragging {
            self.tile_stale = true;
        } else {
            self.tile_stale = false;
            self.render_thumbs(ctx);
        }
        self.schedule_save();
    }

    fn framing_of(&self, target: &CropTarget) -> Framing {
        match target {
            CropTarget::Camera => self.cam_framing,
            CropTarget::Picture(_) => self.framing(),
        }
    }

    fn set_framing_of(&mut self, ctx: &egui::Context, target: &CropTarget, f: Framing, dragging: bool) {
        match target {
            CropTarget::Camera => {
                self.cam_framing = Framing { crop: Framing::clamped(f.crop), ..f };
                self.worker.configure(|c| c.camera_framing = self.cam_framing); // live, from the next frame
                self.schedule_save();
            }
            CropTarget::Picture(_) => self.set_framing(ctx, f, dragging),
        }
    }

    /// The output's shape (w / h): the camera's while running.
    fn frame_aspect(&self) -> f32 {
        self.worker.camera_info().map_or(16.0 / 9.0, |c| c.width as f32 / c.height.max(1) as f32)
    }

    /// Image mode needs a picture: use the first in the library, else ask.
    fn ensure_image(&mut self, ctx: &egui::Context) -> bool {
        if self.bg_path.as_deref().is_some_and(Path::is_file) {
            return true;
        }
        if let Some((p, _)) = library::library().into_iter().next() {
            self.select_image(ctx, p);
            return true;
        }
        self.add_images(ctx)
    }

    // ------------------------------------------------------------- actions --

    fn refresh_vc_status(&mut self) {
        self.vc = virtualcam::status();
    }

    fn vcam_action(&mut self, ctx: &egui::Context, install: bool) {
        if self.busy.is_some() {
            return;
        }
        if !install && self.worker.running() {
            message(MessageLevel::Warning, "Virtual camera", "Stop the camera first.");
            return;
        }
        if !install
            && !confirm(
                "Virtual camera",
                "Uninstall the virtual camera?\n\nMeet, Zoom and Teams won't be able to see vcam until you \
                 set it up again. Windows will ask for administrator permission.",
            )
        {
            return;
        }
        self.busy = Some("Waiting for Windows permission…".into());
        let (tx, rx) = mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(if install { virtualcam::install() } else { virtualcam::uninstall() });
            ctx.request_repaint();
        });
        self.vc_job = Some((install, rx));
    }

    fn toggle(&mut self, ctx: &egui::Context) {
        if self.worker.running() || self.worker.status() == Status::Starting {
            self.worker.stop();
            return;
        }
        if self.mode == BgMode::Image && !self.ensure_image(ctx) {
            return;
        }
        let (_, cw, ch, ow, oh) = SPEED_MODES[self.speed];
        self.worker.configure(|c| {
            c.camera = self.cam;
            c.mode = self.mode;
            c.bg_path = self.bg_path.clone();
            c.framing = self.framing();
            c.camera_framing = self.cam_framing;
            c.blur_image = self.blur_image;
            c.blur_strength = self.blur;
            c.realistic = self.realistic;
            c.mirror = self.mirror;
            c.multi_person = !self.one_person;
            c.exposure = exposure_mode(self.keep_frame_rate);
            c.brightness = self.brightness;
            c.downsample = Some(QUALITY_PRESETS[self.quality].1);
            (c.width, c.height, c.out_width, c.out_height) = (cw, ch, ow, oh);
        });
        self.worker.start();
    }

    fn apply_bg(&mut self, ctx: &egui::Context) {
        if self.mode == BgMode::Image && !self.ensure_image(ctx) {
            self.mode = BgMode::Blur;
        }
        self.worker.set_background(self.mode, self.bg_path.clone(), self.framing(), Some(self.blur_image));
        self.schedule_save();
    }

    /// Per-frame housekeeping: keys, finished background jobs, focus, close.
    fn housekeeping(&mut self, ctx: &egui::Context) {
        let (enter, open, side) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::CTRL, Key::Enter),
                i.consume_key(Modifiers::CTRL, Key::O),
                i.consume_key(Modifiers::CTRL, Key::B),
            )
        });
        if enter {
            self.toggle(ctx);
        }
        if open {
            self.add_images(ctx);
        }
        if side {
            self.side_open = !self.side_open;
            self.schedule_save();
        }
        let dropped: Vec<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        if !dropped.is_empty() {
            self.import_images(ctx, &dropped);
        }

        if let Some((_, rx)) = &self.vc_job {
            if let Ok((ok, msg)) = rx.try_recv() {
                self.vc_job = None;
                self.busy = None;
                self.refresh_vc_status();
                if !ok {
                    message(MessageLevel::Error, "Virtual camera", &msg);
                }
            }
        }
        if let Some(rx) = &self.cams_job {
            if let Ok(cams) = rx.try_recv() {
                self.cams_job = None;
                // Keep the same device selected if it moved in the list.
                let current = self.cameras.get(self.cam).cloned();
                if let Some(i) = current.and_then(|c| cams.iter().position(|n| *n == c)) {
                    self.cam = i;
                }
                self.cameras = cams;
            }
        }

        // Re-check the driver (and cameras) when the user comes back, e.g.
        // after installing OBS.
        let focused = ctx.input(|i| i.focused);
        if focused && !self.was_focused && self.busy.is_none() {
            self.refresh_vc_status();
            if self.cams_job.is_none() {
                let (tx, rx) = mpsc::channel();
                let c = ctx.clone();
                std::thread::spawn(move || {
                    let _ = tx.send(capture::list_cameras());
                    c.request_repaint();
                });
                self.cams_job = Some(rx);
            }
        }
        self.was_focused = focused;

        if ctx.input(|i| i.viewport().close_requested()) {
            if self.worker.running() && !confirm("Quit vcam", "Your virtual camera is on. Quit and turn it off?") {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            } else {
                self.save_settings();
                self.worker.stop();
            }
        }
        if self.save_at.is_some_and(|t| Instant::now() >= t) {
            self.save_settings();
        }

        // The worker's output thread builds the preview already fitted and in
        // RGB, so this thread only uploads it.
        match self.worker.latest_preview() {
            Some(p) if !self.preview_shown.as_ref().is_some_and(|s| Arc::ptr_eq(s, &p)) => {
                let img = egui::ColorImage::from_rgb([p.w, p.h], &p.rgb);
                match &mut self.preview_tex {
                    Some(t) => t.set(img, TextureOptions::LINEAR),
                    None => self.preview_tex = Some(texture(ctx, "preview", img)),
                }
                self.preview_shown = Some(p);
            }
            Some(_) => {}
            None => {
                self.preview_tex = None;
                self.preview_shown = None;
            }
        }
    }

    // ------------------------------------------------------------------ UI --

    fn paint_preview(&mut self, ui: &mut egui::Ui, rect: Rect) {
        let p = self.pal();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 8.0, p.preview);
        let ppp = ui.ctx().pixels_per_point();
        self.worker.set_preview_box(Some(((rect.width() * ppp) as usize, (rect.height() * ppp) as usize)));

        let state = self.worker.status();
        let tex = self.preview_tex.as_ref().filter(|_| state == Status::Running);
        if let Some(t) = tex {
            let size = t.size_vec2() / ppp;
            egui::Image::from_texture(t).corner_radius(8).paint_at(ui, Rect::from_center_size(rect.center(), size));
        } else {
            // Running but no frame yet still reads as starting.
            self.paint_empty_preview(ui, rect, if state == Status::Running { Status::Starting } else { state });
        }

        // Status pill (top-left).
        let pill = match state {
            Status::Running => {
                let dev = self.worker.device().map(|d| format!("  ·  {d}")).unwrap_or_default();
                Some((format!("●  LIVE  ·  {:.0} fps{dev}", self.worker.fps()), hex(0xff5a5a)))
            }
            Status::Starting => Some(("●  Starting camera…".to_string(), p.warn)),
            _ => None,
        };
        if let Some((text, color)) = pill {
            let galley = painter.layout_no_wrap(text, semi(12.0), color);
            let r = Rect::from_min_size(rect.min + vec2(12.0, 12.0), galley.size() + vec2(20.0, 8.0));
            painter.rect_filled(r, 4.0, Color32::from_black_alpha(200));
            painter.galley(r.min + vec2(10.0, 4.0), galley, color);
        }

        // Error bar (bottom), hover for the technical detail.
        if state == Status::Error {
            let raw = self.worker.error().unwrap_or_default();
            let width = (rect.width() - 24.0).max(200.0);
            let galley = painter.layout(friendly_error(&raw), FontId::proportional(14.0), Color32::WHITE, width - 24.0);
            let h = galley.size().y + 16.0;
            let r = Rect::from_min_size(pos2(rect.left() + 12.0, rect.bottom() - 12.0 - h), vec2(width, h));
            painter.rect_filled(r, 4.0, p.danger);
            painter.galley(r.min + vec2(12.0, 8.0), galley, Color32::WHITE);
            ui.interact(r, ui.make_persistent_id("error_bar"), Sense::hover()).on_hover_text(format!("Details: {raw}"));
        }
    }

    /// The preview while there's no video: what's happening, and the one
    /// thing to do next (a Start / Try again button right where you look).
    fn paint_empty_preview(&mut self, ui: &mut egui::Ui, rect: Rect, state: Status) {
        let p = self.pal();
        let painter = ui.painter_at(rect);
        let (head, sub, button) = match state {
            Status::Starting => ("Starting your camera…", "The first start can take a few seconds.", None),
            Status::Error => ("Your camera stopped", "See the message below for what to do.", Some("Try again")),
            _ => ("Your camera is off", "Start it to see your preview and go live in your call.", Some("Start camera")),
        };
        let bright = Color32::from_gray(232);
        let mut y = rect.center().y - if button.is_some() { 92.0 } else { 56.0 };
        let cx = rect.center().x;

        // Icon: a spinner while starting, else a camera (crossed out when off).
        let icon = Rect::from_center_size(pos2(cx, y + 18.0), vec2(56.0, 36.0));
        if state == Status::Starting {
            egui::Spinner::new().size(32.0).color(p.preview_text).paint_at(ui, icon);
        } else {
            let stroke = Stroke::new(2.0, p.preview_text);
            let body = Rect::from_min_max(icon.min + vec2(2.0, 4.0), pos2(icon.left() + 38.0, icon.bottom() - 4.0));
            painter.rect_stroke(body, 6.0, stroke, StrokeKind::Middle);
            let (lx, ly) = (body.right() + 4.0, body.center().y);
            painter.add(egui::Shape::convex_polygon(
                vec![pos2(lx, ly - 4.0), pos2(lx + 12.0, ly - 10.0), pos2(lx + 12.0, ly + 10.0), pos2(lx, ly + 4.0)],
                Color32::TRANSPARENT,
                stroke,
            ));
            let (a, b) = (icon.left_top() + vec2(4.0, -2.0), icon.right_bottom() - vec2(4.0, -2.0));
            painter.line_segment([a, b], Stroke::new(6.0, p.preview));
            painter.line_segment([a, b], stroke);
        }
        y += 36.0 + 18.0;

        let wrap = (rect.width() - 48.0).max(120.0);
        let g = painter.layout(head.into(), semi(19.0), bright, wrap);
        painter.galley(pos2(cx - g.size().x / 2.0, y), g.clone(), bright);
        y += g.size().y + 6.0;
        let g = painter.layout(sub.into(), FontId::proportional(13.5), p.preview_text, wrap);
        painter.galley(pos2(cx - g.size().x / 2.0, y), g.clone(), p.preview_text);
        y += g.size().y + 22.0;

        if let Some(label) = button {
            let r = Rect::from_center_size(pos2(cx, y + 20.0), vec2(180.0, 40.0));
            let resp = ui.interact(r, ui.make_persistent_id("preview_start"), Sense::click());
            resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, label));
            let fill = if resp.hovered() { p.accent_hover } else { p.accent };
            painter.rect_filled(r, 6.0, fill);
            painter.text(r.center(), Align2::CENTER_CENTER, label, semi(15.0), p.on_accent);
            Self::focus_ring(ui, p, &resp, r, 6.0);
            let hint = painter.layout_no_wrap("or press Ctrl+Enter".into(), FontId::proportional(12.0), p.preview_text);
            painter.galley(pos2(cx - hint.size().x / 2.0, r.bottom() + 10.0), hint, p.preview_text);
            if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                self.toggle(ui.ctx());
            }
        }
    }

    fn section(ui: &mut egui::Ui, p: &Palette, title: &str) {
        ui.add_space(14.0);
        ui.label(RichText::new(title).font(semi(12.0)).color(p.muted));
        ui.add_space(2.0);
    }

    fn link(ui: &mut egui::Ui, p: &Palette, text: &str) -> Response {
        ui.add(egui::Button::new(RichText::new(text).font(semi(12.5)).color(p.accent)).frame(false))
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    /// Keyboard focus ring around a custom widget (Tab moves between them).
    fn focus_ring(ui: &egui::Ui, p: &Palette, resp: &Response, rect: Rect, radius: f32) {
        if resp.has_focus() {
            ui.painter().rect_stroke(rect.expand(2.0), radius + 2.0, Stroke::new(2.0, p.accent), StrokeKind::Outside);
        }
    }

    /// Full-width filled action button (Start / Stop), with its shortcut on
    /// the right so it's discoverable without hovering.
    fn big_button(ui: &mut egui::Ui, p: &Palette, text: &str, fill: Color32, hover: Color32) -> Response {
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 46.0), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, text));
        let color = if resp.is_pointer_button_down_on() || resp.hovered() { hover } else { fill };
        let painter = ui.painter();
        painter.rect_filled(rect, 6.0, color);
        painter.text(rect.center(), Align2::CENTER_CENTER, text, semi(16.0), p.on_accent);
        painter.text(rect.right_center() - vec2(14.0, 0.0), Align2::RIGHT_CENTER, "Ctrl+Enter",
                     FontId::proportional(11.5), p.on_accent.gamma_multiply(0.7));
        Self::focus_ring(ui, p, &resp, rect, 6.0);
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    /// A full-row on/off switch: label on the left, switch on the right; the
    /// whole row is clickable (and Space toggles it once focused).
    fn switch(ui: &mut egui::Ui, p: &Palette, on: &mut bool, label: &str) -> Response {
        let (rect, mut resp) = ui.allocate_exact_size(vec2(ui.available_width(), 28.0), Sense::click());
        if resp.clicked() {
            *on = !*on;
            resp.mark_changed();
        }
        let value = *on;
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::CheckBox, true, value, label));
        let t = ui.ctx().animate_bool_responsive(resp.id, value);
        let track = Rect::from_min_size(pos2(rect.right() - 36.0, rect.center().y - 10.0), vec2(36.0, 20.0));
        let hot = resp.hovered();
        let off = if hot { p.muted } else { p.border };
        let on_fill = if hot { p.accent_hover } else { p.accent };
        let painter = ui.painter();
        painter.text(rect.left_center(), Align2::LEFT_CENTER, label, FontId::proportional(13.5), p.text);
        painter.rect_filled(track, 10.0, if value { on_fill } else { off });
        let x = egui::lerp(track.left() + 10.0..=track.right() - 10.0, t);
        painter.circle_filled(pos2(x, track.center().y), 7.0, Color32::WHITE);
        Self::focus_ring(ui, p, &resp, track, 10.0);
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    /// Full-width slider in the app's palette: a rail filled in the accent up to
    /// the value and a round knob. (egui's own draws its rail in the widget fill,
    /// which is the panel colour here, so only the knob showed.) Drag or click
    /// to set; arrow keys step it once focused.
    fn slider(ui: &mut egui::Ui, p: &Palette, value: &mut f32, lo: f32, hi: f32) -> Response {
        let knob = 8.0;
        let (rect, mut resp) = ui.allocate_exact_size(vec2(ui.available_width(), 2.0 * knob + 6.0), Sense::click_and_drag());
        let (x0, x1) = (rect.left() + knob, rect.right() - knob);
        let before = *value;
        if let Some(pos) = resp.interact_pointer_pos() {
            *value = lo + ((pos.x - x0) / (x1 - x0)).clamp(0.0, 1.0) * (hi - lo);
        }
        if resp.has_focus() {
            let step = (hi - lo) / 20.0;
            ui.input(|i| {
                if i.key_pressed(Key::ArrowLeft) || i.key_pressed(Key::ArrowDown) {
                    *value -= step;
                }
                if i.key_pressed(Key::ArrowRight) || i.key_pressed(Key::ArrowUp) {
                    *value += step;
                }
            });
        }
        *value = value.clamp(lo, hi);
        if *value != before {
            resp.mark_changed();
        }

        let hot = resp.hovered() || resp.dragged() || resp.has_focus();
        let cx = x0 + (*value - lo) / (hi - lo) * (x1 - x0);
        let cy = rect.center().y;
        let painter = ui.painter();
        let rail = Rect::from_min_max(pos2(x0, cy - 2.0), pos2(x1, cy + 2.0));
        painter.rect_filled(rail, 2.0, p.border);
        painter.rect_filled(Rect::from_min_max(rail.min, pos2(cx, rail.max.y)), 2.0, p.accent);
        let fill = if resp.dragged() { p.tile_sel } else { p.surface };
        painter.circle(pos2(cx, cy), if hot { knob } else { knob - 1.0 }, fill, Stroke::new(2.0, p.accent));
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    /// A selectable card: optional thumbnail above a label.
    fn tile(ui: &mut egui::Ui, p: &Palette, width: f32, tex: Option<&TextureHandle>, img: Vec2, label: &str,
            selected: bool) -> Response {
        let pad = 6.0;
        let label_h = 18.0;
        let img_h = if tex.is_some() { img.y + 4.0 } else { 0.0 };
        let (rect, resp) = ui.allocate_exact_size(vec2(width, pad * 2.0 + img_h + label_h), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::RadioButton, true, selected, label));
        let hovered = resp.hovered();
        let fill = if selected || hovered { p.tile_sel } else { p.tile };
        let stroke = if selected { Stroke::new(2.0, p.accent) } else { Stroke::new(1.0, if hovered { p.muted } else { p.border }) };
        let painter = ui.painter();
        painter.rect(rect, 6.0, fill, stroke, StrokeKind::Inside);
        if let Some(t) = tex {
            let r = Rect::from_center_size(pos2(rect.center().x, rect.top() + pad + img.y / 2.0), img);
            egui::Image::from_texture(t).corner_radius(3).paint_at(ui, r);
        }
        ui.painter().text(
            pos2(rect.center().x, rect.bottom() - pad - label_h / 2.0),
            Align2::CENTER_CENTER,
            label,
            if selected { semi(12.0) } else { FontId::proportional(12.0) },
            if selected { p.accent } else { p.text },
        );
        Self::focus_ring(ui, p, &resp, rect, 6.0);
        resp.on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    fn ui_sidebar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let p = self.pal();
        let full = ui.available_width();

        // Brand + virtual-camera status.
        ui.horizontal(|ui| {
            if let Some(logo) = &self.logo {
                ui.add(egui::Image::new(logo).fit_to_exact_size(vec2(32.0, 32.0)));
            }
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 0.0;
                ui.label(RichText::new("vcam").font(semi(20.0)));
                let (text, color) = match (&self.busy, self.vc.installed) {
                    (Some(b), _) => (b.clone(), p.muted),
                    (None, true) if self.worker.running() => ("● Live in Meet, Zoom & Teams".into(), p.danger),
                    (None, true) => ("● Ready for Meet, Zoom & Teams".into(), p.success),
                    (None, false) => ("Virtual camera not set up".into(), p.muted),
                };
                ui.label(RichText::new(text).size(12.0).color(color));
            });
        });

        // Setup notice (only while the virtual camera isn't usable yet).
        if !self.vc.installed {
            ui.add_space(12.0);
            egui::Frame::new()
                .fill(p.notice)
                .stroke(Stroke::new(1.0, p.notice_border))
                .corner_radius(6)
                .inner_margin(12)
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.label(RichText::new("ONE-TIME SETUP").font(semi(12.0)).color(p.warn));
                    let text = self.busy.clone().unwrap_or_else(|| {
                        if self.vc.obs_found {
                            "Let Meet, Zoom and Teams see vcam as a camera. Takes one click — Windows will \
                             ask for administrator permission."
                                .into()
                        } else {
                            "vcam uses the free OBS Studio virtual-camera driver. Install OBS Studio, then \
                             come back here. You can still preview without it."
                                .into()
                        }
                    });
                    ui.label(text);
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        let idle = self.busy.is_none();
                        if self.vc.obs_found {
                            if ui.add_enabled(idle, egui::Button::new("Set up virtual camera")).clicked() {
                                self.vcam_action(&ctx, true);
                            }
                        } else {
                            if ui.add_enabled(idle, egui::Button::new("Get OBS Studio")).clicked() {
                                system::open_url(OBS_URL);
                            }
                            if Self::link(ui, p, "Check again").clicked() {
                                self.refresh_vc_status();
                            }
                        }
                    });
                });
        }

        // Start / Stop.
        ui.add_space(14.0);
        let live = self.worker.running() || self.worker.status() == Status::Starting;
        let resp = if live {
            Self::big_button(ui, p, "Stop camera", p.danger, p.danger_hover)
        } else {
            Self::big_button(ui, p, "Start camera", p.accent, p.accent_hover)
        };
        if resp.clicked() {
            self.toggle(&ctx);
        }

        // Camera.
        Self::section(ui, p, "CAMERA");
        let before = self.cam;
        let selected = self.cameras.get(self.cam).cloned().unwrap_or_else(|| "No camera found".into());
        egui::ComboBox::from_id_salt("camera")
            .width(full)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                for (i, name) in self.cameras.iter().enumerate() {
                    ui.selectable_value(&mut self.cam, i, name);
                }
            })
            .response
            .on_hover_text("Using a laptop with an external webcam? Pick it here.");
        if self.cam != before {
            // Camera change -> the worker reopens the device (auto reconnect).
            self.worker.configure(|c| c.camera = self.cam);
            self.schedule_save();
        }
        ui.add_space(2.0);
        if Self::switch(ui, p, &mut self.mirror, "Mirror image")
            .on_hover_text("Flips the video horizontally — for everyone in the call, not just your preview.")
            .changed()
        {
            self.worker.configure(|c| c.mirror = self.mirror);
            self.schedule_save();
        }
        ui.add_space(6.0);
        self.ui_framing(ui, &ctx, CropTarget::Camera);
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label("Brightness").on_hover_text(
                "Darkens or brightens the picture's shadows and midtones; black and white stay as they are. \
                 Double-click the slider to reset.",
            );
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let pct = (self.brightness * 100.0).round();
                let text = if pct == 0.0 { "As captured".to_string() } else { format!("{pct:+.0}%") };
                ui.label(RichText::new(text).size(12.0).color(p.muted));
            });
        });
        let r = Self::slider(ui, p, &mut self.brightness, -1.0, 1.0);
        // Snap to "As captured" near the centre, so it's easy to land on
        // exactly no change.
        if r.double_clicked() || self.brightness.abs() < 0.04 {
            self.brightness = 0.0;
        }
        if r.changed() || r.double_clicked() {
            // Live: applied from the next frame, no reconnect.
            self.worker.configure(|c| c.brightness = self.brightness);
            self.schedule_save();
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Darker").size(12.0).color(p.muted));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(RichText::new("Brighter").size(12.0).color(p.muted));
            });
        });

        // Background.
        Self::section(ui, p, "BACKGROUND");
        // One compact row (like Meet's effects strip), so the picture gallery
        // below is visible without scrolling.
        let gap = 6.0;
        let tile_w = ((full - gap * 3.0) / 4.0).floor();
        let img = vec2(tile_w - 10.0, ((tile_w - 10.0) * 66.0 / 120.0).round());
        let mut picked = None;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = gap;
            for (i, &(mode, _, label)) in MODES.iter().enumerate() {
                let r = Self::tile(ui, p, tile_w, self.tile_tex.get(i), img, label, self.mode == mode);
                let r = r.on_hover_text(match mode {
                    BgMode::None => "Your real camera, no background effect.",
                    BgMode::Blur => "Blur the room behind you.",
                    BgMode::Image => "Replace the room with a picture.",
                    BgMode::Color => "Green screen: solid green, for apps (like OBS) that key it out.",
                });
                if r.clicked() {
                    picked = Some(mode);
                }
            }
        });
        if let Some(m) = picked {
            self.mode = m;
            self.apply_bg(&ctx);
        }

        // The picture options go above the gallery, which can grow long.
        if self.mode == BgMode::Image {
            ui.add_space(6.0);
            if Self::switch(ui, p, &mut self.realistic, "Match my room (realistic)")
                .on_hover_text("Matches the picture to your camera: your room's light and colour, lens focus, \
                                webcam grain, and light spilling onto your edges.")
                .changed()
            {
                self.worker.set_realistic(self.realistic);
                self.schedule_save();
            }
            if Self::switch(ui, p, &mut self.blur_image, "Soften image")
                .on_hover_text("An out-of-focus look, as if the picture were behind you.")
                .changed()
            {
                self.apply_bg(&ctx);
            }
            ui.add_space(6.0);
            if let Some(path) = self.bg_path.clone() {
                self.ui_framing(ui, &ctx, CropTarget::Picture(path));
            }
        }
        if self.mode == BgMode::Blur || (self.mode == BgMode::Image && self.blur_image) {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label("Blur amount");
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let pct = (self.blur - BLUR_MIN) / (BLUR_MAX - BLUR_MIN) * 100.0;
                    ui.label(RichText::new(format!("{pct:.0}%")).size(12.0).color(p.muted));
                });
            });
            let r = Self::slider(ui, p, &mut self.blur, BLUR_MIN, BLUR_MAX);
            if r.changed() {
                // Live: rebuilds the compositor with the new sigma, no reconnect.
                self.worker.set_blur_strength(self.blur);
                self.schedule_save();
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("Subtle").size(12.0).color(p.muted));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(RichText::new("Strong").size(12.0).color(p.muted));
                });
            });
        }
        if self.mode == BgMode::Image {
            ui.add_space(10.0);
            self.ui_gallery(ui, &ctx);
        }

        // Advanced (collapsible).
        ui.add_space(14.0);
        ui.separator();
        let arrow = if self.adv_open { "▾  Advanced settings" } else { "▸  Advanced settings" };
        if Self::link(ui, p, arrow).clicked() {
            self.adv_open = !self.adv_open;
            self.schedule_save();
        }
        if self.adv_open {
            self.ui_advanced(ui, &ctx);
        }
    }

    /// Fill / Fit / Stretch tiles; picks one, or None.
    fn fit_tiles(ui: &mut egui::Ui, p: &Palette, current: FitMode) -> Option<FitMode> {
        let full = ui.available_width();
        let mut picked = None;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for &(fit, label, tip) in &FITS {
                let r = Self::tile(ui, p, (full - 12.0) / 3.0, None, Vec2::ZERO, label, current == fit);
                if r.on_hover_text(tip).clicked() && current != fit {
                    picked = Some(fit);
                }
            }
        });
        picked
    }

    /// How the camera / picture is fitted, and the way into its crop editor.
    fn ui_framing(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, target: CropTarget) {
        let p = self.pal();
        let f = self.framing_of(&target);
        let camera = target == CropTarget::Camera;
        let mut open = false;
        ui.horizontal(|ui| {
            ui.label(if camera { "Camera framing" } else { "Framing" });
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let text = if f.crop == Framing::WHOLE { "Crop…" } else { "Edit crop…" };
                open = Self::link(ui, p, text)
                    .on_hover_text(if camera {
                        "Choose the part of the camera picture to use (a digital zoom); only that part is \
                         processed. Move, resize or zoom the crop."
                    } else {
                        "Choose the part of the picture to use: move, resize or zoom the crop."
                    })
                    .clicked();
            });
        });
        if let Some(fit) = Self::fit_tiles(ui, p, f.fit) {
            self.set_framing_of(ctx, &target, Framing { fit, ..f }, false);
        }
        if open && self.crop_edit.as_ref().is_none_or(|e| e.target != target) {
            self.crop_edit = Some(match target {
                CropTarget::Camera => CropEditor::camera(),
                CropTarget::Picture(path) => CropEditor::picture(ctx, path),
            });
        }
    }

    /// The crop editor window: the whole picture (or live camera picture)
    /// with the crop on top. Every change shows live in the preview.
    fn ui_crop_editor(&mut self, ctx: &egui::Context) {
        self.worker.want_raw(self.crop_edit.as_ref().is_some_and(|e| e.target == CropTarget::Camera));
        let Some(mut ed) = self.crop_edit.take() else { return };
        match &ed.target {
            // A picture editor follows the chosen picture, and closes when
            // there's none.
            CropTarget::Picture(shown) => match self.bg_path.clone() {
                Some(path) if self.mode == BgMode::Image => {
                    if path != *shown {
                        ed = CropEditor::picture(ctx, path);
                    }
                }
                _ => return,
            },
            CropTarget::Camera => {
                let raw = self.worker.latest_raw();
                if let Some(frame) = raw.as_ref().filter(|r| !ed.shown.as_ref().is_some_and(|s| Arc::ptr_eq(s, r))) {
                    let k = (EDITOR_PX as f32 / 1.6 / frame.w.max(frame.h) as f32).min(1.0);
                    let (w, h) = (((frame.w as f32 * k) as usize).max(1), ((frame.h as f32 * k) as usize).max(1));
                    let img = egui::ColorImage::from_rgb([w, h], &imgproc::make_preview(frame, w, h));
                    match &mut ed.tex {
                        Some(t) => t.set(img, TextureOptions::LINEAR),
                        None => ed.tex = Some(texture(ctx, "crop-camera", img)),
                    }
                }
                if raw.is_none() {
                    ed.tex = None;
                }
                ed.shown = raw;
            }
        }
        if let Some(rx) = &ed.job {
            match rx.try_recv() {
                Ok(Some(img)) => {
                    ed.tex = Some(texture(ctx, "crop-editor", img));
                    ed.job = None;
                }
                Ok(None) | Err(mpsc::TryRecvError::Disconnected) => return, // unreadable picture
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        let p = self.pal();
        let aspect = self.frame_aspect();
        let mut f = self.framing_of(&ed.target);
        let camera = ed.target == CropTarget::Camera;
        let mut changed: Option<bool> = None; // Some(still dragging)
        let mut open = !ctx.input(|i| i.key_pressed(Key::Escape));
        let mut done = false;
        egui::Window::new(if camera { "Crop camera" } else { "Crop & position" })
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .pivot(Align2::CENTER_CENTER)
            .default_pos(ctx.content_rect().center())
            .show(ctx, |ui| {
                let box_size = vec2(560.0, 380.0);
                ui.set_width(box_size.x);
                if let Some(fit) = Self::fit_tiles(ui, p, f.fit) {
                    f.fit = fit;
                    changed = Some(false);
                }
                ui.add_space(6.0);
                let Some(tex) = ed.tex.clone() else {
                    let size = vec2(box_size.x, box_size.y * 0.8);
                    if camera {
                        ui.add_sized(size, egui::Label::new(RichText::new("Start the camera to see it here.").color(p.muted)));
                    } else {
                        ui.add_sized(size, egui::Spinner::new());
                    }
                    return;
                };
                let [iw, ih] = tex.size();
                let k = (box_size.x / iw as f32).min(box_size.y / ih as f32);
                let size = vec2(iw as f32 * k, ih as f32 * k);
                // Fill keeps the frame's shape (as a w / h of picture fractions).
                let lock = (f.fit == FitMode::Fill).then(|| aspect * ih as f32 / iw as f32);
                let crop = f.effective_crop(iw, ih, aspect);

                let (rect, resp) = ui
                    .vertical_centered(|ui| ui.allocate_exact_size(size, Sense::click_and_drag()))
                    .inner;
                let to_screen = |c: [f32; 4]| {
                    Rect::from_min_max(rect.min + vec2(c[0], c[1]) * size, rect.min + vec2(c[2], c[3]) * size)
                };
                let cr = to_screen(crop);
                // A new crop; `dragging`: more follow at once.
                let set = |f: &mut Framing, changed: &mut Option<bool>, c: [f32; 4], dragging: bool| {
                    if c != f.crop || !dragging {
                        f.crop = c;
                        *changed = Some(dragging);
                    }
                };

                // Drag a grip; pressing outside the crop first brings it there.
                if resp.drag_started() {
                    if let Some(pos) = resp.interact_pointer_pos() {
                        ed.drag = Some(match grip_at(cr, pos, lock.is_some()) {
                            Some(g) => (g, crop, pos),
                            None => (Grip::MOVE, dragged_crop(crop, Grip::MOVE, (pos - cr.center()) / size, lock), pos),
                        });
                    }
                }
                if let (Some((grip, start, from)), Some(pos)) = (ed.drag, resp.interact_pointer_pos()) {
                    if resp.dragged() {
                        set(&mut f, &mut changed, dragged_crop(start, grip, (pos - from) / size, lock), true);
                    }
                }
                if resp.drag_stopped() {
                    ed.drag = None;
                    let c = f.crop;
                    set(&mut f, &mut changed, c, false);
                }
                if resp.clicked() {
                    if let Some(pos) = resp.interact_pointer_pos().filter(|&pos| grip_at(cr, pos, lock.is_some()).is_none()) {
                        set(&mut f, &mut changed, dragged_crop(crop, Grip::MOVE, (pos - cr.center()) / size, lock), false);
                    }
                }
                if resp.hovered() {
                    let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        set(&mut f, &mut changed, zoomed_crop(crop, (-scroll * 0.002).exp(), lock), true);
                    }
                }
                let hover = resp.hover_pos().and_then(|pos| grip_at(cr, pos, lock.is_some()));
                if let Some(g) = ed.drag.map(|d| d.0).or(hover) {
                    ctx.set_cursor_icon(g.cursor(ed.drag.is_some()));
                }

                // The picture, dimmed outside the crop, with a rule-of-thirds grid.
                let cr = to_screen(f.effective_crop(iw, ih, aspect)); // as changed just now
                egui::Image::from_texture(&tex).paint_at(ui, rect);
                let painter = ui.painter();
                let shade = Color32::from_black_alpha(150);
                for r in [
                    Rect::from_min_max(rect.min, pos2(rect.right(), cr.top())),
                    Rect::from_min_max(pos2(rect.left(), cr.bottom()), rect.max),
                    Rect::from_min_max(pos2(rect.left(), cr.top()), pos2(cr.left(), cr.bottom())),
                    Rect::from_min_max(pos2(cr.right(), cr.top()), pos2(rect.right(), cr.bottom())),
                ] {
                    painter.rect_filled(r, 0.0, shade);
                }
                let faint = Stroke::new(1.0, Color32::from_white_alpha(80));
                for t in [1.0 / 3.0, 2.0 / 3.0] {
                    let (x, y) = (cr.left() + cr.width() * t, cr.top() + cr.height() * t);
                    painter.line_segment([pos2(x, cr.top()), pos2(x, cr.bottom())], faint);
                    painter.line_segment([pos2(cr.left(), y), pos2(cr.right(), y)], faint);
                }
                painter.rect_stroke(cr, 0.0, Stroke::new(1.5, Color32::WHITE), StrokeKind::Inside);
                let mut grips = vec![cr.left_top(), cr.right_top(), cr.left_bottom(), cr.right_bottom()];
                if lock.is_none() {
                    grips.extend([cr.center_top(), cr.center_bottom(), cr.left_center(), cr.right_center()]);
                }
                for g in grips {
                    painter.rect(Rect::from_center_size(g, vec2(9.0, 9.0)), 1.5, Color32::WHITE,
                                 Stroke::new(1.0, Color32::from_black_alpha(160)), StrokeKind::Middle);
                }
                ui.add_space(4.0);
                ui.label(
                    RichText::new(if lock.is_some() {
                        "Drag to move, drag a corner to resize, scroll to zoom. Fill keeps the frame's shape."
                    } else {
                        "Drag to move, drag a corner or edge to resize, scroll to zoom."
                    })
                    .size(12.0)
                    .color(p.muted),
                );

                ui.add_space(6.0);
                let crop = f.effective_crop(iw, ih, aspect);
                ui.horizontal(|ui| {
                    ui.label("Align");
                    for (label, x, y) in [
                        ("Left", Some(-1), None),
                        ("Top", None, Some(-1)),
                        ("Centre", Some(0), Some(0)),
                        ("Bottom", None, Some(1)),
                        ("Right", Some(1), None),
                    ] {
                        if ui.button(label).clicked() {
                            set(&mut f, &mut changed, aligned_crop(crop, x, y), false);
                        }
                    }
                    ui.separator();
                    ui.label("Zoom");
                    if ui.button(" − ").on_hover_text("Zoom out").clicked() {
                        set(&mut f, &mut changed, zoomed_crop(crop, 1.25, lock), false);
                    }
                    if ui.button(" + ").on_hover_text("Zoom in").clicked() {
                        set(&mut f, &mut changed, zoomed_crop(crop, 0.8, lock), false);
                    }
                });
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    if ui.button("Whole picture").on_hover_text("Undo the crop").clicked() {
                        set(&mut f, &mut changed, Framing::WHOLE, false);
                    }
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        done = ui.button("  Done  ").clicked();
                    });
                });
            });
        if let Some(dragging) = changed {
            self.set_framing_of(ctx, &ed.target, f, dragging);
        }
        if open && !done {
            self.crop_edit = Some(ed);
        } else if camera {
            self.worker.want_raw(false);
        } else if self.tile_stale {
            self.tile_stale = false;
            self.render_thumbs(ctx);
        }
    }

    fn ui_gallery(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let p = self.pal();
        let cols = 3;
        let gap = 4.0;
        let cell_w = (ui.available_width() - gap * (cols as f32 - 1.0)) / cols as f32;
        let cell = vec2(cell_w, cell_w * CELL_PX.1 as f32 / CELL_PX.0 as f32);
        let sel = self.bg_path.clone();
        let (mut select, mut remove, mut add) = (None, None, false);
        let n = self.gallery.len() + 1;
        for row in 0..n.div_ceil(cols) {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = gap;
                for i in row * cols..(row * cols + cols).min(n) {
                    let (rect, resp) = ui.allocate_exact_size(cell, Sense::click());
                    let resp = resp.on_hover_cursor(egui::CursorIcon::PointingHand);
                    let Some(item) = self.gallery.get(i) else {
                        // The trailing "+ Add" cell.
                        resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, "Add pictures"));
                        let hot = resp.hovered();
                        let painter = ui.painter();
                        painter.rect(rect, 4.0, if hot { p.tile_sel } else { p.tile },
                                     Stroke::new(1.0, if hot { p.accent } else { p.border }), StrokeKind::Inside);
                        painter.text(rect.center() - vec2(0.0, 7.0), Align2::CENTER_CENTER, "+", semi(18.0), p.accent);
                        painter.text(rect.center() + vec2(0.0, 11.0), Align2::CENTER_CENTER, "Add", semi(11.5), p.accent);
                        Self::focus_ring(ui, p, &resp, rect, 4.0);
                        if resp.on_hover_text("Add your own pictures (Ctrl+O), or drag them onto this window").clicked() {
                            add = true;
                        }
                        continue;
                    };
                    let selected = sel.as_deref().is_some_and(|s| library::same_file(s, &item.path));
                    let name = if item.removable {
                        item.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                    } else {
                        "Built-in background".into()
                    };
                    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::RadioButton, true, selected, &name));
                    egui::Image::from_texture(&item.tex).corner_radius(4).paint_at(ui, rect);
                    let painter = ui.painter();
                    if selected {
                        painter.rect_stroke(rect, 4.0, Stroke::new(2.5, p.accent), StrokeKind::Inside);
                        // A check mark, so the choice reads without relying on colour.
                        let c = rect.left_top() + vec2(11.0, 11.0);
                        painter.circle(c, 7.5, p.accent, Stroke::new(1.5, Color32::WHITE));
                        painter.line(vec![c + vec2(-3.5, 0.2), c + vec2(-1.0, 2.7), c + vec2(3.5, -2.3)],
                                     Stroke::new(1.6, Color32::WHITE));
                    } else if resp.hovered() {
                        painter.rect_stroke(rect, 4.0, Stroke::new(2.0, p.muted), StrokeKind::Inside);
                    }
                    Self::focus_ring(ui, p, &resp, rect, 4.0);
                    // The remove badge only shows on the picture you're pointing
                    // at (Delete also removes the focused one).
                    let badge = Rect::from_min_size(pos2(rect.right() - 21.0, rect.top() + 4.0), vec2(17.0, 17.0));
                    let over_badge = item.removable && resp.hover_pos().is_some_and(|pos| badge.contains(pos));
                    if item.removable && resp.hovered() {
                        let fill = if over_badge { p.danger } else { Color32::from_black_alpha(190) };
                        painter.rect_filled(badge, 4.0, fill);
                        painter.text(badge.center(), Align2::CENTER_CENTER, "✕", FontId::proportional(11.0), Color32::WHITE);
                    }
                    if item.removable && resp.has_focus() && ui.input(|i| i.key_pressed(Key::Delete)) {
                        remove = Some(item.path.clone());
                    }
                    let resp = resp.on_hover_text(if over_badge { "Remove from your backgrounds".to_string() } else { name });
                    if resp.clicked() {
                        if over_badge {
                            remove = Some(item.path.clone());
                        } else {
                            select = Some(item.path.clone());
                        }
                    }
                }
            });
        }
        if let Some(path) = select {
            self.select_image(ctx, path);
        }
        if let Some(path) = remove {
            self.remove_image(ctx, &path);
        }
        if add {
            self.add_images(ctx);
        }
    }

    fn ui_advanced(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let p = self.pal();
        let full = ui.available_width();
        ui.add_space(4.0);
        ui.label("Performance");
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for (i, m) in SPEED_MODES.iter().enumerate() {
                if Self::tile(ui, p, (full - 6.0) / 2.0, None, Vec2::ZERO, m.0, self.speed == i).clicked() && self.speed != i {
                    // Capture/output size change -> the worker reconnects automatically.
                    self.speed = i;
                    let (_, cw, ch, ow, oh) = SPEED_MODES[i];
                    self.worker.configure(|c| (c.width, c.height, c.out_width, c.out_height) = (cw, ch, ow, oh));
                    self.schedule_save();
                }
            }
        });
        ui.label(
            RichText::new("Sharp: full 720p detail. Smooth: steadier motion on slower computers or webcams.")
                .size(12.0)
                .color(p.muted),
        );

        ui.add_space(8.0);
        ui.label("Edge detail");
        let before = self.quality;
        egui::ComboBox::from_id_salt("quality")
            .width(full)
            .selected_text(QUALITY_PRESETS[self.quality].0)
            .show_ui(ui, |ui| {
                for (i, q) in QUALITY_PRESETS.iter().enumerate() {
                    ui.selectable_value(&mut self.quality, i, q.0);
                }
            })
            .response
            .on_hover_text("Finer = crisper hair and edges but more CPU. Applies instantly.");
        if self.quality != before {
            // Live quality<->speed balance; takes effect on the next frame.
            self.worker.set_quality(QUALITY_PRESETS[self.quality].1);
            self.schedule_save();
        }

        ui.add_space(6.0);
        if Self::switch(ui, p, &mut self.one_person, "Only show one person")
            .on_hover_text("Hides people walking behind you. Turn off to keep everyone in frame.")
            .changed()
        {
            self.worker.configure(|c| c.multi_person = !self.one_person);
            self.schedule_save();
        }
        if Self::switch(ui, p, &mut self.keep_frame_rate, "Keep full frame rate in dim light")
            .on_hover_text(
                "Many webcams slow to ~20 fps indoors to brighten the picture. When yours does, \
                 vcam fixes its exposure to hold the full frame rate and brightens the picture itself \
                 to match. Turn off to leave the exposure entirely to the camera.",
            )
            .changed()
        {
            // Reopens the camera with the new setting.
            self.worker.configure(|c| c.exposure = exposure_mode(self.keep_frame_rate));
            self.schedule_save();
        }

        ui.add_space(8.0);
        ui.label("Virtual camera driver");
        ui.horizontal(|ui| {
            let idle = self.busy.is_none();
            if ui.add_enabled(idle && self.vc.obs_found, egui::Button::new("Reinstall")).clicked() {
                self.vcam_action(ctx, true);
            }
            if ui.add_enabled(idle && self.vc.installed, egui::Button::new("Uninstall")).clicked() {
                self.vcam_action(ctx, false);
            }
        });
        ui.add_space(8.0);
        ui.label(
            RichText::new("Changes apply instantly. Changing performance or camera briefly reconnects the camera.")
                .size(12.0)
                .color(p.muted),
        );
    }
}

impl eframe::App for App {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // While minimized nobody sees the preview, so ask for none (saves the
        // CPU for matting). `ui` isn't called then.
        if ctx.input(|i| i.viewport().minimized.unwrap_or(false)) {
            self.worker.set_preview_box(None);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let dark = ctx.theme() == Theme::Dark;
        if dark != self.dark {
            self.dark = dark;
            self.render_thumbs(&ctx); // the image placeholder follows the theme
        }
        self.housekeeping(&ctx);
        let p = self.pal();

        if self.side_open {
            egui::Panel::right("sidebar")
                .resizable(false)
                .exact_size(346.0)
                .show_separator_line(false)
                .frame(egui::Frame::new().fill(p.bg).inner_margin(Margin { left: 0, right: 16, top: 16, bottom: 16 }))
                .show(ui, |ui| {
                    egui::Frame::new()
                        .fill(p.surface)
                        .stroke(Stroke::new(1.0, p.border))
                        .corner_radius(8)
                        .show(ui, |ui| {
                            ui.set_min_size(ui.available_size());
                            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                                egui::Frame::new().inner_margin(Margin::symmetric(18, 16)).show(ui, |ui| {
                                    self.ui_sidebar(ui);
                                });
                            });
                        });
                });
        }

        let right = if self.side_open { 8 } else { 16 };
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(p.bg).inner_margin(Margin { left: 16, right, top: 16, bottom: 8 }))
            .show(ui, |ui| {
                let footer_h = 24.0;
                let avail = ui.available_rect_before_wrap();
                let prect = Rect::from_min_max(avail.min, pos2(avail.max.x, avail.max.y - footer_h - 8.0));
                ui.allocate_rect(prect, Sense::hover());
                self.paint_preview(ui, prect);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label(RichText::new("In Meet, Zoom or Teams, pick ").size(12.0).color(p.muted));
                    ui.label(RichText::new(virtualcam::DEVICE_NAME).font(semi(12.0)).color(p.text))
                        .on_hover_text("Video settings → Camera. Leave vcam running while you're in the call.");
                    ui.label(RichText::new(" as your camera.").size(12.0).color(p.muted));
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let text = if self.side_open { "Hide controls  ▸" } else { "◂  Show controls" };
                        if Self::link(ui, p, text)
                            .on_hover_text("Ctrl+B  ·  Ctrl+Enter still starts / stops the camera")
                            .clicked()
                        {
                            self.side_open = !self.side_open;
                            self.schedule_save();
                        }
                    });
                });
            });

        self.ui_crop_editor(&ctx);
        // The picture tile catches up once a crop drag / zoom settles.
        if self.tile_stale
            && self.crop_edit.as_ref().is_none_or(|e| e.drag.is_none())
            && ctx.input(|i| i.smooth_scroll_delta == Vec2::ZERO)
        {
            self.tile_stale = false;
            self.render_thumbs(&ctx);
        }

        // Pictures dragged over the window: say what dropping them does.
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::unique("drop")));
            let r = ctx.content_rect().shrink(10.0);
            painter.rect(r, 10.0, p.bg.gamma_multiply(0.92), Stroke::new(2.0, p.accent), StrokeKind::Inside);
            painter.text(r.center() - vec2(0.0, 12.0), Align2::CENTER_CENTER, "Drop to add as a background",
                         semi(20.0), p.text);
            painter.text(r.center() + vec2(0.0, 16.0), Align2::CENTER_CENTER, "PNG, JPG, BMP or WebP",
                         FontId::proportional(13.0), p.muted);
        }

        // Poll the worker's status (fps, state) a few times a second; new
        // preview frames request their own repaint.
        let fast = self.worker.status() == Status::Starting || self.busy.is_some() || self.save_at.is_some();
        ctx.request_repaint_after(Duration::from_millis(if fast { 100 } else { 250 }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inside(c: [f32; 4]) -> bool {
        c[0] >= 0.0 && c[1] >= 0.0 && c[2] <= 1.0 + 1e-6 && c[3] <= 1.0 + 1e-6 && c[0] < c[2] && c[1] < c[3]
    }

    #[test]
    fn locked_drag_keeps_shape_and_stays_inside() {
        let a = 1.5;
        let start = [0.2, 0.2, 0.5, 0.4];
        for d in [vec2(0.9, 0.1), vec2(-0.9, -0.9), vec2(0.05, 0.6), vec2(-0.3, 0.0)] {
            for grip in [Grip { x: 1, y: 1 }, Grip { x: -1, y: -1 }, Grip { x: 1, y: -1 }, Grip { x: -1, y: 1 }] {
                let c = dragged_crop(start, grip, d, Some(a));
                assert!(inside(c), "{c:?}");
                assert!(((c[2] - c[0]) / (c[3] - c[1]) - a).abs() < 1e-3, "{c:?}");
            }
        }
    }

    #[test]
    fn moving_and_zooming_stay_inside() {
        let c = dragged_crop([0.2, 0.2, 0.5, 0.4], Grip::MOVE, vec2(5.0, -5.0), None);
        assert!(inside(c) && (c[2], c[1]) == (1.0, 0.0));
        let z = zoomed_crop([0.0, 0.0, 0.5, 0.5], 10.0, None);
        assert!(inside(z) && z[2] - z[0] <= 1.0);
        let z = zoomed_crop([0.4, 0.4, 0.6, 0.6], 0.001, Some(1.0));
        assert!(z[2] - z[0] >= Framing::MIN_SIDE - 1e-6);
        let a = aligned_crop([0.2, 0.2, 0.5, 0.4], Some(1), Some(-1));
        assert!(a.iter().zip([0.7, 0.0, 1.0, 0.2]).all(|(a, b)| (a - b).abs() < 1e-6), "{a:?}");
    }

    #[test]
    fn grips() {
        let r = Rect::from_min_max(pos2(100.0, 100.0), pos2(300.0, 200.0));
        assert_eq!(grip_at(r, pos2(101.0, 199.0), false), Some(Grip { x: -1, y: 1 }));
        assert_eq!(grip_at(r, pos2(200.0, 101.0), false), Some(Grip { x: 0, y: -1 }));
        assert_eq!(grip_at(r, pos2(200.0, 101.0), true), Some(Grip::MOVE));
        assert_eq!(grip_at(r, pos2(50.0, 50.0), false), None);
    }
}

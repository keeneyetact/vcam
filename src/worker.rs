//! The capture -> matte -> composite -> virtual-camera loop on its own thread,
//! so a GUI (or the CLI) can start/stop it and change options live.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use anyhow::Result;

use crate::autotune::{self, AutoQuality};
use crate::capture::{CameraInfo, ExposureMode, FrameGrabber};
use crate::compositor::Compositor;
use crate::imgproc::{self, Frame, Framing};
use crate::{hw, kernels};
use crate::pipeline::OutputPipeline;
use crate::rvm::RvmEngine;
use crate::virtualcam::VirtualCam;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BgMode {
    /// "Original": the camera frame untouched.
    None,
    Blur,
    Image,
    Color,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtualCamUse {
    /// Fail the session if the virtual camera can't be opened (GUI).
    Required,
    /// Warn and carry on preview-only (CLI).
    Optional,
    Off,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub camera: usize,
    /// Capture size (what the camera is asked for).
    pub width: usize,
    pub height: usize,
    /// Output size; None == the capture size. Capture small/fast and upscale
    /// the finished frame for a smooth 30 fps at 720p.
    pub out_width: Option<usize>,
    pub out_height: Option<usize>,
    pub fps: u32,
    pub mode: BgMode,
    pub bg_path: Option<PathBuf>,
    /// Which part of the picture is used, and how it's fitted.
    pub framing: Framing,
    /// Which part of the camera picture is processed (a digital zoom), and
    /// how it's fitted to the frame.
    pub camera_framing: Framing,
    /// Solid colour background, BGR.
    pub color: [u8; 3],
    /// Soften the picture background (out-of-focus look).
    pub blur_image: bool,
    /// Blur sigma multiplier for Blur mode and `blur_image`; 1.0 is the
    /// original fixed amount.
    pub blur_strength: f32,
    /// Match the picture background to the live camera (light, focus,
    /// grain, light wrap) so it reads as the real room.
    pub realistic: bool,
    /// Draw a line around the camera picture in the output (picture
    /// background only).
    pub outline: bool,
    pub mirror: bool,
    pub multi_person: bool,
    /// RVM downsample_ratio; None == automatic (`autotune`: the finest this
    /// PC sustains at the camera's rate).
    pub downsample: Option<f32>,
    pub threads: Option<usize>,
    /// How the webcam's exposure is handled; auto-exposure alone slows many
    /// webcams to ~20 fps in dim light.
    pub exposure: ExposureMode,
    /// Darken or brighten the camera picture, -1 to 1 (0 == as captured);
    /// e.g. to make up for the light a fixed exposure gives up.
    pub brightness: f32,
    pub virtualcam: VirtualCamUse,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera: 0,
            width: 1280,
            height: 720,
            out_width: None,
            out_height: None,
            fps: 30,
            mode: BgMode::Image,
            bg_path: None,
            framing: Framing::default(),
            camera_framing: Framing::default(),
            color: [64, 177, 0],
            blur_image: false,
            blur_strength: 1.0,
            realistic: true,
            outline: false,
            mirror: false,
            multi_person: false,
            downsample: None,
            threads: None,
            exposure: ExposureMode::Adaptive,
            brightness: 0.0,
            virtualcam: VirtualCamUse::Required,
        }
    }
}

impl Config {
    /// Options that need the capture/engine/virtual-cam pipeline rebuilt;
    /// everything else (mirror, brightness, downsample, one-person, background) applies
    /// live on the next frame.
    fn restart_key(&self) -> impl PartialEq {
        (self.camera, self.width, self.height, self.out_width, self.out_height, self.fps, self.threads, self.exposure,
         self.virtualcam)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Stopped,
    Starting,
    Running,
    Error,
}

/// The latest output frame fitted into the GUI's preview box, as RGB.
pub struct Preview {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<u8>,
}

/// How long the webcam keeps running after the last app closes the virtual
/// camera (and after Start, if no app has it open by then). Then it pauses:
/// webcam off, nothing computed, until an app opens the camera again.
const LINGER: Duration = Duration::from_millis(2500);
/// While paused: how often to look for an app opening the virtual camera.
const WATCH_EVERY: Duration = Duration::from_millis(100);
/// Repeat the last frame after this long without one (well under the ~330 ms
/// an app tolerates).
const KEEP_FED_GAP: Duration = Duration::from_millis(100);
/// How long a webcam that fails while an app wants the output is retried, and
/// how often (see `retry_webcam`).
const WEBCAM_RETRY_FOR: Duration = Duration::from_secs(10);
const WEBCAM_RETRY_EVERY: Duration = Duration::from_millis(500);

type SourceKey = (PathBuf, Option<SystemTime>);
type BgKey = (SourceKey, usize, usize, Framing);

/// Decoded pictures are kept at most this big, and this many of them: a
/// re-crop is then just crop + resize (a few ms), never a decode.
const SOURCE_MAX_SIDE: usize = 2560;
const SOURCES_KEPT: usize = 3;

#[derive(Default)]
struct BgState {
    dirty: bool,
    /// Bumps per background change; stale async loads are dropped.
    gen: u64,
    /// A loader thread is building a compositor. One at a time: changes made
    /// meanwhile (e.g. dragging the crop) stay `dirty` and coalesce into the
    /// next build.
    loading: bool,
    /// Compositor finished by a loader thread.
    pending: Option<Arc<Compositor>>,
    /// Decoded pictures, most recent last.
    sources: Vec<(SourceKey, Arc<Frame>)>,
    /// Fitted backgrounds, most recent last (re-selecting one is instant).
    cache: Vec<(BgKey, Arc<Frame>)>,
}

struct Info {
    status: Status,
    running: bool,
    error: Option<String>,
    fps: f32,
    /// The downsample_ratio in use (Auto's current pick), while running.
    quality: Option<f32>,
    device: Option<String>,
    camera: Option<CameraInfo>,
    /// Running, but paused (webcam closed, nothing computed) until an app
    /// opens the virtual camera.
    waiting: bool,
}

struct Shared {
    cfg: Mutex<Config>,
    bg: Mutex<BgState>,
    info: Mutex<Info>,
    stop: AtomicBool,
    restart: AtomicBool,
    /// (w, h) the GUI wants the preview to fit, or None for no preview (e.g.
    /// while minimized).
    preview_box: Mutex<Option<(usize, usize)>>,
    latest: Mutex<(Option<Arc<Frame>>, Option<Arc<Preview>>)>,
    /// The camera picture before its crop, kept only while `want_raw` (the
    /// GUI's camera crop editor shows it).
    want_raw: AtomicBool,
    latest_raw: Mutex<Option<Arc<Frame>>>,
    on_preview: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
}

pub struct VCamWorker {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Default for VCamWorker {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl VCamWorker {
    pub fn new(cfg: Config) -> Self {
        let shared = Shared {
            cfg: Mutex::new(cfg),
            bg: Mutex::new(BgState::default()),
            info: Mutex::new(Info {
                status: Status::Stopped,
                running: false,
                error: None,
                fps: 0.0,
                quality: None,
                device: None,
                camera: None,
                waiting: false,
            }),
            stop: AtomicBool::new(false),
            restart: AtomicBool::new(false),
            preview_box: Mutex::new(None),
            latest: Mutex::new((None, None)),
            want_raw: AtomicBool::new(false),
            latest_raw: Mutex::new(None),
            on_preview: Mutex::new(None),
        };
        Self { shared: Arc::new(shared), thread: Mutex::new(None) }
    }

    // ---- public, thread-safe ----

    pub fn config(&self) -> Config {
        self.shared.cfg.lock().unwrap().clone()
    }

    /// Change options; a pipeline-shape change while live reconnects
    /// automatically (no manual OFF/ON).
    pub fn configure(&self, f: impl FnOnce(&mut Config)) {
        let restart = {
            let mut cfg = self.shared.cfg.lock().unwrap();
            let before = cfg.restart_key();
            f(&mut cfg);
            before != cfg.restart_key()
        };
        if restart && self.running() {
            self.shared.restart.store(true, Ordering::Relaxed);
        }
    }

    pub fn set_background(&self, mode: BgMode, path: Option<PathBuf>, framing: Framing, blur_image: Option<bool>) {
        self.configure(|c| {
            c.mode = mode;
            if path.is_some() {
                c.bg_path = path;
            }
            c.framing = framing;
            if let Some(b) = blur_image {
                c.blur_image = b;
            }
        });
        self.shared.bg.lock().unwrap().dirty = true;
    }

    /// Crop / fit of the picture, live (coalesced while it's being dragged).
    pub fn set_framing(&self, framing: Framing) {
        self.configure(|c| c.framing = framing);
        self.shared.bg.lock().unwrap().dirty = true;
    }

    /// Blur intensity for both Blur mode and a softened picture. Rebuilds the
    /// compositor (the sigma is baked into it), off the frame loop.
    pub fn set_blur_strength(&self, strength: f32) {
        self.configure(|c| c.blur_strength = strength);
        self.shared.bg.lock().unwrap().dirty = true;
    }

    /// Realistic picture matching on/off; rebuilds the compositor off the
    /// frame loop.
    pub fn set_realistic(&self, on: bool) {
        self.configure(|c| c.realistic = on);
        self.shared.bg.lock().unwrap().dirty = true;
    }

    /// Live quality/speed balance (RVM downsample_ratio, None == automatic),
    /// from the next frame.
    pub fn set_quality(&self, downsample: Option<f32>) {
        self.configure(|c| c.downsample = downsample);
    }

    pub fn status(&self) -> Status {
        self.shared.info.lock().unwrap().status
    }

    pub fn running(&self) -> bool {
        self.shared.info.lock().unwrap().running
    }

    pub fn error(&self) -> Option<String> {
        self.shared.info.lock().unwrap().error.clone()
    }

    pub fn fps(&self) -> f32 {
        self.shared.info.lock().unwrap().fps
    }

    /// The downsample_ratio in use, while running.
    pub fn quality(&self) -> Option<f32> {
        self.shared.info.lock().unwrap().quality
    }

    /// Name of the virtual camera being fed, while running.
    pub fn device(&self) -> Option<String> {
        self.shared.info.lock().unwrap().device.clone()
    }

    /// Running but paused until an app (Meet, Zoom, Teams...) opens the
    /// virtual camera: the webcam is off and nothing is computed.
    pub fn waiting(&self) -> bool {
        self.shared.info.lock().unwrap().waiting
    }

    pub fn camera_info(&self) -> Option<CameraInfo> {
        self.shared.info.lock().unwrap().camera.clone()
    }

    /// Keep the camera picture before its crop, for `latest_raw`.
    pub fn want_raw(&self, on: bool) {
        if !self.shared.want_raw.swap(on, Ordering::Relaxed) || on {
            return;
        }
        *self.shared.latest_raw.lock().unwrap() = None;
    }

    /// The latest camera picture (brightened, unmirrored) before its crop.
    pub fn latest_raw(&self) -> Option<Arc<Frame>> {
        self.shared.latest_raw.lock().unwrap().clone()
    }

    pub fn latest_frame(&self) -> Option<Arc<Frame>> {
        self.shared.latest.lock().unwrap().0.clone()
    }

    /// Latest output frame fitted into the preview box, or None. A new Arc per
    /// frame, so `Arc::ptr_eq` tells whether it changed.
    pub fn latest_preview(&self) -> Option<Arc<Preview>> {
        self.shared.latest.lock().unwrap().1.clone()
    }

    pub fn set_preview_box(&self, size: Option<(usize, usize)>) {
        *self.shared.preview_box.lock().unwrap() = size;
    }

    /// Called (on the output thread) whenever a new preview is ready.
    pub fn set_on_preview(&self, f: impl Fn() + Send + Sync + 'static) {
        *self.shared.on_preview.lock().unwrap() = Some(Box::new(f));
    }

    pub fn start(&self) {
        let mut t = self.thread.lock().unwrap();
        if t.as_ref().is_some_and(|h| !h.is_finished()) {
            return;
        }
        crate::system::keep_realtime_when_background(); // keep streaming smoothly when minimized
        self.shared.stop.store(false, Ordering::Relaxed);
        {
            let mut info = self.shared.info.lock().unwrap();
            info.error = None;
            info.status = Status::Starting;
        }
        let sh = self.shared.clone();
        *t = std::thread::Builder::new().name("vcam-worker".into()).spawn(move || thread_main(&sh)).ok();
    }

    /// Stop and wait (up to 3 s) for the session to wind down.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        let mut t = self.thread.lock().unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        while t.as_ref().is_some_and(|h| !h.is_finished()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if t.as_ref().is_some_and(|h| h.is_finished()) {
            let _ = t.take().unwrap().join();
        }
    }
}

impl Drop for VCamWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn thread_main(sh: &Arc<Shared>) {
    // A session runs until stop or until a pipeline-shape change requests a
    // reconnect, then rebuilds -- so any option can change without OFF/ON.
    let r = (|| -> Result<()> {
        while !sh.stop.load(Ordering::Relaxed) {
            sh.restart.store(false, Ordering::Relaxed);
            sh.info.lock().unwrap().status = Status::Starting;
            run_session(sh)?;
        }
        Ok(())
    })();
    let mut info = sh.info.lock().unwrap();
    if let Err(e) = r {
        info.error = Some(format!("{e:#}"));
        info.status = Status::Error;
    } else {
        info.status = Status::Stopped;
    }
    info.running = false;
    info.quality = None;
    info.device = None;
    info.camera = None;
    info.waiting = false;
    drop(info);
    *sh.latest.lock().unwrap() = (None, None);
    *sh.latest_raw.lock().unwrap() = None;
}

fn fitted_background(sh: &Shared, path: &Path, framing: &Framing, w: usize, h: usize) -> Result<Arc<Frame>> {
    let mtime = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    let src_key: SourceKey = (path.to_path_buf(), mtime);
    let key: BgKey = (src_key.clone(), w, h, *framing);
    let source = {
        let mut bg = sh.bg.lock().unwrap();
        if let Some(img) = most_recent(&mut bg.cache, &key) {
            return Ok(img);
        }
        most_recent(&mut bg.sources, &src_key)
    };
    let source = match source {
        Some(s) => s,
        None => {
            // No lock: decoding can be slow.
            let s = Arc::new(imgproc::load_background_source(path, SOURCE_MAX_SIDE)?);
            keep(&mut sh.bg.lock().unwrap().sources, src_key, s.clone(), SOURCES_KEPT);
            s
        }
    };
    let img = Arc::new(imgproc::frame_background(&source, framing, w, h));
    keep(&mut sh.bg.lock().unwrap().cache, key, img.clone(), 8); // ~2.7 MB each at 720p
    Ok(img)
}

/// The entry for `key`, moved to the most-recent end.
fn most_recent<K: PartialEq, V: Clone>(list: &mut Vec<(K, V)>, key: &K) -> Option<V> {
    let i = list.iter().position(|(k, _)| k == key)?;
    let entry = list.remove(i);
    let v = entry.1.clone();
    list.push(entry);
    Some(v)
}

fn keep<K: PartialEq, V>(list: &mut Vec<(K, V)>, key: K, v: V, cap: usize) {
    list.retain(|(k, _)| *k != key);
    list.push((key, v));
    while list.len() > cap {
        list.remove(0);
    }
}

/// Build from a config snapshot (no lock held: image decoding can be slow).
fn build_compositor(sh: &Shared, cfg: &Config, w: usize, h: usize) -> Compositor {
    match cfg.mode {
        BgMode::None => return Compositor::passthrough(),
        BgMode::Color => return Compositor::color(w, h, cfg.color),
        BgMode::Image => {
            if let Some(p) = cfg.bg_path.as_deref().filter(|p| p.is_file()) {
                match fitted_background(sh, p, &cfg.framing, w, h) {
                    Ok(bg) => {
                        let bg = if cfg.blur_image {
                            Arc::new(imgproc::blur_background_image(&bg, cfg.blur_strength))
                        } else {
                            bg
                        };
                        return if cfg.realistic {
                            Compositor::realistic(&bg, cfg.blur_image)
                        } else {
                            Compositor::image(bg)
                        };
                    }
                    Err(e) => crate::log!("{e:#}; blurring instead"), // unreadable image
                }
            }
        }
        BgMode::Blur => {}
    }
    Compositor::blur(w, h, cfg.blur_strength)
}

/// Build the new compositor off the frame loop; the old background keeps
/// streaming until it's ready, and a newer selection supersedes this one.
fn load_compositor_async(sh: &Arc<Shared>, gen: u64, cfg: Config, w: usize, h: usize) {
    let sh = sh.clone();
    std::thread::spawn(move || {
        let comp = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| build_compositor(&sh, &cfg, w, h)));
        let mut bg = sh.bg.lock().unwrap();
        bg.loading = false; // even if it failed, or the next build would never start
        if let Ok(comp) = comp {
            if gen == bg.gen {
                bg.pending = Some(Arc::new(comp));
            }
        }
    });
}

/// Runs on the output thread: keep the frame and build the GUI preview (fitted
/// to the box the GUI asked for, RGB) so the GUI thread only displays it.
fn store_latest(sh: &Shared, out: &Arc<Frame>) {
    let pbox = *sh.preview_box.lock().unwrap();
    let preview = pbox.and_then(|(bw, bh)| {
        let k = (bw as f64 / out.w as f64).min(bh as f64 / out.h as f64);
        let (tw, th) = ((out.w as f64 * k) as usize, (out.h as f64 * k) as usize);
        (tw > 2 && th > 2).then(|| Arc::new(Preview { w: tw, h: th, rgb: imgproc::make_preview(out, tw, th) }))
    });
    let has_preview = preview.is_some();
    *sh.latest.lock().unwrap() = (Some(out.clone()), preview);
    if has_preview {
        if let Some(cb) = &*sh.on_preview.lock().unwrap() {
            cb();
        }
    }
}

fn make_engine(cfg: &Config, w: usize, h: usize) -> Result<RvmEngine> {
    let m = hw::machine();
    crate::log!("CPU: {} ({})", m.cpu, m.summary());
    let dsr = cfg.downsample.unwrap_or_else(|| autotune::initial(m, w, h));
    let engine = RvmEngine::new(w, h, Some(dsr), !cfg.multi_person, cfg.threads)
        .map_err(|e| e.context("RVM engine failed to start"))?;
    crate::log!(
        "Engine: RobustVideoMatting (downsample_ratio={:.2}, CPU x{} threads{})",
        engine.dsr,
        engine.threads,
        if engine.staged() { ", 3-stage pipeline" } else { "" }
    );
    Ok(engine)
}

fn run_session(sh: &Arc<Shared>) -> Result<()> {
    let cfg = sh.cfg.lock().unwrap().clone();
    let (grabber, cam_info) = FrameGrabber::start(cfg.camera, cfg.width, cfg.height, cfg.fps as f64, cfg.exposure)?;
    let (w, h, mut cam_fps) = (cam_info.width, cam_info.height, cam_info.fps);
    crate::log!("Capturing {w}x{h} {} @ {:.0} fps from '{}'", cam_info.format, cam_info.fps, cam_info.name);
    if let Some(e) = cam_info.exposure {
        crate::log!("Exposure fixed at 1/{} s to hold the frame rate", 1u32 << -e.min(0));
    }
    let mut engine = make_engine(&cfg, w, h)?;

    let bg_cfg = {
        let mut bg = sh.bg.lock().unwrap();
        bg.dirty = false;
        bg.pending = None;
        bg.gen += 1;
        sh.cfg.lock().unwrap().clone()
    };
    let mut compositor = Arc::new(build_compositor(sh, &bg_cfg, w, h));

    // Output can be a different (larger) size than capture. NV12 needs even sizes.
    let out_w = cfg.out_width.unwrap_or(w) & !1;
    let out_h = cfg.out_height.unwrap_or(h) & !1;
    let upscale = Some((out_w, out_h)).filter(|&s| s != (w, h));
    if upscale.is_some() {
        crate::log!("Scaling output to {out_w}x{out_h}");
    }
    let cam = match cfg.virtualcam {
        VirtualCamUse::Off => None,
        VirtualCamUse::Required => Some(VirtualCam::open(out_w, out_h, cfg.fps as f64)?),
        VirtualCamUse::Optional => match VirtualCam::open(out_w, out_h, cfg.fps as f64) {
            Ok(c) => Some(c),
            Err(e) => {
                crate::log!("WARNING: could not start a virtual camera: {e:#}");
                crate::log!("Install OBS Studio (provides 'OBS Virtual Camera') and retry.");
                crate::log!("Continuing in preview-only mode.");
                None
            }
        },
    };
    if let Some(c) = &cam {
        crate::log!("Virtual camera live -> '{}' (select it in your video app)", c.device());
    }
    let device = cam.as_ref().map(|c| c.device().to_string());
    let cam = cam.map(|c| Arc::new(Mutex::new(c)));
    // Who has the virtual camera open. None (no virtual camera: preview only)
    // == always running.
    let watcher = cam.as_ref().map(|c| c.lock().unwrap().watcher());
    let keep_fed = cam.clone().map(KeepFed::start);
    let sh2 = sh.clone();
    let pipeline = OutputPipeline::new(cam.clone(), Some(Box::new(move |out: &Arc<Frame>| store_latest(&sh2, out))))?;
    {
        let mut info = sh.info.lock().unwrap();
        info.device = device;
        info.camera = Some(cam_info);
        info.running = true;
        info.status = Status::Running;
    }

    // The webcam; None while paused (no app is using the virtual camera) or
    // while it's being retried.
    let mut grabber = Some(grabber);
    let mut paused = false;
    // When the output was last wanted. Starting counts: the webcam runs for
    // LINGER first, which also shows it works.
    let mut wanted_at = Instant::now();
    // Since when the webcam has been failing, and when to try it next.
    let (mut failing_since, mut next_try) = (None::<Instant>, Instant::now());
    let (mut frames, mut since, mut last_seq) = (0u64, Instant::now(), 0u64);
    // The picture's curve (exposure lift + brightness), rebuilt only when
    // either changes.
    let mut lut: Option<((f32, f32), [u8; 256])> = None;
    // The camera framing the capture thread last measured brightness for.
    let mut measured: Option<Framing> = None;
    // Fits the edge detail to this PC while the quality is on Auto.
    let mut auto: Option<AutoQuality> = None;
    while !sh.stop.load(Ordering::Relaxed) && !sh.restart.load(Ordering::Relaxed) {
        if let Some(e) = pipeline.error() {
            anyhow::bail!(e);
        }
        if let Some(watch) = watcher {
            // An app streaming from the virtual camera wants the output; so
            // does the camera crop editor, which shows the live picture.
            if watch.consumers() > 0 || sh.want_raw.load(Ordering::Relaxed) {
                wanted_at = Instant::now();
            }
            if wanted_at.elapsed() >= LINGER {
                if !paused {
                    paused = true;
                    grabber = None;
                    failing_since = None;
                    crate::log!("No app is using the virtual camera: paused (webcam off)");
                    let mut info = sh.info.lock().unwrap();
                    info.waiting = true;
                    info.fps = 0.0;
                }
                std::thread::sleep(WATCH_EVERY);
                continue;
            }
            if paused {
                paused = false;
                crate::log!("An app opened the virtual camera: resuming");
                // Black, not the picture from when it paused, until the
                // first new frame (the keep-alive repeats it meanwhile).
                cam.as_ref().unwrap().lock().unwrap().blank();
                *sh.latest.lock().unwrap() = (None, None);
                sh.info.lock().unwrap().waiting = false;
            }
            if grabber.is_none() {
                if Instant::now() < next_try {
                    std::thread::sleep(WATCH_EVERY);
                    continue;
                }
                let (g, info) = match FrameGrabber::start(cfg.camera, cfg.width, cfg.height, cfg.fps as f64, cfg.exposure) {
                    Ok(opened) => opened,
                    Err(e) => {
                        retry_webcam(e, &mut failing_since, &mut next_try)?;
                        continue;
                    }
                };
                if (info.width, info.height) != (w, h) {
                    // The engine and background are built for the old size.
                    crate::log!("The webcam came back at {}x{} (was {w}x{h}): rebuilding", info.width, info.height);
                    sh.restart.store(true, Ordering::Relaxed);
                    continue;
                }
                cam_fps = info.fps;
                sh.info.lock().unwrap().camera = Some(info);
                grabber = Some(g);
                // A fresh webcam: its frames count from 0, and its exposure
                // and the speed measurements start over.
                (frames, since, last_seq) = (0, Instant::now(), 0);
                measured = None;
                auto = None;
            }
        }
        let Some(g) = grabber.as_ref() else { continue };
        let Some((seq, frame, lift)) = g.wait_next(last_seq, Duration::from_millis(100)) else {
            if !g.running() {
                let why = g.error().map(|e| format!(": {e}")).unwrap_or_default();
                let e = anyhow::anyhow!("camera stopped delivering frames{why}");
                if watcher.is_none() {
                    return Err(e);
                }
                grabber = None;
                retry_webcam(e, &mut failing_since, &mut next_try)?;
            }
            continue;
        };
        last_seq = seq;
        failing_since = None; // the webcam works
        let (load, mirror, bright, want_dsr, multi, cam_framing, outline) = {
            let mut bg = sh.bg.lock().unwrap();
            let cfg = sh.cfg.lock().unwrap();
            let mut load = None;
            if bg.dirty && !bg.loading {
                bg.dirty = false;
                bg.loading = true;
                bg.gen += 1;
                load = Some((bg.gen, cfg.clone()));
            }
            if let Some(c) = bg.pending.take() {
                compositor = c;
            }
            let outline = cfg.outline && cfg.mode == BgMode::Image;
            (load, cfg.mirror, cfg.brightness, cfg.downsample, cfg.multi_person, cfg.camera_framing, outline)
        };
        if let Some((gen, c)) = load {
            load_compositor_async(sh, gen, c, w, h);
        }
        // Adjusted before matting, so the matte and the room matching see
        // the same picture the meeting does.
        let adjust = bright != 0.0 || lift != 1.0;
        if adjust && lut.as_ref().is_none_or(|l| l.0 != (lift, bright)) {
            lut = Some(((lift, bright), imgproc::camera_lut(lift, bright)));
        }
        let curve = lut.as_ref().filter(|_| adjust).map(|l| &l.1);
        if sh.want_raw.load(Ordering::Relaxed) {
            // The crop editor shows the camera's own picture, brightened as the
            // call sees it but unmirrored: the mirror comes after the framing.
            let whole = match curve {
                Some(c) => Arc::new(kernels::pool().install(|| frame.adjusted(false, Some(c)))),
                None => frame.clone(),
            };
            *sh.latest_raw.lock().unwrap() = Some(whole);
        }
        // The dim-light brightening measures the cropped part.
        if measured != Some(cam_framing) {
            g.set_region(cam_framing.effective_crop(w, h, cam_framing.target_aspect(w as f32 / h as f32)));
            measured = Some(cam_framing);
        }
        // Where the camera picture lands, mirrored with it, for the outline.
        let outline = outline.then(|| {
            let [x, y, pw, ph] = imgproc::camera_rect(&cam_framing, w, h, frame.w, frame.h);
            [if mirror { w - x - pw } else { x }, y, pw, ph]
        });
        // Camera framing first, on the camera's own picture: everything from
        // here on -- brightness, mirror, matting, faces, the room matching,
        // the output -- sees only the crop, sized into its place in a frame of
        // the capture size: the engine and background keep their size, so it
        // changes live (~1 ms at 720p).
        let frame = if cam_framing != Framing::default() {
            Arc::new(kernels::pool().install(|| imgproc::frame_camera(&frame, &cam_framing, w, h)))
        } else {
            frame
        };
        // Then the mirror, which flips the framed picture.
        let frame = if mirror || curve.is_some() {
            Arc::new(kernels::pool().install(|| frame.adjusted(mirror, curve)))
        } else {
            frame
        };
        if compositor.is_passthrough() {
            // Original camera: skip matting entirely (saves the CPU); the
            // engine stays loaded so switching back is instant.
            pipeline.submit(compositor.clone(), frame, None, upscale, outline)?;
        } else {
            // Live, in-place option changes (no rebuild needed).
            match want_dsr {
                Some(d) => {
                    auto = None;
                    engine.set_downsample(d);
                }
                None => {
                    let a = auto.get_or_insert_with(|| AutoQuality::new(engine.dsr, cam_fps, Instant::now()));
                    if let Some(d) = a.poll(Instant::now(), || engine.stage_time()) {
                        crate::log!("Auto quality: downsample_ratio {:.3} -> {d:.3}", engine.dsr);
                        engine.set_downsample(d);
                    }
                }
            }
            engine.set_single_person(!multi);
            // Only inference runs here; the post-processing happens on the
            // output thread, overlapping the next inference.
            let pending = engine.infer(frame.clone())?;
            pipeline.submit(compositor.clone(), frame, Some(pending), upscale, outline)?;
        }
        frames += 1;
        if frames % 15 == 0 {
            // The current rate, not the session's average: it moves when
            // the camera's rate does (e.g. the adaptive exposure kicking in).
            let mut info = sh.info.lock().unwrap();
            info.fps = 15.0 / since.elapsed().as_secs_f32();
            info.quality = Some(engine.dsr);
            drop(info);
            since = Instant::now();
        }
    }
    // Drop order: keep-alive, pipeline (drains), the virtual camera (closes
    // with the last of them), engine, webcam.
    drop(keep_fed);
    drop(pipeline);
    drop(cam);
    drop(engine);
    drop(grabber);
    Ok(())
}

/// The webcam failed while an app wants the output. Often that app is still
/// letting it go: e.g. Zoom switching from the webcam to this camera opens
/// this one before it has released the webcam. So retry for a while, showing
/// black, and give up (with `e`) only if it stays unavailable.
fn retry_webcam(e: anyhow::Error, failing_since: &mut Option<Instant>, next_try: &mut Instant) -> Result<()> {
    let since = *failing_since.get_or_insert_with(|| {
        crate::log!("Webcam not available yet ({e:#}); retrying for up to {} s", WEBCAM_RETRY_FOR.as_secs());
        Instant::now()
    });
    if since.elapsed() >= WEBCAM_RETRY_FOR {
        return Err(e);
    }
    *next_try = Instant::now() + WEBCAM_RETRY_EVERY;
    Ok(())
}

/// While an app is watching but nothing new goes out (paused, or the webcam
/// still starting), repeats the last frame so the app holds the camera open
/// steadily instead of dropping and reopening it (see `VirtualCam::keep_fed`).
struct KeepFed {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl KeepFed {
    fn start(cam: Arc<Mutex<VirtualCam>>) -> Self {
        let watcher = cam.lock().unwrap().watcher();
        let (stop, rx) = mpsc::channel::<()>();
        let thread = std::thread::Builder::new()
            .name("vcam-keepfed".into())
            .spawn(move || {
                while let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(KEEP_FED_GAP) {
                    if watcher.consumers() > 0 {
                        cam.lock().unwrap().keep_fed(KEEP_FED_GAP);
                    }
                }
            })
            .ok();
        Self { stop: Some(stop), thread }
    }
}

impl Drop for KeepFed {
    fn drop(&mut self) {
        self.stop = None; // disconnects: the thread ends at once
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

//! Webcam capture via Media Foundation's source reader.
//!
//! The reader decodes whatever the camera delivers (MJPEG / YUY2 / NV12) to
//! RGB32 for us. Picking the native format matters: many webcams do 1280x720
//! at only ~10 fps uncompressed but 30 fps as MJPEG, so the format is chosen
//! for the requested size first and frame rate second.
//!
//! Even then, in dim light a webcam's auto-exposure lengthens each exposure
//! past the frame interval and quietly drops to ~15-20 fps. Optionally the
//! exposure is pinned to the longest one that still fits a frame (restored when
//! the camera closes), which holds the full frame rate.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use windows::core::{Interface, GUID, PWSTR};
use windows::Win32::Media::DirectShow::{
    CameraControl_Exposure, CameraControl_Flags_Auto, CameraControl_Flags_Manual, IAMCameraControl,
};
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoTaskMemFree, CoUninitialize, COINIT_MULTITHREADED};

use crate::imgproc::Frame;

const FIRST_VIDEO: u32 = MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32;

/// COM (multithreaded apartment) + Media Foundation for the current thread.
struct MfThread;

impl MfThread {
    fn init() -> Result<Self> {
        unsafe {
            // S_FALSE / RPC_E_CHANGED_MODE just mean COM is already set up.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            MFStartup(MF_VERSION, MFSTARTUP_FULL).context("Media Foundation failed to start")?;
        }
        Ok(MfThread)
    }
}

impl Drop for MfThread {
    fn drop(&mut self) {
        unsafe {
            let _ = MFShutdown();
            CoUninitialize();
        }
    }
}

fn is_virtual(name: &str) -> bool {
    // Never offer our own output as an input.
    name.contains("OBS Virtual Camera")
}

/// Video capture devices (excluding the OBS Virtual Camera), in index order.
fn devices() -> Result<Vec<(String, IMFActivate)>> {
    unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1)?;
        let attrs = attrs.context("MFCreateAttributes")?;
        attrs.SetGUID(&MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)?;
        let mut ptr: *mut Option<IMFActivate> = std::ptr::null_mut();
        let mut count = 0u32;
        MFEnumDeviceSources(&attrs, &mut ptr, &mut count)?;
        let mut out = Vec::new();
        if !ptr.is_null() {
            for i in 0..count as usize {
                if let Some(act) = std::ptr::read(ptr.add(i)) {
                    let mut name = PWSTR::null();
                    let mut len = 0u32;
                    let label = if act.GetAllocatedString(&MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &mut name, &mut len).is_ok() {
                        let s = name.to_string().unwrap_or_default();
                        CoTaskMemFree(Some(name.0 as *const _));
                        s
                    } else {
                        format!("Camera {}", i + 1)
                    };
                    if !is_virtual(&label) {
                        out.push((label, act));
                    }
                }
            }
            CoTaskMemFree(Some(ptr as *const _));
        }
        Ok(out)
    }
}

/// Names of the available cameras, in index order.
pub fn list_cameras() -> Vec<String> {
    // On a thread of its own: the caller (e.g. the GUI thread) may already be
    // in a single-threaded COM apartment.
    std::thread::spawn(|| -> Result<Vec<String>> {
        let _mf = MfThread::init()?;
        Ok(devices()?.into_iter().map(|(n, _)| n).collect())
    })
    .join()
    .ok()
    .and_then(Result::ok)
    .unwrap_or_default()
}

#[derive(Clone, Debug)]
pub struct CameraInfo {
    pub name: String,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub format: String,
    /// The pinned exposure, log2 seconds (-5 == 1/32 s); None == the camera's auto.
    pub exposure: Option<i32>,
}

fn split_u64(v: u64) -> (u32, u32) {
    ((v >> 32) as u32, v as u32)
}

fn subtype_name(g: &GUID) -> String {
    let fourcc = g.data1.to_le_bytes();
    if fourcc.iter().all(|c| c.is_ascii_alphanumeric() || *c == b' ') {
        String::from_utf8_lossy(&fourcc).trim().to_string()
    } else {
        format!("{:08x}", g.data1)
    }
}

/// How the webcam's exposure is handled. In dim light a webcam's
/// auto-exposure lengthens each exposure past the frame interval and drops to
/// ~15-20 fps; pinning it to the longest exposure that fits a frame holds the
/// full rate, at the cost of a darker picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureMode {
    /// The camera's own auto-exposure, untouched.
    Camera,
    /// Pinned from the start.
    Fixed,
    /// The camera's auto-exposure while it holds the frame rate; pinned once
    /// dim light slows it, with the picture lifted back to the brightness
    /// auto-exposure was giving (see `Adaptive`).
    Adaptive,
}

/// The camera's exposure control, and what to put back.
struct ExposureControl {
    cc: IAMCameraControl,
    /// The longest exposure that fits a frame, log2 seconds.
    fixed: i32,
    /// The camera's own (value, flags), restored on release.
    orig: (i32, i32),
    pinned: bool,
}

impl ExposureControl {
    /// None if the camera has no manual exposure.
    ///
    /// The setting lives in the camera, so a session that was killed leaves it
    /// pinned. Manual at exactly our value is taken to be such a leftover: it's
    /// put back to auto rather than kept as "the camera's own" setting.
    unsafe fn open(source: &IMFMediaSource, fps: f64) -> Option<Self> {
        let cc: IAMCameraControl = source.cast().ok()?;
        let id = CameraControl_Exposure.0;
        let (mut lo, mut hi, mut step, mut def, mut caps) = (0, 0, 0, 0, 0);
        cc.GetRange(id, &mut lo, &mut hi, &mut step, &mut def, &mut caps).ok()?;
        if caps & CameraControl_Flags_Manual.0 == 0 {
            return None;
        }
        let (mut value, mut flags) = (0, 0);
        cc.Get(id, &mut value, &mut flags).ok()?;
        // Exposure is in log2 seconds: 2^-5 s = 31 ms is the longest under 33 ms.
        let fixed = (-(fps.max(1.0).log2().ceil() as i32)).clamp(lo, hi);
        let mut orig = (value, flags);
        if flags == CameraControl_Flags_Manual.0 && value == fixed && caps & CameraControl_Flags_Auto.0 != 0 {
            orig = (def, CameraControl_Flags_Auto.0);
            let _ = cc.Set(id, orig.0, orig.1);
        }
        Some(Self { cc, fixed, orig, pinned: false })
    }

    fn pin(&mut self) -> bool {
        self.pinned = unsafe { self.cc.Set(CameraControl_Exposure.0, self.fixed, CameraControl_Flags_Manual.0) }.is_ok();
        self.pinned
    }

    fn release(&mut self) {
        if std::mem::take(&mut self.pinned) {
            unsafe {
                let _ = self.cc.Set(CameraControl_Exposure.0, self.orig.0, self.orig.1);
            }
        }
    }
}

/// How long the rate is measured over, and how long to let the camera settle
/// after its exposure changes before judging it.
const RATE_WINDOW: Duration = Duration::from_secs(2);
const SETTLE: Duration = Duration::from_secs(3);
/// Pin when auto-exposure delivers less than this share of the nominal rate.
const SLOW: f64 = 0.85;

/// `ExposureMode::Adaptive`, fed one frame at a time on the camera thread.
///
/// While pinned, the picture is lifted with a gamma curve that maps its mean
/// brightness back to what auto-exposure was giving (black and white stay
/// put), so the switch doesn't darken it; it follows the room from there
/// much as auto-exposure would. Once the room is bright enough that the
/// pinned picture outshines that, auto-exposure would hold the rate itself,
/// and it gets the camera back.
struct Adaptive {
    fps: f64,
    /// Frame arrival times over the last `RATE_WINDOW`.
    times: VecDeque<Instant>,
    /// When the exposure last changed (camera opened, pinned or released).
    changed: Instant,
    /// Smoothed mean luma under auto-exposure: the brightness to keep.
    target: Option<f32>,
    /// Smoothed mean luma while pinned.
    pinned: Option<f32>,
    /// Smoothed exact `lift`, before it's rounded.
    smooth: Option<f32>,
    /// Since when the pinned picture has been clearly brighter than `target`.
    bright_since: Option<Instant>,
    /// Pinning didn't raise the rate (it's slow for some other reason).
    gave_up: bool,
    /// Gamma exponent for the frames; 1 == as captured, < 1 brightens.
    lift: f32,
}

impl Adaptive {
    fn new(fps: f64) -> Self {
        Self {
            fps,
            times: VecDeque::new(),
            changed: Instant::now(),
            target: None,
            pinned: None,
            smooth: None,
            bright_since: None,
            gave_up: false,
            lift: 1.0,
        }
    }

    /// `region`: the part of the frame that's used (x0, y0, x1, y1 as
    /// fractions), whose brightness is the one to keep.
    fn step(&mut self, ctl: &mut ExposureControl, frame: &Frame, region: [f32; 4]) {
        let now = Instant::now();
        let ema = |v: Option<f32>, x: f32| v.map_or(x, |v| v + (x - v) * 0.1);
        let hist = luma_histogram(frame, region);
        let luma = lifted_mean(&hist, 1.0);
        self.times.push_back(now);
        while self.times.front().is_some_and(|&t| now - t > RATE_WINDOW) {
            self.times.pop_front();
        }
        // The window is shorter than the settle time, so it's all post-change.
        let settled = now - self.changed >= SETTLE;
        let slow = settled && (self.times.len() as f64) < SLOW * self.fps * RATE_WINDOW.as_secs_f64();
        if !ctl.pinned {
            self.lift = 1.0;
            self.target = Some(ema(self.target, luma));
            if slow && !self.gave_up && ctl.pin() {
                crate::log!(
                    "Dim light slowed the camera to {:.0} fps: exposure fixed at 1/{} s, picture brightened to match",
                    self.times.len() as f64 / RATE_WINDOW.as_secs_f64(),
                    1u32 << -ctl.fixed.min(0)
                );
                (self.changed, self.pinned, self.smooth, self.bright_since) = (now, None, None, None);
            }
            return;
        }
        if slow {
            crate::log!("Fixing the exposure didn't raise the frame rate; back to the camera's auto-exposure");
            self.gave_up = true;
            self.release(ctl, now);
            return;
        }
        // The new exposure takes a few frames to arrive.
        if now - self.changed < Duration::from_millis(300) {
            return;
        }
        let p = ema(self.pinned, luma);
        self.pinned = Some(p);
        let target = self.target.unwrap_or(p).clamp(16.0, 235.0);
        let e = ema(self.smooth, lift_for(&hist, target));
        self.smooth = Some(e);
        self.lift = (e * 100.0).round() / 100.0; // steps, so the curve isn't rebuilt every frame
        if p > target * 1.15 {
            if now - *self.bright_since.get_or_insert(now) >= SETTLE {
                crate::log!("Enough light for the full frame rate: back to the camera's auto-exposure");
                self.release(ctl, now);
            }
        } else {
            self.bright_since = None;
        }
    }

    fn release(&mut self, ctl: &mut ExposureControl, now: Instant) {
        ctl.release();
        (self.changed, self.lift) = (now, 1.0);
    }
}

/// Luma histogram of `region` of a frame (fractions), sampled every 8th pixel
/// each way (closer for a small region).
fn luma_histogram(f: &Frame, region: [f32; 4]) -> [u32; 256] {
    let px = |v: f32, n: usize| ((v.clamp(0.0, 1.0) * n as f32) as usize).min(n - 1);
    let (x0, y0) = (px(region[0], f.w), px(region[1], f.h));
    let (x1, y1) = (px(region[2], f.w).max(x0) + 1, px(region[3], f.h).max(y0) + 1);
    let step = ((x1 - x0).min(y1 - y0) / 40).clamp(1, 8);
    let mut hist = [0u32; 256];
    for y in (y0..y1).step_by(step) {
        for px in f.data[(y * f.w + x0) * 3..][..(x1 - x0) * 3].chunks_exact(3).step_by(step) {
            let l = (29 * px[0] as u32 + 150 * px[1] as u32 + 77 * px[2] as u32 + 128) >> 8; // BGR
            hist[l as usize] += 1;
        }
    }
    hist
}

/// Mean luma of the histogram's picture after a gamma `lift`.
fn lifted_mean(hist: &[u32; 256], lift: f32) -> f32 {
    let (mut sum, mut n) = (0f64, 0u64);
    for (v, &c) in hist.iter().enumerate().filter(|(_, &c)| c > 0) {
        sum += c as f64 * (v as f64 / 255.0).powf(lift as f64);
        n += c as u64;
    }
    (255.0 * sum / n.max(1) as f64) as f32
}

/// The lift (0.45..=1) that brings the picture's mean luma to `target`. A
/// curve moves the mean less than it moves the mean's own level, so this is
/// solved on the histogram (by bisection: the mean falls as the lift rises).
fn lift_for(hist: &[u32; 256], target: f32) -> f32 {
    let (mut lo, mut hi) = (0.45f32, 1.0f32);
    if lifted_mean(hist, hi) >= target {
        return hi;
    }
    if lifted_mean(hist, lo) <= target {
        return lo;
    }
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if lifted_mean(hist, mid) > target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

struct Camera {
    reader: IMFSourceReader,
    source: IMFMediaSource,
    /// Restored on drop.
    exposure: Option<ExposureControl>,
    w: usize,
    h: usize,
    stride: i32,
}

impl Camera {
    fn open(index: usize, width: usize, height: usize, fps: f64, mode: ExposureMode) -> Result<(Self, CameraInfo)> {
        let busy = || {
            anyhow!(
                "Could not open camera {index}. A webcam can only be used by one app at a time; close \
                 anything else using it (Meet/Zoom/Teams/OBS, or another program), or try a different camera."
            )
        };
        let mut devs = devices()?;
        if index >= devs.len() {
            bail!("Could not open camera {index}: only {} camera(s) found.", devs.len());
        }
        let (name, act) = devs.swap_remove(index);
        unsafe {
            let source: IMFMediaSource = act.ActivateObject().map_err(|_| busy())?;
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 3)?;
            let attrs = attrs.context("MFCreateAttributes")?;
            // Decode/convert to RGB32 inside the reader; keep latency low.
            attrs.SetUINT32(&MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, 1)?;
            attrs.SetUINT32(&MF_LOW_LATENCY, 1)?;
            let reader = MFCreateSourceReaderFromMediaSource(&source, &attrs).map_err(|_| busy())?;
            reader.SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS.0 as u32, false)?;
            reader.SetStreamSelection(FIRST_VIDEO, true)?;

            // Rank the camera's native formats: same aspect ratio (a 4:3 frame
            // can't stand in for 16:9), closest size, enough fps, uncompressed
            // (cheaper to decode), then closest fps.
            let mut cands = Vec::new();
            for i in 0.. {
                let Ok(t) = reader.GetNativeMediaType(FIRST_VIDEO, i) else { break };
                let (Ok(sz), Ok(rate), Ok(sub)) =
                    (t.GetUINT64(&MF_MT_FRAME_SIZE), t.GetUINT64(&MF_MT_FRAME_RATE), t.GetGUID(&MF_MT_SUBTYPE))
                else {
                    continue;
                };
                let ((w, h), (num, den)) = (split_u64(sz), split_u64(rate));
                let f = if den > 0 { num as f64 / den as f64 } else { 0.0 };
                let tag = subtype_name(&sub);
                if matches!(tag.as_str(), "H264" | "HEVC" | "H265") {
                    continue;
                }
                let aspect = (w as f64 / h.max(1) as f64 - width as f64 / height.max(1) as f64).abs() > 0.02;
                let size_err = (w as i64 - width as i64).unsigned_abs() + (h as i64 - height as i64).unsigned_abs();
                let short = ((fps - f).max(0.0) * 10.0).round() as u64;
                let compressed = (sub == MFVideoFormat_MJPG) as u8;
                let over = ((f - fps).abs() * 10.0).round() as u64;
                cands.push(((aspect, size_err, short, compressed, over), t, w, h, f, tag));
            }
            cands.sort_by_key(|c| c.0);
            let mut chosen = None;
            for (_, t, w, h, f, tag) in cands {
                if reader.SetCurrentMediaType(FIRST_VIDEO, None, &t).is_ok() {
                    chosen = Some((w, h, f, tag));
                    break;
                }
            }
            let (_, _, f, tag) = chosen.ok_or_else(busy)?;

            let out = MFCreateMediaType()?;
            out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
            out.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32)?;
            reader
                .SetCurrentMediaType(FIRST_VIDEO, None, &out)
                .context("the camera's video can't be converted to RGB")?;
            let cur = reader.GetCurrentMediaType(FIRST_VIDEO)?;
            let (w, h) = split_u64(cur.GetUINT64(&MF_MT_FRAME_SIZE)?);
            let stride = cur.GetUINT32(&MF_MT_DEFAULT_STRIDE).map(|s| s as i32).unwrap_or(w as i32 * 4);
            // Opened even for `Camera`, to clear a leftover pin.
            let exposure = ExposureControl::open(&source, f).filter(|_| mode != ExposureMode::Camera);
            let exposure = match (mode, exposure) {
                (ExposureMode::Fixed, Some(mut e)) => e.pin().then_some(e),
                (_, e) => e,
            };
            let info = CameraInfo {
                name,
                width: w as usize,
                height: h as usize,
                fps: f,
                format: tag,
                exposure: exposure.as_ref().filter(|e| e.pinned).map(|e| e.fixed),
            };
            Ok((Camera { reader, source, exposure, w: w as usize, h: h as usize, stride }, info))
        }
    }

    /// Next frame (blocks for up to a frame interval).
    fn read(&mut self) -> Result<Frame> {
        unsafe {
            loop {
                let mut flags = 0u32;
                let mut sample = None;
                self.reader.ReadSample(FIRST_VIDEO, 0, None, Some(&mut flags), None, Some(&mut sample))?;
                if flags & (MF_SOURCE_READERF_ERROR.0 as u32 | MF_SOURCE_READERF_ENDOFSTREAM.0 as u32) != 0 {
                    bail!("camera stopped delivering frames");
                }
                let Some(sample) = sample else { continue }; // stream tick / gap
                let buf = sample.ConvertToContiguousBuffer()?;
                return self.copy_out(&buf);
            }
        }
    }

    unsafe fn copy_out(&self, buf: &IMFMediaBuffer) -> Result<Frame> {
        let (w, h) = (self.w, self.h);
        let mut frame = Frame::scratch(w, h);
        let mut convert = |scan0: *const u8, pitch: isize| {
            for y in 0..h {
                let row = std::slice::from_raw_parts(scan0.offset(y as isize * pitch), w * 4);
                let dst = &mut frame.data[y * w * 3..][..w * 3];
                for (d, s) in dst.chunks_exact_mut(3).zip(row.chunks_exact(4)) {
                    d.copy_from_slice(&s[..3]); // RGB32 is B, G, R, X in memory
                }
            }
        };
        if let Ok(b2) = buf.cast::<IMF2DBuffer>() {
            let (mut scan0, mut pitch) = (std::ptr::null_mut(), 0i32);
            b2.Lock2D(&mut scan0, &mut pitch)?;
            convert(scan0, pitch as isize);
            b2.Unlock2D()?;
        } else {
            let (mut p, mut len) = (std::ptr::null_mut(), 0u32);
            buf.Lock(&mut p, None, Some(&mut len))?;
            if (len as usize) < w * h * 4 {
                buf.Unlock()?;
                bail!("short camera frame");
            }
            // A negative default stride means bottom-up rows.
            let start = if self.stride < 0 { p.add((h - 1) * w * 4) } else { p };
            convert(start, self.stride as isize);
            buf.Unlock()?;
        }
        Ok(frame)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        unsafe {
            // Hand the camera back as we found it (other apps share the setting).
            if let Some(e) = &mut self.exposure {
                e.release();
            }
            let _ = self.source.Shutdown();
        }
    }
}

struct GrabState {
    frame: Option<Arc<Frame>>,
    /// The frame's adaptive-exposure lift (1 == none); see `Adaptive`.
    lift: f32,
    /// The part of the frame that's used, whose brightness counts.
    region: [f32; 4],
    seq: u64,
    running: bool,
}

/// Continuously pulls frames on a background thread so the main loop never
/// blocks on the camera: a read can block for a full frame interval, and
/// overlapping it with matting lets the loop run at the camera's real rate.
pub struct FrameGrabber {
    shared: Arc<(Mutex<GrabState>, Condvar)>,
    stop: Arc<AtomicBool>,
    done: mpsc::Receiver<()>,
    thread: Option<JoinHandle<()>>,
}

impl FrameGrabber {
    pub fn start(index: usize, width: usize, height: usize, fps: f64, exposure: ExposureMode) -> Result<(Self, CameraInfo)> {
        let shared = Arc::new((Mutex::new(GrabState { frame: None, lift: 1.0, region: [0.0, 0.0, 1.0, 1.0], seq: 0, running: true }), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (info_tx, info_rx) = mpsc::sync_channel::<Result<CameraInfo>>(1);
        let (done_tx, done) = mpsc::channel();
        let (sh, st) = (shared.clone(), stop.clone());
        let thread = std::thread::Builder::new().name("camera".into()).spawn(move || {
            let _done = done_tx; // dropped (=> signalled) when this thread ends
            let run = || -> Result<()> {
                let _mf = MfThread::init()?;
                let (mut cam, mut adaptive) = match Camera::open(index, width, height, fps, exposure) {
                    Ok((cam, info)) => {
                        let adaptive = (exposure == ExposureMode::Adaptive).then(|| Adaptive::new(info.fps));
                        let _ = info_tx.send(Ok(info));
                        (cam, adaptive)
                    }
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return Ok(());
                    }
                };
                while !st.load(Ordering::Relaxed) {
                    let frame = cam.read()?;
                    let lift = match (&mut adaptive, &mut cam.exposure) {
                        (Some(a), Some(ctl)) => {
                            let region = sh.0.lock().unwrap().region;
                            a.step(ctl, &frame, region);
                            a.lift
                        }
                        _ => 1.0,
                    };
                    let (lock, cv) = &*sh;
                    let mut g = lock.lock().unwrap();
                    g.frame = Some(Arc::new(frame));
                    g.lift = lift;
                    g.seq += 1;
                    cv.notify_all();
                }
                Ok(())
            };
            let _ = run();
            let (lock, cv) = &*sh;
            lock.lock().unwrap().running = false;
            cv.notify_all();
        })?;
        let info = info_rx.recv().map_err(|_| anyhow!("camera thread failed"))??;
        Ok((Self { shared, stop, done, thread: Some(thread) }, info))
    }

    pub fn running(&self) -> bool {
        self.shared.0.lock().unwrap().running
    }

    /// The part of the frame that's used (x0, y0, x1, y1 as fractions): the
    /// dim-light brightening keeps that part's brightness.
    pub fn set_region(&self, region: [f32; 4]) {
        self.shared.0.lock().unwrap().region = region;
    }

    /// Block until a frame newer than `last_seq` arrives (woken by the camera
    /// thread, not by polling). None on timeout or when the camera has stopped.
    /// Also returns the gamma lift the frame needs (1 == none; see `Adaptive`).
    pub fn wait_next(&self, last_seq: u64, timeout: Duration) -> Option<(u64, Arc<Frame>, f32)> {
        let (lock, cv) = &*self.shared;
        let g = lock.lock().unwrap();
        let (g, _) = cv
            .wait_timeout_while(g, timeout, |g| (g.seq == last_seq || g.frame.is_none()) && g.running)
            .unwrap();
        if g.seq == last_seq {
            return None;
        }
        g.frame.clone().map(|f| (g.seq, f, g.lift))
    }
}

impl Drop for FrameGrabber {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // The reader returns within a frame interval; don't hang on a wedged
        // driver (on timeout the thread is detached).
        let ended = !matches!(self.done.recv_timeout(Duration::from_secs(2)), Err(mpsc::RecvTimeoutError::Timeout));
        if let Some(t) = self.thread.take().filter(|_| ended) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_measures_only_the_region() {
        // Left half black, right half white.
        let mut f = Frame::new(320, 180);
        for row in f.data.chunks_exact_mut(320 * 3) {
            row[160 * 3..].fill(255);
        }
        let right = luma_histogram(&f, [0.5, 0.0, 1.0, 1.0]);
        assert!(right[255] > 0 && right[0] == 0);
        let whole = luma_histogram(&f, [0.0, 0.0, 1.0, 1.0]);
        assert!(whole[255] > 0 && whole[0] > 0);
        assert!(luma_histogram(&f, [0.1, 0.1, 0.12, 0.12]).iter().sum::<u32>() > 0); // tiny crop still samples
    }

    #[test]
    fn lift_brings_the_mean_to_target() {
        // A dark, spread-out picture: levels 10..130.
        let mut hist = [0u32; 256];
        for v in 10..130 {
            hist[v] = 5;
        }
        let e = lift_for(&hist, 110.0);
        assert!(e < 1.0 && (lifted_mean(&hist, e) - 110.0).abs() < 0.5, "lift {e}");
        assert_eq!(lift_for(&hist, 40.0), 1.0); // already bright enough: never darkens
        assert_eq!(lift_for(&hist, 250.0), 0.45); // capped
    }
}

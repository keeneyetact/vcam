//! Webcam capture via Media Foundation's source reader.
//!
//! The reader decodes whatever the camera delivers (MJPEG / YUY2 / NV12) to
//! RGB32 for us. Picking the native format matters: many webcams do 1280x720
//! at only ~10 fps uncompressed but 30 fps as MJPEG, so the format is chosen
//! for the requested size first and frame rate second.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use windows::core::{Interface, GUID, PWSTR};
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

struct Camera {
    reader: IMFSourceReader,
    source: IMFMediaSource,
    w: usize,
    h: usize,
    stride: i32,
}

impl Camera {
    fn open(index: usize, width: usize, height: usize, fps: f64) -> Result<(Self, CameraInfo)> {
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
            let info = CameraInfo { name, width: w as usize, height: h as usize, fps: f, format: tag };
            Ok((Camera { reader, source, w: w as usize, h: h as usize, stride }, info))
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
        let mut frame = Frame::new(w, h);
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
            let _ = self.source.Shutdown();
        }
    }
}

struct GrabState {
    frame: Option<Arc<Frame>>,
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
    pub fn start(index: usize, width: usize, height: usize, fps: f64) -> Result<(Self, CameraInfo)> {
        let shared = Arc::new((Mutex::new(GrabState { frame: None, seq: 0, running: true }), Condvar::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (info_tx, info_rx) = mpsc::sync_channel::<Result<CameraInfo>>(1);
        let (done_tx, done) = mpsc::channel();
        let (sh, st) = (shared.clone(), stop.clone());
        let thread = std::thread::Builder::new().name("camera".into()).spawn(move || {
            let _done = done_tx; // dropped (=> signalled) when this thread ends
            let run = || -> Result<()> {
                let _mf = MfThread::init()?;
                let mut cam = match Camera::open(index, width, height, fps) {
                    Ok((cam, info)) => {
                        let _ = info_tx.send(Ok(info));
                        cam
                    }
                    Err(e) => {
                        let _ = info_tx.send(Err(e));
                        return Ok(());
                    }
                };
                while !st.load(Ordering::Relaxed) {
                    let frame = Arc::new(cam.read()?);
                    let (lock, cv) = &*sh;
                    let mut g = lock.lock().unwrap();
                    g.frame = Some(frame);
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

    /// Block until a frame newer than `last_seq` arrives (woken by the camera
    /// thread, not by polling). None on timeout or when the camera has stopped.
    pub fn wait_next(&self, last_seq: u64, timeout: Duration) -> Option<(u64, Arc<Frame>)> {
        let (lock, cv) = &*self.shared;
        let g = lock.lock().unwrap();
        let (g, _) = cv
            .wait_timeout_while(g, timeout, |g| (g.seq == last_seq || g.frame.is_none()) && g.running)
            .unwrap();
        if g.seq == last_seq {
            return None;
        }
        g.frame.clone().map(|f| (g.seq, f))
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

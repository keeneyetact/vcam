//! Face detection (YuNet) for single-person mode. The matte alone can't tell
//! two people apart once they touch -- they come out as one blob -- so the
//! faces say where each person is and `kernels::keep_central` cuts the blob
//! between them.
//!
//! Detection runs on its own thread a few times a second and never blocks the
//! frame path; each frame uses the newest result (heads move little in 100 ms).

use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use crate::imgproc::{self, Frame};
use crate::models;

/// The model's fixed input size.
const SIZE: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
/// Real faces score ~0.9; an open hand held out can score 0.6-0.65, and
/// taken for someone else's face it got the user's own arm cut away.
const MIN_SCORE: f32 = 0.8;
const NMS_IOU: f32 = 0.3;
/// How often the tracker looks for faces.
const INTERVAL: Duration = Duration::from_millis(100);
/// A face the detector stops finding is still reported this long, so one
/// missed look doesn't drop the cut between two people for a moment (the
/// other person popping back in). Faces off the matte are ignored anyway.
const HOLD: Duration = Duration::from_millis(400);

/// A detected face, in frame pixels: centre and size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
}

impl Face {
    fn iou(&self, o: &Face) -> f32 {
        let ix = (self.x + self.w / 2.0).min(o.x + o.w / 2.0) - (self.x - self.w / 2.0).max(o.x - o.w / 2.0);
        let iy = (self.y + self.h / 2.0).min(o.y + o.h / 2.0) - (self.y - self.h / 2.0).max(o.y - o.h / 2.0);
        let inter = ix.max(0.0) * iy.max(0.0);
        inter / (self.w * self.h + o.w * o.h - inter)
    }
}

/// The faces to report after a detection: those `found` now, plus those in
/// `seen` (each with when it was last found) that weren't found again but
/// were found within HOLD. Updates `seen`.
fn hold(seen: &mut Vec<(Face, Instant)>, found: Vec<Face>, now: Instant) -> Vec<Face> {
    let missed: Vec<(Face, Instant)> = seen
        .iter()
        .filter(|(f, t)| now - *t < HOLD && found.iter().all(|g| g.iou(f) <= 0.0))
        .copied()
        .collect();
    *seen = found.iter().map(|&f| (f, now)).chain(missed).collect();
    seen.iter().map(|&(f, _)| f).collect()
}

pub struct FaceDetector {
    sess: Session,
}

impl FaceDetector {
    pub fn new() -> Result<Self> {
        let err = |e: ort::Error<_>| anyhow!("{e}");
        let sess = Session::builder(crate::rvm::ort_env()?)?
            .with_intra_threads(1)
            .map_err(err)?
            .with_intra_op_spinning(false)
            .map_err(err)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(err)?
            .commit_from_memory(&models::face_model_bytes()?)
            .context("loading the face detection model")?;
        Ok(Self { sess })
    }

    /// Faces in `frame`, most confident first.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Face>> {
        if frame.w == 0 || frame.h == 0 {
            return Ok(Vec::new());
        }
        // Letterbox into the top-left of the 640x640 input (BGR, 0..255, NCHW).
        let scale = SIZE as f32 / frame.w.max(frame.h) as f32;
        let (rw, rh) = (
            ((frame.w as f32 * scale).round() as usize).clamp(1, SIZE),
            ((frame.h as f32 * scale).round() as usize).clamp(1, SIZE),
        );
        let small = imgproc::resize(frame, rw, rh);
        let mut input = vec![0f32; 3 * SIZE * SIZE];
        for y in 0..rh {
            for (x, px) in small.data[y * rw * 3..][..rw * 3].chunks_exact(3).enumerate() {
                for c in 0..3 {
                    input[c * SIZE * SIZE + y * SIZE + x] = px[c] as f32;
                }
            }
        }
        let input = Tensor::from_array((vec![1i64, 3, SIZE as i64, SIZE as i64], input))?;
        let out = self.sess.run(ort::inputs!["input" => &input])?;

        let mut faces = Vec::new();
        for s in STRIDES {
            let get = |name: &str| -> Result<&[f32]> {
                let v = out.get(&format!("{name}_{s}")).with_context(|| format!("face model has no {name}_{s}"))?;
                Ok(v.try_extract_tensor::<f32>()?.1)
            };
            let (cls, obj, bbox) = (get("cls")?, get("obj")?, get("bbox")?);
            let cols = SIZE / s;
            for (i, (&c, &o)) in cls.iter().zip(obj).enumerate() {
                let score = (c.clamp(0.0, 1.0) * o.clamp(0.0, 1.0)).sqrt();
                if score < MIN_SCORE {
                    continue;
                }
                let b = &bbox[i * 4..][..4];
                let sf = s as f32;
                faces.push(Face {
                    x: ((i % cols) as f32 + b[0]) * sf / scale,
                    y: ((i / cols) as f32 + b[1]) * sf / scale,
                    w: b[2].exp() * sf / scale,
                    h: b[3].exp() * sf / scale,
                    score,
                });
            }
        }
        faces.sort_by(|a, b| b.score.total_cmp(&a.score));
        let mut kept: Vec<Face> = Vec::new();
        for f in faces {
            if kept.iter().all(|k| k.iou(&f) <= NMS_IOU) {
                kept.push(f);
            }
        }
        Ok(kept)
    }
}

/// The newest detection and the frame size it was made at.
type Latest = Arc<Mutex<((usize, usize), Arc<Vec<Face>>)>>;

/// Runs a `FaceDetector` on its own thread on at most one frame per INTERVAL.
pub struct FaceTracker {
    tx: Option<SyncSender<Arc<Frame>>>,
    latest: Latest,
    last_offer: Option<Instant>,
    thread: Option<JoinHandle<()>>,
}

impl FaceTracker {
    /// Starts the thread; the model loads there, so this never blocks. If it
    /// can't load, the tracker just never reports faces.
    pub fn start() -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<Arc<Frame>>(1);
        let latest: Latest = Arc::new(Mutex::new(((0, 0), Arc::new(Vec::new()))));
        let out = latest.clone();
        let thread = std::thread::Builder::new().name("faces".into()).spawn(move || {
            let mut det = match FaceDetector::new() {
                Ok(d) => d,
                Err(e) => {
                    crate::log!("Face detection unavailable ({e:#}); single-person mode keeps the most central shape");
                    return;
                }
            };
            let mut seen = Vec::new();
            let mut size = (0, 0);
            while let Ok(frame) = rx.recv() {
                if size != (frame.w, frame.h) {
                    seen.clear(); // held faces are in the old frame's pixels
                    size = (frame.w, frame.h);
                }
                match det.detect(&frame) {
                    Ok(found) => {
                        let faces = hold(&mut seen, found, Instant::now());
                        *out.lock().unwrap() = (size, Arc::new(faces));
                    }
                    Err(e) => {
                        crate::log!("Face detection failed ({e:#}); turning it off");
                        *out.lock().unwrap() = ((0, 0), Arc::new(Vec::new()));
                        return;
                    }
                }
            }
        })?;
        Ok(Self { tx: Some(tx), latest, last_offer: None, thread: Some(thread) })
    }

    /// Hands `frame` to the detector if it's due and idle; returns the newest
    /// faces found at this frame size.
    pub fn offer(&mut self, frame: &Arc<Frame>) -> Arc<Vec<Face>> {
        let now = Instant::now();
        if self.last_offer.is_none_or(|t| now - t >= INTERVAL) {
            if let Some(tx) = &self.tx {
                match tx.try_send(frame.clone()) {
                    Ok(()) => self.last_offer = Some(now),
                    Err(TrySendError::Full(_)) => {}
                    Err(TrySendError::Disconnected(_)) => self.tx = None,
                }
            }
        }
        let latest = self.latest.lock().unwrap();
        if latest.0 == (frame.w, frame.h) {
            latest.1.clone()
        } else {
            Arc::new(Vec::new())
        }
    }
}

impl Drop for FaceTracker {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hold_keeps_a_briefly_missed_face() {
        let f = |x: f32| Face { x, y: 50.0, w: 40.0, h: 40.0, score: 0.9 };
        let t0 = Instant::now();
        let mut seen = Vec::new();
        assert_eq!(hold(&mut seen, vec![f(100.0), f(300.0)], t0).len(), 2);
        // The right face is missed once: still reported.
        let faces = hold(&mut seen, vec![f(104.0)], t0 + Duration::from_millis(100));
        assert_eq!(faces, vec![f(104.0), f(300.0)]);
        // A face that moved a little replaces its old self rather than doubling.
        let faces = hold(&mut seen, vec![f(108.0), f(302.0)], t0 + Duration::from_millis(200));
        assert_eq!(faces, vec![f(108.0), f(302.0)]);
        // Gone for longer than HOLD: dropped.
        let later = t0 + Duration::from_millis(200) + HOLD;
        assert_eq!(hold(&mut seen, vec![f(108.0)], later), vec![f(108.0)]);
    }

    /// Two copies of a portrait side by side: two faces, one in each half.
    #[test]
    #[ignore = "needs the face model and VCAM_TEST_FACE_IMAGE (a portrait photo)"]
    fn finds_two_faces_side_by_side() -> Result<()> {
        let path = std::env::var("VCAM_TEST_FACE_IMAGE")?;
        let one = imgproc::rgb_to_frame(&imgproc::load_rgb(std::path::Path::new(&path))?);
        let mut two = Frame::new(one.w * 2, one.h);
        for y in 0..one.h {
            let row = &one.data[y * one.w * 3..][..one.w * 3];
            two.data[y * two.w * 3..][..one.w * 3].copy_from_slice(row);
            two.data[(y * two.w + one.w) * 3..][..one.w * 3].copy_from_slice(row);
        }
        let faces = FaceDetector::new()?.detect(&two)?;
        println!("{faces:?}");
        assert_eq!(faces.len(), 2);
        let mut xs: Vec<f32> = faces.iter().map(|f| f.x).collect();
        xs.sort_by(f32::total_cmp);
        assert!(xs[0] < one.w as f32 && xs[1] > one.w as f32);
        assert!((xs[1] - xs[0] - one.w as f32).abs() < 4.0, "the copies' faces should be one width apart");
        Ok(())
    }
}

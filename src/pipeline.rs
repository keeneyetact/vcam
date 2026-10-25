//! Compositing + virtual-camera send on a worker thread, so that work overlaps
//! the *next* frame's inference instead of waiting behind it.
//!
//! The matte's inference is a serial wall -- each frame feeds the recurrent
//! state into the next -- but the engine's post-processing (guided upsample,
//! single-person isolation) and the compositing + send after it don't depend
//! on the next frame. Frames are handled in submit order and the output is
//! identical to doing it inline; the cost is one frame of latency.

use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{anyhow, bail, Result};

use crate::compositor::Compositor;
use crate::imgproc::{self, Frame};
use crate::rvm::Pending;
use crate::virtualcam::VirtualCam;

pub type OnOutput = Box<dyn Fn(&Arc<Frame>) + Send>;

struct Job {
    compositor: Arc<Compositor>,
    frame: Arc<Frame>,
    /// The engine's deferred post-processing; None == passthrough.
    pending: Option<Pending>,
    upscale: Option<(usize, usize)>,
}

pub struct OutputPipeline {
    tx: Option<SyncSender<Job>>,
    thread: Option<JoinHandle<()>>,
    latest: Arc<Mutex<Option<Arc<Frame>>>>,
    error: Arc<Mutex<Option<String>>>,
}

impl OutputPipeline {
    pub fn new(mut cam: Option<VirtualCam>, on_output: Option<OnOutput>) -> Result<Self> {
        // Small queue => bounded latency + back-pressure on inference.
        let (tx, rx) = mpsc::sync_channel::<Job>(2);
        let latest = Arc::new(Mutex::new(None));
        let error = Arc::new(Mutex::new(None::<String>));
        let (l, e) = (latest.clone(), error.clone());
        let thread = std::thread::Builder::new().name("output".into()).spawn(move || {
            for job in rx {
                if e.lock().unwrap().is_some() {
                    continue; // drain quietly so submit() never blocks
                }
                let r = (|| -> Result<()> {
                    let (frame, alpha) = match job.pending {
                        Some(p) => {
                            let (alpha, fg) = p.finish()?;
                            (Arc::new(fg), Some(alpha))
                        }
                        None => (job.frame, None),
                    };
                    let mut out = job.compositor.render(frame, alpha.as_deref());
                    if let Some(a) = alpha {
                        imgproc::FLOATS.give(a);
                    }
                    if let Some((w, h)) = job.upscale.filter(|&s| s != out.size()) {
                        // Cover-fit: crops rather than stretches if the camera
                        // couldn't deliver the output's aspect ratio.
                        out = Arc::new(imgproc::resize_cover(&out, w, h));
                    }
                    if let Some(cam) = cam.as_mut() {
                        // No extra pacing: the webcam already paces us, and pacing
                        // twice turns into stalls when timers get coarse.
                        cam.send(&out)?;
                    }
                    *l.lock().unwrap() = Some(out.clone());
                    if let Some(cb) = &on_output {
                        cb(&out);
                    }
                    Ok(())
                })();
                if let Err(err) = r {
                    *e.lock().unwrap() = Some(format!("{err:#}"));
                }
            }
            drop(cam); // close the virtual camera on this thread, after the last frame
        })?;
        Ok(Self { tx: Some(tx), thread: Some(thread), latest, error })
    }

    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }

    /// Hand one frame's render + send to the worker. Blocks (back-pressure) if
    /// the worker is two frames behind; fails if the worker has failed.
    pub fn submit(&self, compositor: Arc<Compositor>, frame: Arc<Frame>, pending: Option<Pending>,
                  upscale: Option<(usize, usize)>) -> Result<()> {
        if let Some(e) = self.error() {
            bail!(e);
        }
        self.tx
            .as_ref()
            .ok_or_else(|| anyhow!("output closed"))?
            .send(Job { compositor, frame, pending, upscale })
            .map_err(|_| anyhow!("output thread stopped"))
    }

    /// Most recently sent output frame; None until the first.
    pub fn latest(&self) -> Option<Arc<Frame>> {
        self.latest.lock().unwrap().clone()
    }
}

impl Drop for OutputPipeline {
    fn drop(&mut self) {
        self.tx = None; // drain + join the worker before the camera closes
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

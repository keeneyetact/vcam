//! RobustVideoMatting via ONNX Runtime — high-quality, temporally coherent
//! video matting. The recurrent state (r1..r4, fed back every frame) is what
//! keeps hair/hands/edges stable under motion, so no extra temporal smoothing
//! or guided-filter hacks are needed. Produces a soft alpha and a colour-
//! decontaminated foreground (clean hair edges, no old-background colour bleed).

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor};

use crate::imgproc::Frame;
use crate::kernels;
use crate::models::{self, REC_IN, TRUNK_OUTPUTS};

/// The process-wide ONNX Runtime environment.
pub fn ort_env() -> Result<&'static ort::environment::Environment> {
    static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("vcam").build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| anyhow!("onnxruntime failed to start: {e}"))
}

/// A good intra-op thread count for RVM: enough to use the fast cores, capped
/// because more threads *slow* this small recurrent net (on a 20-core
/// i7-14700, 28 threads = 77 ms vs 8 threads = 48 ms).
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(2, 8)
}

fn build_session(model: &Path, threads: usize) -> Result<Session> {
    let err = |e: ort::Error<_>| anyhow!("{e}");
    Session::builder(ort_env()?)?
        .with_intra_threads(threads)
        .map_err(err)?
        // Sleep between runs instead of spin-waiting. The camera paces us, so
        // each stage idles part of every frame, and spinning through that kept
        // ~20 of 28 logical CPUs busy (vs ~6.5) on an i7-14700 for the same
        // fps -- CPU the video-call app needs. Throughput is unchanged.
        .with_intra_op_spinning(false)
        .map_err(err)?
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(err)?
        .commit_from_file(model)
        .with_context(|| format!("loading {}", model.display()))
}

fn zero_state() -> Result<DynValue> {
    Ok(Tensor::from_array((vec![1i64, 1, 1, 1], vec![0f32]))?.into_dyn())
}

type Tensors = HashMap<String, Arc<DynValue>>;
type AB = (DynValue, DynValue);

enum StageData {
    /// The frame's (1,3,H,W) input, returned to the engine's buffer pool once read.
    Src(Tensor<f32>, Sender<Tensor<f32>>),
    Feats(Tensors),
}

struct Job {
    dsr: f32,
    data: StageData,
    reply: SyncSender<Result<AB>>,
}

/// RVM run as three models in a pipeline, one thread + onnxruntime session
/// each: encoder (state-free) | recurrent decoder core | decoder tail (state-
/// free). Frame N+2's encoder, N+1's core and N's tail run at the same time, so
/// throughput is set by the slowest stage rather than the sum: +14-24% fps end
/// to end on an i7-14700 (onnxruntime stops scaling past ~8 threads on this
/// model, so three sessions use a many-core CPU better). Output is identical
/// to the single model; the cost is two frames more latency. Each frame
/// carries its downsample_ratio, so the core resets the recurrent state exactly
/// at the frame where it changes.
pub struct StagePipeline {
    tx: Option<SyncSender<Job>>,
    threads: Vec<JoinHandle<()>>,
    error: Arc<Mutex<Option<String>>>,
}

impl StagePipeline {
    pub fn new(paths: &[std::path::PathBuf; 3], threads: [usize; 3]) -> Result<Self> {
        let mut sessions = Vec::new();
        for (p, &t) in paths.iter().zip(&threads) {
            sessions.push(build_session(p, t)?);
        }
        let error = Arc::new(Mutex::new(None::<String>));
        // Capacity 1: bounded latency, and back-pressure reaches the caller.
        let (tx0, mut rx) = mpsc::sync_channel::<Job>(1);
        let mut handles = Vec::new();
        for (k, mut sess) in sessions.into_iter().enumerate() {
            let (next_tx, next_rx) = mpsc::sync_channel::<Job>(1);
            let next = (k < 2).then_some(next_tx);
            let input_names: Vec<String> = sess.inputs().iter().map(|i| i.name().to_string()).collect();
            let output_names: Vec<String> = sess.outputs().iter().map(|o| o.name().to_string()).collect();
            let err = error.clone();
            let my_rx = std::mem::replace(&mut rx, next_rx);
            let h = std::thread::Builder::new().name(format!("rvm-stage-{k}")).spawn(move || {
                let mut rec: Option<Vec<Arc<DynValue>>> = None;
                let mut last_dsr = f32::NAN;
                while let Ok(job) = my_rx.recv() {
                    let reply = job.reply.clone();
                    if let Some(e) = err.lock().unwrap().clone() {
                        let _ = reply.send(Err(anyhow!(e))); // an earlier frame failed
                        continue;
                    }
                    let r = run_stage(k, &mut sess, &input_names, &output_names, &mut rec, &mut last_dsr, job);
                    match r {
                        Ok(Some(job)) => {
                            if let Some(n) = &next {
                                let _ = n.send(job);
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            *err.lock().unwrap() = Some(format!("{e:#}"));
                            let _ = reply.send(Err(e));
                        }
                    }
                }
            })?;
            handles.push(h);
        }
        Ok(Self { tx: Some(tx0), threads: handles, error })
    }

    /// Queue one frame's input; returns where its (A, b) will arrive. Blocks
    /// while the pipeline is full.
    fn submit(&self, src: Tensor<f32>, back: Sender<Tensor<f32>>, dsr: f32) -> Result<Receiver<Result<AB>>> {
        if let Some(e) = self.error.lock().unwrap().clone() {
            bail!(e);
        }
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx
            .as_ref()
            .context("pipeline closed")?
            .send(Job { dsr, data: StageData::Src(src, back), reply })
            .map_err(|_| anyhow!("RVM pipeline stopped"))?;
        Ok(rx)
    }
}

impl Drop for StagePipeline {
    fn drop(&mut self) {
        self.tx = None; // each stage exits when its input closes, closing the next
        for h in self.threads.drain(..) {
            let _ = h.join();
        }
    }
}

fn run_named<'s>(sess: &'s mut Session, names: &[String], data: &Tensors) -> Result<ort::session::SessionOutputs<'s>> {
    let inputs: Vec<(std::borrow::Cow<'_, str>, SessionInputValue<'_>)> = names
        .iter()
        .map(|n| {
            let v = data.get(n).with_context(|| format!("missing stage input {n}"))?;
            Ok((n.as_str().into(), SessionInputValue::from(&**v)))
        })
        .collect::<Result<_>>()?;
    Ok(sess.run(inputs)?)
}

fn take(out: &mut ort::session::SessionOutputs<'_>, name: &str) -> Result<DynValue> {
    out.remove(name).with_context(|| format!("model produced no {name}"))
}

fn run_stage(
    k: usize,
    sess: &mut Session,
    input_names: &[String],
    output_names: &[String],
    rec: &mut Option<Vec<Arc<DynValue>>>,
    last_dsr: &mut f32,
    mut job: Job,
) -> Result<Option<Job>> {
    match k {
        0 => {
            let StageData::Src(src, back) = std::mem::replace(&mut job.data, StageData::Feats(HashMap::new())) else {
                bail!("encoder expects the frame input");
            };
            let dsr = Tensor::from_array((vec![1i64], vec![job.dsr]))?;
            let mut out = sess.run(ort::inputs!["src" => &src, "downsample_ratio" => &dsr])?;
            let mut feats = Tensors::new();
            for n in output_names {
                feats.insert(n.clone(), Arc::new(take(&mut out, n)?));
            }
            drop(out);
            let _ = back.send(src);
            job.data = StageData::Feats(feats);
            Ok(Some(job))
        }
        1 => {
            let StageData::Feats(data) = &mut job.data else { bail!("core expects features") };
            if job.dsr != *last_dsr || rec.is_none() {
                // New coarse size: the old state can't carry over.
                *rec = Some((0..4).map(|_| zero_state().map(Arc::new)).collect::<Result<_>>()?);
                *last_dsr = job.dsr;
            }
            for (n, v) in REC_IN.iter().zip(rec.as_ref().unwrap()) {
                data.insert(n.to_string(), v.clone());
            }
            let mut out = run_named(sess, input_names, data)?;
            let mut new_rec = Vec::new();
            for (i, n) in output_names.iter().enumerate() {
                let v = Arc::new(take(&mut out, n)?);
                if i < 4 {
                    new_rec.push(v.clone());
                }
                data.insert(n.clone(), v);
            }
            for n in REC_IN {
                data.remove(n);
            }
            *rec = Some(new_rec);
            Ok(Some(job))
        }
        _ => {
            let StageData::Feats(data) = &job.data else { bail!("tail expects features") };
            let mut out = run_named(sess, input_names, data)?;
            let ab = (take(&mut out, TRUNK_OUTPUTS[0])?, take(&mut out, TRUNK_OUTPUTS[1])?);
            let _ = job.reply.send(Ok(ab));
            Ok(None)
        }
    }
}

enum Backend {
    Staged {
        pipe: StagePipeline,
        pool_tx: Sender<Tensor<f32>>,
        pool_rx: Receiver<Tensor<f32>>,
    },
    /// The stock model in one session (fallback if the stages can't be built).
    Full { sess: Session, rec: Option<Vec<DynValue>>, rec_dsr: f32 },
}

/// One frame's inference in flight; `finish` turns it into (alpha, foreground).
pub enum Pending {
    Staged { rx: Receiver<Result<AB>>, frame: Arc<Frame>, single: bool },
    Full { fgr: DynValue, pha: DynValue, single: bool },
}

impl Pending {
    /// Raw model outputs -> (alpha HxW in [0, 1], foreground BGR). Runs on the
    /// output thread, overlapping the next frame's inference.
    pub fn finish(self) -> Result<(Vec<f32>, Frame)> {
        match self {
            Pending::Staged { rx, frame, single } => {
                let (a, b) = rx.recv_timeout(Duration::from_secs(10)).map_err(|_| anyhow!("RVM timed out"))??;
                let (shape, av) = a.try_extract_tensor::<f32>()?;
                let (_, bv) = b.try_extract_tensor::<f32>()?;
                let (ch, cw) = match **shape {
                    [1, 4, ch, cw] => (ch as usize, cw as usize),
                    _ => bail!("unexpected guided-filter coefficient shape {shape:?}"),
                };
                let (mut alpha, fg) = kernels::guided_upsample(av, bv, ch, cw, &frame);
                if single {
                    kernels::keep_central(&mut alpha, frame.h, frame.w);
                }
                Ok((alpha, fg))
            }
            Pending::Full { fgr, pha, single } => {
                let (shape, p) = pha.try_extract_tensor::<f32>()?;
                let (h, w) = (shape[2] as usize, shape[3] as usize);
                let mut alpha = p.to_vec();
                if single {
                    kernels::keep_central(&mut alpha, h, w);
                }
                let (_, f) = fgr.try_extract_tensor::<f32>()?;
                Ok((alpha, kernels::fgr_to_bgr(f, h, w)))
            }
        }
    }
}

pub struct RvmEngine {
    pub w: usize,
    pub h: usize,
    /// downsample_ratio: the coarse network's resolution, which dominates speed.
    pub dsr: f32,
    pub threads: usize,
    pub single: bool,
    backend: Backend,
}

impl RvmEngine {
    pub fn new(w: usize, h: usize, downsample: Option<f32>, single_person: bool, threads: Option<usize>) -> Result<Self> {
        let threads = threads.filter(|&t| t > 0).unwrap_or_else(default_threads);
        // RVM's recommended ~0.375 (0.25 for >=1080p) keeps the refined full-res
        // edges sharp while running fast.
        let dsr = downsample.unwrap_or(if w.max(h) >= 1600 { 0.25 } else { 0.375 });
        let backend = match models::ensure_rvm_stages()
            .and_then(|p| StagePipeline::new(&p, [(threads / 2).max(2), threads, threads]))
        {
            Ok(pipe) => {
                let (pool_tx, pool_rx) = mpsc::channel();
                Backend::Staged { pipe, pool_tx, pool_rx }
            }
            Err(e) => {
                crate::log!("RVM stage pipeline unavailable ({e:#}); using the full model");
                let full = models::rvm_model_bytes()?;
                let path = crate::paths::writable_model_dir().join(models::RVM_FILE);
                if models::find_model(models::RVM_FILE).is_none() {
                    std::fs::create_dir_all(path.parent().unwrap())?;
                    std::fs::write(&path, &full)?;
                }
                let path = models::find_model(models::RVM_FILE).unwrap_or(path);
                Backend::Full { sess: build_session(&path, threads)?, rec: None, rec_dsr: dsr }
            }
        };
        Ok(Self { w, h, dsr, threads, single: single_person, backend })
    }

    pub fn staged(&self) -> bool {
        matches!(self.backend, Backend::Staged { .. })
    }

    /// The serial half: input conversion + inference. Only this has to run
    /// frame by frame in order (it carries the recurrent state); `finish` can
    /// run on another thread, so the next inference starts while this frame is
    /// still being post-processed.
    pub fn infer(&mut self, frame: Arc<Frame>) -> Result<Pending> {
        let n = 3 * frame.h * frame.w;
        let shape = [1i64, 3, frame.h as i64, frame.w as i64];
        match &mut self.backend {
            Backend::Staged { pipe, pool_tx, pool_rx } => {
                // Reuse a returned input buffer of the right size if there is one
                // (each in-flight frame needs its own).
                let mut src = match pool_rx.try_recv() {
                    Ok(t) if **t.shape() == shape => t,
                    _ => Tensor::from_array((shape.to_vec(), vec![0f32; n]))?,
                };
                kernels::fill_src(&frame, src.extract_tensor_mut().1);
                let rx = pipe.submit(src, pool_tx.clone(), self.dsr)?;
                Ok(Pending::Staged { rx, frame, single: self.single })
            }
            Backend::Full { sess, rec, rec_dsr } => {
                let mut src = vec![0f32; n];
                kernels::fill_src(&frame, &mut src);
                let src = Tensor::from_array((shape.to_vec(), src))?;
                if rec.is_none() || *rec_dsr != self.dsr {
                    *rec = Some((0..4).map(|_| zero_state()).collect::<Result<_>>()?);
                    *rec_dsr = self.dsr;
                }
                let r = rec.as_ref().unwrap();
                let dsr = Tensor::from_array((vec![1i64], vec![self.dsr]))?;
                let mut out = sess.run(ort::inputs![
                    "src" => &src, "r1i" => &r[0], "r2i" => &r[1], "r3i" => &r[2], "r4i" => &r[3],
                    "downsample_ratio" => &dsr
                ])?;
                let fgr = take(&mut out, "fgr")?;
                let pha = take(&mut out, "pha")?;
                let new_rec = models::REC_OUT.iter().map(|n| take(&mut out, n)).collect::<Result<Vec<_>>>()?;
                drop(out);
                *rec = Some(new_rec);
                Ok(Pending::Full { fgr, pha, single: self.single })
            }
        }
    }

    pub fn process(&mut self, frame: Arc<Frame>) -> Result<(Vec<f32>, Frame)> {
        self.infer(frame)?.finish()
    }

    /// Live-tune the quality/speed dial. downsample_ratio is a per-frame model
    /// input, so the change takes effect on the next frame with no session
    /// rebuild (the recurrent state is reset at that frame).
    pub fn set_downsample(&mut self, dsr: f32) {
        self.dsr = dsr;
    }

    /// Live toggle for keeping only the person nearest the centre.
    pub fn set_single_person(&mut self, single: bool) {
        self.single = single;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The staged pipeline + Rust guided-filter tail must match the stock
    /// model run in one session, to within float rounding.
    #[test]
    #[ignore = "needs the RVM model (bundled at build time or downloaded); slow"]
    fn staged_matches_full_model() -> Result<()> {
        let (w, h) = (320, 192);
        // Build the stages here (not whatever is cached) so the test covers it.
        let dir = std::env::temp_dir().join("vcam_test_models");
        std::fs::create_dir_all(&dir)?;
        let bytes = models::rvm_model_bytes()?;
        let full_path = dir.join(models::RVM_FILE);
        std::fs::write(&full_path, &bytes)?;
        let stage_paths = models::STAGE_FILES.map(|f| dir.join(f));
        for (p, b) in stage_paths.iter().zip(models::build_stages(&bytes)?) {
            std::fs::write(p, b)?;
        }
        let engine = |backend| RvmEngine { w, h, dsr: 0.375, threads: 4, single: false, backend };
        let (pool_tx, pool_rx) = mpsc::channel();
        let mut staged = engine(Backend::Staged { pipe: StagePipeline::new(&stage_paths, [2, 4, 4])?, pool_tx, pool_rx });
        let mut full = engine(Backend::Full { sess: build_session(&full_path, 4)?, rec: None, rec_dsr: 0.375 });
        let mut seed = 7u32;
        for _ in 0..3 {
            let mut f = Frame::new(w, h);
            for (i, v) in f.data.iter_mut().enumerate() {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // A smooth gradient plus noise, so the matte isn't all zero.
                *v = ((i / 3 % w) as u32 * 255 / w as u32 / 2 + (seed >> 26)) as u8;
            }
            let f = Arc::new(f);
            let (a1, f1) = staged.process(f.clone())?;
            let (a2, f2) = full.process(f)?;
            let da = a1.iter().zip(&a2).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            let df = f1.data.iter().zip(&f2.data).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
            assert!(da < 1e-4, "alpha differs by {da}");
            assert!(df <= 1, "foreground differs by {df} levels");
        }
        Ok(())
    }
}

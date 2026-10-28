//! The RobustVideoMatting model: locating / downloading it, and splitting it
//! into the three pipeline stages vcam runs.
//!
//! RVM is cut just before its deep-guided-filter tail: the stages output the
//! coarse guided-filter coefficients A (tensor "777") and b ("779") instead of
//! fgr/pha, and `kernels::guided_upsample` does the full-resolution tail -- two
//! full-res Resize ops plus elementwise passes, ~20% of RVM's time at 720p --
//! in one fused pass on the output thread. That truncated model is then split
//! at the recurrent state into encoder | core | tail (see `rvm::StagePipeline`).
//! The stage models are built once from the full model and cached.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::onnx_graph::Model;
use crate::paths;

/// RobustVideoMatting — real-time video matting with temporal coherence.
/// Free/open-source (PeterL1n/RobustVideoMatting).
pub const RVM_FILE: &str = "rvm_mobilenetv3_fp32.onnx";
pub const RVM_URL: &str =
    "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx";

pub const TRUNK_OUTPUTS: [&str; 6] = ["777", "779", "r1o", "r2o", "r3o", "r4o"];
pub const REC_IN: [&str; 4] = ["r1i", "r2i", "r3i", "r4i"];
pub const REC_OUT: [&str; 4] = ["r1o", "r2o", "r3o", "r4o"];
pub const STAGE_FILES: [&str; 3] = [
    "rvm_mobilenetv3_fp32_enc.onnx",
    "rvm_mobilenetv3_fp32_core.onnx",
    "rvm_mobilenetv3_fp32_tail.onnx",
];

/// YuNet face detector (OpenCV Zoo, MIT), for telling people apart in
/// single-person mode. Fixed 640x640 input; ~230 KB.
pub const FACE_FILE: &str = "face_detection_yunet_2023mar.onnx";
pub const FACE_URL: &str =
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx";

#[cfg(embedded_model)]
const EMBEDDED: Option<&[u8]> = Some(include_bytes!(env!("VCAM_EMBEDDED_MODEL")));
#[cfg(not(embedded_model))]
const EMBEDDED: Option<&[u8]> = None;

#[cfg(embedded_face_model)]
const EMBEDDED_FACE: Option<&[u8]> = Some(include_bytes!(env!("VCAM_EMBEDDED_FACE_MODEL")));
#[cfg(not(embedded_face_model))]
const EMBEDDED_FACE: Option<&[u8]> = None;

pub fn find_model(file: &str) -> Option<PathBuf> {
    paths::model_dirs()
        .into_iter()
        .map(|d| d.join(file))
        .find(|p| std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() > 0))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Bytes of the full RVM model: bundled in the binary, a cached / local copy,
/// or downloaded on first use.
pub fn rvm_model_bytes() -> Result<Vec<u8>> {
    if let Some(bytes) = EMBEDDED {
        return Ok(bytes.to_vec());
    }
    if let Some(p) = find_model(RVM_FILE) {
        return Ok(std::fs::read(p)?);
    }
    download("RobustVideoMatting", RVM_URL, RVM_FILE)
}

/// Bytes of the face-detection model, found the same way as RVM's.
pub fn face_model_bytes() -> Result<Vec<u8>> {
    if let Some(bytes) = EMBEDDED_FACE {
        return Ok(bytes.to_vec());
    }
    if let Some(p) = find_model(FACE_FILE) {
        return Ok(std::fs::read(p)?);
    }
    download("face detection", FACE_URL, FACE_FILE)
}

fn download(what: &str, url: &str, file: &str) -> Result<Vec<u8>> {
    let path = paths::writable_model_dir().join(file);
    crate::log!("Downloading {what} model -> {} ...", path.display());
    let bytes = ureq::get(url)
        .call()
        .and_then(|r| r.into_body().with_config().limit(200 << 20).read_to_vec())
        .with_context(|| format!("Failed to download model from {url}"))?;
    write_atomic(&path, &bytes).context("Failed to save the downloaded model")?;
    crate::log!("Model ready.");
    Ok(bytes)
}

/// Paths of the encoder / core / tail stage models, building them if needed.
pub fn ensure_rvm_stages() -> Result<[PathBuf; 3]> {
    if let [Some(a), Some(b), Some(c)] = STAGE_FILES.map(find_model) {
        return Ok([a, b, c]);
    }
    let stages = build_stages(&rvm_model_bytes()?)?;
    let paths = STAGE_FILES.map(|f| paths::writable_model_dir().join(f));
    for (p, bytes) in paths.iter().zip(&stages) {
        write_atomic(p, bytes).with_context(|| format!("Failed to save {}", p.display()))?;
    }
    Ok(paths)
}

/// The tensors at the stage boundaries, from the truncated model's graph.
pub struct Split {
    pub enc_out: Vec<String>,
    pub core_out: Vec<String>,
    pub tail_in: Vec<String>,
}

fn push_unique(v: &mut Vec<String>, s: &str) {
    if !v.iter().any(|x| x == s) {
        v.push(s.to_string());
    }
}

/// Split the truncated model at the recurrent state: the encoder is the
/// state-free part, the core the stateful nodes the next frame's state needs,
/// and the tail the rest of A/b.
pub fn split(trunk: &Model) -> Split {
    let init = &trunk.initializers;
    let prod = trunk.producers();
    // Tensors that depend on the recurrent state (the graph is topologically sorted).
    let mut dep: HashSet<String> = REC_IN.iter().map(|s| s.to_string()).collect();
    for n in &trunk.nodes {
        if n.inputs.iter().any(|i| dep.contains(i)) {
            dep.extend(n.outputs.iter().cloned());
        }
    }
    let stateful: Vec<usize> =
        (0..trunk.nodes.len()).filter(|&i| trunk.nodes[i].outputs.iter().any(|o| dep.contains(o))).collect();
    // Encoder outputs = what stateful nodes read from the state-free part.
    let mut enc_out = Vec::new();
    for &i in &stateful {
        for t in &trunk.nodes[i].inputs {
            if !t.is_empty() && !dep.contains(t) && !init.contains(t) {
                push_unique(&mut enc_out, t);
            }
        }
    }
    let mut core = HashSet::new();
    let mut stack: Vec<String> = REC_OUT.iter().map(|s| s.to_string()).collect();
    while let Some(t) = stack.pop() {
        if let Some(&i) = prod.get(t.as_str()) {
            let n = &trunk.nodes[i];
            if !core.contains(&i) && n.outputs.iter().any(|o| dep.contains(o)) {
                core.insert(i);
                stack.extend(n.inputs.iter().cloned());
            }
        }
    }
    let tail: Vec<usize> = stateful.into_iter().filter(|i| !core.contains(i)).collect();
    let tail_made: HashSet<&str> =
        tail.iter().flat_map(|&i| trunk.nodes[i].outputs.iter().map(String::as_str)).collect();
    let mut tail_in = Vec::new();
    for &i in &tail {
        for t in &trunk.nodes[i].inputs {
            if !t.is_empty() && !init.contains(t) && !tail_made.contains(t.as_str()) {
                push_unique(&mut tail_in, t);
            }
        }
    }
    let mut core_out: Vec<String> = REC_OUT.iter().map(|s| s.to_string()).collect();
    for t in &tail_in {
        if prod.get(t.as_str()).is_some_and(|i| core.contains(i)) && !REC_OUT.contains(&t.as_str()) {
            core_out.push(t.clone());
        }
    }
    Split { enc_out, core_out, tail_in }
}

fn onnx_elem_type(t: ort::value::TensorElementType) -> Option<i32> {
    use ort::value::TensorElementType as T;
    Some(match t {
        T::Float32 => 1,
        T::Uint8 => 2,
        T::Int8 => 3,
        T::Uint16 => 4,
        T::Int16 => 5,
        T::Int32 => 6,
        T::Int64 => 7,
        T::String => 8,
        T::Bool => 9,
        T::Float16 => 10,
        T::Float64 => 11,
        T::Uint32 => 12,
        T::Uint64 => 13,
        _ => return None,
    })
}

/// Element types of intermediate tensors, as onnxruntime infers them: load the
/// model with those tensors added as (untyped) graph outputs and read back.
fn infer_types(model: &Model, names: &[String]) -> Result<HashMap<String, i32>> {
    let probe = model.with_extra_outputs(names)?;
    let session = ort::session::Session::builder(crate::rvm::ort_env()?)?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Disable)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .commit_from_memory(&probe)?;
    let mut types = HashMap::new();
    for o in session.outputs() {
        if let ort::value::ValueType::Tensor { ty, .. } = o.dtype() {
            if let Some(t) = onnx_elem_type(*ty) {
                types.insert(o.name().to_string(), t);
            }
        }
    }
    Ok(types)
}

/// Build the three stage models from the full RVM model.
pub fn build_stages(full_bytes: &[u8]) -> Result<[Vec<u8>; 3]> {
    let full = Model::parse(full_bytes)?;
    let trunk_out: Vec<String> = TRUNK_OUTPUTS.iter().map(|s| s.to_string()).collect();
    let trunk_bytes = full.extract(&full.input_names, &trunk_out, &HashMap::new())?;
    let trunk = Model::parse(&trunk_bytes)?;
    let s = split(&trunk);

    let mut boundary = s.enc_out.clone();
    for t in s.core_out.iter().chain(&s.tail_in) {
        push_unique(&mut boundary, t);
    }
    let types = infer_types(&trunk, &boundary)?;

    let strs = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let enc_in = strs(&["src", "downsample_ratio"]);
    let mut core_in = s.enc_out.clone();
    core_in.extend(strs(&REC_IN));
    Ok([
        trunk.extract(&enc_in, &s.enc_out, &types)?,
        trunk.extract(&core_in, &s.core_out, &types)?,
        trunk.extract(&s.tail_in, &trunk_out[..2], &types)?,
    ])
}

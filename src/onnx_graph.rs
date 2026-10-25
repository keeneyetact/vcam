//! Lossless ONNX sub-graph extraction, working on the protobuf wire format.
//!
//! This is `onnx.utils.extract_model` for what vcam needs: keep the nodes that
//! compute `outputs` from `inputs`, the initializers they read, and new graph
//! inputs/outputs. Only the handful of fields that matter are decoded (node
//! inputs/outputs, initializer names, value-info names); every other byte of
//! the model is copied through untouched, so nothing a hand-written schema
//! forgot to declare can be lost.

use std::collections::{HashMap, HashSet};

use anyhow::{bail, Context, Result};

// Field numbers from onnx.proto.
const MODEL_GRAPH: u32 = 7;
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_VALUE_INFO: u32 = 13;
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const TENSOR_NAME: u32 = 8;
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;
const TYPE_TENSOR: u32 = 1;
const TENSOR_TYPE_ELEM: u32 = 1;

/// onnx TensorProto.DataType values used here.
pub const FLOAT: i32 = 1;
pub const INT64: i32 = 7;

/// One top-level field of a message: its number, the whole encoded field
/// (key included, for copying through) and, if length-delimited, the payload.
struct Field<'a> {
    num: u32,
    raw: &'a [u8],
    payload: Option<&'a [u8]>,
}

fn varint(buf: &[u8], pos: &mut usize) -> Result<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *buf.get(*pos).context("truncated protobuf varint")?;
        *pos += 1;
        v |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(v);
        }
    }
    bail!("protobuf varint too long")
}

fn fields(buf: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let start = pos;
        let key = varint(buf, &mut pos)?;
        let (num, wire) = ((key >> 3) as u32, key & 7);
        let payload = match wire {
            0 => {
                varint(buf, &mut pos)?;
                None
            }
            1 => {
                pos += 8;
                None
            }
            2 => {
                let len = varint(buf, &mut pos)? as usize;
                let p = buf.get(pos..pos + len).context("truncated protobuf field")?;
                pos += len;
                Some(p)
            }
            5 => {
                pos += 4;
                None
            }
            _ => bail!("unsupported protobuf wire type {wire}"),
        };
        if pos > buf.len() {
            bail!("truncated protobuf field");
        }
        out.push(Field { num, raw: &buf[start..pos], payload });
    }
    Ok(out)
}

fn strings(msg: &[u8], num: u32) -> Result<Vec<String>> {
    fields(msg)?
        .into_iter()
        .filter(|f| f.num == num)
        .map(|f| Ok(String::from_utf8(f.payload.context("string field")?.to_vec())?))
        .collect()
}

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_bytes(out: &mut Vec<u8>, num: u32, payload: &[u8]) {
    put_varint(out, ((num as u64) << 3) | 2);
    put_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

/// ValueInfoProto { name, type: tensor(elem_type) } -- no shape, which
/// onnxruntime accepts (and infers) for every tensor.
fn value_info(name: &str, elem_type: Option<i32>) -> Vec<u8> {
    let mut vi = Vec::new();
    put_bytes(&mut vi, VALUE_INFO_NAME, name.as_bytes());
    if let Some(t) = elem_type {
        let mut tensor = Vec::new();
        put_varint(&mut tensor, (TENSOR_TYPE_ELEM as u64) << 3);
        put_varint(&mut tensor, t as u64);
        let mut ty = Vec::new();
        put_bytes(&mut ty, TYPE_TENSOR, &tensor);
        put_bytes(&mut vi, VALUE_INFO_TYPE, &ty);
    }
    vi
}

pub struct Node {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// A parsed model: graph nodes in (topological) file order plus what's needed
/// to re-emit it.
pub struct Model<'a> {
    bytes: &'a [u8],
    pub nodes: Vec<Node>,
    node_raw: Vec<&'a [u8]>,
    pub initializers: HashSet<String>,
    /// Existing graph input/output ValueInfoProtos by name (they carry shapes).
    io_info: HashMap<String, &'a [u8]>,
    pub input_names: Vec<String>,
}

impl<'a> Model<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        let graph = fields(bytes)?
            .into_iter()
            .find(|f| f.num == MODEL_GRAPH)
            .and_then(|f| f.payload)
            .context("ONNX model has no graph")?;
        let mut m = Model {
            bytes,
            nodes: Vec::new(),
            node_raw: Vec::new(),
            initializers: HashSet::new(),
            io_info: HashMap::new(),
            input_names: Vec::new(),
        };
        for f in fields(graph)? {
            let p = f.payload.unwrap_or_default();
            match f.num {
                GRAPH_NODE => {
                    m.nodes.push(Node { inputs: strings(p, NODE_INPUT)?, outputs: strings(p, NODE_OUTPUT)? });
                    m.node_raw.push(p);
                }
                GRAPH_INITIALIZER => {
                    if let Some(n) = strings(p, TENSOR_NAME)?.pop() {
                        m.initializers.insert(n);
                    }
                }
                GRAPH_INPUT | GRAPH_OUTPUT => {
                    if let Some(n) = strings(p, VALUE_INFO_NAME)?.pop() {
                        if f.num == GRAPH_INPUT {
                            m.input_names.push(n.clone());
                        }
                        m.io_info.insert(n, p);
                    }
                }
                _ => {}
            }
        }
        Ok(m)
    }

    /// Node index producing each tensor.
    pub fn producers(&self) -> HashMap<&str, usize> {
        let mut p = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            for o in &n.outputs {
                p.insert(o.as_str(), i);
            }
        }
        p
    }

    /// The model with extra graph outputs (declared without a type, which
    /// onnxruntime fills in) -- used to learn intermediate tensors' types.
    pub fn with_extra_outputs(&self, names: &[String]) -> Result<Vec<u8>> {
        let graph = self.rewrite_graph(None, None, Some(names), None)?;
        self.with_graph(&graph)
    }

    /// Sub-model computing `outputs` from `inputs`. `types` gives the element
    /// type of tensors that aren't already graph inputs/outputs of this model.
    pub fn extract(&self, inputs: &[String], outputs: &[String], types: &HashMap<String, i32>) -> Result<Vec<u8>> {
        let prod = self.producers();
        let stop: HashSet<&str> = inputs.iter().map(String::as_str).collect();
        let mut keep = vec![false; self.nodes.len()];
        let mut stack: Vec<&str> = outputs.iter().map(String::as_str).collect();
        let mut seen = HashSet::new();
        while let Some(t) = stack.pop() {
            if t.is_empty() || stop.contains(t) || !seen.insert(t) {
                continue;
            }
            match prod.get(t) {
                Some(&i) => {
                    if !keep[i] {
                        keep[i] = true;
                        stack.extend(self.nodes[i].inputs.iter().map(String::as_str));
                    }
                }
                None if self.initializers.contains(t) => {}
                None => bail!("tensor {t} is needed but neither computed nor an input"),
            }
        }
        let used: HashSet<&str> = self
            .nodes
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .flat_map(|(n, _)| n.inputs.iter().map(String::as_str))
            .collect();
        // Inputs need an element type; outputs may go without (inferred).
        let info = |name: &str, need_type: bool| -> Result<Vec<u8>> {
            if let Some(raw) = self.io_info.get(name) {
                return Ok(raw.to_vec());
            }
            let t = types.get(name).copied();
            if need_type && t.is_none() {
                bail!("no type known for tensor {name}");
            }
            Ok(value_info(name, t))
        };
        let ins = inputs.iter().map(|n| info(n, true)).collect::<Result<Vec<_>>>()?;
        let outs = outputs.iter().map(|n| info(n, false)).collect::<Result<Vec<_>>>()?;
        let graph = self.rewrite_graph(Some(&keep), Some(&used), None, Some((ins, outs)))?;
        self.with_graph(&graph)
    }

    fn rewrite_graph(
        &self,
        keep: Option<&[bool]>,
        used_inits: Option<&HashSet<&str>>,
        extra_outputs: Option<&[String]>,
        io: Option<(Vec<Vec<u8>>, Vec<Vec<u8>>)>,
    ) -> Result<Vec<u8>> {
        let graph = fields(self.bytes)?
            .into_iter()
            .find(|f| f.num == MODEL_GRAPH)
            .and_then(|f| f.payload)
            .context("ONNX model has no graph")?;
        let mut out = Vec::with_capacity(graph.len());
        let (mut node_i, mut wrote_in, mut wrote_out) = (0usize, false, false);
        let emit = |out: &mut Vec<u8>, num: u32, list: &[Vec<u8>]| {
            for v in list {
                put_bytes(out, num, v);
            }
        };
        for f in fields(graph)? {
            let p = f.payload.unwrap_or_default();
            match f.num {
                GRAPH_NODE => {
                    if keep.map_or(true, |k| k[node_i]) {
                        out.extend_from_slice(f.raw);
                    }
                    node_i += 1;
                }
                GRAPH_INITIALIZER => {
                    let name = strings(p, TENSOR_NAME)?.pop().unwrap_or_default();
                    if used_inits.map_or(true, |u| u.contains(name.as_str())) {
                        out.extend_from_slice(f.raw);
                    }
                }
                GRAPH_INPUT => match &io {
                    Some((ins, _)) => {
                        if !wrote_in {
                            emit(&mut out, GRAPH_INPUT, ins);
                            wrote_in = true;
                        }
                    }
                    None => out.extend_from_slice(f.raw),
                },
                GRAPH_OUTPUT => match &io {
                    Some((_, outs)) => {
                        if !wrote_out {
                            emit(&mut out, GRAPH_OUTPUT, outs);
                            wrote_out = true;
                        }
                    }
                    None => out.extend_from_slice(f.raw),
                },
                // Shape annotations of tensors that may no longer exist; the
                // runtime re-infers them.
                GRAPH_VALUE_INFO if io.is_some() => {}
                _ => out.extend_from_slice(f.raw),
            }
        }
        if let Some((ins, outs)) = &io {
            if !wrote_in {
                emit(&mut out, GRAPH_INPUT, ins);
            }
            if !wrote_out {
                emit(&mut out, GRAPH_OUTPUT, outs);
            }
        }
        for n in extra_outputs.unwrap_or_default() {
            put_bytes(&mut out, GRAPH_OUTPUT, &value_info(n, None));
        }
        Ok(out)
    }

    fn with_graph(&self, graph: &[u8]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.bytes.len());
        for f in fields(self.bytes)? {
            if f.num == MODEL_GRAPH {
                put_bytes(&mut out, MODEL_GRAPH, graph);
            } else {
                out.extend_from_slice(f.raw);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 1 << 40, u64::MAX] {
            let mut b = Vec::new();
            put_varint(&mut b, v);
            let mut pos = 0;
            assert_eq!(varint(&b, &mut pos).unwrap(), v);
            assert_eq!(pos, b.len());
        }
    }

    #[test]
    fn value_info_parses_back() {
        let vi = value_info("abc", Some(INT64));
        assert_eq!(strings(&vi, VALUE_INFO_NAME).unwrap(), vec!["abc".to_string()]);
    }
}

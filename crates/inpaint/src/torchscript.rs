//! Tensors of a TorchScript archive (`torch.jit.save`), read without Python.
//!
//! A TorchScript `.pt` is a zip holding `<name>/data.pkl` (the module tree as a pickle, its
//! tensors as persistent references) and `<name>/data/<n>` (raw storages). The module tree is
//! walked into `state_dict`-style dotted names (`generator.model.1.ffc.convl2l.weight`).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{DType, Device, Tensor};

use crate::{Error, Result};

/// Deepest module nesting walked (LaMa needs 6).
const MAX_DEPTH: usize = 32;
/// Most tensors read from one archive.
const MAX_TENSORS: usize = 100_000;

pub struct Archive {
    path: PathBuf,
    tensors: HashMap<String, TensorInfo>,
}

fn walk(o: Object, prefix: &str, dir: &Path, depth: usize, out: &mut Vec<TensorInfo>) {
    if depth > MAX_DEPTH || out.len() >= MAX_TENSORS {
        return;
    }
    match o {
        Object::Build { args, .. } => walk(*args, prefix, dir, depth + 1, out),
        Object::Dict(kv) => {
            for (k, v) in kv {
                let Object::Unicode(name) = &k else { continue };
                let full = if prefix.is_empty() { name.clone() } else { format!("{prefix}.{name}") };
                if let Ok(Some(t)) = v.clone().into_tensor_info(Object::Unicode(full.clone()), dir) {
                    out.push(t);
                    continue;
                }
                walk(v, &full, dir, depth + 1, out);
            }
        }
        _ => {}
    }
}

impl Archive {
    pub fn open(path: &Path) -> Result<Self> {
        let err = |e: &dyn std::fmt::Display| Error::Model(format!("{}: {e}", path.display()));
        let file = File::open(path).map_err(|e| err(&e))?;
        let mut zip = zip::ZipArchive::new(BufReader::new(file)).map_err(|e| err(&e))?;
        let pkl = zip.file_names().find(|n| n.ends_with("/data.pkl")).map(str::to_owned).ok_or_else(|| err(&"not a TorchScript archive (no data.pkl)"))?;
        let dir = PathBuf::from(pkl.trim_end_matches(".pkl"));
        let reader = zip.by_name(&pkl).map_err(|e| err(&e))?;
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(reader)).map_err(|e| err(&e))?;
        let root = stack.finalize().map_err(|e| err(&e))?;
        let mut infos = Vec::new();
        walk(root, "", &dir, 0, &mut infos);
        if infos.is_empty() {
            return Err(err(&"no tensors found"));
        }
        Ok(Self { path: path.to_owned(), tensors: infos.into_iter().map(|t| (t.name.clone(), t)).collect() })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Tensor `name` (contiguous F32 only, as LaMa's are), on `device`.
    pub fn get(&self, name: &str, device: &Device) -> Result<Tensor> {
        let info = self.tensors.get(name).ok_or_else(|| Error::Model(format!("missing tensor {name}")))?;
        if info.dtype != DType::F32 || !info.layout.is_contiguous() {
            return Err(Error::Model(format!("{name}: unsupported layout or dtype {:?}", info.dtype)));
        }
        let file = File::open(&self.path).map_err(|e| Error::Model(format!("{}: {e}", self.path.display())))?;
        let mut zip = zip::ZipArchive::new(BufReader::new(file)).map_err(|e| Error::Model(e.to_string()))?;
        let mut r = zip.by_name(&info.path).map_err(|e| Error::Model(format!("{name}: {e}")))?;
        let skip = info.layout.start_offset().saturating_mul(4) as u64;
        std::io::copy(&mut r.by_ref().take(skip), &mut std::io::sink()).map_err(|e| Error::Model(e.to_string()))?;
        let n = info.layout.shape().elem_count();
        let mut bytes = vec![0u8; n.checked_mul(4).ok_or_else(|| Error::Model(format!("{name}: too large")))?];
        r.read_exact(&mut bytes).map_err(|e| Error::Model(format!("{name}: {e} (damaged download?)")))?;
        let data: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        if data.iter().any(|v| !v.is_finite()) {
            return Err(Error::Model(format!("{name}: non-finite weights")));
        }
        Ok(Tensor::from_vec(data, info.layout.shape().clone(), device)?)
    }
}

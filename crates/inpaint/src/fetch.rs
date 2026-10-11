//! Downloading the LaMa model, only when the user asks for AI Remove. The downloader itself
//! (mirrors, resuming, checks, the pure-Rust HTTPS client) is the `lightcraft-fetch` crate.

pub use lightcraft_fetch::*;

use std::path::{Path, PathBuf};

/// File name of the model in its folder.
pub const WEIGHTS_FILE: &str = "big-lama.pt";

/// Size of the big-lama TorchScript archive (206 MB).
pub const WEIGHTS_SIZE: u64 = 205_669_692;

/// The model, pinned by size and SHA-256 (the TorchScript export of big-lama published with the
/// Apache-2.0 IOPaint project).
pub const FILES: &[FileSpec] = &[FileSpec {
    name: WEIGHTS_FILE,
    size: Some(WEIGHTS_SIZE),
    sha256: Some("344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"),
    max: WEIGHTS_SIZE,
}];

/// Where the model is downloaded from (`<base>/big-lama.pt`).
pub const DEFAULT_MIRRORS: &[&str] = &["https://github.com/Sanster/models/releases/download/add_big_lama"];

/// Environment variable with extra mirrors, tried before the defaults.
pub const MIRRORS_ENV: &str = "LIGHTCRAFT_LAMA_MIRRORS";

pub fn mirrors(env: Option<&str>) -> Vec<String> {
    lightcraft_fetch::mirrors(env, None, DEFAULT_MIRRORS)
}

/// The folder LightCraft keeps the model in: `<models dir>/lama`.
pub fn model_dir(models: &Path) -> PathBuf {
    models.join("lama")
}

/// The weights file, when it is installed in `dir`.
pub fn installed(dir: &Path) -> Option<PathBuf> {
    let p = dir.join(WEIGHTS_FILE);
    std::fs::metadata(&p).ok().filter(|m| m.len() == WEIGHTS_SIZE).map(|_| p)
}

//! AI Remove: LaMa inpainting (Suvorov et al., "Resolution-robust Large Mask Inpainting with
//! Fourier Convolutions", WACV 2022) in pure Rust on candle.
//!
//! [`Inpainter::fill`] takes an RGB crop and a mask of what to remove and returns the crop with
//! the masked pixels replaced by plausible surroundings. On macOS the model runs on the GPU
//! through Metal; elsewhere on the CPU.
//!
//! The weights are not part of LightCraft: they are the big-lama checkpoint (Apache-2.0, Samsung
//! AI Center), downloaded once when the user asks for AI Remove ([`fetch`]) and read straight
//! from its TorchScript archive (no Python).
//!
//! Z4K Studios addition (branch `z4k`); not part of upstream LightCraft. No UI dependencies (L3).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(target_arch = "wasm32"))]
mod dft;
#[cfg(not(target_arch = "wasm32"))]
pub mod fetch;
#[cfg(not(target_arch = "wasm32"))]
mod lama;
#[cfg(not(target_arch = "wasm32"))]
pub mod torchscript;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("AI Remove model: {0}")]
    Model(String),
    #[error("AI Remove: {0}")]
    Input(String),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("AI Remove: {0}")]
    Candle(#[from] candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Largest crop side the model is run on (larger crops are scaled down by the caller).
pub const MAX_SIDE: usize = 2048;

#[cfg(not(target_arch = "wasm32"))]
pub use native::Inpainter;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::Path;

    use candle_core::{Device, Tensor};

    use super::{Error, Result, MAX_SIDE};

    /// The loaded model.
    pub struct Inpainter {
        lama: crate::lama::Lama,
    }

    /// The GPU on macOS (when Metal is available), the CPU elsewhere.
    pub fn default_device() -> Device {
        #[cfg(target_os = "macos")]
        if let Ok(d) = Device::new_metal(0) {
            return d;
        }
        Device::Cpu
    }

    /// Mirror index `i` (any integer) into `0..n` the way `numpy.pad(mode="symmetric")` does.
    fn sym(i: isize, n: usize) -> usize {
        let n = n as isize;
        if n <= 0 {
            return 0;
        }
        let p = 2 * n;
        let m = i.rem_euclid(p);
        (if m < n { m } else { p - 1 - m }) as usize
    }

    impl Inpainter {
        /// Load the big-lama TorchScript archive at `path` onto [`default_device`].
        pub fn load(path: &Path) -> Result<Self> {
            Self::load_on(path, &default_device())
        }

        pub fn load_on(path: &Path, device: &Device) -> Result<Self> {
            let archive = crate::torchscript::Archive::open(path)?;
            Ok(Self { lama: crate::lama::Lama::load(&archive, device)? })
        }

        /// Fill the masked part of `rgb` (`w × h`, interleaved RGB in 0..1) where `mask`
        /// (`w × h`, 0..1) is above 0.5. Pixels outside the mask come back unchanged.
        pub fn fill(&self, rgb: &[f32], mask: &[f32], w: usize, h: usize) -> Result<Vec<f32>> {
            let n = w.checked_mul(h).ok_or_else(|| Error::Input("image too large".into()))?;
            if w == 0 || h == 0 || rgb.len() != n * 3 || mask.len() != n {
                return Err(Error::Input(format!("{w}×{h}: image and mask sizes don't match")));
            }
            if w > MAX_SIDE || h > MAX_SIDE {
                return Err(Error::Input(format!("{w}×{h}: larger than {MAX_SIDE} px")));
            }
            let pw = w.div_ceil(8).max(4) * 8;
            let ph = h.div_ceil(8).max(4) * 8;
            let np = pw * ph;
            // planar [masked r, g, b, mask], padded by mirroring
            let mut x = vec![0.0f32; 4 * np];
            for y in 0..ph {
                let sy = sym(y as isize, h);
                for xx in 0..pw {
                    let s = sy * w + sym(xx as isize, w);
                    let m = if mask.get(s).copied().unwrap_or(0.0) > 0.5 { 1.0 } else { 0.0 };
                    let o = y * pw + xx;
                    for c in 0..3 {
                        let v = rgb.get(s * 3 + c).copied().unwrap_or(0.0);
                        let v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
                        if let Some(slot) = x.get_mut(c * np + o) {
                            *slot = v * (1.0 - m);
                        }
                    }
                    if let Some(slot) = x.get_mut(3 * np + o) {
                        *slot = m;
                    }
                }
            }
            let t = Tensor::from_vec(x, (1, 4, ph, pw), self.lama.device())?;
            let out = self.lama.generate(&t)?.flatten_all()?.to_vec1::<f32>()?;
            let mut res = rgb.to_vec();
            for y in 0..h {
                for xx in 0..w {
                    let s = y * w + xx;
                    if mask.get(s).copied().unwrap_or(0.0) <= 0.5 {
                        continue;
                    }
                    for c in 0..3 {
                        let v = out.get(c * np + y * pw + xx).copied().unwrap_or(0.0);
                        if let Some(slot) = res.get_mut(s * 3 + c) {
                            *slot = v;
                        }
                    }
                }
            }
            Ok(res)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::sym;
        #[test]
        fn symmetric_padding() {
            let v: Vec<usize> = (-3..8).map(|i| sym(i, 4)).collect();
            assert_eq!(v, [2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0]);
        }
    }
}

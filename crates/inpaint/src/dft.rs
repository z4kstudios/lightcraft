//! 2-D real FFT pair (`rfftn` / `irfftn` over the last two axes, "ortho" norm) as matrix
//! products, so it runs wherever candle's matmul does (Metal included).
//!
//! LaMa's spectral blocks work at 1/8 of the input resolution (64 × 64 for a 512 × 512 crop),
//! where dense DFT matrices are small and a matmul is as fast as an FFT.

use candle_core::{Device, Tensor};

use crate::Result;

/// DFT matrices for an `h × w` grid.
pub struct Dft {
    pub h: usize,
    pub w: usize,
    /// `[w, w/2+1]`: cos and sin of 2π·x·k/w (forward along the width).
    cw: Tensor,
    sw: Tensor,
    /// `[h, h]` (symmetric): cos and sin of 2π·y·k/h.
    ch: Tensor,
    sh: Tensor,
    /// `[w/2+1, w]`: c_k·cos and c_k·sin (inverse along the width; c_k = 1 for the DC and Nyquist bins, else 2).
    icw: Tensor,
    isw: Tensor,
    scale: f64,
}

fn mat(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64, device: &Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            v.push(f(r, c) as f32);
        }
    }
    Ok(Tensor::from_vec(v, (rows, cols), device)?)
}

impl Dft {
    pub fn new(h: usize, w: usize, device: &Device) -> Result<Self> {
        let wf = w / 2 + 1;
        let tau = std::f64::consts::TAU;
        let ang_w = |x: usize, k: usize| tau * ((x * k) % w) as f64 / w as f64;
        let ang_h = |y: usize, k: usize| tau * ((y * k) % h) as f64 / h as f64;
        let ck = |k: usize| if k == 0 || (w % 2 == 0 && k == w / 2) { 1.0 } else { 2.0 };
        Ok(Self {
            h,
            w,
            cw: mat(w, wf, |x, k| ang_w(x, k).cos(), device)?,
            sw: mat(w, wf, |x, k| ang_w(x, k).sin(), device)?,
            ch: mat(h, h, |y, k| ang_h(y, k).cos(), device)?,
            sh: mat(h, h, |y, k| ang_h(y, k).sin(), device)?,
            icw: mat(wf, w, |k, x| ck(k) * ang_w(x, k).cos(), device)?,
            isw: mat(wf, w, |k, x| ck(k) * ang_w(x, k).sin(), device)?,
            scale: 1.0 / ((h * w) as f64).sqrt(),
        })
    }

    /// `x`: `[n, h, w]` real → `(re, im)`, each `[n, h, w/2+1]`.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (n, h, w) = x.dims3()?;
        let wf = self.w / 2 + 1;
        let flat = x.reshape((n * h, w))?;
        // along the width: e^{-iθ} = cos − i·sin
        let ar = flat.matmul(&self.cw)?.reshape((n, h, wf))?;
        let ai = flat.matmul(&self.sw)?.neg()?.reshape((n, h, wf))?;
        // along the height (rows of the transposed layout): (C − iS)(ar + i·ai)
        let tr = ar.transpose(1, 2)?.contiguous()?.reshape((n * wf, h))?;
        let ti = ai.transpose(1, 2)?.contiguous()?.reshape((n * wf, h))?;
        let br = (tr.matmul(&self.ch)? + ti.matmul(&self.sh)?)?;
        let bi = (ti.matmul(&self.ch)? - tr.matmul(&self.sh)?)?;
        let back = |t: Tensor| -> Result<Tensor> { Ok(t.reshape((n, wf, h))?.transpose(1, 2)?.contiguous()?.affine(self.scale, 0.0)?) };
        Ok((back(br)?, back(bi)?))
    }

    /// Inverse of [`Dft::forward`]: `(re, im)` `[n, h, w/2+1]` → `[n, h, w]` real.
    pub fn inverse(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        let (n, h, wf) = re.dims3()?;
        let tr = re.transpose(1, 2)?.contiguous()?.reshape((n * wf, h))?;
        let ti = im.transpose(1, 2)?.contiguous()?.reshape((n * wf, h))?;
        // along the height: (C + iS)(tr + i·ti)
        let yr = (tr.matmul(&self.ch)? - ti.matmul(&self.sh)?)?;
        let yi = (ti.matmul(&self.ch)? + tr.matmul(&self.sh)?)?;
        let yr = yr.reshape((n, wf, h))?.transpose(1, 2)?.contiguous()?.reshape((n * h, wf))?;
        let yi = yi.reshape((n, wf, h))?.transpose(1, 2)?.contiguous()?.reshape((n * h, wf))?;
        // along the width, real output: Σ c_k (Re·cos − Im·sin)
        let x = (yr.matmul(&self.icw)? - yi.matmul(&self.isw)?)?;
        Ok(x.reshape((n, h, self.w))?.affine(self.scale, 0.0)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_known_bins() {
        let d = Device::Cpu;
        let (h, w) = (6, 8);
        let v: Vec<f32> = (0..h * w).map(|i| ((i * 37 % 11) as f32) / 7.0 - 0.6).collect();
        let x = Tensor::from_vec(v.clone(), (1, h, w), &d).unwrap();
        let f = Dft::new(h, w, &d).unwrap();
        let (re, im) = f.forward(&x).unwrap();
        // DC bin = sum / sqrt(hw)
        let dc = re.get(0).unwrap().get(0).unwrap().get(0).unwrap().to_scalar::<f32>().unwrap();
        let sum: f32 = v.iter().sum();
        assert!((dc - sum / ((h * w) as f32).sqrt()).abs() < 1e-4);
        let back = f.inverse(&re, &im).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in back.iter().zip(&v) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }
}

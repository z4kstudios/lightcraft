//! LaMa (big-lama): the Fast-Fourier-Convolution ResNet inpainting generator.
//!
//! Layout of `generator.model` (51 M weights):
//! 0. reflect pad 3 · 1. conv 7×7 4→64 + BN + ReLU
//! 2–4. conv 3×3 stride 2 + BN + ReLU: 64→128→256→512 (the last splits into 128 local + 384 global channels)
//! 5–22. 18 FFC residual blocks over (local, global), each two FFC + BN + ReLU
//! 23. concat · 24–32. 3 × (transposed conv 3×3 stride 2 + BN + ReLU): 512→256→128→64
//! 33–35. reflect pad 3 · conv 7×7 64→3 · sigmoid
//!
//! An FFC maps (l, g) to (l2l(l) + g2l(g), l2g(l) + g2g(g)); g2g is the spectral transform:
//! 1×1 conv + BN + ReLU to half the channels, plus a Fourier unit (2-D real FFT, 1×1 conv + BN +
//! ReLU over the stacked real and imaginary parts, inverse FFT) added to it, then a 1×1 conv.
//! Every 3×3 / 7×7 convolution pads by reflection. Batch norms are folded to a per-channel
//! scale and shift at load time.

use std::sync::Mutex;

use candle_core::{Device, Tensor};

use crate::dft::Dft;
use crate::torchscript::Archive;
use crate::{Error, Result};

const EPS: f64 = 1e-5;
const N_BLOCKS: usize = 18;

/// A batch norm as `x · scale + shift` (per channel, `[1, c, 1, 1]`).
struct Affine {
    scale: Tensor,
    shift: Tensor,
}

impl Affine {
    fn load(a: &Archive, p: &str, d: &Device) -> Result<Self> {
        let w = a.get(&format!("{p}.weight"), d)?;
        let b = a.get(&format!("{p}.bias"), d)?;
        let m = a.get(&format!("{p}.running_mean"), d)?;
        let v = a.get(&format!("{p}.running_var"), d)?;
        let scale = w.div(&v.affine(1.0, EPS)?.sqrt()?)?;
        let shift = (b - m.mul(&scale)?)?;
        let c = scale.dim(0)?;
        Ok(Self { scale: scale.reshape((1, c, 1, 1))?, shift: shift.reshape((1, c, 1, 1))? })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_mul(&self.scale)?.broadcast_add(&self.shift)?)
    }
}

/// Reflection padding of the last two axes.
fn reflect_pad(x: &Tensor, p: usize) -> Result<Tensor> {
    if p == 0 {
        return Ok(x.clone());
    }
    let mut x = x.clone();
    for dim in [2, 3] {
        let n = x.dim(dim)?;
        if n <= p {
            return Err(Error::Input(format!("image too small to pad ({n} ≤ {p})")));
        }
        let dev = x.device().clone();
        let before: Vec<u32> = (1..=p).rev().map(|i| i as u32).collect();
        let after: Vec<u32> = (0..p).map(|i| (n - 2 - i) as u32).collect();
        let b = x.index_select(&Tensor::new(before, &dev)?, dim)?;
        let a = x.index_select(&Tensor::new(after, &dev)?, dim)?;
        x = Tensor::cat(&[&b, &x, &a], dim)?;
    }
    Ok(x)
}

/// A bias-free convolution with reflection padding.
struct Conv {
    w: Tensor,
    pad: usize,
    stride: usize,
}

impl Conv {
    fn load(a: &Archive, name: &str, pad: usize, stride: usize, d: &Device) -> Result<Self> {
        Ok(Self { w: a.get(&format!("{name}.weight"), d)?, pad, stride })
    }
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        Ok(reflect_pad(x, self.pad)?.conv2d(&self.w, 0, self.stride, 1, 1)?)
    }
}

fn relu(x: &Tensor) -> Result<Tensor> {
    Ok(x.relu()?)
}

/// The spectral transform (global → global).
struct Spectral {
    conv1: Tensor,
    bn1: Affine,
    fu_conv: Tensor,
    fu_bn: Affine,
    conv2: Tensor,
}

impl Spectral {
    fn load(a: &Archive, p: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            conv1: a.get(&format!("{p}.conv1.0.weight"), d)?,
            bn1: Affine::load(a, &format!("{p}.conv1.1"), d)?,
            fu_conv: a.get(&format!("{p}.fu.conv_layer.weight"), d)?,
            fu_bn: Affine::load(a, &format!("{p}.fu.bn"), d)?,
            conv2: a.get(&format!("{p}.conv2.weight"), d)?,
        })
    }

    fn apply(&self, x: &Tensor, dft: &Dft) -> Result<Tensor> {
        let x = relu(&self.bn1.apply(&x.conv2d(&self.conv1, 0, 1, 1, 1)?)?)?;
        let (b, c, h, w) = x.dims4()?;
        let wf = w / 2 + 1;
        let (re, im) = dft.forward(&x.reshape((b * c, h, w))?)?;
        // channels interleaved as (c0 re, c0 im, c1 re, …)
        let f = Tensor::stack(&[re.reshape((b, c, h, wf))?, im.reshape((b, c, h, wf))?], 2)?.reshape((b, 2 * c, h, wf))?;
        let f = relu(&self.fu_bn.apply(&f.conv2d(&self.fu_conv, 0, 1, 1, 1)?)?)?;
        let f = f.reshape((b, c, 2, h, wf))?;
        let re = f.narrow(2, 0, 1)?.contiguous()?.reshape((b * c, h, wf))?;
        let im = f.narrow(2, 1, 1)?.contiguous()?.reshape((b * c, h, wf))?;
        let fu = dft.inverse(&re, &im)?.reshape((b, c, h, w))?;
        Ok((x + fu)?.conv2d(&self.conv2, 0, 1, 1, 1)?)
    }
}

/// FFC + BN + ReLU over (local, global).
struct Ffc {
    l2l: Conv,
    l2g: Conv,
    g2l: Conv,
    g2g: Spectral,
    bn_l: Affine,
    bn_g: Affine,
}

impl Ffc {
    fn load(a: &Archive, p: &str, d: &Device) -> Result<Self> {
        Ok(Self {
            l2l: Conv::load(a, &format!("{p}.ffc.convl2l"), 1, 1, d)?,
            l2g: Conv::load(a, &format!("{p}.ffc.convl2g"), 1, 1, d)?,
            g2l: Conv::load(a, &format!("{p}.ffc.convg2l"), 1, 1, d)?,
            g2g: Spectral::load(a, &format!("{p}.ffc.convg2g"), d)?,
            bn_l: Affine::load(a, &format!("{p}.bn_l"), d)?,
            bn_g: Affine::load(a, &format!("{p}.bn_g"), d)?,
        })
    }
    fn apply(&self, l: &Tensor, g: &Tensor, dft: &Dft) -> Result<(Tensor, Tensor)> {
        let ol = (self.l2l.apply(l)? + self.g2l.apply(g)?)?;
        let og = (self.l2g.apply(l)? + self.g2g.apply(g, dft)?)?;
        Ok((relu(&self.bn_l.apply(&ol)?)?, relu(&self.bn_g.apply(&og)?)?))
    }
}

/// A transposed 3×3 convolution, stride 2, padding 1, output padding 1 (PyTorch weight
/// `[c_in, c_out, 3, 3]`), as one ordinary 2×2 convolution producing the four output parities,
/// then a pixel shuffle. Along one axis output `2m` is `x[m]·k1` and output `2m+1` is
/// `x[m]·k2 + x[m+1]·k0` (`x[n]` = 0), so the 2×2 kernel for parity `a` at tap `d` is `k[KTAP[a][d]]`.
/// Same result as `conv_transpose2d`, but on the fast im2col + matmul path (candle's transposed
/// convolution kernel is direct and was the slowest part of the model on Metal).
struct Up {
    w2: Tensor,
    c_out: usize,
    b: Tensor,
    bn: Affine,
}

/// The 3-tap index used by output parity `a` at 2×2 tap `d` (`None`: zero).
const KTAP: [[Option<usize>; 2]; 2] = [[Some(1), None], [Some(2), Some(0)]];

impl Up {
    fn load(a: &Archive, i: usize, d: &Device) -> Result<Self> {
        let m = |s: String| format!("generator.model.{s}");
        let w = a.get(&m(format!("{i}.weight")), d)?;
        let (c_in, c_out, kh, kw) = w.dims4()?;
        if (kh, kw) != (3, 3) {
            return Err(Error::Model(format!("layer {i}: expected a 3×3 kernel, got {kh}×{kw}")));
        }
        let wv = w.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        // [ (a, b, co), ci, dy, dx ]
        let mut w2 = vec![0.0f32; 4 * c_out * c_in * 4];
        for (pa, ra) in KTAP.iter().enumerate() {
            for (pb, rb) in KTAP.iter().enumerate() {
                for co in 0..c_out {
                    for ci in 0..c_in {
                        for (dy, ky) in ra.iter().enumerate() {
                            for (dx, kx) in rb.iter().enumerate() {
                                let (Some(ky), Some(kx)) = (ky, kx) else { continue };
                                let src = ((ci * c_out + co) * 3 + ky) * 3 + kx;
                                let dst = ((((pa * 2 + pb) * c_out + co) * c_in + ci) * 2 + dy) * 2 + dx;
                                if let (Some(v), Some(slot)) = (wv.get(src), w2.get_mut(dst)) {
                                    *slot = *v;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            w2: Tensor::from_vec(w2, (4 * c_out, c_in, 2, 2), d)?,
            c_out,
            b: a.get(&m(format!("{i}.bias")), d)?.reshape((1, (), 1, 1))?,
            bn: Affine::load(a, &m(format!("{}", i + 1)), d)?,
        })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _, h, w) = x.dims4()?;
        let x = x.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
        let y = x.conv2d(&self.w2, 0, 1, 1, 1)?; // [b, 4·c_out, h, w]
        let y = y.reshape((b, 2, 2, self.c_out, h, w))?.permute((0, 3, 4, 1, 5, 2))?.contiguous()?;
        let t = y.reshape((b, self.c_out, 2 * h, 2 * w))?.broadcast_add(&self.b)?;
        relu(&self.bn.apply(&t)?)
    }
}

/// The 7×7 output convolution (64 → 3) without im2col: one matmul to the 3·49 per-tap planes,
/// then the planes summed at their offsets. (im2col of a full-resolution 64-channel image
/// is 3136 values per pixel — gigabytes at 1 MP — for a 3-channel result.)
struct Head {
    /// `[3·k·k, c_in]`, rows ordered (co, ky, kx).
    w: Tensor,
    b: Tensor,
    k: usize,
}

impl Head {
    fn load(a: &Archive, d: &Device) -> Result<Self> {
        let w = a.get("generator.model.34.weight", d)?;
        let (c_out, c_in, kh, kw) = w.dims4()?;
        if kh != kw || c_out != 3 {
            return Err(Error::Model(format!("output layer: unexpected shape {c_out}×{c_in}×{kh}×{kw}")));
        }
        // [co, ci, ky, kx] → [co, ky, kx, ci]
        let w = w.permute((0, 2, 3, 1))?.contiguous()?.reshape((c_out * kh * kw, c_in))?;
        Ok(Self { w, b: a.get("generator.model.34.bias", d)?.reshape((1, 3, 1, 1))?, k: kh })
    }

    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let k = self.k;
        let p = reflect_pad(x, k / 2)?;
        let (_, c, hp, wp) = p.dims4()?;
        let (h, w) = (hp + 1 - k, wp + 1 - k);
        let z = self.w.matmul(&p.reshape((c, hp * wp))?)?.reshape((3, k * k, hp, wp))?;
        let mut acc: Option<Tensor> = None;
        for ky in 0..k {
            for kx in 0..k {
                let t = z.narrow(1, ky * k + kx, 1)?.narrow(2, ky, h)?.narrow(3, kx, w)?;
                acc = Some(match acc {
                    Some(a) => (a + t)?,
                    None => t.contiguous()?,
                });
            }
        }
        let t = acc.ok_or_else(|| Error::Model("empty kernel".into()))?.reshape((1, 3, h, w))?;
        Ok(t.broadcast_add(&self.b)?)
    }
}

pub struct Lama {
    device: Device,
    stem: Conv,
    stem_bn: Affine,
    down: Vec<(Conv, Affine)>,
    /// The last downsampling: local and global halves.
    split: (Conv, Affine, Conv, Affine),
    blocks: Vec<(Ffc, Ffc)>,
    up: Vec<Up>,
    head: Head,
    dft: Mutex<Option<Dft>>,
}

impl Lama {
    pub fn load(a: &Archive, device: &Device) -> Result<Self> {
        let d = device;
        let m = |s: &str| format!("generator.model.{s}");
        let down = (2..=3)
            .map(|i| Ok((Conv::load(a, &m(&format!("{i}.ffc.convl2l")), 1, 2, d)?, Affine::load(a, &m(&format!("{i}.bn_l")), d)?)))
            .collect::<Result<Vec<_>>>()?;
        let split = (
            Conv::load(a, &m("4.ffc.convl2l"), 1, 2, d)?,
            Affine::load(a, &m("4.bn_l"), d)?,
            Conv::load(a, &m("4.ffc.convl2g"), 1, 2, d)?,
            Affine::load(a, &m("4.bn_g"), d)?,
        );
        let blocks = (5..5 + N_BLOCKS)
            .map(|i| Ok((Ffc::load(a, &m(&format!("{i}.conv1")), d)?, Ffc::load(a, &m(&format!("{i}.conv2")), d)?)))
            .collect::<Result<Vec<_>>>()?;
        let up = [24, 27, 30].iter().map(|&i| Up::load(a, i, d)).collect::<Result<Vec<_>>>()?;
        Ok(Self {
            device: device.clone(),
            stem: Conv::load(a, &m("1.ffc.convl2l"), 3, 1, d)?,
            stem_bn: Affine::load(a, &m("1.bn_l"), d)?,
            down,
            split,
            blocks,
            up,
            head: Head::load(a, d)?,
            dft: Mutex::new(None),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The generator on `x` = `[1, 4, h, w]` (masked RGB in 0..1, then the mask); `h` and `w`
    /// multiples of 8, at least 32. Returns `[1, 3, h, w]` in 0..1.
    pub fn generate(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = x.dims4()?;
        if !h.is_multiple_of(8) || !w.is_multiple_of(8) || h < 32 || w < 32 {
            return Err(Error::Input(format!("{w}×{h}: sides must be multiples of 8, at least 32")));
        }
        let mut guard = self.dft.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if guard.as_ref().is_none_or(|d| d.h != h / 8 || d.w != w / 8) {
            *guard = Some(Dft::new(h / 8, w / 8, &self.device)?);
        }
        let dft = guard.as_ref().ok_or_else(|| Error::Model("no DFT".into()))?;
        let mut t = relu(&self.stem_bn.apply(&self.stem.apply(x)?)?)?;
        for (c, bn) in &self.down {
            t = relu(&bn.apply(&c.apply(&t)?)?)?;
        }
        let (cl, bl, cg, bg) = &self.split;
        let mut l = relu(&bl.apply(&cl.apply(&t)?)?)?;
        let mut g = relu(&bg.apply(&cg.apply(&t)?)?)?;
        for (f1, f2) in &self.blocks {
            let (l1, g1) = f1.apply(&l, &g, dft)?;
            let (l2, g2) = f2.apply(&l1, &g1, dft)?;
            l = (l + l2)?;
            g = (g + g2)?;
        }
        let mut t = Tensor::cat(&[&l, &g], 1)?;
        for u in &self.up {
            t = u.apply(&t)?;
        }
        let t = self.head.apply(&t)?;
        Ok(candle_core::Tensor::ones_like(&t)?.broadcast_div(&(t.neg()?.exp()? + 1.0)?)?)
    }
}

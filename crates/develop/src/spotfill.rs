//! AI Remove results stored with a spot ([`SpotFill`]), so rendering never needs the model.
//! (Z4K Studios addition, branch `z4k`.)

use serde::{Deserialize, Serialize};

use crate::segmask::{base64_decode, base64_encode};

/// Largest side a stored fill may claim (hostile input caps the allocation).
pub const MAX_SIDE: usize = 2048;

/// Values are stored as log2 in 32 stops from 2^LOG_MIN, quantized to 16 bits.
const LOG_MIN: f32 = -20.0;
const LOG_SPAN: f32 = 32.0;

/// The pixels an AI Remove spot paints with: `w × h` scene-linear RGB samples over the part
/// `rect` (normalized oriented-image coordinates `[x0, y0, x1, y1]`, the spots' own) of the
/// photo, as the model filled it. Sample `(x, y)` sits at
/// `(x0 + (x + 0.5) / w · (x1 − x0), y0 + (y + 0.5) / h · (y1 − y0))`.
///
/// Stored log-encoded in 16 bits per channel (planar, horizontally delta-coded), deflated and
/// base64-encoded.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct SpotFill {
    pub w: u32,
    pub h: u32,
    pub rect: [f64; 4],
    pub data: String,
}

impl std::fmt::Debug for SpotFill {
    // the data is large: a render key formats spots with `{:?}`, so keep this short (but
    // different for different fills)
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.data.bytes() {
            hash = (hash ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        write!(f, "SpotFill({}×{} {:?} #{hash:016x})", self.w, self.h, self.rect)
    }
}

fn enc(v: f32) -> u16 {
    let l = if v.is_finite() && v > 0.0 { v.log2() } else { LOG_MIN };
    (((l - LOG_MIN) / LOG_SPAN).clamp(0.0, 1.0) * 65535.0).round() as u16
}

fn dec(q: u16) -> f32 {
    if q == 0 {
        return 0.0;
    }
    (f32::from(q) / 65535.0 * LOG_SPAN + LOG_MIN).exp2()
}

impl SpotFill {
    /// From `w × h` interleaved RGB samples over `rect`.
    pub fn new(w: usize, h: usize, rect: [f64; 4], rgb: &[f32]) -> Option<SpotFill> {
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || rgb.len() != w * h * 3 {
            return None;
        }
        let mut bytes = Vec::with_capacity(w * h * 6);
        for c in 0..3 {
            for y in 0..h {
                let mut prev = 0u16;
                for x in 0..w {
                    let q = enc(rgb.get((y * w + x) * 3 + c).copied().unwrap_or(0.0));
                    bytes.extend_from_slice(&q.wrapping_sub(prev).to_le_bytes());
                    prev = q;
                }
            }
        }
        let z = miniz_oxide::deflate::compress_to_vec(&bytes, 6);
        Some(SpotFill { w: w as u32, h: h as u32, rect, data: base64_encode(&z) })
    }

    /// The covered part, normalized; `None` when damaged.
    pub fn bounds(&self) -> Option<[f64; 4]> {
        let r = self.rect;
        (r.iter().all(|v| v.is_finite()) && r[2] > r[0] && r[3] > r[1]).then_some(r)
    }

    /// The samples, interleaved RGB; `None` when the data is damaged or claims an absurd size.
    pub fn pixels(&self) -> Option<Vec<[f32; 3]>> {
        let (w, h) = (self.w as usize, self.h as usize);
        if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE {
            return None;
        }
        let n = w * h;
        let z = base64_decode(&self.data)?;
        let b = miniz_oxide::inflate::decompress_to_vec_with_limit(&z, n * 6).ok()?;
        if b.len() != n * 6 {
            return None;
        }
        let mut out = vec![[0.0f32; 3]; n];
        let mut q = b.as_chunks::<2>().0.iter().map(|p| u16::from_le_bytes(*p));
        for c in 0..3 {
            for y in 0..h {
                let mut prev = 0u16;
                for x in 0..w {
                    prev = prev.wrapping_add(q.next()?);
                    if let Some(px) = out.get_mut(y * w + x) {
                        px[c] = dec(prev);
                    }
                }
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_within_quantization() {
        let (w, h) = (7, 5);
        let rgb: Vec<f32> = (0..w * h * 3).map(|i| 0.001 + (i as f32 * 0.37).sin().abs() * 3.0).collect();
        let f = SpotFill::new(w, h, [0.1, 0.2, 0.3, 0.4], &rgb).unwrap();
        let back = f.pixels().unwrap();
        for (i, p) in back.iter().enumerate() {
            for c in 0..3 {
                let a = rgb[i * 3 + c];
                assert!((p[c] - a).abs() <= a * 0.001, "{} vs {a}", p[c]);
            }
        }
        assert!(format!("{f:?}").len() < 100);
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<SpotFill>(&json).unwrap(), f);
    }

    #[test]
    fn damaged_data_is_none() {
        let mut f = SpotFill::new(2, 2, [0.0, 0.0, 1.0, 1.0], &[0.5; 12]).unwrap();
        f.w = 3;
        assert!(f.pixels().is_none());
        f.w = 100_000;
        assert!(f.pixels().is_none());
        assert!(SpotFill { w: 2, h: 2, rect: [0.0, 0.0, 1.0, 1.0], data: "!!".into() }.pixels().is_none());
        assert!(SpotFill { rect: [0.5, 0.0, 0.2, 1.0], ..SpotFill::new(1, 1, [0.0; 4], &[0.0; 3]).unwrap() }.bounds().is_none());
        assert!(SpotFill::new(0, 1, [0.0; 4], &[]).is_none());
    }
}

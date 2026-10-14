//! Remove / Heal / Clone spots, rendered at output resolution in scene-linear light.
//!
//! - **Clone** copies the source patch with a feathered edge.
//! - **Heal** copies the source's *detail* and keeps the target's *tone*: result = src + blur(target − src)
//!   (frequency separation — a fast, stable approximation of gradient-domain healing).
//! - **Remove** is Heal with an automatically chosen source: candidate offsets on rings around the
//!   spot are scored by how well the source's surrounding annulus matches the target's.
//! - **Fill** (content-aware) rebuilds the spot from patches of its surroundings instead of one
//!   source: PatchMatch-based completion ([`lightcraft_inpaint::complete`]), coarse to fine, so the
//!   structure is chosen at a scale set by the spot's size (the same at preview and export size).
//!   Deterministic; the source offset is ignored.

use lightcraft_develop::{Spot, SpotMode};
use lightcraft_geom::Point;
use lightcraft_raster::Rgb32f;

use crate::geometry::Frame;

#[inline]
fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    // Zero feather is a hard edge, including when the pixel lies exactly on the radius.
    // Dividing 0 by 0 here would give that pixel a NaN alpha, even at zero opacity.
    if e0 >= e1 {
        return if x < e1 { 0.0 } else { 1.0 };
    }
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Most a spot's size counts for, in long edges: a size read from a sidecar isn't held to the slider's range, and one
/// far past the picture (or infinite) made the radius, the blur it heals with and the patch it reads from unbounded
/// (issue #1519). Two long edges cover the whole frame from anywhere in it.
const MAX_SPOT_SIZE: f64 = 2.0;

/// A spot's radius in output pixels, `ppl` output pixels to the long edge: at least a pixel (a size that is not a
/// number too), at most [`MAX_SPOT_SIZE`] long edges.
fn radius_px(size: f64, ppl: f64) -> f64 {
    let size = if size.is_nan() { 0.0 } else { size.min(MAX_SPOT_SIZE) };
    (size * ppl).max(1.0)
}

/// Mean colour of `img` in a disc/annulus.
fn ring_stats(img: &Rgb32f, c: (f32, f32), r0: f32, r1: f32) -> Option<Vec<[f32; 3]>> {
    let mut out = Vec::new();
    let n = 24;
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU;
        for rr in [r0, (r0 + r1) / 2.0, r1] {
            let (x, y) = (c.0 + a.cos() * rr, c.1 + a.sin() * rr);
            if x < 0.0 || y < 0.0 || x >= img.width as f32 || y >= img.height as f32 {
                return None;
            }
            out.push(img.sample_bilinear(x, y));
        }
    }
    Some(out)
}

fn score(img: &Rgb32f, target: (f32, f32), source: (f32, f32), r: f32) -> f32 {
    let (Some(a), Some(b)) = (ring_stats(img, target, r * 1.05, r * 1.6), ring_stats(img, source, r * 1.05, r * 1.6)) else { return f32::MAX };
    // also compare the inside of the source against the target's surroundings (texture continuity)
    let inside = ring_stats(img, source, r * 0.2, r * 0.9);
    let mut s = 0.0;
    for (p, q) in a.iter().zip(&b) {
        for c in 0..3 {
            let (u, v) = (p[c].max(0.0).sqrt(), q[c].max(0.0).sqrt());
            s += (u - v) * (u - v);
        }
    }
    if let Some(ins) = inside {
        let mean_a: f32 = a.iter().map(|p| p[1].max(0.0).sqrt()).sum::<f32>() / a.len() as f32;
        let var: f32 = ins.iter().map(|p| (p[1].max(0.0).sqrt() - mean_a).powi(2)).sum::<f32>() / ins.len() as f32;
        s += var * a.len() as f32 * 0.5;
    }
    s
}

/// Candidate source offsets (output pixels) for a spot of radius `r` at `target`, best first
/// (candidates reaching outside the image are left out).
pub fn ranked_sources(img: &Rgb32f, target: (f32, f32), r: f32) -> Vec<(f32, f32)> {
    let mut c: Vec<(f32, (f32, f32))> = Vec::new();
    for ring in [2.4f32, 3.2, 4.2] {
        for i in 0..16 {
            let a = i as f32 / 16.0 * std::f32::consts::TAU;
            let d = (a.cos() * r * ring, a.sin() * r * ring);
            let s = score(img, target, (target.0 + d.0, target.1 + d.1), r);
            if s < f32::MAX {
                c.push((s, d));
            }
        }
    }
    c.sort_by(|a, b| a.0.total_cmp(&b.0));
    c.into_iter().map(|(_, d)| d).collect()
}

/// Pick a source offset (in output pixels) for a remove/heal spot.
pub fn auto_source(img: &Rgb32f, target: (f32, f32), r: f32) -> (f32, f32) {
    ranked_sources(img, target, r).first().copied().unwrap_or((r * 2.5, 0.0))
}

/// A source offset (normalized, as [`Spot::source_offset`]) for `spot` on `src` developed with
/// `s`: the best match, or with `avoid` (the current offset) the best one at least a spot radius
/// away from it ("refresh source"). `None` when no candidate fits in the image.
pub fn pick_source(
    src: &Rgb32f,
    info: &crate::SourceInfo,
    s: &lightcraft_develop::DevelopSettings,
    spot: &Spot,
    avoid: Option<Point>,
) -> Option<Point> {
    let frame = crate::frame_for(src, info, s, true);
    let (w, h) = frame.fit(512, 512);
    if w == 0 || h == 0 {
        return None;
    }
    let img = frame.sample(src, w, h);
    let (to_out, to_norm) = (frame.norm_to_out(w, h), frame.out_to_norm(w, h));
    let t = to_out.apply(*spot.points.first()?);
    let r = radius_px(spot.size, frame.px_per_long(w)) as f32;
    let avoid = avoid.map(|o| {
        let (a, b) = (to_out.apply(Point::new(0.0, 0.0)), to_out.apply(o));
        ((b.x - a.x) as f32, (b.y - a.y) as f32)
    });
    let far = |d: &(f32, f32)| avoid.is_none_or(|a| ((d.0 - a.0).powi(2) + (d.1 - a.1).powi(2)).sqrt() > r);
    let d = ranked_sources(&img, (t.x as f32, t.y as f32), r).into_iter().find(far)?;
    let (n0, n1) = (to_norm.apply(t), to_norm.apply(Point::new(t.x + d.0 as f64, t.y + d.1 as f64)));
    Some(Point::new(n1.x - n0.x, n1.y - n0.y))
}

/// The most a window grows to take in the spots that touch it.
const MAX_SPOT_SPAN: usize = if cfg!(target_arch = "wasm32") { 4096 } else { 8192 };
/// Strokes with more dabs than this are not examined dab by dab (the render would be too slow anyway).
const MAX_DABS_CONSIDERED: usize = 200_000;

/// `win` (a window of the `full_w × full_h` output of `frame`) grown to hold, for every spot (or
/// Auto Mask stroke) that reaches into it, the whole spot and the patch it reads from: [`apply`] works on the pixels it
/// is given, so a spot cut by the window's edge, or whose source lies outside it, would come out
/// different from the same spot in the whole render. Spots with an automatic source (`None`)
/// search up to 4.2 radii around themselves. Growth that would pass [`MAX_SPOT_SPAN`] pixels is not
/// done (the window comes back as asked; see [`window_for_reads_checked`] to know).
pub fn window_for_reads(
    s: &lightcraft_develop::DevelopSettings,
    frame: &Frame,
    full_w: usize,
    full_h: usize,
    ppl: f64,
    win: crate::PixelWindow,
) -> crate::PixelWindow {
    window_for_reads_checked(s, frame, full_w, full_h, ppl, win).unwrap_or(win)
}

/// [`window_for_reads`], or `None` when holding everything the window reads would take more than
/// [`MAX_SPOT_SPAN`] pixels along an axis: a caller that can show something else (the loupe's
/// whole-frame render) must, because the window alone would come out wrong.
pub fn window_for_reads_checked(
    s: &lightcraft_develop::DevelopSettings,
    frame: &Frame,
    full_w: usize,
    full_h: usize,
    ppl: f64,
    win: crate::PixelWindow,
) -> Option<crate::PixelWindow> {
    let spots = &s.spots;
    let autos: Vec<&lightcraft_develop::BrushStroke> = s
        .masks
        .iter()
        .filter(|m| m.visible)
        .flat_map(|m| &m.components)
        .filter_map(|c| match &c.shape {
            lightcraft_develop::MaskShape::Brush { strokes } => Some(strokes),
            _ => None,
        })
        .flatten()
        .filter(|st| st.auto_mask)
        .collect();
    if spots.is_empty() && autos.is_empty() {
        return Some(win);
    }
    let to_out = frame.norm_to_out(full_w, full_h);
    let origin = to_out.apply(Point::new(0.0, 0.0));
    // (target box, source box) of each spot, in output pixels: [x0, y0, x1, y1]
    let boxes: Vec<([f64; 4], [f64; 4])> = spots
        .iter()
        .filter_map(|spot| {
            let r = radius_px(spot.size, ppl);
            let pts: Vec<_> = spot.points.iter().map(|p| to_out.apply(*p)).collect();
            let first = pts.first()?;
            let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
            for p in &pts {
                (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
            }
            // the box `apply` works in: the stroke plus a radius, plus a radius of padding
            let t = [x0 - 2.0 * r, y0 - 2.0 * r, x1 + 2.0 * r, y1 + 2.0 * r];
            let s = match spot.source_offset {
                _ if spot.fill.is_some() => t,
                Some(o) => {
                    let b = to_out.apply(o);
                    let (dx, dy) = (b.x - origin.x, b.y - origin.y);
                    [t[0] + dx, t[1] + dy, t[2] + dx, t[3] + dy]
                }
                None => [t[0] - 4.2 * r, t[1] - 4.2 * r, t[2] + 4.2 * r, t[3] + 4.2 * r],
            };
            Some((t, s)).filter(|(t, s)| t.iter().chain(s).all(|v| v.is_finite()))
        })
        .collect();
    // Auto Mask strokes weigh each dab by its similarity to the pixel under the dab's centre and
    // refine the stroke over a couple of radii: each dab matters to pixels within 2.5 radii
    let mut dab_boxes: Vec<[f64; 4]> = Vec::new();
    for st in autos {
        let dabs = crate::masks::brush_dabs(st, frame, full_w, full_h);
        let r = dabs.r;
        if dabs.dabs.len() > MAX_DABS_CONSIDERED {
            continue;
        }
        for d in &dabs.dabs {
            let b = [d.x - 2.5 * r, d.y - 2.5 * r, d.x + 2.5 * r, d.y + 2.5 * r];
            if b.iter().all(|v| v.is_finite()) {
                dab_boxes.push(b);
            }
        }
    }
    let (fw, fh) = (full_w as f64, full_h as f64);
    let mut r = [win.x as f64, win.y as f64, (win.x + win.w) as f64, (win.y + win.h) as f64];
    let touches = |r: &[f64; 4], b: &[f64; 4]| b[0] < r[2] && b[2] > r[0] && b[1] < r[3] && b[3] > r[1];
    // a dab reaching the requested window needs its surroundings, once: the pixels that brings in
    // are only read, not drawn on (unlike spots, whose patches other spots may read)
    let asked = r;
    for b in dab_boxes.iter().filter(|b| touches(&asked, b)) {
        r = [r[0].min(b[0]), r[1].min(b[1]), r[2].max(b[2]), r[3].max(b[3])];
    }
    r = [r[0].max(0.0), r[1].max(0.0), r[2].min(fw), r[3].min(fh)];
    if r[2] - r[0] > MAX_SPOT_SPAN as f64 || r[3] - r[1] > MAX_SPOT_SPAN as f64 {
        return None;
    }
    for _ in 0..=boxes.len() {
        let mut grown = r;
        for (t, s) in &boxes {
            if touches(&r, t) {
                for b in [t, s] {
                    grown = [grown[0].min(b[0]), grown[1].min(b[1]), grown[2].max(b[2]), grown[3].max(b[3])];
                }
            }
        }
        let grown = [grown[0].max(0.0), grown[1].max(0.0), grown[2].min(fw), grown[3].min(fh)];
        if grown == r {
            break;
        }
        if grown[2] - grown[0] > MAX_SPOT_SPAN as f64 || grown[3] - grown[1] > MAX_SPOT_SPAN as f64 {
            return None;
        }
        r = grown;
    }
    let (x, y) = (r[0].floor() as usize, r[1].floor() as usize);
    Some(
        crate::PixelWindow { x, y, w: (r[2].ceil() as usize).saturating_sub(x).max(1), h: (r[3].ceil() as usize).saturating_sub(y).max(1) }
            .clamped(full_w, full_h),
    )
}

/// Content-aware fill of the box: `src` becomes `tgt` with every pixel under the stroke (`alpha > 0`)
/// rebuilt from the rest of the box. Falls back to a smooth membrane fill when the box holds no whole
/// patch to copy from.
fn fill_box(src: &mut [[f32; 3]], tgt: &[[f32; 3]], alpha: &[f32], bw: usize, bh: usize) {
    let hole: Vec<bool> = alpha.iter().map(|a| *a > 0.0).collect();
    let flat: Vec<f32> = tgt.iter().flatten().copied().collect();
    let out = lightcraft_inpaint::complete(bw, bh, 3, &flat, &hole, &lightcraft_inpaint::CompleteParams::default())
        .unwrap_or_else(|| lightcraft_inpaint::membrane_fill(bw, bh, 3, &flat, &hole));
    let (pixels, _) = out.as_chunks::<3>();
    for (d, v) in src.iter_mut().zip(pixels) {
        *d = *v;
    }
}

/// A decoded [`lightcraft_develop::SpotFill`].
struct FillImage {
    rect: [f64; 4],
    w: usize,
    h: usize,
    px: Vec<[f32; 3]>,
}

impl FillImage {
    /// Bilinear sample at normalized point `p` (edges extended).
    fn sample(&self, p: Point) -> [f32; 3] {
        let [x0, y0, x1, y1] = self.rect;
        let fx = ((p.x - x0) / (x1 - x0) * self.w as f64 - 0.5).clamp(0.0, (self.w - 1) as f64);
        let fy = ((p.y - y0) / (y1 - y0) * self.h as f64 - 0.5).clamp(0.0, (self.h - 1) as f64);
        let (ix, iy) = (fx.floor() as usize, fy.floor() as usize);
        let (jx, jy) = ((ix + 1).min(self.w - 1), (iy + 1).min(self.h - 1));
        let (tx, ty) = ((fx - ix as f64) as f32, (fy - iy as f64) as f32);
        let g = |x: usize, y: usize| self.px.get(y * self.w + x).copied().unwrap_or([0.0; 3]);
        let (a, b, c, d) = (g(ix, iy), g(jx, iy), g(ix, jy), g(jx, jy));
        std::array::from_fn(|k| (a[k] * (1.0 - tx) + b[k] * tx) * (1.0 - ty) + (c[k] * (1.0 - tx) + d[k] * tx) * ty)
    }
}

/// Apply all spots to `img` (output resolution). `ppl` = output pixels per long-edge unit.
pub fn apply(img: &mut Rgb32f, spots: &[Spot], frame: &Frame, ppl: f64) {
    if spots.is_empty() {
        return;
    }
    let (w, h) = (img.width, img.height);
    let to_out = frame.norm_to_out(w, h);
    for spot in spots {
        let r = radius_px(spot.size, ppl) as f32;
        let pts: Vec<(f32, f32)> = spot
            .points
            .iter()
            .map(|p| {
                let q = to_out.apply(*p);
                (q.x as f32, q.y as f32)
            })
            .collect();
        let Some(&first) = pts.first() else { continue };
        // AI Remove (Z4K): paint the stored fill (sampled through the normalized coordinates it was
        // made in) instead of a source patch
        let ai_fill = spot.fill.as_ref().and_then(|f| Some(FillImage { rect: f.bounds()?, w: f.w as usize, h: f.h as usize, px: f.pixels()? }));
        let to_norm = frame.out_to_norm(w, h);
        let fill = spot.mode == SpotMode::Fill && ai_fill.is_none();
        let offset = match spot.source_offset {
            _ if fill || ai_fill.is_some() => (0.0, 0.0),
            Some(o) => {
                let a = to_out.apply(Point::new(0.0, 0.0));
                let b = to_out.apply(o);
                ((b.x - a.x) as f32, (b.y - a.y) as f32)
            }
            None => auto_source(img, first, r),
        };
        // Bounding box of the stroke.
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for p in &pts {
            x0 = x0.min(p.0 - r);
            y0 = y0.min(p.1 - r);
            x1 = x1.max(p.0 + r);
            y1 = y1.max(p.1 + r);
        }
        // content-aware fill reads the spot's surroundings (the window holds 2 radii around it)
        let pad = if fill { 2.0 * r } else { r };
        let (bx0, by0) = (((x0 - pad).floor().max(0.0)) as usize, ((y0 - pad).floor().max(0.0)) as usize);
        let (bx1, by1) = (((x1 + pad).ceil() as usize).min(w), ((y1 + pad).ceil() as usize).min(h));
        if bx1 <= bx0 || by1 <= by0 {
            continue;
        }
        let (bw, bh) = (bx1 - bx0, by1 - by0);
        let feather = (spot.feather / 100.0).clamp(0.0, 1.0) as f32;
        let opacity = (spot.opacity / 100.0).clamp(0.0, 1.0) as f32;
        // alpha of the stroke and source patch in the box
        let mut alpha = vec![0.0f32; bw * bh];
        let mut src = vec![[0.0f32; 3]; bw * bh];
        let mut tgt = vec![[0.0f32; 3]; bw * bh];
        for y in 0..bh {
            for x in 0..bw {
                let (px, py) = ((bx0 + x) as f32 + 0.5, (by0 + y) as f32 + 0.5);
                let d = pts.iter().map(|p| ((px - p.0).powi(2) + (py - p.1).powi(2)).sqrt()).fold(f32::MAX, f32::min);
                let a = match &ai_fill {
                    Some(_) => 1.0 - smooth(r, r * FILL_FADE_OUT, d),
                    None => 1.0 - smooth(r * (1.0 - feather * 0.8), r, d),
                };
                alpha[y * bw + x] = a * opacity;
                src[y * bw + x] = match &ai_fill {
                    Some(f) => f.sample(to_norm.apply(Point::new(f64::from(px), f64::from(py)))),
                    None => img.sample_bilinear(px + offset.0, py + offset.1),
                };
                tgt[y * bw + x] = img.get(bx0 + x, by0 + y);
            }
        }
        if fill {
            fill_box(&mut src, &tgt, &alpha, bw, bh);
        }
        let heal = matches!(spot.mode, SpotMode::Remove | SpotMode::Heal) || ai_fill.is_some();
        let low = if heal {
            // Tone correction from the spot's *surroundings* only: normalized convolution of
            // (target − source) over pixels outside the stroke (a membrane-like interpolation).
            let outside: Vec<f32> = alpha.iter().map(|a| if *a < 0.02 { 1.0 } else { 0.0 }).collect();
            let diff = Rgb32f {
                width: bw,
                height: bh,
                data: tgt.iter().zip(&src).zip(&outside).map(|((t, s), m)| [(t[0] - s[0]) * m, (t[1] - s[1]) * m, (t[2] - s[2]) * m]).collect(),
            };
            let wts = lightcraft_raster::Plane { width: bw, height: bh, data: outside };
            let sigma = (r * 0.6).max(1.0);
            let (bd, bm) = (lightcraft_raster::blur::gaussian(&diff, sigma), lightcraft_raster::blur::gaussian(&wts, sigma));
            Some(Rgb32f {
                width: bw,
                height: bh,
                data: bd
                    .data
                    .iter()
                    .zip(&bm.data)
                    .map(|(d, m)| {
                        let m = m.max(1e-4);
                        [d[0] / m, d[1] / m, d[2] / m]
                    })
                    .collect(),
            })
        } else {
            None
        };
        let grain = if ai_fill.is_some() { grain_of(&tgt, &alpha, bw, bh) } else { None };
        for y in 0..bh {
            for x in 0..bw {
                let i = y * bw + x;
                let a = alpha[i];
                if a <= 0.0 {
                    continue;
                }
                let mut v = src[i];
                if let Some(g) = grain {
                    // shot-noise-like: the deviation grows with the square root of the level
                    let (gx, gy) = ((bx0 + x) as u32, (by0 + y) as u32);
                    for (c, vc) in v.iter_mut().enumerate() {
                        *vc += g[c] * vc.max(0.0).sqrt() * gauss(gx, gy, c as u32);
                    }
                }
                if let Some(l) = &low {
                    let d = l.data[i];
                    v = [v[0] + d[0], v[1] + d[1], v[2] + d[2]];
                }
                let t = tgt[i];
                img.set(bx0 + x, by0 + y, [t[0] + (v[0] - t[0]) * a, t[1] + (v[1] - t[1]) * a, t[2] + (v[2] - t[2]) * a].map(|c| c.max(0.0)));
            }
        }
    }
}

/// The photo's grain around a filled spot, per channel, as the standard deviation of the noise
/// divided by the square root of the level (shot noise). Measured on the pixels just outside the
/// stroke: the high-pass of the photo (it minus a small blur), robustly (median absolute
/// deviation). The model's fill is smooth; without matching grain it reads as a smudge on a
/// high-ISO photo. `None` when the surroundings are too small to measure.
fn grain_of(tgt: &[[f32; 3]], alpha: &[f32], w: usize, h: usize) -> Option<[f32; 3]> {
    let img = Rgb32f { width: w, height: h, data: tgt.to_vec() };
    let low = lightcraft_raster::blur::gaussian(&img, 1.0);
    let mut out = [0.0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let mut v: Vec<f32> = tgt
            .iter()
            .zip(&low.data)
            .zip(alpha)
            .filter(|(_, a)| **a <= 0.0)
            .map(|((t, l), _)| (t[c] - l[c]) / l[c].max(1e-4).sqrt())
            .filter(|v| v.is_finite())
            .map(f32::abs)
            .collect();
        if v.len() < 64 {
            return None;
        }
        let mid = v.len() / 2;
        let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
        // MAD → σ (×1.4826), and undo the σ = 1 px high-pass's damping of white noise (≈ 0.87)
        *o = (*m * 1.4826 / 0.87).min(1.0);
    }
    Some(out)
}

/// A standard normal number fixed by (x, y, c): the same grain on every render at a given size.
fn gauss(x: u32, y: u32, c: u32) -> f32 {
    let hash = |mut v: u32| {
        v ^= v >> 16;
        v = v.wrapping_mul(0x7feb_352d);
        v ^= v >> 15;
        v = v.wrapping_mul(0x846c_a68b);
        v ^ (v >> 16)
    };
    let a = hash(x.wrapping_mul(0x9e37_79b9) ^ hash(y.wrapping_add(0x632b_e5ab)) ^ c.wrapping_mul(0x85eb_ca6b));
    let b = hash(a ^ 0x2545_f491);
    let u1 = ((a >> 8) as f32 + 0.5) / 16_777_216.0;
    let u2 = (b >> 8) as f32 / 16_777_216.0;
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

/// Longest side of the crop AI Remove runs the model on.
pub const FILL_MAX_SIDE: usize = 1024;

/// How far past the stroke (in brush radii) the model refills: a stroke drawn snugly around an
/// object leaves its edge, shadow and halo just outside, and the model, seeing them, continues
/// them into the hole.
pub const FILL_GROW: f64 = 1.3;
/// Where a filled spot's render fades out (brush radii): solid to the stroke's radius, then a
/// blend into the photo inside the refilled margin (the stroke's feather isn't used: fading the
/// fill out inside the stroke would bring back what was removed).
const FILL_FADE_OUT: f32 = 1.25;

/// What the model needs to fill spot `index` (Z4K AI Remove): a display-encoded crop around the
/// stroke and a mask of it. Made by [`fill_job`]; turned into a stored fill by [`FillJob::finish`].
pub struct FillJob {
    pub w: usize,
    pub h: usize,
    /// Interleaved RGB in 0..1 (sRGB-encoded after scaling by `gain`).
    pub rgb: Vec<f32>,
    /// 1 where the model fills, else 0.
    pub mask: Vec<f32>,
    /// The scene-linear crop (what the spot stage sees), for the pixels the model leaves alone.
    lin: Vec<[f32; 3]>,
    gain: f32,
    /// Output pixel (of the scaled, uncropped full frame) → normalized coordinates, and back.
    to_norm: lightcraft_geom::Affine,
    to_out: lightcraft_geom::Affine,
    origin: (f64, f64),
    rect: [f64; 4],
}

/// The crop and mask AI Remove fills for `s.spots[index]`: the photo as the spot stage sees it
/// (white-balanced, earlier spots applied, uncropped), around the stroke with room for context,
/// scaled to at most [`FILL_MAX_SIDE`]. `None` when the spot is missing or lies outside the photo.
pub fn fill_job(src: &Rgb32f, info: &crate::SourceInfo, s: &lightcraft_develop::DevelopSettings, index: usize) -> Option<FillJob> {
    let effective = s.effective();
    let mut base = effective.sdr_rendition().into_owned();
    let spot = base.spots.get(index)?.clone();
    base.spots.truncate(index);
    let frame = crate::frame_for(src, info, &base, false);
    let (nw, nh) = (frame.ow.round().max(1.0) as usize, frame.oh.round().max(1.0) as usize);
    let (fw, fh) = frame.fit(nw, nh);
    let ppl = frame.px_per_long(fw);
    let to_out = frame.norm_to_out(fw, fh);
    let r = (spot.size * ppl).max(1.0);
    let pts: Vec<Point> = spot.points.iter().map(|p| to_out.apply(*p)).collect();
    let first = pts.first()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x, first.y);
    for p in &pts {
        (x0, y0, x1, y1) = (x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y));
    }
    let (bw, bh) = (x1 - x0 + 2.0 * r, y1 - y0 + 2.0 * r);
    let m = (2.0 * r).max(0.6 * bw.max(bh)).max(32.0);
    let (rx0, ry0) = ((x0 - r - m).max(0.0), (y0 - r - m).max(0.0));
    let (rx1, ry1) = ((x1 + r + m).min(fw as f64), (y1 + r + m).min(fh as f64));
    if !(rx1 > rx0 + 1.0 && ry1 > ry0 + 1.0) || ![rx0, ry0, rx1, ry1].iter().all(|v| v.is_finite()) {
        return None;
    }
    // scale so the crop fits the model
    let k = (FILL_MAX_SIDE as f64 / (rx1 - rx0).max(ry1 - ry0)).min(1.0);
    let (sw, sh) = (((fw as f64 * k).round() as usize).max(1), ((fh as f64 * k).round() as usize).max(1));
    let win = crate::PixelWindow {
        x: (rx0 * k).floor() as usize,
        y: (ry0 * k).floor() as usize,
        w: (((rx1 - rx0) * k).ceil() as usize).max(32),
        h: (((ry1 - ry0) * k).ceil() as usize).max(32),
    }
    .clamped(sw, sh);
    let req =
        crate::RenderRequest { apply_crop: false, depth: crate::OutputDepth::F32Linear, window: Some(win), ..crate::RenderRequest::fit(sw, sh) };
    let plan = crate::plan(src, info, &base, &req);
    let (w, h) = (plan.w, plan.h);
    let mut img = plan.frame.sample(src, w, h);
    crate::lin_cpu(&mut img, info, &plan);
    // the stroke in the scaled frame's pixels
    let to_out_s = frame.norm_to_out(sw, sh);
    let rs = (spot.size * frame.px_per_long(sw)).max(1.0);
    let spts: Vec<(f64, f64)> = spot.points.iter().map(|p| to_out_s.apply(*p)).map(|q| (q.x - win.x as f64, q.y - win.y as f64)).collect();
    let grow = rs * FILL_GROW + 2.0;
    let mut mask = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
            let d = spts.iter().map(|p| ((px - p.0).powi(2) + (py - p.1).powi(2)).sqrt()).fold(f64::MAX, f64::min);
            if d <= grow
                && let Some(v) = mask.get_mut(y * w + x)
            {
                *v = 1.0;
            }
        }
    }
    // exposure for the model: the brightest surroundings (99th percentile) at 0.9
    let mut lum: Vec<f32> =
        img.data.iter().zip(&mask).filter(|(_, m)| **m == 0.0).map(|(p, _)| p[0].max(p[1]).max(p[2])).filter(|v| v.is_finite()).collect();
    let gain = if lum.is_empty() {
        1.0
    } else {
        let i = ((lum.len() - 1) as f64 * 0.99) as usize;
        let (_, v, _) = lum.select_nth_unstable_by(i, |a, b| a.total_cmp(b));
        0.9 / v.max(1e-6)
    };
    let rgb: Vec<f32> = img.data.iter().flat_map(|p| p.map(|c| lightcraft_color::transfer::linear_to_srgb((c * gain).clamp(0.0, 1.0)))).collect();
    let to_norm = frame.out_to_norm(sw, sh);
    let corners =
        [(win.x, win.y), (win.x + w, win.y), (win.x, win.y + h), (win.x + w, win.y + h)].map(|(x, y)| to_norm.apply(Point::new(x as f64, y as f64)));
    let rect = [
        corners.iter().map(|c| c.x).fold(f64::MAX, f64::min),
        corners.iter().map(|c| c.y).fold(f64::MAX, f64::min),
        corners.iter().map(|c| c.x).fold(f64::MIN, f64::max),
        corners.iter().map(|c| c.y).fold(f64::MIN, f64::max),
    ];
    Some(FillJob { w, h, rgb, mask, lin: img.data, gain, to_norm, to_out: to_out_s, origin: (win.x as f64, win.y as f64), rect })
}

impl FillJob {
    /// The stored fill from the model's result `filled` (interleaved RGB in 0..1, as `rgb`).
    pub fn finish(&self, filled: &[f32]) -> Option<lightcraft_develop::SpotFill> {
        let (w, h) = (self.w, self.h);
        if filled.len() != w * h * 3 {
            return None;
        }
        // back to scene-linear: the model's pixels where it filled, the photo's elsewhere
        let lin: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                if self.mask.get(i).copied().unwrap_or(0.0) > 0.5 {
                    std::array::from_fn(|c| lightcraft_color::transfer::srgb_to_linear(filled.get(i * 3 + c).copied().unwrap_or(0.0)) / self.gain)
                } else {
                    self.lin.get(i).copied().unwrap_or([0.0; 3])
                }
            })
            .collect();
        let crop = Rgb32f { width: w, height: h, data: lin };
        // resample onto a grid along the normalized axes (orientation flips/rotations reorder it)
        let [x0, y0, x1, y1] = self.rect;
        let swap = (self.to_norm.apply(Point::new(1.0, 0.0)).y - self.to_norm.apply(Point::new(0.0, 0.0)).y).abs() > 1e-12;
        let (gw, gh) = if swap { (h, w) } else { (w, h) };
        let mut out = Vec::with_capacity(gw * gh * 3);
        for j in 0..gh {
            for i in 0..gw {
                let n = Point::new(x0 + (i as f64 + 0.5) / gw as f64 * (x1 - x0), y0 + (j as f64 + 0.5) / gh as f64 * (y1 - y0));
                let o = self.to_out.apply(n);
                out.extend_from_slice(&crop.sample_bilinear((o.x - self.origin.0) as f32, (o.y - self.origin.1) as f32));
            }
        }
        lightcraft_develop::SpotFill::new(gw, gh, self.rect, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lightcraft_develop::DevelopSettings;

    #[test]
    fn hard_edge_spots_at_zero_opacity_leave_every_pixel_unchanged() {
        for mode in [SpotMode::Clone, SpotMode::Heal, SpotMode::Remove, SpotMode::Fill] {
            for (cx, cy) in [(32.5, 24.5), (0.5, 0.5)] {
                let original = Rgb32f::from_fn(64, 48, |x, y| [0.2 + x as f32 / 128.0, 0.3 + y as f32 / 128.0, 0.4]);
                let mut img = original.clone();
                let frame = Frame::new(64, 48, &DevelopSettings::default(), true);
                let spot = Spot {
                    mode,
                    points: vec![Point::new(cx / 64.0, cy / 48.0)],
                    size: 5.0 / 64.0,
                    feather: 0.0,
                    opacity: 0.0,
                    source_offset: Some(Point::new(8.0 / 64.0, 0.0)),
                    fill: None,
                };
                apply(&mut img, &[spot], &frame, frame.px_per_long(64));
                for (i, (actual, expected)) in img.data.iter().zip(&original.data).enumerate() {
                    assert_eq!(actual, expected, "zero-opacity {mode:?} at ({cx}, {cy}), pixel {i}");
                }
            }
        }
    }

    #[test]
    fn hard_edge_clone_copies_inside_and_preserves_pixels_on_the_radius() {
        for (w, h, cx, cy, radius) in [(64, 48, 32.5, 24.5, 5.0), (128, 96, 64.5, 48.5, 10.0)] {
            let original = Rgb32f::from_fn(w, h, |x, y| [0.2 + x as f32 / w as f32, y as f32 / h as f32, 0.4]);
            let mut img = original.clone();
            let frame = Frame::new(w, h, &DevelopSettings::default(), true);
            let spot = Spot {
                mode: SpotMode::Clone,
                points: vec![Point::new(cx / w as f64, cy / h as f64)],
                size: radius / w as f64,
                feather: 0.0,
                opacity: 100.0,
                source_offset: Some(Point::new(0.125, 0.0)),
                fill: None,
            };
            apply(&mut img, &[spot], &frame, frame.px_per_long(w));
            let (x, y) = (cx as usize, cy as usize);
            assert_eq!(img.get(x, y), original.get(x + w / 8, y), "inside copies the source at {w}px");
            assert_eq!(img.get(x + radius as usize, y), original.get(x + radius as usize, y), "radius stays unchanged at {w}px");
            assert!(img.data.iter().flatten().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn remove_erases_a_dot() {
        // flat grey with a gentle gradient and a black dot in the middle
        let mut img = Rgb32f::from_fn(120, 80, |x, _| [0.2 + x as f32 * 0.001; 3]);
        for y in 36..44 {
            for x in 56..64 {
                img.set(x, y, [0.0; 3]);
            }
        }
        let frame = Frame::new(120, 80, &DevelopSettings::default(), true);
        let spot = Spot { points: vec![Point::new(0.5, 0.5)], size: 8.0 / 120.0, feather: 30.0, ..Default::default() };
        apply(&mut img, &[spot], &frame, frame.px_per_long(120));
        let c = img.get(60, 40);
        assert!((c[0] - 0.26).abs() < 0.03, "{c:?}");
    }

    /// A stand-in for the model: masked pixels become the mean of the unmasked ones.
    fn fake_model(job: &FillJob) -> Vec<f32> {
        let (mut acc, mut n) = ([0.0f32; 3], 0.0f32);
        for i in 0..job.w * job.h {
            if job.mask[i] == 0.0 {
                for c in 0..3 {
                    acc[c] += job.rgb[i * 3 + c];
                }
                n += 1.0;
            }
        }
        let mean = acc.map(|a| a / n);
        (0..job.w * job.h).flat_map(|i| if job.mask[i] > 0.5 { mean } else { [job.rgb[i * 3], job.rgb[i * 3 + 1], job.rgb[i * 3 + 2]] }).collect()
    }

    #[test]
    fn ai_fill_replaces_the_stroke_and_survives_a_reload() {
        let mut src = Rgb32f::from_fn(160, 120, |_, _| [0.3, 0.3, 0.3]);
        for y in 50..70 {
            for x in 70..90 {
                src.set(x, y, [0.9, 0.05, 0.05]);
            }
        }
        let info = crate::SourceInfo::default();
        let mut s = DevelopSettings::default();
        s.spots.push(Spot { points: vec![Point::new(0.5, 0.5)], size: 14.0 / 160.0, feather: 20.0, ..Default::default() });
        let job = fill_job(&src, &info, &s, 0).unwrap();
        assert!(job.mask.iter().any(|m| *m > 0.5) && job.mask.contains(&0.0));
        let fill = job.finish(&fake_model(&job)).unwrap();
        // stored and read back as in a catalog
        let fill: lightcraft_develop::SpotFill = serde_json::from_str(&serde_json::to_string(&fill).unwrap()).unwrap();
        s.spots[0].fill = Some(fill);
        let mut img = src.clone();
        let frame = Frame::new(160, 120, &s, true);
        apply(&mut img, &s.spots, &frame, frame.px_per_long(160));
        let c = img.get(80, 60);
        assert!((c[0] - 0.3).abs() < 0.03 && (c[1] - 0.3).abs() < 0.03, "{c:?}");
        // far from the stroke nothing changes
        assert_eq!(img.get(5, 5), src.get(5, 5));
        // and at another output size it lands in the same place
        let mut small = Rgb32f::from_fn(80, 60, |x, y| src.get(x * 2, y * 2));
        apply(&mut small, &s.spots, &frame, frame.px_per_long(80));
        let c = small.get(40, 30);
        assert!((c[0] - 0.3).abs() < 0.05, "{c:?}");
    }

    /// Vertical stripes (period `period` px) with a black blot over the middle.
    fn striped_with_blot(w: usize, h: usize, period: usize) -> Rgb32f {
        let mut img = Rgb32f::from_fn(w, h, |x, _| if (x / period).is_multiple_of(2) { [0.2; 3] } else { [0.7; 3] });
        let (cx, cy, r) = (w as f32 / 2.0, h as f32 / 2.0, w as f32 * 0.04);
        for y in 0..h {
            for x in 0..w {
                if (x as f32 + 0.5 - cx).hypot(y as f32 + 0.5 - cy) < r {
                    img.set(x, y, [0.0; 3]);
                }
            }
        }
        img
    }

    fn fill_spot() -> Spot {
        // a source offset is ignored by Fill
        Spot {
            mode: SpotMode::Fill,
            points: vec![Point::new(0.5, 0.5)],
            size: 0.06,
            feather: 0.0,
            source_offset: Some(Point::new(0.3, 0.0)),
            ..Default::default()
        }
    }

    #[test]
    fn fill_rebuilds_structure_deterministically() {
        let (w, h) = (200, 160);
        let mut img = striped_with_blot(w, h, 8);
        let frame = Frame::new(w, h, &DevelopSettings::default(), true);
        apply(&mut img, &[fill_spot()], &frame, frame.px_per_long(w));
        // the blot is gone and the stripes continue through it
        let mut err = 0.0f32;
        for x in 90..110 {
            let want = if (x / 8) % 2 == 0 { 0.2 } else { 0.7 };
            err += (img.get(x, 80)[0] - want).abs();
        }
        assert!(err / 20.0 < 0.12, "mean error {}", err / 20.0);
        let mut again = striped_with_blot(w, h, 8);
        apply(&mut again, &[fill_spot()], &frame, frame.px_per_long(w));
        assert_eq!(img.data, again.data);
    }

    #[test]
    fn fill_matches_between_preview_and_export_size() {
        // the same photo at two sizes: the half-size fill matches the full-size one, downsampled
        let (mut big, mut small) = (striped_with_blot(320, 240, 16), striped_with_blot(160, 120, 8));
        let (fb, fs) = (Frame::new(320, 240, &DevelopSettings::default(), true), Frame::new(160, 120, &DevelopSettings::default(), true));
        apply(&mut big, &[fill_spot()], &fb, fb.px_per_long(320));
        apply(&mut small, &[fill_spot()], &fs, fs.px_per_long(160));
        let (mut err, mut n) = (0.0f32, 0);
        for y in 50..70 {
            for x in 70..90 {
                let b: f32 = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().map(|(dx, dy)| big.get(2 * x + dx, 2 * y + dy)[0]).sum::<f32>() / 4.0;
                err += (b - small.get(x, y)[0]).abs();
                n += 1;
            }
        }
        assert!(err / n as f32 <= 0.08, "mean difference {}", err / n as f32);
    }

    #[test]
    fn fill_survives_hostile_spots() {
        let frame = Frame::new(40, 30, &DevelopSettings::default(), true);
        for (p, size) in [((0.0, 0.0), 2.0), ((-5.0, 9.0), 0.1), ((0.5, 0.5), 0.0), ((f64::NAN, 0.5), 0.1), ((0.5, 0.5), f64::INFINITY)] {
            let mut img = Rgb32f::from_fn(40, 30, |x, _| [x as f32 / 40.0; 3]);
            let spot = Spot { mode: SpotMode::Fill, points: vec![Point::new(p.0, p.1)], size, ..Default::default() };
            apply(&mut img, &[spot], &frame, frame.px_per_long(40));
            assert!(img.data.iter().flatten().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn clone_copies_source() {
        let mut img = Rgb32f::from_fn(100, 100, |x, _| if x < 50 { [0.1; 3] } else { [0.9; 3] });
        let frame = Frame::new(100, 100, &DevelopSettings::default(), true);
        let spot = Spot {
            mode: SpotMode::Clone,
            points: vec![Point::new(0.25, 0.5)],
            size: 0.05,
            feather: 0.0,
            source_offset: Some(Point::new(0.5, 0.0)),
            ..Default::default()
        };
        apply(&mut img, &[spot], &frame, frame.px_per_long(100));
        assert!(img.get(25, 50)[0] > 0.8);
        assert!(img.get(5, 50)[0] < 0.2);
    }

    fn with_spots(spots: Vec<Spot>) -> DevelopSettings {
        DevelopSettings { spots, ..Default::default() }
    }

    fn spot_at(x: f64, y: f64, size: f64, offset: Option<Point>) -> Spot {
        Spot { points: vec![Point::new(x, y)], size, source_offset: offset, ..Default::default() }
    }

    #[test]
    fn a_window_grows_to_hold_the_spots_that_touch_it_and_their_sources() {
        use crate::PixelWindow;
        let frame = Frame::new(4000, 3000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(4000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        // none: the window is left alone
        assert_eq!(window_for_reads(&with_spots(vec![]), &frame, 4000, 3000, ppl, win), win);
        // a spot far away is no business of this window
        let far = spot_at(0.9, 0.9, 0.01, Some(Point::new(0.02, 0.0)));
        assert_eq!(window_for_reads(&with_spots(vec![far]), &frame, 4000, 3000, ppl, win), win);
        // a spot inside it, reading from 300 px to the right, widens it to the source
        let near = spot_at(0.30, 0.38, 0.01, Some(Point::new(0.075, 0.0)));
        let g = window_for_reads(&with_spots(vec![near]), &frame, 4000, 3000, ppl, win);
        assert!(g.x <= win.x && g.x + g.w >= 1200 + 300 + 80 && g.y <= win.y && g.y + g.h >= win.y + win.h, "{g:?}");
        // a spot that only reaches in with its edge counts too
        let edge = spot_at(0.24, 0.38, 0.01, Some(Point::new(-0.05, 0.0)));
        assert!(window_for_reads(&with_spots(vec![edge]), &frame, 4000, 3000, ppl, win).x < win.x);
    }

    #[test]
    fn spot_windows_are_bounded_and_survive_hostile_spots() {
        use crate::PixelWindow;
        let frame = Frame::new(40_000, 30_000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(40_000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        // a spot whose source is a long way off would need an enormous window: not granted
        let wide = spot_at(0.03, 0.04, 0.01, Some(Point::new(0.6, 0.0)));
        let g = window_for_reads(&with_spots(vec![wide]), &frame, 40_000, 30_000, ppl, win);
        assert!(g.w <= MAX_SPOT_SPAN && g.h <= MAX_SPOT_SPAN, "{g:?}");
        for bad in [f64::NAN, f64::INFINITY, 1e300, -1e300] {
            let s = spot_at(bad, bad, bad, Some(Point::new(bad, bad)));
            let g = window_for_reads(&with_spots(vec![s]), &frame, 40_000, 30_000, ppl, win);
            assert_eq!(g, win);
        }
        let empty = Spot { points: vec![], ..Default::default() };
        assert_eq!(window_for_reads(&with_spots(vec![empty]), &frame, 40_000, 30_000, ppl, win), win);
    }

    // Given an Auto Mask stroke reaching into the window, the window grows to hold all of it
    #[test]
    fn a_window_grows_to_hold_an_auto_mask_stroke() {
        use crate::PixelWindow;
        use lightcraft_develop::{BrushStroke, Mask, MaskComponent, MaskOp, MaskShape};
        let frame = Frame::new(4000, 3000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(4000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        let stroke =
            |auto| BrushStroke { points: vec![Point::new(0.30, 0.38), Point::new(0.5, 0.38)], size: 0.01, auto_mask: auto, ..Default::default() };
        let mask = |auto| Mask {
            components: vec![MaskComponent { name: None, op: MaskOp::Add, invert: false, shape: MaskShape::Brush { strokes: vec![stroke(auto)] } }],
            ..Default::default()
        };
        let plain = DevelopSettings { masks: vec![mask(false)], ..Default::default() };
        assert_eq!(window_for_reads(&plain, &frame, 4000, 3000, ppl, win), win, "a plain stroke reads nothing around it");
        let auto = DevelopSettings { masks: vec![mask(true)], ..Default::default() };
        let g = window_for_reads(&auto, &frame, 4000, 3000, ppl, win);
        // the dabs near the window, not the whole stroke (it runs on to x = 2000)
        assert!(g.x <= 1000 && g.x + g.w > 1400 && g.x + g.w < 1700, "{g:?}");
        // a window at the far end of a long stroke does not grow back along all of it
        let far = window_for_reads(&auto, &frame, 4000, 3000, ppl, PixelWindow { x: 1900, ..win });
        assert!(far.x > 1500, "{far:?}");
        // a hidden mask reads nothing
        let hidden = DevelopSettings { masks: vec![Mask { visible: false, ..auto.masks[0].clone() }], ..Default::default() };
        assert_eq!(window_for_reads(&hidden, &frame, 4000, 3000, ppl, win), win);
    }

    // Given a window that would have to grow past the cap, there is no answer (never a wrong one)
    #[test]
    fn a_window_that_cannot_hold_what_it_reads_is_refused() {
        use crate::PixelWindow;
        let frame = Frame::new(40_000, 30_000, &DevelopSettings::default(), true);
        let ppl = frame.px_per_long(40_000);
        let win = PixelWindow { x: 1000, y: 1000, w: 400, h: 300 };
        let far = with_spots(vec![spot_at(0.03, 0.04, 0.01, Some(Point::new(0.6, 0.0)))]);
        assert_eq!(window_for_reads_checked(&far, &frame, 40_000, 30_000, ppl, win), None);
        let near = with_spots(vec![spot_at(0.03, 0.04, 0.01, Some(Point::new(0.02, 0.0)))]);
        assert!(window_for_reads_checked(&near, &frame, 40_000, 30_000, ppl, win).is_some());
        // a spot that doesn't touch the window is no business of it, however far it reads
        let elsewhere = with_spots(vec![spot_at(0.9, 0.9, 0.01, Some(Point::new(-0.8, 0.0)))]);
        assert_eq!(window_for_reads_checked(&elsewhere, &frame, 40_000, 30_000, ppl, win), Some(win));
    }
}

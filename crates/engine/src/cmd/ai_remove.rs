//! AI Remove (Z4K Studios addition, branch `z4k`): Remove spots filled by the LaMa model
//! (`lightcraft-inpaint`). The model is downloaded once on request (206 MB, Apache-2.0) into
//! `<config>/models/lama`; the fill it computes is stored with the spot, so renders, exports and
//! other machines never need it.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use lightcraft_develop::Spot;
use lightcraft_geom::Point;
use serde_json::{Value, json};

use super::masks::SPOT_SIZE;
use super::{CommandSpec, always, bad, cmd, f64_or, has_active};
use crate::{Result, Session};

/// Where the model is kept: `$LIGHTCRAFT_LAMA_DIR`, else `<config>/models/lama`.
pub fn model_dir() -> Option<PathBuf> {
    std::env::var_os("LIGHTCRAFT_LAMA_DIR")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| crate::config::config_dir().map(|d| lightcraft_inpaint::fetch::model_dir(&d.join("models"))))
}

#[derive(Default)]
struct Download {
    running: AtomicBool,
    done: AtomicU64,
    total: AtomicU64,
    error: Mutex<Option<String>>,
}

fn download() -> &'static Download {
    static D: OnceLock<Download> = OnceLock::new();
    D.get_or_init(Download::default)
}

fn model() -> &'static Mutex<Option<Arc<lightcraft_inpaint::Inpainter>>> {
    static M: OnceLock<Mutex<Option<Arc<lightcraft_inpaint::Inpainter>>>> = OnceLock::new();
    M.get_or_init(|| Mutex::new(None))
}

fn installed() -> Option<PathBuf> {
    model_dir().and_then(|d| lightcraft_inpaint::fetch::installed(&d))
}

fn status() -> Value {
    let d = download();
    json!({
        "installed": installed().is_some(),
        "downloading": d.running.load(Ordering::SeqCst),
        "done": d.done.load(Ordering::SeqCst),
        "total": d.total.load(Ordering::SeqCst),
        "error": d.error.lock().unwrap_or_else(PoisonError::into_inner).clone(),
        "modelDir": model_dir().map(|p| p.display().to_string()),
        "sizeBytes": lightcraft_inpaint::fetch::WEIGHTS_SIZE,
        "licence": "Apache-2.0 (big-lama, Samsung AI Center)",
    })
}

fn start_download() -> Result<Value> {
    let c = "aiRemove.install";
    let dir = model_dir().ok_or_else(|| bad(c, "no folder to keep the model in"))?;
    let d = download();
    if installed().is_some() || d.running.swap(true, Ordering::SeqCst) {
        return Ok(status());
    }
    *d.error.lock().unwrap_or_else(PoisonError::into_inner) = None;
    d.done.store(0, Ordering::SeqCst);
    d.total.store(lightcraft_inpaint::fetch::WEIGHTS_SIZE, Ordering::SeqCst);
    let spawned = std::thread::Builder::new().name("lama-download".into()).spawn(move || {
        let d = download();
        let mirrors = lightcraft_inpaint::fetch::mirrors(std::env::var(lightcraft_inpaint::fetch::MIRRORS_ENV).ok().as_deref());
        let cancel = AtomicBool::new(false);
        let r = lightcraft_inpaint::fetch::download(
            lightcraft_inpaint::fetch::FILES,
            &mirrors,
            &dir,
            &lightcraft_inpaint::fetch::Options::default(),
            &cancel,
            &mut |p| d.done.store(p.done, Ordering::SeqCst),
        );
        if let Err(e) = r {
            *d.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.to_string());
        }
        d.running.store(false, Ordering::SeqCst);
    });
    if let Err(e) = spawned {
        d.running.store(false, Ordering::SeqCst);
        return Err(bad(c, e.to_string()));
    }
    Ok(status())
}

fn loaded(c: &str) -> Result<Arc<lightcraft_inpaint::Inpainter>> {
    let mut m = model().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(l) = m.as_ref() {
        return Ok(l.clone());
    }
    let path = installed().ok_or_else(|| bad(c, "the AI Remove model isn't installed yet (run aiRemove.install)"))?;
    let l = Arc::new(lightcraft_inpaint::Inpainter::load(&path).map_err(|e| bad(c, e.to_string()))?);
    *m = Some(l.clone());
    Ok(l)
}

/// The fill for spot `i` of `dd` (the active photo's settings).
fn compute(s: &mut Session, dd: &lightcraft_develop::DevelopSettings, i: usize, c: &str) -> Result<lightcraft_develop::SpotFill> {
    let lama = loaded(c)?;
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let src = s.source_now(id, crate::media::SourceLevel::Full).map_err(|e| bad(c, e))?;
    let info = s.source_info(id);
    let job = lightcraft_pipeline::spots::fill_job(&src, &info, dd, i).ok_or_else(|| bad(c, "the spot is outside the photo"))?;
    let filled = lama.fill(&job.rgb, &job.mask, job.w, job.h).map_err(|e| bad(c, e.to_string()))?;
    job.finish(&filled).ok_or_else(|| bad(c, "couldn't store the result"))
}

/// Compute and store the fill of spot `i` of the active photo (one undo step).
pub(crate) fn fill_spot(s: &mut Session, i: usize, c: &str, label: &str) -> Result<Value> {
    let t = std::time::Instant::now();
    let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
    let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
    if i >= dd.spots.len() {
        return Err(bad(c, "no such spot"));
    }
    let fill = compute(s, &dd, i, c)?;
    if let Some(sp) = dd.spots.get_mut(i) {
        sp.fill = Some(fill);
    }
    s.set_develop(id, dd, label)?;
    s.active_spot = Some(i);
    Ok(json!({"index": i, "ms": t.elapsed().as_millis() as u64}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(query "aiRemove.status", "AI Remove Status", [], None, "{} → {installed, downloading, done, total, error, modelDir, sizeBytes, licence}", always, |_, _| Ok(status())),
        cmd!(
            "aiRemove.install",
            "Download AI Remove Model",
            [],
            None,
            "{} — downloads the AI Remove model (LaMa, 206 MB, Apache-2.0) in the background; poll aiRemove.status",
            always,
            |_, _| start_download()
        ),
        cmd!(
            "spot.aiRemove",
            "AI Remove",
            [],
            None,
            "{points: [[x,y],…] normalized (a click or a brushed path), size?: brush radius as a fraction of the long edge, feather?, index?: refill an existing spot instead} — removes what the stroke covers with the AI model (filled from its surroundings, works on RAW, non-destructive); returns {index, ms}",
            has_active,
            |s, p| {
                let c = "spot.aiRemove";
                if let Some(i) = p.get("index").and_then(Value::as_u64) {
                    return fill_spot(s, i as usize, c, "AI Remove");
                }
                let pts: Vec<Point> = p
                    .get("points")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(|q| Some(Point::new(q.get(0)?.as_f64()?, q.get(1)?.as_f64()?))).filter(|q| q.x.is_finite() && q.y.is_finite()).collect())
                    .unwrap_or_default();
                if pts.is_empty() {
                    return Err(bad(c, "missing points"));
                }
                let t = std::time::Instant::now();
                let d = Spot::default();
                let spot = Spot {
                    points: pts,
                    size: f64_or(p, "size", d.size).clamp(SPOT_SIZE.0, SPOT_SIZE.1),
                    feather: f64_or(p, "feather", 30.0).clamp(0.0, 100.0),
                    ..d
                };
                let id = s.active().ok_or_else(|| bad(c, "no active photo"))?;
                let mut dd = (*s.develop_of(id).unwrap_or_default()).clone();
                dd.spots.push(spot);
                let i = dd.spots.len() - 1;
                let fill = compute(s, &dd, i, c)?;
                if let Some(sp) = dd.spots.get_mut(i) {
                    sp.fill = Some(fill);
                }
                s.set_develop(id, dd, "AI Remove")?;
                s.active_spot = Some(i);
                Ok(json!({"index": i, "ms": t.elapsed().as_millis() as u64}))
            }
        ),
    ]
}

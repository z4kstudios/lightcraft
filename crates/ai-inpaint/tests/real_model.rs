//! Compares the Rust LaMa with the PyTorch TorchScript reference on a synthetic 128 × 96 image,
//! on the CPU and on the default device (Metal on macOS).
//! Ignored by default: needs the real model (`LIGHTCRAFT_LAMA_PT`). The reference tensors
//! (`ref_img.f32`, `ref_mask.f32`, `ref_out.f32`, planar) are in `tests/data`, or in `LIGHTCRAFT_LAMA_REF`.
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

fn read(p: PathBuf) -> Vec<f32> {
    std::fs::read(p).unwrap().as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

fn check(device: &candle_core::Device) {
    let model = PathBuf::from(std::env::var("LIGHTCRAFT_LAMA_PT").unwrap());
    let dir =
        std::env::var("LIGHTCRAFT_LAMA_REF").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data"));
    let (w, h) = (128, 96);
    let img = read(dir.join("ref_img.f32"));
    let mask = read(dir.join("ref_mask.f32"));
    let want = read(dir.join("ref_out.f32"));
    let n = w * h;
    let mut rgb = vec![0.0; n * 3];
    for i in 0..n {
        for c in 0..3 {
            rgb[i * 3 + c] = img[c * n + i];
        }
    }
    let t = std::time::Instant::now();
    let lama = lightcraft_ai_inpaint::Inpainter::load_on(&model, device).unwrap();
    eprintln!("{device:?}: load {:?}", t.elapsed());
    for _ in 0..2 {
        let t = std::time::Instant::now();
        let got = lama.fill(&rgb, &mask, w, h).unwrap();
        eprintln!("{device:?}: fill {:?}", t.elapsed());
        let mut max: f32 = 0.0;
        for i in 0..n {
            for c in 0..3 {
                max = max.max((got[i * 3 + c] - want[c * n + i]).abs());
            }
        }
        eprintln!("{device:?}: max abs diff {max}");
        assert!(max < 2e-3, "max abs diff {max}");
    }
}

#[test]
#[ignore]
fn matches_pytorch_reference() {
    check(&candle_core::Device::Cpu);
}

#[test]
#[ignore]
fn matches_pytorch_reference_on_the_default_device() {
    check(&lightcraft_ai_inpaint::default_device());
}

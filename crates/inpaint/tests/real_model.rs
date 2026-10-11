//! Compares the Rust LaMa with the PyTorch TorchScript reference on a synthetic 128 × 96 image.
//! Ignored by default: needs the real model (`LIGHTCRAFT_LAMA_PT`) and the reference tensors
//! (`LIGHTCRAFT_LAMA_REF`, a folder with `ref_img.f32`, `ref_mask.f32`, `ref_out.f32`, planar).
#![cfg(not(target_arch = "wasm32"))]

use std::path::PathBuf;

fn read(p: PathBuf) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

#[test]
#[ignore]
fn matches_pytorch_reference() {
    let model = PathBuf::from(std::env::var("LIGHTCRAFT_LAMA_PT").unwrap());
    let dir = PathBuf::from(std::env::var("LIGHTCRAFT_LAMA_REF").unwrap());
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
    let lama = lightcraft_inpaint::Inpainter::load_on(&model, &candle_core::Device::Cpu).unwrap();
    eprintln!("load {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let got = lama.fill(&rgb, &mask, w, h).unwrap();
    eprintln!("fill {:?}", t.elapsed());
    let mut max: f32 = 0.0;
    for i in 0..n {
        for c in 0..3 {
            max = max.max((got[i * 3 + c] - want[c * n + i]).abs());
        }
    }
    eprintln!("max abs diff {max}");
    assert!(max < 2e-3, "max abs diff {max}");
}

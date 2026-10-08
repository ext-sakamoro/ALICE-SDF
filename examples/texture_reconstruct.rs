//! Fit a procedural noise formula to an image and reconstruct it at another
//! resolution.
//!
//! Run: `cargo run --example texture_reconstruct --features texture-fit`
//!
//! The source image is synthesized from the fitter's own law, so the fit can
//! be compared with a known answer (the fitter is pinned in
//! `tests/test_texture_fit_oracle.rs`).
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "example scene and animation values; the crate output these examples show is computed by the library"
)]

use alice_sdf::texture::{eval_octave, fit_texture, reconstruct, TextureFitConfig};

fn main() {
    let (w, h) = (64u32, 48u32);
    // bias + one octave of the fitter's own noise law
    let pixels: Vec<u8> = (0..w * h)
        .map(|i| {
            let (u, v) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            let value = 0.5 + eval_octave(u, v, 0.2, 4.0, [0.1, 0.3], 0, 0.0);
            (value.clamp(0.0, 1.0) * 255.0).round() as u8
        })
        .collect();
    let path = std::env::temp_dir().join("alice_sdf_texture_reconstruct.png");
    image::GrayImage::from_raw(w, h, pixels.clone())
        .unwrap()
        .save(&path)
        .unwrap();

    let fit = fit_texture(&path, &TextureFitConfig::default()).expect("fit");
    std::fs::remove_file(&path).ok();
    let octaves: usize = fit.octaves.iter().map(Vec::len).sum();
    println!(
        "fit: bias {:.4}, {octaves} octave(s), PSNR {:.2} dB, NMSE {:.5}",
        fit.bias[0], fit.psnr_db, fit.nmse
    );

    // at the source resolution the reconstruction is the fitted signal
    let same = reconstruct(&fit, w as usize, h as usize);
    let mse: f64 = same
        .iter()
        .zip(&pixels)
        .map(|(&r, &p)| (f64::from(r) - f64::from(p) / 255.0).powi(2))
        .sum::<f64>()
        / f64::from(w * h);
    let psnr = 10.0 * (1.0 / mse.max(1e-30)).log10();
    println!("reconstruct {w}x{h}: PSNR against the source {psnr:.2} dB");
    assert!(
        (psnr - f64::from(fit.psnr_db)).abs() < 0.5,
        "reported PSNR is the reconstruction's"
    );

    // at twice the resolution: same formula, so even pixels match the source grid
    let big = reconstruct(&fit, 2 * w as usize, 2 * h as usize);
    let worst = (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (x, y)))
        .map(|(x, y)| (big[2 * x + 2 * y * 2 * w as usize] - same[x + y * w as usize]).abs())
        .fold(0.0f32, f32::max);
    println!(
        "reconstruct {}x{}: worst difference at the shared sample points {worst:.2e}",
        2 * w,
        2 * h
    );
    assert!(worst < 1e-5);
    assert!(big.iter().all(|v| (0.0..=1.0).contains(v)));
}

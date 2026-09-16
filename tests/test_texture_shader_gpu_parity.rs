//! GPU ↔ CPU parity for the shader `texture::generate_shader` emits.
//!
//! `reconstruct` is the CPU rendering of a `TextureFitResult`; the emitted
//! WGSL `procedural_texture(uv)` must produce the same image on the GPU.
//! Since 1.14.0 both sides use the crate's PCG lattice noise (integer hash,
//! `u32 → f32`, trilinear blend), so the only float work is the blend and
//! the rotation — the tolerance is a few ulp, not a "close enough" band.
//! Runs on the CI `gpu-parity` job (lavapipe, `ALICE_SDF_REQUIRE_GPU=1`).
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "gpu", feature = "texture-fit"))]

use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;
use alice_sdf::texture::{
    generate_shader, reconstruct, FittedOctave, ShaderLanguage, TextureFitResult,
};

/// A fit result written by hand: rotated and axis-aligned octaves (the two
/// emit paths), phases off the lattice, seeds 0..3.
fn fit_result() -> TextureFitResult {
    TextureFitResult {
        width: 64,
        height: 48,
        channels: 1,
        bias: vec![0.45],
        octaves: vec![vec![
            FittedOctave {
                amplitude: 0.28,
                frequency: 3.5,
                phase: [0.7, 1.3],
                seed: 0,
                rotation: 0.0,
            },
            FittedOctave {
                amplitude: -0.11,
                frequency: 7.25,
                phase: [2.1, 0.4],
                seed: 1,
                rotation: 0.6,
            },
            FittedOctave {
                amplitude: 0.05,
                frequency: 13.0,
                phase: [0.0, 5.5],
                seed: 2,
                rotation: -1.1,
            },
            FittedOctave {
                amplitude: 0.03,
                frequency: 29.0,
                phase: [1.0, 1.0],
                seed: 3,
                rotation: 0.0,
            },
        ]],
        psnr_db: 0.0,
        nmse: 0.0,
    }
}

/// Wrap the emitted texture function in the compute skeleton `GpuEvaluator`
/// expects (a point buffer, workgroup size 256 = `from_wgsl` default): `p.xy`
/// is the uv.
fn compute_shader(texture_wgsl: &str) -> String {
    format!(
        r"struct InputPoint {{
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}}

struct OutputDistance {{
    distance: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}}

@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;

{texture_wgsl}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{
        return;
    }}
    let point = input_points[idx];
    output_distances[idx].distance = procedural_texture(vec2<f32>(point.x, point.y));
}}
"
    )
}

#[test]
fn emitted_wgsl_matches_reconstruct() {
    let r = fit_result();
    let wgsl = generate_shader(&r, ShaderLanguage::Wgsl, "oracle");
    let gpu = match GpuEvaluator::from_wgsl(&compute_shader(&wgsl)) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
            );
            eprintln!("skipping texture shader GPU parity: {e}");
            return;
        }
    };

    let (w, h) = (r.width as usize, r.height as usize);
    // `reconstruct` samples u = x / width, v = y / height (pixel corners).
    let points: Vec<Vec3> = (0..w * h)
        .map(|i| Vec3::new((i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.0))
        .collect();
    let got = gpu.eval_batch(&points).expect("gpu eval");
    let want = reconstruct(&r, w, h);
    assert_eq!(got.len(), want.len());

    let mut worst = (0.0f32, 0usize);
    for (i, (g, c)) in got.iter().zip(&want).enumerate() {
        let d = (g - c).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    let (x, y) = (worst.1 % w, worst.1 / w);
    eprintln!(
        "texture shader: max |gpu - cpu| = {:.3e} at pixel ({x}, {y}) over {} pixels (cpu {} gpu {})",
        worst.0,
        got.len(),
        want[worst.1],
        got[worst.1]
    );
    // Integer hash → identical corner values; the trilinear blend, rotation
    // and `uv * f + phase` differ by fma / rounding only.
    assert!(
        worst.0 < 2e-5,
        "emitted WGSL drifts from reconstruct by {:.3e} at ({x}, {y})",
        worst.0
    );
    // The image is not degenerate (the octaves actually contribute).
    let (lo, hi) = want
        .iter()
        .fold((1.0f32, 0.0f32), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    assert!(
        hi - lo > 0.3,
        "reconstruction range [{lo}, {hi}] too flat to be a parity test"
    );
}

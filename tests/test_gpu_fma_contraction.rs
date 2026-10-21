//! ALICE-Law G3 on the wgpu path: `GpuEvaluator` must not let the GPU compiler
//! fuse `a * b + c` into one rounding.
//!
//! The discriminator: with a = b = 1 + 2^-12 and c = -(1 + 2^-11),
//! a * b = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so two roundings give exactly 0
//! and one fused rounding gives 2^-24.
//!
//! Measured on Apple silicon through wgpu 24 (Metal backend): the three forms give
//! `0x33800000`, `0xb3800000` and `0x33800000`, the fused values; a WGSL `fma` call
//! gives `0x33800000` as the control. The MSL path (`MslShader`, which puts
//! `#pragma METAL fp contract(off)` first) does not fuse
//! (`tests/test_msl_metal_oracle.rs`). With the `msl` feature on macOS
//! `GpuEvaluator` runs on Metal through `MslShader` (fast math off), and this is a
//! normal test there; on the wgpu path (macOS without `msl`, and every other
//! target, where Vulkan, DX12 and GL are not measured) it stays ignored as a
//! known gap, to be run by hand on those backends.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: mul_add is the reference for the fused value"
)]
#![cfg(feature = "gpu")]

use alice_sdf::compiled::GpuEvaluator;
use glam::Vec3;

fn kernel_wgsl(body: &str) -> String {
    format!(
        "struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 {{ return {body}; }}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{ return; }}
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}}
"
    )
}

fn discriminator() -> Vec3 {
    let (a, b, c) = (
        1.0 + 2f32.powi(-12),
        1.0 + 2f32.powi(-12),
        -(1.0 + 2f32.powi(-11)),
    );
    assert_eq!((a * b + c).to_bits(), 0.0f32.to_bits());
    assert_eq!(a.mul_add(b, c).to_bits(), 2f32.powi(-24).to_bits());
    Vec3::new(a, b, c)
}

/// (WGSL body, the two-roundings result)
const FORMS: [(&str, f32); 3] = [
    ("p.x * p.y + p.z", 0.0),
    ("-(p.x * p.y + p.z)", -0.0),
    ("(p.x * p.y) + p.z", 0.0),
];

#[test]
#[cfg_attr(
    not(all(target_os = "macos", feature = "msl")),
    ignore = "src gap: gpu fma contraction (wgpu path: Metal without msl contracts; Vulkan, DX12 and GL unmeasured)"
)]
fn gpu_evaluator_does_not_fuse_a_multiply_add() {
    let pts = [discriminator()];
    let mut compared = 0;
    for (body, want) in FORMS {
        let evaluator = match GpuEvaluator::from_wgsl(&kernel_wgsl(body)).map(|c| c.into_parts().0)
        {
            Ok(e) => e,
            Err(e) => {
                for var in ["ALICE_SDF_REQUIRE_GPU", "ALICE_SDF_REQUIRE_METAL"] {
                    assert!(
                        std::env::var_os(var).is_none(),
                        "{var} is set but no GPU device was found: {e}"
                    );
                }
                eprintln!("skipping GPU fma contraction: no GPU device ({e})");
                return;
            }
        };
        let got = evaluator.eval_batch(&pts).expect("eval on the GPU")[0];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "`{body}` through GpuEvaluator gave {got:e} (bits {:#010x}); two roundings give {want:e}, one fused rounding gives the 2^-24 magnitude",
            got.to_bits()
        );
        compared += 1;
    }
    assert_eq!(compared, FORMS.len());
}

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
//! normal test there, and `GpuEvaluator` runs that way on macOS whenever `gpu` is on. On Linux it
//! reads marked SPIR-V through Vulkan passthrough (`src/compiled/wgsl/spirv_no_contraction.rs`),
//! and these are normal tests there too. Windows has no device in CI and stays ignored.
//!
//! What cannot be controlled says so: `gpu_context::contraction` answers per path and backend, and
//! the last test holds that answer against the driver.
//!
//! A driver also fuses inside a builtin, and whether it does depends on what else the shader
//! computes: `length(p.xz)` next to `length(p)` shares `x * x`. Measured on Vulkan before the
//! builtins were written out: 259 to 1330 of 4096 values differ, by the builtin. So the second test
//! compares each builtin alone with the same builtin in such a shader, and with its written-out
//! form.
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
@compute @workgroup_size(256)
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
    not(any(target_os = "linux", target_os = "macos")),
    ignore = "src gap: gpu fma contraction (no device on this target in CI; DX12 and GL are refused)"
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

/// A builtin, and the same value written out in marked arithmetic.
const BUILTINS: [(&str, &str); 7] = [
    ("length(p.xz)", "sqrt(p.x * p.x + p.z * p.z)"),
    ("length(p)", "sqrt(p.x * p.x + p.y * p.y + p.z * p.z)"),
    ("dot(p, p)", "p.x * p.x + p.y * p.y + p.z * p.z"),
    (
        "distance(p, vec3<f32>(0.1, 0.2, 0.3))",
        "sqrt((p.x - 0.1) * (p.x - 0.1) + (p.y - 0.2) * (p.y - 0.2) + (p.z - 0.3) * (p.z - 0.3))",
    ),
    (
        "normalize(p).x",
        "p.x / sqrt(p.x * p.x + p.y * p.y + p.z * p.z)",
    ),
    ("cross(p, p.zxy).y", "p.z * p.z - p.x * p.y"),
    ("mix(p.x, p.y, p.z)", "p.x * (1.0 - p.z) + p.y * p.z"),
];

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "src gap: gpu fma contraction (a builtin's sum is written out only on the Vulkan path; MSL with fast math off is measured by the test above)"
)]
fn a_builtin_does_not_depend_on_what_else_the_shader_computes() {
    let mut state = 0x1234_5678_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        2.0 * ((state >> 40) as f32 / (1u64 << 24) as f32) - 1.0
    };
    let pts: Vec<Vec3> = (0..1024)
        .map(|_| Vec3::new(2.0 * next(), 2.0 * next(), 2.0 * next()))
        .collect();
    // never taken, so only what the shader computes changes, and it shares `x * x`
    let beside = |body: &str| {
        format!("select({body}, -1.0, length(p.xz) + length(p.xy) + length(p) > 1.0e30)")
    };
    let run = |body: &str| -> Option<Vec<f32>> {
        match GpuEvaluator::from_wgsl(&kernel_wgsl(body)) {
            Ok(e) => Some(
                e.into_parts()
                    .0
                    .eval_batch(&pts[..1024])
                    .expect("eval on the GPU"),
            ),
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but no GPU device was found: {e}"
                );
                eprintln!("skipping: no GPU device ({e})");
                None
            }
        }
    };
    let mut compared = 0;
    for (builtin, written) in BUILTINS {
        let Some(alone) = run(builtin) else { return };
        let in_context = run(&beside(builtin)).expect("the device was there");
        let out = run(written).expect("the device was there");
        for (i, p) in pts.iter().enumerate() {
            assert_eq!(
                alone[i].to_bits(),
                in_context[i].to_bits(),
                "`{builtin}` at {p}: {} alone, {} beside other lengths",
                alone[i],
                in_context[i]
            );
            assert_eq!(
                alone[i].to_bits(),
                out[i].to_bits(),
                "`{builtin}` at {p}: {}, written out `{written}` gives {}",
                alone[i],
                out[i]
            );
            compared += 1;
        }
    }
    assert_eq!(compared, BUILTINS.len() * pts.len());
}

#[test]
fn the_answer_per_path_and_backend() {
    use alice_sdf::gpu_context::{contraction, Contraction, GpuPath};
    for path in [GpuPath::Evaluator, GpuPath::WgpuShader] {
        assert_eq!(
            contraction(path, wgpu::Backend::Vulkan),
            Contraction::Forbidden
        );
        assert_eq!(contraction(path, wgpu::Backend::Dx12), Contraction::Refused);
        assert_eq!(contraction(path, wgpu::Backend::Gl), Contraction::Refused);
        assert_eq!(
            contraction(path, wgpu::Backend::BrowserWebGpu),
            Contraction::NotControlled
        );
    }
    assert_eq!(
        contraction(GpuPath::WgpuShader, wgpu::Backend::Metal),
        Contraction::NotControlled
    );
    let metal_evaluator = if cfg!(target_os = "macos") {
        Contraction::Forbidden
    } else {
        Contraction::NotControlled
    };
    assert_eq!(
        contraction(GpuPath::Evaluator, wgpu::Backend::Metal),
        metal_evaluator
    );
}

/// GLSL compute reaches the driver through wgpu on every backend, so on Metal it is the known gap
/// and on Vulkan it is forbidden: the query must say which, and the driver must not contradict it.
#[cfg(feature = "glsl")]
#[test]
fn the_wgpu_path_answers_for_itself() {
    use alice_sdf::gpu_context::Contraction;
    let src = "#version 450
layout(local_size_x = 256) in;
struct InputPoint { float x; float y; float z; float pad; };
struct OutputDistance { float distance; float pad1; float pad2; float pad3; };
layout(std430, set = 0, binding = 0) readonly buffer I { InputPoint input_points[]; };
layout(std430, set = 0, binding = 1) buffer O { OutputDistance output_distances[]; };
layout(std140, set = 0, binding = 2) uniform C { uint point_count; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) { return; }
    InputPoint p = input_points[idx];
    output_distances[idx].distance = p.x * p.y + p.z;
}
";
    let evaluator = match GpuEvaluator::from_glsl_compute(src).map(|c| c.into_parts().0) {
        Ok(e) => e,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but no GPU device was found: {e}"
            );
            eprintln!("skipping: {e}");
            return;
        }
    };
    let got = evaluator
        .eval_batch(&[discriminator()])
        .expect("eval on the GPU")[0];
    match evaluator.contraction() {
        Contraction::Forbidden => assert_eq!(got.to_bits(), 0.0f32.to_bits(), "got {got:e}"),
        // KNOWN-DEFECT-ARM: known defect: gpu-wgpu-metal-contraction: wgpu's Metal path has no way to turn fast math off
        Contraction::NotControlled => {
            eprintln!(
                "known defect: gpu-wgpu-metal-contraction: this path may fuse; it gave {got:e}"
            );
        }
        Contraction::Refused => panic!("a refused path compiled a shader"),
    }
}

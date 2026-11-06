//! `GpuEvaluator` dispatches `point_count.div_ceil(size)` workgroups, so a size other than the
//! shader's own loses points or runs extra groups. With too few, the rest of the output stays 0 and
//! the call still returns `Ok`: a wrong value with no error. So the size comes from the shader, and a
//! shader whose size cannot be known is refused.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuError, GpuEvaluator};
use glam::Vec3;

fn kernel(attribute: &str, declarations: &str, entry: &str) -> String {
    format!(
        "struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
{declarations}
@compute {attribute}
fn {entry}(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{ return; }}
    output_distances[idx].distance = input_points[idx].x + 1.0;
}}
"
    )
}

/// 1000 points, no multiple of a size below, each with its own nonzero answer
fn points() -> [Vec3; 1000] {
    std::array::from_fn(|i| Vec3::new(i as f32, 0.0, 0.0))
}

fn built<T>(what: &str, r: Result<T, GpuError>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(GpuError::NoAdapter | GpuError::DeviceCreation(_)) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but no GPU device was found"
            );
            eprintln!("skipping {what}: no GPU device");
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

fn check(what: &str, gpu: &GpuEvaluator) {
    let pts = points();
    let got = gpu.eval_batch(&pts).expect("eval on the GPU");
    assert_eq!(got.len(), pts.len());
    for (i, (v, p)) in got.iter().zip(&pts).enumerate() {
        assert_eq!(
            v.to_bits(),
            (p.x + 1.0).to_bits(),
            "{what}: point {i} gave {v}"
        );
    }
}

#[test]
fn every_point_is_evaluated_at_the_sizes_a_shader_declares() {
    for size in [1, 64, 256] {
        let text = kernel(&format!("@workgroup_size({size})"), "", "main");
        let what = format!("size {size}");
        let Some(c) = built(&what, GpuEvaluator::from_wgsl(&text)) else {
            return;
        };
        check(&what, &c.into_parts().0);
    }
}

#[test]
fn the_async_constructor_reads_the_size_too() {
    let text = kernel("@workgroup_size(64)", "", "main");
    let Some(c) = built(
        "async",
        pollster::block_on(GpuEvaluator::from_wgsl_async(&text)),
    ) else {
        return;
    };
    check("async", &c.into_parts().0);
}

#[test]
fn an_override_with_a_default_is_read() {
    let text = kernel("@workgroup_size(wg)", "override wg: u32 = 64;", "main");
    let Some(c) = built("override", GpuEvaluator::from_wgsl(&text)) else {
        return;
    };
    check("override", &c.into_parts().0);
}

#[cfg(feature = "glsl")]
#[test]
fn a_glsl_shader_declares_its_size_as_well() {
    let text = "#version 450
layout(local_size_x = 64) in;
struct InputPoint { float x; float y; float z; float pad; };
struct OutputDistance { float distance; float pad1; float pad2; float pad3; };
layout(std430, set = 0, binding = 0) readonly buffer I { InputPoint input_points[]; };
layout(std430, set = 0, binding = 1) buffer O { OutputDistance output_distances[]; };
layout(std140, set = 0, binding = 2) uniform C { uint point_count; };
void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) { return; }
    output_distances[idx].distance = input_points[idx].x + 1.0;
}
";
    let Some(c) = built("glsl", GpuEvaluator::from_glsl_compute(text)) else {
        return;
    };
    check("glsl", &c.into_parts().0);
}

/// Refused before any device is asked for, so these hold with or without one
#[test]
fn a_shader_whose_size_cannot_be_known_is_refused() {
    let no_default = kernel("@workgroup_size(wg)", "override wg: u32;", "main");
    let other_entry = kernel("@workgroup_size(64)", "", "other");
    for (what, text) in [
        ("override with no default", no_default),
        ("no `main`", other_entry),
    ] {
        let got = GpuEvaluator::from_wgsl(&text).map(|c| c.into_parts().0);
        assert!(
            matches!(got, Err(GpuError::ShaderCompilation(_))),
            "{what}: {:?}",
            got.err()
        );
    }
}

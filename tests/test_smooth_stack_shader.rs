//! The smooth section stack on the shader backends: the WGSL and GLSL emits parse and validate in
//! naga, and the WGSL / GLSL text run on the GPU gives the CPU tree's value (when an adapter is
//! present; `ALICE_SDF_REQUIRE_GPU=1` makes "no adapter" a failure). The stacks: a circle splitting
//! into two (the branch case, two loops per level from the middle up), a rotated and translated
//! copy (the transforms around it), and a union with a sphere.
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::{Quat, Vec2};
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    // control points of a closed uniform cubic B-spline close to a circle of radius r
    let k = 3.0 / (2.0 + (std::f32::consts::TAU / m as f32).cos());
    (0..m)
        .map(|j| {
            let a = std::f32::consts::TAU * j as f32 / m as f32;
            c + Vec2::new(a.cos(), a.sin()) * (r * k)
        })
        .collect()
}

fn split_stack() -> SdfNode {
    let levels: Vec<f32> = (0..8).map(|i| -0.35 + 0.1 * i as f32).collect();
    let loops: Vec<Vec<Vec<Vec2>>> = levels
        .iter()
        .map(|&y| {
            if y < 0.0 {
                vec![circle(0.30, Vec2::ZERO, 12)]
            } else {
                let s = 0.18 + 0.4 * y;
                vec![circle(0.16, Vec2::new(-s, 0.0), 8), circle(0.16, Vec2::new(s, 0.0), 8)]
            }
        })
        .collect();
    SdfNode::smooth_section_stack(levels, loops, 0.01, 0.1).expect("a valid stack")
}

fn trees() -> Vec<(&'static str, SdfNode)> {
    let st = split_stack();
    vec![
        ("stack", st.clone()),
        ("stack_rotated", st.clone().rotate(Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4)).translate(0.1, -0.2, 0.05)),
        ("stack_union", st.union(SdfNode::sphere(0.2).translate(0.0, 0.45, 0.0))),
    ]
}

fn points(n: usize) -> Vec<Vec3> {
    let mut s: u64 = 0x5e55_0000_0000_0087;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32) * 1.6 - 0.8
    };
    (0..n).map(|_| Vec3::new(r(), r(), r())).collect()
}

fn validate(module: &naga::Module) -> Result<(), String> {
    Validator::new(ValidationFlags::all(), Capabilities::all()).validate(module).map(|_| ()).map_err(|e| format!("{e:?}"))
}

#[test]
fn the_stack_transpiles_to_valid_wgsl() {
    let mut n = 0;
    for (name, node) in trees() {
        let src = WgslShader::transpile(&node, TranspileMode::Hardcoded).source;
        assert!(src.contains("alice_sss_"), "{name}: the stack's functions are not in the WGSL");
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
        validate(&module).unwrap_or_else(|e| panic!("{name}: WGSL validation: {e}"));
        n += 1;
    }
    assert_eq!(n, 3, "every tree validated");
}

#[cfg(feature = "glsl")]
#[test]
fn the_stack_transpiles_to_valid_glsl() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    let mut n = 0;
    for (name, node) in trees() {
        let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        assert!(shader.source.contains("alice_sss_"), "{name}: the stack's functions are not in the GLSL");
        let source = format!("{}\nout vec4 alice_frag;\nvoid main() {{ alice_frag = vec4(sdf_eval(vec3(0.1, 0.2, 0.3))); }}\n", shader.source);
        let mut fe = naga::front::glsl::Frontend::default();
        let opts = naga::front::glsl::Options { stage: naga::ShaderStage::Fragment, defines: naga::FastHashMap::default() };
        let module = fe.parse(&opts, &source).unwrap_or_else(|e| panic!("{name}: GLSL parse: {e:?}"));
        validate(&module).unwrap_or_else(|e| panic!("{name}: GLSL validation: {e}"));
        n += 1;
    }
    assert_eq!(n, 3, "every tree validated");
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use alice_sdf::compiled::GpuEvaluator;

    fn worst(node: &SdfNode, got: &[f32], pts: &[Vec3]) -> (f32, Vec3) {
        let mut w = (0.0f32, Vec3::ZERO);
        for (p, g) in pts.iter().zip(got) {
            let d = (g - eval(node, *p)).abs();
            if d > w.0 {
                w = (d, *p);
            }
        }
        w
    }

    fn no_gpu(e: &dyn std::fmt::Display) {
        assert!(std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(), "ALICE_SDF_REQUIRE_GPU is set but the GPU path failed: {e}");
        eprintln!("skipping the stack's GPU parity: {e}");
    }

    /// The tolerance: the closest-point search runs the same statements, so the GPU and the CPU differ
    /// by rounding only (no fused contraction is assumed either way); 1e-5 in a ±0.8 box
    #[test]
    fn the_wgsl_stack_gives_the_cpu_value() {
        let pts = points(4096);
        for (name, node) in trees() {
            let gpu = match GpuEvaluator::new(&node) {
                Ok(g) => g,
                Err(e) => return no_gpu(&e),
            };
            let got = gpu.eval_batch(&pts).expect("gpu eval");
            let (w, at) = worst(&node, &got, &pts);
            eprintln!("{name}: WGSL max |gpu - cpu| = {w:.3e} over {}", pts.len());
            assert!(w <= 1e-5, "{name}: WGSL/CPU drift {w:.3e} at {at:?}");
        }
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn the_glsl_stack_gives_the_cpu_value() {
        use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
        let pts = points(4096);
        for (name, node) in trees() {
            let lib = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).source;
            let src = format!(
                r"#version 450
layout(local_size_x = 256) in;
struct InputPoint {{ float x; float y; float z; float pad; }};
struct OutputDistance {{ float distance; float pad1; float pad2; float pad3; }};
layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ InputPoint input_points[]; }};
layout(std430, set = 0, binding = 1) buffer OutputDistances {{ OutputDistance output_distances[]; }};
layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};
{lib}
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) {{ return; }}
    InputPoint pt = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));
}}
"
            );
            let gpu = match GpuEvaluator::from_glsl_compute(&src).map(|c| c.into_parts().0) {
                Ok(g) => g,
                Err(e) => return no_gpu(&e),
            };
            let got = gpu.eval_batch(&pts).expect("gpu eval");
            let (w, at) = worst(&node, &got, &pts);
            eprintln!("{name}: GLSL max |gpu - cpu| = {w:.3e} over {}", pts.len());
            assert!(w <= 1e-5, "{name}: GLSL/CPU drift {w:.3e} at {at:?}");
        }
    }
}

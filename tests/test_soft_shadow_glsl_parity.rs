//! `raycast::SOFT_SHADOW_GLSL` against `raycast::soft_shadow` on the same rays.
//!
//! The GLSL is run on the GPU through naga's GLSL front end (the compute shader
//! of the scene, with its output line replaced by the shadow call), and every
//! ray is compared with the CPU law. The rays are chosen so that the set holds
//! fully lit rays (the law returns 1), rays that hit the occluder (0) and rays
//! in the penumbra (strictly between); each class must be non-empty, so the
//! comparison cannot pass by comparing nothing.
//!
//! The full render pipeline must call this function for its key light (the
//! earlier two-sample proxy is gone): `full_pipeline_uses_the_shared_soft_shadow`.

#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode, RenderConfig};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;
use alice_sdf::raycast::{soft_shadow, SOFT_SHADOW_GLSL};

const MINT: f32 = 0.02;
const MAXT: f32 = 20.0;
const K: f32 = 8.0;
/// The march accumulates `t += h`; a last-bit difference in `h` can move a
/// sample by a few ulp of `t`, which the penumbra term scales by `k / t`.
/// Measured worst on a Metal GPU: 5.4e-7 over 3721 rays.
const TOL: f32 = 1e-5;

fn light() -> Vec3 {
    Vec3::new(0.4, 1.0, 0.3).normalize()
}

/// A sphere over a box: the box casts no shadow on the plane below, the sphere does.
fn scene() -> SdfNode {
    SdfNode::sphere(1.0)
        .translate(0.0, 2.0, 0.0)
        .union(SdfNode::box3d(0.6, 0.3, 0.6).translate(3.0, 0.5, -2.0))
}

/// Ray origins on the plane y = 0, on a 61 × 61 grid over [-5, 5]².
fn origins() -> Vec<Vec3> {
    let mut v = Vec::new();
    for i in 0..61 {
        for j in 0..61 {
            let x = -5.0 + 10.0 * i as f32 / 60.0;
            let z = -5.0 + 10.0 * j as f32 / 60.0;
            v.push(Vec3::new(x, 0.0, z));
        }
    }
    v
}

/// The scene's compute shader with its output line replaced by the shadow call.
fn shadow_compute_shader(node: &SdfNode, glsl_fn: &str) -> String {
    let src = GlslShader::transpile(node, GlslTranspileMode::Hardcoded).to_compute_shader();
    let call = "output_distances[idx].distance = sdf_eval(p);";
    assert_eq!(src.matches(call).count(), 1, "compute shader output line");
    let l = light();
    let src = src.replace(
        call,
        &format!(
            "output_distances[idx].distance = sdf_soft_shadow(p, vec3({:?}, {:?}, {:?}), {MINT:?}, {MAXT:?}, {K:?});",
            l.x, l.y, l.z
        ),
    );
    let main = "void main() {";
    assert_eq!(src.matches(main).count(), 1, "compute shader main");
    let src = src.replace(
        main,
        &format!("float sdf_map_lite(vec3 p) {{ return sdf_eval(p); }}\n{glsl_fn}\n{main}"),
    );
    vulkanize(&src)
}

/// The OpenGL-only declarations of the compute shader in their Vulkan form
/// (the same rewrite `tests/test_glsl_export_oracle.rs` documents): the
/// version line, the loose `point_count` uniform as a block, and the
/// write-only qualifier naga does not accept.
fn vulkanize(src: &str) -> String {
    let mut out = String::new();
    for (i, line) in src.lines().enumerate() {
        let t = line.trim();
        if i == 0 && t.starts_with("#version") {
            out.push_str("#version 450\n");
        } else if t == "layout(location = 0) uniform uint point_count;" {
            out.push_str(
                "layout(std140, set = 0, binding = 2) uniform HostUniforms {\n    uint point_count;\n};\n",
            );
        } else if t.starts_with("layout(std430, binding = 0)") {
            out.push_str(&line.replace("binding = 0", "set = 0, binding = 0"));
            out.push('\n');
        } else if t.starts_with("layout(std430, binding = 1)") {
            out.push_str(
                &line
                    .replace("binding = 1", "set = 0, binding = 1")
                    .replace("writeonly buffer", "buffer"),
            );
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

fn validate(src: &str) {
    let module = naga::front::glsl::Frontend::default()
        .parse(
            &naga::front::glsl::Options::from(naga::ShaderStage::Compute),
            src,
        )
        .unwrap_or_else(|e| panic!("soft shadow compute: naga parse failed: {e:?}\n{src}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("soft shadow compute: naga validation failed: {e:?}"));
}

fn gpu_or_skip(src: &str) -> Option<GpuEvaluator> {
    match GpuEvaluator::from_glsl_compute(src).map(|c| c.into_parts().0) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("soft shadow parity: skipped ({e})");
            None
        }
    }
}

/// (lit, shadowed, penumbra, worst |gpu - cpu|)
fn compare(gpu: &[f32], cpu: &[f32]) -> (usize, usize, usize, f32) {
    assert_eq!(gpu.len(), cpu.len());
    let (mut lit, mut dark, mut pen, mut worst) = (0, 0, 0, 0.0f32);
    for (g, c) in gpu.iter().zip(cpu) {
        assert!(g.is_finite(), "GPU shadow is not finite: {g}");
        worst = worst.max((g - c).abs());
        if *c == 1.0 {
            lit += 1;
        } else if *c == 0.0 {
            dark += 1;
        } else {
            pen += 1;
        }
    }
    (lit, dark, pen, worst)
}

#[test]
fn soft_shadow_glsl_matches_the_cpu_law() {
    let node = scene();
    let pts = origins();
    let cpu: Vec<f32> = pts
        .iter()
        .map(|&p| soft_shadow(&node, p, light(), MINT, MAXT, K))
        .collect();
    let src = shadow_compute_shader(&node, SOFT_SHADOW_GLSL);
    validate(&src);
    let Some(gpu) = gpu_or_skip(&src) else {
        return;
    };
    let got = gpu.eval_batch(&pts).unwrap();
    let (lit, dark, pen, worst) = compare(&got, &cpu);
    eprintln!("soft shadow parity: lit {lit}, shadowed {dark}, penumbra {pen}, worst {worst:e}");
    assert!(lit >= 100, "too few lit rays compared: {lit}");
    assert!(dark >= 20, "too few shadowed rays compared: {dark}");
    assert!(pen >= 50, "too few penumbra rays compared: {pen}");
    assert!(
        worst <= TOL,
        "GPU and CPU soft shadow differ by {worst:e} (tolerance {TOL:e})"
    );
}

/// The comparison has teeth: the same shader with the penumbra term's `k`
/// dropped (a plausible slip) must fail the tolerance on the penumbra rays.
#[test]
fn soft_shadow_parity_catches_a_dropped_k() {
    let broken = SOFT_SHADOW_GLSL.replace("k * d / max(t - y, 0.0)", "d / max(t - y, 0.0)");
    assert_ne!(broken, SOFT_SHADOW_GLSL, "the mutation must apply");
    let node = scene();
    let pts = origins();
    let cpu: Vec<f32> = pts
        .iter()
        .map(|&p| soft_shadow(&node, p, light(), MINT, MAXT, K))
        .collect();
    let src = shadow_compute_shader(&node, &broken);
    validate(&src);
    let Some(gpu) = gpu_or_skip(&src) else {
        return;
    };
    let got = gpu.eval_batch(&pts).unwrap();
    let (_, _, pen, worst) = compare(&got, &cpu);
    assert!(pen >= 50, "too few penumbra rays compared: {pen}");
    assert!(worst > TOL, "a dropped k went unnoticed (worst {worst:e})");
}

/// Closed-form limits of the CPU law itself: a ray that starts inside the
/// sphere's shadow column and runs into it returns 0, a ray that leaves the
/// scene without coming near anything returns 1.
#[test]
fn soft_shadow_limits() {
    let node = scene();
    // Straight up from under the sphere's centre: hits it.
    assert_eq!(
        soft_shadow(&node, Vec3::new(0.0, 0.0, 0.0), Vec3::Y, MINT, MAXT, K),
        0.0
    );
    // Straight down from far above the box, away from both: nothing in the way
    // and the nearest surface stays far, so the penumbra term never drops below 1.
    assert_eq!(
        soft_shadow(&node, Vec3::new(-40.0, 0.0, 40.0), -Vec3::Y, MINT, MAXT, K),
        1.0
    );
}

#[test]
fn full_pipeline_uses_the_shared_soft_shadow() {
    let src = GlslShader::transpile(&scene(), GlslTranspileMode::Hardcoded)
        .to_fragment_shader_full(&RenderConfig::default());
    assert!(
        src.contains(SOFT_SHADOW_GLSL),
        "the full pipeline must include raycast::SOFT_SHADOW_GLSL verbatim"
    );
    assert!(
        src.contains("float keyShadow=sdf_soft_shadow("),
        "the key light must be shadowed by sdf_soft_shadow"
    );
    assert!(!src.contains("shadowProxy"), "the two-sample proxy is gone");
}

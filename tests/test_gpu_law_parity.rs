//! GPU ↔ CPU parity for the laws touched by the 1.10.3 / 1.11.0 parity work:
//! cell / sector snapping (`floor(x + 0.5)`, `p * (1 / s)` operands), the
//! `x < 0 ? -1 : 1` sign convention (pyramid, hex prism), scale applied after
//! non-linear blends, the stable exp-smooth form, and the taper law (the
//! WGSL / GLSL / HLSL transpilers emitted `1 + y * f` — a mirrored taper —
//! until 1.11.0).
//!
//! Runs the WGSL path through `GpuEvaluator` when an adapter is available and
//! skips otherwise (CI runners have no GPU; run locally after touching a law,
//! ALICE-SDF-LAWS §5 Port Parity Oracle Rule).
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

mod common;

use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;

/// Deterministic LCG points in a ±3 box, plus the cell / sector / base-plane
/// ties the laws are sensitive to.
fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x6e01_5e00_0000_0001;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 40) as f32) / ((1u64 << 24) as f32)).mul_add(6.0, -3.0)
    };
    let mut pts: Vec<Vec3> = (0..n).map(|_| Vec3::new(next(), next(), next())).collect();
    for i in -6..=6 {
        let v = i as f32 * 0.5;
        pts.push(Vec3::new(v, 0.0, 0.0));
        pts.push(Vec3::new(0.0, v, 0.0));
        pts.push(Vec3::new(v, v, -v));
    }
    pts
}

fn gpu_or_skip(node: &SdfNode) -> Option<GpuEvaluator> {
    match GpuEvaluator::new(node) {
        Ok(g) => Some(g),
        Err(e) => {
            // CI's gpu-parity job (software Vulkan / lavapipe) sets
            // ALICE_SDF_REQUIRE_GPU=1 so that "no adapter" is a failure, not a
            // silent skip — this test is the only oracle for the transpilers.
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
            );
            eprintln!("skipping GPU law parity: {e}");
            None
        }
    }
}

fn assert_gpu_matches_cpu(name: &str, node: &SdfNode, abs_tol: f32) {
    let Some(gpu) = gpu_or_skip(node) else {
        return;
    };
    let pts = points(2048);
    let got = gpu.eval_batch(&pts).expect("gpu eval");
    let mut worst = (0.0_f32, Vec3::ZERO, 0.0_f32, 0.0_f32);
    for (p, g) in pts.iter().zip(&got) {
        let c = eval(node, *p);
        // relative: the taper guard makes |d| ~ 1e6 on its singular plane
        let diff = (g - c).abs() / c.abs().max(1.0);
        if diff > worst.0 {
            worst = (diff, *p, c, *g);
        }
    }
    assert!(
        worst.0 <= abs_tol,
        "{name}: GPU/CPU drift {:.3e} at {:?} (cpu={} gpu={})",
        worst.0,
        worst.1,
        worst.2,
        worst.3
    );
    eprintln!(
        "{name}: max |gpu - cpu| = {:.3e} over {} points",
        worst.0,
        pts.len()
    );
}

fn offset_sphere() -> SdfNode {
    SdfNode::sphere(0.3).translate(0.6, 0.0, 0.0)
}

#[test]
fn taper_gpu_matches_cpu() {
    // pre-1.11.0 the shader mirrored the taper (1 + y f instead of 1 - y f)
    assert_gpu_matches_cpu("taper", &SdfNode::box3d(1.0, 1.0, 1.0).taper(0.5), 1e-4);
    assert_gpu_matches_cpu(
        "taper_negative",
        &SdfNode::box3d(1.0, 1.0, 1.0).taper(-0.5),
        1e-4,
    );
}

#[test]
fn repeat_laws_gpu_match_cpu_at_ties() {
    assert_gpu_matches_cpu(
        "repeat_infinite_offset",
        &offset_sphere().repeat_infinite(2.0, 2.0, 2.0),
        1e-4,
    );
    assert_gpu_matches_cpu(
        "repeat_finite_offset",
        &offset_sphere().repeat_finite([3, 2, 3], Vec3::splat(2.0)),
        1e-4,
    );
    // atan2 on the GPU is not libm, so an *exact* sector tie (count 4 on the
    // diagonals, odd counts at π) can resolve differently on the GPU — that is
    // the documented GPU transcendental domain, not a law drift. Count 6 has no
    // tie on the sample grid; away from ties the sector choice must agree.
    assert_gpu_matches_cpu(
        "polar_repeat_offset",
        &SdfNode::sphere(0.3)
            .translate(0.8, 0.0, 0.2)
            .polar_repeat(6),
        5e-3,
    );
}

/// 3.1.0: the shaders' `alice_atan2` returns the CPU law's exact constants on
/// the axes (`atan2(±0, x < 0) = ±π`, `atan2(y, ±0) = ±π/2`), so a sector
/// boundary that lies on an axis (odd counts at π, count 4 at ±π/2) snaps to
/// the same sector as the CPU. The diagonal ties of count 4 depend on the
/// GPU's `atan` last ulp and stay in the tolerance domain.
#[test]
fn polar_axis_ties_gpu_match_cpu() {
    let Some(_) = gpu_or_skip(&SdfNode::sphere(1.0)) else {
        return;
    };
    let base = SdfNode::sphere(0.3).translate(0.8, 0.0, 0.2);
    for count in [3u32, 5, 7] {
        let node = base.clone().polar_repeat(count);
        let gpu = GpuEvaluator::new(&node).expect("gpu");
        let mut pts = Vec::new();
        for i in 1..=12 {
            let v = i as f32 * 0.25;
            // on the -x axis (angle π), +x axis (0) and both z half-axes (±π/2)
            pts.push(Vec3::new(-v, 0.1, 0.0));
            pts.push(Vec3::new(-v, 0.1, -0.0));
            pts.push(Vec3::new(v, 0.1, 0.0));
            pts.push(Vec3::new(0.0, 0.1, v));
            pts.push(Vec3::new(0.0, 0.1, -v));
            pts.push(Vec3::new(-0.0, 0.1, -v));
        }
        let got = gpu.eval_batch(&pts).expect("gpu eval");
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            assert!(
                (g - c).abs() <= 1e-4 * c.abs().max(1.0),
                "polar_repeat({count}) @ {p:?}: gpu {g} vs cpu {c}"
            );
        }
    }
}

#[test]
fn sign_convention_gpu_matches_cpu() {
    assert_gpu_matches_cpu("pyramid", &SdfNode::pyramid(1.5), 1e-4);
    assert_gpu_matches_cpu("hex_prism", &SdfNode::hex_prism(1.0, 0.5), 1e-4);
}

#[test]
fn scale_after_blend_gpu_matches_cpu() {
    assert_gpu_matches_cpu(
        "scale_exp_smooth_union",
        &SdfNode::sphere(1.0)
            .exp_smooth_union(SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0), 0.8625)
            .scale(0.25),
        1e-4,
    );
    assert_gpu_matches_cpu(
        "scale_smooth_union_round",
        &offset_sphere()
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.4)
            .scale(0.5)
            .round(0.1)
            .scale(2.0),
        1e-4,
    );
}

#[test]
fn exp_smooth_far_gpu_matches_cpu() {
    // d ≫ k: the textbook form underflowed to log(0); the stable form is finite
    assert_gpu_matches_cpu(
        "exp_smooth_union_far",
        &SdfNode::sphere(0.05)
            .exp_smooth_union(SdfNode::box3d(1.5, 1.5, 0.05), 0.125)
            .scale(0.25),
        1e-4,
    );
}

/// The five GDF polyhedra had no shader helper at all until 1.10.3 (the
/// walker emitted `sdf_tetrahedron(...)` but no transpiler defined it), and
/// `ColumnsUnion` emitted a truncated declaration. Both were found by
/// `tests/test_transpiler_naga_validate.rs`; this pins the ported laws.
#[test]
fn polyhedra_and_columns_gpu_match_cpu() {
    assert_gpu_matches_cpu("tetrahedron", &SdfNode::tetrahedron(0.7), 1e-5);
    assert_gpu_matches_cpu("dodecahedron", &SdfNode::dodecahedron(0.7), 1e-5);
    assert_gpu_matches_cpu("icosahedron", &SdfNode::icosahedron(0.7), 1e-5);
    assert_gpu_matches_cpu(
        "truncated_octahedron",
        &SdfNode::truncated_octahedron(0.7),
        1e-5,
    );
    assert_gpu_matches_cpu(
        "truncated_icosahedron",
        &SdfNode::truncated_icosahedron(0.7),
        1e-5,
    );
    assert_gpu_matches_cpu(
        "columns_union",
        &SdfNode::sphere(0.8).columns_union(SdfNode::box3d(1.0, 0.5, 0.5), 0.3, 3.0),
        1e-4,
    );
}

/// Laws re-ported from Inigo Quilez's exact forms in 1.11.0 (the previous
/// versions jumped or were not distance fields; found by the Lipschitz
/// property test). Shader helpers are mirrored line by line.
#[test]
fn iq_exact_ports_gpu_match_cpu() {
    assert_gpu_matches_cpu("egg", &SdfNode::egg(1.0, 0.4), 1e-5);
    assert_gpu_matches_cpu(
        "horseshoe",
        &SdfNode::horseshoe(0.8, 0.9, 0.6, 0.2, 0.3),
        1e-5,
    );
    // pow(x, 1/3) / acos on the GPU are not libm: 1e-4 relative
    assert_gpu_matches_cpu("blobby_cross", &SdfNode::blobby_cross(1.2, 0.5), 1e-4);
    assert_gpu_matches_cpu(
        "sweep_bezier",
        &SdfNode::sphere(0.3).sweep_bezier(
            glam::Vec2::new(-1.5, -0.5),
            glam::Vec2::new(0.0, 1.5),
            glam::Vec2::new(1.5, -0.5),
        ),
        1e-4,
    );
    assert_gpu_matches_cpu("stairs", &SdfNode::stairs(0.4, 0.3, 5, 0.5), 1e-5);
    // sin / cos / atan2 on the GPU are not libm: 1e-4 relative
    assert_gpu_matches_cpu("helix", &SdfNode::helix(1.0, 0.2, 0.7, 1.5), 1e-4);
    assert_gpu_matches_cpu("ellipsoid", &SdfNode::ellipsoid(1.2, 0.6, 0.9), 1e-5);
    assert_gpu_matches_cpu("ellipsoid_flat", &SdfNode::ellipsoid(1.0, 1.0, 0.1), 1e-5);
    assert_gpu_matches_cpu(
        "ellipsoid_axis_aligned_queries",
        &SdfNode::ellipsoid(3.0, 1.0, 0.5).translate(0.5, 0.0, 0.0),
        1e-5,
    );
}

/// The GLSL transpiler's output on the GPU (wgpu compiles it through naga's
/// GLSL front end) against the CPU, over the whole corpus. Until 2.1.0 the
/// GLSL path was only parsed and validated (`test_transpiler_naga_validate`);
/// this is its execution oracle, the same standing as the WGSL one.
#[cfg(feature = "glsl")]
mod glsl_execution {
    use super::*;
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};

    pub fn compute_wrapper(library: &str) -> String {
        format!(
            r"#version 450
layout(local_size_x = 256) in;
struct InputPoint {{ float x; float y; float z; float pad; }};
struct OutputDistance {{ float distance; float pad1; float pad2; float pad3; }};
layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ InputPoint input_points[]; }};
layout(std430, set = 0, binding = 1) buffer OutputDistances {{ OutputDistance output_distances[]; }};
layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};

{library}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) {{ return; }}
    InputPoint pt = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));
}}
"
        )
    }

    #[test]
    fn every_corpus_node_matches_cpu_through_glsl() {
        use alice_sdf::compiled::shader_unsupported_nodes;
        // `corpus()` lives in tests/common; the WGSL tests above use hand-picked
        // laws, the GLSL oracle covers every node the transpilers accept.
        let corpus = crate::common::corpus::corpus();
        // Random points only: the appended ties (`atan2(0, -1) = π` on a sector
        // boundary) are where naga's GLSL `atan(y, x)` lowering and the CPU
        // disagree on the sign of π — a documented platform-dependent tie
        // (`repeat_laws_gpu_match_cpu_at_ties` pins the WGSL side).
        let pts: Vec<Vec3> = points(1024).into_iter().take(1024).collect();
        let mut failures = Vec::new();
        let mut checked = 0usize;
        for (name, node) in corpus {
            // Nodes the transpilers cannot express evaluate their child as-is
            // (`shader_unsupported_nodes` is the single list); they are not a
            // parity question until they are implemented.
            if !shader_unsupported_nodes(&node).is_empty() {
                continue;
            }
            let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
            let gpu = match GpuEvaluator::from_glsl_compute(&compute_wrapper(&shader.source)) {
                Ok(g) => g,
                Err(e) => {
                    assert!(
                        std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                        "{name}: ALICE_SDF_REQUIRE_GPU is set but the GLSL module failed: {e}"
                    );
                    eprintln!("skipping GLSL execution parity ({name}): {e}");
                    return;
                }
            };
            let got = gpu.eval_batch(&pts).expect("gpu eval");
            let mut worst = 0.0f32;
            for (p, g) in pts.iter().zip(&got) {
                let c = eval(&node, *p);
                let tol = 1e-4 * c.abs().max(1.0);
                let diff = (g - c).abs();
                if diff > tol && diff > worst {
                    worst = diff;
                }
            }
            checked += 1;
            if worst > 0.0 {
                failures.push(format!("{name}: GLSL/CPU drift {worst:.3e}"));
            }
        }
        eprintln!("GLSL execution parity: {checked} corpus nodes");
        assert!(checked > 100, "corpus too small: {checked}");
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

/// Every corpus node through the WGSL path. The hand-picked tests above
/// cover the laws touched by specific fixes; this one is the blanket oracle
/// (16 of 142 nodes drifted when it was first run in 2.1.0 — heart / pie /
/// vesica / box frame / three TPMS / columns / bend / extrude / displacement
/// / octant mirror were different laws in the shaders, and rounded cylinder
/// was wrong on the CPU).
#[test]
fn every_corpus_node_matches_cpu_through_wgsl() {
    use alice_sdf::compiled::shader_unsupported_nodes;
    // Random points only — exact ties (a sector boundary at atan2 = π, a
    // columns cell boundary) are platform-dependent on the GPU and are
    // pinned separately by `repeat_laws_gpu_match_cpu_at_ties`.
    let pts: Vec<Vec3> = points(1024).into_iter().take(1024).collect();
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in common::corpus::corpus() {
        if !shader_unsupported_nodes(&node).is_empty() {
            continue;
        }
        let Some(gpu) = gpu_or_skip(&node) else {
            return;
        };
        let got = gpu.eval_batch(&pts).expect("gpu eval");
        let mut worst = 0.0f32;
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            let tol = 1e-4 * c.abs().max(1.0);
            let diff = (g - c).abs();
            if diff > tol && diff > worst {
                worst = diff;
            }
        }
        checked += 1;
        if worst > 0.0 {
            failures.push(format!("{name}: WGSL/CPU drift {worst:.3e}"));
        }
    }
    eprintln!("WGSL execution parity: {checked} corpus nodes");
    assert!(checked > 100, "corpus too small: {checked}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[cfg(feature = "glsl")]
mod glsl_axis_ties {
    use super::*;
    use alice_sdf::compiled::{GlslShader, GlslTranspileMode};

    /// GLSL twin of `polar_axis_ties_gpu_match_cpu`: naga's GLSL `atan(y, x)`
    /// lowering disagreed with the CPU on the sign of π at `atan2(0, -x)`
    /// (why `every_corpus_node_matches_cpu_through_glsl` samples random
    /// points only); `alice_atan2` pins those ties.
    #[test]
    fn polar_axis_ties_glsl_match_cpu() {
        let base = SdfNode::sphere(0.3).translate(0.8, 0.0, 0.2);
        for count in [3u32, 5, 7] {
            let node = base.clone().polar_repeat(count);
            let shader = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
            let gpu = match GpuEvaluator::from_glsl_compute(
                &super::glsl_execution::compute_wrapper(&shader.source),
            ) {
                Ok(g) => g,
                Err(e) => {
                    assert!(
                        std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                        "ALICE_SDF_REQUIRE_GPU is set but the GLSL module failed: {e}"
                    );
                    eprintln!("skipping GLSL axis ties: {e}");
                    return;
                }
            };
            let mut pts = Vec::new();
            for i in 1..=12 {
                let v = i as f32 * 0.25;
                pts.push(Vec3::new(-v, 0.1, 0.0));
                pts.push(Vec3::new(-v, 0.1, -0.0));
                pts.push(Vec3::new(v, 0.1, 0.0));
                pts.push(Vec3::new(0.0, 0.1, v));
                pts.push(Vec3::new(0.0, 0.1, -v));
                pts.push(Vec3::new(-0.0, 0.1, -v));
            }
            let got = gpu.eval_batch(&pts).expect("gpu eval");
            for (p, g) in pts.iter().zip(&got) {
                let c = eval(&node, *p);
                assert!(
                    (g - c).abs() <= 1e-4 * c.abs().max(1.0),
                    "polar_repeat({count}) @ {p:?}: glsl {g} vs cpu {c}"
                );
            }
        }
    }
}

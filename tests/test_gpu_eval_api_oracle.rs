//! Oracles for the `GpuEvaluator` API beyond `eval_batch` — the buffer pool,
//! chunked dispatch, the workgroup size, Dynamic parameters with GPU normals,
//! the async constructors / evaluation, `eval_batch_submit`, the GLSL compute
//! route and the WGSL material function.
//!
//! Every expected value is the closed form of a translated sphere,
//! `d(p) = |p − c| − r` and `n(p) = (p − c) / |p − c|`, computed here in f64;
//! nothing below calls the crate's CPU evaluator to make an expected value.
//!
//! CI's gpu-parity job (lavapipe) sets `ALICE_SDF_REQUIRE_GPU=1`, so a missing
//! adapter fails instead of skipping (`scripts/preflight.sh` does the same on
//! Metal).
//!
//! # Tolerances
//!
//! * distance `DIST_TOL = 1e-5` absolute: `|p − c|` stays below 4 here, so one
//!   f32 ulp of the result is at most 4.8e-7; GPU `sqrt` / `length` are not
//!   required to be correctly rounded (WGSL allows a few ulp), and the shader
//!   subtracts in f32 where the reference works in f64. 1e-5 is ~20 ulp.
//! * normal `NORMAL_TOL = 2e-3` on each component: the shader's tetrahedral
//!   difference uses `e = 0.001`, so the four samples differ from each other
//!   by ~1.7e-3 while each carries an f32 rounding of up to ~2.4e-7 (one ulp at
//!   d ≈ 3) — a relative error of ~5e-4 in the difference, before
//!   normalisation. The truncation error of the scheme on a sphere is
//!   O(e² / |p − c|²) ≤ 4e-6 for `|p − c| ≥ 0.5`, which is negligible.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuError, GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;

const DIST_TOL: f64 = 1e-5;
const NORMAL_TOL: f64 = 2e-3;

/// `r` and `c` of the sphere `sphere(r).translate(c)`.
#[derive(Clone, Copy)]
struct Ball {
    r: f32,
    c: [f32; 3],
}

impl Ball {
    fn node(self) -> SdfNode {
        SdfNode::sphere(self.r).translate(self.c[0], self.c[1], self.c[2])
    }
    fn offset(self, p: Vec3) -> [f64; 3] {
        [
            f64::from(p.x) - f64::from(self.c[0]),
            f64::from(p.y) - f64::from(self.c[1]),
            f64::from(p.z) - f64::from(self.c[2]),
        ]
    }
    fn dist(self, p: Vec3) -> f64 {
        let q = self.offset(p);
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - f64::from(self.r)
    }
    fn normal(self, p: Vec3) -> [f64; 3] {
        let q = self.offset(p);
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        [q[0] / l, q[1] / l, q[2] / l]
    }
}

const A: Ball = Ball {
    r: 1.0,
    c: [0.25, -0.5, 0.125],
};
/// Same tree shape as `A` (one sphere under one translate), other values.
const B: Ball = Ball {
    r: 1.75,
    c: [-0.5, 0.25, 0.0],
};

/// Deterministic points with `0.5 ≤ |p − c| ≤ 3` for both `A` and `B`
/// (a 32-bit LCG; no crate code involved).
fn points(n: usize) -> Vec<Vec3> {
    let mut s: u32 = 0x1234_5678;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = Vec3::new(next() * 2.2, next() * 2.2, next() * 2.2);
        let ok = |b: Ball| {
            let q = b.offset(p);
            let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
            (0.5..=3.0).contains(&l)
        };
        if ok(A) && ok(B) {
            out.push(p);
        }
    }
    out
}

/// `None` (skip) without an adapter, unless `ALICE_SDF_REQUIRE_GPU` is set.
fn require<T>(what: &str, r: Result<T, GpuError>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("{what}: skipped ({e})");
            None
        }
    }
}

/// Compare GPU distances with the closed form; returns the number compared.
fn check_dist(what: &str, ball: Ball, pts: &[Vec3], got: &[f32]) -> usize {
    assert_eq!(got.len(), pts.len(), "{what}: result count");
    let mut worst = (0.0f64, 0usize);
    for (i, (&p, &g)) in pts.iter().zip(got).enumerate() {
        let err = (f64::from(g) - ball.dist(p)).abs();
        if err.is_nan() || err > worst.0 {
            worst = (err, i);
        }
    }
    assert!(
        worst.0 <= DIST_TOL,
        "{what}: |gpu − closed form| = {:e} at point {} {:?} (gpu {}, expected {})",
        worst.0,
        worst.1,
        pts[worst.1],
        got[worst.1],
        ball.dist(pts[worst.1])
    );
    assert!(!pts.is_empty(), "{what}: 0 points compared");
    pts.len()
}

#[test]
fn buffer_pool_grows_and_matches_the_closed_form() {
    let Some(gpu) = require("pool", GpuEvaluator::new(&A.node())) else {
        return;
    };
    let mut pool = gpu.create_buffer_pool(100);
    // The pool never holds fewer than 256 points.
    assert_eq!(pool.capacity, 256);

    let small = points(200);
    let d = gpu.eval_batch_pooled(&small, &mut pool).unwrap();
    check_dist("pooled 200", A, &small, &d);
    assert_eq!(pool.capacity, 256, "no growth below capacity");

    // Growth: max(1.5 × 256, 1000) = 1000.
    let big = points(1000);
    let d = gpu.eval_batch_pooled(&big, &mut pool).unwrap();
    check_dist("pooled 1000", A, &big, &d);
    assert_eq!(pool.capacity, 1000);

    // Reusing the grown pool for a smaller batch returns exactly that many
    // results, not the stale tail of the previous one.
    let d = gpu.eval_batch_pooled(&small, &mut pool).unwrap();
    check_dist("pooled 200 again", A, &small, &d);

    assert!(gpu.eval_batch_pooled(&[], &mut pool).unwrap().is_empty());
}

#[test]
fn eval_batch_auto_splits_large_batches_without_losing_points() {
    let Some(gpu) = require("auto", GpuEvaluator::new(&A.node())) else {
        return;
    };
    let mut pool = gpu.create_buffer_pool(256);
    // Above the 262,144-point single-dispatch threshold: 300,000 points are
    // evaluated as 262,144 + 37,856, and every one must come back in order.
    let pts = points(300_000);
    let d = gpu.eval_batch_auto(&pts, &mut pool).unwrap();
    let n = check_dist("auto 300k", A, &pts, &d);
    assert_eq!(n, 300_000);
    // Below the threshold it is one pooled dispatch.
    let few = points(777);
    let d = gpu.eval_batch_auto(&few, &mut pool).unwrap();
    check_dist("auto 777", A, &few, &d);
}

#[test]
fn every_entry_point_honours_the_shader_workgroup_size() {
    // `with_workgroup_size` rounds down to a power of two in 64..=1024.
    let shader = WgslShader::transpile(&A.node(), TranspileMode::Hardcoded);
    assert_eq!(shader.clone().with_workgroup_size(100).workgroup_size, 64);
    assert_eq!(shader.clone().with_workgroup_size(16).workgroup_size, 64);
    assert_eq!(shader.clone().with_workgroup_size(300).workgroup_size, 256);
    assert_eq!(
        shader.clone().with_workgroup_size(5000).workgroup_size,
        1024
    );
    let wg64 = shader.with_workgroup_size(64);
    assert!(wg64.to_compute_shader().contains("@workgroup_size(64)"));

    let Some(gpu) = require("workgroup 64", GpuEvaluator::from_shader(&wg64)) else {
        return;
    };
    // 1000 points need ceil(1000 / 64) = 16 workgroups of 64; a dispatch that
    // divides by 256 instead launches 4 × 64 = 256 threads and leaves the
    // remaining 744 outputs unwritten.
    let pts = points(1000);
    check_dist("eval_batch wg64", A, &pts, &gpu.eval_batch(&pts).unwrap());
    let mut pool = gpu.create_buffer_pool(1000);
    check_dist(
        "eval_batch_pooled wg64",
        A,
        &pts,
        &gpu.eval_batch_pooled(&pts, &mut pool).unwrap(),
    );
    check_dist(
        "eval_batch_async wg64",
        A,
        &pts,
        &pollster::block_on(gpu.eval_batch_async(&pts)).unwrap(),
    );
}

#[test]
fn dynamic_parameters_update_distances_and_gpu_normals() {
    // Dynamic mode reads the sphere radius and the translation from a uniform
    // buffer; `extract_params` must lay them out the way `transpile` does.
    let la = WgslShader::extract_params(&A.node());
    let lb = WgslShader::extract_params(&B.node());
    assert_eq!(
        la,
        WgslShader::transpile(&A.node(), TranspileMode::Dynamic).param_layout
    );
    assert_eq!(la.len(), lb.len(), "same tree shape, same layout length");
    // The four literals of the tree, each exactly once (order is the
    // transpiler's traversal order, values are bit-exact copies).
    for (layout, ball) in [(&la, A), (&lb, B)] {
        let mut got = layout.clone();
        let mut want = vec![ball.r, ball.c[0], ball.c[1], ball.c[2]];
        got.sort_by(f32::total_cmp);
        want.sort_by(f32::total_cmp);
        assert_eq!(got, want);
    }

    let Some(gpu) = require("dynamic", GpuEvaluator::new_dynamic(&A.node())) else {
        return;
    };
    let pts = points(1500);

    let check_full = |what: &str, ball: Ball| {
        let full = gpu.eval_batch_full(&pts).unwrap();
        assert_eq!(full.len(), pts.len(), "{what}: result count");
        let d: Vec<f32> = full.iter().map(|&(d, _)| d).collect();
        check_dist(what, ball, &pts, &d);
        for (&p, &(_, n)) in pts.iter().zip(&full) {
            let want = ball.normal(p);
            let err = (f64::from(n.x) - want[0])
                .abs()
                .max((f64::from(n.y) - want[1]).abs())
                .max((f64::from(n.z) - want[2]).abs());
            assert!(
                err <= NORMAL_TOL,
                "{what}: normal at {p:?} is {n:?}, expected {want:?} (err {err:e})"
            );
        }
    };

    check_full("full A", A);
    check_dist(
        "dynamic eval_batch A",
        A,
        &pts,
        &gpu.eval_batch(&pts).unwrap(),
    );

    // No recompilation: the same pipelines now see B's radius and centre.
    gpu.update_params(&B.node());
    check_full("full B", B);
    check_dist(
        "dynamic eval_batch B",
        B,
        &pts,
        &gpu.eval_batch(&pts).unwrap(),
    );
    let mut pool = gpu.create_buffer_pool(256);
    check_dist(
        "dynamic eval_batch_pooled B",
        B,
        &pts,
        &gpu.eval_batch_pooled(&pts, &mut pool).unwrap(),
    );
    check_dist(
        "dynamic eval_batch_async B",
        B,
        &pts,
        &pollster::block_on(gpu.eval_batch_async(&pts)).unwrap(),
    );

    // And back.
    gpu.update_params(&A.node());
    check_full("full A again", A);
}

#[test]
fn async_constructors_and_submitted_batches_match_the_closed_form() {
    let Some(gpu) = require(
        "new_async",
        pollster::block_on(GpuEvaluator::new_async(&A.node())),
    ) else {
        return;
    };
    let pts = points(1200);
    check_dist(
        "eval_batch_async",
        A,
        &pts,
        &pollster::block_on(gpu.eval_batch_async(&pts)).unwrap(),
    );
    check_dist(
        "submit + wait",
        A,
        &pts,
        &gpu.eval_batch_submit(pts.clone()).wait().unwrap(),
    );
    check_dist(
        "submit + resolve",
        A,
        &pts,
        &pollster::block_on(gpu.eval_batch_submit(pts.clone()).resolve()).unwrap(),
    );
    assert!(pollster::block_on(gpu.eval_batch_async(&[]))
        .unwrap()
        .is_empty());

    // `from_shader_async` keeps the shader's workgroup size (see
    // `every_entry_point_honours_the_shader_workgroup_size`).
    let wg64 = WgslShader::transpile(&B.node(), TranspileMode::Hardcoded).with_workgroup_size(64);
    let gpu = require(
        "from_shader_async",
        pollster::block_on(GpuEvaluator::from_shader_async(&wg64)),
    )
    .expect("an adapter was found a moment ago");
    check_dist(
        "from_shader_async wg64",
        B,
        &pts,
        &gpu.eval_batch(&pts).unwrap(),
    );

    // `from_wgsl_async` takes a complete compute shader.
    let src = WgslShader::transpile(&B.node(), TranspileMode::Hardcoded).to_compute_shader();
    let gpu = require(
        "from_wgsl_async",
        pollster::block_on(GpuEvaluator::from_wgsl_async(&src)),
    )
    .expect("an adapter was found a moment ago");
    check_dist(
        "from_wgsl_async",
        B,
        &pts,
        &pollster::block_on(gpu.eval_batch_async(&pts)).unwrap(),
    );
}

/// `from_glsl_compute`: a GLSL 450 compute shader around the GLSL
/// transpiler's `sdf_eval`, with the three bindings of the WGSL wrapper,
/// compiled by naga's GLSL front end and run on the GPU.
///
/// The wrapper is written here rather than taken from
/// `GlslShader::to_compute_shader`, whose `layout(location = 0) uniform uint
/// point_count;` is OpenGL-only GLSL (a non-opaque uniform outside a block is
/// not allowed under GL_KHR_vulkan_glsl, which naga and wgpu follow).
#[cfg(feature = "glsl")]
#[test]
fn glsl_compute_shader_runs_on_the_gpu() {
    use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
    let library = GlslShader::transpile(&A.node(), GlslTranspileMode::Hardcoded);
    let src = format!(
        r"#version 450
layout(local_size_x = 256) in;
struct InputPoint {{ float x; float y; float z; float pad; }};
struct OutputDistance {{ float distance; float pad1; float pad2; float pad3; }};
layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ InputPoint input_points[]; }};
layout(std430, set = 0, binding = 1) buffer OutputDistances {{ OutputDistance output_distances[]; }};
layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};
{}
void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) {{ return; }}
    InputPoint pt = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));
}}
",
        library.get_eval_function()
    );
    let Some(gpu) = require("from_glsl_compute", GpuEvaluator::from_glsl_compute(&src)) else {
        return;
    };
    let pts = points(1000);
    check_dist("glsl compute", A, &pts, &gpu.eval_batch(&pts).unwrap());
}

#[test]
fn material_function_returns_the_id_of_the_nearest_subtree() {
    let left = Ball {
        r: 0.5,
        c: [-1.0, 0.0, 0.0],
    };
    let right = Ball {
        r: 0.75,
        c: [1.0, 0.25, 0.0],
    };
    let node = left
        .node()
        .with_material(3)
        .union(right.node().with_material(7));
    // No `WithMaterial` anywhere: the function is not emitted at all.
    assert_eq!(
        WgslShader::transpile_material(&A.node(), TranspileMode::Hardcoded),
        ""
    );

    let main = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    let material = WgslShader::transpile_material(&node, TranspileMode::Hardcoded);
    assert!(material.contains("fn sdf_eval_material(p: vec3<f32>) -> f32"));
    let src = format!(
        r"struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
{}
{}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= point_count) {{ return; }}
    let q = input_points[idx];
    output_distances[idx].distance = sdf_eval_material(vec3<f32>(q.x, q.y, q.z));
}}
",
        main.source, material
    );
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("material WGSL does not parse: {}", e.emit_to_string(&src)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("material WGSL does not validate: {e:?}"));

    let Some(gpu) = require("material", GpuEvaluator::from_wgsl(&src)) else {
        return;
    };
    // Points that are clearly nearer to one ball (no ties within 1e-3).
    let pts: Vec<Vec3> = points(4000)
        .into_iter()
        .filter(|&p| (left.dist(p) - right.dist(p)).abs() > 1e-3)
        .collect();
    assert!(pts.len() > 3000, "{} points kept", pts.len());
    let ids = gpu.eval_batch(&pts).unwrap();
    let mut seen = [0usize; 2];
    for (&p, &id) in pts.iter().zip(&ids) {
        let want = if left.dist(p) < right.dist(p) {
            3.0
        } else {
            7.0
        };
        assert_eq!(id, want, "material at {p:?}");
        seen[usize::from(want == 7.0)] += 1;
    }
    assert!(seen[0] > 0 && seen[1] > 0, "both ids observed: {seen:?}");
}

//! `HexPrism`'s `hex_radius` is the apothem on every backend: the hexagon's flat sides lie at
//! `hex_radius` from the Z axis and its corners at `2·hex_radius/√3`.
//!
//! oracle: the closed-form distances of a regular hexagonal prism of apothem `a` and half height `h`
//! (flat side y = a, corner x = 2a/√3, cap z = h), evaluated on every CPU evaluator (tree, compiled
//! scalar / SIMD / BVH, and the scalar / SIMD JIT under the `jit` feature). The shader helpers (GLSL /
//! HLSL / WGSL) are checked as text only (`py - hex_r` in the helper) and are NOT run here: this file
//! executes the CPU backends, not every backend; a GPU-side oracle for this primitive is still needed
//! to cover the shaders' results. A backend that read `hex_radius` as the corner distance would put
//! the flat side at `a·√3/2` and read about `−0.13·a` at the oracle's flat-side point.

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_bvh, eval_compiled_simd, CompiledSdf, CompiledSdfBvh, Vec3x8,
};
use alice_sdf::eval::eval;
use alice_sdf::SdfNode;
use glam::Vec3;

/// A named evaluator of the field
type Backend = Box<dyn Fn(Vec3) -> f32>;

const A: f32 = 2.0;
const H: f32 = 1.5;

/// (point, closed-form distance): on and off the flat side, at the corner, on the cap
fn oracle_points() -> Vec<(Vec3, f32)> {
    let corner = 2.0 * A / 3.0_f32.sqrt();
    vec![
        (Vec3::new(0.0, A, 0.0), 0.0),
        (Vec3::new(0.0, A + 0.5, 0.0), 0.5),
        (Vec3::new(0.0, A - 0.25, 0.0), -0.25),
        (Vec3::new(corner, 0.0, 0.0), 0.0),
        (Vec3::new(corner + 0.5, 0.0, 0.0), 0.5),
        (Vec3::new(0.0, 0.0, H + 0.5), 0.5),
    ]
}

#[test]
fn every_cpu_backend_puts_the_flat_side_at_hex_radius() {
    let node = SdfNode::hex_prism(A, 2.0 * H); // the constructor takes the full height
    let compiled = CompiledSdf::compile(&node);
    let bvh = CompiledSdfBvh::try_compile(&node).expect("bvh");
    #[allow(unused_mut)]
    let mut backends: Vec<(&str, Backend)> = vec![
        ("tree", Box::new(|p| eval(&node, p))),
        ("compiled scalar", Box::new(|p| eval_compiled(&compiled, p))),
        (
            "compiled simd",
            Box::new(|p| eval_compiled_simd(&compiled, Vec3x8::splat(p)).to_array()[0]),
        ),
        ("compiled bvh", Box::new(|p| eval_compiled_bvh(&bvh, p))),
    ];
    #[cfg(feature = "jit")]
    {
        let jit = alice_sdf::compiled::jit::JitCompiledSdf::compile(&node).expect("jit");
        backends.push(("jit scalar", Box::new(move |p| jit.eval(p))));
        let jit_simd = alice_sdf::compiled::jit::JitSimdSdf::compile(&compiled).expect("jit simd");
        backends.push((
            "jit simd",
            Box::new(move |p| jit_simd.eval_batch(&[p.x; 8], &[p.y; 8], &[p.z; 8])[0]),
        ));
    }
    let mut compared = 0;
    let mut wrong = Vec::new();
    for (name, f) in &backends {
        for (p, want) in oracle_points() {
            compared += 1;
            let got = f(p);
            if (got - want).abs() > 1e-4 {
                wrong.push(format!("{name} at {p}: {got} (closed form {want})"));
            }
        }
    }
    assert!(
        compared >= 4 * oracle_points().len(),
        "only {compared} comparisons"
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
#[test]
fn every_shader_helper_compares_the_flat_side_with_hex_radius() {
    let node = SdfNode::hex_prism(A, 2.0 * H);
    #[allow(unused_mut)]
    let mut sources: Vec<(&str, String)> = Vec::new();
    #[cfg(feature = "glsl")]
    sources.push((
        "glsl",
        alice_sdf::compiled::GlslShader::transpile(
            &node,
            alice_sdf::compiled::GlslTranspileMode::Hardcoded,
        )
        .source,
    ));
    #[cfg(feature = "hlsl")]
    sources.push((
        "hlsl",
        alice_sdf::compiled::HlslShader::transpile(
            &node,
            alice_sdf::compiled::HlslTranspileMode::Hardcoded,
        )
        .source,
    ));
    #[cfg(feature = "gpu")]
    sources.push((
        "wgsl",
        alice_sdf::compiled::WgslShader::transpile(
            &node,
            alice_sdf::compiled::TranspileMode::Hardcoded,
        )
        .source,
    ));
    assert!(!sources.is_empty(), "no shader backend compiled in");
    for (name, src) in &sources {
        assert!(src.contains("sdf_hex_prism"), "{name}: no hex prism helper");
        assert!(
            src.contains("py - hex_r"),
            "{name}: the flat side is not compared with hex_r"
        );
    }
}

#[test]
fn the_bounds_reach_the_corners() {
    let b = alice_sdf::tight_aabb::compute_tight_aabb(&SdfNode::hex_prism(A, 2.0 * H));
    let corner = 2.0 * A / 3.0_f32.sqrt();
    assert!(
        (b.max.x - corner).abs() < 1e-2 && (b.max.y - A).abs() < 1e-2,
        "{b:?}"
    );
}

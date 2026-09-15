//! Rounding at cell boundaries must agree on every evaluation path.
//!
//! The repeat / polar / helix laws snap to a cell with a rounding step. Before
//! 1.10.3 each path used its platform's `round` (ties away from zero on the
//! scalar path, ties to even on AVX / NEON / Cranelift / WGSL), so a point on
//! a cell boundary — every marching-cubes grid whose step divides the spacing
//! has them — landed in a different cell per path and the distance differed by
//! a whole cell (1.2 for the case below, not an ulp). The canonical rule is now
//! `floor(x + 0.5)` (`crate::crispy::round_half_up`) everywhere; this file pins
//! it for the tree, compiled scalar, SIMD, JIT (feature) and the generated
//! shader text.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, eval_compiled_batch_simd, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use glam::Vec3;

const TOL: f32 = 1e-5;

/// Child offset inside the cell so that picking the wrong cell changes the
/// distance (a centred sphere would hide the bug).
fn offset_sphere() -> SdfNode {
    SdfNode::sphere(0.3).translate(0.6, 0.0, 0.0)
}

fn cases() -> Vec<(&'static str, SdfNode)> {
    vec![
        (
            "repeat_infinite",
            offset_sphere().repeat_infinite(2.0, 2.0, 2.0),
        ),
        (
            "repeat_finite",
            offset_sphere().repeat_finite([4, 2, 4], Vec3::splat(2.0)),
        ),
        (
            "polar_repeat",
            SdfNode::sphere(0.3)
                .translate(0.8, 0.0, 0.2)
                .polar_repeat(4),
        ),
        ("helix", SdfNode::helix(0.6, 0.1, 0.5, 0.8)),
    ]
}

/// Spacing 2 → x = odd integers are exact `p / s = k + 0.5` ties. Polar repeat
/// with 4 sectors has its ties on the diagonals; helix with pitch 0.5 on
/// `y = pitch * (k + 0.5)` at theta = 0.
fn tie_points() -> Vec<Vec3> {
    let mut pts: Vec<Vec3> = [-3.0f32, -1.0, 1.0, 3.0, 5.0]
        .iter()
        .flat_map(|&x| [Vec3::new(x, 0.0, 0.0), Vec3::new(x, 1.0, -1.0)])
        .collect();
    pts.push(Vec3::new(1.0, 0.0, 1.0));
    pts.push(Vec3::new(-1.0, 0.0, 1.0));
    pts.push(Vec3::new(0.7, 0.25, 0.0));
    pts.push(Vec3::new(0.7, 0.75, 0.0));
    // pad to a multiple of 8 so the SIMD batch path uses the 8-lane kernel
    while pts.len() % 8 != 0 {
        pts.push(Vec3::new(0.25, 0.0, 0.0));
    }
    pts
}

#[test]
fn tree_compiled_simd_agree_at_cell_boundaries() {
    for (name, node) in cases() {
        let compiled = CompiledSdf::compile(&node);
        let pts = tie_points();
        let simd = eval_compiled_batch_simd(&compiled, &pts);
        for (i, &p) in pts.iter().enumerate() {
            let tree = eval(&node, p);
            let scalar = eval_compiled(&compiled, p);
            assert!(
                (tree - scalar).abs() <= TOL,
                "{name} @ {p:?}: tree {tree} vs compiled {scalar}"
            );
            assert!(
                (tree - simd[i]).abs() <= TOL,
                "{name} @ {p:?}: tree {tree} vs simd {}",
                simd[i]
            );
        }
    }
}

#[test]
fn tie_rule_is_floor_plus_half() {
    // x = 1 with spacing 2: cell = floor(0.5 + 0.5) = 1 → local x = -1,
    // distance to the sphere at +0.6 is |-1 - 0.6| - 0.3 = 1.3 (the
    // ties-to-even rule would give cell 0 → 0.1).
    let node = offset_sphere().repeat_infinite(2.0, 2.0, 2.0);
    let compiled = CompiledSdf::compile(&node);
    for &(x, want) in &[(1.0f32, 1.3f32), (-1.0, 1.3), (3.0, 1.3), (0.25, 0.05)] {
        let p = Vec3::new(x, 0.0, 0.0);
        assert!((eval(&node, p) - want).abs() <= TOL, "tree @ x={x}");
        assert!(
            (eval_compiled(&compiled, p) - want).abs() <= TOL,
            "compiled @ x={x}"
        );
    }
}

#[cfg(feature = "jit")]
#[test]
fn jit_agrees_at_cell_boundaries() {
    use alice_sdf::compiled::jit::{JitCompiledSdf, JitSimdSdf};
    for (name, node) in cases() {
        // polar repeat / helix are not JIT-supported; `compile` reports that.
        let Ok(jit) = JitCompiledSdf::compile(&node) else {
            continue;
        };
        let compiled = CompiledSdf::compile(&node);
        let pts = tie_points();
        let got = jit.eval_batch(&pts);
        for (i, &p) in pts.iter().enumerate() {
            let tree = eval(&node, p);
            assert!(
                (tree - got[i]).abs() <= TOL,
                "{name} @ {p:?}: tree {tree} vs jit {}",
                got[i]
            );
        }
        if let Ok(jit_simd) = JitSimdSdf::compile(&compiled) {
            let xs: Vec<f32> = pts.iter().map(|p| p.x).collect();
            let ys: Vec<f32> = pts.iter().map(|p| p.y).collect();
            let zs: Vec<f32> = pts.iter().map(|p| p.z).collect();
            let got = jit_simd.eval_batch(&xs, &ys, &zs);
            for (i, &p) in pts.iter().enumerate() {
                let tree = eval(&node, p);
                assert!(
                    (tree - got[i]).abs() <= TOL,
                    "{name} @ {p:?}: tree {tree} vs jit simd {}",
                    got[i]
                );
            }
        }
    }
}

/// The shader transpilers must not emit `round(` for the cell snap — GLSL's
/// tie direction is implementation-defined and WGSL / HLSL disagree with each
/// other — but `floor(... + 0.5)`.
#[cfg(feature = "glsl")]
#[test]
fn glsl_text_uses_floor_plus_half() {
    use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
    for (name, node) in cases() {
        let src = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).source;
        assert!(
            !src.contains("round("),
            "{name} glsl: cell snap still uses round()\n{src}"
        );
        assert!(
            src.contains("+ 0.5)"),
            "{name} glsl: expected floor(x + 0.5) snap\n{src}"
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn wgsl_text_uses_floor_plus_half_and_parses() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    for (name, node) in cases() {
        let wgsl = WgslShader::transpile(&node, TranspileMode::Hardcoded).source;
        assert!(
            !wgsl.contains("round("),
            "{name} wgsl: cell snap still uses round()\n{wgsl}"
        );
        assert!(
            wgsl.contains("+ 0.5)"),
            "{name} wgsl: expected floor(x + 0.5) snap\n{wgsl}"
        );
        // must still parse: vector + scalar `+ 0.5` is valid WGSL
        let parsed = naga::front::wgsl::parse_str(&wgsl);
        assert!(
            parsed.is_ok(),
            "{name}: WGSL parse failed:\n{}",
            parsed
                .err()
                .map(|e| e.emit_to_string(&wgsl))
                .unwrap_or_default()
        );
    }
}

#[cfg(feature = "hlsl")]
#[test]
fn hlsl_text_uses_floor_plus_half() {
    use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
    for (name, node) in cases() {
        let src = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source;
        assert!(
            !src.contains("round("),
            "{name} hlsl: cell snap still uses round()\n{src}"
        );
    }
}

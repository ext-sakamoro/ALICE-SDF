//! Integration tests: evaluator parity across every compiled opcode
//!
//! The tree evaluator (`eval`) is the oracle. For every compilable `SdfNode`
//! variant we compare:
//!
//! - `eval_compiled`      — scalar bytecode (shared exhaustive stack machine)
//! - `eval_compiled_simd` — 8-lane SIMD, lane 0 of a splatted point
//! - `eval_compiled_bvh`  — BVH bytecode (same stack machine; since 1.9.2 the
//!   BVH accepts every tree the main compiler does)
//! - `JitCompiledSdf` / `JitSimdSdf` (feature `jit`) — Cranelift scalar JIT and
//!   the SIMD JIT, **or** a loud `Err` from `compile` — never a silent MAX
//!   distance
//!
//! A final test asserts that the corpus below reaches every `OpCode` the
//! compiler can emit, so a new opcode that is not covered here fails CI.
//!
//! Background (1.9.1): before this test the scalar and BVH evaluators were
//! hand-copied with a `_ =>` fallback that turned unknown primitives into a
//! unit sphere and unknown modifiers into a no-op.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::compiled::{CompileError, OpCode};
use alice_sdf::prelude::*;
use common::corpus::{corpus, identity_mat, sphere, square_verts, unit_box};
use glam::Vec2;
use std::collections::BTreeSet;

const TOL: f32 = 1e-4;

fn simd_lane0(sdf: &CompiledSdf, p: Vec3) -> f32 {
    eval_compiled_simd(sdf, Vec3x8::splat(p)).to_array()[0]
}

fn sample_points() -> Vec<Vec3> {
    vec![
        Vec3::ZERO,
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(-2.0, 0.4, -0.9),
        // Cell-boundary ties for the repeat laws (spacing 2 → p / s = ±0.5, 1.5):
        // every evaluator must round the same way (`floor(x + 0.5)`).
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(3.0, -1.0, 1.0),
        // pyramid(1.5) base centre (-0.0 sign argument), and a far point where
        // a leaf-scaled blend width is visibly wrong
        Vec3::new(0.0, -1.5, 0.0),
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(-3.0, -3.0, 3.0),
        Vec3::new(0.0, -2.25, 0.0),
    ]
}

/// Compare tree vs scalar vs SIMD; BVH is compared when it compiles and must
/// otherwise reject loudly. Returns (mismatch lines, opcodes the bytecode emitted).
fn check_parity(name: &str, node: &SdfNode) -> (Vec<String>, BTreeSet<String>) {
    let compiled = CompiledSdf::try_compile(node)
        .unwrap_or_else(|e| panic!("{name}: CompiledSdf::try_compile failed: {e}"));
    let bvh = CompiledSdfBvh::try_compile(node)
        .unwrap_or_else(|e| panic!("{name}: CompiledSdfBvh::try_compile failed: {e}"));
    #[cfg(feature = "jit")]
    let jit_scalar = alice_sdf::compiled::jit::JitCompiledSdf::compile(node).ok();
    #[cfg(feature = "jit")]
    let jit_simd = alice_sdf::compiled::jit::JitSimdSdf::compile(&compiled).ok();
    let mut failures = Vec::new();
    for p in sample_points() {
        let tree = eval(node, p);
        // Comparing non-finite or astronomically large values is meaningless;
        // the degenerate-polygon sentinel (1e10) is covered separately.
        if !tree.is_finite() || tree.abs() > 1e6 {
            continue;
        }
        let scalar = eval_compiled(&compiled, p);
        let simd = simd_lane0(&compiled, p);
        #[allow(unused_mut)]
        let mut paths = vec![
            ("scalar", scalar),
            ("simd", simd),
            ("bvh", eval_compiled_bvh(&bvh, p)),
        ];
        #[cfg(feature = "jit")]
        {
            if let Some(j) = &jit_scalar {
                paths.push(("jit", j.eval(p)));
            }
            if let Some(j) = &jit_simd {
                let out = j.eval_batch(&[p.x; 8], &[p.y; 8], &[p.z; 8]);
                paths.push(("jit_simd", out[0]));
            }
        }
        for (path, v) in paths {
            if (v - tree).abs() > TOL * tree.abs().max(1.0) {
                failures.push(format!("{name} @ {p:?}: tree={tree:.6} {path}={v:.6}"));
            }
        }
    }
    let ops = compiled
        .instructions
        .iter()
        .map(|i| format!("{:?}", i.opcode))
        .collect();
    (failures, ops)
}

#[test]
fn every_compilable_node_matches_tree_eval() {
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        failures.extend(check_parity(name, &node).0);
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The corpus must reach every opcode the compiler can emit, so that adding
/// an opcode without adding a parity case here fails CI.
#[test]
fn corpus_covers_every_emitted_opcode() {
    let mut seen = BTreeSet::new();
    for (name, node) in corpus() {
        seen.extend(check_parity(name, &node).1);
    }
    // 125 OpCode variants minus `Animated` (compiler inlines the child) = 124.
    // `End` and `PopTransform` are emitted and counted.
    let expected = 124;
    assert!(
        !seen.contains("Animated"),
        "compiler started emitting Animated; update the expected count"
    );
    assert_eq!(
        seen.len(),
        expected,
        "corpus reaches {} opcodes, expected {}: {:?}",
        seen.len(),
        expected,
        seen
    );
    // Spot-check the historically missing ones are actually reached.
    for op in [
        "Circle2D",
        "Rect2D",
        "RoundedRect2D",
        "Segment2D",
        "Polygon2D",
        "Annular2D",
        "ExpSmoothUnion",
        "ExpSmoothIntersection",
        "ExpSmoothSubtraction",
        "Shear",
    ] {
        assert!(seen.contains(op), "{op} not reached by corpus");
    }
    let _ = OpCode::Sphere; // keep the import meaningful for readers
}

#[test]
fn unsupported_nodes_are_rejected_loudly_not_silently() {
    // Main compiler: Terrain used to compile to a silent sphere.
    let terrain = SdfNode::Terrain {
        scale: 1.0,
        amplitude: 0.5,
    };
    assert!(matches!(
        CompiledSdf::try_compile(&terrain),
        Err(CompileError::UnsupportedPrimitive(_))
    ));

    // Both compilers reject the same set (no bytecode law): Terrain, Triangle, Bezier.
    let rejected: Vec<(&str, SdfNode)> = vec![
        ("terrain", terrain),
        ("triangle", SdfNode::triangle(Vec3::ZERO, Vec3::X, Vec3::Y)),
        ("bezier", SdfNode::bezier(Vec3::ZERO, Vec3::X, Vec3::Y, 0.1)),
    ];
    for (name, node) in rejected {
        assert!(
            matches!(
                CompiledSdf::try_compile(&node),
                Err(CompileError::UnsupportedPrimitive(_))
            ),
            "{name}: CompiledSdf must reject with UnsupportedPrimitive"
        );
        assert!(
            matches!(
                CompiledSdfBvh::try_compile(&node),
                Err(CompileError::UnsupportedPrimitive(_))
            ),
            "{name}: CompiledSdfBvh must reject with UnsupportedPrimitive"
        );
    }
    // Since 1.9.2 the BVH accepts what used to be silently replaced by sphere(0.001).
    for (name, node) in [
        ("ifs", sphere().ifs(vec![identity_mat()], 2)),
        (
            "projective",
            unit_box().projective_transform(identity_mat(), 1.0),
        ),
        ("icosahedral", unit_box().icosahedral_symmetry()),
        ("circle_2d", SdfNode::circle_2d(0.25, 0.5)),
    ] {
        assert!(
            CompiledSdfBvh::try_compile(&node).is_ok(),
            "{name}: BVH must compile"
        );
    }
}

/// Every primitive's AABB must contain every point where the SDF is ≤ 0.
/// Sampled on a grid so a too-tight conservative bound in `refit::primitive_aabb`
/// fails here instead of silently culling geometry downstream.
#[test]
fn primitive_and_scene_aabbs_are_conservative() {
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        let bvh = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&bvh);
        if !aabb.is_valid() {
            failures.push(format!("{name}: scene AABB is empty/invalid"));
            continue;
        }
        if !aabb.min().is_finite() || !aabb.max().is_finite() {
            continue; // unbounded shape: infinite AABB is trivially conservative
        }
        let steps = 25;
        let extent = 3.0f32;
        for ix in 0..steps {
            for iy in 0..steps {
                for iz in 0..steps {
                    let f = |i: usize| -extent + 2.0 * extent * (i as f32) / ((steps - 1) as f32);
                    let p = Vec3::new(f(ix), f(iy), f(iz));
                    let d = eval(&node, p);
                    if d <= 0.0 && aabb.distance_to_point(p) > 1e-4 {
                        failures.push(format!(
                            "{name}: inside point {p:?} (d={d:.4}) outside AABB [{:?}, {:?}]",
                            aabb.min(),
                            aabb.max()
                        ));
                        break;
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} AABB violations:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn polygon_2d_vertices_survive_compilation() {
    // A triangle and a square must evaluate differently at the same point
    // (previously the compiler dropped the vertices and both were a unit sphere).
    let tri = SdfNode::polygon_2d(
        vec![
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(0.0, 0.5),
        ],
        0.5,
    );
    let sq = SdfNode::polygon_2d(square_verts(), 0.5);
    let p = Vec3::new(0.4, 0.4, 0.0);
    let d_tri = eval_compiled(&CompiledSdf::compile(&tri), p);
    let d_sq = eval_compiled(&CompiledSdf::compile(&sq), p);
    assert!((d_tri - eval(&tri, p)).abs() < TOL);
    assert!((d_sq - eval(&sq, p)).abs() < TOL);
    assert!((d_tri - d_sq).abs() > 0.05, "tri={d_tri} sq={d_sq}");
}

// ============================================================================
// Interval / analytic-gradient representations vs point evaluation
// ============================================================================
//
// `interval::eval_interval` and `eval::gradient::eval_gradient` are two more
// hand-written representations of the same 128-variant law (interval
// arithmetic and derivatives). They are pinned to `eval` here with the same
// corpus: an interval that fails to contain a sampled value, or an analytic
// gradient that disagrees with the central difference, is a law drift.

/// Deterministic LCG in [0, 1).
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[test]
fn interval_eval_contains_point_values() {
    use alice_sdf::interval::{eval_interval, Vec3Interval};
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        let mut rnd = lcg(0x1a7e_0001 ^ name.len() as u64);
        for _ in 0..24 {
            // Random box: centre in ±2.5, half-size 0.05..0.8
            let c = Vec3::new(
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
            );
            let h = Vec3::new(
                rnd().mul_add(0.75, 0.05),
                rnd().mul_add(0.75, 0.05),
                rnd().mul_add(0.75, 0.05),
            );
            let bounds = Vec3Interval::from_bounds(c - h, c + h);
            let iv = eval_interval(&node, bounds);
            if !(iv.lo.is_finite() || iv.lo == f32::NEG_INFINITY)
                || !(iv.hi.is_finite() || iv.hi == f32::INFINITY)
            {
                failures.push(format!(
                    "{name}: non-finite interval {iv:?} for box {c:?}±{h:?}"
                ));
                continue;
            }
            // 8 corners + 8 interior samples
            let mut pts: Vec<Vec3> = (0..8)
                .map(|k| {
                    Vec3::new(
                        if k & 1 == 0 { c.x - h.x } else { c.x + h.x },
                        if k & 2 == 0 { c.y - h.y } else { c.y + h.y },
                        if k & 4 == 0 { c.z - h.z } else { c.z + h.z },
                    )
                })
                .collect();
            for _ in 0..8 {
                pts.push(Vec3::new(
                    (c.x - h.x) + rnd() * 2.0 * h.x,
                    (c.y - h.y) + rnd() * 2.0 * h.y,
                    (c.z - h.z) + rnd() * 2.0 * h.z,
                ));
            }
            for p in pts {
                let d = eval(&node, p);
                if !d.is_finite() {
                    continue;
                }
                let tol = 1e-4 * d.abs().max(1.0);
                if d < iv.lo - tol || d > iv.hi + tol {
                    failures.push(format!(
                        "{name}: eval={d} outside interval [{}, {}] at p={p:?} (box {c:?}±{h:?})",
                        iv.lo, iv.hi
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} interval containment violations:\n{}",
        failures.len(),
        failures
            .iter()
            .take(5000)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn analytic_gradient_matches_numerical() {
    use alice_sdf::eval::gradient::eval_gradient;
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in corpus() {
        let mut rnd = lcg(0x9a4d_0001 ^ name.len() as u64);
        for _ in 0..40 {
            let p = Vec3::new(
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
            );
            // Two-step central difference: keep only points where the field is
            // locally smooth (both step sizes agree), which excludes CSG creases,
            // repetition cell borders and other non-differentiable loci.
            let cd = |e: f32| {
                Vec3::new(
                    eval(&node, p + Vec3::X * e) - eval(&node, p - Vec3::X * e),
                    eval(&node, p + Vec3::Y * e) - eval(&node, p - Vec3::Y * e),
                    eval(&node, p + Vec3::Z * e) - eval(&node, p - Vec3::Z * e),
                ) / (2.0 * e)
            };
            let g1 = cd(1e-3);
            let g2 = cd(4e-3);
            if !g1.is_finite()
                || !g2.is_finite()
                || (g1 - g2).length() > 2e-2 * g1.length().max(1.0)
            {
                continue;
            }
            let ga = eval_gradient(&node, p);
            if !ga.is_finite() {
                failures.push(format!("{name}: non-finite analytic gradient at {p:?}"));
                continue;
            }
            checked += 1;
            let err = (ga - g1).length();
            if err > 2e-2 * g1.length().max(1.0) {
                failures.push(format!(
                    "{name}: analytic {ga:?} vs numerical {g1:?} (err {err:.3e}) at p={p:?}"
                ));
            }
        }
    }
    assert!(
        checked > 1000,
        "too few smooth sample points checked: {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} gradient mismatches:\n{}",
        failures.len(),
        failures
            .iter()
            .take(5000)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// `eval_lipschitz(node)` must bound every difference quotient of `eval` on
/// the exterior: `|f(p + h·u) − f(p − h·u)| / 2h ≤ L` for every unit direction
/// `u` whenever at least one of the two samples is outside (`f ≥ 0`),
/// differentiable or not. Sphere tracing steps by `d / L`, so an
/// under-claimed L (external review 2026-09-15: TPMS 9 variants and Ellipsoid
/// claimed 1.0, measured 1.7–7.0×) makes rays skip the surface. Variants
/// that declare `INFINITY` are skipped (no claim to check).
#[test]
fn lipschitz_claim_bounds_every_difference_quotient() {
    use alice_sdf::interval::eval_lipschitz;
    let dirs: Vec<Vec3> = {
        let mut v = vec![Vec3::X, Vec3::Y, Vec3::Z];
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                v.push(Vec3::new(sx, sy, 1.0).normalize());
            }
        }
        let mut rnd = lcg(0x1f2e_3d4c);
        for _ in 0..6 {
            v.push(
                Vec3::new(
                    rnd().mul_add(2.0, -1.0),
                    rnd().mul_add(2.0, -1.0),
                    rnd().mul_add(2.0, -1.0),
                )
                .normalize(),
            );
        }
        v
    };
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        let claimed = eval_lipschitz(&node);
        if !claimed.is_finite() {
            continue; // unbounded by declaration (e.g. twist of an unbounded child)
        }
        let mut worst = 0.0f32;
        let mut worst_at = Vec3::ZERO;
        let mut rnd = lcg(0x5a5a_0001 ^ name.len() as u64);
        let steps = 13;
        let mut points: Vec<Vec3> = Vec::with_capacity(steps * steps * steps + 400);
        for ix in 0..steps {
            for iy in 0..steps {
                for iz in 0..steps {
                    let f = |i: usize| -3.0 + 6.0 * (i as f32) / ((steps - 1) as f32);
                    points.push(Vec3::new(f(ix), f(iy), f(iz)));
                }
            }
        }
        for _ in 0..400 {
            points.push(Vec3::new(
                rnd().mul_add(6.0, -3.0),
                rnd().mul_add(6.0, -3.0),
                rnd().mul_add(6.0, -3.0),
            ));
        }
        for p in points {
            for &u in &dirs {
                for h in [1e-3f32, 1e-2] {
                    let (fa, fb) = (eval(&node, p + u * h), eval(&node, p - u * h));
                    if fa < 0.0 && fb < 0.0 {
                        continue; // strictly interior pair: outside the bound's domain
                    }
                    let q = (fa - fb).abs() / (2.0 * h);
                    if q.is_finite() && q > worst {
                        worst = q;
                        worst_at = p;
                    }
                }
            }
        }
        // 0.5 % slack for finite-difference rounding
        if worst > claimed * 1.005 + 1e-4 {
            failures.push(format!(
                "{name}: eval_lipschitz claims {claimed:.3}, measured difference quotient {worst:.3} ({:.2}×) near {worst_at:?}",
                worst / claimed
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} unsound Lipschitz claims:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The bound must stay informative: exact primitives claim 1, the TPMS and
/// deformation bounds are finite, and only the documented non-Lipschitz laws
/// declare `INFINITY` (a blanket `INFINITY` would also pass the soundness
/// test above).
#[test]
fn lipschitz_claims_are_finite_where_the_law_is_lipschitz() {
    use alice_sdf::interval::eval_lipschitz;
    let mut infinite = Vec::new();
    let mut finite = 0usize;
    for (name, node) in corpus() {
        if eval_lipschitz(&node).is_finite() {
            finite += 1;
        } else {
            infinite.push(name);
        }
    }
    assert!(
        finite >= 100,
        "only {finite} finite claims; infinite: {infinite:?}"
    );
    assert!(
        infinite.len() <= 24,
        "{} infinite claims (expected ≤ 24): {infinite:?}",
        infinite.len()
    );
    assert_eq!(eval_lipschitz(&SdfNode::sphere(1.0)), 1.0);
    assert!((eval_lipschitz(&SdfNode::gyroid(2.0, 0.1)) - 3f32.sqrt()).abs() < 1e-3);
    assert!((eval_lipschitz(&SdfNode::neovius(0.5, 0.1)) - 7.05).abs() < 1e-3);
    // Twist of a unit box (XZ corner radius √2): v = 2·√2, σ = v/2 + √(1 + v²/4)
    let twisted = SdfNode::box3d(2.0, 2.0, 2.0).twist(2.0);
    let v = 2.0 * 2f32.sqrt();
    let expect = v * 0.5 + (1.0 + v * v * 0.25).sqrt();
    assert!(
        (eval_lipschitz(&twisted) - expect).abs() < 1e-3,
        "twist bound {} vs {expect}",
        eval_lipschitz(&twisted)
    );
    // Same twist of a plane: unbounded child, documented fallback radius 10
    let twisted_plane = SdfNode::plane(Vec3::Y, 0.0).twist(2.0);
    assert!(eval_lipschitz(&twisted_plane).is_finite());
    assert!(eval_lipschitz(&SdfNode::sphere(1.0).repeat_infinite(3.0, 3.0, 3.0)).is_infinite());
}

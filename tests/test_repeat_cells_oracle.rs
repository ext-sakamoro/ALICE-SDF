//! `SdfNode::RepeatCells` (batch 1: mirrored repetition) against a brute-force
//! reference built independently of the node's law.
//!
//! Law: on each axis with spacing `s > 0` the cell is `c = round_half_up(p/s)`
//! (infinite) or `c = clamp(round_half_up(p/s + (n−1)/2), 0, n−1)` for `n`
//! cells centred on the origin; the child is evaluated at `p − centre(c)`, and
//! with `mirror` that axis is negated in odd cells, so neighbouring copies are
//! reflections of each other across the cell boundary. An axis with spacing 0
//! is not repeated.
//!
//! Oracles:
//! * the reference is the minimum of the child's exact distance over every
//!   copy (centre and flip computed here, not by the node);
//! * a child inside its cell: the node equals the reference (mirrored
//!   repetition is exact: for `p` and a child on the same side of a mirror
//!   plane, a reflected copy is never closer than the unreflected one);
//! * a child wider than its cell: the node over-estimates (documented as
//!   approximate); the test measures the error and checks the direction;
//! * `mirror = false` equals `RepeatInfinite` to the bit (one cell law);
//! * the field is continuous across cell boundaries and 1-Lipschitz for a
//!   1-Lipschitz child (sampled difference quotients);
//! * tree and compiled bytecode agree to the bit;
//! * `neighbours` / `variation` (batch 2) are refused with a typed error at
//!   construction, in the readers and by the compiler; the evaluators that
//!   cannot return an error give no value (NaN distance and gradient, an
//!   unbounded interval, a NaN in each shader); an invalid spacing or count is
//!   refused.
//!
//! The three `SdfNode::RepeatCells { .. }` literals below are deliberate: they
//! bypass the checked constructor to show that every other entry (validation,
//! readers, compiler, evaluators) still refuses the batch-2 fields.
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the reference is built with the platform libm"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::modifiers::{CellVariation, RepeatCellsError};
use alice_sdf::prelude::*;

/// A child off-centre in its cell (so that mirroring changes the shape) and
/// inside the cell of spacing (2, 1.5) on x / y.
fn contained_child() -> SdfNode {
    SdfNode::box3d_half_extents(0.3, 0.2, 0.25).translate(0.45, 0.2, 0.0)
}

/// A child that reaches past its cell's boundary on x (x in [0.75, 1.35] with
/// the boundary at 1): for a point at x = 0.5 the own copy is 0.25 away but the
/// mirrored neighbour's copy (x in [0.65, 1.25]) is 0.15 away.
fn wide_child() -> SdfNode {
    SdfNode::box3d_half_extents(0.3, 0.2, 0.25).translate(1.05, 0.0, 0.0)
}

const SPACING: Vec3 = Vec3::new(2.0, 1.5, 0.0);

/// (cell index c, centre in units of spacing) for every copy of one axis.
fn copies(n: Option<u32>) -> Vec<(i32, f32)> {
    match n {
        None => (-8..=8).map(|c| (c, c as f32)).collect(),
        Some(n) => (0..n as i32)
            .map(|c| (c, c as f32 - (n as f32 - 1.0) * 0.5))
            .collect(),
    }
}

/// The minimum over all copies of the child's distance.
fn brute_force(child: &SdfNode, p: Vec3, count: Option<[u32; 3]>, mirror: bool) -> f32 {
    let axis = |i: usize| -> Vec<(i32, f32)> {
        if SPACING[i] > 0.0 {
            copies(count.map(|c| c[i]))
        } else {
            vec![(0, 0.0)]
        }
    };
    let mut best = f32::INFINITY;
    for &(cx, ox) in &axis(0) {
        for &(cy, oy) in &axis(1) {
            for &(cz, oz) in &axis(2) {
                let mut q = p - Vec3::new(ox * SPACING.x, oy * SPACING.y, oz * SPACING.z);
                if mirror {
                    if cx.rem_euclid(2) == 1 {
                        q.x = -q.x;
                    }
                    if cy.rem_euclid(2) == 1 {
                        q.y = -q.y;
                    }
                    if cz.rem_euclid(2) == 1 {
                        q.z = -q.z;
                    }
                }
                best = best.min(eval(child, q));
            }
        }
    }
    best
}

fn points() -> Vec<Vec3> {
    let mut s = 0x2545_f491u32;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s as f32 / u32::MAX as f32
    };
    (0..3000)
        .map(|_| Vec3::new(r() * 12.0 - 6.0, r() * 9.0 - 4.5, r() * 2.0 - 1.0))
        .collect()
}

#[test]
fn mirrored_repetition_of_a_contained_child_is_exact() {
    for count in [None, Some([3, 4, 1]), Some([1, 2, 1])] {
        let child = contained_child();
        let node = child.clone().repeat_cells(SPACING, count, true).unwrap();
        let mut worst = 0.0f32;
        let pts = points();
        for &p in &pts {
            let got = eval(&node, p);
            let want = brute_force(&child, p, count, true);
            worst = worst.max((got - want).abs());
        }
        assert!(pts.len() >= 3000);
        assert!(worst <= 2e-6, "count {count:?}: worst |node − brute force| {worst}");
    }
}

/// A child wider than its cell: the single-cell value is an over-estimate
/// (never below the true distance), and it is wrong by a measurable amount.
#[test]
fn a_child_wider_than_its_cell_is_approximate_and_over_estimates() {
    let child = wide_child();
    let node = child.clone().repeat_cells(SPACING, None, true).unwrap();
    let (mut worst, mut below) = (0.0f32, 0usize);
    for &p in &points() {
        let got = eval(&node, p);
        let want = brute_force(&child, p, None, true);
        worst = worst.max(got - want);
        if got < want - 2e-6 {
            below += 1;
        }
    }
    assert_eq!(below, 0, "the single-cell value went below the true distance");
    assert!(worst > 0.05, "expected a measurable over-estimate, got {worst}");
}

#[test]
fn without_mirror_it_is_repeat_infinite_to_the_bit() {
    let child = contained_child();
    let spacing = Vec3::new(2.0, 1.5, 3.0);
    let a = child.clone().repeat_cells(spacing, None, false).unwrap();
    let b = child.repeat_infinite(spacing.x, spacing.y, spacing.z);
    for &p in &points() {
        assert_eq!(eval(&a, p).to_bits(), eval(&b, p).to_bits(), "{p:?}");
    }
}

#[test]
fn mirrored_field_is_one_lipschitz_across_cell_boundaries() {
    for child in [contained_child(), wide_child()] {
        let node = child.repeat_cells(SPACING, None, true).unwrap();
        let mut pairs = 0;
        let mut worst = 0.0f32;
        for &p in &points() {
            for d in [Vec3::X * 0.013, Vec3::Y * 0.011, Vec3::new(0.007, -0.009, 0.004)] {
                let q = p + d;
                worst = worst.max((eval(&node, p) - eval(&node, q)).abs() / d.length());
                pairs += 1;
            }
        }
        assert!(pairs > 8000);
        assert!(worst <= 1.0 + 1e-3, "difference quotient {worst}");
    }
}

#[test]
fn tree_and_compiled_agree() {
    for (count, mirror) in [(None, true), (Some([3, 4, 1]), true), (Some([2, 2, 1]), false)] {
        let node = contained_child().repeat_cells(SPACING, count, mirror).unwrap();
        let compiled = CompiledSdf::compile(&node);
        for &p in &points() {
            assert_eq!(
                eval(&node, p).to_bits(),
                eval_compiled(&compiled, p).to_bits(),
                "{count:?} {mirror} {p:?}"
            );
        }
    }
}

#[test]
fn invalid_parameters_are_refused() {
    let c = contained_child;
    assert!(matches!(
        c().repeat_cells(Vec3::new(-1.0, 1.0, 0.0), None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(Vec3::new(f32::NAN, 1.0, 0.0), None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(Vec3::ZERO, None, true),
        Err(RepeatCellsError::Spacing(_))
    ));
    assert!(matches!(
        c().repeat_cells(SPACING, Some([0, 2, 1]), true),
        Err(RepeatCellsError::Count(_))
    ));
}

fn literal(neighbours: bool, variation: Option<CellVariation>) -> SdfNode {
    SdfNode::RepeatCells {
        child: std::sync::Arc::new(contained_child()),
        spacing: SPACING,
        count: None,
        mirror: true,
        neighbours,
        variation,
    }
}

/// Batch 2 fields: present in the node (the stored form does not change when
/// they arrive), refused until then — by validation, the readers and the compiler.
#[test]
fn neighbours_and_variation_are_refused_until_they_are_implemented() {
    let node = literal(true, None);
    let varied = literal(false, Some(CellVariation { seed: 7, scale: 0.1 }));
    for n in [&node, &varied] {
        assert!(matches!(
            n.validate_repeat_cells(),
            Err(RepeatCellsError::NotYetSupported(_))
        ));
    }
    use alice_sdf::io::{from_json_string, to_json_string};
    for n in [&node, &varied] {
        let json = to_json_string(&SdfTree::new(n.clone())).unwrap();
        assert!(from_json_string(&json).is_err(), "reader accepted a batch-2 field");
    }
    for (i, n) in [&node, &varied].into_iter().enumerate() {
        let path = std::env::temp_dir().join(format!(
            "alice_sdf_repeat_cells_{}_{i}.asdf",
            std::process::id()
        ));
        alice_sdf::io::save_asdf(&SdfTree::new(n.clone()), &path).unwrap();
        let loaded = alice_sdf::io::load_asdf(&path);
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_err(), "binary reader accepted a batch-2 field");
    }
    for n in [&node, &varied] {
        assert!(CompiledSdf::try_compile(n).is_err());
    }
}

/// The I-packet reader refuses the batch-2 fields too.
#[cfg(feature = "asp")]
#[test]
fn i_packet_reader_refuses_neighbours_and_variation() {
    use alice_sdf::asp_bridge::{create_sdf_i_packet, decode_sdf_i_packet};
    for n in [literal(true, None), literal(false, Some(CellVariation { seed: 7, scale: 0.1 }))] {
        let packet = create_sdf_i_packet(&SdfTree::new(n), 1).unwrap();
        assert!(decode_sdf_i_packet(&packet).is_err());
    }
    // the same packet path accepts a supported node
    let ok = contained_child().repeat_cells(SPACING, None, true).unwrap();
    let packet = create_sdf_i_packet(&SdfTree::new(ok), 1).unwrap();
    assert!(decode_sdf_i_packet(&packet).is_ok());
}

/// The evaluators that cannot return an error (a direct enum literal is not
/// stopped at construction) give "no value" for the batch-2 fields instead of
/// the single-cell value with the field ignored: the tree distance and gradient
/// are NaN (and the NaN reaches a caller through a transform), the interval is
/// unbounded, and each shader returns a quiet NaN for the node. The checked
/// entries (construction, readers, compile) refuse with an error, above.
#[test]
fn the_evaluators_give_no_value_for_neighbours_and_variation() {
    use alice_sdf::interval::{eval_interval, Vec3Interval};
    let supported = contained_child().repeat_cells(SPACING, None, true).unwrap();
    let p = Vec3::new(0.3, -0.2, 0.1);
    assert!(eval(&supported, p).is_finite());
    for n in [literal(true, None), literal(false, Some(CellVariation { seed: 7, scale: 0.1 }))] {
        assert!(eval(&n, p).is_nan());
        assert!(eval(&n.clone().translate(1.0, 0.0, 0.0), p).is_nan(), "NaN lost through a transform");
        assert!(eval_gradient(&n, p).is_nan());
        let iv = eval_interval(&n, Vec3Interval::from_bounds(Vec3::splat(-1.0), Vec3::splat(1.0)));
        assert!(iv.lo == f32::NEG_INFINITY && iv.hi == f32::INFINITY, "{iv:?}");
        shader_nan(&n, &supported);
    }
}

/// Each shader returns a quiet NaN for a node with the batch-2 fields, and the
/// supported form has none.
#[allow(unused_variables)]
fn shader_nan(n: &SdfNode, supported: &SdfNode) {
    #[cfg(feature = "glsl")]
    {
        use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
        let t = |x: &SdfNode| GlslShader::transpile(x, GlslTranspileMode::Hardcoded).source;
        assert!(t(n).contains("uintBitsToFloat(0x7fc00000u)"), "GLSL");
        assert!(!t(supported).contains("0x7fc00000u"));
    }
    #[cfg(feature = "hlsl")]
    {
        use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
        let t = |x: &SdfNode| HlslShader::transpile(x, HlslTranspileMode::Hardcoded).source;
        assert!(t(n).contains("asfloat(0x7fc00000u)"), "HLSL");
        assert!(!t(supported).contains("0x7fc00000u"));
    }
    #[cfg(feature = "gpu")]
    {
        use alice_sdf::compiled::{TranspileMode, WgslShader};
        let t = |x: &SdfNode| WgslShader::transpile(x, TranspileMode::Hardcoded).source;
        assert!(t(n).contains("bitcast<f32>(0x7fc00000u)"), "WGSL");
        assert!(!t(supported).contains("0x7fc00000u"));
    }
}

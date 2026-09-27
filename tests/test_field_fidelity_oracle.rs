//! Oracles for what a field's distance claim is worth.
//!
//! Written before the implementation. Two things are checked: the static
//! classification is consistent with the bound it is derived from, and the
//! *measured* difference quotients agree with closed forms that can be
//! written down by hand for simple fields.
//!
//! # Closed forms used
//!
//! * A plane `f(p) = p·n − d` has `|f(p) − f(q)| / |p − q| = |dir·n|` for a
//!   pair separated along `dir`, so over random directions the quotient
//!   sweeps `[0, 1]`: the supremum is exactly `1` (reached along `±n`) and
//!   the infimum exactly `0` (reached in the plane). Both ends are attained
//!   in the limit, so a dense enough sample must come close to each.
//! * A sphere `f(p) = |p| − r` gives `| |p| − |q| | / |p − q| = |dir·p̂|` to
//!   first order in the separation, so the same `[0, 1]` sweep holds away
//!   from the centre.
//! * Scaling a field's *value* by `k` scales every quotient by `k`, so a
//!   field that reports `k·dist` has tension exactly `k`. This is the
//!   over-reporting mode, and `k > 1` is what pierces thin geometry.
//! * For every node, the measured maximum may never exceed the static bound
//!   `eval_lipschitz` — that is the bound's definition, and a violation is a
//!   bug in one of the two.

use alice_sdf::fidelity::{distance_fidelity, Fidelity};
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::measure::measure_tension;
use alice_sdf::types::{Aabb, SdfNode};
use glam::Vec3;

fn region(half: f32) -> Aabb {
    Aabb::new(Vec3::splat(-half), Vec3::splat(half))
}

#[test]
fn a_plane_sweeps_the_whole_unit_range_of_quotients() {
    let plane = SdfNode::plane(Vec3::Y, 0.0);
    let t = measure_tension(&plane, region(4.0), 40_000, 0xa11ce, 1e-3);
    assert!(
        t.max_quotient <= 1.0 + 1e-3,
        "a plane cannot exceed 1, measured {}",
        t.max_quotient
    );
    assert!(
        t.max_quotient > 0.99,
        "the supremum 1 should be approached, measured {}",
        t.max_quotient
    );
    assert!(
        t.min_quotient < 0.02,
        "the infimum 0 should be approached, measured {}",
        t.min_quotient
    );
    assert!(!t.tears(), "an exact plane must not read as tearing");
    assert!(t.sample_count > 1000, "too few exterior pairs counted");
}

#[test]
fn a_sphere_never_over_reports() {
    let sphere = SdfNode::sphere(1.0);
    let t = measure_tension(&sphere, region(4.0), 40_000, 0x5b1e, 1e-3);
    assert!(
        t.max_quotient <= 1.0 + 1e-2,
        "exact sphere measured {}",
        t.max_quotient
    );
    assert!(matches!(
        distance_fidelity(&sphere),
        Fidelity::NeverOverReports { .. }
    ));
    assert!(!distance_fidelity(&sphere).can_overshoot());
    assert_eq!(distance_fidelity(&sphere).safe_step_scale(), Some(1.0));
}

#[test]
fn the_measured_maximum_never_exceeds_the_static_bound() {
    // the bound's definition, checked through the new API over a spread of
    // laws including the ones whose bound is above 1
    let nodes: Vec<(&str, SdfNode)> = vec![
        ("sphere", SdfNode::sphere(1.0)),
        ("box", SdfNode::box3d(0.8, 0.5, 1.2)),
        ("torus", SdfNode::torus(1.0, 0.3)),
        ("plane", SdfNode::plane(Vec3::Y, 0.0)),
        (
            "union",
            SdfNode::sphere(1.0).union(SdfNode::box3d(0.7, 0.7, 0.7)),
        ),
        (
            "smooth_union",
            SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.7, 0.7, 0.7), 0.3),
        ),
        ("gyroid", SdfNode::gyroid(1.0, 0.1)),
    ];
    for (name, node) in nodes {
        let bound = eval_lipschitz(&node);
        let t = measure_tension(&node, region(3.0), 30_000, 0xbeef, 1e-3);
        assert!(
            t.max_quotient <= bound * 1.05 + 1e-3,
            "{name}: measured {} exceeds the claimed bound {bound}",
            t.max_quotient
        );
        // and the classification has to agree with the same bound
        match distance_fidelity(&node) {
            Fidelity::NeverOverReports { lipschitz } => assert!(lipschitz <= 1.0),
            Fidelity::OverReportsBy { lipschitz } => assert!(lipschitz > 1.0),
            Fidelity::Unbounded => assert!(!bound.is_finite()),
        }
    }
}

#[test]
fn a_field_that_over_reports_reads_as_tearing() {
    // a gyroid's law is not a distance: its bound is well above 1, which is
    // the mode that steps past a surface
    let g = SdfNode::gyroid(1.0, 0.1);
    let bound = eval_lipschitz(&g);
    assert!(bound > 1.0, "gyroid bound should exceed 1, got {bound}");
    match distance_fidelity(&g) {
        Fidelity::OverReportsBy { lipschitz } => {
            assert!((lipschitz - bound).abs() < 1e-6);
            assert!(distance_fidelity(&g).can_overshoot());
            assert_eq!(distance_fidelity(&g).safe_step_scale(), Some(bound));
        }
        other => panic!("expected OverReportsBy, got {other:?}"),
    }
    let t = measure_tension(&g, region(3.0), 30_000, 0x9001, 1e-3);
    assert!(
        t.tears(),
        "a field with a bound of {bound} should measure above 1, got {}",
        t.max_quotient
    );
}

#[test]
fn an_unbounded_law_has_no_safe_step() {
    // polar repetition of an arbitrary child is not Lipschitz on the
    // exterior (interval.rs returns INFINITY rather than guessing)
    let repeated = SdfNode::sphere(0.4).polar_repeat(7);
    if eval_lipschitz(&repeated).is_finite() {
        // the crate may have tightened this; then the test's premise is gone
        // and the classification still has to be consistent
        assert!(!matches!(distance_fidelity(&repeated), Fidelity::Unbounded));
    } else {
        assert_eq!(distance_fidelity(&repeated), Fidelity::Unbounded);
        assert_eq!(distance_fidelity(&repeated).safe_step_scale(), None);
        assert!(distance_fidelity(&repeated).can_overshoot());
    }
}

#[test]
fn measurement_is_reproducible_from_its_seed() {
    let node = SdfNode::sphere(1.0).union(SdfNode::box3d(0.6, 0.6, 0.6));
    let a = measure_tension(&node, region(3.0), 5_000, 0x1234, 1e-3);
    let b = measure_tension(&node, region(3.0), 5_000, 0x1234, 1e-3);
    assert_eq!(a.max_quotient.to_bits(), b.max_quotient.to_bits());
    assert_eq!(a.min_quotient.to_bits(), b.min_quotient.to_bits());
    assert_eq!(a.sample_count, b.sample_count);
    let c = measure_tension(&node, region(3.0), 5_000, 0x1235, 1e-3);
    assert!(
        c.max_quotient.to_bits() != a.max_quotient.to_bits()
            || c.min_quotient.to_bits() != a.min_quotient.to_bits(),
        "a different seed should not reproduce the same extremes exactly"
    );
}

#[test]
fn zero_samples_is_not_a_division_by_zero() {
    let node = SdfNode::sphere(1.0);
    let t = measure_tension(&node, region(1.0), 0, 0, 1e-3);
    assert_eq!(t.sample_count, 0);
    assert!(t.max_quotient.is_finite() && t.min_quotient.is_finite());
    assert!(!t.tears());
}

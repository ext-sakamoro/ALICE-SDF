//! `alice_sdf::interval_gradient::eval_interval_gradient` encloses the field's gradient over a box.
//!
//! oracle: the gradient at a point by central differences of the field itself (`eval`, step 1e-3),
//! independent of the code under test. At random points of random boxes the difference quotient lies
//! inside the box's component intervals (within the difference quotient's own error, 2e-3), for every
//! covered node kind; where two children meet (min / max / smooth blends) it is a blend of both
//! children's gradients and the hull must hold it. An uncovered kind reports `known == false` with
//! `[-1, 1]` on every axis.

use alice_sdf::eval::eval;
use alice_sdf::interval::Vec3Interval;
use alice_sdf::interval_gradient::eval_interval_gradient;
use alice_sdf::SdfNode;
use glam::{Quat, Vec3};

/// A small deterministic generator (the oracle must not depend on a crate's random stream)
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        #[allow(clippy::cast_precision_loss)]
        let v = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
        v
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

fn numeric_gradient(node: &SdfNode, p: Vec3) -> Vec3 {
    let h = 1e-3;
    Vec3::new(
        eval(node, p + Vec3::X * h) - eval(node, p - Vec3::X * h),
        eval(node, p + Vec3::Y * h) - eval(node, p - Vec3::Y * h),
        eval(node, p + Vec3::Z * h) - eval(node, p - Vec3::Z * h),
    ) / (2.0 * h)
}

fn scenes() -> Vec<(&'static str, SdfNode)> {
    let n = Vec3::new(1.0, 2.0, 3.0).normalize();
    let rot = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0).normalize(), 0.7);
    vec![
        ("sphere", SdfNode::sphere(1.0)),
        ("plane", SdfNode::plane(n, 0.3)),
        ("box", SdfNode::box3d(2.0, 1.0, 4.0)),
        ("cylinder", SdfNode::cylinder(0.5, 4.0)),
        ("translate", SdfNode::sphere(0.8).translate(0.5, -0.3, 0.2)),
        ("rotate", SdfNode::box3d(2.0, 1.0, 3.0).rotate(rot)),
        ("rotate cylinder", SdfNode::cylinder(0.4, 3.0).rotate(rot)),
        ("scale", SdfNode::box3d(1.0, 2.0, 1.0).scale(1.7)),
        (
            "union",
            SdfNode::box3d(2.0, 2.0, 2.0).union(SdfNode::sphere(0.9).translate(1.2, 0.0, 0.0)),
        ),
        (
            "intersection",
            SdfNode::box3d(2.0, 2.0, 2.0).intersection(SdfNode::sphere(1.2)),
        ),
        (
            "subtraction",
            SdfNode::box3d(3.0, 3.0, 3.0).subtract(SdfNode::cylinder(0.6, 5.0)),
        ),
        (
            "smooth union",
            SdfNode::sphere(0.8).smooth_union(SdfNode::sphere(0.8).translate(1.2, 0.0, 0.0), 0.4),
        ),
        (
            "smooth intersection",
            SdfNode::box3d(2.0, 2.0, 2.0).smooth_intersection(SdfNode::sphere(1.3), 0.3),
        ),
        (
            "smooth subtraction",
            SdfNode::box3d(3.0, 3.0, 3.0).smooth_subtract(SdfNode::sphere(1.0), 0.3),
        ),
        (
            "pins on a floor",
            SdfNode::box3d(20.0, 2.0, 20.0)
                .translate(0.0, -1.0, 0.0)
                .union(SdfNode::cylinder(0.3, 6.0).translate(-0.8, 3.0, 0.0))
                .union(SdfNode::cylinder(0.3, 6.0).translate(0.8, 3.0, 0.0)),
        ),
    ]
}

#[test]
fn every_covered_kind_encloses_the_point_gradient() {
    let mut rng = Lcg(0x5eed_1234_abcd_0001);
    let mut compared = 0_usize;
    let mut failures = Vec::new();
    for (name, node) in scenes() {
        for _ in 0..200 {
            let c = Vec3::new(
                rng.range(-2.5, 2.5),
                rng.range(-2.5, 2.5),
                rng.range(-2.5, 2.5),
            );
            let half = [0.005_f32, 0.05, 0.3][(rng.next() * 3.0) as usize % 3];
            let lo = c - Vec3::splat(half);
            let hi = c + Vec3::splat(half);
            let g = eval_interval_gradient(&node, Vec3Interval::from_bounds(lo, hi));
            assert!(g.known, "{name}: a covered kind reported unknown");
            for _ in 0..6 {
                let p = Vec3::new(
                    rng.range(lo.x, hi.x),
                    rng.range(lo.y, hi.y),
                    rng.range(lo.z, hi.z),
                );
                let v = numeric_gradient(&node, p);
                if !v.is_finite() {
                    continue;
                }
                let tol = 2e-3;
                let inside =
                    |i: alice_sdf::interval::Interval, x: f32| x >= i.lo - tol && x <= i.hi + tol;
                if !(inside(g.x, v.x) && inside(g.y, v.y) && inside(g.z, v.z))
                    && failures.len() < 12
                {
                    failures.push(format!(
                        "{name}: at {p} gradient {v} outside {g:?} (box {lo}..{hi})"
                    ));
                }
                compared += 1;
            }
        }
    }
    assert!(compared > 10_000, "only {compared} points compared");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn an_uncovered_kind_is_unknown_never_narrow() {
    let g = eval_interval_gradient(
        &SdfNode::torus(1.0, 0.3),
        Vec3Interval::from_bounds(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.1, 0.1, 0.1)),
    );
    assert!(!g.known);
    for i in [g.x, g.y, g.z] {
        assert!(i.lo <= -1.0 && i.hi >= 1.0, "{i:?}");
    }
}

#[test]
fn a_box_away_from_any_ridge_gets_a_narrow_direction() {
    // the plane's side of one pin, far from the other pin and the floor: one direction, narrow
    let pins = SdfNode::box3d(20.0, 2.0, 20.0)
        .translate(0.0, -1.0, 0.0)
        .union(SdfNode::cylinder(0.3, 6.0).translate(-0.8, 3.0, 0.0))
        .union(SdfNode::cylinder(0.3, 6.0).translate(0.8, 3.0, 0.0));
    let g = eval_interval_gradient(
        &pins,
        Vec3Interval::from_bounds(Vec3::new(-1.6, 2.9, -0.05), Vec3::new(-1.5, 3.0, 0.05)),
    );
    assert!(g.known);
    assert!(g.x.hi < -0.9, "points away from the pin along -x: {g:?}");
    // between the pins on the line through both axes: both pins' directions (a ridge)
    let r = eval_interval_gradient(
        &pins,
        Vec3Interval::from_bounds(Vec3::new(-0.02, 2.9, -0.02), Vec3::new(0.02, 3.0, 0.02)),
    );
    assert!(
        r.x.lo < -0.9 && r.x.hi > 0.9,
        "the ridge between the pins holds both: {r:?}"
    );
}

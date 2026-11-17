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
        ("rounded box", SdfNode::rounded_box(1.0, 0.5, 2.0, 0.2)),
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

/// The closed-form gradient of each primitive at a point away from its edges and axes
fn closed_form(kind: &str, p: Vec3) -> Option<Vec3> {
    match kind {
        "sphere" => Some(p / p.length()),
        "plane" => Some(Vec3::new(1.0, 2.0, 3.0).normalize()),
        "box" => {
            let h = Vec3::new(1.0, 0.5, 2.0);
            let q = p.abs() - h;
            let s = Vec3::new(p.x.signum(), p.y.signum(), p.z.signum());
            if q.max_element() > 0.0 {
                let m = q.max(Vec3::ZERO);
                Some(m * s / m.length())
            } else {
                let mut g = Vec3::ZERO;
                if q.x >= q.y && q.x >= q.z {
                    g.x = s.x;
                } else if q.y >= q.z {
                    g.y = s.y;
                } else {
                    g.z = s.z;
                }
                Some(g)
            }
        }
        "cylinder" => {
            let (r, hh) = (0.5_f32, 2.0_f32);
            let rho = (p.x * p.x + p.z * p.z).sqrt();
            let radial = Vec3::new(p.x / rho, 0.0, p.z / rho);
            let axial = Vec3::new(0.0, p.y.signum(), 0.0);
            let (qr, qy) = (rho - r, p.y.abs() - hh);
            if qr > 0.0 || qy > 0.0 {
                let (a, b) = (qr.max(0.0), qy.max(0.0));
                let n = (a * a + b * b).sqrt();
                Some((radial * a + axial * b) / n)
            } else if qr >= qy {
                Some(radial)
            } else {
                Some(axial)
            }
        }
        _ => None,
    }
}

#[test]
fn the_primitives_hold_their_closed_form_gradient() {
    // oracle: the closed forms above, written from the primitives' definitions (not from the code)
    let n = Vec3::new(1.0, 2.0, 3.0).normalize();
    let prims = [
        ("sphere", SdfNode::sphere(1.0)),
        ("plane", SdfNode::plane(n, 0.3)),
        ("box", SdfNode::box3d(2.0, 1.0, 4.0)),
        ("cylinder", SdfNode::cylinder(0.5, 4.0)),
    ];
    let mut rng = Lcg(0x0c10_5edf_0123_4567);
    let mut compared = 0_usize;
    for (kind, node) in &prims {
        for _ in 0..2000 {
            let p = Vec3::new(
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
                rng.range(-3.0, 3.0),
            );
            let half = [0.0_f32, 1e-3, 0.05][(rng.next() * 3.0) as usize % 3];
            let g = eval_interval_gradient(
                node,
                Vec3Interval::from_bounds(p - Vec3::splat(half), p + Vec3::splat(half)),
            );
            let want = closed_form(kind, p).expect("kind");
            if !want.is_finite() {
                continue;
            }
            let tol = 1e-5;
            assert!(
                want.x >= g.x.lo - tol
                    && want.x <= g.x.hi + tol
                    && want.y >= g.y.lo - tol
                    && want.y <= g.y.hi + tol
                    && want.z >= g.z.lo - tol
                    && want.z <= g.z.hi + tol,
                "{kind}: closed form {want} at {p} outside {g:?} (half {half})"
            );
            compared += 1;
        }
    }
    assert!(compared > 7_000, "only {compared} compared");
}

#[test]
fn a_point_box_away_from_edges_is_tight() {
    // a box of zero width off every edge: the enclosure is the closed form within a few ulp
    let g = eval_interval_gradient(
        &SdfNode::sphere(1.0),
        Vec3Interval::from_bounds(Vec3::new(0.6, 0.8, 0.0), Vec3::new(0.6, 0.8, 0.0)),
    );
    assert!(g.x.width() < 1e-5 && g.y.width() < 1e-5, "{g:?}");
    assert!(
        (g.x.lo - 0.6).abs() < 1e-5 && (g.y.lo - 0.8).abs() < 1e-5,
        "{g:?}"
    );
}

/// Scenes where the band rules decide the answer: a slot (the cutter's inside is air), a rounded box
/// next to a plate (the rounding radius shifts the box's value), smooth blends (children above / below
/// the blended value)
fn band_scenes() -> Vec<(&'static str, SdfNode)> {
    let mut v = scenes();
    v.push((
        "slot",
        SdfNode::box3d(6.0, 3.0, 4.0)
            .subtract(SdfNode::box3d(1.0, 4.0, 6.0).translate(0.0, 1.0, 0.0)),
    ));
    v.push((
        "rounded box on a plate",
        SdfNode::rounded_box(1.0, 0.6, 1.0, 0.4)
            .translate(0.0, 0.6, 0.0)
            .union(SdfNode::box3d(6.0, 0.4, 6.0).translate(0.0, -0.2, 0.0)),
    ));
    v.push((
        "smooth union of plates",
        SdfNode::box3d(4.0, 0.3, 4.0)
            .translate(0.0, 0.5, 0.0)
            .smooth_union(SdfNode::box3d(4.0, 0.3, 4.0).translate(0.0, -0.5, 0.0), 0.6),
    ));
    v.push((
        "smooth subtraction slot",
        SdfNode::box3d(6.0, 3.0, 4.0)
            .smooth_subtract(SdfNode::box3d(1.0, 4.0, 6.0).translate(0.0, 1.0, 0.0), 0.3),
    ));
    v.push((
        "scaled slot",
        SdfNode::box3d(6.0, 3.0, 4.0)
            .subtract(SdfNode::box3d(1.0, 4.0, 6.0).translate(0.0, 1.0, 0.0))
            .scale(0.6),
    ));
    v
}

#[test]
fn the_banded_enclosure_holds_every_point_whose_value_is_in_the_band() {
    use alice_sdf::interval::Interval;
    use alice_sdf::interval_gradient::eval_interval_gradient_in_band;
    let mut rng = Lcg(0xba4d_0bad_c0de_0042);
    let (mut compared, mut in_band_boxes) = (0_usize, 0_usize);
    let mut failures = Vec::new();
    for (name, node) in band_scenes() {
        for _ in 0..300 {
            let c = Vec3::new(
                rng.range(-2.5, 2.5),
                rng.range(-2.5, 2.5),
                rng.range(-2.5, 2.5),
            );
            let half = [0.01_f32, 0.08, 0.4][(rng.next() * 3.0) as usize % 3];
            let (lo, hi) = (c - Vec3::splat(half), c + Vec3::splat(half));
            let band = match (rng.next() * 4.0) as usize {
                0 => Interval {
                    lo: 0.0,
                    hi: f32::INFINITY,
                },
                1 => Interval {
                    lo: f32::NEG_INFINITY,
                    hi: 0.0,
                },
                2 => {
                    let a = rng.range(-1.0, 1.0);
                    Interval {
                        lo: a,
                        hi: a + rng.range(0.0, 0.6),
                    }
                }
                _ => {
                    let a = rng.range(-0.3, 0.3);
                    Interval {
                        lo: a,
                        hi: f32::INFINITY,
                    }
                }
            };
            let g = eval_interval_gradient_in_band(&node, Vec3Interval::from_bounds(lo, hi), band);
            for _ in 0..12 {
                let p = Vec3::new(
                    rng.range(lo.x, hi.x),
                    rng.range(lo.y, hi.y),
                    rng.range(lo.z, hi.z),
                );
                let d = eval(&node, p);
                if !(d >= band.lo && d <= band.hi) {
                    continue;
                }
                let v = numeric_gradient(&node, p);
                if !v.is_finite() {
                    continue;
                }
                compared += 1;
                let Some(g) = g else {
                    if failures.len() < 12 {
                        failures.push(format!(
                            "{name}: {p} has value {d} in {band:?} but the box was left out"
                        ));
                    }
                    continue;
                };
                in_band_boxes += 1;
                let tol = 2e-3;
                let inside = |i: Interval, x: f32| x >= i.lo - tol && x <= i.hi + tol;
                if !(inside(g.x, v.x) && inside(g.y, v.y) && inside(g.z, v.z))
                    && failures.len() < 12
                {
                    failures.push(format!(
                        "{name}: at {p} (d {d}, band {band:?}) gradient {v} outside {g:?}"
                    ));
                }
            }
        }
    }
    assert!(
        compared > 20_000 && in_band_boxes > 20_000,
        "only {compared} points compared"
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_air_band_keeps_less_than_the_full_band_and_nothing_inside_the_material() {
    use alice_sdf::interval::Interval;
    use alice_sdf::interval_gradient::eval_interval_gradient_in_band;
    // a box resting on a plate: the air band's enclosure lies within the full one (what holds; a child's
    // band is not intersected with its sibling's value, so it is not as tight as the air alone), and a
    // box wholly inside the material has no air point at all
    let s = SdfNode::box3d(2.0, 2.0, 2.0)
        .translate(0.0, 1.0, 0.0)
        .union(SdfNode::box3d(8.0, 1.0, 8.0).translate(0.0, -0.5, 0.0));
    let air = Interval {
        lo: 0.0,
        hi: f32::INFINITY,
    };
    let b = Vec3Interval::from_bounds(Vec3::new(0.95, -0.05, -0.05), Vec3::new(1.05, 0.05, 0.05));
    let full = eval_interval_gradient(&s, b);
    let a = eval_interval_gradient_in_band(&s, b, air).expect("air in the box");
    for (x, f) in [(a.x, full.x), (a.y, full.y), (a.z, full.z)] {
        assert!(x.lo >= f.lo && x.hi <= f.hi, "air {x:?} outside full {f:?}");
    }
    assert!(
        a.x.lo > -0.1,
        "no air point sees a wall behind it along -x: {a:?}"
    );
    let inside = Vec3Interval::from_bounds(Vec3::new(-0.3, -0.6, -0.3), Vec3::new(0.3, -0.4, 0.3));
    assert!(eval_interval_gradient_in_band(&s, inside, air).is_none());
}

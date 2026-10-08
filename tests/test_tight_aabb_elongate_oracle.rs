//! Oracle for the level-set AABB bound on `Elongate` and the nodes added with it
//! (`Mirror`, `ScaleNonUniform`, `RoundedBox`, `RoundedCylinder`, `SmoothIntersection`,
//! `SmoothSubtraction`, `WithMaterial`, and the exact box of a
//! rotated round leaf)
//!
//! Before: `analytic_aabb` returned `Unsupported` for any tree holding an `Elongate`, the
//! search fell back to interval arithmetic, and a strip built as
//! `translate(rotate_z(elongate_x(rotate_x(cylinder))))` turned by 45 degrees came back
//! as about +-500 (the whole `preset_large` search box)
//!
//! Independent references (none of them calls the bound under test):
//! (a) closed form: the stadium strip from `a` to `b` with in-plane radius `R` and
//!     thickness `w` spans `[min(a, b) - R, max(a, b) + R]` in X / Y and `+-w / 2` in Z
//! (b) closed form: an elongated box / cylinder grows by the elongation per axis
//! (c) soundness: every grid / random point with `f <= 0` lies inside the box
//! (d) the interval enclosure of an `Elongate` contains every point value it encloses
//!
//! Every test counts its comparisons and fails when none were made
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::eval;
use alice_sdf::interval::{eval_interval, Vec3Interval};
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{
    analytic_aabb, compute_tight_aabb_with_config, AnalyticAabb, TightAabbConfig,
};
use glam::{Quat, Vec2, Vec3};
use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

fn bounded(node: &SdfNode) -> Aabb {
    match analytic_aabb(node) {
        AnalyticAabb::Bounded(b) => b,
        other => panic!("expected a bound, got {other:?} for {node:?}"),
    }
}

fn near(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).abs().max_element() <= tol
}

/// The strip of `capsule_polyline_sdf` in ALICE-LOL, one edge: a Z-axis cylinder
/// elongated along local X by half the edge, turned to the edge and moved to its middle
fn strip(a: Vec2, b: Vec2, radius: f32, width: f32) -> SdfNode {
    let edge = b - a;
    let puck = SdfNode::Rotate {
        child: Arc::new(SdfNode::Cylinder {
            radius,
            half_height: width * 0.5,
        }),
        rotation: Quat::from_rotation_x(FRAC_PI_2),
    };
    let elongated = SdfNode::Elongate {
        child: Arc::new(puck),
        amount: Vec3::new(edge.length() * 0.5, 0.0, 0.0),
    };
    let mid = (a + b) * 0.5;
    SdfNode::Rotate {
        child: Arc::new(elongated),
        rotation: Quat::from_rotation_z(edge.y.atan2(edge.x)),
    }
    .translate(mid.x, mid.y, 0.0)
}

fn polyline(pts: &[Vec2], radius: f32, width: f32) -> SdfNode {
    let mut it = pts.windows(2).map(|w| strip(w[0], w[1], radius, width));
    let first = it.next().expect("two points");
    it.fold(first, SdfNode::union)
}

/// Closed form (a): the box of the union of stadium strips
fn polyline_box(pts: &[Vec2], radius: f32, width: f32) -> (Vec3, Vec3) {
    let lo = pts.iter().fold(Vec2::splat(f32::MAX), |m, p| m.min(*p));
    let hi = pts.iter().fold(Vec2::splat(f32::MIN), |m, p| m.max(*p));
    (
        Vec3::new(lo.x - radius, lo.y - radius, -width * 0.5),
        Vec3::new(hi.x + radius, hi.y + radius, width * 0.5),
    )
}

/// Soundness (c): every solid point of a grid over `[lo, hi]` lies in `b` (with `slack`).
/// Returns the number of solid points checked.
fn assert_grid_inside(node: &SdfNode, b: &Aabb, lo: Vec3, hi: Vec3, n: u32, slack: f32) -> usize {
    let mut solid = 0;
    let step = (hi - lo) / n as f32;
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=n {
                let p = lo + step * Vec3::new(i as f32, j as f32, k as f32);
                if eval(node, p) <= 0.0 {
                    solid += 1;
                    assert!(
                        p.cmpge(b.min - Vec3::splat(slack)).all()
                            && p.cmple(b.max + Vec3::splat(slack)).all(),
                        "solid {p:?} outside {b:?}"
                    );
                }
            }
        }
    }
    solid
}

#[test]
fn an_elongated_box_and_cylinder_grow_by_the_elongation() {
    let mut compared = 0;
    // box (1, 2, 3) elongated by (0.5, 0.25, 2): half (1.5, 2.25, 5)
    let node = SdfNode::box3d(2.0, 4.0, 6.0).elongate(0.5, 0.25, 2.0);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(1.5, 2.25, 5.0), 1e-4), "{b:?}");
    assert!(near(b.min, -Vec3::new(1.5, 2.25, 5.0), 1e-4), "{b:?}");
    compared += 2;
    // cylinder r = 2, h = 1 (Y axis) elongated by (3, 0.5, 1.5): half (5, 1.5, 3.5)
    let node = SdfNode::Cylinder {
        radius: 2.0,
        half_height: 1.0,
    }
    .elongate(3.0, 0.5, 1.5);
    let b = bounded(&node);
    assert!(near(b.max, Vec3::new(5.0, 1.5, 3.5), 1e-4), "{b:?}");
    assert!(near(b.min, -Vec3::new(5.0, 1.5, 3.5), 1e-4), "{b:?}");
    compared += 2;
    let solid = assert_grid_inside(&node, &b, Vec3::splat(-7.0), Vec3::splat(7.0), 56, 0.0);
    assert!(solid > 1000, "only {solid} solid points");
    // the solid reaches the box: the farthest solid grid point along +X is within one step
    let reach = (0..=400)
        .map(|i| i as f32 * 0.0125 + 4.9)
        .filter(|x| eval(&node, Vec3::new(*x, 0.0, 0.0)) <= 0.0)
        .fold(0.0f32, f32::max);
    assert!((reach - 5.0).abs() <= 0.0125, "reach {reach}");
    compared += 1;
    assert!(compared > 0);
}

#[test]
fn a_rotated_strip_is_bounded_by_the_stadium_closed_form() {
    let (radius, width, len) = (2.75f32, 5.0f32, 40.0f32);
    let cfg = TightAabbConfig::preset_large();
    let mut compared = 0;
    for deg in [0.0f32, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 170.0] {
        let dir = Vec2::new(deg.to_radians().cos(), deg.to_radians().sin());
        let (a, b) = (Vec2::new(3.0, -2.0), Vec2::new(3.0, -2.0) + dir * len);
        let node = strip(a, b, radius, width);
        let (lo, hi) = polyline_box(&[a, b], radius, width);
        let bound = bounded(&node);
        assert!(
            near(bound.min, lo, 2e-3) && near(bound.max, hi, 2e-3),
            "{deg} deg analytic {bound:?} vs closed form {lo:?}..{hi:?}"
        );
        let aabb = compute_tight_aabb_with_config(&node, &cfg);
        assert!(
            near(aabb.min, lo, 0.05) && near(aabb.max, hi, 0.05),
            "{deg} deg combined {aabb:?} vs closed form {lo:?}..{hi:?}"
        );
        let solid = assert_grid_inside(
            &node,
            &aabb,
            lo - Vec3::splat(4.0),
            hi + Vec3::splat(4.0),
            48,
            0.0,
        );
        assert!(solid > 0, "{deg} deg: no solid grid point");
        compared += 2;
    }
    assert_eq!(compared, 18);
}

#[test]
fn the_four_point_polyline_of_alice_lol_gets_a_box_within_a_few_millimetres() {
    // the hook strip: radius 2.75, thickness 5, four points with 45 degree legs
    let pts = [
        Vec2::new(0.0, 0.0),
        Vec2::new(30.0, 30.0),
        Vec2::new(55.0, 5.0),
        Vec2::new(55.0, -20.0),
    ];
    let (radius, width) = (2.75, 5.0);
    let node = polyline(&pts, radius, width);
    let (lo, hi) = polyline_box(&pts, radius, width);
    let aabb = compute_tight_aabb_with_config(&node, &TightAabbConfig::preset_large());
    // was about +-500 before the fix
    assert!(
        near(aabb.min, lo, 0.05) && near(aabb.max, hi, 0.05),
        "{aabb:?} vs closed form {lo:?}..{hi:?}"
    );
    let solid = assert_grid_inside(
        &node,
        &aabb,
        lo - Vec3::splat(5.0),
        hi + Vec3::splat(5.0),
        64,
        0.0,
    );
    assert!(solid > 500, "only {solid} solid points");
}

#[test]
fn rotated_round_leaves_get_their_exact_box() {
    let mut compared = 0;
    for deg in [0.0f32, 20.0, 45.0, 70.0, 90.0] {
        let t = deg.to_radians();
        let (s, c) = (t.sin().abs(), t.cos().abs());
        let q = Quat::from_rotation_x(t);
        // a sphere is rotation invariant
        let b = bounded(&SdfNode::sphere(2.0).rotate(q));
        assert!(near(b.max, Vec3::splat(2.0), 1e-4), "{deg}: {b:?}");
        // cylinder r, h along Y turned about X: axis (0, cos, sin)
        let (r, h) = (1.5f32, 4.0f32);
        let b = bounded(
            &SdfNode::Cylinder {
                radius: r,
                half_height: h,
            }
            .rotate(q),
        );
        let expect = Vec3::new(r, r * s + h * c, r * c + h * s);
        assert!(near(b.max, expect, 1e-4), "{deg}: {b:?} vs {expect:?}");
        // torus R, r (ring in XZ) turned about X
        let (rr, rm) = (3.0f32, 0.5f32);
        let b = bounded(&SdfNode::torus(rr, rm).rotate(q));
        let expect = Vec3::new(rr + rm, rr * s + rm, rr * c + rm);
        assert!(near(b.max, expect, 1e-4), "{deg}: {b:?} vs {expect:?}");
        // capsule along X from -2 to 2, radius 0.5, turned about Z
        let qz = Quat::from_rotation_z(t);
        let b = bounded(
            &SdfNode::Capsule {
                point_a: Vec3::new(-2.0, 0.0, 0.0),
                point_b: Vec3::new(2.0, 0.0, 0.0),
                radius: 0.5,
            }
            .rotate(qz),
        );
        let expect = Vec3::new(2.0 * c + 0.5, 2.0 * s + 0.5, 0.5);
        assert!(near(b.max, expect, 1e-4), "{deg}: {b:?} vs {expect:?}");
        compared += 4;
    }
    assert_eq!(compared, 20);
}

#[test]
fn the_other_added_nodes_have_closed_form_boxes() {
    let mut compared = 0;
    // a unit sphere at x = 3 mirrored in X covers x in [-4, 4]
    let b = bounded(
        &SdfNode::sphere(1.0)
            .translate(3.0, 0.5, 0.0)
            .mirror(true, false, false),
    );
    assert!(near(b.min, Vec3::new(-4.0, -0.5, -1.0), 1e-4), "{b:?}");
    assert!(near(b.max, Vec3::new(4.0, 1.5, 1.0), 1e-4), "{b:?}");
    compared += 1;
    // a sphere that lies entirely at x < 0 is never reached by |x|: empty
    let gone = SdfNode::sphere(1.0)
        .translate(-3.0, 0.0, 0.0)
        .mirror(true, false, false);
    assert!(matches!(analytic_aabb(&gone), AnalyticAabb::Empty));
    let probe = (-50..=50)
        .flat_map(|i| (-50..=50).map(move |j| Vec3::new(i as f32 * 0.1, j as f32 * 0.1, 0.0)))
        .filter(|p| eval(&gone, *p) <= 0.0)
        .count();
    assert_eq!(probe, 0, "the mirrored set is not empty");
    compared += 1;
    // non-uniform scale (2, 3, 4) of a unit sphere: half (2, 3, 4)
    let b = bounded(&SdfNode::sphere(1.0).scale_xyz(2.0, 3.0, 4.0));
    assert!(near(b.max, Vec3::new(2.0, 3.0, 4.0), 1e-4), "{b:?}");
    compared += 1;
    // rounded box: inner half (1, 2, 3) rounded by 0.5 spans (1.5, 2.5, 3.5)
    let b = bounded(&SdfNode::RoundedBox {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
        round_radius: 0.5,
    });
    assert!(near(b.max, Vec3::new(1.5, 2.5, 3.5), 1e-4), "{b:?}");
    compared += 1;
    // rounded cylinder radius 2, round 0.25, half height 1: (2, 1.25, 2)
    let b = bounded(&SdfNode::RoundedCylinder {
        radius: 2.0,
        round_radius: 0.25,
        half_height: 1.0,
    });
    assert!(near(b.max, Vec3::new(2.0, 1.25, 2.0), 1e-4), "{b:?}");
    compared += 1;
    // smooth max is never below max: the smooth intersection lies in the plain overlap and the
    // smooth subtraction in the minuend; a material tag passes the child through
    let a = SdfNode::box3d(4.0, 4.0, 4.0);
    let b = bounded(
        &a.clone()
            .smooth_intersection(SdfNode::box3d(4.0, 4.0, 4.0).translate(3.0, 0.0, 0.0), 0.8),
    );
    assert!(b.min.x >= 1.0 - 1e-4 && b.max.x <= 2.0 + 1e-4, "{b:?}");
    compared += 1;
    let b = bounded(&a.smooth_subtract(SdfNode::sphere(1.0), 0.8));
    assert!(near(b.max, Vec3::splat(2.0), 1e-4), "{b:?}");
    compared += 1;
    let b = bounded(&SdfNode::WithMaterial {
        child: Arc::new(SdfNode::sphere(1.5).translate(1.0, 0.0, 0.0)),
        material_id: 3,
    });
    assert!(near(b.max, Vec3::new(2.5, 1.5, 1.5), 1e-4), "{b:?}");
    compared += 1;
    // negative / non-finite elongation and a non-positive non-uniform scale make no statement
    for node in [
        SdfNode::sphere(1.0).elongate(-1.0, 0.0, 0.0),
        SdfNode::sphere(1.0).elongate(f32::NAN, 0.0, 0.0),
        SdfNode::sphere(1.0).scale_xyz(1.0, 0.0, 1.0),
        SdfNode::sphere(1.0).scale_xyz(1.0, -2.0, 1.0),
    ] {
        assert!(matches!(analytic_aabb(&node), AnalyticAabb::Unsupported));
        compared += 1;
    }
    assert_eq!(compared, 12);
}

/// Deterministic xorshift for the random trees
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    fn vec(&mut self, lo: f32, hi: f32) -> Vec3 {
        Vec3::new(self.range(lo, hi), self.range(lo, hi), self.range(lo, hi))
    }
    fn quat(&mut self) -> Quat {
        Quat::from_euler(
            glam::EulerRot::XYZ,
            self.range(-3.2, 3.2),
            self.range(-3.2, 3.2),
            self.range(-3.2, 3.2),
        )
    }
}

fn random_tree(rng: &mut Rng, depth: u32) -> SdfNode {
    if depth == 0 {
        return match (rng.next() * 7.0) as u32 {
            0 => SdfNode::sphere(rng.range(0.3, 1.5)),
            1 => SdfNode::Box3d {
                half_extents: rng.vec(0.2, 1.5),
            },
            2 => SdfNode::Cylinder {
                radius: rng.range(0.3, 1.2),
                half_height: rng.range(0.3, 1.5),
            },
            3 => SdfNode::torus(rng.range(0.8, 1.8), rng.range(0.2, 0.6)),
            4 => SdfNode::Capsule {
                point_a: rng.vec(-1.0, 1.0),
                point_b: rng.vec(-1.0, 1.0),
                radius: rng.range(0.2, 0.6),
            },
            5 => SdfNode::RoundedBox {
                half_extents: rng.vec(0.2, 1.2),
                round_radius: rng.range(0.05, 0.4),
            },
            _ => SdfNode::RoundedCylinder {
                radius: rng.range(0.4, 1.2),
                round_radius: rng.range(0.05, 0.3),
                half_height: rng.range(0.3, 1.2),
            },
        };
    }
    let sub = |rng: &mut Rng| random_tree(rng, depth - 1);
    match (rng.next() * 12.0) as u32 {
        0 => {
            let a = sub(rng);
            a.union(sub(rng))
        }
        1 => {
            let a = sub(rng);
            a.subtract(sub(rng))
        }
        2 | 3 => {
            let a = rng.vec(0.0, 1.5);
            sub(rng).elongate(a.x, a.y, a.z)
        }
        4 => {
            let q = rng.quat();
            sub(rng).rotate(q)
        }
        5 => {
            let t = rng.vec(-2.0, 2.0);
            sub(rng).translate(t.x, t.y, t.z)
        }
        6 => {
            let (x, y, z) = (rng.next() < 0.5, rng.next() < 0.5, rng.next() < 0.5);
            sub(rng).mirror(x, y, z)
        }
        7 => {
            let s = rng.vec(0.5, 2.0);
            sub(rng).scale_xyz(s.x, s.y, s.z)
        }
        8 => {
            let r = rng.range(0.05, 0.5);
            sub(rng).round(r)
        }
        9 => {
            let (a, k) = (sub(rng), rng.range(0.1, 1.5));
            a.smooth_intersection(sub(rng), k)
        }
        10 => {
            let (a, k) = (sub(rng), rng.range(0.1, 1.5));
            a.smooth_subtract(sub(rng), k)
        }
        _ => SdfNode::WithMaterial {
            child: Arc::new(sub(rng)),
            material_id: 1,
        },
    }
}

#[test]
fn the_bound_contains_every_solid_point_of_random_elongate_trees() {
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let (mut trees, mut solid) = (0, 0usize);
    for _ in 0..300 {
        let tree = random_tree(&mut rng, 3);
        let region = analytic_aabb(&tree);
        assert!(
            !matches!(region, AnalyticAabb::Unsupported),
            "every node of the generator is covered: {tree:?}"
        );
        trees += 1;
        for _ in 0..4000 {
            let p = rng.vec(-7.0, 7.0);
            if eval(&tree, p) > 0.0 {
                continue;
            }
            solid += 1;
            match region {
                AnalyticAabb::Bounded(b) => assert!(
                    p.cmpge(b.min).all() && p.cmple(b.max).all(),
                    "solid {p:?} outside {b:?} for {tree:?}"
                ),
                _ => panic!("solid {p:?} in a tree reported {region:?}: {tree:?}"),
            }
        }
    }
    assert_eq!(trees, 300);
    assert!(solid > 5_000, "only {solid} solid points were hit");
}

#[test]
fn the_interval_enclosure_of_an_elongate_contains_every_point_value() {
    // the interval path widens `x - clamp(x)` (the two uses of x are not correlated) but must
    // never drop a value: fixed here so the fallback stays sound
    let mut rng = Rng(0x94D0_49BB_1331_11EB);
    let mut checked = 0usize;
    for _ in 0..200 {
        let tree = random_tree(&mut rng, 2);
        let c = rng.vec(-4.0, 4.0);
        let h = rng.vec(0.05, 3.0);
        let iv = eval_interval(&tree, Vec3Interval::from_bounds(c - h, c + h));
        for _ in 0..200 {
            let u = rng.vec(-1.0, 1.0);
            let p = c + h * u;
            let v = eval(&tree, p);
            assert!(
                v >= iv.lo - 1e-4 && v <= iv.hi + 1e-4,
                "f({p:?}) = {v} outside [{}, {}] for {tree:?}",
                iv.lo,
                iv.hi
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 40_000);
}

#[test]
fn the_interval_search_alone_still_contains_an_elongated_strip() {
    // a zero twist keeps the shape but is not covered by the level-set bound, so the result
    // is the interval search alone: wider than the strip under rotation, never smaller
    let (a, b) = (Vec2::new(0.0, 0.0), Vec2::new(20.0, 20.0));
    let node = strip(a, b, 2.75, 5.0).twist(0.0);
    assert!(matches!(analytic_aabb(&node), AnalyticAabb::Unsupported));
    let cfg = TightAabbConfig::try_new(60.0, 24, 16).unwrap();
    let aabb = compute_tight_aabb_with_config(&node, &cfg);
    let (lo, hi) = polyline_box(&[a, b], 2.75, 5.0);
    assert!(
        aabb.min.cmple(lo + Vec3::splat(0.05)).all()
            && aabb.max.cmpge(hi - Vec3::splat(0.05)).all(),
        "{aabb:?} cuts the strip {lo:?}..{hi:?}"
    );
    let solid = assert_grid_inside(
        &node,
        &aabb,
        lo - Vec3::splat(3.0),
        hi + Vec3::splat(3.0),
        48,
        0.0,
    );
    assert!(solid > 200, "only {solid} solid points");
}

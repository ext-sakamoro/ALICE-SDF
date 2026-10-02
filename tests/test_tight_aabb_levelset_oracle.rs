//! Oracle for the level-set AABB bound (`tight_aabb::analytic_aabb`)
//!
//! The interval search behind `compute_tight_aabb` drops the correlation between the axes of
//! a rotated point, so a thin rotated plate came back about twice as large as it is. The
//! level-set bound pushes the accumulated transform down to the leaf and applies it once.
//!
//! Independent references:
//! (a) closed form: a box `half` rotated by `R` is bounded exactly by `|R| * half`
//! (b) composition: three nested rotations equal the single product rotation
//! (c) soundness: every point of the solid `{f <= 0}` lies in the box (random trees)
//! (d) the combined result is never looser than the bound and still contains the solid
//! (e) level algebra: offset, onion, smooth union and uniform scale shift the level

use alice_sdf::eval;
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{
    analytic_aabb, compute_tight_aabb_with_config, AnalyticAabb, TightAabbConfig,
};
use glam::{Mat3, Quat, Vec3};

fn bounded(node: &SdfNode) -> Aabb {
    match analytic_aabb(node) {
        AnalyticAabb::Bounded(b) => b,
        other => panic!("expected a bound, got {other:?}"),
    }
}

fn big() -> TightAabbConfig {
    TightAabbConfig::try_new(40.0, 40, 32).unwrap()
}

/// `|R| * half` — the box that bounds a rotated box
fn rotated_half(q: Quat, half: Vec3) -> Vec3 {
    let m = Mat3::from_quat(q);
    Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs()) * half
}

fn near(a: Vec3, b: Vec3, tol: f32) -> bool {
    (a - b).abs().max_element() <= tol
}

#[test]
fn a_rotated_box_is_bounded_by_the_closed_form() {
    let half = Vec3::new(15.0, 15.0, 0.4);
    for deg in [0.0f32, 15.0, 30.0, 45.0, 60.0, 90.0] {
        let q = Quat::from_rotation_x(deg.to_radians());
        let node = SdfNode::Box3d { half_extents: half }.rotate(q);
        let b = bounded(&node);
        let expect = rotated_half(q, half);
        assert!(
            near((b.max - b.min) * 0.5, expect, 1e-3),
            "{deg} deg: {b:?} vs {expect:?}"
        );
        assert!(near((b.max + b.min) * 0.5, Vec3::ZERO, 1e-3));
    }
}

#[test]
fn nested_rotations_do_not_accumulate() {
    let half = Vec3::new(15.0, 6.0, 0.4);
    let (qx, qy, qz) = (
        Quat::from_rotation_x(0.5),
        Quat::from_rotation_y(0.7),
        Quat::from_rotation_z(1.1),
    );
    // rotate(qz, rotate(qy, rotate(qx, box))) maps a point by qz * qy * qx
    let node = SdfNode::Box3d { half_extents: half }
        .rotate(qx)
        .rotate(qy)
        .rotate(qz);
    let b = bounded(&node);
    let expect = rotated_half(qz * qy * qx, half);
    assert!(
        near((b.max - b.min) * 0.5, expect, 1e-3),
        "{b:?} vs {expect:?}"
    );
}

#[test]
fn a_translated_scaled_rotated_leaf_is_exact() {
    let half = Vec3::new(2.0, 1.0, 0.5);
    let q = Quat::from_rotation_z(0.6);
    // scale(3, translate(1,2,3, rotate(box)))
    let node = SdfNode::Box3d { half_extents: half }
        .rotate(q)
        .translate(1.0, 2.0, 3.0)
        .scale(3.0);
    let b = bounded(&node);
    assert!(near(
        (b.max - b.min) * 0.5,
        rotated_half(q, half) * 3.0,
        1e-3
    ));
    assert!(near((b.max + b.min) * 0.5, Vec3::new(3.0, 6.0, 9.0), 1e-3));
}

#[test]
fn the_search_result_for_a_thin_rotated_plate_is_no_longer_inflated() {
    // the plate that lost 73% of its volume in text-to-print: 30 x 30 x 0.8 turned 30 deg about X
    let half = Vec3::new(15.0, 15.0, 0.4);
    let q = Quat::from_rotation_x(30.0f32.to_radians());
    let node = SdfNode::Box3d { half_extents: half }.rotate(q);
    let aabb = compute_tight_aabb_with_config(&node, &big());
    let exact = rotated_half(q, half);
    let got = (aabb.max - aabb.min) * 0.5;
    assert!(
        got.x <= exact.x * 1.01 + 0.05
            && got.y <= exact.y * 1.01 + 0.05
            && got.z <= exact.z * 1.01 + 0.05,
        "inflated: {got:?} vs exact {exact:?}"
    );
    // and it still contains the plate
    assert!(got.x >= exact.x - 0.05 && got.y >= exact.y - 0.05 && got.z >= exact.z - 0.05);
}

#[test]
fn the_level_shifts_of_the_offset_family_are_exact() {
    // round: f = child - r
    let b = bounded(&SdfNode::box3d(2.0, 2.0, 2.0).round(0.5));
    assert!(near(b.max, Vec3::splat(1.5), 1e-3), "{b:?}");
    // onion: |child| - t <= 0
    let b = bounded(&SdfNode::sphere(2.0).onion(0.25));
    assert!(near(b.max, Vec3::splat(2.25), 1e-3), "{b:?}");
    // smooth union loosens the level by k / 4
    let sm = SdfNode::sphere(1.0).smooth_union(SdfNode::sphere(1.0).translate(4.0, 0.0, 0.0), 2.0);
    let b = bounded(&sm);
    assert!(b.max.x >= 5.0 - 1e-3 && b.max.x <= 5.5 + 1e-3, "{b:?}");
    // uniform scale divides the level it passes down
    let b = bounded(&SdfNode::sphere(1.0).round(0.5).scale(2.0));
    assert!(near(b.max, Vec3::splat(3.0), 1e-3), "{b:?}");
}

#[test]
fn subtraction_bounds_by_the_minuend_and_intersection_by_the_overlap() {
    let a = SdfNode::box3d(4.0, 4.0, 4.0);
    let cutter = SdfNode::box3d(100.0, 100.0, 100.0);
    let b = bounded(&a.clone().subtract(cutter));
    assert!(near(b.max, Vec3::splat(2.0), 1e-3), "{b:?}");
    let overlap = bounded(
        &a.clone()
            .intersection(SdfNode::box3d(4.0, 4.0, 4.0).translate(3.0, 0.0, 0.0)),
    );
    assert!(
        overlap.min.x >= 1.0 - 1e-3 && overlap.max.x <= 2.0 + 1e-3,
        "{overlap:?}"
    );
    // disjoint boxes: the intersection is empty
    let none = a.intersection(SdfNode::box3d(1.0, 1.0, 1.0).translate(50.0, 0.0, 0.0));
    assert!(matches!(analytic_aabb(&none), AnalyticAabb::Empty));
    assert!(matches!(
        analytic_aabb(&SdfNode::sphere(-1.0)),
        AnalyticAabb::Empty
    ));
}

#[test]
fn unsupported_nodes_make_no_statement() {
    let twisted = SdfNode::box3d(2.0, 2.0, 2.0).twist(1.0);
    assert!(matches!(analytic_aabb(&twisted), AnalyticAabb::Unsupported));
    let mixed = SdfNode::sphere(1.0).union(twisted);
    assert!(matches!(analytic_aabb(&mixed), AnalyticAabb::Unsupported));
    // a non-positive or non-finite scale is not folded
    assert!(matches!(
        analytic_aabb(&SdfNode::sphere(1.0).scale(0.0)),
        AnalyticAabb::Unsupported
    ));
    assert!(matches!(
        analytic_aabb(&SdfNode::sphere(1.0).scale(f32::NAN)),
        AnalyticAabb::Unsupported
    ));
    // the search still answers for them (the bound only ever tightens)
    let aabb = compute_tight_aabb_with_config(&SdfNode::sphere(1.0).scale(0.0), &big());
    assert!(aabb.min.x.is_finite() && aabb.max.x.is_finite());
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
}

fn random_tree(rng: &mut Rng, depth: u32) -> SdfNode {
    let leaf = |rng: &mut Rng| match (rng.next() * 5.0) as u32 {
        0 => SdfNode::sphere(rng.range(0.3, 2.0)),
        1 => SdfNode::Box3d {
            half_extents: Vec3::new(
                rng.range(0.2, 2.0),
                rng.range(0.2, 2.0),
                rng.range(0.2, 2.0),
            ),
        },
        2 => SdfNode::Cylinder {
            radius: rng.range(0.3, 1.5),
            half_height: rng.range(0.3, 2.0),
        },
        3 => SdfNode::Torus {
            major_radius: rng.range(1.0, 2.5),
            minor_radius: rng.range(0.2, 0.8),
        },
        _ => SdfNode::Capsule {
            point_a: Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ),
            point_b: Vec3::new(
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
                rng.range(-1.0, 1.0),
            ),
            radius: rng.range(0.2, 0.8),
        },
    };
    if depth == 0 {
        return leaf(rng);
    }
    match (rng.next() * 9.0) as u32 {
        0 => random_tree(rng, depth - 1).union(random_tree(rng, depth - 1)),
        1 => random_tree(rng, depth - 1).intersection(random_tree(rng, depth - 1)),
        2 => random_tree(rng, depth - 1).subtract(random_tree(rng, depth - 1)),
        3 => random_tree(rng, depth - 1)
            .smooth_union(random_tree(rng, depth - 1), rng.range(0.1, 2.0)),
        4 => random_tree(rng, depth - 1).translate(
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
        ),
        5 => random_tree(rng, depth - 1).rotate(Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
            rng.range(-3.0, 3.0),
        )),
        6 => random_tree(rng, depth - 1).scale(rng.range(0.5, 2.5)),
        7 => random_tree(rng, depth - 1).round(rng.range(0.05, 0.6)),
        _ => random_tree(rng, depth - 1).onion(rng.range(0.05, 0.5)),
    }
}

#[test]
fn the_bound_contains_every_solid_point_of_random_trees() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut checked_trees = 0;
    let mut solid_points = 0usize;
    for _ in 0..400 {
        let tree = random_tree(&mut rng, 3);
        let region = analytic_aabb(&tree);
        if matches!(region, AnalyticAabb::Unsupported) {
            continue;
        }
        checked_trees += 1;
        for _ in 0..4000 {
            let p = Vec3::new(
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
                rng.range(-8.0, 8.0),
            );
            if eval(&tree, p) > 0.0 {
                continue;
            }
            solid_points += 1;
            match region {
                AnalyticAabb::Empty => {
                    panic!("solid point {p:?} in a tree reported empty: {tree:?}")
                }
                AnalyticAabb::Bounded(b) => assert!(
                    p.cmpge(b.min).all() && p.cmple(b.max).all(),
                    "solid point {p:?} outside {b:?} for {tree:?}"
                ),
                AnalyticAabb::Unsupported => unreachable!(),
            }
        }
    }
    assert!(checked_trees > 300, "only {checked_trees} supported trees");
    assert!(
        solid_points > 5_000,
        "only {solid_points} solid points were hit"
    );
}

#[test]
fn the_combined_result_contains_the_solid_and_is_never_looser_than_the_bound() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let cfg = big();
    for _ in 0..60 {
        let tree = random_tree(&mut rng, 2);
        let AnalyticAabb::Bounded(analytic) = analytic_aabb(&tree) else {
            continue;
        };
        let aabb = compute_tight_aabb_with_config(&tree, &cfg);
        assert!(
            aabb.min.cmpge(analytic.min - Vec3::splat(1e-3)).all(),
            "{aabb:?} vs {analytic:?}"
        );
        assert!(
            aabb.max.cmple(analytic.max + Vec3::splat(1e-3)).all(),
            "{aabb:?} vs {analytic:?}"
        );
        for _ in 0..400 {
            let p = Vec3::new(
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
                rng.range(-10.0, 10.0),
            );
            if eval(&tree, p) <= 0.0 {
                let slack = Vec3::splat(0.05);
                assert!(
                    p.cmpge(aabb.min - slack).all() && p.cmple(aabb.max + slack).all(),
                    "solid {p:?} outside the combined {aabb:?} for {tree:?}"
                );
            }
        }
    }
}

#[test]
fn a_smooth_union_solid_that_pokes_out_of_the_plain_union_stays_inside_the_bound() {
    // two unit spheres 3 apart, blend width 4: the blend adds material beyond x = 4
    let tree =
        SdfNode::sphere(1.0).smooth_union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0), 4.0);
    let mut edge = 0.0f32;
    for step in 0..3000u16 {
        let x = f32::from(step).mul_add(0.001, 3.0);
        if eval(&tree, Vec3::new(x, 0.0, 0.0)) <= 0.0 {
            edge = x;
        }
    }
    assert!(
        edge > 4.02,
        "the blend should add material beyond the plain union, edge = {edge}"
    );
    let b = bounded(&tree);
    assert!(b.max.x >= edge, "bound {b:?} cuts the solid at {edge}");
    // and not by more than the blend's largest deviation k / 4
    assert!(b.max.x <= 4.0 + 1.0 + 1e-3, "{b:?}");
}

#[test]
fn the_corners_of_a_rotated_box_stay_inside_despite_rounding() {
    // the corners are on the surface (f = 0). `R * corner` and `|R| * half` round differently,
    // so an unpadded bound would cut some of them by an ulp
    let mut rng = Rng(0xA076_1D64_78BD_642F);
    for _ in 0..2000 {
        let half = Vec3::new(
            rng.range(0.1, 20.0),
            rng.range(0.1, 20.0),
            rng.range(0.1, 20.0),
        );
        let q = Quat::from_euler(
            glam::EulerRot::XYZ,
            rng.range(-3.2, 3.2),
            rng.range(-3.2, 3.2),
            rng.range(-3.2, 3.2),
        );
        let t = Vec3::new(
            rng.range(-9.0, 9.0),
            rng.range(-9.0, 9.0),
            rng.range(-9.0, 9.0),
        );
        let node = SdfNode::Box3d { half_extents: half }
            .rotate(q)
            .translate(t.x, t.y, t.z);
        let b = bounded(&node);
        for sx in [-1.0f32, 1.0] {
            for sy in [-1.0f32, 1.0] {
                for sz in [-1.0f32, 1.0] {
                    let corner = q * (half * Vec3::new(sx, sy, sz)) + t;
                    assert!(
                        corner.cmpge(b.min).all() && corner.cmple(b.max).all(),
                        "corner {corner:?} outside {b:?}"
                    );
                }
            }
        }
    }
}

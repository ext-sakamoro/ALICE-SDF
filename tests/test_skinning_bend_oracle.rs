//! `SdfSkinning` against what a bent joint is, in closed form.
//!
//! The part is a capsule along x with half length 3 and radius 0.2, the joint at the origin. The
//! bone of the x > 0 half turns by 90 degrees about z, the x < 0 half stays. Away from the joint
//! each half is rigid, so the posed field is known exactly at a point on or beside an arm.
//!
//! The last test holds the bound of the proposed form, an angle that varies around the joint's
//! axis, on a prototype in this file: `‖J‖₂ ≤ max(1 + 1.5 θ / Δ_in, |1.5 θ / Δ_out − 1|)`.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::eval::eval;
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::transforms::skinning::BoneTransform;
use alice_sdf::types::SdfNode;
use glam::{DMat3, DVec3, Mat4, Vec3};
use std::f64::consts::{PI, TAU};

const HALF: f32 = 3.0;
const RADIUS: f32 = 0.2;

fn bone(inv_bind: Mat4, current: Mat4, weight: f32) -> BoneTransform {
    BoneTransform {
        inv_bind_pose: inv_bind.to_cols_array(),
        current_pose: current.to_cols_array(),
        weight,
    }
}

fn arm() -> SdfNode {
    SdfNode::capsule(
        Vec3::new(-HALF, 0.0, 0.0),
        Vec3::new(HALF, 0.0, 0.0),
        RADIUS,
    )
}

/// Point and exact value of the bent arm where it is rigid: on an axis and on a surface
fn bent_arm_values() -> [(Vec3, f32); 4] {
    [
        (Vec3::new(-1.5, 0.0, 0.0), -RADIUS),
        (Vec3::new(0.0, 1.5, 0.0), -RADIUS),
        (Vec3::new(-1.5, RADIUS, 0.0), 0.0),
        (Vec3::new(RADIUS, 1.5, 0.0), 0.0),
    ]
}

/// A bone's fields say bone → world and world → bone at bind, so a part follows its bone: the
/// forward map is `current · inv_bind` and the posed field reads the rest pose through its inverse.
#[test]
fn a_part_moves_the_way_its_bone_moves() {
    let moved = Vec3::new(2.0, 0.0, 0.0);
    let at_origin = SdfNode::sphere(1.0).sdf_skinning(vec![bone(
        Mat4::IDENTITY,
        Mat4::from_translation(moved),
        1.0,
    )]);
    assert!(
        (eval(&at_origin, moved) + 1.0).abs() < 1e-5,
        "a bone moved by +2: the centre reads {}, expected -1",
        eval(&at_origin, moved)
    );

    let bind = Vec3::new(1.0, 0.0, 0.0);
    let bound_away = SdfNode::sphere(1.0)
        .translate(1.0, 0.0, 0.0)
        .sdf_skinning(vec![bone(
            Mat4::from_translation(-bind),
            Mat4::from_translation(bind + moved),
            1.0,
        )]);
    assert!(
        (eval(&bound_away, bind + moved) + 1.0).abs() < 1e-5,
        "a bone bound at +1 moved by +2: the centre reads {}, expected -1",
        eval(&bound_away, bind + moved)
    );
}

/// Whatever the two weights, and whichever way a bone's matrix is read, one affine map cannot hold
/// both arms: the search reports the best any constant weight reaches.
#[test]
fn a_bend_is_rigid_away_from_the_joint() {
    let turn = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
    let mut best = (f32::INFINITY, 0.0, false);
    for step in 0..=20 {
        let w = step as f32 / 20.0;
        for inverse in [false, true] {
            let current = if inverse { turn.inverse() } else { turn };
            let node = arm().sdf_skinning(vec![
                bone(Mat4::IDENTITY, Mat4::IDENTITY, 1.0 - w),
                bone(Mat4::IDENTITY, current, w),
            ]);
            let error = bent_arm_values()
                .iter()
                .map(|(p, want)| (eval(&node, *p) - want).abs())
                .fold(0.0_f32, f32::max);
            if error < best.0 {
                best = (error, w, inverse);
            }
        }
    }
    assert!(
        best.0 < 1e-4,
        "the bent arm's values: best error {} at constant weight {} (inverse read: {})",
        best.0,
        best.1,
        best.2
    );
}

/// A rigid pose keeps distances, so the field's slope is the child's. The bound of a turned sphere
/// is 1; a Frobenius norm says sqrt(3), and a tracer divides by it.
#[test]
fn a_rigid_pose_has_lipschitz_one() {
    let node = SdfNode::sphere(1.0).sdf_skinning(vec![bone(
        Mat4::IDENTITY,
        Mat4::from_rotation_z(1.0),
        1.0,
    )]);
    let bound = eval_lipschitz(&node);
    assert!((bound - 1.0).abs() < 1e-5, "bound {bound}");
}

fn smooth(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// The turn's share at angle `phi`: 1 on the moved arm's side, 0 on the still arm's side, changing
/// over `inner` around the inner bisector and `outer` around the outer one.
fn share(phi: f64, theta: f64, inner: f64, outer: f64) -> f64 {
    let inner_at = (theta + PI) / 2.0;
    let outer_at = inner_at - PI;
    let rel = (phi - outer_at).rem_euclid(TAU);
    let arc = (inner_at - outer_at).rem_euclid(TAU);
    if rel < outer / 2.0 {
        smooth(rel / outer + 0.5)
    } else if rel > TAU - outer / 2.0 {
        smooth((rel - TAU) / outer + 0.5)
    } else if (rel - arc).abs() < inner / 2.0 {
        1.0 - smooth((rel - arc) / inner + 0.5)
    } else if rel < arc {
        1.0
    } else {
        0.0
    }
}

/// Posed point to rest point, turning by `theta · share` about z
fn by_angle(p: DVec3, theta: f64, inner: f64, outer: f64) -> DVec3 {
    let turn = -theta * share(p.y.atan2(p.x), theta, inner, outer);
    let (s, c) = turn.sin_cos();
    DVec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z)
}

/// Posed point to rest point, blending the two matrices (linear blend skinning)
fn by_matrix(p: DVec3, theta: f64, inner: f64, outer: f64) -> DVec3 {
    let w = share(p.y.atan2(p.x), theta, inner, outer);
    let (s, c) = (-theta).sin_cos();
    p * (1.0 - w) + DVec3::new(c * p.x - s * p.y, s * p.x + c * p.y, p.z) * w
}

fn jacobian(map: &dyn Fn(DVec3) -> DVec3, p: DVec3) -> DMat3 {
    const H: f64 = 1e-6;
    let d = |e: DVec3| (map(p + e * H) - map(p - e * H)) / (2.0 * H);
    DMat3::from_cols(d(DVec3::X), d(DVec3::Y), d(DVec3::Z))
}

fn largest_singular(j: DMat3) -> f64 {
    let m = j.transpose() * j;
    let mut v = DVec3::new(0.6, 0.7, 0.4).normalize();
    for _ in 0..80 {
        v = (m * v).normalize_or_zero();
    }
    (m * v).length().sqrt()
}

/// (largest singular value, smallest determinant) over points that miss the bands' edges and axis
fn survey(map: &dyn Fn(DVec3) -> DVec3) -> (f64, f64) {
    let mut seed = 0x2545_f491_u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 33) as f64 / (1u64 << 31) as f64) * 6.0 - 3.0
    };
    let (mut most, mut det) = (0.0_f64, f64::INFINITY);
    for _ in 0..200_000 {
        let p = DVec3::new(next(), next(), next());
        if p.x.hypot(p.y) < 1e-2 {
            continue;
        }
        let j = jacobian(map, p);
        most = most.max(largest_singular(j));
        det = det.min(j.determinant());
    }
    (most, det)
}

/// The proposed form: the bound is reached, holds at every radius, nothing folds where
/// `Δ_out > 1.5 θ`, and the matrix blend is printed beside it for the design's comparison.
#[test]
fn the_proposed_bound_holds_on_the_prototype() {
    for (theta_deg, inner_deg) in [
        (30.0, 60.0),
        (60.0, 60.0),
        (90.0, 45.0),
        (90.0, 90.0),
        (120.0, 45.0),
    ] {
        let theta = f64::to_radians(theta_deg);
        let inner = f64::to_radians(inner_deg);
        let outer = 1.8 * theta;
        let bound = (1.0 + 1.5 * theta / inner).max((1.5 * theta / outer - 1.0).abs());
        let (angle, angle_det) = survey(&|p| by_angle(p, theta, inner, outer));
        let (matrix, matrix_det) = survey(&|p| by_matrix(p, theta, inner, outer));
        println!(
            "theta {theta_deg} inner {inner_deg}: angle max {angle:.3} bound {bound:.3} min det \
             {angle_det:.3}; matrix max {matrix:.3} min det {matrix_det:.3}"
        );
        assert!(angle <= bound * (1.0 + 1e-3), "{angle} over {bound}");
        assert!(
            angle > bound * 0.97,
            "the bound is loose: {angle} of {bound}"
        );
        assert!(angle_det > 0.0, "the map folds: {angle_det}");
    }
}

/// The prototype passes the same closed-form values the node must, and keeps an arm's thickness at
/// the bend where the matrix blend swells it.
#[test]
fn the_prototype_gives_the_bent_arm_values() {
    let (theta, inner) = (f64::to_radians(90.0), f64::to_radians(45.0));
    let outer = 1.8 * theta;
    let field = |q: DVec3| {
        let x = q.x.clamp(-f64::from(HALF), f64::from(HALF));
        DVec3::new(q.x - x, q.y, q.z).length() - f64::from(RADIUS)
    };
    for (p, want) in bent_arm_values() {
        let got = field(by_angle(p.as_dvec3(), theta, inner, outer));
        assert!((got - f64::from(want)).abs() < 1e-6, "{p}: {got} vs {want}");
    }
    let along_outer = |map: &dyn Fn(DVec3) -> DVec3| -> f64 {
        let dir = DVec3::new(1.0, -1.0, 0.0).normalize();
        (1..4000)
            .map(|i| f64::from(i) * 5e-4)
            .min_by(|a, b| {
                field(map(dir * *a))
                    .abs()
                    .total_cmp(&field(map(dir * *b)).abs())
            })
            .unwrap_or(0.0)
    };
    let angle_at = along_outer(&|p| by_angle(p, theta, inner, outer));
    let matrix_at = along_outer(&|p| by_matrix(p, theta, inner, outer));
    println!("surface on the outer bisector: angle {angle_at:.4}, matrix {matrix_at:.4}");
    assert!((angle_at - f64::from(RADIUS)).abs() < 1e-3, "{angle_at}");
    assert!(
        (matrix_at - f64::from(RADIUS) * 2.0_f64.sqrt()).abs() < 1e-3,
        "the matrix blend swells by 1 / cos(theta / 2): {matrix_at}"
    );
}

//! `SdfSkinning` against what a bent joint is, in closed form.
//!
//! The part is a capsule along x with half length 3 and radius 0.2, the joint at the origin. The
//! bone of the x > 0 half turns by 90 degrees about z, the x < 0 half stays. Away from the joint
//! each half is rigid, so the posed field is known exactly at a point on or beside an arm.
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
use glam::{Mat4, Vec3};

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
#[ignore = "known defect: sdf-skinning-constant-weights: the weight is the same at every point, so the node is one affine map"]
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

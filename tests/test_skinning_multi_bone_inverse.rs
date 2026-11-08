//! `SdfSkinning` with several bones in different poses maps a posed point back to its rest point.
//!
//! oracle: the forward linear blend skinning of the node, posed = Σ wᵢ (currentᵢ · inv_bindᵢ) rest
//! / Σ wᵢ (the weights are one per bone over the whole node, so the blend is one affine map A), built
//! in f64 with glam's `DMat4` outside the crate. The rest point of a posed point is `A⁻¹ posed`.
//! Today the node averages the per-bone inverses (Σ wᵢ Mᵢ⁻¹ / Σ wᵢ), which is that inverse only when
//! the bones share a pose: red on purpose until the exact inverse is in.
//!
//! The allowance per point is derived, not chosen: the posed point is rounded to f32 (half an ulp
//! of each coordinate), which moves its exact rest point by at most ‖A⁻¹‖₂ · √3 · ½ulp(|posed|∞);
//! the result is rounded once more (½ulp(|rest|∞) per coordinate, √3 of it in length); and the
//! inverse itself is formed in f64 and rounded to f32 (each of the 12 entries within ½ulp of its
//! magnitude, applied to |posed|: 12 · ½ulp(max |entry|) · |posed|∞). ‖A⁻¹‖₂ is bounded by its
//! Frobenius norm.
//!
//! Author: Moroya Sakamoto

use alice_sdf::transforms::skinning::{sdf_skinning, BoneTransform};
use glam::{DMat4, DVec3, Vec3};

fn ulp(x: f32) -> f32 {
    let a = x.abs();
    f32::from_bits(a.to_bits() + 1) - a
}

struct Bone {
    inv_bind: DMat4,
    current: DMat4,
    weight: f64,
}

/// An elbow: the upper bone bound and posed at the origin, the lower one bound at (1, 0, 0) and
/// turned about z by `angle` at that joint; each point takes both bones with weights `w` and 1 − w
fn elbow(angle: f64, w: f64) -> Vec<Bone> {
    let joint = DVec3::new(1.0, 0.0, 0.0);
    vec![
        Bone {
            inv_bind: DMat4::IDENTITY,
            current: DMat4::IDENTITY,
            weight: w,
        },
        Bone {
            inv_bind: DMat4::from_translation(joint).inverse(),
            current: DMat4::from_translation(joint) * DMat4::from_rotation_z(angle),
            weight: 1.0 - w,
        },
    ]
}

fn to_f32(m: DMat4) -> [f32; 16] {
    m.to_cols_array().map(|v| v as f32)
}

fn bones_f32(bones: &[Bone]) -> Vec<BoneTransform> {
    bones
        .iter()
        .map(|b| BoneTransform {
            inv_bind_pose: to_f32(b.inv_bind),
            current_pose: to_f32(b.current),
            weight: b.weight as f32,
        })
        .collect()
}

/// The blended forward map A = Σ wᵢ (currentᵢ · inv_bindᵢ) / Σ wᵢ, from the f32 bones (the node's
/// data), in f64
fn blended(bones: &[BoneTransform]) -> DMat4 {
    let total: f64 = bones.iter().map(|b| f64::from(b.weight)).sum();
    bones.iter().fold(DMat4::ZERO, |acc, b| {
        let c = DMat4::from_cols_array(&b.current_pose.map(f64::from));
        let ib = DMat4::from_cols_array(&b.inv_bind_pose.map(f64::from));
        acc + c * ib * (f64::from(b.weight) / total)
    })
}

/// The allowance at a posed point (see the module doc)
fn allowance(a_inv: DMat4, posed: Vec3, rest: Vec3) -> f32 {
    let lin = DMat4::from_cols_array(&{
        let mut m = a_inv.to_cols_array();
        m[12] = 0.0;
        m[13] = 0.0;
        m[14] = 0.0;
        m
    });
    let frob = lin
        .to_cols_array()
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt() as f32;
    let entry_max = a_inv
        .to_cols_array()
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs())) as f32;
    let pm = posed.abs().max_element();
    let r3 = 3f32.sqrt();
    frob * r3 * 0.5 * ulp(pm)
        + r3 * 0.5 * ulp(rest.abs().max_element())
        + 12.0 * 0.5 * ulp(entry_max) * pm
}

fn rest_points() -> Vec<DVec3> {
    let mut out = Vec::new();
    for i in 0..9 {
        for j in -3..=3 {
            for k in -2..=2 {
                out.push(DVec3::new(
                    -0.2 + 0.3 * f64::from(i),
                    0.1 * f64::from(j),
                    0.1 * f64::from(k),
                ));
            }
        }
    }
    out
}

fn worst_over(angle: f64, w: f64) -> (f32, Vec3, Vec3) {
    let bones = bones_f32(&elbow(angle, w));
    let a = blended(&bones);
    let a_inv = a.inverse();
    let mut worst = (0.0f32, Vec3::ZERO, Vec3::ZERO);
    for r in rest_points() {
        let posed = a.transform_point3(r).as_vec3();
        // the exact rest point of the f32 posed point
        let want = a_inv.transform_point3(posed.as_dvec3()).as_vec3();
        let (got, _) = sdf_skinning(posed, &bones);
        let ratio = (got - want).length() / allowance(a_inv, posed, want);
        if ratio > worst.0 {
            worst = (ratio, posed, got - want);
        }
    }
    worst
}

/// Bones in one pose: the per-bone inverses agree, so the node is already exact (the oracle's own
/// check that the allowance is met where the method is right)
#[test]
fn bones_in_the_same_pose_map_back_exactly() {
    let (ratio, at, err) = worst_over(0.0, 0.5);
    assert!(ratio <= 1.0, "{ratio} of the allowance at {at}: {err}");
}

/// Bones in different poses: the rest point is A⁻¹ posed (red until the node inverts the blend)
#[test]
fn bones_in_different_poses_map_back_to_the_rest_point() {
    for (angle, w) in [(0.3, 0.5), (0.7, 0.5), (0.7, 0.25), (1.2, 0.6)] {
        let (ratio, at, err) = worst_over(angle, w);
        assert!(
            ratio <= 1.0,
            "angle {angle}, weight {w}: {ratio:.3e} of the allowance at {at} (error {err})"
        );
    }
}

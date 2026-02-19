//! Integration tests for new Transform variants
//!
//! Tests ProjectiveTransform, LatticeDeform, and SdfSkinning
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::transforms::skinning::BoneTransform;
use alice_sdf::types::SdfNode;
use glam::Vec3;

#[test]
fn test_projective_transform_identity() {
    // Identity projection matrix should leave sphere unchanged
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere = SdfNode::sphere(1.0);
    let transformed = sphere.projective_transform(identity, 1.0);

    // Should evaluate to same distance as original sphere
    let p = Vec3::new(2.0, 0.0, 0.0);
    let d_original = eval(&SdfNode::sphere(1.0), p);
    let d_transformed = eval(&transformed, p);

    assert!(
        (d_original - d_transformed).abs() < 1e-5,
        "Identity projection should not change distance: {} vs {}",
        d_original,
        d_transformed
    );
}

#[test]
fn test_projective_transform_lipschitz_bound() {
    // Test that Lipschitz bound is respected
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere = SdfNode::sphere(1.0);
    let lipschitz = 0.5;
    let transformed = sphere.projective_transform(identity, lipschitz);

    let p = Vec3::new(2.0, 0.0, 0.0);
    let d = eval(&transformed, p);

    // Distance should be scaled by Lipschitz bound
    assert!(d > 0.0, "Should be outside sphere");
}

#[test]
fn test_lattice_deform_identity() {
    // Control points at regular grid = identity deformation
    let mut control_points = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                control_points.push(Vec3::new(i as f32 / 3.0, j as f32 / 3.0, k as f32 / 3.0));
            }
        }
    }

    let sphere = SdfNode::sphere(0.3);
    let deformed = sphere.lattice_deform(control_points, 3, 3, 3, Vec3::ZERO, Vec3::ONE);

    // Points inside the lattice should evaluate similarly
    let p = Vec3::new(0.5, 0.5, 0.5);
    let d = eval(&deformed, p);

    // Should be close to sphere at (0.5, 0.5, 0.5) - radius 0.3 = distance
    let expected = ((0.5_f32 * 0.5 + 0.5 * 0.5 + 0.5 * 0.5).sqrt() - 0.3).abs();
    assert!(
        (d - expected).abs() < 0.2,
        "Identity lattice should preserve approximate distance: {} vs {}",
        d,
        expected
    );
}

#[test]
fn test_lattice_deform_warp() {
    // Control points with slight offset = warped space
    let mut control_points = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let x = i as f32 / 3.0;
                let y = j as f32 / 3.0;
                let z = k as f32 / 3.0;
                // Add a slight wave
                let offset = Vec3::new(0.1 * (y * 3.14).sin(), 0.0, 0.0);
                control_points.push(Vec3::new(x, y, z) + offset);
            }
        }
    }

    let sphere = SdfNode::sphere(0.3);
    let deformed = sphere.lattice_deform(control_points, 3, 3, 3, Vec3::ZERO, Vec3::ONE);

    // Should still evaluate (warped but valid)
    let p = Vec3::new(0.5, 0.5, 0.5);
    let d = eval(&deformed, p);

    // Just check it's finite
    assert!(d.is_finite(), "Warped lattice should give finite distance");
}

#[test]
fn test_sdf_skinning_identity() {
    // Identity bone transform
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let bone = BoneTransform {
        inv_bind_pose: identity,
        current_pose: identity,
        weight: 1.0,
    };

    let sphere = SdfNode::sphere(1.0);
    let skinned = sphere.sdf_skinning(vec![bone]);

    // Should evaluate to same as original sphere
    let p = Vec3::new(2.0, 0.0, 0.0);
    let d_original = eval(&SdfNode::sphere(1.0), p);
    let d_skinned = eval(&skinned, p);

    assert!(
        (d_original - d_skinned).abs() < 1e-4,
        "Identity skinning should not change distance: {} vs {}",
        d_original,
        d_skinned
    );
}

#[test]
fn test_sdf_skinning_translation() {
    // Bone with translation
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let translated = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    ];

    let bone = BoneTransform {
        inv_bind_pose: identity,
        current_pose: translated,
        weight: 1.0,
    };

    let sphere = SdfNode::sphere(1.0);
    let skinned = sphere.sdf_skinning(vec![bone]);

    // Should evaluate
    let p = Vec3::new(2.0, 0.0, 0.0);
    let d = eval(&skinned, p);

    assert!(
        d.is_finite(),
        "Translated skinning should give finite distance"
    );
}

#[test]
fn test_sdf_skinning_multi_bone() {
    // Multiple bones with weights
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let bone1 = BoneTransform {
        inv_bind_pose: identity,
        current_pose: identity,
        weight: 0.5,
    };

    let bone2 = BoneTransform {
        inv_bind_pose: identity,
        current_pose: identity,
        weight: 0.5,
    };

    let sphere = SdfNode::sphere(1.0);
    let skinned = sphere.sdf_skinning(vec![bone1, bone2]);

    // Should evaluate to same as original (both bones are identity)
    let p = Vec3::new(2.0, 0.0, 0.0);
    let d_original = eval(&SdfNode::sphere(1.0), p);
    let d_skinned = eval(&skinned, p);

    assert!(
        (d_original - d_skinned).abs() < 1e-4,
        "Multi-bone identity skinning should not change distance: {} vs {}",
        d_original,
        d_skinned
    );
}

#[test]
fn test_combined_transforms() {
    // Test combining new transforms with existing ones
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere = SdfNode::sphere(1.0)
        .translate(1.0, 0.0, 0.0)
        .projective_transform(identity, 1.0)
        .scale(2.0);

    let p = Vec3::new(0.0, 0.0, 0.0);
    let d = eval(&sphere, p);

    // Should evaluate without panicking
    assert!(
        d.is_finite(),
        "Combined transforms should give finite distance"
    );
}

#[test]
fn test_csg_with_new_transforms() {
    // Test CSG operations with new transforms
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere1 = SdfNode::sphere(1.0).projective_transform(identity, 1.0);
    let sphere2 = SdfNode::sphere(1.0).translate(0.5, 0.0, 0.0);
    let union = sphere1.union(sphere2);

    let p = Vec3::new(0.0, 0.0, 0.0);
    let d = eval(&union, p);

    // Should evaluate without panicking
    assert!(
        d.is_finite(),
        "CSG with new transforms should give finite distance"
    );
    assert!(d < 0.0, "Point should be inside at least one sphere");
}

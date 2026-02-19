//! Example demonstrating the new Transform variants
//!
//! Shows how to use ProjectiveTransform, LatticeDeform, and SdfSkinning
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::transforms::skinning::BoneTransform;
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn main() {
    println!("=== ALICE-SDF New Transform Variants Demo ===\n");

    // 1. Projective Transform Example
    println!("1. Projective Transform (Perspective Warp)");
    let identity_matrix = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere = SdfNode::sphere(1.0).projective_transform(identity_matrix, 1.0);

    let test_point = Vec3::new(2.0, 0.0, 0.0);
    let distance = eval(&sphere, test_point);
    println!(
        "   Sphere with identity projection at {:?}: distance = {}",
        test_point, distance
    );
    println!("   (Should be ~1.0 since point is 2 units from origin, radius is 1)\n");

    // 2. Lattice Deformation Example
    println!("2. Lattice Deformation (Free-Form Deformation)");

    // Create a 4x4x4 control point lattice
    let mut control_points = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let x = i as f32 / 3.0;
                let y = j as f32 / 3.0;
                let z = k as f32 / 3.0;
                // Add a slight wave deformation
                let wave = Vec3::new(0.1 * (y * 6.28).sin(), 0.0, 0.0);
                control_points.push(Vec3::new(x, y, z) + wave);
            }
        }
    }

    let box_deformed = SdfNode::box3d(0.4, 0.4, 0.4).lattice_deform(
        control_points.clone(),
        3,
        3,
        3,
        Vec3::ZERO,
        Vec3::ONE,
    );

    let test_point = Vec3::new(0.5, 0.5, 0.5);
    let distance = eval(&box_deformed, test_point);
    println!(
        "   Box with wave lattice at {:?}: distance = {}",
        test_point, distance
    );
    println!("   (Lattice deforms the box with a sinusoidal wave)\n");

    // 3. SDF Skinning Example
    println!("3. SDF Skinning (Bone-Weight Deformation)");

    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    // Bone 1: Identity
    let bone1 = BoneTransform {
        inv_bind_pose: identity,
        current_pose: identity,
        weight: 0.5,
    };

    // Bone 2: Translated along X
    let translated = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0,
    ];

    let bone2 = BoneTransform {
        inv_bind_pose: identity,
        current_pose: translated,
        weight: 0.5,
    };

    let capsule_skinned =
        SdfNode::capsule(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.2)
            .sdf_skinning(vec![bone1, bone2]);

    let test_point = Vec3::new(0.0, 0.0, 0.0);
    let distance = eval(&capsule_skinned, test_point);
    println!(
        "   Capsule with two-bone skinning at {:?}: distance = {}",
        test_point, distance
    );
    println!("   (Blends between identity and translated poses)\n");

    // 4. Combined Example: CSG with new transforms
    println!("4. Combined Example: CSG + New Transforms");

    let sphere1 = SdfNode::sphere(0.8).projective_transform(identity_matrix, 1.0);

    let sphere2 = SdfNode::sphere(0.6)
        .translate(0.5, 0.0, 0.0)
        .lattice_deform(control_points, 3, 3, 3, Vec3::ZERO, Vec3::ONE);

    let csg_shape = sphere1.union(sphere2);

    let test_point = Vec3::new(0.0, 0.0, 0.0);
    let distance = eval(&csg_shape, test_point);
    println!(
        "   Union of projected sphere + lattice sphere at {:?}: distance = {}",
        test_point, distance
    );
    println!("   (CSG works seamlessly with new transforms)\n");

    println!("=== Demo Complete ===");
    println!("\nKey Features:");
    println!("  • ProjectiveTransform: Apply perspective warps with Lipschitz correction");
    println!("  • LatticeDeform: Free-form deformation via Bézier control points");
    println!("  • SdfSkinning: Linear blend skinning for character animation");
    println!("  • All three integrate seamlessly with existing transforms and CSG operations");
}

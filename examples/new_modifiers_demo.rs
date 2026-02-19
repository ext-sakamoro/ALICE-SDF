//! Demo of the 4 new modifiers: IcosahedralSymmetry, IFS, HeightmapDisplacement, SurfaceRoughness
//!
//! Usage: cargo run --example new_modifiers_demo

use alice_sdf::prelude::*;

fn main() {
    println!("ALICE-SDF New Modifiers Demo\n");

    // 1. Icosahedral Symmetry (120-fold)
    println!("1. Icosahedral Symmetry");
    let ico_sphere = SdfNode::sphere(1.0).icosahedral_symmetry();
    let p1 = Vec3::new(1.0, 0.0, 0.0);
    let p2 = Vec3::new(0.0, 1.0, 0.0);
    println!("   Distance at (1,0,0): {:.4}", eval(&ico_sphere, p1));
    println!("   Distance at (0,1,0): {:.4}", eval(&ico_sphere, p2));
    println!("   (Should be similar due to symmetry)\n");

    // 2. Iterated Function System (IFS)
    println!("2. IFS - Fractal Folding");
    let scale_half = [
        0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let ifs_sphere = SdfNode::sphere(0.3).ifs(vec![scale_half], 5);
    let p = Vec3::new(0.5, 0.5, 0.5);
    println!("   Distance at (0.5,0.5,0.5): {:.4}", eval(&ifs_sphere, p));
    println!("   (Creates self-similar fractal patterns)\n");

    // 3. Heightmap Displacement
    println!("3. Heightmap Displacement");
    // Simple 4x4 heightmap with a bump
    let heightmap = vec![
        0.0, 0.1, 0.1, 0.0, 0.1, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, 0.1, 0.0, 0.1, 0.1, 0.0,
    ];
    let displaced_sphere = SdfNode::sphere(1.0).heightmap_displacement(heightmap, 4, 4, 0.2, 1.0);
    println!(
        "   Distance at origin: {:.4}",
        eval(&displaced_sphere, Vec3::ZERO)
    );
    println!(
        "   Distance at (1,0,0): {:.4}",
        eval(&displaced_sphere, Vec3::X)
    );
    println!("   (Heightmap modulates surface)\n");

    // 4. Surface Roughness (FBM)
    println!("4. Surface Roughness");
    let smooth_sphere = SdfNode::sphere(1.0);
    let rough_sphere = SdfNode::sphere(1.0).surface_roughness(5.0, 0.03, 4);
    let test_point = Vec3::new(1.0, 0.0, 0.0);
    println!(
        "   Smooth sphere at (1,0,0): {:.4}",
        eval(&smooth_sphere, test_point)
    );
    println!(
        "   Rough sphere at (1,0,0): {:.4}",
        eval(&rough_sphere, test_point)
    );
    println!("   (Adds procedural micro-detail)\n");

    // 5. Combined Example
    println!("5. Combined Modifiers");
    let complex = SdfNode::box3d(1.0, 1.0, 1.0)
        .icosahedral_symmetry()
        .surface_roughness(10.0, 0.05, 3);
    println!(
        "   Distance at (0.5,0.5,0.5): {:.4}",
        eval(&complex, Vec3::new(0.5, 0.5, 0.5))
    );
    println!("   (Icosahedral symmetry + roughness)\n");

    println!("Demo complete! All 4 new modifiers working.");
}

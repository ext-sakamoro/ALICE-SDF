use alice_sdf::prelude::*;

#[test]
fn test_icosahedral_symmetry() {
    let sphere = SdfNode::sphere(1.0).icosahedral_symmetry();

    // Evaluate at different points
    let p1 = Vec3::new(1.0, 0.0, 0.0);
    let p2 = Vec3::new(0.0, 1.0, 0.0);
    let p3 = Vec3::new(0.0, 0.0, 1.0);

    let d1 = eval(&sphere, p1);
    let d2 = eval(&sphere, p2);
    let d3 = eval(&sphere, p3);

    // Due to symmetry, all should be close
    assert!(
        (d1 - d2).abs() < 0.1,
        "Icosahedral symmetry failed: {} vs {}",
        d1,
        d2
    );
    assert!(
        (d2 - d3).abs() < 0.1,
        "Icosahedral symmetry failed: {} vs {}",
        d2,
        d3
    );
}

#[test]
fn test_ifs() {
    // Scale by 0.5 transform
    let half = [
        0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let sphere = SdfNode::sphere(1.0).ifs(vec![half], 3);

    // Should evaluate without panicking
    let p = Vec3::new(0.5, 0.5, 0.5);
    let d = eval(&sphere, p);
    assert!(d.is_finite(), "IFS distance should be finite: got {}", d);
}

#[test]
fn test_heightmap_displacement() {
    // Create a simple heightmap
    let heightmap = vec![0.0, 0.5, 0.5, 1.0];

    let sphere = SdfNode::sphere(1.0).heightmap_displacement(heightmap, 2, 2, 0.1, 1.0);

    // Should evaluate without panicking
    let p = Vec3::new(0.0, 0.0, 0.0);
    let d = eval(&sphere, p);
    assert!(
        d.is_finite(),
        "Heightmap displacement should be finite: got {}",
        d
    );
}

#[test]
fn test_surface_roughness() {
    let sphere = SdfNode::sphere(1.0).surface_roughness(2.0, 0.05, 3);

    // Should evaluate without panicking
    let p = Vec3::new(1.0, 0.0, 0.0);
    let d = eval(&sphere, p);
    assert!(
        d.is_finite(),
        "Surface roughness should be finite: got {}",
        d
    );

    // Roughness should be small compared to base sphere
    let base_d = eval(&SdfNode::sphere(1.0), p);
    assert!(
        (d - base_d).abs() < 0.2,
        "Roughness too large: {} vs {}",
        d,
        base_d
    );
}

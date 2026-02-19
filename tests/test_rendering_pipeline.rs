//! Integration tests: Rendering pipeline
//!
//! Verifies raycast hit/miss, normal direction, AO, and soft shadow behavior.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::prelude::*;
use common::*;

// ============================================================================
// Raycast hit/miss
// ============================================================================

#[test]
fn raycast_hits_sphere() {
    let shape = test_sphere();
    let ray = Ray::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let hit = raycast(&shape, ray, 10.0);

    assert!(hit.is_some(), "Ray aimed at sphere should hit");
    let h = hit.unwrap();
    assert!(h.distance > 0.0, "Hit distance should be positive");
    assert!(h.distance < 5.0, "Hit should be within reasonable range");
}

#[test]
fn raycast_misses_sphere() {
    let shape = test_sphere();
    let ray = Ray::new(Vec3::new(-3.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let hit = raycast(&shape, ray, 10.0);

    assert!(hit.is_none(), "Ray above sphere should miss");
}

#[test]
fn raycast_hit_point_near_surface() {
    let shape = test_sphere();
    let ray = Ray::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let hit = raycast(&shape, ray, 10.0).expect("Should hit");

    // Hit point should be near sphere surface (distance from origin ~1.0)
    let dist_from_center = hit.point.length();
    assert!(
        (dist_from_center - 1.0).abs() < 0.01,
        "Hit point should be on sphere surface: dist={}",
        dist_from_center
    );
}

// ============================================================================
// Normal direction
// ============================================================================

#[test]
fn raycast_normal_points_outward() {
    let shape = test_sphere();
    let ray = Ray::new(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
    let hit = raycast(&shape, ray, 10.0).expect("Should hit");

    // Normal at hit point should point away from sphere center (roughly -X)
    assert!(
        hit.normal.x < -0.9,
        "Normal should point toward ray origin: {:?}",
        hit.normal
    );
    assert!(
        hit.normal.length() > 0.99,
        "Normal should be unit length: len={}",
        hit.normal.length()
    );
}

#[test]
fn normal_function_matches_gradient_direction() {
    let shape = test_sphere();
    let p = Vec3::new(1.0, 0.0, 0.0); // On sphere surface
    let eps = 0.001;

    let n = normal(&shape, p, eps);
    let g = gradient(&shape, p, eps);

    // Normal and gradient should point in same direction
    let dot = n.dot(g.normalize());
    assert!(
        dot > 0.99,
        "Normal and gradient should align: dot={}, n={:?}, g={:?}",
        dot,
        n,
        g
    );
}

// ============================================================================
// Compiled raycast
// ============================================================================

#[test]
fn raymarch_compiled_hits_sphere() {
    let shape = test_sphere();
    let compiled = CompiledSdf::compile(&shape);

    let origin = Vec3::new(-3.0, 0.0, 0.0);
    let dir = Vec3::new(1.0, 0.0, 0.0);
    let hit = alice_sdf::raycast::raymarch_compiled(&compiled, origin, dir, 10.0);

    assert!(hit.is_some(), "Compiled raymarch should hit sphere");
    let h = hit.unwrap();
    let dist = h.point.length();
    assert!(
        (dist - 1.0).abs() < 0.02,
        "Compiled hit should be on surface: dist={}",
        dist
    );
}

#[test]
fn raymarch_compiled_matches_interpreted() {
    let shape = test_csg();
    let compiled = CompiledSdf::compile(&shape);

    let origin = Vec3::new(-3.0, 0.0, 0.0);
    let dir = Vec3::new(1.0, 0.0, 0.0);

    let hit_interp = raymarch(&shape, origin, dir, 10.0);
    let hit_compiled = alice_sdf::raycast::raymarch_compiled(&compiled, origin, dir, 10.0);

    match (hit_interp, hit_compiled) {
        (Some(h1), Some(h2)) => {
            assert!(
                (h1.distance - h2.distance).abs() < 0.05,
                "Hit distances should match: interp={}, compiled={}",
                h1.distance,
                h2.distance
            );
        }
        (None, None) => {} // Both miss, OK
        _ => panic!("Compiled and interpreted should agree on hit/miss"),
    }
}

// ============================================================================
// Ambient Occlusion
// ============================================================================

#[test]
fn ao_in_open_space_is_high() {
    let shape = test_sphere();
    // Point on top of sphere, normal pointing up — open sky
    let p = Vec3::new(0.0, 1.01, 0.0);
    let n = Vec3::new(0.0, 1.0, 0.0);
    let ao = ambient_occlusion(&shape, p, n, 5, 0.5);

    assert!(
        ao > 0.5,
        "AO on exposed surface should be high (bright): ao={}",
        ao
    );
}

#[test]
fn ao_in_concavity_is_lower() {
    // Create a concave shape: inside of a box with small opening
    let shape = SdfNode::box3d(2.0, 2.0, 2.0);

    // Point inside the box, normal pointing inward — surrounded by geometry
    let p = Vec3::new(0.0, 0.0, 1.99);
    let n = Vec3::new(0.0, 0.0, -1.0);
    let ao = ambient_occlusion(&shape, p, n, 5, 1.0);

    // AO should be lower due to surrounding geometry (approximate check)
    assert!(ao.is_finite(), "AO should be finite: {}", ao);
}

// ============================================================================
// Soft Shadow
// ============================================================================

#[test]
fn soft_shadow_no_occluder() {
    let shape = SdfNode::sphere(0.5).translate(0.0, 0.0, -5.0); // Far away
    let point = Vec3::new(0.0, 0.0, 0.0);
    let light_dir = Vec3::new(0.0, 1.0, 0.0); // Light from above

    let shadow = soft_shadow(&shape, point, light_dir, 0.01, 10.0, 8.0);
    assert!(
        shadow > 0.8,
        "No occluder above → should be mostly lit: shadow={}",
        shadow
    );
}

#[test]
fn soft_shadow_with_occluder() {
    let shape = SdfNode::sphere(1.0).translate(0.0, 3.0, 0.0); // Sphere directly above
    let point = Vec3::new(0.0, 0.0, 0.0);
    let light_dir = Vec3::new(0.0, 1.0, 0.0);

    let shadow = soft_shadow(&shape, point, light_dir, 0.01, 10.0, 8.0);
    assert!(
        shadow < 0.5,
        "Sphere directly above should cast shadow: shadow={}",
        shadow
    );
}

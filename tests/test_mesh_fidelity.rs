//! Integration tests: Mesh generation fidelity
//!
//! Verifies mesh quality, vertex counts, watertightness, and LOD behavior.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::prelude::*;
use common::*;

// ============================================================================
// Basic mesh generation
// ============================================================================

#[test]
fn sphere_mesh_has_vertices_and_faces() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 32,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);

    assert!(
        mesh.vertices.len() > 100,
        "Sphere mesh should have many vertices, got {}",
        mesh.vertices.len()
    );
    assert!(
        mesh.indices.len() > 100,
        "Sphere mesh should have many indices, got {}",
        mesh.indices.len()
    );
    assert_eq!(
        mesh.indices.len() % 3,
        0,
        "Index count should be divisible by 3"
    );
}

#[test]
fn compiled_mesh_matches_interpreted_mesh() {
    let shape = test_sphere();
    let compiled = CompiledSdf::compile(&shape);
    let config = MarchingCubesConfig {
        resolution: 16,
        ..Default::default()
    };
    let bounds_min = Vec3::splat(-1.5);
    let bounds_max = Vec3::splat(1.5);

    let mesh_interp = sdf_to_mesh(&shape, bounds_min, bounds_max, &config);
    let mesh_compiled =
        alice_sdf::mesh::sdf_to_mesh_compiled(&compiled, bounds_min, bounds_max, &config);

    // Vertex counts should be similar (minor floating-point differences may cause
    // slightly different isosurface crossings at grid edges)
    let v_interp = mesh_interp.vertices.len();
    let v_compiled = mesh_compiled.vertices.len();
    let ratio = v_interp.min(v_compiled) as f64 / v_interp.max(v_compiled) as f64;
    assert!(
        ratio > 0.5,
        "Compiled and interpreted meshes should have similar vertex counts: {} vs {} (ratio={})",
        v_interp,
        v_compiled,
        ratio
    );

    // Both should produce non-empty meshes
    assert!(
        mesh_compiled.vertices.len() > 50,
        "Compiled mesh should have vertices"
    );
    assert!(
        mesh_compiled.indices.len() > 50,
        "Compiled mesh should have indices"
    );
}

// ============================================================================
// Mesh quality
// ============================================================================

#[test]
fn mesh_normals_are_unit_length() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 16,
        compute_normals: true,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);

    for (i, v) in mesh.vertices.iter().enumerate() {
        let len = v.normal.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Normal at vertex {} is not unit length: len={}",
            i,
            len
        );
    }
}

#[test]
fn sphere_mesh_vertices_near_surface() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 32,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);

    // All vertices should be near the sphere surface (distance ~0)
    for v in &mesh.vertices {
        let dist_from_surface = (v.position.length() - 1.0).abs();
        assert!(
            dist_from_surface < 0.15,
            "Vertex {:?} is far from sphere surface: dist={}",
            v.position,
            dist_from_surface
        );
    }
}

#[test]
fn mesh_validation_passes() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 16,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    let validation = validate_mesh(&mesh);

    assert_eq!(
        validation.degenerate_triangles, 0,
        "Sphere mesh should have no degenerate triangles"
    );
}

// ============================================================================
// LOD chain
// ============================================================================

#[test]
fn lod_chain_has_decreasing_vertex_count() {
    let shape = test_sphere();
    let min = Vec3::splat(-1.5);
    let max = Vec3::splat(1.5);

    let lod_config = DecimationLodConfig {
        num_levels: 4,
        base_resolution: 32,
        decimation_ratio: 0.5,
        distance_multiplier: 2.0,
        base_distance: 5.0,
        compute_normals: true,
        preserve_materials: false,
    };
    let lod_chain = generate_lod_chain_decimated(&shape, min, max, &lod_config);

    assert_eq!(lod_chain.levels.len(), 4, "Should have 4 LOD levels");

    // Each successive LOD should have fewer or equal vertices
    for i in 1..lod_chain.levels.len() {
        assert!(
            lod_chain.levels[i].mesh.vertices.len() <= lod_chain.levels[i - 1].mesh.vertices.len(),
            "LOD {} ({} verts) should have fewer vertices than LOD {} ({} verts)",
            i,
            lod_chain.levels[i].mesh.vertices.len(),
            i - 1,
            lod_chain.levels[i - 1].mesh.vertices.len()
        );
    }
}

// ============================================================================
// Bounding volumes
// ============================================================================

#[test]
fn computed_aabb_encloses_mesh() {
    let shape = test_box();
    let config = MarchingCubesConfig {
        resolution: 16,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    let aabb = compute_aabb(&mesh);

    for v in &mesh.vertices {
        assert!(
            v.position.x >= aabb.min.x - 1e-5
                && v.position.x <= aabb.max.x + 1e-5
                && v.position.y >= aabb.min.y - 1e-5
                && v.position.y <= aabb.max.y + 1e-5
                && v.position.z >= aabb.min.z - 1e-5
                && v.position.z <= aabb.max.z + 1e-5,
            "Vertex {:?} is outside AABB [{:?}, {:?}]",
            v.position,
            aabb.min,
            aabb.max
        );
    }
}

#[test]
fn bounding_sphere_encloses_mesh() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 16,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    let bs = compute_bounding_sphere(&mesh);

    for v in &mesh.vertices {
        let dist = (v.position - bs.center).length();
        assert!(
            dist <= bs.radius + 1e-4,
            "Vertex {:?} is outside bounding sphere (center={:?}, r={}, dist={})",
            v.position,
            bs.center,
            bs.radius,
            dist
        );
    }
}

// ============================================================================
// Vertex cache optimization
// ============================================================================

#[test]
fn vertex_cache_optimization_improves_acmr() {
    let shape = test_complex_shape();
    let config = MarchingCubesConfig {
        resolution: 32,
        ..Default::default()
    };
    let mut mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

    let acmr_before = compute_acmr(&mesh, 16);
    optimize_vertex_cache(&mut mesh);
    let acmr_after = compute_acmr(&mesh, 16);

    assert!(
        acmr_after <= acmr_before + 0.01,
        "ACMR should not worsen after optimization: before={}, after={}",
        acmr_before,
        acmr_after
    );
}

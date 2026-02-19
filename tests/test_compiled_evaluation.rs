//! Integration tests: Compiled vs Interpreted evaluation consistency
//!
//! Verifies CompiledSdf, CompiledSdfBvh, SIMD batch, and SoA backends
//! produce results consistent with the interpreter.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::prelude::*;
use common::*;

// ============================================================================
// CompiledSdf vs Interpreter
// ============================================================================

#[test]
fn compiled_sphere_matches_interpreted() {
    assert_compiled_matches_interpreted(&test_sphere(), 1e-5);
}

#[test]
fn compiled_box_matches_interpreted() {
    assert_compiled_matches_interpreted(&test_box(), 1e-5);
}

#[test]
fn compiled_csg_matches_interpreted() {
    assert_compiled_matches_interpreted(&test_csg(), 1e-4);
}

#[test]
fn compiled_complex_matches_interpreted() {
    assert_compiled_matches_interpreted(&test_complex_shape(), 1e-4);
}

#[test]
fn compiled_with_modifiers_matches_interpreted() {
    let shape = SdfNode::sphere(1.0).round(0.1).onion(0.05);
    assert_compiled_matches_interpreted(&shape, 1e-4);
}

// ============================================================================
// SIMD batch vs scalar
// ============================================================================

#[test]
fn simd_batch_matches_scalar() {
    let shape = test_complex_shape();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_points();

    let batch_results = eval_compiled_batch_simd(&compiled, &points);

    for (i, p) in points.iter().enumerate() {
        let scalar = eval_compiled(&compiled, *p);
        assert!(
            (batch_results[i] - scalar).abs() < 1e-5,
            "SIMD batch mismatch at point {}: batch={}, scalar={}",
            i,
            batch_results[i],
            scalar
        );
    }
}

#[test]
fn simd_parallel_matches_sequential() {
    let shape = test_complex_shape();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_grid_points(10); // 1000 points

    let seq = eval_compiled_batch_simd(&compiled, &points);
    let par = eval_compiled_batch_simd_parallel(&compiled, &points);

    assert_eq!(seq.len(), par.len());
    for i in 0..seq.len() {
        assert!(
            (seq[i] - par[i]).abs() < 1e-6,
            "Parallel mismatch at index {}: seq={}, par={}",
            i,
            seq[i],
            par[i]
        );
    }
}

// ============================================================================
// SoA batch evaluation
// ============================================================================

#[test]
fn soa_batch_matches_simd_batch() {
    let shape = test_csg();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_grid_points(8); // 512 points

    let soa_points = SoAPoints::from_vec3_slice(&points);
    let soa_results = eval_compiled_batch_soa(&compiled, &soa_points);
    let simd_results = eval_compiled_batch_simd(&compiled, &points);

    assert_eq!(soa_results.len(), simd_results.len());
    for i in 0..soa_results.len() {
        assert!(
            (soa_results[i] - simd_results[i]).abs() < 1e-5,
            "SoA/SIMD mismatch at index {}: soa={}, simd={}",
            i,
            soa_results[i],
            simd_results[i]
        );
    }
}

#[test]
fn soa_parallel_matches_sequential() {
    let shape = test_sphere();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_grid_points(10);

    let soa_points = SoAPoints::from_vec3_slice(&points);
    let seq = eval_compiled_batch_soa(&compiled, &soa_points);
    let par = eval_compiled_batch_soa_parallel(&compiled, &soa_points);

    for i in 0..seq.len() {
        assert!(
            (seq[i] - par[i]).abs() < 1e-6,
            "SoA parallel mismatch at {}: seq={}, par={}",
            i,
            seq[i],
            par[i]
        );
    }
}

// ============================================================================
// BVH evaluation
// ============================================================================

#[test]
fn bvh_matches_compiled() {
    let shape = test_csg();
    let compiled = CompiledSdf::compile(&shape);
    let bvh = CompiledSdfBvh::compile(&shape);

    for p in test_points() {
        let d_compiled = eval_compiled(&compiled, p);
        let d_bvh = eval_compiled_bvh(&bvh, p);
        assert!(
            (d_compiled - d_bvh).abs() < 1e-4,
            "BVH mismatch at {:?}: compiled={}, bvh={}",
            p,
            d_compiled,
            d_bvh
        );
    }
}

#[test]
fn bvh_scene_aabb_is_valid() {
    let shape = SdfNode::sphere(1.0).translate(2.0, 3.0, 0.0);
    let bvh = CompiledSdfBvh::compile(&shape);
    let aabb = get_scene_aabb(&bvh);

    // The sphere at (2,3,0) with radius 1 should have AABB roughly [1,2,-1] to [3,4,1]
    assert!(
        aabb.min_x <= 1.5 && aabb.max_x >= 2.5,
        "AABB X range invalid: min_x={}, max_x={}",
        aabb.min_x,
        aabb.max_x
    );
    assert!(
        aabb.min_y <= 2.5 && aabb.max_y >= 3.5,
        "AABB Y range invalid: min_y={}, max_y={}",
        aabb.min_y,
        aabb.max_y
    );
}

// ============================================================================
// try_compile error handling
// ============================================================================

#[test]
fn try_compile_success_on_supported_shapes() {
    let shape = test_complex_shape();
    let result = CompiledSdf::try_compile(&shape);
    assert!(
        result.is_ok(),
        "try_compile should succeed for standard shapes"
    );
}

// ============================================================================
// Compiled normals
// ============================================================================

#[test]
fn compiled_normal_direction_is_outward() {
    let shape = test_sphere();
    let compiled = CompiledSdf::compile(&shape);

    // Normal at surface point on X-axis should point roughly in +X direction
    let p = Vec3::new(1.0, 0.0, 0.0);
    let n = eval_compiled_normal(&compiled, p, 0.001);
    assert!(n.x > 0.9, "Normal should point outward: {:?}", n);
    assert!(
        n.length() > 0.99,
        "Normal should be normalized: len={}",
        n.length()
    );
}

#[test]
fn compiled_distance_and_normal_consistency() {
    let shape = test_box();
    let compiled = CompiledSdf::compile(&shape);

    let p = Vec3::new(1.0, 0.0, 0.0);
    let (dist, norm) = eval_compiled_distance_and_normal(&compiled, p, 0.001);

    let d_standalone = eval_compiled(&compiled, p);
    let n_standalone = eval_compiled_normal(&compiled, p, 0.001);

    assert_close(dist, d_standalone, 1e-5, "distance mismatch");
    assert_close(norm.x, n_standalone.x, 1e-4, "normal X mismatch");
    assert_close(norm.y, n_standalone.y, 1e-4, "normal Y mismatch");
    assert_close(norm.z, n_standalone.z, 1e-4, "normal Z mismatch");
}

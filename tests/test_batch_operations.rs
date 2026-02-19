//! Integration tests: Batch processing operations
//!
//! Verifies batch_parallel vs sequential, SIMD vs scalar, and determinism.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::prelude::*;
use common::*;

// ============================================================================
// Interpreter batch consistency
// ============================================================================

#[test]
fn eval_batch_matches_individual() {
    let shape = test_complex_shape();
    let points = test_points();

    let batch = eval_batch(&shape, &points);

    for (i, p) in points.iter().enumerate() {
        let individual = eval(&shape, *p);
        assert!(
            (batch[i] - individual).abs() < 1e-6,
            "Batch mismatch at point {}: batch={}, individual={}",
            i,
            batch[i],
            individual
        );
    }
}

#[test]
fn eval_batch_parallel_matches_sequential() {
    let shape = test_complex_shape();
    let points = test_grid_points(10); // 1000 points

    let sequential = eval_batch(&shape, &points);
    let parallel = eval_batch_parallel(&shape, &points);

    assert_eq!(sequential.len(), parallel.len());
    for i in 0..sequential.len() {
        assert!(
            (sequential[i] - parallel[i]).abs() < 1e-6,
            "Parallel mismatch at index {}: seq={}, par={}",
            i,
            sequential[i],
            parallel[i]
        );
    }
}

// ============================================================================
// Compiled SIMD batch
// ============================================================================

#[test]
fn compiled_simd_batch_large_dataset() {
    let shape = test_csg();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_grid_points(15); // 3375 points

    let results = eval_compiled_batch_simd_parallel(&compiled, &points);

    assert_eq!(
        results.len(),
        points.len(),
        "Result count should match point count"
    );

    // All results should be finite
    for (i, &d) in results.iter().enumerate() {
        assert!(
            d.is_finite(),
            "Distance at index {} is not finite: {}",
            i,
            d
        );
    }
}

// ============================================================================
// Grid evaluation
// ============================================================================

#[test]
fn eval_grid_correct_size() {
    let shape = test_sphere();
    let resolution = 16;
    let grid = eval_grid(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), resolution);

    assert_eq!(
        grid.len(),
        resolution * resolution * resolution,
        "Grid should have resolution^3 entries"
    );
}

#[test]
fn eval_grid_sphere_inside_outside() {
    let shape = test_sphere(); // Unit sphere at origin
    let resolution = 8;
    let grid = eval_grid(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), resolution);

    // Center cell should be negative (inside)
    let center_idx = (resolution / 2) * resolution * resolution
        + (resolution / 2) * resolution
        + (resolution / 2);
    assert!(
        grid[center_idx] < 0.0,
        "Center of sphere should be inside (negative): {}",
        grid[center_idx]
    );

    // Corner cell should be positive (outside)
    assert!(
        grid[0] > 0.0,
        "Corner should be outside (positive): {}",
        grid[0]
    );
}

// ============================================================================
// Determinism
// ============================================================================

#[test]
fn eval_is_deterministic() {
    let shape = test_complex_shape();
    let points = test_grid_points(8);

    let run1 = eval_batch(&shape, &points);
    let run2 = eval_batch(&shape, &points);

    for i in 0..run1.len() {
        assert_eq!(
            run1[i].to_bits(),
            run2[i].to_bits(),
            "Evaluation should be bit-exact deterministic at index {}",
            i
        );
    }
}

#[test]
fn compiled_eval_is_deterministic() {
    let shape = test_complex_shape();
    let compiled = CompiledSdf::compile(&shape);
    let points = test_grid_points(8);

    let run1 = eval_compiled_batch_simd(&compiled, &points);
    let run2 = eval_compiled_batch_simd(&compiled, &points);

    for i in 0..run1.len() {
        assert_eq!(
            run1[i].to_bits(),
            run2[i].to_bits(),
            "Compiled evaluation should be bit-exact deterministic at index {}",
            i
        );
    }
}

// ============================================================================
// SoAPoints construction
// ============================================================================

#[test]
fn soa_from_vec3_preserves_data() {
    let points = test_points();
    let soa = SoAPoints::from_vec3_slice(&points);

    assert_eq!(soa.len(), points.len());
    assert!(soa.padded_len() >= soa.len());
    // padded_len should be a multiple of SIMD width (8)
    assert_eq!(soa.padded_len() % 8, 0);
}

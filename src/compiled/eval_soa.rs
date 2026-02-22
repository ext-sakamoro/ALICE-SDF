//! SoA-Optimized Batch Evaluation
//!
//! This module provides SIMD-optimized batch evaluation functions that
//! leverage SoA (Structure of Arrays) memory layout for maximum throughput.
//!
//! # Performance Characteristics
//!
//! | Method | Memory Layout | SIMD Load | Cache Efficiency |
//! |--------|---------------|-----------|------------------|
//! | `eval_compiled_batch` | AoS (`Vec<Vec3>`) | Gather/Shuffle | Poor |
//! | `eval_compiled_batch_soa` | SoA | Direct Load | Excellent |
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::soa::SoAPoints;
//! use alice_sdf::compiled::{CompiledSdf, eval_compiled_batch_soa};
//!
//! let shape = SdfNode::sphere(1.0);
//! let compiled = CompiledSdf::compile(&shape);
//!
//! // Create SoA points
//! let points: SoAPoints = (0..10000)
//!     .map(|i| Vec3::new(i as f32 * 0.01, 0.0, 0.0))
//!     .collect();
//!
//! // Evaluate with SoA optimization
//! let distances = eval_compiled_batch_soa(&compiled, &points);
//! ```
//!
//! Author: Moroya Sakamoto

use super::eval_simd::eval_compiled_simd;
use super::simd::Vec3x8;
use super::CompiledSdf;
use crate::soa::{SoADistances, SoAPoints, SIMD_WIDTH};
use rayon::prelude::*;
use wide::f32x8;

/// Evaluate compiled SDF at many points using SoA layout (single-threaded)
///
/// This function leverages the SoA memory layout for direct SIMD loading,
/// eliminating gather/shuffle operations that are costly on modern CPUs.
///
/// # Arguments
/// * `sdf` - The compiled SDF to evaluate
/// * `points` - Points in SoA layout
///
/// # Returns
/// Vector of distances, one per input point
pub fn eval_compiled_batch_soa(sdf: &CompiledSdf, points: &SoAPoints) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }

    let len = points.len();
    let padded_len = points.padded_len();
    let mut results = vec![0.0f32; padded_len];

    // Process 8 points at a time using direct SIMD loads
    for chunk_start in (0..padded_len).step_by(SIMD_WIDTH) {
        // Direct load from SoA - no shuffle needed!
        let (x, y, z) = points.load_simd(chunk_start).unwrap_or_else(|| {
            // Fallback for partial chunks (shouldn't happen with proper padding)
            (f32x8::splat(0.0), f32x8::splat(0.0), f32x8::splat(0.0))
        });

        let p = Vec3x8 { x, y, z };

        // Evaluate SIMD batch
        let distances = eval_compiled_simd(sdf, p);

        // Store results
        let dist_arr: [f32; 8] = distances.into();
        results[chunk_start..chunk_start + 8].copy_from_slice(&dist_arr);
    }

    // Truncate to actual length
    results.truncate(len);
    results
}

/// Evaluate compiled SDF at many points using SoA layout (multi-threaded)
///
/// Uses Rayon for parallel processing across CPU cores while maintaining
/// SIMD efficiency within each thread.
///
/// # Arguments
/// * `sdf` - The compiled SDF to evaluate
/// * `points` - Points in SoA layout
///
/// # Returns
/// Vector of distances, one per input point
pub fn eval_compiled_batch_soa_parallel(sdf: &CompiledSdf, points: &SoAPoints) -> Vec<f32> {
    if points.is_empty() {
        return Vec::new();
    }

    let len = points.len();
    let padded_len = points.padded_len();

    // Threshold for parallel processing (overhead not worth it for small batches)
    const PARALLEL_THRESHOLD: usize = 256;

    if padded_len < PARALLEL_THRESHOLD {
        return eval_compiled_batch_soa(sdf, points);
    }

    // Pre-allocate output buffer
    let mut results = SoADistances::with_capacity(len);

    // Get raw slices for parallel access
    let (x_slice, y_slice, z_slice) = points.as_slices();

    // Process in chunks of SIMD_WIDTH, parallelized
    let chunk_count = padded_len / SIMD_WIDTH;

    let chunk_results: Vec<[f32; 8]> = (0..chunk_count)
        .into_par_iter()
        .map(|chunk_idx| {
            let offset = chunk_idx * SIMD_WIDTH;

            // Direct load from SoA slices
            let x = f32x8::from(&x_slice[offset..offset + 8]);
            let y = f32x8::from(&y_slice[offset..offset + 8]);
            let z = f32x8::from(&z_slice[offset..offset + 8]);

            let p = Vec3x8 { x, y, z };
            let distances = eval_compiled_simd(sdf, p);

            distances.into()
        })
        .collect();

    // Flatten results
    for (chunk_idx, chunk_result) in chunk_results.into_iter().enumerate() {
        let offset = chunk_idx * SIMD_WIDTH;
        let copy_len = (len - offset).min(SIMD_WIDTH);
        results.distances[offset..offset + copy_len].copy_from_slice(&chunk_result[..copy_len]);
    }

    results.to_vec()
}

/// Evaluate and store results directly into pre-allocated SoA output
///
/// This is the most efficient variant for repeated evaluations where
/// you want to avoid allocation overhead.
///
/// # Arguments
/// * `sdf` - The compiled SDF to evaluate
/// * `points` - Input points in SoA layout
/// * `output` - Pre-allocated output buffer (must have capacity >= points.len())
///
/// # Safety
/// Output must have capacity for at least `points.padded_len()` elements.
pub fn eval_compiled_batch_soa_into(
    sdf: &CompiledSdf,
    points: &SoAPoints,
    output: &mut SoADistances,
) {
    let padded_len = points.padded_len();

    // Ensure output has space
    if output.distances.len() < padded_len {
        output.distances.resize(padded_len, 0.0);
    }

    let (x_slice, y_slice, z_slice) = points.as_slices();

    for chunk_start in (0..padded_len).step_by(SIMD_WIDTH) {
        let x = f32x8::from(&x_slice[chunk_start..chunk_start + 8]);
        let y = f32x8::from(&y_slice[chunk_start..chunk_start + 8]);
        let z = f32x8::from(&z_slice[chunk_start..chunk_start + 8]);

        let p = Vec3x8 { x, y, z };
        let distances = eval_compiled_simd(sdf, p);

        // Store directly
        // SAFETY: `chunk_start` is bounded by the loop condition
        // (chunk_start + SIMD_WIDTH <= padded_len). The output buffer was
        // resized to at least `padded_len` elements above.
        unsafe {
            output.store_simd_unchecked(chunk_start, distances);
        }
    }
}

/// Ultra-fast evaluation using raw pointers (unsafe but maximum performance)
///
/// This function provides the absolute minimum overhead for SDF evaluation.
/// Use when you have hot paths that need every bit of performance.
///
/// # Safety
/// - `x_ptr`, `y_ptr`, `z_ptr` must point to valid memory of at least `count` f32s
/// - `out_ptr` must point to valid memory of at least `count` f32s
/// - All pointers should be aligned to 32 bytes for best performance
/// - `count` should be a multiple of 8
#[inline(never)]
pub unsafe fn eval_compiled_batch_soa_raw(
    sdf: &CompiledSdf,
    x_ptr: *const f32,
    y_ptr: *const f32,
    z_ptr: *const f32,
    out_ptr: *mut f32,
    count: usize,
) {
    let aligned_count = (count + 7) & !7; // Round up to multiple of 8

    for i in (0..aligned_count).step_by(8) {
        // SAFETY: Caller guarantees x_ptr, y_ptr, z_ptr each point to at least
        // `count` valid f32 elements (rounded up to multiple of 8). The loop
        // index `i` ranges from 0 to aligned_count in steps of 8, so
        // `ptr.add(i)` through `ptr.add(i + 7)` are within bounds.
        let x = f32x8::from(std::slice::from_raw_parts(x_ptr.add(i), 8));
        let y = f32x8::from(std::slice::from_raw_parts(y_ptr.add(i), 8));
        let z = f32x8::from(std::slice::from_raw_parts(z_ptr.add(i), 8));

        let p = Vec3x8 { x, y, z };
        let distances = eval_compiled_simd(sdf, p);

        // SAFETY: Caller guarantees out_ptr points to at least `count` writable
        // f32 elements. Source (stack array) and destination (out_ptr) do not overlap.
        let arr: [f32; 8] = distances.into();
        std::ptr::copy_nonoverlapping(arr.as_ptr(), out_ptr.add(i), 8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;
    use crate::types::SdfNode;
    use glam::Vec3;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_soa_batch_eval_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = eval_compiled_batch_soa(&compiled, &soa);

        assert_eq!(results.len(), points.len());

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&sphere, *p);
            assert!(
                approx_eq(results[i], expected, 0.001),
                "Mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_soa_batch_eval_parallel() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(2.0, 0.0, 0.0),
            0.2,
        );
        let compiled = CompiledSdf::compile(&shape);

        // Generate many points
        let points: Vec<Vec3> = (0..1000)
            .map(|i| Vec3::new(i as f32 * 0.01 - 5.0, 0.0, 0.0))
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);

        // Compare single-threaded and parallel results
        let results_st = eval_compiled_batch_soa(&compiled, &soa);
        let results_mt = eval_compiled_batch_soa_parallel(&compiled, &soa);

        assert_eq!(results_st.len(), results_mt.len());

        for i in 0..points.len() {
            assert!(
                approx_eq(results_st[i], results_mt[i], 0.0001),
                "Mismatch at {}: st={}, mt={}",
                i,
                results_st[i],
                results_mt[i]
            );
        }
    }

    #[test]
    fn test_soa_batch_eval_into() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);

        let points = vec![Vec3::ZERO, Vec3::new(2.0, 0.0, 0.0)];
        let soa = SoAPoints::from_vec3_slice(&points);

        let mut output = SoADistances::with_capacity(points.len());
        eval_compiled_batch_soa_into(&compiled, &soa, &mut output);

        let results = output.to_vec();

        assert!(approx_eq(results[0], -1.0, 0.001)); // Inside sphere
        assert!(approx_eq(results[1], 1.0, 0.001)); // Outside sphere
    }

    #[test]
    fn test_soa_vs_aos_consistency() {
        let shape = SdfNode::Sphere { radius: 1.0 }
            .union(SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            })
            .translate(1.0, 0.0, 0.0);

        let compiled = CompiledSdf::compile(&shape);

        let points: Vec<Vec3> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0 * std::f32::consts::TAU;
                Vec3::new(t.cos() * 2.0, t.sin() * 2.0, 0.0)
            })
            .collect();

        // SoA evaluation
        let soa = SoAPoints::from_vec3_slice(&points);
        let soa_results = eval_compiled_batch_soa(&compiled, &soa);

        // Standard evaluation
        for (i, p) in points.iter().enumerate() {
            let expected = eval(&shape, *p);
            assert!(
                approx_eq(soa_results[i], expected, 0.01),
                "SoA/CPU mismatch at {}: soa={}, cpu={}",
                i,
                soa_results[i],
                expected
            );
        }
    }

    #[test]
    fn test_soa_large_batch() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);

        // Large batch
        let points: Vec<Vec3> = (0..10000)
            .map(|i| {
                let t = i as f32 / 10000.0;
                Vec3::new(t * 4.0 - 2.0, (t * 10.0).sin(), (t * 10.0).cos())
            })
            .collect();

        let soa = SoAPoints::from_vec3_slice(&points);
        let results = eval_compiled_batch_soa_parallel(&compiled, &soa);

        assert_eq!(results.len(), 10000);

        // Spot check
        for i in [0, 1000, 5000, 9999] {
            let expected = eval(&sphere, points[i]);
            assert!(
                approx_eq(results[i], expected, 0.01),
                "Mismatch at {}: got {}, expected {}",
                i,
                results[i],
                expected
            );
        }
    }

    #[test]
    fn test_soa_raw_eval() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&sphere);

        let points = vec![
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ];

        let soa = SoAPoints::from_vec3_slice(&points);
        let (px, py, pz) = soa.as_ptrs();

        let mut output = vec![0.0f32; 8];

        // SAFETY: soa.as_ptrs() returns valid pointers into the SoA allocation, each with 8
        // elements (padded). output has 8 elements. All pointers remain valid for this call.
        unsafe {
            eval_compiled_batch_soa_raw(&compiled, px, py, pz, output.as_mut_ptr(), 8);
        }

        for (i, p) in points.iter().enumerate() {
            let expected = eval(&sphere, *p);
            assert!(
                approx_eq(output[i], expected, 0.001),
                "Raw mismatch at {}: got {}, expected {}",
                i,
                output[i],
                expected
            );
        }
    }
}

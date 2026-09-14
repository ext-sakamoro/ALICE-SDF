//! SIMD-accelerated SDF evaluation (8 points at once)
//!
//! This module provides 8-wide SIMD evaluation using AVX2/AVX-512/NEON,
//! achieving up to 8x speedup over scalar evaluation. Since 1.10.0 the
//! evaluator itself is the shared generic stack machine in `eval_core`; this
//! file keeps the batch / gradient entry points.
//!
//! Author: Moroya Sakamoto

use super::compiler::CompiledSdf;
use super::simd::Vec3x8;
use glam::Vec3;
use wide::f32x8;

/// Evaluate compiled SDF at 8 points simultaneously
///
/// Instantiation of the shared stack machine ([`super::eval_core`]) with
/// `R = wide::f32x8`: every transform / modifier / post-processing law is the
/// same code the scalar evaluator runs.
///
/// # Arguments
/// * `sdf` - Compiled SDF bytecode
/// * `points` - 8 points packed as Vec3x8
///
/// # Returns
/// 8 distance values as f32x8
#[inline]
pub fn eval_compiled_simd(sdf: &CompiledSdf, points: Vec3x8) -> f32x8 {
    super::eval_core::eval_bytecode::<f32x8>(&sdf.instructions, &sdf.aux_data, points.into())
}

/// Batch evaluate compiled SDF using SIMD (8 points at a time)
///
/// This is the main entry point for SIMD-accelerated batch evaluation.
/// Points are processed 8 at a time, with any remainder handled by
/// scalar evaluation.
pub fn eval_compiled_batch_simd(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    let n = points.len();
    let mut results = vec![0.0f32; n];

    // Process 8 points at a time
    let chunks = n / 8;
    for i in 0..chunks {
        let base = i * 8;
        let p = Vec3x8::from_vecs([
            points[base],
            points[base + 1],
            points[base + 2],
            points[base + 3],
            points[base + 4],
            points[base + 5],
            points[base + 6],
            points[base + 7],
        ]);

        let d = eval_compiled_simd(sdf, p);
        let arr = d.to_array();
        results[base..base + 8].copy_from_slice(&arr);
    }

    // Handle remainder with scalar evaluation
    let remainder = n % 8;
    if remainder > 0 {
        let base = chunks * 8;
        for i in 0..remainder {
            results[base + i] = super::eval::eval_compiled(sdf, points[base + i]);
        }
    }

    results
}

/// Parallel batch evaluate compiled SDF using SIMD
///
/// Combines SIMD (8-wide) with multi-threading for maximum performance.
pub fn eval_compiled_batch_simd_parallel(sdf: &CompiledSdf, points: &[Vec3]) -> Vec<f32> {
    use rayon::prelude::*;

    let n = points.len();
    if n < 64 {
        // Not worth parallelizing for small inputs
        return eval_compiled_batch_simd(sdf, points);
    }

    // Process in chunks of 64 (8 SIMD lanes * 8 iterations per thread)
    let chunk_size = 64;
    let mut results = vec![0.0f32; n];

    results
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let base = chunk_idx * chunk_size;
            let chunk_len = chunk.len();

            // Process 8 points at a time within this chunk
            let simd_iters = chunk_len / 8;
            for i in 0..simd_iters {
                let p_base = base + i * 8;
                let p = Vec3x8::from_vecs([
                    points[p_base],
                    points[p_base + 1],
                    points[p_base + 2],
                    points[p_base + 3],
                    points[p_base + 4],
                    points[p_base + 5],
                    points[p_base + 6],
                    points[p_base + 7],
                ]);

                let d = eval_compiled_simd(sdf, p);
                let arr = d.to_array();
                chunk[i * 8..(i + 1) * 8].copy_from_slice(&arr);
            }

            // Handle remainder within chunk
            let remainder = chunk_len % 8;
            if remainder > 0 {
                let r_base = simd_iters * 8;
                for i in 0..remainder {
                    chunk[r_base + i] = super::eval::eval_compiled(sdf, points[base + r_base + i]);
                }
            }
        });

    results
}

// ============================================================================
// Gradient (Normal) Computation - SIMD
// ============================================================================

/// Compute gradient (normal direction) using finite differences - SIMD 8-wide
///
/// Returns unnormalized gradient vectors (gx, gy, gz) for 8 points simultaneously.
/// The caller can normalize if needed.
///
/// # Performance
/// - 6 SDF evaluations per call (central differences)
/// - ~6x slower than distance-only, but still SIMD-accelerated
/// - For 100K points: ~0.6ms on modern CPU
#[inline]
pub fn eval_gradient_simd(sdf: &CompiledSdf, p: Vec3x8, epsilon: f32) -> (f32x8, f32x8, f32x8) {
    let e = f32x8::splat(epsilon);
    let ne = f32x8::splat(-epsilon);

    // Tetrahedral method: 4 evaluations instead of 6
    let v0 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + e,
            y: p.y + ne,
            z: p.z + ne,
        },
    ); // (+,-,-)
    let v1 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + ne,
            y: p.y + ne,
            z: p.z + e,
        },
    ); // (-,-,+)
    let v2 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + ne,
            y: p.y + e,
            z: p.z + ne,
        },
    ); // (-,+,-)
    let v3 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + e,
            y: p.y + e,
            z: p.z + e,
        },
    ); // (+,+,+)

    let gx = v0 - v1 - v2 + v3;
    let gy = -v0 - v1 + v2 + v3;
    let gz = -v0 + v1 - v2 + v3;

    (gx, gy, gz)
}

/// Compute both distance and gradient in one call - SIMD 8-wide
///
/// More efficient when you need both values, as it shares the center evaluation.
#[inline]
pub fn eval_distance_and_gradient_simd(
    sdf: &CompiledSdf,
    p: Vec3x8,
    epsilon: f32,
) -> (f32x8, f32x8, f32x8, f32x8) {
    // Center distance
    let d_center = eval_compiled_simd(sdf, p);

    // Tetrahedral method: 4 evaluations for gradient
    let e = f32x8::splat(epsilon);
    let ne = f32x8::splat(-epsilon);

    let v0 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + e,
            y: p.y + ne,
            z: p.z + ne,
        },
    ); // (+,-,-)
    let v1 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + ne,
            y: p.y + ne,
            z: p.z + e,
        },
    ); // (-,-,+)
    let v2 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + ne,
            y: p.y + e,
            z: p.z + ne,
        },
    ); // (-,+,-)
    let v3 = eval_compiled_simd(
        sdf,
        Vec3x8 {
            x: p.x + e,
            y: p.y + e,
            z: p.z + e,
        },
    ); // (+,+,+)

    let gx = v0 - v1 - v2 + v3;
    let gy = -v0 - v1 + v2 + v3;
    let gz = -v0 + v1 - v2 + v3;

    (d_center, gx, gy, gz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiled::CompiledSdf;
    use crate::eval::eval;
    use crate::types::SdfNode;

    #[test]
    fn test_simd_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Check against scalar evaluation
        assert!((arr[0] - (-1.0)).abs() < 0.001); // origin
        assert!(arr[1].abs() < 0.001); // on surface
        assert!((arr[2] - 1.0).abs() < 0.001); // outside
    }

    #[test]
    fn test_simd_box() {
        let node = SdfNode::box3d(1.0, 1.0, 1.0);
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(1.5, 1.5, 0.0),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Compare with scalar
        let test_vecs = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.001,
                "Mismatch at {}: simd={}, scalar={}",
                i,
                arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_union() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&node);

        let points = Vec3x8::from_vecs([
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        // Verify against scalar
        let test_vecs = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.01,
                "Union mismatch at {}: simd={}, scalar={}",
                i,
                arr[i],
                scalar
            );
        }
    }

    #[test]
    fn test_simd_batch() {
        let node = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1);
        let compiled = CompiledSdf::compile(&node);

        let points: Vec<Vec3> = (0..100)
            .map(|i| {
                let t = i as f32 / 100.0;
                Vec3::new(
                    (t * 12.34).sin() * 2.0,
                    (t * 23.45).sin() * 2.0,
                    (t * 34.56).sin() * 2.0,
                )
            })
            .collect();

        let simd_results = eval_compiled_batch_simd(&compiled, &points);
        let scalar_results: Vec<f32> = points.iter().map(|p| eval(&node, *p)).collect();

        for (i, (simd, scalar)) in simd_results.iter().zip(scalar_results.iter()).enumerate() {
            assert!(
                (simd - scalar).abs() < 0.01,
                "Batch mismatch at {}: simd={}, scalar={}",
                i,
                simd,
                scalar
            );
        }
    }

    #[test]
    fn test_simd_complex() {
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
            .translate(0.5, 0.0, 0.0)
            .scale(1.5);
        let compiled = CompiledSdf::compile(&node);

        let test_vecs = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        let points = Vec3x8::from_vecs([
            test_vecs[0],
            test_vecs[1],
            test_vecs[2],
            test_vecs[3],
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::ZERO,
        ]);

        let d = eval_compiled_simd(&compiled, points);
        let arr = d.to_array();

        for (i, v) in test_vecs.iter().enumerate() {
            let scalar = eval(&node, *v);
            assert!(
                (arr[i] - scalar).abs() < 0.05,
                "Complex mismatch at {:?}: simd={}, scalar={}",
                v,
                arr[i],
                scalar
            );
        }
    }
}

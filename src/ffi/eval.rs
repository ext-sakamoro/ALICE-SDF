//! FFI functions for evaluating SDFs (single point, batch, SoA, gradient, animated).
//!
//! Author: Moroya Sakamoto

use super::registry::{get_compiled, get_node};
use super::types::*;
use crate::animation::AnimationParams;
use crate::compiled::{eval_compiled, CompiledSdf};
use std::slice;

// ============================================================================
// Evaluation (Deep Fried)
// ============================================================================

/// Evaluate SDF at a single point
///
/// For bulk evaluation, use `alice_sdf_eval_compiled_batch` instead.
#[no_mangle]
pub extern "C" fn alice_sdf_eval(node: SdfHandle, x: f32, y: f32, z: f32) -> f32 {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return f32::MAX,
    };
    crate::eval(&sdf_node, glam::Vec3::new(x, y, z))
}

/// Evaluate compiled SDF at a single point
#[no_mangle]
pub extern "C" fn alice_sdf_eval_compiled(compiled: CompiledHandle, x: f32, y: f32, z: f32) -> f32 {
    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return f32::MAX,
    };
    eval_compiled(&comp, glam::Vec3::new(x, y, z))
}

/// Batch evaluate SDF at multiple points (parallel, zero-copy output)
///
/// # Parameters
/// - `node`: SDF handle (will be compiled internally if not already)
/// - `points`: Pointer to array of floats [x0, y0, z0, x1, y1, z1, ...]
/// - `distances`: Output array (must be pre-allocated with `count` elements)
/// - `count`: Number of points to evaluate
///
/// # Safety
///
/// - `points` must point to a valid array of at least `count * 3` f32 values (AoS layout:
///   [x0, y0, z0, x1, y1, z1, ...]). The pointer must be non-null and properly aligned.
/// - `distances` must point to a writable array of at least `count` f32 values.
/// - Both pointers must remain valid and non-aliasing for the entire duration of the call.
///
/// # Deep Fried Optimization
/// Compiles the SDF internally and uses parallel evaluation.
/// For maximum performance, use `alice_sdf_compile` + `alice_sdf_eval_compiled_batch`.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_batch(
    node: SdfHandle,
    points: *const f32,
    distances: *mut f32,
    count: u32,
) -> BatchResult {
    if points.is_null() || distances.is_null() {
        return BatchResult {
            count: 0,
            result: SdfResult::NullPointer,
        };
    }

    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => {
            return BatchResult {
                count: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let count_usize = count as usize;
    // SAFETY: Caller guarantees points is a non-null, valid pointer to at least count * 3 f32
    // values in AoS layout. Null check performed above. Pointer remains valid for this call.
    let points_slice = slice::from_raw_parts(points, count_usize * 3);
    // SAFETY: Caller guarantees distances is a non-null, writable pointer to at least count f32
    // values. Null check performed above. No other reference aliases this slice during the call.
    let distances_slice = slice::from_raw_parts_mut(distances, count_usize);

    // Compile once
    let compiled = CompiledSdf::compile(&sdf_node);

    // Parallel evaluation with Rayon
    use rayon::prelude::*;

    if count_usize >= 256 {
        // Parallel for large batches
        distances_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, d)| {
                let p = glam::Vec3::new(
                    points_slice[i * 3],
                    points_slice[i * 3 + 1],
                    points_slice[i * 3 + 2],
                );
                *d = eval_compiled(&compiled, p);
            });
    } else {
        // Sequential for small batches (Rayon overhead not worth it)
        for i in 0..count_usize {
            let p = glam::Vec3::new(
                points_slice[i * 3],
                points_slice[i * 3 + 1],
                points_slice[i * 3 + 2],
            );
            distances_slice[i] = eval_compiled(&compiled, p);
        }
    }

    BatchResult {
        count,
        result: SdfResult::Ok,
    }
}

/// Batch evaluate compiled SDF (fastest path for AoS data)
///
/// # Parameters
/// - `compiled`: Pre-compiled SDF handle from `alice_sdf_compile`
/// - `points`: Pointer to array of floats [x0, y0, z0, x1, y1, z1, ...]
/// - `distances`: Output array (must be pre-allocated with `count` elements)
/// - `count`: Number of points to evaluate
///
/// # Safety
///
/// - `points` must point to a valid array of at least `count * 3` f32 values (AoS layout:
///   [x0, y0, z0, x1, y1, z1, ...]). The pointer must be non-null and properly aligned.
/// - `distances` must point to a writable array of at least `count` f32 values.
/// - Both pointers must remain valid and non-aliasing for the entire duration of the call.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_compiled_batch(
    compiled: CompiledHandle,
    points: *const f32,
    distances: *mut f32,
    count: u32,
) -> BatchResult {
    if points.is_null() || distances.is_null() {
        return BatchResult {
            count: 0,
            result: SdfResult::NullPointer,
        };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => {
            return BatchResult {
                count: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let count_usize = count as usize;
    // SAFETY: Caller guarantees points is a non-null, valid pointer to at least count * 3 f32
    // values in AoS layout. Null check performed above. Pointer remains valid for this call.
    let points_slice = slice::from_raw_parts(points, count_usize * 3);
    // SAFETY: Caller guarantees distances is a non-null, writable pointer to at least count f32
    // values. Null check performed above. No other reference aliases this slice during the call.
    let distances_slice = slice::from_raw_parts_mut(distances, count_usize);

    use rayon::prelude::*;

    if count_usize >= 256 {
        distances_slice
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, d)| {
                let p = glam::Vec3::new(
                    points_slice[i * 3],
                    points_slice[i * 3 + 1],
                    points_slice[i * 3 + 2],
                );
                *d = eval_compiled(&comp, p);
            });
    } else {
        for i in 0..count_usize {
            let p = glam::Vec3::new(
                points_slice[i * 3],
                points_slice[i * 3 + 1],
                points_slice[i * 3 + 2],
            );
            distances_slice[i] = eval_compiled(&comp, p);
        }
    }

    BatchResult {
        count,
        result: SdfResult::Ok,
    }
}

/// Batch evaluate using SoA (Structure of Arrays) layout - THE FASTEST PATH
///
/// SoA layout enables SIMD vectorization and better cache utilization.
/// X, Y, Z coordinates are stored in separate contiguous arrays.
///
/// # Parameters
/// - `compiled`: Pre-compiled SDF handle
/// - `x`: Pointer to X coordinates array
/// - `y`: Pointer to Y coordinates array
/// - `z`: Pointer to Z coordinates array
/// - `distances`: Output array (caller-allocated)
/// - `count`: Number of points
///
/// # Safety
///
/// - `x`, `y`, and `z` must each point to a valid array of at least `count` f32 values.
/// - `distances` must point to a writable array of at least `count` f32 values.
/// - All four pointers must be non-null and remain valid for the entire duration of the call.
/// - Arrays should be 32-byte aligned for AVX2 (not required, but faster).
///
/// # Performance
/// This is the fastest evaluation path. Use for:
/// - Physics simulations
/// - Particle systems
/// - Real-time SDf tracing
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_soa(
    compiled: CompiledHandle,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    distances: *mut f32,
    count: u32,
) -> BatchResult {
    if x.is_null() || y.is_null() || z.is_null() || distances.is_null() {
        return BatchResult {
            count: 0,
            result: SdfResult::NullPointer,
        };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => {
            return BatchResult {
                count: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let count_usize = count as usize;
    // SAFETY: Caller guarantees x, y, z are non-null pointers to valid arrays of at least count
    // f32 values each (SoA layout). Null checks performed above. Pointers remain valid for this
    // call. The three slices do not overlap and are not aliased by dist_slice.
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    // SAFETY: Caller guarantees distances is a non-null, writable pointer to at least count f32
    // values. Null check performed above. Does not alias x, y, or z slices.
    let dist_slice = slice::from_raw_parts_mut(distances, count_usize);

    use rayon::prelude::*;

    // Threshold for parallelization
    const PARALLEL_THRESHOLD: usize = 1024;

    // Use true SIMD path: eval_compiled_batch_soa_raw processes 8 points
    // at a time with direct f32x8 loads from SoA arrays
    let _aligned_count = (count_usize + 7) & !7;

    // Ensure we have enough padding for SIMD (read up to aligned_count)
    // The raw function handles the 8-wide alignment internally
    if count_usize >= PARALLEL_THRESHOLD {
        // Parallel: split into chunks, each chunk uses SIMD internally
        const CHUNK_SIZE: usize = 4096;

        dist_slice
            .par_chunks_mut(CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * CHUNK_SIZE;
                let chunk_len = chunk.len();
                let _simd_count = (chunk_len + 7) & !7;

                // Use SIMD for full 8-wide groups
                let simd_iters = chunk_len / 8;
                for s in 0..simd_iters {
                    let base = start + s * 8;
                    let local_base = s * 8;

                    let vx = wide::f32x8::from([
                        x_slice[base],
                        x_slice[base + 1],
                        x_slice[base + 2],
                        x_slice[base + 3],
                        x_slice[base + 4],
                        x_slice[base + 5],
                        x_slice[base + 6],
                        x_slice[base + 7],
                    ]);
                    let vy = wide::f32x8::from([
                        y_slice[base],
                        y_slice[base + 1],
                        y_slice[base + 2],
                        y_slice[base + 3],
                        y_slice[base + 4],
                        y_slice[base + 5],
                        y_slice[base + 6],
                        y_slice[base + 7],
                    ]);
                    let vz = wide::f32x8::from([
                        z_slice[base],
                        z_slice[base + 1],
                        z_slice[base + 2],
                        z_slice[base + 3],
                        z_slice[base + 4],
                        z_slice[base + 5],
                        z_slice[base + 6],
                        z_slice[base + 7],
                    ]);

                    let p = crate::compiled::Vec3x8 {
                        x: vx,
                        y: vy,
                        z: vz,
                    };
                    let distances = crate::compiled::eval_compiled_simd(&comp, p);
                    let arr: [f32; 8] = distances.into();
                    chunk[local_base..local_base + 8].copy_from_slice(&arr);
                }

                // Handle remainder scalar
                let remainder_start = simd_iters * 8;
                for local_i in remainder_start..chunk_len {
                    let i = start + local_i;
                    let p = glam::Vec3::new(x_slice[i], y_slice[i], z_slice[i]);
                    chunk[local_i] = eval_compiled(&comp, p);
                }
            });
    } else {
        // Sequential: use raw SIMD path for full groups, scalar for remainder
        crate::compiled::eval_compiled_batch_soa_raw(&comp, x, y, z, distances, count_usize);
    }

    BatchResult {
        count,
        result: SdfResult::Ok,
    }
}

// ============================================================================
// Gradient (Normal) Evaluation - SoA Layout (THE DEEP FRIED PATH)
// ============================================================================

/// Batch evaluate Distance AND Gradient using SoA layout - THE ULTIMATE DEEP FRIED PATH
///
/// This function computes both distance and gradient (surface normal direction)
/// for all points in a single call, using SIMD acceleration.
///
/// # Arguments
/// - `compiled`: Compiled SDF handle
/// - `x`, `y`, `z`: Input position arrays (SoA layout)
/// - `nx`, `ny`, `nz`: Output gradient/normal arrays (SoA layout)
/// - `dist`: Output distance array (optional, can be null)
/// - `count`: Number of points
///
/// # Safety
///
/// - `x`, `y`, `z` must each point to a valid array of at least `count` f32 values.
/// - `nx`, `ny`, `nz` must each point to a writable array of at least `count` f32 values.
/// - All six of the above pointers must be non-null and remain valid for the duration of
///   the call. The output slices must not alias each other or the input slices.
/// - `dist` may be null (output is skipped), but if non-null it must point to a writable
///   array of at least `count` f32 values.
/// - Arrays should be 32-byte aligned for AVX2 (recommended).
///
/// # Performance
/// - ~4x slower than distance-only (7 SDF evals vs 1)
/// - Still achieves 150M+ points/sec on modern CPU
/// - For 1M points: ~6ms
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_gradient_soa(
    compiled: CompiledHandle,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    nx: *mut f32,
    ny: *mut f32,
    nz: *mut f32,
    dist: *mut f32,
    count: u32,
) -> BatchResult {
    // Null pointer checks
    if x.is_null() || y.is_null() || z.is_null() || nx.is_null() || ny.is_null() || nz.is_null() {
        return BatchResult {
            count: 0,
            result: SdfResult::NullPointer,
        };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => {
            return BatchResult {
                count: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    let count_usize = count as usize;
    // SAFETY: Caller guarantees x, y, z are non-null pointers to valid arrays of at least count
    // f32 values each (SoA layout). Null checks performed above for x, y, z. Pointers remain
    // valid for the duration of this call and do not alias the output slices.
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    // SAFETY: Caller guarantees nx, ny, nz are non-null, writable pointers to at least count f32
    // values each. Null checks performed above for nx, ny, nz. Pointers remain valid and do not
    // alias each other or the input slices.
    let nx_slice = slice::from_raw_parts_mut(nx, count_usize);
    let ny_slice = slice::from_raw_parts_mut(ny, count_usize);
    let nz_slice = slice::from_raw_parts_mut(nz, count_usize);
    // SAFETY: dist may legitimately be null (optional output). If non-null, caller guarantees it
    // points to a writable array of at least count f32 values that remains valid for this call.
    let mut dist_slice = if dist.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(dist, count_usize))
    };

    use crate::compiled::{eval_distance_and_gradient_simd, Vec3x8};
    use rayon::prelude::*;
    use wide::f32x8;

    const EPSILON: f32 = 0.001;
    const PARALLEL_THRESHOLD: usize = 1024;
    const CHUNK_SIZE: usize = 4096;

    if count_usize >= PARALLEL_THRESHOLD {
        // Parallel processing with safe chunk access
        let chunks: Vec<(usize, usize)> = (0..count_usize)
            .step_by(CHUNK_SIZE)
            .map(|start| (start, (start + CHUNK_SIZE).min(count_usize)))
            .collect();

        chunks.into_par_iter().for_each(|(start, end)| {
            let chunk_len = end - start;
            let simd_iters = chunk_len / 8;

            // SIMD loop (8 points at a time)
            for s in 0..simd_iters {
                let base = start + s * 8;

                // Load 8 points from SoA arrays
                let vx = f32x8::from([
                    x_slice[base],
                    x_slice[base + 1],
                    x_slice[base + 2],
                    x_slice[base + 3],
                    x_slice[base + 4],
                    x_slice[base + 5],
                    x_slice[base + 6],
                    x_slice[base + 7],
                ]);
                let vy = f32x8::from([
                    y_slice[base],
                    y_slice[base + 1],
                    y_slice[base + 2],
                    y_slice[base + 3],
                    y_slice[base + 4],
                    y_slice[base + 5],
                    y_slice[base + 6],
                    y_slice[base + 7],
                ]);
                let vz = f32x8::from([
                    z_slice[base],
                    z_slice[base + 1],
                    z_slice[base + 2],
                    z_slice[base + 3],
                    z_slice[base + 4],
                    z_slice[base + 5],
                    z_slice[base + 6],
                    z_slice[base + 7],
                ]);

                let p = Vec3x8 {
                    x: vx,
                    y: vy,
                    z: vz,
                };

                // Compute distance and gradient in one call
                let (d, gx, gy, gz) = eval_distance_and_gradient_simd(&comp, p, EPSILON);

                // Store results
                let d_arr: [f32; 8] = d.to_array();
                let gx_arr: [f32; 8] = gx.to_array();
                let gy_arr: [f32; 8] = gy.to_array();
                let gz_arr: [f32; 8] = gz.to_array();

                for i in 0..8 {
                    // Normalize gradient to get unit normal
                    let len =
                        (gx_arr[i] * gx_arr[i] + gy_arr[i] * gy_arr[i] + gz_arr[i] * gz_arr[i])
                            .sqrt();
                    let inv_len = if len > 1e-8 { 1.0 / len } else { 0.0 };

                    // SAFETY: Output slices were created from raw pointers with validated count.
                    // Loop bounds (base + i < count) ensure we stay within the allocated region.
                    // cast_mut() on a shared-slice pointer is safe here because Rayon chunk
                    // partitioning ensures no two tasks write to the same index concurrently.
                    *nx_slice.as_ptr().add(base + i).cast_mut() = gx_arr[i] * inv_len;
                    *ny_slice.as_ptr().add(base + i).cast_mut() = gy_arr[i] * inv_len;
                    *nz_slice.as_ptr().add(base + i).cast_mut() = gz_arr[i] * inv_len;
                }

                if let Some(ref ds) = dist_slice {
                    for i in 0..8 {
                        // SAFETY: Output slices were created from raw pointers with validated count.
                        // Loop bounds (base + i < count) ensure we stay within the allocated region.
                        *ds.as_ptr().add(base + i).cast_mut() = d_arr[i];
                    }
                }
            }

            // Handle remainder (non-SIMD)
            let remainder_start = start + simd_iters * 8;
            for i in remainder_start..end {
                let p = glam::Vec3::new(x_slice[i], y_slice[i], z_slice[i]);
                let d = eval_compiled(&comp, p);

                // Finite difference gradient
                let eps = EPSILON;
                let dx = eval_compiled(&comp, p + glam::Vec3::X * eps)
                    - eval_compiled(&comp, p - glam::Vec3::X * eps);
                let dy = eval_compiled(&comp, p + glam::Vec3::Y * eps)
                    - eval_compiled(&comp, p - glam::Vec3::Y * eps);
                let dz = eval_compiled(&comp, p + glam::Vec3::Z * eps)
                    - eval_compiled(&comp, p - glam::Vec3::Z * eps);

                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let inv_len = if len > 1e-8 { 1.0 / len } else { 0.0 };

                // SAFETY: Output slices were created from raw pointers with validated count.
                // Loop bounds (remainder_start <= i < end <= count) ensure we stay within the
                // allocated region. cast_mut() is safe because Rayon chunk partitioning ensures
                // no two tasks write to the same index concurrently.
                *nx_slice.as_ptr().add(i).cast_mut() = dx * inv_len;
                *ny_slice.as_ptr().add(i).cast_mut() = dy * inv_len;
                *nz_slice.as_ptr().add(i).cast_mut() = dz * inv_len;

                if let Some(ref ds) = dist_slice {
                    // SAFETY: Output slices were created from raw pointers with validated count.
                    // Loop bounds (remainder_start <= i < end <= count) ensure we stay in bounds.
                    *ds.as_ptr().add(i).cast_mut() = d;
                }
            }
        });
    } else {
        // Sequential for small batches
        for i in 0..count_usize {
            let p = glam::Vec3::new(x_slice[i], y_slice[i], z_slice[i]);
            let d = eval_compiled(&comp, p);

            let eps = EPSILON;
            let dx = eval_compiled(&comp, p + glam::Vec3::X * eps)
                - eval_compiled(&comp, p - glam::Vec3::X * eps);
            let dy = eval_compiled(&comp, p + glam::Vec3::Y * eps)
                - eval_compiled(&comp, p - glam::Vec3::Y * eps);
            let dz = eval_compiled(&comp, p + glam::Vec3::Z * eps)
                - eval_compiled(&comp, p - glam::Vec3::Z * eps);

            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            let inv_len = if len > 1e-8 { 1.0 / len } else { 0.0 };

            nx_slice[i] = dx * inv_len;
            ny_slice[i] = dy * inv_len;
            nz_slice[i] = dz * inv_len;

            if let Some(ref mut ds) = dist_slice {
                ds[i] = d;
            }
        }
    }

    BatchResult {
        count,
        result: SdfResult::Ok,
    }
}

// ============================================================================
// Animated Compiled Evaluation (Zero-Copy)
// ============================================================================

/// Evaluate compiled SDF with animation transform (zero-allocation per frame)
///
/// Instead of rebuilding the SDF tree every frame, this applies the inverse
/// transform to the query point and evaluates against the pre-compiled base shape.
///
/// # Parameters
/// - `compiled`: Pre-compiled base shape
/// - `params`: Pointer to AnimationParams struct (36 bytes)
/// - `x`, `y`, `z`: Query point coordinates
///
/// # Safety
///
/// - `params` must point to a valid, properly initialized `AnimationParams` struct that remains
///   valid for the duration of the call. The pointer must be non-null and correctly aligned.
///
/// # Returns
/// Signed distance at the query point
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_animated_compiled(
    compiled: CompiledHandle,
    params: *const AnimationParams,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    if params.is_null() {
        return f32::MAX;
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return f32::MAX,
    };

    // SAFETY: Caller guarantees params is a non-null pointer to a valid, initialized
    // AnimationParams struct. Null check performed above. The struct remains valid for
    // this call. We only read through the reference; no mutation occurs.
    let params_ref = &*params;
    crate::animation::eval_animated_compiled(&comp, params_ref, glam::Vec3::new(x, y, z))
}

/// Batch evaluate compiled SDF with animation transform using SoA layout
///
/// Combines zero-copy animation with SIMD SoA evaluation for maximum
/// throughput on animated particle systems.
///
/// # Parameters
/// - `compiled`: Pre-compiled base shape
/// - `params`: Pointer to AnimationParams struct (shared for all points)
/// - `x`, `y`, `z`: Input position arrays (SoA layout)
/// - `distances`: Output array (caller-allocated)
/// - `count`: Number of points
///
/// # Safety
///
/// - `params` must point to a valid, properly initialized `AnimationParams` struct that remains
///   valid for the duration of the call. Must be non-null and correctly aligned.
/// - `x`, `y`, and `z` must each point to a valid array of at least `count` f32 values.
/// - `distances` must point to a writable array of at least `count` f32 values.
/// - All five pointers must be non-null and remain valid for the entire duration of the call.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_animated_batch_soa(
    compiled: CompiledHandle,
    params: *const AnimationParams,
    x: *const f32,
    y: *const f32,
    z: *const f32,
    distances: *mut f32,
    count: u32,
) -> BatchResult {
    if params.is_null() || x.is_null() || y.is_null() || z.is_null() || distances.is_null() {
        return BatchResult {
            count: 0,
            result: SdfResult::NullPointer,
        };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => {
            return BatchResult {
                count: 0,
                result: SdfResult::InvalidHandle,
            }
        }
    };

    // SAFETY: Caller guarantees params is a non-null pointer to a valid, initialized
    // AnimationParams struct. Null check performed above. The struct remains valid for
    // this call. We only read through the reference; no mutation occurs.
    let params_ref = &*params;
    let count_usize = count as usize;
    // SAFETY: Caller guarantees x, y, z are non-null pointers to valid arrays of at least count
    // f32 values each (SoA layout). Null checks performed above. Pointers remain valid for this
    // call and do not alias the output distances slice.
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    // SAFETY: Caller guarantees distances is a non-null, writable pointer to at least count f32
    // values. Null check performed above. Does not alias x, y, or z slices.
    let dist_slice = slice::from_raw_parts_mut(distances, count_usize);

    // Apply inverse animation transform to each point, then evaluate via SIMD
    let has_transform =
        params_ref.has_translation() || params_ref.has_rotation() || params_ref.has_scale();

    if !has_transform {
        // No animation transform: use direct SIMD SoA path
        crate::compiled::eval_compiled_batch_soa_raw(&comp, x, y, z, distances, count_usize);
    } else {
        // Transform points then evaluate
        // For SIMD efficiency, we transform in 8-wide batches
        let simd_iters = count_usize / 8;

        for s in 0..simd_iters {
            let base = s * 8;

            // Load 8 points
            let mut px = [0.0f32; 8];
            let mut py = [0.0f32; 8];
            let mut pz = [0.0f32; 8];
            let mut scale_corrections = [1.0f32; 8];

            for i in 0..8 {
                let point =
                    glam::Vec3::new(x_slice[base + i], y_slice[base + i], z_slice[base + i]);
                let (tp, sc) = params_ref.transform_point(point);
                px[i] = tp.x;
                py[i] = tp.y;
                pz[i] = tp.z;
                scale_corrections[i] = sc;
            }

            // Evaluate transformed points via SIMD
            let vx = wide::f32x8::from(px);
            let vy = wide::f32x8::from(py);
            let vz = wide::f32x8::from(pz);
            let p = crate::compiled::Vec3x8 {
                x: vx,
                y: vy,
                z: vz,
            };
            let raw_distances = crate::compiled::eval_compiled_simd(&comp, p);
            let raw_arr: [f32; 8] = raw_distances.into();

            // Apply scale correction
            for i in 0..8 {
                dist_slice[base + i] = raw_arr[i] * scale_corrections[i];
            }
        }

        // Handle remainder
        let remainder_start = simd_iters * 8;
        for i in remainder_start..count_usize {
            let point = glam::Vec3::new(x_slice[i], y_slice[i], z_slice[i]);
            dist_slice[i] = crate::animation::eval_animated_compiled(&comp, params_ref, point);
        }
    }

    BatchResult {
        count,
        result: SdfResult::Ok,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::compilation::{alice_sdf_compile, alice_sdf_free_compiled};
    use crate::ffi::memory::alice_sdf_free;
    use crate::ffi::primitives::alice_sdf_sphere;

    #[test]
    fn test_compile_and_eval() {
        let sphere = alice_sdf_sphere(1.0);
        assert!(!sphere.is_null());

        let compiled = alice_sdf_compile(sphere);
        assert!(!compiled.is_null());

        // Eval at origin (inside)
        let d = alice_sdf_eval_compiled(compiled, 0.0, 0.0, 0.0);
        assert!((d + 1.0).abs() < 0.001);

        // Eval on surface
        let d = alice_sdf_eval_compiled(compiled, 1.0, 0.0, 0.0);
        assert!(d.abs() < 0.001);

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }

    #[test]
    fn test_batch_eval() {
        let sphere = alice_sdf_sphere(1.0);
        let compiled = alice_sdf_compile(sphere);

        let points: [f32; 9] = [
            0.0, 0.0, 0.0, // inside
            1.0, 0.0, 0.0, // on surface
            2.0, 0.0, 0.0, // outside
        ];
        let mut distances: [f32; 3] = [0.0; 3];

        // SAFETY: points is a valid local array of 9 f32 (3 points x 3 floats). distances is a
        // valid local array of 3 f32. Both remain valid for the duration of the call. compiled is
        // a valid handle obtained above.
        unsafe {
            let result =
                alice_sdf_eval_compiled_batch(compiled, points.as_ptr(), distances.as_mut_ptr(), 3);
            assert_eq!(result.result, SdfResult::Ok);
            assert_eq!(result.count, 3);
        }

        assert!((distances[0] + 1.0).abs() < 0.001); // inside
        assert!(distances[1].abs() < 0.001); // surface
        assert!((distances[2] - 1.0).abs() < 0.001); // outside

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }

    #[test]
    fn test_animated_compiled_eval() {
        let sphere = alice_sdf_sphere(1.0);
        let compiled = alice_sdf_compile(sphere);

        // No animation: origin should be inside sphere
        let params = AnimationParams {
            scale: 1.0,
            ..Default::default()
        };
        // SAFETY: params is a valid, initialized AnimationParams local struct. compiled is a
        // valid handle. The reference &params lives for the duration of the call.
        unsafe {
            let d = alice_sdf_eval_animated_compiled(compiled, &params, 0.0, 0.0, 0.0);
            assert!((d + 1.0).abs() < 0.01);
        }

        // Translated by 5 on X: origin should be outside
        let params = AnimationParams {
            translate_x: 5.0,
            scale: 1.0,
            ..Default::default()
        };
        // SAFETY: params is a valid, initialized AnimationParams local struct. compiled is a
        // valid handle. The reference &params lives for the duration of the call.
        unsafe {
            let d = alice_sdf_eval_animated_compiled(compiled, &params, 0.0, 0.0, 0.0);
            assert!(d > 0.0);
        }

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }

    #[test]
    fn test_animated_batch_soa() {
        let sphere = alice_sdf_sphere(1.0);
        let compiled = alice_sdf_compile(sphere);

        let params = AnimationParams {
            translate_x: 5.0,
            scale: 1.0,
            ..Default::default()
        };

        let x: [f32; 4] = [0.0, 5.0, 10.0, 5.0];
        let y: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
        let z: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        let mut distances: [f32; 4] = [0.0; 4];

        // SAFETY: params, x, y, z, distances are all valid local arrays/structs. x, y, z each
        // have 4 f32 elements; distances has 4 f32 elements; all are valid for the call duration.
        // compiled is a valid handle obtained above.
        unsafe {
            let result = alice_sdf_eval_animated_batch_soa(
                compiled,
                &params,
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                distances.as_mut_ptr(),
                4,
            );
            assert_eq!(result.result, SdfResult::Ok);
        }

        // x=0 with translate_x=5 -> sphere center at 5, query at 0 -> distance ~4
        assert!(distances[0] > 3.0);
        // x=5 with translate_x=5 -> sphere center at 5, query at 5 -> inside (-1)
        assert!((distances[1] + 1.0).abs() < 0.01);

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }

    #[test]
    fn test_soa_eval() {
        let sphere = alice_sdf_sphere(1.0);
        let compiled = alice_sdf_compile(sphere);

        let x: [f32; 4] = [0.0, 1.0, 2.0, 0.5];
        let y: [f32; 4] = [0.0, 0.0, 0.0, 0.5];
        let z: [f32; 4] = [0.0, 0.0, 0.0, 0.5];
        let mut distances: [f32; 4] = [0.0; 4];

        // SAFETY: x, y, z each have 4 f32 elements (SoA layout). distances has 4 f32 elements.
        // All local arrays are valid and live for the duration of the call. compiled is a valid
        // handle obtained above.
        unsafe {
            let result = alice_sdf_eval_soa(
                compiled,
                x.as_ptr(),
                y.as_ptr(),
                z.as_ptr(),
                distances.as_mut_ptr(),
                4,
            );
            assert_eq!(result.result, SdfResult::Ok);
        }

        assert!((distances[0] + 1.0).abs() < 0.001); // origin
        assert!(distances[1].abs() < 0.001); // surface
        assert!((distances[2] - 1.0).abs() < 0.001); // outside

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }
}

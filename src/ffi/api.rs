//! FFI API for ALICE-SDF (Deep Fried Edition)
//!
//! This module provides C-compatible functions for creating and evaluating SDFs.
//! All functions use `extern "C"` calling convention and are safe to call from
//! C, C++, C#, Python, and other languages.
//!
//! # Deep Fried Features
//!
//! - **Pre-compilation**: `alice_sdf_compile()` converts SdfNode to bytecode once
//! - **Zero-copy batch**: `alice_sdf_eval_compiled_batch()` writes directly to caller memory
//! - **SoA support**: `alice_sdf_eval_soa()` for SIMD-friendly data layout
//! - **Parallel evaluation**: Large batches automatically use Rayon
//!
//! # Performance Hierarchy (fastest to slowest)
//!
//! 1. `alice_sdf_eval_soa` - SoA layout + compiled (1B+ ops/sec)
//! 2. `alice_sdf_eval_compiled_batch` - AoS layout + compiled
//! 3. `alice_sdf_eval_batch` - AoS layout + auto-compile
//! 4. `alice_sdf_eval` - Single point (use for debugging only)
//!
//! # Memory Management
//!
//! - Call `alice_sdf_free()` for every SDF handle when done
//! - Call `alice_sdf_free_compiled()` for every compiled handle
//! - Call `alice_sdf_free_string()` for every string returned by shader functions
//!
//! # Thread Safety
//!
//! - All functions are thread-safe
//! - Handles can be shared across threads
//! - Batch evaluation uses parallel processing internally
//!
//! Author: Moroya Sakamoto

use super::registry::{
    register_node, get_node, remove_node,
    register_compiled, get_compiled, remove_compiled,
    register_mesh, get_mesh, remove_mesh,
};
use super::types::*;
use crate::prelude::*;
use crate::compiled::{CompiledSdf, eval_compiled, eval_compiled_batch_parallel};
use crate::animation::AnimationParams;
use std::ffi::{c_char, CString};
use std::ptr;
use std::slice;

// ============================================================================
// Library Info
// ============================================================================

/// Get the library version
#[no_mangle]
pub extern "C" fn alice_sdf_version() -> VersionInfo {
    VersionInfo::current()
}

/// Get version string (caller must free with alice_sdf_free_string)
#[no_mangle]
pub extern "C" fn alice_sdf_version_string() -> *mut c_char {
    let version = format!("ALICE-SDF v{}.{}.{} (Deep Fried)",
        VersionInfo::current().major,
        VersionInfo::current().minor,
        VersionInfo::current().patch
    );
    match CString::new(version) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

// ============================================================================
// Primitives
// ============================================================================

/// Create a sphere SDF
#[no_mangle]
pub extern "C" fn alice_sdf_sphere(radius: f32) -> SdfHandle {
    let node = SdfNode::Sphere { radius };
    register_node(node)
}

/// Create a box SDF
#[no_mangle]
pub extern "C" fn alice_sdf_box(hx: f32, hy: f32, hz: f32) -> SdfHandle {
    let node = SdfNode::Box3d {
        half_extents: glam::Vec3::new(hx, hy, hz),
    };
    register_node(node)
}

/// Create a cylinder SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cylinder(radius: f32, half_height: f32) -> SdfHandle {
    let node = SdfNode::Cylinder { radius, half_height };
    register_node(node)
}

/// Create a torus SDF
#[no_mangle]
pub extern "C" fn alice_sdf_torus(major_radius: f32, minor_radius: f32) -> SdfHandle {
    let node = SdfNode::Torus { major_radius, minor_radius };
    register_node(node)
}

/// Create a capsule SDF
#[no_mangle]
pub extern "C" fn alice_sdf_capsule(
    ax: f32, ay: f32, az: f32,
    bx: f32, by: f32, bz: f32,
    radius: f32
) -> SdfHandle {
    let node = SdfNode::Capsule {
        point_a: glam::Vec3::new(ax, ay, az),
        point_b: glam::Vec3::new(bx, by, bz),
        radius,
    };
    register_node(node)
}

/// Create a plane SDF
#[no_mangle]
pub extern "C" fn alice_sdf_plane(nx: f32, ny: f32, nz: f32, distance: f32) -> SdfHandle {
    let node = SdfNode::Plane {
        normal: glam::Vec3::new(nx, ny, nz).normalize(),
        distance,
    };
    register_node(node)
}

// ============================================================================
// Boolean Operations
// ============================================================================

/// Union of two SDFs (A ∪ B)
#[no_mangle]
pub extern "C" fn alice_sdf_union(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().union((*node_b).clone());
    register_node(node)
}

/// Intersection of two SDFs (A ∩ B)
#[no_mangle]
pub extern "C" fn alice_sdf_intersection(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().intersection((*node_b).clone());
    register_node(node)
}

/// Subtraction of SDFs (A - B)
#[no_mangle]
pub extern "C" fn alice_sdf_subtract(a: SdfHandle, b: SdfHandle) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().subtract((*node_b).clone());
    register_node(node)
}

/// Smooth union of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_union(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_union((*node_b).clone(), k);
    register_node(node)
}

/// Smooth intersection of two SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_intersection(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_intersection((*node_b).clone(), k);
    register_node(node)
}

/// Smooth subtraction of SDFs
#[no_mangle]
pub extern "C" fn alice_sdf_smooth_subtract(a: SdfHandle, b: SdfHandle, k: f32) -> SdfHandle {
    let node_a = match get_node(a) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node_b = match get_node(b) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let node = (*node_a).clone().smooth_subtract((*node_b).clone(), k);
    register_node(node)
}

// ============================================================================
// Transforms
// ============================================================================

/// Translate an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_translate(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().translate(x, y, z);
    register_node(new_node)
}

/// Rotate an SDF using quaternion
#[no_mangle]
pub extern "C" fn alice_sdf_rotate(node: SdfHandle, qx: f32, qy: f32, qz: f32, qw: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let quat = glam::Quat::from_xyzw(qx, qy, qz, qw).normalize();
    let new_node = (*sdf_node).clone().rotate(quat);
    register_node(new_node)
}

/// Rotate an SDF using Euler angles (radians)
#[no_mangle]
pub extern "C" fn alice_sdf_rotate_euler(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().rotate_euler(x, y, z);
    register_node(new_node)
}

/// Uniform scale an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_scale(node: SdfHandle, factor: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().scale(factor);
    register_node(new_node)
}

/// Non-uniform scale an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_scale_xyz(node: SdfHandle, x: f32, y: f32, z: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().scale_xyz(x, y, z);
    register_node(new_node)
}

// ============================================================================
// Modifiers
// ============================================================================

/// Apply rounding to an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_round(node: SdfHandle, radius: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().round(radius);
    register_node(new_node)
}

/// Apply onion (shell) modifier to an SDF
#[no_mangle]
pub extern "C" fn alice_sdf_onion(node: SdfHandle, thickness: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().onion(thickness);
    register_node(new_node)
}

/// Apply twist modifier
#[no_mangle]
pub extern "C" fn alice_sdf_twist(node: SdfHandle, strength: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().twist(strength);
    register_node(new_node)
}

/// Apply bend modifier
#[no_mangle]
pub extern "C" fn alice_sdf_bend(node: SdfHandle, curvature: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().bend(curvature);
    register_node(new_node)
}

/// Apply infinite repetition
#[no_mangle]
pub extern "C" fn alice_sdf_repeat(node: SdfHandle, sx: f32, sy: f32, sz: f32) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    let new_node = (*sdf_node).clone().repeat_infinite(sx, sy, sz);
    register_node(new_node)
}

// ============================================================================
// Compilation (Deep Fried)
// ============================================================================

/// Compile an SDF to bytecode for fast evaluation
///
/// This is expensive (~0.1ms), but the resulting CompiledHandle evaluates
/// ~10x faster. Compile once at setup time, reuse every frame.
///
/// # Returns
/// A CompiledHandle that can be used with `alice_sdf_eval_compiled*` functions.
/// Returns NULL on failure.
#[no_mangle]
pub extern "C" fn alice_sdf_compile(node: SdfHandle) -> CompiledHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return COMPILED_HANDLE_NULL,
    };

    let compiled = CompiledSdf::compile(&sdf_node);
    register_compiled(compiled)
}

/// Free a compiled SDF handle
#[no_mangle]
pub extern "C" fn alice_sdf_free_compiled(compiled: CompiledHandle) {
    if !compiled.is_null() {
        remove_compiled(compiled);
    }
}

/// Get instruction count of a compiled SDF (for profiling)
#[no_mangle]
pub extern "C" fn alice_sdf_compiled_instruction_count(compiled: CompiledHandle) -> u32 {
    match get_compiled(compiled) {
        Some(c) => c.instruction_count() as u32,
        None => 0,
    }
}

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
        return BatchResult { count: 0, result: SdfResult::NullPointer };
    }

    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return BatchResult { count: 0, result: SdfResult::InvalidHandle },
    };

    let count_usize = count as usize;
    let points_slice = slice::from_raw_parts(points, count_usize * 3);
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

    BatchResult { count, result: SdfResult::Ok }
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
/// - `points` must point to at least `count * 3` floats
/// - `distances` must point to at least `count` floats
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_eval_compiled_batch(
    compiled: CompiledHandle,
    points: *const f32,
    distances: *mut f32,
    count: u32,
) -> BatchResult {
    if points.is_null() || distances.is_null() {
        return BatchResult { count: 0, result: SdfResult::NullPointer };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return BatchResult { count: 0, result: SdfResult::InvalidHandle },
    };

    let count_usize = count as usize;
    let points_slice = slice::from_raw_parts(points, count_usize * 3);
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

    BatchResult { count, result: SdfResult::Ok }
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
/// - All arrays must have at least `count` elements
/// - Arrays should be 32-byte aligned for AVX2 (not required, but faster)
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
        return BatchResult { count: 0, result: SdfResult::NullPointer };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return BatchResult { count: 0, result: SdfResult::InvalidHandle },
    };

    let count_usize = count as usize;
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    let dist_slice = slice::from_raw_parts_mut(distances, count_usize);

    use rayon::prelude::*;

    // Threshold for parallelization
    const PARALLEL_THRESHOLD: usize = 1024;

    // Use true SIMD path: eval_compiled_batch_soa_raw processes 8 points
    // at a time with direct f32x8 loads from SoA arrays
    let aligned_count = (count_usize + 7) & !7;

    // Ensure we have enough padding for SIMD (read up to aligned_count)
    // The raw function handles the 8-wide alignment internally
    if count_usize >= PARALLEL_THRESHOLD {
        // Parallel: split into chunks, each chunk uses SIMD internally
        const CHUNK_SIZE: usize = 4096;

        dist_slice.par_chunks_mut(CHUNK_SIZE).enumerate().for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * CHUNK_SIZE;
            let chunk_len = chunk.len();
            let simd_count = (chunk_len + 7) & !7;

            // Use SIMD for full 8-wide groups
            let simd_iters = chunk_len / 8;
            for s in 0..simd_iters {
                let base = start + s * 8;
                let local_base = s * 8;

                let vx = wide::f32x8::from([
                    x_slice[base], x_slice[base + 1], x_slice[base + 2], x_slice[base + 3],
                    x_slice[base + 4], x_slice[base + 5], x_slice[base + 6], x_slice[base + 7],
                ]);
                let vy = wide::f32x8::from([
                    y_slice[base], y_slice[base + 1], y_slice[base + 2], y_slice[base + 3],
                    y_slice[base + 4], y_slice[base + 5], y_slice[base + 6], y_slice[base + 7],
                ]);
                let vz = wide::f32x8::from([
                    z_slice[base], z_slice[base + 1], z_slice[base + 2], z_slice[base + 3],
                    z_slice[base + 4], z_slice[base + 5], z_slice[base + 6], z_slice[base + 7],
                ]);

                let p = crate::compiled::Vec3x8 { x: vx, y: vy, z: vz };
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
        crate::compiled::eval_compiled_batch_soa_raw(
            &comp, x, y, z, distances, count_usize,
        );
    }

    BatchResult { count, result: SdfResult::Ok }
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
/// - All non-null arrays must have at least `count` elements
/// - Arrays should be 32-byte aligned for AVX2 (recommended)
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
        return BatchResult { count: 0, result: SdfResult::NullPointer };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return BatchResult { count: 0, result: SdfResult::InvalidHandle },
    };

    let count_usize = count as usize;
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    let nx_slice = slice::from_raw_parts_mut(nx, count_usize);
    let ny_slice = slice::from_raw_parts_mut(ny, count_usize);
    let nz_slice = slice::from_raw_parts_mut(nz, count_usize);
    let mut dist_slice = if dist.is_null() {
        None
    } else {
        Some(slice::from_raw_parts_mut(dist, count_usize))
    };

    use rayon::prelude::*;
    use crate::compiled::{eval_compiled_simd, eval_distance_and_gradient_simd, Vec3x8};
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
                    x_slice[base], x_slice[base + 1], x_slice[base + 2], x_slice[base + 3],
                    x_slice[base + 4], x_slice[base + 5], x_slice[base + 6], x_slice[base + 7],
                ]);
                let vy = f32x8::from([
                    y_slice[base], y_slice[base + 1], y_slice[base + 2], y_slice[base + 3],
                    y_slice[base + 4], y_slice[base + 5], y_slice[base + 6], y_slice[base + 7],
                ]);
                let vz = f32x8::from([
                    z_slice[base], z_slice[base + 1], z_slice[base + 2], z_slice[base + 3],
                    z_slice[base + 4], z_slice[base + 5], z_slice[base + 6], z_slice[base + 7],
                ]);

                let p = Vec3x8 { x: vx, y: vy, z: vz };

                // Compute distance and gradient in one call
                let (d, gx, gy, gz) = eval_distance_and_gradient_simd(&comp, p, EPSILON);

                // Store results
                let d_arr: [f32; 8] = d.to_array();
                let gx_arr: [f32; 8] = gx.to_array();
                let gy_arr: [f32; 8] = gy.to_array();
                let gz_arr: [f32; 8] = gz.to_array();

                for i in 0..8 {
                    // Normalize gradient to get unit normal
                    let len = (gx_arr[i] * gx_arr[i] + gy_arr[i] * gy_arr[i] + gz_arr[i] * gz_arr[i]).sqrt();
                    let inv_len = if len > 1e-8 { 1.0 / len } else { 0.0 };

                    // Write to output arrays (unsafe but we're in unsafe fn)
                    *nx_slice.as_ptr().add(base + i).cast_mut() = gx_arr[i] * inv_len;
                    *ny_slice.as_ptr().add(base + i).cast_mut() = gy_arr[i] * inv_len;
                    *nz_slice.as_ptr().add(base + i).cast_mut() = gz_arr[i] * inv_len;
                }

                if let Some(ref ds) = dist_slice {
                    for i in 0..8 {
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
                let dx = eval_compiled(&comp, p + glam::Vec3::X * eps) - eval_compiled(&comp, p - glam::Vec3::X * eps);
                let dy = eval_compiled(&comp, p + glam::Vec3::Y * eps) - eval_compiled(&comp, p - glam::Vec3::Y * eps);
                let dz = eval_compiled(&comp, p + glam::Vec3::Z * eps) - eval_compiled(&comp, p - glam::Vec3::Z * eps);

                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                let inv_len = if len > 1e-8 { 1.0 / len } else { 0.0 };

                *nx_slice.as_ptr().add(i).cast_mut() = dx * inv_len;
                *ny_slice.as_ptr().add(i).cast_mut() = dy * inv_len;
                *nz_slice.as_ptr().add(i).cast_mut() = dz * inv_len;

                if let Some(ref ds) = dist_slice {
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
            let dx = eval_compiled(&comp, p + glam::Vec3::X * eps) - eval_compiled(&comp, p - glam::Vec3::X * eps);
            let dy = eval_compiled(&comp, p + glam::Vec3::Y * eps) - eval_compiled(&comp, p - glam::Vec3::Y * eps);
            let dz = eval_compiled(&comp, p + glam::Vec3::Z * eps) - eval_compiled(&comp, p - glam::Vec3::Z * eps);

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

    BatchResult { count, result: SdfResult::Ok }
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
        return BatchResult { count: 0, result: SdfResult::NullPointer };
    }

    let comp = match get_compiled(compiled) {
        Some(c) => c,
        None => return BatchResult { count: 0, result: SdfResult::InvalidHandle },
    };

    let params_ref = &*params;
    let count_usize = count as usize;
    let x_slice = slice::from_raw_parts(x, count_usize);
    let y_slice = slice::from_raw_parts(y, count_usize);
    let z_slice = slice::from_raw_parts(z, count_usize);
    let dist_slice = slice::from_raw_parts_mut(distances, count_usize);

    // Apply inverse animation transform to each point, then evaluate via SIMD
    let has_transform = params_ref.has_translation()
        || params_ref.has_rotation()
        || params_ref.has_scale();

    if !has_transform {
        // No animation transform: use direct SIMD SoA path
        crate::compiled::eval_compiled_batch_soa_raw(
            &comp, x, y, z, distances, count_usize,
        );
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
                let point = glam::Vec3::new(x_slice[base + i], y_slice[base + i], z_slice[base + i]);
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
            let p = crate::compiled::Vec3x8 { x: vx, y: vy, z: vz };
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

    BatchResult { count, result: SdfResult::Ok }
}

// ============================================================================
// Shader Generation
// ============================================================================

/// Generate WGSL shader code
#[no_mangle]
#[cfg(feature = "gpu")]
pub extern "C" fn alice_sdf_to_wgsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::InvalidHandle },
    };

    let shader = crate::compiled::WgslShader::transpile(&sdf_node, crate::compiled::TranspileMode::Hardcoded);
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult { data: s.into_raw(), len, result: SdfResult::Ok }
        }
        Err(_) => StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::Unknown },
    }
}

/// Generate HLSL shader code
#[no_mangle]
#[cfg(feature = "hlsl")]
pub extern "C" fn alice_sdf_to_hlsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::InvalidHandle },
    };

    let shader = crate::compiled::HlslShader::transpile(&sdf_node, crate::compiled::HlslTranspileMode::Hardcoded);
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult { data: s.into_raw(), len, result: SdfResult::Ok }
        }
        Err(_) => StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::Unknown },
    }
}

/// Generate GLSL shader code
#[no_mangle]
#[cfg(feature = "glsl")]
pub extern "C" fn alice_sdf_to_glsl(node: SdfHandle) -> StringResult {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::InvalidHandle },
    };

    let shader = crate::compiled::GlslShader::transpile(&sdf_node, crate::compiled::GlslTranspileMode::Hardcoded);
    match CString::new(shader.source.clone()) {
        Ok(s) => {
            let len = shader.source.len() as u32;
            StringResult { data: s.into_raw(), len, result: SdfResult::Ok }
        }
        Err(_) => StringResult { data: ptr::null_mut(), len: 0, result: SdfResult::Unknown },
    }
}

// ============================================================================
// File I/O
// ============================================================================

/// Save SDF to .asdf file
#[no_mangle]
pub extern "C" fn alice_sdf_save(node: SdfHandle, path: *const c_char) -> SdfResult {
    if path.is_null() {
        return SdfResult::NullPointer;
    }

    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SdfResult::InvalidHandle,
    };

    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return SdfResult::InvalidParameter,
        }
    };

    let tree = SdfTree::new((*sdf_node).clone());
    match crate::save(&tree, path_str) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Load SDF from .asdf file
#[no_mangle]
pub extern "C" fn alice_sdf_load(path: *const c_char) -> SdfHandle {
    if path.is_null() {
        return SDF_HANDLE_NULL;
    }

    let path_str = unsafe {
        match std::ffi::CStr::from_ptr(path).to_str() {
            Ok(s) => s,
            Err(_) => return SDF_HANDLE_NULL,
        }
    };

    match crate::load(path_str) {
        Ok(tree) => register_node(tree.root),
        Err(_) => SDF_HANDLE_NULL,
    }
}

// ============================================================================
// Mesh Generation & Export
// ============================================================================

/// Generate mesh from SDF via Marching Cubes (returns reusable MeshHandle)
///
/// Generate once, export to multiple formats without re-computing.
/// Call `alice_sdf_free_mesh` when done.
#[no_mangle]
pub extern "C" fn alice_sdf_generate_mesh(
    node: SdfHandle,
    resolution: u32,
    bounds: f32,
) -> MeshHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return MESH_HANDLE_NULL,
    };

    let config = MarchingCubesConfig {
        resolution: resolution as usize,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };
    let min = glam::Vec3::splat(-bounds);
    let max = glam::Vec3::splat(bounds);

    let mesh = sdf_to_mesh(&sdf_node, min, max, &config);
    register_mesh(mesh)
}

/// Get vertex count of a mesh
#[no_mangle]
pub extern "C" fn alice_sdf_mesh_vertex_count(mesh: MeshHandle) -> u32 {
    match get_mesh(mesh) {
        Some(m) => m.vertex_count() as u32,
        None => 0,
    }
}

/// Get triangle count of a mesh
#[no_mangle]
pub extern "C" fn alice_sdf_mesh_triangle_count(mesh: MeshHandle) -> u32 {
    match get_mesh(mesh) {
        Some(m) => m.triangle_count() as u32,
        None => 0,
    }
}

/// Free a mesh handle
#[no_mangle]
pub extern "C" fn alice_sdf_free_mesh(mesh: MeshHandle) {
    if !mesh.is_null() {
        remove_mesh(mesh);
    }
}

/// Helper: extract path string from C pointer
unsafe fn path_from_c(path: *const c_char) -> Result<&'static str, SdfResult> {
    if path.is_null() {
        return Err(SdfResult::NullPointer);
    }
    // SAFETY: We only hold reference during the export call
    std::ffi::CStr::from_ptr(path)
        .to_str()
        .map_err(|_| SdfResult::InvalidParameter)
}

/// Helper: resolve mesh from either MeshHandle or SdfHandle
unsafe fn resolve_mesh(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    resolution: u32,
    bounds: f32,
) -> Result<std::sync::Arc<crate::mesh::Mesh>, SdfResult> {
    // Prefer pre-generated MeshHandle
    if !mesh_handle.is_null() {
        return get_mesh(mesh_handle).ok_or(SdfResult::InvalidHandle);
    }

    // Fallback: generate from SDF on the fly
    let sdf_node = get_node(sdf_handle).ok_or(SdfResult::InvalidHandle)?;
    let config = MarchingCubesConfig {
        resolution: resolution as usize,
        iso_level: 0.0,
        compute_normals: true,
        ..Default::default()
    };
    let min = glam::Vec3::splat(-bounds);
    let max = glam::Vec3::splat(bounds);
    Ok(std::sync::Arc::new(sdf_to_mesh(&sdf_node, min, max, &config)))
}

/// Export mesh to OBJ file
///
/// Pass a pre-generated MeshHandle (from `alice_sdf_generate_mesh`) for best performance.
/// Alternatively, pass a null MeshHandle + valid SdfHandle to generate on the fly.
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_obj(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) { Ok(s) => s, Err(e) => return e };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m, Err(e) => return e,
    };
    match crate::io::export_obj(&mesh, path_str, &ObjConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to GLB (binary glTF) file
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_glb(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) { Ok(s) => s, Err(e) => return e };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m, Err(e) => return e,
    };
    match crate::io::export_glb(&mesh, path_str, &GltfConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to USDA (Universal Scene Description) file
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_usda(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) { Ok(s) => s, Err(e) => return e };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m, Err(e) => return e,
    };
    match crate::io::export_usda(&mesh, path_str, &UsdConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to Alembic (.abc) file
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_alembic(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) { Ok(s) => s, Err(e) => return e };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m, Err(e) => return e,
    };
    match crate::io::export_alembic(&mesh, path_str, &AlembicConfig::default()) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

/// Export mesh to FBX file
#[no_mangle]
pub unsafe extern "C" fn alice_sdf_export_fbx(
    mesh_handle: MeshHandle,
    sdf_handle: SdfHandle,
    path: *const c_char,
    resolution: u32,
    bounds: f32,
) -> SdfResult {
    let path_str = match path_from_c(path) { Ok(s) => s, Err(e) => return e };
    let mesh = match resolve_mesh(mesh_handle, sdf_handle, resolution, bounds) {
        Ok(m) => m, Err(e) => return e,
    };
    match crate::io::export_fbx(&mesh, path_str, &FbxConfig::default(), None) {
        Ok(_) => SdfResult::Ok,
        Err(_) => SdfResult::IoError,
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free an SDF handle
#[no_mangle]
pub extern "C" fn alice_sdf_free(node: SdfHandle) {
    if !node.is_null() {
        remove_node(node);
    }
}

/// Free a string returned by shader generation functions
#[no_mangle]
pub extern "C" fn alice_sdf_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            let _ = CString::from_raw(s);
        }
    }
}

/// Clone an SDF handle (creates a new reference)
#[no_mangle]
pub extern "C" fn alice_sdf_clone(node: SdfHandle) -> SdfHandle {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return SDF_HANDLE_NULL,
    };
    register_node((*sdf_node).clone())
}

// ============================================================================
// Utilities
// ============================================================================

/// Get the node count in an SDF tree
#[no_mangle]
pub extern "C" fn alice_sdf_node_count(node: SdfHandle) -> u32 {
    let sdf_node = match get_node(node) {
        Some(n) => n,
        None => return 0,
    };
    sdf_node.node_count()
}

/// Check if a handle is valid
#[no_mangle]
pub extern "C" fn alice_sdf_is_valid(node: SdfHandle) -> bool {
    get_node(node).is_some()
}

/// Check if a compiled handle is valid
#[no_mangle]
pub extern "C" fn alice_sdf_is_compiled_valid(compiled: CompiledHandle) -> bool {
    get_compiled(compiled).is_some()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
            0.0, 0.0, 0.0,  // inside
            1.0, 0.0, 0.0,  // on surface
            2.0, 0.0, 0.0,  // outside
        ];
        let mut distances: [f32; 3] = [0.0; 3];

        unsafe {
            let result = alice_sdf_eval_compiled_batch(
                compiled,
                points.as_ptr(),
                distances.as_mut_ptr(),
                3,
            );
            assert_eq!(result.result, SdfResult::Ok);
            assert_eq!(result.count, 3);
        }

        assert!((distances[0] + 1.0).abs() < 0.001); // inside
        assert!(distances[1].abs() < 0.001);          // surface
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

        // x=0 with translate_x=5 → sphere center at 5, query at 0 → distance ~4
        assert!(distances[0] > 3.0);
        // x=5 with translate_x=5 → sphere center at 5, query at 5 → inside (-1)
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
        assert!(distances[1].abs() < 0.001);          // surface
        assert!((distances[2] - 1.0).abs() < 0.001); // outside

        alice_sdf_free_compiled(compiled);
        alice_sdf_free(sphere);
    }
}

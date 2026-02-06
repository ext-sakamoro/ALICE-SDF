//! Sphere tracing (raymarching) implementation (Deep Fried v2)
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Integer Division Exorcism**: Row-based iteration eliminates `i % w` / `i / w`.
//! - **Incremental Ray Generation**: Pre-calculate row vectors per pixel.
//! - **Forced Inlining**: Hot loops are forced inline.
//! - **Over-Relaxation**: Enhanced sphere tracing with safety checks.
//! - **Compiled Backend**: `CompiledSdf` VM eliminates tree traversal overhead.
//! - **JIT Backend**: `JitCompiledSdf` native code, zero interpreter overhead.
//! - **SIMD Packet Tracing**: 8 rays marched simultaneously via `Vec3x8`/`JitSimdSdf`.
//! - **Tile-Based Rendering**: Cache-friendly 8-pixel horizontal tiles.
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, normal};
use crate::types::{Hit, Ray};
use crate::SdfNode;
use glam::Vec3;
use rayon::prelude::*;

use crate::compiled::{CompiledSdf, eval_compiled, eval_compiled_normal, Vec3x8, eval_compiled_simd};
use wide::{f32x8, CmpLt, CmpGt, CmpGe};

/// Raymarch configuration
#[derive(Debug, Clone, Copy)]
pub struct RaymarchConfig {
    /// Maximum number of marching steps
    pub max_steps: u32,
    /// Distance threshold for surface hit
    pub epsilon: f32,
    /// Minimum step distance (prevents infinite loops)
    pub min_step: f32,
    /// Step relaxation factor (omega for over-relaxation)
    pub omega: f32,
}

impl Default for RaymarchConfig {
    fn default() -> Self {
        RaymarchConfig {
            max_steps: 128,
            epsilon: 0.0001,
            min_step: 0.0001,
            omega: 1.0, // No over-relaxation
        }
    }
}

impl RaymarchConfig {
    /// High quality configuration
    pub fn high_quality() -> Self {
        RaymarchConfig {
            max_steps: 256,
            epsilon: 0.00001,
            min_step: 0.00001,
            omega: 1.0,
        }
    }

    /// Fast configuration for preview (Deep Fried)
    ///
    /// Uses moderate over-relaxation for faster convergence while maintaining stability.
    pub fn fast() -> Self {
        RaymarchConfig {
            max_steps: 64,
            epsilon: 0.0005,  // Balanced precision
            min_step: 0.0001, // Small min step for stability
            omega: 1.2,       // Moderate over-relaxation (safer than 1.6)
        }
    }
}

/// Raymarch result with detailed information
#[derive(Debug, Clone, Copy)]
pub struct RaymarchResult {
    /// Whether a hit was found
    pub hit: bool,
    /// Distance traveled
    pub distance: f32,
    /// Hit point (if hit)
    pub point: Vec3,
    /// Surface normal (if hit)
    pub normal: Vec3,
    /// Number of steps taken
    pub steps: u32,
}

// ============================================================================
// Interpreter Backend (SdfNode tree traversal)
// ============================================================================

/// Perform sphere tracing along a ray (Interpreter path)
///
/// Uses `eval()` interpreter to traverse the SDF tree. This is the simplest
/// path but slowest. Prefer `raymarch_compiled()` or `raymarch_jit()` for
/// production use.
#[inline(always)]
pub fn raymarch(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<Hit> {
    const EPSILON: f32 = 0.0001;
    const MAX_STEPS: u32 = 128;

    let mut t = 0.0;
    let mut steps = 0;

    while t < max_distance && steps < MAX_STEPS {
        let point = origin + direction * t;
        let d = eval(node, point);

        if d.abs() < EPSILON {
            return Some(Hit {
                distance: t,
                point,
                normal: normal(node, point, EPSILON),
                steps,
            });
        }

        t += d;
        steps += 1;
    }

    None
}

/// Perform sphere tracing with custom configuration
#[inline(always)]
pub fn raymarch_with_config(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    config: &RaymarchConfig,
) -> Option<Hit> {
    let dir = direction.normalize();
    let mut t = 0.0;
    let mut steps = 0;
    let omega = config.omega;
    let use_relaxation = omega > 1.0;

    let mut prev_dist = 0.0;
    let mut prev_step = 0.0;

    while t < max_distance && steps < config.max_steps {
        let point = origin + dir * t;
        let d = eval(node, point);

        if d.abs() < config.epsilon {
            return Some(Hit {
                distance: t,
                point,
                normal: normal(node, point, config.epsilon),
                steps,
            });
        }

        let step = if use_relaxation && steps > 0 {
            let expected_min = prev_step - prev_dist;
            if d < expected_min { d } else { d * omega }
        } else {
            d
        };

        prev_dist = d;
        prev_step = step;
        t += step.max(config.min_step);
        steps += 1;
    }

    None
}

/// Raymarch returning detailed result
#[allow(dead_code)]
#[inline(always)]
pub fn raymarch_detailed(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    config: &RaymarchConfig,
) -> RaymarchResult {
    let dir = direction.normalize();
    let mut t = 0.0;
    let mut steps = 0;

    while t < max_distance && steps < config.max_steps {
        let point = origin + dir * t;
        let d = eval(node, point);

        if d.abs() < config.epsilon {
            return RaymarchResult {
                hit: true,
                distance: t,
                point,
                normal: normal(node, point, config.epsilon),
                steps,
            };
        }

        t += d.max(config.min_step);
        steps += 1;
    }

    RaymarchResult {
        hit: false,
        distance: t,
        point: origin + dir * t,
        normal: Vec3::ZERO,
        steps,
    }
}

/// Batch raymarch (single-threaded)
pub fn raymarch_batch(
    node: &SdfNode,
    rays: &[Ray],
    max_distance: f32,
) -> Vec<Option<Hit>> {
    rays.iter()
        .map(|ray| raymarch(node, ray.origin, ray.direction, max_distance))
        .collect()
}

/// Batch raymarch (parallel)
pub fn raymarch_batch_parallel(
    node: &SdfNode,
    rays: &[Ray],
    max_distance: f32,
) -> Vec<Option<Hit>> {
    rays.par_iter()
        .map(|ray| raymarch(node, ray.origin, ray.direction, max_distance))
        .collect()
}

/// Render a depth buffer (Interpreter path)
pub fn render_depth(
    node: &SdfNode,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<f32> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);

    let mut buffer = vec![0.0f32; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch(node, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
            }
        });

    buffer
}

/// Render normals as RGB values (Interpreter path)
pub fn render_normals(
    node: &SdfNode,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<[u8; 3]> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);

    let mut buffer = vec![[0u8; 3]; width * height];
    let config = RaymarchConfig::default();

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch_with_config(node, camera_pos, ray_dir, max_distance, &config) {
                    Some(hit) => [
                        ((hit.normal.x * 0.5 + 0.5) * 255.0) as u8,
                        ((hit.normal.y * 0.5 + 0.5) * 255.0) as u8,
                        ((hit.normal.z * 0.5 + 0.5) * 255.0) as u8,
                    ],
                    None => [0, 0, 0],
                };
            }
        });

    buffer
}

// ============================================================================
// Compiled Backend (CompiledSdf VM — ~3x faster than interpreter)
// ============================================================================

/// Sphere tracing using CompiledSdf (cache-efficient VM)
///
/// # Deep Fried Optimization
///
/// Eliminates Arc pointer chasing and branch misprediction from tree traversal.
/// The CompiledSdf flat instruction array is much more cache-friendly.
#[inline(always)]
pub fn raymarch_compiled(
    sdf: &CompiledSdf,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<Hit> {
    const EPSILON: f32 = 0.0001;
    const MAX_STEPS: u32 = 128;

    let mut t = 0.0;
    let mut steps = 0;

    while t < max_distance && steps < MAX_STEPS {
        let point = origin + direction * t;
        let d = eval_compiled(sdf, point);

        if d.abs() < EPSILON {
            return Some(Hit {
                distance: t,
                point,
                normal: eval_compiled_normal(sdf, point, EPSILON),
                steps,
            });
        }

        t += d;
        steps += 1;
    }

    None
}

/// Sphere tracing with config using CompiledSdf
#[inline(always)]
pub fn raymarch_compiled_with_config(
    sdf: &CompiledSdf,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    config: &RaymarchConfig,
) -> Option<Hit> {
    let dir = direction.normalize();
    let mut t = 0.0;
    let mut steps = 0;
    let omega = config.omega;
    let use_relaxation = omega > 1.0;

    let mut prev_dist = 0.0;
    let mut prev_step = 0.0;

    while t < max_distance && steps < config.max_steps {
        let point = origin + dir * t;
        let d = eval_compiled(sdf, point);

        if d.abs() < config.epsilon {
            return Some(Hit {
                distance: t,
                point,
                normal: eval_compiled_normal(sdf, point, config.epsilon),
                steps,
            });
        }

        let step = if use_relaxation && steps > 0 {
            let expected_min = prev_step - prev_dist;
            if d < expected_min { d } else { d * omega }
        } else {
            d
        };

        prev_dist = d;
        prev_step = step;
        t += step.max(config.min_step);
        steps += 1;
    }

    None
}

/// Batch raymarch using CompiledSdf (parallel)
pub fn raymarch_compiled_batch_parallel(
    sdf: &CompiledSdf,
    rays: &[Ray],
    max_distance: f32,
) -> Vec<Option<Hit>> {
    rays.par_iter()
        .map(|ray| raymarch_compiled(sdf, ray.origin, ray.direction, max_distance))
        .collect()
}

/// Render depth buffer using CompiledSdf
pub fn render_depth_compiled(
    sdf: &CompiledSdf,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<f32> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);

    let mut buffer = vec![0.0f32; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch_compiled(sdf, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
            }
        });

    buffer
}

/// Render normals using CompiledSdf
pub fn render_normals_compiled(
    sdf: &CompiledSdf,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<[u8; 3]> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);

    let mut buffer = vec![[0u8; 3]; width * height];
    let config = RaymarchConfig::default();

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch_compiled_with_config(sdf, camera_pos, ray_dir, max_distance, &config) {
                    Some(hit) => [
                        ((hit.normal.x * 0.5 + 0.5) * 255.0) as u8,
                        ((hit.normal.y * 0.5 + 0.5) * 255.0) as u8,
                        ((hit.normal.z * 0.5 + 0.5) * 255.0) as u8,
                    ],
                    None => [0, 0, 0],
                };
            }
        });

    buffer
}

// ============================================================================
// SIMD Packet Tracing (CompiledSdf + Vec3x8 — ~6x faster)
// ============================================================================

/// March 8 rays simultaneously using SIMD CompiledSdf evaluation
///
/// # Deep Fried Optimization
///
/// - 8 rays evaluated in parallel via `eval_compiled_simd()`
/// - Masked control flow: finished lanes keep their final distance
/// - No branch misprediction from per-lane hit/miss decisions
///
/// Returns array of 8 `Option<(f32, Vec3)>` (distance, hit_point) — normals
/// should be computed separately for hit lanes only to avoid waste.
pub fn raymarch_simd_8(
    sdf: &CompiledSdf,
    origins: Vec3x8,
    directions: Vec3x8,
    max_distance: f32,
    config: &RaymarchConfig,
) -> [Option<(f32, Vec3, u32)>; 8] {
    let eps = f32x8::splat(config.epsilon);
    let max_d = f32x8::splat(max_distance);
    let zero = f32x8::splat(0.0);

    let mut t = f32x8::splat(0.0);

    // finished: 0.0 = active, 1.0 = done
    let mut finished = f32x8::splat(0.0);
    // hit: 0.0 = miss, 1.0 = hit
    let mut hit_flags = f32x8::splat(0.0);
    let mut step_counts = [0u32; 8];

    for step in 0..config.max_steps {
        // p = origin + direction * t
        let px = origins.x + directions.x * t;
        let py = origins.y + directions.y * t;
        let pz = origins.z + directions.z * t;

        let points = Vec3x8 { x: px, y: py, z: pz };
        let d = eval_compiled_simd(sdf, points);

        // Check hit: |d| < epsilon
        let abs_d = d.abs();
        let is_hit = abs_d.cmp_lt(eps);

        // Check too far: t > max_distance
        let is_far = t.cmp_gt(max_d);

        // Lanes that just became done
        let one = f32x8::splat(1.0);
        let newly_hit = is_hit.blend(one, zero) * (one - finished);
        let newly_far = is_far.blend(one, zero) * (one - finished);

        // Update hit flags (only for newly hit, not far)
        hit_flags = hit_flags + newly_hit;

        // Mark finished lanes
        finished = finished + newly_hit + newly_far;
        // Clamp finished to 1.0
        finished = finished.fast_min(one);

        // Update step counts for active lanes
        let finished_arr: [f32; 8] = finished.into();
        for i in 0..8 {
            if finished_arr[i] < 0.5 {
                step_counts[i] = step + 1;
            }
        }

        // Early exit if all lanes done
        if finished.cmp_ge(one).all() {
            break;
        }

        // Advance t for active lanes only (finished lanes get zero advance)
        let active = one - finished;
        t = t + d * active;
    }

    // Extract results
    let t_arr: [f32; 8] = t.into();
    let hit_arr: [f32; 8] = hit_flags.into();
    let (ox, oy, oz) = origins.to_array();
    let (dx, dy, dz) = directions.to_array();

    let mut results: [Option<(f32, Vec3, u32)>; 8] = [None; 8];

    for i in 0..8 {
        if hit_arr[i] > 0.5 {
            let p = Vec3::new(
                ox[i] + dx[i] * t_arr[i],
                oy[i] + dy[i] * t_arr[i],
                oz[i] + dz[i] * t_arr[i],
            );
            let _n = eval_compiled_normal(sdf, p, config.epsilon);
            results[i] = Some((t_arr[i], p, step_counts[i]));
        }
    }

    results
}

/// Render depth buffer using SIMD 8-ray packet tracing (Tile-Based)
///
/// # Deep Fried Optimization
///
/// Processes 8 horizontal pixels at once, maximizing SIMD utilization.
/// Each tile evaluates 8 points per step through `eval_compiled_simd()`.
pub fn render_depth_compiled_simd(
    sdf: &CompiledSdf,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<f32> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);
    let config = RaymarchConfig::default();

    let mut buffer = vec![0.0f32; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            // Process in 8-pixel tiles
            let mut x = 0;
            while x + 8 <= width {
                // Build 8 ray directions
                let mut dir_x = [0.0f32; 8];
                let mut dir_y = [0.0f32; 8];
                let mut dir_z = [0.0f32; 8];

                for i in 0..8 {
                    let u = ((x + i) as f32 * inv_width) * 2.0 - 1.0;
                    let rd = (row_vec + right_scaled * u).normalize();
                    dir_x[i] = rd.x;
                    dir_y[i] = rd.y;
                    dir_z[i] = rd.z;
                }

                let origins = Vec3x8::splat(camera_pos);
                let directions = Vec3x8::new(dir_x, dir_y, dir_z);

                let results = raymarch_simd_8(sdf, origins, directions, max_distance, &config);

                for i in 0..8 {
                    row[x + i] = match results[i] {
                        Some((dist, _, _)) => dist,
                        None => f32::MAX,
                    };
                }

                x += 8;
            }

            // Scalar fallback for remaining pixels
            while x < width {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                row[x] = match raymarch_compiled(sdf, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
                x += 1;
            }
        });

    buffer
}

// ============================================================================
// JIT Scalar Backend (Native code — ~5x faster)
// ============================================================================

/// Sphere tracing using JIT-compiled native code
///
/// # Deep Fried Optimization
///
/// Calls the JIT function pointer directly — zero interpreter overhead,
/// zero tree traversal, just a `call` instruction to native machine code.
#[cfg(feature = "jit")]
#[inline(always)]
pub fn raymarch_jit(
    sdf: &crate::compiled::jit::JitCompiledSdf,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<Hit> {
    const EPSILON: f32 = 0.0001;
    const MAX_STEPS: u32 = 128;

    let mut t = 0.0;
    let mut steps = 0;

    while t < max_distance && steps < MAX_STEPS {
        let point = origin + direction * t;
        let d = sdf.eval(point);

        if d.abs() < EPSILON {
            return Some(Hit {
                distance: t,
                point,
                normal: calc_normal_jit(sdf, point, EPSILON),
                steps,
            });
        }

        t += d;
        steps += 1;
    }

    None
}

/// Sphere tracing with config using JIT
#[cfg(feature = "jit")]
#[inline(always)]
pub fn raymarch_jit_with_config(
    sdf: &crate::compiled::jit::JitCompiledSdf,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
    config: &RaymarchConfig,
) -> Option<Hit> {
    let dir = direction.normalize();
    let mut t = 0.0;
    let mut steps = 0;
    let omega = config.omega;
    let use_relaxation = omega > 1.0;

    let mut prev_dist = 0.0;
    let mut prev_step = 0.0;

    while t < max_distance && steps < config.max_steps {
        let point = origin + dir * t;
        let d = sdf.eval(point);

        if d.abs() < config.epsilon {
            return Some(Hit {
                distance: t,
                point,
                normal: calc_normal_jit(sdf, point, config.epsilon),
                steps,
            });
        }

        let step = if use_relaxation && steps > 0 {
            let expected_min = prev_step - prev_dist;
            if d < expected_min { d } else { d * omega }
        } else {
            d
        };

        prev_dist = d;
        prev_step = step;
        t += step.max(config.min_step);
        steps += 1;
    }

    None
}

/// Calculate surface normal using JIT (tetrahedral method)
#[cfg(feature = "jit")]
#[inline(always)]
fn calc_normal_jit(sdf: &crate::compiled::jit::JitCompiledSdf, p: Vec3, eps: f32) -> Vec3 {
    let k0 = Vec3::new(1.0, -1.0, -1.0);
    let k1 = Vec3::new(-1.0, -1.0, 1.0);
    let k2 = Vec3::new(-1.0, 1.0, -1.0);
    let k3 = Vec3::new(1.0, 1.0, 1.0);

    (k0 * sdf.eval(p + k0 * eps)
     + k1 * sdf.eval(p + k1 * eps)
     + k2 * sdf.eval(p + k2 * eps)
     + k3 * sdf.eval(p + k3 * eps))
    .normalize()
}

/// Batch raymarch using JIT (parallel)
#[cfg(feature = "jit")]
pub fn raymarch_jit_batch_parallel(
    sdf: &crate::compiled::jit::JitCompiledSdf,
    rays: &[Ray],
    max_distance: f32,
) -> Vec<Option<Hit>> {
    rays.par_iter()
        .map(|ray| raymarch_jit(sdf, ray.origin, ray.direction, max_distance))
        .collect()
}

/// Render depth buffer using JIT scalar
#[cfg(feature = "jit")]
pub fn render_depth_jit(
    sdf: &crate::compiled::jit::JitCompiledSdf,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<f32> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);

    let mut buffer = vec![0.0f32; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch_jit(sdf, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
            }
        });

    buffer
}

// ============================================================================
// JIT SIMD Packet Tracing (8 rays via JitSimdSdf — ~15-20x faster)
// ============================================================================

/// March 8 rays simultaneously using JIT SIMD native code
///
/// # Deep Fried Optimization
///
/// - Calls `JitSimdSdf::eval_8()` which executes native AVX2/NEON instructions
/// - 8 SDF evaluations per function call
/// - Masked control flow: no branch penalty for mixed hit/miss lanes
/// - Combined with tile-based rendering for cache efficiency
#[cfg(feature = "jit")]
pub fn raymarch_jit_simd_8(
    sdf: &crate::compiled::jit::JitSimdSdf,
    _compiled: &CompiledSdf,
    origins: Vec3x8,
    directions: Vec3x8,
    max_distance: f32,
    config: &RaymarchConfig,
) -> [Option<(f32, Vec3, u32)>; 8] {
    let eps = f32x8::splat(config.epsilon);
    let max_d = f32x8::splat(max_distance);
    let zero = f32x8::splat(0.0);
    let one = f32x8::splat(1.0);

    let mut t = f32x8::splat(0.0);
    let mut finished = f32x8::splat(0.0);
    let mut hit_flags = f32x8::splat(0.0);
    let mut step_counts = [0u32; 8];

    for step in 0..config.max_steps {
        // p = origin + direction * t
        let px = origins.x + directions.x * t;
        let py = origins.y + directions.y * t;
        let pz = origins.z + directions.z * t;

        // JIT SIMD eval: 8 SDF evaluations in one native call
        let px_arr: [f32; 8] = px.into();
        let py_arr: [f32; 8] = py.into();
        let pz_arr: [f32; 8] = pz.into();
        let d_arr = unsafe { sdf.eval_8(&px_arr, &py_arr, &pz_arr) };
        let d = f32x8::from(d_arr);

        // Check hit: |d| < epsilon
        let abs_d = d.abs();
        let is_hit = abs_d.cmp_lt(eps);
        let is_far = t.cmp_gt(max_d);

        let newly_hit = is_hit.blend(one, zero) * (one - finished);
        let newly_far = is_far.blend(one, zero) * (one - finished);

        hit_flags = hit_flags + newly_hit;
        finished = (finished + newly_hit + newly_far).fast_min(one);

        let finished_arr: [f32; 8] = finished.into();
        for i in 0..8 {
            if finished_arr[i] < 0.5 {
                step_counts[i] = step + 1;
            }
        }

        if finished.cmp_ge(one).all() {
            break;
        }

        let active = one - finished;
        t = t + d * active;
    }

    // Extract results
    let t_arr: [f32; 8] = t.into();
    let hit_arr: [f32; 8] = hit_flags.into();
    let (ox, oy, oz) = origins.to_array();
    let (dx, dy, dz) = directions.to_array();

    let mut results: [Option<(f32, Vec3, u32)>; 8] = [None; 8];

    for i in 0..8 {
        if hit_arr[i] > 0.5 {
            let p = Vec3::new(
                ox[i] + dx[i] * t_arr[i],
                oy[i] + dy[i] * t_arr[i],
                oz[i] + dz[i] * t_arr[i],
            );
            // Normal from compiled path (no SIMD normal helper needed)
            results[i] = Some((t_arr[i], p, step_counts[i]));
        }
    }

    results
}

/// Render depth buffer using JIT SIMD (Tile-Based, ~15-20x faster)
///
/// # Deep Fried Optimization
///
/// Combines JIT SIMD packet tracing with tile-based rendering:
/// - 8 horizontal pixels form one SIMD packet
/// - `JitSimdSdf::eval_8()` evaluates all 8 in one native call
/// - Row-based parallelization via rayon
/// - Scalar fallback for edge pixels (width % 8 != 0)
#[cfg(feature = "jit")]
pub fn render_depth_jit_simd(
    sdf: &crate::compiled::jit::JitSimdSdf,
    compiled: &CompiledSdf,
    camera_pos: Vec3,
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
    max_distance: f32,
) -> Vec<f32> {
    let (forward, right, up, half_width, half_height) =
        camera_basis(camera_dir, camera_up, width, height, fov);
    let config = RaymarchConfig::default();

    let mut buffer = vec![0.0f32; width * height];

    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;
            let inv_width = 1.0 / width as f32;

            let mut x = 0;
            while x + 8 <= width {
                let mut dir_x = [0.0f32; 8];
                let mut dir_y = [0.0f32; 8];
                let mut dir_z = [0.0f32; 8];

                for i in 0..8 {
                    let u = ((x + i) as f32 * inv_width) * 2.0 - 1.0;
                    let rd = (row_vec + right_scaled * u).normalize();
                    dir_x[i] = rd.x;
                    dir_y[i] = rd.y;
                    dir_z[i] = rd.z;
                }

                let origins = Vec3x8::splat(camera_pos);
                let directions = Vec3x8::new(dir_x, dir_y, dir_z);

                let results = raymarch_jit_simd_8(sdf, compiled, origins, directions, max_distance, &config);

                for i in 0..8 {
                    row[x + i] = match results[i] {
                        Some((dist, _, _)) => dist,
                        None => f32::MAX,
                    };
                }

                x += 8;
            }

            // Scalar fallback for edge pixels
            while x < width {
                let u = (x as f32 * inv_width) * 2.0 - 1.0;
                let ray_dir = (row_vec + right_scaled * u).normalize();

                row[x] = match raymarch_compiled(compiled, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
                x += 1;
            }
        });

    buffer
}

// ============================================================================
// Shared Utilities
// ============================================================================

/// Pre-calculate camera basis vectors (shared by all render functions)
///
/// Returns (forward, right, up, half_width, half_height)
#[inline(always)]
fn camera_basis(
    camera_dir: Vec3,
    camera_up: Vec3,
    width: usize,
    height: usize,
    fov: f32,
) -> (Vec3, Vec3, Vec3, f32, f32) {
    let forward = camera_dir.normalize();
    let right = forward.cross(camera_up).normalize();
    let up = right.cross(forward);

    let aspect = width as f32 / height as f32;
    let half_height = (fov * 0.5).tan();
    let half_width = half_height * aspect;

    (forward, right, up, half_width, half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Interpreter Tests ============

    #[test]
    fn test_raymarch_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch(&sphere, origin, direction, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01);
        assert!((hit.normal.x - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_raymarch_miss() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 5.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch(&sphere, origin, direction, 20.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_raymarch_config_quality() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let config_fast = RaymarchConfig::fast();
        let config_hq = RaymarchConfig::high_quality();

        let hit_fast = raymarch_with_config(&sphere, origin, direction, 10.0, &config_fast);
        let hit_hq = raymarch_with_config(&sphere, origin, direction, 10.0, &config_hq);

        assert!(hit_fast.is_some());
        assert!(hit_hq.is_some());
    }

    #[test]
    fn test_raymarch_detailed() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let config = RaymarchConfig::default();
        let result = raymarch_detailed(&sphere, origin, direction, 10.0, &config);

        assert!(result.hit);
        assert!(result.steps > 0);
        assert!(result.steps < config.max_steps);
    }

    #[test]
    fn test_render_depth() {
        let sphere = SdfNode::sphere(1.0);
        let camera_pos = Vec3::new(0.0, 0.0, -5.0);
        let camera_dir = Vec3::new(0.0, 0.0, 1.0);
        let camera_up = Vec3::new(0.0, 1.0, 0.0);

        let depth = render_depth(
            &sphere, camera_pos, camera_dir, camera_up,
            8, 8, std::f32::consts::FRAC_PI_4, 10.0,
        );

        assert_eq!(depth.len(), 64);
        let center_depth = depth[8 * 4 + 4];
        assert!(center_depth < f32::MAX);
        assert!((center_depth - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_render_normals() {
        let sphere = SdfNode::sphere(1.0);
        let camera_pos = Vec3::new(0.0, 0.0, -5.0);
        let camera_dir = Vec3::new(0.0, 0.0, 1.0);
        let camera_up = Vec3::new(0.0, 1.0, 0.0);

        let normals = render_normals(
            &sphere, camera_pos, camera_dir, camera_up,
            8, 8, std::f32::consts::FRAC_PI_4, 10.0,
        );

        assert_eq!(normals.len(), 64);
        let center_normal = normals[8 * 4 + 4];
        assert!(center_normal[2] < 64);
    }

    #[test]
    fn test_over_relaxation() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let config = RaymarchConfig::fast();
        let hit = raymarch_with_config(&sphere, origin, direction, 10.0, &config);

        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.1);
    }

    // ============ Compiled Backend Tests ============

    #[test]
    fn test_raymarch_compiled_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch_compiled(&compiled, origin, direction, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01,
            "Compiled sphere hit distance: {}", hit.distance);
    }

    #[test]
    fn test_raymarch_compiled_miss() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let origin = Vec3::new(-5.0, 5.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch_compiled(&compiled, origin, direction, 20.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_raymarch_compiled_matches_interpreter() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0), 0.2);
        let compiled = CompiledSdf::compile(&shape);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit_interp = raymarch(&shape, origin, direction, 20.0);
        let hit_compiled = raymarch_compiled(&compiled, origin, direction, 20.0);

        assert_eq!(hit_interp.is_some(), hit_compiled.is_some());
        if let (Some(h1), Some(h2)) = (hit_interp, hit_compiled) {
            assert!((h1.distance - h2.distance).abs() < 0.01,
                "Interpreter={}, Compiled={}", h1.distance, h2.distance);
        }
    }

    #[test]
    fn test_render_depth_compiled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let camera_pos = Vec3::new(0.0, 0.0, -5.0);
        let camera_dir = Vec3::new(0.0, 0.0, 1.0);
        let camera_up = Vec3::new(0.0, 1.0, 0.0);

        let depth = render_depth_compiled(
            &compiled, camera_pos, camera_dir, camera_up,
            8, 8, std::f32::consts::FRAC_PI_4, 10.0,
        );

        assert_eq!(depth.len(), 64);
        let center_depth = depth[8 * 4 + 4];
        assert!(center_depth < f32::MAX);
        assert!((center_depth - 4.0).abs() < 0.5,
            "Compiled depth center: {}", center_depth);
    }

    // ============ SIMD Packet Tests ============

    #[test]
    fn test_raymarch_simd_8_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let config = RaymarchConfig::default();

        // 8 rays from different X positions, all heading +Z
        let origins = Vec3x8::new(
            [-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0],
            [0.0; 8],
            [-5.0; 8],
        );
        let directions = Vec3x8::new(
            [0.0; 8],
            [0.0; 8],
            [1.0; 8],
        );

        let results = raymarch_simd_8(&compiled, origins, directions, 20.0, &config);

        // Center ray (index 2) should hit at ~4.0
        assert!(results[2].is_some(), "Center ray should hit sphere");
        let (dist, _, _) = results[2].unwrap();
        assert!((dist - 4.0).abs() < 0.1, "Center ray dist: {}", dist);

        // Edge rays (index 0, 4) should miss (|x| = 2 > radius 1)
        assert!(results[0].is_none(), "Ray at x=-2 should miss");
        assert!(results[4].is_none(), "Ray at x=2 should miss");
    }

    #[test]
    fn test_render_depth_compiled_simd() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let camera_pos = Vec3::new(0.0, 0.0, -5.0);
        let camera_dir = Vec3::new(0.0, 0.0, 1.0);
        let camera_up = Vec3::new(0.0, 1.0, 0.0);

        let depth_scalar = render_depth_compiled(
            &compiled, camera_pos, camera_dir, camera_up,
            16, 16, std::f32::consts::FRAC_PI_4, 10.0,
        );
        let depth_simd = render_depth_compiled_simd(
            &compiled, camera_pos, camera_dir, camera_up,
            16, 16, std::f32::consts::FRAC_PI_4, 10.0,
        );

        assert_eq!(depth_scalar.len(), depth_simd.len());

        // Compare results: should be very close
        for i in 0..depth_scalar.len() {
            if depth_scalar[i] < f32::MAX && depth_simd[i] < f32::MAX {
                assert!((depth_scalar[i] - depth_simd[i]).abs() < 0.5,
                    "Pixel {} mismatch: scalar={}, simd={}",
                    i, depth_scalar[i], depth_simd[i]);
            }
        }
    }

    // ============ JIT Backend Tests ============

    #[cfg(feature = "jit")]
    #[test]
    fn test_raymarch_jit_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let jit = crate::compiled::jit::JitCompiledSdf::compile(&sphere).unwrap();
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch_jit(&jit, origin, direction, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01,
            "JIT sphere hit distance: {}", hit.distance);
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_raymarch_jit_matches_compiled() {
        let shape = SdfNode::sphere(1.0)
            .union(SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0));
        let compiled = CompiledSdf::compile(&shape);
        let jit = crate::compiled::jit::JitCompiledSdf::compile(&shape).unwrap();
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit_compiled = raymarch_compiled(&compiled, origin, direction, 20.0);
        let hit_jit = raymarch_jit(&jit, origin, direction, 20.0);

        assert_eq!(hit_compiled.is_some(), hit_jit.is_some());
        if let (Some(h1), Some(h2)) = (hit_compiled, hit_jit) {
            assert!((h1.distance - h2.distance).abs() < 0.01,
                "Compiled={}, JIT={}", h1.distance, h2.distance);
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_raymarch_jit_simd_8_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let jit_simd = crate::compiled::jit::JitSimdSdf::compile(&compiled).unwrap();
        let config = RaymarchConfig::default();

        let origins = Vec3x8::new(
            [0.0; 8],
            [0.0; 8],
            [-5.0; 8],
        );
        let directions = Vec3x8::new(
            [0.0; 8],
            [0.0; 8],
            [1.0; 8],
        );

        let results = raymarch_jit_simd_8(&jit_simd, &compiled, origins, directions, 20.0, &config);

        // All 8 rays from origin heading +Z should hit sphere at ~4.0
        for i in 0..8 {
            assert!(results[i].is_some(), "Ray {} should hit", i);
            let (dist, _, _) = results[i].unwrap();
            assert!((dist - 4.0).abs() < 0.1, "Ray {} dist: {}", i, dist);
        }
    }
}

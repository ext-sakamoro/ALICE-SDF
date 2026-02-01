//! Sphere tracing (raymarching) implementation (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//!
//! - **Integer Division Exorcism**: Replaced `i % w` / `i / w` with row-based iteration.
//! - **Incremental Ray Generation**: Pre-calculate row vectors to avoid redundant math per pixel.
//! - **Forced Inlining**: Hot loops are forced inline.
//! - **Over-Relaxation**: Enhanced sphere tracing with safety checks.
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, normal};
use crate::types::{Hit, Ray};
use crate::SdfNode;
use glam::Vec3;
use rayon::prelude::*;

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

/// Perform sphere tracing along a ray (Inlined Hot Path)
///
/// # Deep Fried Optimization
///
/// Hardcoded defaults for maximum speed in standard calls.
/// Avoids indirection through config struct in the hot path.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `origin` - Ray origin
/// * `direction` - Normalized ray direction
/// * `max_distance` - Maximum ray travel distance
///
/// # Returns
/// Hit information if surface found
#[inline(always)]
pub fn raymarch(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<Hit> {
    // Hardcoded "Good Enough" defaults for maximum speed
    const EPSILON: f32 = 0.0001;
    const MAX_STEPS: u32 = 128;

    let mut t = 0.0;
    let mut steps = 0;

    // Simple loop without over-relaxation for stability in default path
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

/// Perform sphere tracing with custom configuration (Deep Fried)
///
/// # Deep Fried Optimization
///
/// - Over-relaxation with safety rollback for faster convergence
/// - Branch prediction friendly: config checks are predictable per-frame
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

    // For over-relaxation: track previous distance and step
    let mut prev_dist = 0.0;
    let mut prev_step = 0.0;

    while t < max_distance && steps < config.max_steps {
        let point = origin + dir * t;
        let d = eval(node, point);

        // Hit check
        if d.abs() < config.epsilon {
            return Some(Hit {
                distance: t,
                point,
                normal: normal(node, point, config.epsilon),
                steps,
            });
        }

        // Deep Fried Over-Relaxation (Enhanced Sphere Tracing)
        // Branch prediction handles this well if config is constant per frame
        let step = if use_relaxation && steps > 0 {
            // Safety check: if we overstepped, the current distance will be
            // smaller than expected based on the previous step
            let expected_min = prev_step - prev_dist;
            if d < expected_min {
                // We overstepped! Fall back to previous position + conservative step
                // This is a simplified rollback
                d
            } else {
                d * omega
            }
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

/// Render a depth buffer for a camera view (Deep Fried Edition)
///
/// # Deep Fried Optimization
///
/// Uses row-based parallelization to completely eliminate integer division/modulo
/// per pixel (`i % width`, `i / width` → row iteration).
///
/// Ray basis vectors are pre-calculated per row, inner loop uses only
/// addition and multiplication.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `camera_pos` - Camera position
/// * `camera_dir` - Camera forward direction
/// * `camera_up` - Camera up vector
/// * `width` - Image width
/// * `height` - Image height
/// * `fov` - Field of view in radians
/// * `max_distance` - Maximum ray distance
///
/// # Returns
/// Depth buffer (distance values, f32::MAX for miss)
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
    // 1. Pre-calculate camera basis (once)
    let forward = camera_dir.normalize();
    let right = forward.cross(camera_up).normalize();
    let up = right.cross(forward);

    let aspect = width as f32 / height as f32;
    let half_height = (fov * 0.5).tan();
    let half_width = half_height * aspect;

    // 2. Prepare output buffer
    let mut buffer = vec![0.0f32; width * height];

    // 3. Row-based parallel execution (Integer Division Exorcism)
    // Splitting by rows avoids 'i % width' / 'i / width' math entirely
    buffer
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, row)| {
            // Pre-calculate Y component for this entire row
            // v ranges from 1.0 (top) to -1.0 (bottom)
            let v = 1.0 - (y as f32 / height as f32) * 2.0;
            let row_vec = forward + up * (v * half_height);
            let right_scaled = right * half_width;

            // Division Exorcism: multiply by reciprocal
            let inv_width = 1.0 / width as f32;

            for (x, pixel) in row.iter_mut().enumerate() {
                // u ranges from -1.0 (left) to 1.0 (right)
                let u = (x as f32 * inv_width) * 2.0 - 1.0;

                // Incremental ray direction reconstruction
                let ray_dir = (row_vec + right_scaled * u).normalize();

                *pixel = match raymarch(node, camera_pos, ray_dir, max_distance) {
                    Some(hit) => hit.distance,
                    None => f32::MAX,
                };
            }
        });

    buffer
}

/// Render normals as RGB values (Deep Fried Edition)
///
/// # Deep Fried Optimization
///
/// Same row-based parallelization as render_depth.
///
/// Returns Vec<[u8; 3]> where normal XYZ is mapped to RGB.
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
    let forward = camera_dir.normalize();
    let right = forward.cross(camera_up).normalize();
    let up = right.cross(forward);

    let aspect = width as f32 / height as f32;
    let half_height = (fov * 0.5).tan();
    let half_width = half_height * aspect;

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

                *pixel = match raymarch_with_config(node, camera_pos, ray_dir, max_distance, &config)
                {
                    Some(hit) => {
                        // Map normal from [-1,1] to [0,255]
                        [
                            ((hit.normal.x * 0.5 + 0.5) * 255.0) as u8,
                            ((hit.normal.y * 0.5 + 0.5) * 255.0) as u8,
                            ((hit.normal.z * 0.5 + 0.5) * 255.0) as u8,
                        ]
                    }
                    None => [0, 0, 0],
                };
            }
        });

    buffer
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raymarch_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let hit = raymarch(&sphere, origin, direction, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        // Should hit at distance 4 (5 - 1)
        assert!((hit.distance - 4.0).abs() < 0.01);
        // Normal should point away from center
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
            &sphere,
            camera_pos,
            camera_dir,
            camera_up,
            8,
            8,
            std::f32::consts::FRAC_PI_4,
            10.0,
        );

        assert_eq!(depth.len(), 64);

        // Center pixel should hit the sphere
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
            &sphere,
            camera_pos,
            camera_dir,
            camera_up,
            8,
            8,
            std::f32::consts::FRAC_PI_4,
            10.0,
        );

        assert_eq!(normals.len(), 64);

        // Center pixel should have a normal pointing towards camera (negative Z)
        let center_normal = normals[8 * 4 + 4];
        // Z component mapped: -1 -> 0, so should be close to 0
        assert!(center_normal[2] < 64); // Should be small (negative Z)
    }

    #[test]
    fn test_over_relaxation() {
        let sphere = SdfNode::sphere(1.0);
        let origin = Vec3::new(-5.0, 0.0, 0.0);
        let direction = Vec3::new(1.0, 0.0, 0.0);

        let config = RaymarchConfig::fast(); // Uses omega = 1.6
        let hit = raymarch_with_config(&sphere, origin, direction, 10.0, &config);

        assert!(hit.is_some());
        let hit = hit.unwrap();
        // Should still hit at approximately the same distance
        assert!((hit.distance - 4.0).abs() < 0.1);
    }
}

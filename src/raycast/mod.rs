//! Raymarching for SDFs (Deep Fried Edition)
//!
//! Implements sphere tracing algorithm for ray-SDF intersection.
//!
//! # Deep Fried Optimizations
//!
//! - **Specialized Loops**: Dedicated loops for Shadow/AO that don't compute normals.
//! - **Early Exit**: Hard shadow detection exits immediately.
//! - **Iteration Limits**: Prevents infinite loops in degenerate SDFs.
//! - **Forced Inlining**: Hot paths are forced inline.
//!
//! Author: Moroya Sakamoto

mod march;

pub use march::{
    raymarch, raymarch_batch, raymarch_batch_parallel, raymarch_with_config,
    render_depth, render_normals,
    RaymarchConfig, RaymarchResult,
};

use crate::eval::eval;
use crate::types::{Hit, Ray};
use crate::SdfNode;
use glam::Vec3;

/// Simple raycast with default parameters
///
/// # Arguments
/// * `node` - The SDF tree
/// * `ray` - Ray to cast
/// * `max_distance` - Maximum ray distance
///
/// # Returns
/// Hit information if intersection found
#[inline(always)]
pub fn raycast(node: &SdfNode, ray: Ray, max_distance: f32) -> Option<Hit> {
    raymarch(node, ray.origin, ray.direction, max_distance)
}

/// Cast multiple rays in parallel
pub fn raycast_batch(node: &SdfNode, rays: &[Ray], max_distance: f32) -> Vec<Option<Hit>> {
    raymarch_batch_parallel(node, rays, max_distance)
}

/// Ambient occlusion estimation (Deep Fried Edition)
///
/// Estimates how occluded a point is by sampling along the normal.
///
/// # Deep Fried Optimization
///
/// - Pure `eval()` calls only (no normal computation, no hit struct allocation)
/// - Fixed step sampling for predictable performance
/// - Branchless occlusion accumulation
///
/// # Arguments
/// * `node` - The SDF tree
/// * `point` - Surface point
/// * `normal` - Surface normal
/// * `samples` - Number of samples
/// * `max_distance` - Maximum sampling distance
///
/// # Returns
/// Occlusion factor (0 = fully occluded, 1 = fully exposed)
#[inline(always)]
pub fn ambient_occlusion(
    node: &SdfNode,
    point: Vec3,
    normal: Vec3,
    samples: u32,
    max_distance: f32,
) -> f32 {
    let mut occlusion = 0.0;
    let step = max_distance / samples as f32;

    // Fixed-step sampling: predictable performance, no early exit
    for i in 1..=samples {
        let t = i as f32 * step;
        let sample_point = point + normal * t;

        // Pure eval, no overhead (no hit check, no normal)
        let d = eval(node, sample_point);

        // Branchless accumulation
        // If d >= t (outside), contribution is zero or negative (clamped)
        // If d < t (occluded), contribution is positive
        let contribution = (t - d.max(0.0)) / t;
        occlusion += contribution;
    }

    // Intensity factor k=1.0 baked in
    (1.0 - occlusion / samples as f32).clamp(0.0, 1.0)
}

/// Soft shadow estimation (Deep Fried Edition)
///
/// Calculates soft shadow factor using ray marching.
///
/// # Deep Fried Optimization
///
/// - Early exit for hard shadows (h < 0.001 returns 0 immediately)
/// - Iteration limit prevents infinite loops in weird SDFs
/// - Improved penumbra calculation (Inigo Quilez technique)
///
/// # Arguments
/// * `node` - The SDF tree
/// * `origin` - Shadow ray origin (on surface)
/// * `direction` - Direction toward light
/// * `min_t` - Minimum distance (to avoid self-intersection)
/// * `max_t` - Maximum distance to light
/// * `k` - Softness factor (higher = softer shadows)
///
/// # Returns
/// Shadow factor (0 = fully shadowed, 1 = fully lit)
#[inline(always)]
pub fn soft_shadow(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    min_t: f32,
    max_t: f32,
    k: f32,
) -> f32 {
    let mut res: f32 = 1.0;
    let mut t = min_t;
    let mut ph = 1e10; // Large initial value

    // Deep Fried: Iteration limit prevents infinite loops
    const MAX_STEPS: u32 = 64;

    for _ in 0..MAX_STEPS {
        if t >= max_t {
            break;
        }

        let h = eval(node, origin + direction * t);

        // Deep Fried: Early exit for hard shadows
        // If we're inside geometry, we're definitely shadowed
        if h < 0.001 {
            return 0.0;
        }

        // Improved soft shadow penumbra (Inigo Quilez)
        // https://iquilezles.org/articles/rmshadows/
        let y = h * h / (2.0 * ph);
        let d = (h * h - y * y).sqrt();
        res = res.min(k * d / (t - y).max(0.0));

        ph = h;
        t += h; // h is the safe step distance
    }

    res.clamp(0.0, 1.0)
}

/// Hard shadow test (Deep Fried Edition)
///
/// Fast binary shadow test - just checks if anything blocks the ray.
///
/// # Deep Fried Optimization
///
/// - No penumbra calculation
/// - Returns immediately on any intersection
/// - Minimal computation per step
///
/// # Arguments
/// * `node` - The SDF tree
/// * `origin` - Shadow ray origin
/// * `direction` - Direction toward light
/// * `min_t` - Minimum distance (to avoid self-intersection)
/// * `max_t` - Maximum distance to light
///
/// # Returns
/// `true` if in shadow, `false` if lit
#[inline(always)]
pub fn hard_shadow(
    node: &SdfNode,
    origin: Vec3,
    direction: Vec3,
    min_t: f32,
    max_t: f32,
) -> bool {
    const EPSILON: f32 = 0.001;
    const MAX_STEPS: u32 = 64;

    let mut t = min_t;

    for _ in 0..MAX_STEPS {
        if t >= max_t {
            return false; // Reached light, not in shadow
        }

        let h = eval(node, origin + direction * t);

        if h < EPSILON {
            return true; // Hit something, in shadow
        }

        t += h;
    }

    false // Exceeded steps, assume not in shadow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raycast_hit() {
        let sphere = SdfNode::sphere(1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

        let hit = raycast(&sphere, ray, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_raycast_miss() {
        let sphere = SdfNode::sphere(1.0);
        let ray = Ray::new(Vec3::new(-5.0, 5.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

        let hit = raycast(&sphere, ray, 10.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_ambient_occlusion() {
        let sphere = SdfNode::sphere(1.0);
        let point = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::new(1.0, 0.0, 0.0);

        let ao = ambient_occlusion(&sphere, point, normal, 5, 0.5);
        // On a convex surface, should be mostly unoccluded
        assert!(ao > 0.5);
    }

    #[test]
    fn test_soft_shadow_no_occluder() {
        let sphere = SdfNode::sphere(1.0).translate(0.0, 0.0, 5.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let shadow = soft_shadow(&sphere, origin, to_light, 0.01, 10.0, 8.0);
        // No occluder in the way
        assert!((shadow - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_soft_shadow_with_occluder() {
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let shadow = soft_shadow(&sphere, origin, to_light, 0.01, 10.0, 8.0);
        // Occluder in the way, should be significantly shadowed
        assert!(shadow < 0.5);
    }

    #[test]
    fn test_hard_shadow() {
        // Sphere blocking the light
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let in_shadow = hard_shadow(&sphere, origin, to_light, 0.01, 10.0);
        assert!(in_shadow);

        // No sphere blocking
        let to_side = Vec3::new(1.0, 0.0, 0.0).normalize();
        let not_in_shadow = hard_shadow(&sphere, origin, to_side, 0.01, 10.0);
        assert!(!not_in_shadow);
    }

    #[test]
    fn test_hard_shadow_no_occluder() {
        let sphere = SdfNode::sphere(1.0).translate(0.0, 0.0, 5.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let in_shadow = hard_shadow(&sphere, origin, to_light, 0.01, 10.0);
        assert!(!in_shadow);
    }
}

//! Raymarching for SDFs (Deep Fried v2 Edition)
//!
//! Implements sphere tracing algorithm for ray-SDF intersection.
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Specialized Loops**: Dedicated loops for Shadow/AO that don't compute normals.
//! - **Early Exit**: Hard shadow detection exits immediately.
//! - **Iteration Limits**: Prevents infinite loops in degenerate SDFs.
//! - **Forced Inlining**: Hot paths are forced inline.
//! - **Compiled Backend**: `CompiledSdf` VM eliminates tree traversal overhead.
//! - **JIT Backend**: `JitCompiledSdf` native code, zero interpreter overhead.
//! - **SIMD Packet Tracing**: 8 rays marched simultaneously via `Vec3x8`/`JitSimdSdf`.
//! - **Tile-Based Rendering**: Cache-friendly 8-pixel horizontal tiles.
//!
//! Author: Moroya Sakamoto

mod march;

// === Interpreter Backend (original) ===
pub use march::{
    raymarch, raymarch_batch, raymarch_batch_parallel, raymarch_with_config, render_depth,
    render_normals, RaymarchConfig, RaymarchResult,
};

// === Compiled Backend ===
pub use march::{
    raymarch_compiled, raymarch_compiled_batch_parallel, raymarch_compiled_with_config,
    render_depth_compiled, render_normals_compiled,
};

// === SIMD Packet Tracing (Compiled) ===
pub use march::{raymarch_simd_8, render_depth_compiled_simd};

// === JIT Backend ===
#[cfg(feature = "jit")]
pub use march::{
    raymarch_jit, raymarch_jit_batch_parallel, raymarch_jit_with_config, render_depth_jit,
};

// === JIT SIMD Packet Tracing ===
#[cfg(feature = "jit")]
pub use march::{raymarch_jit_simd_8, render_depth_jit_simd};

use crate::compiled::{eval_compiled, CompiledSdf};
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

// =========================================================================
// Interpreter Backend: AO / Shadow
// =========================================================================

/// Ambient occlusion estimation (Interpreter)
///
/// Estimates how occluded a point is by sampling along the normal.
///
/// - Pure `eval()` calls only (no normal computation, no hit struct allocation)
/// - Fixed step sampling for predictable performance
/// - Branchless occlusion accumulation
#[inline(always)]
pub fn ambient_occlusion(
    node: &SdfNode,
    point: Vec3,
    normal: Vec3,
    samples: u32,
    max_distance: f32,
) -> f32 {
    let mut occlusion = 0.0;
    let inv_samples = 1.0 / samples as f32;
    let step = max_distance * inv_samples;

    for i in 1..=samples {
        let t = i as f32 * step;
        let sample_point = point + normal * t;
        let d = eval(node, sample_point);
        let contribution = (t - d.max(0.0)) / t;
        occlusion += contribution;
    }

    (1.0 - occlusion * inv_samples).clamp(0.0, 1.0)
}

/// Soft shadow estimation (Interpreter)
///
/// Calculates soft shadow factor using ray marching.
/// Improved penumbra calculation (Inigo Quilez technique).
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
    let mut ph = 1e10;
    const MAX_STEPS: u32 = 64;

    for _ in 0..MAX_STEPS {
        if t >= max_t {
            break;
        }

        let h = eval(node, origin + direction * t);

        if h < 0.001 {
            return 0.0;
        }

        let y = h * h / (2.0 * ph);
        let d = (h * h - y * y).sqrt();
        res = res.min(k * d / (t - y).max(0.0));

        ph = h;
        t += h;
    }

    res.clamp(0.0, 1.0)
}

/// Hard shadow test (Interpreter)
///
/// Fast binary shadow test - just checks if anything blocks the ray.
#[inline(always)]
pub fn hard_shadow(node: &SdfNode, origin: Vec3, direction: Vec3, min_t: f32, max_t: f32) -> bool {
    const EPSILON: f32 = 0.001;
    const MAX_STEPS: u32 = 64;

    let mut t = min_t;

    for _ in 0..MAX_STEPS {
        if t >= max_t {
            return false;
        }

        let h = eval(node, origin + direction * t);

        if h < EPSILON {
            return true;
        }

        t += h;
    }

    false
}

// =========================================================================
// Compiled Backend: AO / Shadow
// =========================================================================

/// Ambient occlusion estimation (Compiled Backend)
///
/// Same algorithm as `ambient_occlusion()` but uses `eval_compiled()`.
/// ~3x faster than interpreter version.
#[inline(always)]
pub fn ambient_occlusion_compiled(
    sdf: &CompiledSdf,
    point: Vec3,
    normal: Vec3,
    samples: u32,
    max_distance: f32,
) -> f32 {
    let mut occlusion = 0.0;
    let inv_samples = 1.0 / samples as f32;
    let step = max_distance * inv_samples;

    for i in 1..=samples {
        let t = i as f32 * step;
        let sample_point = point + normal * t;
        let d = eval_compiled(sdf, sample_point);
        let contribution = (t - d.max(0.0)) / t;
        occlusion += contribution;
    }

    (1.0 - occlusion * inv_samples).clamp(0.0, 1.0)
}

/// Soft shadow estimation (Compiled Backend)
///
/// Same algorithm as `soft_shadow()` but uses `eval_compiled()`.
/// ~3x faster than interpreter version.
#[inline(always)]
pub fn soft_shadow_compiled(
    sdf: &CompiledSdf,
    origin: Vec3,
    direction: Vec3,
    min_t: f32,
    max_t: f32,
    k: f32,
) -> f32 {
    let mut res: f32 = 1.0;
    let mut t = min_t;
    let mut ph = 1e10;
    const MAX_STEPS: u32 = 64;

    for _ in 0..MAX_STEPS {
        if t >= max_t {
            break;
        }

        let h = eval_compiled(sdf, origin + direction * t);

        if h < 0.001 {
            return 0.0;
        }

        let y = h * h / (2.0 * ph);
        let d = (h * h - y * y).sqrt();
        res = res.min(k * d / (t - y).max(0.0));

        ph = h;
        t += h;
    }

    res.clamp(0.0, 1.0)
}

/// Hard shadow test (Compiled Backend)
///
/// Same algorithm as `hard_shadow()` but uses `eval_compiled()`.
/// ~3x faster than interpreter version.
#[inline(always)]
pub fn hard_shadow_compiled(
    sdf: &CompiledSdf,
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
            return false;
        }

        let h = eval_compiled(sdf, origin + direction * t);

        if h < EPSILON {
            return true;
        }

        t += h;
    }

    false
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
        assert!(ao > 0.5);
    }

    #[test]
    fn test_soft_shadow_no_occluder() {
        let sphere = SdfNode::sphere(1.0).translate(0.0, 0.0, 5.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let shadow = soft_shadow(&sphere, origin, to_light, 0.01, 10.0, 8.0);
        assert!((shadow - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_soft_shadow_with_occluder() {
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let shadow = soft_shadow(&sphere, origin, to_light, 0.01, 10.0, 8.0);
        assert!(shadow < 0.5);
    }

    #[test]
    fn test_hard_shadow() {
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let in_shadow = hard_shadow(&sphere, origin, to_light, 0.01, 10.0);
        assert!(in_shadow);

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

    // === Compiled Backend Tests ===

    #[test]
    fn test_ao_compiled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let point = Vec3::new(1.0, 0.0, 0.0);
        let normal = Vec3::new(1.0, 0.0, 0.0);

        let ao = ambient_occlusion_compiled(&compiled, point, normal, 5, 0.5);
        assert!(ao > 0.5);

        // Should match interpreter result
        let ao_interp = ambient_occlusion(&sphere, point, normal, 5, 0.5);
        assert!((ao - ao_interp).abs() < 0.01);
    }

    #[test]
    fn test_soft_shadow_compiled() {
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let compiled = CompiledSdf::compile(&sphere);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let shadow = soft_shadow_compiled(&compiled, origin, to_light, 0.01, 10.0, 8.0);
        assert!(shadow < 0.5);

        let shadow_interp = soft_shadow(&sphere, origin, to_light, 0.01, 10.0, 8.0);
        assert!((shadow - shadow_interp).abs() < 0.01);
    }

    #[test]
    fn test_hard_shadow_compiled() {
        let sphere = SdfNode::sphere(0.5).translate(0.0, 2.0, 0.0);
        let compiled = CompiledSdf::compile(&sphere);
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let to_light = Vec3::new(0.0, 1.0, 0.0).normalize();

        let in_shadow = hard_shadow_compiled(&compiled, origin, to_light, 0.01, 10.0);
        assert!(in_shadow);

        let to_side = Vec3::new(1.0, 0.0, 0.0).normalize();
        let not_in_shadow = hard_shadow_compiled(&compiled, origin, to_side, 0.01, 10.0);
        assert!(!not_in_shadow);
    }
}

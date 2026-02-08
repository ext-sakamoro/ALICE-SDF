//! Cone Tracing: March cones through SVO for indirect lighting (Deep Fried Edition)
//!
//! A cone trace samples the SVO at increasing radius as it marches along
//! a direction. The cone footprint grows with distance, naturally integrating
//! over a larger area (approximating a pre-filtered radiance lookup).
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::{direct_lighting, sky_color, DirectionalLight};
use crate::svo::{svo_query_point, SparseVoxelOctree};

/// Configuration for cone tracing
#[derive(Debug, Clone)]
pub struct ConeTraceConfig {
    /// Number of cones for hemisphere trace (5 or 9 typical)
    pub num_cones: u32,
    /// Maximum trace distance
    pub max_distance: f32,
    /// Initial cone half-angle (radians)
    pub cone_angle: f32,
    /// Step multiplier (larger = faster but less accurate)
    pub step_multiplier: f32,
    /// Minimum step size (world units)
    pub min_step: f32,
    /// Offset from surface to start tracing (avoid self-intersection)
    pub surface_offset: f32,
    /// Ambient occlusion weight (0.0 = no AO, 1.0 = full AO)
    pub ao_weight: f32,
}

impl Default for ConeTraceConfig {
    fn default() -> Self {
        ConeTraceConfig {
            num_cones: 5,
            max_distance: 20.0,
            cone_angle: 0.5, // ~28 degrees half-angle
            step_multiplier: 1.0,
            min_step: 0.05,
            surface_offset: 0.1,
            ao_weight: 0.5,
        }
    }
}

/// Result of a single cone trace
#[derive(Debug, Clone, Copy)]
pub struct ConeTraceResult {
    /// Accumulated color/radiance
    pub color: Vec3,
    /// Accumulated occlusion (0=fully occluded, 1=fully open)
    pub occlusion: f32,
}

/// Trace a single cone through the SVO
///
/// Marches along `direction` from `origin`, sampling the SVO at each step.
/// The cone radius grows with distance: `radius = t * tan(cone_angle)`.
/// Occlusion accumulates as the cone passes through surfaces.
pub fn cone_trace(
    svo: &SparseVoxelOctree,
    origin: Vec3,
    direction: Vec3,
    config: &ConeTraceConfig,
    light: Option<&DirectionalLight>,
) -> ConeTraceResult {
    let tan_angle = config.cone_angle.tan();
    let mut t = config.surface_offset;
    let mut color = Vec3::ZERO;
    let mut alpha = 0.0f32;

    while t < config.max_distance && alpha < 0.95 {
        let pos = origin + direction * t;
        let cone_radius = (t * tan_angle).max(config.min_step);

        // Sample SVO distance at this position
        let dist = svo_query_point(svo, pos);

        if dist < cone_radius {
            // We're near or inside a surface
            let opacity = (1.0 - dist / cone_radius).clamp(0.0, 1.0);

            // Estimate surface radiance at this point
            let radiance = estimate_radiance(svo, pos, direction, light);

            // Front-to-back compositing
            color += radiance * opacity * (1.0 - alpha);
            alpha += opacity * (1.0 - alpha);
        }

        // Step forward: at least min_step, or distance-based
        let step = (dist.abs().max(cone_radius) * config.step_multiplier).max(config.min_step);
        t += step;
    }

    // Add sky contribution for unoccluded portion
    let sky = sky_color(direction);
    color += sky * (1.0 - alpha);

    ConeTraceResult {
        color,
        occlusion: 1.0 - alpha,
    }
}

/// Trace a hemisphere of cones from a surface point
///
/// Distributes `num_cones` cones in a cosine-weighted hemisphere around
/// the surface normal and accumulates indirect illumination.
pub fn trace_hemisphere(
    svo: &SparseVoxelOctree,
    position: Vec3,
    normal: Vec3,
    config: &ConeTraceConfig,
    light: Option<&DirectionalLight>,
) -> Vec3 {
    let cones = generate_cosine_cones(normal, config.num_cones);
    let origin = position + normal * config.surface_offset;

    let mut total_color = Vec3::ZERO;
    let mut total_weight = 0.0f32;

    for (dir, weight) in &cones {
        let result = cone_trace(svo, origin, *dir, config, light);

        // Mix radiance with AO
        let ao_factor = 1.0 - config.ao_weight * (1.0 - result.occlusion);
        total_color += result.color * *weight * ao_factor;
        total_weight += weight;
    }

    if total_weight > 0.0 {
        total_color * (1.0 / total_weight)
    } else {
        Vec3::ZERO
    }
}

/// Estimate surface radiance at a point (simple diffuse model)
fn estimate_radiance(
    svo: &SparseVoxelOctree,
    pos: Vec3,
    _incoming_dir: Vec3,
    light: Option<&DirectionalLight>,
) -> Vec3 {
    let e = 0.05;
    let k0 = Vec3::new(1.0, -1.0, -1.0);
    let k1 = Vec3::new(-1.0, -1.0, 1.0);
    let k2 = Vec3::new(-1.0, 1.0, -1.0);
    let k3 = Vec3::new(1.0, 1.0, 1.0);
    let normal = (k0 * svo_query_point(svo, pos + k0 * e)
        + k1 * svo_query_point(svo, pos + k1 * e)
        + k2 * svo_query_point(svo, pos + k2 * e)
        + k3 * svo_query_point(svo, pos + k3 * e))
    .normalize_or_zero();

    // Diffuse surface albedo (gray)
    let albedo = Vec3::splat(0.5);

    // Direct lighting from sun
    let direct = match light {
        Some(l) => direct_lighting(normal, l),
        None => Vec3::ZERO,
    };

    // Simple ambient
    let ambient = Vec3::splat(0.05);

    albedo * (direct + ambient)
}

/// Generate cosine-weighted cone directions on a hemisphere
///
/// Uses a fixed pattern for reproducibility.
fn generate_cosine_cones(normal: Vec3, num_cones: u32) -> Vec<(Vec3, f32)> {
    let (tangent, bitangent) = make_orthonormal_basis(normal);
    let mut cones = Vec::with_capacity(num_cones as usize);

    if num_cones == 1 {
        cones.push((normal, 1.0));
        return cones;
    }

    // Center cone along normal (highest weight)
    cones.push((normal, 0.4));

    // Ring of cones at ~60 degrees from normal
    let ring_count = (num_cones - 1) as f32;
    let cone_elevation = 0.5f32; // cos(60°) = 0.5, sin(60°) = 0.866
    let sin_elev = (1.0 - cone_elevation * cone_elevation).sqrt();

    for i in 0..(num_cones - 1) {
        let angle = (i as f32 / ring_count) * std::f32::consts::TAU;
        let dir = normal * cone_elevation
            + tangent * (angle.cos() * sin_elev)
            + bitangent * (angle.sin() * sin_elev);
        let weight = cone_elevation; // Cosine weight
        cones.push((dir.normalize(), weight));
    }

    cones
}

/// Create an orthonormal basis from a normal vector
fn make_orthonormal_basis(normal: Vec3) -> (Vec3, Vec3) {
    let up = if normal.y.abs() < 0.9 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let tangent = normal.cross(up).normalize();
    let bitangent = tangent.cross(normal).normalize();
    (tangent, bitangent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::svo::{SparseVoxelOctree, SvoBuildConfig};
    use crate::types::SdfNode;

    fn make_test_svo() -> SparseVoxelOctree {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 4,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };
        SparseVoxelOctree::build(&sphere, &config)
    }

    #[test]
    fn test_cone_trace_basic() {
        let svo = make_test_svo();
        let config = ConeTraceConfig {
            max_distance: 5.0,
            ..Default::default()
        };

        let result = cone_trace(&svo, Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y, &config, None);

        assert!(result.color.length() > 0.0, "Should have some color");
        assert!(result.occlusion >= 0.0 && result.occlusion <= 1.0);
    }

    #[test]
    fn test_cone_trace_miss() {
        let svo = make_test_svo();
        let config = ConeTraceConfig {
            max_distance: 5.0,
            ..Default::default()
        };

        // Trace away from sphere
        let result = cone_trace(&svo, Vec3::new(0.0, 3.0, 0.0), Vec3::Y, &config, None);

        // Should get mostly sky color (low occlusion)
        assert!(
            result.occlusion > 0.5,
            "Tracing away should have low occlusion"
        );
    }

    #[test]
    fn test_trace_hemisphere() {
        let svo = make_test_svo();
        let config = ConeTraceConfig {
            num_cones: 5,
            max_distance: 5.0,
            ..Default::default()
        };

        let color = trace_hemisphere(
            &svo,
            Vec3::new(0.0, 1.1, 0.0), // Just above sphere surface
            Vec3::Y,
            &config,
            Some(&DirectionalLight::default()),
        );

        assert!(color.length() > 0.0, "Should have some indirect light");
    }

    #[test]
    fn test_generate_cones() {
        let cones = generate_cosine_cones(Vec3::Y, 5);
        assert_eq!(cones.len(), 5);

        // All directions should be in the upper hemisphere
        for (dir, weight) in &cones {
            assert!(
                dir.y > -0.1,
                "Cone should be in upper hemisphere, y={}",
                dir.y
            );
            assert!(*weight > 0.0);
            assert!(
                (dir.length() - 1.0).abs() < 0.01,
                "Direction should be normalized"
            );
        }
    }

    #[test]
    fn test_cone_trace_with_light() {
        let svo = make_test_svo();
        let config = ConeTraceConfig::default();
        let light = DirectionalLight::default();

        let with_light = cone_trace(
            &svo,
            Vec3::new(0.0, 3.0, 0.0),
            Vec3::NEG_Y,
            &config,
            Some(&light),
        );

        let without_light = cone_trace(&svo, Vec3::new(0.0, 3.0, 0.0), Vec3::NEG_Y, &config, None);

        // With light should be brighter (or at least as bright)
        assert!(with_light.color.length() >= without_light.color.length() - 0.1);
    }

    #[test]
    fn test_orthonormal_basis() {
        let (t, b) = make_orthonormal_basis(Vec3::Y);
        assert!((t.dot(Vec3::Y)).abs() < 0.001);
        assert!((b.dot(Vec3::Y)).abs() < 0.001);
        assert!((t.dot(b)).abs() < 0.001);
    }

    #[test]
    fn test_config_default() {
        let config = ConeTraceConfig::default();
        assert_eq!(config.num_cones, 5);
        assert!(config.max_distance > 0.0);
        assert!(config.cone_angle > 0.0);
    }
}

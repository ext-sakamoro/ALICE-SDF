//! Cone Tracing Global Illumination (Deep Fried Edition)
//!
//! Real-time GI using SDF cone tracing through Sparse Voxel Octrees.
//! Lumen-inspired approach: trace cones through the SVO to approximate
//! indirect lighting, then store in an irradiance probe grid.
//!
//! # Architecture
//!
//! - **Cone Trace**: March a cone through the SVO, accumulating radiance
//! - **Irradiance Probes**: 3D grid of L1 spherical harmonics probes
//! - **Probe Bake**: Fill probes by cone tracing from each probe position
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::gi::*;
//! use alice_sdf::svo::*;
//!
//! let svo = SparseVoxelOctree::build(&shape, &config);
//!
//! // Bake irradiance probes
//! let grid = bake_irradiance_grid(&svo, &BakeGiConfig::default());
//!
//! // Query indirect lighting at a point
//! let indirect = grid.sample(point, normal);
//! ```
//!
//! Author: Moroya Sakamoto

pub mod cone_trace;
pub mod irradiance;
pub mod probe_bake;

use glam::Vec3;

pub use cone_trace::{cone_trace, trace_hemisphere, ConeTraceConfig, ConeTraceResult};
pub use irradiance::{IrradianceGrid, IrradianceProbe};
pub use probe_bake::{bake_irradiance_grid, BakeGiConfig};

/// A directional light source for GI computation
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    /// Light direction (normalized, pointing toward the light)
    pub direction: Vec3,
    /// Light color/intensity (RGB, HDR)
    pub color: Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        DirectionalLight {
            direction: Vec3::new(0.5, 1.0, 0.3).normalize(),
            color: Vec3::new(1.0, 0.95, 0.8),
        }
    }
}

/// A point light source for GI computation
#[derive(Debug, Clone, Copy)]
pub struct PointLight {
    /// World-space position
    pub position: Vec3,
    /// Light color/intensity (RGB, HDR)
    pub color: Vec3,
    /// Attenuation radius
    pub radius: f32,
}

/// Simple sky color model for ambient lighting
///
/// Returns a color based on the direction (hemisphere blend).
#[inline]
pub fn sky_color(direction: Vec3) -> Vec3 {
    let t = direction.y * 0.5 + 0.5; // 0=ground, 1=sky
    let ground = Vec3::new(0.1, 0.08, 0.05);
    let sky = Vec3::new(0.4, 0.6, 1.0);
    ground + (sky - ground) * t
}

/// Compute direct lighting from a directional light at a surface point
#[inline]
pub fn direct_lighting(normal: Vec3, light: &DirectionalLight) -> Vec3 {
    let ndotl = normal.dot(light.direction).max(0.0);
    light.color * ndotl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sky_color() {
        let up = sky_color(Vec3::Y);
        let down = sky_color(Vec3::NEG_Y);
        // Sky should be bluer/brighter than ground
        assert!(up.z > down.z);
    }

    #[test]
    fn test_direct_lighting() {
        let light = DirectionalLight::default();
        let lit = direct_lighting(Vec3::Y, &light);
        assert!(lit.length() > 0.0);

        let unlit = direct_lighting(Vec3::NEG_Y, &light);
        assert!(unlit.length() < lit.length());
    }

    #[test]
    fn test_directional_light_default() {
        let light = DirectionalLight::default();
        assert!((light.direction.length() - 1.0).abs() < 0.001);
    }
}

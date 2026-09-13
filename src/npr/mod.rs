//! NPR (Non-Photorealistic Rendering) primitives
//!
//! Procedural stylized-shading building blocks derived only from SDF value,
//! surface normal, view direction, and light direction. No texture atlas
//! dependency: every output is a closed-form function of its inputs.
//!
//! # Categories
//!
//! - [`toon`]: N-band cel shading, soft ramps, two-tone, posterize
//! - [`outline`]: SDF-derived, curvature-derived, and depth-step outlines
//! - [`sky`]: Sky gradient bands, cloud coverage, distance color quantize,
//!   light shafts, sun disc
//!
//! Author: Moroya Sakamoto

pub mod compiled_color;
pub mod composition;
pub mod distortion;
pub mod dsl;
pub mod dsl_shader;
pub mod hatch;
pub mod motion;
pub mod noise;
pub mod outline;
pub mod palette;
pub mod rim;
#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
pub mod scene_composer;
pub mod sdf_integration;
pub mod shader_glue;
pub mod sky;
pub mod toon;

use glam::Vec3;

/// Common input bundle for NPR primitives that need multiple surface terms
///
/// Primitives are free to take fewer arguments; this struct is provided for
/// callers composing several primitives at the same shading point.
#[derive(Debug, Clone, Copy)]
pub struct NprInput {
    /// Signed distance at the shading point
    pub sdf: f32,
    /// Surface normal (assumed unit length)
    pub normal: Vec3,
    /// Direction from surface to camera (assumed unit length)
    pub view: Vec3,
    /// Direction from surface to light (assumed unit length)
    pub light: Vec3,
    /// World-space position of the shading point
    pub position: Vec3,
}

impl NprInput {
    /// Construct from all six components
    #[inline]
    #[must_use]
    pub const fn new(sdf: f32, normal: Vec3, view: Vec3, light: Vec3, position: Vec3) -> Self {
        Self {
            sdf,
            normal,
            view,
            light,
            position,
        }
    }

    /// Unclamped `normal . light`
    #[inline]
    #[must_use]
    pub fn n_dot_l(&self) -> f32 {
        self.normal.dot(self.light)
    }

    /// Unclamped `normal . view`
    #[inline]
    #[must_use]
    pub fn n_dot_v(&self) -> f32 {
        self.normal.dot(self.view)
    }
}

/// Linear-space RGB color used by NPR primitives (channels in `[0, 1]`)
pub type NprColor = Vec3;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npr_input_n_dot_l_matches_manual() {
        let n = Vec3::new(0.0, 1.0, 0.0);
        let l = Vec3::new(0.0, 1.0, 0.0);
        let input = NprInput::new(0.0, n, Vec3::Z, l, Vec3::ZERO);
        assert!((input.n_dot_l() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn npr_input_n_dot_v_matches_manual() {
        let n = Vec3::new(1.0, 0.0, 0.0);
        let v = Vec3::new(1.0, 0.0, 0.0);
        let input = NprInput::new(0.0, n, v, Vec3::Y, Vec3::ZERO);
        assert!((input.n_dot_v() - 1.0).abs() < 1e-6);
    }
}

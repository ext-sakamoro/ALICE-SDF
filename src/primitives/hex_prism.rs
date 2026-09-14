//! Hexagonal Prism SDF (Deep Fried Edition)
//!
//! Exact SDF for a hexagonal prism centered at origin.
//! Hexagon in the XY plane, extruded along Z-axis.
//!
//! Based on Inigo Quilez's sdHexPrism formula.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Hexagonal prism along Z (generic over [`Real`]).
#[inline(always)]
pub fn sdf_hex_prism_r<R: Real>(p: Vec3R<R>, hex_radius: f32, half_height: f32) -> R {
    let kx = R::splat(-0.8660254);
    let ky = R::splat(0.5);
    let kz = 0.57735027 * hex_radius;
    let mut px = p.x.abs();
    let mut py = p.y.abs();
    let pz = p.z.abs();
    let reflect = R::splat(2.0) * (kx * px + ky * py).min(R::zero());
    px = px - reflect * kx;
    py = py - reflect * ky;
    let clamped_x = px.clamp(R::splat(-kz), R::splat(kz));
    let dx = px - clamped_x;
    let dy = py - R::splat(hex_radius);
    let d_xy = (dx * dx + dy * dy).sqrt() * dy.signum();
    let d_z = pz - R::splat(half_height);
    let ox = d_xy.max(R::zero());
    let oz = d_z.max(R::zero());
    d_xy.max(d_z).min(R::zero()) + (ox * ox + oz * oz).sqrt()
}

/// Hexagonal prism along Z.
#[inline(always)]
pub fn sdf_hex_prism(p: Vec3, hex_radius: f32, half_height: f32) -> f32 {
    sdf_hex_prism_r::<f32>(p.into(), hex_radius, half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_prism_origin_inside() {
        let d = sdf_hex_prism(Vec3::ZERO, 1.0, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_hex_prism_vertex() {
        // At (0, hex_radius, 0), should be on surface
        let d = sdf_hex_prism(Vec3::new(0.0, 1.0, 0.0), 1.0, 1.0);
        assert!(
            d.abs() < 0.001,
            "Hex vertex should be on surface, got {}",
            d
        );
    }

    #[test]
    fn test_hex_prism_z_cap() {
        // At (0, 0, half_height), should be on surface
        let d = sdf_hex_prism(Vec3::new(0.0, 0.0, 1.0), 1.0, 1.0);
        assert!(d.abs() < 0.001, "Z cap should be on surface, got {}", d);
    }

    #[test]
    fn test_hex_prism_outside() {
        let d = sdf_hex_prism(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

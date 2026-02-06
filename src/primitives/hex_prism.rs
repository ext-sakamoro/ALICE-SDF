//! Hexagonal Prism SDF (Deep Fried Edition)
//!
//! Exact SDF for a hexagonal prism centered at origin.
//! Hexagon in the XY plane, extruded along Z-axis.
//!
//! Based on Inigo Quilez's sdHexPrism formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a hexagonal prism centered at origin
///
/// - Regular hexagon of given radius in XY plane
/// - Extruded along Z-axis by half_height
#[inline(always)]
pub fn sdf_hex_prism(p: Vec3, hex_radius: f32, half_height: f32) -> f32 {
    // k = vec3(-sqrt(3)/2, 0.5, 1/sqrt(3))
    let kx: f32 = -0.8660254;
    let ky: f32 = 0.5;
    let kz: f32 = 0.57735027;

    let mut px = p.x.abs();
    let mut py = p.y.abs();
    let pz = p.z.abs();

    // Reflect across hex symmetry
    let dot_kxy = kx * px + ky * py;
    let reflect = 2.0 * dot_kxy.min(0.0);
    px -= reflect * kx;
    py -= reflect * ky;

    // Distance in XY
    let clamped_x = px.clamp(-kz * hex_radius, kz * hex_radius);
    let dx = px - clamped_x;
    let dy = py - hex_radius;
    let d_xy = (dx * dx + dy * dy).sqrt() * dy.signum();

    // Distance along Z
    let d_z = pz - half_height;

    d_xy.max(d_z).min(0.0) + glam::Vec2::new(d_xy.max(0.0), d_z.max(0.0)).length()
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
        assert!(d.abs() < 0.001, "Hex vertex should be on surface, got {}", d);
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

//! Rounded X SDF (Deep Fried Edition)
//!
//! X-shaped cross with rounding in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's 2D Rounded X formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a rounded X shape, extruded along Y-axis
///
/// - `width`: arm length of the X in XZ plane
/// - `round_radius`: rounding radius
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_rounded_x(p: Vec3, width: f32, round_radius: f32, half_height: f32) -> f32 {
    // 2D Rounded X in XZ plane (IQ formula)
    let q = Vec2::new(p.x.abs(), p.z.abs());
    let s = (q.x + q.y).min(width) * 0.5;
    let d_2d = (q - Vec2::splat(s)).length() - round_radius;
    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_x_center_inside() {
        let d = sdf_rounded_x(Vec3::ZERO, 1.0, 0.2, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_rounded_x_far_outside() {
        let d = sdf_rounded_x(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.2, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_rounded_x_symmetry_xz() {
        let d1 = sdf_rounded_x(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.2, 0.5);
        let d2 = sdf_rounded_x(Vec3::new(-0.3, 0.1, -0.2), 1.0, 0.2, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in XZ");
    }

    #[test]
    fn test_rounded_x_symmetry_y() {
        let d1 = sdf_rounded_x(Vec3::new(0.3, 0.2, 0.1), 1.0, 0.2, 0.5);
        let d2 = sdf_rounded_x(Vec3::new(0.3, -0.2, 0.1), 1.0, 0.2, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

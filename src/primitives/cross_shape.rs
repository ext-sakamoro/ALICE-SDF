//! Cross shape SDF (Deep Fried Edition)
//!
//! 2D cross/plus shape in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdCross formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a cross (plus) shape, extruded along Y-axis
///
/// - `length`: half-length of the cross arms
/// - `thickness`: half-thickness of the cross arms
/// - `round_radius`: rounding radius (0 = sharp corners)
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_cross_shape(
    p: Vec3,
    length: f32,
    thickness: f32,
    round_radius: f32,
    half_height: f32,
) -> f32 {
    // 2D cross in XZ plane
    let qx = p.x.abs();
    let qz = p.z.abs();

    // Union of two rectangles: horizontal (length x thickness) and vertical (thickness x length)
    let d_h = Vec2::new(qx - length, qz - thickness);
    let d_v = Vec2::new(qx - thickness, qz - length);

    let d_h_sdf = Vec2::new(d_h.x.max(0.0), d_h.y.max(0.0)).length() + d_h.x.max(d_h.y).min(0.0);
    let d_v_sdf = Vec2::new(d_v.x.max(0.0), d_v.y.max(0.0)).length() + d_v.x.max(d_v.y).min(0.0);

    let d_2d = d_h_sdf.min(d_v_sdf) - round_radius;

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_shape_center_inside() {
        let d = sdf_cross_shape(Vec3::ZERO, 1.0, 0.3, 0.0, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_cross_shape_far_outside() {
        let d = sdf_cross_shape(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.3, 0.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_cross_shape_symmetry_x() {
        let d1 = sdf_cross_shape(Vec3::new(0.5, 0.1, 0.1), 1.0, 0.3, 0.05, 0.5);
        let d2 = sdf_cross_shape(Vec3::new(-0.5, 0.1, 0.1), 1.0, 0.3, 0.05, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_cross_shape_symmetry_z() {
        let d1 = sdf_cross_shape(Vec3::new(0.1, 0.1, 0.5), 1.0, 0.3, 0.05, 0.5);
        let d2 = sdf_cross_shape(Vec3::new(0.1, 0.1, -0.5), 1.0, 0.3, 0.05, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

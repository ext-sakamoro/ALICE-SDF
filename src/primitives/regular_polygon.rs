//! Regular polygon prism SDF (Deep Fried Edition)
//!
//! Regular N-sided polygon in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdPolygon formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a regular N-sided polygon, extruded along Y-axis
///
/// - `radius`: circumscribed circle radius (center to vertex)
/// - `n_sides`: number of sides (as f32, truncated to integer)
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_regular_polygon(p: Vec3, radius: f32, n_sides: f32, half_height: f32) -> f32 {
    let qx = p.x.abs();
    let qz = p.z;
    let n = n_sides.max(3.0);
    let an = std::f32::consts::PI / n;
    let he = radius * an.cos();

    // Rotate to first sector
    let angle = qx.atan2(qz);
    let bn = an * ((angle + an) / (2.0 * an)).floor();
    let cos_b = bn.cos();
    let sin_b = bn.sin();
    let rx = cos_b * qx + sin_b * qz;

    let d_2d = rx - he;

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_polygon_center_inside() {
        let d = sdf_regular_polygon(Vec3::ZERO, 1.0, 6.0, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_regular_polygon_far_outside() {
        let d = sdf_regular_polygon(Vec3::new(5.0, 0.0, 0.0), 1.0, 6.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_regular_polygon_symmetry_y() {
        let d1 = sdf_regular_polygon(Vec3::new(0.2, 0.2, 0.3), 1.0, 6.0, 0.5);
        let d2 = sdf_regular_polygon(Vec3::new(0.2, -0.2, 0.3), 1.0, 6.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }

    #[test]
    fn test_regular_polygon_symmetry_x() {
        let d1 = sdf_regular_polygon(Vec3::new(0.3, 0.1, 0.2), 1.0, 6.0, 0.5);
        let d2 = sdf_regular_polygon(Vec3::new(-0.3, 0.1, 0.2), 1.0, 6.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

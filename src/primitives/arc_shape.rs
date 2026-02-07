//! Arc shape SDF (Deep Fried Edition)
//!
//! 2D arc (thick ring sector) in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdArc formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for an arc (thick ring sector), extruded along Y-axis
///
/// - `aperture`: half opening angle in radians
/// - `radius`: arc center radius
/// - `thickness`: ring thickness (half)
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_arc_shape(p: Vec3, aperture: f32, radius: f32, thickness: f32, half_height: f32) -> f32 {
    // 2D arc in XZ plane (IQ's sdArc)
    let qx = p.x.abs();
    let qz = p.z;
    let sc = Vec2::new(aperture.sin(), aperture.cos());

    let d_2d = if sc.y * qx > sc.x * qz {
        // Inside the angular span
        (Vec2::new(qx, qz) - sc * radius).length() - thickness
    } else {
        // Outside: distance to the ring
        (Vec2::new(qx, qz).length() - radius).abs() - thickness
    };

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_shape_inside() {
        // Point on the arc center line at z=0.5 (within 45deg aperture, radius=1.0)
        let d = sdf_arc_shape(Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        assert!(d < 0.0, "Point on arc center should be inside, got {}", d);
    }

    #[test]
    fn test_arc_shape_far_outside() {
        let d = sdf_arc_shape(Vec3::new(5.0, 0.0, 0.0), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_arc_shape_symmetry_x() {
        let d1 = sdf_arc_shape(Vec3::new(0.3, 0.1, 0.8), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        let d2 = sdf_arc_shape(Vec3::new(-0.3, 0.1, 0.8), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_arc_shape_symmetry_y() {
        let d1 = sdf_arc_shape(Vec3::new(0.1, 0.2, 0.8), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        let d2 = sdf_arc_shape(Vec3::new(0.1, -0.2, 0.8), std::f32::consts::FRAC_PI_4, 1.0, 0.2, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

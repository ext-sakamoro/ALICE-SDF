//! Tube SDF (Deep Fried Edition)
//!
//! Hollow cylinder (pipe) along Y-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a tube (hollow cylinder) along Y-axis
///
/// - `outer_radius`: radius of the tube center-line
/// - `thickness`: half-wall thickness
/// - `half_height`: half the tube height
#[inline(always)]
pub fn sdf_tube(p: Vec3, outer_radius: f32, thickness: f32, half_height: f32) -> f32 {
    let r = Vec2::new(p.x, p.z).length();
    let d_ring = (r - outer_radius).abs() - thickness;
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_ring.max(0.0), d_y.max(0.0));
    d_ring.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tube_wall_inside() {
        let d = sdf_tube(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.1, 1.0);
        assert!(d < 0.0, "Point on tube wall ring should be inside, got {}", d);
    }

    #[test]
    fn test_tube_far_outside() {
        let d = sdf_tube(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.1, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_tube_symmetry() {
        let d1 = sdf_tube(Vec3::new(0.8, 0.3, 0.0), 1.0, 0.1, 1.0);
        let d2 = sdf_tube(Vec3::new(-0.8, -0.3, 0.0), 1.0, 0.1, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric");
    }

    #[test]
    fn test_tube_hollow_center() {
        let d = sdf_tube(Vec3::ZERO, 1.0, 0.1, 1.0);
        assert!(d > 0.0, "Center of hollow tube should be outside, got {}", d);
    }
}

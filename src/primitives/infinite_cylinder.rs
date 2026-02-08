//! Infinite Cylinder SDF (Deep Fried Edition)
//!
//! Infinite cylinder along Y-axis (no caps).
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for an infinite cylinder along Y-axis
///
/// - `radius`: cylinder radius
#[inline(always)]
pub fn sdf_infinite_cylinder(p: Vec3, radius: f32) -> f32 {
    (p.x * p.x + p.z * p.z).sqrt() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinite_cylinder_origin() {
        let d = sdf_infinite_cylinder(Vec3::ZERO, 1.0);
        assert!(
            (d + 1.0).abs() < 0.001,
            "Origin should be -radius, got {}",
            d
        );
    }

    #[test]
    fn test_infinite_cylinder_on_surface() {
        let d = sdf_infinite_cylinder(Vec3::new(1.0, 100.0, 0.0), 1.0);
        assert!(d.abs() < 0.001, "On surface should be ~0, got {}", d);
    }

    #[test]
    fn test_infinite_cylinder_outside() {
        let d = sdf_infinite_cylinder(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!((d - 4.0).abs() < 0.001, "Should be 4.0, got {}", d);
    }

    #[test]
    fn test_infinite_cylinder_y_invariant() {
        let d1 = sdf_infinite_cylinder(Vec3::new(0.5, 0.0, 0.0), 1.0);
        let d2 = sdf_infinite_cylinder(Vec3::new(0.5, 999.0, 0.0), 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be Y-invariant");
    }
}

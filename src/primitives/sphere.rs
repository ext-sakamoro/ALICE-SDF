//! Sphere primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: Zero call overhead.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Signed distance to a sphere centered at origin
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `radius` - Sphere radius
///
/// # Returns
/// Signed distance (negative inside, positive outside)
#[inline(always)]
pub fn sdf_sphere(point: Vec3, radius: f32) -> f32 {
    point.length() - radius
}

/// Signed distance to a sphere at arbitrary center
#[inline(always)]
pub fn sdf_sphere_at(point: Vec3, center: Vec3, radius: f32) -> f32 {
    (point - center).length() - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_origin() {
        // Center of sphere
        assert!((sdf_sphere(Vec3::ZERO, 1.0) + 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_sphere_surface() {
        // On surface
        assert!((sdf_sphere(Vec3::new(1.0, 0.0, 0.0), 1.0)).abs() < 0.0001);
        assert!((sdf_sphere(Vec3::new(0.0, 1.0, 0.0), 1.0)).abs() < 0.0001);
        assert!((sdf_sphere(Vec3::new(0.0, 0.0, 1.0), 1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_sphere_outside() {
        // Outside
        let d = sdf_sphere(Vec3::new(2.0, 0.0, 0.0), 1.0);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_sphere_inside() {
        // Inside
        let d = sdf_sphere(Vec3::new(0.5, 0.0, 0.0), 1.0);
        assert!((d + 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_sphere_at() {
        let center = Vec3::new(1.0, 2.0, 3.0);
        let d = sdf_sphere_at(center, center, 1.0);
        assert!((d + 1.0).abs() < 0.0001);
    }
}

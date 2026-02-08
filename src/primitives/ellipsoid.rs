//! Ellipsoid SDF (Deep Fried Edition)
//!
//! Approximate but high-quality SDF for an ellipsoid.
//! Based on Inigo Quilez's formula: k0*(k0-1)/k1
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for an ellipsoid centered at origin
///
/// Uses Inigo Quilez's formula which provides a very good approximation.
/// The error is negligible for most practical uses.
///
/// # Arguments
/// * `p` - Point to evaluate
/// * `radii` - Semi-axes lengths (x, y, z)
#[inline(always)]
pub fn sdf_ellipsoid(p: Vec3, radii: Vec3) -> f32 {
    // Guard against zero radii (prevents division by zero in p/radii)
    let safe_radii = Vec3::new(radii.x.max(1e-10), radii.y.max(1e-10), radii.z.max(1e-10));
    let k0 = (p / safe_radii).length();
    let k1 = (p / (safe_radii * safe_radii)).length();
    if k1 < 1e-10 {
        return -safe_radii.x.min(safe_radii.y).min(safe_radii.z);
    }
    k0 * (k0 - 1.0) / k1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ellipsoid_origin() {
        // At origin, should be inside (negative distance)
        let d = sdf_ellipsoid(Vec3::ZERO, Vec3::new(1.0, 2.0, 1.0));
        assert!(d < 0.0, "Origin should be inside ellipsoid, got {}", d);
    }

    #[test]
    fn test_ellipsoid_on_surface_x() {
        // On the x-axis surface
        let d = sdf_ellipsoid(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        assert!(d.abs() < 0.01, "Should be on surface at x=rx, got {}", d);
    }

    #[test]
    fn test_ellipsoid_on_surface_y() {
        // On the y-axis surface
        let d = sdf_ellipsoid(Vec3::new(0.0, 2.0, 0.0), Vec3::new(1.0, 2.0, 1.0));
        assert!(d.abs() < 0.01, "Should be on surface at y=ry, got {}", d);
    }

    #[test]
    fn test_ellipsoid_sphere_equivalence() {
        // With equal radii, should approximate a sphere
        let r = 1.5;
        let p = Vec3::new(r, 0.0, 0.0);
        let d = sdf_ellipsoid(p, Vec3::splat(r));
        assert!(
            d.abs() < 0.01,
            "Equal radii should be like a sphere, got {}",
            d
        );
    }

    #[test]
    fn test_ellipsoid_outside() {
        let d = sdf_ellipsoid(Vec3::new(5.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

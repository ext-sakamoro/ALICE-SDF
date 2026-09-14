//! Ellipsoid SDF (Deep Fried Edition)
//!
//! Approximate but high-quality SDF for an ellipsoid.
//! Based on Inigo Quilez's formula: k0*(k0-1)/k1
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Ellipsoid with `radii` (generic over [`Real`]); returns `-min(radii)` at the centre.
#[inline(always)]
pub fn sdf_ellipsoid_r<R: Real>(p: Vec3R<R>, radii: Vec3) -> R {
    let safe = Vec3::new(radii.x.max(1e-10), radii.y.max(1e-10), radii.z.max(1e-10));
    let inv = Vec3R::<R>::splat(Vec3::new(1.0 / safe.x, 1.0 / safe.y, 1.0 / safe.z));
    let inv2 = Vec3R::<R>::splat(Vec3::new(
        1.0 / (safe.x * safe.x),
        1.0 / (safe.y * safe.y),
        1.0 / (safe.z * safe.z),
    ));
    let k0 = p.mul_vec(inv).length();
    let k1 = p.mul_vec(inv2).length();
    let eps = R::splat(1e-10);
    let d = k0 * (k0 - R::one()) / k1.max(eps);
    let centre = R::splat(-safe.x.min(safe.y).min(safe.z));
    R::select(k1.lt(eps), centre, d)
}

/// Ellipsoid with `radii`; returns `-min(radii)` at the centre.
#[inline(always)]
pub fn sdf_ellipsoid(p: Vec3, radii: Vec3) -> f32 {
    sdf_ellipsoid_r::<f32>(p.into(), radii)
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

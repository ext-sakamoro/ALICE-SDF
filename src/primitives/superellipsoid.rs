//! Superellipsoid SDF (Deep Fried Edition)
//!
//! Generalized ellipsoid that smoothly morphs between sphere, box, and cylinder.
//!
//! Parameters e1 and e2 control the shape:
//! - e1=1, e2=1: ellipsoid
//! - e1→0, e2→0: box-like
//! - e1=2, e2=2: concave pinch
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a superellipsoid
///
/// - `half_extents`: semi-axis radii
/// - `e1`: north-south roundness (Y axis)
/// - `e2`: east-west roundness (XZ plane)
#[inline(always)]
pub fn sdf_superellipsoid(p: Vec3, half_extents: Vec3, e1: f32, e2: f32) -> f32 {
    let e1 = e1.max(0.02);
    let e2 = e2.max(0.02);
    let q = Vec3::new(
        (p.x / half_extents.x).abs().max(1e-10),
        (p.y / half_extents.y).abs().max(1e-10),
        (p.z / half_extents.z).abs().max(1e-10),
    );
    let m1 = 2.0 / e2;
    let m2 = 2.0 / e1;
    let w = q.x.powf(m1) + q.z.powf(m1);
    let v = w.powf(e2 / e1) + q.y.powf(m2);
    let f = v.powf(e1 * 0.5);
    let min_extent = half_extents.x.min(half_extents.y.min(half_extents.z));
    (f - 1.0) * min_extent * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_superellipsoid_center_inside() {
        let d = sdf_superellipsoid(Vec3::ZERO, Vec3::splat(1.0), 1.0, 1.0);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_superellipsoid_far_outside() {
        let d = sdf_superellipsoid(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0), 1.0, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_superellipsoid_symmetry() {
        let d1 = sdf_superellipsoid(Vec3::new(0.3, 0.4, 0.2), Vec3::splat(1.0), 0.5, 0.5);
        let d2 = sdf_superellipsoid(Vec3::new(-0.3, -0.4, -0.2), Vec3::splat(1.0), 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric");
    }

    #[test]
    fn test_superellipsoid_near_surface() {
        let d = sdf_superellipsoid(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(1.0), 1.0, 1.0);
        assert!(d.abs() < 0.1, "e1=e2=1 should be near surface at r=1, got {}", d);
    }
}

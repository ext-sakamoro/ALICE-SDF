//! Plane primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: Zero call overhead.
//! - **Minimal Operations**: Axis-aligned planes use single component access.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Signed distance to an infinite plane
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `normal` - Plane normal (should be normalized)
/// * `distance` - Distance from origin to plane along normal
///
/// # Returns
/// Signed distance (negative below plane, positive above)
#[inline(always)]
pub fn sdf_plane(point: Vec3, normal: Vec3, distance: f32) -> f32 {
    point.dot(normal) - distance
}

/// Signed distance to the XY plane (Z = 0)
#[inline(always)]
pub fn sdf_plane_xy(point: Vec3) -> f32 {
    point.z
}

/// Signed distance to the XZ plane (Y = 0)
#[inline(always)]
pub fn sdf_plane_xz(point: Vec3) -> f32 {
    point.y
}

/// Signed distance to the YZ plane (X = 0)
#[inline(always)]
pub fn sdf_plane_yz(point: Vec3) -> f32 {
    point.x
}

/// Signed distance to a plane defined by three points
#[inline(always)]
pub fn sdf_plane_from_points(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let normal = ab.cross(ac).normalize();
    let distance = normal.dot(a);
    sdf_plane(point, normal, distance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_horizontal() {
        // Horizontal plane at Y = 0
        let normal = Vec3::new(0.0, 1.0, 0.0);

        // Above plane
        let d = sdf_plane(Vec3::new(0.0, 1.0, 0.0), normal, 0.0);
        assert!((d - 1.0).abs() < 0.0001);

        // On plane
        let d = sdf_plane(Vec3::new(5.0, 0.0, -3.0), normal, 0.0);
        assert!(d.abs() < 0.0001);

        // Below plane
        let d = sdf_plane(Vec3::new(0.0, -2.0, 0.0), normal, 0.0);
        assert!((d + 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_plane_offset() {
        // Plane at Y = 1
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let d = sdf_plane(Vec3::new(0.0, 1.0, 0.0), normal, 1.0);
        assert!(d.abs() < 0.0001);

        let d = sdf_plane(Vec3::new(0.0, 2.0, 0.0), normal, 1.0);
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_plane_diagonal() {
        // 45-degree plane
        let normal = Vec3::new(1.0, 1.0, 0.0).normalize();

        let d = sdf_plane(Vec3::ZERO, normal, 0.0);
        assert!(d.abs() < 0.0001);

        // Point at (1, 1, 0) should be at distance sqrt(2) from origin along normal
        let d = sdf_plane(Vec3::new(1.0, 1.0, 0.0), normal, 0.0);
        let expected = 2.0_f32.sqrt();
        assert!((d - expected).abs() < 0.0001);
    }

    #[test]
    fn test_axis_planes() {
        assert!((sdf_plane_xy(Vec3::new(0.0, 0.0, 1.0)) - 1.0).abs() < 0.0001);
        assert!((sdf_plane_xz(Vec3::new(0.0, 1.0, 0.0)) - 1.0).abs() < 0.0001);
        assert!((sdf_plane_yz(Vec3::new(1.0, 0.0, 0.0)) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_plane_from_points() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 0.0, 1.0);

        // This creates the XZ plane (Y = 0)
        let d = sdf_plane_from_points(Vec3::new(0.0, 1.0, 0.0), a, b, c);
        // Note: normal direction depends on winding order
        assert!((d.abs() - 1.0).abs() < 0.0001);
    }
}

//! Box primitive SDF (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Branchless Logic**: Uses standard max/min logic for interior/exterior.
//! - **Forced Inlining**: Zero call overhead.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Signed distance to an axis-aligned box centered at origin
///
/// # Arguments
/// * `point` - Point to evaluate
/// * `half_extents` - Half-size in each dimension (width/2, height/2, depth/2)
///
/// # Returns
/// Signed distance (negative inside, positive outside)
#[inline(always)]
pub fn sdf_box3d(point: Vec3, half_extents: Vec3) -> f32 {
    let q = point.abs() - half_extents;
    // Branchless combine of interior (negative) and exterior (positive) distance
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
}

/// Signed distance to a box at arbitrary center
#[inline(always)]
pub fn sdf_box3d_at(point: Vec3, center: Vec3, half_extents: Vec3) -> f32 {
    sdf_box3d(point - center, half_extents)
}

/// Signed distance to a rounded box (box with rounded edges)
#[inline(always)]
pub fn sdf_rounded_box3d(point: Vec3, half_extents: Vec3, radius: f32) -> f32 {
    sdf_box3d(point, half_extents - Vec3::splat(radius)) - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_origin() {
        // Center of unit box
        let d = sdf_box3d(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0));
        assert!((d + 1.0).abs() < 0.0001); // Distance to surface is -1
    }

    #[test]
    fn test_box_surface() {
        // On surface (face center)
        let d = sdf_box3d(Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(d.abs() < 0.0001);
    }

    #[test]
    fn test_box_outside() {
        // Outside along X axis
        let d = sdf_box3d(Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!((d - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_box_corner() {
        // Outside at corner - distance to corner is sqrt(3)
        let d = sdf_box3d(Vec3::new(2.0, 2.0, 2.0), Vec3::new(1.0, 1.0, 1.0));
        let expected = (3.0_f32).sqrt(); // Distance from (2,2,2) to (1,1,1)
        assert!((d - expected).abs() < 0.0001);
    }

    #[test]
    fn test_box_inside() {
        // Inside
        let d = sdf_box3d(Vec3::new(0.5, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0));
        assert!((d + 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_rounded_box() {
        let d = sdf_rounded_box3d(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), 0.1);
        // Center distance should be approximately -(1 - 0.1) - 0.1 = -1.0
        assert!(d < 0.0);
    }
}

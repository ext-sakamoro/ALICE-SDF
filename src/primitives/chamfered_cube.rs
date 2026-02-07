//! Chamfered Cube SDF (Deep Fried Edition)
//!
//! Axis-aligned box with chamfered (beveled) edges.
//! Intersection of a box and an octahedral constraint.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a chamfered cube
///
/// - `half_extents`: half-size along each axis
/// - `chamfer`: chamfer amount (0 = regular box, larger = more chamfer)
#[inline(always)]
pub fn sdf_chamfered_cube(p: Vec3, half_extents: Vec3, chamfer: f32) -> f32 {
    let ap = p.abs();
    // Box SDF
    let q = ap - half_extents;
    let d_box = q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0);
    // Octahedral chamfer plane
    let s = half_extents.x + half_extents.y + half_extents.z;
    let inv_sqrt3 = 0.57735027;
    let d_chamfer = (ap.x + ap.y + ap.z - s + chamfer) * inv_sqrt3;
    d_box.max(d_chamfer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chamfered_cube_center_inside() {
        let d = sdf_chamfered_cube(Vec3::ZERO, Vec3::splat(1.0), 0.3);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_chamfered_cube_far_outside() {
        let d = sdf_chamfered_cube(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0), 0.3);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_chamfered_cube_symmetry() {
        let d1 = sdf_chamfered_cube(Vec3::new(0.5, 0.3, 0.2), Vec3::splat(1.0), 0.3);
        let d2 = sdf_chamfered_cube(Vec3::new(-0.5, -0.3, -0.2), Vec3::splat(1.0), 0.3);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric");
    }

    #[test]
    fn test_chamfered_cube_corner_cut() {
        let d_no_chamfer = sdf_chamfered_cube(Vec3::new(0.9, 0.9, 0.9), Vec3::splat(1.0), 0.0);
        let d_with_chamfer = sdf_chamfered_cube(Vec3::new(0.9, 0.9, 0.9), Vec3::splat(1.0), 0.5);
        assert!(d_with_chamfer > d_no_chamfer, "Chamfer should cut corners");
    }
}

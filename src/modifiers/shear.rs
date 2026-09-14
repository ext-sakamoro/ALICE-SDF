//! Shear modifier (single source of truth)
//!
//! Inverse shear applied to the evaluation point so the child appears sheared.
//! Used by the tree evaluator, compiled scalar / BVH evaluator and SIMD path.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Inverse shear: `shear = (xy, xz, yz)`.
///
/// `y' = y - xy * x`, `z' = z - xz * x - yz * y`.
#[inline(always)]
pub fn modifier_shear(point: Vec3, shear: Vec3) -> Vec3 {
    Vec3::new(
        point.x,
        shear.x.mul_add(-point.x, point.y),
        shear
            .z
            .mul_add(-point.y, shear.y.mul_add(-point.x, point.z)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_shear_is_identity() {
        let p = Vec3::new(0.3, -0.7, 1.1);
        assert_eq!(modifier_shear(p, Vec3::ZERO), p);
    }

    #[test]
    fn shear_xy_moves_y_by_x() {
        let p = Vec3::new(2.0, 1.0, 0.0);
        let q = modifier_shear(p, Vec3::new(0.5, 0.0, 0.0));
        assert!((q.y - 0.0).abs() < 1e-6);
        assert_eq!(q.x, 2.0);
        assert_eq!(q.z, 0.0);
    }
}

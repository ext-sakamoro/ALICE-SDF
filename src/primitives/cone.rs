//! Cone SDF (Deep Fried Edition)
//!
//! Exact SDF for a capped cone along Y-axis.
//! Base at y = -half_height with given radius, tip at y = half_height.
//!
//! Based on Inigo Quilez's exact cone SDF formula.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Cone with base `radius` and `half_height`, apex up (generic over [`Real`]).
#[inline(always)]
pub fn sdf_cone_r<R: Real>(p: Vec3R<R>, radius: f32, half_height: f32) -> R {
    let q_x = (p.x * p.x + p.z * p.z).sqrt();
    let q_y = p.y;
    let h = R::splat(half_height);
    let radius = R::splat(radius);
    let k2x = -radius;
    let k2y = h + h;
    let ca_r = R::select(q_y.lt(R::zero()), radius, R::zero());
    let ca_x = q_x - q_x.min(ca_r);
    let ca_y = q_y.abs() - h;
    let diff_x = -q_x;
    let diff_y = h - q_y;
    // (diff · k2) / (k2 · k2): projection parameter, both products are dot products
    #[allow(clippy::suspicious_operation_groupings)]
    let t = ((diff_x * k2x + diff_y * k2y) / (k2x * k2x + k2y * k2y)).clamp(R::zero(), R::one());
    let cb_x = q_x + k2x * t;
    let cb_y = q_y - h + k2y * t;
    let both_neg = R::mask_and(cb_x.lt(R::zero()), ca_y.lt(R::zero()));
    let s = R::select(both_neg, R::splat(-1.0), R::one());
    let d2 = (ca_x * ca_x + ca_y * ca_y).min(cb_x * cb_x + cb_y * cb_y);
    s * d2.sqrt()
}

/// Cone with base `radius` and `half_height`, apex up.
#[inline(always)]
pub fn sdf_cone(p: Vec3, radius: f32, half_height: f32) -> f32 {
    sdf_cone_r::<f32>(p.into(), radius, half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cone_origin_inside() {
        // Origin should be inside the cone
        let d = sdf_cone(Vec3::ZERO, 1.0, 1.0);
        assert!(d < 0.0, "Origin should be inside cone, got {}", d);
    }

    #[test]
    fn test_cone_tip() {
        // At the tip (0, half_height, 0), distance should be ~0
        let d = sdf_cone(Vec3::new(0.0, 1.0, 0.0), 1.0, 1.0);
        assert!(d.abs() < 0.001, "Tip should be on surface, got {}", d);
    }

    #[test]
    fn test_cone_base_edge() {
        // At (radius, -half_height, 0), should be on surface
        let d = sdf_cone(Vec3::new(1.0, -1.0, 0.0), 1.0, 1.0);
        assert!(d.abs() < 0.001, "Base edge should be on surface, got {}", d);
    }

    #[test]
    fn test_cone_outside() {
        // Far outside
        let d = sdf_cone(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0);
        assert!(d > 0.0, "Point far outside should be positive, got {}", d);
    }
}

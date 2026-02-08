//! Capped Cone SDF (Deep Fried Edition)
//!
//! Truncated cone (frustum) centered at origin along Y-axis.
//! Bottom radius r1 at y = -half_height, top radius r2 at y = +half_height.
//!
//! Based on Inigo Quilez's sdCappedCone formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a capped cone centered at origin
///
/// - `half_height`: half the cone height
/// - `r1`: bottom radius (at y = -half_height)
/// - `r2`: top radius (at y = +half_height)
#[inline(always)]
pub fn sdf_capped_cone(p: Vec3, half_height: f32, r1: f32, r2: f32) -> f32 {
    let h = half_height;
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), p.y);
    let k1 = Vec2::new(r2, h);
    let k2 = Vec2::new(r2 - r1, 2.0 * h);

    let min_r = if q.y < 0.0 { r1 } else { r2 };
    let ca = Vec2::new(q.x - q.x.min(min_r), q.y.abs() - h);
    let k2_dot = k2.dot(k2);
    let t = if k2_dot > 0.0001 {
        ((k1 - q).dot(k2) / k2_dot).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let cb = q - k1 + k2 * t;

    let s = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    s * ca.dot(ca).min(cb.dot(cb)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capped_cone_origin() {
        let d = sdf_capped_cone(Vec3::ZERO, 1.0, 0.5, 0.3);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_capped_cone_outside() {
        let d = sdf_capped_cone(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 0.3);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_capped_cone_cylinder_degenerate() {
        // When r1 == r2, it's a cylinder
        let d = sdf_capped_cone(Vec3::new(0.5, 0.0, 0.0), 1.0, 1.0, 1.0);
        assert!(d < 0.0, "Inside cylinder should be negative, got {}", d);
    }

    #[test]
    fn test_capped_cone_symmetry() {
        let d1 = sdf_capped_cone(Vec3::new(0.3, 0.5, 0.0), 1.0, 0.5, 0.3);
        let d2 = sdf_capped_cone(Vec3::new(-0.3, 0.5, 0.0), 1.0, 0.5, 0.3);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

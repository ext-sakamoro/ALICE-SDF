//! Cone SDF (Deep Fried Edition)
//!
//! Exact SDF for a capped cone along Y-axis.
//! Base at y = -half_height with given radius, tip at y = half_height.
//!
//! Based on Inigo Quilez's exact cone SDF formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a capped cone along Y-axis
///
/// - Base circle at y = -half_height with given radius
/// - Tip at y = half_height
///
/// # Deep Fried: branchless min/max, no sqrt until final
#[inline(always)]
pub fn sdf_cone(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let q_x = (p.x * p.x + p.z * p.z).sqrt();
    let q_y = p.y;

    let h = half_height;

    // k1 = tip position in 2D (0, h)
    // k2 = direction from tip to base edge (-radius, 2*h)
    let k2x = -radius;
    let k2y = 2.0 * h;

    // ca: closest point on caps
    let ca_r = if q_y < 0.0 { radius } else { 0.0 };
    let ca_x = q_x - q_x.min(ca_r);
    let ca_y = q_y.abs() - h;

    // cb: closest point on mantle
    // cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0, 1)
    let diff_x = -q_x; // k1.x - q.x = 0 - q_x
    let diff_y = h - q_y;
    let t = ((diff_x * k2x + diff_y * k2y) / (k2x * k2x + k2y * k2y)).clamp(0.0, 1.0);
    let cb_x = q_x + k2x * t; // q.x - k1.x + k2.x * t = q_x + k2x * t
    let cb_y = q_y - h + k2y * t;

    let s = if cb_x < 0.0 && ca_y < 0.0 { -1.0 } else { 1.0 };
    let d2 = (ca_x * ca_x + ca_y * ca_y).min(cb_x * cb_x + cb_y * cb_y);

    s * d2.sqrt()
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

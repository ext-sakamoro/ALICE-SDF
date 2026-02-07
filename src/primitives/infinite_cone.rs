//! Infinite Cone SDF (Deep Fried Edition)
//!
//! Infinite cone along Y-axis (extends upward from origin).
//!
//! Based on Inigo Quilez's sdCone (infinite version) formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for an infinite cone along Y-axis
///
/// - `angle`: half-angle of the cone in radians
#[inline(always)]
pub fn sdf_infinite_cone(p: Vec3, angle: f32) -> f32 {
    let c = Vec2::new(angle.sin(), angle.cos());
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), -p.y);
    let d = (q - c * q.dot(c).max(0.0)).length();
    d * if q.x * c.y - q.y * c.x < 0.0 { -1.0 } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infinite_cone_origin() {
        let d = sdf_infinite_cone(Vec3::ZERO, 0.5);
        assert!(d.abs() < 0.001, "Origin is on the tip, got {}", d);
    }

    #[test]
    fn test_infinite_cone_inside() {
        // Cone opens downward (q.y = -p.y), so negative Y axis is inside
        let d = sdf_infinite_cone(Vec3::new(0.0, -1.0, 0.0), 0.5);
        assert!(d < 0.0, "On-axis below should be inside, got {}", d);
    }

    #[test]
    fn test_infinite_cone_outside() {
        // Point far to the side
        let d = sdf_infinite_cone(Vec3::new(5.0, 0.0, 0.0), 0.3);
        assert!(d > 0.0, "Far side should be outside, got {}", d);
    }
}

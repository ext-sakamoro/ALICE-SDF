//! Solid Angle SDF (Deep Fried Edition)
//!
//! A 3D cone sector (like a spotlight cone shape).
//!
//! Based on Inigo Quilez's sdSolidAngle formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a solid angle centered at origin
///
/// - `angle`: half-angle of the cone in radians
/// - `radius`: sphere radius bounding the solid angle
#[inline(always)]
pub fn sdf_solid_angle(p: Vec3, angle: f32, radius: f32) -> f32 {
    let c = Vec2::new(angle.sin(), angle.cos());
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), p.y);
    let l = q.length() - radius;
    let m = (q - c * q.dot(c).clamp(0.0, radius)).length();
    let sign = if c.y * q.x - c.x * q.y < 0.0 {
        -1.0
    } else {
        1.0
    };
    l.max(m * sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solid_angle_origin() {
        // Origin is on the cone tip, distance = max(-radius, 0) = 0
        let d = sdf_solid_angle(Vec3::ZERO, 0.5, 1.0);
        assert!(
            d <= 0.001,
            "Origin should be on or inside surface, got {}",
            d
        );
    }

    #[test]
    fn test_solid_angle_inside() {
        // Point slightly inside the cone
        let d = sdf_solid_angle(Vec3::new(0.0, 0.5, 0.0), 1.0, 2.0);
        assert!(d < 0.0, "Inside cone should be negative, got {}", d);
    }

    #[test]
    fn test_solid_angle_outside() {
        let d = sdf_solid_angle(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_solid_angle_above() {
        // Point straight up inside the cone angle
        let d = sdf_solid_angle(Vec3::new(0.0, 0.5, 0.0), 1.0, 1.0);
        assert!(d < 0.0, "Inside cone should be negative, got {}", d);
    }
}

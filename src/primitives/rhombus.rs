//! Rhombus SDF (Deep Fried Edition)
//!
//! 3D rhombus (diamond shape) centered at origin.
//!
//! Based on Inigo Quilez's sdRhombus formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Dot product variation: a.x*b.x - a.y*b.y
#[inline(always)]
fn ndot(a: Vec2, b: Vec2) -> f32 {
    a.x * b.x - a.y * b.y
}

/// Exact SDF for a rhombus centered at origin
///
/// - `la`: half-diagonal length along X
/// - `lb`: half-diagonal length along Z
/// - `half_height`: half the height along Y
/// - `round_radius`: edge rounding
#[inline(always)]
pub fn sdf_rhombus(p: Vec3, la: f32, lb: f32, half_height: f32, round_radius: f32) -> f32 {
    let p = p.abs();
    let b = Vec2::new(la, lb);
    let f = ndot(b, b - 2.0 * Vec2::new(p.x, p.z));
    let f = (f / b.dot(b)).clamp(-1.0, 1.0);

    let qx = (Vec2::new(p.x, p.z) - 0.5 * b * Vec2::new(1.0 - f, 1.0 + f)).length()
        * (p.x * b.y + p.z * b.x - b.x * b.y).signum()
        - round_radius;
    let qy = p.y - half_height;

    let d = Vec2::new(qx, qy);
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rhombus_origin() {
        let d = sdf_rhombus(Vec3::ZERO, 1.0, 0.5, 0.3, 0.05);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_rhombus_outside() {
        let d = sdf_rhombus(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 0.3, 0.05);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_rhombus_symmetry() {
        let d1 = sdf_rhombus(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.5, 0.3, 0.05);
        let d2 = sdf_rhombus(Vec3::new(-0.3, -0.1, -0.2), 1.0, 0.5, 0.3, 0.05);
        assert!((d1 - d2).abs() < 0.001, "Should be fully symmetric");
    }
}

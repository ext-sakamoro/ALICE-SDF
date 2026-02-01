//! Twist modifier for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Simultaneous Trig**: Uses `sin_cos()` for single-instruction computation.
//! - **Forced Inlining**: `#[inline(always)]`.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Twist space around the Y-axis
#[inline(always)]
pub fn modifier_twist(point: Vec3, strength: f32) -> Vec3 {
    let (s, c) = (point.y * strength).sin_cos();
    Vec3::new(point.x * c - point.z * s, point.y, point.x * s + point.z * c)
}

/// Twist space around the X-axis
#[inline(always)]
pub fn modifier_twist_x(point: Vec3, strength: f32) -> Vec3 {
    let (s, c) = (point.x * strength).sin_cos();
    Vec3::new(point.x, point.y * c - point.z * s, point.y * s + point.z * c)
}

/// Twist space around the Z-axis
#[inline(always)]
pub fn modifier_twist_z(point: Vec3, strength: f32) -> Vec3 {
    let (s, c) = (point.z * strength).sin_cos();
    Vec3::new(point.x * c - point.y * s, point.x * s + point.y * c, point.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_twist_equivalence() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let s = 0.5;
        let t1 = modifier_twist(p, s);
        // Manual calc
        let a = p.y * s;
        let t2 = Vec3::new(p.x * a.cos() - p.z * a.sin(), p.y, p.x * a.sin() + p.z * a.cos());
        assert!((t1 - t2).length() < 1e-6);
    }

    #[test]
    fn test_twist_at_origin() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = modifier_twist(point, 1.0);
        assert!((result - point).length() < 0.0001);
    }

    #[test]
    fn test_twist_90_degrees() {
        let point = Vec3::new(1.0, PI / 2.0, 0.0);
        let result = modifier_twist(point, 1.0);
        let expected = Vec3::new(0.0, PI / 2.0, 1.0);
        assert!((result - expected).length() < 0.0001);
    }
}

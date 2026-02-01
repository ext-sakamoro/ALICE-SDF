//! Repetition modifiers for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Fast Modulo**: Replaced general modulo with `x - s * round(x/s)` logic.
//!   This relies on `round` (single instruction) vs multiple ops for floor/mod.
//! - **Forced Inlining**: `#[inline(always)]`.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Infinite repetition along all axes (Deep Fried)
#[inline(always)]
pub fn modifier_repeat_infinite(point: Vec3, spacing: Vec3) -> Vec3 {
    // p - s * round(p / s) maps to [-s/2, s/2] range
    // Much faster than standard euclidean modulo logic
    Vec3::new(
        point.x - spacing.x * (point.x / spacing.x).round(),
        point.y - spacing.y * (point.y / spacing.y).round(),
        point.z - spacing.z * (point.z / spacing.z).round(),
    )
}

/// Finite repetition along all axes (Deep Fried)
#[inline(always)]
pub fn modifier_repeat_finite(point: Vec3, count: [u32; 3], spacing: Vec3) -> Vec3 {
    let limit = Vec3::new(count[0] as f32, count[1] as f32, count[2] as f32) * 0.5;
    // clamp(round(p/s), -limit, limit)
    let cell = (point / spacing).round().clamp(-limit, limit);
    point - cell * spacing
}

/// Infinite repetition along a single axis (X)
#[inline(always)]
pub fn modifier_repeat_x(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        point.x - spacing * (point.x / spacing).round(),
        point.y,
        point.z,
    )
}

/// Infinite repetition along a single axis (Y)
#[inline(always)]
pub fn modifier_repeat_y(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        point.x,
        point.y - spacing * (point.y / spacing).round(),
        point.z,
    )
}

/// Infinite repetition along a single axis (Z)
#[inline(always)]
pub fn modifier_repeat_z(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        point.x,
        point.y,
        point.z - spacing * (point.z / spacing).round(),
    )
}

/// Polar repetition around Y axis
#[inline(always)]
pub fn modifier_repeat_polar(point: Vec3, count: u32) -> Vec3 {
    // Polar repeat is inherently somewhat heavy (atan2, sin/cos)
    // We minimize overheads around it.
    let angle = point.z.atan2(point.x);
    let radius = (point.x * point.x + point.z * point.z).sqrt();

    let sector = std::f32::consts::TAU / count as f32;
    // Align to sector center: a - s * round(a/s)
    let sector_angle = angle - sector * (angle / sector).round();

    let (s, c) = sector_angle.sin_cos();
    Vec3::new(radius * c, point.y, radius * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repeat_equivalence() {
        let p = Vec3::new(3.2, 0.0, 0.0);
        let s = Vec3::splat(2.0);
        let r = modifier_repeat_infinite(p, s);
        // 3.2 - 2.0 * round(1.6) = 3.2 - 4.0 = -0.8
        assert!((r.x - (-0.8)).abs() < 1e-6);
    }

    #[test]
    fn test_repeat_infinite_origin() {
        let spacing = Vec3::splat(2.0);
        let result = modifier_repeat_infinite(Vec3::ZERO, spacing);
        assert!((result - Vec3::ZERO).length() < 0.0001);
    }

    #[test]
    fn test_repeat_finite_center() {
        let count = [3, 3, 3];
        let spacing = Vec3::splat(2.0);
        let result = modifier_repeat_finite(Vec3::ZERO, count, spacing);
        assert!((result - Vec3::ZERO).length() < 0.0001);
    }

    #[test]
    fn test_repeat_x() {
        let point = Vec3::new(3.2, 1.0, 2.0);
        let result = modifier_repeat_x(point, 2.0);
        // 3.2 - 2.0 * round(1.6) = 3.2 - 4.0 = -0.8
        assert!((result.x - (-0.8)).abs() < 0.0001);
        assert!((result.y - 1.0).abs() < 0.0001);
        assert!((result.z - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_repeat_polar() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = modifier_repeat_polar(point, 6);
        assert!((result.y).abs() < 0.0001);
    }
}

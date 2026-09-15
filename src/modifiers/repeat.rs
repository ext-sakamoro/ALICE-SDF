//! Repetition modifiers for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Fast Modulo**: Replaced general modulo with `x - s * round(x/s)` logic,
//!   where `round` is [`round_half_up`] (`floor(x + 0.5)`) so that the tree
//!   evaluator ties the same way as SIMD / JIT / shaders at cell boundaries.
//! - **Forced Inlining**: `#[inline(always)]`.
//!
//! Author: Moroya Sakamoto

use crate::crispy::{round_half_up, round_half_up_vec3};
use glam::Vec3;

/// Infinite repetition along all axes (Deep Fried)
#[inline(always)]
pub fn modifier_repeat_infinite(point: Vec3, spacing: Vec3) -> Vec3 {
    // p - s * round(p / s) maps to [-s/2, s/2] range
    // Much faster than standard euclidean modulo logic
    Vec3::new(
        spacing
            .x
            .mul_add(-round_half_up(point.x / spacing.x), point.x),
        spacing
            .y
            .mul_add(-round_half_up(point.y / spacing.y), point.y),
        spacing
            .z
            .mul_add(-round_half_up(point.z / spacing.z), point.z),
    )
}

/// Infinite repetition — Division Exorcism edition.
///
/// Takes precomputed `recip_spacing = 1.0 / spacing` to eliminate 3 divisions
/// from the hot path. `p * recip` replaces `p / spacing`.
#[inline(always)]
pub fn modifier_repeat_infinite_rk(point: Vec3, spacing: Vec3, recip_spacing: Vec3) -> Vec3 {
    Vec3::new(
        spacing
            .x
            .mul_add(-round_half_up(point.x * recip_spacing.x), point.x),
        spacing
            .y
            .mul_add(-round_half_up(point.y * recip_spacing.y), point.y),
        spacing
            .z
            .mul_add(-round_half_up(point.z * recip_spacing.z), point.z),
    )
}

/// Finite repetition along all axes (Deep Fried)
#[inline(always)]
pub fn modifier_repeat_finite(point: Vec3, count: [u32; 3], spacing: Vec3) -> Vec3 {
    let limit = Vec3::new(count[0] as f32, count[1] as f32, count[2] as f32) * 0.5;
    // clamp(round(p/s), -limit, limit)
    let cell = round_half_up_vec3(point / spacing).clamp(-limit, limit);
    point - cell * spacing
}

/// Infinite repetition along a single axis (X)
#[inline(always)]
pub fn modifier_repeat_x(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        spacing.mul_add(-round_half_up(point.x / spacing), point.x),
        point.y,
        point.z,
    )
}

/// Infinite repetition along a single axis (Y)
#[inline(always)]
pub fn modifier_repeat_y(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        point.x,
        spacing.mul_add(-round_half_up(point.y / spacing), point.y),
        point.z,
    )
}

/// Infinite repetition along a single axis (Z)
#[inline(always)]
pub fn modifier_repeat_z(point: Vec3, spacing: f32) -> Vec3 {
    Vec3::new(
        point.x,
        point.y,
        spacing.mul_add(-round_half_up(point.z / spacing), point.z),
    )
}

/// Polar repetition around Y axis
#[inline(always)]
pub fn modifier_repeat_polar(point: Vec3, count: u32) -> Vec3 {
    // One law: same operands as the compiled PolarRepeat instruction.
    super::polar_repeat::modifier_polar_repeat(point, count)
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

//! Bend modifier for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Branchless**: Removed `if curvature < epsilon` check.
//!   Pipeline consistency > saving a few FLOPs on identity transform.
//! - **Simultaneous Trig**: Uses `sin_cos()`.
//! - **Forced Inlining**: `#[inline(always)]`.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Bend space around the Z-axis (bending in the XY plane)
#[inline(always)]
pub fn modifier_bend(point: Vec3, curvature: f32) -> Vec3 {
    // k = 0 case is handled naturally by cos(0)=1, sin(0)=0
    // k * x -> 0, so c=1, s=0 -> x'=x, y'=y
    let (s, c) = (curvature * point.x).sin_cos();
    Vec3::new(
        c * point.x - s * point.y,
        s * point.x + c * point.y,
        point.z,
    )
}

/// Bend space around the Y-axis (bending in the XZ plane)
#[inline(always)]
pub fn modifier_bend_x(point: Vec3, curvature: f32) -> Vec3 {
    let (s, c) = (curvature * point.y).sin_cos();
    Vec3::new(
        c * point.x - s * point.z,
        point.y,
        s * point.x + c * point.z,
    )
}

/// Bend space around the X-axis (bending in the YZ plane)
#[inline(always)]
pub fn modifier_bend_z(point: Vec3, curvature: f32) -> Vec3 {
    let (s, c) = (curvature * point.y).sin_cos();
    Vec3::new(
        point.x,
        c * point.y - s * point.z,
        s * point.y + c * point.z,
    )
}

/// Cheap bend approximation using polynomial (Optimized)
#[inline(always)]
pub fn modifier_bend_cheap(point: Vec3, curvature: f32) -> Vec3 {
    let bend = curvature * point.y * point.y;
    Vec3::new(point.x + bend, point.y, point.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bend_zero() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let r = modifier_bend(p, 0.0);
        assert!((r - p).length() < 1e-6);
    }

    #[test]
    fn test_bend_at_origin() {
        let point = Vec3::new(0.0, 1.0, 0.0);
        let result = modifier_bend(point, 1.0);
        assert!((result.x - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_bend_cheap() {
        let point = Vec3::new(0.0, 2.0, 0.0);
        let result = modifier_bend_cheap(point, 0.5);
        assert!((result.x - 2.0).abs() < 0.0001);
        assert!((result.y - 2.0).abs() < 0.0001);
    }
}

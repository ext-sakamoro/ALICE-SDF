//! Rotation transform for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//!
//! - **Conjugate vs Inverse**: For unit quaternions |q|=1, inverse equals
//!   conjugate (just sign flip, no division by norm²). Saves ~10 cycles.
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};

/// Transform a point by inverse rotation for SDF evaluation (Deep Fried)
///
/// To evaluate an SDF that has been rotated by `rotation`,
/// we need to rotate the query point by the inverse rotation.
///
/// # Deep Fried Optimization
///
/// For unit quaternions (|q| = 1), the inverse equals the conjugate:
/// ```text
/// q⁻¹ = q̄ / |q|² = q̄ / 1 = q̄
/// ```
/// `conjugate()` just flips signs (x,y,z → -x,-y,-z), no division needed.
///
/// # Arguments
/// * `point` - Query point
/// * `rotation` - Rotation quaternion (must be normalized/unit quaternion)
///
/// # Returns
/// Transformed point for child SDF evaluation
#[inline(always)]
pub fn transform_rotate(point: Vec3, rotation: Quat) -> Vec3 {
    // Deep Fried: conjugate() instead of inverse()
    // Unit quaternions: |q|=1, so q⁻¹ = q̄ (no norm calculation)
    rotation.conjugate() * point
}

/// Transform a point by rotation (forward transform)
#[inline(always)]
pub fn transform_rotate_inverse(point: Vec3, rotation: Quat) -> Vec3 {
    rotation * point
}

/// Transform a point by inverse Euler rotation (Deep Fried)
///
/// # Arguments
/// * `point` - Query point
/// * `x` - Rotation around X axis (radians)
/// * `y` - Rotation around Y axis (radians)
/// * `z` - Rotation around Z axis (radians)
#[inline(always)]
pub fn transform_rotate_euler(point: Vec3, x: f32, y: f32, z: f32) -> Vec3 {
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, x, y, z);
    transform_rotate(point, rotation)
}

/// Create a rotation quaternion around an arbitrary axis
#[allow(dead_code)]
#[inline(always)]
pub fn rotation_axis_angle(axis: Vec3, angle: f32) -> Quat {
    Quat::from_axis_angle(axis.normalize(), angle)
}

/// Create a rotation quaternion from a "look at" direction
#[allow(dead_code)]
#[inline(always)]
pub fn rotation_look_at(direction: Vec3, up: Vec3) -> Quat {
    let forward = direction.normalize();
    let right = up.cross(forward).normalize();
    let up = forward.cross(right);
    Quat::from_mat3(&glam::Mat3::from_cols(right, up, forward))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_rotate_identity() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::IDENTITY;
        let result = transform_rotate(point, rotation);
        assert!((result - point).length() < 0.0001);
    }

    #[test]
    fn test_rotate_90_y() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let rotation = Quat::from_rotation_y(PI / 2.0);
        let result = transform_rotate(point, rotation);
        // Inverse of CCW 90° Y rotation is CW 90°, which moves X to +Z
        let expected = Vec3::new(0.0, 0.0, 1.0);
        assert!((result - expected).length() < 0.0001);
    }

    #[test]
    fn test_rotate_inverse_roundtrip() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.5, 0.3, 0.7);

        let transformed = transform_rotate(point, rotation);
        let restored = transform_rotate_inverse(transformed, rotation);

        assert!((restored - point).length() < 0.0001);
    }

    #[test]
    fn test_rotate_euler() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = transform_rotate_euler(point, 0.0, PI / 2.0, 0.0);
        // Inverse of CCW 90° Y rotation is CW 90°, which moves X to +Z
        let expected = Vec3::new(0.0, 0.0, 1.0);
        assert!((result - expected).length() < 0.0001);
    }

    #[test]
    fn test_axis_angle() {
        let rotation = rotation_axis_angle(Vec3::Y, PI);
        let point = Vec3::new(1.0, 0.0, 0.0);
        let result = rotation * point;
        let expected = Vec3::new(-1.0, 0.0, 0.0);
        assert!((result - expected).length() < 0.0001);
    }
}

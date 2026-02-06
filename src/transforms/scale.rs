//! Scale transform for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//!
//! - **SIMD Vector Division**: `point / factors` uses vdivps instruction
//!   instead of 3 separate scalar divisions
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Transform a point by inverse uniform scale for SDF evaluation
///
/// For uniform scaling, we divide the point by the scale factor,
/// then multiply the resulting distance by the scale factor.
///
/// # Arguments
/// * `point` - Query point
/// * `factor` - Scale factor
///
/// # Returns
/// (transformed_point, distance_multiplier)
#[inline(always)]
pub fn transform_scale(point: Vec3, factor: f32) -> (Vec3, f32) {
    (point / factor, factor)
}

/// Transform a point by scale (forward transform)
#[inline(always)]
pub fn transform_scale_inverse(point: Vec3, factor: f32) -> Vec3 {
    point * factor
}

/// Transform a point by inverse non-uniform scale (Deep Fried)
///
/// Non-uniform scaling distorts the SDF and requires distance correction.
/// The distance should be multiplied by the minimum scale factor as an
/// approximation (this is not exact but works for raymarching).
///
/// # Deep Fried Optimization
///
/// Uses vector division `point / factors` which compiles to a single
/// vdivps SIMD instruction instead of 3 scalar divisions:
/// ```text
/// Before: point.x / factors.x, point.y / factors.y, point.z / factors.z
/// After:  point / factors  →  vdivps xmm0, xmm1
/// ```
///
/// # Arguments
/// * `point` - Query point
/// * `factors` - Scale factors for each axis (x, y, z)
///
/// # Returns
/// (transformed_point, approximate_distance_multiplier)
#[inline(always)]
pub fn transform_scale_nonuniform(point: Vec3, factors: Vec3) -> (Vec3, f32) {
    // Deep Fried: vector division instead of component-wise scalar division
    // Compiles to single vdivps SIMD instruction
    let transformed = point / factors;
    let min_factor = factors.x.min(factors.y.min(factors.z));
    (transformed, min_factor)
}

/// Apply distance correction after non-uniform scale evaluation
///
/// This provides a more accurate distance estimate for non-uniform scaling
/// by considering the gradient direction.
#[allow(dead_code)]
#[inline(always)]
pub fn correct_distance_nonuniform(distance: f32, factors: Vec3, gradient: Vec3) -> f32 {
    // Deep Fried: vector multiplication instead of component-wise scalar
    let scaled_gradient = gradient * factors;
    distance * scaled_gradient.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_uniform() {
        let point = Vec3::new(2.0, 4.0, 6.0);
        let (transformed, mult) = transform_scale(point, 2.0);
        assert!((transformed - Vec3::new(1.0, 2.0, 3.0)).length() < 0.0001);
        assert!((mult - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_scale_inverse() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let factor = 2.0;

        let (transformed, _) = transform_scale(point, factor);
        let restored = transform_scale_inverse(transformed, factor);

        assert!((restored - point).length() < 0.0001);
    }

    #[test]
    fn test_scale_identity() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let (transformed, mult) = transform_scale(point, 1.0);
        assert!((transformed - point).length() < 0.0001);
        assert!((mult - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_scale_nonuniform() {
        let point = Vec3::new(2.0, 4.0, 6.0);
        let factors = Vec3::new(2.0, 2.0, 3.0);
        let (transformed, mult) = transform_scale_nonuniform(point, factors);
        assert!((transformed - Vec3::new(1.0, 2.0, 2.0)).length() < 0.0001);
        assert!((mult - 2.0).abs() < 0.0001); // Min of 2, 2, 3
    }

    #[test]
    fn test_distance_correction() {
        let distance = 1.0;
        let factors = Vec3::new(1.0, 2.0, 1.0);
        let gradient = Vec3::new(0.0, 1.0, 0.0); // Pointing in Y direction

        let corrected = correct_distance_nonuniform(distance, factors, gradient);
        // Gradient scaled by Y factor (2.0), length is 2.0
        assert!((corrected - 2.0).abs() < 0.0001);
    }
}

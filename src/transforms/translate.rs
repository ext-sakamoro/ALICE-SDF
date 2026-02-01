//! Translation transform for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//!
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead
//!   Translation is a single SIMD subtract, call overhead would dominate.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Transform a point by inverse translation for SDF evaluation (Deep Fried)
///
/// To evaluate an SDF that has been translated by `offset`,
/// we need to transform the query point by `-offset`.
///
/// # Deep Fried Optimization
///
/// Translation is just `vsubps xmm0, xmm1` (3 cycles). Without forced
/// inlining, function call overhead (~15 cycles) would be 5x the actual work.
///
/// # Arguments
/// * `point` - Query point
/// * `offset` - Translation offset
///
/// # Returns
/// Transformed point for child SDF evaluation
#[inline(always)]
pub fn transform_translate(point: Vec3, offset: Vec3) -> Vec3 {
    point - offset
}

/// Transform a point by translation (forward transform)
///
/// Useful for transforming results back to world space.
#[inline(always)]
pub fn transform_translate_inverse(point: Vec3, offset: Vec3) -> Vec3 {
    point + offset
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_origin() {
        let offset = Vec3::new(1.0, 2.0, 3.0);
        let result = transform_translate(offset, offset);
        assert!((result - Vec3::ZERO).length() < 0.0001);
    }

    #[test]
    fn test_translate_inverse() {
        let point = Vec3::new(1.0, 2.0, 3.0);
        let offset = Vec3::new(0.5, 0.5, 0.5);

        let transformed = transform_translate(point, offset);
        let restored = transform_translate_inverse(transformed, offset);

        assert!((restored - point).length() < 0.0001);
    }
}

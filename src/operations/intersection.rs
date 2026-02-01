//! Intersection operation for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

/// Intersection of two SDFs (maximum distance)
///
/// # Returns
/// Maximum of the two distances
#[inline(always)]
pub fn sdf_intersection(d1: f32, d2: f32) -> f32 {
    d1.max(d2)
}

/// Intersection of multiple SDFs
#[inline(always)]
pub fn sdf_intersection_multi(distances: &[f32]) -> f32 {
    distances
        .iter()
        .copied()
        .reduce(|a, b| a.max(b))
        .unwrap_or(f32::MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        assert_eq!(sdf_intersection(1.0, 2.0), 2.0);
    }

    #[test]
    fn test_intersection_multi() {
        let distances = vec![-0.5, -0.3, -0.8, -0.2];
        assert!((sdf_intersection_multi(&distances) + 0.2).abs() < 0.0001);
    }

    #[test]
    fn test_intersection_multi_empty() {
        assert_eq!(sdf_intersection_multi(&[]), f32::MIN);
    }
}

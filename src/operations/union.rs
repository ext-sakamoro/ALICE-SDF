//! Union operation for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

/// Union of two SDFs (minimum distance)
///
/// # Returns
/// Minimum of the two distances
#[inline(always)]
pub fn sdf_union(d1: f32, d2: f32) -> f32 {
    d1.min(d2)
}

/// Union of multiple SDFs
#[inline(always)]
pub fn sdf_union_multi(distances: &[f32]) -> f32 {
    // reduce() produces optimal code for slice iteration
    distances
        .iter()
        .copied()
        .reduce(|a, b| a.min(b))
        .unwrap_or(f32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_union() {
        assert_eq!(sdf_union(1.0, 2.0), 1.0);
    }

    #[test]
    fn test_union_multi() {
        let distances = vec![3.0, 1.0, 2.0, 5.0];
        assert!((sdf_union_multi(&distances) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_union_multi_empty() {
        assert_eq!(sdf_union_multi(&[]), f32::MAX);
    }
}

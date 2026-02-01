//! Subtraction operation for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

/// Subtraction of B from A (A minus B)
///
/// # Returns
/// max(d1, -d2)
#[inline(always)]
pub fn sdf_subtraction(d1: f32, d2: f32) -> f32 {
    d1.max(-d2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subtraction() {
        assert_eq!(sdf_subtraction(1.0, 2.0), 1.0); // max(1, -2)
        assert_eq!(sdf_subtraction(1.0, -2.0), 2.0); // max(1, 2)
    }
}

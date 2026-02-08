//! XOR operation for SDFs (Deep Fried Edition)
//!
//! Produces the symmetric difference of two shapes:
//! the region inside exactly one of the two shapes.
//!
//! Author: Moroya Sakamoto

/// XOR of two SDFs (symmetric difference)
///
/// Returns the region that is inside exactly one shape but not both.
/// Formula: max(min(a, b), -max(a, b))
#[inline(always)]
pub fn sdf_xor(a: f32, b: f32) -> f32 {
    a.min(b).max(-a.max(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_outside_both() {
        // Both positive (outside both) => should be positive
        let d = sdf_xor(1.0, 2.0);
        assert!(d > 0.0, "Outside both should be positive, got {}", d);
    }

    #[test]
    fn test_xor_inside_one() {
        // Inside a but outside b => should be negative
        let d = sdf_xor(-1.0, 2.0);
        assert!(d < 0.0, "Inside one should be negative, got {}", d);
    }

    #[test]
    fn test_xor_inside_both() {
        // Inside both => should be positive (XOR excludes overlap)
        let d = sdf_xor(-1.0, -2.0);
        assert!(d > 0.0, "Inside both should be positive, got {}", d);
    }

    #[test]
    fn test_xor_symmetry() {
        let d1 = sdf_xor(0.5, -0.3);
        let d2 = sdf_xor(-0.3, 0.5);
        assert!((d1 - d2).abs() < 1e-6, "XOR should be symmetric");
    }
}

//! Engrave operation for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpEngrave. Engraves shape b into shape a.
//!
//! Author: Moroya Sakamoto

/// Engrave operation on two SDFs
///
/// Engraves shape b into the surface of shape a.
/// Formula: `max(a, (a + r - abs(b)) * SQRT_0_5)`
/// - `r`: engrave depth
#[inline(always)]
pub fn sdf_engrave(a: f32, b: f32, r: f32) -> f32 {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    a.max((a + r - b.abs()) * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engrave_far_outside() {
        // Far outside a: engrave has no effect
        let d = sdf_engrave(5.0, 0.5, 0.1);
        assert!(
            (d - 5.0).abs() < 0.01,
            "Far outside should be ~a, got {}",
            d
        );
    }

    #[test]
    fn test_engrave_finite() {
        let d = sdf_engrave(0.1, -0.2, 0.3);
        assert!(d.is_finite());
    }

    #[test]
    fn test_engrave_preserves_outside() {
        // When a is large positive, result should be a
        let d = sdf_engrave(10.0, 0.0, 0.1);
        assert!((d - 10.0).abs() < 0.01);
    }
}

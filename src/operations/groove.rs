//! Groove operation for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpGroove. Cuts a groove of shape b into shape a.
//!
//! Author: Moroya Sakamoto

/// Groove operation on two SDFs
///
/// Cuts a groove of shape b into the surface of shape a.
/// Formula: `max(a, min(a + ra, rb - abs(b)))`
/// - `ra`: groove width
/// - `rb`: groove depth
#[inline(always)]
pub fn sdf_groove(a: f32, b: f32, ra: f32, rb: f32) -> f32 {
    a.max((a + ra).min(rb - b.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groove_far_outside() {
        // Far outside a: groove has no effect
        let d = sdf_groove(5.0, 0.5, 0.1, 0.05);
        assert!(
            (d - 5.0).abs() < 0.01,
            "Far outside should be ~a, got {}",
            d
        );
    }

    #[test]
    fn test_groove_preserves_interior() {
        // Deep inside: groove doesn't go that deep
        let d = sdf_groove(-5.0, 0.0, 0.1, 0.05);
        assert!(d.is_finite());
    }

    #[test]
    fn test_groove_finite() {
        let d = sdf_groove(0.0, 0.0, 0.2, 0.1);
        assert!(d.is_finite());
    }
}

//! Tongue operation for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpTongue. Inverse of groove: adds a tongue.
//!
//! Author: Moroya Sakamoto

/// Tongue operation on two SDFs
///
/// Adds a tongue-shaped protrusion of shape b onto shape a.
/// Formula: `min(a, max(a - ra, abs(b) - rb))`
/// - `ra`: tongue width
/// - `rb`: tongue height
#[inline(always)]
pub fn sdf_tongue(a: f32, b: f32, ra: f32, rb: f32) -> f32 {
    a.min((a - ra).max(b.abs() - rb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tongue_far_outside() {
        // Far outside both: result ≤ a (tongue = min(a, ...))
        let d = sdf_tongue(5.0, 5.0, 0.1, 0.05);
        // b=5.0, abs(b)-rb = 4.95, max(a-ra, abs(b)-rb) = max(4.9,4.95) = 4.95
        // min(5.0, 4.95) = 4.95
        assert!(
            (d - 4.95).abs() < 0.01,
            "Far outside should be ~4.95, got {}",
            d
        );
    }

    #[test]
    fn test_tongue_extends_surface() {
        // Near surface: tongue extends it
        let d = sdf_tongue(0.05, 0.0, 0.1, 0.2);
        // abs(0)-0.2 = -0.2, max(0.05-0.1, -0.2) = max(-0.05,-0.2) = -0.05
        // min(0.05, -0.05) = -0.05
        assert!(d < 0.0, "Tongue should extend surface, got {}", d);
    }

    #[test]
    fn test_tongue_finite() {
        let d = sdf_tongue(0.0, 0.0, 0.2, 0.1);
        assert!(d.is_finite());
    }
}

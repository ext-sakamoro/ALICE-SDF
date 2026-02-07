//! Taper modifier — shrink cross-section along Y-axis
//!
//! Ported from Baker's `opTaper(p, factor)`.
//! `factor > 0` = narrower at top (positive Y).
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Taper the evaluation point: shrink XZ cross-section based on Y position.
///
/// `factor > 0` narrows the shape at positive Y.
/// Returns the transformed point (pass to child SDF).
///
/// Formula: `s = 1 / (1 - p.y * factor); return (p.x * s, p.y, p.z * s)`
#[inline(always)]
pub fn modifier_taper(p: Vec3, factor: f32) -> Vec3 {
    let s = 1.0 / (1.0 - p.y * factor);
    Vec3::new(p.x * s, p.y, p.z * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_taper_at_origin() {
        // At y=0, taper should be identity
        let p = Vec3::new(1.0, 0.0, 2.0);
        let q = modifier_taper(p, 0.5);
        assert!((q.x - 1.0).abs() < 0.001);
        assert!((q.y - 0.0).abs() < 0.001);
        assert!((q.z - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_taper_positive_y() {
        // At y>0 with positive factor, XZ should be expanded (inverse scale)
        let p = Vec3::new(1.0, 1.0, 1.0);
        let q = modifier_taper(p, 0.5);
        // s = 1/(1-0.5) = 2.0
        assert!((q.x - 2.0).abs() < 0.001);
        assert!((q.y - 1.0).abs() < 0.001);
        assert!((q.z - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_taper_zero_factor() {
        // Factor=0 should be identity
        let p = Vec3::new(3.0, 5.0, 7.0);
        let q = modifier_taper(p, 0.0);
        assert!((q - p).length() < 0.001);
    }
}

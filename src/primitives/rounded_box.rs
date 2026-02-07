//! Rounded Box SDF (Deep Fried Edition)
//!
//! Box with rounded edges. More accurate than Box + Round modifier.
//!
//! Based on Inigo Quilez's sdRoundBox formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a rounded box centered at origin
///
/// - `half_extents`: half-size along each axis (before rounding)
/// - `round_radius`: edge rounding radius
#[inline(always)]
pub fn sdf_rounded_box(p: Vec3, half_extents: Vec3, round_radius: f32) -> f32 {
    let q = p.abs() - half_extents;
    q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0) - round_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_box_origin() {
        let d = sdf_rounded_box(Vec3::ZERO, Vec3::splat(1.0), 0.1);
        assert!(d < -0.5, "Origin should be deep inside, got {}", d);
    }

    #[test]
    fn test_rounded_box_on_face() {
        // On face center (1.0, 0, 0) with half_extents=1.0, round=0.1
        // Should be approximately -round_radius
        let d = sdf_rounded_box(Vec3::new(1.0, 0.0, 0.0), Vec3::splat(1.0), 0.1);
        assert!((d + 0.1).abs() < 0.01, "On face should be ~-0.1, got {}", d);
    }

    #[test]
    fn test_rounded_box_outside() {
        let d = sdf_rounded_box(Vec3::new(5.0, 0.0, 0.0), Vec3::splat(1.0), 0.1);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_rounded_box_zero_round() {
        // With zero rounding, should match regular box
        let d = sdf_rounded_box(Vec3::new(2.0, 0.0, 0.0), Vec3::splat(1.0), 0.0);
        assert!((d - 1.0).abs() < 0.001, "Should match box, got {}", d);
    }
}

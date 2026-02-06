//! Revolution modifier (Deep Fried Edition)
//!
//! Revolves a shape around the Y-axis to create rotational symmetry.
//! The child SDF is evaluated in a 2D cross-section plane.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Revolution around Y-axis
///
/// Maps the point to (length(p.xz) - offset, p.y, 0) before
/// evaluating the child SDF. This creates a shape of revolution.
///
/// # Arguments
/// * `p` - Point to transform
/// * `offset` - Distance from Y-axis to the center of the revolved shape
///
/// # Examples
/// - Revolution of a circle (sphere) with offset = ring radius → Torus
/// - Revolution of a square (box) → Ring with square cross-section
#[inline(always)]
pub fn modifier_revolution(p: Vec3, offset: f32) -> Vec3 {
    let q = (p.x * p.x + p.z * p.z).sqrt() - offset;
    Vec3::new(q, p.y, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_revolution_on_axis() {
        // Point on Y-axis, offset=2: should map to (-2, y, 0)
        let p = Vec3::new(0.0, 1.0, 0.0);
        let r = modifier_revolution(p, 2.0);
        assert!((r.x - (-2.0)).abs() < 0.001);
        assert!((r.y - 1.0).abs() < 0.001);
        assert_eq!(r.z, 0.0);
    }

    #[test]
    fn test_revolution_at_offset() {
        // Point at distance=offset from Y-axis: should map to (0, y, 0)
        let p = Vec3::new(2.0, 1.0, 0.0);
        let r = modifier_revolution(p, 2.0);
        assert!(r.x.abs() < 0.001);
        assert!((r.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_revolution_rotational_symmetry() {
        // Points at same distance from Y-axis should map to same result
        let p1 = Vec3::new(3.0, 0.0, 0.0);
        let p2 = Vec3::new(0.0, 0.0, 3.0);
        let p3 = Vec3::new(2.121, 0.0, 2.121); // ~3.0 distance
        let offset = 1.0;

        let r1 = modifier_revolution(p1, offset);
        let r2 = modifier_revolution(p2, offset);
        let r3 = modifier_revolution(p3, offset);

        assert!((r1.x - r2.x).abs() < 0.001);
        assert!((r1.x - r3.x).abs() < 0.01);
    }
}

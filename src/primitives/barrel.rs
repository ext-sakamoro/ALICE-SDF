//! Barrel SDF (Deep Fried Edition)
//!
//! Cylinder with a parabolic radial bulge along Y-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Approximate SDF for a barrel shape along Y-axis
///
/// - `radius`: base radius (at top and bottom caps)
/// - `half_height`: half the barrel height
/// - `bulge`: additional radius at the middle (y=0)
#[inline(always)]
pub fn sdf_barrel(p: Vec3, radius: f32, half_height: f32, bulge: f32) -> f32 {
    let r = Vec2::new(p.x, p.z).length();
    let y_norm = (p.y / half_height).clamp(-1.0, 1.0);
    let effective_r = radius + bulge * (1.0 - y_norm * y_norm);
    let d_r = r - effective_r;
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_r.max(0.0), d_y.max(0.0));
    d_r.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_barrel_center_inside() {
        let d = sdf_barrel(Vec3::ZERO, 1.0, 1.0, 0.2);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_barrel_far_outside() {
        let d = sdf_barrel(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0, 0.2);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_barrel_bulge_wider_at_middle() {
        let d_middle = sdf_barrel(Vec3::new(1.1, 0.0, 0.0), 1.0, 1.0, 0.2);
        let d_top = sdf_barrel(Vec3::new(1.1, 1.0, 0.0), 1.0, 1.0, 0.2);
        assert!(d_middle < d_top, "Middle should be wider than top");
    }

    #[test]
    fn test_barrel_symmetry() {
        let d1 = sdf_barrel(Vec3::new(0.5, 0.3, 0.2), 1.0, 1.0, 0.2);
        let d2 = sdf_barrel(Vec3::new(-0.5, -0.3, -0.2), 1.0, 1.0, 0.2);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric");
    }
}

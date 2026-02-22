//! Egg SDF (Deep Fried Edition)
//!
//! 3D egg shape (revolution body) along Y-axis.
//!
//! Based on Inigo Quilez's Egg SDF formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for an egg shape
///
/// Revolution body around Y-axis.
/// - `ra`: overall size / base radius
/// - `rb`: top deformation (controls pointiness)
#[inline(always)]
pub fn sdf_egg(p: Vec3, ra: f32, rb: f32) -> f32 {
    // IQ egg formula (revolution body)
    let px = Vec2::new(p.x, p.z).length();
    let py = p.y;
    let r = ra - rb;

    if py < 0.0 {
        Vec2::new(px, py).length() - r
    } else if px * ra < py * rb {
        // Near the top (narrow part)
        Vec2::new(px, py - ra).length()
    } else {
        Vec2::new(px + rb, py).length() - ra
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_egg_center_inside() {
        let d = sdf_egg(Vec3::ZERO, 1.0, 0.2);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_egg_far_outside() {
        let d = sdf_egg(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.2);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_egg_symmetry_xz() {
        let d1 = sdf_egg(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.2);
        let d2 = sdf_egg(Vec3::new(-0.3, 0.1, -0.2), 1.0, 0.2);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in XZ");
    }

    #[test]
    fn test_egg_revolution_symmetry() {
        let d1 = sdf_egg(Vec3::new(0.3, 0.1, 0.0), 1.0, 0.2);
        let d2 = sdf_egg(Vec3::new(0.0, 0.1, 0.3), 1.0, 0.2);
        assert!((d1 - d2).abs() < 0.001, "Should have revolution symmetry");
    }
}

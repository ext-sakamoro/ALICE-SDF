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
/// Revolution body around Y-axis — Inigo Quilez's exact `sdEgg` in the
/// (|xz|, y) half-plane: a disc of radius `ra` below y = 0, two arcs of
/// radius `2·(ra − rb)` centred at (∓(ra − rb), 0) meeting a cap of radius
/// `rb` at the apex `y = √3·(ra − rb) + ra`. Every branch is a distance to a
/// circular arc, so `|∇f| = 1` everywhere (exact SDF, Lipschitz 1).
/// - `ra`: base (bottom) radius
/// - `rb`: apex (top) radius, `0 < rb < ra`
///
/// Until 1.10.3 this was a three-branch approximation that reported positive
/// distances for interior points on the axis (`egg(1, 0.5)` at (0, 0.1, 0)
/// = +0.9) and jumped by `ra − rb` across the origin.
#[inline(always)]
pub fn sdf_egg(p: Vec3, ra: f32, rb: f32) -> f32 {
    const K: f32 = 1.732_050_8; // √3
    let px = Vec2::new(p.x, p.z).length();
    let py = p.y;
    let r = ra - rb;
    let d = if py < 0.0 {
        Vec2::new(px, py).length() - r
    } else if K * (px + r) < py {
        Vec2::new(px, K * -r + py).length() - r
    } else {
        2.0f32 * -r + Vec2::new(px + r, py).length()
    };
    d - rb
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
    fn test_egg_axis_interior_and_apex() {
        // interior points on the axis are negative (the old law returned +0.9 here)
        assert!(sdf_egg(Vec3::new(0.0, 0.1, 0.0), 1.0, 0.5) < 0.0);
        // apex is on the surface: y = √3·(ra − rb) + ra
        let apex = 1.732_050_8f32.mul_add(0.5, 1.0);
        assert!(sdf_egg(Vec3::new(0.0, apex, 0.0), 1.0, 0.5).abs() < 1e-5);
        // base circle radius ra
        assert!(sdf_egg(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.5).abs() < 1e-5);
        assert!(sdf_egg(Vec3::new(0.0, -1.0, 0.0), 1.0, 0.5).abs() < 1e-5);
        // continuity across the origin (the old law jumped by ra - rb)
        let a = sdf_egg(Vec3::new(0.0, 1e-4, 0.0), 1.0, 0.5);
        let b = sdf_egg(Vec3::new(0.0, -1e-4, 0.0), 1.0, 0.5);
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
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

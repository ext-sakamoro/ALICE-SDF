//! RoundedCone SDF (Deep Fried Edition)
//!
//! Exact SDF for a rounded cone along Y-axis.
//! Bottom at y = -half_height with radius r1,
//! Top at y = half_height with radius r2.
//! Smoothly blended (rounded) edges at both caps.
//!
//! Based on Inigo Quilez's sdRoundCone formula.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Rounded cone: spheres of `r1` / `r2` at `y = ∓half_height` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_rounded_cone_r<R: Real>(p: Vec3R<R>, r1: f32, r2: f32, half_height: f32) -> R {
    let h = half_height * 2.0;
    let q_x = (p.x * p.x + p.z * p.z).sqrt();
    let q_y = p.y + R::splat(half_height);
    let b = (r1 - r2) / h;
    let a = (1.0 - b * b).sqrt();
    let (ra, rb, rh) = (R::splat(a), R::splat(b), R::splat(h));
    let k = q_y * ra - q_x * rb;
    let d_bottom = (q_x * q_x + q_y * q_y).sqrt() - R::splat(r1);
    let dy = q_y - rh;
    let d_top = (q_x * q_x + dy * dy).sqrt() - R::splat(r2);
    let d_mantle = q_x * ra + q_y * rb - R::splat(r1);
    R::select(
        k.lt(R::zero()),
        d_bottom,
        R::select(k.gt(ra * rh), d_top, d_mantle),
    )
}

/// Rounded cone: spheres of `r1` / `r2` at `y = ∓half_height`.
#[inline(always)]
pub fn sdf_rounded_cone(p: Vec3, r1: f32, r2: f32, half_height: f32) -> f32 {
    sdf_rounded_cone_r::<f32>(p.into(), r1, r2, half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_cone_origin_inside() {
        let d = sdf_rounded_cone(Vec3::ZERO, 1.0, 0.5, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_rounded_cone_bottom() {
        // Bottom sphere center is at y=-half_height, radius=r1
        // Surface point at bottom pole: y = -half_height - r1 = -2.0
        let d = sdf_rounded_cone(Vec3::new(0.0, -2.0, 0.0), 1.0, 0.5, 1.0);
        assert!(
            d.abs() < 0.001,
            "Bottom pole should be on surface, got {}",
            d
        );
    }

    #[test]
    fn test_rounded_cone_top() {
        // Top sphere center is at y=half_height, radius=r2
        // Surface point at top pole: y = half_height + r2 = 1.5
        let d = sdf_rounded_cone(Vec3::new(0.0, 1.5, 0.0), 1.0, 0.5, 1.0);
        assert!(d.abs() < 0.001, "Top pole should be on surface, got {}", d);
    }

    #[test]
    fn test_rounded_cone_outside() {
        let d = sdf_rounded_cone(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

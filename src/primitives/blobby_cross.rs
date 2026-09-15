//! Blobby cross SDF (Deep Fried Edition)
//!
//! 2D blobby/organic cross shape in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdBlobbyCross formula using sqrt-based blending.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Blobbiness of the cross arms (Inigo Quilez's `he`): the arms are
/// parabola segments `y = he·(1 − 2x²)` in the folded octant; 0.5 gives the
/// canonical blobby cross whose arms reach the unit square's mid-edges.
pub const BLOBBY_CROSS_HE: f32 = 0.5;

/// SDF for a blobby (organic) cross shape, extruded along Y-axis
///
/// Inigo Quilez's exact `sdBlobbyCross` (distance to the parabola-arm
/// curve, obtained by solving the depressed cubic for the nearest
/// parameter) evaluated on `|xz| / size` and scaled back by `size`, then
/// extruded along Y. Exact SDF, Lipschitz 1.
///
/// - `size`: overall size of the blobby cross (arms reach `±size`)
/// - `half_height`: half the extrusion height along Y
///
/// Until 1.10.3 this was a home-grown "sqrt blend" that jumped by up to
/// 23× the sample spacing between its two regions (Lipschitz property test).
#[inline(always)]
pub fn sdf_blobby_cross(p: Vec3, size: f32, half_height: f32) -> f32 {
    let d_2d = blobby_cross_2d(Vec2::new(p.x, p.z) / size, BLOBBY_CROSS_HE) * size;
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

/// IQ `sdBlobbyCross` in 2D (shared by the interval / gradient paths).
#[inline(always)]
pub fn blobby_cross_2d(pos: Vec2, he: f32) -> f32 {
    let pos = pos.abs();
    let pos =
        Vec2::new((pos.x - pos.y).abs(), 1.0 - pos.x - pos.y) * std::f32::consts::FRAC_1_SQRT_2;
    let p = (he - pos.y - 0.25 / he) / (6.0 * he);
    let q = pos.x / (he * he * 16.0);
    let h = q.mul_add(q, -(p * p * p));
    let x = if h >= 0.0 {
        // one real root (`h = 0`: double root, avoids the 0/0 of the trig form)
        let r = h.sqrt();
        (q + r).powf(1.0 / 3.0) - (q - r).abs().powf(1.0 / 3.0) * (r - q).signum()
    } else {
        let r = p.sqrt();
        2.0 * r * ((q / (p * r)).acos() / 3.0).cos()
    };
    let x = x.min(std::f32::consts::FRAC_1_SQRT_2);
    let z = Vec2::new(x, he * (1.0 - 2.0 * x * x)) - pos;
    z.length() * z.y.signum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blobby_cross_center_inside() {
        let d = sdf_blobby_cross(Vec3::ZERO, 1.0, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_blobby_cross_arm_tips_and_eikonal() {
        // arm tips on the axes are on the surface (size 1)
        for p in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
            let d = sdf_blobby_cross(p, 1.0, 1.0);
            assert!(d.abs() < 1e-4, "{p:?}: {d}");
        }
        // exact SDF: central differences have unit length away from the axes / kinks
        let h = 1e-3;
        for (x, z) in [(0.3, 1.2), (1.4, 0.2), (0.9, 0.7), (0.2, 0.25), (1.5, 1.3)] {
            let p = Vec3::new(x, 0.0, z);
            let gx = sdf_blobby_cross(p + Vec3::X * h, 1.0, 5.0)
                - sdf_blobby_cross(p - Vec3::X * h, 1.0, 5.0);
            let gz = sdf_blobby_cross(p + Vec3::Z * h, 1.0, 5.0)
                - sdf_blobby_cross(p - Vec3::Z * h, 1.0, 5.0);
            let g = (gx * gx + gz * gz).sqrt() / (2.0 * h);
            assert!((g - 1.0).abs() < 2e-2, "|grad| = {g} at {p:?}");
        }
    }

    #[test]
    fn test_blobby_cross_far_outside() {
        let d = sdf_blobby_cross(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_blobby_cross_symmetry_x() {
        let d1 = sdf_blobby_cross(Vec3::new(0.3, 0.1, 0.1), 1.0, 0.5);
        let d2 = sdf_blobby_cross(Vec3::new(-0.3, 0.1, 0.1), 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_blobby_cross_symmetry_z() {
        let d1 = sdf_blobby_cross(Vec3::new(0.1, 0.1, 0.3), 1.0, 0.5);
        let d2 = sdf_blobby_cross(Vec3::new(0.1, 0.1, -0.3), 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

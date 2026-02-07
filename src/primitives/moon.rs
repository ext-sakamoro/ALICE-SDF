//! Moon (crescent) SDF (Deep Fried Edition)
//!
//! 2D crescent/moon shape in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdMoon formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a crescent moon shape, extruded along Y-axis
///
/// - `d`: distance between the two circle centers
/// - `ra`: outer circle radius
/// - `rb`: inner (subtracted) circle radius
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_moon(p: Vec3, d: f32, ra: f32, rb: f32, half_height: f32) -> f32 {
    // 2D moon in XZ plane (IQ's sdMoon)
    let qx = p.x.abs();
    let qz = p.z;
    let q = Vec2::new(qx, qz);

    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b_sq = ra * ra - a * a;
    let b = if b_sq > 0.0 { b_sq.sqrt() } else { 0.0 };

    let d_2d = if d * (qx * b - qz * a) > d * d * 0.0_f32.max(qx * b - qz * a).min(1.0).min(0.0) {
        // Simplified: check which region we're in
        let d_outer = q.length() - ra;
        let d_inner = (q - Vec2::new(d, 0.0)).length() - rb;
        d_outer.max(-d_inner)
    } else {
        let d_outer = q.length() - ra;
        let d_inner = (q - Vec2::new(d, 0.0)).length() - rb;
        d_outer.max(-d_inner)
    };

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moon_inside() {
        // Point on the crescent body (opposite side from inner circle)
        let d = sdf_moon(Vec3::new(0.0, 0.0, -0.8), 0.5, 1.0, 0.8, 0.5);
        assert!(d < 0.0, "Should be inside the crescent, got {}", d);
    }

    #[test]
    fn test_moon_far_outside() {
        let d = sdf_moon(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0, 0.8, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_moon_symmetry_x() {
        let d1 = sdf_moon(Vec3::new(0.3, 0.1, -0.5), 0.5, 1.0, 0.8, 0.5);
        let d2 = sdf_moon(Vec3::new(-0.3, 0.1, -0.5), 0.5, 1.0, 0.8, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_moon_symmetry_y() {
        let d1 = sdf_moon(Vec3::new(0.1, 0.2, -0.5), 0.5, 1.0, 0.8, 0.5);
        let d2 = sdf_moon(Vec3::new(0.1, -0.2, -0.5), 0.5, 1.0, 0.8, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

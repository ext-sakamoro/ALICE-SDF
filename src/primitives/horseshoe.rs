//! Horseshoe SDF (Deep Fried Edition)
//!
//! U-shaped (horseshoe) cross-section in XY plane with Z depth.
//!
//! Based on Inigo Quilez's sdHorseshoe formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a horseshoe shape centered at origin
///
/// Inigo Quilez's `sdHorseshoe` in the XY plane (a ring arc opened by
/// `angle` on each side, extended by two straight legs of length
/// `half_length`, band half-width `width`), extruded along Z by
/// `thickness`. Every branch measures a distance to the band's centre
/// curve, so the field is an exact SDF (Lipschitz 1).
///
/// - `angle`: opening half-angle in radians
/// - `radius`: ring radius
/// - `half_length`: length of each straight leg beyond the arc
/// - `width`: band half-width
/// - `thickness`: half-thickness along Z
///
/// Until 1.10.3 the port mixed an `abs(qx)` leg mirror with the width /
/// thickness terms and was not a distance field (difference quotients up
/// to √2, found by the Lipschitz property test).
#[inline(always)]
pub fn sdf_horseshoe(
    p: Vec3,
    angle: f32,
    radius: f32,
    half_length: f32,
    width: f32,
    thickness: f32,
) -> f32 {
    let c = Vec2::new(alice_det_math::cos(angle), alice_det_math::sin(angle));
    let px = p.x.abs();
    let l = alice_det_math::hypot(px, p.y);
    // rotate into the arc frame: mat2(-c.x, c.y, c.y, c.x) * p
    let qx = (-c.x) * px + (c.y * p.y);
    let qy = c.y * px + (c.x * p.y);
    let qx = if qy > 0.0 || qx > 0.0 {
        qx
    } else {
        l * (-c.x).signum()
    };
    let qy = if qx > 0.0 { qy } else { l };
    // box of half-size (half_length, width) around the leg centre line
    let bx = qx - half_length;
    let by = (qy - radius).abs() - width;
    let d2 = Vec2::new(bx.max(0.0), by.max(0.0)).length() + bx.max(by).min(0.0);
    // extrude along Z
    let dz = p.z.abs() - thickness;
    Vec2::new(d2.max(0.0), dz.max(0.0)).length() + d2.max(dz).min(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horseshoe_exact_points() {
        // angle π/2: the arc is the lower half ring (opening upwards), legs
        // rise from (±r, 0) by half_length. On the band centre at the bottom
        // of the arc the distance is -width.
        let a = std::f32::consts::FRAC_PI_2;
        let d = sdf_horseshoe(Vec3::new(0.0, -0.5, 0.0), a, 0.5, 0.3, 0.1, 0.2);
        assert!((d + 0.1).abs() < 1e-5, "{d}");
        // 0.05 above a leg's flat end (legs end at y = half_length)
        let d = sdf_horseshoe(Vec3::new(0.5, 0.35, 0.0), a, 0.5, 0.3, 0.1, 0.2);
        assert!((d - 0.05).abs() < 1e-5, "{d}");
        // z extrusion: 0.05 beyond the thickness over the arc centre line
        let d = sdf_horseshoe(Vec3::new(0.0, -0.5, 0.25), a, 0.5, 0.3, 0.1, 0.2);
        assert!((d - 0.05).abs() < 1e-5, "{d}");
    }

    #[test]
    fn test_horseshoe_outside() {
        let d = sdf_horseshoe(Vec3::new(5.0, 5.0, 5.0), 1.0, 1.0, 0.5, 0.2, 0.1);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_horseshoe_symmetry_x() {
        let d1 = sdf_horseshoe(Vec3::new(0.5, 0.3, 0.0), 1.0, 1.0, 0.5, 0.2, 0.1);
        let d2 = sdf_horseshoe(Vec3::new(-0.5, 0.3, 0.0), 1.0, 1.0, 0.5, 0.2, 0.1);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

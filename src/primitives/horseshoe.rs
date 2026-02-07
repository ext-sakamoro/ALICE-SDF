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
/// - `angle`: opening half-angle in radians
/// - `radius`: ring radius
/// - `half_length`: half the straight extension length
/// - `width`: cross-section half-width
/// - `thickness`: cross-section half-thickness (Z depth)
#[inline(always)]
pub fn sdf_horseshoe(
    p: Vec3,
    angle: f32,
    radius: f32,
    half_length: f32,
    width: f32,
    thickness: f32,
) -> f32 {
    let c = Vec2::new(angle.cos(), angle.sin());
    let px = p.x.abs();
    let l = (px * px + p.y * p.y).sqrt();

    // Rotate by the angle
    let mut qx = -c.x * px + c.y * p.y;
    let mut qy = c.y * px + c.x * p.y;

    if !(qy > 0.0 || qx > 0.0) {
        qx = l * (-c.x).signum();
    }
    if qx <= 0.0 {
        qy = l;
    }

    qx = qx.abs();
    qy -= radius;

    let rx = (qx - half_length).max(0.0);
    let ry = qy;
    let inner = Vec2::new(rx, ry);
    let inner_len = inner.length() + qx.min(0.0).max(qy.min(0.0));

    let d = Vec2::new(
        (inner_len - width).max(0.0),
        (p.z.abs() - thickness).max(0.0),
    );
    -width.min(thickness) + d.length() + (inner_len - width).max(p.z.abs() - thickness).min(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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

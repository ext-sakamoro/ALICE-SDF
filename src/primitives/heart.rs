//! Heart SDF (Deep Fried Edition)
//!
//! 3D heart shape — revolution of 2D heart contour around Y-axis.
//!
//! Based on Inigo Quilez's 2D heart formula, extended to 3D via revolution.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

#[inline(always)]
fn dot2(v: Vec2) -> f32 {
    v.dot(v)
}

/// Approximate SDF for a 3D heart shape centered at origin
///
/// - `size`: overall size scale
#[inline(always)]
pub fn sdf_heart(p: Vec3, size: f32) -> f32 {
    let sp = p / size;
    // Use revolution: r = length(p.xz), y = p.y
    let mut q = Vec2::new((sp.x * sp.x + sp.z * sp.z).sqrt(), sp.y);

    // Shift so the heart center is at origin
    q.y -= 0.5;
    let qx = q.x.abs();

    if qx + q.y > 1.0 {
        (dot2(Vec2::new(qx - 0.25, q.y - 0.75)).sqrt() - std::f32::consts::SQRT_2 * 0.25) * size
    } else {
        let d1 = dot2(Vec2::new(qx, q.y - 1.0));
        let t = (qx + q.y).max(0.0) * 0.5;
        let d2 = dot2(Vec2::new(qx - t, q.y - t));
        let sign = if qx > q.y { 1.0 } else { -1.0 };
        d1.min(d2).sqrt() * sign * size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heart_center_inside() {
        // The heart shape centers around y=0.5; test a clearly interior point
        let d = sdf_heart(Vec3::new(0.0, 0.6, 0.0), 1.0);
        assert!(d < 0.0, "Center area should be inside, got {}", d);
    }

    #[test]
    fn test_heart_far_outside() {
        let d = sdf_heart(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_heart_scale() {
        let d1 = sdf_heart(Vec3::new(2.0, 0.0, 0.0), 1.0);
        let d2 = sdf_heart(Vec3::new(4.0, 0.0, 0.0), 2.0);
        assert!(
            (d1 / 1.0 - d2 / 2.0).abs() < 0.2,
            "Scale should be proportional"
        );
    }

    #[test]
    fn test_heart_symmetry_xz() {
        let d1 = sdf_heart(Vec3::new(0.3, 0.4, 0.1), 1.0);
        let d2 = sdf_heart(Vec3::new(-0.3, 0.4, -0.1), 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in XZ");
    }
}

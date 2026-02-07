//! Blobby cross SDF (Deep Fried Edition)
//!
//! 2D blobby/organic cross shape in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdBlobbyCross formula using sqrt-based blending.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a blobby (organic) cross shape, extruded along Y-axis
///
/// - `size`: overall size of the blobby cross
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_blobby_cross(p: Vec3, size: f32, half_height: f32) -> f32 {
    // 2D blobby cross in XZ plane (IQ formula)
    let qx = p.x.abs();
    let qz = p.z.abs();
    let q = Vec2::new(qx, qz) / size;

    // sqrt-based organic blending
    let n = q.x + q.y;
    let d_2d = if n < 1.0 {
        let t = 1.0 - n;
        let b = q.x * q.y;
        (-(t * t - 2.0 * b).max(0.0).sqrt() + n - 1.0) * size * 0.5_f32.sqrt()
    } else {
        let d1 = (q.x - 1.0).max(0.0);
        let d2 = (q.y - 1.0).max(0.0);
        let dx = Vec2::new(q.x - 1.0, q.y);
        let dz = Vec2::new(q.x, q.y - 1.0);
        dx.length().min(dz.length()).min((d1 * d1 + d2 * d2).sqrt()) * size
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
    fn test_blobby_cross_center_inside() {
        let d = sdf_blobby_cross(Vec3::ZERO, 1.0, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
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

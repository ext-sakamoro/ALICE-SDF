//! Vesica SDF (Deep Fried Edition)
//!
//! 3D vesica (lens/almond shape) — revolution of 2D vesica piscis around Y-axis.
//!
//! Based on Inigo Quilez's sdVesicaSegment formula (simplified, centered).
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a 3D vesica centered at origin (revolved around Y)
///
/// - `radius`: radius of the two arcs forming the vesica
/// - `half_dist`: half the distance between the two arc centers
#[inline(always)]
pub fn sdf_vesica(p: Vec3, radius: f32, half_dist: f32) -> f32 {
    // 2D cross section in (r, y) where r = length(p.xz)
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), p.y.abs());
    let b = (radius * radius - half_dist * half_dist).max(0.0).sqrt();

    if (q.y - b) * half_dist > q.x * b {
        (q - Vec2::new(0.0, b)).length()
    } else {
        (q - Vec2::new(-half_dist, 0.0)).length() - radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vesica_origin() {
        let d = sdf_vesica(Vec3::ZERO, 1.0, 0.5);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_vesica_outside() {
        let d = sdf_vesica(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_vesica_symmetry_y() {
        let d1 = sdf_vesica(Vec3::new(0.2, 0.3, 0.1), 1.0, 0.5);
        let d2 = sdf_vesica(Vec3::new(0.2, -0.3, 0.1), 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }

    #[test]
    fn test_vesica_sphere_degenerate() {
        // When half_dist = 0, vesica degenerates towards sphere
        let d = sdf_vesica(Vec3::new(2.0, 0.0, 0.0), 1.0, 0.0);
        let d_sphere = Vec3::new(2.0, 0.0, 0.0).length() - 1.0;
        assert!((d - d_sphere).abs() < 0.1, "Zero dist should be ~sphere, got {}", d);
    }
}

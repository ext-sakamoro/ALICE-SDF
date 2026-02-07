//! Uneven Capsule prism SDF (Deep Fried Edition)
//!
//! 2D uneven capsule shape in XY plane, extruded along Z-axis.
//!
//! Based on Inigo Quilez's 2D Uneven Capsule formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for an uneven capsule prism
///
/// Two circles of different radii connected by tangent lines, in XY plane,
/// extruded along Z.
/// - `r1`: bottom circle radius
/// - `r2`: top circle radius
/// - `cap_height`: half-height between circle centers
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_uneven_capsule(p: Vec3, r1: f32, r2: f32, cap_height: f32, half_depth: f32) -> f32 {
    // 2D Uneven Capsule in XY (IQ formula)
    let px = p.x.abs();
    let h = cap_height * 2.0;
    let b = (r1 - r2) / h;
    let a = (1.0 - b * b).max(0.0).sqrt();
    let k = Vec2::new(-b, a).dot(Vec2::new(px, p.y));
    let d_2d = if k < 0.0 {
        Vec2::new(px, p.y).length() - r1
    } else if k > a * h {
        Vec2::new(px, p.y - h).length() - r2
    } else {
        Vec2::new(px, p.y).dot(Vec2::new(a, b)) - r1
    };
    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.max(0.0), d_z.max(0.0));
    d_2d.max(d_z).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uneven_capsule_center_inside() {
        let d = sdf_uneven_capsule(Vec3::ZERO, 0.5, 0.3, 0.5, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_uneven_capsule_far_outside() {
        let d = sdf_uneven_capsule(Vec3::new(5.0, 0.0, 0.0), 0.5, 0.3, 0.5, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_uneven_capsule_symmetry_x() {
        let d1 = sdf_uneven_capsule(Vec3::new(0.2, 0.1, 0.1), 0.5, 0.3, 0.5, 0.5);
        let d2 = sdf_uneven_capsule(Vec3::new(-0.2, 0.1, 0.1), 0.5, 0.3, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_uneven_capsule_symmetry_z() {
        let d1 = sdf_uneven_capsule(Vec3::new(0.1, 0.1, 0.2), 0.5, 0.3, 0.5, 0.5);
        let d2 = sdf_uneven_capsule(Vec3::new(0.1, 0.1, -0.2), 0.5, 0.3, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

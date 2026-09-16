//! Rounded Cylinder SDF (Deep Fried Edition)
//!
//! Cylinder with rounded edges along Y-axis.
//!
//! Based on Inigo Quilez's sdRoundedCylinder formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a rounded cylinder centered at origin
///
/// - `radius`: main cylinder radius (the outer radius, rounding included)
/// - `round_radius`: edge rounding radius
/// - `half_height`: half the cylinder height
///
/// `ρ − radius + round_radius` in the cross section: IQ's `sdRoundedCylinder`
/// writes `− 2·ra` for its own parametrisation, and the CPU paths carried
/// that literally until 2.1.0 (a cylinder of radius 0.4 rendered as 0.8 on
/// the CPU while the shaders drew 0.4). The shader helpers and the AABB use
/// this form.
#[inline(always)]
pub fn sdf_rounded_cylinder(p: Vec3, radius: f32, round_radius: f32, half_height: f32) -> f32 {
    let d = Vec2::new(
        p.x.hypot(p.z) - radius + round_radius,
        p.y.abs() - half_height,
    );
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length() - round_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_cylinder_origin() {
        let d = sdf_rounded_cylinder(Vec3::ZERO, 1.0, 0.1, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_rounded_cylinder_outside() {
        let d = sdf_rounded_cylinder(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.1, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_rounded_cylinder_symmetry_y() {
        let d1 = sdf_rounded_cylinder(Vec3::new(0.5, 0.8, 0.0), 1.0, 0.1, 1.0);
        let d2 = sdf_rounded_cylinder(Vec3::new(0.5, -0.8, 0.0), 1.0, 0.1, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

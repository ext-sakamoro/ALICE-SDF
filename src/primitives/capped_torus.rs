//! Capped Torus SDF (Deep Fried Edition)
//!
//! Torus arc (partial torus) in XZ plane.
//! The cap angle defines how much of the torus is visible.
//!
//! Based on Inigo Quilez's sdCappedTorus formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a capped torus centered at origin in XZ plane
///
/// - `major_radius`: distance from center to tube center
/// - `minor_radius`: tube radius
/// - `cap_angle`: half opening angle in radians (0 = point, PI = full torus)
#[inline(always)]
pub fn sdf_capped_torus(p: Vec3, major_radius: f32, minor_radius: f32, cap_angle: f32) -> f32 {
    let sc = (cap_angle.sin(), cap_angle.cos());
    let px = p.x.abs();
    let k = if sc.1 * px > sc.0 * p.y {
        px * sc.0 + p.y * sc.1
    } else {
        (px * px + p.y * p.y).sqrt()
    };
    (p.x * p.x + p.y * p.y + p.z * p.z + major_radius * major_radius - 2.0 * major_radius * k)
        .sqrt()
        - minor_radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capped_torus_full() {
        // Full torus (cap_angle = PI)
        let d = sdf_capped_torus(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.3, std::f32::consts::PI);
        assert!(
            (d + 0.3).abs() < 0.01,
            "On tube center should be ~-0.3, got {}",
            d
        );
    }

    #[test]
    fn test_capped_torus_origin() {
        let d = sdf_capped_torus(Vec3::ZERO, 1.0, 0.3, std::f32::consts::FRAC_PI_2);
        assert!(d > 0.0, "Origin should be outside, got {}", d);
    }

    #[test]
    fn test_capped_torus_outside() {
        let d = sdf_capped_torus(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.3, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_capped_torus_symmetry_x() {
        let d1 = sdf_capped_torus(Vec3::new(0.5, 0.3, 0.2), 1.0, 0.3, 1.0);
        let d2 = sdf_capped_torus(Vec3::new(-0.5, 0.3, 0.2), 1.0, 0.3, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

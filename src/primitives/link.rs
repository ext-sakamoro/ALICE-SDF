//! Link (Chain Link) SDF (Deep Fried Edition)
//!
//! Exact SDF for a chain link shape centered at origin.
//! A torus stretched along Y-axis.
//!
//! Based on Inigo Quilez's sdLink formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a chain link shape centered at origin
///
/// - `half_length`: half the straight section length along Y
/// - `r1`: major radius (distance from center to tube center)
/// - `r2`: minor radius (tube thickness)
#[inline(always)]
pub fn sdf_link(p: Vec3, half_length: f32, r1: f32, r2: f32) -> f32 {
    let qx = p.x;
    let qy = (p.y.abs() - half_length).max(0.0);
    let qz = p.z;

    let xy_len = (qx * qx + qy * qy).sqrt() - r1;
    (xy_len * xy_len + qz * qz).sqrt() - r2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_link_origin_outside() {
        // Origin is outside the tube (torus-like shape). Distance = r1 - r2
        let d = sdf_link(Vec3::ZERO, 0.5, 1.0, 0.3);
        assert!((d - 0.7).abs() < 0.001, "Origin distance should be r1-r2=0.7, got {}", d);
    }

    #[test]
    fn test_link_on_tube() {
        // At (r1, 0, 0), distance should be approximately -r2
        let d = sdf_link(Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0, 0.3);
        assert!((d + 0.3).abs() < 0.01, "On major radius, d should be ~-r2, got {}", d);
    }

    #[test]
    fn test_link_outside() {
        let d = sdf_link(Vec3::new(5.0, 0.0, 0.0), 0.5, 1.0, 0.3);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_link_symmetry_y() {
        let d1 = sdf_link(Vec3::new(0.5, 1.0, 0.2), 0.5, 1.0, 0.3);
        let d2 = sdf_link(Vec3::new(0.5, -1.0, 0.2), 0.5, 1.0, 0.3);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

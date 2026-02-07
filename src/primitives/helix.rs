//! Helix SDF (Deep Fried Edition)
//!
//! 3D helix (spiral tube) shape along Y-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a helix (spiral tube) along Y-axis
///
/// - `major_r`: major radius (distance from Y-axis to helix center)
/// - `minor_r`: minor radius (tube thickness)
/// - `pitch`: vertical distance per full revolution
/// - `half_height`: half the height along Y (caps the helix)
#[inline(always)]
pub fn sdf_helix(p: Vec3, major_r: f32, minor_r: f32, pitch: f32, half_height: f32) -> f32 {
    let r_xz = Vec2::new(p.x, p.z).length();
    let theta = p.z.atan2(p.x);
    let py = p.y;
    let tau = std::f32::consts::TAU;

    // Radial distance from the helix center circle
    let d_radial = r_xz - major_r;

    // Find nearest helix wrap
    // At angle theta, helix y = (theta + 2*PI*k) * pitch / (2*PI) = theta*pitch/tau + k*pitch
    let y_at_theta = theta * pitch / tau;
    let k = ((py - y_at_theta) / pitch).round();

    // Check k-1, k, k+1 for closest
    let mut d_tube = f32::MAX;
    for dk in [-1.0_f32, 0.0, 1.0] {
        let kk = k + dk;
        let y_helix = y_at_theta + kk * pitch;
        let dy = py - y_helix;
        let d = Vec2::new(d_radial, dy).length() - minor_r;
        d_tube = d_tube.min(d);
    }

    // Cap at half_height
    let d_cap = py.abs() - half_height;
    d_tube.max(d_cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helix_on_tube() {
        // Point on the helix tube at (major_r, 0, 0)
        let d = sdf_helix(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d < 0.0, "Point on helix tube should be inside, got {}", d);
    }

    #[test]
    fn test_helix_far_outside() {
        let d = sdf_helix(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_helix_center_outside() {
        // Center of helix (on the axis) should be outside
        let d = sdf_helix(Vec3::ZERO, 1.0, 0.2, 1.0, 2.0);
        assert!(d > 0.0, "Center (on axis) should be outside, got {}", d);
    }

    #[test]
    fn test_helix_wrap_at_pitch() {
        // At theta=0 (x=major_r, z=0), helix wraps are at y=k*pitch
        // So point (major_r, pitch, 0) should be on the tube (k=1 wrap)
        let d = sdf_helix(Vec3::new(1.0, 1.0, 0.0), 1.0, 0.2, 1.0, 2.0);
        assert!(d < 0.0, "Point on helix wrap should be inside, got {}", d);
    }
}

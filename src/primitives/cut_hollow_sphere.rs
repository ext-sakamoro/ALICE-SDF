//! Cut Hollow Sphere SDF (Deep Fried Edition)
//!
//! Hollow sphere shell with a planar cut.
//!
//! Based on Inigo Quilez's sdCutHollowSphere formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a cut hollow sphere centered at origin
///
/// - `radius`: sphere radius
/// - `cut_height`: Y height of the cut plane
/// - `thickness`: shell thickness
#[inline(always)]
pub fn sdf_cut_hollow_sphere(p: Vec3, radius: f32, cut_height: f32, thickness: f32) -> f32 {
    let w = (radius * radius - cut_height * cut_height).max(0.0).sqrt();
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), p.y);

    if cut_height * q.x < w * q.y {
        (q - Vec2::new(w, cut_height)).length() - thickness
    } else {
        (q.length() - radius).abs() - thickness
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_hollow_sphere_shell() {
        // On the sphere surface (x=radius), should be approximately -thickness
        let d = sdf_cut_hollow_sphere(Vec3::new(1.0, 0.0, 0.0), 1.0, 0.5, 0.05);
        assert!(d.abs() < 0.06, "On surface should be ~thickness, got {}", d);
    }

    #[test]
    fn test_cut_hollow_sphere_origin() {
        // Origin is inside the hollow part
        let d = sdf_cut_hollow_sphere(Vec3::ZERO, 1.0, 0.5, 0.05);
        assert!(d > 0.0, "Origin should be outside (hollow), got {}", d);
    }

    #[test]
    fn test_cut_hollow_sphere_far() {
        let d = sdf_cut_hollow_sphere(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 0.05);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

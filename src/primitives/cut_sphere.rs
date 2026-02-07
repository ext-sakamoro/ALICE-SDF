//! Cut Sphere SDF (Deep Fried Edition)
//!
//! Sphere with a planar cut at a given height.
//!
//! Based on Inigo Quilez's sdCutSphere formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a cut sphere centered at origin
///
/// - `radius`: sphere radius
/// - `cut_height`: Y height of the cut plane (must be in -radius..radius)
#[inline(always)]
pub fn sdf_cut_sphere(p: Vec3, radius: f32, cut_height: f32) -> f32 {
    let w = (radius * radius - cut_height * cut_height).max(0.0).sqrt();
    let q = Vec2::new((p.x * p.x + p.z * p.z).sqrt(), p.y);
    let s1 = (cut_height - radius) * q.x * q.x + w * w * (cut_height + radius - 2.0 * q.y);
    let s2 = cut_height * q.x - w * q.y;
    let s = s1.max(s2);

    if s < 0.0 {
        q.length() - radius
    } else if q.x < w {
        cut_height - q.y
    } else {
        (q - Vec2::new(w, cut_height)).length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cut_sphere_inside() {
        // IQ's cut sphere keeps the cap ABOVE the cut plane
        // Point at (0, 0.8, 0) with r=1, h=0.5 should be inside the cap
        let d = sdf_cut_sphere(Vec3::new(0.0, 0.8, 0.0), 1.0, 0.5);
        assert!(d < 0.0, "Inside cap should be negative, got {}", d);
    }

    #[test]
    fn test_cut_sphere_below_cut() {
        // Point below the cut plane is outside the cap
        let d = sdf_cut_sphere(Vec3::ZERO, 1.0, 0.5);
        assert!(d > 0.0, "Below cut should be positive, got {}", d);
    }

    #[test]
    fn test_cut_sphere_negative_cut() {
        // With h = -0.5 (cut below center), most of sphere is the cap
        let d = sdf_cut_sphere(Vec3::ZERO, 1.0, -0.5);
        assert!(d < 0.0, "Origin should be inside cap with low cut, got {}", d);
    }

    #[test]
    fn test_cut_sphere_outside() {
        let d = sdf_cut_sphere(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

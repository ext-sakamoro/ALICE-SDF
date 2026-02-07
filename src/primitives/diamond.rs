//! Diamond SDF (Deep Fried Edition)
//!
//! Bipyramid (double-cone) with revolution symmetry around Y-axis.
//! Useful for crystals, gems, and collectible items.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a diamond (bipyramid) centered at origin
///
/// - `radius`: equator radius
/// - `half_height`: half the total height (apex to apex)
#[inline(always)]
pub fn sdf_diamond(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let q = Vec2::new(Vec2::new(p.x, p.z).length(), p.y.abs());
    // Line segment from A=(radius, 0) to B=(0, half_height)
    let ba = Vec2::new(-radius, half_height);
    let qa = q - Vec2::new(radius, 0.0);
    let t = (qa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    let closest = Vec2::new(radius, 0.0) + ba * t;
    let dist = (q - closest).length();
    // Sign: inside if r/radius + y/half_height < 1
    if q.x * half_height + q.y * radius < radius * half_height {
        -dist
    } else {
        dist
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diamond_center_inside() {
        let d = sdf_diamond(Vec3::ZERO, 1.0, 1.0);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_diamond_far_outside() {
        let d = sdf_diamond(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_diamond_symmetry_xz() {
        let d1 = sdf_diamond(Vec3::new(0.3, 0.2, 0.1), 1.0, 1.5);
        let d2 = sdf_diamond(Vec3::new(-0.3, 0.2, -0.1), 1.0, 1.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in XZ");
    }

    #[test]
    fn test_diamond_symmetry_y() {
        let d1 = sdf_diamond(Vec3::new(0.3, 0.5, 0.1), 1.0, 1.5);
        let d2 = sdf_diamond(Vec3::new(0.3, -0.5, 0.1), 1.0, 1.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

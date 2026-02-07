//! Death Star SDF (Deep Fried Edition)
//!
//! Sphere with a spherical indentation (like the Death Star).
//! A large sphere with a smaller sphere subtracted from one side.
//!
//! Based on Inigo Quilez's sdDeathStar formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Exact SDF for a Death Star shape centered at origin
///
/// - `ra`: main sphere radius
/// - `rb`: carving sphere radius
/// - `d`: distance between sphere centers along X-axis
#[inline(always)]
pub fn sdf_death_star(p: Vec3, ra: f32, rb: f32, d: f32) -> f32 {
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = (ra * ra - a * a).max(0.0).sqrt();

    let p2 = Vec2::new(p.x, (p.y * p.y + p.z * p.z).sqrt());

    if p2.x * b - p2.y * a > d * (b - p2.y).max(0.0) {
        (p2 - Vec2::new(a, b)).length()
    } else {
        (p2.length() - ra).max(-(Vec2::new(p2.x - d, p2.y).length() - rb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_death_star_back_inside() {
        // Back side (away from carving) should be inside
        let d = sdf_death_star(Vec3::new(-0.8, 0.0, 0.0), 1.0, 0.8, 0.7);
        assert!(d < 0.0, "Back inside should be negative, got {}", d);
    }

    #[test]
    fn test_death_star_carving_outside() {
        // Origin is inside the carving sphere (d=0.7, rb=0.8), so carved out
        let d = sdf_death_star(Vec3::ZERO, 1.0, 0.8, 0.7);
        assert!(d > 0.0, "Origin near carving should be outside, got {}", d);
    }

    #[test]
    fn test_death_star_back() {
        // Back side (negative X), should be like a regular sphere
        let d = sdf_death_star(Vec3::new(-1.0, 0.0, 0.0), 1.0, 0.8, 0.7);
        assert!(d.abs() < 0.01, "Back surface should be ~0, got {}", d);
    }

    #[test]
    fn test_death_star_far() {
        let d = sdf_death_star(Vec3::new(0.0, 5.0, 0.0), 1.0, 0.8, 0.7);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

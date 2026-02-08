//! Truncated Octahedron SDF (Deep Fried Edition)
//!
//! Archimedean solid using GDF with cube + octahedron normals.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use super::gdf_vectors::{GDF_CUBE, GDF_OCTAHEDRON};

/// SDF for a truncated octahedron
///
/// - `radius`: distance from center to face
#[inline(always)]
pub fn sdf_truncated_octahedron(p: Vec3, radius: f32) -> f32 {
    // Uses cube normals [0..3] + octahedron normals [3..7]
    let mut d = f32::NEG_INFINITY;
    for n in &GDF_CUBE {
        d = d.max(p.dot(*n).abs());
    }
    for n in &GDF_OCTAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    d - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_octahedron_origin() {
        let d = sdf_truncated_octahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_truncated_octahedron_outside() {
        let d = sdf_truncated_octahedron(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_truncated_octahedron_face_surface() {
        // Along cube normal (1,0,0) at distance = radius
        let d = sdf_truncated_octahedron(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!(d.abs() < 0.01, "Should be on surface, got {}", d);
    }

    #[test]
    fn test_truncated_octahedron_symmetry() {
        let r = 1.0;
        let d1 = sdf_truncated_octahedron(Vec3::new(0.5, 0.3, 0.2), r);
        let d2 = sdf_truncated_octahedron(Vec3::new(0.5, -0.3, 0.2), r);
        assert!((d1 - d2).abs() < 1e-6);
    }
}

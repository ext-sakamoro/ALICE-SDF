//! Icosahedron SDF (Deep Fried Edition)
//!
//! Regular icosahedron using GDF (Generalized Distance Function).
//! Uses octahedron + icosahedron normals from gdf_vectors.
//!
//! Author: Moroya Sakamoto

use super::gdf_vectors::{GDF_ICOSAHEDRON, GDF_OCTAHEDRON};
use glam::Vec3;

/// SDF for a regular icosahedron
///
/// - `radius`: distance from center to face
#[inline(always)]
pub fn sdf_icosahedron(p: Vec3, radius: f32) -> f32 {
    // Icosahedron uses octahedron normals [3..7] + icosahedron normals [7..13]
    let mut d = f32::NEG_INFINITY;
    for n in &GDF_OCTAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    for n in &GDF_ICOSAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    d - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icosahedron_origin() {
        let d = sdf_icosahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_icosahedron_outside() {
        let d = sdf_icosahedron(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_icosahedron_symmetry() {
        let r = 1.0;
        let d1 = sdf_icosahedron(Vec3::new(0.5, 0.5, 0.5), r);
        let d2 = sdf_icosahedron(Vec3::new(-0.5, 0.5, 0.5), r);
        assert!((d1 - d2).abs() < 1e-6);
    }
}

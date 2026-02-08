//! Truncated Icosahedron SDF (Deep Fried Edition)
//!
//! Soccer ball / football shape. Archimedean solid using GDF with
//! octahedron + icosahedron + dodecahedron normals.
//!
//! Author: Moroya Sakamoto

use super::gdf_vectors::{GDF_DODECAHEDRON, GDF_ICOSAHEDRON, GDF_OCTAHEDRON};
use glam::Vec3;

/// SDF for a truncated icosahedron (soccer ball)
///
/// - `radius`: distance from center to face
#[inline(always)]
pub fn sdf_truncated_icosahedron(p: Vec3, radius: f32) -> f32 {
    // Uses octahedron [3..7] + icosahedron [7..13] + dodecahedron [13..19] normals
    let mut d = f32::NEG_INFINITY;
    for n in &GDF_OCTAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    for n in &GDF_ICOSAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    for n in &GDF_DODECAHEDRON {
        d = d.max(p.dot(*n).abs());
    }
    d - radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncated_icosahedron_origin() {
        let d = sdf_truncated_icosahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_truncated_icosahedron_outside() {
        let d = sdf_truncated_icosahedron(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_truncated_icosahedron_symmetry() {
        let r = 1.0;
        let d1 = sdf_truncated_icosahedron(Vec3::new(0.5, 0.5, 0.5), r);
        let d2 = sdf_truncated_icosahedron(Vec3::new(-0.5, 0.5, 0.5), r);
        assert!((d1 - d2).abs() < 1e-6);
    }

    #[test]
    fn test_truncated_icosahedron_more_faces_than_icosahedron() {
        // Truncated icosahedron has more constraining planes, so it's smaller
        // for the same radius parameter
        let p = Vec3::new(0.7, 0.7, 0.0);
        let d_ico = {
            let mut d = f32::NEG_INFINITY;
            for n in &super::super::gdf_vectors::GDF_OCTAHEDRON {
                d = d.max(p.dot(*n).abs());
            }
            for n in &super::super::gdf_vectors::GDF_ICOSAHEDRON {
                d = d.max(p.dot(*n).abs());
            }
            d - 1.0
        };
        let d_trunc = sdf_truncated_icosahedron(p, 1.0);
        // Truncated version adds more planes, so distance >= icosahedron distance
        assert!(d_trunc >= d_ico - 1e-6);
    }
}

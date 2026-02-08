//! Dodecahedron SDF (Deep Fried Edition)
//!
//! Regular dodecahedron using GDF (Generalized Distance Function).
//! Uses dodecahedron face normals from gdf_vectors.
//!
//! Author: Moroya Sakamoto

use super::gdf_vectors::{gdf_eval, GDF_DODECAHEDRON};
use glam::Vec3;

/// SDF for a regular dodecahedron
///
/// - `radius`: distance from center to face
#[inline(always)]
pub fn sdf_dodecahedron(p: Vec3, radius: f32) -> f32 {
    gdf_eval(p, &GDF_DODECAHEDRON, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dodecahedron_origin() {
        let d = sdf_dodecahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_dodecahedron_outside() {
        let d = sdf_dodecahedron(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_dodecahedron_surface() {
        // A face normal direction at distance = radius should be on surface
        let n = Vec3::new(0.0, 0.8506508083520400, 0.5257311121191336);
        let d = sdf_dodecahedron(n * 1.0, 1.0);
        assert!(
            d.abs() < 0.01,
            "Face center should be on surface, got {}",
            d
        );
    }

    #[test]
    fn test_dodecahedron_symmetry() {
        let r = 1.0;
        let d1 = sdf_dodecahedron(Vec3::new(0.5, 0.5, 0.5), r);
        let d2 = sdf_dodecahedron(Vec3::new(-0.5, 0.5, 0.5), r);
        // Due to abs in GDF, these should be equal
        assert!((d1 - d2).abs() < 1e-6);
    }
}

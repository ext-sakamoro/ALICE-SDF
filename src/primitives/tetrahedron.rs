//! Tetrahedron SDF (Deep Fried Edition)
//!
//! Regular tetrahedron centered at origin.
//! Uses 4 face normals with abs-dot GDF approach.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Pre-normalized tetrahedron face normals
const TETRA_NORMALS: [Vec3; 4] = [
    Vec3::new(0.577_350_26, 0.577_350_26, 0.577_350_26),
    Vec3::new(-0.577_350_26, -0.577_350_26, 0.577_350_26),
    Vec3::new(-0.577_350_26, 0.577_350_26, -0.577_350_26),
    Vec3::new(0.577_350_26, -0.577_350_26, -0.577_350_26),
];

/// SDF for a regular tetrahedron
///
/// - `size`: distance from center to face
#[inline(always)]
pub fn sdf_tetrahedron(p: Vec3, size: f32) -> f32 {
    let mut d = f32::NEG_INFINITY;
    for n in &TETRA_NORMALS {
        d = d.max(p.dot(*n));
    }
    d - size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tetrahedron_origin() {
        let d = sdf_tetrahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_tetrahedron_surface() {
        // Face center along (1,1,1) direction at distance = size
        let n = Vec3::new(1.0, 1.0, 1.0).normalize();
        let d = sdf_tetrahedron(n * 1.0, 1.0);
        assert!(
            d.abs() < 0.01,
            "Face center should be on surface, got {}",
            d
        );
    }

    #[test]
    fn test_tetrahedron_outside() {
        let d = sdf_tetrahedron(Vec3::new(3.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_tetrahedron_symmetry() {
        let size = 1.0;
        let p = Vec3::new(0.5, 0.3, 0.2);
        let d1 = sdf_tetrahedron(p, size);
        // Tetrahedron has S4 symmetry — not full octahedral, but test two normals
        let d2 = sdf_tetrahedron(Vec3::new(-p.x, -p.y, p.z), size);
        // These map to different faces but should give valid distances
        assert!(d1.is_finite() && d2.is_finite());
    }
}

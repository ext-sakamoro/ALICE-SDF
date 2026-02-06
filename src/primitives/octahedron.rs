//! Octahedron SDF (Deep Fried Edition)
//!
//! Exact SDF for a regular octahedron centered at origin.
//!
//! Based on Inigo Quilez's sdOctahedron (exact) formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a regular octahedron centered at origin
///
/// - Vertices at (±s, 0, 0), (0, ±s, 0), (0, 0, ±s)
/// - Parameter `s` controls the size (distance from center to vertex)
#[inline(always)]
pub fn sdf_octahedron(p: Vec3, s: f32) -> f32 {
    let p = Vec3::new(p.x.abs(), p.y.abs(), p.z.abs());
    let m = p.x + p.y + p.z - s;

    let q;
    if 3.0 * p.x < m {
        q = p;
    } else if 3.0 * p.y < m {
        q = Vec3::new(p.y, p.z, p.x);
    } else if 3.0 * p.z < m {
        q = Vec3::new(p.z, p.x, p.y);
    } else {
        return m * 0.57735027; // 1/sqrt(3)
    }

    let k = (0.5 * (q.z - q.y + s)).clamp(0.0, s);
    Vec3::new(q.x, q.y - s + k, q.z - k).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_octahedron_origin_inside() {
        let d = sdf_octahedron(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_octahedron_vertex() {
        // At vertex (1, 0, 0), should be on surface
        let d = sdf_octahedron(Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!(d.abs() < 0.001, "Vertex should be on surface, got {}", d);
    }

    #[test]
    fn test_octahedron_symmetry() {
        let s = 1.5;
        let d1 = sdf_octahedron(Vec3::new(0.5, 0.3, 0.2), s);
        let d2 = sdf_octahedron(Vec3::new(-0.5, 0.3, 0.2), s);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric");
    }

    #[test]
    fn test_octahedron_outside() {
        let d = sdf_octahedron(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

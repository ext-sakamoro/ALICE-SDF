//! Octahedron SDF (Deep Fried Edition)
//!
//! Exact SDF for a regular octahedron centered at origin.
//!
//! Based on Inigo Quilez's sdOctahedron (exact) formula.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{Real, Vec3R};
use glam::Vec3;

/// Octahedron of size `s` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_octahedron_r<R: Real>(p: Vec3R<R>, s: f32) -> R {
    let rs = R::splat(s);
    let three = R::splat(3.0);
    let (ax, ay, az) = (p.x.abs(), p.y.abs(), p.z.abs());
    let m = ax + ay + az - rs;
    let d_flat = m * R::splat(0.57735027);
    let m1 = (three * ax).lt(m);
    let m2 = (three * ay).lt(m);
    let m3 = (three * az).lt(m);
    // Priority: m1, then m2, then m3, else flat (same as the branch chain)
    let qx = R::select(m1, ax, R::select(m2, ay, az));
    let qy = R::select(m1, ay, R::select(m2, az, ax));
    let qz = R::select(m1, az, R::select(m2, ax, ay));
    let k = (R::splat(0.5) * (qz - qy + rs)).clamp(R::zero(), rs);
    let vy = qy - rs + k;
    let vz = qz - k;
    let d_edge = (qx * qx + vy * vy + vz * vz).sqrt();
    let any = R::mask_or(m1, R::mask_or(m2, m3));
    R::select(any, d_edge, d_flat)
}

/// Octahedron of size `s`.
#[inline(always)]
pub fn sdf_octahedron(p: Vec3, s: f32) -> f32 {
    sdf_octahedron_r::<f32>(p.into(), s)
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

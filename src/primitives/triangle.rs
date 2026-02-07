//! Triangle SDF — exact unsigned distance to a 3D triangle
//!
//! Ported from Baker's `sdTriangle(p, a, b, c)` (IQ's formulation).
//! Returns unsigned distance since a triangle is infinitely thin.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Squared length of a vector (avoids sqrt)
#[inline(always)]
fn dot2(v: Vec3) -> f32 {
    v.dot(v)
}

/// Exact unsigned distance from point `p` to triangle `(a, b, c)`.
///
/// Since a triangle has zero volume, there is no meaningful "inside",
/// so this returns unsigned (always >= 0) distance.
#[inline]
pub fn sdf_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ba = b - a;
    let pa = p - a;
    let cb = c - b;
    let pb = p - b;
    let ac = a - c;
    let pc = p - c;
    let nor = ba.cross(ac);

    let sign_a = ba.cross(nor).dot(pa).signum();
    let sign_b = cb.cross(nor).dot(pb).signum();
    let sign_c = ac.cross(nor).dot(pc).signum();

    let d2 = if sign_a + sign_b + sign_c < 2.0 {
        // Point projects outside the triangle — distance to nearest edge
        let d_ab = dot2(ba * (ba.dot(pa) / ba.dot(ba)).clamp(0.0, 1.0) - pa);
        let d_bc = dot2(cb * (cb.dot(pb) / cb.dot(cb)).clamp(0.0, 1.0) - pb);
        let d_ca = dot2(ac * (ac.dot(pc) / ac.dot(ac)).clamp(0.0, 1.0) - pc);
        d_ab.min(d_bc).min(d_ca)
    } else {
        // Point projects onto the triangle plane — perpendicular distance
        let d = nor.dot(pa);
        d * d / nor.dot(nor)
    };

    d2.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_on_vertex() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);

        // On vertex a
        let d = sdf_triangle(a, a, b, c);
        assert!(d.abs() < 0.001, "On vertex a: {}", d);
    }

    #[test]
    fn test_triangle_above_center() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);

        // Center of triangle, lifted by 1.0 along Z
        let center = (a + b + c) / 3.0 + Vec3::new(0.0, 0.0, 1.0);
        let d = sdf_triangle(center, a, b, c);
        assert!((d - 1.0).abs() < 0.001, "Above center: {}", d);
    }

    #[test]
    fn test_triangle_on_edge() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 2.0, 0.0);

        // Midpoint of edge AB
        let mid = (a + b) * 0.5;
        let d = sdf_triangle(mid, a, b, c);
        assert!(d.abs() < 0.001, "On edge: {}", d);
    }

    #[test]
    fn test_triangle_always_positive() {
        let a = Vec3::new(-1.0, -1.0, 0.0);
        let b = Vec3::new(1.0, -1.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);

        for i in 0..100 {
            let x = (i as f32 * 0.137) - 3.0;
            let y = (i as f32 * 0.291) - 3.0;
            let z = (i as f32 * 0.473) - 3.0;
            let d = sdf_triangle(Vec3::new(x, y, z), a, b, c);
            assert!(d >= 0.0, "Distance should be >= 0, got {}", d);
        }
    }
}

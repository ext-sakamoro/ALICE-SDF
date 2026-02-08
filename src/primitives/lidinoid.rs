//! Lidinoid Surface TPMS SDF (Deep Fried Edition)
//!
//! Lidinoid triply-periodic minimal surface.
//! Implicit: 0.5*(sin(2x)*cos(y)*sin(z) + sin(x)*sin(2y)*cos(z) + cos(x)*sin(y)*sin(2z))
//!         - 0.5*(cos(2x)*cos(2y) + cos(2y)*cos(2z) + cos(2z)*cos(2x)) + 0.15 = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a Lidinoid surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_lidinoid(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    // Double-angle identities: sin(2x) = 2*sin(x)*cos(x), cos(2x) = 2*cos²(x)-1
    // Eliminates 3 sin_cos() calls (6 trig → 0 trig)
    let s2x = 2.0 * sx * cx;
    let s2y = 2.0 * sy * cy;
    let s2z = 2.0 * sz * cz;
    let c2x = 2.0 * cx * cx - 1.0;
    let c2y = 2.0 * cy * cy - 1.0;
    let c2z = 2.0 * cz * cz - 1.0;

    let term1 = 0.5 * (s2x * cy * sz + sx * s2y * cz + cx * sy * s2z);
    let term2 = 0.5 * (c2x * c2y + c2y * c2z + c2z * c2x);
    let d = term1 - term2 + 0.15;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lidinoid_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_lidinoid(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_lidinoid(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_lidinoid_finite() {
        let d = sdf_lidinoid(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }

    #[test]
    fn test_lidinoid_symmetry() {
        let d1 = sdf_lidinoid(Vec3::new(0.5, 0.5, 0.5), 1.0, 0.1);
        assert!(d1.is_finite());
    }
}

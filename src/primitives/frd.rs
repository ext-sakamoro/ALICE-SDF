//! FRD Surface TPMS SDF (Deep Fried Edition)
//!
//! F-RD (Face-centered cubic, Rhombic Dodecahedron) triply-periodic minimal surface.
//! Implicit: cos(2x)*sin(y)*cos(z) + cos(x)*cos(2y)*sin(z) + sin(x)*cos(y)*cos(2z) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for an FRD surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_frd(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    // Double-angle identity: cos(2x) = 2*cos²(x) - 1 — eliminates 3 trig calls
    let c2x = 2.0 * cx * cx - 1.0;
    let c2y = 2.0 * cy * cy - 1.0;
    let c2z = 2.0 * cz * cz - 1.0;
    let d = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frd_origin() {
        // At origin: sin=0, cos=1: d = 1*0*1 + 1*1*0 + 0*1*1 = 0
        let d = sdf_frd(Vec3::ZERO, 1.0, 0.1);
        assert!((d + 0.1).abs() < 0.01, "Origin on surface minus thickness, got {}", d);
    }

    #[test]
    fn test_frd_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_frd(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_frd(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_frd_finite() {
        let d = sdf_frd(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }
}

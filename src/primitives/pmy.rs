//! PMY Surface TPMS SDF (Deep Fried Edition)
//!
//! PMY triply-periodic minimal surface.
//! Implicit: 2*cos(x)*cos(y)*cos(z) + sin(2x)*sin(y) + sin(x)*sin(2z) + sin(2y)*sin(z) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a PMY surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_pmy(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    // Double-angle identity: sin(2x) = 2*sin(x)*cos(x) — eliminates 3 trig calls
    let s2x = 2.0 * sx * cx;
    let s2y = 2.0 * sy * cy;
    let s2z = 2.0 * sz * cz;
    let d = 2.0 * cx * cy * cz + s2x * sy + sx * s2z + s2y * sz;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pmy_origin() {
        // At origin: cos=1, sin=0: d = 2*1*1*1 + 0 + 0 + 0 = 2
        let d = sdf_pmy(Vec3::ZERO, 1.0, 0.1);
        assert!((d - 1.9).abs() < 0.01, "Expected ~1.9, got {}", d);
    }

    #[test]
    fn test_pmy_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_pmy(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_pmy(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_pmy_finite() {
        let d = sdf_pmy(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }
}

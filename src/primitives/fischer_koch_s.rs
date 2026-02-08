//! Fischer-Koch S Surface TPMS SDF (Deep Fried Edition)
//!
//! Fischer-Koch S triply-periodic minimal surface.
//! Uses the same implicit as FRD but with a level-set offset of 0.4.
//! Implicit: cos(2x)*sin(y)*cos(z) + cos(x)*cos(2y)*sin(z) + sin(x)*cos(y)*cos(2z) - 0.4 = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a Fischer-Koch S surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_fischer_koch_s(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    // Double-angle identity: cos(2x) = 2*cos²(x) - 1 — eliminates 3 trig calls
    let c2x = 2.0 * cx * cx - 1.0;
    let c2y = 2.0 * cy * cy - 1.0;
    let c2z = 2.0 * cz * cz - 1.0;
    let d = c2x * sy * cz + cx * c2y * sz + sx * cy * c2z - 0.4;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fischer_koch_s_origin() {
        // At origin: d = 0 - 0.4 = -0.4, |d|/1 - thickness = 0.4 - 0.1 = 0.3
        let d = sdf_fischer_koch_s(Vec3::ZERO, 1.0, 0.1);
        assert!((d - 0.3).abs() < 0.01, "Expected ~0.3, got {}", d);
    }

    #[test]
    fn test_fischer_koch_s_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_fischer_koch_s(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_fischer_koch_s(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_fischer_koch_s_differs_from_frd() {
        // Should give different values than FRD at same point
        use super::super::frd::sdf_frd;
        let p = Vec3::new(0.5, 0.5, 0.5);
        let d_fk = sdf_fischer_koch_s(p, 1.0, 0.1);
        let d_frd = sdf_frd(p, 1.0, 0.1);
        assert!((d_fk - d_frd).abs() > 0.01, "Should differ from FRD");
    }
}

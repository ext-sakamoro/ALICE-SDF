//! IWP Surface TPMS SDF (Deep Fried Edition)
//!
//! I-WP (I-graph and Wrapped Package) triply-periodic minimal surface.
//! Implicit: 2*(cos(x)*cos(y)+cos(y)*cos(z)+cos(z)*cos(x)) - (cos(2x)+cos(2y)+cos(2z)) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for an IWP surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_iwp(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let cx = sp.x.cos();
    let cy = sp.y.cos();
    let cz = sp.z.cos();
    // Double-angle identity: cos(2x) = 2*cos²(x) - 1 — eliminates 3 trig calls
    let c2x = 2.0 * cx * cx - 1.0;
    let c2y = 2.0 * cy * cy - 1.0;
    let c2z = 2.0 * cz * cz - 1.0;
    let d = 2.0 * (cx * cy + cy * cz + cz * cx) - (c2x + c2y + c2z);
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iwp_origin() {
        // cos(0)=1, cos(0)=1: d = 2*(1+1+1) - (1+1+1) = 3
        let d = sdf_iwp(Vec3::ZERO, 1.0, 0.1);
        assert!((d - 2.9).abs() < 0.01, "Expected ~2.9, got {}", d);
    }

    #[test]
    fn test_iwp_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_iwp(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_iwp(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_iwp_finite() {
        let d = sdf_iwp(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }
}

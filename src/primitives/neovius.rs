//! Neovius Surface TPMS SDF (Deep Fried Edition)
//!
//! Neovius triply-periodic minimal surface.
//! Implicit: 3*(cos(x)+cos(y)+cos(z)) + 4*cos(x)*cos(y)*cos(z) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a Neovius surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_neovius(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let cx = sp.x.cos();
    let cy = sp.y.cos();
    let cz = sp.z.cos();
    let d = 3.0 * (cx + cy + cz) + 4.0 * cx * cy * cz;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neovius_origin() {
        // At origin: cos(0)=1, d = 3*(1+1+1) + 4*1*1*1 = 13
        let d = sdf_neovius(Vec3::ZERO, 1.0, 0.1);
        assert!((d - 12.9).abs() < 0.01, "Expected ~12.9, got {}", d);
    }

    #[test]
    fn test_neovius_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_neovius(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_neovius(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_neovius_finite() {
        let d = sdf_neovius(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }
}

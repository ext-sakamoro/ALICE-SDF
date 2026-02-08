//! Diamond Surface TPMS SDF (Deep Fried Edition)
//!
//! Schwarz Diamond triply-periodic minimal surface.
//! Implicit: sin(x)*sin(y)*sin(z) + sin(x)*cos(y)*cos(z)
//!         + cos(x)*sin(y)*cos(z) + cos(x)*cos(y)*sin(z) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a Diamond surface (TPMS)
///
/// - `scale`: spatial frequency
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_diamond_surface(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let (sx, cx) = sp.x.sin_cos();
    let (sy, cy) = sp.y.sin_cos();
    let (sz, cz) = sp.z.sin_cos();
    let d = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz;
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diamond_surface_origin() {
        // At origin: all sin=0, cos=1, d = 0+0+0+1*1*0 = 0
        // Actually: sin(0)*sin(0)*sin(0) + sin(0)*cos(0)*cos(0) + cos(0)*sin(0)*cos(0) + cos(0)*cos(0)*sin(0) = 0
        let d = sdf_diamond_surface(Vec3::ZERO, 1.0, 0.1);
        assert!(
            (d + 0.1).abs() < 0.01,
            "Origin on surface minus thickness, got {}",
            d
        );
    }

    #[test]
    fn test_diamond_surface_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_diamond_surface(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_diamond_surface(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic");
    }

    #[test]
    fn test_diamond_surface_finite() {
        let d = sdf_diamond_surface(Vec3::new(1.0, 2.0, 3.0), 2.0, 0.05);
        assert!(d.is_finite());
    }
}

//! Schwarz P Surface SDF (Deep Fried Edition)
//!
//! Schwarz P triply-periodic minimal surface (TPMS).
//! Surface satisfies: cos(x) + cos(y) + cos(z) = 0
//!
//! Popular for 3D printing lattices, SF architecture, and procedural art.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a Schwarz P surface
///
/// - `scale`: spatial frequency (larger = more repetitions)
/// - `thickness`: shell half-thickness
#[inline(always)]
pub fn sdf_schwarz_p(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let d = sp.x.cos() + sp.y.cos() + sp.z.cos();
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarz_p_on_surface() {
        let half_pi = std::f32::consts::FRAC_PI_2;
        let d = sdf_schwarz_p(Vec3::splat(half_pi), 1.0, 0.1);
        assert!(
            (d + 0.1).abs() < 0.01,
            "Should be on surface minus thickness, got {}",
            d
        );
    }

    #[test]
    fn test_schwarz_p_origin() {
        // cos(0)+cos(0)+cos(0) = 3, d = |3|/1 - 0.1 = 2.9
        let d = sdf_schwarz_p(Vec3::ZERO, 1.0, 0.1);
        assert!((d - 2.9).abs() < 0.01, "Expected ~2.9, got {}", d);
    }

    #[test]
    fn test_schwarz_p_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_schwarz_p(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_schwarz_p(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic with 2*PI");
    }

    #[test]
    fn test_schwarz_p_scale() {
        let d1 = sdf_schwarz_p(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_schwarz_p(Vec3::new(0.5, 0.3, 0.7), 2.0, 0.1);
        assert!(d1.is_finite() && d2.is_finite());
    }
}

//! Gyroid SDF (Deep Fried Edition)
//!
//! Gyroid triply-periodic minimal surface (TPMS).
//! Popular in procedural art, 3D printing lattices, and architectural design.
//!
//! The gyroid surface satisfies: sin(x)*cos(y) + sin(y)*cos(z) + sin(z)*cos(x) = 0
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Approximate SDF for a gyroid surface
///
/// - `scale`: spatial frequency (larger = more repetitions)
/// - `thickness`: shell thickness (half)
#[inline(always)]
pub fn sdf_gyroid(p: Vec3, scale: f32, thickness: f32) -> f32 {
    let sp = p * scale;
    let d = sp.x.sin() * sp.y.cos() + sp.y.sin() * sp.z.cos() + sp.z.sin() * sp.x.cos();
    // Normalize by scale for proper distance metric
    d.abs() / scale - thickness
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gyroid_on_surface() {
        // At (PI/2, 0, 0): sin(PI/2)*cos(0) + sin(0)*cos(0) + sin(0)*cos(PI/2) = 1
        // So distance from surface is (1/scale - thickness)
        let d = sdf_gyroid(Vec3::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0), 1.0, 0.5);
        assert!((d - 0.5).abs() < 0.01, "Expected ~0.5, got {}", d);
    }

    #[test]
    fn test_gyroid_origin() {
        // At origin: sin(0)*cos(0) + sin(0)*cos(0) + sin(0)*cos(0) = 0
        // Should be on the surface minus thickness
        let d = sdf_gyroid(Vec3::ZERO, 1.0, 0.1);
        assert!(
            (d + 0.1).abs() < 0.01,
            "Origin should be -thickness, got {}",
            d
        );
    }

    #[test]
    fn test_gyroid_periodicity() {
        let tau = 2.0 * std::f32::consts::PI;
        let d1 = sdf_gyroid(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_gyroid(Vec3::new(0.5 + tau, 0.3 + tau, 0.7 + tau), 1.0, 0.1);
        assert!((d1 - d2).abs() < 0.01, "Should be periodic with 2*PI");
    }

    #[test]
    fn test_gyroid_scale() {
        // Higher scale = more frequent oscillations
        let d1 = sdf_gyroid(Vec3::new(0.5, 0.3, 0.7), 1.0, 0.1);
        let d2 = sdf_gyroid(Vec3::new(0.5, 0.3, 0.7), 2.0, 0.1);
        // Just check both are finite
        assert!(d1.is_finite() && d2.is_finite());
    }
}

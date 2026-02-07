//! Triangular Prism SDF (Deep Fried Edition)
//!
//! Equilateral triangular prism along Z-axis.
//!
//! Based on Inigo Quilez's sdTriPrism formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a triangular prism centered at origin
///
/// - `width`: half-width of the equilateral triangle cross-section
/// - `half_depth`: half the prism depth along Z
#[inline(always)]
pub fn sdf_triangular_prism(p: Vec3, width: f32, half_depth: f32) -> f32 {
    let q = p.abs();
    // 0.866025 = sqrt(3)/2
    (q.z - half_depth).max((q.x * 0.866025 + p.y * 0.5).max(-p.y) - width * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_prism_origin() {
        let d = sdf_triangular_prism(Vec3::ZERO, 1.0, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_triangular_prism_outside() {
        let d = sdf_triangular_prism(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_triangular_prism_symmetry_z() {
        let d1 = sdf_triangular_prism(Vec3::new(0.2, 0.1, 0.5), 1.0, 1.0);
        let d2 = sdf_triangular_prism(Vec3::new(0.2, 0.1, -0.5), 1.0, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

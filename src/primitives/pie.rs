//! Pie (sector prism) SDF (Deep Fried Edition)
//!
//! 2D pie/sector shape in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's 2D Pie formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a pie (sector) shape, extruded along Y-axis
///
/// - `angle`: half opening angle in radians
/// - `radius`: pie radius
/// - `half_height`: half the extrusion height along Y
///
/// The sector opens around the +Z direction in the XZ plane.
/// Note: the apex (origin in XZ) is on the boundary, so d=0 there.
#[inline(always)]
pub fn sdf_pie(p: Vec3, angle: f32, radius: f32, half_height: f32) -> f32 {
    // 2D Pie in XZ plane (IQ formula)
    let qx = p.x.abs();
    let qz = p.z;
    let q = Vec2::new(qx, qz);
    let sc = Vec2::new(angle.sin(), angle.cos());
    let l = q.length() - radius;
    let dot_qc = q.dot(sc).clamp(0.0, radius);
    let m = (q - sc * dot_qc).length();
    let cross_val = sc.y * qx - sc.x * qz;
    let s = if cross_val > 0.0 { 1.0 } else if cross_val < 0.0 { -1.0 } else { 0.0 };
    let d_2d = l.max(m * s);
    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pie_inside() {
        // Point inside the sector (along +Z, away from the apex)
        let d = sdf_pie(Vec3::new(0.0, 0.0, 0.5), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        assert!(d < 0.0, "Interior point should be inside, got {}", d);
    }

    #[test]
    fn test_pie_far_outside() {
        let d = sdf_pie(Vec3::new(5.0, 0.0, 0.0), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_pie_symmetry_y() {
        let d1 = sdf_pie(Vec3::new(0.1, 0.2, 0.5), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        let d2 = sdf_pie(Vec3::new(0.1, -0.2, 0.5), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }

    #[test]
    fn test_pie_symmetry_x() {
        let d1 = sdf_pie(Vec3::new(0.2, 0.1, 0.5), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        let d2 = sdf_pie(Vec3::new(-0.2, 0.1, 0.5), std::f32::consts::FRAC_PI_4, 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

//! Pyramid SDF (Deep Fried Edition)
//!
//! Exact SDF for a 4-sided pyramid centered at origin.
//! Base square (side = 1) at y = -half_height,
//! Tip at y = half_height.
//!
//! Based on Inigo Quilez's sdPyramid formula.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

/// Exact SDF for a 4-sided pyramid centered at origin
///
/// - Square base (side = 1) at y = -half_height
/// - Tip at y = half_height
/// - Use Scale/ScaleNonUniform to change base size
#[inline(always)]
pub fn sdf_pyramid(p: Vec3, half_height: f32) -> f32 {
    let h = half_height * 2.0;
    let m2 = h * h + 0.25;

    // Shift to base at y=0
    let py = p.y + half_height;

    let mut px = p.x.abs();
    let mut pz = p.z.abs();
    if pz > px {
        std::mem::swap(&mut px, &mut pz);
    }
    px -= 0.5;
    pz -= 0.5;

    let qx = pz;
    let qy = h * py - 0.5 * px;
    let qz = h * px + 0.5 * py;

    let s = (-qx).max(0.0);
    let t = ((qy - 0.5 * pz) / (m2 + 0.25)).clamp(0.0, 1.0);

    let a = m2 * (qx + s) * (qx + s) + qy * qy;
    let b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);

    let d2 = if qy.min(-qx * m2 - qy * 0.5) > 0.0 {
        0.0
    } else {
        a.min(b)
    };

    ((d2 + qz * qz) / m2).sqrt() * qz.max(-py).signum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pyramid_origin_inside() {
        let d = sdf_pyramid(Vec3::ZERO, 1.0);
        assert!(d < 0.0, "Origin should be inside, got {}", d);
    }

    #[test]
    fn test_pyramid_tip() {
        let d = sdf_pyramid(Vec3::new(0.0, 1.0, 0.0), 1.0);
        assert!(d.abs() < 0.001, "Tip should be on surface, got {}", d);
    }

    #[test]
    fn test_pyramid_outside() {
        let d = sdf_pyramid(Vec3::new(5.0, 0.0, 0.0), 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }
}

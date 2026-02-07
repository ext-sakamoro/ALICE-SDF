//! Parabola segment SDF (Deep Fried Edition)
//!
//! 2D parabolic arch in XY plane, extruded along Z-axis.
//!
//! The arch shape: y = para_height * (1 - (x/width)^2) for |x| <= width.
//! Interior is the region between y=0 and the arch curve.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a parabolic arch segment, extruded along Z-axis
///
/// Arch curve: y = para_height * (1 - (x/width)^2).
/// Interior: 0 <= y <= arch curve, |x| <= width.
/// - `width`: half-width of the arch base
/// - `para_height`: height of the arch at center (x=0)
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_parabola_segment(p: Vec3, width: f32, para_height: f32, half_depth: f32) -> f32 {
    let px = p.x.abs();
    let py = p.y;
    let w = width;
    let h = para_height;

    // Arch curve: y = h * (1 - (x/w)^2), for x in [0, w]
    // Inside check
    let y_arch = if px <= w { h * (1.0 - (px / w) * (px / w)) } else { 0.0 };
    let inside = px <= w && py >= 0.0 && py <= y_arch;

    // Distance to parabola curve via Newton's method
    // Find closest point on (t, h*(1-(t/w)^2)) for t in [0, w]
    let ww = w * w;
    let mut t = px.clamp(0.0, w);
    for _ in 0..8 {
        let ft = h * (1.0 - t * t / ww);
        let dft = -2.0 * h * t / ww;
        let ex = px - t;
        let ey = py - ft;
        // Minimize dist^2: d/dt[(px-t)^2 + (py-ft)^2] = -2*ex + 2*ey*dft = 0
        let f = -ex + ey * dft;
        let df = 1.0 + dft * dft + ey * (-2.0 * h / ww);
        if df.abs() > 1e-10 {
            t = (t - f / df).clamp(0.0, w);
        }
    }
    let closest_y = h * (1.0 - t * t / ww);
    let d_para = Vec2::new(px - t, py - closest_y).length();

    // Distance to base (y=0 line from x=0 to x=w)
    let d_base = if px <= w {
        py.abs()
    } else {
        Vec2::new(px - w, py).length()
    };

    let d_unsigned = d_para.min(d_base);
    let d_2d = if inside { -d_unsigned } else { d_unsigned };

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let ext = Vec2::new(d_2d.max(0.0), d_z.max(0.0));
    d_2d.max(d_z).min(0.0) + ext.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parabola_segment_inside() {
        // Point at center, below the arch top (h=1)
        let d = sdf_parabola_segment(Vec3::new(0.0, 0.3, 0.0), 1.0, 1.0, 0.5);
        assert!(d < 0.0, "Interior point should be inside, got {}", d);
    }

    #[test]
    fn test_parabola_segment_far_outside() {
        let d = sdf_parabola_segment(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_parabola_segment_symmetry_x() {
        let d1 = sdf_parabola_segment(Vec3::new(0.3, 0.2, 0.1), 1.0, 1.0, 0.5);
        let d2 = sdf_parabola_segment(Vec3::new(-0.3, 0.2, 0.1), 1.0, 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_parabola_segment_symmetry_z() {
        let d1 = sdf_parabola_segment(Vec3::new(0.2, 0.3, 0.2), 1.0, 1.0, 0.5);
        let d2 = sdf_parabola_segment(Vec3::new(0.2, 0.3, -0.2), 1.0, 1.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

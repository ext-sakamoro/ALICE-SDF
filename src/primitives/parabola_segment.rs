//! Parabola segment SDF (Deep Fried Edition)
//!
//! 2D parabolic arch in XY plane, extruded along Z-axis.
//!
//! The arch shape: y = para_height * (1 - (x/width)^2) for |x| <= width.
//! Interior is the region between y=0 and the arch curve.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};

/// SDF for a parabolic arch segment, extruded along Z-axis
///
/// Newton steps from the upper bound of the nearest parameter. Measured against an f64 truth over
/// 62,460 points: 3 steps leave 1.6e-3, 4 and more leave only f32 rounding; the fifth is margin.
const NEWTON_STEPS: usize = 5;

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
    let y_arch = if px <= w {
        h * ((px / w) * -(px / w) + 1.0)
    } else {
        0.0
    };
    let inside = px <= w && py >= 0.0 && py <= y_arch;

    // Nearest point of the arch: with a = h / w^2 the stationary points of the squared distance are the
    // roots of g(t) = k t^3 - m t - px (k = 2 a^2, m = 2 a (h - py) - 1). g is convex on t >= 0 and
    // g(0) <= 0, so it has one root on t > 0, which is the global minimum on [0, w], and Newton from any
    // t right of the root descends to it without overshooting. sqrt(m / k) + cbrt(px / k) is above the
    // root (substitute it into g), and the arch's end is the start when that is further or k = 0.
    let ww = w * w;
    let a = h / ww;
    let k = 2.0 * a * a;
    let m = 2.0 * a * (h - py) - 1.0;
    let mut t = w;
    if k > 0.0 {
        let bound = (m / k).law_max(0.0).sqrt() + alice_det_math::cbrt(px / k);
        if bound < w {
            t = bound;
        }
    }
    for _ in 0..NEWTON_STEPS {
        let g = k * t * t * t - m * t - px;
        let gp = 3.0 * k * t * t - m;
        if gp > 0.0 {
            t = super::clamp_total(t - g / gp, 0.0, w);
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

    let d_unsigned = d_para.law_min(d_base);
    let d_2d = if inside { -d_unsigned } else { d_unsigned };

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let ext = Vec2::new(d_2d.law_max(0.0), d_z.law_max(0.0));
    d_2d.law_max(d_z).law_min(0.0) + ext.length()
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

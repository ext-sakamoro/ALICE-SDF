//! Parallelogram prism SDF (Deep Fried Edition)
//!
//! 2D parallelogram shape in XY plane, extruded along Z-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Point-to-segment distance helper
#[inline(always)]
fn dist_to_seg(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    let t = if len_sq > 0.0 {
        ((px - ax) * dx + (py - ay) * dy) / len_sq
    } else {
        0.0
    };
    let t = t.clamp(0.0, 1.0);
    let cx = ax + dx * t;
    let cy = ay + dy * t;
    ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt()
}

/// Cross product (p-a) x (b-a) for inside test
#[inline(always)]
fn cross_2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    (px - ax) * (by - ay) - (py - ay) * (bx - ax)
}

/// SDF for a parallelogram prism
///
/// 2D parallelogram in XY plane extruded along Z.
/// - `width`: half-width of the parallelogram
/// - `para_height`: half-height of the 2D parallelogram
/// - `skew`: horizontal skew amount
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_parallelogram(p: Vec3, width: f32, para_height: f32, skew: f32, half_depth: f32) -> f32 {
    let px = p.x;
    let py = p.y;
    let he = para_height;
    let wi = width;
    let sk = skew;

    // Vertices (counter-clockwise):
    // D = (wi - sk, -he)   bottom-right
    // A = (wi + sk,  he)   top-right
    // B = (-wi + sk, he)   top-left
    // C = (-wi - sk, -he)  bottom-left
    let (dx, dy) = (wi - sk, -he);
    let (ax, ay) = (wi + sk, he);
    let (bx, by) = (-wi + sk, he);
    let (cx, cy) = (-wi - sk, -he);

    // Distance to 4 edge segments
    let d1 = dist_to_seg(px, py, dx, dy, ax, ay); // right
    let d2 = dist_to_seg(px, py, ax, ay, bx, by); // top
    let d3 = dist_to_seg(px, py, bx, by, cx, cy); // left
    let d4 = dist_to_seg(px, py, cx, cy, dx, dy); // bottom

    let d_unsigned = d1.min(d2).min(d3).min(d4);

    // Inside: (P-A)×(B-A) is negative for interior of CCW polygon
    let c1 = cross_2d(px, py, dx, dy, ax, ay);
    let c2 = cross_2d(px, py, ax, ay, bx, by);
    let c3 = cross_2d(px, py, bx, by, cx, cy);
    let c4 = cross_2d(px, py, cx, cy, dx, dy);
    let inside = c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0 && c4 <= 0.0;

    let d_2d = if inside { -d_unsigned } else { d_unsigned };

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.max(0.0), d_z.max(0.0));
    d_2d.max(d_z).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallelogram_center_inside() {
        let d = sdf_parallelogram(Vec3::ZERO, 1.0, 0.5, 0.3, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_parallelogram_far_outside() {
        let d = sdf_parallelogram(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 0.3, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_parallelogram_symmetry_z() {
        let d1 = sdf_parallelogram(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.5, 0.3, 0.5);
        let d2 = sdf_parallelogram(Vec3::new(0.3, 0.1, -0.2), 1.0, 0.5, 0.3, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }

    #[test]
    fn test_parallelogram_scale() {
        let d1 = sdf_parallelogram(Vec3::ZERO, 1.0, 0.5, 0.3, 0.5);
        let d2 = sdf_parallelogram(Vec3::ZERO, 2.0, 1.0, 0.6, 1.0);
        assert!(d1 < 0.0 && d2 < 0.0, "Both should be inside");
        assert!(d2 < d1, "Larger shape should have more negative distance at center");
    }
}

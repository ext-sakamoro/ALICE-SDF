//! Trapezoid prism SDF (Deep Fried Edition)
//!
//! 2D trapezoid shape in XY plane, extruded along Z-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Point-to-segment squared distance helper
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

/// SDF for a trapezoid prism
///
/// 2D trapezoid in XY plane extruded along Z.
/// - `r1`: half-width at bottom
/// - `r2`: half-width at top
/// - `trap_height`: half-height of the 2D trapezoid (Y-axis)
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_trapezoid(p: Vec3, r1: f32, r2: f32, trap_height: f32, half_depth: f32) -> f32 {
    let px = p.x.abs();
    let py = p.y;
    let he = trap_height;

    // Distance to three edge segments (right half after abs):
    // Bottom: (0,-he) to (r1,-he)
    let d_bot = dist_to_seg(px, py, 0.0, -he, r1, -he);
    // Slant: (r1,-he) to (r2,he)
    let d_slant = dist_to_seg(px, py, r1, -he, r2, he);
    // Top: (r2,he) to (0,he)
    let d_top = dist_to_seg(px, py, r2, he, 0.0, he);

    let d_unsigned = d_bot.min(d_slant).min(d_top);

    // Sign: inside check via half-plane test
    // Inside if: |py| <= he AND px <= interpolated width at py
    let d_y_inside = py >= -he && py <= he;
    // Slant outward normal: perpendicular to (r2-r1, 2*he) = (2*he, r1-r2)
    let nx = 2.0 * he;
    let ny = r1 - r2;
    let d_slant_plane = (px - r1) * nx + (py + he) * ny;
    let inside = d_y_inside && d_slant_plane <= 0.0;

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
    fn test_trapezoid_center_inside() {
        let d = sdf_trapezoid(Vec3::ZERO, 1.0, 0.5, 0.5, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_trapezoid_far_outside() {
        let d = sdf_trapezoid(Vec3::new(5.0, 0.0, 0.0), 1.0, 0.5, 0.5, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_trapezoid_symmetry_x() {
        let d1 = sdf_trapezoid(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.5, 0.5, 0.5);
        let d2 = sdf_trapezoid(Vec3::new(-0.3, 0.1, 0.2), 1.0, 0.5, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_trapezoid_symmetry_z() {
        let d1 = sdf_trapezoid(Vec3::new(0.3, 0.1, 0.2), 1.0, 0.5, 0.5, 0.5);
        let d2 = sdf_trapezoid(Vec3::new(0.3, 0.1, -0.2), 1.0, 0.5, 0.5, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

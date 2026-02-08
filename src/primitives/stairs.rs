//! Stairs SDF (Deep Fried Edition)
//!
//! Staircase profile in XY plane, extruded along Z-axis.
//! The staircase is a union of progressively taller rectangular steps.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF of a single step box in local coordinates
#[inline(always)]
fn step_box(lx: f32, ly: f32, s: f32, sw: f32, sh: f32) -> f32 {
    let cx = s * sw + sw * 0.5;
    let hy = (s + 1.0) * sh * 0.5;
    let dx = (lx - cx).abs() - sw * 0.5;
    let dy = (ly - hy).abs() - hy;
    Vec2::new(dx.max(0.0), dy.max(0.0)).length() + dx.max(dy).min(0.0)
}

/// SDF for a staircase shape, extruded along Z-axis
///
/// The staircase has n_steps, each step_width wide and step_height tall.
/// Step i has height (i+1)*step_height. Centered at origin.
/// - `step_width`: width of each step
/// - `step_height`: height of each step
/// - `n_steps`: number of steps (as f32)
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_stairs(
    p: Vec3,
    step_width: f32,
    step_height: f32,
    n_steps: f32,
    half_depth: f32,
) -> f32 {
    let sw = step_width;
    let sh = step_height;
    let n = n_steps.max(1.0);
    let tw = n * sw;
    let th = n * sh;

    // Center coordinates: local x in [0, tw], y in [0, th]
    let lx = p.x + tw * 0.5;
    let ly = p.y + th * 0.5;

    // Find candidate step indices from x and y
    let si = (lx / sw).floor().clamp(0.0, n - 1.0);
    let sj = ((ly / sh).ceil() - 1.0).clamp(0.0, n - 1.0);

    // Check the nearest candidate steps (si-1, si, si+1, sj)
    let mut d_2d = step_box(lx, ly, si, sw, sh);

    if si > 0.0 {
        d_2d = d_2d.min(step_box(lx, ly, si - 1.0, sw, sh));
    }
    if si < n - 1.0 {
        d_2d = d_2d.min(step_box(lx, ly, si + 1.0, sw, sh));
    }
    if sj != si && sj != si - 1.0 && sj != si + 1.0 {
        d_2d = d_2d.min(step_box(lx, ly, sj, sw, sh));
    }

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.max(0.0), d_z.max(0.0));
    d_2d.max(d_z).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stairs_center_inside() {
        // Point clearly inside a step (offset from step edges)
        let d = sdf_stairs(Vec3::new(0.1, -0.3, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d < 0.0, "Point inside step should be negative, got {}", d);
    }

    #[test]
    fn test_stairs_far_outside() {
        let d = sdf_stairs(Vec3::new(5.0, 0.0, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_stairs_symmetry_z() {
        let d1 = sdf_stairs(Vec3::new(0.1, 0.1, 0.2), 0.5, 0.5, 4.0, 0.5);
        let d2 = sdf_stairs(Vec3::new(0.1, 0.1, -0.2), 0.5, 0.5, 4.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }

    #[test]
    fn test_stairs_step_shape() {
        // Point on the first step (low-left) should be inside
        let d = sdf_stairs(Vec3::new(-0.8, -0.8, 0.0), 0.5, 0.5, 4.0, 0.5);
        assert!(d < 0.0, "Point on first step should be inside, got {}", d);
    }
}

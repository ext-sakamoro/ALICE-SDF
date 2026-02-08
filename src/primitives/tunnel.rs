//! Tunnel SDF (Deep Fried Edition)
//!
//! D-shaped arch cross-section in XY plane (rectangle + semicircle on top),
//! extruded along Z-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a tunnel shape
///
/// A D-shaped arch: rectangle (|x| <= w, -h <= y <= h) with a semicircular
/// dome (radius w, centered at y=h) on top. Extruded along Z.
/// - `width`: half-width of the tunnel (also dome radius)
/// - `height_2d`: half-height of the rectangular part
/// - `half_depth`: half the extrusion depth along Z
#[inline(always)]
pub fn sdf_tunnel(p: Vec3, width: f32, height_2d: f32, half_depth: f32) -> f32 {
    // 2D: union of rectangle and half-circle
    let px = p.x.abs();
    let py = p.y;

    // SDF of rectangle: |x| <= width, -height_2d <= y <= height_2d
    let dx = px - width;
    let dy_rect = py.abs() - height_2d;
    let d_rect = Vec2::new(dx.max(0.0), dy_rect.max(0.0)).length() + dx.max(dy_rect).min(0.0);

    // SDF of semicircle dome: center at (0, height_2d), radius = width, y >= height_2d
    let d_circle = Vec2::new(px, py - height_2d).length() - width;

    // Union (min) for the top dome region only
    let d_2d = if py > height_2d {
        d_rect.min(d_circle)
    } else {
        d_rect
    };

    // Extrude along Z
    let d_z = p.z.abs() - half_depth;
    let w = Vec2::new(d_2d.max(0.0), d_z.max(0.0));
    d_2d.max(d_z).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tunnel_center_inside() {
        let d = sdf_tunnel(Vec3::ZERO, 1.0, 1.0, 1.0);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_tunnel_far_outside() {
        let d = sdf_tunnel(Vec3::new(5.0, 0.0, 0.0), 1.0, 1.0, 1.0);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_tunnel_symmetry_x() {
        let d1 = sdf_tunnel(Vec3::new(0.3, 0.1, 0.2), 1.0, 1.0, 1.0);
        let d2 = sdf_tunnel(Vec3::new(-0.3, 0.1, 0.2), 1.0, 1.0, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_tunnel_symmetry_z() {
        let d1 = sdf_tunnel(Vec3::new(0.3, 0.1, 0.2), 1.0, 1.0, 1.0);
        let d2 = sdf_tunnel(Vec3::new(0.3, 0.1, -0.2), 1.0, 1.0, 1.0);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Z");
    }
}

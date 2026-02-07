//! Star polygon SDF (Deep Fried Edition)
//!
//! Star polygon in XZ plane, extruded along Y-axis.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a star polygon, extruded along Y-axis
///
/// - `radius`: outer vertex radius
/// - `n_points`: number of star points (as f32, truncated)
/// - `m`: inner vertex radius (spike depth)
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_star_polygon(p: Vec3, radius: f32, n_points: f32, m: f32, half_height: f32) -> f32 {
    let qx = p.x.abs();
    let qz = p.z;
    let n = n_points.max(3.0);
    let an = std::f32::consts::PI / n;

    // Fold to one half-sector [0, an]
    let r = Vec2::new(qx, qz).length();
    let mut angle = qx.atan2(qz);
    // Modulo to [0, 2*an], then reflect to [0, an]
    angle = ((angle % (2.0 * an)) + 2.0 * an) % (2.0 * an);
    if angle > an {
        angle = 2.0 * an - angle;
    }

    // Point in folded polar-to-Cartesian
    let pt = Vec2::new(r * angle.cos(), r * angle.sin());

    // Edge from outer vertex A=(radius, 0) to inner vertex B=(m*cos(an), m*sin(an))
    let a = Vec2::new(radius, 0.0);
    let b = Vec2::new(m * an.cos(), m * an.sin());
    let ab = b - a;
    let ap = pt - a;

    // Closest point on segment AB
    let t = (ap.dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
    let closest = a + ab * t;
    let dist = (pt - closest).length();

    // Sign via cross product: AB × AP
    let cross = ab.x * ap.y - ab.y * ap.x;
    let d_2d = if cross > 0.0 { -dist } else { dist };

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_polygon_center_inside() {
        let d = sdf_star_polygon(Vec3::ZERO, 1.0, 5.0, 0.4, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_star_polygon_far_outside() {
        let d = sdf_star_polygon(Vec3::new(5.0, 0.0, 0.0), 1.0, 5.0, 0.4, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_star_polygon_symmetry_x() {
        let d1 = sdf_star_polygon(Vec3::new(0.2, 0.1, 0.5), 1.0, 5.0, 0.4, 0.5);
        let d2 = sdf_star_polygon(Vec3::new(-0.2, 0.1, 0.5), 1.0, 5.0, 0.4, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }

    #[test]
    fn test_star_polygon_symmetry_y() {
        let d1 = sdf_star_polygon(Vec3::new(0.1, 0.2, 0.3), 1.0, 5.0, 0.4, 0.5);
        let d2 = sdf_star_polygon(Vec3::new(0.1, -0.2, 0.3), 1.0, 5.0, 0.4, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }
}

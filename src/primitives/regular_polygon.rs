//! Regular polygon prism SDF (Deep Fried Edition)
//!
//! Regular N-sided polygon in XZ plane, extruded along Y-axis.
//!
//! Based on Inigo Quilez's sdPolygon formula.
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// SDF for a regular N-sided polygon, extruded along Y-axis
///
/// - `radius`: circumscribed circle radius (center to vertex)
/// - `n_sides`: number of sides (as f32, truncated to integer)
/// - `half_height`: half the extrusion height along Y
#[inline(always)]
pub fn sdf_regular_polygon(p: Vec3, radius: f32, n_sides: f32, half_height: f32) -> f32 {
    // Inigo Quilez "Regular Polygon" (exact), in the XZ plane with `radius` as the
    // circumradius. Before 1.9.2 the sector fold only handled z >= 0 and the 2D
    // term was a half-plane distance, which made the shape unbounded.
    let n = n_sides.max(3.0).trunc();
    let an = std::f32::consts::PI / n;
    let (acs_s, acs_c) = alice_det_math::sin_cos(an);

    // Reduce to the first sector: angle in [-an, an)
    let bn = (alice_det_math::atan2(p.x, p.z)).rem_euclid(2.0 * an) - an;
    let r = alice_det_math::hypot(p.x, p.z);
    let (bs, bc) = alice_det_math::sin_cos(bn);
    let mut q = Vec2::new(r * bc, (r * bs).abs());

    // Distance to the edge line through the vertex at angle `an`
    q -= Vec2::new(radius * acs_c, radius * acs_s);
    q.y += (-q.y).clamp(0.0, radius * acs_s);
    let d_2d = q.length() * q.x.signum();

    // Extrude along Y
    let d_y = p.y.abs() - half_height;
    let w = Vec2::new(d_2d.max(0.0), d_y.max(0.0));
    d_2d.max(d_y).min(0.0) + w.length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regular_polygon_center_inside() {
        let d = sdf_regular_polygon(Vec3::ZERO, 1.0, 6.0, 0.5);
        assert!(d < 0.0, "Center should be inside, got {}", d);
    }

    #[test]
    fn test_regular_polygon_far_outside() {
        let d = sdf_regular_polygon(Vec3::new(5.0, 0.0, 0.0), 1.0, 6.0, 0.5);
        assert!(d > 0.0, "Far point should be outside, got {}", d);
    }

    #[test]
    fn test_regular_polygon_bounded_in_every_sector() {
        // Regression: the pre-1.9.2 sector fold left z < 0 unbounded
        for (x, z) in [
            (-3.0, -3.0),
            (0.0, -3.0),
            (3.0, -3.0),
            (-3.0, 3.0),
            (0.0, 3.0),
        ] {
            let d = sdf_regular_polygon(Vec3::new(x, 0.0, z), 1.0, 6.0, 0.5);
            assert!(d > 1.5, "({x}, {z}) must be far outside, got {d}");
        }
    }

    #[test]
    fn test_regular_polygon_vertex_and_apothem() {
        // Hexagon circumradius 1: vertex on surface, apothem cos(30°) on surface
        let d_v = sdf_regular_polygon(Vec3::new(0.0, 0.0, 1.0), 1.0, 6.0, 0.5);
        assert!(d_v.abs() < 1e-4, "vertex should be on surface, got {d_v}");
        let ap = (std::f32::consts::PI / 6.0).cos();
        let d_a = sdf_regular_polygon(
            Vec3::new(
                ap * (std::f32::consts::PI / 6.0).sin(),
                0.0,
                ap * (std::f32::consts::PI / 6.0).cos(),
            ),
            1.0,
            6.0,
            0.5,
        );
        assert!(
            d_a.abs() < 1e-4,
            "apothem point should be on surface, got {d_a}"
        );
    }

    #[test]
    fn test_regular_polygon_symmetry_y() {
        let d1 = sdf_regular_polygon(Vec3::new(0.2, 0.2, 0.3), 1.0, 6.0, 0.5);
        let d2 = sdf_regular_polygon(Vec3::new(0.2, -0.2, 0.3), 1.0, 6.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in Y");
    }

    #[test]
    fn test_regular_polygon_symmetry_x() {
        let d1 = sdf_regular_polygon(Vec3::new(0.3, 0.1, 0.2), 1.0, 6.0, 0.5);
        let d2 = sdf_regular_polygon(Vec3::new(-0.3, 0.1, 0.2), 1.0, 6.0, 0.5);
        assert!((d1 - d2).abs() < 0.001, "Should be symmetric in X");
    }
}

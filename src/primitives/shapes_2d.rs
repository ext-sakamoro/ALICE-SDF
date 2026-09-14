//! 2D primitives extruded along Z (single source of truth)
//!
//! Tree evaluator, compiled scalar / BVH evaluator and SIMD per-lane path all
//! call these functions so the four evaluation paths cannot drift.
//!
//! Every shape computes a 2D signed distance in the XY plane and combines it
//! with the Z extrusion via [`extrude_2d`].
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

/// Combine a 2D distance with a Z half-height extrusion.
#[inline(always)]
pub fn extrude_2d(d2d: f32, dz: f32) -> f32 {
    d2d.max(dz).min(0.0) + Vec2::new(d2d.max(0.0), dz.max(0.0)).length()
}

/// Circle of `radius` in XY, extruded `half_height` along Z.
#[inline(always)]
pub fn sdf_circle_2d(p: Vec3, radius: f32, half_height: f32) -> f32 {
    let d2d = Vec2::new(p.x, p.y).length() - radius;
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Axis-aligned rectangle with `half_extents` in XY, extruded along Z.
#[inline(always)]
pub fn sdf_rect_2d(p: Vec3, half_extents: Vec2, half_height: f32) -> f32 {
    let d = Vec2::new(p.x.abs() - half_extents.x, p.y.abs() - half_extents.y);
    let d2d = Vec2::new(d.x.max(0.0), d.y.max(0.0)).length() + d.x.max(d.y).min(0.0);
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Rectangle with rounded corners (`round_radius`), extruded along Z.
#[inline(always)]
pub fn sdf_rounded_rect_2d(
    p: Vec3,
    half_extents: Vec2,
    round_radius: f32,
    half_height: f32,
) -> f32 {
    let d = Vec2::new(
        p.x.abs() - half_extents.x + round_radius,
        p.y.abs() - half_extents.y + round_radius,
    );
    let d2d = Vec2::new(d.x.max(0.0), d.y.max(0.0)).length() + d.x.max(d.y).min(0.0) - round_radius;
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Line segment `a`–`b` in XY with `thickness`, extruded along Z.
#[inline(always)]
pub fn sdf_segment_2d(p: Vec3, a: Vec2, b: Vec2, thickness: f32, half_height: f32) -> f32 {
    let p2 = Vec2::new(p.x, p.y);
    let pa = p2 - a;
    let ba = b - a;
    let h = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    let d2d = (pa - ba * h).length() - thickness;
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Annulus (ring) with `outer_radius` and radial `thickness`, extruded along Z.
#[inline(always)]
pub fn sdf_annular_2d(p: Vec3, outer_radius: f32, thickness: f32, half_height: f32) -> f32 {
    let d2d = (Vec2::new(p.x, p.y).length() - outer_radius).abs() - thickness;
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Signed distance to a closed 2D polygon in the XY plane (no extrusion).
///
/// Returns `1e10` for degenerate polygons (< 3 vertices), matching the
/// tree evaluator's historical behaviour.
#[inline]
pub fn sdf_polygon_2d_xy(p2: Vec2, vertices: &[Vec2]) -> f32 {
    let n = vertices.len();
    if n < 3 {
        return 1e10;
    }
    let mut d = (p2 - vertices[0]).length_squared();
    let mut s = 1.0_f32;
    let mut j = n - 1;
    for i in 0..n {
        let e = vertices[j] - vertices[i];
        let w = p2 - vertices[i];
        let b_proj = w - e * (w.dot(e) / e.dot(e)).clamp(0.0, 1.0);
        d = d.min(b_proj.dot(b_proj));
        let c = [
            p2.y >= vertices[i].y,
            p2.y < vertices[j].y,
            e.x * w.y > e.y * w.x,
        ];
        if c.iter().all(|x| *x) || c.iter().all(|x| !*x) {
            s = -s;
        }
        j = i;
    }
    s * d.sqrt()
}

/// Closed polygon in XY, extruded along Z.
#[inline]
pub fn sdf_polygon_2d(p: Vec3, vertices: &[Vec2], half_height: f32) -> f32 {
    let d2d = sdf_polygon_2d_xy(Vec2::new(p.x, p.y), vertices);
    extrude_2d(d2d, p.z.abs() - half_height)
}

/// Closed polygon in XY from a flat `[x0, y0, x1, y1, ...]` slice, extruded along Z.
///
/// Used by the compiled evaluators, which store vertices in `aux_data`.
#[inline]
pub fn sdf_polygon_2d_flat(p: Vec3, flat: &[f32], half_height: f32) -> f32 {
    let n = flat.len() / 2;
    if n < 3 {
        return extrude_2d(1e10, p.z.abs() - half_height);
    }
    let p2 = Vec2::new(p.x, p.y);
    let v = |i: usize| Vec2::new(flat[i * 2], flat[i * 2 + 1]);
    let mut d = (p2 - v(0)).length_squared();
    let mut s = 1.0_f32;
    let mut j = n - 1;
    for i in 0..n {
        let vi = v(i);
        let vj = v(j);
        let e = vj - vi;
        let w = p2 - vi;
        let b_proj = w - e * (w.dot(e) / e.dot(e)).clamp(0.0, 1.0);
        d = d.min(b_proj.dot(b_proj));
        let c = [p2.y >= vi.y, p2.y < vj.y, e.x * w.y > e.y * w.x];
        if c.iter().all(|x| *x) || c.iter().all(|x| !*x) {
            s = -s;
        }
        j = i;
    }
    extrude_2d(s * d.sqrt(), p.z.abs() - half_height)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_2d_matches_analytic() {
        // Inside the disc on the mid-plane: distance is the larger of the two inside distances
        assert!((sdf_circle_2d(Vec3::ZERO, 0.5, 1.0) + 0.5).abs() < 1e-6);
        // Outside radially, inside Z
        assert!((sdf_circle_2d(Vec3::new(1.5, 0.0, 0.0), 0.5, 1.0) - 1.0).abs() < 1e-6);
        // Outside along Z only
        assert!((sdf_circle_2d(Vec3::new(0.0, 0.0, 2.0), 0.5, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn polygon_flat_matches_vec2() {
        let verts = [
            Vec2::new(-1.0, -1.0),
            Vec2::new(1.0, -1.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(-1.0, 1.0),
        ];
        let flat: Vec<f32> = verts.iter().flat_map(|v| [v.x, v.y]).collect();
        for p in [
            Vec3::ZERO,
            Vec3::new(0.5, 0.25, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.5),
            Vec3::new(-3.0, 3.0, -0.2),
        ] {
            let a = sdf_polygon_2d(p, &verts, 0.5);
            let b = sdf_polygon_2d_flat(p, &flat, 0.5);
            assert!((a - b).abs() < 1e-6, "{p:?}: {a} vs {b}");
        }
        // Square of half-extent 1 behaves like rect_2d
        let r = sdf_rect_2d(Vec3::new(0.5, 0.25, 0.0), Vec2::splat(1.0), 0.5);
        assert!((sdf_polygon_2d(Vec3::new(0.5, 0.25, 0.0), &verts, 0.5) - r).abs() < 1e-6);
    }

    #[test]
    fn degenerate_polygon_is_far() {
        assert!(sdf_polygon_2d(Vec3::ZERO, &[Vec2::ZERO, Vec2::X], 0.5) > 1e9);
        assert!(sdf_polygon_2d_flat(Vec3::ZERO, &[0.0, 0.0, 1.0, 0.0], 0.5) > 1e9);
    }
}

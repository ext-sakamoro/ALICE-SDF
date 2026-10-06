//! Axis-aligned and point-defined primitives vs closed forms computed in `f64`.
//!
//! The functions checked here (`sdf_capsule_vertical`, `sdf_capsule_horizontal`,
//! `sdf_cylinder_capped`, `sdf_cylinder_infinite`, the three coordinate planes,
//! `sdf_plane_from_points`, `sdf_torus_capped`) are specialised forms of the
//! node primitives. The references below do not call the crate:
//!
//! * capsules: distance to a segment minus the radius,
//! * capped cylinder: the 2D box law in (radial, axial) coordinates of the
//!   segment `a b`, which is exact for a solid cylinder,
//! * plane through three points: `(p - a) . n` with `n = (b - a) x (c - a) / |..|`
//!   (right-handed),
//! * capped torus: brute-force minimum over the arc `ra (sin t, 0, cos t)`,
//!   `|t| <= angle`, sampled finely enough that the sampling error is below
//!   `1e-8`, minus the tube radius,
//! * `SdfNode::box3d_half_extents`: the exact box distance,
//!   `|max(q, 0)| + min(max(q.x, q.y, q.z), 0)` with `q = |p| - h`.
//!
//! Every test counts its comparisons and fails if none were made.
//!
//! Author: Moroya Sakamoto
#![allow(clippy::float_cmp)]

use alice_sdf::primitives::{
    sdf_capsule_horizontal, sdf_capsule_vertical, sdf_cylinder_capped, sdf_cylinder_infinite,
    sdf_plane_from_points, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz, sdf_torus_capped,
};
use alice_sdf::types::SdfCategory;
use alice_sdf::{eval, SdfNode};
use glam::Vec3;

type V = [f64; 3];

fn sub(a: V, b: V) -> V {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn len(a: V) -> f64 {
    dot(a, a).sqrt()
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
const fn v64(p: Vec3) -> V {
    [p.x as f64, p.y as f64, p.z as f64]
}

/// Distance from `p` to the segment `a b`.
fn seg_dist(p: V, a: V, b: V) -> f64 {
    let ab = sub(b, a);
    let l2 = dot(ab, ab);
    // a zero-length segment is the point `a`
    let t = if l2 == 0.0 {
        0.0
    } else {
        (dot(sub(p, a), ab) / l2).clamp(0.0, 1.0)
    };
    len(sub(
        p,
        [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t],
    ))
}

/// Exact signed distance to the solid cylinder of radius `r` around segment `a b`.
fn capped_cylinder(p: V, a: V, b: V, r: f64) -> f64 {
    let ab = sub(b, a);
    let l = len(ab);
    let u = [ab[0] / l, ab[1] / l, ab[2] / l];
    let ap = sub(p, a);
    let t = dot(ap, u);
    let radial = len(sub(ap, [u[0] * t, u[1] * t, u[2] * t]));
    let dx = radial - r;
    let dy = (t - l * 0.5).abs() - l * 0.5;
    dx.max(dy).min(0.0) + (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt()
}

fn points() -> Vec<Vec3> {
    let mut out = Vec::new();
    for ix in -6..=6 {
        for iy in -6..=6 {
            for iz in -6..=6 {
                // irrational-ish offsets keep the grid off the axes and the caps
                out.push(Vec3::new(
                    ix as f32 * 0.37 + 0.013,
                    iy as f32 * 0.41 - 0.021,
                    iz as f32 * 0.29 + 0.007,
                ));
            }
        }
    }
    out
}

fn assert_close(what: &str, p: Vec3, got: f32, want: f64, tol: f64) {
    assert!(
        (got as f64 - want).abs() <= tol * (1.0 + want.abs()),
        "{what} at {p:?}: got {got}, closed form {want}"
    );
}

#[test]
fn axis_capsules_are_segment_distance_minus_radius() {
    let mut n = 0;
    for &(h, r) in &[(0.8_f32, 0.3_f32), (1.5, 0.6), (0.0, 0.5)] {
        for p in points() {
            let hv = h as f64;
            let want_v = seg_dist(v64(p), [0.0, -hv, 0.0], [0.0, hv, 0.0]) - r as f64;
            assert_close(
                "sdf_capsule_vertical",
                p,
                sdf_capsule_vertical(p, h, r),
                want_v,
                1e-5,
            );
            let want_h = seg_dist(v64(p), [-hv, 0.0, 0.0], [hv, 0.0, 0.0]) - r as f64;
            assert_close(
                "sdf_capsule_horizontal",
                p,
                sdf_capsule_horizontal(p, h, r),
                want_h,
                1e-5,
            );
            n += 2;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn capped_cylinder_between_points_is_the_radial_axial_box_law() {
    let cases = [
        (Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.5_f32),
        (Vec3::new(-0.7, 0.2, 0.3), Vec3::new(0.9, -0.4, 0.8), 0.35),
        (Vec3::new(0.1, 0.1, -1.2), Vec3::new(0.1, 0.1, 1.2), 0.8),
    ];
    let mut n = 0;
    for &(a, b, r) in &cases {
        for p in points() {
            let want = capped_cylinder(v64(p), v64(a), v64(b), r as f64);
            assert_close(
                "sdf_cylinder_capped",
                p,
                sdf_cylinder_capped(p, a, b, r),
                want,
                2e-5,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn infinite_cylinder_is_radial_distance_minus_radius() {
    let mut n = 0;
    for &r in &[0.25_f32, 1.0, 1.7] {
        for p in points() {
            let want = (p.x as f64).hypot(p.z as f64) - r as f64;
            assert_close(
                "sdf_cylinder_infinite",
                p,
                sdf_cylinder_infinite(p, r),
                want,
                1e-6,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn coordinate_planes_and_plane_through_three_points() {
    let tri = [
        (
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ),
        (
            Vec3::new(0.3, -0.2, 0.5),
            Vec3::new(1.1, 0.4, -0.3),
            Vec3::new(-0.6, 0.9, 0.2),
        ),
    ];
    let mut n = 0;
    for p in points() {
        assert_eq!(sdf_plane_xy(p), p.z);
        assert_eq!(sdf_plane_xz(p), p.y);
        assert_eq!(sdf_plane_yz(p), p.x);
        n += 3;
        for &(a, b, c) in &tri {
            let nrm = cross(sub(v64(b), v64(a)), sub(v64(c), v64(a)));
            let l = len(nrm);
            let want = dot(sub(v64(p), v64(a)), [nrm[0] / l, nrm[1] / l, nrm[2] / l]);
            assert_close(
                "sdf_plane_from_points",
                p,
                sdf_plane_from_points(p, a, b, c),
                want,
                1e-5,
            );
            n += 1;
        }
    }
    // the three defining points lie on the plane
    for &(a, b, c) in &tri {
        for q in [a, b, c] {
            assert!(sdf_plane_from_points(q, a, b, c).abs() < 1e-6);
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn capped_torus_is_distance_to_the_arc_minus_the_tube_radius() {
    const SAMPLES: usize = 40_000;
    let mut n = 0;
    for &(ra, rb, angle) in &[
        (1.0_f32, 0.25_f32, 0.9_f32),
        (1.4, 0.4, 2.2),
        (0.8, 0.2, std::f32::consts::PI),
    ] {
        let a = angle as f64;
        let arc: Vec<V> = (0..=SAMPLES)
            .map(|i| {
                let t = -a + 2.0 * a * i as f64 / SAMPLES as f64;
                [ra as f64 * t.sin(), 0.0, ra as f64 * t.cos()]
            })
            .collect();
        for p in points().into_iter().step_by(7) {
            let q = v64(p);
            let near = arc
                .iter()
                .map(|&c| len(sub(q, c)))
                .fold(f64::INFINITY, f64::min);
            let want = near - rb as f64;
            assert_close(
                "sdf_torus_capped",
                p,
                sdf_torus_capped(p, ra, rb, angle),
                want,
                1e-5,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn box3d_half_extents_node_is_the_exact_box_distance() {
    let mut n = 0;
    for &(hx, hy, hz) in &[(0.5_f32, 0.5_f32, 0.5_f32), (1.2, 0.3, 0.7)] {
        let node = SdfNode::box3d_half_extents(hx, hy, hz);
        for p in points() {
            let q = [
                (p.x as f64).abs() - hx as f64,
                (p.y as f64).abs() - hy as f64,
                (p.z as f64).abs() - hz as f64,
            ];
            let outside = len([q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)]);
            let inside = q[0].max(q[1]).max(q[2]).min(0.0);
            assert_close(
                "box3d_half_extents",
                p,
                eval(&node, p),
                outside + inside,
                1e-5,
            );
            n += 1;
        }
    }
    assert!(n > 0, "no comparisons made");
}

#[test]
fn node_category_follows_the_variant_family() {
    let s = SdfNode::sphere(1.0);
    let cases = [
        (
            SdfNode::box3d_half_extents(1.0, 1.0, 1.0),
            SdfCategory::Primitive,
        ),
        (s.clone(), SdfCategory::Primitive),
        (
            s.clone().union(SdfNode::sphere(0.5)),
            SdfCategory::Operation,
        ),
        (s.clone().translate(1.0, 0.0, 0.0), SdfCategory::Transform),
        (s.twist(1.0), SdfCategory::Modifier),
    ];
    for (node, want) in &cases {
        assert_eq!(node.category(), *want, "{node:?}");
    }
    assert!(!cases.is_empty());
}

//! 2D SDF oracle: closed-form distances computed here, independently.
//!
//! - circle `|p − c| − r`, ring `||p| − R| − t`, segment capsule
//!   `|p − a − clamp(t)·(b − a)| − th`
//! - rectangle / rounded rectangle: the exact box SDF (outside length +
//!   inside max), rounded by `r`
//! - regular polygon: the exact convex-polygon SDF from its vertices
//!   (interior = max of edge half-plane distances, exterior = nearest edge)
//! - star / ellipse are documented as approximations, so only their exact
//!   points are checked: tips, notches and axis vertices lie on the zero set,
//!   with the right sign inside and outside
//! - CSG: `min`, `max(a, −b)`, `max`, polynomial smooth min
//!   `min(a, b) − h²k/4, h = max(k − |a − b|, 0)/k`
//! - transforms: a translated / rotated / scaled shape equals the shape
//!   built directly at that place / orientation / size
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::sdf2d::{eval_2d, eval_2d_batch, eval_2d_normal, Sdf2dNode};

fn grid() -> Vec<[f32; 2]> {
    let mut v = Vec::new();
    for i in 0..21 {
        for j in 0..21 {
            v.push([i as f32 * 0.2 - 2.0 + 0.013, j as f32 * 0.2 - 2.0 - 0.007]);
        }
    }
    v
}

fn len(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

fn box_sdf(p: [f32; 2], hw: f32, hh: f32) -> f32 {
    let (dx, dy) = (p[0].abs() - hw, p[1].abs() - hh);
    len(dx.max(0.0), dy.max(0.0)) + dx.max(dy).min(0.0)
}

fn segment_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (bx, by) = (b[0] - a[0], b[1] - a[1]);
    let (px, py) = (p[0] - a[0], p[1] - a[1]);
    let (dot, len2) = (px * bx + py * by, bx * bx + by * by);
    let t = (dot / len2).clamp(0.0, 1.0);
    len(px - t * bx, py - t * by)
}

fn check(node: &Sdf2dNode, oracle: impl Fn([f32; 2]) -> f32, tol: f32, name: &str) -> usize {
    let mut n = 0;
    for p in grid() {
        let (got, expected) = (eval_2d(node, p), oracle(p));
        assert!(
            (got - expected).abs() <= tol,
            "{name} at {p:?}: {got} vs {expected}"
        );
        n += 1;
    }
    n
}

#[test]
fn primitives_match_closed_forms() {
    let mut n = 0;
    n += check(
        &Sdf2dNode::circle(0.8),
        |p| len(p[0], p[1]) - 0.8,
        1e-5,
        "circle",
    );
    n += check(
        &Sdf2dNode::rect(1.0, 0.5),
        |p| box_sdf(p, 1.0, 0.5),
        1e-5,
        "rect",
    );
    n += check(
        &Sdf2dNode::rounded_rect(1.0, 0.6, 0.25),
        |p| box_sdf(p, 0.75, 0.35) - 0.25,
        1e-5,
        "rounded_rect",
    );
    let (a, b) = ([-1.0, -0.5], [1.2, 0.7]);
    n += check(
        &Sdf2dNode::line(a, b, 0.1),
        |p| segment_dist(p, a, b) - 0.1,
        1e-5,
        "line",
    );
    n += check(
        &Sdf2dNode::ring(1.0, 0.2),
        |p| (len(p[0], p[1]) - 1.0).abs() - 0.2,
        1e-5,
        "ring",
    );
    assert!(n > 0);
}

/// Exact SDF of a convex polygon given counter-clockwise vertices.
fn convex_polygon(p: [f32; 2], verts: &[[f32; 2]]) -> f32 {
    let n = verts.len();
    let mut inside = true;
    let mut max_plane = f32::MIN;
    let mut min_edge = f32::MAX;
    for i in 0..n {
        let (a, b) = (verts[i], verts[(i + 1) % n]);
        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
        let l = len(ex, ey);
        // outward normal of a CCW edge is (ey, −ex)/l
        let plane = ((p[0] - a[0]) * ey - (p[1] - a[1]) * ex) / l;
        if plane > 0.0 {
            inside = false;
        }
        max_plane = max_plane.max(plane);
        min_edge = min_edge.min(segment_dist(p, a, b));
    }
    if inside {
        max_plane
    } else {
        min_edge
    }
}

#[test]
fn regular_polygon_is_the_exact_convex_polygon_sdf() {
    let mut n = 0;
    for &(sides, r) in &[(3_u32, 1.2_f32), (5, 1.0), (6, 1.1), (8, 0.9)] {
        // vertices at angles k·2π/n (a vertex on +X)
        let verts: Vec<[f32; 2]> = (0..sides)
            .map(|k| {
                let a = k as f32 * std::f32::consts::TAU / sides as f32;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        n += check(
            &Sdf2dNode::regular_polygon(r, sides),
            |p| convex_polygon(p, &verts),
            2e-5,
            &format!("polygon n={sides}"),
        );
    }
    assert!(n > 0);
}

#[test]
fn star_and_ellipse_exact_points() {
    let (outer, inner, k) = (1.0_f32, 0.4_f32, 5_u32);
    let star = Sdf2dNode::star(outer, inner, k);
    let s = std::f32::consts::PI / k as f32;
    let mut n = 0;
    for i in 0..k {
        let notch = 2.0 * s * i as f32;
        let tip = notch + s;
        assert!(eval_2d(&star, [inner * notch.cos(), inner * notch.sin()]).abs() < 1e-5);
        assert!(eval_2d(&star, [outer * tip.cos(), outer * tip.sin()]).abs() < 1e-5);
        // just inside the notch / outside the tip
        assert!(
            eval_2d(
                &star,
                [0.9 * inner * notch.cos(), 0.9 * inner * notch.sin()]
            ) < 0.0
        );
        assert!(eval_2d(&star, [1.1 * outer * tip.cos(), 1.1 * outer * tip.sin()]) > 0.0);
        n += 4;
    }
    assert!(eval_2d(&star, [0.0, 0.0]) < 0.0);

    let (a, b) = (1.5_f32, 0.5_f32);
    let e = Sdf2dNode::ellipse(a, b);
    for v in [[a, 0.0], [-a, 0.0], [0.0, b], [0.0, -b]] {
        assert!(eval_2d(&e, v).abs() < 1e-5, "{v:?}");
        n += 1;
    }
    for p in grid() {
        let f = (p[0] / a).powi(2) + (p[1] / b).powi(2);
        if (f - 1.0).abs() > 1e-3 {
            assert_eq!(eval_2d(&e, p) < 0.0, f < 1.0, "ellipse sign at {p:?}");
            n += 1;
        }
    }
    assert!(n > 0);
}

#[test]
fn csg_and_onion() {
    let a = || Sdf2dNode::circle(1.0);
    let b = || Sdf2dNode::rect(0.5, 1.2).translate(0.8, 0.0);
    let da = |p: [f32; 2]| len(p[0], p[1]) - 1.0;
    let db = |p: [f32; 2]| box_sdf([p[0] - 0.8, p[1]], 0.5, 1.2);
    let mut n = 0;
    n += check(&a().union(b()), |p| da(p).min(db(p)), 1e-5, "union");
    n += check(&a().subtract(b()), |p| da(p).max(-db(p)), 1e-5, "subtract");
    n += check(&a().intersect(b()), |p| da(p).max(db(p)), 1e-5, "intersect");
    let k = 0.3;
    n += check(
        &a().smooth_union(b(), k),
        |p| {
            let (x, y) = (da(p), db(p));
            let h = (k - (x - y).abs()).max(0.0) / k;
            x.min(y) - h * h * k / 4.0
        },
        1e-5,
        "smooth_union",
    );
    n += check(&a().onion(0.1), |p| da(p).abs() - 0.1, 1e-5, "onion");
    assert!(n > 0);
}

#[test]
fn transforms_equal_the_directly_built_shape() {
    let mut n = 0;
    // translate
    let t = Sdf2dNode::circle(0.5).translate(0.7, -0.3);
    let direct = Sdf2dNode::Circle {
        center: [0.7, -0.3],
        radius: 0.5,
    };
    n += check(&t, |p| eval_2d(&direct, p), 1e-5, "translate");
    // rotate a 1 × 0.4 rectangle by 90°: a 0.4 × 1 rectangle
    let r = Sdf2dNode::rect(1.0, 0.4).rotate(std::f32::consts::FRAC_PI_2);
    n += check(&r, |p| box_sdf(p, 0.4, 1.0), 1e-5, "rotate");
    // rotation is counter-clockwise: a circle at (1, 0) rotated by +90° sits at (0, 1)
    let rc = Sdf2dNode::circle(0.3)
        .translate(1.0, 0.0)
        .rotate(std::f32::consts::FRAC_PI_2);
    n += check(&rc, |p| len(p[0], p[1] - 1.0) - 0.3, 1e-5, "rotate ccw");
    // scale a circle by 2.5: radius 0.4 → 1.0
    let s = Sdf2dNode::circle(0.4).scale(2.5);
    n += check(&s, |p| len(p[0], p[1]) - 1.0, 1e-5, "scale");
    assert!(n > 0);
}

#[test]
fn batch_and_normal() {
    let c = Sdf2dNode::circle(1.0).translate(0.25, 0.0);
    let pts = grid();
    let batch = eval_2d_batch(&c, &pts);
    assert_eq!(batch.len(), pts.len());
    let mut n = 0;
    for (p, d) in pts.iter().zip(&batch) {
        assert_eq!(d.to_bits(), eval_2d(&c, *p).to_bits());
        let (x, y) = (p[0] - 0.25, p[1]);
        let r = len(x, y);
        if r > 0.05 {
            let g = eval_2d_normal(&c, *p);
            assert!(
                (g[0] - x / r).abs() < 2e-3 && (g[1] - y / r).abs() < 2e-3,
                "{p:?}: {g:?}"
            );
            n += 1;
        }
    }
    assert!(n > 0);
}

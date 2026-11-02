//! Loft: closed sections swept along a spline axis, the section interpolated between stations.
//!
//! Oracles, each independent of the implementation:
//! - a straight axis with one constant regular n-gon section is a capped prism: distance by the
//!   closed form of the 2D polygon distance extruded between the two end planes (written here);
//! - a straight axis with a constant square section is a box: the box's closed form;
//! - on a curved axis with sections that change size and shape, the field never grows faster
//!   than 1 per unit length (finite differences over random pairs), and `eval_lipschitz` says
//!   so;
//! - the sign: points on the axis inside the section are negative, points far away positive;
//! - every malformed input is refused by name;
//! - the compiled bytecode gives the same value as the tree.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]

use alice_sdf::interval::eval_lipschitz;
use alice_sdf::prelude::*;
use glam::Vec2;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn ngon(n: usize, r: f32) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / n as f32;
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// The 2D signed distance to a convex polygon (counter-clockwise), written here by edges: the
/// largest signed edge-plane distance inside, the distance to the nearest edge segment outside
fn polygon_2d_reference(q: Vec2, v: &[Vec2]) -> f32 {
    let n = v.len();
    let mut inside = true;
    let mut best = f32::INFINITY;
    let mut plane = f32::NEG_INFINITY;
    for i in 0..n {
        let a = v[i];
        let b = v[(i + 1) % n];
        let e = b - a;
        let out = Vec2::new(e.y, -e.x).normalize();
        let s = (q - a).dot(out);
        plane = plane.max(s);
        if s > 0.0 {
            inside = false;
        }
        let t = ((q - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
        best = best.min((q - (a + e * t)).length());
    }
    if inside {
        plane
    } else {
        best
    }
}

/// The capped prism along +y from y = 0 to y = h, section in (x, z) = (u, v): extrusion of the
/// 2D distance between the two end planes
fn prism_reference(p: Vec3, section: &[Vec2], h: f32) -> f32 {
    let d2 = polygon_2d_reference(Vec2::new(p.x, p.z), section);
    let dw = (-p.y).max(p.y - h);
    let o = Vec2::new(d2.max(0.0), dw.max(0.0));
    d2.max(dw).min(0.0) + o.length()
}

/// A straight axis along +y: up = +x, so N = +x and B = T x N = y x x = -z; the reference uses
/// (u, v) = (x, -z)
fn straight(h: f32, section: Vec<Vec2>) -> SdfNode {
    SdfNode::loft(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![0.0, 1.0],
        vec![section.clone(), section],
    )
    .expect("a straight loft")
}

#[test]
fn a_straight_loft_of_a_constant_ngon_is_the_capped_prism() {
    let h = 1.5;
    let sec = ngon(40, 0.3);
    let node = straight(h, sec.clone());
    let mut r = lcg(1);
    let mut worst = 0.0f32;
    for _ in 0..4000 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 2.4 - 0.45, r() * 1.4 - 0.7);
        let want = prism_reference(Vec3::new(p.x, p.y, -p.z), &sec, h);
        worst = worst.max((eval(&node, p) - want).abs());
    }
    assert!(worst < 2e-5, "largest difference {worst}");
}

#[test]
fn a_straight_loft_of_a_constant_square_is_the_box() {
    let h = 2.0;
    let a = 0.25;
    let sq = vec![Vec2::new(-a, -a), Vec2::new(a, -a), Vec2::new(a, a), Vec2::new(-a, a)];
    let node = straight(h, sq);
    let mut r = lcg(2);
    let mut worst = 0.0f32;
    for _ in 0..4000 {
        let p = Vec3::new(r() * 1.2 - 0.6, r() * 3.0 - 0.5, r() * 1.2 - 0.6);
        // box centred at (0, h/2, 0), half extents (a, h/2, a): closed form
        let q = (p - Vec3::new(0.0, h / 2.0, 0.0)).abs() - Vec3::new(a, h / 2.0, a);
        let want = q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0);
        worst = worst.max((eval(&node, p) - want).abs());
    }
    assert!(worst < 2e-5, "largest difference {worst}");
}

/// A curved axis with sections that grow, shrink and squash: ellipses of 32 vertices
fn curved() -> SdfNode {
    let axis = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.05, 0.3, 0.02),
        Vec3::new(0.12, 0.6, 0.0),
        Vec3::new(0.15, 0.9, -0.05),
    ];
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> {
        (0..32)
            .map(|k| {
                let t = std::f32::consts::TAU * k as f32 / 32.0;
                Vec2::new(a * t.cos(), b * t.sin())
            })
            .collect()
    };
    SdfNode::loft(
        axis,
        Vec3::Z,
        vec![0.0, 1.2, 3.0],
        vec![ellipse(0.12, 0.09), ellipse(0.08, 0.1), ellipse(0.06, 0.05)],
    )
    .expect("a curved loft")
}

#[test]
fn the_field_is_one_lipschitz_on_a_curved_varying_loft() {
    let node = curved();
    let l = eval_lipschitz(&node);
    assert!(l.is_finite() && l <= 1.0, "eval_lipschitz {l}");
    let mut r = lcg(3);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.05;
        let q = p + dir * step;
        worst = worst.max((eval(&node, q) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

#[test]
fn inside_on_the_axis_and_outside_far_away() {
    let node = curved();
    for s in [0.05f32, 0.4, 0.75] {
        // points on the axis polyline's chords stay within the sections for this gentle axis
        let p = Vec3::new(0.15 * s, 0.9 * s, 0.0);
        assert!(eval(&node, p) < 0.0, "axis point {p:?} reads {}", eval(&node, p));
    }
    for p in [Vec3::new(1.0, 0.5, 0.0), Vec3::new(0.0, -0.5, 0.0), Vec3::new(0.0, 1.5, 0.0)] {
        assert!(eval(&node, p) > 0.2, "far point {p:?} reads {}", eval(&node, p));
    }
}

#[test]
fn malformed_input_is_refused_by_name() {
    let sec = ngon(8, 0.2);
    let ax = vec![Vec3::ZERO, Vec3::Y];
    let err = |r: Result<SdfNode, alice_sdf::primitives::loft::LoftError>| r.err().expect("refused").to_string();
    assert!(err(SdfNode::loft(vec![Vec3::ZERO], Vec3::X, vec![0.0], vec![sec.clone()])).contains("axis"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, vec![0.0], vec![sec.clone()])).contains("station"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, vec![0.0, 0.5], vec![sec.clone(), sec.clone()])).contains("station"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, vec![0.0, 1.0], vec![sec.clone(), ngon(9, 0.2)])).contains("vertex count"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::X, vec![0.0, 1.0], vec![ngon(2, 0.2), ngon(2, 0.2)])).contains("vertex count"));
    assert!(err(SdfNode::loft(ax.clone(), Vec3::Y, vec![0.0, 1.0], vec![sec.clone(), sec.clone()])).contains("up"));
    let nan = vec![Vec3::ZERO, Vec3::new(f32::NAN, 1.0, 0.0)];
    assert!(err(SdfNode::loft(nan, Vec3::X, vec![0.0, 1.0], vec![sec.clone(), sec])).contains("finite"));
}

#[test]
fn the_compiled_bytecode_gives_the_trees_value() {
    let node = curved();
    let compiled = alice_sdf::compiled::CompiledSdf::compile(&node);
    let mut r = lcg(5);
    for _ in 0..2000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3);
        let a = eval(&node, p);
        let b = alice_sdf::compiled::eval_compiled(&compiled, p);
        assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: tree {a}, bytecode {b}");
    }
}

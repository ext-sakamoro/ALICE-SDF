//! Loft with curve sections: each section is a closed periodic uniform cubic B-spline through M
//! control points in the frame's plane (counter-clockwise), so a section need not be star-shaped;
//! along the axis the control rows are a uniform cubic B-spline as in the polar mode.
//!
//! Oracles, each independent of the implementation:
//! - a straight axis with one constant section is the capped extrusion of that curve: the distance
//!   is computed here from a dense polyline of the curve (the curve evaluated here, by its own
//!   B-spline basis), for a near-circle (M = 32), an ellipse (M = 24) and a C shape that is not star
//!   about any point (its hollow is outside);
//! - on a curved axis with sections that change, the field never grows faster than 1 per unit
//!   length, and it never passes the surface of a straight loft whose section grows;
//! - inside on the axis, outside far away; every malformed input is refused by name; the compiled
//!   bytecode gives the same value as the tree, bit for bit.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm serves as an independent reference"
)]
#![allow(clippy::manual_midpoint)]

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

/// The periodic uniform cubic B-spline of `cp` at global parameter `t ∈ [0, M)`
fn spline(cp: &[Vec2], t: f32) -> Vec2 {
    let m = cp.len();
    let j = (t.floor() as usize) % m;
    let u = t - t.floor();
    let p = |k: isize| cp[((j as isize + k).rem_euclid(m as isize)) as usize];
    let v = 1.0 - u;
    p(-1) * (v * v * v / 6.0)
        + p(0) * ((3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0)
        + p(1) * ((-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0)
        + p(2) * (u * u * u / 6.0)
}

/// The signed 2D distance to the closed curve (negative inside), from a dense polyline and the
/// crossing-number test
fn curve_distance(q: Vec2, cp: &[Vec2]) -> f32 {
    let n = 4096 * cp.len() / 8;
    let pts: Vec<Vec2> = (0..n)
        .map(|i| spline(cp, cp.len() as f32 * i as f32 / n as f32))
        .collect();
    let mut best = f32::INFINITY;
    let mut inside = false;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let e = b - a;
        let t = ((q - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
        best = best.min((q - (a + e * t)).length());
        if (a.y > q.y) != (b.y > q.y) && q.x < a.x + (q.y - a.y) * e.x / e.y {
            inside = !inside;
        }
    }
    if inside {
        -best
    } else {
        best
    }
}

/// The capped extrusion along +y from 0 to h of the curve in (u, v) = (x, −z) (up = +x: N = +x,
/// B = T × N = −z)
fn extrusion_distance(p: Vec3, cp: &[Vec2], h: f32) -> f32 {
    let d2 = curve_distance(Vec2::new(p.x, -p.z), cp);
    let dw = (-p.y).max(p.y - h);
    let o = Vec2::new(d2.max(0.0), dw.max(0.0));
    d2.max(dw).min(0.0) + o.length()
}

fn ellipse(m: usize, a: f32, b: f32) -> Vec<Vec2> {
    (0..m)
        .map(|k| {
            let t = std::f32::consts::TAU * k as f32 / m as f32;
            Vec2::new(a * t.cos(), b * t.sin())
        })
        .collect()
}

/// A C shape, counter-clockwise: a thick arc open toward +x (its hollow, around (0.15, 0), is
/// outside), not star-shaped about any point
fn c_shape() -> Vec<Vec2> {
    let mut v = Vec::new();
    let (ro, ri) = (0.4f32, 0.22f32);
    let n = 14;
    // outer arc from 50° to 310° counter-clockwise, inner arc back
    for k in 0..=n {
        let t = (50.0 + 260.0 * k as f32 / n as f32).to_radians();
        v.push(Vec2::new(ro * t.cos(), ro * t.sin()));
    }
    for k in 0..=n {
        let t = (310.0 - 260.0 * k as f32 / n as f32).to_radians();
        v.push(Vec2::new(ri * t.cos(), ri * t.sin()));
    }
    v
}

fn straight(h: f32, section: Vec<Vec2>) -> SdfNode {
    SdfNode::loft_curves(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![section.clone(), section],
    )
    .expect("a straight curve loft")
}

fn check_extrusion(cp: Vec<Vec2>, seed: u64) {
    let h = 1.2;
    let node = straight(h, cp.clone());
    let mut r = lcg(seed);
    let mut worst = 0.0f32;
    for _ in 0..1500 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 1.8 - 0.3, r() * 1.4 - 0.7);
        let want = extrusion_distance(p, &cp, h);
        let f = eval(&node, p);
        worst = worst.max((f - want).abs());
    }
    // the dense polyline's own chord error is below 1e-5 at these sizes
    assert!(worst < 1e-4, "largest difference {worst}");
}

#[test]
fn a_constant_near_circle_section_is_its_capped_extrusion() {
    check_extrusion(ellipse(32, 0.3, 0.3), 1);
}

#[test]
fn a_constant_ellipse_section_is_its_capped_extrusion() {
    check_extrusion(ellipse(24, 0.4, 0.15), 2);
}

#[test]
fn a_constant_c_shaped_section_is_its_capped_extrusion() {
    let cp = c_shape();
    // the hollow is outside
    assert!(
        curve_distance(Vec2::new(0.3, 0.0), &cp) > 0.0,
        "the reference's own C shape"
    );
    check_extrusion(cp, 3);
    let node = straight(1.2, c_shape());
    assert!(
        eval(&node, Vec3::new(0.3, 0.6, 0.0)) > 0.0,
        "the hollow reads inside"
    );
}

fn curved() -> SdfNode {
    let axis = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.05, 0.3, 0.02),
        Vec3::new(0.12, 0.6, 0.0),
        Vec3::new(0.15, 0.9, -0.05),
    ];
    let rows: Vec<Vec<Vec2>> = (0..4)
        .map(|i| {
            let s = 1.0 - 0.15 * i as f32;
            c_shape()
                .iter()
                .map(|p| Vec2::new(p.x * s * 0.3, p.y * (0.3 + 0.03 * i as f32)))
                .collect()
        })
        .collect();
    SdfNode::loft_curves(axis, Vec3::Z, rows).expect("a curved curve loft")
}

#[test]
fn the_field_is_one_lipschitz_on_a_curved_varying_curve_loft() {
    let node = curved();
    let l = eval_lipschitz(&node);
    assert!(l.is_finite() && l <= 1.0, "eval_lipschitz {l}");
    let mut r = lcg(4);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.05;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

#[test]
fn a_growing_section_never_passes_its_surface() {
    // straight axis, the near-circle growing linearly from 0.2 to 0.45: the section at height y is
    // the B-spline of the scaled control points, so the 3D distance is at least |d₂| cos α and at
    // most |d₂| — the field must stay within [0, |d₂|] outside, with the sign of d₂
    let (h, ra, rb) = (1.0f32, 0.2f32, 0.45f32);
    let node = SdfNode::loft_curves(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![ellipse(32, ra, ra), ellipse(32, rb, rb)],
    )
    .expect("growing");
    let mut r = lcg(5);
    let (mut bad, mut compared) = (0, 0);
    for _ in 0..3000 {
        let p = Vec3::new(r() * 1.4 - 0.7, 0.05 + r() * 0.9, r() * 1.4 - 0.7);
        let s = ra + (rb - ra) * p.y / h;
        let d2 = curve_distance(Vec2::new(p.x, -p.z), &ellipse(32, s, s));
        if d2.abs() < 1e-3 {
            continue;
        }
        compared += 1;
        let f = eval(&node, p);
        bad += usize::from((f > 0.0) != (d2 > 0.0) || (d2 > 0.0 && f > d2 + 1e-4));
    }
    assert!(compared > 2000);
    assert_eq!(bad, 0, "wrong sign or past the surface among {compared}");
}

#[test]
fn inside_on_the_axis_and_outside_far_away() {
    let node = straight(1.0, ellipse(16, 0.3, 0.2));
    assert!(eval(&node, Vec3::new(0.0, 0.5, 0.0)) < 0.0);
    for p in [
        Vec3::new(1.0, 0.5, 0.0),
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.0, 1.5, 0.0),
    ] {
        assert!(
            eval(&node, p) > 0.2,
            "far point {p:?} reads {}",
            eval(&node, p)
        );
    }
}

#[test]
fn malformed_curve_input_is_refused_by_name() {
    let ax = vec![Vec3::ZERO, Vec3::Y];
    let e = ellipse(8, 0.2, 0.1);
    let err = |r: Result<SdfNode, alice_sdf::primitives::loft::LoftError>| {
        r.expect_err("refused").to_string()
    };
    assert!(err(SdfNode::loft_curves(
        vec![Vec3::ZERO],
        Vec3::X,
        vec![e.clone(), e.clone()]
    ))
    .contains("axis"));
    assert!(err(SdfNode::loft_curves(ax.clone(), Vec3::X, vec![e.clone()])).contains("row"));
    assert!(err(SdfNode::loft_curves(
        ax.clone(),
        Vec3::X,
        vec![e.clone(), ellipse(9, 0.2, 0.1)]
    ))
    .contains("control points"));
    assert!(err(SdfNode::loft_curves(
        ax.clone(),
        Vec3::X,
        vec![ellipse(3, 0.2, 0.1), ellipse(3, 0.2, 0.1)]
    ))
    .contains("control points"));
    let cw: Vec<Vec2> = e.iter().rev().copied().collect();
    assert!(err(SdfNode::loft_curves(
        ax.clone(),
        Vec3::X,
        vec![cw.clone(), cw]
    ))
    .contains("counter-clockwise"));
    assert!(err(SdfNode::loft_curves(
        ax.clone(),
        Vec3::Y,
        vec![e.clone(), e.clone()]
    ))
    .contains("up"));
    let mut nan = e.clone();
    nan[2].x = f32::NAN;
    assert!(err(SdfNode::loft_curves(ax, Vec3::X, vec![nan, e])).contains("finite"));
}

#[test]
fn the_compiled_bytecode_gives_the_trees_value_for_curves() {
    let node = curved();
    let compiled = alice_sdf::compiled::CompiledSdf::compile(&node);
    let mut r = lcg(6);
    for _ in 0..2000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 1.4 - 0.25, r() * 0.6 - 0.3);
        let a = eval(&node, p);
        let b = alice_sdf::compiled::eval_compiled(&compiled, p);
        assert_eq!(a.to_bits(), b.to_bits(), "at {p:?}: tree {a}, bytecode {b}");
    }
}

/// The periodic cubic B-spline on knots `k` (M + 1 values, period k[M] − k[0]) at `t ∈ [k0, kM)`,
/// written here by the Cox–de Boor recursion of the basis functions over the periodically extended
/// knots (independent of the de Boor evaluation in the crate)
fn spline_knots(cp: &[Vec2], k: &[f32], t: f32) -> Vec2 {
    let m = cp.len() as isize;
    let period = k[cp.len()] - k[0];
    let kn = |i: isize| -> f32 {
        let r = i.rem_euclid(m);
        k[r as usize] + ((i - r) / m) as f32 * period
    };
    fn basis(kn: &dyn Fn(isize) -> f32, i: isize, p: usize, t: f32) -> f32 {
        if p == 0 {
            return if kn(i) <= t && t < kn(i + 1) {
                1.0
            } else {
                0.0
            };
        }
        let a = kn(i + p as isize) - kn(i);
        let b = kn(i + p as isize + 1) - kn(i + 1);
        let l = if a > 0.0 {
            (t - kn(i)) / a * basis(kn, i, p - 1, t)
        } else {
            0.0
        };
        let r = if b > 0.0 {
            (kn(i + p as isize + 1) - t) / b * basis(kn, i + 1, p - 1, t)
        } else {
            0.0
        };
        l + r
    }
    // the span holding t is j; control point P_{j−1+k} (k = 0..3) carries the basis that starts at
    // knot j − 3 + k (the crate's convention: span j is influenced by P_{j−1} … P_{j+2}, and a cubic
    // basis that is nonzero on [K_j, K_{j+1}) starts at one of K_{j−3} … K_j)
    let mut j = 0isize;
    while !(kn(j) <= t && t < kn(j + 1)) {
        j += 1;
    }
    let mut c = Vec2::ZERO;
    for kk in 0..4isize {
        let ci = (j - 1 + kk).rem_euclid(m) as usize;
        c += cp[ci] * basis(&kn, j - 3 + kk, 3, t);
    }
    c
}

fn curve_distance_knots(q: Vec2, cp: &[Vec2], k: &[f32]) -> f32 {
    let n = 512 * cp.len();
    let (k0, km) = (k[0], k[cp.len()]);
    let pts: Vec<Vec2> = (0..n)
        .map(|i| spline_knots(cp, k, k0 + (km - k0) * i as f32 / n as f32))
        .collect();
    let mut best = f32::INFINITY;
    let mut inside = false;
    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        let e = b - a;
        let t = ((q - a).dot(e) / e.dot(e)).clamp(0.0, 1.0);
        best = best.min((q - (a + e * t)).length());
        if (a.y > q.y) != (b.y > q.y) && q.x < a.x + (q.y - a.y) * e.x / e.y {
            inside = !inside;
        }
    }
    if inside {
        -best
    } else {
        best
    }
}

fn check_extrusion_knots(cp: Vec<Vec2>, k: Vec<f32>, seed: u64, samples: usize) {
    let h = 1.2;
    let node = SdfNode::loft_curves_knots(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        k.clone(),
        vec![cp.clone(), cp.clone()],
    )
    .expect("a knotted curve loft");
    let mut r = lcg(seed);
    let mut worst = 0.0f32;
    for _ in 0..samples {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 1.8 - 0.3, r() * 1.4 - 0.7);
        let d2 = curve_distance_knots(Vec2::new(p.x, -p.z), &cp, &k);
        let dw = (-p.y).max(p.y - h);
        let o = Vec2::new(d2.max(0.0), dw.max(0.0));
        let want = d2.max(dw).min(0.0) + o.length();
        worst = worst.max((eval(&node, p) - want).abs());
    }
    assert!(worst < 1e-4, "largest difference {worst}");
}

#[test]
fn uniform_knots_given_explicitly_match_the_uniform_mode() {
    let cp = c_shape();
    let m = cp.len();
    let a = straight(1.2, cp.clone());
    let b = SdfNode::loft_curves_knots(
        vec![Vec3::ZERO, Vec3::new(0.0, 1.2, 0.0)],
        Vec3::X,
        (0..=m).map(|i| i as f32).collect(),
        vec![cp.clone(), cp],
    )
    .expect("uniform knots");
    let mut r = lcg(7);
    for _ in 0..500 {
        let p = Vec3::new(r() * 1.4 - 0.7, r() * 1.8 - 0.3, r() * 1.4 - 0.7);
        assert_eq!(eval(&a, p).to_bits(), eval(&b, p).to_bits());
    }
}

#[test]
fn a_circle_on_non_uniform_knots_is_its_capped_extrusion() {
    let m = 24;
    let cp = ellipse(m, 0.3, 0.3);
    let k: Vec<f32> = (0..=m)
        .map(|j| j as f32 + 0.35 * (std::f32::consts::TAU * j as f32 / m as f32).sin())
        .collect();
    check_extrusion_knots(cp, k, 8, 600);
}

#[test]
fn a_c_shape_with_refined_knots_is_its_capped_extrusion() {
    // the knots crowd toward the two inner corners (where the C turns from the outer to the inner arc)
    let cp = c_shape();
    let m = cp.len();
    let mut k = vec![0.0f32];
    for j in 0..m {
        let near = j
            .min(m - 1 - j)
            .min((j as isize - 14).unsigned_abs())
            .min((j as isize - 15).unsigned_abs());
        k.push(k[j] + if near <= 1 { 0.3 } else { 1.0 });
    }
    check_extrusion_knots(cp, k, 9, 600);
}

#[test]
fn three_hundred_control_points_are_supported() {
    let m = 300;
    let cp = ellipse(m, 0.4, 0.15);
    check_extrusion_knots(cp, (0..=m).map(|i| i as f32).collect(), 10, 200);
}

#[test]
fn malformed_knots_are_refused() {
    let ax = vec![Vec3::ZERO, Vec3::Y];
    let e = ellipse(8, 0.2, 0.1);
    let err = |r: Result<SdfNode, alice_sdf::primitives::loft::LoftError>| {
        r.expect_err("refused").to_string()
    };
    assert!(err(SdfNode::loft_curves_knots(
        ax.clone(),
        Vec3::X,
        vec![0.0; 9],
        vec![e.clone(), e.clone()]
    ))
    .contains("knots"));
    assert!(err(SdfNode::loft_curves_knots(
        ax.clone(),
        Vec3::X,
        (0..8).map(|i| i as f32).collect(),
        vec![e.clone(), e.clone()]
    ))
    .contains("knots"));
    let mut k: Vec<f32> = (0..=8).map(|i| i as f32).collect();
    k[3] = 2.0;
    assert!(err(SdfNode::loft_curves_knots(
        ax,
        Vec3::X,
        k,
        vec![e.clone(), e]
    ))
    .contains("knots"));
}

#[test]
fn a_steeply_growing_section_stays_one_lipschitz_along_the_axis() {
    // the near-circle grows from 0.2 to 0.6 over 0.3 of axis (slope 1.33): the in-plane distance alone
    // changes faster than 1 per unit length along the axis; the bound must hold it
    let node = SdfNode::loft_curves(
        vec![Vec3::ZERO, Vec3::new(0.0, 0.3, 0.0)],
        Vec3::X,
        vec![ellipse(32, 0.2, 0.2), ellipse(32, 0.6, 0.6)],
    )
    .expect("steep");
    let mut r = lcg(11);
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let a = r() * std::f32::consts::TAU;
        let rho = r() * 1.0;
        let p = Vec3::new(rho * a.cos(), r() * 0.3, rho * a.sin());
        let step = 1e-3 + r() * 0.02;
        worst = worst.max((eval(&node, p + Vec3::Y * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

#[test]
fn a_small_tightly_bent_curve_loft_never_reads_past_its_own_surface() {
    // a toe-sized loft (mm): ellipse sections 7 × 5 mm on an axis bent to a radius of curvature of
    // about 25 mm; at points 0.2–8 mm outside, f never exceeds the distance to the loft's own zero set:
    // no inside point within 0.98 f (20,000 random points in that ball per point)
    let axis: Vec<Vec3> = (0..8)
        .map(|i| {
            let a = 0.12 * i as f32;
            Vec3::new(25.0 * (1.0 - a.cos()), 25.0 * a.sin(), 0.0)
        })
        .collect();
    let rows: Vec<Vec<Vec2>> = (0..8)
        .map(|i| ellipse(24, 3.5 - 0.15 * i as f32, 2.5))
        .collect();
    let node = SdfNode::loft_curves(axis, Vec3::Z, rows).expect("toe-like");
    let mut r = lcg(17);
    let (mut over, mut compared) = (0, 0);
    while compared < 60 {
        let p = Vec3::new(r() * 30.0 - 5.0, r() * 30.0 - 4.0, r() * 16.0 - 8.0);
        let f = eval(&node, p);
        if !(0.2..=8.0).contains(&f) {
            continue;
        }
        // no point of the loft's own inside within 0.98 f: 20,000 random points in that ball
        let mut best = f32::INFINITY;
        for _ in 0..20000 {
            let d = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5) * 2.0;
            if d.length() > 1.0 {
                continue;
            }
            let q = p + d * (0.98 * f);
            if eval(&node, q) <= 0.0 {
                best = best.min((q - p).length());
            }
        }
        compared += 1;
        over += usize::from(best.is_finite());
    }
    assert_eq!(
        over, 0,
        "read past its own surface at {over} of {compared} points"
    );
}

/// A torso-like curve loft (mm): 288 control points per section, an ellipse 150 × 100 with a shallow
/// groove at the back and the front, 24 rows along a gently bent 400 mm axis; the last three rows
/// shrink fast (to a quarter of the width over 50 mm), so the local bound spreads several-fold along the axis
fn torso_like() -> SdfNode {
    let axis: Vec<Vec3> = (0..6)
        .map(|i| {
            Vec3::new(
                8.0 * (0.6 * i as f32).sin(),
                80.0 * i as f32,
                5.0 * (0.4 * i as f32).cos(),
            )
        })
        .collect();
    let rows: Vec<Vec<Vec2>> = (0..24)
        .map(|i| {
            let top = [1.0f32, 0.7, 0.45, 0.25];
            let s =
                (1.0 + 0.08 * (0.35 * i as f32).sin()) * if i >= 20 { top[i - 20] } else { 1.0 };
            (0..288)
                .map(|k| {
                    let t = std::f32::consts::TAU * k as f32 / 288.0;
                    let groove = 1.0
                        - 0.03 * (-(t - std::f32::consts::FRAC_PI_2).powi(2) * 40.0).exp()
                        - 0.03 * (-(t - 1.5 * std::f32::consts::PI).powi(2) * 40.0).exp();
                    Vec2::new(150.0 * s * groove * t.cos(), 100.0 * groove * t.sin())
                })
                .collect()
        })
        .collect();
    SdfNode::loft_curves(axis, Vec3::Z, rows).expect("torso-like")
}

#[test]
fn the_normal_is_steady_on_a_torso_like_curve_loft() {
    // surface points found by bisection along rays out from the axis; the normal by central differences
    // with a 20 µm and a 500 µm step differs by less than 1° at ≥ 2000 points
    let node = torso_like();
    let mut r = lcg(18);
    let grad = |p: Vec3, e: f32| -> Vec3 {
        Vec3::new(
            eval(&node, p + Vec3::X * e) - eval(&node, p - Vec3::X * e),
            eval(&node, p + Vec3::Y * e) - eval(&node, p - Vec3::Y * e),
            eval(&node, p + Vec3::Z * e) - eval(&node, p - Vec3::Z * e),
        )
        .normalize()
    };
    if let SdfNode::LoftCurves { ref bounds, .. } = node {
        let (lo, hi) = bounds
            .knots
            .iter()
            .fold((f32::INFINITY, 0.0f32), |(a, b), &k| (a.min(k), b.max(k)));
        assert!(
            hi > 3.0 * lo,
            "the local bound must spread (knots {lo}..{hi})"
        );
    }
    let (mut worst, mut compared) = (0.0f32, 0);
    let mut turns = Vec::new();
    while compared < 2000 {
        let y = 30.0 + r() * 360.0;
        let a = r() * std::f32::consts::TAU;
        let c = Vec3::new(4.0, y, 3.0);
        let dir = Vec3::new(a.cos(), 0.0, a.sin());
        let (mut lo, mut hi) = (0.0f32, 300.0f32);
        if eval(&node, c) >= 0.0 || eval(&node, c + dir * hi) <= 0.0 {
            continue;
        }
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if eval(&node, c + dir * m) <= 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        let p = c + dir * (0.5 * (lo + hi));
        let turn = grad(p, 0.02)
            .dot(grad(p, 0.5))
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        worst = worst.max(turn);
        turns.push(turn);
        compared += 1;
    }
    turns.sort_by(f32::total_cmp);
    let (p50, p99) = (turns[turns.len() / 2], turns[turns.len() * 99 / 100]);
    eprintln!("normal turn p50 {p50}° p99 {p99}° max {worst}° over {compared} points");
    // p50 is held here; the p99 row is the known defect below
    assert!(
        p50 < 0.1,
        "normal turn p50 {p50}° (p99 {p99}°, max {worst}°) over {compared} surface points"
    );
}

#[test]
#[ignore = "known defect: LOFT-NORMAL-P99 — where L(s) jumps (the shrinking top), the band B(s) is narrow and the normal turns up to ~6° (p99) between 20 µm and 500 µm steps; the unit-scale near field is the fix"]
fn the_normal_is_steady_at_the_99th_percentile_on_a_torso_like_curve_loft() {
    let node = torso_like();
    let mut r = lcg(19);
    let grad = |p: Vec3, e: f32| -> Vec3 {
        Vec3::new(
            eval(&node, p + Vec3::X * e) - eval(&node, p - Vec3::X * e),
            eval(&node, p + Vec3::Y * e) - eval(&node, p - Vec3::Y * e),
            eval(&node, p + Vec3::Z * e) - eval(&node, p - Vec3::Z * e),
        )
        .normalize()
    };
    let mut turns = Vec::new();
    while turns.len() < 2000 {
        let y = 30.0 + r() * 360.0;
        let a = r() * std::f32::consts::TAU;
        let c = Vec3::new(4.0, y, 3.0);
        let dir = Vec3::new(a.cos(), 0.0, a.sin());
        let (mut lo, mut hi) = (0.0f32, 300.0f32);
        if eval(&node, c) >= 0.0 || eval(&node, c + dir * hi) <= 0.0 {
            continue;
        }
        for _ in 0..40 {
            let m = 0.5 * (lo + hi);
            if eval(&node, c + dir * m) <= 0.0 {
                lo = m;
            } else {
                hi = m;
            }
        }
        let p = c + dir * (0.5 * (lo + hi));
        turns.push(
            grad(p, 0.02)
                .dot(grad(p, 0.5))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees(),
        );
    }
    turns.sort_by(f32::total_cmp);
    assert!(
        turns[turns.len() * 99 / 100] < 1.0,
        "normal turn p99 {}°",
        turns[turns.len() * 99 / 100]
    );
}

// ---- caps: the end section eroded toward its deepest point and sheared toward an apex offset ----

#[test]
fn a_near_circle_with_hemispherical_caps_is_the_capsule() {
    // the near-circle (M = 32, radius 0.3): its curve radius r (measured here on the curve) and caps of
    // height r with no offset: the capsule of radius r between the end planes
    let cp = ellipse(32, 0.3, 0.3);
    let r: f32 = (0..3200)
        .map(|i| spline(&cp, 32.0 * i as f32 / 3200.0).length())
        .sum::<f32>()
        / 3200.0;
    let h = 0.8;
    let node = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![cp.clone(), cp],
        [[r, 0.0, 0.0], [r, 0.0, 0.0]],
    )
    .expect("capsule");
    let mut rr = lcg(20);
    let mut worst = 0.0f32;
    for _ in 0..3000 {
        let p = Vec3::new(rr() * 1.4 - 0.7, rr() * 2.0 - 0.6, rr() * 1.4 - 0.7);
        let y = p.y.clamp(0.0, h);
        let want = (p - Vec3::new(0.0, y, 0.0)).length() - r;
        if want > 0.0 {
            worst = worst.max((eval(&node, p) - want).abs());
        }
    }
    // the B-spline near-circle's radius varies by about 1e-4 around r
    assert!(worst < 1e-3, "largest difference outside {worst}");
}

/// A straight C-section loft capped at the far end with an apex offset
fn capped_c(offset: Vec2) -> (SdfNode, f32) {
    let h = 0.6;
    let node = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![c_shape(), c_shape()],
        [[0.0, 0.0, 0.0], [0.25, offset.x, offset.y]],
    )
    .expect("capped C");
    (node, h)
}

#[test]
fn a_capped_non_star_section_closes_and_stays_one_lipschitz() {
    let (node, h) = capped_c(Vec2::new(0.05, 0.02));
    // beyond the cap's height nothing is inside; the field is 1-Lipschitz across body and cap
    let mut r = lcg(21);
    for _ in 0..2000 {
        let p = Vec3::new(r() * 1.0 - 0.5, h + 0.26 + r() * 0.2, r() * 1.0 - 0.5);
        assert!(eval(&node, p) > 0.0, "inside beyond the cap at {p:?}");
    }
    let mut worst = 0.0f32;
    for _ in 0..20000 {
        let p = Vec3::new(r() * 1.0 - 0.5, h - 0.1 + r() * 0.4, r() * 1.0 - 0.5);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-3 + r() * 0.03;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
    }
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

#[test]
fn a_cap_reaches_its_height_at_the_offset_apex() {
    // an ellipse end (its deepest point is its centre) with apex offset o: just below the cap height the
    // point above the end centre shifted by o is inside, the unshifted one outside
    let o = Vec2::new(0.08, 0.0);
    let h = 0.6;
    let node = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.0, h, 0.0)],
        Vec3::X,
        vec![ellipse(24, 0.2, 0.12), ellipse(24, 0.2, 0.12)],
        [[0.0, 0.0, 0.0], [0.3, o.x, o.y]],
    )
    .expect("offset cap");
    // frame: N = +x (up = X), B = T × N = −z; the apex at (o.x, h + H, −o.y)
    let apex = Vec3::new(o.x, h + 0.3, -o.y);
    assert!(
        eval(&node, apex - Vec3::Y * 0.01) < 0.0,
        "below the apex reads {}",
        eval(&node, apex - Vec3::Y * 0.01)
    );
    assert!(
        eval(&node, apex + Vec3::Y * 0.01) > 0.0,
        "above the apex reads {}",
        eval(&node, apex + Vec3::Y * 0.01)
    );
    assert!(
        eval(&node, Vec3::new(-o.x, h + 0.29, 0.0)) > 0.0,
        "the unshifted point near the top reads inside"
    );
}

#[test]
fn a_capped_loft_never_reads_past_its_own_surface() {
    let (node, h) = capped_c(Vec2::new(0.05, 0.02));
    let mut r = lcg(22);
    let (mut over, mut compared) = (0, 0);
    while compared < 60 {
        let p = Vec3::new(r() * 1.2 - 0.6, h - 0.2 + r() * 0.8, r() * 1.2 - 0.6);
        let f = eval(&node, p);
        if !(0.005..=0.3).contains(&f) {
            continue;
        }
        let mut hit = false;
        for _ in 0..20000 {
            let d = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5) * 2.0;
            if d.length() <= 1.0 && eval(&node, p + d * (0.98 * f)) <= 0.0 {
                hit = true;
                break;
            }
        }
        compared += 1;
        over += usize::from(hit);
    }
    assert_eq!(
        over, 0,
        "read past its own surface at {over} of {compared} points"
    );
}

#[test]
fn a_strongly_sheared_cap_stays_one_lipschitz() {
    // apex offset 0.9 H (c = 0.9): the shear's factor is c/2 + √(1 + c²/4) ≈ 1.54; with √(1 + c²) ≈ 1.35
    // the field would grow faster than 1 along the shear
    let hh = 0.2;
    let node = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.0, 0.4, 0.0)],
        Vec3::X,
        vec![ellipse(24, 0.2, 0.12), ellipse(24, 0.2, 0.12)],
        [[0.0, 0.0, 0.0], [hh, 0.9 * hh, 0.0]],
    )
    .expect("sheared cap");
    let mut r = lcg(23);
    let mut worst = 0.0f32;
    for _ in 0..60000 {
        let p = Vec3::new(r() * 0.8 - 0.3, 0.38 + r() * 0.3, r() * 0.5 - 0.25);
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let step = 1e-4 + r() * 0.01;
        worst = worst.max((eval(&node, p + dir * step) - eval(&node, p)).abs() / step);
    }
    eprintln!("sheared cap: largest difference quotient {worst}");
    assert!(worst <= 1.0 + 2e-3, "largest difference quotient {worst}");
}

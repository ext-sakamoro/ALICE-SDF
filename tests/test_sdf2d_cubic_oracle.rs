//! Oracle: the 2D cubic Bezier (`Sdf2dNode::Bezier`) never states a distance larger than the true distance
//! to its curve, for glyph-like strokes, straight lines written as cubics, loops and retraced lines.
//!
//! The truth is computed independently in f64: dense samples, then golden-section search on the bracket of
//! every local minimum of the samples. The law may not over-state the distance (a sphere tracer steps
//! through, a rasterised stroke loses its edge). The measured defect: 16 uniform samples over-stated it by
//! 5.4e-2 em on glyph strokes whose half-width is 0.046 em.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: f64 reference distance computed with the platform math"
)]

use alice_sdf::sdf2d::{eval_2d, Sdf2dNode};

type C = [[f64; 2]; 4];

fn curve(c: &C, t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    [0, 1].map(|i| {
        u * u * u * c[0][i]
            + 3.0 * u * u * t * c[1][i]
            + 3.0 * u * t * t * c[2][i]
            + t * t * t * c[3][i]
    })
}

fn true_distance(c: &C, p: [f64; 2]) -> f64 {
    const N: usize = 4096;
    let f = |t: f64| {
        let q = curve(c, t);
        (q[0] - p[0]).hypot(q[1] - p[1])
    };
    let d: Vec<f64> = (0..=N).map(|i| f(i as f64 / N as f64)).collect();
    let mut best = d.iter().copied().fold(f64::INFINITY, f64::min);
    for i in 0..=N {
        let left = i == 0 || d[i - 1] >= d[i];
        let right = i == N || d[i + 1] >= d[i];
        if !(left && right) {
            continue;
        }
        let (mut lo, mut hi) = (
            i.saturating_sub(1) as f64 / N as f64,
            (i + 1).min(N) as f64 / N as f64,
        );
        let g = 0.618_033_988_749_894_9;
        for _ in 0..80 {
            let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
            if f(x1) < f(x2) {
                hi = x2;
            } else {
                lo = x1;
            }
        }
        best = best.min(f(0.5 * (lo + hi)));
    }
    best
}

fn node(c: &[[f32; 2]; 4]) -> Sdf2dNode {
    Sdf2dNode::Bezier {
        p0: c[0],
        p1: c[1],
        p2: c[2],
        p3: c[3],
        thickness: 0.0,
    }
}

fn widen(c: &[[f32; 2]; 4]) -> C {
    c.map(|p| p.map(f64::from))
}

/// over-statement allowed: f32 rounding of the distance at these coordinates (|p| < 3)
const OVER_TOL: f64 = 1e-6;
/// under-statement allowed: the law returns the distance to a point of the curve, so only f32 rounding
const UNDER_TOL: f64 = 1e-6;

#[derive(Default)]
struct Worst {
    over: f64,
    under: f64,
    n: usize,
    at: String,
}

fn check(c: &[[f32; 2]; 4], p: [f32; 2], w: &mut Worst) {
    let truth = true_distance(&widen(c), p.map(f64::from));
    let got = f64::from(eval_2d(&node(c), p));
    let e = got - truth;
    w.n += 1;
    if e > w.over {
        w.over = e;
        w.at = format!("{c:?} p {p:?}: eval {got} true {truth}");
    }
    w.under = w.under.min(e);
}

/// Points on the curve and on both sides of it, at parameters that fall on no sample grid
fn around(c: &[[f32; 2]; 4], w: &mut Worst) {
    let d = widen(c);
    for k in 0..24 {
        let t = (k as f64 + 0.37) / 24.0;
        let q = curve(&d, t);
        let (a, b) = (
            curve(&d, (t - 1e-4).max(0.0)),
            curve(&d, (t + 1e-4).min(1.0)),
        );
        let m = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
        let n = [-(b[1] - a[1]) / m, (b[0] - a[0]) / m];
        for o in [0.0, 0.002, 0.02, 0.046, 0.08, -0.002, -0.02, -0.046, -0.08] {
            check(c, [(q[0] + n[0] * o) as f32, (q[1] + n[1] * o) as f32], w);
        }
    }
}

fn finish(tag: &str, w: &Worst) {
    assert!(w.n > 0, "{tag}: compared 0 points");
    eprintln!(
        "{tag}: {} points, worst over {:.3e} ({}), worst under {:.3e}",
        w.n, w.over, w.at, w.under
    );
    assert!(
        w.over <= OVER_TOL,
        "{tag}: over-states by {:.3e}: {}",
        w.over,
        w.at
    );
    assert!(
        w.under >= -UNDER_TOL,
        "{tag}: under-states by {:.3e}",
        -w.under
    );
}

/// The measured stroke (control points of a glyph): 16 samples gave 0.0540 at a point on the curve
#[test]
fn measured_glyph_stroke() {
    let c = [[0.2, 0.85], [0.18, 0.5], [0.2, 0.2], [0.85, 0.15]];
    let mut w = Worst::default();
    around(&c, &mut w);
    finish("glyph stroke", &w);
    let got = f64::from(eval_2d(&node(&c), [0.796_205_8, 0.154_852_54]));
    let truth = true_distance(
        &widen(&c),
        [f64::from(0.796_205_8f32), f64::from(0.154_852_54f32)],
    );
    assert!(truth < 1e-6, "the independent truth {truth}");
    assert!(
        got <= truth + OVER_TOL,
        "eval {got} over-states the distance {truth}"
    );
}

/// A straight line written as a cubic against the closed-form capsule, for both ways of placing the handles
#[test]
fn straight_lines_match_the_segment_closed_form() {
    let (a, b) = ([0.05f32, 0.1f32], [0.9f32, 0.7f32]);
    let third = |s: f32| [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s];
    let mut n = 0usize;
    let mut worst = 0f64;
    for c in [
        [a, third(1.0 / 3.0), third(2.0 / 3.0), b],
        [a, a, b, b],
        [a, third(0.9), third(0.1), b],
    ] {
        for i in 0..=40 {
            for j in 0..=40 {
                let p = [i as f32 * 0.03 - 0.1, j as f32 * 0.03 - 0.1];
                let (ax, ay) = (f64::from(a[0]), f64::from(a[1]));
                let (bx, by) = (f64::from(b[0]) - ax, f64::from(b[1]) - ay);
                let (px, py) = (f64::from(p[0]) - ax, f64::from(p[1]) - ay);
                let t = ((px * bx + py * by) / (bx * bx + by * by)).clamp(0.0, 1.0);
                let want = (px - bx * t).hypot(py - by * t);
                let got = f64::from(eval_2d(&node(&c), p));
                worst = worst.max((got - want).abs());
                n += 1;
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("straight lines: {n} points, worst |law - segment| {worst:.3e}");
    assert!(worst <= 2e-6, "|law - segment| {worst:.3e}");
}

/// Random cubics, cubics that retrace a line (a cusp in the parameter) and closed loops, at any point
#[test]
fn hard_cubics_at_any_point() {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    let mut r = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let mut w = Worst::default();
    for k in 0..300 {
        let mut c = [[r(), r()], [r(), r()], [r(), r()], [r(), r()]];
        match k % 3 {
            1 => {
                let (dx, dy) = (c[3][0] - c[0][0], c[3][1] - c[0][1]);
                c[1] = [c[0][0] + 3.0 * dx, c[0][1] + 3.0 * dy];
                c[2] = [c[3][0] - 3.0 * dx, c[3][1] - 3.0 * dy];
            }
            2 => c[3] = c[0],
            _ => {}
        }
        for _ in 0..12 {
            check(&c, [r() * 2.0 - 0.5, r() * 2.0 - 0.5], &mut w);
        }
        around(&c, &mut w);
    }
    finish("hard cubics", &w);
}

/// A curve collapsed to a point is the distance to that point, and the law stays finite
#[test]
fn a_collapsed_curve_is_a_point() {
    let c = [[0.5f32, 0.25f32]; 4];
    let got = eval_2d(&node(&c), [1.5, 0.25]);
    assert_eq!(got, 1.0);
}

/// The thickness subtracts exactly
#[test]
fn thickness_is_subtracted() {
    let c = [[0.2, 0.85], [0.18, 0.5], [0.2, 0.2], [0.85, 0.15]];
    let p = [0.4, 0.4];
    let thick = Sdf2dNode::Bezier {
        p0: c[0],
        p1: c[1],
        p2: c[2],
        p3: c[3],
        thickness: 0.05,
    };
    assert_eq!(eval_2d(&thick, p), eval_2d(&node(&c), p) - 0.05);
}

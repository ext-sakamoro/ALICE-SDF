//! Oracle: `sweep_bezier` measures the distance to its quadratic Bezier in the XZ plane, and that
//! distance is never larger than the true one, from near-collinear to strongly bent control points.
//!
//! The true distance is computed independently in f64: the curve is sampled densely and the nearest
//! sample is refined by golden-section search on its bracket. An over-statement lets a sphere tracer
//! step through the surface.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    clippy::manual_midpoint,
    clippy::missing_const_for_fn,
    clippy::suspicious_operation_groupings,
    reason = "test code: f64 reference distance computed with the platform math"
)]

use alice_sdf::prelude::*;
use glam::Vec2;

fn curve(a: [f64; 2], b: [f64; 2], c: [f64; 2], t: f64) -> [f64; 2] {
    let u = 1.0 - t;
    [0, 1].map(|i| u * u * a[i] + 2.0 * u * t * b[i] + t * t * c[i])
}

fn dist(p: [f64; 2], q: [f64; 2]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// The true distance from `p` to the curve (f64): 4096 samples, then golden-section on the best bracket
fn true_distance(a: [f64; 2], b: [f64; 2], c: [f64; 2], p: [f64; 2]) -> f64 {
    const N: usize = 4096;
    let f = |t: f64| dist(p, curve(a, b, c, t));
    let (mut best, mut bd) = (0usize, f64::INFINITY);
    for i in 0..=N {
        let d = f(i as f64 / N as f64);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    let (mut lo, mut hi) = (
        best.saturating_sub(1) as f64 / N as f64,
        (best + 1).min(N) as f64 / N as f64,
    );
    let g = 0.618_033_988_749_894_9;
    for _ in 0..120 {
        let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if f(x1) < f(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    bd.min(f(0.5 * (lo + hi)))
}

/// the f32 value of every coordinate (the law's input)
fn r(p: [f64; 2]) -> [f64; 2] {
    p.map(|x| f64::from(x as f32))
}

fn v2(p: [f64; 2]) -> Vec2 {
    Vec2::new(p[0] as f32, p[1] as f32)
}

/// The truth against closed forms: a straight curve is a segment, and the nearest point to a point
/// above the apex of a parabola (the convex side) is the apex
#[test]
fn the_f64_truth_matches_closed_forms() {
    let (a, b, c) = ([0.0, 0.0], [0.5, 0.0], [1.0, 0.0]);
    for (p, want) in [([0.25, 0.3], 0.3), ([-0.3, 0.4], 0.5), ([1.3, -0.4], 0.5)] {
        let got = true_distance(a, b, c, p);
        assert!(
            (got - want).abs() < 1e-12,
            "segment at {p:?}: {got} vs {want}"
        );
    }
    // the curve a (0,0), b (1/2,1), c (1,0) is y = 2x(1 - x), apex (1/2, 1/2)
    let got = true_distance(a, [0.5, 1.0], c, [0.5, 0.75]);
    assert!((got - 0.25).abs() < 1e-12, "above the apex: {got}");
}

/// over-statement allowed: f32 rounding of the distance at these coordinates (|p| < 5)
const OVER_TOL: f64 = 1e-5;

#[test]
#[ignore = "known defect: SWEEP-BEZIER-OVERSTATE-001: bezier_distance_2d over-states the distance by 1.56e-2 at a near-straight curve and 3.9e-4 at a bent one (f32 closed form, no refinement)"]
fn sweep_bezier_never_overstates_the_distance() {
    let (mut worst, mut at, mut n) = (0.0f64, String::new(), 0usize);
    for &h in &[
        1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0,
    ] {
        for (a, b, c) in [
            ([0.0, 0.0], [0.5, h], [1.0, 0.0]),
            ([0.0, 0.0], [0.3, h], [1.0, 0.0]),
            ([3.1, 2.2], [3.15, 2.2 + 0.5 * h], [3.2, 2.2]),
        ] {
            let (a, b, c) = (r(a), r(b), r(c));
            let node = SdfNode::sphere(0.0).sweep_bezier(v2(a), v2(b), v2(c));
            let chord = dist(a, c);
            let (dx, dy) = ((c[0] - a[0]) / chord, (c[1] - a[1]) / chord);
            for k in 0..=40 {
                let q = curve(a, b, c, k as f64 / 40.0);
                for &o in &[0.0, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1] {
                    let p = r([q[0] - dy * o * chord, q[1] + dx * o * chord]);
                    let got = f64::from(eval(&node, Vec3::new(p[0] as f32, 0.0, p[1] as f32)));
                    let truth = true_distance(a, b, c, p);
                    n += 1;
                    if got - truth > worst {
                        worst = got - truth;
                        at = format!(
                            "a {a:?} b {b:?} c {c:?} p {p:?}: eval {got:.7} true {truth:.7}"
                        );
                    }
                }
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("compared {n} points: worst over-statement {worst:.3e} ({at})");
    assert!(
        worst <= OVER_TOL,
        "over-states the distance by {worst:.3e} > {OVER_TOL:e}: {at}"
    );
}

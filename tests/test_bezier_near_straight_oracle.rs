//! Oracle: the quadratic Bezier tube (`SdfNode::bezier`) never states a distance larger than the true
//! distance to its curve, from near-collinear to strongly bent control points.
//!
//! The true distance is computed independently in f64: the curve is sampled densely and the nearest
//! sample is refined by golden-section search on its bracket. The law may under-state the distance
//! (a sphere tracer stays safe), but an over-statement lets a tracer step through the surface. The
//! measured defect: with |a - 2b + c| = 0.06 a point 0.000126 from the curve evaluated to 0.004022.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

/// f64 point of the quadratic Bezier at `t`
fn curve(a: [f64; 3], b: [f64; 3], c: [f64; 3], t: f64) -> [f64; 3] {
    let u = 1.0 - t;
    [0, 1, 2].map(|i| u * u * a[i] + 2.0 * u * t * b[i] + t * t * c[i])
}

fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

/// The true distance from `p` to the curve (f64): 4096 samples, then golden-section on the best bracket
fn true_distance(a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3]) -> f64 {
    const N: usize = 4096;
    let f = |t: f64| dist(p, curve(a, b, c, t));
    let mut best = 0usize;
    let mut bd = f64::INFINITY;
    for i in 0..=N {
        let d = f(i as f64 / N as f64);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    let (mut lo, mut hi) = (
        (best.saturating_sub(1)) as f64 / N as f64,
        ((best + 1).min(N)) as f64 / N as f64,
    );
    let g = 0.618_033_988_749_894_9;
    for _ in 0..120 {
        let x1 = hi - g * (hi - lo);
        let x2 = lo + g * (hi - lo);
        if f(x1) < f(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    bd.min(f(0.5 * (lo + hi))).min(f(0.0)).min(f(1.0))
}

fn v3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

struct Worst {
    over: f64,
    under: f64,
    at: String,
    n: usize,
}

/// Compare the law at points around the curve a-b-c: along the curve (t grid) offset in the plane by
/// the normal and out of the plane, and beyond both ends
fn sweep(a: [f64; 3], b: [f64; 3], c: [f64; 3], w: &mut Worst) {
    let node = SdfNode::bezier(v3(a), v3(b), v3(c), 0.0);
    let chord = dist(a, c);
    let offsets = [0.0, 1e-4, 1e-3, 1e-2, 0.1];
    for k in 0..=40 {
        let t = k as f64 / 40.0;
        let q = curve(a, b, c, t);
        // a normal in the curve's plane: the chord rotated a quarter turn about z (the sweeps lie in z = const)
        let (dx, dy) = ((c[0] - a[0]) / chord, (c[1] - a[1]) / chord);
        for &o in &offsets {
            for &s in &[1.0, -1.0] {
                for p in [
                    [q[0] - dy * o * s * chord, q[1] + dx * o * s * chord, q[2]],
                    [q[0], q[1], q[2] + o * s * chord],
                ] {
                    check(&node, a, b, c, p, w);
                }
            }
        }
    }
    for &e in &[0.01, 0.1, 0.5] {
        for p in [
            [a[0] - (c[0] - a[0]) * e, a[1] - (c[1] - a[1]) * e + 1e-3 * chord, a[2]],
            [c[0] + (c[0] - a[0]) * e, c[1] + (c[1] - a[1]) * e - 1e-3 * chord, c[2]],
        ] {
            check(&node, a, b, c, p, w);
        }
    }
}

fn check(node: &SdfNode, a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3], w: &mut Worst) {
    // the law sees the f32 control points and point: the truth uses the same values
    let (a, b, c, p) = (r(a), r(b), r(c), r(p));
    let truth = true_distance(a, b, c, p);
    let got = f64::from(eval(node, v3(p)));
    let e = got - truth;
    w.n += 1;
    if e > w.over {
        w.over = e;
        w.at = format!("a {a:?} b {b:?} c {c:?} p {p:?}: eval {got:.7} true {truth:.7}");
    }
    w.under = w.under.min(e);
}

/// the f32 value of every coordinate (the law's input)
fn r(p: [f64; 3]) -> [f64; 3] {
    p.map(|x| f64::from(x as f32))
}

/// over-statement allowed: f32 rounding of the distance at these coordinates (|p| < 5)
const OVER_TOL: f64 = 1e-5;
/// under-statement allowed (a safe tracer, but a law that returns ~0 everywhere is wrong too)
const UNDER_TOL: f64 = 1e-3;

#[test]
fn bezier_never_overstates_the_distance_from_near_straight_to_bent() {
    let mut w = Worst { over: 0.0, under: 0.0, at: String::new(), n: 0 };
    // bend |a - 2b + c| = 2h for a unit chord on the x axis
    for &h in &[1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0] {
        sweep([0.0, 0.0, 0.0], [0.5, h, 0.0], [1.0, 0.0, 0.0], &mut w);
        // control point off the middle (an unsymmetric parametrisation)
        sweep([0.0, 0.0, 0.0], [0.3, h, 0.0], [1.0, 0.0, 0.0], &mut w);
        // a glyph-scale stroke far from the origin (the measured case: chord ~0.1, bend 0.06 units)
        sweep([3.1, 2.2, 0.4], [3.15, 2.2 + 0.5 * h, 0.4], [3.2, 2.2, 0.4], &mut w);
    }
    assert!(w.n > 0, "compared 0 points");
    eprintln!("compared {} points: worst over-statement {:.3e} ({}), worst under-statement {:.3e}", w.n, w.over, w.at, w.under);
    assert!(w.over <= OVER_TOL, "the law over-states the distance by {:.3e} > {OVER_TOL:e}: {}", w.over, w.at);
    assert!(w.under >= -UNDER_TOL, "the law under-states the distance by {:.3e}", -w.under);
}

/// The measured case (a glyph stroke as a midpoint quadratic, em units): the law returned 0.004021526
/// where the curve is 0.000126199 away (f64 truth over 200,001 samples)
#[test]
fn bezier_measured_glyph_stroke_case() {
    let f = |x: u32| f32::from_bits(x);
    let q0 = Vec3::new(f(0x3e99_999a), f(0x3f59_999a), 0.0);
    let q1 = Vec3::new(f(0x3e91_47ae), f(0x3f1f_999a), 0.0);
    let q2 = Vec3::new(f(0x3e9c_28f6), f(0x3ee3_3334), 0.0);
    let p = Vec3::new(f(0x3e98_a09c), f(0x3f51_cfb3), 0.0);
    let d = |v: Vec3| [f64::from(v.x), f64::from(v.y), f64::from(v.z)];
    let truth = true_distance(d(q0), d(q1), d(q2), d(p));
    assert!((truth - 0.000_126_199).abs() < 2e-9, "the independent truth {truth}");
    let got = f64::from(eval(&SdfNode::bezier(q0, q1, q2, 0.0), p));
    assert!(got <= truth + OVER_TOL, "eval {got} over-states the distance {truth}");
}

//! Oracle: the blobby cross law against an independent f64 distance to its own curve (the parabola arc
//! `y = he (1 - 2 x^2)`, `x` in `[0, 1/sqrt 2]`, in the folded frame), and against the zero set in the plane.
//!
//! The measured defect: the f32 closed form lost `q - r` near the arm tips, so the law over-stated the
//! distance by 2.0e-3 at size 1 and returned the wrong side of the curve there.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: f64 reference distance computed with the platform math"
)]

use alice_sdf::prelude::*;

const SQ: f64 = std::f64::consts::FRAC_1_SQRT_2;
const HE: f64 = 0.5;

fn fold(a: f64, b: f64) -> (f64, f64) {
    let (a, b) = (a.abs(), b.abs());
    ((a - b).abs() * SQ, (1.0 - a - b) * SQ)
}

fn arch(x: f64) -> f64 {
    HE * (1.0 - 2.0 * x * x)
}

/// Signed distance to the arc in the folded frame: positive where the arc is above the point in the folded `y`
fn true_signed(a: f64, b: f64) -> f64 {
    let (px, py) = fold(a, b);
    let f = |x: f64| (x - px).hypot(arch(x) - py);
    const N: usize = 2048;
    let d: Vec<f64> = (0..=N).map(|i| f(SQ * i as f64 / N as f64)).collect();
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..=N {
        if (i > 0 && d[i - 1] < d[i]) || (i < N && d[i + 1] < d[i]) {
            continue;
        }
        let (mut lo, mut hi) = (
            SQ * i.saturating_sub(1) as f64 / N as f64,
            SQ * (i + 1).min(N) as f64 / N as f64,
        );
        let g = 0.618_033_988_749_894_9;
        for _ in 0..70 {
            let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
            if f(x1) < f(x2) {
                hi = x2;
            } else {
                lo = x1;
            }
        }
        let x = 0.5 * (lo + hi);
        if f(x) < best.0 {
            best = (f(x), x);
        }
    }
    let side = if arch(best.1) - py >= 0.0 { 1.0 } else { -1.0 };
    best.0 * side
}

/// The grid, the boundary of the arc at several offsets, and the x axis on both sides of the arm tip
fn points() -> Vec<[f32; 2]> {
    let mut v = Vec::new();
    for i in -80..=80 {
        for j in -80..=80 {
            v.push([i as f32 * 0.0171, j as f32 * 0.0171]);
        }
    }
    for k in 0..=400 {
        let x = SQ * k as f64 / 400.0;
        let u = 1.0 - std::f64::consts::SQRT_2 * arch(x);
        let c = [
            0.5 * (u + std::f64::consts::SQRT_2 * x),
            0.5 * (u - std::f64::consts::SQRT_2 * x),
        ];
        for o in [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 0.1] {
            for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                v.push([(c[0] + o * dx) as f32, (c[1] + o * dy) as f32]);
            }
        }
    }
    for e in [0.0f32, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 2.0] {
        for o in [0.0f32, 1e-5, 1e-4, 1e-3, 1e-2, 0.1] {
            v.push([1.0 + e, o]);
            v.push([1.0 - e, o]);
        }
    }
    v
}

const OVER_TOL: f64 = 1e-6;

#[test]
fn the_arc_truth_matches_closed_forms() {
    // the waist: nearest point of the arc from the fold's origin line is its apex
    let (px, py) = fold(0.146_446_609_406_726_24, 0.146_446_609_406_726_24);
    assert!(px.abs() < 1e-12 && (py - HE).abs() < 1e-12, "({px}, {py})");
    assert!(true_signed(0.146_446_609_406_726_24, 0.146_446_609_406_726_24).abs() < 1e-9);
    // the arm tip is on the arc
    assert!(true_signed(1.0, 0.0).abs() < 1e-9);
    // beyond the tip on the axis the distance is to the tip
    assert!((true_signed(1.25, 0.0).abs() - 0.25).abs() < 1e-9);
}

#[test]
fn blobby_cross_matches_the_distance_to_its_curve() {
    let (mut over, mut under, mut flips, mut n, mut at) =
        (0.0f64, 0.0f64, 0usize, 0usize, String::new());
    for size in [1.0f32, 3.0] {
        let node = SdfNode::blobby_cross(size, 1000.0);
        for p in points() {
            let (x, z) = (p[0] * size, p[1] * size);
            let got = f64::from(eval(&node, Vec3::new(x, 0.0, z)));
            let t = true_signed(
                f64::from(x) / f64::from(size),
                f64::from(z) / f64::from(size),
            ) * f64::from(size);
            let e = got.abs() - t.abs();
            n += 1;
            if e > over {
                over = e;
                at = format!("size {size} ({x}, {z}): eval {got} true {t}");
            }
            under = under.min(e);
            // the sign only counts away from rounding of the curve itself
            if t.abs() > 1e-5 * f64::from(size) && (got < 0.0) != (t < 0.0) {
                flips += 1;
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("blobby_cross: {n} points, worst over {over:.3e} ({at}), worst under {under:.3e}, sign flips {flips}");
    assert!(over <= OVER_TOL * 3.0, "over-states by {over:.3e}: {at}");
    assert!(
        under >= -OVER_TOL * 3.0,
        "under-states the distance to its curve by {under:.3e}"
    );
    assert_eq!(flips, 0, "points on the wrong side");
}

/// The zero set in the plane is the part of the arcs in the first quadrant, mirrored, and the arm tips: the
/// law may be below the distance to it (it measures a longer curve), never above it
#[test]
fn blobby_cross_never_overstates_the_distance_to_the_surface() {
    let node = SdfNode::blobby_cross(1.0, 1000.0);
    let (mut over, mut n) = (0.0f64, 0usize);
    for p in points() {
        let (a, b) = (f64::from(p[0]).abs(), f64::from(p[1]).abs());
        let mut best = (a - 1.0).hypot(b).min(a.hypot(b - 1.0));
        for side in [1.0, -1.0] {
            for i in 0..=2048 {
                let x = SQ * f64::from(i) / 2048.0;
                let u = 1.0 - std::f64::consts::SQRT_2 * arch(x);
                let d = side * std::f64::consts::SQRT_2 * x;
                let c = [0.5 * (u + d), 0.5 * (u - d)];
                if c[0] >= 0.0 && c[1] >= 0.0 {
                    best = best.min((c[0] - a).hypot(c[1] - b));
                }
            }
        }
        // the truth's own spacing on the arc bounds how far above the true distance it can be
        let got = f64::from(eval(&node, Vec3::new(p[0], 0.0, p[1]))).abs();
        over = over.max(got - best);
        n += 1;
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("blobby_cross against the surface: {n} points, worst over {over:.3e}");
    assert!(over <= OVER_TOL, "over-states by {over:.3e}");
}

/// The WGSL helper against the CPU law; with `ALICE_SDF_REQUIRE_GPU` set a missing adapter is a failure
#[cfg(feature = "gpu")]
#[test]
fn blobby_cross_gpu_matches_the_cpu_law() {
    use alice_sdf::compiled::GpuEvaluator;
    let (mut par, mut n) = (0.0f64, 0usize);
    for size in [1.0f32, 3.0] {
        let node = SdfNode::blobby_cross(size, 1000.0);
        let gpu = match GpuEvaluator::new(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
                );
                eprintln!("skipping blobby_cross GPU parity: {e}");
                return;
            }
        };
        let all: Vec<Vec3> = points()
            .iter()
            .step_by(13)
            .map(|p| Vec3::new(p[0] * size, 0.0, p[1] * size))
            .collect();
        for pts in all.chunks(2048) {
            for (p, g) in pts.iter().zip(gpu.eval_batch(pts).expect("gpu eval")) {
                par = par.max((f64::from(g) - f64::from(eval(&node, *p))).abs());
                n += 1;
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("blobby_cross GPU: {n} points, max |gpu - cpu| {par:.3e}");
    assert!(par <= 2e-5, "GPU/CPU drift {par:.3e}");
}

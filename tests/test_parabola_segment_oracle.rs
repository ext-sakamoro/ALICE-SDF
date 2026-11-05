//! Oracle: `SdfNode::parabola_segment` returns the distance to its boundary (the arch
//! y = h (1 - (x / w)^2) over |x| <= w and the base y = 0), negative inside, on both sides of the
//! arch and from flat to tall arches.
//!
//! The true distance is computed independently in f64: the arch is sampled densely and the nearest
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
#[cfg(feature = "gpu")]
#[path = "common/gpu_batches.rs"]
mod gpu_batches;
#[cfg(feature = "gpu")]
use gpu_batches::eval_in_batches;

fn arch(w: f64, h: f64, t: f64) -> f64 {
    h * (1.0 - (t / w).powi(2))
}

/// Distance from (px >= 0, py) to the arch over 0 <= t <= w: 4096 samples, then golden-section
fn arch_distance(w: f64, h: f64, px: f64, py: f64) -> f64 {
    const N: usize = 4096;
    let f = |t: f64| ((px - t).powi(2) + (py - arch(w, h, t)).powi(2)).sqrt();
    let (mut best, mut bd) = (0usize, f64::INFINITY);
    for i in 0..=N {
        let d = f(w * i as f64 / N as f64);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    let (mut lo, mut hi) = (
        w * best.saturating_sub(1) as f64 / N as f64,
        w * (best + 1).min(N) as f64 / N as f64,
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

/// The true signed distance in the cross-section
fn true_signed(w: f64, h: f64, x: f64, y: f64) -> f64 {
    let px = x.abs();
    let base = if px <= w {
        y.abs()
    } else {
        ((px - w).powi(2) + y * y).sqrt()
    };
    let d = arch_distance(w, h, px, y).min(base);
    if px <= w && y >= 0.0 && y <= arch(w, h, px) {
        -d
    } else {
        d
    }
}

/// The truth against closed forms: above the apex of a shallow arch (the convex side, within the
/// curvature radius or not) the nearest point is the apex, below the base it is the base, and past
/// the base corner it is the corner
#[test]
fn the_f64_truth_matches_closed_forms() {
    for (x, y, want) in [
        (0.0, 1.5, 0.5),
        (0.0, 1.25, 0.25),
        (0.5, -0.3, 0.3),
        (1.3, -0.4, 0.5),
        (0.0, 0.25, -0.25),
    ] {
        let got = true_signed(1.0, 1.0, x, y);
        assert!((got - want).abs() < 1e-12, "({x}, {y}): {got} vs {want}");
    }
}

/// over-statement and sign tolerance: f32 rounding of the distance at these coordinates (|p| < 10)
const OVER_TOL: f64 = 1e-5;

/// Cross-sections sampled around the shape and along both sides of the arch: at most 2048 points
#[cfg(feature = "gpu")]
fn gpu_points(w: f32, h: f32) -> Vec<Vec3> {
    let (wf, hf) = (f64::from(w), f64::from(h));
    let mut v = Vec::new();
    for i in -15i32..=15 {
        for j in -15i32..=20 {
            v.push(Vec3::new(
                i as f32 * 0.1 * w,
                j as f32 * 0.1 * h.max(w * 0.1),
                0.0,
            ));
        }
    }
    for k in 0..=40 {
        let x = wf * k as f64 / 40.0;
        for o in [0.0, 1e-3, -1e-3, 1e-2, 0.1, 0.5, 2.0] {
            v.push(Vec3::new(x as f32, (arch(wf, hf, x) + o) as f32, 0.0));
        }
    }
    v
}

/// The WGSL helper against the CPU law and the truth; with `ALICE_SDF_REQUIRE_GPU` set a missing adapter
/// is a failure, not a skip
#[cfg(feature = "gpu")]
#[test]
fn parabola_segment_gpu_matches_the_cpu_law_and_the_truth() {
    use alice_sdf::compiled::GpuEvaluator;
    let (mut par, mut over, mut n) = (0.0f64, 0.0f64, 0usize);
    for (w, h) in [
        (1.0f32, 1.0f32),
        (1.0, 0.1),
        (1.0, 0.01),
        (1.0, 3.0),
        (0.2, 2.0),
        (3.0, 0.3),
    ] {
        let node = SdfNode::parabola_segment(w, h, 1000.0);
        let gpu = match GpuEvaluator::new(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
                );
                eprintln!("skipping parabola_segment GPU parity: {e}");
                return;
            }
        };
        let pts = gpu_points(w, h);
        assert!(pts.len() <= 2048, "{} points in one dispatch", pts.len());
        for (p, g) in pts
            .iter()
            .zip(eval_in_batches(&gpu, &pts).expect("gpu eval"))
        {
            let g = f64::from(g);
            par = par.max((g - f64::from(eval(&node, *p))).abs());
            let t = true_signed(f64::from(w), f64::from(h), f64::from(p.x), f64::from(p.y));
            over = over.max(g.abs() - t.abs());
            n += 1;
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("parabola_segment GPU: {n} points, max |gpu - cpu| {par:.3e}, worst over-statement {over:.3e}");
    assert!(par <= 2e-5, "GPU/CPU drift {par:.3e}");
    assert!(
        over <= OVER_TOL,
        "the GPU over-states the distance by {over:.3e}"
    );
}

#[test]
fn parabola_segment_never_overstates_the_distance() {
    let (mut worst, mut at, mut n, mut flipped) = (0.0f64, String::new(), 0usize, 0usize);
    for (w, h) in [
        (1.0f32, 1.0f32),
        (1.0, 0.1),
        (1.0, 0.01),
        (1.0, 1e-3),
        (1.0, 3.0),
        (0.2, 2.0),
        (3.0, 0.3),
    ] {
        let node = SdfNode::parabola_segment(w, h, 1000.0);
        let (wf, hf) = (f64::from(w), f64::from(h));
        let mut pts: Vec<(f32, f32)> = Vec::new();
        // the plane around the shape
        for i in -30i32..=30 {
            for j in -30i32..=40 {
                pts.push((i as f32 * 0.05 * w, j as f32 * 0.05 * h.max(w * 0.1)));
            }
        }
        // along the arch, on both sides
        for k in 0..=80 {
            let x = wf * k as f64 / 80.0;
            for o in [0.0, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, 0.5, 2.0] {
                pts.push((x as f32, (arch(wf, hf, x) + o) as f32));
            }
        }
        for (x, y) in pts {
            let got = f64::from(eval(&node, Vec3::new(x, y, 0.0)));
            let truth = true_signed(wf, hf, f64::from(x), f64::from(y));
            n += 1;
            if truth.abs() > OVER_TOL && got * truth < 0.0 {
                flipped += 1;
            }
            let e = got.abs() - truth.abs();
            if e > worst {
                worst = e;
                at = format!("w {w} h {h} ({x}, {y}): eval {got:.7} true {truth:.7}");
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!(
        "compared {n} points: worst over-statement {worst:.3e} ({at}), {flipped} flipped signs"
    );
    assert_eq!(flipped, 0, "points on the wrong side");
    assert!(
        worst <= OVER_TOL,
        "over-states the distance by {worst:.3e} > {OVER_TOL:e}: {at}"
    );
}

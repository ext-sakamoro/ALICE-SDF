//! Oracle: the helix (`SdfNode::helix`) with no tube radius never states a distance larger than the true
//! distance to its curve, computed independently in f64 (dense samples of the curve parameter, then
//! golden-section search on the bracket of every local minimum).
//!
//! The measured defect: `r^2 + R^2 - 2rR cos(a)` cancels on the curve, so at R = 3 a point on the curve
//! evaluated 1.38e-3 from it.
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

fn true_distance(p: [f64; 3], major: f64, pitch: f64, near: f64) -> f64 {
    let c = pitch / std::f64::consts::TAU;
    let f = |ph: f64| {
        ((p[0] - major * ph.cos()).powi(2)
            + (p[2] - major * ph.sin()).powi(2)
            + (p[1] - c * ph).powi(2))
        .sqrt()
    };
    let w = 3.0 * std::f64::consts::TAU;
    const N: usize = 8192;
    let st = 2.0 * w / N as f64;
    let d: Vec<f64> = (0..=N).map(|i| f(near - w + st * i as f64)).collect();
    let mut best = f64::INFINITY;
    for i in 0..=N {
        if (i > 0 && d[i - 1] < d[i]) || (i < N && d[i + 1] < d[i]) {
            continue;
        }
        let (mut lo, mut hi) = (
            near - w + st * (i as f64 - 1.0),
            near - w + st * (i as f64 + 1.0),
        );
        let g = 0.618_033_988_749_894_9;
        for _ in 0..70 {
            let (a, b) = (hi - g * (hi - lo), lo + g * (hi - lo));
            if f(a) < f(b) {
                hi = b;
            } else {
                lo = a;
            }
        }
        best = best.min(f(0.5 * (lo + hi))).min(d[i]);
    }
    best
}

/// Points on the curve and around it, radially, along the axis and along z
fn points(major: f32, pitch: f32) -> Vec<([f32; 3], f64)> {
    let (rd, pd) = (f64::from(major), f64::from(pitch));
    let mut v = Vec::new();
    for k in 0..120 {
        let ph = -6.0 + k as f64 * 0.1037;
        let cc = [
            rd * ph.cos(),
            pd / std::f64::consts::TAU * ph,
            rd * ph.sin(),
        ];
        for o in [
            0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, -1e-4, -1e-3, -1e-2, -0.05,
        ] {
            for dir in 0..3 {
                let mut q = cc;
                if dir == 0 {
                    q[0] = (rd + o) * ph.cos();
                    q[2] = (rd + o) * ph.sin();
                } else {
                    q[dir] += o;
                }
                v.push(([q[0] as f32, q[1] as f32, q[2] as f32], ph));
            }
        }
    }
    v
}

/// allowed: f32 rounding of the coordinates of a point on a curve at radius 3 (the worst of the set)
const OVER_TOL: f64 = 2.5e-6;

#[test]
fn the_truth_matches_closed_forms() {
    // on the curve; on the axis (every curve point is the same distance in the plane) at the height of a point
    assert!(true_distance([1.0, 0.0, 0.0], 1.0, 1.0, 0.0) < 1e-10);
    assert!((true_distance([0.0, 0.0, 0.0], 1.0, 1e-9, 0.0) - 1.0).abs() < 1e-9);
    // a small radial step off the curve is that step when the pitch is flat
    assert!((true_distance([1.25, 0.0, 0.0], 1.0, 1e-9, 0.0) - 0.25).abs() < 1e-9);
}

#[test]
fn helix_never_overstates_the_distance_to_its_curve() {
    let (mut n, mut over, mut under, mut at) = (0usize, 0.0f64, 0.0f64, String::new());
    for major in [0.3f32, 1.0, 3.0] {
        for pitch in [0.2f32, 0.5, 1.0, 3.0] {
            let node = SdfNode::helix(major, 0.0, pitch, 1000.0);
            for (p, ph) in points(major, pitch) {
                let t = true_distance(p.map(f64::from), f64::from(major), f64::from(pitch), ph);
                let got = f64::from(eval(&node, Vec3::new(p[0], p[1], p[2])));
                let e = got - t;
                n += 1;
                if e > over {
                    over = e;
                    at = format!("R {major} pitch {pitch} {p:?}: eval {got} true {t}");
                }
                under = under.min(e);
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("helix: {n} points, worst over {over:.3e} ({at}), worst under {under:.3e}");
    assert!(over <= OVER_TOL, "over-states by {over:.3e}: {at}");
    assert!(under >= -OVER_TOL, "under-states by {under:.3e}");
}

/// The WGSL helper against the CPU law; with `ALICE_SDF_REQUIRE_GPU` set a missing adapter is a failure
#[cfg(feature = "gpu")]
#[test]
fn helix_gpu_matches_the_cpu_law() {
    use alice_sdf::compiled::GpuEvaluator;
    let (mut par, mut n) = (0.0f64, 0usize);
    for (major, pitch) in [(0.3f32, 3.0f32), (1.0, 0.5), (3.0, 0.2)] {
        let node = SdfNode::helix(major, 0.0, pitch, 1000.0);
        let gpu = match GpuEvaluator::new(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
                );
                eprintln!("skipping helix GPU parity: {e}");
                return;
            }
        };
        let all: Vec<Vec3> = points(major, pitch)
            .iter()
            .map(|(p, _)| Vec3::new(p[0], p[1], p[2]))
            .collect();
        for (p, g) in all
            .iter()
            .zip(eval_in_batches(&gpu, &all).expect("gpu eval"))
        {
            par = par.max((f64::from(g) - f64::from(eval(&node, *p))).abs());
            n += 1;
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("helix GPU: {n} points, max |gpu - cpu| {par:.3e}");
    assert!(par <= 2e-5, "GPU/CPU drift {par:.3e}");
}

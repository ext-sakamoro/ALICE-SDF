//! GPU ↔ CPU parity of the quadratic Bezier tube (`SdfNode::bezier`) from near-collinear to strongly
//! bent control points, and the GPU's distance against an independent f64 truth.
//!
//! The WGSL helper `sdf_bezier` carries the CPU law's Newton refinement of each closed-form root
//! (src/primitives/bezier.rs `refine`); before it, the f32 closed form over-stated the distance on
//! near-straight curves (tests/test_bezier_near_straight_oracle.rs). Runs through `GpuEvaluator`;
//! with `ALICE_SDF_REQUIRE_GPU` set a missing adapter is a failure, not a skip.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: f64 reference distance computed with the platform math"
)]
#![cfg(feature = "gpu")]

use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;

fn curve(a: [f64; 3], b: [f64; 3], c: [f64; 3], t: f64) -> [f64; 3] {
    let u = 1.0 - t;
    [0, 1, 2].map(|i| u * u * a[i] + 2.0 * u * t * b[i] + t * t * c[i])
}

fn dist(p: [f64; 3], q: [f64; 3]) -> f64 {
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
}

/// independent f64 distance to the curve: 4096 samples, then golden-section on the best bracket
fn true_distance(a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3]) -> f64 {
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
    let (mut lo, mut hi) = (best.saturating_sub(1) as f64 / N as f64, (best + 1).min(N) as f64 / N as f64);
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

fn v3(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

fn r(p: [f64; 3]) -> [f64; 3] {
    p.map(|x| f64::from(x as f32))
}

#[test]
fn bezier_gpu_matches_cpu_and_never_overstates() {
    let mut compared = 0usize;
    let (mut worst_par, mut worst_over) = (0.0f64, 0.0f64);
    let mut ran = false;
    for &h in &[1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0] {
        for (a, b, c) in [
            ([0.0, 0.0, 0.0], [0.5, h, 0.0], [1.0, 0.0, 0.0]),
            ([0.0, 0.0, 0.0], [0.3, h, 0.0], [1.0, 0.0, 0.0]),
            ([3.1, 2.2, 0.4], [3.15, 2.2 + 0.5 * h, 0.4], [3.2, 2.2, 0.4]),
        ] {
            let (a, b, c) = (r(a), r(b), r(c));
            let node = SdfNode::bezier(v3(a), v3(b), v3(c), 0.0);
            let gpu = match GpuEvaluator::new(&node) {
                Ok(g) => g,
                Err(e) => {
                    assert!(
                        std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                        "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
                    );
                    eprintln!("skipping bezier GPU parity: {e}");
                    return;
                }
            };
            ran = true;
            let chord = dist(a, c);
            let (dx, dy) = ((c[0] - a[0]) / chord, (c[1] - a[1]) / chord);
            let mut pts64 = Vec::new();
            for k in 0..=40 {
                let q = curve(a, b, c, k as f64 / 40.0);
                for &o in &[0.0, 1e-4, 1e-3, 1e-2] {
                    for &s in &[1.0, -1.0] {
                        pts64.push(r([q[0] - dy * o * s * chord, q[1] + dx * o * s * chord, q[2]]));
                        pts64.push(r([q[0], q[1], q[2] + o * s * chord]));
                    }
                }
            }
            let pts: Vec<Vec3> = pts64.iter().map(|&p| v3(p)).collect();
            let got = gpu.eval_batch(&pts).expect("gpu eval");
            for ((p, p64), g) in pts.iter().zip(&pts64).zip(&got) {
                let cpu = f64::from(eval(&node, *p));
                let g = f64::from(*g);
                worst_par = worst_par.max((g - cpu).abs() / cpu.abs().max(1.0));
                worst_over = worst_over.max(g - true_distance(a, b, c, *p64));
                compared += 1;
            }
        }
    }
    if !ran {
        return;
    }
    assert!(compared > 0, "compared 0 points");
    eprintln!("bezier GPU: {compared} points, max |gpu - cpu| {worst_par:.3e}, worst GPU over-statement {worst_over:.3e}");
    assert!(worst_par <= 2e-5, "GPU/CPU drift {worst_par:.3e}");
    assert!(worst_over <= 1e-5, "the GPU over-states the distance by {worst_over:.3e}");
}

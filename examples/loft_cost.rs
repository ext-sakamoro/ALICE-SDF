//! Local measurement only: GPU cost per evaluation of a polar loft at body sizes
#![allow(clippy::disallowed_methods)]
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;
use std::time::Instant;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn loft(n_axis: usize, rows: usize, k: u32, radius: f32, length: f32) -> SdfNode {
    let axis: Vec<Vec3> = (0..n_axis)
        .map(|i| {
            let t = i as f32 / (n_axis - 1) as f32;
            Vec3::new(0.03 * (t * 6.0).sin(), t * length, 0.02 * (t * 4.0).cos())
        })
        .collect();
    let mut r = lcg(7);
    let w = 2 * k as usize + 1;
    let rows: Vec<Vec<f32>> = (0..rows)
        .map(|j| {
            let t = j as f32 / (rows - 1) as f32;
            let mut row = vec![0.0; w];
            row[0] = radius * (1.0 + 0.2 * (t * 5.0).sin());
            for h in 1..w {
                row[h] = radius * 0.02 * (r() - 0.5) / (1 + h / 2) as f32;
            }
            row
        })
        .collect();
    SdfNode::loft(axis, Vec3::Z, k, rows).expect("loft")
}

fn main() {
    let cases = [
        ("limb  n10  rows40  K16", 10, 40, 16, 0.05, 0.4),
        ("limb  n10  rows40  K32", 10, 40, 32, 0.05, 0.4),
        ("limb  n20  rows80  K64", 20, 80, 64, 0.05, 0.4),
        ("torso n20  rows140 K32", 20, 140, 32, 0.15, 0.6),
        ("torso n20  rows140 K64", 20, 140, 64, 0.15, 0.6),
        ("torso n70  rows140 K64", 70, 140, 64, 0.15, 0.6),
        ("torso n140 rows140 K64", 140, 140, 64, 0.15, 0.6),
    ];
    let mut r = lcg(1);
    let pts: Vec<Vec3> = (0..1_000_000)
        .map(|_| Vec3::new(r() * 0.6 - 0.3, r() * 0.8 - 0.1, r() * 0.6 - 0.3))
        .collect();
    for (name, n, rows, k, rad, len) in cases {
        let node = loft(n, rows, k, rad, len);
        let gpu = GpuEvaluator::new(&node).expect("gpu");
        let _ = gpu.eval_batch(&pts[..10_000]).expect("warm");
        let t = Instant::now();
        let reps = 3;
        for _ in 0..reps {
            let _ = gpu.eval_batch(&pts).expect("eval");
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (reps * pts.len()) as f64;
        // one frame at 949x862 with ~40 evals per pixel on average (march + normal)
        let frame_ms = ns * 949.0 * 862.0 * 40.0 / 1e6;
        println!("{name}: {ns:.1} ns/eval (GPU throughput), ~{frame_ms:.0} ms per 949x862 frame at 40 evals/px");
    }
}

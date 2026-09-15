//! Neural SDF oracle: the network is an approximation of a field with an
//! analytic answer, so the test measures it against that answer — RMSE on
//! held-out points, sign agreement away from the surface — and pins the
//! properties any trainer must have: determinism (same seed → identical
//! weights) and a lossless save / load round trip.
//!
//! Author: Moroya Sakamoto

use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
use alice_sdf::prelude::*;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn held_out(n: usize) -> Vec<Vec3> {
    let mut r = lcg(0xC0FFEE);
    (0..n)
        .map(|_| {
            Vec3::new(
                r().mul_add(3.0, -1.5),
                r().mul_add(3.0, -1.5),
                r().mul_add(3.0, -1.5),
            )
        })
        .collect()
}

/// 2 × 32 network, 1024-point batches: ~4 s per 60 epochs in a debug build,
/// which is what CI runs. The default learning rate (1e-2 since 1.13.0) is
/// part of what is under test — 1e-3 left this sphere at RMSE 0.21 after
/// 120 epochs, 1e-2 reaches 0.09.
fn small_config(seed: u64, epochs: usize) -> NeuralSdfConfig {
    NeuralSdfConfig {
        hidden_layers: 2,
        hidden_width: 32,
        epochs,
        batch_size: 1024,
        seed,
        ..Default::default()
    }
}

#[test]
fn trained_sphere_tracks_the_analytic_distance() {
    let node = SdfNode::sphere(1.0);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let nn = NeuralSdf::train(&node, lo, hi, &small_config(7, 120));
    let pts = held_out(4000);
    let mut sum2 = 0.0f32;
    let mut worst = 0.0f32;
    let mut sign_wrong = 0usize;
    let mut counted = 0usize;
    for &p in &pts {
        let truth = eval(&node, p);
        let got = nn.eval(p);
        let e = (got - truth).abs();
        sum2 += e * e;
        worst = worst.max(e);
        if truth.abs() > 0.15 {
            counted += 1;
            if (got > 0.0) != (truth > 0.0) {
                sign_wrong += 1;
            }
        }
    }
    let rmse = (sum2 / pts.len() as f32).sqrt();
    eprintln!("neural sphere: rmse {rmse:.4} worst {worst:.4} sign errors {sign_wrong}/{counted}");
    assert!(
        rmse < 0.12,
        "rmse {rmse} against the analytic sphere (measured 0.095)"
    );
    assert!(
        sign_wrong * 20 <= counted,
        "{sign_wrong} of {counted} points more than 0.15 from the surface have the wrong sign"
    );
    // more epochs must not make it worse (the optimiser converges): 60 → 120
    let nn_short = NeuralSdf::train(&node, lo, hi, &small_config(7, 60));
    let rmse_short = {
        let s: f32 = pts
            .iter()
            .map(|&p| (nn_short.eval(p) - eval(&node, p)).powi(2))
            .sum();
        (s / pts.len() as f32).sqrt()
    };
    eprintln!("neural sphere 60 epochs: rmse {rmse_short:.4}");
    assert!(
        rmse <= rmse_short,
        "120 epochs rmse {rmse} worse than 60 epochs {rmse_short}"
    );
}

#[test]
fn training_is_deterministic_for_a_seed() {
    let node = SdfNode::box3d(1.2, 0.8, 1.0)
        .smooth_union(SdfNode::sphere(0.7).translate(0.5, 0.5, 0.0), 0.2);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let a = NeuralSdf::train(&node, lo, hi, &small_config(3, 5));
    let b = NeuralSdf::train(&node, lo, hi, &small_config(3, 5));
    let c = NeuralSdf::train(&node, lo, hi, &small_config(4, 5));
    let mut differs_by_seed = false;
    for p in held_out(200) {
        assert_eq!(
            a.eval(p).to_bits(),
            b.eval(p).to_bits(),
            "same seed, different output at {p:?}"
        );
        if a.eval(p).to_bits() != c.eval(p).to_bits() {
            differs_by_seed = true;
        }
    }
    assert!(
        differs_by_seed,
        "different seeds produced identical networks"
    );
}

#[test]
fn save_load_round_trip_is_lossless() {
    let node = SdfNode::torus(0.8, 0.3);
    let nn = NeuralSdf::train(
        &node,
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &small_config(11, 3),
    );
    let mut buf = Vec::new();
    nn.save(&mut buf).expect("save");
    let back = NeuralSdf::load(&mut buf.as_slice()).expect("load");
    for p in held_out(500) {
        assert_eq!(
            nn.eval(p).to_bits(),
            back.eval(p).to_bits(),
            "round trip changed the output at {p:?}"
        );
    }
    let batch = back.eval_batch(&held_out(64));
    for (p, v) in held_out(64).iter().zip(batch) {
        assert_eq!(v.to_bits(), back.eval(*p).to_bits(), "eval_batch != eval");
    }
}

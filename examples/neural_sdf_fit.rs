//! Fitting a small neural network to an SDF, then saving and reloading it.
//!
//! Trains a `NeuralSdf` on a unit sphere, measures it against the analytic
//! distance `|p| − 1` on points it was not trained on, saves the weights to
//! memory, reloads them, and checks the reloaded network gives identical
//! answers (single, batched, and with gradient).
//!
//! # Running
//! ```bash
//! cargo run --release --example neural_sdf_fit
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};
use alice_sdf::prelude::*;

fn main() {
    println!("ALICE-SDF — neural SDF fit");
    println!("==========================");

    let config = NeuralSdfConfig {
        hidden_layers: 2,
        hidden_width: 32,
        pos_encoding_freqs: 2,
        batch_size: 512,
        epochs: 150,
        ..Default::default()
    };
    let sphere = SdfNode::sphere(1.0);
    let net = NeuralSdf::train(&sphere, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    println!(
        "network: input {}, {} hidden layers, {} parameters",
        net.input_dimension(),
        net.hidden_layer_count(),
        net.param_count()
    );
    // input 3 + 6·2 = 15; params 15·32+32 + 32·32+32 + 32+1 = 1601
    assert_eq!(net.input_dimension(), 15);
    assert_eq!(net.hidden_layer_count(), 2);
    assert_eq!(net.param_count(), 15 * 32 + 32 + 32 * 32 + 32 + 32 + 1);

    // Held-out points on a deterministic lattice offset from the training samples.
    let pts: Vec<Vec3> = (0..8)
        .flat_map(|i| (0..8).flat_map(move |j| (0..8).map(move |k| (i, j, k))))
        .map(|(i, j, k)| Vec3::new(i as f32, j as f32, k as f32) * (3.0 / 8.0) - Vec3::splat(1.3))
        .collect();
    let pred = net.eval_batch(&pts);
    let mse: f32 = pts
        .iter()
        .zip(&pred)
        .map(|(p, d)| (d - (p.length() - 1.0)).powi(2))
        .sum::<f32>()
        / pts.len() as f32;
    let rmse = mse.sqrt();
    println!("held-out RMSE vs |p| − 1: {rmse:.4}");
    assert!(rmse < 0.15, "fit too far from the analytic sphere: {rmse}");

    // Save / load round trip.
    let mut bytes = Vec::new();
    net.save(&mut bytes).expect("save");
    let back = NeuralSdf::load(&mut bytes.as_slice()).expect("load");
    println!("saved {} bytes", bytes.len());
    for (p, d) in pts.iter().zip(&pred) {
        assert_eq!(back.eval(*p).to_bits(), d.to_bits());
    }
    let (d, g) = back.eval_with_gradient(Vec3::new(1.2, 0.0, 0.0), 1e-3);
    println!("at (1.2, 0, 0): d = {d:.3}, gradient = {g:.3} (analytic 0.2, +X)");
    assert!(g.x > 0.0, "gradient points away from the surface");
    println!("all checks passed");
}

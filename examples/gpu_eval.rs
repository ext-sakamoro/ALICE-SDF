//! GPU Evaluation Example
//!
//! This example demonstrates how to use WebGPU compute shaders
//! for massively parallel SDF evaluation.
//!
//! # Requirements
//! - Build with `--features gpu`
//! - WebGPU-capable GPU (Metal, Vulkan, DX12, or WebGPU)
//!
//! # Running
//! ```bash
//! cargo run --example gpu_eval --features gpu
//! ```
//!
//! Author: Moroya Sakamoto

#[allow(unused_imports)]
use alice_sdf::prelude::*;
#[allow(unused_imports)]
use std::time::Instant;

#[cfg(feature = "gpu")]
use alice_sdf::compiled::{GpuEvaluator, WgslShader};

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("This example requires the 'gpu' feature.");
        eprintln!("Run with: cargo run --example gpu_eval --features gpu");
        std::process::exit(1);
    }

    #[cfg(feature = "gpu")]
    run_gpu_example();
}

#[cfg(feature = "gpu")]
fn run_gpu_example() {
    println!("=== ALICE-SDF GPU Evaluation Example ===\n");

    // Create a complex SDF shape
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .twist(0.1)
        .translate(0.5, 0.0, 0.0);

    println!("Shape: Sphere + Box - Cylinder with Twist");
    println!("Node count: {}\n", shape.node_count());

    // === Method 1: View Generated WGSL ===
    println!("--- Generated WGSL Shader ---");
    let shader = WgslShader::transpile(&shape);
    println!("WGSL source length: {} bytes", shader.source.len());
    println!("First 500 chars:\n{}\n...\n", &shader.source[..500.min(shader.source.len())]);

    // === Method 2: Create GPU Evaluator ===
    println!("--- Creating GPU Evaluator ---");
    let start = Instant::now();
    let gpu = match GpuEvaluator::new(&shape) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Failed to create GPU evaluator: {}", e);
            eprintln!("Make sure you have a WebGPU-capable GPU.");
            std::process::exit(1);
        }
    };
    println!("GPU initialization: {:?}\n", start.elapsed());

    // === Method 3: Evaluate Single Point ===
    println!("--- Single Point Evaluation ---");
    let point = Vec3::new(0.5, 0.5, 0.5);

    // CPU evaluation for comparison
    let cpu_distance = eval(&shape, point);
    println!("CPU distance at {:?}: {:.6}", point, cpu_distance);

    // GPU evaluation (batch of 1)
    let gpu_distances = gpu.eval_batch(&[point]).unwrap();
    println!("GPU distance at {:?}: {:.6}", point, gpu_distances[0]);
    println!("Difference: {:.9}\n", (cpu_distance - gpu_distances[0]).abs());

    // === Method 4: Batch Evaluation ===
    println!("--- Batch Evaluation Comparison ---");

    for batch_size in [1_000, 10_000, 100_000, 1_000_000] {
        let points: Vec<Vec3> = (0..batch_size)
            .map(|i| {
                let t = i as f32 / batch_size as f32;
                Vec3::new(
                    (t * 123.456).sin() * 2.0,
                    (t * 234.567).sin() * 2.0,
                    (t * 345.678).sin() * 2.0,
                )
            })
            .collect();

        // CPU evaluation
        let start = Instant::now();
        let _cpu_results = eval_batch_parallel(&shape, &points);
        let cpu_time = start.elapsed();

        // GPU evaluation
        let start = Instant::now();
        let _gpu_results = gpu.eval_batch(&points).unwrap();
        let gpu_time = start.elapsed();

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        let winner = if speedup > 1.0 { "GPU" } else { "CPU" };

        println!(
            "Batch {:>7}: CPU {:>8.2?} | GPU {:>8.2?} | {:.2}x ({} wins)",
            batch_size, cpu_time, gpu_time, speedup.max(1.0 / speedup), winner
        );
    }

    println!("\n--- Crossover Analysis ---");
    println!("GPU typically wins when batch_size > ~5,000-10,000 points");
    println!("For smaller batches, CPU SIMD is faster due to GPU dispatch overhead\n");

    // === Method 5: Accuracy Verification ===
    println!("--- Accuracy Verification ---");
    let test_points: Vec<Vec3> = (0..100)
        .map(|i| {
            let t = i as f32 / 100.0;
            Vec3::new(t * 4.0 - 2.0, t * 4.0 - 2.0, t * 4.0 - 2.0)
        })
        .collect();

    let cpu_results: Vec<f32> = test_points.iter().map(|&p| eval(&shape, p)).collect();
    let gpu_results = gpu.eval_batch(&test_points).unwrap();

    let max_error: f32 = cpu_results
        .iter()
        .zip(gpu_results.iter())
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);

    let avg_error: f32 = cpu_results
        .iter()
        .zip(gpu_results.iter())
        .map(|(c, g)| (c - g).abs())
        .sum::<f32>() / test_points.len() as f32;

    println!("Max error: {:.9}", max_error);
    println!("Avg error: {:.9}", avg_error);
    println!("Status: {}", if max_error < 0.001 { "PASS ✓" } else { "FAIL ✗" });

    println!("\n=== Example Complete ===");
}

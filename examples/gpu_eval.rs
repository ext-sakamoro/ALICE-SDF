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
use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};

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
    let shader = WgslShader::transpile(&shape, TranspileMode::Hardcoded);
    println!("WGSL source length: {} bytes", shader.source.len());
    println!(
        "First 500 chars:\n{}\n...\n",
        &shader.source[..500.min(shader.source.len())]
    );

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
    println!(
        "Difference: {:.9}\n",
        (cpu_distance - gpu_distances[0]).abs()
    );

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
            batch_size,
            cpu_time,
            gpu_time,
            speedup.max(1.0 / speedup),
            winner
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
        .sum::<f32>()
        / test_points.len() as f32;

    println!("Max error: {:.9}", max_error);
    println!("Avg error: {:.9}", avg_error);
    println!(
        "Status: {}",
        if max_error < 0.001 {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );

    run_api_tour();

    println!("\n=== Example Complete ===");
}

/// Closed form of `sphere(r).translate(c)`.
#[cfg(feature = "gpu")]
fn ball(r: f32, c: Vec3, p: Vec3) -> f32 {
    (p - c).length() - r
}

#[cfg(feature = "gpu")]
fn assert_close(what: &str, got: &[f32], want: impl Iterator<Item = f32>) {
    let mut worst = 0.0f32;
    let mut n = 0usize;
    for (g, w) in got.iter().zip(want) {
        worst = worst.max((g - w).abs());
        n += 1;
    }
    assert_eq!(n, got.len(), "{what}: result count");
    assert!(n > 0, "{what}: nothing compared");
    assert!(worst < 1e-4, "{what}: max |gpu - closed form| = {worst}");
    println!("{what:<34} {n:>7} points, max error {worst:.2e}");
}

/// The rest of the GpuEvaluator API, each result checked against the closed
/// form of a translated sphere.
#[cfg(feature = "gpu")]
fn run_api_tour() {
    use alice_sdf::compiled::{GpuBufferPool, GpuEvalFuture};

    println!("\n--- API tour (checked against |p - c| - r) ---");
    let (r, c) = (1.0f32, Vec3::new(0.25, -0.5, 0.125));
    let node = SdfNode::sphere(r).translate(c.x, c.y, c.z);
    let pts: Vec<Vec3> = (0..2000)
        .map(|i| {
            let t = i as f32 * 0.618_034;
            Vec3::new(t.sin() * 2.0, (t * 1.7).cos() * 2.0, (t * 2.3).sin() * 2.0)
        })
        .collect();
    let want = |r: f32, c: Vec3| pts.iter().map(move |&p| ball(r, c, p));

    // Persistent buffers: allocated once, grown on demand, chunked above 256K.
    let gpu = GpuEvaluator::new(&node).expect("GPU evaluator");
    let mut pool: GpuBufferPool = gpu.create_buffer_pool(512);
    let d = gpu.eval_batch_pooled(&pts, &mut pool).unwrap();
    assert_close("eval_batch_pooled", &d, want(r, c));
    println!("pool capacity after 2000 points: {}", pool.capacity);
    let d = gpu.eval_batch_auto(&pts, &mut pool).unwrap();
    assert_close("eval_batch_auto", &d, want(r, c));

    // A device-dependent workgroup size; every entry point dispatches with it.
    let shader = WgslShader::transpile(&node, TranspileMode::Hardcoded).with_workgroup_size(64);
    let gpu64 = GpuEvaluator::from_shader(&shader).expect("GPU evaluator (wg 64)");
    let d = gpu64
        .eval_batch_pooled(&pts, &mut gpu64.create_buffer_pool(0))
        .unwrap();
    assert_close("workgroup 64, pooled", &d, want(r, c));

    // Dynamic parameters: change the radius and the centre without rebuilding
    // the pipeline, and get GPU normals (tetrahedral difference) as well.
    let dynamic = GpuEvaluator::new_dynamic(&node).expect("dynamic GPU evaluator");
    let full = dynamic.eval_batch_full(&pts).unwrap();
    let d: Vec<f32> = full.iter().map(|&(d, _)| d).collect();
    assert_close("new_dynamic + eval_batch_full", &d, want(r, c));
    let worst_normal = pts
        .iter()
        .zip(&full)
        .filter(|(&p, _)| (p - c).length() > 0.5)
        .map(|(&p, &(_, n))| (n - (p - c).normalize()).length())
        .fold(0.0f32, f32::max);
    assert!(worst_normal < 5e-3, "GPU normal error {worst_normal}");
    println!("GPU normals: max |n - (p-c)/|p-c|| = {worst_normal:.2e}");
    let (r2, c2) = (1.5f32, Vec3::new(-0.5, 0.25, 0.0));
    let moved = SdfNode::sphere(r2).translate(c2.x, c2.y, c2.z);
    println!(
        "parameters after the move: {:?}",
        WgslShader::extract_params(&moved)
    );
    dynamic.update_params(&moved);
    assert_close(
        "update_params, eval_batch",
        &dynamic.eval_batch(&pts).unwrap(),
        want(r2, c2),
    );
    let normals_wgsl =
        WgslShader::transpile(&node, TranspileMode::Dynamic).to_compute_shader_with_normals();
    println!(
        "distance + normal compute shader: {} bytes",
        normals_wgsl.len()
    );

    // Async construction and evaluation (any executor; pollster here).
    let gpu_async =
        pollster::block_on(GpuEvaluator::new_async(&node)).expect("async GPU evaluator");
    let d = pollster::block_on(gpu_async.eval_batch_async(&pts)).unwrap();
    assert_close("new_async + eval_batch_async", &d, want(r, c));
    // `eval_batch_submit` defers the evaluation to `wait` / `resolve`.
    let future: GpuEvalFuture = gpu_async.eval_batch_submit(pts.clone());
    let d = future.wait().unwrap();
    assert_close("eval_batch_submit + wait", &d, want(r, c));
    let d = pollster::block_on(gpu_async.eval_batch_submit(pts.clone()).resolve()).unwrap();
    assert_close("eval_batch_submit + resolve", &d, want(r, c));
    let from_shader = pollster::block_on(GpuEvaluator::from_shader_async(&shader)).unwrap();
    assert_close(
        "from_shader_async",
        &from_shader.eval_batch(&pts).unwrap(),
        want(r, c),
    );
    let compute = WgslShader::transpile(&moved, TranspileMode::Hardcoded).to_compute_shader();
    let from_wgsl = pollster::block_on(GpuEvaluator::from_wgsl_async(&compute)).unwrap();
    assert_close(
        "from_wgsl_async",
        &from_wgsl.eval_batch(&pts).unwrap(),
        want(r2, c2),
    );

    // Material ids: `sdf_eval_material(p)` returns the id of the nearest
    // `with_material` subtree.
    let materials = SdfNode::sphere(0.5)
        .translate(-1.0, 0.0, 0.0)
        .with_material(3)
        .union(
            SdfNode::sphere(0.5)
                .translate(1.0, 0.0, 0.0)
                .with_material(7),
        );
    let material_fn = WgslShader::transpile_material(&materials, TranspileMode::Hardcoded);
    assert!(material_fn.contains("fn sdf_eval_material"));
    println!("material function: {} bytes", material_fn.len());

    // The GLSL transpiler's sdf_eval, run through naga's GLSL front end.
    #[cfg(feature = "glsl")]
    {
        use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
        let glsl = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        let src = format!(
            "#version 450\n\
             layout(local_size_x = 256) in;\n\
             struct InputPoint {{ float x; float y; float z; float pad; }};\n\
             struct OutputDistance {{ float distance; float pad1; float pad2; float pad3; }};\n\
             layout(std430, set = 0, binding = 0) readonly buffer InputPoints {{ InputPoint input_points[]; }};\n\
             layout(std430, set = 0, binding = 1) buffer OutputDistances {{ OutputDistance output_distances[]; }};\n\
             layout(std140, set = 0, binding = 2) uniform PointCount {{ uint point_count; }};\n\
             {}\n\
             void main() {{\n\
                 uint idx = gl_GlobalInvocationID.x;\n\
                 if (idx >= point_count) {{ return; }}\n\
                 InputPoint pt = input_points[idx];\n\
                 output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));\n\
             }}\n",
            glsl.get_eval_function()
        );
        let gpu_glsl = GpuEvaluator::from_glsl_compute(&src).expect("GLSL compute shader");
        assert_close(
            "from_glsl_compute",
            &gpu_glsl.eval_batch(&pts).unwrap(),
            want(r, c),
        );
    }
}

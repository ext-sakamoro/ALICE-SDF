//! GPU vs CPU Benchmark: Crossover Point Analysis
//!
//! This benchmark compares different evaluation modes to find the
//! optimal crossover point where GPU becomes faster than CPU.
//!
//! # Modes Compared
//! - CPU Interpreted: Recursive tree evaluation
//! - CPU Compiled: Stack machine evaluation
//! - CPU SIMD: 8-wide AVX2/NEON evaluation
//! - CPU SIMD Parallel: SIMD + Rayon multi-threading
//! - GPU Compute: WebGPU compute shader
//!
//! # Expected Results
//! - Small batches (< 1K): CPU wins (GPU dispatch overhead)
//! - Medium batches (1K-10K): Crossover zone
//! - Large batches (> 10K): GPU wins (massive parallelism)
//!
//! Author: Moroya Sakamoto

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use alice_sdf::prelude::*;
use alice_sdf::compiled::{
    CompiledSdf,
    eval_compiled_batch_parallel,
    eval_compiled_batch_simd_parallel,
};
use alice_sdf::soa::SoAPoints;

#[cfg(feature = "gpu")]
use alice_sdf::compiled::GpuEvaluator;

#[cfg(feature = "jit")]
use alice_sdf::compiled::jit::JitSimdSdf;

/// Generate random-ish test points
fn generate_points(count: usize) -> Vec<Vec3> {
    (0..count)
        .map(|i| {
            let t = i as f32 / count as f32;
            Vec3::new(
                (t * 123.456).sin() * 2.0,
                (t * 234.567).sin() * 2.0,
                (t * 345.678).sin() * 2.0,
            )
        })
        .collect()
}

/// Benchmark: GPU vs CPU crossover point (Deep Fried Edition)
fn bench_crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");

    // Test shape: medium complexity
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .twist(0.1)
        .translate(0.5, 0.0, 0.0);

    let compiled = CompiledSdf::compile(&shape);

    #[cfg(feature = "jit")]
    let jit = JitSimdSdf::compile(&compiled).expect("Failed to JIT compile");

    #[cfg(feature = "gpu")]
    let gpu = GpuEvaluator::new(&shape).expect("Failed to create GPU evaluator");

    // Test batch sizes: 100 to 1M
    for size in [100, 1_000, 10_000, 100_000, 1_000_000] {
        let points = generate_points(size);
        let soa = SoAPoints::from_vec3_slice(&points);

        group.throughput(Throughput::Elements(size as u64));

        // CPU Interpreted (parallel)
        group.bench_with_input(
            BenchmarkId::new("cpu_interpreted", size),
            &points,
            |b, points| {
                b.iter(|| eval_batch_parallel(black_box(&shape), black_box(points)))
            },
        );

        // CPU Compiled (parallel)
        group.bench_with_input(
            BenchmarkId::new("cpu_compiled", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_parallel(black_box(&compiled), black_box(points)))
            },
        );

        // CPU SIMD (parallel)
        group.bench_with_input(
            BenchmarkId::new("cpu_simd", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_simd_parallel(black_box(&compiled), black_box(points)))
            },
        );

        // CPU JIT SIMD (parallel)
        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("cpu_jit_simd", size),
            &soa,
            |b, soa| {
                b.iter(|| jit.eval_soa(black_box(soa)))
            },
        );

        // GPU Compute
        #[cfg(feature = "gpu")]
        group.bench_with_input(
            BenchmarkId::new("gpu_compute", size),
            &points,
            |b, points| {
                b.iter(|| gpu.eval_batch(black_box(points)).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark: Simple shape (Sphere) - GPU overhead more visible
fn bench_simple_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_shape");

    let shape = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&shape);

    #[cfg(feature = "jit")]
    let jit = JitSimdSdf::compile(&compiled).expect("Failed to JIT compile");

    #[cfg(feature = "gpu")]
    let gpu = GpuEvaluator::new(&shape).expect("Failed to create GPU evaluator");

    for size in [1_000, 10_000, 100_000] {
        let points = generate_points(size);
        let soa = SoAPoints::from_vec3_slice(&points);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu_simd", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_simd_parallel(black_box(&compiled), black_box(points)))
            },
        );

        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("cpu_jit_simd", size),
            &soa,
            |b, soa| {
                b.iter(|| jit.eval_soa(black_box(soa)))
            },
        );

        #[cfg(feature = "gpu")]
        group.bench_with_input(
            BenchmarkId::new("gpu_compute", size),
            &points,
            |b, points| {
                b.iter(|| gpu.eval_batch(black_box(points)).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark: Complex shape - GPU benefits more
fn bench_complex_shape(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_shape");

    // Very complex shape: 20+ nodes
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0).rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            0.1,
        )
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0).rotate_euler(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            0.1,
        )
        .smooth_union(
            SdfNode::torus(0.8, 0.2).translate(0.0, 1.0, 0.0),
            0.1,
        )
        .twist(0.1)
        .round(0.02)
        .translate(0.0, 0.5, 0.0);

    let compiled = CompiledSdf::compile(&shape);

    #[cfg(feature = "jit")]
    let jit = JitSimdSdf::compile(&compiled).expect("Failed to JIT compile");

    #[cfg(feature = "gpu")]
    let gpu = GpuEvaluator::new(&shape).expect("Failed to create GPU evaluator");

    for size in [1_000, 10_000, 100_000] {
        let points = generate_points(size);
        let soa = SoAPoints::from_vec3_slice(&points);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("cpu_simd", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_simd_parallel(black_box(&compiled), black_box(points)))
            },
        );

        #[cfg(feature = "jit")]
        group.bench_with_input(
            BenchmarkId::new("cpu_jit_simd", size),
            &soa,
            |b, soa| {
                b.iter(|| jit.eval_soa(black_box(soa)))
            },
        );

        #[cfg(feature = "gpu")]
        group.bench_with_input(
            BenchmarkId::new("gpu_compute", size),
            &points,
            |b, points| {
                b.iter(|| gpu.eval_batch(black_box(points)).unwrap())
            },
        );
    }

    group.finish();
}

/// Benchmark: GPU initialization overhead
#[cfg(feature = "gpu")]
fn bench_gpu_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_init");

    let simple = SdfNode::sphere(1.0);
    let complex = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .twist(0.2)
        .translate(1.0, 0.0, 0.0);

    group.bench_function("simple_shape", |b| {
        b.iter(|| GpuEvaluator::new(black_box(&simple)).unwrap())
    });

    group.bench_function("complex_shape", |b| {
        b.iter(|| GpuEvaluator::new(black_box(&complex)).unwrap())
    });

    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(
    benches,
    bench_crossover,
    bench_simple_shape,
    bench_complex_shape,
    bench_gpu_init,
);

#[cfg(not(feature = "gpu"))]
criterion_group!(
    benches,
    bench_crossover,
    bench_simple_shape,
    bench_complex_shape,
);

criterion_main!(benches);

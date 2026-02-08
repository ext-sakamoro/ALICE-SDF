//! Benchmarks for SDF evaluation
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_parallel, eval_compiled_batch_simd,
    eval_compiled_batch_simd_parallel, eval_compiled_batch_soa_parallel, eval_compiled_bvh,
    eval_compiled_simd, CompiledSdf, CompiledSdfBvh, Vec3x8,
};
use alice_sdf::prelude::*;
use alice_sdf::soa::SoAPoints;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "jit")]
use alice_sdf::compiled::jit::JitSimdSdf;

fn bench_primitives(c: &mut Criterion) {
    let mut group = c.benchmark_group("primitives");

    let point = Vec3::new(0.5, 0.5, 0.5);

    group.bench_function("sphere", |b| {
        let sphere = SdfNode::sphere(1.0);
        b.iter(|| eval(black_box(&sphere), black_box(point)))
    });

    group.bench_function("box3d", |b| {
        let box3d = SdfNode::box3d(1.0, 1.0, 1.0);
        b.iter(|| eval(black_box(&box3d), black_box(point)))
    });

    group.bench_function("cylinder", |b| {
        let cylinder = SdfNode::cylinder(0.5, 1.0);
        b.iter(|| eval(black_box(&cylinder), black_box(point)))
    });

    group.bench_function("torus", |b| {
        let torus = SdfNode::torus(1.0, 0.3);
        b.iter(|| eval(black_box(&torus), black_box(point)))
    });

    group.finish();
}

fn bench_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("operations");

    let point = Vec3::new(0.5, 0.5, 0.5);
    let a = SdfNode::sphere(1.0);
    let b = SdfNode::box3d(1.0, 1.0, 1.0);

    group.bench_function("union", |b_iter| {
        let union = a.clone().union(b.clone());
        b_iter.iter(|| eval(black_box(&union), black_box(point)))
    });

    group.bench_function("intersection", |b_iter| {
        let intersection = a.clone().intersection(b.clone());
        b_iter.iter(|| eval(black_box(&intersection), black_box(point)))
    });

    group.bench_function("subtraction", |b_iter| {
        let subtraction = a.clone().subtract(b.clone());
        b_iter.iter(|| eval(black_box(&subtraction), black_box(point)))
    });

    group.bench_function("smooth_union", |b_iter| {
        let smooth = a.clone().smooth_union(b.clone(), 0.2);
        b_iter.iter(|| eval(black_box(&smooth), black_box(point)))
    });

    group.finish();
}

fn bench_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("transforms");

    let point = Vec3::new(0.5, 0.5, 0.5);
    let base = SdfNode::sphere(1.0);

    group.bench_function("translate", |b| {
        let translated = base.clone().translate(1.0, 2.0, 3.0);
        b.iter(|| eval(black_box(&translated), black_box(point)))
    });

    group.bench_function("rotate", |b| {
        let rotated = base.clone().rotate_euler(0.5, 0.5, 0.5);
        b.iter(|| eval(black_box(&rotated), black_box(point)))
    });

    group.bench_function("scale", |b| {
        let scaled = base.clone().scale(2.0);
        b.iter(|| eval(black_box(&scaled), black_box(point)))
    });

    group.finish();
}

fn bench_modifiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("modifiers");

    let point = Vec3::new(0.5, 0.5, 0.5);
    let base = SdfNode::box3d(1.0, 2.0, 1.0);

    group.bench_function("twist", |b| {
        let twisted = base.clone().twist(0.5);
        b.iter(|| eval(black_box(&twisted), black_box(point)))
    });

    group.bench_function("bend", |b| {
        let bent = base.clone().bend(0.3);
        b.iter(|| eval(black_box(&bent), black_box(point)))
    });

    group.bench_function("repeat_infinite", |b| {
        let repeated = SdfNode::sphere(0.3).repeat_infinite(1.0, 1.0, 1.0);
        b.iter(|| eval(black_box(&repeated), black_box(point)))
    });

    group.bench_function("noise", |b| {
        let noisy = base.clone().noise(0.1, 1.0, 42);
        b.iter(|| eval(black_box(&noisy), black_box(point)))
    });

    group.finish();
}

fn bench_complex_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_tree");

    let point = Vec3::new(0.5, 0.5, 0.5);

    // Tree with 5 nodes
    let tree_5 = SdfNode::sphere(1.0)
        .subtract(SdfNode::box3d(0.8, 0.8, 0.8))
        .translate(1.0, 0.0, 0.0);

    // Tree with 10 nodes
    let tree_10 = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .subtract(SdfNode::cylinder(0.3, 2.0))
        .twist(0.2)
        .translate(1.0, 0.0, 0.0)
        .scale(1.5);

    // Tree with 20+ nodes
    let tree_20 = SdfNode::sphere(1.0)
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
        .twist(0.1)
        .round(0.02)
        .translate(0.0, 0.5, 0.0);

    group.bench_function("5_nodes", |b| {
        b.iter(|| eval(black_box(&tree_5), black_box(point)))
    });

    group.bench_function("10_nodes", |b| {
        b.iter(|| eval(black_box(&tree_10), black_box(point)))
    });

    group.bench_function("20_nodes", |b| {
        b.iter(|| eval(black_box(&tree_20), black_box(point)))
    });

    group.finish();
}

fn bench_batch_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_eval");

    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2)
        .twist(0.3);

    for size in [100, 1000, 10000, 100000] {
        let points: Vec<Vec3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Vec3::new(
                    (t * 123.456).sin() * 2.0,
                    (t * 234.567).sin() * 2.0,
                    (t * 345.678).sin() * 2.0,
                )
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("parallel", size), &points, |b, points| {
            b.iter(|| eval_batch_parallel(black_box(&shape), black_box(points)))
        });
    }

    group.finish();
}

fn bench_raymarching(c: &mut Criterion) {
    let mut group = c.benchmark_group("raymarching");

    let sphere = SdfNode::sphere(1.0);
    let complex = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .twist(0.2);

    let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

    group.bench_function("sphere", |b| {
        b.iter(|| raycast(black_box(&sphere), black_box(ray), 10.0))
    });

    group.bench_function("complex", |b| {
        b.iter(|| raycast(black_box(&complex), black_box(ray), 10.0))
    });

    group.finish();
}

fn bench_marching_cubes(c: &mut Criterion) {
    let mut group = c.benchmark_group("marching_cubes");
    group.sample_size(10); // Fewer samples for slow benchmarks

    let sphere = SdfNode::sphere(1.0);
    let min = Vec3::splat(-2.0);
    let max = Vec3::splat(2.0);

    for res in [16, 32, 64] {
        let config = MarchingCubesConfig {
            resolution: res,
            iso_level: 0.0,
            compute_normals: true,
        };

        group.bench_with_input(BenchmarkId::new("resolution", res), &config, |b, config| {
            b.iter(|| sdf_to_mesh(black_box(&sphere), min, max, config))
        });
    }

    group.finish();
}

/// Benchmark: Interpreted vs Compiled SDF evaluation
fn bench_interpreted_vs_compiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpreted_vs_compiled");

    let point = Vec3::new(0.5, 0.5, 0.5);

    // Simple shape: single sphere
    let simple = SdfNode::sphere(1.0);
    let simple_compiled = CompiledSdf::compile(&simple);

    group.bench_function("simple/interpreted", |b| {
        b.iter(|| eval(black_box(&simple), black_box(point)))
    });

    group.bench_function("simple/compiled", |b| {
        b.iter(|| eval_compiled(black_box(&simple_compiled), black_box(point)))
    });

    // Medium complexity: union + translate
    let medium = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .translate(1.0, 0.0, 0.0);
    let medium_compiled = CompiledSdf::compile(&medium);

    group.bench_function("medium/interpreted", |b| {
        b.iter(|| eval(black_box(&medium), black_box(point)))
    });

    group.bench_function("medium/compiled", |b| {
        b.iter(|| eval_compiled(black_box(&medium_compiled), black_box(point)))
    });

    // Complex shape: many operations
    let complex = SdfNode::sphere(1.0)
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
        .twist(0.1)
        .round(0.02)
        .translate(0.0, 0.5, 0.0);
    let complex_compiled = CompiledSdf::compile(&complex);

    group.bench_function("complex/interpreted", |b| {
        b.iter(|| eval(black_box(&complex), black_box(point)))
    });

    group.bench_function("complex/compiled", |b| {
        b.iter(|| eval_compiled(black_box(&complex_compiled), black_box(point)))
    });

    group.finish();
}

/// Benchmark: Batch evaluation (interpreted vs compiled)
fn bench_batch_compiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_compiled");

    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2)
        .twist(0.3);
    let compiled = CompiledSdf::compile(&shape);

    let size = 100000;
    let points: Vec<Vec3> = (0..size)
        .map(|i| {
            let t = i as f32 / size as f32;
            Vec3::new(
                (t * 123.456).sin() * 2.0,
                (t * 234.567).sin() * 2.0,
                (t * 345.678).sin() * 2.0,
            )
        })
        .collect();

    group.bench_function("interpreted_parallel", |b| {
        b.iter(|| eval_batch_parallel(black_box(&shape), black_box(&points)))
    });

    group.bench_function("compiled_parallel", |b| {
        b.iter(|| eval_compiled_batch_parallel(black_box(&compiled), black_box(&points)))
    });

    group.finish();
}

/// Benchmark: SIMD 8-wide evaluation
fn bench_simd_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_evaluation");

    // Complex shape for meaningful comparison
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .translate(0.5, 0.0, 0.0);
    let compiled = CompiledSdf::compile(&shape);

    // 8 test points
    let points_8 = Vec3x8::from_vecs([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(1.5, 0.0, 0.0),
        Vec3::new(0.0, 1.5, 0.0),
    ]);

    group.bench_function("8_points_scalar", |b| {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(1.5, 0.0, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
        ];
        b.iter(|| {
            let mut sum = 0.0f32;
            for p in &pts {
                sum += eval_compiled(black_box(&compiled), black_box(*p));
            }
            sum
        })
    });

    group.bench_function("8_points_simd", |b| {
        b.iter(|| eval_compiled_simd(black_box(&compiled), black_box(points_8)))
    });

    group.finish();
}

/// Benchmark: Batch evaluation comparison (interpreted vs compiled vs SIMD)
fn bench_batch_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_all");

    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2)
        .twist(0.3);
    let compiled = CompiledSdf::compile(&shape);

    for size in [1000, 10000, 100000] {
        let points: Vec<Vec3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Vec3::new(
                    (t * 123.456).sin() * 2.0,
                    (t * 234.567).sin() * 2.0,
                    (t * 345.678).sin() * 2.0,
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("interpreted_parallel", size),
            &points,
            |b, points| b.iter(|| eval_batch_parallel(black_box(&shape), black_box(points))),
        );

        group.bench_with_input(
            BenchmarkId::new("compiled_parallel", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_parallel(black_box(&compiled), black_box(points)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_batch", size),
            &points,
            |b, points| {
                b.iter(|| eval_compiled_batch_simd(black_box(&compiled), black_box(points)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("simd_parallel", size),
            &points,
            |b, points| {
                b.iter(|| {
                    eval_compiled_batch_simd_parallel(black_box(&compiled), black_box(points))
                })
            },
        );
    }

    group.finish();
}

/// Benchmark: BVH evaluation (sparse scene with distant objects)
fn bench_bvh_sparse_scene(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_sparse_scene");

    // Create a sparse scene: multiple spheres spread far apart
    let mut shape = SdfNode::sphere(0.5);
    for i in 1..10 {
        let offset = i as f32 * 5.0; // 5 units apart
        shape = shape.union(SdfNode::sphere(0.5).translate(offset, 0.0, 0.0));
    }

    let compiled = CompiledSdf::compile(&shape);
    let compiled_bvh = CompiledSdfBvh::compile(&shape);

    // Test point near the first sphere (should be fast for BVH)
    let point_near = Vec3::new(0.0, 0.0, 0.0);
    // Test point in the middle (needs to check more spheres)
    let point_middle = Vec3::new(22.5, 0.0, 0.0);
    // Test point far from all (BVH should help)
    let point_far = Vec3::new(0.0, 10.0, 0.0);

    group.bench_function("interpreted/near", |b| {
        b.iter(|| eval(black_box(&shape), black_box(point_near)))
    });

    group.bench_function("compiled/near", |b| {
        b.iter(|| eval_compiled(black_box(&compiled), black_box(point_near)))
    });

    group.bench_function("bvh/near", |b| {
        b.iter(|| eval_compiled_bvh(black_box(&compiled_bvh), black_box(point_near)))
    });

    group.bench_function("interpreted/middle", |b| {
        b.iter(|| eval(black_box(&shape), black_box(point_middle)))
    });

    group.bench_function("compiled/middle", |b| {
        b.iter(|| eval_compiled(black_box(&compiled), black_box(point_middle)))
    });

    group.bench_function("bvh/middle", |b| {
        b.iter(|| eval_compiled_bvh(black_box(&compiled_bvh), black_box(point_middle)))
    });

    group.bench_function("interpreted/far", |b| {
        b.iter(|| eval(black_box(&shape), black_box(point_far)))
    });

    group.bench_function("compiled/far", |b| {
        b.iter(|| eval_compiled(black_box(&compiled), black_box(point_far)))
    });

    group.bench_function("bvh/far", |b| {
        b.iter(|| eval_compiled_bvh(black_box(&compiled_bvh), black_box(point_far)))
    });

    group.finish();
}

/// Benchmark: BVH with complex nested transforms
fn bench_bvh_complex(c: &mut Criterion) {
    let mut group = c.benchmark_group("bvh_complex");

    // Complex scene with transforms
    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .translate(5.0, 0.0, 0.0)
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0)
                .rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0)
                .translate(-5.0, 0.0, 0.0),
            0.2,
        )
        .smooth_union(SdfNode::torus(1.0, 0.3).translate(0.0, 5.0, 0.0), 0.1);

    let compiled = CompiledSdf::compile(&shape);
    let compiled_bvh = CompiledSdfBvh::compile(&shape);

    let point = Vec3::new(0.0, 0.0, 0.0);

    group.bench_function("interpreted", |b| {
        b.iter(|| eval(black_box(&shape), black_box(point)))
    });

    group.bench_function("compiled", |b| {
        b.iter(|| eval_compiled(black_box(&compiled), black_box(point)))
    });

    group.bench_function("bvh", |b| {
        b.iter(|| eval_compiled_bvh(black_box(&compiled_bvh), black_box(point)))
    });

    group.finish();
}

/// Benchmark: SoA throughput (Deep Fried Edition)
fn bench_soa_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("soa_throughput");

    let shape = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .translate(0.5, 0.0, 0.0);

    let compiled = CompiledSdf::compile(&shape);

    #[cfg(feature = "jit")]
    let jit = JitSimdSdf::compile(&compiled).expect("Failed to JIT compile");

    for size in [10_000, 100_000, 1_000_000] {
        let points: Vec<Vec3> = (0..size)
            .map(|i| {
                let t = i as f32 / size as f32;
                Vec3::new(
                    (t * 123.456).sin() * 2.0,
                    (t * 234.567).sin() * 2.0,
                    (t * 345.678).sin() * 2.0,
                )
            })
            .collect();
        let soa = SoAPoints::from_vec3_slice(&points);

        group.throughput(Throughput::Elements(size as u64));

        // AoS SIMD (standard)
        group.bench_with_input(BenchmarkId::new("aos_simd", size), &points, |b, points| {
            b.iter(|| eval_compiled_batch_simd_parallel(black_box(&compiled), black_box(points)))
        });

        // SoA SIMD (optimized layout)
        group.bench_with_input(BenchmarkId::new("soa_simd", size), &soa, |b, soa| {
            b.iter(|| eval_compiled_batch_soa_parallel(black_box(&compiled), black_box(soa)))
        });

        // JIT + SoA (maximum throughput)
        #[cfg(feature = "jit")]
        group.bench_with_input(BenchmarkId::new("jit_soa", size), &soa, |b, soa| {
            b.iter(|| jit.eval_soa(black_box(soa)))
        });
    }

    group.finish();
}

/// Benchmark: JIT compilation overhead
#[cfg(feature = "jit")]
fn bench_jit_compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("jit_compile");

    let simple = SdfNode::sphere(1.0);
    let medium = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .translate(1.0, 0.0, 0.0);
    let complex = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
        .smooth_subtract(SdfNode::cylinder(0.3, 2.0), 0.05)
        .smooth_union(
            SdfNode::cylinder(0.3, 2.0).rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            0.1,
        )
        .twist(0.1)
        .translate(0.5, 0.0, 0.0);

    group.bench_function("simple", |b| {
        let compiled = CompiledSdf::compile(&simple);
        b.iter(|| JitSimdSdf::compile(black_box(&compiled)).unwrap())
    });

    group.bench_function("medium", |b| {
        let compiled = CompiledSdf::compile(&medium);
        b.iter(|| JitSimdSdf::compile(black_box(&compiled)).unwrap())
    });

    group.bench_function("complex", |b| {
        let compiled = CompiledSdf::compile(&complex);
        b.iter(|| JitSimdSdf::compile(black_box(&compiled)).unwrap())
    });

    group.finish();
}

#[cfg(feature = "jit")]
criterion_group!(
    benches,
    bench_primitives,
    bench_operations,
    bench_transforms,
    bench_modifiers,
    bench_complex_tree,
    bench_batch_eval,
    bench_raymarching,
    bench_marching_cubes,
    bench_interpreted_vs_compiled,
    bench_batch_compiled,
    bench_simd_evaluation,
    bench_batch_all,
    bench_bvh_sparse_scene,
    bench_bvh_complex,
    bench_soa_throughput,
    bench_jit_compile,
);

#[cfg(not(feature = "jit"))]
criterion_group!(
    benches,
    bench_primitives,
    bench_operations,
    bench_transforms,
    bench_modifiers,
    bench_complex_tree,
    bench_batch_eval,
    bench_raymarching,
    bench_marching_cubes,
    bench_interpreted_vs_compiled,
    bench_batch_compiled,
    bench_simd_evaluation,
    bench_batch_all,
    bench_bvh_sparse_scene,
    bench_bvh_complex,
    bench_soa_throughput,
);

criterion_main!(benches);

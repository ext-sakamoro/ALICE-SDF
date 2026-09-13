//! Benchmarks for the NPR primitive set
//!
//! Measures per-call latency for representative primitives across the
//! nine categories plus the noise implementations. Not exhaustive; the
//! set is chosen so a regression in any category is visible.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_toon(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_toon");
    group.bench_function("toon_ramp", |b| {
        b.iter(|| toon_ramp(black_box(0.42), black_box(3)));
    });
    group.bench_function("soft_toon_ramp", |b| {
        b.iter(|| soft_toon_ramp(black_box(0.42), black_box(3), black_box(0.05)));
    });
    group.bench_function("posterize_color", |b| {
        let color = Vec3::new(0.3, 0.6, 0.9);
        b.iter(|| posterize_color(black_box(color), black_box(4)));
    });
    group.finish();
}

fn bench_outline(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_outline");
    group.bench_function("distance_field_outline_soft", |b| {
        b.iter(|| distance_field_outline_soft(black_box(0.03), black_box(0.02), black_box(0.1)));
    });
    group.bench_function("curvature_outline", |b| {
        b.iter(|| curvature_outline(black_box(0.4), black_box(1.0), black_box(0.2)));
    });
    group.finish();
}

fn bench_sky(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_sky");
    let palette = [
        Vec3::new(0.9, 0.6, 0.4),
        Vec3::new(0.6, 0.7, 0.9),
        Vec3::new(0.2, 0.3, 0.6),
    ];
    let dir = Vec3::new(0.3, 0.7, 0.5).normalize();
    let sun = Vec3::new(0.4, 0.7, -0.6).normalize();
    group.bench_function("sky_gradient_bands_3", |b| {
        b.iter(|| sky_gradient_bands(black_box(dir), black_box(&palette)));
    });
    group.bench_function("puffy_cloud_layer", |b| {
        b.iter(|| puffy_cloud_layer(black_box(0.5), black_box(0.6), black_box(0.15)));
    });
    group.bench_function("sun_disc", |b| {
        b.iter(|| {
            sun_disc(
                black_box(dir),
                black_box(sun),
                black_box(0.02),
                black_box(0.05),
            )
        });
    });
    group.bench_function("light_shaft_beam", |b| {
        b.iter(|| light_shaft_beam(black_box(dir), black_box(sun), black_box(8.0)));
    });
    group.finish();
}

fn bench_rim(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_rim");
    let n_view = Vec3::new(0.3, 0.7, 0.5).normalize();
    group.bench_function("fresnel_rim", |b| {
        b.iter(|| fresnel_rim(black_box(0.6), black_box(2.0), black_box(1.0)));
    });
    group.bench_function("procedural_matcap", |b| {
        b.iter(|| {
            procedural_matcap(
                black_box(n_view),
                black_box(Vec3::new(0.1, 0.1, 0.3)),
                black_box(Vec3::new(0.6, 0.2, 0.2)),
                black_box(Vec3::new(0.2, 0.5, 0.7)),
                black_box(Vec3::new(0.95, 0.9, 0.85)),
            )
        });
    });
    group.finish();
}

fn bench_composition(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_composition");
    group.bench_function("vignette", |b| {
        b.iter(|| {
            vignette(
                black_box(0.7),
                black_box(0.3),
                black_box(0.3),
                black_box(0.2),
            )
        });
    });
    group.bench_function("bloom_toon", |b| {
        let color = Vec3::new(0.8, 0.6, 0.4);
        b.iter(|| bloom_toon(black_box(color), black_box(0.5), black_box(0.8)));
    });
    group.finish();
}

fn bench_motion(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_motion");
    group.bench_function("speed_line", |b| {
        b.iter(|| {
            speed_line(
                black_box(0.7),
                black_box(0.3),
                black_box(0.5),
                black_box(0.5),
                black_box(12),
                black_box(0.05),
            )
        });
    });
    group.bench_function("impact_flash", |b| {
        b.iter(|| impact_flash(black_box(0.15), black_box(0.3), black_box(1.0)));
    });
    group.finish();
}

fn bench_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_noise");
    let point = Vec3::new(1.5, 2.7, 3.9);
    let hash = HashNoise::new(42);
    let perlin = PerlinNoise::new(42);
    let worley = WorleyNoise::new(42);
    group.bench_function("hash_noise", |b| {
        b.iter(|| hash.sample_scalar(black_box(point)));
    });
    group.bench_function("perlin_noise", |b| {
        b.iter(|| perlin.sample_scalar(black_box(point)));
    });
    group.bench_function("worley_noise", |b| {
        b.iter(|| worley.sample_scalar(black_box(point)));
    });
    group.bench_function("perlin_fbm_4_octaves", |b| {
        b.iter(|| fbm(&perlin, black_box(point), 4, 2.0, 0.5));
    });
    group.finish();
}

fn bench_sdf_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_sdf_integration");
    let sphere = SdfNode::sphere(1.0);
    let point = Vec3::new(1.0, 0.0, 0.0);
    let light = Vec3::new(0.5, 0.8, 0.3).normalize();
    group.bench_function("toon_shade_from_node", |b| {
        b.iter(|| {
            toon_shade_from_node(
                black_box(&sphere),
                black_box(point),
                black_box(light),
                black_box(3),
            )
        });
    });
    group.bench_function("distance_outline_from_node", |b| {
        b.iter(|| {
            distance_outline_from_node(
                black_box(&sphere),
                black_box(point),
                black_box(0.01),
                black_box(0.05),
            )
        });
    });
    group.bench_function("curvature_outline_from_node", |b| {
        b.iter(|| {
            curvature_outline_from_node(
                black_box(&sphere),
                black_box(point),
                black_box(1e-3),
                black_box(0.5),
                black_box(0.1),
            )
        });
    });
    group.finish();
}

fn bench_color_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("npr_dsl");
    let ctx = NprColorContext {
        sdf: 0.0,
        normal: Vec3::new(0.0, 1.0, 0.0),
        view: Vec3::new(0.0, 0.0, 1.0),
        light: Vec3::new(0.4, 0.8, 0.4).normalize(),
        uv: glam::Vec2::new(0.5, 0.5),
        time: 0.0,
    };
    let toon = NprColorNode::Toon {
        shadow: Vec3::new(0.2, 0.18, 0.35),
        light: Vec3::new(0.95, 0.88, 0.75),
        bands: 3,
    };
    let composed = toon.clone().with_outline(Vec3::ZERO, 0.7);
    // Deep composition: 6-level tree touching hit-branch primitives
    let deep = toon
        .clone()
        .with_outline(Vec3::ZERO, 0.15)
        .with_fresnel(Vec3::new(0.9, 0.9, 1.0), 2.0)
        .vignetted(0.6, 0.25)
        .saturate(0.9)
        .tonemap_reinhard(1.0);
    let compiled_toon = toon.compile();
    let compiled_composed = composed.compile();
    let compiled_deep = deep.compile();
    group.bench_function("toon_eval", |b| {
        b.iter(|| toon.eval(black_box(&ctx)));
    });
    group.bench_function("toon_with_outline_eval", |b| {
        b.iter(|| composed.eval(black_box(&ctx)));
    });
    group.bench_function("toon_compiled_eval", |b| {
        b.iter(|| compiled_toon.eval(black_box(&ctx)));
    });
    group.bench_function("toon_with_outline_compiled_eval", |b| {
        b.iter(|| compiled_composed.eval(black_box(&ctx)));
    });
    group.bench_function("deep_composition_eval", |b| {
        b.iter(|| deep.eval(black_box(&ctx)));
    });
    group.bench_function("deep_composition_compiled_eval", |b| {
        b.iter(|| compiled_deep.eval(black_box(&ctx)));
    });

    // Phase 13 — SIMD batch evaluator: bench the 8-lane batched path.
    // The reported per-iter time is the cost of one 8-lane call; a rough
    // per-lane figure is `time / 8`.
    let ctxs_8 = [ctx; 8];
    let batch = NprBatchContext8::from_contexts(&ctxs_8);
    group.bench_function("toon_batch8_eval", |b| {
        b.iter(|| compiled_toon.eval_batch8(black_box(&batch)));
    });
    group.bench_function("toon_with_outline_batch8_eval", |b| {
        b.iter(|| compiled_composed.eval_batch8(black_box(&batch)));
    });
    group.bench_function("deep_composition_batch8_eval", |b| {
        b.iter(|| compiled_deep.eval_batch8(black_box(&batch)));
    });
    group.finish();
}

criterion_group!(
    npr_benches,
    bench_toon,
    bench_outline,
    bench_sky,
    bench_rim,
    bench_composition,
    bench_motion,
    bench_noise,
    bench_sdf_integration,
    bench_color_pipeline,
);
criterion_main!(npr_benches);

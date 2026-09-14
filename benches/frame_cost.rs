use alice_sdf::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench(c: &mut Criterion) {
    let p = Vec3::new(0.3, 0.2, 0.1);
    let base = SdfNode::sphere(1.0);
    let mut t5 = base.clone();
    for _ in 0..5 {
        t5 = t5.translate(0.1, 0.0, 0.0);
    }
    let mut r5 = base.clone();
    for _ in 0..5 {
        r5 = r5.rotate(Quat::from_rotation_y(0.3));
    }
    let mut rd5 = base.clone();
    for _ in 0..5 {
        rd5 = rd5.round(0.01);
    }
    let mut tw5 = base.clone();
    for _ in 0..5 {
        tw5 = tw5.twist(0.1);
    }
    let mut u4 = base.clone();
    for i in 0..4 {
        u4 = u4.smooth_union(SdfNode::sphere(0.5).translate(i as f32, 0.0, 0.0), 0.1);
    }
    let cases = [
        ("sphere", base),
        ("translate5", t5),
        ("rotate5", r5),
        ("round5", rd5),
        ("twist5", tw5),
        ("smooth_union4", u4),
    ];
    let mut g = c.benchmark_group("frame_cost");
    for (name, node) in cases.iter() {
        let compiled = CompiledSdf::compile(node);
        g.bench_function(*name, |b| {
            b.iter(|| eval_compiled(black_box(&compiled), black_box(p)))
        });
        let ps = Vec3x8::splat(p);
        g.bench_function(format!("{name}/simd"), |b| {
            b.iter(|| eval_compiled_simd(black_box(&compiled), black_box(ps)))
        });
    }
    g.finish();
}
criterion_group!(benches, bench);
criterion_main!(benches);

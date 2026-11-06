//! Local diagnosis only, not for main. Builds on the commit before the change too.
#![cfg(feature = "gpu")]
#![allow(missing_docs, clippy::all, dead_code)]
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;
use std::time::Instant;

const KERNEL: &str = "struct InputPoint { x: f32, y: f32, z: f32, _pad: f32, }
struct OutputDistance { distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 { return BODY; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= point_count) { return; }
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}
";
fn run(body: &str, pts: &[Vec3]) -> Vec<f32> {
    GpuEvaluator::from_wgsl(&KERNEL.replace("BODY", body))
        .unwrap_or_else(|e| panic!("{body}: {e}"))
        .into_parts()
        .0
        .eval_batch(pts)
        .unwrap()
}
fn points(n: usize, r: f32, seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    let mut f = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .map(|_| {
            Vec3::new(
                r * (2.0 * f() - 1.0),
                r * (2.0 * f() - 1.0),
                r * (2.0 * f() - 1.0),
            )
        })
        .collect()
}
fn n(a: &[f32], b: &[f32]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count()
}
fn ulps(a: f32, b: f32) -> u64 {
    let k = |x: f32| {
        let i = x.to_bits() as i64;
        if i & 0x8000_0000 != 0 {
            -(i & 0x7fff_ffff)
        } else {
            i
        }
    };
    (k(a) - k(b)).unsigned_abs()
}

#[test]
fn each_builtin() {
    let pts = points(4096, 2.0, 0x1234_5678);
    let share = "length(p.xz) + length(p.xy) + length(p)";
    for (name, alone, explicit) in [
        ("length2", "length(p.xz)", "sqrt(p.x * p.x + p.z * p.z)"),
        ("length3", "length(p)", "sqrt(p.x * p.x + p.y * p.y + p.z * p.z)"),
        ("dot3", "dot(p, p)", "p.x * p.x + p.y * p.y + p.z * p.z"),
        ("distance", "distance(p, vec3<f32>(0.1, 0.2, 0.3))", "sqrt((p.x - 0.1) * (p.x - 0.1) + (p.y - 0.2) * (p.y - 0.2) + (p.z - 0.3) * (p.z - 0.3))"),
        ("normalize.x", "normalize(p).x", "p.x / sqrt(p.x * p.x + p.y * p.y + p.z * p.z)"),
        ("cross.y", "cross(p, p.zxy).y", "p.z * p.z - p.x * p.y"),
        ("mix-scalar", "mix(p.x, p.y, p.z)", "p.x * (1.0 - p.z) + p.y * p.z"),
        ("mix-vector", "mix(p, p.yzx, p.z).x", "p.x * (1.0 - p.z) + p.y * p.z"),
    ] {
        let a = run(alone, &pts);
        let ctx = run(&format!("select({alone}, -1.0, {share} > 1.0e30)"), &pts);
        let e = run(explicit, &pts);
        let ectx = run(&format!("select({explicit}, -1.0, {share} > 1.0e30)"), &pts);
        println!("\nRB builtin {name:12} alone-vs-context {} explicit {} explicit-in-context {} of {}", n(&a, &ctx), n(&a, &e), n(&e, &ectx), pts.len());
    }
}

#[test]
fn overflow() {
    for (v, label) in [
        (1.0e20f32, "1e20"),
        (1.0e19, "1e19"),
        (2.0e19, "2e19"),
        (1.0e-23, "1e-23"),
        (1.0e-25, "1e-25"),
        (1.0e-30, "1e-30"),
        (1.0e-45, "denormal"),
    ] {
        let pts = [Vec3::splat(v), Vec3::new(v, 0.0, 0.0)];
        let l = run("length(p)", &pts);
        let d = run("dot(p, p)", &pts);
        let cpu = (pts[0].length(), pts[1].length());
        println!(
            "\nRB overflow {label:9} length {:e} {:e} dot {:e} {:e} cpu-length {:e} {:e}",
            l[0], l[1], d[0], d[1], cpu.0, cpu.1
        );
    }
    let z = run("normalize(p).x", &[Vec3::ZERO]);
    println!("\nRB normalize(0).x {:e} bits {:#x}", z[0], z[0].to_bits());
}

fn shapes() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("sphere", SdfNode::sphere(1.0)),
        ("box", SdfNode::box3d(1.0, 0.5, 0.25)),
        ("torus", SdfNode::torus(1.0, 0.25)),
        ("capsule", SdfNode::capsule(Vec3::ZERO, Vec3::Y, 0.25)),
        ("twist", SdfNode::box3d(0.5, 1.0, 0.25).twist(2.0)),
        (
            "smooth",
            SdfNode::sphere(1.0)
                .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(0.5, 0.0, 0.0), 0.25),
        ),
        (
            "union",
            SdfNode::torus(1.0, 0.25).union(SdfNode::sphere(0.5).translate(0.0, 1.0, 0.0)),
        ),
    ]
}

#[test]
fn gpu_vs_cpu() {
    let pts = points(4096, 2.0, 0xabcd_ef01);
    for (name, node) in shapes() {
        let g = GpuEvaluator::new(&node).unwrap().eval_batch(&pts).unwrap();
        let c: Vec<f32> = pts.iter().map(|p| eval(&node, *p)).collect();
        let max = g.iter().zip(&c).map(|(a, b)| ulps(*a, *b)).max().unwrap();
        let abs = g
            .iter()
            .zip(&c)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "\nRB gpu-vs-cpu {name:8} differ {} of {} max {max} ulp max-abs {abs:e}",
            n(&g, &c),
            pts.len()
        );
    }
}

#[test]
fn speed() {
    let pts = points(1 << 16, 2.0, 0x55aa_55aa);
    for (name, node) in shapes() {
        let t = Instant::now();
        let e = GpuEvaluator::new(&node).unwrap();
        let build = t.elapsed();
        let _ = e.eval_batch(&pts).unwrap();
        let mut times: Vec<f64> = (0..9)
            .map(|_| {
                let t = Instant::now();
                let r = e.eval_batch(&pts).unwrap();
                std::hint::black_box(r);
                t.elapsed().as_secs_f64() * 1e3
            })
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("\nRB speed {name:8} build {:.1} ms, eval median {:.3} ms, min {:.3} ms of 65536 points", build.as_secs_f64() * 1e3, times[4], times[0]);
    }
}

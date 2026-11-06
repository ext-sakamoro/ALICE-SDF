//! One large dispatch of each long program through the GLSL path and the WGSL path, unsplit, with the
//! exactly-zero results and the time printed: which points came back 0, and whether the work had finished
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "glsl"))]
#![allow(
    clippy::disallowed_methods,
    reason = "a diagnostic: it reports how long a dispatch took, nothing depends on the time"
)]

mod common;

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;

const WRAPPER: &str = r"#version 450
layout(local_size_x = 256) in;
struct InputPoint { float x; float y; float z; float pad; };
struct OutputDistance { float distance; float pad1; float pad2; float pad3; };
layout(std430, set = 0, binding = 0) readonly buffer InputPoints { InputPoint input_points[]; };
layout(std430, set = 0, binding = 1) buffer OutputDistances { OutputDistance output_distances[]; };
layout(std140, set = 0, binding = 2) uniform PointCount { uint point_count; };

LIBRARY

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) { return; }
    InputPoint pt = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3(pt.x, pt.y, pt.z));
}
";

fn ring(a: f32, b: f32, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

/// A tree of a whole body's size: 17 closed-curve lofts of 32 points in 20 rows, two limbs of 40 rows
/// with 32 harmonics, a torso of 140 rows with 64 harmonics, and a stack of 8 levels with two loops
fn body() -> SdfNode {
    let mut parts: Vec<SdfNode> = Vec::new();
    for i in 0..17 {
        let x = -0.8 + 0.1 * i as f32;
        let axis: Vec<Vec3> = (0..20)
            .map(|r| Vec3::new(x, 0.03 * r as f32, 0.002 * r as f32))
            .collect();
        let rows: Vec<Vec<Vec2>> = (0..20)
            .map(|r| {
                ring(
                    0.03 + 0.0004 * r as f32,
                    0.025 + 0.0003 * (i + r) as f32,
                    32,
                )
            })
            .collect();
        parts.push(
            SdfNode::loft_curves_capped(axis, Vec3::X, rows, [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]])
                .expect("a curve loft"),
        );
    }
    let polar = |rows: usize, k: usize, x: f32, r: f32| {
        let axis: Vec<Vec3> = (0..rows)
            .map(|i| Vec3::new(x, 0.006 * i as f32, 0.5))
            .collect();
        let values: Vec<Vec<f32>> = (0..rows)
            .map(|i| {
                let mut v = vec![0.0f32; 2 * k + 1];
                v[0] = r + 0.0001 * i as f32;
                for (h, c) in v.iter_mut().enumerate().skip(1) {
                    *c = 0.002 / h as f32 * (1.0 + 0.001 * (i % 7) as f32);
                }
                v
            })
            .collect();
        SdfNode::loft_capped(axis, Vec3::Z, k as u32, values, [0.05, 0.05]).expect("a polar loft")
    };
    parts.push(polar(40, 32, -0.4, 0.05));
    parts.push(polar(40, 32, 0.4, 0.05));
    parts.push(polar(140, 64, 0.0, 0.15));
    let levels: Vec<f32> = (0..8).map(|i| -1.0 + 0.05 * i as f32).collect();
    parts.push(
        SdfNode::smooth_section_stack(
            levels.clone(),
            levels
                .iter()
                .map(|_| {
                    let a: Vec<Vec2> = ring(0.05, 0.05, 12)
                        .iter()
                        .map(|p| *p + Vec2::new(-0.1, 0.0))
                        .collect();
                    let b: Vec<Vec2> = ring(0.05, 0.05, 12)
                        .iter()
                        .map(|p| *p + Vec2::new(0.1, 0.0))
                        .collect();
                    vec![a, b]
                })
                .collect(),
            0.005,
            0.05,
        )
        .expect("a stack"),
    );
    let mut all = parts.remove(0);
    for p in parts {
        all = all.union(p);
    }
    all
}

fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x6e01_5e00_0000_0001;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 40) as f32) / ((1u64 << 24) as f32)).mul_add(6.0, -3.0)
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

fn report(lane: &str, name: &str, node: &SdfNode, pts: &[Vec3], gpu: &GpuEvaluator) {
    let t = std::time::Instant::now();
    eprintln!("DIAG {lane} {name}: dispatch of {} points", pts.len());
    let got = gpu.eval_batch(pts);
    let secs = t.elapsed().as_secs_f32();
    let got = match got {
        Ok(g) => g,
        Err(e) => {
            eprintln!("DIAG {lane} {name}: Err after {secs:.2} s: {e}");
            return;
        }
    };
    let zeros: Vec<usize> = got
        .iter()
        .enumerate()
        .filter(|(_, g)| **g == 0.0)
        .map(|(i, _)| i)
        .collect();
    let mut worst = (0.0f32, 0usize);
    let mut wrong = 0;
    for (i, (p, g)) in pts.iter().zip(&got).enumerate() {
        let d = (g - alice_sdf::eval::eval(node, *p)).abs();
        if d > 1e-4 {
            wrong += 1;
        }
        if d > worst.0 {
            worst = (d, i);
        }
    }
    eprintln!(
        "DIAG {lane} {name}: {secs:.2} s, {} results, {} exactly zero (first {:?}), {wrong} off by > 1e-4, worst {:.3e} at {}",
        got.len(),
        zeros.len(),
        &zeros[..zeros.len().min(24)],
        worst.0,
        worst.1
    );
}

#[test]
fn large_dispatches() {
    let pts = points(1024);
    for (name, node) in common::corpus::corpus() {
        if !(name.starts_with("loft") || name.contains("stack")) {
            continue;
        }
        match GpuEvaluator::new(&node) {
            Ok(gpu) => report("wgsl", name, &node, &pts, &gpu),
            Err(e) => eprintln!("DIAG wgsl {name}: {e}"),
        }
        let s = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
        let src = WRAPPER.replace("LIBRARY", &s.source);
        eprintln!("DIAG glsl {name}: {} bytes", src.len());
        let gpu = GpuEvaluator::from_glsl_compute(&src)
            .expect("the GLSL module")
            .into_parts()
            .0;
        report("glsl", name, &node, &pts, &gpu);
        for submit in [1024usize, 64] {
            let t = std::time::Instant::now();
            let got: Vec<f32> = pts
                .chunks(submit)
                .flat_map(|c| gpu.eval_batch(c).expect("eval"))
                .collect();
            eprintln!(
                "DIAG glsl {name} again in {submit}: {:.2} s, {} exactly zero",
                t.elapsed().as_secs_f32(),
                got.iter().filter(|g| **g == 0.0).count()
            );
        }
    }
}

/// One case per process, so a hang costs only its own time limit: the node, the points in the dispatch
#[test]
fn one_case() {
    let (Ok(node_name), Ok(n)) = (std::env::var("DIAG_NODE"), std::env::var("DIAG_POINTS")) else {
        return;
    };
    let n: usize = n.parse().expect("a count");
    let pts = points(n);
    if node_name == "mini" {
        let values: Vec<String> = (0..8192).map(|i| format!("{}.0", i % 7)).collect();
        let lib = format!(
            "const float big[8192] = float[8192]({});\nfloat sdf_eval(vec3 p) {{\n    float s = 0.0;\n    for (int i = 0; i < 512; i++) {{\n        s = s + big[(i * 13 + int(abs(p.x) * 100.0)) % 8192];\n    }}\n    return s;\n}}\n",
            values.join(", ")
        );
        let gpu = GpuEvaluator::from_glsl_compute(&WRAPPER.replace("LIBRARY", &lib))
            .expect("the module")
            .into_parts()
            .0;
        let t = std::time::Instant::now();
        let r = gpu.eval_batch(&pts);
        eprintln!(
            "CASE mini {n}: {:.2} s {:?}",
            t.elapsed().as_secs_f32(),
            r.map(|v| v.iter().filter(|x| **x == 0.0).count())
                .map_err(|e| e.to_string())
        );
        until_the_cpu_stops(gpu, t);
    }
    let node = if node_name == "body" {
        body()
    } else {
        common::corpus::corpus()
            .into_iter()
            .find(|(c, _)| *c == node_name)
            .expect("a corpus name")
            .1
    };
    // the same program through WGSL: its arrays as module constants, or in the node data buffer
    if let Ok(form) = std::env::var("DIAG_WGSL") {
        let shader = if form == "const" {
            WgslShader::transpile(&node, TranspileMode::Hardcoded)
        } else if form == "private" {
            let mut w = WgslShader::transpile(&node, TranspileMode::Hardcoded);
            let edited: Vec<String> = w
                .source
                .lines()
                .map(|l| {
                    if l.starts_with("const ") && l.contains(": array<") {
                        l.replacen("const ", "var<private> ", 1)
                    } else {
                        l.to_string()
                    }
                })
                .collect();
            w.source = edited.join("\n");
            eprintln!(
                "CASE private: {} arrays now variables",
                w.source.matches("var<private> ").count()
            );
            w
        } else {
            WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded)
        };
        eprintln!(
            "CASE wgsl {form} {node_name}: {} bytes, {} values in the buffer",
            shader.source.len(),
            shader.node_data.len()
        );
        if let Ok(dir) = std::env::var("DIAG_DUMP") {
            std::fs::write(
                format!("{dir}/{node_name}.{form}.json"),
                format!("{:?}", shader.node_data),
            )
            .expect("write");
            std::fs::write(
                format!("{dir}/{node_name}.{form}.wgsl"),
                shader.to_compute_shader(),
            )
            .expect("write");
            return;
        }
        let made = std::time::Instant::now();
        let gpu = match GpuEvaluator::from_shader(&shader) {
            Ok(g) => g,
            Err(e) => {
                eprintln!(
                    "CASE module refused after {:.2} s: {e}",
                    made.elapsed().as_secs_f32()
                );
                return;
            }
        };
        with_limit(&gpu);
        eprintln!(
            "CASE pipeline created in {:.3} s",
            made.elapsed().as_secs_f32()
        );
        let t = std::time::Instant::now();
        let r = gpu.eval_batch(&pts);
        if r.is_ok() {
            run_time(&gpu, &format!("wgsl {form} {node_name}"));
        }
        eprintln!(
            "CASE wgsl {form} {node_name} {n}: {:.2} s {:?}",
            t.elapsed().as_secs_f32(),
            r.map(|v| v.iter().filter(|x| **x == 0.0).count())
                .map_err(|e| e.to_string())
        );
        until_the_cpu_stops(gpu, t);
    }
    let s = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
    let library: Vec<String> = s
        .source
        .lines()
        .map(|l| {
            if std::env::var_os("DIAG_GLSL_VAR").is_some()
                && l.starts_with("const float ")
                && l.contains('[')
            {
                eprintln!("CASE glsl array now a variable: {}", &l[..l.len().min(40)]);
                l.replacen("const float ", "float ", 1)
            } else {
                l.to_string()
            }
        })
        .collect();
    let src = WRAPPER.replace("LIBRARY", &library.join("\n"));
    if let Ok(dir) = std::env::var("DIAG_DUMP") {
        std::fs::write(format!("{dir}/{node_name}.comp"), &src).expect("write");
        let w = alice_sdf::compiled::GpuEvaluator::new(&node).is_ok();
        eprintln!("CASE dumped {node_name}, wgsl evaluator {w}");
        return;
    }
    let made = std::time::Instant::now();
    let gpu = match GpuEvaluator::from_glsl_compute(&src) {
        Ok(g) => g.into_parts().0,
        Err(e) => {
            eprintln!(
                "CASE module refused after {:.2} s: {e}",
                made.elapsed().as_secs_f32()
            );
            return;
        }
    };
    with_limit(&gpu);
    eprintln!(
        "CASE pipeline created in {:.3} s",
        made.elapsed().as_secs_f32()
    );
    let t = std::time::Instant::now();
    let r = gpu.eval_batch(&pts);
    if r.is_ok() {
        run_time(&gpu, &format!("glsl {node_name}"));
    }
    eprintln!(
        "CASE {node_name} {n}: {:.2} s {:?}",
        t.elapsed().as_secs_f32(),
        r.map(|v| v.iter().filter(|x| **x == 0.0).count())
            .map_err(|e| e.to_string())
    );
    until_the_cpu_stops(gpu, t);
}

/// The time of a later dispatch, the driver's compile being done: the fastest of 5, 1024 points
fn run_time(gpu: &GpuEvaluator, what: &str) {
    let pts = points(1024);
    let mut best = f32::MAX;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        gpu.eval_batch(&pts).expect("a later dispatch");
        best = best.min(t.elapsed().as_secs_f32());
    }
    eprintln!("CASE run {what}: {:.3} ms per 1024 points", best * 1000.0);
}

fn with_limit(gpu: &GpuEvaluator) {
    if let Ok(s) = std::env::var("DIAG_LIMIT_S") {
        gpu.set_wait_limit(std::time::Duration::from_secs(s.parse().expect("seconds")));
    }
}

/// CPU seconds of this process so far (Linux)
fn cpu_seconds() -> f32 {
    let Ok(stat) = std::fs::read_to_string("/proc/self/stat") else {
        return 0.0;
    };
    let rest = stat.rsplit(')').next().expect("the fields after the name");
    let field: Vec<&str> = rest.split_whitespace().collect();
    let ticks: u64 =
        field[11].parse::<u64>().expect("utime") + field[12].parse::<u64>().expect("stime");
    ticks as f32 / 100.0
}

/// Keep the device alive (dropping it panics while its queue is busy) until this process stops using
/// the CPU, then say how long the driver's work took in all: the time a first dispatch needs
#[allow(clippy::disallowed_types, reason = "a diagnostic's elapsed time")]
fn until_the_cpu_stops(gpu: GpuEvaluator, t: std::time::Instant) -> ! {
    std::mem::forget(gpu);
    let mut last = cpu_seconds();
    let mut idle = 0;
    while idle < 3 {
        std::thread::sleep(std::time::Duration::from_secs(2));
        let now = cpu_seconds();
        idle = if now - last < 0.05 { idle + 1 } else { 0 };
        last = now;
    }
    eprintln!(
        "CASE cpu stopped: {:.1} s elapsed before the idle, {last:.1} s of CPU",
        t.elapsed().as_secs_f32() - 6.0
    );
    std::process::exit(0);
}

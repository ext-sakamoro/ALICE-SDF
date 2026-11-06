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
    let (_, node) = common::corpus::corpus()
        .into_iter()
        .find(|(c, _)| *c == node_name)
        .expect("a corpus name");
    // the same program through WGSL: its arrays as module constants, or in the node data buffer
    if let Ok(form) = std::env::var("DIAG_WGSL") {
        let shader = if form == "const" {
            WgslShader::transpile(&node, TranspileMode::Hardcoded)
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
                format!("{dir}/{node_name}.{form}.wgsl"),
                shader.to_compute_shader(),
            )
            .expect("write");
            return;
        }
        let gpu = GpuEvaluator::from_shader(&shader).expect("the module");
        let t = std::time::Instant::now();
        let r = gpu.eval_batch(&pts);
        eprintln!(
            "CASE wgsl {form} {node_name} {n}: {:.2} s {:?}",
            t.elapsed().as_secs_f32(),
            r.map(|v| v.iter().filter(|x| **x == 0.0).count())
                .map_err(|e| e.to_string())
        );
        until_the_cpu_stops(gpu, t);
    }
    let s = GlslShader::transpile(&node, GlslTranspileMode::Hardcoded);
    let src = WRAPPER.replace("LIBRARY", &s.source);
    if let Ok(dir) = std::env::var("DIAG_DUMP") {
        std::fs::write(format!("{dir}/{node_name}.comp"), &src).expect("write");
        let w = alice_sdf::compiled::GpuEvaluator::new(&node).is_ok();
        eprintln!("CASE dumped {node_name}, wgsl evaluator {w}");
        return;
    }
    let gpu = GpuEvaluator::from_glsl_compute(&src)
        .expect("the module")
        .into_parts()
        .0;
    let t = std::time::Instant::now();
    let r = gpu.eval_batch(&pts);
    eprintln!(
        "CASE {node_name} {n}: {:.2} s {:?}",
        t.elapsed().as_secs_f32(),
        r.map(|v| v.iter().filter(|x| **x == 0.0).count())
            .map_err(|e| e.to_string())
    );
    until_the_cpu_stops(gpu, t);
}

/// CPU seconds of this process so far (Linux)
fn cpu_seconds() -> f32 {
    let stat = std::fs::read_to_string("/proc/self/stat").expect("/proc/self/stat");
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

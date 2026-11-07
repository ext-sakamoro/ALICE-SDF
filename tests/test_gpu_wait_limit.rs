//! A read that reaches its wait limit leaves the device and its buffers alive for the work still
//! running, refuses further work on that evaluator, and leaves the process able to use a new one.
//!
//! The limit is zero: one poll right after the submit, while the first dispatch is at most starting
//! (a driver that compiles at the first dispatch, as lavapipe does, is seconds away from finishing).
//! The evaluator is dropped at once. Its device must then stay alive until the work ends: dropping it
//! panics in the queue's drop on that driver, and freed buffers show as a signal, which fails the run.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", not(feature = "msl")))]
#![allow(
    clippy::disallowed_methods,
    reason = "the test sleeps while the driver's work runs out, nothing is timed"
)]

use alice_sdf::compiled::{GpuError, GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use std::time::Duration;

/// A long constant array, different on every run so that no shader cache answers the first dispatch
fn slow_source(run: u32) -> String {
    let values: Vec<String> = (0..8192u32)
        .map(|i| format!("{}.0", (i ^ run) % 7))
        .collect();
    format!(
        r"const big: array<f32, 8192> = array<f32, 8192>({});

struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}

@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{
        return;
    }}
    var s = input_points[idx].x;
    for (var i = 0u; i < 4096u; i += 1u) {{
        s += big[(i * 3u + idx) & 8191u];
    }}
    output_distances[idx].distance = s;
}}
",
        values.join(", ")
    )
}

/// CPU seconds of this process so far, where the system reports them
fn cpu_seconds() -> Option<f32> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let rest = stat.rsplit(')').next()?;
    let field: Vec<&str> = rest.split_whitespace().collect();
    let ticks = field.get(11)?.parse::<u64>().ok()? + field.get(12)?.parse::<u64>().ok()?;
    Some(ticks as f32 / 100.0)
}

/// Until this process stops using the CPU: the abandoned work has run out by then. Without a CPU
/// reading, a fixed time.
fn until_the_cpu_stops() {
    let Some(mut last) = cpu_seconds() else {
        std::thread::sleep(Duration::from_secs(20));
        return;
    };
    let mut idle = 0;
    while idle < 3 {
        std::thread::sleep(Duration::from_secs(2));
        let now = cpu_seconds().unwrap_or(last);
        idle = if now - last < 0.05 { idle + 1 } else { 0 };
        last = now;
    }
}

#[test]
fn a_read_over_its_limit_keeps_the_device_for_the_running_work() {
    let points: Vec<Vec3> = (0..2048).map(|i| Vec3::splat(i as f32 * 0.001)).collect();
    let (slow, _) = GpuEvaluator::from_wgsl(&slow_source(std::process::id()))
        .expect("a device and the shader")
        .into_parts();
    slow.set_wait_limit(Duration::ZERO);

    let over = slow.eval_batch(&points);
    let Err(GpuError::BufferMapping(why)) = &over else {
        panic!("the read did not stop at its limit: {over:?}");
    };
    assert!(why.contains("did not finish"), "another error: {why}");

    let again = slow.eval_batch(&points);
    assert!(again.is_err(), "the evaluator took more work: {again:?}");

    // the work is still running: its buffers and device must survive this
    drop(slow);

    let sphere = SdfNode::sphere(1.0);
    let fresh =
        GpuEvaluator::from_shader(&WgslShader::transpile(&sphere, TranspileMode::Hardcoded))
            .expect("a new evaluator after an abandoned one");
    let pts = [Vec3::ZERO, Vec3::new(3.0, 0.0, 0.0)];
    let got = fresh.eval_batch(&pts).expect("a read on the new evaluator");
    assert_eq!(got, vec![-1.0, 2.0]);
    drop(fresh);

    until_the_cpu_stops();
}

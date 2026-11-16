//! The GPU guard on a device: bounded work runs and gives the CPU's answer, and the shaders the guard exists for are
//! refused with their message instead of reaching the device. A run on a device is wrapped in a wall-clock
//! timeout by its caller (CI's step timeout, or `timeout` around it), so a failure cannot hold the GPU.
//!
//! Runs where this crate's tests may use a GPU (`ALICE_GPU_SLOT` set, or the software fallback adapter asked for with
//! `ALICE_SDF_TEST_FALLBACK_ADAPTER=1`); elsewhere every request is refused by the slot guard, which the test reports
//! and counts as not run (it does not pass silently: it prints `[gpu-guard-on-device] not run`). On a run it prints
//! the adapter and `[gpu-guard-on-device] ran <n>` with the number of device checks done.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::compiled::{GpuError, GpuEvaluator};
use alice_sdf::gpu_context::{refusal, GpuContext};
use alice_sdf::gpu_loop_bound::HostMaxima;
use alice_sdf::SdfNode;
use glam::Vec3;

fn wgsl_module(body: &str) -> String {
    format!(
        "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n@compute @workgroup_size(1)\nfn main() {{\n var acc = 0.0;\n{body}\n o[0] = acc;\n}}\n"
    )
}

fn on_a_device() -> bool {
    std::env::var("ALICE_GPU_SLOT").is_ok_and(|v| !v.is_empty())
        || std::env::var("ALICE_SDF_TEST_FALLBACK_ADAPTER").as_deref() == Ok("1")
}

#[test]
fn bounded_work_runs_and_the_unbounded_shader_is_refused_on_the_device() {
    if !on_a_device() {
        println!("[gpu-guard-on-device] not run: no GPU slot and no fallback adapter asked for");
        return;
    }
    let mut ran = 0usize;

    // positive: an evaluator on the device gives the CPU's distances
    let tree = SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.2, 0.0, 0.0));
    let gpu = GpuEvaluator::new(&tree).expect("an evaluator on the device");
    let points: Vec<Vec3> = (0..256)
        .map(|i| {
            let t = i as f32 / 255.0;
            Vec3::new(-2.0 + 4.0 * t, 0.3 * t, -0.2)
        })
        .collect();
    let got = gpu.eval_batch(&points).expect("evaluated on the device");
    let worst = points
        .iter()
        .zip(&got)
        .map(|(p, g)| (alice_sdf::eval(&tree, *p) - g).abs())
        .fold(0.0_f32, f32::max);
    println!("[gpu-guard-on-device] evaluator: 256 points, worst |gpu - cpu| = {worst:e}");
    assert!(
        worst <= 1e-4,
        "the device's distances differ from the CPU's by {worst}"
    );
    ran += 1;

    let context = GpuContext::request("gpu guard on device").expect("a context on the device");
    let info = context.adapter_info();
    println!(
        "[gpu-guard-on-device] adapter {} type {:?} backend {:?}",
        info.name, info.device_type, info.backend
    );

    // positive: a bounded module is created
    let bounded = wgsl_module("for (var i = 0u; i < 30u; i = i + 1u) { acc = acc + 1.0; }");
    context
        .shader_module(
            "bounded",
            wgpu::ShaderSource::Wgsl(bounded.into()),
            &HostMaxima::new(),
        )
        .expect("a bounded module is created on the device");
    ran += 1;

    // negative: a loop without a bound (what made the GPU run without end) is refused, with its message
    let unbounded = wgsl_module("while (acc < 10.0) { acc = acc + 0.5; }");
    let e = context
        .shader_module(
            "unbounded",
            wgpu::ShaderSource::Wgsl(unbounded.into()),
            &HostMaxima::new(),
        )
        .expect_err("an unbounded loop is refused");
    let GpuError::ShaderCompilation(message) = &e else {
        panic!("{e:?}")
    };
    assert!(
        message.starts_with(refusal::UNBOUNDED_SHADER_LOOP),
        "{message}"
    );
    println!("[gpu-guard-on-device] unbounded refused: {message}");
    ran += 1;

    // negative: bounded loops whose product is over the per-invocation ceiling are refused too
    let too_much = wgsl_module(
        "for (var i = 0u; i < 300u; i = i + 1u) { for (var j = 0u; j < 300u; j = j + 1u) { acc = acc + 1.0; } }",
    );
    let e = context
        .shader_module(
            "over the ceiling",
            wgpu::ShaderSource::Wgsl(too_much.into()),
            &HostMaxima::new(),
        )
        .expect_err("a product over the ceiling is refused");
    println!("[gpu-guard-on-device] over the ceiling refused: {e}");
    assert!(
        e.to_string().contains(refusal::UNBOUNDED_SHADER_LOOP),
        "{e}"
    );
    ran += 1;

    // negative: caller text through the evaluator's text entry is checked before any device is requested
    let e =
        GpuEvaluator::from_wgsl(&wgsl_module("loop { acc = acc + 1.0; }")).expect_err("refused");
    println!("[gpu-guard-on-device] caller text refused: {e}");
    assert!(
        e.to_string().contains(refusal::UNBOUNDED_SHADER_LOOP),
        "{e}"
    );
    ran += 1;

    println!("[gpu-guard-on-device] ran {ran}");
    assert_eq!(ran, 5);
}

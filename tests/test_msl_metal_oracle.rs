//! MSL oracle: every corpus node's emitted MSL must **compile on Metal** and
//! **agree with the CPU law**.
//!
//! MSL is derived from the WGSL emit through naga (`compiled::msl`), so this
//! oracle answers two different questions that the WGSL parity test cannot:
//!
//! 1. does the translation produce MSL the Metal compiler accepts, and
//! 2. does the translated kernel compute the same distances the CPU does
//!
//! A WGSL law fix cannot drift from MSL (single source), but a **translation**
//! bug or a wrong **binding map** can, and both show up here and nowhere else.
//!
//! The Metal runtime compiler (`MTLDevice::newLibraryWithSource`) is used, not
//! `xcrun metal`: the Xcode Metal Toolchain is a separately downloaded
//! component (`xcodebuild -downloadComponent MetalToolchain`) and is absent on
//! a stock machine, while the runtime compiler ships with the OS.
//!
//! Point set and tolerance mirror `tests/test_gpu_law_parity.rs` so a drift
//! here is directly comparable with the WGSL/Vulkan numbers.
//!
//! Set `ALICE_SDF_REQUIRE_METAL=1` to make "no Metal device" a failure instead
//! of a skip (the same contract `ALICE_SDF_REQUIRE_GPU` has for the WGSL
//! oracle, ALICE-SDF-LAWS §5 Port Parity Oracle Rule).
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "msl", target_os = "macos"))]

mod common;

use alice_sdf::compiled::msl::MslShader;
use alice_sdf::compiled::TranspileMode;
use alice_sdf::prelude::*;
use common::corpus::corpus;
use metal::{Device, MTLResourceOptions, MTLSize};

/// Relative tolerance, as in the WGSL oracle: the taper guard pushes `|d|` to
/// ~1e6 on its singular plane, so an absolute bound is meaningless there.
const REL_TOL: f32 = 1e-4;

/// Deterministic LCG points in a ±3 box — byte-for-byte the generator in
/// `tests/test_gpu_law_parity.rs`, so the two oracles sample the same laws.
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

/// A Metal device, or `None` when the host has none (skip unless the caller
/// demanded Metal).
fn device_or_skip() -> Option<Device> {
    match Device::system_default() {
        Some(d) => Some(d),
        None => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_METAL").is_none(),
                "ALICE_SDF_REQUIRE_METAL is set but no Metal device was found"
            );
            eprintln!("skipping MSL Metal oracle: no Metal device");
            None
        }
    }
}

/// Compile `shader` and evaluate `pts` on the GPU.
fn run_on_metal(device: &Device, shader: &MslShader, pts: &[Vec3]) -> Result<Vec<f32>, String> {
    let library = device
        .new_library_with_source(&shader.source, &metal::CompileOptions::new())
        .map_err(|e| format!("Metal compile failed: {e}"))?;
    let function = library
        .get_function(&shader.entry_point, None)
        .map_err(|e| format!("entry point {} not found: {e}", shader.entry_point))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline creation failed: {e}"))?;

    let n = pts.len();
    // `InputPoint` / `OutputDistance` are both 4 x f32 (x, y, z, pad).
    let stride = 16usize;
    let byte_len = (n * stride) as u64;

    let input: Vec<[f32; 4]> = pts.iter().map(|p| [p.x, p.y, p.z, 0.0]).collect();
    let in_buf = device.new_buffer_with_data(
        input.as_ptr().cast(),
        byte_len,
        MTLResourceOptions::StorageModeShared,
    );
    let out_buf = device.new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
    let count = u32::try_from(n).map_err(|_| "point count exceeds u32".to_string())?;
    let count_buf = device.new_buffer_with_data(
        std::ptr::addr_of!(count).cast(),
        4,
        MTLResourceOptions::StorageModeShared,
    );
    // naga's `_mslBufferSizes` members hold **byte** sizes. Both runtime-sized
    // arrays here have the same stride and element count, so filling every slot
    // with the same byte length is correct whichever member maps to which
    // global, and it stays correct if naga renumbers them.
    let sizes = [u32::try_from(byte_len).map_err(|_| "buffer too large".to_string())?; 8];
    let sizes_buf = device.new_buffer_with_data(
        sizes.as_ptr().cast(),
        (sizes.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&in_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&count_buf), 0);
    enc.set_buffer(u64::from(shader.sizes_buffer_slot), Some(&sizes_buf), 0);

    let threads = u64::from(shader.workgroup_size[0].max(1));
    let groups = (n as u64).div_ceil(threads);
    enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Shared storage: the contents are visible without an explicit blit.
    let raw = unsafe { std::slice::from_raw_parts(out_buf.contents().cast::<f32>(), n * 4) };
    Ok(raw.chunks_exact(4).map(|c| c[0]).collect())
}

#[test]
fn every_corpus_node_compiles_as_metal() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in corpus() {
        let shader = match MslShader::transpile(&node, TranspileMode::Hardcoded) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{name}: MSL emit failed: {e}"));
                continue;
            }
        };
        if let Err(e) =
            device.new_library_with_source(&shader.source, &metal::CompileOptions::new())
        {
            failures.push(format!("{name}: Metal compile failed: {e}"));
            continue;
        }
        checked += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {} corpus nodes failed to compile as MSL:\n{}",
        failures.len(),
        checked + failures.len(),
        failures.join("\n")
    );
    assert!(checked > 100, "corpus unexpectedly small: {checked}");
    eprintln!("MSL compiled on Metal for {checked} corpus nodes");
}

#[test]
fn metal_matches_cpu_for_every_corpus_node() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let pts = points(2048);
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut global_worst = (0.0_f32, String::new());

    for (name, node) in corpus() {
        let shader = match MslShader::transpile(&node, TranspileMode::Hardcoded) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{name}: MSL emit failed: {e}"));
                continue;
            }
        };
        let got = match run_on_metal(&device, &shader, &pts) {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        let mut worst = (0.0_f32, Vec3::ZERO, 0.0_f32, 0.0_f32);
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            let diff = (g - c).abs() / c.abs().max(1.0);
            if diff > worst.0 {
                worst = (diff, *p, c, *g);
            }
        }
        if worst.0 > global_worst.0 {
            global_worst = (worst.0, name.to_string());
        }
        if worst.0 > REL_TOL {
            failures.push(format!(
                "{name}: Metal/CPU drift {:.3e} at {:?} (cpu={} metal={})",
                worst.0, worst.1, worst.2, worst.3
            ));
        }
        checked += 1;
    }

    assert!(
        failures.is_empty(),
        "{} corpus node(s) drifted between Metal and CPU:\n{}",
        failures.len(),
        failures.join("\n")
    );
    eprintln!(
        "Metal/CPU parity over {checked} corpus nodes x {} points: worst {:.3e} ({})",
        pts.len(),
        global_worst.0,
        global_worst.1
    );
}

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
//! oracle, the port parity rule).
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
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
/// The options a caller is told to use: `MslShader::REQUIRED_COMPILE_OPTIONS`.
fn required_options() -> metal::CompileOptions {
    let req = MslShader::REQUIRED_COMPILE_OPTIONS;
    assert_eq!(
        req.language_version,
        (2, 0),
        "map the new version to metal::MTLLanguageVersion here"
    );
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(req.fast_math_enabled);
    options.set_language_version(metal::MTLLanguageVersion::V2_0);
    options
}

/// Metal's defaults (fast math on), what a caller gets without setting anything.
fn default_options() -> metal::CompileOptions {
    metal::CompileOptions::new()
}

fn run_on_metal(
    device: &Device,
    shader: &MslShader,
    pts: &[Vec3],
    options: &metal::CompileOptions,
) -> Result<Vec<f32>, String> {
    let library = device
        .new_library_with_source(&shader.source, options)
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
        if let Err(e) = device.new_library_with_source(&shader.source, &required_options()) {
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
        let got = match run_on_metal(&device, &shader, &pts, &required_options()) {
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

/// The compute layout of the transpiler's WGSL with `sdf_eval` replaced by `body`,
/// so `run_on_metal` drives it exactly as it drives a corpus node.
fn kernel_wgsl(body: &str) -> String {
    format!(
        "struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 {{ return {body}; }}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{ return; }}
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}}
"
    )
}

/// The discriminator: with a = b = 1 + 2^-12 and c = -(1 + 2^-11),
/// a * b = 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so two roundings give exactly 0
/// and one fused rounding gives 2^-24.
fn discriminator() -> Vec3 {
    let (a, b, c) = (
        1.0 + 2f32.powi(-12),
        1.0 + 2f32.powi(-12),
        -(1.0 + 2f32.powi(-11)),
    );
    assert_eq!((a * b + c).to_bits(), 0.0f32.to_bits());
    assert_eq!(a.mul_add(b, c).to_bits(), 2f32.powi(-24).to_bits());
    Vec3::new(a, b, c)
}

/// (WGSL body, the two-roundings result)
const FORMS: [(&str, f32); 3] = [
    ("p.x * p.y + p.z", 0.0),
    ("-(p.x * p.y + p.z)", -0.0),
    ("(p.x * p.y) + p.z", 0.0),
];

/// ALICE-Law G3 on the GPU: the emitted MSL does not let the Metal compiler fuse
/// `a * b + c`, even with Metal's default options (fast math on), which fuse it
/// without the `#pragma METAL fp contract(off)` that `MslShader` puts first
/// (measured on Apple silicon). Here only the fused value is ruled out: the sign of
/// a zero is the next test's business. Runs where Metal exists:
/// scripts/preflight.sh on macOS sets ALICE_SDF_REQUIRE_METAL=1; the macOS CI
/// runner skips when it has no device (printed); Linux and Windows do not build
/// this file.
#[test]
fn metal_does_not_fuse_a_multiply_add() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let pts = [discriminator()];
    let fused = 2f32.powi(-24);
    let mut compared = 0;
    for (body, _) in FORMS {
        let shader = MslShader::from_wgsl(&kernel_wgsl(body))
            .expect("transpile")
            .into_parts()
            .0;
        assert!(shader
            .source
            .starts_with("#pragma METAL fp contract(off)\n"));
        let got =
            run_on_metal(&device, &shader, &pts, &default_options()).expect("run on Metal")[0];
        assert!(
            got == 0.0,
            "`{body}` on Metal (default options) gave {got:e} (bits {:#010x}); the fused value is {fused:e} or its negation, two roundings give a zero",
            got.to_bits()
        );
        compared += 1;
    }
    assert_eq!(compared, FORMS.len());
    eprintln!("Metal, default options: {compared} multiply-add forms not fused");
}

/// The sign of a zero needs `MslShader::REQUIRED_COMPILE_OPTIONS` (fast math off):
/// with Metal's defaults `-(a * b + c)` of a zero comes back `+0.0`, and no
/// construct of the emitted source keeps the sign in general (measured on Apple
/// silicon; a bit-flip negation does for that one operator only). With the
/// required options every form gives the two-roundings bits exactly.
#[test]
fn metal_signed_zero_needs_fast_math_off() {
    // checked with or without a device: the default keeps fast math on and loses
    // the sign of zero (`metal_default_fast_math_loses_the_sign_of_zero`)
    let required = std::hint::black_box(MslShader::REQUIRED_COMPILE_OPTIONS);
    assert!(
        !required.fast_math_enabled,
        "REQUIRED_COMPILE_OPTIONS must turn Metal's fast math off"
    );
    let Some(device) = device_or_skip() else {
        return;
    };
    let pts = [discriminator()];
    let mut compared = 0;
    for (body, want) in FORMS {
        let shader = MslShader::from_wgsl(&kernel_wgsl(body))
            .expect("transpile")
            .into_parts()
            .0;
        let got =
            run_on_metal(&device, &shader, &pts, &required_options()).expect("run on Metal")[0];
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "`{body}` on Metal (required options) gave {got:e} (bits {:#010x}), want {want:e}",
            got.to_bits()
        );
        compared += 1;
    }
    assert_eq!(compared, FORMS.len());
    eprintln!("Metal, required options: {compared} multiply-add forms bit-exact");
}

/// Why `REQUIRED_COMPILE_OPTIONS` turns fast math off: with Metal's defaults (fast
/// math on) the negation of a zero sum loses its sign. Measured on Apple silicon
/// (Metal language 2.0, the discriminator input): `-(p.x * p.y + p.z)` gives
/// `0x00000000` with the defaults and `0x80000000` with the required options; the
/// other two forms give `+0` under both. This asserts the observed default
/// behaviour, so it pins a driver, not this crate: it is ignored in CI and run by
/// hand where Metal exists. If it fails, a driver stopped losing the sign and the
/// note on `REQUIRED_COMPILE_OPTIONS` can be revisited.
#[test]
#[ignore = "driver observation: Metal default fast math loses the sign of zero"]
fn metal_default_fast_math_loses_the_sign_of_zero() {
    let Some(device) = device_or_skip() else {
        return;
    };
    let pts = [discriminator()];
    let mut compared = 0;
    for (body, want) in FORMS {
        let shader = MslShader::from_wgsl(&kernel_wgsl(body))
            .expect("transpile")
            .into_parts()
            .0;
        let got =
            run_on_metal(&device, &shader, &pts, &default_options()).expect("run on Metal")[0];
        // fast math drops the sign of a zero: every form comes back +0
        assert_eq!(
            got.to_bits(),
            0.0f32.to_bits(),
            "`{body}` on Metal (default options) gave {got:e} (bits {:#010x}); observed was +0",
            got.to_bits()
        );
        compared += 1;
        if want.is_sign_negative() {
            eprintln!("Metal, default options: `{body}` lost the sign (want {want:e}, got +0)");
        }
    }
    assert_eq!(compared, FORMS.len());
}

//! A module-scope array is never longer than one SPIR-V instruction can hold.
//!
//! A SPIR-V instruction holds at most 65535 words, so a constant composite of more than 65532
//! elements cannot be encoded. naga writes it anyway, with a word count of 0, and the module is invalid
//! on every Vulkan driver (lavapipe loses the device at the next pipeline creation). A loft at a body's
//! size carries about 85000 values, so the emitters write large arrays as consecutive chunks, read
//! through one function (`transpiler_common::split_large_arrays`).
//!
//! Here: a data array of more than 65532 values, emitted in WGSL (Hardcoded) and in GLSL (the Vulkan
//! compute form), each turned into SPIR-V by naga and walked word by word. No instruction may have a
//! word count of 0, the last instruction must end exactly at the module's end, and the chunks must
//! hold more values than one instruction could. The same field on the GPU must match the CPU.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "glsl", feature = "gpu"))]

use alice_sdf::compiled::{GlslShader, GlslTranspileMode, GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;

mod common;
use common::spirv::{instructions, spirv};

/// `words` written to `$ALICE_SDF_SPIRV_DIR/name` when that is set (the CI lane runs spirv-val on them)
fn keep(name: &str, words: &[u32]) {
    if let Some(dir) = std::env::var_os("ALICE_SDF_SPIRV_DIR") {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(std::path::Path::new(&dir).join(name), bytes).expect("write SPIR-V");
    }
}

/// A sphere displaced by a 300 x 300 height map: one data array of 90000 values, more than one
/// SPIR-V instruction can hold (a loft at a body's size reaches the same size, but takes minutes to
/// build; the splitting acts on the emitted text, whichever node wrote the array)
fn large_field() -> SdfNode {
    let (w, h) = (300u32, 300u32);
    let map: Vec<f32> = (0..w * h)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    SdfNode::sphere(0.5).heightmap_displacement(map, w, h, 0.02, 1.0)
}

/// The count of OpConstant (opcode 43) instructions: the data values are float constants
fn constants(ins: &[(u16, u16)]) -> usize {
    ins.iter().filter(|(op, _)| *op == 43).count()
}

#[test]
fn a_large_data_array_gives_spirv_whose_instructions_fit() {
    let node = large_field();
    let wgsl = WgslShader::try_transpile(&node, TranspileMode::Hardcoded).expect("wgsl");
    let largest = wgsl
        .source
        .lines()
        .filter_map(|l| {
            l.strip_prefix("const ")?
                .split_once(": array<f32, ")?
                .1
                .split_once('>')?
                .0
                .parse::<usize>()
                .ok()
        })
        .max()
        .expect("the data is a module array");
    assert!(largest <= 16384, "a module array of {largest} elements");
    assert!(
        wgsl.source.contains("_c1"),
        "the data is not large enough to be chunked"
    );
    let module =
        naga::front::wgsl::parse_str(&wgsl.to_compute_shader()).expect("naga parses the WGSL");
    let words = spirv(&module);
    keep("large_field.wgsl.spv", &words);
    let ins = instructions(&words).expect("WGSL to SPIR-V");
    assert!(ins.len() > 100, "{} instructions", ins.len());
    let data: usize = wgsl
        .source
        .lines()
        .filter(|l| l.starts_with("const ") && l.contains("_c"))
        .map(|l| l.matches(", ").count() + 1)
        .sum();
    assert!(
        data > 65532,
        "the chunks hold {data} values, not more than one instruction could"
    );
    eprintln!(
        "wgsl: {} instructions, {} float constants, {data} values in chunks",
        ins.len(),
        constants(&ins)
    );

    let glsl = GlslShader::try_transpile(&node, GlslTranspileMode::Hardcoded).expect("glsl");
    let text = glsl.to_vulkan_compute_shader();
    let mut fe = naga::front::glsl::Frontend::default();
    let module = fe
        .parse(
            &naga::front::glsl::Options::from(naga::ShaderStage::Compute),
            &text,
        )
        .expect("naga parses the GLSL");
    let words = spirv(&module);
    keep("large_field.glsl.spv", &words);
    let ins = instructions(&words).expect("GLSL to SPIR-V");
    assert!(ins.len() > 100, "{} instructions", ins.len());
    eprintln!("glsl: {} instructions", ins.len());
}

#[test]
fn the_walker_finds_an_overflowed_word_count() {
    // the shape naga writes for 65533 constituents: the count field holds 0
    let mut words = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
    words.extend([(4u32 << 16) | 0x2c, 1, 2, 3, 0]); // opcode 44, OpConstantComposite
    let e = instructions(&words).unwrap_err();
    assert!(e.contains("word count of 0"), "{e}");
    let mut short = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
    short.extend([(5u32 << 16) | 0x2c, 1, 2]);
    assert!(instructions(&short).unwrap_err().contains("ends at word"));
}

#[test]
fn a_chunked_array_on_the_gpu_matches_the_cpu() {
    let node = large_field();
    // the Hardcoded emit, whose data is chunked module arrays (GpuEvaluator::new uses the node
    // data buffer instead)
    let wgsl = WgslShader::try_transpile(&node, TranspileMode::Hardcoded).expect("wgsl");
    assert!(
        wgsl.source.contains("_c1"),
        "the data is not large enough to be chunked"
    );
    let Ok(gpu) = GpuEvaluator::from_shader(&wgsl) else {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but there is no GPU"
        );
        eprintln!("skipping: no GPU");
        return;
    };
    let pts: Vec<Vec3> = (0..2048)
        .map(|i| {
            let t = i as f32 / 2048.0;
            let (s, c) = alice_det_math::sin_cos(37.0 * t);
            Vec3::new(0.35 * c, 1.7 * t - 0.1, 0.35 * s)
        })
        .collect();
    let got = gpu.eval_batch(&pts).expect("gpu");
    let mut worst = 0.0f32;
    for (p, g) in pts.iter().zip(&got) {
        worst = worst.max((eval(&node, *p) - g).abs());
    }
    eprintln!(
        "wgsl: worst |gpu - cpu| {worst:e} over {} points",
        pts.len()
    );
    assert!(worst < 1e-4, "wgsl: worst |gpu - cpu| {worst}");
    // the GLSL Vulkan compute form writes its module arrays as variables (private copies). Metal
    // refuses that at this size ("Compute function exceeds available stack space"); the form is for
    // Vulkan, so its GPU leg runs where wgpu's backend is Vulkan (the lavapipe lane). Its SPIR-V is
    // checked on every platform by `a_large_data_array_gives_spirv_whose_instructions_fit`
    if cfg!(target_os = "macos") {
        eprintln!("glsl: GPU leg not run on Metal (variable arrays at this size exceed its stack)");
        return;
    }
    let glsl = GlslShader::try_transpile(&node, GlslTranspileMode::Hardcoded).expect("glsl");
    let gl = GpuEvaluator::from_glsl_compute(&glsl.to_vulkan_compute_shader())
        .map(|c| c.into_parts().0)
        .expect("glsl evaluator");
    let got = gl.eval_batch(&pts).expect("gpu glsl");
    let mut worst = 0.0f32;
    for (p, g) in pts.iter().zip(&got) {
        worst = worst.max((eval(&node, *p) - g).abs());
    }
    eprintln!("glsl: worst |gpu - cpu| {worst:e}");
    assert!(worst < 1e-4, "glsl: worst |gpu - cpu| {worst}");
}

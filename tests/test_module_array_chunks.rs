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

/// The height map's side (the field reads element `y * W + x`)
const W: u32 = 300;

/// A point whose height-map lookup reads texel `(x, y)` and its +1 neighbours with non-zero
/// weights: on the XY face (|z| dominant), `u = x + 0.25`, `v = y + 0.25` in texel units
fn point_reading(x: u32, y: u32) -> Vec3 {
    let side = (W - 1) as f32;
    let to = |t: u32| ((t as f32 + 0.25) / side) * 2.0 - 1.0;
    Vec3::new(to(x), to(y), 1.5)
}

/// The elements the point's bilinear lookup reads (`heightmap_displacement` on the XY face)
fn elements_read(p: Vec3) -> [usize; 4] {
    let side = (W - 1) as f32;
    let u = ((p.x * 0.5 + 0.5) * side).clamp(0.0, side);
    let v = ((p.y * 0.5 + 0.5) * side).clamp(0.0, side);
    let (u0, v0) = (u.floor() as u32, v.floor() as u32);
    let (u1, v1) = ((u0 + 1).min(W - 1), (v0 + 1).min(W - 1));
    let at = |x: u32, y: u32| (y * W + x) as usize;
    [at(u0, v0), at(u1, v0), at(u0, v1), at(u1, v1)]
}

/// The points: for every chunk boundary `b` the elements `b - 1` and `b`, and the last element.
/// The property is that the chunked arrays read back the right element on the GPU, so every chunk
/// and both sides of every boundary are read, not many points (2048 points of this Hardcoded form
/// did not finish in 600 s on llvmpipe, which copies the module arrays per invocation)
fn boundary_points(chunk: usize, len: usize) -> Vec<Vec3> {
    let mut want: Vec<usize> = Vec::new();
    let mut b = chunk;
    while b < len {
        want.push(b - 1);
        want.push(b);
        b += chunk;
    }
    want.push(len - 1);
    let mut pts: Vec<Vec3> = want
        .iter()
        .map(|&e| {
            let (x, y) = ((e as u32) % W, (e as u32) / W);
            // the last column and row have no +1 neighbour: read them as the +1 side
            point_reading(x.min(W - 2), y.min(W - 2))
        })
        .collect();
    // the element itself must be among the four read; at the last column / row it is the +1 one
    for (p, e) in pts.iter().zip(&want) {
        assert!(
            elements_read(*p).contains(e),
            "a point for element {e} does not read it"
        );
    }
    pts.dedup();
    pts
}

#[test]
fn a_chunked_array_on_the_gpu_matches_the_cpu() {
    let node = large_field();
    // the Hardcoded emit, whose data is chunked module arrays (GpuEvaluator::new uses the node
    // data buffer instead)
    let wgsl = WgslShader::try_transpile(&node, TranspileMode::Hardcoded).expect("wgsl");
    // the chunk size and the array length, read from the chunks' declared types
    // (`const NAME_cK: array<f32, N> = ...`)
    let sizes: Vec<usize> = wgsl
        .source
        .lines()
        .filter(|l| l.starts_with("const ") && l.contains("_c"))
        .filter_map(|l| {
            l.split_once(": array<f32, ")?
                .1
                .split_once('>')?
                .0
                .parse::<usize>()
                .ok()
        })
        .collect();
    assert!(
        sizes.len() >= 2,
        "the data is not large enough to be chunked ({} chunks)",
        sizes.len()
    );
    let (chunk, len) = (sizes[0], sizes.iter().sum::<usize>());
    assert_eq!(
        len,
        (W * W) as usize,
        "the chunks hold the whole height map"
    );
    let pts = boundary_points(chunk, len);
    assert!(!pts.is_empty(), "no point");
    // every chunk is read, and both sides of every boundary
    let read: std::collections::BTreeSet<usize> =
        pts.iter().flat_map(|p| elements_read(*p)).collect();
    let chunks_read: std::collections::BTreeSet<usize> = read.iter().map(|e| e / chunk).collect();
    assert_eq!(
        chunks_read.len(),
        sizes.len(),
        "chunks read {chunks_read:?} of {}",
        sizes.len()
    );
    let mut b = chunk;
    while b < len {
        assert!(
            read.contains(&(b - 1)) && read.contains(&b),
            "boundary {b} not read on both sides"
        );
        b += chunk;
    }
    assert!(read.contains(&(len - 1)), "the last element is not read");
    let Ok(gpu) = GpuEvaluator::from_shader(&wgsl) else {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but there is no GPU"
        );
        eprintln!("skipping: no GPU");
        return;
    };
    let got = gpu.eval_batch(&pts).expect("gpu");
    eprintln!(
        "wgsl: {} points over {} chunks of {chunk}",
        pts.len(),
        sizes.len()
    );
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

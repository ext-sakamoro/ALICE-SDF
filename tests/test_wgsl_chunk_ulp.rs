//! Splitting WGSL data arrays at 2047 elements instead of 16384 does not take the GPU field further
//! from the CPU field.
//!
//! The chunk size changes only the function that reads the data (more, shorter branches); the
//! values and the order of every read are the same. A driver that compiles with fast math (Metal
//! through wgpu does) may still round differently around the read, so the GPU bits can move by an
//! ulp. The rule here is the one for every GPU rounding change: per tree, the largest GPU - CPU
//! difference after is at most the one before.
//!
//! Here: every corpus tree whose WGSL has a split array, and a 300 x 300 height map, on the GPU in
//! the emitted form (chunks of 2047) and in the same text regrouped into chunks of 16384 (the form
//! before), each against the CPU at the same points. Comparing 0 trees fails.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

mod common;
#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use std::collections::BTreeMap;

/// The chunk size of the WGSL data arrays before the spec limit was applied
const BEFORE_CHUNK: usize = 16384;

/// The chunk size the WGSL emitter writes (W3C WGSL, section 2.4 "Limits": 2047 elements per array
/// value constructor)
const EMITTED_CHUNK: usize = 2047;

fn large_field() -> SdfNode {
    let (w, h) = (300u32, 300u32);
    let map: Vec<f32> = (0..w * h)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    SdfNode::sphere(0.5).heightmap_displacement(map, w, h, 0.02, 1.0)
}

/// The split arrays of `src` (`NAME_c0`, `NAME_c1`, … joined back per name, in order) and `src` with
/// each array's chunks and reader function `NAME_r` replaced by one marker line `\0NAME` (where the
/// transpiler wrote them)
fn unsplit(src: &str) -> (BTreeMap<String, Vec<String>>, String) {
    let chunk_of = |line: &str| -> Option<(String, usize)> {
        let rest = line.strip_prefix("const ")?;
        let (name, _) = rest.split_once(": array<f32, ")?;
        let (base, k) = name.rsplit_once("_c")?;
        Some((base.to_owned(), k.parse().ok()?))
    };
    let mut chunks: BTreeMap<String, Vec<(usize, Vec<String>)>> = BTreeMap::new();
    let mut rest = String::new();
    let mut in_reader = false;
    for line in src.lines() {
        if in_reader {
            in_reader = line != "}";
            continue;
        }
        if let Some((base, k)) = chunk_of(line) {
            let open = line.find(">(").expect("constructor") + 2;
            let close = line.rfind(')').expect("constructor end");
            let items = line[open..close].split(", ").map(str::to_owned).collect();
            if !chunks.contains_key(&base) {
                rest.push_str(&format!("\0{base}\n"));
            }
            chunks.entry(base).or_default().push((k, items));
            continue;
        }
        let reader = line
            .strip_prefix("fn ")
            .and_then(|r| r.split_once("_r(i: i32) -> f32 {"))
            .is_some_and(|(b, _)| chunks.contains_key(b));
        if reader {
            in_reader = true;
            continue;
        }
        rest.push_str(line);
        rest.push('\n');
    }
    let arrays = chunks
        .into_iter()
        .map(|(b, mut v)| {
            v.sort_by_key(|x| x.0);
            (b, v.into_iter().flat_map(|x| x.1).collect())
        })
        .collect();
    (arrays, rest)
}

/// `name`'s chunks of at most `chunk` elements and its reader, as the transpiler writes them
fn chunked(name: &str, items: &[String], chunk: usize) -> String {
    let parts = items.len().div_ceil(chunk);
    let mut out = String::new();
    let mut body = String::new();
    for k in 0..parts {
        let it = &items[k * chunk..((k + 1) * chunk).min(items.len())];
        let n = it.len();
        out.push_str(&format!(
            "const {name}_c{k}: array<f32, {n}> = array<f32, {n}>({});\n",
            it.join(", ")
        ));
        let read = format!("{name}_c{k}[i - {}]", k * chunk);
        if k + 1 < parts {
            body.push_str(&format!(
                "    if (i < {}) {{ return {read}; }}\n",
                (k + 1) * chunk
            ));
        } else {
            body.push_str(&format!("    return {read};\n"));
        }
    }
    out.push_str(&format!("fn {name}_r(i: i32) -> f32 {{\n{body}}}\n"));
    out
}

/// `src` with its split arrays regrouped into chunks of `chunk`, or `None` when it has none
fn regrouped(src: &str, chunk: usize) -> Option<String> {
    let (arrays, mut out) = unsplit(src);
    if arrays.is_empty() {
        return None;
    }
    for (b, items) in &arrays {
        if items.len() > chunk {
            out = out.replacen(&format!("\0{b}\n"), &chunked(b, items, chunk), 1);
        } else {
            // an array that fits one chunk is written whole and read directly, as the transpiler does
            let n = items.len();
            let decl = format!(
                "const {b}: array<f32, {n}> = array<f32, {n}>({});\n",
                items.join(", ")
            );
            out = direct_reads(&out.replacen(&format!("\0{b}\n"), &decl, 1), b);
        }
    }
    // the emitted text may not end with a newline
    if !src.ends_with('\n') {
        out.pop();
    }
    Some(out)
}

/// `src` with every read `NAME_r(i32(e))` written back as `NAME[e]`
fn direct_reads(src: &str, name: &str) -> String {
    let pat = format!("{name}_r(i32(");
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while let Some(rel) = src[i..].find(&pat) {
        let at = i + rel;
        out.push_str(&src[i..at]);
        let open = at + pat.len();
        let (mut depth, mut j) = (1usize, open);
        let bytes = src.as_bytes();
        while depth > 0 {
            match bytes[j] {
                b'(' => depth += 1,
                b')' => depth -= 1,
                _ => {}
            }
            j += 1;
        }
        assert_eq!(bytes[j], b')', "a read of {name} not closed");
        out.push_str(&format!("{name}[{}]", &src[open..j - 1]));
        i = j + 1;
    }
    out.push_str(&src[i..]);
    out
}

fn points() -> Vec<Vec3> {
    (0..4096u32)
        .map(|i| {
            let h = i.wrapping_mul(2_654_435_761);
            let g = |s: u32| ((h >> s) & 1023) as f32 / 1023.0 * 1.6 - 0.8;
            Vec3::new(g(0), g(10), g(20))
        })
        .collect()
}

/// The largest |GPU - CPU| of `shader` at `pts` and the number of points whose bits differ from the
/// CPU's, or `None` without a GPU
fn deviation(shader: &WgslShader, cpu: &[f32], pts: &[Vec3]) -> Option<(f32, usize)> {
    let g = match GpuEvaluator::from_shader(shader) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("skipping: {e}");
            return None;
        }
    };
    let gpu = gpu_batches::eval_in_batches(&g, pts).expect("dispatch");
    assert_eq!(gpu.len(), pts.len());
    let max = cpu
        .iter()
        .zip(&gpu)
        .map(|(c, g)| (c - g).abs())
        .fold(0.0f32, f32::max);
    let differ = cpu
        .iter()
        .zip(&gpu)
        .filter(|(c, g)| c.to_bits() != g.to_bits())
        .count();
    Some((max, differ))
}

#[test]
fn the_smaller_wgsl_chunks_keep_the_gpu_as_close_to_the_cpu() {
    let pts = points();
    let mut trees: Vec<(String, SdfNode)> = common::corpus::corpus()
        .into_iter()
        .map(|(n, x)| (n.to_string(), x))
        .collect();
    trees.push(("height map 300 x 300".into(), large_field()));
    let (mut compared, mut worse) = (0usize, Vec::new());
    for (name, node) in &trees {
        let after = WgslShader::transpile(node, TranspileMode::Hardcoded);
        let Some(before_src) = regrouped(&after.source, BEFORE_CHUNK) else {
            continue;
        };
        // the regrouping is lossless: regrouped at the emitted size it is the emitted text
        assert_eq!(
            regrouped(&after.source, EMITTED_CHUNK).as_deref(),
            Some(after.source.as_str()),
            "{name}: the regrouping does not give back the emitted text"
        );
        let mut before = after.clone();
        before.source = before_src;
        let cpu: Vec<f32> = pts.iter().map(|p| eval(node, *p)).collect();
        let Some((b, nb)) = deviation(&before, &cpu, &pts) else {
            return;
        };
        let (a, na) = deviation(&after, &cpu, &pts).expect("the GPU built the other form");
        eprintln!(
            "{name}: max |gpu - cpu| {b:e} -> {a:e}, points off the cpu bits {nb} -> {na} of {}",
            pts.len()
        );
        if a > b {
            worse.push(format!("{name}: {b:e} -> {a:e}"));
        }
        compared += 1;
    }
    assert!(compared > 0, "no tree with a split WGSL array was compared");
    assert!(
        worse.is_empty(),
        "the 2047-element chunks take the GPU further from the CPU:\n{}",
        worse.join("\n")
    );
}

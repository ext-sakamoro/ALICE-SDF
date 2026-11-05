//! `SectionStack` in the shaders: the WGSL, GLSL and HLSL emits evaluate the same field as the
//! CPU law (`SectionStack::eval`).
//!
//! The shared corpus (`tests/common/corpus.rs`) lists the nodes the bytecode compiler accepts; a
//! section stack is refused there, so its shader oracles live here:
//!
//! | output | oracle |
//! |---|---|
//! | WGSL (`GpuEvaluator::new`) | runs on the GPU, every point within `TOL` of the CPU |
//! | GLSL (`to_vulkan_compute_shader`) | runs on the GPU through naga's GLSL front end, within `TOL` |
//! | HLSL (`HlslShader::transpile`) | compiled as C++ with `tests/common/hlsl_cpu_shim.h`, within `TOL` |
//!
//! The stacks exercise every branch of the law: several loops on one level (even-odd over all of
//! them, a hole inside a loop), loops that merge between levels, a concave loop, points below the
//! first level, above the last and between levels, inside and outside, and a stack placed in a tree
//! by a rotation and a translation beside a polygon. The points are dense at the levels and caps.
//! A GPU or compiler that is missing skips, unless `ALICE_SDF_REQUIRE_GPU` / `ALICE_SDF_REQUIRE_CXX`
//! is set; every oracle asserts it compared a non-zero number of points.
//!
//! Tolerance: `TOL = 2e-5` absolute (the fields stay within a few units; GPU `sqrt` / `length` are
//! not correctly rounded, the even-odd sign and the level choice are exact comparisons).
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "glsl", feature = "hlsl"))]
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm builds the loop vertices"
)]

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use alice_sdf::compiled::{
    GlslShader, GlslTranspileMode, GpuEvaluator, HlslShader, HlslTranspileMode,
};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const TOL: f32 = 2e-5;

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

fn circle(c: Vec2, r: f32, n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let a = std::f32::consts::TAU * k as f32 / n as f32;
            c + Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

fn star(n: usize, r_out: f32, r_in: f32) -> Vec<Vec2> {
    (0..2 * n)
        .map(|k| {
            let a = std::f32::consts::PI * k as f32 / n as f32;
            let r = if k % 2 == 0 { r_out } else { r_in };
            Vec2::new(r * a.cos(), r * a.sin())
        })
        .collect()
}

/// Two separate loops at the bottom, one loop with a hole in the middle, a concave star at the top
fn stack() -> SdfNode {
    SdfNode::section_stack(
        vec![0.0, 0.3, 0.55, 0.9],
        vec![
            vec![
                circle(Vec2::new(-0.25, 0.0), 0.15, 24),
                circle(Vec2::new(0.25, 0.0), 0.15, 24),
            ],
            vec![
                circle(Vec2::new(-0.1, 0.0), 0.22, 30),
                circle(Vec2::new(0.1, 0.0), 0.22, 30),
            ],
            vec![circle(Vec2::ZERO, 0.35, 40), circle(Vec2::ZERO, 0.12, 20)],
            vec![star(7, 0.3, 0.15)],
        ],
    )
    .expect("a stack")
}

/// The stack placed in a tree: rotated, moved, unioned with a polygon extrusion
fn scene() -> SdfNode {
    stack()
        .rotate_euler(0.3, -0.5, 0.2)
        .translate(0.1, -0.2, 0.05)
        .union(SdfNode::polygon_2d(star(5, 0.2, 0.1), 0.1).translate(0.6, 0.4, 0.0))
}

/// Points around the stack, half of them on or near the levels and the caps
fn points(n: usize) -> Vec<Vec3> {
    let mut r = lcg(0x5EC7);
    let levels = [0.0f32, 0.3, 0.55, 0.9];
    (0..n)
        .map(|i| {
            let x = r() * 1.4 - 0.7;
            let z = r() * 1.4 - 0.7;
            let y = if i % 2 == 0 {
                levels[(r() * 4.0) as usize % 4] + (r() - 0.5) * 0.01
            } else {
                r() * 1.5 - 0.3
            };
            Vec3::new(x, y, z)
        })
        .collect()
}

fn worst(node: &SdfNode, pts: &[Vec3], got: &[f32]) -> (f32, Vec3) {
    assert_eq!(got.len(), pts.len());
    assert!(!pts.is_empty(), "no point compared");
    let mut w = (0.0f32, Vec3::ZERO);
    for (p, g) in pts.iter().zip(got) {
        let d = (eval(node, *p) - g).abs();
        assert!(
            d.is_finite(),
            "non-finite at {p:?}: cpu {}, shader {g}",
            eval(node, *p)
        );
        if d > w.0 {
            w = (d, *p);
        }
    }
    w
}

fn require_gpu<T, E: std::fmt::Display>(what: &str, r: Result<T, E>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "{what}: ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("{what}: skipped ({e})");
            None
        }
    }
}

#[test]
fn the_stack_has_both_signs_and_every_branch_among_the_points() {
    // the oracles below would pass vacuously on points that never reach a branch
    let node = stack();
    let pts = points(20_000);
    let inside = pts.iter().filter(|p| eval(&node, **p) < 0.0).count();
    let below = pts.iter().filter(|p| p.y < 0.0).count();
    let above = pts.iter().filter(|p| p.y > 0.9).count();
    let hole = pts
        .iter()
        .filter(|p| (0.5..0.6).contains(&p.y) && p.x.hypot(p.z) < 0.1)
        .count();
    assert!(
        inside > 500 && below > 500 && above > 500 && hole > 20,
        "{inside} {below} {above} {hole}"
    );
}

#[test]
fn wgsl_matches_the_cpu_on_the_gpu() {
    for (what, node) in [("stack", stack()), ("scene", scene())] {
        let Some(gpu) = require_gpu(what, GpuEvaluator::new(&node)) else {
            return;
        };
        let pts = points(20_000);
        let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("gpu eval");
        let (w, at) = worst(&node, &pts, &got);
        assert!(
            w < TOL,
            "{what}: WGSL differs from the CPU by {w} at {at:?}"
        );
    }
}

#[test]
fn glsl_matches_the_cpu_on_the_gpu() {
    for (what, node) in [("stack", stack()), ("scene", scene())] {
        let src =
            GlslShader::transpile(&node, GlslTranspileMode::Hardcoded).to_vulkan_compute_shader();
        let Some(gpu) = require_gpu(
            what,
            GpuEvaluator::from_glsl_compute(&src).map(|c| c.into_parts().0),
        ) else {
            return;
        };
        let pts = points(20_000);
        let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("gpu eval");
        let (w, at) = worst(&node, &pts, &got);
        assert!(
            w < TOL,
            "{what}: GLSL differs from the CPU by {w} at {at:?}"
        );
    }
}

// ---- HLSL as C++ (the route of tests/test_hlsl_blinkscript_parity.rs) ----

const SHIM: &str = include_str!("common/hlsl_cpu_shim.h");
const HOST_MAIN: &str = r"
static bool read_exact(void *dst, unsigned long n) {
    char *p = (char *)dst; unsigned long have = 0;
    while (have < n) { long r = read(0, p + have, n - have); if (r <= 0) { return false; } have += (unsigned long)r; }
    return true;
}
static bool write_exact(const void *src, unsigned long n) {
    const char *p = (const char *)src; unsigned long sent = 0;
    while (sent < n) { long r = write(1, p + sent, n - sent); if (r <= 0) { return false; } sent += (unsigned long)r; }
    return true;
}
int main() {
    ALICE_SDF_SET_BINARY_IO();
    unsigned int count = 0;
    if (!read_exact(&count, 4)) { return 2; }
    for (unsigned int i = 0; i < count; ++i) {
        float buf[3];
        if (!read_exact(buf, 12)) { return 3; }
        float d = sdf_eval(float3(buf[0], buf[1], buf[2]));
        if (!write_exact(&d, 4)) { return 4; }
    }
    return 0;
}
";

fn cxx_or_skip() -> Option<String> {
    for cand in ["c++", "clang++", "g++"] {
        if Command::new(cand)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
        {
            return Some(cand.to_string());
        }
    }
    assert!(
        std::env::var_os("ALICE_SDF_REQUIRE_CXX").is_none(),
        "ALICE_SDF_REQUIRE_CXX is set but no C++ compiler was found"
    );
    eprintln!("skipping the HLSL oracle: no C++ compiler");
    None
}

fn compile_and_run(cxx: &str, dir: &Path, name: &str, src: &str, pts: &[Vec3]) -> Vec<f32> {
    let cpp = dir.join(format!("{name}.cpp"));
    let bin: PathBuf = dir.join(name);
    std::fs::write(&cpp, src).expect("write cpp");
    let out = Command::new(cxx)
        .args([
            "-std=c++17",
            "-O1",
            "-ffp-contract=off",
            "-fno-strict-aliasing",
            "-w",
        ])
        .arg(&cpp)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("run the compiler");
    assert!(
        out.status.success(),
        "C++ compile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .take(15)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mut payload = (pts.len() as u32).to_ne_bytes().to_vec();
    for p in pts {
        for c in [p.x, p.y, p.z] {
            payload.extend_from_slice(&c.to_ne_bytes());
        }
    }
    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let writer = std::thread::spawn(move || stdin.write_all(&payload));
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .expect("stdout")
        .read_to_end(&mut raw)
        .expect("read");
    writer.join().expect("writer").expect("write");
    assert!(child.wait().expect("wait").success(), "evaluator failed");
    assert_eq!(raw.len(), pts.len() * 4);
    raw.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn hlsl_matches_the_cpu_compiled_as_cpp() {
    let Some(cxx) = cxx_or_skip() else { return };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("section_stack_hlsl");
    std::fs::create_dir_all(&dir).expect("scratch");
    for (what, node) in [("stack", stack()), ("scene", scene())] {
        let emit = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source;
        let src = format!("{SHIM}\n{emit}\n{HOST_MAIN}");
        let pts = points(20_000);
        let got = compile_and_run(&cxx, &dir, what, &src, &pts);
        let (w, at) = worst(&node, &pts, &got);
        assert!(
            w < TOL,
            "{what}: HLSL differs from the CPU by {w} at {at:?}"
        );
    }
}

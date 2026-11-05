//! The bound skip in the shaders: a union operand whose proven lower bound is already no closer
//! than the other operand is not evaluated, and the result is bit-identical to evaluating it.
//!
//! | oracle | how |
//! |---|---|
//! | the bound holds | for every subtree `lower_bound` covers, the CPU field is ≥ `scale·dist(p, box) − margin` wherever that is positive (points on and around every box face) |
//! | skip = no skip, HLSL | the emit compiled as C++ (`-ffp-contract=off`, deterministic), ≥ 1e6 points, bit for bit |
//! | skip = no skip, WGSL | both shaders on the GPU, ≥ 1e6 points, bit for bit |
//! | not vacuous | the scenes emit skip tests, the reference emits none, and points fall near and far |
//!
//! Points: half on the faces of the skip boxes (± 2 cm, so both sides of each face), the rest in
//! the scene's hull, where the operands overlap.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "hlsl"))]
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm builds the outlines"
)]

use alice_sdf::compiled::transpiler_common::{lower_bound, LowerBound, BOUND_SKIP_EPS};
use alice_sdf::compiled::{GpuEvaluator, HlslShader, HlslTranspileMode, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
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

fn ellipse(a: f32, b: f32, n: usize) -> Vec<Vec2> {
    (0..n)
        .map(|k| {
            let t = std::f32::consts::TAU * k as f32 / n as f32;
            Vec2::new(a * t.cos(), b * t.sin())
        })
        .collect()
}

/// A hull like the body's parts: the intersection of extruded outlines seen from rotated
/// directions, cut by a plane
fn hull(seed: u64, at: Vec3) -> SdfNode {
    let mut r = lcg(seed);
    let mut node = SdfNode::polygon_2d(ellipse(0.12 + 0.05 * r(), 0.3, 24), 0.2);
    for k in 0..3 {
        let side = SdfNode::polygon_2d(star(5 + k, 0.25, 0.18), 0.2).rotate_euler(
            r() * 3.0,
            r() * 3.0,
            r() * 3.0,
        );
        node = node.intersection(side);
    }
    node.intersection(SdfNode::plane(Vec3::new(0.0, -1.0, 0.0), 0.25))
        .translate(at.x, at.y, at.z)
}

/// Five parts joined like the body: a smooth-union chain whose operands are unions of hulls
fn body_like() -> SdfNode {
    let parts: Vec<SdfNode> = (0..5)
        .map(|i| {
            let at = Vec3::new(-0.6 + 0.3 * i as f32, 0.1 * (i % 2) as f32, 0.0);
            hull(10 + i as u64, at).union(hull(20 + i as u64, at + Vec3::new(0.05, 0.2, 0.0)))
        })
        .collect();
    let mut node = parts[0].clone();
    for p in &parts[1..] {
        node = node.smooth_union(p.clone(), 0.01);
    }
    node
}

/// A plain union chain of a section stack and rotated, translated stars
fn union_chain() -> SdfNode {
    let st = SdfNode::section_stack(
        vec![0.0, 0.2, 0.5],
        vec![
            vec![ellipse(0.2, 0.1, 30)],
            vec![ellipse(0.1, 0.2, 30)],
            vec![star(6, 0.2, 0.1)],
        ],
    )
    .expect("stack");
    let mut node = st.translate(0.0, -0.3, 0.0);
    for i in 0..6 {
        let s = SdfNode::polygon_2d(star(40, 0.15, 0.07), 0.05 + 0.01 * i as f32)
            .rotate_euler(0.4 * i as f32, 0.2, -0.3 * i as f32)
            .translate(0.25 * i as f32 - 0.6, 0.3, 0.1 * (i % 3) as f32);
        node = node.union(s);
    }
    node
}

/// A capped polar loft along a gently curved axis, one non-circular section along it
fn limb(seed: u64, from: Vec3, to: Vec3, r0: f32, k: u32) -> SdfNode {
    let mut r = lcg(seed);
    let axis: Vec<Vec3> = (0..6)
        .map(|i| {
            let t = i as f32 / 5.0;
            from + (to - from) * t + Vec3::new(0.02 * (t * 3.0).sin(), 0.0, 0.015 * (t * 2.0).cos())
        })
        .collect();
    // one section along the whole loft: the loft law's constant-bound branch (m = 0); a section
    // that varies along the axis gives a negative clamp m, an open defect of the loft emit
    let w = 2 * k as usize + 1;
    let mut row = vec![0.0; w];
    row[0] = r0;
    for h in 1..w {
        row[h] = r0 * 0.03 * (r() - 0.5) / (1 + h / 2) as f32;
    }
    let rows: Vec<Vec<f32>> = vec![row; 12];
    SdfNode::loft_capped(axis, Vec3::Z, k, rows, [r0, r0]).expect("loft")
}

/// Six lofts joined like the body's smooth parts: a smooth-union chain
fn loft_body() -> SdfNode {
    let parts = [
        limb(
            1,
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::new(0.0, 1.4, 0.0),
            0.14,
            16,
        ),
        limb(
            2,
            Vec3::new(0.15, 1.35, 0.0),
            Vec3::new(0.7, 1.35, 0.0),
            0.045,
            8,
        ),
        limb(
            3,
            Vec3::new(-0.15, 1.35, 0.0),
            Vec3::new(-0.7, 1.35, 0.0),
            0.045,
            8,
        ),
        limb(
            4,
            Vec3::new(0.08, 0.85, 0.0),
            Vec3::new(0.1, 0.05, 0.0),
            0.06,
            8,
        ),
        limb(
            5,
            Vec3::new(-0.08, 0.85, 0.0),
            Vec3::new(-0.1, 0.05, 0.0),
            0.06,
            8,
        ),
        limb(
            6,
            Vec3::new(0.0, 1.45, 0.0),
            Vec3::new(0.0, 1.62, 0.02),
            0.09,
            12,
        ),
    ];
    let mut node = parts[0].clone();
    for p in &parts[1..] {
        node = node.smooth_union(p.clone(), 0.01);
    }
    node
}

fn scenes() -> Vec<(&'static str, SdfNode)> {
    vec![("body_like", body_like()), ("union_chain", union_chain())]
}

/// Every subtree's proven bound
fn bounds(node: &SdfNode, out: &mut Vec<(SdfNode, LowerBound)>) {
    if let Some(b) = lower_bound(node) {
        out.push((node.clone(), b));
    }
    match node {
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. } => {
            bounds(a, out);
            bounds(b, out);
        }
        SdfNode::Translate { child, .. } | SdfNode::Rotate { child, .. } => bounds(child, out),
        _ => {}
    }
}

/// The distance to the box inflated by `BOUND_SKIP_EPS`, as the shader's test computes it
fn box_dist(p: Vec3, b: &LowerBound) -> f32 {
    let mut s = 0.0f32;
    for k in 0..3 {
        let c = 0.5 * (b.lo[k] + b.hi[k]);
        let h = 0.5 * (b.hi[k] - b.lo[k]) + BOUND_SKIP_EPS;
        let d = ((p[k] - c).abs() - h).max(0.0);
        s += d * d;
    }
    s.sqrt()
}

/// Points on and around every box face, and in the hull
fn points(node: &SdfNode, n: usize, seed: u64) -> Vec<Vec3> {
    let mut bs = Vec::new();
    bounds(node, &mut bs);
    assert!(!bs.is_empty(), "no bounded subtree");
    let all = lower_bound(node).unwrap_or(bs[0].1);
    let mut r = lcg(seed);
    (0..n)
        .map(|i| {
            let mut p = Vec3::ZERO;
            if i % 2 == 0 {
                let b = &bs[(r() * bs.len() as f32) as usize % bs.len()].1;
                let axis = (r() * 3.0) as usize % 3;
                for k in 0..3 {
                    p[k] = b.lo[k] + r() * (b.hi[k] - b.lo[k]);
                }
                p[axis] = if r() < 0.5 { b.lo[axis] } else { b.hi[axis] } + (r() - 0.5) * 0.04;
            } else {
                for k in 0..3 {
                    let pad = 0.1;
                    p[k] = all.lo[k] - pad + r() * (all.hi[k] - all.lo[k] + 2.0 * pad);
                }
            }
            p
        })
        .collect()
}

#[test]
fn every_bound_holds_on_the_cpu() {
    for (what, node) in scenes() {
        let mut bs = Vec::new();
        bounds(&node, &mut bs);
        let pts = points(&node, 40_000, 1);
        let mut checked = 0usize;
        for (sub, b) in &bs {
            for p in pts.iter().step_by(7) {
                // the bound the shader tests: inflated box, margin + eps
                let lb = b.scale * box_dist(*p, b) - (b.margin + BOUND_SKIP_EPS);
                if lb > 0.0 {
                    let f = eval(sub, *p);
                    assert!(f >= lb, "{what}: field {f} < bound {lb} at {p:?} for {b:?}");
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 10_000,
            "{what}: only {checked} points outside a box"
        );
    }
}

#[test]
fn the_scenes_emit_skip_tests_and_the_reference_none() {
    for (what, node) in scenes() {
        let skip = WgslShader::transpile(&node, TranspileMode::Hardcoded).source;
        let plain =
            WgslShader::transpile_without_bound_skip(&node, TranspileMode::Hardcoded).source;
        let tests = skip.matches("> 0.0 &&").count();
        assert!(tests >= 4, "{what}: {tests} skip tests");
        assert_eq!(
            plain.matches("> 0.0 &&").count(),
            0,
            "{what}: the reference skips"
        );
    }
}

// ---- HLSL as C++ ----

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

fn cxx() -> Option<String> {
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
    eprintln!("skipping: no C++ compiler");
    None
}

fn run_cpp(cxx: &str, dir: &Path, name: &str, emit: &str, pts: &[Vec3]) -> Vec<u32> {
    let cpp = dir.join(format!("{name}.cpp"));
    let bin = dir.join(name);
    std::fs::write(&cpp, format!("{SHIM}\n{emit}\n{HOST_MAIN}")).expect("write");
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
        .expect("compiler");
    assert!(
        out.status.success(),
        "{name}: C++ compile failed:\n{}",
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
    assert!(child.wait().expect("wait").success());
    assert_eq!(raw.len(), pts.len() * 4);
    raw.chunks_exact(4)
        .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn hlsl_with_skips_equals_hlsl_without_bit_for_bit() {
    let Some(cxx) = cxx() else { return };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("bound_skip_hlsl");
    std::fs::create_dir_all(&dir).expect("scratch");
    for (what, node) in scenes() {
        let pts = points(&node, 1_000_000, 2);
        let skip_src = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source;
        let plain_src =
            HlslShader::transpile_without_bound_skip(&node, HlslTranspileMode::Hardcoded).source;
        let skip = run_cpp(&cxx, &dir, &format!("{what}_skip"), &skip_src, &pts);
        let plain = run_cpp(&cxx, &dir, &format!("{what}_plain"), &plain_src, &pts);
        if let Some(i) = skip.iter().zip(&plain).position(|(a, b)| a != b) {
            panic!(
                "{what}: at {:?} the skipping shader gives {} and the full one {}",
                pts[i],
                f32::from_bits(skip[i]),
                f32::from_bits(plain[i])
            );
        }
        let near = plain.iter().filter(|b| f32::from_bits(**b) < 0.05).count();
        assert!(
            near > 10_000 && near < pts.len() - 10_000,
            "{what}: {near} near points"
        );
    }
}

#[test]
fn wgsl_with_skips_equals_wgsl_without_bit_for_bit_on_the_gpu() {
    for (what, node) in scenes() {
        let skip = WgslShader::transpile(&node, TranspileMode::Hardcoded);
        let plain = WgslShader::transpile_without_bound_skip(&node, TranspileMode::Hardcoded);
        let (a, b) = match (
            GpuEvaluator::from_shader(&skip),
            GpuEvaluator::from_shader(&plain),
        ) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(e), _) | (_, Err(e)) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
                );
                eprintln!("{what}: skipped ({e})");
                return;
            }
        };
        let pts = points(&node, 1_000_000, 3);
        let ga = a.eval_batch(&pts).expect("gpu");
        let gb = b.eval_batch(&pts).expect("gpu");
        assert_eq!(ga.len(), pts.len());
        if let Some(i) = ga
            .iter()
            .zip(&gb)
            .position(|(x, y)| x.to_bits() != y.to_bits())
        {
            panic!("{what}: at {:?} skip {} vs full {}", pts[i], ga[i], gb[i]);
        }
    }
}

/// The loft body through the same oracles: the bound on the CPU, skip tests emitted, and the
/// skipping shader bit-equal to the full one on the GPU on 1e6 points
#[test]
fn loft_body_skips_bit_for_bit() {
    let node = loft_body();
    let mut all = Vec::new();
    bounds(&node, &mut all);
    for (n, _) in &all {
        if let SdfNode::Loft { bounds, .. } = n {
            assert!(
                bounds.m >= 0.0,
                "loft clamp m = {} (the emit prints `--m`)",
                bounds.m
            );
        }
    }
    let mut bs = Vec::new();
    bounds(&node, &mut bs);
    let pts = points(&node, 40_000, 4);
    for (sub, b) in &bs {
        for p in pts.iter().step_by(7) {
            let lb = b.scale * box_dist(*p, b) - (b.margin + BOUND_SKIP_EPS);
            if lb > 0.0 {
                assert!(eval(sub, *p) >= lb);
            }
        }
    }
    let skip = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    let plain = WgslShader::transpile_without_bound_skip(&node, TranspileMode::Hardcoded);
    assert!(skip.source.matches("> 0.0 &&").count() >= 4);
    let (a, b) = (
        GpuEvaluator::from_shader(&skip).expect("gpu"),
        GpuEvaluator::from_shader(&plain).expect("gpu"),
    );
    let pts = points(&node, 1_000_000, 5);
    let (ga, gb) = (
        a.eval_batch(&pts).expect("gpu"),
        b.eval_batch(&pts).expect("gpu"),
    );
    assert!(ga.iter().zip(&gb).all(|(x, y)| x.to_bits() == y.to_bits()));
}

/// The loft's bound on the CPU (independent of the emit defect above): the field is at least the
/// distance to the box of the axis's Bézier control points grown by the reach, wherever positive
#[test]
fn the_loft_bound_holds_on_the_cpu() {
    let node = loft_body();
    let mut bs = Vec::new();
    bounds(&node, &mut bs);
    let lofts = bs
        .iter()
        .filter(|(n, _)| matches!(n, SdfNode::Loft { .. }))
        .count();
    assert_eq!(lofts, 6, "every loft has a bound");
    let pts = points(&node, 60_000, 6);
    let mut checked = 0usize;
    for (sub, b) in &bs {
        for p in &pts {
            let lb = b.scale * box_dist(*p, b) - (b.margin + BOUND_SKIP_EPS);
            if lb > 0.0 {
                let f = eval(sub, *p);
                assert!(f >= lb, "field {f} < bound {lb} at {p:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 50_000, "only {checked} points outside a box");
}

/// Random lofts (polar and curve sections, flat ends, domes, curve caps with an offset apex, bent
/// axes) and random points out to several reaches: the field is at least the bound wherever the
/// bound is positive (`lower_bound_factors`' derivation), and the bound is positive at a fair share
/// of the points (a bound that never bites would pass vacuously)
#[test]
fn the_loft_bound_holds_on_random_lofts() {
    let mut r = lcg(0x10F7_B0);
    let (mut checked, mut lofts) = (0usize, 0usize);
    for case in 0..48 {
        let n = 2 + (r() * 4.0) as usize;
        let axis: Vec<Vec3> = (0..n)
            .map(|i| Vec3::new((r() - 0.5) * 0.3, 0.3 * i as f32, (r() - 0.5) * 0.3))
            .collect();
        let cap = |r: &mut dyn FnMut() -> f32| if r() < 0.5 { 0.0 } else { 0.02 + 0.08 * r() };
        let node = if case % 2 == 0 {
            let k = 1 + (r() * 3.0) as u32;
            let rows: Vec<Vec<f32>> = (0..n.max(3))
                .map(|_| {
                    let mut row = vec![0.08 + 0.06 * r()];
                    for _ in 0..k {
                        row.push((r() - 0.5) * 0.02);
                        row.push((r() - 0.5) * 0.02);
                    }
                    row
                })
                .collect();
            let caps = [cap(&mut r), cap(&mut r)];
            SdfNode::loft_capped(axis, Vec3::Z, k, rows, caps)
        } else {
            let m = 6 + (r() * 4.0) as usize;
            let rows: Vec<Vec<Vec2>> = (0..n.max(3))
                .map(|_| {
                    let (a, b) = (0.06 + 0.06 * r(), 0.05 + 0.05 * r());
                    (0..m)
                        .map(|j| {
                            let t = std::f32::consts::TAU * j as f32 / m as f32;
                            Vec2::new(a * t.cos(), b * t.sin())
                        })
                        .collect()
                })
                .collect();
            let c0 = cap(&mut r);
            let c1 = cap(&mut r);
            let caps = [[c0, (r() - 0.5) * c0, (r() - 0.5) * c0], [c1, 0.0, 0.0]];
            SdfNode::loft_curves_capped(axis, Vec3::Z, rows, caps)
        };
        let Ok(node) = node else { continue };
        let b = lower_bound(&node).expect("a loft has a bound");
        lofts += 1;
        let span = (0..3)
            .map(|k| b.hi[k] - b.lo[k])
            .fold(0.0f32, f32::max)
            .max(0.1);
        for _ in 0..4000 {
            let reach = 0.2 + 4.0 * r() * span;
            let p = Vec3::new(
                0.5 * (b.lo[0] + b.hi[0]) + (r() - 0.5) * 2.0 * reach,
                0.5 * (b.lo[1] + b.hi[1]) + (r() - 0.5) * 2.0 * reach,
                0.5 * (b.lo[2] + b.hi[2]) + (r() - 0.5) * 2.0 * reach,
            );
            let lb = b.scale * box_dist(p, &b) - (b.margin + BOUND_SKIP_EPS);
            if lb > 0.0 {
                let f = eval(&node, p);
                assert!(
                    f >= lb,
                    "case {case}: field {f} < bound {lb} at {p:?} (scale {}, margin {})",
                    b.scale,
                    b.margin
                );
                checked += 1;
            }
        }
    }
    assert!(lofts >= 24, "only {lofts} random lofts were valid");
    assert!(
        checked >= lofts * 400,
        "the bound was positive at only {checked} points"
    );
}

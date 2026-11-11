//! The loft's shader emit evaluates the CPU law: the HLSL text (the same emitter as WGSL and GLSL,
//! `src/compiled/loft_shader.rs`) compiled as C++ with `tests/common/hlsl_cpu_shim.h` and run on the
//! CPU, compared point by point with `eval`. No GPU is needed, so this runs in every test job.
//!
//! The lofts and points reach every part of the law that the emitter ports: the near field, the
//! saturated correction with its per-knot width B(s) (lofts whose local bound changes along the
//! axis), the far bound of the plane pieces joined C1 above the near field (points out to several
//! reaches), the fade, caps, and curve and polar sections. A port that keeps an older form of one of
//! them (a single width for B, `max` instead of the C1 join) differs from the CPU there.
//!
//! Tolerance: `|shader − cpu| ≤ TOL · max(1, |cpu|)` (the fields stay within a few units near the
//! surface; far away the value grows with the distance and the bound is relative).
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "hlsl")]
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm builds the section points"
)]

mod common;

use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const TOL: f32 = 2e-5;
const SHIM: &str = include_str!("common/hlsl_cpu_shim.h");

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// A torso-like curve loft: a wide middle that shrinks fast at the top, so the local bound jumps
/// between knots and the saturation width varies per knot
fn torso() -> SdfNode {
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> {
        (0..10)
            .map(|k| {
                let t = std::f32::consts::TAU * k as f32 / 10.0;
                Vec2::new(a * t.cos(), b * t.sin())
            })
            .collect()
    };
    let axis = (0..6)
        .map(|i| Vec3::new(0.02 * i as f32, 0.2 * i as f32, 0.0))
        .collect();
    let rows = [
        (0.3, 0.2),
        (0.34, 0.22),
        (0.36, 0.24),
        (0.3, 0.2),
        (0.12, 0.1),
        (0.08, 0.07),
    ]
    .iter()
    .map(|&(a, b)| ellipse(a, b))
    .collect();
    SdfNode::loft_curves(axis, Vec3::Z, rows).expect("torso-like loft")
}

/// A curve loft closed by caps at both ends, one with an offset apex (the sheared dome)
fn curve_capped() -> SdfNode {
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> {
        (0..12)
            .map(|k| {
                let t = std::f32::consts::TAU * k as f32 / 12.0;
                Vec2::new(a * t.cos(), b * t.sin())
            })
            .collect()
    };
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.03, 0.3, 0.0),
        Vec3::new(0.05, 0.6, 0.02),
    ];
    let rows = vec![ellipse(0.2, 0.12), ellipse(0.18, 0.13), ellipse(0.15, 0.1)];
    SdfNode::loft_curves_capped(axis, Vec3::X, rows, [[0.1, 0.0, 0.0], [0.15, 0.06, -0.03]])
        .expect("capped curve loft")
}

/// A polar loft with domes at both ends
fn polar_capped() -> SdfNode {
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.02, 0.3, 0.0),
        Vec3::new(0.0, 0.6, 0.03),
    ];
    let rows = vec![
        vec![0.15, 0.01, 0.0],
        vec![0.17, 0.0, 0.01],
        vec![0.12, 0.0, 0.0],
    ];
    SdfNode::loft_capped(axis, Vec3::Z, 1, rows, [0.08, 0.12]).expect("capped polar loft")
}

fn lofts() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("curves_c", common::corpus::loft_curves_c()),
        ("fade_straight", common::corpus::loft_fade(false)),
        ("fade_curved", common::corpus::loft_fade(true)),
        ("faded", common::corpus::loft_faded()),
        ("torso", torso()),
        (
            "torso_moved",
            torso()
                .rotate_euler(0.3, -0.2, 0.5)
                .translate(0.1, -0.3, 0.2),
        ),
        ("curve_caps", curve_capped()),
        ("polar_caps", polar_capped()),
    ]
}

/// Points near the surface (the near field and the correction) and out to several reaches (the far
/// bound and its C1 join)
fn points(node: &SdfNode, n: usize, seed: u64) -> Vec<Vec3> {
    let mut r = lcg(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let scale = match out.len() % 4 {
            0 | 1 => 1.0,
            2 => 3.0,
            _ => 8.0,
        };
        let p = Vec3::new(r() - 0.5, r() * 1.4 - 0.3, r() - 0.5) * 2.0 * scale;
        let keep = scale > 1.0 || eval(node, p).abs() < 0.15 || out.len() % 8 == 0;
        if keep {
            out.push(p);
        }
    }
    out
}

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
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_ne_bytes(*c))
        .collect()
}

#[test]
fn the_loft_hlsl_text_evaluates_the_cpu_law() {
    let Some(cxx) = cxx_or_skip() else { return };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("loft_shader_cpu_agreement");
    std::fs::create_dir_all(&dir).expect("scratch");
    let mut compared = 0usize;
    for (what, node) in lofts() {
        let emit = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded).source;
        let src = format!("{SHIM}\n{emit}\n{HOST_MAIN}");
        let pts = points(&node, 20_000, 0x10F7 ^ what.len() as u64);
        let got = compile_and_run(&cxx, &dir, what, &src, &pts);
        assert_eq!(got.len(), pts.len());
        let (mut worst, mut at) = (0.0f32, Vec3::ZERO);
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            assert!(
                g.is_finite() && c.is_finite(),
                "{what}: non-finite at {p:?}: cpu {c}, shader {g}"
            );
            let e = (c - g).abs() / c.abs().max(1.0);
            if e > worst {
                (worst, at) = (e, *p);
            }
            compared += 1;
        }
        assert!(
            worst <= TOL,
            "{what}: the shader differs from the CPU by {worst} (relative) at {at:?}"
        );
    }
    assert!(compared > 0, "no point compared");
}

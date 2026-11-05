//! Every array read the loft, smooth stack and section stack shaders make stays inside the array:
//! the clamps of the node data buffer (WGSL) are GPU safety only and must never change a value.
//!
//! The emitters write the same index expressions in every dialect (the WGSL reads are
//! `base + (index)` over the node's range, the HLSL reads `index` into the node's own arrays), and
//! every read carries the index range the emit proved for it. `HlslShader::try_transpile_checked`
//! writes each read as `ALICE_RD(array, index, lo, hi)`; compiled as C++
//! (`tests/common/hlsl_cpu_shim.h`) with a counting macro, the shader is run on dense points around
//! each scene and the reads outside their own proven range must be 0, while the count of reads is
//! not (a check that reads nothing proves nothing). An index that leaves its range is counted
//! whether or not it stays inside the array, so a clamp could not hide it.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "hlsl")]

use alice_sdf::compiled::{HlslShader, HlslTranspileMode};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

const SHIM: &str = include_str!("common/hlsl_cpu_shim.h");

const COUNTER: &str = r"
static unsigned long alice_reads = 0;
static unsigned long alice_out_of_range = 0;
template <typename T> static T alice_rd(const T *a, long i, long lo, long hi) {
    alice_reads++;
    if (i < lo || i > hi) { alice_out_of_range++; i = i < lo ? lo : hi; }
    return a[i];
}
#define ALICE_RD(a, i, lo, hi) alice_rd(a, (long)(i), (long)(lo), (long)(hi))
";

const HOST_MAIN: &str = r#"
static bool read_exact(void *dst, unsigned long n) {
    char *p = (char *)dst; unsigned long have = 0;
    while (have < n) { long r = read(0, p + have, n - have); if (r <= 0) { return false; } have += (unsigned long)r; }
    return true;
}
int main() {
    ALICE_SDF_SET_BINARY_IO();
    unsigned int count = 0;
    if (!read_exact(&count, 4)) { return 2; }
    for (unsigned int i = 0; i < count; ++i) {
        float buf[3];
        if (!read_exact(buf, 12)) { return 3; }
        volatile float d = sdf_eval(float3(buf[0], buf[1], buf[2]));
        (void)d;
    }
    char line[64];
    int len = snprintf(line, sizeof line, "%lu %lu\n", alice_reads, alice_out_of_range);
    if (write(1, line, (unsigned long)len) != len) { return 4; }
    return 0;
}
"#;

fn cxx() -> Option<String> {
    for cand in ["c++", "clang++", "g++"] {
        if Command::new(cand).arg("--version").stdout(Stdio::null()).stderr(Stdio::null()).status().is_ok_and(|s| s.success()) {
            return Some(cand.to_string());
        }
    }
    assert!(std::env::var_os("ALICE_SDF_REQUIRE_CXX").is_none(), "ALICE_SDF_REQUIRE_CXX is set but no C++ compiler was found");
    eprintln!("skipping: no C++ compiler");
    None
}

/// (reads, out-of-range reads) of the scene's HLSL over the points
fn run(cxx: &str, dir: &Path, name: &str, node: &SdfNode, pts: &[Vec3]) -> (u64, u64) {
    let counted = HlslShader::try_transpile_checked(node, HlslTranspileMode::Hardcoded).expect("the emit proves its reads").source;
    assert!(counted.contains("ALICE_RD("), "{name}: the emit has no checked read");
    let cpp = dir.join(format!("{name}.cpp"));
    let bin = dir.join(name);
    std::fs::write(&cpp, format!("{SHIM}\n{COUNTER}\n{counted}\n{HOST_MAIN}")).expect("write");
    let out = Command::new(cxx)
        .args(["-std=c++17", "-O1", "-ffp-contract=off", "-fno-strict-aliasing", "-w"])
        .arg(&cpp)
        .arg("-o")
        .arg(&bin)
        .output()
        .expect("compiler");
    assert!(out.status.success(), "{name}: C++ compile failed:\n{}", String::from_utf8_lossy(&out.stderr).lines().take(15).collect::<Vec<_>>().join("\n"));
    let mut payload = (pts.len() as u32).to_ne_bytes().to_vec();
    for p in pts {
        for c in [p.x, p.y, p.z] {
            payload.extend_from_slice(&c.to_ne_bytes());
        }
    }
    let mut child = Command::new(&bin).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let writer = std::thread::spawn(move || stdin.write_all(&payload));
    let mut text = String::new();
    child.stdout.take().expect("stdout").read_to_string(&mut text).expect("read");
    writer.join().expect("writer").expect("write");
    assert!(child.wait().expect("wait").success());
    let v: Vec<u64> = text.split_whitespace().map(|x| x.parse().expect("a count")).collect();
    (v[0], v[1])
}

fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, co) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            c + Vec2::new(co, s) * r
        })
        .collect()
}

fn scenes() -> Vec<(&'static str, SdfNode)> {
    let ellipse = |a: f32, b: f32| -> Vec<Vec2> { circle(1.0, Vec2::ZERO, 12).iter().map(|p| Vec2::new(a * p.x, b * p.y)).collect() };
    let levels: Vec<f32> = (0..8).map(|i| -0.35 + 0.1 * i as f32).collect();
    let smooth = SdfNode::smooth_section_stack(
        levels.clone(),
        levels.iter().map(|&y| if y < 0.0 { vec![circle(0.3, Vec2::ZERO, 12)] } else { vec![circle(0.16, Vec2::new(-0.2, 0.0), 8), circle(0.16, Vec2::new(0.2, 0.0), 8)] }).collect(),
        0.01,
        0.1,
    )
    .expect("smooth stack");
    let stack = SdfNode::section_stack(
        vec![0.0, 0.2, 0.4, 0.6],
        [0.0f32, 0.2, 0.4, 0.6].iter().map(|&y| vec![circle(0.2 + 0.1 * y, Vec2::ZERO, 24), circle(0.05, Vec2::new(0.05, 0.0), 8)]).collect(),
    )
    .expect("section stack");
    let curves = SdfNode::loft_curves_capped(
        vec![Vec3::ZERO, Vec3::new(0.03, 0.3, 0.0), Vec3::new(0.05, 0.6, 0.02), Vec3::new(0.04, 0.9, 0.0)],
        Vec3::X,
        vec![ellipse(0.2, 0.12), ellipse(0.18, 0.13), ellipse(0.15, 0.1), ellipse(0.1, 0.08)],
        [[0.1, 0.0, 0.0], [0.15, 0.06, -0.03]],
    )
    .expect("capped curve loft");
    let polar = SdfNode::loft_capped(
        vec![Vec3::ZERO, Vec3::new(0.02, 0.3, 0.0), Vec3::new(0.0, 0.6, 0.03)],
        Vec3::Z,
        2,
        vec![vec![0.15, 0.01, 0.0, 0.01, 0.0], vec![0.17, 0.0, 0.01, 0.0, 0.0], vec![0.12, 0.0, 0.0, 0.0, 0.01]],
        [0.08, 0.12],
    )
    .expect("capped polar loft");
    vec![("smooth_stack", smooth), ("section_stack", stack), ("curve_loft", curves), ("polar_loft", polar)]
}

/// Points out to beyond every end and far field: below and above the levels and the axis ends, on
/// and around the surface, and far away
fn points(n: usize) -> Vec<Vec3> {
    let mut s: u64 = 0x2EAD_5EE5;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..n)
        .map(|i| {
            let k = [0.6f32, 1.5, 6.0][i % 3];
            Vec3::new((r() - 0.5) * 2.0 * k, (r() - 0.3) * 2.0 * k, (r() - 0.5) * 2.0 * k)
        })
        .collect()
}

#[test]
fn every_array_read_is_inside_its_array() {
    let Some(cxx) = cxx() else { return };
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("shader_read_ranges");
    std::fs::create_dir_all(&dir).expect("scratch");
    let pts = points(30_000);
    for (name, node) in scenes() {
        let (reads, bad) = run(&cxx, &dir, name, &node, &pts);
        assert!(reads > pts.len() as u64, "{name}: only {reads} array reads over {} points", pts.len());
        assert_eq!(bad, 0, "{name}: {bad} of {reads} array reads outside the range the emit proved for them");
    }
}

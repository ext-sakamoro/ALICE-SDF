//! Oracles for the HLSL / BlinkScript outputs other than the bare `sdf_eval`
//! body (whose values `tests/test_hlsl_blinkscript_parity.rs` checks):
//!
//! | output | oracle |
//! |---|---|
//! | `HlslShader::export_ue5_material_function` | compiled with a C++ compiler against `tests/common/hlsl_cpu_shim.h` and run: `AliceSdf_Eval` is the closed-form distance, `AliceSdf_Normal` the closed-form normal |
//! | `HlslShader::extract_params` | the literals of the tree, bit-exact; A's Dynamic material function fed B's parameters evaluates B |
//! | `BlinkScriptShader::extract_params` | the HLSL layout (the body is the HLSL body); A's Dynamic body fed B's parameters evaluates B |
//! | `BlinkScriptShader::get_eval_function` | the body itself, which compiles and evaluates the closed form |
//!
//! The independence argument is the one of `test_hlsl_blinkscript_parity.rs`:
//! the parse is clang / gcc, every intrinsic of the shim is written from the
//! HLSL reference, and every expected value here is a closed form computed in
//! f64 — no crate evaluator is called.
//!
//! Dynamic mode declares `cbuffer SdfParams : register(b1)`, which is HLSL
//! only; the harness replaces that one declaration by a `static const float4
//! params[]` with the values under test (the addressing `params[i / 4].c` is
//! what is checked).
//!
//! # Tolerances
//!
//! * distance `DIST_TOL = 1e-5` absolute: `|p − c| < 4` so one f32 ulp is
//!   ≤ 4.8e-7; C++ promotes unsuffixed literals to double where HLSL keeps
//!   float, which moves a result by a fraction of an ulp.
//! * normal `NORMAL_TOL = 1e-3` per component: `AliceSdf_Normal` is a central
//!   difference with `e = 0.001`, so each component is a difference of two
//!   f32 distances ~2e-3 apart, each rounded to ≤ 2.4e-7 (d < 4): a relative
//!   error of ≤ 2.4e-4 before normalisation. The truncation error of the
//!   central difference on a sphere is O(e² / |p − c|²) ≤ 4e-6.
//!
//! `ALICE_SDF_REQUIRE_CXX=1` turns "no C++ compiler" into a failure (set in
//! CI's HLSL step).
//!
//! Author: Moroya Sakamoto

#![cfg(all(feature = "hlsl", feature = "blinkscript"))]

use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
use alice_sdf::compiled::{BlinkScriptShader, BlinkScriptTranspileMode};
use alice_sdf::prelude::*;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SHIM: &str = include_str!("common/hlsl_cpu_shim.h");
const DIST_TOL: f64 = 1e-5;
const NORMAL_TOL: f64 = 1e-3;

#[derive(Clone, Copy)]
struct Ball {
    r: f32,
    c: [f32; 3],
}

impl Ball {
    fn node(self) -> SdfNode {
        SdfNode::sphere(self.r).translate(self.c[0], self.c[1], self.c[2])
    }
    fn q(self, p: Vec3) -> [f64; 3] {
        [
            f64::from(p.x) - f64::from(self.c[0]),
            f64::from(p.y) - f64::from(self.c[1]),
            f64::from(p.z) - f64::from(self.c[2]),
        ]
    }
    fn dist(self, p: Vec3) -> f64 {
        let q = self.q(p);
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - f64::from(self.r)
    }
    fn normal(self, p: Vec3) -> [f64; 3] {
        let q = self.q(p);
        let l = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        [q[0] / l, q[1] / l, q[2] / l]
    }
}

const A: Ball = Ball {
    r: 1.0,
    c: [0.25, -0.5, 0.125],
};
const B: Ball = Ball {
    r: 1.75,
    c: [-0.5, 0.25, 0.0],
};

/// Points with `0.5 ≤ |p − c| ≤ 3` for both balls.
fn points(n: usize) -> Vec<Vec3> {
    let mut s: u32 = 0x5eed_0f0f;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (s >> 8) as f32 / (1u32 << 24) as f32 * 4.4 - 2.2
    };
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let p = Vec3::new(next(), next(), next());
        let ok = |b: Ball| (0.5..=3.0).contains(&(b.dist(p) + f64::from(b.r)));
        if ok(A) && ok(B) {
            out.push(p);
        }
    }
    out
}

fn cxx_or_skip() -> Option<String> {
    let candidates: Vec<String> = std::env::var("ALICE_SDF_CXX")
        .map(|c| vec![c])
        .unwrap_or_else(|_| vec!["c++".into(), "clang++".into(), "g++".into()]);
    for cand in &candidates {
        let ok = Command::new(cand)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        if ok {
            return Some(cand.clone());
        }
    }
    assert!(
        std::env::var_os("ALICE_SDF_REQUIRE_CXX").is_none(),
        "ALICE_SDF_REQUIRE_CXX is set but no C++ compiler was found (tried {})",
        candidates.join(", ")
    );
    eprintln!("skipping: no C++ compiler");
    None
}

fn scratch() -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("hlsl_export_oracle");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// Host: reads `count` then `count` points from stdin, writes `OUT` floats
/// per point produced by `host_eval(float3 p, float *out)`.
fn host(out_per_point: usize, body: &str) -> String {
    format!(
        r"
static bool read_exact(void *dst, unsigned long n) {{
    char *p = (char *)dst; unsigned long have = 0;
    while (have < n) {{ long r = read(0, p + have, n - have); if (r <= 0) return false; have += (unsigned long)r; }}
    return true;
}}
static bool write_exact(const void *src, unsigned long n) {{
    const char *p = (const char *)src; unsigned long sent = 0;
    while (sent < n) {{ long r = write(1, p + sent, n - sent); if (r <= 0) return false; sent += (unsigned long)r; }}
    return true;
}}
static void host_eval(float3 p, float *out) {{ {body} }}
int main() {{
    ALICE_SDF_SET_BINARY_IO();
    unsigned int count = 0;
    if (!read_exact(&count, 4)) return 2;
    for (unsigned int i = 0; i < count; ++i) {{
        float buf[3]; float out[{out_per_point}];
        if (!read_exact(buf, 12)) return 3;
        host_eval(float3(buf[0], buf[1], buf[2]), out);
        if (!write_exact(out, sizeof(out))) return 4;
    }}
    return 0;
}}
"
    )
}

/// Compile `src` and run it over `pts`; `per` floats come back per point.
fn compile_and_run(cxx: &str, name: &str, src: &str, pts: &[Vec3], per: usize) -> Vec<f32> {
    let dir = scratch();
    let cpp = dir.join(format!("{name}.cpp"));
    let bin = dir.join(name);
    std::fs::write(&cpp, src).unwrap();
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
        .unwrap_or_else(|e| panic!("cannot run {cxx}: {e}"));
    assert!(
        out.status.success(),
        "{name}: C++ compile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
            .lines()
            .take(12)
            .collect::<Vec<_>>()
            .join("\n")
    );
    let mut payload = (pts.len() as u32).to_ne_bytes().to_vec();
    for p in pts {
        for v in [p.x, p.y, p.z] {
            payload.extend_from_slice(&v.to_ne_bytes());
        }
    }
    let mut child = Command::new(&bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&payload));
    let mut raw = Vec::new();
    child.stdout.take().unwrap().read_to_end(&mut raw).unwrap();
    writer.join().unwrap().unwrap();
    assert!(child.wait().unwrap().success(), "{name}: evaluator failed");
    assert_eq!(raw.len(), pts.len() * per * 4, "{name}: output size");
    raw.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `layout` as the `static const float4 params[]` the Dynamic emit reads.
fn params_global(layout: &[f32]) -> String {
    let mut padded = layout.to_vec();
    while padded.len() % 4 != 0 {
        padded.push(0.0);
    }
    let v: Vec<String> = padded
        .chunks_exact(4)
        .map(|c| format!("float4({:e}f, {:e}f, {:e}f, {:e}f)", c[0], c[1], c[2], c[3]))
        .collect();
    format!(
        "static const float4 params[{}] = {{ {} }};\n",
        v.len(),
        v.join(", ")
    )
}

/// Replace the Dynamic `cbuffer SdfParams { ... };` by `params_global(layout)`.
fn with_params(src: &str, layout: &[f32]) -> String {
    let start = src
        .find("cbuffer SdfParams")
        .expect("cbuffer in a Dynamic emit");
    let end = start + src[start..].find("};").unwrap() + 2;
    format!("{}{}{}", &src[..start], params_global(layout), &src[end..])
}

fn check_dist(what: &str, ball: Ball, pts: &[Vec3], got: &[f32]) {
    assert_eq!(got.len(), pts.len());
    assert!(!pts.is_empty(), "{what}: 0 points compared");
    for (&p, &g) in pts.iter().zip(got) {
        let err = (f64::from(g) - ball.dist(p)).abs();
        assert!(
            err <= DIST_TOL,
            "{what}: at {p:?} got {g}, closed form {} (err {err:e})",
            ball.dist(p)
        );
    }
}

fn literal_multiset(layout: &[f32], ball: Ball) {
    let mut got = layout.to_vec();
    let mut want = vec![ball.r, ball.c[0], ball.c[1], ball.c[2]];
    got.sort_by(f32::total_cmp);
    want.sort_by(f32::total_cmp);
    assert_eq!(got, want, "the four literals of the tree, bit-exact");
}

#[test]
fn ue5_material_function_evaluates_distance_and_normal() {
    let Some(cxx) = cxx_or_skip() else { return };
    let mf = HlslShader::transpile(&A.node(), HlslTranspileMode::Hardcoded)
        .export_ue5_material_function();
    assert!(mf.contains("float AliceSdf_Eval(float3 WorldPosition)"));
    assert!(mf.contains("float3 AliceSdf_Normal(float3 p)"));
    let pts = points(800);
    let src = format!(
        "{SHIM}\n{mf}\n{}",
        host(
            4,
            "out[0] = AliceSdf_Eval(p); float3 n = AliceSdf_Normal(p); out[1] = n.x; out[2] = n.y; out[3] = n.z;"
        )
    );
    let got = compile_and_run(&cxx, "ue5_mf", &src, &pts, 4);
    let d: Vec<f32> = got.chunks_exact(4).map(|c| c[0]).collect();
    check_dist("AliceSdf_Eval", A, &pts, &d);
    for (&p, c) in pts.iter().zip(got.chunks_exact(4)) {
        let want = A.normal(p);
        for k in 0..3 {
            let err = (f64::from(c[k + 1]) - want[k]).abs();
            assert!(
                err <= NORMAL_TOL,
                "AliceSdf_Normal at {p:?}: {:?} vs {want:?}",
                &c[1..]
            );
        }
    }
}

#[test]
fn hlsl_extract_params_drives_the_dynamic_material_function() {
    let la = HlslShader::extract_params(&A.node());
    let lb = HlslShader::extract_params(&B.node());
    let dyn_a = HlslShader::transpile(&A.node(), HlslTranspileMode::Dynamic);
    assert_eq!(la, dyn_a.param_layout);
    literal_multiset(&la, A);
    literal_multiset(&lb, B);
    let Some(cxx) = cxx_or_skip() else { return };
    let mf = dyn_a.export_ue5_material_function();
    assert!(mf.contains("cbuffer SdfParams : register(b1)"));
    let pts = points(800);
    for (tag, layout, ball) in [("a", &la, A), ("b", &lb, B)] {
        let src = format!(
            "{SHIM}\n{}\n{}",
            with_params(&mf, layout),
            host(1, "out[0] = AliceSdf_Eval(p);")
        );
        let d = compile_and_run(&cxx, &format!("ue5_mf_dyn_{tag}"), &src, &pts, 1);
        check_dist(&format!("Dynamic A fed {tag}'s params"), ball, &pts, &d);
    }
}

#[test]
fn blinkscript_params_and_body() {
    let la = BlinkScriptShader::extract_params(&A.node());
    let lb = BlinkScriptShader::extract_params(&B.node());
    assert_eq!(la, HlslShader::extract_params(&A.node()));
    literal_multiset(&la, A);
    literal_multiset(&lb, B);
    let hard = BlinkScriptShader::transpile(&A.node(), BlinkScriptTranspileMode::Hardcoded);
    assert_eq!(hard.get_eval_function(), hard.source);
    let dyn_a = BlinkScriptShader::transpile(&A.node(), BlinkScriptTranspileMode::Dynamic);
    assert_eq!(dyn_a.param_layout, la);
    let Some(cxx) = cxx_or_skip() else { return };
    let pts = points(800);
    let src = format!(
        "{SHIM}\n{}\n{}",
        hard.get_eval_function(),
        host(1, "out[0] = sdf_eval(p);")
    );
    check_dist(
        "BlinkScript body",
        A,
        &pts,
        &compile_and_run(&cxx, "blink_hard", &src, &pts, 1),
    );
    // The Dynamic body declares no buffer of its own (the kernel's params
    // panel provides it): prepend B's layout.
    let src = format!(
        "{SHIM}\n{}{}\n{}",
        params_global(&lb),
        dyn_a.get_eval_function(),
        host(1, "out[0] = sdf_eval(p);")
    );
    check_dist(
        "BlinkScript Dynamic A fed B's params",
        B,
        &pts,
        &compile_and_run(&cxx, "blink_dyn", &src, &pts, 1),
    );
}

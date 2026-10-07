//! HLSL / BlinkScript **value** oracle: every corpus node's emitted source is
//! compiled by a C++ compiler and executed, then compared with the CPU law.
//!
//! # Why this exists
//!
//! WGSL is validated by `naga` and run on the GPU (`test_gpu_law_parity.rs`),
//! MSL is compiled and run on Metal (`test_msl_metal_oracle.rs`), GLSL is at
//! least parsed by naga's `glsl-in` (`test_transpiler_naga_validate.rs`).
//! HLSL had **no parser at all** in the test tree — `naga` has no HLSL
//! front-end — so its 1,888-line transpiler was covered only by assertions on
//! the emitted *text* (`tests/test_round_tie_parity.rs`
//! `hlsl_text_uses_floor_plus_half`), and BlinkScript appeared in neither
//! `tests/` nor `ci.yml`. Not being parseable by naga is not a reason for the
//! *values* to go unchecked.
//!
//! # The route, and why it is independent of the CPU implementation
//!
//! `dxc` is not installed and would not help on a Mac anyway (DXIL needs
//! Windows/WARP, and the SPIR-V path would still need a Vulkan driver). What
//! *is* universally available — on this host and on every CI runner — is a
//! C++ compiler, and the emit is a C-like subset. So:
//!
//! ```text
//! HlslShader::transpile(node)           emitted `float sdf_eval(float3 p)`
//!   -> + tests/common/hlsl_cpu_shim.h   HLSL intrinsics, written from the
//!                                       HLSL reference (NOT from the Rust
//!                                       evaluator)
//!   -> clang++ / g++                    an independent parser
//!   -> native binary, run on 2048 pts
//!   -> compare against `eval(&node, p)` the CPU law
//! ```
//!
//! Independence rests on two things: the **parse** is done by a third-party
//! compiler, and every intrinsic in the shim is defined from its HLSL
//! specification rather than by reading `src/compiled/eval_core.rs`. A
//! transpiler that emits `max` for `min`, swaps operands, drops a factor or
//! writes the wrong constant cannot agree with the shim by construction. What
//! this *cannot* catch is a law that is wrong identically in
//! `transpiler_common` and in the CPU evaluator — the same blind spot every
//! parity oracle has, and the reason the analytic oracles
//! (`test_npr_analytic.rs`, `test_validity_oracle.rs`, ...) exist alongside.
//!
//! A second, fully independent gate parses the emit with **glslang's HLSL
//! front-end** and validates the resulting SPIR-V with `spirv-val`
//! (`hlsl_compute_shader_parses_with_glslang`), which is the closest analogue
//! to what naga does for WGSL and GLSL.
//!
//! # Anti-vacuity
//!
//! `oracle_flags_a_perturbed_emit_for_every_node` re-runs the whole parity
//! sweep against a deliberately mutated emit and requires **every** node to be
//! reported. A skipped node, a dropped comparison or an unread output stream
//! therefore fails the suite instead of passing quietly — the failure mode this
//! crate has hit repeatedly (a gate that runs and measures nothing).
//! `hlsl_shim_matches_hand_computed_intrinsics` pins the shim itself against
//! hand-computed values, so the oracle's own reference cannot drift.
//!
//! # Tolerance
//!
//! `REL_TOL` and the point set are byte-for-byte those of
//! `test_msl_metal_oracle.rs`, so a drift here is directly comparable with the
//! Metal and Vulkan numbers. Note that C++ promotes unsuffixed literals to
//! `double` where HLSL keeps them `float`, so a sub-ulp difference is expected
//! by construction; `-ffp-contract=off` keeps FMA from adding more.
//!
//! # Skips
//!
//! `ALICE_SDF_REQUIRE_CXX=1` turns "no C++ compiler" into a failure and
//! `ALICE_SDF_REQUIRE_GLSLANG=1` does the same for `glslangValidator` /
//! `spirv-val`, mirroring the `ALICE_SDF_REQUIRE_GPU` / `_METAL` contract
//! (the port parity rule). Without a CI job that sets
//! them, a feature-gated oracle is indistinguishable from an absent one.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "hlsl")]

mod common;

use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
use alice_sdf::prelude::*;
use common::corpus::corpus;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Relative tolerance, as in the Metal and Vulkan oracles, so that a drift
/// reported here is directly comparable with the Metal and Vulkan numbers.
const REL_TOL: f32 = 1e-4;

/// The bar every node that does **not** amplify rounding is actually held to.
///
/// Measured 2026-09-30 over the full corpus at 2048 points: 45 of 152 nodes are
/// **bit-exact** against the CPU law, 82.3% of all (node, point) pairs are
/// bit-exact, and the worst node other than `lattice_deform` sits at 6.56e-7
/// (~5 f32 ulps). `1e-5` therefore leaves ~15x headroom for a different libm
/// (`sin` / `cos` / `pow` / `atan2` differ by a few ulps between Apple libm and
/// glibc) while staying 10x tighter than `REL_TOL` — so the single exemption
/// below cannot quietly become a general slackening.
const TIGHT_TOL: f32 = 1e-5;

/// Per-node relative tolerance.
///
/// `lattice_deform` is the one node that exceeds `REL_TOL`, at 1.276e-4, and the
/// cause is in the law rather than in the emit:
///
/// ```text
/// d2 = max(length(ffd(p + (0.001,0,0)) - ffd(p - (0.001,0,0))) / 0.002, 0.1)
/// ```
///
/// is a central finite difference whose subtraction cancels ~4 significant
/// digits and is then divided by `2e-3`, so f32 rounding in `alice_ffd_0` is
/// amplified ~500x before it reaches the returned distance through `d4 / d2`.
///
/// Three measurements say this is rounding and not a wrong emit:
///
/// 1. Of 2048 points, the **1970 outside** the lattice box — where the `if` is
///    not taken and no differencing happens — are **bit-exact**, drift exactly
///    `0`. The emitted box algebra is therefore byte-faithful.
/// 2. The 78 points inside the box are the only ones that move, and `d1 =
///    alice_ffd_0(p)` also feeds the distance *directly*; an algebraic error in
///    the deformation would show up there at O(1), not at 1e-4.
/// 3. `tests/test_msl_metal_oracle.rs` reports **the same node as its worst**,
///    at 8.744e-5. Two independent implementations of the same emit (Apple
///    Metal on the GPU, clang on the CPU) disagreeing with the CPU law by
///    *different* amounts is rounding; a genuine law mismatch would put both at
///    the same value.
///
/// The root fix is a larger `eps` or an analytic Jacobian in
/// `src/compiled/transpiler_common.rs`, which is a law change with its own
/// `CHANGELOG` consequences and is deliberately not made here.
fn tolerance(name: &str) -> f32 {
    match name {
        "lattice_deform" => 2e-4,
        _ => REL_TOL,
    }
}

/// Number of sample points per node (same count as the Metal oracle).
const POINTS: usize = 2048;

/// The HLSL intrinsic shim, written from the HLSL reference.
const SHIM: &str = include_str!("common/hlsl_cpu_shim.h");

/// Streams `float[3]` triples in on stdin and `float` distances out on stdout.
/// Partial pipe reads and writes are handled explicitly; a short read is an
/// error rather than a silently truncated result set.
const HOST_MAIN: &str = r"
static bool read_exact(void *dst, unsigned long n) {
    char *p = (char *)dst;
    unsigned long have = 0;
    while (have < n) {
        long r = read(0, p + have, n - have);
        if (r <= 0) { return false; }
        have += (unsigned long)r;
    }
    return true;
}

static bool write_exact(const void *src, unsigned long n) {
    const char *p = (const char *)src;
    unsigned long sent = 0;
    while (sent < n) {
        long r = write(1, p + sent, n - sent);
        if (r <= 0) { return false; }
        sent += (unsigned long)r;
    }
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

/// Deterministic LCG points in a ±3 box — byte-for-byte the generator in
/// `tests/test_gpu_law_parity.rs` and `tests/test_msl_metal_oracle.rs`, so all
/// three oracles sample the same laws.
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

/// A working C++ compiler, or `None` when the host has none (skip unless the
/// caller demanded one). Override the command with `ALICE_SDF_CXX`.
fn cxx_or_skip() -> Option<String> {
    // An explicit `ALICE_SDF_CXX` is honoured alone: falling back to another
    // compiler would silently measure something the caller did not ask for.
    let candidates: Vec<String> = std::env::var("ALICE_SDF_CXX")
        .map(|c| vec![c])
        .unwrap_or_else(|_| vec!["c++".to_string(), "clang++".to_string(), "g++".to_string()]);
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
    eprintln!("skipping HLSL CPU oracle: no C++ compiler");
    None
}

/// Scratch directory for generated sources and binaries.
fn scratch(tag: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(tag);
    std::fs::create_dir_all(&dir).expect("cannot create scratch dir");
    dir
}

/// Assemble a translation unit: shim, optional globals, the emit, the host.
fn translation_unit(emit: &str, globals: &str) -> String {
    format!("{SHIM}\n{globals}\n{emit}\n{HOST_MAIN}")
}

/// Compile `src` and return the executable's path.
fn compile(cxx: &str, dir: &Path, name: &str, src: &str) -> Result<PathBuf, String> {
    let cpp = dir.join(format!("{name}.cpp"));
    let bin = dir.join(name);
    std::fs::write(&cpp, src).map_err(|e| format!("write {}: {e}", cpp.display()))?;
    let out = Command::new(cxx)
        .args([
            "-std=c++17",
            "-O1",
            // The emit must round exactly where HLSL rounds: no FMA fusion.
            "-ffp-contract=off",
            // The swizzle members alias the vector's storage, as in HLSL.
            "-fno-strict-aliasing",
            // Anonymous structs in unions are a compiler extension both clang
            // and gcc accept; this harness judges by values, not diagnostics.
            "-w",
        ])
        .arg(&cpp)
        .arg("-o")
        .arg(&bin)
        .output()
        .map_err(|e| format!("cannot run {cxx}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "C++ compile failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
                .lines()
                .take(12)
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    Ok(bin)
}

/// Run a compiled evaluator over `pts`.
fn run(bin: &Path, pts: &[Vec3]) -> Result<Vec<f32>, String> {
    let mut payload = Vec::with_capacity(4 + pts.len() * 12);
    payload.extend_from_slice(&(pts.len() as u32).to_ne_bytes());
    for p in pts {
        payload.extend_from_slice(&p.x.to_ne_bytes());
        payload.extend_from_slice(&p.y.to_ne_bytes());
        payload.extend_from_slice(&p.z.to_ne_bytes());
    }
    let mut child = Command::new(bin)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot spawn {}: {e}", bin.display()))?;
    let mut stdin = child.stdin.take().expect("piped stdin");
    // A dedicated writer avoids the classic both-pipes-full deadlock.
    let writer = std::thread::spawn(move || stdin.write_all(&payload));
    let mut raw = Vec::new();
    child
        .stdout
        .take()
        .expect("piped stdout")
        .read_to_end(&mut raw)
        .map_err(|e| format!("read stdout: {e}"))?;
    writer
        .join()
        .map_err(|_| "stdin writer panicked".to_string())?
        .map_err(|e| format!("write stdin: {e}"))?;
    let status = child.wait().map_err(|e| format!("wait: {e}"))?;
    if !status.success() {
        return Err(format!("evaluator exited with {status}"));
    }
    if raw.len() != pts.len() * 4 {
        return Err(format!(
            "expected {} bytes of distances, got {}",
            pts.len() * 4,
            raw.len()
        ));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `param_layout` as an HLSL `float4 params[]` global, matching the
/// `params[idx / 4].{x,y,z,w}` addressing `TranspilerCtx::param` emits.
fn params_global(layout: &[f32]) -> String {
    if layout.is_empty() {
        return String::new();
    }
    let mut padded = layout.to_vec();
    while padded.len() % 4 != 0 {
        padded.push(0.0);
    }
    let vecs: Vec<String> = padded
        .chunks_exact(4)
        .map(|c| format!("float4({:e}f, {:e}f, {:e}f, {:e}f)", c[0], c[1], c[2], c[3]))
        .collect();
    format!(
        "static const float4 params[{}] = {{ {} }};\n",
        vecs.len(),
        vecs.join(", ")
    )
}

/// One node's translation unit, ready to compile.
struct Unit {
    name: String,
    src: String,
}

/// Compile every unit, spreading the work over the available cores. ~300 C++
/// translation units per sweep is the wall-clock cost of this oracle, so the
/// compiles run concurrently; results come back in `units` order.
fn compile_all(cxx: &str, dir: &Path, units: &[Unit]) -> Vec<Result<PathBuf, String>> {
    let lanes = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
    let mut lane_results: Vec<Vec<(usize, Result<PathBuf, String>)>> = Vec::new();
    std::thread::scope(|s| {
        // Every lane has to be spawned before the first `join`, or the lanes
        // run one after another and the concurrency is lost.
        let mut handles = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            handles.push(s.spawn(move || {
                units
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % lanes == lane)
                    .map(|(i, u)| (i, compile(cxx, dir, &u.name, &u.src)))
                    .collect::<Vec<_>>()
            }));
        }
        for h in handles {
            lane_results.push(h.join().expect("compile lane panicked"));
        }
    });
    let mut all: Vec<(usize, Result<PathBuf, String>)> =
        lane_results.into_iter().flatten().collect();
    all.sort_by_key(|(i, _)| *i);
    all.into_iter().map(|(_, r)| r).collect()
}

/// Worst relative drift of `got` against the CPU law, with the witness.
fn worst_drift(node: &SdfNode, pts: &[Vec3], got: &[f32]) -> (f32, Vec3, f32, f32) {
    let mut worst = (0.0_f32, Vec3::ZERO, 0.0_f32, 0.0_f32);
    for (p, g) in pts.iter().zip(got) {
        let c = eval(node, *p);
        let diff = (g - c).abs() / c.abs().max(1.0);
        if diff > worst.0 {
            worst = (diff, *p, c, *g);
        }
    }
    worst
}

/// Emit every corpus node in `mode`, apply `mutate` to the **emitted source**
/// and pair it with the globals it needs.
///
/// `mutate` deliberately sees only the emit, never the assembled translation
/// unit: anything it appends has to land *before* the host `main`, or the
/// harness fails to compile instead of measuring.
fn hlsl_units(mode: HlslTranspileMode, emit: &impl Fn(&HlslShader) -> String) -> Vec<Unit> {
    corpus()
        .into_iter()
        .map(|(name, node)| {
            let sh = HlslShader::transpile(&node, mode);
            Unit {
                name: name.to_string(),
                src: translation_unit(&emit(&sh), &params_global(&sh.param_layout)),
            }
        })
        .collect()
}

/// Run a full sweep. Returns `(per-node drift report, worst drift, its node,
/// nodes measured)`. `mutate` rewrites the emitted source before it is compiled.
fn sweep(
    cxx: &str,
    tag: &str,
    mode: HlslTranspileMode,
    mutate: impl Fn(&str) -> String,
) -> (Vec<String>, f32, String, usize) {
    sweep_emit(cxx, tag, mode, |sh| mutate(&sh.source))
}

/// [`sweep`] over an arbitrary rendering of each node's `HlslShader`: `emit`
/// must define `float sdf_eval(float3 p)` for the host to call (the engine
/// wrappers below define it by calling their entry point).
fn sweep_emit(
    cxx: &str,
    tag: &str,
    mode: HlslTranspileMode,
    emit: impl Fn(&HlslShader) -> String,
) -> (Vec<String>, f32, String, usize) {
    let dir = scratch(tag);
    let pts = points(POINTS);
    let units = hlsl_units(mode, &emit);
    let bins = compile_all(cxx, &dir, &units);

    let mut drifted = Vec::new();
    let mut global_worst = (0.0_f32, String::new());
    let mut checked = 0usize;
    for ((name, node), bin) in corpus().into_iter().zip(bins) {
        let bin = match bin {
            Ok(b) => b,
            Err(e) => {
                drifted.push(format!("{name}: {e}"));
                continue;
            }
        };
        let got = match run(&bin, &pts) {
            Ok(v) => v,
            Err(e) => {
                drifted.push(format!("{name}: {e}"));
                continue;
            }
        };
        let w = worst_drift(&node, &pts, &got);
        if w.0 > global_worst.0 {
            global_worst = (w.0, name.to_string());
        }
        // Exempt nodes get their documented bound; everything else is held to
        // `TIGHT_TOL`, 10x below the headline tolerance, so the exemption list
        // cannot spread by accident.
        let bar = if (tolerance(name) - REL_TOL).abs() > f32::EPSILON {
            tolerance(name)
        } else {
            TIGHT_TOL
        };
        if w.0 > bar {
            drifted.push(format!(
                "{name}: HLSL/CPU drift {:.3e} (bar {bar:.1e}) at {:?} (cpu={} hlsl={})",
                w.0, w.1, w.2, w.3
            ));
        }
        checked += 1;
    }
    (drifted, global_worst.0, global_worst.1, checked)
}

// ===========================================================================
// The shim's own oracle
// ===========================================================================

/// Hand-computed HLSL intrinsic results. Every value here is derived from the
/// HLSL specification with pen and paper, never by running ALICE-SDF.
///
/// The `fmod` pair is the load-bearing one: HLSL `fmod` is **truncated**
/// (`-7 - 3 * trunc(-7/3) = -1`) while the floor modulo the CPU law uses and
/// `HlslLang::modulo_expr` emits explicitly gives `+2`. If the shim were to
/// implement floor modulo as `fmod`, a transpiler that regressed to `fmod`
/// would agree with it and the parity sweep would pass on a wrong emit.
#[test]
fn hlsl_shim_matches_hand_computed_intrinsics() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    let probes: &[(&str, &str, f32)] = &[
        // Swizzles are probed the way the emit consumes them — handed to a
        // function that takes the narrower vector — and the basis vectors pin
        // *which lane* landed where, so a transposed alias cannot pass.
        (
            "swizzle_xz_lane0",
            "dot(float3(1.0f, 2.0f, 3.0f).xz, float2(1.0f, 0.0f))",
            1.0,
        ),
        (
            "swizzle_xz_lane1",
            "dot(float3(1.0f, 2.0f, 3.0f).xz, float2(0.0f, 1.0f))",
            3.0,
        ),
        (
            "swizzle_xy_lane1",
            "dot(float3(1.0f, 2.0f, 3.0f).xy, float2(0.0f, 1.0f))",
            2.0,
        ),
        (
            "swizzle_yz_lane0",
            "dot(float3(1.0f, 2.0f, 3.0f).yz, float2(1.0f, 0.0f))",
            2.0,
        ),
        (
            "swizzle_xyz_lane2",
            "dot(float4(1.0f, 2.0f, 3.0f, 4.0f).xyz, float3(0.0f, 0.0f, 1.0f))",
            3.0,
        ),
        (
            "swizzle_xz_of_float4_lane1",
            "dot(float4(1.0f, 2.0f, 3.0f, 4.0f).xz, float2(0.0f, 1.0f))",
            3.0,
        ),
        ("float4_w", "float4(1.0f, 2.0f, 3.0f, 4.0f).w", 4.0),
        ("subscript_1", "float3(7.0f, 8.0f, 9.0f)[1]", 8.0),
        ("fmod_truncated", "fmod(-7.0f, 3.0f)", -1.0),
        (
            "floor_mod_via_emit_form",
            "(-7.0f) - (3.0f) * floor((-7.0f) / (3.0f))",
            2.0,
        ),
        ("sign_neg_zero", "sign(-0.0f)", 0.0),
        ("sign_neg", "sign(-4.0f)", -1.0),
        ("lerp_quarter", "lerp(2.0f, 10.0f, 0.25f)", 4.0),
        ("trunc_neg", "trunc(-2.7f)", -2.0),
        ("floor_neg", "floor(-2.7f)", -3.0),
        ("length3", "length(float3(3.0f, 4.0f, 12.0f))", 13.0),
        ("dot2", "dot(float2(3.0f, 4.0f), float2(5.0f, 6.0f))", 39.0),
        (
            "cross_z",
            "cross(float3(1.0f, 0.0f, 0.0f), float3(0.0f, 1.0f, 0.0f)).z",
            1.0,
        ),
        ("clamp_hi", "clamp(5.0f, 0.0f, 1.0f)", 1.0),
        ("step_eq", "step(0.5f, 0.5f)", 1.0),
        ("normalize2_y", "normalize(float2(0.0f, 5.0f)).y", 1.0),
        ("asint_one", "(float)asint(1.0f)", 1_065_353_216.0),
        ("asuint_neg_two", "(float)asuint(-2.0f)", 3_221_225_472.0),
        (
            "max_vec_scalar",
            "max(float3(1.0f, 5.0f, 2.0f), 3.0f).y",
            5.0,
        ),
        (
            "min_vec_scalar",
            "min(float3(1.0f, 5.0f, 2.0f), 3.0f).z",
            2.0,
        ),
        ("abs_vec", "abs(float2(-3.0f, 4.0f)).x", 3.0),
        ("pow_cube", "pow(2.0f, 3.0f)", 8.0),
        ("acos_one", "acos(1.0f)", 0.0),
        ("atan2_quadrant", "atan2(0.0f, -1.0f)", std::f32::consts::PI),
    ];

    let exprs: Vec<String> = probes.iter().map(|(_, e, _)| format!("    {e},")).collect();
    let src = format!(
        "{SHIM}\nstatic const float kProbes[] = {{\n{}\n}};\n\
         int main() {{\n    ALICE_SDF_SET_BINARY_IO();\n    \
         write(1, kProbes, sizeof(kProbes));\n    return 0;\n}}\n",
        exprs.join("\n")
    );
    let dir = scratch("hlsl_shim_selftest");
    let bin = compile(&cxx, &dir, "shim_selftest", &src).expect("shim self-test must compile");
    let out = Command::new(&bin)
        .output()
        .expect("shim self-test must run");
    assert!(out.status.success(), "shim self-test exited {}", out.status);
    assert_eq!(
        out.stdout.len(),
        probes.len() * 4,
        "shim self-test produced {} bytes for {} probes",
        out.stdout.len(),
        probes.len()
    );

    let mut wrong = Vec::new();
    for ((name, expr, want), chunk) in probes.iter().zip(out.stdout.chunks_exact(4)) {
        let got = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if (got - want).abs() > 1e-6 * want.abs().max(1.0) {
            wrong.push(format!("{name}: `{expr}` = {got}, expected {want}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "the HLSL shim disagrees with {} hand-computed value(s):\n{}",
        wrong.len(),
        wrong.join("\n")
    );
    eprintln!(
        "HLSL shim matches {} hand-computed intrinsic results",
        probes.len()
    );
}

// ===========================================================================
// Value parity
// ===========================================================================

#[test]
fn hlsl_hardcoded_matches_cpu_for_every_corpus_node() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    let (drifted, worst, worst_name, checked) =
        sweep(&cxx, "hlsl_hardcoded", HlslTranspileMode::Hardcoded, |s| {
            s.to_string()
        });
    assert!(
        drifted.is_empty(),
        "{} corpus node(s) drifted between HLSL and CPU:\n{}",
        drifted.len(),
        drifted.join("\n")
    );
    assert_eq!(
        checked,
        corpus().len(),
        "not every corpus node was measured"
    );
    eprintln!(
        "HLSL(Hardcoded)/CPU parity over {checked} corpus nodes x {POINTS} points: \
         worst {worst:.3e} ({worst_name})"
    );
}

/// `Dynamic` mode routes every constant through `params[i].c` instead of
/// baking it in. It is a second emit path with its own indexing arithmetic and
/// was never executed anywhere; a mis-numbered parameter shows up only here.
#[test]
fn hlsl_dynamic_matches_cpu_for_every_corpus_node() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    let (drifted, worst, worst_name, checked) =
        sweep(&cxx, "hlsl_dynamic", HlslTranspileMode::Dynamic, |s| {
            s.to_string()
        });
    assert!(
        drifted.is_empty(),
        "{} corpus node(s) drifted between HLSL(Dynamic) and CPU:\n{}",
        drifted.len(),
        drifted.join("\n")
    );
    assert_eq!(
        checked,
        corpus().len(),
        "not every corpus node was measured"
    );
    eprintln!(
        "HLSL(Dynamic)/CPU parity over {checked} corpus nodes x {POINTS} points: \
         worst {worst:.3e} ({worst_name})"
    );
}

// ===========================================================================
// Engine wrappers (UE5 Custom node, Unity Shader Graph Custom Function)
// ===========================================================================

/// `HlslShader::to_ue5_custom_node` is a function **body**: UE pastes it into
/// a generated function with the input `p`, so the harness does the same.
fn ue5_custom_node_unit(sh: &HlslShader) -> String {
    format!(
        "float sdf_eval(float3 p) {{\n{}\n}}\n",
        sh.to_ue5_custom_node()
    )
}

/// A Unity Custom Function file, made compilable as C++ and wrapped so the
/// host's `sdf_eval` goes through `entry`.
///
/// Two HLSL-only spellings are mapped to their C++ equivalents: `out T x`
/// parameters become `T &x`, and `half` / `half3` are `float` (the shim has no
/// half type). The file sits in a namespace because it defines its own
/// `sdf_eval`. In Dynamic mode its `float4 _SdfParams[1024];` declaration is
/// replaced by the parameter values under test.
fn unity_unit(file: &str, layout: &[f32], entry: &str) -> String {
    let mut file = file
        .replace("out float3 ", "float3 &")
        .replace("out half3 ", "half3 &")
        .replace("out float ", "float &")
        .replace("out half ", "half &");
    if let Some(start) = file.find("float4 _SdfParams[1024];") {
        let decl = params_global(layout).replace("params[", "_SdfParams[");
        file.replace_range(start..start + "float4 _SdfParams[1024];".len(), &decl);
    }
    // The harness also declares the module-scope `params` the bare emit
    // reads. Poison that name so the file can only see its own `_SdfParams`
    // (through its own `#define params _SdfParams`).
    format!(
        "typedef float half;\ntypedef float3 half3;\n#define params alice_sdf_params_outside_the_file\n\
         namespace unity {{\n{file}\n}}\n#undef params\n\
         float sdf_eval(float3 p) {{ {entry} }}\n"
    )
}

fn assert_sweep_clean(
    what: &str,
    (drifted, worst, worst_name, checked): (Vec<String>, f32, String, usize),
) {
    assert!(
        drifted.is_empty(),
        "{what}: {} corpus node(s) drifted from the CPU law:\n{}",
        drifted.len(),
        drifted.join("\n")
    );
    assert_eq!(
        checked,
        corpus().len(),
        "{what}: not every corpus node was measured"
    );
    assert!(checked > 0, "{what}: 0 nodes compared");
    eprintln!("{what}: {checked} corpus nodes x {POINTS} points, worst {worst:.3e} ({worst_name})");
}

/// UE5 Custom node body, Hardcoded and Dynamic, for every corpus node. The
/// corpus includes the nodes with module-scope data arrays (`lattice_deform`,
/// heightmap) that the body moves into member functions.
#[test]
fn ue5_custom_node_matches_cpu_for_every_corpus_node() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    for (tag, mode) in [
        ("ue5_custom_hardcoded", HlslTranspileMode::Hardcoded),
        ("ue5_custom_dynamic", HlslTranspileMode::Dynamic),
    ] {
        assert_sweep_clean(tag, sweep_emit(&cxx, tag, mode, ue5_custom_node_unit));
    }
}

/// `HlslShader::to_unity_custom_function` (Hardcoded) and
/// `HlslShader::export_unity_shader_graph` (Dynamic, `_SdfParams`) for every
/// corpus node, through their Shader Graph entry points.
#[test]
fn unity_custom_functions_match_cpu_for_every_corpus_node() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    assert_sweep_clean(
        "unity_custom_function_hardcoded",
        sweep_emit(
            &cxx,
            "unity_cf_hardcoded",
            HlslTranspileMode::Hardcoded,
            |sh| {
                unity_unit(
                    &sh.to_unity_custom_function(),
                    &sh.param_layout,
                    "float d; unity::SdfEval_float(p, d); return d;",
                )
            },
        ),
    );
    assert_sweep_clean(
        "unity_shader_graph_dynamic",
        sweep_emit(&cxx, "unity_sg_dynamic", HlslTranspileMode::Dynamic, |sh| {
            unity_unit(
                &sh.export_unity_shader_graph(),
                &sh.param_layout,
                "float d; float3 n; unity::AliceSdf_float(p, d, n); return d;",
            )
        }),
    );
}

// ===========================================================================
// Anti-vacuity
// ===========================================================================

/// Re-run the whole sweep against a mutated emit and require **every** node to
/// be reported. This is what distinguishes "the oracle agrees" from "the
/// oracle measured nothing": a node whose binary never ran, whose output was
/// discarded or whose comparison was skipped would pass the parity test and
/// fail here.
///
/// The mutation renames the entry point and interposes a wrapper, which
/// compiles for every node and shifts every distance by far more than
/// `REL_TOL`. `sdf_eval` occurs exactly once per emit (in its signature —
/// helpers never call it), so the rename cannot miss.
#[test]
fn oracle_flags_a_perturbed_emit_for_every_node() {
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    for (name, node) in corpus() {
        let sh = HlslShader::transpile(&node, HlslTranspileMode::Hardcoded);
        assert_eq!(
            sh.source.matches("sdf_eval").count(),
            1,
            "{name}: expected exactly one `sdf_eval` occurrence to rename"
        );
    }

    let (drifted, _, _, checked) = sweep(&cxx, "hlsl_mutant", HlslTranspileMode::Hardcoded, |s| {
        s.replace(
            "float sdf_eval(float3 p)",
            "float sdf_eval_unmutated(float3 p)",
        ) + "\nfloat sdf_eval(float3 p) { return sdf_eval_unmutated(p) + 0.125f; }\n"
    });
    let total = corpus().len();
    assert_eq!(checked, total, "not every mutated node was measured");
    assert_eq!(
        drifted.len(),
        total,
        "the oracle flagged only {} of {total} mutated nodes — the rest are not \
         actually being compared. Unflagged nodes are the ones missing from:\n{}",
        drifted.len(),
        drifted.join("\n")
    );
    eprintln!("mutated emit was rejected for all {total} corpus nodes");
}

// ===========================================================================
// BlinkScript
// ===========================================================================

/// BlinkScript delegates its function body to `HlslShader` on purpose
/// (`src/compiled/blinkscript/mod.rs`, "Why not a separate `ShaderLang`
/// impl?"). Pin that contract so the delegation cannot quietly become a
/// second, unverified copy of the law.
#[test]
#[cfg(feature = "blinkscript")]
fn blinkscript_body_is_byte_identical_to_hlsl() {
    use alice_sdf::compiled::blinkscript::{BlinkScriptShader, BlinkScriptTranspileMode};
    let mut differing = Vec::new();
    for (name, node) in corpus() {
        for (bm, hm) in [
            (
                BlinkScriptTranspileMode::Hardcoded,
                HlslTranspileMode::Hardcoded,
            ),
            (
                BlinkScriptTranspileMode::Dynamic,
                HlslTranspileMode::Dynamic,
            ),
        ] {
            let blink = BlinkScriptShader::transpile(&node, bm);
            let hlsl = HlslShader::transpile(&node, hm);
            if blink.source != hlsl.source {
                differing.push(format!("{name} ({bm:?})"));
            }
            if blink.param_layout != hlsl.param_layout {
                differing.push(format!("{name} ({bm:?}): param_layout differs"));
            }
        }
    }
    assert!(
        differing.is_empty(),
        "BlinkScript body diverged from HLSL for: {}",
        differing.join(", ")
    );
}

/// ... and check the BlinkScript emit's values on their own, rather than
/// inferring them from the byte-equality above: if the delegation is ever
/// replaced by a real `ShaderLang` impl, this test keeps holding the line.
#[test]
#[cfg(feature = "blinkscript")]
fn blinkscript_matches_cpu_for_every_corpus_node() {
    use alice_sdf::compiled::blinkscript::{BlinkScriptShader, BlinkScriptTranspileMode};
    let Some(cxx) = cxx_or_skip() else {
        return;
    };
    let dir = scratch("blinkscript");
    let pts = points(POINTS);
    let units: Vec<Unit> = corpus()
        .into_iter()
        .map(|(name, node)| {
            let sh = BlinkScriptShader::transpile(&node, BlinkScriptTranspileMode::Hardcoded);
            Unit {
                name: name.to_string(),
                src: translation_unit(&sh.source, &params_global(&sh.param_layout)),
            }
        })
        .collect();
    let bins = compile_all(&cxx, &dir, &units);

    let mut drifted = Vec::new();
    let mut checked = 0usize;
    let mut worst_all = (0.0_f32, String::new());
    for ((name, node), bin) in corpus().into_iter().zip(bins) {
        match bin.and_then(|b| run(&b, &pts)) {
            Ok(got) => {
                let w = worst_drift(&node, &pts, &got);
                if w.0 > worst_all.0 {
                    worst_all = (w.0, name.to_string());
                }
                let bar = tolerance(name);
                if w.0 > bar {
                    drifted.push(format!(
                        "{name}: BlinkScript/CPU drift {:.3e} (bar {bar:.1e}) at {:?} \
                         (cpu={} blink={})",
                        w.0, w.1, w.2, w.3
                    ));
                }
                checked += 1;
            }
            Err(e) => drifted.push(format!("{name}: {e}")),
        }
    }
    assert!(
        drifted.is_empty(),
        "{} corpus node(s) drifted between BlinkScript and CPU:\n{}",
        drifted.len(),
        drifted.join("\n")
    );
    assert_eq!(
        checked,
        corpus().len(),
        "not every corpus node was measured"
    );
    eprintln!(
        "BlinkScript/CPU parity over {checked} corpus nodes x {POINTS} points: \
         worst {:.3e} ({})",
        worst_all.0, worst_all.1
    );
}

/// The Nuke kernel container is the part BlinkScript does *not* share with
/// HLSL, so check its shape: the `ImageComputationKernel` declaration, the
/// `define()` / `process()` pair Nuke requires, the params it promises, and
/// the verbatim inclusion of the eval body.
#[test]
#[cfg(feature = "blinkscript")]
fn blinkscript_kernel_wraps_the_body_in_a_nuke_kernel() {
    use alice_sdf::compiled::blinkscript::{BlinkScriptShader, BlinkScriptTranspileMode};
    let node = corpus()
        .into_iter()
        .find(|(n, _)| *n == "sphere")
        .expect("corpus must contain `sphere`")
        .1;
    let sh = BlinkScriptShader::transpile(&node, BlinkScriptTranspileMode::Hardcoded);
    let kernel = sh.to_kernel(0.25, (-2.0, 2.0));
    for needle in [
        "kernel AliceSdfSliceKernel : ImageComputationKernel<eComponentWise>",
        "Image<eWrite, eAccessPoint> dst;",
        "void define() {",
        "void process(int2 pos) {",
        "defineParam(z_slice, \"z_slice\", 0.25)",
        "defineParam(bounds_min, \"bounds_min\", -2)",
        "defineParam(bounds_max, \"bounds_max\", 2)",
        "dst() = sdf_eval(p);",
    ] {
        assert!(
            kernel.contains(needle),
            "BlinkScript kernel is missing `{needle}`:\n{kernel}"
        );
    }
    assert!(
        kernel.contains(sh.get_eval_function()),
        "BlinkScript kernel does not embed its own eval body verbatim"
    );
    // Balanced braces: an unbalanced kernel is rejected by Nuke's compiler and
    // there is no Nuke on a CI runner to tell us.
    assert_eq!(
        kernel.matches('{').count(),
        kernel.matches('}').count(),
        "BlinkScript kernel braces are unbalanced"
    );
}

// ===========================================================================
// An independent parse: glslang's HLSL front-end + spirv-val
// ===========================================================================

/// `naga` cannot parse HLSL, but **glslang** can (`-D` selects the HLSL
/// front-end), and the SPIR-V it produces can be checked with `spirv-val`.
/// This is the HLSL equivalent of what `test_transpiler_naga_validate.rs` does
/// for WGSL and GLSL, and it exercises `to_compute_shader` — the resource
/// declarations, the `[numthreads]` attribute and the entry point — which the
/// value oracle above deliberately does not compile.
#[test]
fn hlsl_compute_shader_parses_with_glslang() {
    let have = |tool: &str| {
        Command::new(tool)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    };
    if !have("glslangValidator") || !have("spirv-val") {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GLSLANG").is_none(),
            "ALICE_SDF_REQUIRE_GLSLANG is set but glslangValidator / spirv-val were not found"
        );
        eprintln!("skipping HLSL glslang parse gate: glslangValidator / spirv-val absent");
        return;
    }

    let dir = scratch("hlsl_glslang");
    let mut failures = Vec::new();
    let mut checked = 0usize;
    for (name, node) in corpus() {
        for mode in [HlslTranspileMode::Hardcoded, HlslTranspileMode::Dynamic] {
            let shader = HlslShader::transpile(&node, mode).to_compute_shader();
            let hlsl = dir.join(format!("{name}_{mode:?}.hlsl"));
            let spv = dir.join(format!("{name}_{mode:?}.spv"));
            std::fs::write(&hlsl, &shader).expect("cannot write HLSL");
            let out = Command::new("glslangValidator")
                .args([
                    "-D",
                    "-e",
                    "main",
                    "-S",
                    "comp",
                    "--target-env",
                    "vulkan1.0",
                    "-o",
                ])
                .arg(&spv)
                .arg(&hlsl)
                .output()
                .expect("cannot run glslangValidator");
            if !out.status.success() {
                failures.push(format!(
                    "{name} ({mode:?}): glslang rejected the HLSL:\n{}",
                    String::from_utf8_lossy(&out.stdout)
                        .lines()
                        .chain(String::from_utf8_lossy(&out.stderr).lines())
                        .filter(|l| l.contains("ERROR"))
                        .take(6)
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
                continue;
            }
            let val = Command::new("spirv-val")
                .arg(&spv)
                .output()
                .expect("cannot run spirv-val");
            if !val.status.success() {
                failures.push(format!(
                    "{name} ({mode:?}): spirv-val rejected the SPIR-V:\n{}",
                    String::from_utf8_lossy(&val.stderr)
                        .lines()
                        .take(6)
                        .collect::<Vec<_>>()
                        .join("\n")
                ));
                continue;
            }
            checked += 1;
        }
    }
    assert!(
        failures.is_empty(),
        "{} HLSL compute shader(s) failed the glslang / spirv-val gate:\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    assert_eq!(
        checked,
        corpus().len() * 2,
        "not every corpus node was validated in both modes"
    );
    eprintln!("glslang + spirv-val accepted {checked} HLSL compute shaders");
}

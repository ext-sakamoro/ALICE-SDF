//! GLSL run on the CPU: the transpiler's output compiled by a C++ compiler against
//! `tests/common/glsl_cpu_shim.h` (every built-in written from the GLSL specification) and
//! evaluated over points, so GLSL values are checked without a GPU.
//!
//! The harness rewrites the two GLSL forms C++ cannot spell (the array constructor
//! `const float a[N] = float[N](..);` and the `ALICE_PRECISE` version test) and puts each tree in
//! its own namespace, so one binary evaluates the whole corpus.
//!
//! - `the_shim_runs_sdf_eval_like_the_cpu`: every corpus tree's `sdf_eval` agrees with `eval`
//!   (1e-4 relative, the transcendentals are libm's on one side and the crate's on the other),
//!   which checks the shim before anything is built on it.
//!
//! Author: Moroya Sakamoto

#![cfg(feature = "glsl")]

mod common;

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use alice_sdf::compiled::{GlslShader, GlslTranspileMode};
use alice_sdf::eval::eval;
use alice_sdf::SdfNode;
use glam::Vec3;

const SHIM: &str = include_str!("common/glsl_cpu_shim.h");

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
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("glsl_cpu_shim");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The GLSL forms C++ cannot spell, rewritten: the array constructor
/// `const T a[N] = T[N](..);` (module or function scope, any element type) becomes a brace
/// initializer; every line gets C++ float literals ([`float_literals`])
fn cxx_from_glsl(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        out.push_str(&float_literals(&array_ctor(line)));
        out.push('\n');
    }
    out
}

/// `const T name[N] = T[N](items);` → `static const T name[N] = {items};`, other lines unchanged;
/// a `T[` constructor anywhere else is a form the harness does not handle and fails
fn array_ctor(line: &str) -> String {
    let t = line.trim_start();
    let indent = &line[..line.len() - t.len()];
    if let Some(rest) = t.strip_prefix("const ") {
        if let Some(sp) = rest.find(' ') {
            let ty = &rest[..sp];
            let decl = &rest[sp + 1..];
            let ctor = format!("] = {ty}[");
            if let (Some(eq), true) = (decl.find(&ctor), decl.ends_with(");")) {
                let open = decl[eq..].find('(').expect("array constructor") + eq;
                return format!(
                    "{indent}static const {ty} {}] = {{{}}};",
                    &decl[..eq],
                    &decl[open + 1..decl.len() - 2]
                );
            }
        }
    }
    for ty in ["float[", "uint[", "int["] {
        assert!(
            !t.contains(ty) || t.starts_with("//"),
            "an array form the harness does not rewrite: {t}"
        );
    }
    line.to_owned()
}

/// GLSL's float literals (`0.5`, `1e-5`, `2.`) as C++ `float` literals (`0.5f`): unsuffixed they
/// are `double` in C++, which would compute the expression in double precision; integer literals
/// stay `int`, as in GLSL
fn float_literals(line: &str) -> String {
    let b = line.as_bytes();
    let mut out = String::with_capacity(line.len() + 16);
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let starts_number = (c.is_ascii_digit()
            || (c == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit()))
            && (i == 0
                || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_' || b[i - 1] == b'.'));
        if !starts_number {
            out.push(c as char);
            i += 1;
            continue;
        }
        let start = i;
        let mut is_float = false;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i < b.len() && b[i] == b'.' {
            is_float = true;
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            is_float = true;
            i += 1;
            if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                i += 1;
            }
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        out.push_str(&line[start..i]);
        let suffixed =
            i < b.len() && (b[i] == b'f' || b[i] == b'F' || b[i] == b'u' || b[i] == b'U');
        if is_float && !suffixed {
            out.push('f');
        }
    }
    out
}

/// One C++ program: the shim, each source in namespace `t<i>`, and a host that reads points and
/// writes, per point and per source, the floats `calls(i)` produces (`calls` writes into `out`).
pub fn program(sources: &[String], per_source: usize, calls: impl Fn(usize) -> String) -> String {
    let mut s = String::from("#define ALICE_PRECISE\n");
    s.push_str(SHIM);
    for (i, src) in sources.iter().enumerate() {
        s.push_str(&format!("namespace t{i} {{\n{}\n}}\n", cxx_from_glsl(src)));
    }
    let per = sources.len() * per_source;
    let mut body = String::new();
    for i in 0..sources.len() {
        body.push_str(&format!(
            "    {{ float *o = out + {}; {} }}\n",
            i * per_source,
            calls(i)
        ));
    }
    s.push_str(&format!(
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
static float out[{per}];
int main() {{
    ALICE_SDF_SET_BINARY_IO();
    unsigned int count = 0;
    if (!read_exact(&count, 4)) return 2;
    for (unsigned int k = 0; k < count; ++k) {{
        float buf[3];
        if (!read_exact(buf, 12)) return 3;
        vec3 p = vec3(buf[0], buf[1], buf[2]);
{body}        if (!write_exact(out, sizeof(out))) return 4;
    }}
    return 0;
}}
"
    ));
    s
}

/// Compile `src` and run it over `pts`; `per` floats come back per point.
pub fn compile_and_run(cxx: &str, name: &str, src: &str, pts: &[Vec3], per: usize) -> Vec<f32> {
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
            .take(16)
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
    raw.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect()
}

/// Deterministic points in [−1.5, 1.5]³ (xorshift32)
pub fn points(n: usize, seed: u32) -> Vec<Vec3> {
    let mut s = seed;
    let mut r = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        (s >> 8) as f32 / (1u32 << 24) as f32
    };
    (0..n)
        .map(|_| Vec3::new(-1.5 + 3.0 * r(), -1.5 + 3.0 * r(), -1.5 + 3.0 * r()))
        .collect()
}

#[test]
fn the_literal_rewrite_suffixes_floats_only() {
    assert_eq!(
        float_literals("x = clamp(a / 0.89, 0.0, 1.0) + 1e-5 * b2 + v.x + 3 + 2. - .5;"),
        "x = clamp(a / 0.89f, 0.0f, 1.0f) + 1e-5f * b2 + v.x + 3 + 2.f - .5f;"
    );
    assert_eq!(
        float_literals("d12 + 5 * (d7 + 3 * d8)"),
        "d12 + 5 * (d7 + 3 * d8)"
    );
}

#[test]
fn the_shim_runs_sdf_eval_like_the_cpu() {
    let Some(cxx) = cxx_or_skip() else { return };
    let trees: Vec<(&str, SdfNode)> = common::corpus::corpus();
    let sources: Vec<String> = trees
        .iter()
        .map(|(_, n)| GlslShader::transpile(n, GlslTranspileMode::Hardcoded).source)
        .collect();
    let src = program(&sources, 1, |i| format!("o[0] = t{i}::sdf_eval(p);"));
    let pts = points(64, 0x2545_f491);
    let got = compile_and_run(&cxx, "shim_corpus", &src, &pts, trees.len());
    let mut compared = 0;
    let mut bad = Vec::new();
    for (k, p) in pts.iter().enumerate() {
        for (i, (name, node)) in trees.iter().enumerate() {
            let (g, c) = (got[k * trees.len() + i], eval(node, *p));
            let ok = (g.is_nan() && c.is_nan()) || (g - c).abs() <= 1e-4 * c.abs().max(1.0);
            if !ok {
                bad.push(format!("{name} at {p:?}: glsl {g} vs eval {c}"));
            }
            compared += 1;
        }
    }
    assert!(compared > 90 * 64, "{compared}");
    assert!(
        bad.is_empty(),
        "{} of {compared} differ, e.g.\n{}",
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );
}

/// The slab on a sheet squeezed to a tenth in y (Λ ≈ 10), as in `tests/test_trace_hit_scale.rs`
fn squeezed() -> SdfNode {
    use alice_sdf::transforms::{grid_warp_from_triangles, GridWarp};
    let n = 9usize;
    let mut rest = Vec::new();
    for j in 0..n {
        for i in 0..n {
            rest.push(Vec3::new(i as f32 / 8.0, j as f32 / 8.0, 0.0));
        }
    }
    let mut tris = Vec::new();
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let (b, c, d) = (a + 1, a + n as u32, a + n as u32 + 1);
            tris.push([a, b, d]);
            tris.push([a, d, c]);
        }
    }
    let posed: Vec<Vec3> = rest
        .iter()
        .map(|p| Vec3::new(p.x, 0.1 * p.y, p.z))
        .collect();
    // the posed box, a cell of 0.1 and a reach of 0.3, no shell bound (no floor)
    let data = grid_warp_from_triangles(
        &rest,
        &posed,
        &tris,
        Vec3::new(-0.3, -0.3, -0.3),
        Vec3::new(1.3, 0.4, 0.3),
        0.1,
        0.3,
    )
    .expect("builds");
    let warp = GridWarp::new(data).expect("valid");
    SdfNode::box3d(1.0, 1.0, 0.04)
        .translate(0.5, 0.5, 0.0)
        .grid_warp(warp)
}

/// The corpus and the trees whose scale is not 1: a grid warp alone, under a union, a smooth
/// union, a subtraction, a translate and a uniform scale, and a non-uniform scale over it
fn scaled_trees() -> Vec<(String, SdfNode)> {
    let mut t: Vec<(String, SdfNode)> = common::corpus::corpus()
        .into_iter()
        .map(|(n, node)| (n.to_owned(), node))
        .collect();
    let ball = || SdfNode::sphere(0.3).translate(0.5, 0.05, 0.4);
    t.push((
        "projective_nonidentity".into(),
        common::corpus::projective_nonidentity(),
    ));
    t.push(("warp".into(), squeezed()));
    t.push(("warp_union".into(), squeezed().union(ball())));
    t.push((
        "warp_smooth_union".into(),
        squeezed().smooth_union(ball(), 0.2),
    ));
    t.push(("warp_subtract".into(), squeezed().subtract(ball())));
    t.push((
        "warp_moved".into(),
        squeezed().translate(0.1, -0.2, 0.05).scale(1.3),
    ));
    t.push(("scaled_warp".into(), squeezed().scale_xyz(1.0, 0.5, 2.0)));
    t
}

#[test]
fn sdf_eval_scaled_is_sdf_eval_with_the_cpu_scale() {
    // oracle: `.x` is the library's own `sdf_eval` bit for bit (the same emitted body); `.y` is
    // `eval::eval_scaled`'s scale, computed independently on the CPU
    let Some(cxx) = cxx_or_skip() else { return };
    let trees = scaled_trees();
    let sources: Vec<String> = trees
        .iter()
        .map(|(_, n)| GlslShader::transpile_with_scale(n, GlslTranspileMode::Hardcoded).source)
        .collect();
    let src = program(&sources, 3, |i| {
        format!(
            "o[0] = t{i}::sdf_eval(p); vec2 v = t{i}::sdf_eval_scaled(p); o[1] = v.x; o[2] = v.y;"
        )
    });
    let mut pts = points(64, 0x9e37_79b9);
    pts.extend(
        points(64, 0x1234_5678)
            .into_iter()
            .map(|p| Vec3::new(0.5 + 0.4 * p.x, 0.05 + 0.04 * p.y, 0.1 * p.z)),
    );
    // just outside the warp's box (z = 0.3 + 1e-6): the box distance is tiny next to the value,
    // so the value is still the child's / Λ to the last bit, but the box decides it (scale 1)
    for j in 0..8 {
        for i in 0..8 {
            pts.push(Vec3::new(
                0.05 + 0.12 * i as f32,
                0.01 * j as f32,
                0.3 + 1e-6,
            ));
        }
    }
    let got = compile_and_run(&cxx, "scaled_corpus", &src, &pts, trees.len() * 3);
    let (mut compared, mut not_one) = (0, 0);
    let mut bad = Vec::new();
    for (k, p) in pts.iter().enumerate() {
        for (i, (name, node)) in trees.iter().enumerate() {
            let o = &got[(k * trees.len() + i) * 3..][..3];
            if o[1].to_bits() != o[0].to_bits() {
                bad.push(format!(
                    "{name} at {p:?}: sdf_eval_scaled.x {} vs sdf_eval {}",
                    o[1], o[0]
                ));
            }
            let (_, s) = alice_sdf::eval::eval_scaled(node, *p);
            if (o[2] - s).abs() > 1e-4 * s.abs().max(1.0) {
                bad.push(format!("{name} at {p:?}: scale glsl {} vs cpu {s}", o[2]));
            }
            if s != 1.0 {
                not_one += 1;
            }
            compared += 1;
        }
    }
    assert!(compared > 95 * 192, "{compared}");
    assert!(
        not_one > 100,
        "the scale was 1 almost everywhere: {not_one}"
    );
    assert!(
        bad.is_empty(),
        "{} differ, e.g.\n{}",
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );
}

#[test]
fn the_scaled_source_starts_with_the_unchanged_library() {
    // `transpile_with_scale` adds a function after `transpile`'s output and changes nothing in it
    let mut n = 0;
    for (name, node) in scaled_trees() {
        for mode in [GlslTranspileMode::Hardcoded, GlslTranspileMode::Dynamic] {
            let plain = GlslShader::transpile(&node, mode);
            let scaled = GlslShader::transpile_with_scale(&node, mode);
            assert!(
                scaled.source.starts_with(&plain.source),
                "{name} {mode:?}: the library part differs"
            );
            assert!(
                scaled.source.contains("vec2 sdf_eval_scaled(vec3 p) {"),
                "{name}"
            );
            assert_eq!(scaled.param_layout, plain.param_layout, "{name} {mode:?}");
            n += 1;
        }
    }
    assert!(n > 190, "{n}");
}

/// A box under two pulls, in the plane
fn pulled() -> (SdfNode, alice_sdf::transforms::BellPull) {
    let pull = alice_sdf::transforms::BellPull::new(
        [[0.5, 0.5, 0.2, -0.1], [-0.2, 0.3, -0.05, 0.1]],
        0.34,
    )
    .expect("one to one");
    (
        SdfNode::box3d(0.8, 0.8, 0.16).translate(0.2, 0.2, 0.0),
        pull,
    )
}

#[test]
fn a_bell_pull_in_the_shader_is_the_bell_pull_on_the_cpu() {
    // oracle: eval_scaled, computed on the CPU. Literal values, and the live source with the uniforms
    // set to the same values
    let Some(cxx) = cxx_or_skip() else { return };
    let (child, pull) = pulled();
    let tree = child.bell_pull(pull);
    let literal = GlslShader::transpile_with_scale(&tree, GlslTranspileMode::Hardcoded).source;
    let live = GlslShader::transpile_with_live_pull(&tree).source;
    assert!(live.contains("uniform vec4 uPull;") && !literal.contains("uPull"));
    let [a, b] = pull.pulls();
    let set = format!(
        "static vec4 uPull = vec4({:e}f, {:e}f, {:e}f, {:e}f);\nstatic vec4 uPull2 = vec4({:e}f, {:e}f, {:e}f, {:e}f);\nstatic float uPullSigma = {:e}f;",
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], pull.sigma()
    );
    let live = live.replace(
        "uniform vec4 uPull;\nuniform vec4 uPull2;\nuniform float uPullSigma;",
        &set,
    );
    assert!(!live.contains("uniform"), "the uniforms were not replaced");
    let src = program(&[literal, live], 2, |i| {
        format!("vec2 v = t{i}::sdf_eval_scaled(p); o[0] = v.x; o[1] = v.y;")
    });
    let mut pts = points(128, 0x7fed_7fed);
    pts.extend(
        points(128, 0x5bd1_e995)
            .into_iter()
            .map(|p| Vec3::new(0.4 + 0.5 * p.x, 0.4 + 0.5 * p.y, 0.1 * p.z)),
    );
    let got = compile_and_run(&cxx, "bell_pull", &src, &pts, 4);
    let mut bad = Vec::new();
    for (k, p) in pts.iter().enumerate() {
        let (d, s) = alice_sdf::eval::eval_scaled(&tree, *p);
        for i in 0..2 {
            let (gd, gs) = (got[k * 4 + i * 2], got[k * 4 + i * 2 + 1]);
            if (gd - d).abs() > 1e-4 * d.abs().max(1.0) || (gs - s).abs() > 1e-4 * s.abs().max(1.0)
            {
                bad.push(format!("source {i} at {p:?}: ({gd}, {gs}) vs ({d}, {s})"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "{} differ, e.g.\n{}",
        bad.len(),
        bad[..bad.len().min(8)].join("\n")
    );
    // nothing moved is the child, value and scale
    let still = alice_sdf::transforms::BellPull::new([[0.0; 4]; 2], 0.34).unwrap();
    let (c, _) = pulled();
    for p in points(64, 0x1234_abcd) {
        let (d, s) = alice_sdf::eval::eval_scaled(&c.clone().bell_pull(still), p);
        assert_eq!(d.to_bits(), eval(&c, p).to_bits(), "{p:?}");
        assert_eq!(s, 1.0);
    }
}

#[test]
fn a_bell_pull_that_could_fold_is_refused() {
    use alice_sdf::transforms::{BellPull, BellPullError};
    // |v| = 0.6 with σ 0.34: slope 0.6 · 0.6065 / 0.34 = 1.07
    assert!(matches!(
        BellPull::new([[0.0, 0.0, 0.6, 0.0], [0.0; 4]], 0.34),
        Err(BellPullError::Folds(s)) if (s - 1.0704).abs() < 1e-3
    ));
    assert!(BellPull::new([[0.0, 0.0, 0.46, 0.0], [0.0; 4]], 0.34).is_ok());
    assert_eq!(BellPull::new([[0.0; 4]; 2], 0.0), Err(BellPullError::Bad));
    assert_eq!(
        BellPull::new([[f32::NAN, 0.0, 0.0, 0.0], [0.0; 4]], 0.3),
        Err(BellPullError::Bad)
    );
    // stored: read through the same check (a pull is read as the field of a tree node)
    let fits = SdfNode::sphere(0.5)
        .bell_pull(BellPull::new([[0.0, 0.0, 0.46, 0.0], [0.0; 4]], 0.34).expect("one to one"));
    let json = serde_json::to_string(&fits).unwrap();
    let folds = json.replace("0.46", "0.6");
    assert_ne!(folds, json);
    assert!(serde_json::from_str::<SdfNode>(&folds).is_err());
    let (child, pull) = pulled();
    let tree = child.bell_pull(pull);
    let round: SdfNode = serde_json::from_str(&serde_json::to_string(&tree).unwrap()).unwrap();
    let p = Vec3::new(0.4, 0.3, 0.05);
    assert_eq!(eval(&round, p).to_bits(), eval(&tree, p).to_bits());
}

#[test]
fn a_bell_pull_never_steps_past_the_surface() {
    // sampled slope of the value and scale, against nothing from the law's bound: the value's slope
    // stays at or below 1 for a 1-Lipschitz child, so a step equal to the value cannot cross
    let (child, pull) = pulled();
    let tree = child.bell_pull(pull);
    let (mut worst, mut compared) = (0.0f32, 0);
    let pts = points(4000, 0xabcd_0123);
    let dirs = points(4000, 0x0123_abcd);
    for (p, d) in pts.iter().zip(&dirs) {
        let d = d.normalize_or_zero();
        if d == Vec3::ZERO {
            continue;
        }
        let (a, b) = (eval(&tree, *p), eval(&tree, *p + d * 2e-3));
        worst = worst.max((a - b).abs() / 2e-3);
        compared += 1;
    }
    assert!(compared > 3000, "{compared}");
    assert!(worst <= 1.0 + 1e-2, "slope {worst}");
    assert!(worst > 0.3, "the check saw almost no slope: {worst}");
}

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
    raw.chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
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
    use alice_sdf::transforms::{grid_warp_from_triangles, GridWarp, WarpGrid};
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
    let grid = WarpGrid {
        box_min: Vec3::new(-0.3, -0.3, -0.3),
        box_max: Vec3::new(1.3, 0.4, 0.3),
        cell: 0.1,
        reach: 0.3,
        shell: None,
        overhang: None,
    };
    let warp =
        GridWarp::new(grid_warp_from_triangles(&rest, &posed, &tris, &grid).expect("builds"))
            .expect("valid");
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


/// A slab on a sheet at rest, and on the same sheet with one corner pulled out: the shape and the
/// grid are the same, the maps are not
fn live_warps() -> (SdfNode, alice_sdf::transforms::GridWarp, alice_sdf::transforms::GridWarp) {
    use alice_sdf::transforms::{grid_warp_from_triangles, GridWarp, WarpGrid};
    let n = 5usize;
    let mut rest = Vec::new();
    for j in 0..n {
        for i in 0..n {
            rest.push(Vec3::new(i as f32 / 4.0, j as f32 / 4.0, 0.0));
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
        .map(|p| *p + Vec3::new(0.3, 0.2, 0.05) * (p.x * p.y).powi(2))
        .collect();
    let grid = WarpGrid {
        box_min: Vec3::new(-0.3, -0.3, -0.3),
        box_max: Vec3::new(1.5, 1.5, 0.3),
        cell: 0.1,
        reach: 0.3,
        shell: Some(0.06),
        overhang: None,
    };
    let at = |p: &[Vec3]| {
        GridWarp::new(grid_warp_from_triangles(&rest, p, &tris, &grid).expect("builds")).expect("valid")
    };
    let child = SdfNode::box3d(0.8, 0.8, 0.08).translate(0.5, 0.5, 0.0);
    (child, at(&rest), at(&posed))
}

/// The live source with `data` in its texture, as C++
fn live_source(source: &str, data: &[f32]) -> String {
    let w = alice_sdf::compiled::GLSL_NODE_DATA_WIDTH;
    let texels: Vec<String> = data.iter().map(|v| format!("{:e}", v)).collect();
    let decl = "uniform highp sampler2D alice_node_data_tex;";
    assert!(source.contains(decl), "the source does not read the texture");
    let read = format!(
        "static const float alice_texels[{}] = {{{}}};\nstruct sampler2D {{ int unused; }};\nstatic sampler2D alice_node_data_tex;\nstruct alice_texel {{ float r; }};\nstatic alice_texel texelFetch(sampler2D, ivec2 c, int) {{ return alice_texel{{alice_texels[c.y * {w} + c.x]}}; }}",
        data.len(),
        texels.join(",")
    );
    source
        .replace(decl, &read)
        .replace("out float ", "float &")
}

#[test]
fn a_live_warp_read_from_the_texture_is_the_posed_warp_on_the_cpu() {
    // oracle: the shader is built once, at rest; with the posed warp's data in its texture its
    // value and scale are eval_scaled of the tree holding the posed warp
    let Some(cxx) = cxx_or_skip() else { return };
    let (child, rest, posed) = live_warps();
    let (shader, at_build) = GlslShader::transpile_with_live_data(&child.clone().grid_warp(rest.clone()));
    assert_eq!(at_build, rest.live_node_data(), "the built data is the warp's layout");
    assert!(at_build.len() > 4096, "{} floats: past a uniform block", at_build.len());
    assert_eq!(shader.source.matches("vec3 alice_warp_").count(), 1, "one function per warp");
    let mut pts = points(96, 0x0bad_cafe);
    pts.extend(
        points(160, 0x1357_9bdf)
            .into_iter()
            .map(|p| Vec3::new(0.5 + 0.4 * p.x, 0.5 + 0.4 * p.y, 0.1 * p.z)),
    );
    let mut differs = 0;
    for (name, warp) in [("rest", &rest), ("posed", &posed)] {
        let tree = child.clone().grid_warp(warp.clone());
        let data = warp.live_node_data();
        assert_eq!(data.len(), at_build.len(), "the layout does not move");
        let src = program(&[live_source(&shader.source, &data)], 2, |i| {
            format!("vec2 v = t{i}::sdf_eval_scaled(p); o[0] = v.x; o[1] = v.y;")
        });
        let got = compile_and_run(&cxx, &format!("live_{name}"), &src, &pts, 2);
        let mut bad = Vec::new();
        for (k, p) in pts.iter().enumerate() {
            let (d, s) = alice_sdf::eval::eval_scaled(&tree, *p);
            let (gd, gs) = (got[k * 2], got[k * 2 + 1]);
            if (gd - d).abs() > 1e-4 * d.abs().max(1.0) || (gs - s).abs() > 1e-4 * s.abs().max(1.0) {
                bad.push(format!("{name} at {p:?}: glsl ({gd}, {gs}) vs cpu ({d}, {s})"));
            }
            if name == "posed" && (d - alice_sdf::eval::eval(&child.clone().grid_warp(rest.clone()), *p)).abs() > 1e-3 {
                differs += 1;
            }
        }
        assert!(bad.is_empty(), "{} differ, e.g.\n{}", bad.len(), bad[..bad.len().min(8)].join("\n"));
    }
    assert!(differs > 50, "the pose hardly moved the field: {differs}");
}

#[test]
fn a_live_warp_keeps_its_ranges_when_two_arrays_hold_the_same_values() {
    // two warps equal today can differ next frame: each has its own range and its own function
    let (child, rest, _) = live_warps();
    let one = rest.live_node_data().len();
    let tree = child.clone().grid_warp(rest.clone()).union(child.grid_warp(rest));
    let (shader, data) = GlslShader::transpile_with_live_data(&tree);
    assert_eq!(data.len(), 2 * one);
    assert_eq!(shader.source.matches("vec3 alice_warp_").count(), 2);
}

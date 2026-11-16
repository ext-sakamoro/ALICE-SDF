//! `repeat_finite` places whole copies of its child on the lattice of its spacing, on every path.
//!
//! A small sphere repeated with spacing 10 is sampled along each axis (the other two coordinates
//! 0) in steps of 1/40 of the sphere's diameter; each run of samples inside the shape is a copy
//! seen along that axis. Every copy must be a whole child (its inside run as wide as the sphere's
//! diameter) centred on a multiple of the spacing, and the copies must be symmetric about the
//! origin. The tree evaluator, the compiled bytecode (scalar and SIMD batch) and the scalar and
//! SIMD JIT (feature `jit`) are each checked against this, not against each other; the interval
//! evaluator is checked to hold the tree's distance, and the shader and Rust emits to clamp to
//! ±floor(count/2). `SITES` lists every place in `src/` that names the node, so a new evaluator
//! fails here until it is classified and checked.
//!
//! The cell index is clamped to ±floor(count/2), so an axis has 2·floor(count/2) + 1 whole copies.
//! Even counts were already whole; odd counts are the case this guards: a clamp to ±count/2 (not
//! an integer for an odd count) cut the child in half at ±1.5 spacing (count 3) or ±0.5 spacing
//! (count 1).
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, eval_compiled_batch_simd, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use glam::Vec3;

const SPACING: f32 = 10.0;
const RADIUS: f32 = 0.5;
const STEP: f32 = 2.0 * RADIUS / 40.0;

fn repeated(count: u32) -> SdfNode {
    SdfNode::sphere(RADIUS).repeat_finite([count; 3], Vec3::splat(SPACING))
}

/// the sample points along `axis` (0 = x) over the whole extent the count could reach, and beyond
fn line(axis: usize, count: u32) -> Vec<Vec3> {
    let reach = (count as f32 + 2.0) * SPACING;
    let n = (2.0 * reach / STEP) as i32;
    (0..=n)
        .map(|i| {
            let mut p = Vec3::ZERO;
            p[axis] = -reach + i as f32 * STEP;
            p
        })
        .collect()
}

/// the inside runs of `values` along `points` as (start, end) coordinates on `axis`
fn runs(points: &[Vec3], values: &[f32], axis: usize) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    let mut start: Option<f32> = None;
    for (p, &d) in points.iter().zip(values) {
        match (d < 0.0, start) {
            (true, None) => start = Some(p[axis]),
            (false, Some(s)) => {
                out.push((s, p[axis] - STEP));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, points.last().map_or(s, |p| p[axis])));
    }
    out
}

/// every run is a whole sphere centred on a multiple of the spacing, and the runs are symmetric
fn check(path: &str, count: u32, axis: usize, runs: &[(f32, f32)]) -> Result<usize, String> {
    if runs.is_empty() {
        return Err(format!("{path} count {count} axis {axis}: no copy at all"));
    }
    for &(a, b) in runs {
        let width = b - a + STEP;
        let centre = f32::midpoint(a, b);
        let lattice = (centre / SPACING).round() * SPACING;
        if (width - 2.0 * RADIUS).abs() > 2.0 * STEP || (centre - lattice).abs() > STEP {
            return Err(format!(
                "{path} count {count} axis {axis}: a copy over [{a:.3}, {b:.3}] (width {width:.3}, \
                 centre {centre:.3}) is not a whole child on the lattice of {SPACING}; copies: {runs:?}"
            ));
        }
    }
    let centres: Vec<f32> = runs.iter().map(|&(a, b)| f32::midpoint(a, b)).collect();
    let mirrored: Vec<f32> = centres.iter().rev().map(|c| -c).collect();
    if centres
        .iter()
        .zip(&mirrored)
        .any(|(c, m)| (c - m).abs() > STEP)
    {
        return Err(format!(
            "{path} count {count} axis {axis}: copies are not symmetric about the origin: {centres:?}"
        ));
    }
    Ok(runs.len())
}

fn check_all(path: &str, counts: &[u32], eval_line: impl Fn(&SdfNode, &[Vec3]) -> Vec<f32>) {
    let mut failures = Vec::new();
    let mut compared = 0;
    for &count in counts {
        let node = repeated(count);
        for axis in 0..3 {
            let points = line(axis, count);
            let values = eval_line(&node, &points);
            match check(path, count, axis, &runs(&points, &values, axis)) {
                // 2·floor(count/2) + 1 whole copies on every axis
                Ok(n) if n == 2 * (count / 2) as usize + 1 => compared += 1,
                Ok(n) => failures.push(format!(
                    "{path} count {count} axis {axis}: {n} copies, not 2·floor(count/2) + 1 = {}",
                    2 * (count / 2) + 1
                )),
                Err(e) => failures.push(e),
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(
        compared,
        counts.len() * 3,
        "{path}: every count and axis compared"
    );
}

const EVEN: [u32; 3] = [2, 4, 6];
const ODD: [u32; 4] = [1, 3, 5, 7];

fn tree(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    points.iter().map(|&p| eval(node, p)).collect()
}

fn compiled(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    let c = CompiledSdf::compile(node);
    points.iter().map(|&p| eval_compiled(&c, p)).collect()
}

fn simd(node: &SdfNode, points: &[Vec3]) -> Vec<f32> {
    eval_compiled_batch_simd(&CompiledSdf::compile(node), points)
}

#[test]
fn even_counts_place_whole_copies_on_the_lattice_on_every_path() {
    check_all("tree", &EVEN, tree);
    check_all("compiled", &EVEN, compiled);
    check_all("simd", &EVEN, simd);
}

#[test]
fn odd_counts_place_whole_copies_on_the_lattice_in_the_tree_evaluator() {
    check_all("tree", &ODD, tree);
}

#[test]
fn odd_counts_place_whole_copies_on_the_lattice_in_the_compiled_bytecode() {
    check_all("compiled", &ODD, compiled);
    check_all("simd", &ODD, simd);
}

#[cfg(feature = "jit")]
#[test]
fn odd_and_even_counts_place_whole_copies_on_the_lattice_in_the_jit() {
    use alice_sdf::compiled::jit::JitCompiledSdf;
    let jit = |node: &SdfNode, points: &[Vec3]| {
        JitCompiledSdf::compile(node)
            .expect("repeat_finite of a sphere is JIT-supported")
            .eval_batch(points)
    };
    check_all("jit", &EVEN, jit);
    check_all("jit", &ODD, jit);
    let jit_simd = |node: &SdfNode, points: &[Vec3]| {
        let xs: Vec<f32> = points.iter().map(|p| p.x).collect();
        let ys: Vec<f32> = points.iter().map(|p| p.y).collect();
        let zs: Vec<f32> = points.iter().map(|p| p.z).collect();
        alice_sdf::compiled::jit::JitSimdSdf::compile(&CompiledSdf::compile(node))
            .expect("repeat_finite of a sphere is SIMD-JIT-supported")
            .eval_batch(&xs, &ys, &zs)
    };
    check_all("jit_simd", &EVEN, jit_simd);
    check_all("jit_simd", &ODD, jit_simd);
}

/// the clamp limit `floor(count / 2)` per axis for `repeated(count)` with `count` on x, `count + 1`
/// on y and `count + 2` on z (so one scene holds an odd and an even axis)
#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu", feature = "rust"))]
fn mixed(count: u32) -> (SdfNode, [f32; 3]) {
    let c = [count, count + 1, count + 2];
    (
        SdfNode::sphere(RADIUS).repeat_finite(c, Vec3::splat(SPACING)),
        // the law's form: the count as f32, halved, floored (the bytecode holds the count as f32)
        c.map(|n| (n as f32 * 0.5).floor()),
    )
}

#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu", feature = "rust"))]
// 2^25 + 3 and up: `(n / 2) as f32` (integer halving) and `(n as f32 * 0.5).floor()` differ, so
// every emit must use the law's form
const MIXED: [u32; 5] = [1, 2, 3, 4, (1 << 25) + 3];

/// The interval evaluator over a box of one point holds the tree's distance at that point, on the
/// sample lines of every odd and even count: a clamp other than ±floor(count/2) puts a copy where
/// the tree has none (or the reverse) and the point falls outside the interval
#[test]
fn the_interval_evaluator_holds_the_tree_distance_for_odd_and_even_counts() {
    use alice_sdf::interval::{eval_interval, Interval, Vec3Interval};
    let mut compared = 0;
    for &count in EVEN.iter().chain(&ODD) {
        let node = repeated(count);
        for axis in 0..3 {
            for p in line(axis, count).into_iter().step_by(7) {
                let d = eval(&node, p);
                let iv = eval_interval(
                    &node,
                    Vec3Interval {
                        x: Interval::new(p.x, p.x),
                        y: Interval::new(p.y, p.y),
                        z: Interval::new(p.z, p.z),
                    },
                );
                assert!(
                    iv.lo <= d && d <= iv.hi,
                    "count {count} at {p}: tree {d} outside [{}, {}]",
                    iv.lo,
                    iv.hi
                );
                compared += 1;
            }
        }
    }
    assert!(compared > 1000, "compared {compared}");
}

/// the limits of every `clamp(floor(..), -limit, limit)` in shader text, as written
#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
fn shader_limits(src: &str) -> Vec<[f32; 3]> {
    if src.contains("metal::clamp(metal::floor(") {
        return msl_limits(src);
    }
    let mut out = Vec::new();
    for line in src.lines().filter(|l| l.contains("clamp(floor(")) {
        let after = &line[line.find("clamp(floor(").unwrap()..];
        let neg = after.find(", -").expect("a negative limit");
        let open = neg + after[neg..].find('(').expect("a vector constructor");
        let close = open + after[open..].find(')').expect("the end of the vector");
        let v: Vec<f32> = after[open + 1..close]
            .split(',')
            .map(|s| s.trim().trim_end_matches('f').parse().expect("a number"))
            .collect();
        out.push([v[0], v[1], v[2]]);
    }
    out
}

/// the upper limits of every `metal::clamp(metal::floor(..), lower, upper)` in naga's MSL
#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
fn msl_limits(src: &str) -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for line in src
        .lines()
        .filter(|l| l.contains("metal::clamp(metal::floor("))
    {
        let after = &line[line.find("metal::clamp(metal::floor(").unwrap()..];
        let open = after.rfind("metal::float3(").expect("the upper limit") + "metal::float3(".len();
        let close = open + after[open..].find(')').expect("the end of the vector");
        let v: Vec<f32> = after[open..close]
            .split(',')
            .map(|s| s.trim().parse().expect("a number"))
            .collect();
        out.push([v[0], v[1], v[2]]);
    }
    out
}

/// Each shader language the crate emits clamps the cell index of a finite repetition to
/// ±floor(count/2) per axis (GLSL / HLSL / WGSL share `transpiler_common`; each is read as emitted)
#[cfg(any(feature = "glsl", feature = "hlsl", feature = "gpu"))]
#[test]
fn every_shader_emit_clamps_to_half_the_count_rounded_down() {
    let mut compared = 0;
    for count in MIXED {
        let (node, want) = mixed(count);
        #[allow(unused_mut)]
        let mut emitted: Vec<(&str, String)> = Vec::new();
        #[cfg(feature = "glsl")]
        emitted.push((
            "glsl",
            alice_sdf::compiled::GlslShader::transpile(
                &node,
                alice_sdf::compiled::GlslTranspileMode::Hardcoded,
            )
            .source,
        ));
        #[cfg(feature = "hlsl")]
        emitted.push((
            "hlsl",
            alice_sdf::compiled::HlslShader::transpile(
                &node,
                alice_sdf::compiled::HlslTranspileMode::Hardcoded,
            )
            .source,
        ));
        #[cfg(feature = "msl")]
        emitted.push((
            "msl",
            alice_sdf::compiled::msl::MslShader::transpile(
                &node,
                alice_sdf::compiled::TranspileMode::Hardcoded,
            )
            .expect("repeat_finite of a sphere emits MSL")
            .source,
        ));
        #[cfg(feature = "gpu")]
        emitted.push((
            "wgsl",
            alice_sdf::compiled::WgslShader::transpile(
                &node,
                alice_sdf::compiled::TranspileMode::Hardcoded,
            )
            .source,
        ));
        for (lang, src) in emitted {
            let limits = shader_limits(&src);
            assert_eq!(limits, vec![want], "{lang} count {count}");
            compared += 1;
        }
    }
    assert!(compared >= MIXED.len(), "compared {compared}");
}

/// The emitted Rust (`fn sdf`) clamps each axis to ±floor(count/2)
#[cfg(feature = "rust")]
#[test]
fn the_emitted_rust_clamps_to_half_the_count_rounded_down() {
    let mut compared = 0;
    for count in MIXED {
        let (node, want) = mixed(count);
        let src = alice_sdf::compiled::RustSource::transpile(&node).expect("transpiles");
        let got: Vec<f32> = src
            .source()
            .lines()
            .filter(|l| l.contains(").floor().max(("))
            .map(|l| {
                let at = l.find(".min(").expect("an upper limit") + 5;
                let end = at + l[at..].find("_f32").expect("an f32 literal");
                l[at..end].parse::<f32>().expect("a number")
            })
            .collect();
        assert_eq!(got, want.to_vec(), "count {count}");
        compared += 1;
    }
    assert_eq!(compared, MIXED.len());
}

/// Every place in `src/` that names a finite repetition, by file, with what it does with it. A
/// new place (a sixth evaluator, a new emit) changes a count here and has to be classified: an
/// evaluator gets a check in this file, anything else says why it computes nothing
const SITES: &[(&str, usize, &str)] = &[
    (
        "src/compiled/real.rs",
        5,
        "the law: real::repeat_finite (tree, bytecode scalar and SIMD batch, gradient)",
    ),
    (
        "src/compiled/eval_core.rs",
        2,
        "bytecode op → real::repeat_finite (compiled, simd)",
    ),
    (
        "src/modifiers/repeat.rs",
        4,
        "modifier_repeat_finite → real::repeat_finite (tree)",
    ),
    (
        "src/eval/gradient.rs",
        2,
        "modifier_repeat_finite (tree gradient)",
    ),
    (
        "src/compiled/jit/codegen.rs",
        1,
        "scalar JIT (checked: jit)",
    ),
    (
        "src/compiled/jit/simd/mod.rs",
        4,
        "SIMD JIT, two builders (checked: jit_simd)",
    ),
    (
        "src/compiled/rust/transpiler.rs",
        3,
        "Rust emit (checked: the_emitted_rust_…)",
    ),
    (
        "src/compiled/transpiler_common.rs",
        2,
        "GLSL / HLSL / WGSL / MSL emit (checked: every_shader_emit_…) and the scale-kind variant list",
    ),
    (
        "src/interval.rs",
        2,
        "interval evaluator (checked: the_interval_evaluator_…)",
    ),
    (
        "src/compiled/refit.rs",
        7,
        "AABB refit: count·spacing, a superset of ±floor(count/2)·spacing",
    ),
    (
        "src/compiled/compiler.rs",
        4,
        "bytecode encoding (count as f32) and traversal",
    ),
    ("src/compiled/instruction.rs", 2, "instruction constructor"),
    ("src/compiled/opcode.rs", 3, "opcode tables"),
    ("src/compiled/wgsl/transpiler.rs", 1, "traversal only"),
    (
        "src/eval/mod.rs",
        8,
        "descend and eval_material (apply modifier_repeat_finite), variant lists, a node of the \
         eval_rec arm test",
    ),
    ("src/eval/scaled.rs", 1, "variant list"),
    ("src/diff.rs", 2, "tree diff"),
    ("src/llm_schema.rs", 2, "schema text and traversal"),
    ("src/optimize.rs", 2, "rebuilds the node"),
    ("src/types/mod.rs", 3, "the variant and its category"),
    ("src/types/modifiers.rs", 2, "constructor"),
    ("src/types/drop.rs", 1, "drop traversal"),
    ("src/modifiers/mod.rs", 1, "re-export"),
    ("src/ffi/modifiers.rs", 2, "FFI constructor"),
    ("src/python/transforms.rs", 2, "Python constructor"),
];

#[test]
fn every_place_in_src_that_names_a_finite_repetition_is_classified() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for e in std::fs::read_dir(dir).expect("read src") {
            let p = e.expect("an entry").path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root.join("src"), &mut files);
    let mut found: Vec<(String, usize)> = files
        .iter()
        .filter_map(|p| {
            let text = std::fs::read_to_string(p).expect("utf-8 source");
            let n = text
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .filter(|l| l.contains("RepeatFinite") || l.contains("repeat_finite"))
                .count();
            let rel = p
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            (n > 0).then_some((rel, n))
        })
        .collect();
    found.sort();
    let mut want: Vec<(String, usize)> = SITES
        .iter()
        .map(|(f, n, _)| ((*f).to_owned(), *n))
        .collect();
    want.sort();
    assert!(files.len() > 100, "read {} source files", files.len());
    assert_eq!(
        found, want,
        "a place that names RepeatFinite was added, removed or changed: classify it in SITES"
    );
}

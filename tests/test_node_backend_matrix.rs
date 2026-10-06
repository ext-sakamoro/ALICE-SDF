//! Which `SdfNode` variant reaches which backend, checked against the ledger
//! `docs/node-support.md`.
//!
//! The evaluators and transpilers dispatch on `SdfNode` with `match`. Where a
//! `match` has no `_ =>` arm, rustc already refuses a variant that is not
//! handled (GLSL / WGSL / HLSL / BlinkScript). Where it has one, a new variant
//! compiles and falls into it silently: the tree evaluator returns `f32::MAX`,
//! the interval evaluator returns `Interval::EVERYTHING`. This test runs every
//! variant through the backends whose outcome can be observed at run time and
//! records the result per variant:
//!
//! | column     | `ok` means                                     | otherwise              |
//! |------------|------------------------------------------------|------------------------|
//! | `eval`     | a finite distance other than `f32::MAX`        | `fallback`             |
//! | `interval` | a bounded enclosure over a ±2 box              | `everything`           |
//! | `compile`  | `CompiledSdf::try_compile` returns `Ok`        | `err`                  |
//! | `jit`      | `JitCompiledSdf::compile` returns `Ok` (`jit`) | `err`                  |
//! | `jit_simd` | `JitSimdSdf::compile` of the compiled tree     | `err`, `-` (no bytecode) |
//! | `msl`      | `MslShader::transpile` returns `Ok` (`msl`)    | `err`                  |
//! | `rust`     | `RustSource::transpile` returns `Ok` (`rust`)  | `err`                  |
//! | `ffi`      | a constructor `alice_sdf_<name>` in `src/ffi/` | `missing`              |
//! | `python`   | a constructor `fn <name>(` in `src/python/`    | `missing`              |
//!
//! The two binding columns are matched by name (the variant in snake case,
//! `2D` as `_2d`, `Subtraction` as `subtract`, and the few names in
//! `binding_aliases`), so they say whether a constructor of that name exists,
//! not that it builds the right variant.
//!
//! Checks (each one fails the test):
//! - every variant of `SdfNode` (read from `src/types/mod.rs`) is the top-level
//!   variant of at least one entry (the shared corpus, plus `Triangle` and
//!   `Bezier`, which the compiler rejects and the corpus therefore leaves out), so a new variant without a corpus
//!   entry is caught here instead of falling through every `_ =>` arm unseen
//! - the measured cells equal the committed ledger, in both directions: a backend
//!   that stops handling a variant fails, and so does one that starts handling
//!   it until the ledger records the change
//! - the ledger lists exactly the variants of the enum, and at least one cell
//!   was compared
//!
//! Columns of a feature that is off in this build are not compared. CI runs the
//! test with `jit,msl,rust` so every column is compared there.
//!
//! Regenerate the ledger after an intended change (all three features on):
//! `ALICE_SDF_WRITE_NODE_SUPPORT=1 cargo test --features jit,msl,rust --test test_node_backend_matrix`
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::interval::{eval_interval, Vec3Interval};
use alice_sdf::prelude::*;
use common::corpus::corpus;
use std::collections::BTreeMap;

const LEDGER: &str = "docs/node-support.md";
const COLUMNS: [&str; 9] = [
    "eval", "interval", "compile", "jit", "jit_simd", "msl", "rust", "ffi", "python",
];

/// Variant names of `SdfNode`, in declaration order, read from the source.
fn enum_variants() -> Vec<String> {
    let src = include_str!("../src/types/mod.rs");
    let start = src
        .find("pub enum SdfNode {")
        .expect("pub enum SdfNode in src/types/mod.rs");
    let body = &src[start..];
    let end = body.find("\n}\n").expect("end of enum SdfNode");
    let mut out = Vec::new();
    for line in body[..end].lines().skip(1) {
        // a variant starts at four spaces of indentation with an upper-case name
        if let Some(rest) = line.strip_prefix("    ") {
            if rest.starts_with(' ') {
                continue;
            }
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                out.push(name);
            }
        }
    }
    out
}

/// The top-level variant of a node, from its `Debug` form.
fn variant_of(node: &SdfNode) -> String {
    format!("{node:?}")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn eval_cell(node: &SdfNode) -> &'static str {
    let pts = [
        Vec3::new(0.3, -0.2, 0.1),
        Vec3::new(1.7, 0.4, -0.9),
        Vec3::ZERO,
    ];
    let ok = pts.iter().any(|&p| {
        let d = eval(node, p);
        d.is_finite() && d != f32::MAX
    });
    if ok {
        "ok"
    } else {
        "fallback"
    }
}

fn interval_cell(node: &SdfNode) -> &'static str {
    let b = Vec3Interval::from_bounds(Vec3::splat(-2.0), Vec3::splat(2.0));
    let i = eval_interval(node, b);
    if i.lo.is_finite() || i.hi.is_finite() {
        "ok"
    } else {
        "everything"
    }
}

fn compile_cell(node: &SdfNode) -> &'static str {
    if CompiledSdf::try_compile(node).is_ok() {
        "ok"
    } else {
        "err"
    }
}

#[cfg(feature = "jit")]
fn jit_cell(node: &SdfNode) -> Option<&'static str> {
    Some(
        if alice_sdf::compiled::jit::JitCompiledSdf::compile(node).is_ok() {
            "ok"
        } else {
            "err"
        },
    )
}
#[cfg(not(feature = "jit"))]
const fn jit_cell(_: &SdfNode) -> Option<&'static str> {
    None
}

#[cfg(feature = "jit")]
fn jit_simd_cell(node: &SdfNode) -> Option<&'static str> {
    let Ok(c) = CompiledSdf::try_compile(node) else {
        return Some("-");
    };
    Some(
        if alice_sdf::compiled::jit::JitSimdSdf::compile(&c).is_ok() {
            "ok"
        } else {
            "err"
        },
    )
}
#[cfg(not(feature = "jit"))]
const fn jit_simd_cell(_: &SdfNode) -> Option<&'static str> {
    None
}

#[cfg(feature = "msl")]
fn msl_cell(node: &SdfNode) -> Option<&'static str> {
    use alice_sdf::compiled::{msl::MslShader, TranspileMode};
    Some(
        if MslShader::transpile(node, TranspileMode::Hardcoded).is_ok() {
            "ok"
        } else {
            "err"
        },
    )
}
#[cfg(not(feature = "msl"))]
const fn msl_cell(_: &SdfNode) -> Option<&'static str> {
    None
}

#[cfg(feature = "rust")]
fn rust_cell(node: &SdfNode) -> Option<&'static str> {
    Some(
        if alice_sdf::compiled::rust::RustSource::transpile(node).is_ok() {
            "ok"
        } else {
            "err"
        },
    )
}
#[cfg(not(feature = "rust"))]
const fn rust_cell(_: &SdfNode) -> Option<&'static str> {
    None
}

/// `Circle2D` -> `circle_2d`, `ExpSmoothSubtraction` -> `exp_smooth_subtract`.
fn constructor_name(variant: &str) -> String {
    let v = variant
        .replace("2D", "_2d")
        .replace("Subtraction", "Subtract");
    // `_` before an upper-case letter that follows a lower-case letter or a digit,
    // so an acronym stays one word (`IWP` -> `iwp`, `FischerKochS` -> `fischer_koch_s`)
    let mut out = String::new();
    let mut prev: Option<char> = None;
    for c in v.chars() {
        if c.is_ascii_uppercase()
            && prev.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            out.push('_');
        }
        out.push(c.to_ascii_lowercase());
        prev = Some(c);
    }
    out
}

/// Constructor names that differ from `constructor_name` in one binding:
/// (variant, ffi name, python name).
fn binding_aliases(variant: &str) -> (Option<&'static str>, Option<&'static str>) {
    match variant {
        "Box3d" => (Some("box"), None),
        "SdfSkinning" => (Some("skinning"), None),
        "RepeatInfinite" => (Some("repeat"), Some("repeat")),
        "ScaleNonUniform" => (None, Some("scale_xyz")),
        "Diamond" => (None, Some("diamond_shape")),
        "Stairs" => (None, Some("stairs_shape")),
        _ => (None, None),
    }
}

fn read_tree(dir: &str) -> String {
    fn walk(p: &std::path::Path, out: &mut String) {
        let mut entries: Vec<_> = std::fs::read_dir(p)
            .expect("read binding dir")
            .flatten()
            .collect();
        entries.sort_by_key(std::fs::DirEntry::path);
        for e in entries {
            let path = e.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                *out += &std::fs::read_to_string(&path).expect("read binding source");
                out.push('\n');
            }
        }
    }
    let mut s = String::new();
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(dir),
        &mut s,
    );
    assert!(s.len() > 1000, "{dir}: no source read");
    s
}

fn has_fn(src: &str, name: &str) -> bool {
    // `fn name(` / `fn name<` as a whole word
    let pat = format!("fn {name}");
    src.match_indices(&pat).any(|(i, _)| {
        let before_ok = i == 0 || !src.as_bytes()[i - 1].is_ascii_alphanumeric();
        let after = src[i + pat.len()..].trim_start();
        before_ok && (after.starts_with('(') || after.starts_with('<'))
    })
}

/// (ffi, python) cells of one variant.
fn binding_cells(variant: &str, ffi: &str, py: &str) -> (&'static str, &'static str) {
    let base = constructor_name(variant);
    let (fa, pa) = binding_aliases(variant);
    let f = has_fn(ffi, &format!("alice_sdf_{base}"))
        || fa.is_some_and(|a| has_fn(ffi, &format!("alice_sdf_{a}")));
    let p = has_fn(py, &base) || pa.is_some_and(|a| has_fn(py, a));
    (
        if f { "ok" } else { "missing" },
        if p { "ok" } else { "missing" },
    )
}

/// The shared corpus holds only compilable trees (the parity tests compile every
/// entry). Variants the compiler rejects on purpose are added here, so their
/// row records that rejection instead of being missing.
fn entries() -> Vec<(&'static str, SdfNode)> {
    let mut v = corpus();
    v.push((
        "triangle",
        SdfNode::triangle(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.6, 0.2),
        ),
    ));
    v.push((
        "bezier",
        SdfNode::bezier(
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::new(0.6, 0.0, 0.0),
            0.1,
        ),
    ));
    v
}

/// Per variant, per column: the outcomes of every corpus entry of that variant
/// (joined with `/` when they differ). `None` for a column whose feature is off.
fn measure() -> BTreeMap<String, [Option<String>; 9]> {
    let mut acc: BTreeMap<String, [Vec<&'static str>; 9]> = BTreeMap::new();
    let mut seen_feature = [true, true, true, false, false, false, false, true, true];
    let ffi = read_tree("src/ffi");
    let py = read_tree("src/python");
    for (_name, node) in entries() {
        let (f, p) = binding_cells(&variant_of(&node), &ffi, &py);
        let cells = [
            Some(eval_cell(&node)),
            Some(interval_cell(&node)),
            Some(compile_cell(&node)),
            jit_cell(&node),
            jit_simd_cell(&node),
            msl_cell(&node),
            rust_cell(&node),
            Some(f),
            Some(p),
        ];
        let row = acc.entry(variant_of(&node)).or_default();
        for (i, c) in cells.iter().enumerate() {
            if let Some(c) = c {
                seen_feature[i] = true;
                if !row[i].contains(c) {
                    row[i].push(c);
                }
            }
        }
    }
    acc.into_iter()
        .map(|(k, v)| {
            let cols = std::array::from_fn(|i| {
                seen_feature[i].then(|| {
                    let mut s = v[i].clone();
                    s.sort_unstable();
                    s.join("/")
                })
            });
            (k, cols)
        })
        .collect()
}

fn render(variants: &[String], m: &BTreeMap<String, [Option<String>; 9]>) -> String {
    let mut out = String::from(
        "# SdfNode backend support\n\n\
_Generated by `tests/test_node_backend_matrix.rs` (`ALICE_SDF_WRITE_NODE_SUPPORT=1 cargo test --features jit,msl,rust --test test_node_backend_matrix`); \
no timestamp, the file changes only when its content does._\n\n\
Every `SdfNode` variant, run through the backends whose outcome can be observed at run time, \
using the shared test corpus (`tests/common/corpus.rs`). CI fails when a cell changes and this file does not.\n\n\
| Column | `ok` | otherwise |\n|--------|------|-----------|\n\
| `eval` | the tree evaluator returns a finite distance other than `f32::MAX` | `fallback`: the unknown-variant arm |\n\
| `interval` | `eval_interval` returns a bounded enclosure over a ±2 box | `everything`: no pruning for this variant |\n\
| `compile` | `CompiledSdf::try_compile` succeeds | `err` |\n\
| `jit` | `JitCompiledSdf::compile` (scalar tree JIT) succeeds (`jit` feature) | `err` |\n\
| `jit_simd` | `JitSimdSdf::compile` of the compiled tree succeeds (`jit` feature) | `err`; `-` when the tree does not compile |\n\
| `msl` | `MslShader::transpile` succeeds (`msl` feature) | `err` |\n\
| `rust` | `RustSource::transpile` succeeds (`rust` feature) | `err` |\n\
| `ffi` | a constructor `alice_sdf_<name>` exists in `src/ffi/` (matched by name) | `missing` |\n\
| `python` | a constructor `fn <name>(` exists in `src/python/` (matched by name) | `missing` |\n\n\
The GLSL, WGSL, HLSL and BlinkScript transpilers are not listed: they share one dispatch \
(`transpile_node_inner` in `src/compiled/transpiler_common.rs`) whose `match` has no wildcard arm, so rustc \
rejects a variant it does not handle (`transpiler_dispatch_has_no_wildcard_arm` keeps it that way).\n\n\
| Variant | eval | interval | compile | jit | jit_simd | msl | rust | ffi | python |\n|---------|------|----------|---------|-----|----------|-----|------|-----|--------|\n",
    );
    for v in variants {
        let row = m.get(v).expect("every variant is measured");
        let cells: Vec<String> = row
            .iter()
            .map(|c| c.clone().unwrap_or_else(|| "?".into()))
            .collect();
        out += &format!("| `{v}` | {} |\n", cells.join(" | "));
    }
    out
}

/// Ledger rows: variant -> cells.
fn parse_ledger(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("| `") else {
            continue;
        };
        let Some((name, tail)) = rest.split_once("` |") else {
            continue;
        };
        if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
            continue; // the column legend rows start with a lower-case name
        }
        let cells: Vec<String> = tail
            .split('|')
            .map(|c| c.trim().to_string())
            .filter(|c| !c.is_empty())
            .collect();
        out.insert(name.to_string(), cells);
    }
    out
}

#[test]
fn every_variant_reaches_the_backends_the_ledger_records() {
    let variants = enum_variants();
    assert!(
        variants.len() >= 100,
        "read only {} variants from src/types/mod.rs",
        variants.len()
    );
    let m = measure();

    let missing: Vec<&String> = variants.iter().filter(|v| !m.contains_key(*v)).collect();
    assert!(
        missing.is_empty(),
        "SdfNode variants with no entry in tests/common/corpus.rs (add one): {missing:?}"
    );
    let unknown: Vec<&String> = m.keys().filter(|k| !variants.contains(k)).collect();
    assert!(
        unknown.is_empty(),
        "corpus top-level variants not in the enum: {unknown:?}"
    );

    let path = format!("{}/{LEDGER}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var_os("ALICE_SDF_WRITE_NODE_SUPPORT").is_some() {
        assert!(
            m.values().all(|r| r.iter().all(Option::is_some)),
            "write the ledger with every feature column on: --features jit,msl,rust"
        );
        std::fs::write(&path, render(&variants, &m)).expect("write ledger");
        return;
    }

    let ledger =
        parse_ledger(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{LEDGER}: {e}")));
    let ledger_names: Vec<&String> = ledger.keys().collect();
    let mut sorted_variants: Vec<&String> = variants.iter().collect();
    sorted_variants.sort();
    assert_eq!(
        ledger_names, sorted_variants,
        "{LEDGER} does not list exactly the SdfNode variants"
    );

    let mut diffs = Vec::new();
    let mut compared = 0usize;
    for v in &variants {
        let row = &m[v];
        let want = &ledger[v];
        assert_eq!(
            want.len(),
            COLUMNS.len(),
            "{LEDGER}: row `{v}` has {} cells",
            want.len()
        );
        for (i, col) in COLUMNS.iter().enumerate() {
            if let Some(got) = &row[i] {
                compared += 1;
                if got != &want[i] {
                    diffs.push(format!("{v}.{col}: measured `{got}`, ledger `{}`", want[i]));
                }
            }
        }
    }
    assert!(
        compared >= variants.len() * 3,
        "compared only {compared} cells"
    );
    assert!(
        diffs.is_empty(),
        "backend support changed ({} cells); if intended, regenerate {LEDGER} \
         (ALICE_SDF_WRITE_NODE_SUPPORT=1 cargo test --features jit,msl,rust --test test_node_backend_matrix):\n  {}",
        diffs.len(),
        diffs.join("\n  ")
    );
}

#[test]
fn the_enum_reader_finds_the_variants_it_should() {
    let v = enum_variants();
    for name in [
        "Sphere",
        "Box3d",
        "MetricBall",
        "MetricBlend",
        "Union",
        "Translate",
        "Twist",
        "WithMaterial",
        "Terrain",
    ] {
        assert!(
            v.contains(&name.to_string()),
            "{name} not read from src/types/mod.rs"
        );
    }
    // field names (indented deeper) are not variants
    assert!(!v
        .iter()
        .any(|n| n == "Radius" || n.chars().next().is_some_and(char::is_lowercase)));
}

#[test]
fn constructor_names_follow_the_binding_convention() {
    assert_eq!(constructor_name("Circle2D"), "circle_2d");
    assert_eq!(constructor_name("RoundedRect2D"), "rounded_rect_2d");
    assert_eq!(
        constructor_name("ExpSmoothSubtraction"),
        "exp_smooth_subtract"
    );
    assert_eq!(constructor_name("Box3d"), "box3d");
    assert_eq!(constructor_name("IWP"), "iwp");
    assert_eq!(constructor_name("FischerKochS"), "fischer_koch_s");
    assert!(has_fn(
        "pub extern \"C\" fn alice_sdf_box(x: f32)",
        "alice_sdf_box"
    ));
    assert!(!has_fn(
        "pub extern \"C\" fn alice_sdf_box_frame(x: f32)",
        "alice_sdf_box"
    ));
    assert!(!has_fn("fn my_alice_sdf_box(", "alice_sdf_box"));
}

/// The body of the first `match node {` after `fn <name>`, with comments removed.
fn match_body(src: &str, fn_name: &str) -> String {
    let code: String = src
        .lines()
        .map(|l| l.split("//").next().unwrap_or(""))
        .collect::<Vec<_>>()
        .join("\n");
    let f = code
        .find(&format!("fn {fn_name}"))
        .unwrap_or_else(|| panic!("fn {fn_name} not found"));
    let m = f + code[f..]
        .find("match node {")
        .unwrap_or_else(|| panic!("no `match node` in {fn_name}"));
    let start = m + "match node {".len();
    let mut depth = 1;
    for (i, c) in code[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return code[start..start + i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced match in {fn_name}");
}

/// Whether a `match` body has a `_ =>` arm at its top level.
fn has_top_level_wildcard(body: &str) -> bool {
    let mut depth = 0i32;
    let mut top = String::new();
    for c in body.chars() {
        match c {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ if depth == 0 => top.push(c),
            _ => {}
        }
    }
    // an arm may carry attributes (`#[allow(..)] _ => ..`): their brackets are
    // already dropped above, so skip the `#` that is left, and a leading `|`
    top.split([',', '\n']).any(|arm| {
        let a = arm.trim_start_matches(|c: char| c.is_whitespace() || c == '#' || c == '|');
        a.starts_with("_ =>") || a.starts_with("_ if ") || a == "_"
    })
}

#[test]
fn transpiler_dispatch_has_no_wildcard_arm() {
    let body = match_body(
        include_str!("../src/compiled/transpiler_common.rs"),
        "transpile_node_inner",
    );
    assert!(
        body.matches("SdfNode::").count() >= 100,
        "transpile_node_inner no longer dispatches on SdfNode"
    );
    assert!(
        !has_top_level_wildcard(&body),
        "transpile_node_inner gained a `_ =>` arm: a new SdfNode variant would compile without a shader law"
    );
    // the scanner itself: a wildcard at the top level is found, one inside an arm is not
    assert!(has_top_level_wildcard(
        "SdfNode::A { .. } => 1,\n    _ => 0,"
    ));
    assert!(!has_top_level_wildcard(
        "SdfNode::A { x } => match x { _ => 0 },"
    ));
    assert!(has_top_level_wildcard(
        "SdfNode::A { .. } => 1,\n    #[allow(unreachable_patterns)] _ => 0,"
    ));
    assert!(has_top_level_wildcard(
        "SdfNode::A { .. } => 1,\n    _ if false => 0,"
    ));
}

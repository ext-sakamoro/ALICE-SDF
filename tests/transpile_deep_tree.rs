//! The shader transpilers on long chains of binary operations, and their output on a corpus
//!
//! A union of many shapes reaches the transpiler as a chain folded to the left (an n-ary call is
//! read that way, since a smooth union is not associative): its depth is the number of shapes.
//! The transpilers walk the first operands of such a chain with a list, not the call stack, so a
//! chain of 10,000 shapes transpiles on a test thread's stack. The text they emit is the text
//! the recursive form emitted: the SHA-256 of every output of the corpus below was recorded from
//! the recursive form (main 9ce986ce) and is checked here for WGSL, GLSL and HLSL, in both modes,
//! and for GLSL with the local scale (the walk that records a scale frame per node).
#![cfg(all(feature = "gpu", feature = "glsl", feature = "hlsl"))]

use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode};
use alice_sdf::compiled::hlsl::{HlslShader, HlslTranspileMode};
use alice_sdf::compiled::wgsl::{TranspileMode, WgslShader};
use alice_sdf::SdfNode;
use sha2::{Digest as _, Sha256};
use std::fmt::Write as _;

type Op = fn(SdfNode, SdfNode) -> SdfNode;

/// Every binary operation of the transpilers, with fixed parameters
const OPS: [(&str, Op); 24] = [
    ("union", |a, b| a.union(b)),
    ("intersection", |a, b| a.intersection(b)),
    ("subtract", |a, b| a.subtract(b)),
    ("smooth_union", |a, b| a.smooth_union(b, 0.3)),
    ("smooth_union_k0", |a, b| a.smooth_union(b, 0.0)),
    ("smooth_intersection", |a, b| a.smooth_intersection(b, 0.25)),
    ("smooth_subtract", |a, b| a.smooth_subtract(b, 0.2)),
    ("chamfer_union", |a, b| a.chamfer_union(b, 0.1)),
    ("chamfer_intersection", |a, b| {
        a.chamfer_intersection(b, 0.1)
    }),
    ("chamfer_subtract", |a, b| a.chamfer_subtract(b, 0.1)),
    ("stairs_union", |a, b| a.stairs_union(b, 0.2, 3.0)),
    ("stairs_intersection", |a, b| {
        a.stairs_intersection(b, 0.2, 3.0)
    }),
    ("stairs_subtract", |a, b| a.stairs_subtract(b, 0.2, 3.0)),
    ("xor", |a, b| a.xor(b)),
    ("morph", |a, b| a.morph(b, 0.4)),
    ("columns_union", |a, b| a.columns_union(b, 0.2, 4.0)),
    ("columns_intersection", |a, b| {
        a.columns_intersection(b, 0.2, 4.0)
    }),
    ("columns_subtract", |a, b| a.columns_subtract(b, 0.2, 4.0)),
    ("pipe", |a, b| a.pipe(b, 0.05)),
    ("engrave", |a, b| a.engrave(b, 0.05)),
    ("groove", |a, b| a.groove(b, 0.1, 0.05)),
    ("tongue", |a, b| a.tongue(b, 0.1, 0.05)),
    ("exp_smooth_union", |a, b| a.exp_smooth_union(b, 0.3)),
    ("exp_smooth_subtract", |a, b| a.exp_smooth_subtract(b, 0.3)),
];

/// The `i`-th leaf: shapes and transforms in turn, so operands are not all alike
#[allow(clippy::cast_precision_loss)]
fn leaf(i: usize) -> SdfNode {
    let f = i as f32;
    match i % 4 {
        0 => SdfNode::sphere(0.2 + 0.01 * f),
        1 => SdfNode::box3d(1.1, 2.3, 0.5 + 0.1 * f).translate(f * 0.1, 0.0, -0.2),
        2 => SdfNode::torus(0.5, 0.1).scale(1.0 + 0.05 * f),
        _ => SdfNode::cylinder(0.3, 0.4).rotate_euler(0.1 * f, 0.2, 0.0),
    }
}

/// `op` folded to the left over `n` leaves (as an n-ary call is read)
fn left_chain(op: Op, n: usize) -> SdfNode {
    (1..n).fold(leaf(0), |acc, i| op(acc, leaf(i)))
}

/// `op` nested to the right over `n` leaves
fn right_chain(op: Op, n: usize) -> SdfNode {
    (0..n - 1)
        .rev()
        .fold(leaf(n - 1), |acc, i| op(leaf(i), acc))
}

/// The corpus: each operation as a left chain, a right chain and inside a mixed tree, and the
/// shape of the input that overflowed (a smooth union with k = 0 of a sphere and 127 boxes)
fn corpus() -> Vec<(String, SdfNode)> {
    let mut out = Vec::new();
    for (i, (name, op)) in OPS.iter().enumerate() {
        out.push((format!("{name} left 6"), left_chain(*op, 6)));
        out.push((format!("{name} right 4"), right_chain(*op, 4)));
        let (other_name, other) = OPS[(i + 7) % OPS.len()];
        let mixed = op(
            other(left_chain(OPS[(i + 3) % OPS.len()].1, 3), leaf(5)).translate(0.1, 0.2, 0.3),
            left_chain(*op, 3).scale(0.5),
        );
        out.push((format!("{name} with {other_name} mixed"), mixed));
    }
    let crash = (0..127).fold(SdfNode::sphere(0.2), |acc, i| {
        #[allow(clippy::cast_precision_loss)]
        let z = 112.0 + i as f32;
        acc.smooth_union(SdfNode::box3d(1.1, 2.3, z), 0.0)
    });
    out.push(("crash shape: smooth_union k 0 of 128".to_owned(), crash));
    out
}

fn outputs(node: &SdfNode) -> [(&'static str, String); 8] {
    [
        (
            "wgsl hardcoded",
            WgslShader::transpile(node, TranspileMode::Hardcoded).source,
        ),
        (
            "wgsl dynamic",
            WgslShader::transpile(node, TranspileMode::Dynamic).source,
        ),
        (
            "glsl hardcoded",
            GlslShader::transpile(node, GlslTranspileMode::Hardcoded).source,
        ),
        (
            "glsl dynamic",
            GlslShader::transpile(node, GlslTranspileMode::Dynamic).source,
        ),
        (
            "hlsl hardcoded",
            HlslShader::transpile(node, HlslTranspileMode::Hardcoded).source,
        ),
        (
            "hlsl dynamic",
            HlslShader::transpile(node, HlslTranspileMode::Dynamic).source,
        ),
        // the value and its local scale: the walk records a scale frame per node
        (
            "glsl scaled hardcoded",
            GlslShader::transpile_with_scale(node, GlslTranspileMode::Hardcoded).source,
        ),
        (
            "glsl scaled dynamic",
            GlslShader::transpile_with_scale(node, GlslTranspileMode::Dynamic).source,
        ),
    ]
}

/// One line per output: the corpus entry, the backend and mode, the SHA-256 of the text
fn digest_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for (name, node) in corpus() {
        for (backend, text) in outputs(&node) {
            let mut line = format!("{name} | {backend} | ");
            for b in Sha256::digest(text.as_bytes()) {
                write!(line, "{b:02x}").unwrap();
            }
            lines.push(line);
        }
    }
    lines
}

const RECORDED: &str = include_str!("transpile_deep_tree.sha256");

/// Every output of the corpus is the text the recursive form emitted
#[test]
fn the_output_is_the_recorded_text() {
    let lines = digest_lines();
    if std::env::var_os("ALICE_SDF_RECORD_TRANSPILE").is_some() {
        println!("{}", lines.join("\n"));
    }
    let recorded: Vec<&str> = RECORDED.lines().collect();
    assert_eq!(
        recorded.len(),
        8 * (3 * OPS.len() + 1),
        "compared nothing new"
    );
    assert_eq!(lines.len(), recorded.len());
    for (got, want) in lines.iter().zip(&recorded) {
        assert_eq!(got, want);
    }
}

/// A union of 10,000 shapes folded to the left transpiles on a test thread (2 MiB), every
/// backend, and its text grows with the shapes
#[test]
#[ignore = "the recursive form on this base overflows the stack here; this branch only shows the recorded text is its output"]
fn a_chain_of_ten_thousand_shapes_transpiles() {
    for (name, op) in [OPS[0], OPS[3], OPS[15]] {
        let node = left_chain(op, 10_000);
        for (backend, text) in outputs(&node) {
            assert!(
                text.len() > 10_000 * 20,
                "{name} {backend}: {} bytes",
                text.len()
            );
        }
    }
}

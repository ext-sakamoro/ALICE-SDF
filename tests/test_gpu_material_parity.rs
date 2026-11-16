//! GPU ↔ CPU parity of the material id (`WgslShader::transpile_material` against
//! `eval_material`) over the corpus.
//!
//! The corpus has almost no `WithMaterial`, so every tree with children is given
//! materials here: each leaf is wrapped in `WithMaterial` with its own id (1, 2, …
//! in serialisation order) and the whole tree in one more (999, which an inner tag must win
//! over everywhere). The wrapping goes through the stored form (`serde_json`):
//! a leaf is an object whose fields hold no node, so no list of variants is kept here
//! and a new variant is tagged without touching this file.
//!
//! The id at a point is compared where the CPU id is the same at the point and at its
//! six neighbours 2e-3 away; elsewhere the point is on a boundary between two
//! materials (a tie of the rule that picks the operand) and is skipped. The skipped
//! share is reported per tree and fails above `MAX_SKIPPED` (a rule that is wrong only
//! on a symmetric tree would otherwise hide behind ties), as does a tree with fewer
//! than `MIN_COMPARED` compared points.
//!
//! CI's gpu-parity job (lavapipe) sets `ALICE_SDF_REQUIRE_GPU=1`, so a missing
//! adapter fails instead of skipping.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

mod common;

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use serde_json::{Map, Value};

/// Largest share of a tree's points that may be skipped as material boundaries
const MAX_SKIPPED: f64 = 0.25;
/// Fewest compared points per tree
const MIN_COMPARED: usize = 64;
/// Neighbour offset of the boundary test
const EPS: f32 = 2e-3;

/// The node-holding fields of the stored form
const CHILD_FIELDS: [&str; 5] = ["a", "b", "child", "inner", "outer"];

fn is_node(v: &Value) -> bool {
    matches!(v, Value::Object(m) if m.len() == 1) || matches!(v, Value::String(_))
}

/// Wraps every leaf of the stored node `v` in `WithMaterial`; returns the number of leaves
fn tag_leaves(v: &mut Value, next: &mut u32) -> usize {
    let Value::Object(outer) = v else {
        // a unit variant: a leaf
        return wrap(v, next);
    };
    let (_, body) = outer.iter_mut().next().expect("externally tagged");
    let mut leaves = 0;
    if let Value::Object(fields) = body {
        for key in CHILD_FIELDS {
            if let Some(child) = fields.get_mut(key) {
                if is_node(child) {
                    leaves += tag_leaves(child, next);
                }
            }
        }
    }
    if leaves == 0 {
        wrap(v, next)
    } else {
        leaves
    }
}

fn wrap(v: &mut Value, next: &mut u32) -> usize {
    let mut body = Map::new();
    body.insert("child".into(), v.take());
    body.insert("material_id".into(), Value::from(*next));
    *next += 1;
    let mut outer = Map::new();
    outer.insert("WithMaterial".into(), Value::Object(body));
    *v = Value::Object(outer);
    1
}

/// The id of the tag around the whole tree: every leaf is tagged, so it shows only if an inner
/// tag stops winning over an outer one
const OUTER: u32 = 999;

fn tagged(node: &SdfNode) -> (SdfNode, usize) {
    let mut v = serde_json::to_value(node).expect("serialise");
    let mut next = 1;
    let leaves = tag_leaves(&mut v, &mut next);
    let mut outer = next;
    let _ = wrap(&mut v, &mut outer);
    if let Value::Object(m) = &mut v {
        if let Some(Value::Object(body)) = m.get_mut("WithMaterial") {
            body.insert("material_id".into(), Value::from(OUTER));
        }
    }
    (
        serde_json::from_value(v).expect("deserialise the tagged tree"),
        leaves,
    )
}

fn points(n: usize) -> Vec<Vec3> {
    let mut state: u64 = 0x6d61_7465_7269_616c;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 40) as f32) / ((1u64 << 24) as f32)) * 4.0 - 2.0
    };
    (0..n).map(|_| Vec3::new(next(), next(), next())).collect()
}

fn material_shader(node: &SdfNode) -> String {
    let main = WgslShader::transpile(node, TranspileMode::Hardcoded);
    let material = WgslShader::transpile_material(node, TranspileMode::Hardcoded);
    format!(
        r"struct InputPoint {{ x: f32, y: f32, z: f32, _pad: f32, }}
struct OutputDistance {{ distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }}
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
{}
{}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= point_count) {{ return; }}
    let q = input_points[idx];
    output_distances[idx].distance = sdf_eval_material(vec3<f32>(q.x, q.y, q.z));
}}
",
        main.source, material
    )
}

struct Row {
    name: &'static str,
    compared: usize,
    skipped: usize,
    wrong: usize,
    first_wrong: Option<(Vec3, u32, u32)>,
}

#[test]
fn gpu_material_matches_cpu_over_the_corpus() {
    let pts = points(4096);
    let offsets = [
        Vec3::X * EPS,
        -Vec3::X * EPS,
        Vec3::Y * EPS,
        -Vec3::Y * EPS,
        Vec3::Z * EPS,
        -Vec3::Z * EPS,
    ];
    // "no adapter" is decided once, on a plain shader; past this, a shader error is a failure
    if let Err(e) = GpuEvaluator::new(&SdfNode::sphere(1.0)) {
        assert!(
            std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
            "ALICE_SDF_REQUIRE_GPU is set but no GPU adapter was found: {e}"
        );
        eprintln!("skipping GPU material parity: {e}");
        return;
    }
    let mut rows = Vec::new();
    for (name, node) in common::corpus::corpus() {
        let (node, leaves) = tagged(&node);
        if leaves < 2 {
            // one material everywhere: nothing to pick between
            continue;
        }
        let src = material_shader(&node);
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
            panic!(
                "{name}: material WGSL does not parse: {}",
                e.emit_to_string(&src)
            )
        });
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: material WGSL does not validate: {e:?}"));
        let gpu = GpuEvaluator::from_wgsl(&src)
            .unwrap_or_else(|e| panic!("{name}: the GPU refused the material shader: {e}"))
            .into_parts()
            .0;
        let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("gpu eval");
        let mut row = Row {
            name,
            compared: 0,
            skipped: 0,
            wrong: 0,
            first_wrong: None,
        };
        for (&p, &g) in pts.iter().zip(&got) {
            let want = eval_material(&node, p);
            if offsets.iter().any(|&o| eval_material(&node, p + o) != want) {
                row.skipped += 1;
                continue;
            }
            row.compared += 1;
            if g != want as f32 {
                row.wrong += 1;
                row.first_wrong.get_or_insert((p, want, g as u32));
            }
        }
        rows.push(row);
    }
    let mut failures = Vec::new();
    for r in &rows {
        let share = r.skipped as f64 / (r.compared + r.skipped) as f64;
        eprintln!(
            "{:<32} compared {:>5} skipped {:>5} ({:>5.1} %) wrong {:>5}",
            r.name,
            r.compared,
            r.skipped,
            100.0 * share,
            r.wrong
        );
        if r.wrong > 0 {
            let (p, want, got) = r.first_wrong.expect("recorded");
            failures.push(format!(
                "{}: {} of {} wrong (first at {p:?}: CPU {want}, GPU {got})",
                r.name, r.wrong, r.compared
            ));
        }
        if share > MAX_SKIPPED {
            failures.push(format!("{}: {:.1} % skipped", r.name, 100.0 * share));
        }
        if r.compared < MIN_COMPARED {
            failures.push(format!("{}: only {} compared", r.name, r.compared));
        }
    }
    assert!(!rows.is_empty(), "no corpus tree has two leaves");
    eprintln!("{} trees compared", rows.len());
    assert!(
        failures.is_empty(),
        "{} trees:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

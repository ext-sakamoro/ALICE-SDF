//! No WGSL text the transpiler writes has an array constructor of more than 2047 elements.
//!
//! The WGSL limits table (W3C WGSL, section 2.4 "Limits"): "Maximum number of elements in value
//! constructor expression of array type: 2047". An implementation is only required to support
//! shaders within the limits, so a constructor over it is outside the language even where naga
//! accepts it (naga does not check this limit; a constructor of more than 65532 elements is written
//! to SPIR-V with a word count of 0).
//!
//! Here: every WGSL form of every corpus tree and of a 300 x 300 height map is parsed by naga.
//! Every `Compose` expression of an array type, at module scope and in every function, is counted,
//! and the largest must be at most `WGSL_ARRAY_CONSTRUCTOR_LIMIT`. Reading 0 shaders or 0 array
//! constructors fails.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

mod common;

use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use common::corpus::corpus;

/// W3C WGSL, section 2.4 "Limits": the elements of one array value constructor
const WGSL_ARRAY_CONSTRUCTOR_LIMIT: usize = 2047;

/// A sphere displaced by a 300 x 300 height map: one data array of 90000 values
fn large_field() -> SdfNode {
    let (w, h) = (300u32, 300u32);
    let map: Vec<f32> = (0..w * h)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect();
    SdfNode::sphere(0.5).heightmap_displacement(map, w, h, 0.02, 1.0)
}

/// The trees: the corpus (every kind with a data array: lofts, section stacks, polygons, height maps,
/// lattices) and a height map large enough to be split
fn trees() -> Vec<(String, SdfNode)> {
    let mut t: Vec<(String, SdfNode)> = corpus()
        .into_iter()
        .map(|(n, node)| (n.to_string(), node))
        .collect();
    t.push(("height map 300 x 300".into(), large_field()));
    t
}

/// Every WGSL text a tree is written as: the evaluator function's module in each mode and data form,
/// and the compute / normals shaders built from them
fn forms(node: &SdfNode) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for (label, shader) in [
        (
            "hardcoded",
            WgslShader::try_transpile(node, TranspileMode::Hardcoded).ok(),
        ),
        (
            "dynamic",
            WgslShader::try_transpile(node, TranspileMode::Dynamic).ok(),
        ),
        (
            "node data",
            WgslShader::try_transpile_with_node_data(node, TranspileMode::Hardcoded).ok(),
        ),
        (
            "union bounds",
            Some(WgslShader::transpile_with_union_bounds(
                node,
                TranspileMode::Hardcoded,
            )),
        ),
    ] {
        let Some(s) = shader else { continue };
        out.push((label, s.to_compute_shader()));
        out.push((label, s.to_compute_shader_with_normals()));
    }
    out
}

/// The element counts of the array constructors of `module`, at module scope and in every function
fn array_constructor_lengths(module: &naga::Module) -> Vec<usize> {
    let is_array = |ty: naga::Handle<naga::Type>| {
        matches!(module.types[ty].inner, naga::TypeInner::Array { .. })
    };
    let lengths = |arena: &naga::Arena<naga::Expression>| -> Vec<usize> {
        arena
            .iter()
            .filter_map(|(_, e)| match e {
                naga::Expression::Compose { ty, components } if is_array(*ty) => {
                    Some(components.len())
                }
                _ => None,
            })
            .collect()
    };
    let mut out = lengths(&module.global_expressions);
    for (_, f) in module.functions.iter() {
        out.extend(lengths(&f.expressions));
    }
    for ep in &module.entry_points {
        out.extend(lengths(&ep.function.expressions));
    }
    out
}

/// The largest array constructor of `text`, or the parse error
fn largest_constructor(text: &str) -> Result<Vec<usize>, String> {
    naga::front::wgsl::parse_str(text)
        .map(|m| array_constructor_lengths(&m))
        .map_err(|e| e.emit_to_string(text))
}

#[test]
fn no_wgsl_array_constructor_is_over_the_spec_limit() {
    let (mut shaders, mut constructors, mut largest) = (0usize, 0usize, (0usize, String::new()));
    let mut over = Vec::new();
    let mut failures = Vec::new();
    for (name, node) in trees() {
        for (form, text) in forms(&node) {
            match largest_constructor(&text) {
                Err(e) => failures.push(format!("{name} ({form}): parse failed\n{e}")),
                Ok(lengths) => {
                    shaders += 1;
                    constructors += lengths.len();
                    for n in lengths {
                        if n > largest.0 {
                            largest = (n, format!("{name} ({form})"));
                        }
                        if n > WGSL_ARRAY_CONSTRUCTOR_LIMIT {
                            over.push(format!("{name} ({form}): {n} elements"));
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "array constructors: {shaders} shaders, {constructors} constructors, largest {} in {}",
        largest.0, largest.1
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(shaders > 0, "no shader was read");
    assert!(constructors > 0, "no array constructor was seen");
    // the large height map has to reach the limit, or the check never sees a split array
    assert!(
        largest.0 >= WGSL_ARRAY_CONSTRUCTOR_LIMIT / 2,
        "the largest constructor has {} elements, the trees do not exercise the limit",
        largest.0
    );
    over.dedup();
    assert!(
        over.is_empty(),
        "{} array constructors over {WGSL_ARRAY_CONSTRUCTOR_LIMIT} elements:\n{}",
        over.len(),
        over.join("\n")
    );
}

/// The counter's teeth: a constructor of 2048 elements is over the limit, one of 2047 is not
#[test]
fn the_counter_sees_a_constructor_one_over_the_limit() {
    let decl = |n: usize| {
        let items = vec!["0.0"; n].join(", ");
        format!(
            "const A: array<f32, {n}> = array<f32, {n}>({items});\nfn f(i: i32) -> f32 {{ return A[i]; }}\n"
        )
    };
    let at = largest_constructor(&decl(WGSL_ARRAY_CONSTRUCTOR_LIMIT)).expect("parse");
    let over = largest_constructor(&decl(WGSL_ARRAY_CONSTRUCTOR_LIMIT + 1)).expect("parse");
    assert_eq!(at, vec![WGSL_ARRAY_CONSTRUCTOR_LIMIT]);
    assert_eq!(over, vec![WGSL_ARRAY_CONSTRUCTOR_LIMIT + 1]);
}

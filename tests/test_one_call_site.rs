//! A tree's shader has one call site per node kind: the call sites of the emitted WGSL (node data
//! buffer form) do not grow with the number of lofts, stacks or polygons in a union, and neither does
//! the time to build the pipeline and run the first dispatch.
//!
//! Why: a driver's compile time follows the call sites, not the text. With one copy of the loft's
//! functions per loft, a body of 17 lofts took 60-313 s to compile on real drivers, and 400 extruded
//! polygons took 643 s on an Orin and 7.6 GB on llvmpipe (more than a hosted CI runner holds). With
//! one call site per kind, in a loop over the nodes' offsets in the node data buffer, the same body
//! compiled in 6-19 s.
//!
//! The call sites are counted on the text: every function the module defines, every place it is
//! called. The time is measured on whatever adapter runs the test; the limit is a growth ratio, not a
//! clock, so it holds on a slow machine.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
use alice_sdf::prelude::*;
use glam::Vec2;
use std::time::Instant;

fn ring(a: f32, b: f32, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

fn star(n: usize, r: f32, dr: f32) -> Vec<Vec2> {
    (0..n)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / n as f32);
            let rr = if j % 2 == 0 { r } else { r - dr };
            Vec2::new(rr * c, rr * s)
        })
        .collect()
}

/// A balanced union, as a body's parts are, so no recursion goes deep
fn union_of(mut level: Vec<SdfNode>) -> SdfNode {
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter();
        while let Some(a) = it.next() {
            next.push(match it.next() {
                Some(b) => a.union(b),
                None => a,
            });
        }
        level = next;
    }
    level.pop().expect("a part")
}

fn curve_lofts(n: usize) -> SdfNode {
    union_of(
        (0..n)
            .map(|i| {
                let x = -0.8 + 0.1 * i as f32;
                let axis: Vec<Vec3> = (0..8)
                    .map(|r| Vec3::new(x, 0.03 * r as f32, 0.002 * r as f32))
                    .collect();
                let rows: Vec<Vec<Vec2>> = (0..8)
                    .map(|r| {
                        ring(
                            0.03 + 0.0004 * r as f32,
                            0.025 + 0.0003 * (i + r) as f32,
                            12,
                        )
                    })
                    .collect();
                SdfNode::loft_curves_capped(
                    axis,
                    Vec3::X,
                    rows,
                    [[0.01, 0.0, 0.0], [0.01, 0.0, 0.0]],
                )
                .expect("a curve loft")
                .rotate_euler(0.1 * i as f32, 0.2, 0.0)
            })
            .collect(),
    )
}

fn polar_lofts(n: usize) -> SdfNode {
    union_of(
        (0..n)
            .map(|i| {
                let x = -0.8 + 0.1 * i as f32;
                let axis: Vec<Vec3> = (0..6).map(|r| Vec3::new(x, 0.05 * r as f32, 0.5)).collect();
                let rows: Vec<Vec<f32>> = (0..6)
                    .map(|r| vec![0.04 + 0.002 * ((i + r) % 5) as f32, 0.003, 0.0, 0.0, 0.002])
                    .collect();
                SdfNode::loft_capped(axis, Vec3::Z, 2, rows, [0.02, 0.0]).expect("a polar loft")
            })
            .collect(),
    )
}

fn polygons(n: usize) -> SdfNode {
    union_of(
        (0..n as u32)
            .map(|k| {
                let (i, j) = ((k % 20) as f32, (k / 20) as f32);
                SdfNode::polygon_2d(star(46, 0.04 + 0.002 * (k % 7) as f32, 0.02), 0.03).translate(
                    i * 0.1,
                    j * 0.1,
                    0.0,
                )
            })
            .collect(),
    )
}

fn stacks(n: usize) -> SdfNode {
    union_of(
        (0..n)
            .map(|i| {
                let levels = vec![0.0, 0.1, 0.2, 0.3];
                let rings: Vec<Vec<Vec<Vec2>>> = levels
                    .iter()
                    .map(|_| vec![ring(0.05 + 0.002 * (i % 4) as f32, 0.04, 10)])
                    .collect();
                SdfNode::section_stack(levels, rings)
                    .expect("a section stack")
                    .translate(0.15 * i as f32, 0.0, 0.0)
            })
            .collect(),
    )
}

fn smooth_stacks(n: usize) -> SdfNode {
    union_of(
        (0..n)
            .map(|i| {
                let levels = vec![0.0, 0.1, 0.2, 0.3];
                let rings: Vec<Vec<Vec<Vec2>>> = levels
                    .iter()
                    .map(|_| vec![ring(0.05 + 0.002 * (i % 4) as f32, 0.04, 10)])
                    .collect();
                SdfNode::smooth_section_stack(levels, rings, 0.01, 0.1)
                    .expect("a smooth section stack")
                    .translate(0.15 * i as f32, 0.0, 0.0)
            })
            .collect(),
    )
}

type Build = fn(usize) -> SdfNode;

const KINDS: [(&str, Build); 5] = [
    ("curve lofts", curve_lofts),
    ("polar lofts", polar_lofts),
    ("polygons", polygons),
    ("section stacks", stacks),
    ("smooth section stacks", smooth_stacks),
];

/// The functions the module defines, and the call sites of those functions (each place a defined
/// name is followed by `(`, not counting its own definition)
fn call_sites(src: &str) -> (usize, usize) {
    let names: Vec<&str> = src
        .match_indices("fn ")
        .filter_map(|(at, _)| {
            let rest = &src[at + 3..];
            let end = rest.find('(')?;
            let name = rest[..end].trim();
            (!name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
                .then_some(name)
        })
        .collect();
    let bytes = src.as_bytes();
    let mut calls = 0;
    for name in &names {
        let pat = format!("{name}(");
        for (at, _) in src.match_indices(&pat) {
            let prev_ident =
                at > 0 && (bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
            let definition = src[..at].ends_with("fn ");
            if !prev_ident && !definition {
                calls += 1;
            }
        }
    }
    (names.len(), calls)
}

fn shader(node: &SdfNode) -> WgslShader {
    WgslShader::try_transpile_with_node_data(node, TranspileMode::Hardcoded).expect("transpile")
}

#[test]
fn the_call_sites_do_not_grow_with_the_nodes_of_a_kind() {
    let mut failures = Vec::new();
    for (kind, build) in KINDS {
        let (f4, c4) = call_sites(&shader(&build(4)).to_compute_shader());
        let (f16, c16) = call_sites(&shader(&build(16)).to_compute_shader());
        eprintln!("{kind}: 4 nodes {f4} functions {c4} call sites, 16 nodes {f16} functions {c16} call sites");
        assert!(c4 > 0 && c16 > 0, "{kind}: counted no call site");
        if c16 != c4 || f16 != f4 {
            failures.push(format!(
                "{kind}: {c4} -> {c16} call sites, {f4} -> {f16} functions"
            ));
        }
    }
    assert!(failures.is_empty(), "grows with the nodes: {failures:#?}");
}

#[test]
fn the_counter_sees_call_sites_that_grow() {
    // the counter's own teeth: one function called from N places counts N
    let src = |n: usize| {
        let calls: String = (0..n).map(|_| "    d = min(d, f(p));\n").collect();
        format!("fn f(p: vec3<f32>) -> f32 {{ return 0.0; }}\nfn sdf_eval(p: vec3<f32>) -> f32 {{\n    var d = 1.0;\n{calls}    return d;\n}}\n")
    };
    assert_eq!(call_sites(&src(3)), (2, 3));
    assert_eq!(call_sites(&src(9)), (2, 9));
}

/// Seconds from the shader's text to the first dispatch's values (pipeline creation and the first
/// read, where a driver that compiles lazily does its work)
fn build_and_first_dispatch(node: &SdfNode) -> Option<f64> {
    let s = shader(node);
    let t = Instant::now();
    let gpu = match GpuEvaluator::from_shader(&s) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("skipping: {e}");
            return None;
        }
    };
    let v = gpu
        .eval_batch(&[Vec3::ZERO, Vec3::splat(0.1)])
        .expect("first dispatch");
    assert_eq!(v.len(), 2);
    Some(t.elapsed().as_secs_f64())
}

#[test]
fn the_pipeline_time_does_not_grow_with_the_nodes_of_a_kind() {
    let mut failures = Vec::new();
    for (kind, build, small, large) in [
        ("curve lofts", curve_lofts as Build, 8, 32),
        ("polygons", polygons as Build, 100, 400),
    ] {
        let Some(a) = build_and_first_dispatch(&build(small)) else {
            return;
        };
        let Some(b) = build_and_first_dispatch(&build(large)) else {
            return;
        };
        eprintln!(
            "{kind}: {small} nodes {a:.2} s, {large} nodes {b:.2} s, ratio {:.2}",
            b / a
        );
        assert!(a > 0.0 && b > 0.0, "{kind}: timed nothing");
        if b > 2.0 * a {
            failures.push(format!("{kind}: {a:.2} s -> {b:.2} s"));
        }
    }
    assert!(
        failures.is_empty(),
        "the pipeline time grows with the nodes: {failures:#?}"
    );
}

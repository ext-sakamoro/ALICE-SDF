//! `update_params` writes parameter values and nothing else: it refuses a tree that differs in anything
//! the shader was compiled with, and a parameter-only update gives the new tree's values.
//!
//! oracle: the CPU evaluator on the new tree (an independent code path from the shader)
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;
use glam::Vec2;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// `eval_batch` against the CPU: the transpiled law agrees within this (see the agreement oracles)
const BUDGET: f32 = 2e-5;

fn ring(m: usize, r: f32) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(r * c, r * s)
        })
        .collect()
}

fn points() -> Vec<Vec3> {
    let mut s = 7u64;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..512)
        .map(|_| Vec3::new(r() - 0.5, r() * 1.2 - 0.3, r() - 0.5))
        .collect()
}

fn levels() -> Vec<f32> {
    (0..6).map(|i| 0.15 * i as f32).collect()
}

fn axis() -> Vec<Vec3> {
    (0..4)
        .map(|i| Vec3::new(0.0, 0.3 * i as f32, 0.0))
        .collect()
}

fn polygon(m: usize, r: f32) -> SdfNode {
    SdfNode::Polygon2D {
        vertices: ring(m, r),
        half_height: 0.2,
    }
}

fn curves(r: f32) -> SdfNode {
    SdfNode::loft_curves(
        axis(),
        Vec3::X,
        axis().iter().map(|_| ring(12, r)).collect(),
    )
    .expect("curve loft")
}

fn polar(r: f32) -> SdfNode {
    SdfNode::loft(
        axis(),
        Vec3::Z,
        1,
        axis().iter().map(|_| vec![r, 0.0, 0.0]).collect(),
    )
    .expect("polar loft")
}

fn section(r: f32) -> SdfNode {
    SdfNode::section_stack(
        levels(),
        levels().iter().map(|_| vec![ring(12, r)]).collect(),
    )
    .expect("section stack")
}

fn smooth(r: f32) -> SdfNode {
    SdfNode::smooth_section_stack(
        levels(),
        levels().iter().map(|_| vec![ring(12, r)]).collect(),
        0.01,
        0.1,
    )
    .expect("smooth stack")
}

fn evaluator(n: &SdfNode) -> Option<GpuEvaluator> {
    match GpuEvaluator::new_dynamic(n) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "no GPU: {e:?}"
            );
            eprintln!("skipping: {e:?}");
            None
        }
    }
}

fn worst(g: &GpuEvaluator, n: &SdfNode, p: &[Vec3]) -> f32 {
    let v = gpu_batches::eval_in_batches(g, p).expect("eval");
    p.iter()
        .zip(&v)
        .map(|(q, x)| (x - alice_sdf::eval::eval(n, *q)).abs())
        .fold(0.0, f32::max)
}

#[test]
fn a_tree_with_other_data_is_refused() {
    let p = points();
    let cases: Vec<(&str, SdfNode, SdfNode)> = vec![
        ("curve loft rows", curves(0.2), curves(0.3)),
        ("polar loft rows", polar(0.2), polar(0.3)),
        ("section stack loops", section(0.2), section(0.3)),
        ("smooth stack loops", smooth(0.2), smooth(0.3)),
        (
            "another form",
            SdfNode::sphere(0.2),
            SdfNode::box3d(0.2, 0.2, 0.2),
        ),
        (
            "a polygon of another size",
            polygon(12, 0.2),
            polygon(13, 0.2),
        ),
    ];
    let mut refused = 0;
    for (name, a, b) in cases {
        let Some(g) = evaluator(&a) else { return };
        let before = worst(&g, &a, &p);
        assert!(before <= BUDGET, "{name}: {before} before the update");
        let r = catch_unwind(AssertUnwindSafe(|| g.update_params(&b)));
        let e = r
            .err()
            .unwrap_or_else(|| panic!("{name}: the update was accepted"));
        let text = e
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| e.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            text.contains("more than parameter values"),
            "{name}: {text}"
        );
        // refused means unchanged: still the old tree
        let after = worst(&g, &a, &p);
        assert!(after <= BUDGET, "{name}: {after} after the refusal");
        refused += 1;
    }
    assert_eq!(refused, 6);
}

#[test]
fn a_parameter_update_gives_the_new_tree() {
    let p = points();
    let moved = |n: fn(f32) -> SdfNode| {
        (
            n(0.2).translate(0.1, 0.0, 0.0),
            n(0.2).translate(-0.05, 0.1, 0.02),
        )
    };
    let cases: Vec<(&str, SdfNode, SdfNode)> = vec![
        ("sphere radius", SdfNode::sphere(0.2), SdfNode::sphere(0.3)),
        ("polygon of 5 vertices", polygon(5, 0.2), polygon(5, 0.3)),
        ("polygon of 9 vertices", polygon(9, 0.2), polygon(9, 0.3)),
        ("polygon of 12 vertices", polygon(12, 0.2), polygon(12, 0.3)),
        ("polygon of 41 vertices", polygon(41, 0.2), polygon(41, 0.3)),
        (
            "translate over a curve loft",
            moved(curves).0,
            moved(curves).1,
        ),
        (
            "translate over a polar loft",
            moved(polar).0,
            moved(polar).1,
        ),
        (
            "translate over a section stack",
            moved(section).0,
            moved(section).1,
        ),
        (
            "translate over a smooth stack",
            moved(smooth).0,
            moved(smooth).1,
        ),
        (
            "rotate and scale",
            SdfNode::box3d(0.2, 0.1, 0.1)
                .rotate_euler(0.1, 0.2, 0.3)
                .scale(1.0),
            SdfNode::box3d(0.3, 0.1, 0.2)
                .rotate_euler(0.4, 0.1, 0.0)
                .scale(1.5),
        ),
        (
            "smooth union",
            SdfNode::sphere(0.2).smooth_union(SdfNode::sphere(0.1).translate(0.1, 0.0, 0.0), 0.05),
            SdfNode::sphere(0.25).smooth_union(SdfNode::sphere(0.15).translate(0.2, 0.0, 0.0), 0.1),
        ),
    ];
    for (name, a, b) in cases {
        let Some(g) = evaluator(&a) else { return };
        let difference = p
            .iter()
            .map(|q| (alice_sdf::eval::eval(&a, *q) - alice_sdf::eval::eval(&b, *q)).abs())
            .fold(0.0, f32::max);
        assert!(difference > 1e-3, "{name}: the two trees are the same");
        g.update_params(&b);
        let new = worst(&g, &b, &p);
        assert!(
            new <= BUDGET,
            "{name}: {new} from the new tree after the update"
        );
    }
}

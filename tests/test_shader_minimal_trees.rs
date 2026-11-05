//! The smallest trees each constructor accepts reach every shader dialect and evaluate as the CPU.
//!
//! Emit ranges and loop bounds come from the tree's sizes, so a site can be empty or a loop can have
//! no iteration only at the smallest sizes, which larger scenes never reach. Each family is swept
//! upward in one size at a time from where the constructor first accepts it: every tree must emit in
//! WGSL (constants, node data, checked), GLSL and HLSL (plain and checked); on a GPU, the checked
//! shader must agree with the CPU and no read may leave its proven range.
//!
//! Author: Moroya Sakamoto
#![cfg(all(feature = "gpu", feature = "glsl", feature = "hlsl"))]

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

use alice_sdf::compiled::{
    GlslShader, GlslTranspileMode, GpuEvaluator, HlslShader, HlslTranspileMode, TranspileMode,
    WgslShader,
};
use alice_sdf::prelude::*;
use glam::Vec2;

/// The sizes swept per family, enough to pass every smallest case
const SIZES: std::ops::RangeInclusive<usize> = 1..=7;

fn ring(m: usize, a: f32, b: f32) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * j as f32 / m as f32);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

fn levels(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| -0.4 + 0.8 * i as f32 / (n.max(2) - 1) as f32)
        .collect()
}

fn axis(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| Vec3::new(0.0, 0.8 * i as f32 / (n.max(2) - 1) as f32 - 0.4, 0.0))
        .collect()
}

/// (family, size, tree) of every size the family's constructor accepts
fn trees() -> Vec<(&'static str, usize, SdfNode)> {
    type Build = Box<dyn Fn(usize) -> Option<SdfNode>>;
    let families: Vec<(&'static str, Build)> = vec![
        (
            "smooth stack: levels",
            Box::new(|n| {
                SdfNode::smooth_section_stack(
                    levels(n),
                    vec![vec![ring(8, 0.2, 0.15)]; n],
                    0.01,
                    0.1,
                )
                .ok()
            }),
        ),
        (
            "smooth stack: loop points",
            Box::new(|m| {
                SdfNode::smooth_section_stack(
                    levels(4),
                    vec![vec![ring(m, 0.2, 0.15)]; 4],
                    0.01,
                    0.1,
                )
                .ok()
            }),
        ),
        (
            "section stack: levels",
            Box::new(|n| SdfNode::section_stack(levels(n), vec![vec![ring(8, 0.2, 0.15)]; n]).ok()),
        ),
        (
            "section stack: loop points",
            Box::new(|m| SdfNode::section_stack(levels(2), vec![vec![ring(m, 0.2, 0.15)]; 2]).ok()),
        ),
        (
            "section stack: loops",
            Box::new(|k| {
                let set: Vec<Vec<Vec2>> = (0..k)
                    .map(|i| {
                        ring(6, 0.05, 0.05)
                            .iter()
                            .map(|p| *p + Vec2::new(0.15 * i as f32, 0.0))
                            .collect()
                    })
                    .collect();
                SdfNode::section_stack(levels(2), vec![set; 2]).ok()
            }),
        ),
        (
            "polar loft: rows",
            Box::new(|n| SdfNode::loft(axis(n), Vec3::Z, 1, vec![vec![0.15, 0.01, 0.0]; n]).ok()),
        ),
        (
            "polar loft: harmonics",
            Box::new(|h| {
                let mut row = vec![0.005; 2 * (h - 1) + 1];
                row[0] = 0.15;
                SdfNode::loft(axis(2), Vec3::Z, (h - 1) as u32, vec![row; 2]).ok()
            }),
        ),
        (
            "capped polar loft: rows",
            Box::new(|n| {
                SdfNode::loft_capped(
                    axis(n),
                    Vec3::Z,
                    1,
                    vec![vec![0.15, 0.01, 0.0]; n],
                    [0.05, 0.05],
                )
                .ok()
            }),
        ),
        (
            "curve loft: rows",
            Box::new(|n| SdfNode::loft_curves(axis(n), Vec3::X, vec![ring(8, 0.15, 0.1); n]).ok()),
        ),
        (
            "curve loft: section points",
            Box::new(|m| SdfNode::loft_curves(axis(2), Vec3::X, vec![ring(m, 0.15, 0.1); 2]).ok()),
        ),
        (
            "capped curve loft: rows",
            Box::new(|n| {
                SdfNode::loft_curves_capped(
                    axis(n),
                    Vec3::X,
                    vec![ring(8, 0.15, 0.1); n],
                    [[0.05, 0.0, 0.0]; 2],
                )
                .ok()
            }),
        ),
    ];
    let mut out = Vec::new();
    for (family, build) in &families {
        let built: Vec<(usize, SdfNode)> = SIZES.filter_map(|s| build(s).map(|t| (s, t))).collect();
        assert!(
            built.len() >= 2,
            "{family}: {} sizes built, the sweep is too short",
            built.len()
        );
        assert!(built[0].0 >= 1, "{family}");
        for (s, t) in built {
            out.push((*family, s, t));
        }
    }
    out
}

#[test]
fn the_smallest_trees_emit_in_every_dialect() {
    let list = trees();
    for (family, size, node) in &list {
        let at = format!("{family} {size}");
        WgslShader::try_transpile(node, TranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: WGSL: {e}"));
        WgslShader::try_transpile_with_node_data(node, TranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: WGSL node data: {e}"));
        WgslShader::try_transpile_with_node_data_checked(node, TranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: WGSL checked: {e}"));
        GlslShader::try_transpile(node, GlslTranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: GLSL: {e}"));
        HlslShader::try_transpile(node, HlslTranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: HLSL: {e}"));
        HlslShader::try_transpile_checked(node, HlslTranspileMode::Hardcoded)
            .unwrap_or_else(|e| panic!("{at}: HLSL checked: {e}"));
    }
    println!("{} trees emitted", list.len());
    assert!(list.len() >= 30, "only {} trees", list.len());
}

/// The four-level smooth stack has no span search: its one span is every height
#[test]
fn the_smallest_smooth_stack_has_no_search_loop_and_a_larger_one_does() {
    let build = |n| {
        SdfNode::smooth_section_stack(levels(n), vec![vec![ring(8, 0.2, 0.15)]; n], 0.01, 0.1)
            .expect("a stack")
    };
    let search = |n| {
        WgslShader::transpile(&build(n), TranspileMode::Hardcoded)
            .source
            .contains("k < 2")
    };
    assert!(!search(4));
    assert!(search(5));
}

#[test]
fn the_smallest_trees_evaluate_as_the_cpu_on_the_gpu() {
    let mut s: u64 = 0x5EED_0F_5A11;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let pts: Vec<Vec3> = (0..256)
        .map(|_| Vec3::new(r() * 1.2 - 0.6, r() * 1.4 - 0.7, r() * 1.2 - 0.6))
        .collect();
    let mut checked = 0usize;
    for (family, size, node) in trees() {
        let at = format!("{family} {size}");
        let gpu = match GpuEvaluator::new_checked(&node) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "{at}: no GPU: {e}"
                );
                eprintln!("skipping: no GPU ({e})");
                return;
            }
        };
        let got = gpu_batches::eval_in_batches(&gpu, &pts).expect("gpu eval");
        assert_eq!(
            gpu.read_hits().expect("the counter"),
            Some(0),
            "{at}: a read left its proven range"
        );
        assert_eq!(got.len(), pts.len());
        for (p, g) in pts.iter().zip(&got) {
            let c = eval(&node, *p);
            assert!(
                (c - g).abs() <= 2e-5 * c.abs().max(1.0),
                "{at}: GPU {g} vs CPU {c} at {p:?}"
            );
            checked += 1;
        }
    }
    println!("{checked} values compared");
    assert!(checked >= 30 * pts.len(), "only {checked} values compared");
}

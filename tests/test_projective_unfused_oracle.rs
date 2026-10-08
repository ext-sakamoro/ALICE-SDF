//! `ProjectiveTransform` evaluates `M⁻¹·[p, 1]` with every `a * b + c` rounded
//! twice, never fused.
//!
//! oracle: the homogeneous product written out below with plain `*` and `+`
//! (Rust never contracts them into an FMA), divided by `w`, and the child
//! scaled by `min(|1/w|, lipschitz_bound)` — the law `src/eval/mod.rs` states.
//! It does not call `transforms::projective`, so a fused or reordered
//! implementation disagrees with it.
//!
//! The corpus node in `tests/common/corpus.rs` uses the identity matrix, where
//! every product is exact and a fused and an unfused evaluation give the same
//! bits; that node cannot tell the two apart. The matrix here has entries
//! with full 24-bit mantissas, and the test also asserts that a fused
//! evaluation of the same product differs at some of the points, so the
//! scene keeps its ability to see a fused implementation.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_simd, eval_compiled_bvh, CompiledSdf, CompiledSdfBvh,
};
use alice_sdf::prelude::*;

/// A perspective-like inverse matrix (column-major), every entry inexact in
/// binary, with `w` staying in `[0.8, 1.3]` over the sample box.
const INV_M: [f32; 16] = [
    1.137_469_3,
    0.071_329_14,
    -0.043_917_6,
    0.031_415_93,
    -0.058_213_77,
    0.913_782_4,
    0.087_123_51,
    -0.027_182_82,
    0.019_283_74,
    -0.064_738_29,
    1.071_828_2,
    0.044_721_36,
    0.123_456_79,
    -0.098_765_43,
    0.056_789_01,
    1.012_345_7,
];
const BOUND: f32 = 1.25;
const RADIUS: f32 = 0.7;

/// `M⁻¹·[p, 1]` with every product and sum rounded on its own
fn reference_unfused(p: Vec3, m: &[f32; 16]) -> (Vec3, f32) {
    let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    let inv_w = 1.0 / w;
    let x = (m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12]) * inv_w;
    let y = (m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13]) * inv_w;
    let z = (m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]) * inv_w;
    (Vec3::new(x, y, z), inv_w.abs())
}

/// The same product with fused multiply-adds, used only to show the scene
/// can tell the two apart
fn reference_fused(p: Vec3, m: &[f32; 16]) -> Vec3 {
    let w = m[11].mul_add(p.z, m[3].mul_add(p.x, m[7] * p.y)) + m[15];
    let inv_w = 1.0 / w;
    Vec3::new(
        (m[8].mul_add(p.z, m[0].mul_add(p.x, m[4] * p.y)) + m[12]) * inv_w,
        (m[9].mul_add(p.z, m[1].mul_add(p.x, m[5] * p.y)) + m[13]) * inv_w,
        (m[10].mul_add(p.z, m[2].mul_add(p.x, m[6] * p.y)) + m[14]) * inv_w,
    )
}

/// 1000 LCG points in a ±2 box
fn points() -> Vec<Vec3> {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32) * 4.0 - 2.0
    };
    (0..1000)
        .map(|_| Vec3::new(next(), next(), next()))
        .collect()
}

fn scene() -> SdfNode {
    SdfNode::sphere(RADIUS).projective_transform(INV_M, BOUND)
}

fn expected(p: Vec3) -> f32 {
    let (q, correction) = reference_unfused(p, &INV_M);
    eval(&SdfNode::sphere(RADIUS), q) * correction.min(BOUND)
}

#[test]
fn scene_distinguishes_fused_from_unfused() {
    let differ = points()
        .into_iter()
        .filter(|&p| reference_fused(p, &INV_M) != reference_unfused(p, &INV_M).0)
        .count();
    assert!(
        differ >= 100,
        "only {differ} of 1000 points tell a fused product from an unfused one"
    );
}

/// Every point whose result is not bit-identical to the unfused reference
fn mismatches(results: &[f32]) -> Vec<String> {
    points()
        .into_iter()
        .zip(results)
        .filter_map(|(p, &got)| {
            let want = expected(p);
            (got.to_bits() != want.to_bits()).then(|| format!("{p:?}: got {got:e} want {want:e}"))
        })
        .collect()
}

fn assert_matches(path: &str, results: &[f32]) {
    assert_eq!(results.len(), 1000, "{path}: wrong number of results");
    let bad = mismatches(results);
    assert!(
        bad.is_empty(),
        "{path}: {} of 1000 points differ from the unfused reference, first: {}",
        bad.len(),
        bad.first().map_or("", String::as_str)
    );
}

#[test]
fn tree_evaluator_matches_unfused_reference() {
    let node = scene();
    let got: Vec<f32> = points().into_iter().map(|p| eval(&node, p)).collect();
    assert_matches("tree", &got);
}

#[test]
fn compiled_scalar_matches_unfused_reference() {
    let compiled = CompiledSdf::compile(&scene());
    let got: Vec<f32> = points()
        .into_iter()
        .map(|p| eval_compiled(&compiled, p))
        .collect();
    assert_matches("compiled scalar", &got);
}

/// The 8-lane evaluator: 1000 points are 125 full chunks, so every point
/// goes through `f32x8` (no scalar tail)
#[test]
fn compiled_simd_matches_unfused_reference() {
    let compiled = CompiledSdf::compile(&scene());
    let got = eval_compiled_batch_simd(&compiled, &points());
    assert_matches("compiled f32x8", &got);
}

#[test]
fn compiled_bvh_matches_unfused_reference() {
    let bvh = CompiledSdfBvh::compile(&scene());
    let got: Vec<f32> = points()
        .into_iter()
        .map(|p| eval_compiled_bvh(&bvh, p))
        .collect();
    assert_matches("compiled BVH", &got);
}

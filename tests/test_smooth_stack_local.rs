//! The smooth section stack's local bound: the knots spread a large bound only over the reach, the
//! slope claim bounds the field's measured slope (and is more than 1 where the knots change), and
//! the interval evaluator and the Lipschitz report use the claim
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use glam::Vec2;

const REACH: f32 = 0.03;

fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let a = std::f32::consts::TAU * j as f32 / m as f32;
            let (s, co) = alice_det_math::sin_cos(a);
            c + Vec2::new(co, s) * r
        })
        .collect()
}

/// A column whose section jumps sideways between two close levels, so that one interval's bound is
/// far larger than the others'
fn jumping() -> SdfNode {
    let levels: Vec<f32> = (0..24).map(|i| 0.04 * i as f32).collect();
    let loops = (0..24)
        .map(|i| {
            let x = if i < 12 { 0.0 } else { 0.12 };
            vec![circle(0.2, Vec2::new(x, 0.0), 10)]
        })
        .collect();
    SdfNode::smooth_section_stack_local(levels, loops, REACH).expect("a valid stack")
}

fn stack(node: &SdfNode) -> &alice_sdf::primitives::SmoothSectionStack {
    match node {
        SdfNode::SmoothSectionStack(st) => st,
        _ => panic!("a stack"),
    }
}

#[test]
fn neighbouring_knots_differ_by_the_stated_limit_and_the_jump_spreads() {
    let node = jumping();
    let st = stack(&node);
    let k = st.knots();
    let (lo, hi) = k
        .iter()
        .fold((f32::INFINITY, 0.0f32), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(hi > 2.0 * lo, "the bound must vary (knots {lo}..{hi})");
    let mut limited = 0;
    for i in 0..k.len() - 1 {
        let d = st.levels[i + 1] - st.levels[i];
        let lim = k[i].min(k[i + 1]) * d / (REACH / 3.0);
        assert!(
            (k[i + 1] - k[i]).abs() <= lim * (1.0 + 1e-5),
            "knots {i}, {}: {} past {lim}",
            i + 1,
            (k[i + 1] - k[i]).abs()
        );
        // the raw window takes the large bound or not: a step wider than the limit shows the limit acted
        if (k[i + 1] - k[i]).abs() > 0.99 * lim {
            limited += 1;
        }
    }
    assert!(limited > 0, "no pair at the limit: the limit is not tested");
}

#[test]
fn the_slope_claim_bounds_the_measured_slope_and_is_needed() {
    let node = jumping();
    let claim = stack(&node).slope_claim();
    assert!(claim > 1.0, "the knots vary, so the claim is above 1");
    let mut s: u64 = 0x10CA_1B0D;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let (mut worst, mut n) = (0.0f32, 0u32);
    for _ in 0..60_000 {
        let p = Vec3::new(r() * 0.8 - 0.3, r() * 0.9, r() * 0.6 - 0.3);
        let a = alice_sdf::eval::eval(&node, p);
        // the clamp's plateau says nothing about the slope
        if a.abs() >= REACH {
            continue;
        }
        let dir = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize_or_zero();
        let h = 1e-3;
        let b = alice_sdf::eval::eval(&node, p + dir * h);
        worst = worst.max((b - a).abs() / h);
        n += 1;
    }
    assert!(n > 5_000, "only {n} points inside the reach");
    assert!(
        worst <= claim * (1.0 + 1e-2),
        "measured slope {worst} past the claim {claim}"
    );
}

#[test]
fn the_lipschitz_report_and_the_interval_use_the_claim() {
    let node = jumping();
    let claim = stack(&node).slope_claim();
    assert!(claim > 1.0);
    assert_eq!(alice_sdf::interval::eval_lipschitz(&node), claim);
    // at a point the enclosure's half width grows with the slope claim
    let c = Vec3::new(0.5, 0.45, 0.0);
    let h = 0.01;
    let iv = alice_sdf::interval::eval_interval(
        &node,
        alice_sdf::interval::Vec3Interval::from_bounds(c - Vec3::splat(h), c + Vec3::splat(h)),
    );
    let width = iv.hi - iv.lo;
    let plain = 2.0 * Vec3::splat(h).length();
    assert!(
        width >= plain * claim * 0.999,
        "width {width} for slope {claim}, unit slope {plain}"
    );
}

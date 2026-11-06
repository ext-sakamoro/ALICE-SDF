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
    jumping_at(0.04, 24)
}

/// [`jumping`] with `n` levels `dy` apart
fn jumping_at(dy: f32, n: usize) -> SdfNode {
    let levels: Vec<f32> = (0..n).map(|i| dy * i as f32).collect();
    let loops = (0..n)
        .map(|i| {
            let x = if i < n / 2 { 0.0 } else { 0.12 };
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

/// The binary form is positional: a stack with and without a reach comes back the same, with a node
/// stored after it
#[test]
fn both_bounds_survive_the_binary_form() {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let levels: Vec<f32> = (0..8).map(|i| 0.05 * i as f32).collect();
    let loops = |_: usize| (0..8).map(|_| vec![circle(0.2, Vec2::ZERO, 10)]).collect::<Vec<_>>();
    let global = SdfNode::smooth_section_stack(levels.clone(), loops(0), 0.01, 0.1).expect("global");
    let local = SdfNode::smooth_section_stack_local(levels, loops(0), REACH).expect("local");
    for (name, node) in [("global", global), ("local", local)] {
        let tree = SdfTree::new(node.union(SdfNode::sphere(0.25)));
        let path = dir.join(format!("stack_{name}.asdf"));
        alice_sdf::io::save_asdf(&tree, &path).expect("save");
        let back = alice_sdf::io::load_asdf(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        for i in 0..200 {
            let t = i as f32 * 0.01;
            let p = Vec3::new(t - 1.0, t * 0.4, 0.1);
            let (a, b) = (alice_sdf::eval::eval(&tree.root, p), alice_sdf::eval::eval(&back.root, p));
            assert_eq!(a.to_bits(), b.to_bits(), "{name} at {p}");
        }
    }
}

/// Stepping by the value over the slope claim against a fixed small step: the same rays hit, and the
/// stepping never lands past the first crossing. Independent of the bound: the reference reads only
/// the sign of the field
#[test]
fn stepping_agrees_with_a_fixed_small_step() {
    for (dy, n) in [(0.04, 24), (0.008, 48), (0.002, 160)] {
        let node = jumping_at(dy, n);
        let claim = stack(&node).slope_claim();
        let top = dy * (n as f32 - 1.0);
        let h = 2e-4f32;
        let (mut hits, mut later) = (0, 0);
        for iy in 0..40 {
            for iz in 0..20 {
                let o = Vec3::new(-0.6, top * (iy as f32 + 0.5) / 40.0, -0.28 + 0.028 * iz as f32);
                let dir = Vec3::X;
                let mut rt = None;
                let mut t = 0.0f32;
                while t < 1.4 {
                    if alice_sdf::eval::eval(&node, o + dir * t) <= 0.0 {
                        rt = Some(t);
                        break;
                    }
                    t += h;
                }
                let mut st = None;
                let mut t = 0.0f32;
                for _ in 0..4096 {
                    if t >= 1.4 {
                        break;
                    }
                    let d = alice_sdf::eval::eval(&node, o + dir * t);
                    if d < 1e-4 {
                        st = Some(t);
                        break;
                    }
                    t += d / claim;
                }
                if let Some(a) = rt {
                    hits += 1;
                    let b = st.unwrap_or_else(|| panic!("dy {dy}: the stepping missed a crossing at {a}, from {o}"));
                    if b > a + 2.0 * h {
                        later += 1;
                    }
                }
            }
        }
        assert!(hits > 0, "dy {dy}: no ray hit");
        assert_eq!(later, 0, "dy {dy}: {later} of {hits} rays landed past the crossing");
    }
}

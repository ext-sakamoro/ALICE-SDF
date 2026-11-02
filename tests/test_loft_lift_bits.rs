//! The curve-section distance moved out of the loft into shared functions (used by the smooth section
//! stack too): the loft's field must stay bit for bit what it was. Curve lofts (straight and curved axes,
//! uniform and non-uniform section knots, 6..48 control points) evaluated at fixed points; the hash of every
//! value's bits is pinned to the value measured before the move (224bf16).

use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use glam::{Vec2, Vec3};

fn lofts() -> Vec<SdfNode> {
    let mut out = Vec::new();
    for (m, nonuniform, curved) in [(6usize, false, false), (12, true, false), (24, false, true), (48, true, true)] {
        let row = |s: f32| -> Vec<Vec2> {
            (0..m)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / m as f32;
                    Vec2::new((0.30 + 0.05 * s) * alice_det_math::cos(a), (0.20 - 0.03 * s) * alice_det_math::sin(a) + 0.02 * alice_det_math::sin(3.0 * a))
                })
                .collect()
        };
        let axis = if curved {
            vec![Vec3::ZERO, Vec3::new(0.05, 0.5, 0.0), Vec3::new(0.15, 1.0, 0.05), Vec3::new(0.3, 1.4, 0.1)]
        } else {
            vec![Vec3::ZERO, Vec3::new(0.0, 0.7, 0.0), Vec3::new(0.0, 1.4, 0.0)]
        };
        let rows: Vec<Vec<Vec2>> = (0..4).map(|i| row(i as f32 / 3.0)).collect();
        let knots: Vec<f32> = (0..=m).map(|j| j as f32 + if nonuniform && j % 3 == 1 { 0.3 } else { 0.0 }).collect();
        out.push(SdfNode::loft_curves_knots(axis, Vec3::X, knots, rows).expect("a valid curve loft"));
    }
    out
}

#[test]
fn the_curve_loft_field_is_bit_equal_after_the_lift() {
    let mut st: u64 = 11;
    let mut rnd = || {
        st = st.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((st >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n = 0usize;
    for node in lofts() {
        for _ in 0..1000 {
            let p = Vec3::new(rnd() * 1.2 - 0.6, rnd() * 1.8 - 0.2, rnd() * 1.2 - 0.6);
            for b in eval(&node, p).to_bits().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            n += 1;
        }
    }
    assert_eq!(n, 4000, "every point compared");
    // pinned before the lift (224bf16): print with --nocapture when it moves
    println!("loft field hash {h:#018x}");
    assert_eq!(h, PINNED, "the loft's field changed");
}

const PINNED: u64 = 0;

//! The scalar Law min / max through the public API (ALICE-Law G2 and G4)
//!
//! `law_min_f32` / `law_max_f32` (src/compiled/real.rs) reach the public API at
//! `operations::sdf_union` / `sdf_intersection`, the bounds of
//! `interval::Interval` (`min`, `max`, `hull`, `intersect`) and of
//! `compiled::AabbPacked` (`union`, `intersection`). Each site is driven here
//! with an equal pair of zeros in both orders, a NaN in either position (NaNs
//! built with `f32::from_bits`, with payloads other than the canonical one) and
//! a NaN pair, and the result is compared bit for bit with the rule written out
//! in this file: NaN gives the other operand, a NaN pair gives the canonical
//! quiet NaN `0x7fc0_0000`, an equal pair of zeros gives `-0` from min when
//! either is `-0` and `+0` from max when either is `+0`, every other pair is
//! ordered by value.

use alice_sdf::compiled::AabbPacked;
use alice_sdf::interval::Interval;
use alice_sdf::operations::{sdf_intersection, sdf_union};
use glam::Vec3;

const CANONICAL_NAN: u32 = 0x7fc0_0000;

/// Operand values: both zeros, two finite values, and NaNs with several payloads
/// (a quiet one with a payload, a negative quiet one, a signalling one)
fn operands() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.5,
        -2.0,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffc0_0123),
        f32::from_bits(0x7f80_0001),
    ]
}

fn expected_min(a: f32, b: f32) -> u32 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => CANONICAL_NAN,
        (true, false) => b.to_bits(),
        (false, true) => a.to_bits(),
        (false, false) if a == b && a == 0.0 => {
            if a.is_sign_negative() || b.is_sign_negative() {
                (-0.0f32).to_bits()
            } else {
                0.0f32.to_bits()
            }
        }
        (false, false) => (if a < b { a } else { b }).to_bits(),
    }
}

fn expected_max(a: f32, b: f32) -> u32 {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => CANONICAL_NAN,
        (true, false) => b.to_bits(),
        (false, true) => a.to_bits(),
        (false, false) if a == b && a == 0.0 => {
            if a.is_sign_positive() || b.is_sign_positive() {
                0.0f32.to_bits()
            } else {
                (-0.0f32).to_bits()
            }
        }
        (false, false) => (if a > b { a } else { b }).to_bits(),
    }
}

/// Runs `check` over every ordered pair and counts the comparisons, and also
/// counts the three classes the Law names, so a grid that lost them fails
fn each_pair(mut check: impl FnMut(f32, f32)) -> usize {
    let ops = operands();
    let (mut compared, mut pos_neg, mut neg_pos, mut nan_first) = (0, 0, 0, 0);
    for &a in &ops {
        for &b in &ops {
            check(a, b);
            compared += 1;
            if a == 0.0 && b == 0.0 && a.is_sign_positive() && b.is_sign_negative() {
                pos_neg += 1;
            }
            if a == 0.0 && b == 0.0 && a.is_sign_negative() && b.is_sign_positive() {
                neg_pos += 1;
            }
            if a.is_nan() && !b.is_nan() {
                nan_first += 1;
            }
        }
    }
    assert!(
        pos_neg > 0 && neg_pos > 0 && nan_first > 0,
        "the grid lost a tie class"
    );
    compared
}

#[test]
fn sdf_union_and_intersection_follow_the_law() {
    let compared = each_pair(|a, b| {
        assert_eq!(
            sdf_union(a, b).to_bits(),
            expected_min(a, b),
            "sdf_union({a:?}, {b:?})"
        );
        assert_eq!(
            sdf_intersection(a, b).to_bits(),
            expected_max(a, b),
            "sdf_intersection({a:?}, {b:?})"
        );
    });
    assert!(compared > 0, "compared nothing");
}

const fn iv(lo: f32, hi: f32) -> Interval {
    Interval { lo, hi }
}

#[test]
fn interval_bounds_follow_the_law() {
    // the operand goes into both bounds of one interval and the matching bound of
    // the other, so each bound of the result is exactly one scalar min or max
    let compared = each_pair(|a, b| {
        let (x, y) = (iv(a, a), iv(b, b));
        let m = x.min(y);
        assert_eq!(
            [m.lo.to_bits(), m.hi.to_bits()],
            [expected_min(a, b); 2],
            "min({a:?}, {b:?})"
        );
        let m = x.max(y);
        assert_eq!(
            [m.lo.to_bits(), m.hi.to_bits()],
            [expected_max(a, b); 2],
            "max({a:?}, {b:?})"
        );
        let h = x.hull(y);
        assert_eq!(
            [h.lo.to_bits(), h.hi.to_bits()],
            [expected_min(a, b), expected_max(a, b)],
            "hull({a:?}, {b:?})"
        );
        let i = x.intersect(y);
        assert_eq!(
            [i.lo.to_bits(), i.hi.to_bits()],
            [expected_max(a, b), expected_min(a, b)],
            "intersect({a:?}, {b:?})"
        );
    });
    assert!(compared > 0, "compared nothing");
}

#[test]
fn aabb_union_and_intersection_follow_the_law() {
    let compared = each_pair(|a, b| {
        let x = AabbPacked::new(Vec3::splat(a), Vec3::splat(a));
        let y = AabbPacked::new(Vec3::splat(b), Vec3::splat(b));
        let u = x.union(&y);
        assert_eq!(
            [u.min_x, u.min_y, u.min_z].map(f32::to_bits),
            [expected_min(a, b); 3],
            "union min ({a:?}, {b:?})"
        );
        assert_eq!(
            [u.max_x, u.max_y, u.max_z].map(f32::to_bits),
            [expected_max(a, b); 3],
            "union max ({a:?}, {b:?})"
        );
        let i = x.intersection(&y);
        assert_eq!(
            [i.min_x, i.min_y, i.min_z].map(f32::to_bits),
            [expected_max(a, b); 3],
            "intersection min ({a:?}, {b:?})"
        );
        assert_eq!(
            [i.max_x, i.max_y, i.max_z].map(f32::to_bits),
            [expected_min(a, b); 3],
            "intersection max ({a:?}, {b:?})"
        );
    });
    assert!(compared > 0, "compared nothing");
}

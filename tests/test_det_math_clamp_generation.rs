//! alice-det-math 0.5.2 at 39507d1b changed `metric::clamp` (and only it, of the functions this
//! crate calls): it is now `min(max(x, low), high)` with the G2 max / min, where 0.5.1 and 0.5.0
//! compared (`clamp(-0, +0, hi)` kept `-0`). This crate reaches the clamp only through
//! `metric::smoothstep`, whose clamp has the bounds `0` and `1`. This test shows that
//! `smoothstep` gives the same bits under both clamps for every input of a lattice of special
//! values, so the distances this crate computes did not move: the reason the identifiers of the
//! earlier generations stay in `PREVIOUS_SDF_SEMANTICS_IDS` (`tests/sdf_semantics_id.rs` names
//! this test for each of them).
//!
//! oracle: `clamp_0_5_1` is alice-det-math 0.5.1's clamp, copied from its source (6384856), and
//! `smoothstep_0_5_1` is `smoothstep` built on it the way alice-det-math writes it; neither calls
//! alice-det-math's clamp.

/// alice-det-math 0.5.1's `metric::clamp`
fn clamp_0_5_1(x: f32, low: f32, high: f32) -> f32 {
    if x.is_nan() || low.is_nan() || high.is_nan() {
        return f32::NAN;
    }
    let y = if x > high { high } else { x };
    if y < low {
        low
    } else {
        y
    }
}

/// `metric::smoothstep` with alice-det-math 0.5.1's clamp
fn smoothstep_0_5_1(edge0: f32, edge1: f32, x: f32) -> f32 {
    if x.is_nan() || edge0.is_nan() || edge1.is_nan() {
        return f32::NAN;
    }
    if edge1 <= edge0 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = clamp_0_5_1((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    let r = t * t * (3.0 - 2.0 * t);
    if r.is_nan() {
        f32::NAN
    } else {
        r
    }
}

fn lattice() -> Vec<f32> {
    let mut v = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        2.0,
        1e-30,
        -1e-30,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0001),
        0.25,
        0.75,
        3.0,
    ];
    for i in -8..=8 {
        v.push(i as f32 / 8.0);
    }
    v
}

#[test]
fn smoothstep_gives_the_same_bits_under_both_clamps() {
    let xs = lattice();
    let mut compared = 0usize;
    for &e0 in &xs {
        for &e1 in &xs {
            for &x in &xs {
                let new = alice_det_math::metric::smoothstep(e0, e1, x);
                let old = smoothstep_0_5_1(e0, e1, x);
                assert_eq!(
                    new.to_bits(),
                    old.to_bits(),
                    "smoothstep({e0:?}, {e1:?}, {x:?}): {new:?} now, {old:?} with the 0.5.1 clamp"
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, xs.len().pow(3));
    assert!(compared > 0, "compared nothing");
}

/// The lattice reaches the one input the two clamps treat differently (`-0` against the `+0`
/// bound, which `smoothstep` gives its clamp when `x` is `-0` and `edge0` is `+0`), so the
/// comparison above is not vacuous
#[test]
fn the_lattice_reaches_where_the_two_clamps_differ() {
    let t = -0.0f32 - 0.0f32;
    assert_eq!(t.to_bits(), (-0.0f32).to_bits());
    let new = alice_det_math::metric::clamp(t, 0.0, 1.0);
    let old = clamp_0_5_1(t, 0.0, 1.0);
    assert_ne!(
        new.to_bits(),
        old.to_bits(),
        "the two clamps agree at -0: the comparison has no tooth"
    );
    assert!(lattice().iter().any(|v| v.to_bits() == (-0.0f32).to_bits()));
    assert!(lattice().iter().any(|v| v.to_bits() == 0.0f32.to_bits()));
}

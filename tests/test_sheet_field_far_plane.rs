//! A plane seen nearly head-on from far away: the march's step lands within an f32 rounding of the surface, so the
//! sign of the side channel at the floor step's start depends on how the point is rounded. The march samples the step's
//! ends at the f32 point `o + d t`; the cell walk builds its cubic from the same point in f64. At t = 600 an ulp of t
//! is 6e-5, so the two can disagree on the start's sign, and the crossing then lies in the gap between the two
//! roundings: no root inside the walk's interval exists for it to find. A crossing exists when the step's ends have
//! opposite signs (the march's own values); the walk only locates it.
//!
//! Oracle: the fixed-step reference (`RootMode::FixedFloor`, every floor step re-walked in 20 sub-steps) on the same
//! field, and the plane's analytic crossing. Per ray the default march must decide (never `Undecided`), hit where the
//! reference hits, and agree by `SheetField::hit_agrees_with_reference` (`|t - t_ref| <= REFERENCE_T_TOL`, or `|s|`
//! within `JUMP_S` all along between the two). A second test gives that check its teeth: a hit through the near surface
//! is refused, and a pair inside one stretch where `|s| <= JUMP_S` is accepted.
//!
//! The scene does test the claim: every ray hits under the reference, within the quantiser's bound of the plane. Measured
//! before this test: without the end-sign rule 84 of 9216 rays miss; with a bounded dip search as the root finder,
//! 108 are undecided.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{
    MarchResult, RootMode, SheetField, ValidRule, BRICK, JUMP_S, Q_MAX, REFERENCE_T_TOL,
};
use glam::Vec3;

const H: f32 = 0.27;
const DIST: f32 = 600.0;
const N: usize = 96;

/// How far the field's zero can sit from the plane: each brick stores its nodes as bytes over its own value range,
/// which for `s = z - 0.5` is the brick's height `BRICK * H`, so a decoded node is within half a code; at the decoded
/// zero `z - 0.5 = -error` (plus 1e-3 for the f32 encode).
fn decode_bound() -> f32 {
    0.5 * (BRICK as f32 * H) / Q_MAX * (1.0 + 1e-3)
}

fn plane() -> SheetField {
    let lo = Vec3::new(-2.0 + 0.013, -2.0 + 0.021, 0.5 - 2.0 + 0.007);
    let nb = (4.0 / (8.0 * H)).ceil() as u32;
    let lv = level_from_fn(
        lo,
        H,
        [nb, nb, nb],
        |_c, _e| true,
        |p| vec![p.z - 0.5, -1.0],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// (origin, direction): aimed at points of the plane z = 0.5, tilted up to 0.05 rad from head-on in varied azimuths
fn rays() -> Vec<(Vec3, Vec3)> {
    let mut out = Vec::with_capacity(N * N);
    for i in 0..N {
        for j in 0..N {
            let p = Vec3::new(
                -1.2 + 2.4 * i as f32 / N as f32 + 0.000_037 * j as f32,
                -1.2 + 2.4 * j as f32 / N as f32 + 0.000_29 * i as f32,
                0.5,
            );
            let tilt = 0.05 * (0.2 + 0.8 * ((i * 7 + j * 13) % 97) as f32 / 97.0);
            let az = (i * 31 + j * 17) as f32 * 0.37;
            let d = Vec3::new(tilt * az.cos(), tilt * az.sin(), -1.0).normalize();
            out.push((p - d * DIST, d));
        }
    }
    out
}

#[test]
fn a_far_plane_seen_head_on_is_hit_where_the_fixed_step_reference_hits() {
    let f = plane();
    let mut fref = f.clone();
    fref.set_root_mode(RootMode::FixedFloor);
    let tmax = DIST + 3.0;
    let (mut hits, mut far, mut undecided, mut misses) = (0usize, 0usize, 0usize, 0usize);
    for (o, d) in rays() {
        let MarchResult::Hit(hr) = fref.march_decided(o, d, tmax) else {
            panic!("the reference does not hit at {o} {d}")
        };
        let off = (o + d * hr.t).z - 0.5;
        assert!(
            off.abs() <= decode_bound(),
            "reference {} is {off} off the plane",
            hr.t
        );
        match f.march_decided(o, d, tmax) {
            MarchResult::Hit(h) => {
                hits += 1;
                if !f.hit_agrees_with_reference(h.sheet, o, d, h.t, hr.t) {
                    far += 1;
                }
            }
            MarchResult::Undecided => undecided += 1,
            MarchResult::Miss => misses += 1,
        }
    }
    assert_eq!(
        (misses, undecided, far),
        (0, 0, 0),
        "of {} rays: {misses} miss, {undecided} undecided, {far} hit away from the reference ({hits} hit)",
        N * N
    );
}

/// The agreement check's two parts, each with a case it must decide the other way when broken. Oracle: a slab of
/// thickness 4 (faces z = 2.5 and z = -1.5, stored `s = |z - 0.5| - 2`) and a plane seen at a grazing slope, by their
/// geometry.
/// - a hit on the far face of the slab (`t` 4 / cos past the near one) is not the near crossing: `|s|` reaches 2
///   between (refused; a check that accepted every pair would accept it);
/// - two points 0.3 apart along a ray that grazes a plane at slope 0.01 (`|s| <= 0.0015` all along) are the same
///   crossing up to the field's quantisation (accepted; a check with only the `t` part would refuse it);
/// - `t` within `REFERENCE_T_TOL` is accepted, just past it with `|s|` large between is refused.
#[test]
fn the_reference_agreement_check_refuses_a_hit_through_the_near_surface_and_accepts_one_stretch() {
    let lo = Vec3::new(-2.0 + 0.013, -2.0 + 0.021, -3.5 + 0.007);
    let nb = (8.0 / (8.0 * H)).ceil() as u32;
    let lv = level_from_fn(
        lo,
        H,
        [nb, nb, nb],
        |_c, _e| true,
        |p| vec![(p.z - 0.5).abs() - 2.0, -1.0],
    )
    .unwrap();
    let slab = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
    let d = Vec3::new(0.05, 0.03, -1.0).normalize();
    let o = Vec3::new(0.1, 0.2, 2.5) - d * 10.0;
    // near face z = 2.5 at t 10, far face z = -1.5 four units below
    let (near, far) = (10.0, 10.0 + 4.0 / -d.z);
    assert!(
        !slab.hit_agrees_with_reference(0, o, d, far, near),
        "a hit through the slab was accepted"
    );
    assert!(slab.hit_agrees_with_reference(0, o, d, near + 0.5 * REFERENCE_T_TOL, near));
    assert!(!slab.hit_agrees_with_reference(0, o, d, near + 2.0 * REFERENCE_T_TOL + 0.05, near));

    let lo = Vec3::new(-2.0 + 0.013, -2.0 + 0.021, 0.5 - 2.0 + 0.007);
    let nb = (4.0 / (8.0 * H)).ceil() as u32;
    let lv = level_from_fn(
        lo,
        H,
        [nb, nb, nb],
        |_c, _e| true,
        |p| vec![p.z - 0.5, -1.0],
    )
    .unwrap();
    let plane = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
    let g = Vec3::new(1.0, 0.0, -0.01).normalize();
    let o = Vec3::new(-1.0, 0.1, 0.5 + 0.0015);
    // s along the ray: 0.0015 - 0.01 t (per unit t, to first order): t in [0, 0.3] keeps |s| <= 0.0015
    assert!(
        plane.hit_agrees_with_reference(0, o, g, 0.3, 0.0),
        "one stretch within JUMP_S was refused"
    );
    assert!(0.3 > REFERENCE_T_TOL * 100.0 && 0.0015 < JUMP_S);
}

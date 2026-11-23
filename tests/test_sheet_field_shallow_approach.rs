//! Shallow approach: a ray that closes on a sheet at a small angle spends many floor steps (`MIN_STEP`) within a
//! fraction of a step of the sheet before it crosses. The march's default must decide every such ray (a hit or a
//! miss, never `Undecided`) and hit where the fixed-step reference (`RootMode::FixedFloor`) hits.
//!
//! The scene stands in for a scanned rock's face seen head-on at its rim: a plane sheet `s = z` (cells of 0.005)
//! and rays along +x descending at slopes from 1.125e-3 to 2e-3 from heights of 0.0003 to 0.0006. Each crosses the plane
//! after tens to hundreds of floor steps with both ends above it. A search that bounds the work per step and gives
//! up past the bound (`Undecided`) leaves such rays undecided; a consumer that reads a non-hit as a miss then loses
//! them (measured on a scanned slate part before this test: 217 of 640000 rays, all undecided).
//!
//! Checks, per ray: the default march's answer equals the reference's (both hit and agree by
//! `SheetField::hit_agrees_with_reference`, or both miss), and no ray is `Undecided`. The scene does test the claim:
//! every ray hits under the reference, at a point within the quantiser's bound of the plane (`decode_bound`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, RootMode, SheetField, ValidRule, BRICK, Q_MAX};
use glam::Vec3;

const H: f32 = 0.005;

/// How far the field's zero can sit from the plane: each brick stores its nodes as bytes over its own value range,
/// which for `s = z` is the brick's height `BRICK * H`, so a decoded node is within half a code of `z`. Trilinear
/// interpolation keeps that bound, and at the decoded zero `z = -error`, so `|z| <= half a code` (plus 1e-3 for the
/// f32 encode). A bound in `t` would have to divide by the ray's slope.
fn decode_bound() -> f32 {
    0.5 * (BRICK as f32 * H) / Q_MAX * (1.0 + 1e-3)
}

fn plane() -> SheetField {
    let lo = Vec3::new(-0.04 + 0.0013, -0.04 + 0.0021, -0.04 + 0.0007);
    // 8-cell bricks of 0.005: 0.04 per brick; 18 bricks along x cover every crossing below (at most 0.6 from x = -0.035)
    let lv = level_from_fn(lo, H, [18, 2, 2], |_c, _e| true, |p| vec![p.z, -1.0]).unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// (origin, direction, analytic crossing) for slopes 2e-3 .. 1.125e-3 and heights 0.0003 .. 0.0006, crossings <= 0.6
fn rays() -> Vec<(Vec3, Vec3, f32)> {
    let mut out = Vec::new();
    for i in 0..8 {
        let slope = 2e-3 - 1.25e-4 * i as f32;
        for j in 0..4 {
            let z0 = 0.0003 + 0.0001 * j as f32;
            let d = Vec3::new(1.0, 0.0, -slope).normalize();
            let o = Vec3::new(-0.035, 0.0013 + 0.0007 * j as f32, z0);
            let t = z0 / (slope / (1.0 + slope * slope).sqrt());
            if t < 0.6 {
                out.push((o, d, t));
            }
        }
    }
    out
}

#[test]
fn shallow_approaches_are_decided_and_hit_where_the_fixed_step_reference_hits() {
    let f = plane();
    let mut fref = f.clone();
    fref.set_root_mode(RootMode::FixedFloor);
    let rs = rays();
    assert!(rs.len() >= 20, "only {} rays", rs.len());
    let tmax = 0.7;
    let (mut undecided, mut hits) = (0usize, 0usize);
    for (o, d, t_an) in &rs {
        let want = fref.march_decided(*o, *d, tmax);
        let MarchResult::Hit(hr) = want else {
            panic!("the reference does not hit at {o} (analytic {t_an}): {want:?}")
        };
        let off = (*o + *d * hr.t).z;
        assert!(
            off.abs() <= decode_bound(),
            "reference {} (analytic {t_an}) is {off} off the plane",
            hr.t
        );
        match f.march_decided(*o, *d, tmax) {
            MarchResult::Hit(h) => {
                assert!(
                    f.hit_agrees_with_reference(h.sheet, *o, *d, h.t, hr.t),
                    "{o} {d}: hit {} reference {}",
                    h.t,
                    hr.t
                );
                hits += 1;
            }
            MarchResult::Undecided => undecided += 1,
            m => panic!("{o} {d}: {m:?}, the reference hits at {}", hr.t),
        }
    }
    assert_eq!(
        undecided,
        0,
        "{undecided} of {} rays undecided ({hits} hit)",
        rs.len()
    );
    assert_eq!(hits, rs.len());
}

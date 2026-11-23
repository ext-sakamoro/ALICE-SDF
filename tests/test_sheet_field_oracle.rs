//! Oracles for `sheet_field` against closed forms (no value here comes from the implementation under test).
//!
//! The sheet is the square patch `|x| <= 5, |y| <= 5, z = 0`, normal +z. Its channels are exact by construction:
//! `s = z` and `d_b` = the signed in-plane distance to the square (negative inside), so
//! `f = sqrt(z^2 + max(d_b, 0)^2)` is the Euclidean distance to the patch (oracle: point-to-rectangle distance).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::build::{level_from_fn, quantize, sheet};
use alice_sdf::sheet_field::{Level, MarchResult, SheetField, SheetFieldError, ValidRule, BRICK};
use glam::Vec3;

const H: f32 = 0.5;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 5.0;
    let dy = y.abs() - 5.0;
    let out = Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length();
    out + dx.max(dy).min(0.0)
}

/// oracle: Euclidean distance from p to the closed square patch |x|,|y| <= 5, z = 0
fn patch_distance(p: Vec3) -> f32 {
    let qx = p.x.abs() - 5.0;
    let qy = p.y.abs() - 5.0;
    Vec3::new(qx.max(0.0), qy.max(0.0), p.z).length()
}

fn patch_field(uv: bool, rule: ValidRule) -> SheetField {
    // grid origin off the patch's lines (0.37 / 0.21 / 0.13 off the nodes), 3 bricks of 4 mm around
    let lo = Vec3::new(-8.0 + 0.37, -8.0 + 0.21, -6.0 + 0.13);
    let nb = [4u32, 4, 3];
    let lv = level_from_fn(
        lo,
        H,
        nb,
        |_c, _e| true,
        |p| {
            let mut v = vec![p.z, square_db(p.x, p.y)];
            if uv {
                v.push(0.5 + 0.01 * p.x);
                v.push(0.25 + 0.02 * p.y);
            }
            v
        },
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], rule)
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*seed >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Distance values match the closed form within h/2 plus one quantization step of the brick (range 8 mm / 255
/// < 0.04). h/2 is a measured margin, not a derived bound: `s = z` is linear, so trilinear interpolation reproduces it
/// and only `d_b` errs; `d_b` does not depend on z, so its error is the bilinear one in the plane, which for a
/// 1-Lipschitz channel is at most the distance to the farthest cell corner, sqrt(2)/2 h. The patch's corners and
/// edges keep the measured worst below h/2, and the test holds it there.
#[test]
fn distance_matches_the_patch_closed_form() {
    let f = patch_field(false, ValidRule::CellV1);
    let mut seed = 7u64;
    let mut worst = 0.0f32;
    let mut n = 0;
    for _ in 0..4000 {
        let p = Vec3::new(
            lcg(&mut seed) * 12.0 - 6.0,
            lcg(&mut seed) * 12.0 - 6.0,
            lcg(&mut seed) * 2.0 - 1.0,
        );
        let (v, _, k) = f.eval(p);
        assert_eq!(k, Some(0));
        let s = f.sheets[0].sample(p, f.rule);
        if s.masked {
            assert!(
                v <= patch_distance(p) + 1e-4,
                "a masked value must be a lower bound: {v} vs {}",
                patch_distance(p)
            );
            continue;
        }
        worst = worst.max((v - patch_distance(p)).abs());
        n += 1;
    }
    assert!(n > 1000, "compared {n} points");
    assert!(worst <= 0.5 * H + 0.04, "worst |f - distance| {worst}");
}

/// Rays straight down hit the patch at t = 10 - 0 inside the square (|x|,|y| <= 4.5) and miss it beyond one
/// cell outside (|x| >= 5.6); none runs out of steps.
#[test]
fn rays_hit_the_patch_and_miss_beyond_it() {
    let f = patch_field(false, ValidRule::CellV1);
    let d = Vec3::new(0.0, 0.0, -1.0);
    let mut hits = 0;
    for i in 0..41 {
        for j in 0..41 {
            let x = -7.0 + 14.0 * i as f32 / 40.0;
            let y = -7.0 + 14.0 * j as f32 / 40.0;
            let r = f.march(Vec3::new(x, y, 10.0), d, 30.0, 4000);
            if x.abs() <= 4.5 && y.abs() <= 4.5 {
                let MarchResult::Hit(h) = r else {
                    panic!("({x}, {y}) inside the square: {r:?}")
                };
                assert!((h.t - 10.0).abs() < 0.02, "({x}, {y}): t {}", h.t);
                assert!(h.normal.z.abs() > 0.99, "normal {:?}", h.normal);
                hits += 1;
            } else if x.abs() >= 5.6 || y.abs() >= 5.6 {
                assert_eq!(r, MarchResult::Miss, "({x}, {y}) outside the square");
            }
        }
    }
    assert!(hits > 500, "hits {hits}");
}

/// A sign jump away from the surface: the side channel is `s = z` near the sheet, but on the left half
/// (x < 0) its sign flips where |z| > 2 (as when the nearest point switches to another part of an open sheet
/// seen from the other side). The trilinear s then crosses zero at z ~ 2 above the left half, a crossing 2 mm
/// from the sheet. The cell rule masks every cell there (|s| ~ 2 > sqrt(3)/2 h + q), so rays down the left
/// half (started inside the jump's band) reach the true sheet at z = 0; without a mask they stop at the phantom (red). Rays down the
/// right half hit z = 0 under both rules (the mask never removes a true hit).
#[test]
fn the_cell_rule_removes_a_far_sign_jump_and_keeps_true_hits() {
    let build = |rule| {
        let lo = Vec3::new(-8.0 + 0.37, -8.0 + 0.21, -6.0 + 0.13);
        let lv = level_from_fn(
            lo,
            H,
            [4, 4, 3],
            |_c, _e| true,
            |p| {
                vec![
                    if p.x < 0.0 && p.z.abs() > 2.0 {
                        -p.z
                    } else {
                        p.z
                    },
                    square_db(p.x, p.y),
                ]
            },
        )
        .unwrap();
        SheetField::new(vec![sheet(vec![lv]).unwrap()], rule)
    };
    let d = Vec3::new(0.0, 0.0, -1.0);
    // left-half rays start inside the jump's band (z = 2.2): from far away a sphere-tracing step of |s| would
    // jump over the band in one step and never sample the phantom (vacuous); starting in it, an unmasked field
    // stops at once and a masked one steps on to the sheet
    const Z0: f32 = 2.2;
    let near_jump = |f: &SheetField| {
        (0..21).filter(|i| {
        let x = -4.0 + 3.5 * f32::from(*i as u8) / 20.0;
        matches!(f.march(Vec3::new(x, 1.0, Z0), d, 30.0, 4000), MarchResult::Hit(h) if (h.t - Z0).abs() > 0.5)
    }).count()
    };
    let masked = build(ValidRule::CellV1);
    let unmasked = build(ValidRule::None);
    assert_eq!(
        near_jump(&masked),
        0,
        "cell rule: phantom hits above the left half"
    );
    // without a mask a left-half ray either stops at a phantom (the upper jump, or the lower one at z = -2 when it
    // starts in the flipped region and the true crossing reads negative on both sides) or steps over the band; measured
    // 2026-11-13: most rays stop at a phantom. the guard asks for at least 5 of 21 so the mutant is visibly red
    let ph = near_jump(&unmasked);
    assert!(
        ph >= 5,
        "no mask: only {ph} of 21 left-half rays stopped at a phantom (the test would be vacuous)"
    );
    for x in [0.5f32, 1.5, 3.0, 4.0] {
        for f in [&masked, &unmasked] {
            let MarchResult::Hit(h) = f.march(Vec3::new(x, 1.0, 10.0), d, 30.0, 4000) else {
                panic!("x {x}: no hit")
            };
            assert!((h.t - 10.0).abs() < 0.02, "x {x}: t {}", h.t);
        }
    }
    for i in 0..21 {
        let x = -4.0 + 3.5 * i as f32 / 20.0;
        let MarchResult::Hit(h) = masked.march(Vec3::new(x, 1.0, Z0), d, 30.0, 4000) else {
            panic!("left x {x}: no hit")
        };
        assert!((h.t - Z0).abs() < 0.02, "left x {x}: t {}", h.t);
    }
}

/// A ray with too few steps to arrive is undecided, never a hit (and never a miss).
#[test]
fn a_ray_out_of_steps_is_undecided() {
    let f = patch_field(false, ValidRule::CellV1);
    let r = f.march(
        Vec3::new(0.0, 0.0, 400.0),
        Vec3::new(0.0, 0.0, -1.0),
        800.0,
        3,
    );
    assert_eq!(r, MarchResult::Undecided);
}

/// UV channels carry a linear map exactly up to quantization (range / 255 per brick).
#[test]
fn uv_channels_carry_the_surface_map() {
    let f = patch_field(true, ValidRule::CellV1);
    let MarchResult::Hit(h) = f.march(
        Vec3::new(1.3, -2.1, 10.0),
        Vec3::new(0.0, 0.0, -1.0),
        30.0,
        4000,
    ) else {
        panic!("no hit")
    };
    let uv = h.uv.expect("uv channels");
    assert!((uv[0] - (0.5 + 0.01 * 1.3)).abs() < 2e-4, "u {}", uv[0]);
    assert!((uv[1] - (0.25 - 0.02 * 2.1)).abs() < 4e-4, "v {}", uv[1]);
    assert!(patch_field(false, ValidRule::CellV1).sheets[0]
        .sample(Vec3::ZERO, ValidRule::CellV1)
        .uv
        .is_none());
}

#[test]
fn unknown_rule_names_are_refused() {
    assert_eq!(ValidRule::from_name("cell-v1"), Ok(ValidRule::CellV1));
    for bad in ["", "none", "us:1.0", "Cell-V1"] {
        assert!(
            matches!(
                ValidRule::from_name(bad),
                Err(SheetFieldError::UnknownRule(_))
            ),
            "{bad}"
        );
    }
}

/// Degenerate builds are refused with an error (not a panic, not an empty sheet).
#[test]
fn degenerate_builds_are_refused() {
    let lo = Vec3::ZERO;
    let ch = |p: Vec3| vec![p.z, -1.0];
    assert!(level_from_fn(lo, 0.0, [1, 1, 1], |_c, _e| true, ch).is_err());
    assert!(level_from_fn(lo, -1.0, [1, 1, 1], |_c, _e| true, ch).is_err());
    assert!(level_from_fn(lo, f32::NAN, [1, 1, 1], |_c, _e| true, ch).is_err());
    assert!(level_from_fn(lo, 1.0, [0, 1, 1], |_c, _e| true, ch).is_err());
    assert!(level_from_fn(lo, 1.0, [1, 1, 1], |_c, _e| false, ch).is_err());
    assert!(level_from_fn(lo, 1.0, [1, 1, 1], |_c, _e| true, |_p| vec![0.0; 3]).is_err());
    assert!(sheet(vec![]).is_err());
    // a channel value that is not finite is refused, not folded into the brick's range
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let e = level_from_fn(
            lo,
            1.0,
            [1, 1, 1],
            |_c, _e| true,
            move |p| vec![if p.x > 0.5 { bad } else { p.z }, -1.0],
        )
        .expect_err("a non-finite channel value was quantized");
        assert!(e.to_string().contains("not finite"), "{e}");
        let mut brick = vec![0.0f32; (BRICK + 1).pow(3)];
        brick[3] = bad;
        assert!(quantize(&[brick], false).is_err());
    }
    assert!(quantize(&[vec![0.0; 5]], false).is_err());
}

fn layout_err(lv: Level) -> String {
    let e = sheet(vec![lv.clone()]).expect_err("a malformed level was accepted");
    let SheetFieldError::Shape(m) = &e else {
        panic!("not a shape error: {e:?}")
    };
    // the same layout assembled from the public fields is refused by `Sheet::validate`
    let mut sh = sheet(vec![patch_level()]).unwrap();
    sh.levels = vec![lv];
    assert!(
        sh.validate().is_err(),
        "validate accepted what sheet refused: {m}"
    );
    m.clone()
}

fn patch_level() -> Level {
    patch_field(false, ValidRule::CellV1).sheets[0].levels[0].clone()
}

/// Every reader indexes a level by its index, rows and kept counts, so a level that does not fit them is refused
/// where it is made (`sheet`, which the ASHF reader also uses) and by `Sheet::validate` (which the GPU upload calls).
/// A dropped `s` row would read 0 at every node: the cell rule would mask the brick and a ray would step straight
/// through a surface it may contain, so `s` must be stored for every active brick.
#[test]
fn malformed_level_layouts_and_a_dropped_s_row_are_refused() {
    let good = patch_level();
    assert!(sheet(vec![good.clone()]).unwrap().validate().is_ok());
    // a hand-built brick whose s row is dropped (row -1, the form d_b uses for a brick outside the boundary)
    let mut lv = good.clone();
    lv.channels[0].rows[0] = -1;
    assert!(layout_err(lv).contains("channel 0 (s) drops a brick"));
    // d_b may drop a brick: same level with d_b's first row dropped is accepted
    let mut lv = good.clone();
    lv.channels[1].rows[0] = -1;
    assert!(sheet(vec![lv]).is_ok());
    let mut lv = good.clone();
    lv.index.pop();
    assert!(layout_err(lv).contains("index of"));
    let mut lv = good.clone();
    let active = lv.channels[0].rows.len() as i32;
    lv.index[0] = active;
    assert!(layout_err(lv).contains("index entry"));
    let mut lv = good.clone();
    lv.index[0] = -2;
    assert!(layout_err(lv).contains("index entry"));
    let mut lv = good.clone();
    lv.channels[1].rows.pop();
    assert!(layout_err(lv).contains("rows for"));
    let mut lv = good.clone();
    lv.channels[1].rows[0] = lv.channels[1].scale.len() as i32;
    assert!(layout_err(lv).contains("kept rows"));
    let mut lv = good.clone();
    lv.channels[0].scale[0] = f32::NAN;
    assert!(layout_err(lv).contains("not finite and positive"));
    let mut lv = good.clone();
    lv.channels[0].offset[0] = f32::INFINITY;
    assert!(layout_err(lv).contains("not finite and positive"));
    let mut lv = good.clone();
    lv.channels.truncate(1);
    assert!(layout_err(lv).contains("channels (2, 4 or 7)"));
    let mut lv = good.clone();
    lv.lo.x = f32::NAN;
    assert!(layout_err(lv).contains("level h"));
    let mut lv = good;
    lv.h = 0.0;
    assert!(layout_err(lv).contains("level h"));
}

/// A point that is not finite is in no brick: the sample is the outside one, not brick 0's value.
#[test]
fn a_point_that_is_not_finite_is_outside() {
    let f = patch_field(false, ValidRule::CellV1);
    for p in [
        Vec3::new(f32::NAN, 0.0, 0.0),
        Vec3::new(0.0, f32::INFINITY, 0.0),
        Vec3::new(0.0, 0.0, f32::NEG_INFINITY),
    ] {
        let s = f.sheets[0].sample(p, f.rule);
        assert!(!s.inside, "{p} answered inside");
        assert_eq!(s.uv, None);
    }
}

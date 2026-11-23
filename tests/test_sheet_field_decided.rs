//! A ray always ends decided: with the step budget [`decided_steps`] of `tmax` the march returns a hit or a miss,
//! never `Undecided`, however long it grazes a sheet.
//!
//! Every step advances at least the floor [`MIN_STEP`] (less the f32 rounding of `t`), so `tmax` is covered within
//! `decided_steps(tmax)` steps: `Undecided` is unreachable by construction, not tuned away by a large constant.
//!
//! Oracle (geometry, not the field's own output): the flat square sheet `z = 0`, `|x|, |y| <= 60`, and a ray parallel
//! to it at height 0.02 mm along `+x` over its full length. The ray never meets the sheet, so the answer is a miss.
//! Its distance to the sheet stays 0.02 mm for 120 mm, every step is 0.02 mm (the distance) and the ray needs ~6000
//! steps: a fixed budget of 4000 (the earlier caller's) ends it undecided. A tilted ray through the same sheet must
//! still hit at the analytic `t` (the budget changes nothing for rays that end early).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{decided_steps, MarchResult, SheetField, ValidRule, MIN_STEP};
use glam::Vec3;

const HALF: f32 = 60.0;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - HALF;
    let dy = y.abs() - HALF;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn flat() -> SheetField {
    let h = 0.25;
    let lo = Vec3::new(-64.0 + 0.013, -2.0 + 0.021, -1.0 + 0.007);
    // 2 bricks across y and z: the rays stay near y = 0, z = 0
    let lv = level_from_fn(
        lo,
        h,
        [64, 2, 1],
        |_c, _e| true,
        |p| vec![p.z, square_db(p.x, p.y)],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// the grazing rays: start just outside the sheet's x range, run along +x the whole length at z = 0.02
fn grazing() -> (Vec<Vec3>, Vec3, f32) {
    let os = (0..8)
        .map(|i| Vec3::new(-HALF - 1.0, -0.4 + 0.1 * i as f32 + 0.003, 0.02))
        .collect();
    (os, Vec3::X, 2.0 * HALF + 2.0)
}

fn check_miss(label: &str, r: &[MarchResult], wrong: &mut Vec<String>) {
    for (i, m) in r.iter().enumerate() {
        if *m != MarchResult::Miss {
            wrong.push(format!("{label} ray {i}: {m:?}, want Miss"));
        }
    }
}

#[test]
fn the_budget_covers_tmax_at_the_floor_step() {
    for tmax in [0.5f32, 1.0, 122.0, 974.8, 4000.0, 8000.0] {
        let n = decided_steps(tmax).unwrap();
        // walk t the way the march does (f32 adds of the floor) and count the steps to reach tmax; at 8000 mm the
        // f32 rounding makes the walk 809002 steps, 9002 more than tmax / MIN_STEP
        let (mut t, mut k) = (0.0f32, 0u32);
        loop {
            if t >= tmax {
                break;
            }
            t += MIN_STEP;
            k += 1;
        }
        assert!(n >= k, "tmax {tmax}: budget {n} < {k} floor steps");
    }
}

#[test]
fn a_ray_grazing_a_sheet_for_its_full_length_ends_in_a_miss() {
    let f = flat();
    let (os, d, tmax) = grazing();
    let steps = decided_steps(tmax).unwrap();
    let mut wrong = Vec::new();
    let cpu: Vec<MarchResult> = os
        .iter()
        .map(|o| f.march_decided(*o, d, tmax).unwrap())
        .collect();
    check_miss("cpu", &cpu, &mut wrong);
    // the same rays with the earlier fixed budget run out of steps: the grazing is real, not a trivial miss
    let short: Vec<MarchResult> = os.iter().map(|o| f.march(*o, d, tmax, 4000)).collect();
    assert!(
        short.iter().all(|m| *m == MarchResult::Undecided),
        "4000 steps already decide these rays (the oracle grazes too little): {short:?}"
    );
    #[cfg(feature = "gpu")]
    match alice_sdf::sheet_field::gpu::GpuSheetField::new(&f) {
        Ok(g) => {
            let gpu = g.march_batch_decided(&os, d, tmax).unwrap();
            check_miss("gpu", &gpu, &mut wrong);
            assert_eq!(gpu, cpu, "gpu vs cpu");
        }
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
        }
    }
    assert!(steps > 4000, "budget {steps}");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn a_ray_through_the_sheet_still_hits_at_the_analytic_distance() {
    let f = flat();
    // from z = 0.6 down at 30 degrees: meets z = 0 at t = 0.6 / sin(30 deg) = 1.2
    let d = Vec3::new(0.866_025_4, 0.0, -0.5);
    let o = Vec3::new(10.003, 0.107, 0.6);
    match f.march_decided(o, d, 50.0).unwrap() {
        // surface distance |dt| sin 30 within one u8 step of s (the brick spans 2 mm of s: 2 / 255 mm)
        MarchResult::Hit(h) => assert!((h.t - 1.2).abs() * 0.5 <= 2.0 / 255.0, "t {}", h.t),
        m => panic!("{m:?}"),
    }
}

/// A `tmax` so far that the f32 rounding of `t` reaches the floor step (from about 83,886 mm: `MIN_STEP - tmax *
/// f32::EPSILON <= 0`) has no deciding budget: it is refused as a range error on the CPU and the GPU, never a panic and
/// never a march that silently stops short. Just inside the range the budget is still computed.
#[test]
fn a_tmax_no_budget_decides_is_refused_not_panicked() {
    use alice_sdf::sheet_field::SheetFieldError;
    let edge = MIN_STEP / f32::EPSILON;
    for tmax in [1e5f32, edge * 1.01, f32::MAX] {
        assert!(
            matches!(decided_steps(tmax), Err(SheetFieldError::Range(_))),
            "tmax {tmax}"
        );
        let f = flat();
        assert!(matches!(
            f.march_decided(Vec3::ZERO, -Vec3::Z, tmax),
            Err(SheetFieldError::Range(_))
        ));
        assert!(matches!(
            f.march_profile(Vec3::ZERO, -Vec3::Z, tmax),
            Err(SheetFieldError::Range(_))
        ));
    }
    // inside: the budget covers tmax at the reduced least step, so it exceeds tmax / MIN_STEP
    let near = edge * 0.5;
    let n = decided_steps(near).unwrap();
    assert!(f64::from(n) >= f64::from(near) / f64::from(MIN_STEP), "{n}");
    // the non-finite and non-positive cases still end at once
    for tmax in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        assert_eq!(decided_steps(tmax), Ok(1), "tmax {tmax}");
    }
    #[cfg(feature = "gpu")]
    match alice_sdf::sheet_field::gpu::GpuSheetField::new(&flat()) {
        Ok(g) => {
            let e = g
                .march_batch_decided(&[Vec3::ZERO], -Vec3::Z, 1e5)
                .expect_err("a tmax no budget decides was marched on the GPU");
            assert!(e.to_string().contains("out of range"), "{e}");
        }
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
        }
    }
}

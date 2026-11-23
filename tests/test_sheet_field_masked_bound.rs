//! The masked cell's lower bound, per corner (`max_c (|s_c| - |p - c|) - scale`, module docs): it lets a ray take
//! longer steps through masked cells, and it must never step a ray past a sheet's surface out of the sheet's box
//! (the only way a crossing can be lost: a crossing is a sign change of the sheet's own value between two steps
//! inside its box).
//!
//! Oracles (none is the march's own output):
//! - two flat sheets 0.03 apart (closer than the 0.05 floor), each in a box only 8 cells deep with its surface
//!   one cell above the box bottom, rays from inside the boxes' masked top region: every ray hits the upper sheet
//!   at the analytic depth (origin z minus the sheet's z, within the bisection's resolution), on the CPU and the
//!   GPU, and the masked bound did set steps longer than the old bound would have (the field took fewer steps);
//! - the bound never exceeds the true distance to the sheet at sampled points inside masked cells (the analytic
//!   distance of a flat sheet is |z - z_sheet|).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;

const H: f32 = 0.0625;
/// One storage step of `s` in these boxes (bytes over about 8 cells of range: 0.5 / 255 ~ 0.002), and a margin:
/// the stored field differs from the analytic one by up to a step, so hit depths and distances are compared at that.
const STEP: f32 = 0.003;

/// A flat square sheet (half side 0.7) at height `z` in a box 2 wide (4 x 4 bricks) and 8 cells deep whose bottom is
/// one cell below the sheet; the sheet stays 0.3 inside the box sides (over the 3 cells the outside bound assumes).
fn flat(z: f32) -> Sheet {
    let lo = Vec3::new(-1.0 + 0.013, -1.0 + 0.021, z - H);
    let lv = level_from_fn(
        lo,
        H,
        [4, 4, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = (p.x.abs() - 0.7, p.y.abs() - 0.7);
            vec![
                p.z - z,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

const ZB: f32 = 0.2;
const ZC: f32 = ZB - 0.03;

fn field() -> SheetField {
    SheetField::new(vec![flat(ZB), flat(ZC)], ValidRule::CellV1)
}

/// Rays straight down from just under the top of the upper sheet's box (7 cells above it, in masked cells).
fn origins() -> Vec<Vec3> {
    let top = ZB - H + 8.0 * H - 0.01;
    (0..400)
        .map(|i| {
            let (a, b) = ((i % 20) as f32 / 20.0, (i / 20) as f32 / 20.0);
            Vec3::new(-0.6 + 1.2 * a + 0.003, -0.6 + 1.2 * b + 0.007, top)
        })
        .collect()
}

#[test]
fn every_ray_hits_the_upper_sheet_at_its_depth_on_cpu_and_gpu() {
    let f = field();
    let os = origins();
    let tmax = 1.0;
    let expect = |o: Vec3| o.z - ZB;
    let mut steps_total = 0usize;
    for &o in &os {
        let (m, rec) = f.march_profile(o, -Vec3::Z, tmax).unwrap();
        steps_total += rec.len();
        match m {
            MarchResult::Hit(h) => {
                assert_eq!(h.sheet, 0, "{o}: hit the lower sheet ({m:?})");
                assert!(
                    (h.t - expect(o)).abs() < STEP,
                    "{o}: t {} expected {}",
                    h.t,
                    expect(o)
                );
            }
            other => panic!("{o}: {other:?}, expected a hit on the upper sheet"),
        }
    }
    // 7 cells of masked approach: with the old bound (umin - scale - 0.866 h, about 6 cells at the start) the
    // approach takes a handful of steps; the per-corner bound must not take more
    assert!(
        steps_total <= os.len() * 12,
        "{steps_total} steps for {} rays",
        os.len()
    );

    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        let g = match GpuSheetField::new(&f) {
            Ok(g) => g,
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
                );
                eprintln!("gpu-test: skipped (no GPU: {e})");
                return;
            }
        };
        let r = g.march_batch_decided(&os, -Vec3::Z, tmax).unwrap();
        for (o, m) in os.iter().zip(&r) {
            match m {
                MarchResult::Hit(h) => {
                    assert_eq!(h.sheet, 0, "{o}: gpu hit the lower sheet ({m:?})");
                    assert!(
                        (h.t - expect(*o)).abs() < STEP,
                        "{o}: gpu t {} expected {}",
                        h.t,
                        expect(*o)
                    );
                }
                other => panic!("{o}: gpu {other:?}, expected a hit on the upper sheet"),
            }
        }
    }
}

/// The tilted plane's unit normal: a slight tilt puts the stored node values between quantisation steps (a flat
/// sheet's nodes land on the steps exactly, so the storage term would never be needed there).
fn nrm() -> Vec3 {
    Vec3::new(0.03, 0.02, 1.0).normalize()
}

/// A tilted square sheet `n . p = d` (half side 0.7 in x and y) in the same box shape as [`flat`], with `s` its
/// exact signed distance.
fn tilted(d: f32) -> Sheet {
    let n = nrm();
    let lo = Vec3::new(-1.0 + 0.013, -1.0 + 0.021, d - H);
    let lv = level_from_fn(
        lo,
        H,
        [4, 4, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = (p.x.abs() - 0.7, p.y.abs() - 0.7);
            vec![
                n.dot(p) - d,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

/// Soundness without a tolerance for storage error: in a masked cell the bound must be at most the true distance
/// exactly (up to f32 rounding, 1e-5, against a storage step of about 0.002), because the bound already subtracts
/// the storage step (`- scale`). Each point lies a short way along the normal from a grid node at least 2.5 cells
/// from the sheet, so that node's per-corner term equals the true distance up to the node's storage error: a bound
/// without `- scale` exceeds the distance wherever that node's stored value was rounded away from the sheet.
#[test]
fn the_masked_bound_never_exceeds_the_true_distance() {
    let n = nrm();
    let (db, dc) = (ZB, ZB - 0.03);
    let f = SheetField::new(vec![tilted(db), tilted(dc)], ValidRule::CellV1);
    let lo = Vec3::new(-1.0 + 0.013, -1.0 + 0.021, db - H);
    let (mut checked, mut tight) = (0usize, 0usize);
    for i in 6..26 {
        for j in 6..26 {
            for k in 0..9 {
                let c = lo + H * Vec3::new(i as f32, j as f32, k as f32);
                if n.dot(c) - db < 2.5 * H {
                    continue;
                }
                for e in [0.003f32, 0.011, 0.023] {
                    let p = c + n * e;
                    if p.z > lo.z + 8.0 * H - 0.001 {
                        continue;
                    }
                    let (fv, _, ks) = f.eval(p);
                    let truth = (n.dot(p) - db).abs().min((n.dot(p) - dc).abs());
                    assert!(
                        fv <= truth + 1e-5,
                        "{p}: bound {fv} over the distance {truth} (sheet {ks:?})"
                    );
                    checked += 1;
                    if truth - fv < 0.01 {
                        tight += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 2000, "only {checked} points checked");
    assert!(
        tight > checked / 2,
        "the bound was tight at only {tight} of {checked} points: not the per-corner term"
    );
}

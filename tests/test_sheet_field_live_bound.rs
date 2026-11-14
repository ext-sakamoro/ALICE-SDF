//! The GPU march's per-ray list of live sheets is sized per field from a proved bound, so a deep stack of
//! overlapping sheet boxes marches like the CPU instead of being refused, and a shallow field compiles the same
//! program as before.
//!
//! Oracles (none is the shader's own output):
//! - bound: for sampled points (box centres and corners pulled inside), the number of sheet boxes holding the
//!   point, counted directly from the boxes, never exceeds [`live_bound`];
//! - buckets: the compiled list size is 64 up to 64, then the next power of two up to [`MAX_LIVE_CAP`], and a
//!   bound over the cap is refused (typed);
//! - same program: a field whose bound is at most 64 gets exactly the shader text of the fixed 64 list;
//! - geometry: 100 flat squares stacked 0.01 mm apart (every point of the stack is in all 100 boxes), rays
//!   straight down: the first hit is the top square at the analytic t, on the GPU as on the CPU.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{
    live_bound, march_shader, max_live_for, GpuSheetField, MAX_LIVE_CAP, MAX_LIVE_SHEETS,
};
use alice_sdf::sheet_field::{MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;

const H: f32 = 0.25;

fn square(z: f32, cx: f32, cy: f32, half: f32) -> Sheet {
    let lo = Vec3::new(cx - 8.0 + 0.013, cy - 8.0 + 0.021, z - 1.0 + 0.007);
    let lv = level_from_fn(
        lo,
        H,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = ((p.x - cx).abs() - half, (p.y - cy).abs() - half);
            vec![
                p.z - z,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

/// 100 squares, z = 0.01 k, same footprint: all 100 boxes overlap
fn deep() -> SheetField {
    SheetField::new(
        (0..100)
            .map(|k| square(0.01 * k as f32, 0.0, 0.0, 5.0))
            .collect(),
        ValidRule::CellV1,
    )
}

/// 12 squares scattered in x, y and z (depth well under 64)
fn shallow() -> SheetField {
    SheetField::new(
        (0..12)
            .map(|k| {
                square(
                    10.0 - 0.7 * k as f32,
                    ((k * 5) % 7) as f32 - 3.0,
                    ((k * 3) % 5) as f32 - 2.0,
                    2.0 + (k % 4) as f32 * 0.9,
                )
            })
            .collect(),
        ValidRule::CellV1,
    )
}

/// the most boxes holding one of the sampled points, counted directly
fn sampled_depth(f: &SheetField) -> usize {
    let mut pts = Vec::new();
    for s in &f.sheets {
        pts.push((s.lo + s.hi) * 0.5);
        for c in 0..8u32 {
            let pick = |i: u32, lo: f32, hi: f32| {
                if c >> i & 1 == 1 {
                    hi - 1e-3
                } else {
                    lo + 1e-3
                }
            };
            pts.push(Vec3::new(
                pick(0, s.lo.x, s.hi.x),
                pick(1, s.lo.y, s.hi.y),
                pick(2, s.lo.z, s.hi.z),
            ));
        }
    }
    assert!(!pts.is_empty());
    pts.iter()
        .map(|p| {
            f.sheets
                .iter()
                .filter(|s| p.cmpge(s.lo).all() && p.cmple(s.hi).all())
                .count()
        })
        .max()
        .unwrap()
}

#[test]
fn the_bound_holds_on_sampled_points_and_the_buckets_follow_it() {
    for (name, f) in [("deep", deep()), ("shallow", shallow())] {
        let d = sampled_depth(&f);
        let b = live_bound(&f);
        assert!(
            d > 0 && d <= b as usize,
            "{name}: sampled depth {d} over the bound {b}"
        );
        let m = max_live_for(b).unwrap();
        assert!(
            m >= b && m >= MAX_LIVE_SHEETS,
            "{name}: list {m} under bound {b}"
        );
    }
    assert_eq!(sampled_depth(&deep()), 100);
    assert_eq!(max_live_for(1).unwrap(), 64);
    assert_eq!(max_live_for(64).unwrap(), 64);
    assert_eq!(max_live_for(65).unwrap(), 128);
    assert_eq!(max_live_for(100).unwrap(), 128);
    assert_eq!(max_live_for(129).unwrap(), 256);
    assert_eq!(max_live_for(MAX_LIVE_CAP).unwrap(), MAX_LIVE_CAP);
    assert!(
        max_live_for(MAX_LIVE_CAP + 1).is_err(),
        "a bound over the cap is refused"
    );
}

#[test]
fn a_field_with_bound_at_most_64_compiles_the_program_of_the_fixed_list() {
    let f = shallow();
    let b = live_bound(&f);
    assert!(
        b <= MAX_LIVE_SHEETS,
        "the shallow field is not shallow: {b}"
    );
    assert_eq!(
        march_shader(max_live_for(b).unwrap()),
        march_shader(MAX_LIVE_SHEETS)
    );
    match GpuSheetField::new(&f) {
        Ok(g) => assert_eq!(g.max_live(), MAX_LIVE_SHEETS),
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
        }
    }
}

#[test]
fn a_stack_of_100_boxes_marches_on_the_gpu_like_the_cpu() {
    let f = deep();
    assert!(live_bound(&f) >= 100);
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
    assert_eq!(g.max_live(), 128);
    let origins: Vec<Vec3> = (0..64)
        .map(|i| {
            Vec3::new(
                -4.0 + 8.0 * ((i % 8) as f32 + 0.37) / 8.0,
                -4.0 + 8.0 * ((i / 8) as f32 + 0.61) / 8.0,
                5.0,
            )
        })
        .collect();
    let gpu = g
        .march_batch_decided(&origins, -Vec3::Z, 10.0)
        .expect("the deep stack is marched, not refused");
    for (o, m) in origins.iter().zip(&gpu) {
        let c = f.march_decided(*o, -Vec3::Z, 10.0);
        match (m, c) {
            (MarchResult::Hit(a), MarchResult::Hit(b)) => {
                assert_eq!(
                    a.sheet, 99,
                    "ray {o}: the top square (z 0.99) is the first hit"
                );
                assert_eq!(a.sheet, b.sheet, "ray {o}");
                // analytic plane z = 0.99 from z = 5: t = 4.01 within one s step (2 mm / 255)
                assert!((a.t - 4.01).abs() <= 2.0 / 255.0, "ray {o}: t {}", a.t);
            }
            (m, c) => panic!("ray {o}: gpu {m:?} cpu {c:?}"),
        }
    }
}

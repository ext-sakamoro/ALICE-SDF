//! GPU parity for `sheet_field`: the WGSL evaluator and marcher (`sheet_field::gpu::GpuSheetField`) return the
//! CPU reference (`SheetField`), which `test_sheet_field_oracle` pins to closed forms.
//!
//! Fields: (a) the square patch with UV channels, (b) the same patch with a far sign jump of the side channel
//! (where the cell rule masks cells). Checks: the union value, its signed companion and the masked flag at
//! 4,000 points per field (the masked flags must agree point for point, so a shader that skips the rule
//! differs from the CPU on field (b)); ray marches over a grid: hit / miss / undecided agree, and t, normal
//! and UV agree within f32 evaluation-order tolerance.
//!
//! CI's gpu-parity job sets ALICE_SDF_REQUIRE_GPU=1 so a missing adapter fails instead of skipping.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::GpuSheetField;
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule};
use glam::Vec3;

/// f32 evaluation order differs between the CPU loop and the shader (fused or reordered sums).
const TOL: f32 = 1e-4;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 5.0;
    let dy = y.abs() - 5.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn field(jump: bool) -> SheetField {
    let lo = Vec3::new(-8.0 + 0.37, -8.0 + 0.21, -6.0 + 0.13);
    let lv = level_from_fn(
        lo,
        0.5,
        [4, 4, 3],
        |_c, _e| true,
        |p| {
            let s = if jump && p.x < 0.0 && p.z.abs() > 2.0 {
                -p.z
            } else {
                p.z
            };
            vec![s, square_db(p.x, p.y), 0.5 + 0.01 * p.x, 0.25 + 0.02 * p.y]
        },
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

fn gpu(f: &SheetField) -> Option<GpuSheetField> {
    match GpuSheetField::new(f) {
        Ok(g) => Some(g),
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
            None
        }
    }
}

fn points(seed: u64) -> Vec<Vec3> {
    let mut s = seed;
    let mut r = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..4000)
        .map(|_| Vec3::new(r() * 16.0 - 8.0, r() * 16.0 - 8.0, r() * 12.0 - 6.0))
        .collect()
}

#[test]
fn values_and_masks_match_the_cpu() {
    for jump in [false, true] {
        let f = field(jump);
        let Some(g) = gpu(&f) else { return };
        let pts = points(11 + u64::from(jump));
        let out = g.eval_batch(&pts).unwrap();
        assert_eq!(out.len(), pts.len());
        let mut masked = 0;
        for (p, o) in pts.iter().zip(&out) {
            let (fc, gc, _) = f.eval(*p);
            let sc = f.sheets[0].sample(*p, f.rule);
            assert!(
                (o.f - fc).abs() <= TOL * fc.abs().max(1.0),
                "jump {jump} {p:?}: f gpu {} cpu {fc}",
                o.f
            );
            assert!(
                (o.g - gc).abs() <= TOL * gc.abs().max(1.0),
                "jump {jump} {p:?}: g gpu {} cpu {gc}",
                o.g
            );
            assert_eq!(
                o.masked,
                sc.masked && sc.inside,
                "jump {jump} {p:?}: masked flag"
            );
            masked += usize::from(o.masked);
        }
        if jump {
            assert!(
                masked > 100,
                "the sign-jump field must have masked cells to compare ({masked})"
            );
        }
    }
}

#[test]
fn marches_match_the_cpu() {
    for jump in [false, true] {
        let f = field(jump);
        let Some(g) = gpu(&f) else { return };
        let mut origins = Vec::new();
        for i in 0..41 {
            for j in 0..41 {
                let z = if jump && i < 20 { 2.2 } else { 10.0 };
                origins.push(Vec3::new(
                    -7.0 + 14.0 * i as f32 / 40.0,
                    -7.0 + 14.0 * j as f32 / 40.0,
                    z,
                ));
            }
        }
        let d = Vec3::new(0.0, 0.0, -1.0);
        let out = g.march_batch(&origins, d, 30.0, 4000).unwrap();
        assert_eq!(out.len(), origins.len());
        let mut hits = 0;
        for (o, r) in origins.iter().zip(&out) {
            let c = f.march(*o, d, 30.0, 4000);
            match (c, *r) {
                (MarchResult::Hit(a), MarchResult::Hit(b)) => {
                    hits += 1;
                    assert!(
                        (a.t - b.t).abs() <= 1e-3,
                        "{o:?}: t cpu {} gpu {}",
                        a.t,
                        b.t
                    );
                    assert!(
                        a.normal.dot(b.normal) >= 0.999,
                        "{o:?}: normal cpu {:?} gpu {:?}",
                        a.normal,
                        b.normal
                    );
                    let (ua, ub) = (a.uv.unwrap(), b.uv.unwrap());
                    assert!(
                        (ua[0] - ub[0]).abs() <= 2e-4 && (ua[1] - ub[1]).abs() <= 2e-4,
                        "{o:?}: uv {ua:?} {ub:?}"
                    );
                }
                (a, b) => assert_eq!(
                    std::mem::discriminant(&a),
                    std::mem::discriminant(&b),
                    "{o:?}: cpu {a:?} gpu {b:?}"
                ),
            }
        }
        assert!(hits > 400, "jump {jump}: hits {hits}");
    }
}

/// Out of steps is undecided on the GPU too (never a hit).
#[test]
fn a_gpu_ray_out_of_steps_is_undecided() {
    let f = field(false);
    let Some(g) = gpu(&f) else { return };
    let out = g
        .march_batch(
            &[Vec3::new(0.0, 0.0, 400.0)],
            Vec3::new(0.0, 0.0, -1.0),
            800.0,
            3,
        )
        .unwrap();
    assert_eq!(out[0], MarchResult::Undecided);
}

/// 300 small sheets (the BVH oracle's cloud): the GPU's BVH walks return the CPU's answers (values, masked
/// flags, owning sheets via the march).
fn cloud() -> SheetField {
    let mut s = 3u64;
    let mut r = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let mut sheets = Vec::new();
    for _ in 0..300 {
        let c = Vec3::new(r(), r(), r()) * 40.0;
        let n = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5).normalize();
        let lv = level_from_fn(
            c - Vec3::splat(2.13),
            0.5,
            [1, 1, 1],
            |_c, _e| true,
            move |p| {
                let q = p - c;
                let along = q.dot(n);
                vec![along, (q - n * along).length() - 1.5]
            },
        )
        .unwrap();
        sheets.push(sheet(vec![lv]).unwrap());
    }
    SheetField::new(sheets, ValidRule::CellV1)
}

#[test]
fn many_sheets_match_the_cpu() {
    let f = cloud();
    let Some(g) = gpu(&f) else { return };
    let pts: Vec<Vec3> = points(41)
        .iter()
        .map(|p| *p * 2.5 + Vec3::splat(20.0))
        .collect();
    let out = g.eval_batch(&pts).unwrap();
    for (p, o) in pts.iter().zip(&out) {
        let (fc, gc, _) = f.eval(*p);
        assert!(
            (o.f - fc).abs() <= TOL * fc.abs().max(1.0),
            "{p:?}: f gpu {} cpu {fc}",
            o.f
        );
        assert!(
            (o.g - gc).abs() <= TOL * gc.abs().max(1.0),
            "{p:?}: g gpu {} cpu {gc}",
            o.g
        );
    }
    let origins: Vec<Vec3> = (0..400)
        .map(|i| {
            Vec3::new(
                (i % 20) as f32 * 2.0 + 0.5,
                (i / 20) as f32 * 2.0 + 0.5,
                60.0,
            )
        })
        .collect();
    let d = Vec3::new(0.05, -0.03, -1.0).normalize();
    let rays = g.march_batch(&origins, d, 120.0, 4000).unwrap();
    let mut hits = 0;
    for (o, r) in origins.iter().zip(&rays) {
        match (f.march(*o, d, 120.0, 4000), *r) {
            (MarchResult::Hit(a), MarchResult::Hit(b)) => {
                hits += 1;
                assert!(
                    (a.t - b.t).abs() <= 1e-3 && a.sheet == b.sheet,
                    "{o:?}: cpu {} #{} gpu {} #{}",
                    a.t,
                    a.sheet,
                    b.t,
                    b.sheet
                );
            }
            (a, b) => assert_eq!(
                std::mem::discriminant(&a),
                std::mem::discriminant(&b),
                "{o:?}: cpu {a:?} gpu {b:?}"
            ),
        }
    }
    assert!(hits > 20, "hits {hits}");
}

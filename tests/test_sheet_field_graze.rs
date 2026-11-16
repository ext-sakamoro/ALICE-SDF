//! Grazes: a ray that cuts a sheet over a chord shorter than the march's floor step (`MIN_STEP`, 0.01) has `g` of
//! the same sign at both ends of the step that contains the chord, so a march that only compares the step's ends
//! misses it. The dip search (module docs) finds it: a step whose ends satisfy `|s_a| + |s_b| <= L (t_b - t_a)`
//! is searched, and no other step can hold a zero.
//!
//! Oracle (not the march): a fine scan of the stored field along each ray, `g` sampled every 1e-5 (a hundredth of
//! the chords below), the first sign change bisected. The scene is a thin cylinder (radius R = 0.02, axis along y,
//! cells of 0.005) and vertical rays at offset x0 from the axis, chosen around the grazing range:
//! - every ray: the march's hit / miss equals the scan's, and a hit lies within 1e-4 of the scan's crossing;
//! - the scene does test the claim: at least 6 rays cross with a chord (entry to exit, by the scan) shorter than
//!   `MIN_STEP`, and at least 6 rays miss.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule};
use glam::Vec3;

const R: f32 = 0.02;
const H: f32 = 0.005;

fn cylinder() -> SheetField {
    let lo = Vec3::new(-0.04 + 0.0013, -0.04 + 0.0021, -0.04 + 0.0007);
    let lv = level_from_fn(
        lo,
        H,
        [2, 2, 2],
        |_c, _e| true,
        |p| vec![(p.x * p.x + p.z * p.z).sqrt() - R, -1.0],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// x0 for vertical rays starting at z = 0.035: around the radius, from chords of about 0.012 to clear misses
fn rays() -> Vec<f32> {
    (0..48).map(|i| R - 0.0012 + 0.00005 * i as f32).collect()
}

/// The scan: first sign change of the sheet's `g` along the ray (step 1e-5), bisected; and the chord to the next one
fn scan(f: &SheetField, o: Vec3, tmax: f32) -> Option<(f32, f32)> {
    let sh = &f.sheets[0];
    let g = |t: f32| sh.sample(o - Vec3::Z * t, f.rule).g;
    let n = (tmax / 1e-5) as usize;
    let mut prev = g(0.0);
    let mut entry = None;
    for i in 1..=n {
        let t = i as f32 * 1e-5;
        let v = g(t);
        if (v < 0.0) != (prev < 0.0) {
            let (mut a, mut b, mut ga) = (t - 1e-5, t, prev);
            for _ in 0..30 {
                let m = 0.5 * (a + b);
                let gm = g(m);
                if (gm < 0.0) == (ga < 0.0) {
                    a = m;
                    ga = gm;
                } else {
                    b = m;
                }
            }
            match entry {
                None => entry = Some(0.5 * (a + b)),
                Some(e) => return Some((e, 0.5 * (a + b) - e)),
            }
        }
        prev = v;
    }
    entry.map(|e| (e, f32::INFINITY))
}

const Z0: f32 = 0.035;

#[test]
fn grazing_chords_shorter_than_the_floor_step_are_found_and_near_misses_stay_misses() {
    let f = cylinder();
    let tmax = 0.07;
    let rs = rays();
    let refs: Vec<Option<(f32, f32)>> = rs
        .iter()
        .map(|&x0| scan(&f, Vec3::new(x0, 0.0013, Z0), tmax))
        .collect();
    let short = refs
        .iter()
        .filter(|r| r.is_some_and(|(_, c)| c < alice_sdf::sheet_field::MIN_STEP))
        .count();
    let misses = refs.iter().filter(|r| r.is_none()).count();
    assert!(
        short >= 6,
        "only {short} rays cross with a chord under MIN_STEP: the scene does not test grazes"
    );
    assert!(misses >= 6, "only {misses} rays miss");
    let check = |name: &str, x0: f32, want: Option<(f32, f32)>, m: &MarchResult| match (want, m) {
        (Some((t, _)), MarchResult::Hit(h)) => {
            assert!(
                (h.t - t).abs() < 1e-4,
                "{name} x0 {x0}: hit at t {} the scan's crossing {t}",
                h.t
            );
        }
        (None, MarchResult::Miss) => {}
        (w, m) => panic!("{name} x0 {x0}: the scan says {w:?}, the march {m:?}"),
    };
    for (&x0, &want) in rs.iter().zip(&refs) {
        let m = f.march_decided(Vec3::new(x0, 0.0013, Z0), -Vec3::Z, tmax);
        check("cpu", x0, want, &m);
    }

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
    let os: Vec<Vec3> = rs.iter().map(|&x0| Vec3::new(x0, 0.0013, Z0)).collect();
    let r = g.march_batch_decided(&os, -Vec3::Z, tmax).unwrap();
    for ((x0, want), m) in rs.iter().zip(&refs).zip(&r) {
        check("gpu", *x0, *want, m);
    }
}

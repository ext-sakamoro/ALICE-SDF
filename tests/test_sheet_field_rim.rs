//! An open sheet ends at its rim: a ray that crosses the sheet's plane outside the rim misses, one that crosses
//! inside hits, at every angle (CPU reference and GPU alike).
//!
//! Oracle (geometry, not the field's output): the square patch `|x| <= 1, |y| <= 1` on `z = 0`; `s = z`,
//! `d_b` = the signed in-plane distance to the square's boundary. Rays cross `z = 0` at `x = 1 + delta`
//! (`y = 0.2`), so the crossing is `delta` outside the rim (`delta > 0`) or inside it (`delta < 0`), from
//! `L = 1.2` away along the ray: a hit must be at `t = L`.
//!
//! - `delta` = 0.005, 0.01, 0.02, 0.05 mm outside: miss (never a hit, never undecided).
//! - `delta` = -0.005, -0.05 mm inside: hit at `t = L` within 1e-3 mm.
//!
//! Directions: across the rim outward and inward, and along the rim, at 90, 45, 10, 2 and 0.5 degrees from
//! the plane (edge-on and grazing views). `h = 0.0625` keeps the u8 rounding of `d_b` (half a step,
//! ~0.0014 mm here) below the smallest tested distance, so the expected answers are decidable from the data.
//! A march that accepts a crossing by the distance `f = hypot(s, max(d_b, 0)) < 0.02`, or stops where the
//! union `f < 0.01`, hits the rays 0.005-0.02 mm outside (the rim dilated by that tolerance).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule};
use glam::Vec3;

const L: f32 = 1.2;
const OUTSIDE: [f32; 4] = [0.005, 0.01, 0.02, 0.05];
const INSIDE: [f32; 2] = [-0.005, -0.05];
const ANGLES_DEG: [f32; 5] = [90.0, 45.0, 10.0, 2.0, 0.5];

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 1.0;
    let dy = y.abs() - 1.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn patch() -> SheetField {
    let lo = Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007);
    let lv = level_from_fn(
        lo,
        0.0625,
        [6, 6, 2],
        |_c, _e| true,
        |p| vec![p.z, square_db(p.x, p.y)],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// (label, direction) for every family and angle
fn directions() -> Vec<(String, Vec3)> {
    let mut out = Vec::new();
    for a in ANGLES_DEG {
        let (c, s) = (a.to_radians().cos(), a.to_radians().sin());
        out.push((format!("across outward {a} deg"), Vec3::new(c, 0.0, -s)));
        out.push((format!("across inward {a} deg"), Vec3::new(-c, 0.0, -s)));
        out.push((format!("along the rim {a} deg"), Vec3::new(0.0, c, -s)));
    }
    out
}

/// the ray origin that crosses z = 0 at x = 1 + delta, y = 0.2 after L along `d`
fn origin(d: Vec3, delta: f32) -> Vec3 {
    Vec3::new(1.0 + delta, 0.2, 0.0) - d * L
}

#[test]
fn rays_outside_the_rim_miss_and_inside_hit() {
    let f = patch();
    let mut checked = 0;
    let mut wrong = Vec::new();
    for (label, d) in directions() {
        for delta in OUTSIDE {
            let r = f.march(origin(d, delta), d, 2.0 * L, 4000);
            if r != MarchResult::Miss {
                wrong.push(format!("{label}, {delta} outside: {r:?}"));
            }
            checked += 1;
        }
        for delta in INSIDE {
            match f.march(origin(d, delta), d, 2.0 * L, 4000) {
                MarchResult::Hit(h) if (h.t - L).abs() <= 1e-3 && h.sheet == 0 => {}
                r => wrong.push(format!("{label}, {} inside: {r:?}", -delta)),
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 15 * 6, "rays checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} rays wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_marcher_ends_the_sheet_at_its_rim_too() {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    let f = patch();
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
    let mut checked = 0;
    let mut wrong = Vec::new();
    for (label, d) in directions() {
        let deltas: Vec<f32> = OUTSIDE.iter().chain(INSIDE.iter()).copied().collect();
        let origins: Vec<Vec3> = deltas.iter().map(|&dl| origin(d, dl)).collect();
        let got = g.march_batch(&origins, d, 2.0 * L, 4000).unwrap();
        for ((dl, o), r) in deltas.iter().zip(&origins).zip(&got) {
            let cpu = f.march(*o, d, 2.0 * L, 4000);
            let ok = if *dl > 0.0 {
                *r == MarchResult::Miss
            } else {
                matches!(r, MarchResult::Hit(h) if (h.t - L).abs() <= 1e-3 && h.sheet == 0)
            };
            let agree = match (r, &cpu) {
                (MarchResult::Hit(a), MarchResult::Hit(b)) => {
                    (a.t - b.t).abs() <= 1e-4 && a.sheet == b.sheet
                }
                (a, b) => a == b,
            };
            if !ok || !agree {
                wrong.push(format!("{label}, delta {dl}: gpu {r:?}, cpu {cpu:?}"));
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 15 * 6, "rays checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} rays wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

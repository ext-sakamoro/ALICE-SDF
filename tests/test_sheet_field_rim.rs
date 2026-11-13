//! An open sheet ends at its rim: a ray that crosses the sheet's plane outside the rim misses, one that crosses
//! inside hits, at every angle (CPU reference and GPU alike).
//!
//! Oracle (geometry, not the field's output): the square patch `|x| <= 1, |y| <= 1` on `z = 0`; `s = z`,
//! `d_b` = the signed in-plane distance to the square's boundary. Rays cross `z = 0` at `x = 1 + delta`
//! (`y = 0.2`), so the crossing is `delta` outside the rim (`delta > 0`) or inside it (`delta < 0`), from
//! `L = 1.2` away along the ray: a hit must be at `t = L`.
//!
//! - `delta` = 0.005, 0.01, 0.02, 0.05 mm outside: miss (never a hit, never undecided).
//! - `delta` = -0.005, -0.05 mm inside: hit at `t = L`, the hit point within 1e-3 mm of the plane
//!   (`|t - L| * sin(angle) <= 1e-3`: along the ray the position is ill-conditioned at grazing angles, so the
//!   oracle bounds the distance to the surface, which is what a hit has to get right).
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
/// (angle from the plane in degrees, its tangent): directions are built from the slope (the crate forbids
/// f32::sin / cos, even in tests); 90 degrees is straight down
const ANGLES: [(f32, f32); 5] = [
    (90.0, f32::INFINITY),
    (45.0, 1.0),
    (10.0, 0.176_326_98),
    (2.0, 0.034_920_77),
    (0.5, 0.008_726_867),
];

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
    for (a, m) in ANGLES {
        // unit (cos, sin) of the angle from its tangent
        let (c, s) = if m.is_infinite() {
            (0.0, 1.0)
        } else {
            let n = (1.0 + m * m).sqrt();
            (1.0 / n, m / n)
        };
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
                MarchResult::Hit(h) if (h.t - L).abs() * -d.z <= 1e-3 && h.sheet == 0 => {}
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
                matches!(r, MarchResult::Hit(h) if (h.t - L).abs() * -d.z <= 1e-3 && h.sheet == 0)
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

/// A sign jump of `s` where one level ends and another answers is not a crossing: the fine level (`x < 0`) has
/// its surface at `z = 0`, the coarse level at `z = -0.1`; a ray parallel to both at `z = -0.05` sees `s` flip
/// from -0.05 to +0.05 at the level seam but crosses no surface, so it misses. The check is `|s| <= 0.02` at the
/// bisected point (a ramp inside one level passes through 0 and is a true crossing; a jump between levels
/// stays at |s| ~ 0.05). A ray that does cross the fine surface still hits.
#[test]
fn a_sign_jump_between_levels_is_not_a_crossing() {
    let fine = level_from_fn(
        Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007),
        0.0625,
        [3, 6, 2],
        |_c, _e| true,
        |p| vec![p.z, square_db(p.x, p.y)],
    )
    .unwrap();
    let coarse = level_from_fn(
        Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007),
        0.125,
        [3, 3, 1],
        |_c, _e| true,
        |p| vec![p.z + 0.1, square_db(p.x, p.y)],
    )
    .unwrap();
    let f = SheetField::new(vec![sheet(vec![fine, coarse]).unwrap()], ValidRule::CellV1);
    let parallel = f.march(Vec3::new(-0.8, 0.2, -0.05), Vec3::X, 1.2, 4000);
    assert_eq!(
        parallel,
        MarchResult::Miss,
        "the level seam's sign jump read as a hit"
    );
    match f.march(Vec3::new(-0.5, 0.2, 0.3), -Vec3::Z, 1.0, 4000) {
        MarchResult::Hit(h) => assert!((h.t - 0.3).abs() <= 1e-3, "t {}", h.t),
        r => panic!("a ray through the fine surface: {r:?}"),
    }
    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        match GpuSheetField::new(&f) {
            Ok(g) => {
                let par = g
                    .march_batch(&[Vec3::new(-0.8, 0.2, -0.05)], Vec3::X, 1.2, 4000)
                    .unwrap();
                assert_eq!(
                    par[0],
                    MarchResult::Miss,
                    "gpu: the level seam's sign jump read as a hit"
                );
                let thr = g
                    .march_batch(&[Vec3::new(-0.5, 0.2, 0.3)], -Vec3::Z, 1.0, 4000)
                    .unwrap();
                assert!(
                    matches!(thr[0], MarchResult::Hit(h) if (h.t - 0.3).abs() <= 1e-3),
                    "gpu: {:?}",
                    thr[0]
                );
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
}

/// Near its rim the sheet is still flat: a hit 0.005 / 0.01 / 0.03 mm inside the rim (within the 0.05 mm
/// differencing reach of the normal) has the plane's normal `+z` (the direction `s` grows), `dot >= 0.9999`,
/// for every direction, CPU and GPU. Oracle: the plane `z = 0`. A normal differenced from the distance
/// (`g = sign(s) hypot(s, max(d_b, 0))`) bends toward the rim there, because `d_b > 0` on the outer half of
/// the stencil; the surface normal is the gradient of `s` alone.
#[test]
fn the_normal_near_the_rim_is_the_plane_normal() {
    let f = patch();
    #[cfg(feature = "gpu")]
    let g = alice_sdf::sheet_field::gpu::GpuSheetField::new(&f).ok();
    #[cfg(feature = "gpu")]
    if g.is_none() {
        assert!(
            std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
            "ALICE_SDF_REQUIRE_GPU set but no GPU"
        );
    }
    let deltas = [-0.005f32, -0.01, -0.03];
    let mut checked = 0;
    let mut wrong = Vec::new();
    for (label, d) in directions() {
        let origins: Vec<Vec3> = deltas.iter().map(|&dl| origin(d, dl)).collect();
        // pushed to only with the gpu feature
        #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
        let mut results: Vec<(&str, Vec<MarchResult>)> = vec![(
            "cpu",
            origins
                .iter()
                .map(|o| f.march(*o, d, 2.0 * L, 4000))
                .collect(),
        )];
        #[cfg(feature = "gpu")]
        if let Some(g) = &g {
            results.push(("gpu", g.march_batch(&origins, d, 2.0 * L, 4000).unwrap()));
        }
        for (who, rs) in &results {
            for (dl, r) in deltas.iter().zip(rs) {
                match r {
                    MarchResult::Hit(h) if h.normal.z >= 0.9999 => {}
                    r => wrong.push(format!("{who} {label}, {} inside: {r:?}", -dl)),
                }
                checked += 1;
            }
        }
    }
    assert!(checked >= 15 * 3, "rays checked {checked}");
    assert!(
        wrong.is_empty(),
        "{} of {checked} normals wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

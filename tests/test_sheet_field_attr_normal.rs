//! Author-normal channels: a sheet built with 7 channels (`s`, `d_b`, `u`, `v`, `nx`, `ny`, `nz`) returns, at a
//! hit, the stored normal of the surface point (normalized) as `Hit::attr_normal`, CPU and GPU alike; with 2 or 4
//! channels it returns `None`. The geometric normal (`Hit::normal`, the gradient of `s`) is a different quantity:
//! an author normal can tilt far from it, and a renderer that shades like the source mesh needs the author's.
//!
//! Oracle: the square patch `z = 0` with the author normal field `n(p) = (0.2 x, -0.1 y, 1)` (linear, so trilinear
//! interpolation reproduces it up to u8 rounding), tilted up to ~17 degrees from `+z` across the patch. At the hit
//! of a ray straight down at `(x, y)`, `attr_normal . normalize(n(x, y, 0)) >= 0.9999`.
//! Channel counts other than 2, 4 or 7 are refused when the level is built.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule};
use glam::Vec3;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 1.0;
    let dy = y.abs() - 1.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn author(x: f32, y: f32) -> Vec3 {
    Vec3::new(0.2 * x, -0.1 * y, 1.0)
}

fn patch(nch: usize) -> SheetField {
    let lo = Vec3::new(-1.5 + 0.013, -1.5 + 0.021, -0.5 + 0.007);
    let lv = level_from_fn(
        lo,
        0.0625,
        [6, 6, 2],
        |_c, _e| true,
        move |p| {
            let n = author(p.x, p.y);
            let all = [
                p.z,
                square_db(p.x, p.y),
                0.5 + 0.01 * p.x,
                0.25 + 0.02 * p.y,
                n.x,
                n.y,
                n.z,
            ];
            all[..nch].to_vec()
        },
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// ray origins straight above a grid of points inside the patch (not too close to the rim)
fn origins() -> Vec<Vec3> {
    let mut o = Vec::new();
    for i in 0..9 {
        for j in 0..9 {
            o.push(Vec3::new(
                -0.8 + 0.2 * i as f32 + 0.013,
                -0.8 + 0.2 * j as f32 + 0.007,
                0.4,
            ));
        }
    }
    o
}

fn check(label: &str, rs: &[MarchResult], os: &[Vec3], wrong: &mut Vec<String>) -> usize {
    let mut checked = 0;
    for (o, r) in os.iter().zip(rs) {
        let want = author(o.x, o.y).normalize();
        match r {
            MarchResult::Hit(h) => match h.attr_normal {
                Some(n) if n.dot(want) >= 0.9999 && (n.length() - 1.0).abs() <= 1e-5 => {}
                other => wrong.push(format!(
                    "{label} {o:?}: attr_normal {other:?}, want {want:?}"
                )),
            },
            other => wrong.push(format!("{label} {o:?}: {other:?}")),
        }
        checked += 1;
    }
    checked
}

#[test]
fn a_seven_channel_sheet_returns_the_author_normal_at_the_hit() {
    let f = patch(7);
    let os = origins();
    let mut wrong = Vec::new();
    let cpu: Vec<MarchResult> = os
        .iter()
        .map(|o| f.march(*o, -Vec3::Z, 1.0, 4000))
        .collect();
    // added to only with the gpu feature
    #[cfg_attr(not(feature = "gpu"), allow(unused_mut))]
    let mut checked = check("cpu", &cpu, &os, &mut wrong);
    #[cfg(feature = "gpu")]
    match alice_sdf::sheet_field::gpu::GpuSheetField::new(&f) {
        Ok(g) => {
            let gpu = g.march_batch(&os, -Vec3::Z, 1.0, 4000).unwrap();
            checked += check("gpu", &gpu, &os, &mut wrong);
        }
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
        }
    }
    assert!(checked >= 81, "checked {checked}");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

#[test]
fn two_or_four_channels_carry_no_author_normal() {
    for nch in [2, 4] {
        let f = patch(nch);
        let o = Vec3::new(0.113, 0.207, 0.4);
        match f.march(o, -Vec3::Z, 1.0, 4000) {
            MarchResult::Hit(h) => assert_eq!(h.attr_normal, None, "{nch} channels"),
            r => panic!("{nch} channels: {r:?}"),
        }
        #[cfg(feature = "gpu")]
        if let Ok(g) = alice_sdf::sheet_field::gpu::GpuSheetField::new(&f) {
            match g.march_batch(&[o], -Vec3::Z, 1.0, 4000).unwrap()[0] {
                MarchResult::Hit(h) => assert_eq!(h.attr_normal, None, "gpu, {nch} channels"),
                r => panic!("gpu, {nch} channels: {r:?}"),
            }
        }
    }
}

#[test]
fn other_channel_counts_are_refused() {
    for nch in [1usize, 3, 5, 6, 8] {
        let r = level_from_fn(
            Vec3::ZERO,
            0.5,
            [1, 1, 1],
            |_c, _e| true,
            move |p| vec![p.z; nch],
        );
        assert!(r.is_err(), "{nch} channels accepted");
    }
}

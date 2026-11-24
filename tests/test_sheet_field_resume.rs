//! A march ray that runs out of its slice (the per-dispatch work budget) after completing a step continues in a
//! later dispatch from its last completed step boundary, and ends with the result of the same ray marched in one
//! dispatch.
//!
//! - one step per dispatch: every ray stopped at every step boundary and continued ends with the one-shot result
//!   bit for bit (kind, t, sheet, normal and attributes), and its work summed over the slices equals the one-shot
//!   work. The reference is the one-shot march of the same field (budget raised to the cap with the test hook, so the
//!   grazing rays decide in one dispatch); the scenes are the 100-sheet stack with head-on, oblique and grazing rays.
//! - the slice limit: a ray that needs more dispatches than the limit ends undecided after exactly that many.
//! - progress: a dispatch in which a ray completes no step returns it undecided, never continued (no ray is resumed
//!   without progress).
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{GpuSheetField, MARCH_WORK_BUDGET_CAP};
use alice_sdf::sheet_field::{decided_steps, MarchResult, Sheet, SheetField, ValidRule};
use glam::Vec3;
use std::time::Duration;

fn square(z: f32, half: f32) -> Sheet {
    let lo = Vec3::new(-8.0 + 0.013, -8.0 + 0.021, z - 1.0 + 0.007);
    let lv = level_from_fn(
        lo,
        0.25,
        [8, 8, 1],
        |_c, _e| true,
        |p| {
            let (dx, dy) = (p.x.abs() - half, p.y.abs() - half);
            vec![
                p.z - z,
                Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0),
            ]
        },
    )
    .unwrap();
    sheet(vec![lv]).unwrap()
}

/// The 100-sheet stack: sheets 0.01 apart, every box overlapping
fn deep() -> SheetField {
    SheetField::new(
        (0..100).map(|k| square(0.01 * k as f32, 5.0)).collect(),
        ValidRule::CellV1,
    )
}

/// Origins and directions: x = -7, about 1e-3 downward in z, so each ray crosses the stack over many metres
const RAYS: &[([f32; 3], [f32; 3])] = &[
    (
        [-7.0, 1.0167718, 0.07286668],
        [0.932739, -0.36055, -0.0013063821],
    ),
    (
        [-7.0, -2.5520484, 0.6362223],
        [0.932739, -0.36055, -0.0013063821],
    ),
    (
        [-7.0, 4.0448155, 0.64838463],
        [0.96755993, -0.2526406, -0.00064476667],
    ),
    (
        [-7.0, -3.7878048, 0.75843745],
        [0.96755993, -0.2526406, -0.00064476667],
    ),
    (
        [-7.0, 0.24888378, 0.4050771],
        [0.9789988, 0.20386113, -0.0014044347],
    ),
    (
        [-7.0, -1.6487458, 0.51086],
        [0.9789988, 0.20386113, -0.0014044347],
    ),
    (
        [-7.0, 0.97896713, 0.70958877],
        [0.89985186, -0.43619442, -0.0010622086],
    ),
    (
        [-7.0, -0.9220914, 0.19609874],
        [0.89985186, -0.43619442, -0.0010622086],
    ),
    (
        [-7.0, -0.3282246, 0.8415045],
        [0.99520856, 0.09777014, -0.00095463154],
    ),
    (
        [-7.0, 3.4949188, 0.7750247],
        [0.99520856, 0.09777014, -0.00095463154],
    ),
    (
        [-7.0, 4.3239746, 0.71438664],
        [0.9622284, -0.27224204, -0.0009456583],
    ),
    (
        [-7.0, 0.9472055, 0.011855185],
        [0.9622284, -0.27224204, -0.0009456583],
    ),
    (
        [-7.0, -1.0421, 0.7018742],
        [0.9913312, 0.1313752, -0.0017245769],
    ),
    (
        [-7.0, 1.5148505, 0.1962673],
        [0.9913312, 0.1313752, -0.0017245769],
    ),
    (
        [-7.0, 3.9529083, 0.64766914],
        [0.9500032, -0.31223944, -0.00053506216],
    ),
    (
        [-7.0, 4.476365, 0.10405451],
        [0.9500032, -0.31223944, -0.00053506216],
    ),
    (
        [-7.0, 0.89213866, 0.59471625],
        [0.94471765, -0.32787988, -0.001834236],
    ),
    (
        [-7.0, 0.32139838, 0.967318],
        [0.94471765, -0.32787988, -0.001834236],
    ),
    (
        [-7.0, -4.236829, 0.4610837],
        [0.9976164, -0.06899924, -0.00085431413],
    ),
    (
        [-7.0, -0.33217388, 0.5610703],
        [0.9976164, -0.06899924, -0.00085431413],
    ),
    (
        [-7.0, -1.3020275, 0.99693227],
        [0.9940509, 0.10889911, -0.0019064717],
    ),
    (
        [-7.0, 0.5988756, 0.5664503],
        [0.9940509, 0.10889911, -0.0019064717],
    ),
    (
        [-7.0, 1.039445, 0.44205046],
        [0.9918886, -0.12710525, -0.001202514],
    ),
    (
        [-7.0, -1.283334, 0.9619782],
        [0.9918886, -0.12710525, -0.001202514],
    ),
    (
        [-7.0, -0.44085377, 0.49510592],
        [0.94426864, 0.32917544, -0.00050643896],
    ),
    (
        [-7.0, 3.6062777, 0.9360616],
        [0.94426864, 0.32917544, -0.00050643896],
    ),
    (
        [-7.0, 3.9605937, 0.76803005],
        [0.95365673, -0.30089584, -0.0007404223],
    ),
    (
        [-7.0, 4.3850417, 0.70042473],
        [0.95365673, -0.30089584, -0.0007404223],
    ),
    (
        [-7.0, 2.2779963, 0.31402582],
        [0.933188, 0.3593868, -0.0011174774],
    ),
    (
        [-7.0, -1.4636831, 0.12301403],
        [0.933188, 0.3593868, -0.0011174774],
    ),
    (
        [-7.0, -0.000908196, 0.96639264],
        [0.8955334, -0.44499296, -0.0011708236],
    ),
    (
        [-7.0, -3.0389085, 0.67602223],
        [0.8955334, -0.44499296, -0.0011708236],
    ),
];
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

/// The result as bits: kind, then t, sheet, normal, uv and attribute normal of a hit
fn bits(m: &MarchResult) -> (u8, Vec<u32>) {
    match m {
        MarchResult::Hit(h) => {
            let mut v = vec![h.t.to_bits(), h.sheet as u32];
            v.extend(h.normal.to_array().map(f32::to_bits));
            v.extend(h.uv.map_or([u32::MAX; 2], |u| u.map(f32::to_bits)));
            v.extend(
                h.attr_normal
                    .map_or([u32::MAX; 3], |n| n.to_array().map(f32::to_bits)),
            );
            (1, v)
        }
        MarchResult::Miss => (0, Vec::new()),
        MarchResult::Undecided => (2, Vec::new()),
    }
}

/// The one-shot reference and the one-step slices of one set of rays (all in one direction per call)
fn compare(f: &SheetField, name: &str, set: &[(Vec3, Vec3)], one_step: bool) -> usize {
    let (Some(one), Some(sliced)) = (gpu(f), gpu(f)) else {
        return 0;
    };
    one.set_dispatch_wait(Duration::from_secs(30));
    sliced.set_dispatch_wait(Duration::from_secs(30));
    one.set_work_budget_up_to_cap(MARCH_WORK_BUDGET_CAP)
        .unwrap();
    if one_step {
        sliced.set_slice_steps(1);
        sliced.set_max_march_slices(u32::MAX);
    }
    let steps = decided_steps(30.0).unwrap();
    let mut compared = 0;
    for (k, (o, d)) in set.iter().enumerate() {
        let a = one.march_batch(&[*o], *d, 30.0, steps).unwrap();
        let sa = one.last_march_stats();
        assert_eq!(
            sa.ray_slices,
            vec![1],
            "{name} ray {k}: the reference took more than one dispatch"
        );
        assert!(
            !matches!(a[0], MarchResult::Undecided),
            "{name} ray {k}: the reference is undecided"
        );
        let b = sliced.march_batch(&[*o], *d, 30.0, steps).unwrap();
        let sb = sliced.last_march_stats();
        assert_eq!(
            bits(&b[0]),
            bits(&a[0]),
            "{name} ray {k} at {o}: sliced {:?}, one-shot {:?}",
            b[0],
            a[0]
        );
        assert_eq!(
            sb.ray_work, sa.ray_work,
            "{name} ray {k}: summed slice work differs from the one-shot work"
        );
        assert_eq!(
            u64::from(sb.ray_slices[0]) - 1,
            sb.resumed_rays,
            "{name} ray {k}"
        );
        compared += 1;
    }
    eprintln!("{name}: {compared} rays equal bit for bit (one step per dispatch: {one_step})");
    compared
}

#[test]
fn a_ray_continued_after_every_step_ends_with_the_one_shot_result_and_work() {
    let f = deep();
    let head: Vec<(Vec3, Vec3)> = (0..24)
        .map(|i| {
            (
                Vec3::new(-6.0 + 0.5 * i as f32, -4.0 + 0.37 * i as f32, 5.0),
                -Vec3::Z,
            )
        })
        .collect();
    let oblique: Vec<(Vec3, Vec3)> = (0..24)
        .map(|i| {
            (
                Vec3::new(-6.0 + 0.5 * i as f32, -4.0 + 0.37 * i as f32, 5.0),
                Vec3::new(0.3, 0.2, -1.0).normalize(),
            )
        })
        .collect();
    let graze: Vec<(Vec3, Vec3)> = RAYS
        .iter()
        .take(4)
        .map(|(o, d)| (Vec3::from_array(*o), Vec3::from_array(*d)))
        .collect();
    let n = compare(&f, "head-on", &head, true)
        + compare(&f, "oblique", &oblique, true)
        + compare(&f, "graze", &graze, true);
    if n == 0 {
        return;
    }
    assert_eq!(n, 52);
}

/// The default slice (the work budget, no step limit): a ray that runs out of it in the middle of a step is continued
/// from the boundary before that step, and ends with the one-shot result and work. A boundary taken after a cut step
/// (its crossing check or list copy cut short) would lose work or a crossing here.
#[test]
fn a_ray_out_of_its_slice_mid_step_ends_with_the_one_shot_result_and_work() {
    let f = deep();
    let graze: Vec<(Vec3, Vec3)> = RAYS
        .iter()
        .map(|(o, d)| (Vec3::from_array(*o), Vec3::from_array(*d)))
        .collect();
    let n = compare(&f, "graze at the default slice", &graze, false);
    if n == 0 {
        return;
    }
    assert_eq!(n, RAYS.len());
}

#[test]
fn a_ray_that_needs_more_slices_than_the_limit_ends_undecided_after_exactly_that_many() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    g.set_max_march_slices(2);
    // the ray of the set with the most work (9,045,629): over 4 slices, so 2 cannot decide it
    let (o, d) = RAYS[9];
    let r = g
        .march_batch(
            &[Vec3::from_array(o)],
            Vec3::from_array(d),
            30.0,
            decided_steps(30.0).unwrap(),
        )
        .unwrap();
    let st = g.last_march_stats();
    assert_eq!(r[0], MarchResult::Undecided);
    assert_eq!(st.ray_slices, vec![2]);
    assert_eq!(st.over_slices, 1);
    assert_eq!(st.resumed_rays, 1);
}

#[test]
fn a_dispatch_in_which_a_ray_completes_no_step_returns_it_undecided_not_continued() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    // the 100 live sheets at the origin take more than this before the first step completes
    g.set_work_budget(50).unwrap();
    let r = g
        .march_batch(
            &[Vec3::new(0.3, 0.2, 0.5)],
            -Vec3::Z,
            10.0,
            decided_steps(10.0).unwrap(),
        )
        .unwrap();
    let st = g.last_march_stats();
    assert_eq!(r[0], MarchResult::Undecided);
    assert_eq!(st.ray_slices, vec![1]);
    assert_eq!(st.resumed_rays, 0);
    assert_eq!(st.over_budget, 1);
}

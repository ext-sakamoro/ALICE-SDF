//! Rays grazing a deep stack of sheets need more work than the default per-ray budget ([`MARCH_WORK_BUDGET`]): today
//! the GPU march returns them undecided where the CPU march decides them.
//!
//! The scene is the 100-sheet flat stack of the dispatch tests (sheets 0.01 apart, every box overlapping) and 32 rays
//! that enter it nearly parallel to the sheets. The decision reference is the CPU march (`SheetField::march`), which
//! has no work budget; the work count is the GPU's own (the per-ray work in `last_march_stats`).
//!
//! - the measured work: with the budget raised to [`MARCH_WORK_BUDGET_CAP`] (a test hook, under a 30 s wait bound so
//!   the device speed rule still holds) every ray decides, and its work equals the recorded table exactly. The work
//!   is a count, so it is the same on every device: a device that differs is a finding, not a tolerance. Every entry
//!   is over the default budget.
//!   A device too slow to run one ray at the cap within a third of 30 s runs the cheapest rays instead (at most 2,
//!   under a wait raised for them up to 600 s) and asserts their work exactly; one that cannot run even those fails
//!   under `ALICE_SDF_REQUIRE_GPU`. The test prints how many rays it compared, and the CI step checks that line.
//! - the decision: the same kind and sheet as the CPU march, and `t` within 1e-3 (the parity rule of
//!   `test_sheet_field_gpu_parity.rs`; the CPU and GPU `t` differ in the float arithmetic, measured at most 8.13e-5 on
//!   the A6000). The GPU `t` bits are printed per ray so devices can be compared.
//! - the open defect, pinned and ignored: at the default budget the GPU decides the same as the CPU march. It does not
//!   today (every one of these rays is undecided); a march that resumes over dispatches is the planned fix, and the
//!   ignore is listed in the oracle status ledger as open work.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{GpuSheetField, MARCH_WORK_BUDGET, MARCH_WORK_BUDGET_CAP};
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
/// The GPU's work per ray at [`MARCH_WORK_BUDGET_CAP`], in the order of `RAYS` (measured on the A6000, 2026-11-24)
const T_TABLE: [u32; 32] = [
    3783143, 4882709, 3942006, 3261072, 4623273, 2489785, 2697860, 6459483, 4636166, 9045629,
    6853053, 4912660, 2258361, 4060834, 4005774, 8301422, 3117975, 4095192, 6513761, 4967001,
    3802124, 3649921, 3809246, 3732550, 8377420, 2568863, 2932548, 5240046, 5411894, 4708048,
    6934871, 2563007,
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

/// The parity rule: the same kind, the same sheet, and `t` within 1e-3
fn same_decision(gpu: &MarchResult, cpu: &MarchResult) -> bool {
    match (gpu, cpu) {
        (MarchResult::Hit(a), MarchResult::Hit(b)) => {
            a.sheet == b.sheet && (a.t - b.t).abs() <= 1e-3
        }
        (MarchResult::Miss, MarchResult::Miss) => true,
        _ => false,
    }
}

#[test]
fn grazing_rays_decide_at_the_cap_with_the_recorded_work_and_the_cpu_decision() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    // the rays this device runs: all 32 when one ray at the cap fits a third of a 30 s wait; on a slower device the
    // cheapest rays, under a wait raised for them (at most 600 s, at most 2 rays). Every ray that runs is asserted
    // exactly, so a slow lane still checks the work count against the table
    let ns = g.ns_per_tick();
    let full_s = ns * f64::from(MARCH_WORK_BUDGET_CAP) * 1e-9;
    let (wait_s, budget, chosen): (f64, u32, Vec<usize>) = if full_s * 3.0 <= 30.0 {
        (30.0, MARCH_WORK_BUDGET_CAP, (0..RAYS.len()).collect())
    } else {
        let mut order: Vec<usize> = (0..RAYS.len()).collect();
        order.sort_by_key(|&i| T_TABLE[i]);
        let chosen: Vec<usize> = order
            .into_iter()
            .take(2)
            .filter(|&i| ns * f64::from(T_TABLE[i]) * 1e-9 * 3.3 <= 600.0)
            .collect();
        let budget = chosen.iter().map(|&i| T_TABLE[i]).max().unwrap_or(0);
        (
            (ns * f64::from(budget) * 1e-9 * 3.3).max(30.0),
            budget,
            chosen,
        )
    };
    if chosen.is_empty() {
        assert!(
            std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
            "ALICE_SDF_REQUIRE_GPU set but the device ({ns} ns per tick) cannot run even the cheapest grazing ray within 600 s"
        );
        eprintln!("gpu-test: not run (device too slow: {ns} ns per tick)");
        return;
    }
    g.set_dispatch_wait(Duration::from_secs_f64(wait_s));
    if let Err(e) = g.set_work_budget_up_to_cap(budget) {
        panic!(
            "the cap hook refused budget {budget} under a {wait_s} s wait ({ns} ns per tick): {e}"
        );
    }
    let steps = decided_steps(30.0).unwrap();
    let (mut compared, mut max_dt) = (0, 0.0f32);
    for &i in &chosen {
        let (o, d) = RAYS[i];
        let (o, d) = (Vec3::from_array(o), Vec3::from_array(d));
        let r = g.march_batch(&[o], d, 30.0, steps).unwrap();
        let work = g.last_march_stats().max_work;
        let cpu = f.march(o, d, 30.0, steps);
        if let (MarchResult::Hit(a), MarchResult::Hit(b)) = (&r[0], &cpu) {
            max_dt = max_dt.max((a.t - b.t).abs());
            eprintln!(
                "ray {i}: work {work} gpu t bits {:#010x} ({}) cpu t {}",
                a.t.to_bits(),
                a.t,
                b.t
            );
        } else {
            eprintln!("ray {i}: work {work} gpu {:?} cpu {:?}", r[0], cpu);
        }
        assert!(
            !matches!(r[0], MarchResult::Undecided),
            "ray {i} at {o}: undecided at the cap (work {work})"
        );
        assert!(
            same_decision(&r[0], &cpu),
            "ray {i} at {o}: GPU {:?}, CPU {:?}",
            r[0],
            cpu
        );
        assert_eq!(
            work, T_TABLE[i],
            "ray {i} at {o}: the work differs from the recorded table"
        );
        assert!(
            work > MARCH_WORK_BUDGET,
            "ray {i}: work {work} is within the default budget"
        );
        compared += 1;
    }
    eprintln!(
        "compared {compared} rays of {}, max |t gpu - t cpu| {max_dt:e}, {ns} ns per tick",
        RAYS.len()
    );
    assert_eq!(compared, chosen.len());
    assert!(compared > 0);
}

#[test]
#[ignore = "known defect: sheet-field-grazing-budget"]
fn grazing_rays_decide_at_the_default_budget_as_on_the_cpu() {
    let f = deep();
    let Some(g) = gpu(&f) else { return };
    let steps = decided_steps(30.0).unwrap();
    let mut compared = 0;
    for (i, (o, d)) in RAYS.iter().enumerate() {
        let (o, d) = (Vec3::from_array(*o), Vec3::from_array(*d));
        let r = g.march_batch(&[o], d, 30.0, steps).unwrap();
        let cpu = f.march(o, d, 30.0, steps);
        assert!(
            same_decision(&r[0], &cpu),
            "ray {i} at {o}: GPU {:?}, CPU {:?}",
            r[0],
            cpu
        );
        compared += 1;
    }
    assert_eq!(compared, RAYS.len());
}

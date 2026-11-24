//! The GPU march's per-ray work budget is one fixed number on every device ([`MARCH_WORK_BUDGET`]): a device too
//! slow to run one ray at that budget within a third of the dispatch wait bound is refused when a field is uploaded,
//! never given a smaller budget, so a ray's result does not depend on the device that marched it.
//!
//! Oracles (geometry and arithmetic, not the march):
//! - the refusal edge is `ns_per_tick x MARCH_WORK_BUDGET <= DISPATCH_WAIT / 3`, computed here from the constants;
//! - a ray grazing the flat sheet `z = 0` (`|x|, |y| <= 500`) at height 0.02 never meets it, so its answer is a miss;
//!   grazing about 1000 mm it needs more work than the budget (about 3,000 ticks per mm, the speed probe's sheet),
//!   so the GPU ends it undecided and counts it over budget, while a graze of about 500 mm stays under the budget and
//!   ends a miss. Both are marched on the same device admitted as a fast one and as a slow one: the results are
//!   identical.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::compiled::GpuError;
use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{check_device_speed, DISPATCH_WAIT, MARCH_WORK_BUDGET};
use alice_sdf::sheet_field::{SheetField, ValidRule};
use glam::Vec3;

const HALF: f32 = 500.0;

fn flat() -> SheetField {
    let h = 0.25;
    let nbx = ((2.0 * HALF + 8.0) / (8.0 * h)).ceil() as u32;
    let lo = Vec3::new(-HALF - 4.0 + 0.013, -2.0 + 0.021, -1.0 + 0.007);
    let db = |x: f32, y: f32| {
        let dx = x.abs() - HALF;
        let dy = y.abs() - HALF;
        Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
    };
    let lv = level_from_fn(
        lo,
        h,
        [nbx, 2, 1],
        |_c, _e| true,
        move |p| vec![p.z, db(p.x, p.y)],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

/// The ns per tick at which one ray at the full budget takes exactly a third of the wait bound.
fn edge_ns() -> f64 {
    DISPATCH_WAIT.as_secs_f64() / 3.0 / f64::from(MARCH_WORK_BUDGET) * 1e9
}

#[test]
fn a_device_too_slow_for_the_budget_is_refused_with_both_times() {
    let edge = edge_ns();
    assert!(check_device_speed(edge * 0.999, DISPATCH_WAIT).is_ok());
    match check_device_speed(edge * 1.001, DISPATCH_WAIT) {
        Err(GpuError::DeviceTooSlow {
            ns_per_tick,
            budget,
            budget_ms,
            limit_ms,
        }) => {
            assert_eq!(budget, MARCH_WORK_BUDGET);
            assert!((ns_per_tick - edge * 1.001).abs() < 1e-9);
            // a third of the 4 s wait bound, and the full budget's time just over it
            assert_eq!(
                limit_ms,
                (DISPATCH_WAIT.as_secs_f64() / 3.0 * 1e3).floor() as u64
            );
            assert!(budget_ms > limit_ms, "{budget_ms} {limit_ms}");
        }
        r => panic!("{r:?}"),
    }
    // a speed that was not measured is refused, not admitted
    for ns in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                check_device_speed(ns, DISPATCH_WAIT),
                Err(GpuError::InvalidFormat(_))
            ),
            "{ns}"
        );
    }
    // the upload refuses the slow device before it builds anything (no GPU needed for the refusal)
    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        let r = GpuSheetField::new_with_ns_per_tick(&flat(), edge * 2.0);
        assert!(
            matches!(r, Err(GpuError::DeviceTooSlow { .. })),
            "{:?}",
            r.err()
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
fn the_budget_and_the_results_do_not_follow_the_device_speed() {
    use alice_sdf::sheet_field::gpu::GpuSheetField;
    use alice_sdf::sheet_field::MarchResult;
    let f = flat();
    let o = [Vec3::new(-HALF - 1.0, 0.003, 0.02)];
    let mut runs = Vec::new();
    // a fast device and a slow one still within the edge
    for ns in [1.0, edge_ns() * 0.95] {
        let g = match GpuSheetField::new_with_ns_per_tick(&f, ns) {
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
        assert_eq!(g.ns_per_tick(), ns);
        // about 500 mm of graze: under the budget, a miss
        let mid = g.march_batch_decided(&o, Vec3::X, 500.0).unwrap();
        let mid_st = g.last_march_stats();
        // about 1000 mm of graze: over the budget, undecided
        let long = g
            .march_batch_decided(&o, Vec3::X, 2.0 * HALF + 2.0)
            .unwrap();
        let long_st = g.last_march_stats();
        eprintln!(
            "ns/tick {ns}: mid work {} {:?}, long work {} {:?}",
            mid_st.max_work, mid, long_st.max_work, long
        );
        assert_eq!(mid, vec![MarchResult::Miss], "ns {ns}");
        // the mid ray is near the budget: a budget scaled down by half would end it undecided
        assert!(
            mid_st.max_work > MARCH_WORK_BUDGET / 2 && mid_st.max_work <= MARCH_WORK_BUDGET,
            "ns {ns}: mid work {}",
            mid_st.max_work
        );
        assert_eq!(long, vec![MarchResult::Undecided], "ns {ns}");
        assert_eq!(long_st.over_budget, 1, "ns {ns}");
        // the long ray's own work is under twice the budget: a budget scaled up by two would decide it
        assert!(long_st.max_work > MARCH_WORK_BUDGET, "ns {ns}");
        runs.push((mid, long));
    }
    assert_eq!(
        runs[0], runs[1],
        "the fast and the slow device gave different results"
    );
    // the CPU, with no work budget, ends the long graze a miss: the geometry's answer
    assert_eq!(
        f.march_decided(o[0], Vec3::X, 2.0 * HALF + 2.0).unwrap(),
        MarchResult::Miss
    );
}

/// The device-speed probe: prints this device's march time per tick and the full budget's time (the CI's software
/// adapter step reads these to see whether that adapter fits the wait bound). The upload measures the same way.
#[cfg(feature = "gpu")]
#[test]
fn the_device_speed_probe_reports_ns_per_tick() {
    use alice_sdf::sheet_field::gpu::{measure_ns_per_tick, GpuSheetField};
    let ns = match measure_ns_per_tick() {
        Ok(ns) => ns,
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "ALICE_SDF_REQUIRE_GPU set but no GPU: {e}"
            );
            eprintln!("gpu-test: skipped (no GPU: {e})");
            return;
        }
    };
    eprintln!(
        "device speed: {ns:.2} ns per tick, full budget {:.3} s, limit {:.3} s",
        ns * f64::from(MARCH_WORK_BUDGET) * 1e-9,
        DISPATCH_WAIT.as_secs_f64() / 3.0
    );
    assert!(ns.is_finite() && ns > 0.0, "{ns}");
    match GpuSheetField::new(&flat()) {
        Ok(g) => assert!(g.ns_per_tick().is_finite() && g.ns_per_tick() > 0.0),
        Err(e) => panic!("upload on this device: {e}"),
    }
}

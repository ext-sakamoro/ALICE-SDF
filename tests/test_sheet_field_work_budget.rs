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
        // about 1000 mm of graze: over the budget, so it is continued in a second dispatch and decided
        let long = g
            .march_batch_decided(&o, Vec3::X, 2.0 * HALF + 2.0)
            .unwrap();
        let long_st = g.last_march_stats();
        // and with one dispatch a ray, the same ray ends undecided
        g.set_max_march_slices(1);
        let cut = g
            .march_batch_decided(&o, Vec3::X, 2.0 * HALF + 2.0)
            .unwrap();
        let cut_st = g.last_march_stats();
        g.set_max_march_slices(alice_sdf::sheet_field::gpu::MAX_MARCH_SLICES);
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
        assert_eq!(long, vec![MarchResult::Miss], "ns {ns}");
        assert_eq!(long_st.ray_slices, vec![2], "ns {ns}");
        assert_eq!(cut, vec![MarchResult::Undecided], "ns {ns}");
        assert_eq!(cut_st.over_slices, 1, "ns {ns}");
        // the long ray's own work is over the budget and under twice it: two slices decide it
        assert!(
            long_st.ray_work[0] > u64::from(MARCH_WORK_BUDGET),
            "ns {ns}"
        );
        assert!(
            long_st.ray_work[0] < 2 * u64::from(MARCH_WORK_BUDGET),
            "ns {ns}"
        );
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

/// The speed table the upload refuses by: parsed from the crate's file, at least one row, every row made by the
/// measure command (whose example exists); an empty or malformed table is refused, not read as "no adapter".
#[test]
fn the_adapter_speed_table_parses_and_names_its_command() {
    use alice_sdf::sheet_field::gpu::{
        adapter_speeds, parse_adapter_speeds, ADAPTER_SPEED_COMMAND,
    };
    let rows = adapter_speeds().expect("the tables parse");
    assert!(
        !rows.is_empty(),
        "no row in the crate's table or the one {} names",
        alice_sdf::sheet_field::gpu::ADAPTER_SPEEDS_VAR
    );
    for r in &rows {
        assert_eq!(r.command, ADAPTER_SPEED_COMMAND, "{r:?}");
        assert!(r.ns_per_tick.is_finite() && r.ns_per_tick > 0.0, "{r:?}");
        assert_eq!(r.measured.len(), 10, "{r:?}");
    }
    let example = ADAPTER_SPEED_COMMAND
        .split_whitespace()
        .skip_while(|w| *w != "--example")
        .nth(1)
        .expect("the command names an example");
    assert!(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("examples/{example}.rs"))
            .is_file(),
        "{example}"
    );
    assert!(parse_adapter_speeds("").is_err());
    assert!(parse_adapter_speeds("# header only\n\n").is_err());
    assert!(parse_adapter_speeds("0x10de\t0x20b0\tVulkan\tX\t\t-1\t2026-11-24\tc").is_err());
    assert!(parse_adapter_speeds("0x10de\t0x20b0\tVulkan\tX").is_err());
    let one =
        parse_adapter_speeds("0x10de\t0x20b0\tVulkan\tX\tdrv 1\t320.5\t2026-11-24\tc").unwrap();
    assert_eq!(
        (one[0].vendor, one[0].device, one[0].ns_per_tick),
        (0x10de, 0x20b0, 320.5)
    );
    // a row prints back as the line it was read from
    assert_eq!(
        parse_adapter_speeds(&one[0].tsv()).unwrap(),
        one,
        "{}",
        one[0].tsv()
    );
}

fn info(vendor: u32, device: u32, name: &str, driver_info: &str) -> wgpu::AdapterInfo {
    wgpu::AdapterInfo {
        name: name.to_string(),
        vendor,
        device,
        device_type: wgpu::DeviceType::DiscreteGpu,
        driver: String::new(),
        driver_info: driver_info.to_string(),
        backend: wgpu::Backend::Vulkan,
    }
}

/// The lookup is the exact adapter's row; any other adapter (another device, driver or name) is refused with the
/// command that measures it.
#[test]
fn the_speed_lookup_is_exact_and_an_unknown_adapter_is_refused() {
    use alice_sdf::sheet_field::gpu::{
        parse_adapter_speeds, table_ns_per_tick, ADAPTER_SPEED_COMMAND,
    };
    let rows = parse_adapter_speeds(
        "0x10de\t0x20b0\tVulkan\tGPU A\tdrv 1\t320\t2026-11-24\tc\n0x10de\t0x20b0\tVulkan\tGPU A\tdrv 2\t900\t2026-11-24\tc",
    )
    .unwrap();
    assert_eq!(
        table_ns_per_tick(&rows, &info(0x10de, 0x20b0, "GPU A", "drv 1")).unwrap(),
        320.0
    );
    assert_eq!(
        table_ns_per_tick(&rows, &info(0x10de, 0x20b0, "GPU A", "drv 2")).unwrap(),
        900.0
    );
    for other in [
        info(0x10de, 0x20b1, "GPU A", "drv 1"),
        info(0x1002, 0x20b0, "GPU A", "drv 1"),
        info(0x10de, 0x20b0, "GPU B", "drv 1"),
        info(0x10de, 0x20b0, "GPU A", "drv 3"),
    ] {
        match table_ns_per_tick(&rows, &other) {
            Err(GpuError::AdapterSpeedUnknown { command, var, .. }) => {
                assert_eq!(command, ADAPTER_SPEED_COMMAND);
                assert_eq!(var, alice_sdf::sheet_field::gpu::ADAPTER_SPEEDS_VAR);
            }
            r => panic!("{other:?}: {r:?}"),
        }
    }
    let mut metal = info(0x10de, 0x20b0, "GPU A", "drv 1");
    metal.backend = wgpu::Backend::Metal;
    assert!(table_ns_per_tick(&rows, &metal).is_err());
}

/// This machine's adapter: measured and printed as a table row (the CI's GPU steps read it), and it must have a row
/// in the crate's table, or no field can be uploaded here.
#[cfg(feature = "gpu")]
#[test]
fn this_adapter_is_measured_and_has_a_row() {
    use alice_sdf::sheet_field::gpu::{adapter_speeds, measure_adapter_speed, table_ns_per_tick};
    let row = match measure_adapter_speed("0000-00-00") {
        Ok(r) => r,
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
        "adapter speed row (date to fill): {}\nfull budget {:.3} s, limit {:.3} s",
        row.tsv(),
        row.ns_per_tick * f64::from(MARCH_WORK_BUDGET) * 1e-9,
        DISPATCH_WAIT.as_secs_f64() / 3.0
    );
    let info = wgpu::AdapterInfo {
        name: row.name.clone(),
        vendor: row.vendor,
        device: row.device,
        device_type: wgpu::DeviceType::Other,
        driver: String::new(),
        driver_info: row.driver_info.clone(),
        backend: match row.backend.as_str() {
            "Vulkan" => wgpu::Backend::Vulkan,
            "Metal" => wgpu::Backend::Metal,
            "Dx12" => wgpu::Backend::Dx12,
            "Gl" => wgpu::Backend::Gl,
            _ => wgpu::Backend::Empty,
        },
    };
    if let Err(e) = table_ns_per_tick(&adapter_speeds().unwrap(), &info) {
        panic!("{e}");
    }
}

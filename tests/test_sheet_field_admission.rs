//! Admission of a device by its measured march speed, on the device the job runs on.
//!
//! With `ALICE_SDF_EXPECT_SPEED_REFUSAL=1` (the CI's lavapipe job, whose measured row is in the file
//! `ALICE_SDF_ADAPTER_SPEEDS` names), uploading a field must be refused as too slow for the work budget, with the
//! measured speed and both times in the error; anything else (an upload, another error) fails. Without the variable
//! the test only reports what the device's admission returned. The sheet field GPU suites run on real adapters,
//! where admission passes.
#![cfg(feature = "gpu")]

use alice_sdf::compiled::GpuError;
use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::GpuSheetField;
use alice_sdf::sheet_field::{SheetField, ValidRule};
use glam::Vec3;

fn plane() -> SheetField {
    let lv = level_from_fn(
        Vec3::new(-2.0 + 0.013, -2.0 + 0.021, -1.0 + 0.007),
        0.25,
        [2, 2, 1],
        |_c, _e| true,
        |p| vec![p.z, -10.0],
    )
    .unwrap();
    SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1)
}

#[test]
fn a_device_too_slow_for_the_work_budget_is_refused_at_upload_with_its_measured_speed() {
    let expect_refusal = std::env::var("ALICE_SDF_EXPECT_SPEED_REFUSAL").is_ok_and(|v| v == "1");
    match GpuSheetField::new(&plane()) {
        Err(GpuError::DeviceTooSlow {
            ns_per_tick,
            budget,
            budget_ms,
            limit_ms,
        }) => {
            println!(
                "admission refused: {ns_per_tick:.2} ns per tick, budget {budget} ticks = {budget_ms} ms over the \
                 limit {limit_ms} ms"
            );
            assert!(ns_per_tick > 0.0 && budget > 0 && budget_ms > limit_ms);
        }
        Ok(g) => {
            assert!(
                !expect_refusal,
                "ALICE_SDF_EXPECT_SPEED_REFUSAL=1 but the device was admitted ({} live sheets)",
                g.max_live()
            );
            println!("admission passed");
        }
        Err(e) => {
            assert!(
                !expect_refusal,
                "ALICE_SDF_EXPECT_SPEED_REFUSAL=1 but the upload failed otherwise: {e}"
            );
            println!("admission not reached: {e}");
        }
    }
}

//! A wait bound of zero: the march's dispatch cannot have returned, so the call is a typed `DispatchDidNotFinish`
//! naming the rays, and the field refuses further work.
//!
//! This test has a test binary (a process) of its own. Its dispatch is abandoned on purpose and keeps running on
//! the device, and while it runs the process refuses every new device (`refusal::DEVICE_ABANDONED`): that refusal is
//! the intended behaviour (no new GPU work starts behind work this process gave up waiting for). Next to other GPU
//! tests in one binary it made a later test's device creation fail, depending on whether the abandoned dispatch
//! had ended; here no other test shares the process, and the refusal itself is asserted.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::compiled::GpuError;
use alice_sdf::gpu_context::refusal;
use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::GpuSheetField;
use alice_sdf::sheet_field::{Sheet, SheetField, ValidRule};
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

/// 100 flat squares 0.01 apart, every box overlapping: the deepest live list the tests build
fn deep() -> SheetField {
    SheetField::new(
        (0..100).map(|k| square(0.01 * k as f32, 5.0)).collect(),
        ValidRule::CellV1,
    )
}

fn origins(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let (a, b) = (
                (i % 211) as f32 / 211.0,
                (i / 211) as f32 / (n / 211 + 1) as f32,
            );
            Vec3::new(-6.0 + 12.0 * a, -6.0 + 12.0 * b, 5.0)
        })
        .collect()
}

#[test]
fn a_wait_bound_of_zero_is_a_typed_error_and_the_field_refuses_more() {
    let f = deep();
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
    g.set_dispatch_wait(Duration::ZERO);
    let os = origins(20_000);
    match g.march_batch_decided(&os, -Vec3::Z, 10.0) {
        Err(GpuError::DispatchDidNotFinish { rays, limit_ms }) => {
            assert!(rays > 0 && rays <= os.len(), "rays {rays}");
            assert_eq!(limit_ms, 0);
        }
        other => panic!("not the typed error: {other:?}"),
    }
    assert!(
        g.march_batch_decided(&os[..10], -Vec3::Z, 10.0).is_err(),
        "the field took more work"
    );
    // the process-wide guard: a new device is refused while the abandoned dispatch runs; it may have ended by now
    // (the device finished it), and then a new device is allowed again, so either outcome is checked and named
    match GpuSheetField::new(&f) {
        Err(GpuError::DeviceCreation(m)) => {
            assert!(
                m.contains(refusal::DEVICE_ABANDONED),
                "another refusal: {m}"
            );
            eprintln!("abandon guard: a new device was refused while the abandoned dispatch ran");
        }
        Ok(_) => {
            eprintln!("abandon guard: the abandoned dispatch had ended; a new device was allowed");
        }
        Err(e) => panic!("another error: {e}"),
    }
}

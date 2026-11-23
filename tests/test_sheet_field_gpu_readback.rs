//! `GpuSheetField`'s reads of its mapped results are this batch's, at volume (the readback dependency that
//! `GpuContext::before_host_read` records; see `tests/test_gpu_readback_volume.rs` for the driver where its
//! absence returned another dispatch's old results at 16,384 points and up).
//!
//! 40 batches of 32,768 points, 3 times. Every point lies beyond the sheet's box along +x at its own
//! distance (the union is the box distance there, oracle: the CPU reference), so a value left from another
//! batch differs by at least 32 mm, far outside f32 evaluation noise.
//!
//! Long: it runs only with `ALICE_SDF_VOLUME_READBACK=1` (the GPU job in CI, and the device lane), and
//! needs a GPU then.
#![cfg(feature = "gpu")]
#![allow(clippy::unwrap_used)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::GpuSheetField;
use alice_sdf::sheet_field::{SheetField, ValidRule};
use glam::Vec3;

const BATCHES: usize = 40;
const RUNS: usize = 3;
const N: usize = 32_768;

#[test]
fn every_sheet_field_value_read_back_is_this_batch() {
    if std::env::var_os("ALICE_SDF_VOLUME_READBACK").is_none() {
        eprintln!("skipping: set ALICE_SDF_VOLUME_READBACK=1 to run the volume readback test");
        return;
    }
    let lv = level_from_fn(
        Vec3::splat(-2.13),
        0.5,
        [1, 1, 1],
        |_c, _e| true,
        |p| vec![p.z, p.x.abs().max(p.y.abs()) - 1.0],
    )
    .unwrap();
    let field = SheetField::new(vec![sheet(vec![lv]).unwrap()], ValidRule::CellV1);
    let gpu = GpuSheetField::new(&field)
        .expect("ALICE_SDF_VOLUME_READBACK is set but no GPU device was found");
    let hi = field.sheets[0].hi;
    let (mut wrong, mut checked) = (0u64, 0u64);
    for _run in 0..RUNS {
        for batch in 0..BATCHES {
            let pts: Vec<Vec3> = (0..N)
                .map(|i| Vec3::new(hi.x + 10.0 + 0.001 * (batch * N + i) as f32, 0.0, 0.0))
                .collect();
            let got = gpu.eval_batch(&pts).unwrap();
            assert_eq!(got.len(), pts.len());
            for (p, g) in pts.iter().zip(&got) {
                let want = field.eval(*p).0;
                if (g.f - want).abs() > 1e-5 * want {
                    wrong += 1;
                }
                checked += 1;
            }
        }
    }
    assert_eq!(
        checked,
        (RUNS * BATCHES * N) as u64,
        "the test read nothing"
    );
    assert_eq!(
        wrong, 0,
        "{wrong} values of {checked} were not this batch's"
    );
}

//! What the CPU reads from a mapped buffer is what the GPU last wrote, at volume.
//!
//! wgpu 24 records no dependency from the GPU's writes to the CPU's reads of a mapped buffer
//! (<https://github.com/gfx-rs/wgpu/issues/9306>). On one integrated-GPU Vulkan driver (540.4.0)
//! the first mapped read then returned another dispatch's old
//! results for whole 64-byte lines, at 16,384 points a dispatch and up, a few sets in ten, while the
//! memory held the new ones. `GpuContext::before_host_read` records the dependency. This test sends
//! 40 batches of 32,768 points 30 times; every point has its own answer (`x + 0.5`, exact in f32),
//! so a value left over from another batch shows as a wrong answer, not as noise.
//!
//! Long: it runs only with `ALICE_SDF_VOLUME_READBACK=1` (the GPU job in CI, and the device lane),
//! and needs a GPU then.
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gpu")]

use alice_sdf::compiled::{GpuError, GpuEvaluator};
use glam::Vec3;

const KERNEL: &str = "struct InputPoint { x: f32, y: f32, z: f32, _pad: f32, }
struct OutputDistance { distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= point_count) { return; }
    output_distances[idx].distance = input_points[idx].x + 0.5;
}
";

const BATCHES: u32 = 40;
const RUNS: u32 = 30;

#[test]
fn every_value_read_back_is_this_batch() {
    if std::env::var_os("ALICE_SDF_VOLUME_READBACK").is_none() {
        eprintln!("skipping: set ALICE_SDF_VOLUME_READBACK=1 to run the volume readback test");
        return;
    }
    let gpu = match GpuEvaluator::from_wgsl(KERNEL) {
        Ok(c) => c.into_parts().0,
        Err(GpuError::NoAdapter | GpuError::DeviceCreation(_)) => {
            panic!("ALICE_SDF_VOLUME_READBACK is set but no GPU device was found")
        }
        Err(e) => panic!("{e}"),
    };
    let (mut wrong, mut batches_wrong, mut checked) = (0_u64, 0_u64, 0_u64);
    let mut first = None;
    for run in 0..RUNS {
        for batch in 0..BATCHES {
            let base = batch * 32_768;
            let pts: Vec<Vec3> = (0..32_768)
                .map(|i| Vec3::new((base + i) as f32, 0.0, 0.0))
                .collect();
            let got = gpu.eval_batch(&pts).expect("eval on the GPU");
            assert_eq!(got.len(), pts.len());
            let bad = got
                .iter()
                .zip(&pts)
                .filter(|(v, p)| v.to_bits() != (p.x + 0.5).to_bits())
                .count() as u64;
            checked += pts.len() as u64;
            if bad > 0 {
                wrong += bad;
                batches_wrong += 1;
                first.get_or_insert((run, batch));
            }
        }
    }
    assert_eq!(
        checked,
        u64::from(RUNS * BATCHES) * 32_768,
        "the test read nothing"
    );
    assert_eq!(
        wrong, 0,
        "{wrong} values of {checked} were not this batch's, in {batches_wrong} batches (first: run, batch {first:?})"
    );
}

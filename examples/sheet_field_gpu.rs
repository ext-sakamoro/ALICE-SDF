//! Sheet field on the GPU: the square sheet of `sheet_field_patch`, evaluated and ray-marched by the WGSL
//! evaluator (`sheet_field::gpu::GpuSheetField`), compared with the CPU reference ray by ray, by the march with a
//! given step budget and by the decided march; then the dispatch controls (dispatch size target, wait bound,
//! ray order, per-ray work budget) and the march statistics.
//!
//! Run: `cargo run --example sheet_field_gpu --features gpu`

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::{
    storage_binding_limit, GpuSheetField, DISPATCH_WAIT, SPREAD_BLOCK,
};
use alice_sdf::sheet_field::{MarchResult, SheetField, ValidRule};
use glam::Vec3;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 5.0;
    let dy = y.abs() - 5.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let level = level_from_fn(
        Vec3::new(-7.63, -7.79, -5.87),
        0.5,
        [4, 4, 3],
        |_c, _e| true,
        |p| vec![p.z, square_db(p.x, p.y), 0.5 + 0.01 * p.x, 0.5 + 0.01 * p.y],
    )?;
    let field = SheetField::new(vec![sheet(vec![level])?], ValidRule::from_name("cell-v1")?);
    let gpu = GpuSheetField::new(&field)?;
    println!(
        "device: storage binding up to {} bytes; march compiled for {} live sheets",
        storage_binding_limit()?,
        gpu.max_live()
    );
    let origins: Vec<Vec3> = (0..225)
        .map(|i| Vec3::new(-7.0 + (i % 15) as f32, -7.0 + (i / 15) as f32, 10.0))
        .collect();
    let rays = gpu.march_batch(&origins, Vec3::NEG_Z, 30.0, 4000)?;
    let samples = gpu.eval_batch(&origins)?;
    let mut agree = 0;
    for (o, r) in origins.iter().zip(&rays) {
        let same = match (field.march(*o, Vec3::NEG_Z, 30.0, 4000), r) {
            (MarchResult::Hit(a), MarchResult::Hit(b)) => (a.t - b.t).abs() < 1e-3,
            (a, b) => std::mem::discriminant(&a) == std::mem::discriminant(b),
        };
        agree += usize::from(same);
    }
    println!(
        "rays 225: GPU agrees with the CPU on {agree}; first sample f {:.4}",
        samples[0].f
    );
    let decided = gpu.march_batch_decided(&origins, Vec3::NEG_Z, 30.0)?;
    let same = origins
        .iter()
        .zip(&decided)
        .filter(|(o, r)| {
            match (
                field
                    .march_decided(**o, Vec3::NEG_Z, 30.0)
                    .expect("tmax within the decided range"),
                r,
            ) {
                (MarchResult::Hit(a), MarchResult::Hit(b)) => (a.t - b.t).abs() < 1e-3,
                (a, b) => std::mem::discriminant(&a) == std::mem::discriminant(b),
            }
        })
        .count();
    println!("decided march: GPU agrees with the CPU on {same} of 225 rays");
    let stats = gpu.last_march_stats();
    println!(
        "dispatches {} (cut {}, re-issued rays {}), largest work of one ray {}",
        stats.dispatches.len(),
        stats.cut_dispatches,
        stats.reissued_rays,
        stats.max_work
    );

    // the dispatch controls: shorter dispatches and a longer wait for a slow device, the rays in their own order
    // instead of spread blocks (same results), and a per-ray work budget too small for any ray (all undecided)
    gpu.set_dispatch_target(std::time::Duration::from_millis(50));
    gpu.set_dispatch_wait(2 * DISPATCH_WAIT);
    gpu.set_spread_block(0);
    let contiguous = gpu.march_batch_decided(&origins, Vec3::NEG_Z, 30.0)?;
    gpu.set_spread_block(SPREAD_BLOCK);
    println!(
        "contiguous order: same as spread blocks on {} of 225 rays",
        contiguous
            .iter()
            .zip(&decided)
            .filter(|(a, b)| a == b)
            .count()
    );
    gpu.set_work_budget(1)?;
    let starved = gpu.march_batch_decided(&origins, Vec3::NEG_Z, 30.0)?;
    println!(
        "work budget 1: undecided on {} of 225 rays",
        starved
            .iter()
            .filter(|r| **r == MarchResult::Undecided)
            .count()
    );
    Ok(())
}

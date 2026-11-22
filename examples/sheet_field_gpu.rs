//! Sheet field on the GPU: the square sheet of `sheet_field_patch`, evaluated and ray-marched by the WGSL
//! evaluator (`sheet_field::gpu::GpuSheetField`), compared with the CPU reference ray by ray.
//!
//! Run: `cargo run --example sheet_field_gpu --features gpu`

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::gpu::GpuSheetField;
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
    Ok(())
}

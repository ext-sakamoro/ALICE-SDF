//! Sheet field example: one open square sheet with a UV map, built from its channel function, then a grid of
//! rays cast straight down. Prints the hits (t, normal, uv), the misses beyond the rim, and the rule's name check,
//! and the same rays by the decided march (a hit or a miss for every ray).
//!
//! Run: `cargo run --example sheet_field_patch --no-default-features`

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{MarchResult, RootMode, SheetField, ValidRule};
use glam::Vec3;

fn square_db(x: f32, y: f32) -> f32 {
    let dx = x.abs() - 5.0;
    let dy = y.abs() - 5.0;
    Vec3::new(dx.max(0.0), dy.max(0.0), 0.0).length() + dx.max(dy).min(0.0)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rule = ValidRule::from_name("cell-v1")?;
    let level = level_from_fn(
        Vec3::new(-7.63, -7.79, -5.87),
        0.5,
        [4, 4, 3],
        |_center, _edge| true,
        |p| vec![p.z, square_db(p.x, p.y), 0.5 + 0.01 * p.x, 0.5 + 0.01 * p.y],
    )?;
    let field = SheetField::new(vec![sheet(vec![level])?], rule);
    let (mut hits, mut misses, mut undecided) = (0, 0, 0);
    for i in 0..15 {
        for j in 0..15 {
            let origin = Vec3::new(-7.0 + i as f32, -7.0 + j as f32, 10.0);
            match field.march(origin, Vec3::NEG_Z, 30.0, 4000) {
                MarchResult::Hit(h) => {
                    hits += 1;
                    if i == 7 && j == 7 {
                        println!(
                            "centre ray: t {:.4}, normal {:?}, uv {:?}",
                            h.t, h.normal, h.uv
                        );
                    }
                }
                MarchResult::Miss => misses += 1,
                MarchResult::Undecided => undecided += 1,
            }
        }
    }
    println!("rays 225: hits {hits}, misses {misses}, undecided {undecided}");
    let agree = (0..225)
        .filter(|i| {
            let o = Vec3::new(-7.0 + (i % 15) as f32, -7.0 + (i / 15) as f32, 10.0);
            field.march(o, Vec3::NEG_Z, 30.0, 4000)
                == field.march_linear(o, Vec3::NEG_Z, 30.0, 4000)
        })
        .count();
    println!("BVH march equals the linear march on {agree} of 225 rays");
    // the decided march takes its step budget from the ray's length, so it ends in a hit or a miss
    let decided_undecided = (0..225)
        .filter(|i| {
            let o = Vec3::new(-7.0 + (i % 15) as f32, -7.0 + (i / 15) as f32, 10.0);
            field
                .march_decided(o, Vec3::NEG_Z, 30.0)
                .expect("tmax within the decided range")
                == MarchResult::Undecided
        })
        .count();
    println!("decided march: undecided on {decided_undecided} of 225 rays");
    // the centre ray's steps, recorded by the same march
    let (centre, steps) = field
        .march_profile(Vec3::new(0.0, 0.0, 10.0), Vec3::NEG_Z, 30.0)
        .expect("tmax within the decided range");
    println!(
        "centre ray profiled: {} steps ({} from masked cells), hit {}",
        steps.len(),
        steps.iter().filter(|s| s.dd_masked).count(),
        matches!(centre, MarchResult::Hit(_))
    );
    // one row of rays against the references: the fixed-step march (0.0005 steps, no bound) and the march whose
    // floor steps are re-walked in fixed sub-steps
    let mut reference = field.clone();
    reference.set_root_mode(RootMode::FixedFloor);
    let row: Vec<Vec3> = (0..15)
        .map(|i| Vec3::new(-7.0 + i as f32, 0.3, 10.0))
        .collect();
    let same = |a: &MarchResult, b: &MarchResult| match (a, b) {
        (MarchResult::Hit(x), MarchResult::Hit(y)) => (x.t - y.t).abs() < 1e-3,
        _ => std::mem::discriminant(a) == std::mem::discriminant(b),
    };
    let (mut fixed, mut floor) = (0, 0);
    for o in &row {
        let m = field
            .march_decided(*o, Vec3::NEG_Z, 30.0)
            .expect("tmax within the decided range");
        fixed += usize::from(same(
            &m,
            &field.march_fixed_step(*o, Vec3::NEG_Z, 30.0, 0.0005),
        ));
        // the reference's own agreement rule: within REFERENCE_T_TOL, or the same crossing within JUMP_S between
        floor += usize::from(
            match (
                m,
                reference
                    .march_decided(*o, Vec3::NEG_Z, 30.0)
                    .expect("tmax within the decided range"),
            ) {
                (MarchResult::Hit(a), MarchResult::Hit(b)) => {
                    a.sheet == b.sheet
                        && field.hit_agrees_with_reference(a.sheet, *o, Vec3::NEG_Z, a.t, b.t)
                }
                (a, b) => std::mem::discriminant(&a) == std::mem::discriminant(&b),
            },
        );
    }
    println!("row of 15 rays: same as the fixed-step march on {fixed}, as the fixed-floor march on {floor}");
    Ok(())
}

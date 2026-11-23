//! The floor of a masked cell's lower bound is `min(MASKED_FLOOR, h/2)`: masking only says every corner is more than
//! `sqrt(3)/2 * h` from the surface, and a surface point just past the centre of one of the cell's faces is that far
//! from all eight corners while only a little over `h/2` from the face. A floor of `sqrt(3)/2 * h` answers more than
//! the true distance there.
//!
//! Oracle: a small sphere (radius `R = 0.01 h`) whose surface is `0.6 h` outside the centre of a face of one cell, in
//! cells of `h = 0.02` (so `h/2` is under `MASKED_FLOOR`). The true distance at `p` is `|p - centre| - R`. That cell is
//! masked (its near corners are about `0.92 h` from the sphere, over `sqrt(3)/2 * h` plus the storage step), and every
//! sampled point of it must answer at most its true distance, on the CPU and, with the `gpu` feature, on the GPU. The
//! scene tests the claim: the sample nearest the face is under `sqrt(3)/2 * h` from the sphere.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_precision_loss)]

use alice_sdf::sheet_field::build::{level_from_fn, sheet};
use alice_sdf::sheet_field::{masked_floor, Sheet, ValidRule, MASKED_FLOOR};
use glam::Vec3;

const H: f32 = 0.02;
const R: f32 = 0.01 * H;

/// The lower corner of the masked cell (cell (3, 3, 3) of one brick) and the sphere's centre, `0.6 h + R` above the
/// centre of the cell's top face.
fn scene() -> (Sheet, Vec3, Vec3) {
    let lo = Vec3::new(0.0013, 0.0021, 0.0007);
    let cell = lo + Vec3::splat(3.0 * H);
    let centre = cell + Vec3::new(0.5 * H, 0.5 * H, H + 0.6 * H + R);
    let lv = level_from_fn(
        lo,
        H,
        [1, 1, 1],
        |_c, _e| true,
        |p| vec![(p - centre).length() - R, -1.0],
    )
    .unwrap();
    (sheet(vec![lv]).unwrap(), cell, centre)
}

/// Points inside the masked cell: a 5 x 5 x 5 grid, the top layer a thousandth of a cell under the face.
fn points(cell: Vec3) -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                let u = |n: i32| (0.1 + 0.2 * n as f32).min(0.999);
                let z = if k == 4 { 0.999 } else { u(k) };
                out.push(cell + Vec3::new(u(i), u(j), z) * H);
            }
        }
    }
    out
}

fn truth(p: Vec3, centre: Vec3) -> f32 {
    (p - centre).length() - R
}

#[test]
fn a_masked_cell_next_to_a_surface_point_past_a_face_centre_answers_at_most_its_distance() {
    let (sh, cell, centre) = scene();
    assert!(
        masked_floor(H) < MASKED_FLOOR,
        "the scene must test a cell under the floor's range"
    );
    let ps = points(cell);
    let top = cell + Vec3::new(0.5, 0.5, 0.999) * H;
    assert!(
        truth(top, centre) < 0.866 * H,
        "the scene does not separate h/2 from sqrt(3)/2 h"
    );

    let mut masked = 0;
    for p in &ps {
        let a = sh.sample(*p, ValidRule::CellV1);
        if a.masked {
            masked += 1;
            assert!(
                a.f <= truth(*p, centre),
                "cpu {p}: masked bound {} over the distance {}",
                a.f,
                truth(*p, centre)
            );
        }
    }
    assert_eq!(masked, ps.len(), "every point of the cell must be masked");
    let a = sh.sample(top, ValidRule::CellV1);
    assert!(
        a.masked && a.f <= truth(top, centre),
        "cpu face centre: {} over {}",
        a.f,
        truth(top, centre)
    );

    #[cfg(feature = "gpu")]
    {
        use alice_sdf::sheet_field::gpu::GpuSheetField;
        use alice_sdf::sheet_field::SheetField;
        let f = SheetField::new(vec![sh], ValidRule::CellV1);
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
        let mut all = ps;
        all.push(top);
        let r = g.eval_batch(&all).unwrap();
        for (p, s) in all.iter().zip(&r) {
            assert!(s.masked, "gpu {p}: not masked");
            assert!(
                s.f <= truth(*p, centre),
                "gpu {p}: masked bound {} over the distance {}",
                s.f,
                truth(*p, centre)
            );
        }
    }
}

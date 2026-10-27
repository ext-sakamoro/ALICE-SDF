//! Surface plate — a tileable stone surface, drawn from above
//!
//! Stones from `voronoi_cell`, a tile grid from `grid_cell`, grain from `hash_noise_2d_periodic`;
//! the height is lowest in the joints between stones. `render_plate` draws one tile seen from
//! straight above (the normal from the heights by `normal_from_height`), and the example checks
//! that the surface repeats: the sample at `u` and at `u + tile` is the same.
//!
//! # Running
//! ```bash
//! cargo run --example surface_plate
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::surface::{
    grid_cell, hash_noise_2d_periodic, normal_from_height, render_plate, voronoi_cell, Light,
    PlateView, SurfaceSample,
};

/// One tile is 4 × 4 stones of size 0.25, and 2 × 2 grid cells of size 0.5
const TILE: f64 = 1.0;
const STONE: f64 = 0.25;
const GRAIN: u32 = 16;

fn sample(u: f64, v: f64) -> SurfaceSample {
    let stone = voronoi_cell(u, v, STONE, 0.8, [4, 4], 7);
    let grid = grid_cell(u, v, [0.5, 0.5], 0.0, [2, 2], 11);
    #[allow(clippy::cast_possible_truncation)]
    let grain = f64::from(hash_noise_2d_periodic(
        (u * f64::from(GRAIN)) as f32,
        (v * f64::from(GRAIN)) as f32,
        [GRAIN, GRAIN],
        3,
    ));
    // a joint 0.02 wide between stones, the stone tops tilted a little per stone
    let joint = (stone.edge / 0.02).min(1.0);
    let height = 0.01 * joint + 0.002 * stone.numbers[0] + 0.0005 * grain;
    let tint = 0.55 + 0.25 * stone.numbers[1] + 0.05 * grid.numbers[0];
    SurfaceSample {
        height,
        albedo: [tint, tint * 0.96, tint * 0.9],
        roughness: 0.6 + 0.3 * joint * stone.numbers[2],
    }
}

fn main() {
    let view = PlateView {
        width: 64,
        height: 64,
        extent: [TILE, TILE],
    };
    let light = Light {
        elevation: 0.9,
        azimuth: 0.6,
        irradiance: 1.0,
    };
    let image = render_plate(&view, &light, &sample, 1e-3);
    let mean: f64 = image.iter().map(|p| p[1]).sum::<f64>() / image.len() as f64;
    let (lo, hi) = image
        .iter()
        .map(|p| p[1])
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), g| {
            (lo.min(g), hi.max(g))
        });
    println!("ALICE-SDF — surface plate");
    println!(
        "{} × {} pixels, green radiance mean {mean:.4} (min {lo:.4}, max {hi:.4})",
        view.width, view.height
    );

    // the surface repeats every tile: the same sample one tile to the right and one tile up
    let mut worst = 0.0_f64;
    for i in 0..32 {
        let (u, v) = (0.03 + 0.031 * f64::from(i), 0.07 + 0.027 * f64::from(i));
        let a = sample(u, v).height;
        worst = worst.max((a - sample(u + TILE, v).height).abs());
        worst = worst.max((a - sample(u, v + TILE).height).abs());
    }
    let n = normal_from_height(&|u, v| sample(u, v).height, 0.5, 0.5, 1e-3);
    println!("tiling: largest height difference one tile apart {worst:.3e}");
    println!(
        "normal at the centre ({:.4}, {:.4}, {:.4})",
        n[0], n[1], n[2]
    );
    assert!(worst < 1e-6, "the plate repeats every tile");
}

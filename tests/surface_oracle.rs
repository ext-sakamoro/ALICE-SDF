//! Oracles for `alice_sdf::surface`: the periodic value noise, the cell patterns and the plate
//! render a material is drawn on
//!
//! Expected values come from closed forms written here, never from the function under test (the
//! closed forms use alice-det-math for sin / cos, which is not the code under test):
//! - periodic noise: at the lattice points the value is the corner hash, so a lattice point and
//!   the same point one period on give the same bits; across one period sampled on N pixels the
//!   step across the wrap (the last pixel to the first) is no larger than the steps inside
//! - the non-periodic noise (the control): the same image has a step across the wrap far larger
//!   than inside, so the seam check has teeth
//! - grid cells: the distance to the nearest edge is min(fu, pu − fu, fv, pv − fv) in the cell's
//!   own coordinates, a row shifted by `shift` of a cell per row; the cell numbers repeat with the
//!   tile's cell count
//! - voronoi cells with no jitter: the feature points sit at the square centres, so the cells and
//!   the edge distance are those of the grid
//! - the normal of a height h = a·sin(2πu/P) by central difference is (−h′, 0, 1)/|·| with
//!   h′ = a·2π/P·cos(2πu/P), within the O(step²) error of the difference
//! - a flat plate of albedo ρ lit by a directional light of irradiance E at elevation θ and seen
//!   from above gives the Lambert radiance ρ·E·sin θ/π (the plate's specular term is weighted by
//!   1 − roughness, so roughness 1 is Lambert alone), the same bits on every render
#![allow(clippy::unwrap_used, clippy::float_cmp)]

use std::f64::consts::PI;

use alice_det_math::{cos64, sin64};
use alice_sdf::modifiers::hash_noise_3d;
use alice_sdf::surface::{
    grid_cell, hash_noise_2d_periodic, normal_from_height, render_plate, voronoi_cell, Light,
    PlateView, SurfaceSample,
};

fn hash_noise_3d_cpu(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    hash_noise_3d(glam::Vec3::new(x, y, z), seed)
}

#[test]
fn periodic_noise_repeats_at_the_lattice_points() {
    for (period, seed) in [([4u32, 4u32], 7u32), ([3, 5], 11), ([16, 1], 0)] {
        for i in 0..period[0] as i32 + 2 {
            for j in 0..period[1] as i32 + 2 {
                let a = hash_noise_2d_periodic(i as f32, j as f32, period, seed);
                let b = hash_noise_2d_periodic(
                    (i + period[0] as i32) as f32,
                    (j + period[1] as i32) as f32,
                    period,
                    seed,
                );
                let c =
                    hash_noise_2d_periodic((i - period[0] as i32) as f32, j as f32, period, seed);
                assert_eq!(a.to_bits(), b.to_bits(), "{period:?} ({i}, {j})");
                assert_eq!(
                    a.to_bits(),
                    c.to_bits(),
                    "{period:?} ({i}, {j}) one period back"
                );
                assert!((-1.0..=1.0).contains(&a));
            }
        }
    }
}

#[test]
fn periodic_noise_is_the_crate_noise_inside_one_period() {
    // inside the first period no corner wraps, so the value is the crate's one noise law
    for (x, y) in [(0.25f32, 0.5f32), (1.75, 2.125), (2.5, 0.0)] {
        assert_eq!(
            hash_noise_2d_periodic(x, y, [4, 4], 3).to_bits(),
            hash_noise_3d_cpu(x, y, 0.0, 3).to_bits(),
            "({x}, {y})"
        );
    }
}

/// The largest step across the wrap of each row and the 95th percentile of the steps inside
fn seam_and_interior(img: &[f32], n: usize) -> (f32, f32) {
    let mut inside = Vec::new();
    let mut seam = 0.0f32;
    for y in 0..n {
        for x in 0..n {
            let a = img[y * n + x];
            let b = img[y * n + (x + 1) % n];
            if x + 1 == n {
                seam = seam.max((a - b).abs());
            } else {
                inside.push((a - b).abs());
            }
        }
    }
    inside.sort_by(f32::total_cmp);
    (seam, inside[inside.len() * 95 / 100])
}

#[test]
fn a_periodic_image_has_no_seam_and_the_plain_noise_has_one() {
    let (n, p) = (64usize, 4u32);
    let at = |k: usize| k as f32 * p as f32 / n as f32;
    let periodic: Vec<f32> = (0..n * n)
        .map(|i| hash_noise_2d_periodic(at(i % n), at(i / n), [p, p], 5))
        .collect();
    let (seam, inside) = seam_and_interior(&periodic, n);
    assert!(
        seam <= inside,
        "periodic: seam {seam} vs interior p95 {inside}"
    );
    // control: the crate's plain noise over the same samples does not wrap
    let plain: Vec<f32> = (0..n * n)
        .map(|i| hash_noise_3d_cpu(at(i % n), at(i / n), 0.0, 5))
        .collect();
    let (seam, inside) = seam_and_interior(&plain, n);
    assert!(
        seam > 4.0 * inside,
        "plain: seam {seam} vs interior p95 {inside}"
    );
}

#[test]
fn grid_cells_edge_distance_and_numbers() {
    // 0.25 m cells, no shift, a tile of 4 x 4 cells
    let c = grid_cell(0.30, 0.10, [0.25, 0.25], 0.0, [4, 4], 9);
    assert_eq!(c.id, [1, 0]);
    let (fu, fv) = (0.30_f64 - 0.25, 0.10_f64);
    assert!(
        (c.edge - fu.min(0.25 - fu).min(fv).min(0.25 - fv)).abs() < 1e-15,
        "{c:?}"
    );
    // the cell numbers repeat with the tile's cell count, and differ between cells
    let same = grid_cell(0.30 + 1.0, 0.10 + 2.0, [0.25, 0.25], 0.0, [4, 4], 9);
    assert_eq!(same.id, [1, 0]);
    assert_eq!(c.numbers, same.numbers);
    let other = grid_cell(0.55, 0.10, [0.25, 0.25], 0.0, [4, 4], 9);
    assert_ne!(c.numbers, other.numbers);
    for x in c.numbers {
        assert!((0.0..1.0).contains(&x));
    }
    // another seed, other numbers
    assert_ne!(
        grid_cell(0.30, 0.10, [0.25, 0.25], 0.0, [4, 4], 10).numbers,
        c.numbers
    );
    // the edge nearest from the far side of the cell
    let far = grid_cell(0.48, 0.10, [0.25, 0.25], 0.0, [4, 4], 9);
    assert!(
        (far.edge - (0.25 - (0.48_f64 - 0.25))).abs() < 1e-15,
        "{far:?}"
    );
    // half a cell of shift per row: row 1 starts at 0.125 (the point sits mid-row, so the
    // shifted column decides the edge)
    let s = grid_cell(0.20, 0.375, [0.25, 0.25], 0.5, [4, 4], 9);
    assert_eq!(s.id, [0, 1]);
    let fu = 0.20_f64 - 0.125;
    let fv = 0.375_f64 - 0.25;
    assert!(
        (s.edge - fu.min(0.25 - fu).min(fv).min(0.25 - fv)).abs() < 1e-15,
        "{s:?}"
    );
    // one tile further along v: 4 rows of half a cell move the columns by 2 cells, and the cell
    // is the same cell of the tile
    let next = grid_cell(0.20, 0.375 + 1.0, [0.25, 0.25], 0.5, [4, 4], 9);
    assert_eq!((next.id, next.numbers), (s.id, s.numbers));
    assert!((next.edge - s.edge).abs() < 1e-12);
    // on an edge the distance is 0
    assert!(
        grid_cell(0.25, 0.10, [0.25, 0.25], 0.0, [4, 4], 9)
            .edge
            .abs()
            < 1e-15
    );
}

#[test]
fn voronoi_without_jitter_is_the_grid() {
    for (u, v) in [(0.30, 0.10), (0.61, 0.87), (0.05, 0.95), (0.99, 0.51)] {
        let g = grid_cell(u, v, [0.25, 0.25], 0.0, [4, 4], 3);
        let w = voronoi_cell(u, v, 0.25, 0.0, [4, 4], 3);
        assert_eq!(w.id, g.id, "({u}, {v})");
        assert!(
            (w.edge - g.edge).abs() < 1e-12,
            "({u}, {v}): {} vs {}",
            w.edge,
            g.edge
        );
    }
    // with jitter the cell numbers still repeat with the tile
    let a = voronoi_cell(0.4, 0.7, 0.25, 0.8, [4, 4], 3);
    let b = voronoi_cell(0.4 + 1.0, 0.7 - 1.0, 0.25, 0.8, [4, 4], 3);
    assert_eq!(a.id, b.id);
    assert_eq!(a.numbers, b.numbers);
    assert!((a.edge - b.edge).abs() < 1e-12);
}

#[test]
fn the_normal_of_a_sine_height() {
    let (amp, period, step) = (0.003, 0.5, 1e-4);
    let h = |u: f64, _v: f64| amp * sin64(2.0 * PI * u / period);
    for u in [0.0, 0.07, 0.125, 0.31] {
        let n = normal_from_height(&h, u, 0.2, step);
        let d = amp * 2.0 * PI / period * cos64(2.0 * PI * u / period);
        let len = (d * d + 1.0).sqrt();
        let want = [-d / len, 0.0, 1.0 / len];
        for k in 0..3 {
            assert!((n[k] - want[k]).abs() < 1e-6, "u {u}: {n:?} vs {want:?}");
        }
        assert!((n[0] * n[0] + n[1] * n[1] + n[2] * n[2] - 1.0).abs() < 1e-12);
    }
}

#[test]
fn a_flat_plate_is_lambert_and_the_same_bits_twice() {
    let albedo = [0.5, 0.25, 0.125];
    let flat = |_u: f64, _v: f64| SurfaceSample {
        height: 0.0,
        albedo,
        roughness: 1.0,
    };
    let view = PlateView {
        width: 8,
        height: 4,
        extent: [1.0, 0.5],
    };
    for elevation_deg in [90.0f64, 45.0, 20.0] {
        let light = Light {
            elevation: elevation_deg.to_radians(),
            azimuth: 0.3,
            irradiance: 3.0,
        };
        let a = render_plate(&view, &light, &flat, 1e-4);
        let b = render_plate(&view, &light, &flat, 1e-4);
        assert_eq!(a.len(), 8 * 4);
        assert_eq!(
            a.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let s = sin64(elevation_deg.to_radians());
        for px in &a {
            for k in 0..3 {
                let want = albedo[k] * 3.0 * s / PI;
                assert!(
                    (px[k] - want).abs() < 1e-12,
                    "elev {elevation_deg}: {px:?} vs {want}"
                );
            }
        }
    }
    // a light below the plate gives black
    let under = Light {
        elevation: -0.2,
        azimuth: 0.0,
        irradiance: 3.0,
    };
    assert!(render_plate(&view, &under, &flat, 1e-4)
        .iter()
        .flatten()
        .all(|v| *v == 0.0));
}

#[test]
fn a_rough_and_a_glossy_plate_differ_only_in_the_specular() {
    // seen from above with the light above, the glossy plate is brighter (a specular lobe), the
    // rough one is the Lambert value
    let view = PlateView {
        width: 2,
        height: 2,
        extent: [1.0, 1.0],
    };
    let light = Light {
        elevation: PI / 2.0,
        azimuth: 0.0,
        irradiance: 1.0,
    };
    let rough = |_: f64, _: f64| SurfaceSample {
        height: 0.0,
        albedo: [0.2; 3],
        roughness: 1.0,
    };
    let glossy = |_: f64, _: f64| SurfaceSample {
        height: 0.0,
        albedo: [0.2; 3],
        roughness: 0.2,
    };
    let r = render_plate(&view, &light, &rough, 1e-4)[0][0];
    let g = render_plate(&view, &light, &glossy, 1e-4)[0][0];
    assert!((r - 0.2 / PI).abs() < 1e-12, "{r}");
    assert!(g > r * 1.5, "glossy {g} vs rough {r}");
}

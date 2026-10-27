//! Surfaces drawn on a plate: the periodic value noise, the cell patterns and the plate render
//!
//! What a surface material is built from, without knowing how a material combines it:
//!
//! - [`hash_noise_2d_periodic`]: the crate's one value-noise law (`modifiers::hash_noise_3d` at
//!   z = 0) with its lattice wrapped every `period` cells, so an image of whole periods repeats
//!   without a seam; inside the first period it is the plain noise, bit for bit
//! - [`grid_cell`] / [`voronoi_cell`]: the cell a point lies in, its distance to the cell's
//!   nearest edge and three numbers in `0..1` per cell, the cell numbers repeating with the tile
//! - [`normal_from_height`]: the normal of a height field by central difference
//! - [`render_plate`]: a flat plate seen from straight above under one directional light, every
//!   pixel from a sample closure: Lambert diffuse plus a GGX specular lobe weighted by
//!   `1 − roughness` (Fresnel–Schlick with F0 0.04, Smith–Schlick shadowing), in linear radiance
//!
//! Everything but the noise is `f64` with `alice-det-math` for every function, so a render is
//! the same bits on every platform.

use std::f64::consts::PI;

use alice_det_math::{cos64, sin64, sqrt64};
use glam::Vec3;

use crate::modifiers::{surface_roughness_hash3, surface_roughness_pcg};

/// The cell a point lies in
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cell {
    /// the cell's column and row, wrapped to the tile's cell count
    pub id: [i64; 2],
    /// the distance (same unit as the point) to the nearest edge of the cell
    pub edge: f64,
    /// three numbers in `0..1` that the cell's id and the seed give
    pub numbers: [f64; 3],
}

/// What a material gives at one point of the plate
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSample {
    /// height above the plate (the unit of the plate's extent)
    pub height: f64,
    /// linear RGB reflectance, each in `0..=1`
    pub albedo: [f64; 3],
    /// `0` mirror-like .. `1` matte
    pub roughness: f64,
}

/// The pixels and the region of the plate they cover, seen from straight above
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlateView {
    /// pixels along u
    pub width: usize,
    /// pixels along v
    pub height: usize,
    /// the plate region `[0, extent[0]] × [0, extent[1]]` the pixels cover
    pub extent: [f64; 2],
}

/// A directional light
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    /// radians above the plate (π/2 = straight above, ≤ 0 = below it)
    pub elevation: f64,
    /// radians around the plate's normal, from +u towards +v
    pub azimuth: f64,
    /// irradiance on a surface facing the light
    pub irradiance: f64,
}

/// The lattice coordinate `i` wrapped to `0..period` (as the float the lattice hash reads)
fn wrap(i: f32, period: u32) -> f32 {
    // the lattice coordinates are integer valued floats; the period is at most 2^24, so the
    // remainder is exact and +0.0 (never -0.0)
    #[allow(clippy::cast_precision_loss)]
    let p = period as f32;
    let r = i.rem_euclid(p);
    if r == p {
        0.0
    } else {
        r + 0.0
    }
}

/// The crate's value noise at `(x, y, 0)` with its lattice wrapped every `period` cells along x
/// and y, in `[-1, 1]`
///
/// The value at `(x + k·period[0], y + m·period[1])` is the value at `(x, y)` for integers k, m
/// (bit for bit at the lattice points, within the rounding of `x − floor(x)` between them); for
/// `0 <= x < period[0] − 1`, `0 <= y < period[1] − 1` it is `modifiers::hash_noise_3d` itself.
///
/// # Panics
///
/// If a period is 0 or above 2^24 (the lattice coordinates would no longer be exact floats)
#[must_use]
pub fn hash_noise_2d_periodic(x: f32, y: f32, period: [u32; 2], seed: u32) -> f32 {
    assert!(
        period.iter().all(|p| (1..=1 << 24).contains(p)),
        "a noise period is 1 ..= 2^24 lattice cells"
    );
    let (ix, iy) = (x.floor(), y.floor());
    let f = Vec3::new(x - ix, y - iy, 0.0);
    let u = f * f * (Vec3::splat(3.0) - f * 2.0);
    let (x0, x1) = (wrap(ix, period[0]), wrap(ix + 1.0, period[0]));
    let (y0, y1) = (wrap(iy, period[1]), wrap(iy + 1.0, period[1]));
    let h = |a: f32, b: f32| surface_roughness_hash3(a, b, 0.0, seed);
    // the same operations as `hash_noise_3d` at z = 0, where u.z = 0 leaves the z = 0 face
    let (n000, n100, n010, n110) = (h(x0, y0), h(x1, y0), h(x0, y1), h(x1, y1));
    let c00 = (n100 - n000) * u.x + n000;
    let c10 = (n110 - n010) * u.x + n010;
    let c0 = (c10 - c00) * u.y + c00;
    // hash_noise_3d computes ((c1 − c0)·u.z + c0)·2 − 1 with u.z = 0: the z = 1 face drops out
    // exactly (its product is +0 or −0, and c0 + ±0 = c0)
    c0 * 2.0 + -1.0
}

/// Three numbers in `0..1` for a cell id and a seed (PCG over the id)
fn cell_numbers(id: [i64; 2], seed: u32, salt: u32) -> [f64; 3] {
    // the ids are wrapped to the tile (non-negative, at most u32::MAX cells per side)
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (a, b) = (id[0] as u32, id[1] as u32);
    let base =
        surface_roughness_pcg(a ^ surface_roughness_pcg(b ^ surface_roughness_pcg(seed ^ salt)));
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let h = surface_roughness_pcg(base ^ (k as u32).wrapping_mul(0x9e37_79b9));
        *o = f64::from(h) / 4_294_967_296.0;
    }
    out
}

/// The grid cell of `(u, v)`
///
/// Cells of `period` along u and v, row r shifted along u by `r · shift · period[0]`, ids wrapped
/// to `tile_cells` (a tile of that many columns and rows, whose rows times the shift is a whole
/// number of cells).
///
/// # Panics
///
/// If a period is not positive and finite, the shift is outside `0..1`, or a cell count is 0
#[must_use]
pub fn grid_cell(
    u: f64,
    v: f64,
    period: [f64; 2],
    shift: f64,
    tile_cells: [u32; 2],
    seed: u32,
) -> Cell {
    assert!(
        period.iter().all(|p| p.is_finite() && *p > 0.0),
        "a cell period is positive"
    );
    assert!((0.0..1.0).contains(&shift), "a row shift is in 0..1");
    assert!(
        tile_cells.iter().all(|c| *c > 0),
        "a tile has at least one cell per side"
    );
    let [pu, pv] = period;
    let row = (v / pv).floor();
    let fv = v - row * pv;
    let us = u - row * shift * pu;
    let col = (us / pu).floor();
    let fu = us - col * pu;
    let (cols, rows) = (i64::from(tile_cells[0]), i64::from(tile_cells[1]));
    // the cell coordinates are integer valued
    #[allow(clippy::cast_possible_truncation)]
    let (r, c) = (row as i64, col as i64);
    // a whole tile of rows moves the columns by rows · shift cells
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let wrap_shift = (rows as f64 * shift).round() as i64;
    let c = c + r.div_euclid(rows) * wrap_shift;
    let id = [c.rem_euclid(cols), r.rem_euclid(rows)];
    let edge = fu.min(pu - fu).min(fv).min(pv - fv);
    Cell {
        id,
        edge,
        numbers: cell_numbers(id, seed, 0x6772_6964),
    }
}

/// The voronoi cell of `(u, v)`
///
/// One feature point per `period` square, at the square's centre moved by up to `jitter`
/// (`0..=1`) of the square, the squares' ids wrapped to `tile_cells`; the edge is the distance to
/// the nearest bisector with a neighbouring feature point.
///
/// # Panics
///
/// If the period is not positive and finite, the jitter is outside `0..=1`, or a cell count is 0
#[must_use]
pub fn voronoi_cell(
    u: f64,
    v: f64,
    period: f64,
    jitter: f64,
    tile_cells: [u32; 2],
    seed: u32,
) -> Cell {
    assert!(
        period.is_finite() && period > 0.0,
        "a cell period is positive"
    );
    assert!((0.0..=1.0).contains(&jitter), "a jitter is in 0..=1");
    assert!(
        tile_cells.iter().all(|c| *c > 0),
        "a tile has at least one cell per side"
    );
    let (cols, rows) = (i64::from(tile_cells[0]), i64::from(tile_cells[1]));
    #[allow(clippy::cast_possible_truncation)]
    let (si, sj) = ((u / period).floor() as i64, (v / period).floor() as i64);
    let feature = |a: i64, b: i64| -> ([f64; 2], [i64; 2]) {
        let id = [a.rem_euclid(cols), b.rem_euclid(rows)];
        let j = cell_numbers(id, seed, 0x6a69_7474);
        #[allow(clippy::cast_precision_loss)]
        let p = [
            (a as f64 + 0.5 + jitter * (j[0] - 0.5)) * period,
            (b as f64 + 0.5 + jitter * (j[1] - 0.5)) * period,
        ];
        (p, id)
    };
    // a feature point stays in its square, so the nearest one is within two squares
    let mut pts = Vec::with_capacity(25);
    for db in -2..=2 {
        for da in -2..=2 {
            pts.push(feature(si + da, sj + db));
        }
    }
    let d2 = |p: [f64; 2]| (p[0] - u) * (p[0] - u) + (p[1] - v) * (p[1] - v);
    let (k, _) = pts.iter().enumerate().map(|(k, (p, _))| (k, d2(*p))).fold(
        (0, f64::INFINITY),
        |best, x| if x.1 < best.1 { x } else { best },
    );
    let (a, id) = pts[k];
    let mut edge = f64::INFINITY;
    for (m, (b, _)) in pts.iter().enumerate() {
        if m == k {
            continue;
        }
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = sqrt64(d[0] * d[0] + d[1] * d[1]);
        if len == 0.0 {
            continue;
        }
        let mid = [f64::midpoint(a[0], b[0]), f64::midpoint(a[1], b[1])];
        let dist = ((mid[0] - u) * d[0] + (mid[1] - v) * d[1]) / len;
        edge = edge.min(dist);
    }
    Cell {
        id,
        edge,
        numbers: cell_numbers(id, seed, 0x766f_726f),
    }
}

/// The unit normal of the height field `h` at `(u, v)` by central difference at `step`
#[must_use]
pub fn normal_from_height(h: &dyn Fn(f64, f64) -> f64, u: f64, v: f64, step: f64) -> [f64; 3] {
    let du = (h(u + step, v) - h(u - step, v)) / (2.0 * step);
    let dv = (h(u, v + step) - h(u, v - step)) / (2.0 * step);
    let len = sqrt64(du * du + dv * dv + 1.0);
    [-du / len, -dv / len, 1.0 / len]
}

/// The plate seen from straight above, one linear-radiance RGB value per pixel
///
/// Row-major, row 0 at v near 0, each pixel sampled at its centre; the normal comes from the
/// samples' heights by central difference at `normal_step`.
#[must_use]
pub fn render_plate(
    view: &PlateView,
    light: &Light,
    sample: &dyn Fn(f64, f64) -> SurfaceSample,
    normal_step: f64,
) -> Vec<[f64; 3]> {
    let l = [
        cos64(light.elevation) * cos64(light.azimuth),
        cos64(light.elevation) * sin64(light.azimuth),
        sin64(light.elevation),
    ];
    let height = |u: f64, v: f64| sample(u, v).height;
    let mut out = Vec::with_capacity(view.width * view.height);
    for y in 0..view.height {
        for x in 0..view.width {
            #[allow(clippy::cast_precision_loss)]
            let (u, v) = (
                (x as f64 + 0.5) / view.width as f64 * view.extent[0],
                (y as f64 + 0.5) / view.height as f64 * view.extent[1],
            );
            let s = sample(u, v);
            let n = normal_from_height(&height, u, v, normal_step);
            out.push(shade(&s, n, l, light.irradiance));
        }
    }
    out
}

/// Lambert plus `(1 − roughness)` · GGX, seen along +z
fn shade(s: &SurfaceSample, n: [f64; 3], l: [f64; 3], irradiance: f64) -> [f64; 3] {
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let ndl = dot(n, l);
    if ndl <= 0.0 {
        return [0.0; 3];
    }
    let view = [0.0, 0.0, 1.0];
    let ndv = n[2].max(1e-6);
    let hv = [l[0] + view[0], l[1] + view[1], l[2] + view[2]];
    let hl = sqrt64(dot(hv, hv));
    let hn = [hv[0] / hl, hv[1] / hl, hv[2] / hl];
    let ndh = dot(n, hn).max(0.0);
    let vdh = dot(view, hn).max(0.0);
    let alpha = (s.roughness * s.roughness).max(1e-4);
    let a2 = alpha * alpha;
    let den = ndh * ndh * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * den * den);
    let f0 = 0.04;
    let m = 1.0 - vdh;
    let fresnel = f0 + (1.0 - f0) * (m * m * m * m * m);
    let k = alpha / 2.0;
    let g1 = |x: f64| x / (x * (1.0 - k) + k);
    let spec = d * fresnel * g1(ndl) * g1(ndv) / (4.0 * ndl * ndv) * (1.0 - s.roughness);
    let e = irradiance * ndl;
    [
        (s.albedo[0] / PI + spec) * e,
        (s.albedo[1] / PI + spec) * e,
        (s.albedo[2] / PI + spec) * e,
    ]
}

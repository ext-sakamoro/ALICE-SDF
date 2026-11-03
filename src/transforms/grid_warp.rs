//! Grid warp: a domain warp by a field of affine maps on a regular grid
//!
//! The grid stores at each node an affine map `x = A p + b` (row-major 3x4, 12 floats). At a point
//! `p` the maps of the 8 nodes of its cell are blended trilinearly and applied to the point:
//! `x(p) = Σ w_i(p) (A_i p + b_i)`. A field that is skinned in a rest pose is posed by a grid that
//! stores, in the posed space, the map from the posed space back to the rest pose (for example
//! the inverse of a linear blend skinning matrix, sampled from the skinned mesh): the child is the
//! rest-pose SDF and is evaluated at `x(p)`.
//!
//! **Distance bound (local).** The warp's slope is bounded per cell from the stored maps when
//! the warp is built (never read from input): `λ_c = max_i ‖A_i‖₂ + sqrt(Σ_k (D_k / h)²)`, where
//! `D_k` bounds `‖(A_i p + b_i) − (A_j p + b_j)‖` over the cell for the 4 edges along axis `k` and
//! `h` is the cell size. Each node holds `Λ_n`, the largest `λ_c` of the cells within
//! `reach + h·√3` of it (an index cube that covers that ball), and `Λ(p)` is their trilinear blend.
//! For `p` in a cell, every corner's ball contains the ball of radius `reach` around `p`, so
//! `Λ(p)` bounds the slope of `g = child ∘ x` on that ball (a 1-Lipschitz child). The node returns
//! `v = clamp(g / Λ, −reach, reach)`: when `|g| / Λ < reach` no zero of `g` is nearer than
//! `|g| / Λ`, otherwise none is nearer than `reach`, so `|v|` never exceeds the distance to the
//! surface (sphere tracing is safe with step `v`) and is the distance itself where the warp is
//! rigid nearby, however steep the warp is far away. The field's slope is bounded by
//! [`GridWarp::lipschitz`] = `max_c (1 + reach · |∇Λ|_c / min Λ_c)` (the second term from `Λ`
//! varying, with `|g| / Λ ≤ reach`).
//!
//! **Outside the box.** `q` = `p` clamped to the box, `d = v(q)`, `e = |p − q|`; the
//! node returns `d + hypot(e, max(d, 0)) − max(d, 0)`. Inside (`e = 0`) this is `d`; outside with
//! `d ≥ 0` it is `hypot(e, d)`, a lower bound of the distance when the warped surface lies inside
//! the box (the angle at `q` between `p` and any point of the box is at least 90°), and
//! 1-Lipschitz (the clamp removes the gradient component along `p − q`). **Contract:** the
//! surface of the warped child lies inside the box; outside points with `d < 0` give `d + e`
//! (continuous, not a distance).
//!
//! Every `a * b + c` is rounded twice (no `mul_add`), in the order the emitted shaders use.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// Floats per grid node (a row-major 3x4 affine map)
pub const GRID_WARP_NODE_FLOATS: usize = 12;

/// The stored form of a [`GridWarp`] (what is serialized): the bound is recomputed on load
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridWarpData {
    /// Position of grid node (0, 0, 0)
    pub origin: [f32; 3],
    /// Cell size (the same on the three axes), > 0
    pub cell: f32,
    /// Nodes per axis, each ≥ 2
    pub dims: [u32; 3],
    /// Node maps, x fastest then y then z, 12 floats each (row-major 3x4)
    pub maps: Vec<f32>,
    /// Radius over which the local slope bound is taken, > 0: the value is clamped to ±`reach`
    /// (a larger reach gives larger values far from the surface and a larger `Λ`)
    pub reach: f32,
}

/// A validated grid warp with its Lipschitz bound (see the module doc)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(try_from = "GridWarpData", into = "GridWarpData")]
pub struct GridWarp {
    data: GridWarpData,
    /// `Λ_n` per node (computed)
    node_lambda: Vec<f32>,
    /// slope bound of the node's field (computed)
    slope: f32,
}

/// Why a grid warp was refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridWarpError {
    /// The cell size is not finite and positive
    BadCell,
    /// An axis has fewer than 2 nodes, or the node count overflows
    BadDims,
    /// `maps.len()` is not 12 × the node count
    BadLength {
        /// Floats expected
        expected: usize,
        /// Floats given
        got: usize,
    },
    /// A stored value or the origin is not finite
    NotFinite,
    /// The computed bound is not finite or is zero (a degenerate map)
    BadBound,
    /// The reach is not finite and positive
    BadReach,
}

impl std::fmt::Display for GridWarpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadCell => write!(f, "grid warp: the cell size must be finite and > 0"),
            Self::BadDims => write!(f, "grid warp: every axis needs at least 2 nodes"),
            Self::BadLength { expected, got } => {
                write!(
                    f,
                    "grid warp: {got} map floats, expected {expected} (12 per node)"
                )
            }
            Self::NotFinite => write!(
                f,
                "grid warp: the origin and every map value must be finite"
            ),
            Self::BadReach => write!(f, "grid warp: the reach must be finite and > 0"),
            Self::BadBound => write!(
                f,
                "grid warp: the Lipschitz bound is not finite and positive"
            ),
        }
    }
}

impl std::error::Error for GridWarpError {}

impl TryFrom<GridWarpData> for GridWarp {
    type Error = GridWarpError;
    fn try_from(data: GridWarpData) -> Result<Self, GridWarpError> {
        Self::new(data)
    }
}

impl From<GridWarp> for GridWarpData {
    fn from(w: GridWarp) -> Self {
        w.data
    }
}

/// Spectral norm of a 3x3 matrix (rows): an upper bound of the square root of the largest
/// eigenvalue of AᵀA. AᵀA is diagonalised by cyclic Jacobi rotations (square roots and the four
/// operations only, a fixed 12 sweeps), then the largest Gershgorin bound `a_ii + Σ_j≠i |a_ij|` of
/// the result is taken (an upper bound of every eigenvalue), with a 1e-9 relative margin for the
/// rounding of the rotations, and never above the Frobenius norm
#[allow(
    clippy::needless_range_loop,
    reason = "a Jacobi rotation updates two rows / columns of the same matrix by index"
)]
fn spectral_norm(a: &[f32; 9]) -> f64 {
    let m: Vec<f64> = a.iter().map(|&v| f64::from(v)).collect();
    let mut s = [[0.0f64; 3]; 3];
    for (i, row) in s.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| m[k * 3 + i] * m[k * 3 + j]).sum();
        }
    }
    let frob: f64 = m.iter().map(|v| v * v).sum::<f64>().sqrt();
    for _ in 0..12 {
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if s[p][q] == 0.0 {
                continue;
            }
            let theta = (s[q][q] - s[p][p]) / (2.0 * s[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * c;
            for k in 0..3 {
                let (skp, skq) = (s[k][p], s[k][q]);
                s[k][p] = c * skp - sn * skq;
                s[k][q] = sn * skp + c * skq;
            }
            for k in 0..3 {
                let (spk, sqk) = (s[p][k], s[q][k]);
                s[p][k] = c * spk - sn * sqk;
                s[q][k] = sn * spk + c * sqk;
            }
        }
    }
    let emax = (0..3)
        .map(|i| {
            s[i][i]
                + (0..3)
                    .filter(|&j| j != i)
                    .map(|j| s[i][j].abs())
                    .sum::<f64>()
        })
        .fold(0.0f64, f64::max);
    (emax.max(0.0).sqrt() * (1.0 + 1e-9)).min(frob)
}

impl GridWarp {
    /// Validate the data and compute the Lipschitz bound
    ///
    /// # Errors
    ///
    /// [`GridWarpError`] for a bad cell, dims, length, a non-finite value or a degenerate bound.
    pub fn new(data: GridWarpData) -> Result<Self, GridWarpError> {
        if !(data.cell.is_finite() && data.cell > 0.0) {
            return Err(GridWarpError::BadCell);
        }
        if data.dims.iter().any(|&n| n < 2) {
            return Err(GridWarpError::BadDims);
        }
        let n = data
            .dims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .ok_or(GridWarpError::BadDims)?;
        let expected = n
            .checked_mul(GRID_WARP_NODE_FLOATS)
            .ok_or(GridWarpError::BadDims)?;
        if data.maps.len() != expected {
            return Err(GridWarpError::BadLength {
                expected,
                got: data.maps.len(),
            });
        }
        if data.origin.iter().chain(&data.maps).any(|v| !v.is_finite()) {
            return Err(GridWarpError::NotFinite);
        }
        if !(data.reach.is_finite() && data.reach > 0.0) {
            return Err(GridWarpError::BadReach);
        }
        let cells = Self::cell_bounds(&data);
        if cells.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(GridWarpError::BadBound);
        }
        let (node_lambda, slope) = Self::local_bounds(&data, &cells);
        if !(slope.is_finite() && slope >= 1.0)
            || node_lambda.iter().any(|v| !(v.is_finite() && *v > 0.0))
        {
            return Err(GridWarpError::BadBound);
        }
        Ok(Self {
            data,
            node_lambda,
            slope,
        })
    }

    /// `Λ_n` per node (the cube of cells within `reach + h·√3`, separable max filters, then the 8
    /// cells around the node) and the slope bound of the field
    fn local_bounds(d: &GridWarpData, cells: &[f64]) -> (Vec<f32>, f32) {
        let [nx, ny, nz] = d.dims.map(|v| v as usize);
        let (cx, cy, cz) = (nx - 1, ny - 1, nz - 1);
        let h = f64::from(d.cell);
        let k = ((f64::from(d.reach) + h * 3f64.sqrt()) / h).ceil() as usize;
        let idx = |i: usize, j: usize, l: usize| i + cx * (j + cy * l);
        let mut a = cells.to_vec();
        let mut b = vec![0.0f64; a.len()];
        for axis in 0..3 {
            let n = [cx, cy, cz][axis];
            for l in 0..cz {
                for j in 0..cy {
                    for i in 0..cx {
                        let t = [i, j, l][axis];
                        let (lo, hi) = (t.saturating_sub(k), (t + k).min(n - 1));
                        let mut m = 0.0f64;
                        for u in lo..=hi {
                            let (ii, jj, ll) = match axis {
                                0 => (u, j, l),
                                1 => (i, u, l),
                                _ => (i, j, u),
                            };
                            m = m.max(a[idx(ii, jj, ll)]);
                        }
                        b[idx(i, j, l)] = m;
                    }
                }
            }
            std::mem::swap(&mut a, &mut b);
        }
        let up = |v: f64| {
            let f = v as f32;
            if f64::from(f) < v {
                f32::from_bits(f.to_bits() + 1)
            } else {
                f
            }
        };
        let mut node = vec![0.0f32; nx * ny * nz];
        for l in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut m = 0.0f64;
                    for (ci, cj, cl) in [
                        (i, j, l),
                        (i.wrapping_sub(1), j, l),
                        (i, j.wrapping_sub(1), l),
                        (i, j, l.wrapping_sub(1)),
                        (i.wrapping_sub(1), j.wrapping_sub(1), l),
                        (i.wrapping_sub(1), j, l.wrapping_sub(1)),
                        (i, j.wrapping_sub(1), l.wrapping_sub(1)),
                        (i.wrapping_sub(1), j.wrapping_sub(1), l.wrapping_sub(1)),
                    ] {
                        if ci < cx && cj < cy && cl < cz {
                            m = m.max(a[idx(ci, cj, cl)]);
                        }
                    }
                    node[i + nx * (j + ny * l)] = up(m);
                }
            }
        }
        // slope: 1 + reach · |∇Λ| / min Λ over each cell (Λ trilinear: |∂Λ/∂k| ≤ max edge diff / h)
        let r = f64::from(d.reach);
        let nd = |i: usize, j: usize, l: usize| f64::from(node[i + nx * (j + ny * l)]);
        let mut slope = 1.0f64;
        for l in 0..cz {
            for j in 0..cy {
                for i in 0..cx {
                    let mut dk = [0.0f64; 3];
                    let mut lmin = f64::INFINITY;
                    for u in 0..2 {
                        for v in 0..2 {
                            dk[0] =
                                dk[0].max((nd(i + 1, j + u, l + v) - nd(i, j + u, l + v)).abs());
                            dk[1] =
                                dk[1].max((nd(i + u, j + 1, l + v) - nd(i + u, j, l + v)).abs());
                            dk[2] =
                                dk[2].max((nd(i + u, j + v, l + 1) - nd(i + u, j + v, l)).abs());
                            for w in 0..2 {
                                lmin = lmin.min(nd(i + u, j + v, l + w));
                            }
                        }
                    }
                    let g = [dk[0] / h, dk[1] / h, dk[2] / h];
                    let grad = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                    slope = slope.max(1.0 + r * grad / lmin);
                }
            }
        }
        (node, up(slope))
    }

    /// `λ_c` per cell (x fastest), see the module doc
    fn cell_bounds(d: &GridWarpData) -> Vec<f64> {
        let [nx, ny, nz] = d.dims.map(|v| v as usize);
        let h = f64::from(d.cell);
        let at = |i: usize, j: usize, k: usize| (i + nx * (j + ny * k)) * GRID_WARP_NODE_FLOATS;
        let norms: Vec<f64> = (0..nx * ny * nz)
            .map(|n| {
                let m = &d.maps[n * 12..n * 12 + 12];
                spectral_norm(&[m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]])
            })
            .collect();
        // ‖(A_i − A_j)(p − c) + ((A_i − A_j) c + b_i − b_j)‖ ≤ ‖A_i − A_j‖_F · r + ‖… at c‖, with
        // c the cell centre and r = h·√3/2 the distance from c to a corner
        let r = h * 3f64.sqrt() / 2.0;
        let diff = |a: usize, b: usize, c: [f64; 3]| -> f64 {
            let (ma, mb) = (&d.maps[a..a + 12], &d.maps[b..b + 12]);
            let mut fro = 0.0;
            let mut at_c = 0.0;
            for row in 0..3 {
                let mut v = f64::from(ma[row * 4 + 3]) - f64::from(mb[row * 4 + 3]);
                for col in 0..3 {
                    let e = f64::from(ma[row * 4 + col]) - f64::from(mb[row * 4 + col]);
                    fro += e * e;
                    v += e * c[col];
                }
                at_c += v * v;
            }
            fro.sqrt() * r + at_c.sqrt()
        };
        let o = d.origin.map(f64::from);
        let mut out = Vec::with_capacity((nx - 1) * (ny - 1) * (nz - 1));
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let c = [
                        o[0] + (i as f64 + 0.5) * h,
                        o[1] + (j as f64 + 0.5) * h,
                        o[2] + (k as f64 + 0.5) * h,
                    ];
                    let mut amax = 0.0f64;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                amax = amax.max(norms[(i + dx) + nx * ((j + dy) + ny * (k + dz))]);
                            }
                        }
                    }
                    let mut dk = [0.0f64; 3];
                    for a in 0..2 {
                        for b in 0..2 {
                            dk[0] =
                                dk[0].max(diff(at(i, j + a, k + b), at(i + 1, j + a, k + b), c));
                            dk[1] =
                                dk[1].max(diff(at(i + a, j, k + b), at(i + a, j + 1, k + b), c));
                            dk[2] =
                                dk[2].max(diff(at(i + a, j + b, k), at(i + a, j + b, k + 1), c));
                        }
                    }
                    let g = [dk[0] / h, dk[1] / h, dk[2] / h];
                    let grad = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                    out.push(amax + grad);
                }
            }
        }
        out
    }

    /// The slope bound of the node's field for a 1-Lipschitz child (computed, see the module doc)
    #[must_use]
    pub const fn lipschitz(&self) -> f32 {
        self.slope
    }

    /// `Λ_n` per node, x fastest (computed, see the module doc)
    #[must_use]
    pub fn node_lambda(&self) -> &[f32] {
        &self.node_lambda
    }

    /// The stored data
    #[must_use]
    pub const fn data(&self) -> &GridWarpData {
        &self.data
    }

    /// Box minimum (node (0, 0, 0))
    #[must_use]
    pub fn box_min(&self) -> Vec3 {
        Vec3::from(self.data.origin)
    }

    /// Box maximum (the last node)
    #[must_use]
    pub fn box_max(&self) -> Vec3 {
        let d = self.data.dims;
        let c = self.data.cell;
        Vec3::from(self.data.origin)
            + Vec3::new(
                (d[0] - 1) as f32 * c,
                (d[1] - 1) as f32 * c,
                (d[2] - 1) as f32 * c,
            )
    }

    /// The warped point `x(q)` for `q` = `p` clamped to the box, `e = |p − q|` and `Λ(q)`
    #[inline]
    #[must_use]
    pub fn warp(&self, p: Vec3) -> (Vec3, f32, f32) {
        let d = &self.data;
        let bmin = self.box_min();
        let q = p.clamp(bmin, self.box_max());
        let e = (p - q).length();
        let inv = 1.0 / d.cell;
        let g = (q - bmin) * inv;
        let [nx, ny, nz] = d.dims;
        let i0 = (g.x.floor() as i64).clamp(0, i64::from(nx) - 2) as usize;
        let j0 = (g.y.floor() as i64).clamp(0, i64::from(ny) - 2) as usize;
        let k0 = (g.z.floor() as i64).clamp(0, i64::from(nz) - 2) as usize;
        let f = Vec3::new(g.x - i0 as f32, g.y - j0 as f32, g.z - k0 as f32);
        let (nx, ny) = (nx as usize, ny as usize);
        let mut m = [0.0f32; 12];
        let mut lam = 0.0f32;
        // corner order (dx, dy, dz) = 000, 100, 010, 110, 001, 101, 011, 111
        for c in 0..8 {
            let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let wx = if dx == 1 { f.x } else { 1.0 - f.x };
            let wy = if dy == 1 { f.y } else { 1.0 - f.y };
            let wz = if dz == 1 { f.z } else { 1.0 - f.z };
            let w = wx * wy * wz;
            let base = ((i0 + dx) + nx * ((j0 + dy) + ny * (k0 + dz))) * GRID_WARP_NODE_FLOATS;
            for (t, mt) in m.iter_mut().enumerate() {
                *mt += w * d.maps[base + t];
            }
            lam += w * self.node_lambda[base / GRID_WARP_NODE_FLOATS];
        }
        let x = Vec3::new(
            m[0] * q.x + m[1] * q.y + m[2] * q.z + m[3],
            m[4] * q.x + m[5] * q.y + m[6] * q.z + m[7],
            m[8] * q.x + m[9] * q.y + m[10] * q.z + m[11],
        );
        (x, e, lam)
    }

    /// The node's value from the child's value `c` at the warped point and `e`, `Λ` from
    /// [`GridWarp::warp`]
    #[inline]
    #[must_use]
    pub fn finish(&self, c: f32, e: f32, lam: f32) -> f32 {
        let r = self.data.reach;
        let d = (c / lam).clamp(-r, r);
        let dp = d.max(0.0);
        d + ((e * e + dp * dp).sqrt() - dp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(
        dims: [u32; 3],
        cell: f32,
        origin: [f32; 3],
        map: impl Fn(Vec3) -> [f32; 12],
    ) -> GridWarpData {
        let mut maps = Vec::new();
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = Vec3::from(origin) + Vec3::new(i as f32, j as f32, k as f32) * cell;
                    maps.extend_from_slice(&map(p));
                }
            }
        }
        GridWarpData {
            origin,
            cell,
            dims,
            maps,
            reach: 4.0 * cell,
        }
    }
    const ID: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    #[test]
    fn identity_grid_is_the_identity_with_bound_one() {
        let w = GridWarp::new(grid([4, 5, 3], 0.1, [-0.2, -0.2, -0.1], |_| ID)).unwrap();
        assert!((w.lipschitz() - 1.0).abs() < 1e-6, "{}", w.lipschitz());
        let p = Vec3::new(0.03, 0.11, -0.05);
        let (x, e, _) = w.warp(p);
        assert!((x - p).length() < 1e-6 && e == 0.0);
    }

    #[test]
    fn a_rotation_grid_has_bound_one_and_rotates() {
        let (s, c) = alice_det_math::sin_cos(0.6f32);
        let rot = [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0];
        let w = GridWarp::new(grid([3, 3, 3], 0.2, [-0.2; 3], |_| rot)).unwrap();
        assert!((w.lipschitz() - 1.0).abs() < 1e-6, "{}", w.lipschitz());
        let p = Vec3::new(0.1, 0.05, -0.07);
        let (x, _, _) = w.warp(p);
        let want = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
        assert!((x - want).length() < 1e-6);
    }

    #[test]
    fn the_bound_holds_on_a_blended_grid() {
        // maps rotate by an angle that grows along y: the blend term must enter the bound
        let w = GridWarp::new(grid([4, 6, 4], 0.05, [-0.1, -0.1, -0.1], |p| {
            let (s, c) = alice_det_math::sin_cos(p.y * 8.0);
            [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0]
        }))
        .unwrap();
        let lmax = w.node_lambda().iter().copied().fold(0.0f32, f32::max);
        assert!(lmax > 1.05, "blend must raise the node bounds: {lmax}");
        let mut seen = 0.0f32;
        let mut n = 0;
        for a in 0..40 {
            for b in 0..40 {
                let p = Vec3::new(
                    -0.09 + 0.0045 * a as f32,
                    -0.09 + 0.0062 * b as f32,
                    0.013 * ((a * b) % 7) as f32 - 0.04,
                );
                let dp = Vec3::new(1e-3, -0.7e-3, 0.4e-3);
                let (x0, _, _) = w.warp(p);
                let (x1, _, _) = w.warp(p + dp);
                seen = seen.max((x1 - x0).length() / dp.length());
                n += 1;
            }
        }
        assert!(n == 1600 && seen > 1.0, "compared {n}, slope {seen}");
        // the warp's own slope is bounded by the node bounds Λ (each covers its cells)
        let lmax = w.node_lambda().iter().copied().fold(0.0f32, f32::max);
        assert!(
            seen <= lmax,
            "slope {seen} above the largest node bound {lmax}"
        );
    }

    #[test]
    fn outside_is_hypot_and_continuous() {
        let w = GridWarp::new(grid([2, 2, 2], 1.0, [0.0; 3], |_| ID)).unwrap();
        // the bound of an identity grid is 1 rounded up by one f32 step (an upper bound)
        assert!((w.finish(0.3, 0.0, 1.0) - 0.3).abs() < 1e-6);
        assert!((w.finish(-0.3, 0.0, 1.0) + 0.3).abs() < 1e-6);
        assert!((w.finish(0.3, 0.4, 1.0) - 0.5).abs() < 1e-6);
        assert!((w.finish(-0.3, 0.4, 1.0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn bad_data_is_refused() {
        let ok = grid([2, 2, 2], 1.0, [0.0; 3], |_| ID);
        let mut d = ok.clone();
        d.cell = 0.0;
        assert_eq!(GridWarp::new(d).unwrap_err(), GridWarpError::BadCell);
        let mut d = ok.clone();
        d.dims = [1, 2, 2];
        assert_eq!(GridWarp::new(d).unwrap_err(), GridWarpError::BadDims);
        let mut d = ok.clone();
        d.maps.pop();
        assert!(matches!(
            GridWarp::new(d).unwrap_err(),
            GridWarpError::BadLength { .. }
        ));
        let mut d = ok.clone();
        d.maps[3] = f32::NAN;
        assert_eq!(GridWarp::new(d).unwrap_err(), GridWarpError::NotFinite);
        let mut d = ok.clone();
        d.reach = 0.0;
        assert_eq!(GridWarp::new(d).unwrap_err(), GridWarpError::BadReach);
        let mut d = ok;
        d.maps.iter_mut().for_each(|v| *v = 0.0);
        assert_eq!(GridWarp::new(d).unwrap_err(), GridWarpError::BadBound);
    }

    #[test]
    fn load_recomputes_the_bound_and_validates() {
        let w = GridWarp::new(grid([3, 3, 3], 0.05, [0.0; 3], |p| {
            let (s, c) = alice_det_math::sin_cos(p.x * 9.0);
            [c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        }))
        .unwrap();
        let json = serde_json::to_string(&w).unwrap();
        assert!(!json.contains("lambda"), "the bound must not be stored");
        let back: GridWarp = serde_json::from_str(&json).unwrap();
        assert_eq!(back.lipschitz().to_bits(), w.lipschitz().to_bits());
        let bad = json.replacen("\"cell\":0.05", "\"cell\":-1.0", 1);
        assert!(serde_json::from_str::<GridWarp>(&bad).is_err());
    }
}

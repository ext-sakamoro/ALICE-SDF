//! A [`GridWarpData`] from a surface given twice: at rest and posed (a cloth after its
//! physics step, a skinned mesh in a pose).
//!
//! Each posed triangle has one affine map from the posed space back to the rest space
//! ([`affine_posed_to_rest`]): it sends the three posed vertices to the rest vertices and the
//! posed unit normal to the rest unit normal, so distances along the normal keep their length
//! (a stretched sheet keeps its thickness). Each grid node of the posed box takes the map of
//! the posed triangle nearest to it (ties to the lower triangle index, so the grid is a
//! function of the input alone). The child of the resulting [`GridWarp`] is the rest-space
//! SDF; [`GridWarp`] blends the node maps and bounds the distance, so the surface follows the
//! pose.
//!
//! [`GridWarp`]: crate::transforms::GridWarp
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::{law_max_f32, law_min_f32};
use crate::transforms::grid_warp::{GridWarpData, GRID_WARP_NODE_FLOATS};
use glam::{Mat3, Vec3};

/// Why [`grid_warp_from_triangles`] built no grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridWarpBuildError {
    /// The rest and posed vertex lists differ in length
    LengthMismatch {
        /// Rest vertices
        rest: usize,
        /// Posed vertices
        posed: usize,
    },
    /// No triangle was given
    NoTriangles,
    /// A triangle names a vertex past the end of the lists
    IndexOutOfRange {
        /// The triangle
        triangle: usize,
        /// The index it names
        index: u32,
    },
    /// A triangle has no area at rest or posed (it has no map)
    Degenerate {
        /// The triangle
        triangle: usize,
    },
    /// The box is not finite or is empty on an axis
    BadBox,
    /// The cell size is not finite and positive
    BadCell,
    /// A vertex is not finite
    NotFinite,
    /// The shell bound is not finite or is negative
    BadShell,
}

impl core::fmt::Display for GridWarpBuildError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { rest, posed } => {
                write!(f, "{rest} rest vertices but {posed} posed vertices")
            }
            Self::NoTriangles => write!(f, "no triangle"),
            Self::IndexOutOfRange { triangle, index } => {
                write!(f, "triangle {triangle} names vertex {index}, past the end")
            }
            Self::Degenerate { triangle } => write!(f, "triangle {triangle} has no area"),
            Self::BadBox => write!(f, "the box is not finite or is empty on an axis"),
            Self::BadCell => write!(f, "the cell size is not finite and positive"),
            Self::NotFinite => write!(f, "a vertex is not finite"),
            Self::BadShell => write!(f, "the shell bound is not finite or is negative"),
        }
    }
}

impl std::error::Error for GridWarpBuildError {}

/// Unit normal of a triangle, `None` without area (or not finite).
fn unit_normal(t: [Vec3; 3]) -> Option<Vec3> {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let n = e1.cross(e2);
    let len = n.length();
    // no area relative to the edges (also catches a non-finite product)
    if !(len.is_finite() && len > 1e-12 * e1.length() * e2.length() && len > 0.0) {
        return None;
    }
    Some(n / len)
}

/// The affine map that sends a posed triangle back to its rest triangle.
///
/// Row-major 3×4, the layout of [`GridWarpData::maps`]. It sends the posed vertices to the rest
/// vertices and the posed unit normal (from the first vertex) to the rest unit normal. `None`
/// if either triangle has no area.
#[must_use]
pub fn affine_posed_to_rest(rest: [Vec3; 3], posed: [Vec3; 3]) -> Option<[f32; 12]> {
    let nr = unit_normal(rest)?;
    let np = unit_normal(posed)?;
    let r = Mat3::from_cols(rest[1] - rest[0], rest[2] - rest[0], nr);
    let p = Mat3::from_cols(posed[1] - posed[0], posed[2] - posed[0], np);
    let det = p.determinant();
    if !(det.is_finite() && det != 0.0) {
        return None;
    }
    let a = r * p.inverse();
    let b = rest[0] - a * posed[0];
    let m = [
        a.x_axis.x, a.y_axis.x, a.z_axis.x, b.x, //
        a.x_axis.y, a.y_axis.y, a.z_axis.y, b.y, //
        a.x_axis.z, a.y_axis.z, a.z_axis.z, b.z,
    ];
    m.iter().all(|v| v.is_finite()).then_some(m)
}

/// Squared distance from `p` to the triangle `abc` (closest point by Voronoi regions).
fn point_triangle_distance_sq(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length_squared();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length_squared();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length_squared();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length_squared()
}

/// The grid [`grid_warp_from_triangles`] builds: the posed box it covers, its cell size, the
/// reach of [`GridWarpData::reach`], and the optional shell bound that turns on the floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WarpGrid {
    /// Lower corner of the posed box
    pub box_min: Vec3,
    /// Upper corner of the posed box
    pub box_max: Vec3,
    /// Cell size, finite and positive
    pub cell: f32,
    /// See [`GridWarpData::reach`]
    pub reach: f32,
    /// The caller's bound: every point of the child lies within this distance of the rest sheet
    /// (`None`: no floor, no fold handling)
    pub shell: Option<f32>,
}

/// Upper bound on the largest singular value of the linear part of a row-major 3×4 map: the
/// square root of the largest absolute row sum of `AᵀA` (Gershgorin; exact for a rotation).
fn spectral_bound(m: &[f32; 12]) -> f32 {
    let a = [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]];
    let mut worst = 0.0f32;
    for i in 0..3 {
        let mut row = 0.0f32;
        for j in 0..3 {
            let ata = a[0][i] * a[0][j] + a[1][i] * a[1][j] + a[2][i] * a[2][j];
            row += ata.abs();
        }
        worst = law_max_f32(worst, row);
    }
    worst.sqrt()
}

/// A [`GridWarpData`] that warps a rest-space SDF onto the posed surface.
///
/// The grid covers the posed box `[box_min, box_max]` with cells of size `cell`
/// (`ceil(extent / cell) + 1` nodes per axis from `box_min`); each node holds the
/// [`affine_posed_to_rest`] map of the posed triangle nearest to it, and the grid has the given
/// `reach` (see [`GridWarpData::reach`]). `rest` and `posed` list the same vertices;
/// `triangles` index them.
///
/// **Shell and floor.** Where the posed sheet folds, a node's map can send a point to a rest
/// point whose image is elsewhere, and the warped field shows surfaces that are not there.
/// With `grid.shell = Some(t)` (the caller's bound: every point of the child lies within `t` of the
/// rest sheet) each node also gets a floor ([`GridWarpData::floor`]) that removes them:
///
/// - for a child point `x` whose nearest rest triangle is `T` with foot `f` (`|x − f| ≤ t`),
///   the triangle's own rest → posed map `F_T` gives `|F_T(x) − F_T(f)| ≤ σ_T·|x − f| ≤ σ_T·t`
///   with `F_T(f)` on the posed sheet and `σ_T` the largest singular value of its linear part
///   (≥ 1: the unit normal is kept, so stretch in the plane only raises it); hence every point
///   of the posed shape lies within `σ·t` of the posed sheet, `σ = max_T σ_T`;
/// - so for a node `n`, `dist(n, shape) ≥ dist(n, posed sheet) − σ·t`;
/// - the node's floor is that minus the cell diagonal `√3·cell`: the distance is 1-Lipschitz,
///   so every corner's floor is at most the distance at any point of the cell, and so is their
///   trilinear blend, which [`GridWarp`](crate::transforms::GridWarp) applies as a lower bound.
///
/// `σ_T` is bounded from above by the Gershgorin bound on `AᵀA` (sound, exact for a rotation).
/// `grid.shell = None` leaves the floor empty (no fold handling).
///
/// # Errors
///
/// [`GridWarpBuildError`] for mismatched or out-of-range input, a triangle without area, a
/// non-finite vertex, an empty or non-finite box, or a cell size that is not finite and
/// positive. The reach is checked by [`GridWarp::new`](crate::transforms::GridWarp::new).
pub fn grid_warp_from_triangles(
    rest: &[Vec3],
    posed: &[Vec3],
    triangles: &[[u32; 3]],
    grid: &WarpGrid,
) -> Result<GridWarpData, GridWarpBuildError> {
    let WarpGrid {
        box_min,
        box_max,
        cell,
        reach,
        shell,
    } = *grid;
    if rest.len() != posed.len() {
        return Err(GridWarpBuildError::LengthMismatch {
            rest: rest.len(),
            posed: posed.len(),
        });
    }
    if triangles.is_empty() {
        return Err(GridWarpBuildError::NoTriangles);
    }
    if rest.iter().chain(posed).any(|v| !v.is_finite()) {
        return Err(GridWarpBuildError::NotFinite);
    }
    if !(cell.is_finite() && cell > 0.0) {
        return Err(GridWarpBuildError::BadCell);
    }
    if !(box_min.is_finite() && box_max.is_finite())
        || box_max.x <= box_min.x
        || box_max.y <= box_min.y
        || box_max.z <= box_min.z
    {
        return Err(GridWarpBuildError::BadBox);
    }
    if let Some(t) = shell {
        if !(t.is_finite() && t >= 0.0) {
            return Err(GridWarpBuildError::BadShell);
        }
    }
    let mut tris = Vec::with_capacity(triangles.len());
    let mut sigma = 0.0f32;
    for (k, t) in triangles.iter().enumerate() {
        let mut r = [Vec3::ZERO; 3];
        let mut p = [Vec3::ZERO; 3];
        for (slot, &i) in t.iter().enumerate() {
            let v = i as usize;
            if v >= rest.len() {
                return Err(GridWarpBuildError::IndexOutOfRange {
                    triangle: k,
                    index: i,
                });
            }
            r[slot] = rest[v];
            p[slot] = posed[v];
        }
        let m = affine_posed_to_rest(r, p).ok_or(GridWarpBuildError::Degenerate { triangle: k })?;
        let forward =
            affine_posed_to_rest(p, r).ok_or(GridWarpBuildError::Degenerate { triangle: k })?;
        sigma = law_max_f32(sigma, spectral_bound(&forward));
        tris.push((p, m));
    }
    let extent = box_max - box_min;
    let nodes = |e: f32| -> Result<u32, GridWarpBuildError> {
        let n = (e / cell).ceil() + 1.0;
        if n.is_finite() && n <= u32::MAX as f32 {
            Ok(n as u32)
        } else {
            Err(GridWarpBuildError::BadCell)
        }
    };
    let dims = [nodes(extent.x)?, nodes(extent.y)?, nodes(extent.z)?];
    let count = dims
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .ok_or(GridWarpBuildError::BadCell)?;
    let mut maps = Vec::with_capacity(count * GRID_WARP_NODE_FLOATS);
    let mut floor = Vec::with_capacity(if shell.is_some() { count } else { 0 });
    let diagonal = cell * 3.0f32.sqrt();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let q = box_min + Vec3::new(i as f32, j as f32, k as f32) * cell;
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for (t, (p, _)) in tris.iter().enumerate() {
                    let d = point_triangle_distance_sq(q, p[0], p[1], p[2]);
                    // strictly smaller: a tie keeps the lower index
                    if d < best_d {
                        best_d = law_min_f32(best_d, d);
                        best = t;
                    }
                }
                maps.extend_from_slice(&tris[best].1);
                if let Some(t) = shell {
                    floor.push(best_d.sqrt() - sigma * t - diagonal);
                }
            }
        }
    }
    Ok(GridWarpData {
        origin: box_min.to_array(),
        cell,
        dims,
        maps,
        reach,
        floor,
    })
}

//! Topology-independent sign determination for mesh-derived SDFs.
//!
//! The unsigned distance to a triangle soup is always well defined. The sign is
//! not: taking it from the face normal of the closest triangle makes the field a
//! function of the input **winding**, so an open sheet grows a phantom interior,
//! reversing the index order negates the whole field, and a mesh assembled from
//! independently wound parts reports the wrong side inside some of them.
//!
//! This module decides the sign from the geometry instead:
//!
//! ```text
//! UDF(x) = min ‖x − y‖               y on the mesh
//! T(x)   ∈ {0, 1}                    1 iff x is reachable from outside the
//!                                    padded bounding box without crossing the
//!                                    surface
//! SDF(x) = (1 − 2·T(x)) · UDF(x)
//! ```
//!
//! `T` comes from a 6-connected flood fill over a uniform voxel grid, seeded on
//! the grid boundary. A cell is *blocked* when the unsigned distance at its
//! centre is within half a cell diagonal, which is exactly the condition that
//! covers every cell the surface passes through — so for a closed surface the
//! blocked set seals the fill and cannot leak inside.
//!
//! Nothing here reads a triangle normal, so reversing the winding of any part of
//! the mesh leaves the field unchanged.
//!
//! # Resolution is the accuracy knob
//!
//! The method resolves a feature only when the grid does. A wall or a gap
//! thinner than roughly two cells is closed over: a thin gap reads as solid, a
//! thin shell loses its cavity. Raise
//! [`MeshToSdfConfig::sign_flood_fill_resolution`](crate::mesh::MeshToSdfConfig)
//! until the thinnest feature spans several cells.
//!
//! # Determinism
//!
//! Cell occupancy is an independent per-cell distance query (no reduction), the
//! flood fill walks integer indices, and every float operation is add / sub /
//! mul / div / compare / sqrt. The result is bit-identical across targets.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

use crate::mesh::bvh::MeshBvh;
use crate::mesh::MeshInputError;

/// How the sign of a mesh-derived signed distance field is decided.
///
/// `#[non_exhaustive]` so a future sign rule is not a breaking change; the unit
/// variants stay nameable and constructible from outside the crate, only a
/// wildcard-less `match` is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MeshSignMode {
    /// Sign taken from the face normal of the closest triangle.
    ///
    /// Correct only for a closed, manifold, consistently wound mesh, and even
    /// then only where the closest feature is a face rather than an edge or a
    /// vertex. Kept as the default so existing callers are unaffected.
    #[default]
    NearestFaceNormal,
    /// Sign taken from a volumetric flood fill seeded outside the bounding box.
    ///
    /// Independent of winding, of whether the surface is closed, and of how many
    /// connected components it has. See the module docs for the resolution
    /// caveat.
    ExteriorFloodFill,
}

/// Voxelised "is this point outside the mesh" classifier.
///
/// Build it once per mesh with [`ExteriorField::build`], then ask
/// [`ExteriorField::signed_distance`] per query point.
#[derive(Debug, Clone)]
pub struct ExteriorField {
    origin: Vec3,
    cell: f32,
    dims: [usize; 3],
    /// The surface may pass through this cell, so its inside/outside label is
    /// not decided by the fill.
    blocked: Vec<bool>,
    /// Reachable from the grid boundary without crossing a blocked cell.
    exterior: Vec<bool>,
}

impl ExteriorField {
    /// Cells of empty space kept on every side of the mesh bounding box.
    ///
    /// Two cells put every boundary cell centre at least 1.5 cells away from the
    /// mesh, which is more than the half cell diagonal (≈0.87 cells) that marks
    /// a cell blocked — so the seed cells are guaranteed to be free.
    pub const PADDING_CELLS: usize = 2;

    /// How far the narrow-band walk (see [`Self::is_exterior`]) may step before
    /// giving up. The blocked band is about two cells thick, so six steps clears
    /// it with margin.
    pub const MAX_BAND_STEPS: usize = 6;

    /// Upper bound on the grid size, to keep a large `resolution` on an
    /// elongated mesh from asking for an unbounded allocation.
    pub const MAX_CELLS: usize = 1 << 24;

    /// Build the classifier for `bvh`.
    ///
    /// `resolution` is the number of cells along the **longest** bounding box
    /// axis; cells are cubic, so the other axes get proportionally fewer.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or degenerate (zero-extent, non-finite)
    /// mesh, and when `resolution` would need more than 16.7M cells.
    pub fn build(bvh: &MeshBvh, resolution: u32) -> Result<Self, MeshInputError> {
        let bounds = bvh.bounds().ok_or(MeshInputError {
            reason: "exterior flood fill needs a non-empty mesh",
        })?;
        let extent = bounds.max - bounds.min;
        let longest = extent.x.max(extent.y).max(extent.z);
        if !longest.is_finite() || longest <= 0.0 {
            return Err(MeshInputError {
                reason: "exterior flood fill needs a mesh with finite, non-zero extent",
            });
        }

        let res = f64::from(resolution.max(4));
        let cell = (f64::from(longest) / res) as f32;
        if cell <= 0.0 || !cell.is_finite() {
            return Err(MeshInputError {
                reason: "exterior flood fill resolution underflows the cell size",
            });
        }

        let pad = Self::PADDING_CELLS;
        let axis_dim = |e: f32| -> usize {
            let n = (f64::from(e) / f64::from(cell)).ceil();
            let n = if n.is_finite() && n >= 1.0 {
                n as usize
            } else {
                1
            };
            n + 2 * pad
        };
        let dims = [axis_dim(extent.x), axis_dim(extent.y), axis_dim(extent.z)];
        let total = dims[0]
            .checked_mul(dims[1])
            .and_then(|n| n.checked_mul(dims[2]))
            .ok_or(MeshInputError {
                reason: "exterior flood fill grid size overflows",
            })?;
        if total > Self::MAX_CELLS {
            return Err(MeshInputError {
                reason: "exterior flood fill resolution exceeds the 16.7M cell budget",
            });
        }

        let origin = bounds.min - Vec3::splat(cell * pad as f32);

        // A cell whose centre is within half a cell diagonal of the surface may
        // contain surface, so block it. Every cell the surface actually crosses
        // satisfies this, which is what makes the fill leak-proof.
        let half_diag = cell * 0.5 * 3.0_f32.sqrt();
        let skeleton = Self {
            origin,
            cell,
            dims,
            blocked: Vec::new(),
            exterior: Vec::new(),
        };
        let blocked: Vec<bool> = (0..total)
            .into_par_iter()
            .map(|idx| bvh.unsigned_distance(skeleton.cell_center(idx)) <= half_diag)
            .collect();

        let exterior = Self::fill_from_boundary(&blocked, dims);

        Ok(Self {
            blocked,
            exterior,
            ..skeleton
        })
    }

    /// 6-connected flood fill through free cells, seeded on every grid boundary
    /// cell. Iterative (explicit stack) so deep grids cannot blow the call
    /// stack, and index-only so the result does not depend on float rounding.
    fn fill_from_boundary(blocked: &[bool], dims: [usize; 3]) -> Vec<bool> {
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let mut exterior = vec![false; blocked.len()];
        let mut stack: Vec<usize> = Vec::new();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let on_boundary =
                        x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz;
                    if !on_boundary {
                        continue;
                    }
                    let idx = x + y * nx + z * nx * ny;
                    if !blocked[idx] && !exterior[idx] {
                        exterior[idx] = true;
                        stack.push(idx);
                    }
                }
            }
        }

        while let Some(idx) = stack.pop() {
            let x = idx % nx;
            let y = (idx / nx) % ny;
            let z = idx / (nx * ny);

            let mut neighbours = [usize::MAX; 6];
            let mut n = 0;
            let push = |v: usize, n: &mut usize, buf: &mut [usize; 6]| {
                buf[*n] = v;
                *n += 1;
            };
            if x > 0 {
                push(idx - 1, &mut n, &mut neighbours);
            }
            if x + 1 < nx {
                push(idx + 1, &mut n, &mut neighbours);
            }
            if y > 0 {
                push(idx - nx, &mut n, &mut neighbours);
            }
            if y + 1 < ny {
                push(idx + nx, &mut n, &mut neighbours);
            }
            if z > 0 {
                push(idx - nx * ny, &mut n, &mut neighbours);
            }
            if z + 1 < nz {
                push(idx + nx * ny, &mut n, &mut neighbours);
            }

            for &nb in &neighbours[..n] {
                if !blocked[nb] && !exterior[nb] {
                    exterior[nb] = true;
                    stack.push(nb);
                }
            }
        }

        exterior
    }

    #[inline]
    fn cell_center(&self, idx: usize) -> Vec3 {
        let (nx, ny) = (self.dims[0], self.dims[1]);
        let x = idx % nx;
        let y = (idx / nx) % ny;
        let z = idx / (nx * ny);
        self.origin
            + Vec3::new(
                (x as f32 + 0.5) * self.cell,
                (y as f32 + 0.5) * self.cell,
                (z as f32 + 0.5) * self.cell,
            )
    }

    /// Index of the cell containing `point`, or `None` when `point` lies outside
    /// the padded grid (which, by construction, means outside the mesh).
    #[inline]
    fn cell_of(&self, point: Vec3) -> Option<usize> {
        let local = (point - self.origin) / self.cell;
        // `is_finite()` first so a NaN coordinate is rejected rather than
        // sliding through a bare `<` comparison.
        let axis = |v: f32, n: usize| -> Option<usize> {
            if !v.is_finite() || v < 0.0 {
                return None;
            }
            let i = v.floor();
            if !i.is_finite() || i >= n as f32 {
                return None;
            }
            Some(i as usize)
        };
        let x = axis(local.x, self.dims[0])?;
        let y = axis(local.y, self.dims[1])?;
        let z = axis(local.z, self.dims[2])?;
        Some(x + y * self.dims[0] + z * self.dims[0] * self.dims[1])
    }

    /// Whether `point` is outside the mesh.
    ///
    /// Cells the fill classified answer directly. For a point inside the blocked
    /// band around the surface, the classification is taken from the first
    /// classified cell reached by stepping **away from the closest surface
    /// point**, one cell at a time — a purely geometric direction, so this stays
    /// winding-independent while resolving the sign below cell size.
    ///
    /// Three cases answer `true` by definition rather than by measurement, and
    /// all three are outside-or-irrelevant: a point beyond the padded grid, a
    /// point sitting exactly on the surface (where the signed distance is 0
    /// either way), and a band point whose walk never leaves the band within
    /// [`Self::MAX_BAND_STEPS`] — geometry finer than the grid, which the module
    /// docs call out as the resolution limit.
    pub fn is_exterior(&self, bvh: &MeshBvh, point: Vec3) -> bool {
        let Some(idx) = self.cell_of(point) else {
            return true;
        };
        if !self.blocked[idx] {
            return self.exterior[idx];
        }

        let Some(closest) = bvh.closest_point(point) else {
            return true;
        };
        let away = point - closest;
        let len = away.length();
        if !len.is_finite() || len <= 0.0 {
            return true;
        }
        let dir = away / len;

        for step in 1..=Self::MAX_BAND_STEPS {
            let probe = point + dir * (self.cell * step as f32);
            match self.cell_of(probe) {
                None => return true,
                Some(j) if !self.blocked[j] => return self.exterior[j],
                Some(_) => {}
            }
        }
        true
    }

    /// `(1 − 2·T(x)) · UDF(x)`: negative inside, positive outside, zero on the
    /// surface.
    #[inline]
    pub fn signed_distance(&self, bvh: &MeshBvh, point: Vec3) -> f32 {
        let udf = bvh.unsigned_distance(point);
        if self.is_exterior(bvh, point) {
            udf
        } else {
            -udf
        }
    }

    /// Cell size of the grid (cubic).
    #[inline]
    pub const fn cell_size(&self) -> f32 {
        self.cell
    }

    /// Grid dimensions in cells, including the empty padding on each side.
    #[inline]
    pub const fn dims(&self) -> [usize; 3] {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed unit cube, outward CCW winding.
    fn cube() -> (Vec<Vec3>, Vec<u32>) {
        let mut verts = Vec::with_capacity(8);
        for &sz in &[-1.0f32, 1.0] {
            for &sy in &[-1.0f32, 1.0] {
                for &sx in &[-1.0f32, 1.0] {
                    verts.push(Vec3::new(sx, sy, sz));
                }
            }
        }
        let mut indices = Vec::new();
        let mut quad = |a: u32, b: u32, c: u32, d: u32| {
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        };
        quad(0, 2, 3, 1);
        quad(4, 5, 7, 6);
        quad(0, 4, 6, 2);
        quad(1, 3, 7, 5);
        quad(0, 1, 5, 4);
        quad(2, 6, 7, 3);
        (verts, indices)
    }

    #[test]
    fn padding_cells_are_never_blocked() {
        let (v, i) = cube();
        let bvh = MeshBvh::build(&v, &i, 4);
        let field = ExteriorField::build(&bvh, 16).expect("cube is a valid mesh");
        let [nx, ny, nz] = field.dims();
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz {
                        let idx = x + y * nx + z * nx * ny;
                        assert!(
                            !field.blocked[idx],
                            "boundary cell ({x},{y},{z}) is blocked"
                        );
                        assert!(
                            field.exterior[idx],
                            "boundary cell ({x},{y},{z}) is not filled"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cube_interior_is_not_reached_by_the_fill() {
        let (v, i) = cube();
        let bvh = MeshBvh::build(&v, &i, 4);
        let field = ExteriorField::build(&bvh, 24).expect("cube is a valid mesh");
        assert!(!field.is_exterior(&bvh, Vec3::ZERO));
        assert!(field.is_exterior(&bvh, Vec3::new(0.0, 0.0, 3.0)));
    }

    #[test]
    fn degenerate_mesh_is_rejected() {
        // all vertices coincident -> zero extent
        let verts = vec![Vec3::ZERO; 3];
        let bvh = MeshBvh::build(&verts, &[0, 1, 2], 4);
        assert!(ExteriorField::build(&bvh, 16).is_err());
    }
}

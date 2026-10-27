//! Dual Contouring mesh generation
//!
//! Unlike Marching Cubes, Dual Contouring places one vertex per cell
//! using QEF (Quadric Error Function) minimization, preserving sharp
//! edges and corners. Uses existing Hermite data infrastructure.
//!
//! # Deep Fried Optimizations
//! - **Parallel grid evaluation**: Rayon par_iter for SDF sampling.
//! - **Parallel QEF solve**: Each cell solved independently.
//! - **Forced inlining**: Hot-path helpers use `#[inline(always)]`.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use crate::compiled::real::LawVec;
use crate::compiled::{eval_compiled, CompiledSdf};
use crate::eval::{eval, normal};
use crate::mesh::{Mesh, Vertex};
use crate::SdfNode;
use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

/// Configuration for Dual Contouring
#[derive(Debug, Clone, Copy)]
pub struct DualContouringConfig {
    /// Grid resolution along each axis (default: 64)
    pub resolution: usize,
    /// Epsilon for gradient/normal computation (default: 0.001)
    pub gradient_epsilon: f32,
    /// Number of bisection iterations for edge intersection (default: 6)
    pub bisection_iterations: u32,
    /// Whether to compute vertex normals (default: true)
    pub compute_normals: bool,
    /// Whether to generate triplanar UV coordinates (default: false)
    pub compute_uvs: bool,
    /// UV tiling scale (default: 1.0)
    pub uv_scale: f32,
    /// Whether to compute tangent vectors (default: false)
    pub compute_tangents: bool,
    /// Whether to evaluate material IDs (default: false)
    pub compute_materials: bool,
    /// Clamp QEF vertex to cell bounds (default: true)
    ///
    /// When true, the solved vertex is clamped to its cell AABB,
    /// preventing degenerate geometry from ill-conditioned QEF systems.
    pub clamp_to_cell: bool,
}

impl Default for DualContouringConfig {
    fn default() -> Self {
        Self {
            resolution: 64,
            gradient_epsilon: 0.001,
            bisection_iterations: 6,
            compute_normals: true,
            compute_uvs: false,
            uv_scale: 1.0,
            compute_tangents: false,
            compute_materials: false,
            clamp_to_cell: true,
        }
    }
}

impl DualContouringConfig {
    /// AAA preset with all features enabled
    pub fn aaa(resolution: usize) -> Self {
        Self {
            resolution,
            compute_normals: true,
            compute_uvs: true,
            uv_scale: 1.0,
            compute_tangents: true,
            compute_materials: true,
            ..Default::default()
        }
    }
}

// ── QEF solver ──────────────────────────────────────────────────────

/// Solve the Quadric Error Function: find point p that minimizes
/// Σ (n_i · (p - x_i))² over all Hermite intersection (x_i, n_i).
///
/// Uses AtA pseudo-inverse via SVD-free Cramer's rule (3×3 system).
/// Falls back to mass-point (centroid) if the system is singular.
#[inline(always)]
fn qef_solve(intersections: &[(Vec3, Vec3)], cell_min: Vec3, cell_max: Vec3, clamp: bool) -> Vec3 {
    if intersections.is_empty() {
        return (cell_min + cell_max) * 0.5;
    }

    // Mass point (centroid of intersections) — used as origin shift for numerical stability
    let mass_point =
        intersections.iter().map(|(p, _)| *p).sum::<Vec3>() / intersections.len() as f32;

    // Build AᵀA and Aᵀb (3×3 symmetric system)
    let mut ata = [[0.0f32; 3]; 3];
    let mut atb = [0.0f32; 3];

    for &(point, normal) in intersections {
        let n = [normal.x, normal.y, normal.z];
        let d = point - mass_point;
        let rhs = normal.dot(d);

        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += n[i] * n[j];
            }
            atb[i] += n[i] * rhs;
        }
    }

    // Solve via Cramer's rule with regularization
    let result = solve_3x3_regularized(&ata, &atb);

    let vertex = result.map_or(mass_point, |v| mass_point + Vec3::new(v[0], v[1], v[2]));

    if clamp {
        vertex.law_clamp(cell_min, cell_max)
    } else {
        vertex
    }
}

/// Solve 3×3 linear system with Tikhonov regularization.
/// Returns None if still degenerate after regularization.
#[inline(always)]
fn solve_3x3_regularized(ata: &[[f32; 3]; 3], atb: &[f32; 3]) -> Option<[f32; 3]> {
    // Add small regularization to diagonal (Tikhonov, λ = 0.01)
    let lambda = 0.01f32;
    let a = [
        [ata[0][0] + lambda, ata[0][1], ata[0][2]],
        [ata[1][0], ata[1][1] + lambda, ata[1][2]],
        [ata[2][0], ata[2][1], ata[2][2] + lambda],
    ];

    // Determinant
    let det = a[0][2] * (a[1][0] * a[2][1] + (-(a[1][1] * a[2][0])))
        + (a[0][0] * (a[1][1] * a[2][2] + (-(a[1][2] * a[2][1])))
            + (-(a[0][1] * (a[1][0] * a[2][2] + (-(a[1][2] * a[2][0]))))));

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;

    // Cramer's rule
    let x = (a[0][2] * (atb[1] * a[2][1] + (-(a[1][1] * atb[2])))
        + (atb[0] * (a[1][1] * a[2][2] + (-(a[1][2] * a[2][1])))
            + (-(a[0][1] * (atb[1] * a[2][2] + (-(a[1][2] * atb[2])))))))
        * inv_det;

    let y = (a[0][2] * (a[1][0] * atb[2] + (-(atb[1] * a[2][0])))
        + (a[0][0] * (atb[1] * a[2][2] + (-(a[1][2] * atb[2])))
            + (-(atb[0] * (a[1][0] * a[2][2] + (-(a[1][2] * a[2][0])))))))
        * inv_det;

    let z = (atb[0] * (a[1][0] * a[2][1] + (-(a[1][1] * a[2][0])))
        + (a[0][0] * (a[1][1] * atb[2] + (-(atb[1] * a[2][1])))
            + (-(a[0][1] * (a[1][0] * atb[2] + (-(atb[1] * a[2][0])))))))
        * inv_det;

    if x.is_finite() && y.is_finite() && z.is_finite() {
        Some([x, y, z])
    } else {
        None
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Find surface intersection along an edge using bisection
#[inline(always)]
fn bisect_edge(
    sdf_fn: impl Fn(Vec3) -> f32,
    a: Vec3,
    b: Vec3,
    da: f32,
    db: f32,
    iters: u32,
) -> Vec3 {
    let mut lo = a;
    let mut hi = b;
    let mut dlo = da;
    let mut dhi = db;

    for _ in 0..iters {
        let t = dlo / (dlo - dhi);
        let mid = lo.lerp(hi, t.clamp(0.0, 1.0));
        let dm = sdf_fn(mid);

        if dm.abs() < 1e-6 {
            return mid;
        }

        if (dlo < 0.0) == (dm < 0.0) {
            lo = mid;
            dlo = dm;
        } else {
            hi = mid;
            dhi = dm;
        }
    }

    let t = dlo / (dlo - dhi);
    lo.lerp(hi, t.clamp(0.0, 1.0))
}

/// Compute gradient (normal) via central differences
#[inline(always)]
fn gradient_cd(sdf_fn: impl Fn(Vec3) -> f32, p: Vec3, eps: f32) -> Vec3 {
    Vec3::new(
        sdf_fn(p + Vec3::X * eps) - sdf_fn(p - Vec3::X * eps),
        sdf_fn(p + Vec3::Y * eps) - sdf_fn(p - Vec3::Y * eps),
        sdf_fn(p + Vec3::Z * eps) - sdf_fn(p - Vec3::Z * eps),
    )
    .normalize_or_zero()
}

/// Triplanar UV from position and normal
#[inline(always)]
fn triplanar_uv(position: Vec3, normal: Vec3, scale: f32) -> Vec2 {
    let abs_n = normal.abs();
    let inv_scale = 1.0 / scale;

    if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
        Vec2::new(position.y * inv_scale, position.z * inv_scale)
    } else if abs_n.y >= abs_n.z {
        Vec2::new(position.x * inv_scale, position.z * inv_scale)
    } else {
        Vec2::new(position.x * inv_scale, position.y * inv_scale)
    }
}

/// Compute tangent from normal
#[inline(always)]
fn compute_tangent(normal: Vec3) -> Vec4 {
    let abs_n = normal.abs();
    let tangent = if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
        let t = Vec3::new(0.0, 1.0, 0.0);
        (t - normal * normal.dot(t)).normalize_or_zero()
    } else {
        let t = Vec3::new(1.0, 0.0, 0.0);
        (t - normal * normal.dot(t)).normalize_or_zero()
    };
    Vec4::new(tangent.x, tangent.y, tangent.z, 1.0)
}

// ── Cell topology ───────────────────────────────────────────────────

/// How many vertices a cell gets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DcTopology {
    /// One vertex per cell that the surface crosses (the classic scheme).
    /// Where two sheets pass through one cell they share that vertex, and
    /// where a grid face has alternating corner signs (+ − + −) on both of
    /// its cells, four quads meet on one vertex pair: a non-manifold edge.
    #[default]
    OneVertexPerCell,
    /// One vertex per sheet in a cell (manifold dual contouring). The
    /// crossed edges of a cell are grouped by the loops the surface draws on
    /// the cell's faces: on a face with two crossed edges they belong to one
    /// loop; on a face with four (alternating corners) the asymptotic decider
    /// of the bilinear interpolant on that face decides which pair of
    /// diagonal corners is connected through the face, and the crossings are
    /// paired around the two corners that are cut off. The decider reads
    /// only the face's four values, so the two cells sharing a face pair its
    /// edges the same way. Each loop gets its own QEF vertex and owns its
    /// edges; a quad takes from each of its four cells the vertex owning its
    /// edge. A sheet narrower than a cell that passes a face twice in both of
    /// its cells (a tube thinner than the grid) still joins one vertex pair
    /// twice: the grid does not resolve it. Loops are grouped by the faces
    /// only: where the trilinear interpolant joins two loops of one cell
    /// through its interior (a tunnel), they stay two sheets, so the mesh is
    /// manifold but its topology inside that cell can differ from the
    /// trilinear surface's.
    Manifold,
}

/// Per-cell output: the vertices (position, normal) and, for each of the 12
/// local edges, the index of the vertex owning it (`u8::MAX`: not crossed)
struct CellVertices {
    vertices: Vec<(Vec3, Vec3)>,
    edge_vertex: [u8; 12],
}

/// The 12 edges of a cell: corner indices and the corners' grid offsets.
/// Corner k sits at offset (k & 1, (k >> 1) & 1, k >> 2).
const EDGES: [(usize, usize, [usize; 3], [usize; 3]); 12] = [
    // Bottom face (z=0)
    (0, 1, [0, 0, 0], [1, 0, 0]),
    (2, 3, [0, 1, 0], [1, 1, 0]),
    (0, 2, [0, 0, 0], [0, 1, 0]),
    (1, 3, [1, 0, 0], [1, 1, 0]),
    // Top face (z=1)
    (4, 5, [0, 0, 1], [1, 0, 1]),
    (6, 7, [0, 1, 1], [1, 1, 1]),
    (4, 6, [0, 0, 1], [0, 1, 1]),
    (5, 7, [1, 0, 1], [1, 1, 1]),
    // Vertical edges
    (0, 4, [0, 0, 0], [0, 0, 1]),
    (1, 5, [1, 0, 0], [1, 0, 1]),
    (2, 6, [0, 1, 0], [0, 1, 1]),
    (3, 7, [1, 1, 0], [1, 1, 1]),
];

/// The 6 faces of a cell: corners a, b, c, d in cyclic order and the edges
/// ab, bc, cd, da (indices into `EDGES`). Opposite faces list their corners
/// at the same in-plane offsets, so a face reads the same four values in the
/// same order from both of its cells.
const FACES: [([usize; 4], [usize; 4]); 6] = [
    ([0, 1, 3, 2], [0, 3, 1, 2]),   // z = 0
    ([4, 5, 7, 6], [4, 7, 5, 6]),   // z = 1
    ([0, 1, 5, 4], [0, 9, 4, 8]),   // y = 0
    ([2, 3, 7, 6], [1, 11, 5, 10]), // y = 1
    ([0, 2, 6, 4], [2, 10, 6, 8]),  // x = 0
    ([1, 3, 7, 5], [3, 11, 7, 9]),  // x = 1
];

/// Group a cell's crossed edges into sheets: the label of each edge's loop
/// (the smallest edge index on the loop; uncrossed edges keep their own)
fn sheet_of_edges(corners: &[f32; 8]) -> [usize; 12] {
    let mut parent: [usize; 12] = core::array::from_fn(|i| i);
    const fn root(parent: &mut [usize; 12], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut join = |a: usize, b: usize| {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi] = lo;
        }
    };
    let inside = |v: f32| v < 0.0;
    for (fc, fe) in FACES {
        let v = fc.map(|k| corners[k]);
        let crossed = [0, 1, 2, 3].map(|i| inside(v[i]) != inside(v[(i + 1) % 4]));
        match crossed.iter().filter(|&&c| c).count() {
            2 => {
                let mut pair = (0..4).filter(|&i| crossed[i]).map(|i| fe[i]);
                let (a, b) = (pair.next().unwrap_or(0), pair.next().unwrap_or(0));
                join(a, b);
            }
            4 => {
                // Asymptotic decider: the bilinear interpolant's value at its
                // saddle. a, c are on one side of 0 and b, d on the other
                // (exactly 0 counts as outside), so a + c and b + d differ in
                // sign and the denominator is never 0.
                let saddle = (v[0] * v[2] - v[1] * v[3]) / (v[0] + v[2] - v[1] - v[3]);
                if inside(saddle) == inside(v[0]) {
                    // a and c are connected: the loops cut off b and d
                    join(fe[0], fe[1]);
                    join(fe[2], fe[3]);
                } else {
                    // b and d are connected: the loops cut off a and c
                    join(fe[3], fe[0]);
                    join(fe[1], fe[2]);
                }
            }
            _ => {}
        }
    }
    core::array::from_fn(|i| root(&mut parent, i))
}

// ── Core algorithm (generic over SDF evaluator) ─────────────────────

/// Run Dual Contouring with a generic SDF evaluator and normal function.
fn dual_contouring_impl(
    sdf_fn: &(dyn Fn(Vec3) -> f32 + Sync),
    normal_fn: &(dyn Fn(Vec3) -> Vec3 + Sync),
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
    topology: DcTopology,
) -> Mesh {
    let res = config.resolution;
    let size = max - min;
    let cell_size = size / res as f32;
    let grid = res + 1; // number of grid vertices per axis

    // 1. Evaluate SDF at all grid vertices (parallel)
    let total = grid * grid * grid;
    let values: Vec<f32> = (0..total)
        .into_par_iter()
        .map(|i| {
            let x = i % grid;
            let y = (i / grid) % grid;
            let z = i / (grid * grid);
            let p = min + Vec3::new(x as f32, y as f32, z as f32) * cell_size;
            sdf_fn(p)
        })
        .collect();

    let idx = |x: usize, y: usize, z: usize| -> usize { z * grid * grid + y * grid + x };
    let pos = |x: usize, y: usize, z: usize| -> Vec3 {
        min + Vec3::new(x as f32, y as f32, z as f32) * cell_size
    };

    // 2. For each cell, find edge crossings and solve QEF (parallel)
    //    cell (cx, cy, cz) has corners at grid vertices (cx..cx+1, cy..cy+1, cz..cz+1).
    //    Only cells with a sign change produce output; `collect` keeps the
    //    cell order whatever the thread count.
    let cell_count = res * res * res;
    let crossed: Vec<(usize, CellVertices)> = (0..cell_count)
        .into_par_iter()
        .filter_map(|ci| {
            let cx = ci % res;
            let cy = (ci / res) % res;
            let cz = ci / (res * res);

            // Gather 8 corner values (corner k is at offset (k & 1, k >> 1 & 1, k >> 2))
            let corners = [
                values[idx(cx, cy, cz)],
                values[idx(cx + 1, cy, cz)],
                values[idx(cx, cy + 1, cz)],
                values[idx(cx + 1, cy + 1, cz)],
                values[idx(cx, cy, cz + 1)],
                values[idx(cx + 1, cy, cz + 1)],
                values[idx(cx, cy + 1, cz + 1)],
                values[idx(cx + 1, cy + 1, cz + 1)],
            ];

            // Check if any edge has a sign change
            // inside = d < 0 (a sample exactly on the surface counts as outside,
            // consistently with the edge tests below)
            let sign0 = corners[0] < 0.0;
            let has_crossing = corners[1..].iter().any(|&v| (v < 0.0) != sign0);
            if !has_crossing {
                return None;
            }

            let mut hermite: Vec<(Vec3, Vec3)> = Vec::with_capacity(12);
            let mut crossed_edges: Vec<usize> = Vec::with_capacity(12);

            for (e, &(c0, c1, off_a, off_b)) in EDGES.iter().enumerate() {
                let da = corners[c0];
                let db = corners[c1];
                if (da < 0.0) == (db < 0.0) {
                    continue; // no sign change
                }

                let pa = pos(cx + off_a[0], cy + off_a[1], cz + off_a[2]);
                let pb = pos(cx + off_b[0], cy + off_b[1], cz + off_b[2]);

                let intersection = bisect_edge(sdf_fn, pa, pb, da, db, config.bisection_iterations);
                let n = normal_fn(intersection);

                hermite.push((intersection, n));
                crossed_edges.push(e);
            }

            if hermite.is_empty() {
                return None;
            }

            let cell_min = pos(cx, cy, cz);
            let cell_max = pos(cx + 1, cy + 1, cz + 1);
            let solve = |points: &[(Vec3, Vec3)]| {
                let vertex_pos = qef_solve(points, cell_min, cell_max, config.clamp_to_cell);
                // Average normal from intersections
                let avg_normal = points
                    .iter()
                    .map(|(_, n)| *n)
                    .sum::<Vec3>()
                    .normalize_or_zero();
                (vertex_pos, avg_normal)
            };

            let mut out = CellVertices {
                vertices: Vec::with_capacity(1),
                edge_vertex: [u8::MAX; 12],
            };
            match topology {
                DcTopology::OneVertexPerCell => {
                    out.vertices.push(solve(&hermite));
                    for &e in &crossed_edges {
                        out.edge_vertex[e] = 0;
                    }
                }
                DcTopology::Manifold => {
                    let sheet = sheet_of_edges(&corners);
                    for &e in &crossed_edges {
                        if out.edge_vertex[e] != u8::MAX {
                            continue;
                        }
                        let k = out.vertices.len() as u8;
                        let points: Vec<(Vec3, Vec3)> = crossed_edges
                            .iter()
                            .zip(&hermite)
                            .filter(|(&f, _)| sheet[f] == sheet[e])
                            .map(|(_, &h)| h)
                            .collect();
                        out.vertices.push(solve(&points));
                        for &f in &crossed_edges {
                            if sheet[f] == sheet[e] {
                                out.edge_vertex[f] = k;
                            }
                        }
                    }
                }
            }
            Some((ci, out))
        })
        .collect();

    // 3. Assign vertex indices: cells in order, a cell's sheets in the order
    //    of their lowest edge
    let mut cell_slot: Vec<u32> = vec![u32::MAX; cell_count];
    let mut first_vertex: Vec<u32> = Vec::with_capacity(crossed.len());
    let mut vertices: Vec<Vertex> = Vec::new();

    for (slot, (ci, cell)) in crossed.iter().enumerate() {
        cell_slot[*ci] = slot as u32;
        first_vertex.push(vertices.len() as u32);
        for &(vpos, vnormal) in &cell.vertices {
            let mut vert = Vertex::new(vpos, vnormal);
            if config.compute_uvs {
                vert.uv = triplanar_uv(vpos, vnormal, config.uv_scale);
            }
            if config.compute_tangents && config.compute_normals && config.compute_uvs {
                vert.tangent = compute_tangent(vnormal);
            }
            vertices.push(vert);
        }
    }

    // The vertex of cell `ci` that owns its local edge `e` (u32::MAX: none)
    let vertex_of = |ci: usize, e: usize| -> u32 {
        let slot = cell_slot[ci];
        if slot == u32::MAX {
            return u32::MAX;
        }
        let k = crossed[slot as usize].1.edge_vertex[e];
        if k == u8::MAX {
            u32::MAX
        } else {
            first_vertex[slot as usize] + u32::from(k)
        }
    };

    // 4. Generate quads → triangles from shared sign-change edges
    //    For each internal edge, the 4 cells sharing that edge form a quad;
    //    each cell contributes the vertex that owns the edge (its local
    //    index in `EDGES` is given next to the cell).
    //    Orientation: the quad's cells are listed counter-clockwise about the
    //    +axis direction, so the winding must be reversed when the *start*
    //    of the edge is outside (outward is −axis). Until 1.12.0 the branches
    //    were the other way round and every DC triangle faced inward (found by
    //    `tests/test_dual_contouring_invariants.rs`). "Inside" is `d < 0`
    //    everywhere: a sample exactly on the surface (a face on a grid plane)
    //    is outside, so no edge lying *in* the surface counts as a crossing —
    //    such in-plane quads have no meaningful orientation.
    let mut quads: Vec<[u32; 4]> = Vec::new();

    let cell_idx = |cx: usize, cy: usize, cz: usize| -> usize { cz * res * res + cy * res + cx };

    let emit_quad = |quads: &mut Vec<[u32; 4]>, q: [u32; 4]| quads.push(q);

    // X-edges: edge along X between grid vertices (x, y, z) and (x+1, y, z)
    // Shared by cells: (x, y-1, z-1), (x, y, z-1), (x, y-1, z), (x, y, z)
    // The loops run x, y, z over 0..res, so an edge's own axis reaches the last
    // grid line (x + 1 = res) and the other two stay inside (1..res): every edge
    // with four cells. The edges of the last layer (x = res - 1 and the like) used
    // to be skipped, and a surface within a cell of the max bound had holes.
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let d0 = values[idx(x, y, z)];

                // X-edge: (x,y,z) → (x+1,y,z), shared by 4 cells if y>0 and z>0
                if y > 0 && z > 0 {
                    let d1 = values[idx(x + 1, y, z)];
                    if (d0 < 0.0) != (d1 < 0.0) {
                        let v0 = vertex_of(cell_idx(x, y - 1, z - 1), 5);
                        let v1 = vertex_of(cell_idx(x, y, z - 1), 4);
                        let v2 = vertex_of(cell_idx(x, y, z), 0);
                        let v3 = vertex_of(cell_idx(x, y - 1, z), 1);

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 < 0.0 {
                                emit_quad(&mut quads, [v0, v1, v2, v3]);
                            } else {
                                emit_quad(&mut quads, [v0, v3, v2, v1]);
                            }
                        }
                    }
                }

                // Y-edge: (x,y,z) → (x,y+1,z), shared by 4 cells if x>0 and z>0
                if x > 0 && z > 0 {
                    let d1 = values[idx(x, y + 1, z)];
                    if (d0 < 0.0) != (d1 < 0.0) {
                        let v0 = vertex_of(cell_idx(x - 1, y, z - 1), 7);
                        let v1 = vertex_of(cell_idx(x, y, z - 1), 6);
                        let v2 = vertex_of(cell_idx(x, y, z), 2);
                        let v3 = vertex_of(cell_idx(x - 1, y, z), 3);

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 < 0.0 {
                                emit_quad(&mut quads, [v0, v3, v2, v1]);
                            } else {
                                emit_quad(&mut quads, [v0, v1, v2, v3]);
                            }
                        }
                    }
                }

                // Z-edge: (x,y,z) → (x,y,z+1), shared by 4 cells if x>0 and y>0
                if x > 0 && y > 0 {
                    let d1 = values[idx(x, y, z + 1)];
                    if (d0 < 0.0) != (d1 < 0.0) {
                        let v0 = vertex_of(cell_idx(x - 1, y - 1, z), 11);
                        let v1 = vertex_of(cell_idx(x, y - 1, z), 10);
                        let v2 = vertex_of(cell_idx(x, y, z), 8);
                        let v3 = vertex_of(cell_idx(x - 1, y, z), 9);

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 < 0.0 {
                                emit_quad(&mut quads, [v0, v1, v2, v3]);
                            } else {
                                emit_quad(&mut quads, [v0, v3, v2, v1]);
                            }
                        }
                    }
                }
            }
        }
    }
    let indices = triangulate_quads(&mut vertices, quads);

    Mesh { vertices, indices }
}

/// Split the dual quads into triangles.
///
/// A dual quad is generally non-planar (saddle cells on the inner ring of a
/// torus): splitting along a fixed diagonal can fold one of the two triangles
/// against the surface. Each quad is split along the diagonal whose two
/// triangles both face the quad's mean vertex normal; the (0, 2) diagonal
/// wins ties. The reference is the mean of the four dual-vertex normals (each
/// sampled on the surface); a field sample at the quad centre is wrong across
/// a CSG crease, where the centre can fall on the other operand.
///
/// Where the surface is tangent to a grid plane (the inner equator of a torus
/// on a grid plane) the cells on both sides of that plane see a sign change
/// and both get a dual vertex, so the two rows of vertices are a fraction of a
/// cell apart and the quads between them are fins: thin, and folded whichever
/// diagonal is chosen. Such a quad (both diagonals fold) is collapsed along
/// its two short edges — the vertex pairs are merged, which is a local edge
/// collapse well inside the cell — and drops out as degenerate. Until 1.14.0
/// these fins were emitted and pointed inward.
fn triangulate_quads(vertices: &mut [Vertex], quads: Vec<[u32; 4]>) -> Vec<u32> {
    let mut remap: Vec<u32> = (0..vertices.len() as u32).collect();
    const fn find(remap: &mut [u32], mut i: u32) -> u32 {
        while remap[i as usize] != i {
            let next = remap[i as usize];
            remap[i as usize] = remap[next as usize];
            i = next;
        }
        i
    }
    let facing = |p: &[Vec3; 4], n_ref: Vec3, a: usize, b: usize, c: usize| {
        (p[b] - p[a]).cross(p[c] - p[a]).dot(n_ref)
    };
    let splits = |p: &[Vec3; 4], n_ref: Vec3| {
        let split_02 = facing(p, n_ref, 0, 1, 2).law_min(facing(p, n_ref, 0, 2, 3));
        let split_13 = facing(p, n_ref, 1, 2, 3).law_min(facing(p, n_ref, 1, 3, 0));
        (split_02, split_13)
    };

    // Pass 1: fins. Both diagonals fold ⇒ merge the shorter pair of opposite
    // edges (endpoints snap to the pair's mean).
    for q in &quads {
        let p = [
            vertices[q[0] as usize].position,
            vertices[q[1] as usize].position,
            vertices[q[2] as usize].position,
            vertices[q[3] as usize].position,
        ];
        let n_ref = vertices[q[0] as usize].normal
            + vertices[q[1] as usize].normal
            + vertices[q[2] as usize].normal
            + vertices[q[3] as usize].normal;
        let (split_02, split_13) = splits(&p, n_ref);
        if split_02 >= 0.0 || split_13 >= 0.0 {
            continue;
        }
        let len_01_23 = (p[1] - p[0]).length() + (p[3] - p[2]).length();
        let len_12_30 = (p[2] - p[1]).length() + (p[0] - p[3]).length();
        let pairs: [(usize, usize); 2] = if len_01_23 < len_12_30 {
            [(0, 1), (2, 3)]
        } else {
            [(1, 2), (3, 0)]
        };
        for (a, b) in pairs {
            let ra = find(&mut remap, q[a]);
            let rb = find(&mut remap, q[b]);
            if ra == rb {
                continue;
            }
            let (keep, drop) = if ra < rb { (ra, rb) } else { (rb, ra) };
            let mid = (vertices[keep as usize].position + vertices[drop as usize].position) * 0.5;
            let n = (vertices[keep as usize].normal + vertices[drop as usize].normal)
                .normalize_or_zero();
            vertices[keep as usize].position = mid;
            vertices[keep as usize].normal = n;
            remap[drop as usize] = keep;
        }
    }

    // Pass 2: triangulate the surviving quads with the remapped indices.
    let mut indices: Vec<u32> = Vec::with_capacity(quads.len() * 6);
    for q in &quads {
        let q = [
            find(&mut remap, q[0]),
            find(&mut remap, q[1]),
            find(&mut remap, q[2]),
            find(&mut remap, q[3]),
        ];
        let p = [
            vertices[q[0] as usize].position,
            vertices[q[1] as usize].position,
            vertices[q[2] as usize].position,
            vertices[q[3] as usize].position,
        ];
        let n_ref = vertices[q[0] as usize].normal
            + vertices[q[1] as usize].normal
            + vertices[q[2] as usize].normal
            + vertices[q[3] as usize].normal;
        let (split_02, split_13) = splits(&p, n_ref);
        let tris: [[u32; 3]; 2] = if split_13 > split_02 {
            [[q[1], q[2], q[3]], [q[1], q[3], q[0]]]
        } else {
            [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
        };
        for t in tris {
            if t[0] != t[1] && t[1] != t[2] && t[2] != t[0] {
                indices.extend_from_slice(&t);
            }
        }
    }
    indices
}

// ── Public API ──────────────────────────────────────────────────────

/// Generate a mesh from an SDF using Dual Contouring
///
/// Dual Contouring places one vertex per cell at the QEF-optimal position,
/// producing meshes that preserve sharp edges and corners better than
/// Marching Cubes. Works best with SDFs that have well-defined gradients.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `min` - Minimum corner of bounding box
/// * `max` - Maximum corner of bounding box
/// * `config` - Dual Contouring configuration
///
/// # Returns
/// Generated mesh with quads triangulated
pub fn dual_contouring(
    node: &SdfNode,
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
) -> Mesh {
    let eps = config.gradient_epsilon;
    dual_contouring_impl(
        &|p| eval(node, p),
        &|p| {
            let n = normal(node, p, eps);
            if n.length_squared() < 1e-10 {
                gradient_cd(|q| eval(node, q), p, eps)
            } else {
                n
            }
        },
        min,
        max,
        config,
        DcTopology::OneVertexPerCell,
    )
}

/// Generate a mesh from a compiled SDF using Dual Contouring
///
/// Uses compiled SDF evaluation for 2-5x speedup over interpreted version.
pub fn dual_contouring_compiled(
    sdf: &CompiledSdf,
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
) -> Mesh {
    let eps = config.gradient_epsilon;
    dual_contouring_impl(
        &|p| eval_compiled(sdf, p),
        &|p| gradient_cd(|q| eval_compiled(sdf, q), p, eps),
        min,
        max,
        config,
        DcTopology::OneVertexPerCell,
    )
}

/// Generate a mesh from an SDF using Dual Contouring with a chosen cell
/// topology
///
/// `DcTopology::OneVertexPerCell` is exactly [`dual_contouring`];
/// `DcTopology::Manifold` gives one vertex per surface sheet in a cell.
pub fn dual_contouring_with_topology(
    node: &SdfNode,
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
    topology: DcTopology,
) -> Mesh {
    let eps = config.gradient_epsilon;
    dual_contouring_impl(
        &|p| eval(node, p),
        &|p| {
            let n = normal(node, p, eps);
            if n.length_squared() < 1e-10 {
                gradient_cd(|q| eval(node, q), p, eps)
            } else {
                n
            }
        },
        min,
        max,
        config,
        topology,
    )
}

/// Generate a mesh of the zero set of any scalar field using Dual Contouring
///
/// For fields that are not an `SdfNode`: a distance to a mesh, a sampled
/// volume, or an offset surface `|p| f(p) - h`. Inside is `field < 0`.
/// Normals are central differences of the field with step
/// `config.gradient_epsilon`.
pub fn dual_contouring_field(
    field: &(dyn Fn(Vec3) -> f32 + Sync),
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
    topology: DcTopology,
) -> Mesh {
    let eps = config.gradient_epsilon;
    dual_contouring_impl(
        field,
        &|p| gradient_cd(field, p, eps),
        min,
        max,
        config,
        topology,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_dc_sphere_produces_mesh() {
        let sphere = SdfNode::sphere(1.0);
        let config = DualContouringConfig {
            resolution: 16,
            ..Default::default()
        };
        let mesh = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 0, "Should produce vertices");
        assert!(mesh.triangle_count() > 0, "Should produce triangles");
        assert_eq!(mesh.indices.len() % 3, 0, "Indices should be multiple of 3");
    }

    #[test]
    fn test_dc_box_produces_mesh() {
        let box3 = SdfNode::box3d(1.0, 1.0, 1.0);
        let config = DualContouringConfig {
            resolution: 16,
            ..Default::default()
        };
        let mesh = dual_contouring(&box3, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_dc_smooth_union() {
        let shape = SdfNode::sphere(0.8).smooth_union(SdfNode::box3d(0.6, 0.6, 0.6), 0.2);
        let config = DualContouringConfig {
            resolution: 16,
            ..Default::default()
        };
        let mesh = dual_contouring(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 10);
        assert!(mesh.triangle_count() > 10);
    }

    #[test]
    fn test_dc_vertices_near_surface() {
        let sphere = SdfNode::sphere(1.0);
        let config = DualContouringConfig {
            resolution: 24,
            ..Default::default()
        };
        let mesh = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // DC vertices should be near the iso-surface
        let max_dist = mesh
            .vertices
            .iter()
            .map(|v| (v.position.length() - 1.0).abs())
            .fold(0.0f32, f32::max);

        // At resolution 24, cell size = 4/24 ≈ 0.167, so vertices should be within ~cell diagonal
        assert!(
            max_dist < 0.5,
            "Vertices should be near surface, max_dist={}",
            max_dist
        );
    }

    #[test]
    fn test_dc_normals_outward() {
        let sphere = SdfNode::sphere(1.0);
        let config = DualContouringConfig {
            resolution: 16,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // All normals should point outward (dot with position > 0)
        let outward_count = mesh
            .vertices
            .iter()
            .filter(|v| v.normal.dot(v.position.normalize_or_zero()) > 0.0)
            .count();

        let ratio = outward_count as f32 / mesh.vertex_count() as f32;
        assert!(
            ratio > 0.9,
            "Most normals should point outward, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_dc_compiled_matches() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let config = DualContouringConfig {
            resolution: 12,
            ..Default::default()
        };

        let mesh_interp = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        let mesh_compiled =
            dual_contouring_compiled(&compiled, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Both should produce similar vertex counts (exact match expected for same resolution)
        assert_eq!(
            mesh_interp.vertex_count(),
            mesh_compiled.vertex_count(),
            "Compiled should match interpreted vertex count"
        );
        assert_eq!(
            mesh_interp.triangle_count(),
            mesh_compiled.triangle_count(),
            "Compiled should match interpreted triangle count"
        );
    }

    #[test]
    fn test_dc_uvs_computed() {
        let sphere = SdfNode::sphere(1.0);
        let config = DualContouringConfig {
            resolution: 12,
            compute_uvs: true,
            ..Default::default()
        };
        let mesh = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // At least some UVs should be non-zero
        let has_uv = mesh.vertices.iter().any(|v| v.uv != Vec2::ZERO);
        assert!(has_uv, "Should have computed UVs");
    }

    #[test]
    fn test_dc_aaa_preset() {
        let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let config = DualContouringConfig::aaa(16);
        let mesh = dual_contouring(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);

        // Check tangent is set
        let has_tangent = mesh.vertices.iter().any(|v| v.tangent != Vec4::ZERO);
        assert!(has_tangent, "AAA preset should compute tangents");
    }

    #[test]
    fn test_dc_higher_resolution_more_detail() {
        let sphere = SdfNode::sphere(1.0);

        let config_lo = DualContouringConfig {
            resolution: 8,
            ..Default::default()
        };
        let config_hi = DualContouringConfig {
            resolution: 24,
            ..Default::default()
        };

        let mesh_lo = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config_lo);
        let mesh_hi = dual_contouring(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config_hi);

        assert!(
            mesh_hi.vertex_count() > mesh_lo.vertex_count(),
            "Higher resolution should produce more vertices: hi={} lo={}",
            mesh_hi.vertex_count(),
            mesh_lo.vertex_count()
        );
    }

    #[test]
    fn test_qef_solve_planar() {
        // Three planes meeting at origin — QEF should solve to near origin
        let intersections = vec![
            (Vec3::new(0.1, 0.0, 0.0), Vec3::X),
            (Vec3::new(0.0, 0.1, 0.0), Vec3::Y),
            (Vec3::new(0.0, 0.0, 0.1), Vec3::Z),
        ];

        let cell_min = Vec3::splat(-1.0);
        let cell_max = Vec3::splat(1.0);

        let result = qef_solve(&intersections, cell_min, cell_max, true);
        assert!(
            result.length() < 0.2,
            "QEF should find near-origin: {:?}",
            result
        );
    }
}

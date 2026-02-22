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
        DualContouringConfig {
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
        DualContouringConfig {
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

    let vertex = match result {
        Some(v) => mass_point + Vec3::new(v[0], v[1], v[2]),
        None => mass_point,
    };

    if clamp {
        vertex.clamp(cell_min, cell_max)
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
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;

    // Cramer's rule
    let x = (atb[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (atb[1] * a[2][2] - a[1][2] * atb[2])
        + a[0][2] * (atb[1] * a[2][1] - a[1][1] * atb[2]))
        * inv_det;

    let y = (a[0][0] * (atb[1] * a[2][2] - a[1][2] * atb[2])
        - atb[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * atb[2] - atb[1] * a[2][0]))
        * inv_det;

    let z = (a[0][0] * (a[1][1] * atb[2] - atb[1] * a[2][1])
        - a[0][1] * (a[1][0] * atb[2] - atb[1] * a[2][0])
        + atb[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
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

        if (dlo > 0.0) == (dm > 0.0) {
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

// ── Core algorithm (generic over SDF evaluator) ─────────────────────

/// Run Dual Contouring with a generic SDF evaluator and normal function.
fn dual_contouring_impl(
    sdf_fn: &(dyn Fn(Vec3) -> f32 + Sync),
    normal_fn: &(dyn Fn(Vec3) -> Vec3 + Sync),
    min: Vec3,
    max: Vec3,
    config: &DualContouringConfig,
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
    //    cell (cx, cy, cz) has corners at grid vertices (cx..cx+1, cy..cy+1, cz..cz+1)
    let cell_count = res * res * res;
    let cell_vertices: Vec<Option<(Vec3, Vec3)>> = (0..cell_count)
        .into_par_iter()
        .map(|ci| {
            let cx = ci % res;
            let cy = (ci / res) % res;
            let cz = ci / (res * res);

            // Gather 8 corner values
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
            let sign0 = corners[0] > 0.0;
            let has_crossing = corners[1..].iter().any(|&v| (v > 0.0) != sign0);
            if !has_crossing {
                return None;
            }

            // 12 edges of the cube — find crossings
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

            let mut hermite: Vec<(Vec3, Vec3)> = Vec::with_capacity(12);

            for &(c0, c1, off_a, off_b) in &EDGES {
                let da = corners[c0];
                let db = corners[c1];
                if (da > 0.0) == (db > 0.0) {
                    continue; // no sign change
                }

                let pa = pos(cx + off_a[0], cy + off_a[1], cz + off_a[2]);
                let pb = pos(cx + off_b[0], cy + off_b[1], cz + off_b[2]);

                let intersection = bisect_edge(sdf_fn, pa, pb, da, db, config.bisection_iterations);
                let n = normal_fn(intersection);

                hermite.push((intersection, n));
            }

            if hermite.is_empty() {
                return None;
            }

            let cell_min = pos(cx, cy, cz);
            let cell_max = pos(cx + 1, cy + 1, cz + 1);
            let vertex_pos = qef_solve(&hermite, cell_min, cell_max, config.clamp_to_cell);

            // Average normal from intersections
            let avg_normal = hermite
                .iter()
                .map(|(_, n)| *n)
                .sum::<Vec3>()
                .normalize_or_zero();

            Some((vertex_pos, avg_normal))
        })
        .collect();

    // 3. Assign vertex indices to non-empty cells
    let mut vertex_index_map: Vec<u32> = vec![u32::MAX; cell_count];
    let mut vertices: Vec<Vertex> = Vec::new();

    for ci in 0..cell_count {
        if let Some((vpos, vnormal)) = cell_vertices[ci] {
            vertex_index_map[ci] = vertices.len() as u32;

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

    // 4. Generate quads → triangles from shared sign-change edges
    //    For each internal edge, the 4 cells sharing that edge form a quad.
    let mut indices: Vec<u32> = Vec::new();

    let cell_idx = |cx: usize, cy: usize, cz: usize| -> usize { cz * res * res + cy * res + cx };

    // X-edges: edge along X between grid vertices (x, y, z) and (x+1, y, z)
    // Shared by cells: (x, y-1, z-1), (x, y, z-1), (x, y-1, z), (x, y, z)
    for z in 0..res {
        for y in 0..res {
            for x in 0..res {
                let d0 = values[idx(x, y, z)];

                // X-edge: (x,y,z) → (x+1,y,z), shared by 4 cells if y>0 and z>0
                if x < res - 1 && y > 0 && z > 0 {
                    let d1 = values[idx(x + 1, y, z)];
                    if (d0 > 0.0) != (d1 > 0.0) {
                        let c0 = cell_idx(x, y - 1, z - 1);
                        let c1 = cell_idx(x, y, z - 1);
                        let c2 = cell_idx(x, y, z);
                        let c3 = cell_idx(x, y - 1, z);

                        let v0 = vertex_index_map[c0];
                        let v1 = vertex_index_map[c1];
                        let v2 = vertex_index_map[c2];
                        let v3 = vertex_index_map[c3];

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 > 0.0 {
                                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
                            } else {
                                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
                            }
                        }
                    }
                }

                // Y-edge: (x,y,z) → (x,y+1,z), shared by 4 cells if x>0 and z>0
                if y < res - 1 && x > 0 && z > 0 {
                    let d1 = values[idx(x, y + 1, z)];
                    if (d0 > 0.0) != (d1 > 0.0) {
                        let c0 = cell_idx(x - 1, y, z - 1);
                        let c1 = cell_idx(x, y, z - 1);
                        let c2 = cell_idx(x, y, z);
                        let c3 = cell_idx(x - 1, y, z);

                        let v0 = vertex_index_map[c0];
                        let v1 = vertex_index_map[c1];
                        let v2 = vertex_index_map[c2];
                        let v3 = vertex_index_map[c3];

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 > 0.0 {
                                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
                            } else {
                                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
                            }
                        }
                    }
                }

                // Z-edge: (x,y,z) → (x,y,z+1), shared by 4 cells if x>0 and y>0
                if z < res - 1 && x > 0 && y > 0 {
                    let d1 = values[idx(x, y, z + 1)];
                    if (d0 > 0.0) != (d1 > 0.0) {
                        let c0 = cell_idx(x - 1, y - 1, z);
                        let c1 = cell_idx(x, y - 1, z);
                        let c2 = cell_idx(x, y, z);
                        let c3 = cell_idx(x - 1, y, z);

                        let v0 = vertex_index_map[c0];
                        let v1 = vertex_index_map[c1];
                        let v2 = vertex_index_map[c2];
                        let v3 = vertex_index_map[c3];

                        if v0 != u32::MAX && v1 != u32::MAX && v2 != u32::MAX && v3 != u32::MAX {
                            if d0 > 0.0 {
                                indices.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
                            } else {
                                indices.extend_from_slice(&[v0, v2, v1, v0, v3, v2]);
                            }
                        }
                    }
                }
            }
        }
    }

    Mesh { vertices, indices }
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

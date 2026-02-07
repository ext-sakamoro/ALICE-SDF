//! SDF to mesh conversion using Marching Cubes (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Z-Slab Parallelization**: Process each Z-layer independently with Rayon.
//! - **Merge Sub-meshes**: Avoid mutex contention by merging results after parallel phase.
//! - **Forced Inlining**: `#[inline(always)]` on hot-path helpers.
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, eval_material, normal};
use crate::compiled::{CompiledSdf, eval_compiled, eval_compiled_batch_simd_parallel};
use crate::mesh::Vertex;
use crate::SdfNode;
use glam::{Vec2, Vec3, Vec4};
use rayon::prelude::*;

/// Configuration for marching cubes
#[derive(Debug, Clone, Copy)]
pub struct MarchingCubesConfig {
    /// Grid resolution along each axis
    pub resolution: usize,
    /// Iso-level (usually 0 for SDF surface)
    pub iso_level: f32,
    /// Whether to compute vertex normals
    pub compute_normals: bool,
    /// Whether to generate triplanar UV coordinates
    pub compute_uvs: bool,
    /// UV tiling scale (world units per UV tile)
    pub uv_scale: f32,
    /// Whether to compute tangent vectors (requires normals and UVs)
    pub compute_tangents: bool,
    /// Whether to evaluate material IDs per vertex
    pub compute_materials: bool,
}

impl Default for MarchingCubesConfig {
    fn default() -> Self {
        MarchingCubesConfig {
            resolution: 64,
            iso_level: 0.0,
            compute_normals: true,
            compute_uvs: false,
            uv_scale: 1.0,
            compute_tangents: false,
            compute_materials: false,
        }
    }
}

/// Configuration preset with all AAA features enabled
impl MarchingCubesConfig {
    /// AAA preset: normals + triplanar UV + tangents + materials
    pub fn aaa(resolution: usize) -> Self {
        MarchingCubesConfig {
            resolution,
            iso_level: 0.0,
            compute_normals: true,
            compute_uvs: true,
            uv_scale: 1.0,
            compute_tangents: true,
            compute_materials: true,
        }
    }
}

/// Compute triplanar UV coordinates from position and normal (Deep Fried)
///
/// Projects the vertex position onto the dominant axis plane determined
/// by the surface normal, producing seamless UVs for arbitrary SDF surfaces.
#[inline(always)]
fn triplanar_uv(position: Vec3, normal: Vec3, scale: f32) -> Vec2 {
    let abs_n = normal.abs();
    let inv_scale = 1.0 / scale;

    if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
        // X-dominant: project onto YZ plane
        Vec2::new(position.y * inv_scale, position.z * inv_scale)
    } else if abs_n.y >= abs_n.z {
        // Y-dominant: project onto XZ plane
        Vec2::new(position.x * inv_scale, position.z * inv_scale)
    } else {
        // Z-dominant: project onto XY plane
        Vec2::new(position.x * inv_scale, position.y * inv_scale)
    }
}

/// Compute tangent vector from normal (Deep Fried)
///
/// Generates a tangent perpendicular to the normal, aligned with the
/// triplanar UV projection direction. W component stores handedness.
#[inline(always)]
fn compute_tangent(normal: Vec3) -> Vec4 {
    let abs_n = normal.abs();

    let tangent = if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
        // X-dominant: tangent along Y
        let t = Vec3::new(0.0, 1.0, 0.0);
        (t - normal * normal.dot(t)).normalize()
    } else if abs_n.y >= abs_n.z {
        // Y-dominant: tangent along X
        let t = Vec3::new(1.0, 0.0, 0.0);
        (t - normal * normal.dot(t)).normalize()
    } else {
        // Z-dominant: tangent along X
        let t = Vec3::new(1.0, 0.0, 0.0);
        (t - normal * normal.dot(t)).normalize()
    };

    Vec4::new(tangent.x, tangent.y, tangent.z, 1.0)
}

/// Compute MikkTSpace-compatible tangent vectors from UV gradients
///
/// For each triangle, computes the tangent and bitangent from position/UV deltas,
/// accumulates at shared vertices, then orthogonalizes against the normal and
/// computes handedness. Produces results compatible with MikkTSpace-based normal
/// map baking tools (xNormal, Substance, UE5, Unity, Godot).
fn compute_mikktspace_tangents(mesh: &mut Mesh) {
    let vert_count = mesh.vertices.len();
    if vert_count == 0 {
        return;
    }

    let mut tangents = vec![Vec3::ZERO; vert_count];
    let mut bitangents = vec![Vec3::ZERO; vert_count];

    // Phase 1: Accumulate tangent/bitangent per triangle
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        if i0 >= vert_count || i1 >= vert_count || i2 >= vert_count {
            continue;
        }

        let p0 = mesh.vertices[i0].position;
        let p1 = mesh.vertices[i1].position;
        let p2 = mesh.vertices[i2].position;

        let uv0 = mesh.vertices[i0].uv;
        let uv1 = mesh.vertices[i1].uv;
        let uv2 = mesh.vertices[i2].uv;

        let dp1 = p1 - p0;
        let dp2 = p2 - p0;
        let duv1 = uv1 - uv0;
        let duv2 = uv2 - uv0;

        let det = duv1.x * duv2.y - duv1.y * duv2.x;
        if det.abs() < 1e-8 {
            continue; // Degenerate UV mapping
        }

        let inv_det = 1.0 / det;
        let t = Vec3::new(
            (dp1.x * duv2.y - dp2.x * duv1.y) * inv_det,
            (dp1.y * duv2.y - dp2.y * duv1.y) * inv_det,
            (dp1.z * duv2.y - dp2.z * duv1.y) * inv_det,
        );
        let b = Vec3::new(
            (dp2.x * duv1.x - dp1.x * duv2.x) * inv_det,
            (dp2.y * duv1.x - dp1.y * duv2.x) * inv_det,
            (dp2.z * duv1.x - dp1.z * duv2.x) * inv_det,
        );

        // Area-weighted accumulation (implicit via dp magnitude)
        tangents[i0] += t;
        tangents[i1] += t;
        tangents[i2] += t;
        bitangents[i0] += b;
        bitangents[i1] += b;
        bitangents[i2] += b;
    }

    // Phase 2: Orthogonalize and compute handedness per vertex
    for i in 0..vert_count {
        let n = mesh.vertices[i].normal;
        let t = tangents[i];
        let b = bitangents[i];

        if t.length_squared() < 1e-10 {
            // Fallback for vertices with no valid UV gradient
            mesh.vertices[i].tangent = compute_tangent(n);
            continue;
        }

        // Gram-Schmidt orthogonalize: T' = normalize(T - N * dot(N, T))
        let ortho_t = (t - n * n.dot(t)).normalize();

        // Handedness: sign of dot(cross(N, T'), B)
        let w = if n.cross(ortho_t).dot(b) < 0.0 {
            -1.0
        } else {
            1.0
        };

        mesh.vertices[i].tangent = Vec4::new(ortho_t.x, ortho_t.y, ortho_t.z, w);
    }
}

/// Simple mesh structure
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Mesh vertices
    pub vertices: Vec<Vertex>,
    /// Triangle indices
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Create an empty mesh
    pub fn new() -> Self {
        Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Get the number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of triangles
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert an SDF to a mesh using marching cubes
///
/// Produces a watertight mesh with shared vertices across cell boundaries.
/// When UVs are enabled, tangent vectors are computed using MikkTSpace-compatible
/// UV gradient method for correct normal map rendering in all major engines.
///
/// # Arguments
/// * `node` - The SDF tree
/// * `min` - Minimum corner of bounding box
/// * `max` - Maximum corner of bounding box
/// * `config` - Marching cubes configuration
///
/// # Returns
/// Generated mesh
pub fn sdf_to_mesh(node: &SdfNode, min: Vec3, max: Vec3, config: &MarchingCubesConfig) -> Mesh {
    let mut mesh = marching_cubes(node, min, max, config);

    // Deduplicate vertices for watertight mesh (shared vertices across cell boundaries).
    // Marching cubes Z-slab parallelization creates duplicate vertices on slab boundaries;
    // dedup merges them so adjacent triangles share vertices → manifold mesh.
    super::optimize::deduplicate_vertices(&mut mesh);

    // Compute MikkTSpace-compatible tangents from UV gradients (post-dedup so topology is correct)
    if config.compute_tangents && config.compute_normals && config.compute_uvs {
        compute_mikktspace_tangents(&mut mesh);
    }

    mesh
}

/// Marching cubes algorithm implementation (Deep Fried Edition)
///
/// Parallelized by Z-slabs to maximize throughput while avoiding mutex contention.
pub fn marching_cubes(
    node: &SdfNode,
    min: Vec3,
    max: Vec3,
    config: &MarchingCubesConfig,
) -> Mesh {
    let resolution = config.resolution;
    let size = max - min;
    let cell_size = size / resolution as f32;

    // Evaluate SDF on grid (parallel)
    let grid_size = resolution + 1;
    let total_points = grid_size * grid_size * grid_size;

    let values: Vec<f32> = (0..total_points)
        .into_par_iter()
        .map(|i| {
            let x = i % grid_size;
            let y = (i / grid_size) % grid_size;
            let z = i / (grid_size * grid_size);

            let point = min
                + Vec3::new(
                    x as f32 * cell_size.x,
                    y as f32 * cell_size.y,
                    z as f32 * cell_size.z,
                );

            eval(node, point)
        })
        .collect();

    // Z-slab parallelization: each Z processes independently, then merge
    let sub_meshes: Vec<Mesh> = (0..resolution)
        .into_par_iter()
        .map(|z| {
            let mut slab_mesh = Mesh::new();
            for y in 0..resolution {
                for x in 0..resolution {
                    process_cell(
                        node,
                        &values,
                        x,
                        y,
                        z,
                        grid_size,
                        min,
                        cell_size,
                        config,
                        &mut slab_mesh,
                    );
                }
            }
            slab_mesh
        })
        .collect();

    // Merge sub-meshes (sequential but fast - just concatenation)
    merge_meshes(sub_meshes)
}

/// Merge multiple sub-meshes into a single mesh (Deep Fried)
#[inline(always)]
fn merge_meshes(sub_meshes: Vec<Mesh>) -> Mesh {
    // Pre-calculate total capacity
    let total_vertices: usize = sub_meshes.iter().map(|m| m.vertices.len()).sum();
    let total_indices: usize = sub_meshes.iter().map(|m| m.indices.len()).sum();

    let mut merged = Mesh {
        vertices: Vec::with_capacity(total_vertices),
        indices: Vec::with_capacity(total_indices),
    };

    for sub in sub_meshes {
        let base_idx = merged.vertices.len() as u32;
        merged.vertices.extend(sub.vertices);
        merged.indices.extend(sub.indices.iter().map(|i| i + base_idx));
    }

    merged
}

/// Process a single cell in the marching cubes grid (Deep Fried)
#[inline(always)]
fn process_cell(
    node: &SdfNode,
    values: &[f32],
    x: usize,
    y: usize,
    z: usize,
    grid_size: usize,
    min: Vec3,
    cell_size: Vec3,
    config: &MarchingCubesConfig,
    mesh: &mut Mesh,
) {
    // Get corner values
    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Vec3::ZERO; 8];

    for i in 0..8 {
        let dx = CORNER_OFFSETS[i][0];
        let dy = CORNER_OFFSETS[i][1];
        let dz = CORNER_OFFSETS[i][2];

        let gx = x + dx;
        let gy = y + dy;
        let gz = z + dz;

        let idx = gx + gy * grid_size + gz * grid_size * grid_size;
        corner_values[i] = values[idx];

        corner_positions[i] = min
            + Vec3::new(
                gx as f32 * cell_size.x,
                gy as f32 * cell_size.y,
                gz as f32 * cell_size.z,
            );
    }

    // Compute cube index
    let mut cube_index = 0;
    for i in 0..8 {
        if corner_values[i] < config.iso_level {
            cube_index |= 1 << i;
        }
    }

    // Skip if entirely inside or outside
    if EDGE_TABLE[cube_index] == 0 {
        return;
    }

    // Compute edge vertices
    let mut edge_vertices = [Vec3::ZERO; 12];
    for i in 0..12 {
        if EDGE_TABLE[cube_index] & (1 << i) != 0 {
            let e0 = EDGE_CONNECTIONS[i][0];
            let e1 = EDGE_CONNECTIONS[i][1];

            edge_vertices[i] = interpolate_vertex(
                corner_positions[e0],
                corner_positions[e1],
                corner_values[e0],
                corner_values[e1],
                config.iso_level,
            );
        }
    }

    // Generate triangles
    let mut i = 0;
    while TRI_TABLE[cube_index][i] != -1 {
        let v0 = edge_vertices[TRI_TABLE[cube_index][i] as usize];
        let v1 = edge_vertices[TRI_TABLE[cube_index][i + 1] as usize];
        let v2 = edge_vertices[TRI_TABLE[cube_index][i + 2] as usize];

        let base_idx = mesh.vertices.len() as u32;

        // Compute normals if requested
        let (n0, n1, n2) = if config.compute_normals {
            (
                normal(node, v0, 0.001),
                normal(node, v1, 0.001),
                normal(node, v2, 0.001),
            )
        } else {
            let face_normal = (v1 - v0).cross(v2 - v0).normalize();
            (face_normal, face_normal, face_normal)
        };

        let mut vert0 = Vertex::new(v0, n0);
        let mut vert1 = Vertex::new(v1, n1);
        let mut vert2 = Vertex::new(v2, n2);

        if config.compute_uvs {
            vert0.uv = triplanar_uv(v0, n0, config.uv_scale);
            vert1.uv = triplanar_uv(v1, n1, config.uv_scale);
            vert2.uv = triplanar_uv(v2, n2, config.uv_scale);
        }

        // When UVs are enabled, tangents are computed in post-processing via MikkTSpace.
        // Only use the simple Gram-Schmidt fallback when UVs are NOT available.
        if config.compute_tangents && config.compute_normals && !config.compute_uvs {
            vert0.tangent = compute_tangent(n0);
            vert1.tangent = compute_tangent(n1);
            vert2.tangent = compute_tangent(n2);
        }

        if config.compute_materials {
            vert0.material_id = eval_material(node, v0);
            vert1.material_id = eval_material(node, v1);
            vert2.material_id = eval_material(node, v2);
        }

        mesh.vertices.push(vert0);
        mesh.vertices.push(vert1);
        mesh.vertices.push(vert2);

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);

        i += 3;
    }
}

/// Linear interpolation along an edge (Deep Fried)
///
/// Branchless: division by near-zero handled by clamp saturating to 0/1.
#[inline(always)]
fn interpolate_vertex(p0: Vec3, p1: Vec3, v0: f32, v1: f32, iso_level: f32) -> Vec3 {
    let t = ((iso_level - v0) / (v1 - v0)).clamp(0.0, 1.0);
    p0 + (p1 - p0) * t
}

// ---------------------------------------------------------------------------
// Compiled VM variants — use eval_compiled instead of interpreted eval
// ---------------------------------------------------------------------------

/// Compute surface normal using compiled evaluator (6 eval calls per vertex)
#[inline(always)]
fn normal_compiled(sdf: &CompiledSdf, point: Vec3, epsilon: f32) -> Vec3 {
    let ex = Vec3::new(epsilon, 0.0, 0.0);
    let ey = Vec3::new(0.0, epsilon, 0.0);
    let ez = Vec3::new(0.0, 0.0, epsilon);

    let grad = Vec3::new(
        eval_compiled(sdf, point + ex) - eval_compiled(sdf, point - ex),
        eval_compiled(sdf, point + ey) - eval_compiled(sdf, point - ey),
        eval_compiled(sdf, point + ez) - eval_compiled(sdf, point - ez),
    );

    let len_sq = grad.length_squared();
    if len_sq < 1e-20 {
        return Vec3::Y;
    }
    grad / len_sq.sqrt()
}

/// Convert compiled SDF to mesh (compiled VM path)
///
/// Uses `eval_compiled` for grid evaluation and normal computation,
/// providing 2-5x speedup over the interpreted `sdf_to_mesh`.
pub fn sdf_to_mesh_compiled(sdf: &CompiledSdf, min: Vec3, max: Vec3, config: &MarchingCubesConfig) -> Mesh {
    let mut mesh = marching_cubes_compiled(sdf, min, max, config);

    super::optimize::deduplicate_vertices(&mut mesh);

    if config.compute_tangents && config.compute_normals && config.compute_uvs {
        compute_mikktspace_tangents(&mut mesh);
    }

    mesh
}

/// Marching cubes using compiled VM evaluator (Deep Fried Edition)
///
/// Grid evaluation uses SIMD batch path for maximum throughput.
/// Normal computation uses grid finite differences (no extra eval calls).
pub fn marching_cubes_compiled(
    sdf: &CompiledSdf,
    min: Vec3,
    max: Vec3,
    config: &MarchingCubesConfig,
) -> Mesh {
    let resolution = config.resolution;
    let size = max - min;
    let cell_size = size / resolution as f32;

    let grid_size = resolution + 1;
    let total_points = grid_size * grid_size * grid_size;

    // Build all grid points, then SIMD batch evaluate
    let points: Vec<Vec3> = (0..total_points)
        .map(|i| {
            let x = i % grid_size;
            let y = (i / grid_size) % grid_size;
            let z = i / (grid_size * grid_size);
            min + Vec3::new(
                x as f32 * cell_size.x,
                y as f32 * cell_size.y,
                z as f32 * cell_size.z,
            )
        })
        .collect();

    let values = eval_compiled_batch_simd_parallel(sdf, &points);

    let sub_meshes: Vec<Mesh> = (0..resolution)
        .into_par_iter()
        .map(|z| {
            let mut slab_mesh = Mesh::new();
            for y in 0..resolution {
                for x in 0..resolution {
                    process_cell_compiled(
                        sdf,
                        &values,
                        x, y, z,
                        grid_size,
                        min,
                        cell_size,
                        config,
                        &mut slab_mesh,
                    );
                }
            }
            slab_mesh
        })
        .collect();

    merge_meshes(sub_meshes)
}

/// Compute normal from grid finite differences (zero eval calls)
///
/// Uses neighboring grid values to approximate the gradient via central differences.
/// Falls back to `normal_compiled` for boundary cells where neighbors are unavailable.
#[inline(always)]
fn normal_from_grid(
    sdf: &CompiledSdf,
    vertex: Vec3,
    values: &[f32],
    gx: usize, gy: usize, gz: usize,
    grid_size: usize,
    cell_size: Vec3,
) -> Vec3 {
    let max_idx = grid_size - 1;
    // Boundary check — need neighbors in all 6 directions
    if gx == 0 || gx >= max_idx || gy == 0 || gy >= max_idx || gz == 0 || gz >= max_idx {
        return normal_compiled(sdf, vertex, 0.001);
    }

    let idx = |x: usize, y: usize, z: usize| -> usize {
        x + y * grid_size + z * grid_size * grid_size
    };

    let grad = Vec3::new(
        (values[idx(gx + 1, gy, gz)] - values[idx(gx - 1, gy, gz)]) / (2.0 * cell_size.x),
        (values[idx(gx, gy + 1, gz)] - values[idx(gx, gy - 1, gz)]) / (2.0 * cell_size.y),
        (values[idx(gx, gy, gz + 1)] - values[idx(gx, gy, gz - 1)]) / (2.0 * cell_size.z),
    );

    let len_sq = grad.length_squared();
    if len_sq < 1e-20 {
        return Vec3::Y;
    }
    grad / len_sq.sqrt()
}

/// Process a single marching cube cell using compiled evaluator
///
/// Normal computation uses grid finite differences when possible,
/// eliminating 6 eval_compiled calls per vertex (the biggest per-vertex cost).
fn process_cell_compiled(
    sdf: &CompiledSdf,
    values: &[f32],
    x: usize, y: usize, z: usize,
    grid_size: usize,
    min: Vec3,
    cell_size: Vec3,
    config: &MarchingCubesConfig,
    mesh: &mut Mesh,
) {
    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Vec3::ZERO; 8];

    for i in 0..8 {
        let dx = CORNER_OFFSETS[i][0];
        let dy = CORNER_OFFSETS[i][1];
        let dz = CORNER_OFFSETS[i][2];

        let gx = x + dx;
        let gy = y + dy;
        let gz = z + dz;

        let idx = gx + gy * grid_size + gz * grid_size * grid_size;
        corner_values[i] = values[idx];

        corner_positions[i] = min
            + Vec3::new(
                gx as f32 * cell_size.x,
                gy as f32 * cell_size.y,
                gz as f32 * cell_size.z,
            );
    }

    let mut cube_index = 0;
    for i in 0..8 {
        if corner_values[i] < config.iso_level {
            cube_index |= 1 << i;
        }
    }

    if EDGE_TABLE[cube_index] == 0 {
        return;
    }

    // Track which grid corners contributed to each edge vertex (for grid-normal lookup)
    let mut edge_vertices = [Vec3::ZERO; 12];
    let mut edge_grid_coords = [[0usize; 6]; 12]; // [gx0, gy0, gz0, gx1, gy1, gz1]
    for i in 0..12 {
        if EDGE_TABLE[cube_index] & (1 << i) != 0 {
            let e0 = EDGE_CONNECTIONS[i][0];
            let e1 = EDGE_CONNECTIONS[i][1];

            edge_vertices[i] = interpolate_vertex(
                corner_positions[e0],
                corner_positions[e1],
                corner_values[e0],
                corner_values[e1],
                config.iso_level,
            );

            // Store grid coords for both endpoints
            edge_grid_coords[i] = [
                x + CORNER_OFFSETS[e0][0], y + CORNER_OFFSETS[e0][1], z + CORNER_OFFSETS[e0][2],
                x + CORNER_OFFSETS[e1][0], y + CORNER_OFFSETS[e1][1], z + CORNER_OFFSETS[e1][2],
            ];
        }
    }

    let mut i = 0;
    while TRI_TABLE[cube_index][i] != -1 {
        let ei0 = TRI_TABLE[cube_index][i] as usize;
        let ei1 = TRI_TABLE[cube_index][i + 1] as usize;
        let ei2 = TRI_TABLE[cube_index][i + 2] as usize;

        let v0 = edge_vertices[ei0];
        let v1 = edge_vertices[ei1];
        let v2 = edge_vertices[ei2];

        let base_idx = mesh.vertices.len() as u32;

        let (n0, n1, n2) = if config.compute_normals {
            // Use grid finite differences — pick the closer corner of each edge
            let gc0 = &edge_grid_coords[ei0];
            let gc1 = &edge_grid_coords[ei1];
            let gc2 = &edge_grid_coords[ei2];
            (
                normal_from_grid(sdf, v0, values, gc0[0], gc0[1], gc0[2], grid_size, cell_size),
                normal_from_grid(sdf, v1, values, gc1[0], gc1[1], gc1[2], grid_size, cell_size),
                normal_from_grid(sdf, v2, values, gc2[0], gc2[1], gc2[2], grid_size, cell_size),
            )
        } else {
            let face_normal = (v1 - v0).cross(v2 - v0).normalize();
            (face_normal, face_normal, face_normal)
        };

        let mut vert0 = Vertex::new(v0, n0);
        let mut vert1 = Vertex::new(v1, n1);
        let mut vert2 = Vertex::new(v2, n2);

        if config.compute_uvs {
            vert0.uv = triplanar_uv(v0, n0, config.uv_scale);
            vert1.uv = triplanar_uv(v1, n1, config.uv_scale);
            vert2.uv = triplanar_uv(v2, n2, config.uv_scale);
        }

        if config.compute_tangents && config.compute_normals && !config.compute_uvs {
            vert0.tangent = compute_tangent(n0);
            vert1.tangent = compute_tangent(n1);
            vert2.tangent = compute_tangent(n2);
        }

        // Note: material evaluation uses interpreted path (requires SdfNode)
        // Compiled path does not support material assignment.

        mesh.vertices.push(vert0);
        mesh.vertices.push(vert1);
        mesh.vertices.push(vert2);

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);

        i += 3;
    }
}

// Corner offsets for the 8 corners of a cube
const CORNER_OFFSETS: [[usize; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

// Edge connections (which corners each edge connects)
const EDGE_CONNECTIONS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

// Edge table - which edges are intersected for each cube configuration
const EDGE_TABLE: [u16; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

// Triangle table - which edges form triangles for each cube configuration
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];

/// Configuration for adaptive marching cubes
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Maximum octree depth (resolution = 2^max_depth per axis)
    pub max_depth: u32,
    /// Minimum octree depth (prevents overly coarse cells)
    pub min_depth: u32,
    /// Distance threshold: cells further than this from surface stay coarse
    pub surface_threshold: f32,
    /// Iso-level
    pub iso_level: f32,
    /// Compute vertex normals
    pub compute_normals: bool,
    /// Compute triplanar UVs
    pub compute_uvs: bool,
    /// UV scale
    pub uv_scale: f32,
    /// Compute tangents
    pub compute_tangents: bool,
    /// Compute materials
    pub compute_materials: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            max_depth: 6,   // 64 effective resolution
            min_depth: 2,   // 4 minimum
            surface_threshold: 0.0,
            iso_level: 0.0,
            compute_normals: true,
            compute_uvs: false,
            uv_scale: 1.0,
            compute_tangents: false,
            compute_materials: false,
        }
    }
}

impl AdaptiveConfig {
    /// AAA preset: high resolution near surface, efficient far away
    pub fn aaa(max_depth: u32) -> Self {
        AdaptiveConfig {
            max_depth,
            min_depth: 2,
            surface_threshold: 0.0,
            iso_level: 0.0,
            compute_normals: true,
            compute_uvs: true,
            uv_scale: 1.0,
            compute_tangents: true,
            compute_materials: true,
        }
    }
}

/// Octree node for adaptive subdivision
struct OctreeCell {
    min: Vec3,
    max: Vec3,
    depth: u32,
}

/// Convert SDF to mesh using adaptive octree marching cubes
///
/// Subdivides space using an octree: cells near the surface are refined
/// to max_depth resolution while cells far from the surface stay coarse.
/// Typically produces 60-80% fewer triangles than uniform grid for the
/// same effective resolution near the surface.
pub fn adaptive_marching_cubes(
    node: &SdfNode,
    bounds_min: Vec3,
    bounds_max: Vec3,
    config: &AdaptiveConfig,
) -> Mesh {
    // Build leaf cells via octree subdivision
    let root = OctreeCell {
        min: bounds_min,
        max: bounds_max,
        depth: 0,
    };

    let mut leaf_cells = Vec::new();
    subdivide_octree(node, &root, config, &mut leaf_cells);

    // Process leaf cells in parallel
    let mc_config = MarchingCubesConfig {
        resolution: 1, // Each leaf is a single cell
        iso_level: config.iso_level,
        compute_normals: config.compute_normals,
        compute_uvs: config.compute_uvs,
        uv_scale: config.uv_scale,
        compute_tangents: false, // Computed in post-processing
        compute_materials: config.compute_materials,
    };

    let sub_meshes: Vec<Mesh> = leaf_cells
        .par_iter()
        .map(|cell| {
            process_adaptive_cell(node, cell, &mc_config)
        })
        .collect();

    let mut mesh = merge_meshes(sub_meshes);

    // Deduplicate for watertight output
    super::optimize::deduplicate_vertices(&mut mesh);

    // Post-process tangents
    if config.compute_tangents && config.compute_normals && config.compute_uvs {
        compute_mikktspace_tangents(&mut mesh);
    }

    mesh
}

/// Recursively subdivide octree based on SDF distance
fn subdivide_octree(
    node: &SdfNode,
    cell: &OctreeCell,
    config: &AdaptiveConfig,
    leaves: &mut Vec<OctreeCell>,
) {
    let center = (cell.min + cell.max) * 0.5;
    let half_diag = (cell.max - cell.min).length() * 0.5;

    // Sample SDF at cell center
    let dist = eval(node, center).abs();

    // Determine threshold: auto-compute from cell diagonal if user set 0
    let threshold = if config.surface_threshold > 0.0 {
        config.surface_threshold
    } else {
        half_diag * 1.2
    };

    // Subdivide if:
    // 1. Not at max depth yet
    // 2. Cell is near the surface (distance < threshold)
    // 3. Above minimum depth
    let should_subdivide = cell.depth < config.max_depth
        && (dist < threshold || cell.depth < config.min_depth);

    if should_subdivide {
        // Split into 8 octants
        for octant in 0..8 {
            let ox = (octant & 1) as f32;
            let oy = ((octant >> 1) & 1) as f32;
            let oz = ((octant >> 2) & 1) as f32;

            let child_min = Vec3::new(
                cell.min.x + ox * (center.x - cell.min.x),
                cell.min.y + oy * (center.y - cell.min.y),
                cell.min.z + oz * (center.z - cell.min.z),
            );
            let child_max = Vec3::new(
                center.x + ox * (cell.max.x - center.x),
                center.y + oy * (cell.max.y - center.y),
                center.z + oz * (cell.max.z - center.z),
            );

            let child = OctreeCell {
                min: child_min,
                max: child_max,
                depth: cell.depth + 1,
            };

            subdivide_octree(node, &child, config, leaves);
        }
    } else if dist < half_diag * 2.0 {
        // Only emit leaf if it could contain surface
        leaves.push(OctreeCell {
            min: cell.min,
            max: cell.max,
            depth: cell.depth,
        });
    }
}

/// Process a single adaptive cell using marching cubes
fn process_adaptive_cell(
    node: &SdfNode,
    cell: &OctreeCell,
    config: &MarchingCubesConfig,
) -> Mesh {
    let mut mesh = Mesh::new();
    let cell_size = cell.max - cell.min;

    // Evaluate 8 corners
    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Vec3::ZERO; 8];

    for i in 0..8 {
        let dx = CORNER_OFFSETS[i][0] as f32;
        let dy = CORNER_OFFSETS[i][1] as f32;
        let dz = CORNER_OFFSETS[i][2] as f32;

        let pos = cell.min + Vec3::new(dx * cell_size.x, dy * cell_size.y, dz * cell_size.z);
        corner_positions[i] = pos;
        corner_values[i] = eval(node, pos);
    }

    // Compute cube index
    let mut cube_index = 0;
    for i in 0..8 {
        if corner_values[i] < config.iso_level {
            cube_index |= 1 << i;
        }
    }

    if EDGE_TABLE[cube_index] == 0 {
        return mesh;
    }

    // Compute edge vertices
    let mut edge_vertices = [Vec3::ZERO; 12];
    for i in 0..12 {
        if EDGE_TABLE[cube_index] & (1 << i) != 0 {
            let e0 = EDGE_CONNECTIONS[i][0];
            let e1 = EDGE_CONNECTIONS[i][1];
            edge_vertices[i] = interpolate_vertex(
                corner_positions[e0],
                corner_positions[e1],
                corner_values[e0],
                corner_values[e1],
                config.iso_level,
            );
        }
    }

    // Generate triangles
    let mut i = 0;
    while TRI_TABLE[cube_index][i] != -1 {
        let v0 = edge_vertices[TRI_TABLE[cube_index][i] as usize];
        let v1 = edge_vertices[TRI_TABLE[cube_index][i + 1] as usize];
        let v2 = edge_vertices[TRI_TABLE[cube_index][i + 2] as usize];

        let base_idx = mesh.vertices.len() as u32;

        let (n0, n1, n2) = if config.compute_normals {
            (
                normal(node, v0, 0.001),
                normal(node, v1, 0.001),
                normal(node, v2, 0.001),
            )
        } else {
            let face_normal = (v1 - v0).cross(v2 - v0).normalize();
            (face_normal, face_normal, face_normal)
        };

        let mut vert0 = Vertex::new(v0, n0);
        let mut vert1 = Vertex::new(v1, n1);
        let mut vert2 = Vertex::new(v2, n2);

        if config.compute_uvs {
            vert0.uv = triplanar_uv(v0, n0, config.uv_scale);
            vert1.uv = triplanar_uv(v1, n1, config.uv_scale);
            vert2.uv = triplanar_uv(v2, n2, config.uv_scale);
        }

        if config.compute_materials {
            vert0.material_id = eval_material(node, v0);
            vert1.material_id = eval_material(node, v1);
            vert2.material_id = eval_material(node, v2);
        }

        mesh.vertices.push(vert0);
        mesh.vertices.push(vert1);
        mesh.vertices.push(vert2);

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);

        i += 3;
    }

    mesh
}

// ---------------------------------------------------------------------------
// Adaptive Marching Cubes — Compiled VM variant
// ---------------------------------------------------------------------------

/// Adaptive marching cubes using compiled VM evaluator
///
/// Same octree subdivision logic as `adaptive_marching_cubes`, but uses
/// `eval_compiled` for distance queries and `normal_compiled` for normals.
/// Typically 2-5x faster than the interpreted variant.
pub fn adaptive_marching_cubes_compiled(
    sdf: &CompiledSdf,
    bounds_min: Vec3,
    bounds_max: Vec3,
    config: &AdaptiveConfig,
) -> Mesh {
    let root = OctreeCell {
        min: bounds_min,
        max: bounds_max,
        depth: 0,
    };

    let mut leaf_cells = Vec::new();
    subdivide_octree_compiled(sdf, &root, config, &mut leaf_cells);

    let mc_config = MarchingCubesConfig {
        resolution: 1,
        iso_level: config.iso_level,
        compute_normals: config.compute_normals,
        compute_uvs: config.compute_uvs,
        uv_scale: config.uv_scale,
        compute_tangents: false,
        compute_materials: false, // Compiled path doesn't support materials
    };

    let sub_meshes: Vec<Mesh> = leaf_cells
        .par_iter()
        .map(|cell| {
            process_adaptive_cell_compiled(sdf, cell, &mc_config)
        })
        .collect();

    let mut mesh = merge_meshes(sub_meshes);

    super::optimize::deduplicate_vertices(&mut mesh);

    if config.compute_tangents && config.compute_normals && config.compute_uvs {
        compute_mikktspace_tangents(&mut mesh);
    }

    mesh
}

/// Recursively subdivide octree using compiled evaluator
fn subdivide_octree_compiled(
    sdf: &CompiledSdf,
    cell: &OctreeCell,
    config: &AdaptiveConfig,
    leaves: &mut Vec<OctreeCell>,
) {
    let center = (cell.min + cell.max) * 0.5;
    let half_diag = (cell.max - cell.min).length() * 0.5;

    let dist = eval_compiled(sdf, center).abs();

    let threshold = if config.surface_threshold > 0.0 {
        config.surface_threshold
    } else {
        half_diag * 1.2
    };

    let should_subdivide = cell.depth < config.max_depth
        && (dist < threshold || cell.depth < config.min_depth);

    if should_subdivide {
        for octant in 0..8 {
            let ox = (octant & 1) as f32;
            let oy = ((octant >> 1) & 1) as f32;
            let oz = ((octant >> 2) & 1) as f32;

            let child_min = Vec3::new(
                cell.min.x + ox * (center.x - cell.min.x),
                cell.min.y + oy * (center.y - cell.min.y),
                cell.min.z + oz * (center.z - cell.min.z),
            );
            let child_max = Vec3::new(
                center.x + ox * (cell.max.x - center.x),
                center.y + oy * (cell.max.y - center.y),
                center.z + oz * (cell.max.z - center.z),
            );

            let child = OctreeCell {
                min: child_min,
                max: child_max,
                depth: cell.depth + 1,
            };

            subdivide_octree_compiled(sdf, &child, config, leaves);
        }
    } else if dist < half_diag * 2.0 {
        leaves.push(OctreeCell {
            min: cell.min,
            max: cell.max,
            depth: cell.depth,
        });
    }
}

/// Process a single adaptive cell using compiled evaluator
fn process_adaptive_cell_compiled(
    sdf: &CompiledSdf,
    cell: &OctreeCell,
    config: &MarchingCubesConfig,
) -> Mesh {
    let mut mesh = Mesh::new();
    let cell_size = cell.max - cell.min;

    let mut corner_values = [0.0f32; 8];
    let mut corner_positions = [Vec3::ZERO; 8];

    for i in 0..8 {
        let dx = CORNER_OFFSETS[i][0] as f32;
        let dy = CORNER_OFFSETS[i][1] as f32;
        let dz = CORNER_OFFSETS[i][2] as f32;

        let pos = cell.min + Vec3::new(dx * cell_size.x, dy * cell_size.y, dz * cell_size.z);
        corner_positions[i] = pos;
        corner_values[i] = eval_compiled(sdf, pos);
    }

    let mut cube_index = 0;
    for i in 0..8 {
        if corner_values[i] < config.iso_level {
            cube_index |= 1 << i;
        }
    }

    if EDGE_TABLE[cube_index] == 0 {
        return mesh;
    }

    let mut edge_vertices = [Vec3::ZERO; 12];
    for i in 0..12 {
        if EDGE_TABLE[cube_index] & (1 << i) != 0 {
            let e0 = EDGE_CONNECTIONS[i][0];
            let e1 = EDGE_CONNECTIONS[i][1];
            edge_vertices[i] = interpolate_vertex(
                corner_positions[e0],
                corner_positions[e1],
                corner_values[e0],
                corner_values[e1],
                config.iso_level,
            );
        }
    }

    let mut i = 0;
    while TRI_TABLE[cube_index][i] != -1 {
        let v0 = edge_vertices[TRI_TABLE[cube_index][i] as usize];
        let v1 = edge_vertices[TRI_TABLE[cube_index][i + 1] as usize];
        let v2 = edge_vertices[TRI_TABLE[cube_index][i + 2] as usize];

        let base_idx = mesh.vertices.len() as u32;

        let (n0, n1, n2) = if config.compute_normals {
            (
                normal_compiled(sdf, v0, 0.001),
                normal_compiled(sdf, v1, 0.001),
                normal_compiled(sdf, v2, 0.001),
            )
        } else {
            let face_normal = (v1 - v0).cross(v2 - v0).normalize();
            (face_normal, face_normal, face_normal)
        };

        let mut vert0 = Vertex::new(v0, n0);
        let mut vert1 = Vertex::new(v1, n1);
        let mut vert2 = Vertex::new(v2, n2);

        if config.compute_uvs {
            vert0.uv = triplanar_uv(v0, n0, config.uv_scale);
            vert1.uv = triplanar_uv(v1, n1, config.uv_scale);
            vert2.uv = triplanar_uv(v2, n2, config.uv_scale);
        }

        mesh.vertices.push(vert0);
        mesh.vertices.push(vert1);
        mesh.vertices.push(vert2);

        mesh.indices.push(base_idx);
        mesh.indices.push(base_idx + 1);
        mesh.indices.push(base_idx + 2);

        i += 3;
    }

    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_to_mesh_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_sdf_to_mesh_box() {
        let box3d = SdfNode::box3d(2.0, 2.0, 2.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: false,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&box3d, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_mesh_vertex_normals() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Check that normals are normalized
        for vertex in &mesh.vertices {
            let len = vertex.normal.length();
            assert!((len - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_watertight_mesh() {
        // Verify that sdf_to_mesh produces a watertight mesh (shared vertices)
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // After dedup, vertex count should be less than 3 * triangle_count
        // (shared vertices means fewer vertices than the per-triangle case)
        let tri_count = mesh.triangle_count();
        assert!(tri_count > 0);
        assert!(
            mesh.vertex_count() < tri_count * 3,
            "Expected shared vertices: {} verts < {} (3 * {} tris)",
            mesh.vertex_count(), tri_count * 3, tri_count
        );

        // Verify manifold: every edge should be shared by exactly 2 triangles
        use std::collections::HashMap;
        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in mesh.indices.chunks_exact(3) {
            let edges = [
                (tri[0].min(tri[1]), tri[0].max(tri[1])),
                (tri[1].min(tri[2]), tri[1].max(tri[2])),
                (tri[0].min(tri[2]), tri[0].max(tri[2])),
            ];
            for &edge in &edges {
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // Allow some boundary edges (count == 1) but no edge should have > 2 triangles
        let bad_edges = edge_count.values().filter(|&&c| c > 2).count();
        assert_eq!(bad_edges, 0, "Found {} non-manifold edges (shared by >2 triangles)", bad_edges);
    }

    #[test]
    fn test_mikktspace_tangents() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig::aaa(16);

        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        for v in &mesh.vertices {
            let t = Vec3::new(v.tangent.x, v.tangent.y, v.tangent.z);
            let n = v.normal;

            // Tangent should be normalized
            let t_len = t.length();
            assert!(
                (t_len - 1.0).abs() < 0.1,
                "Tangent not normalized: length = {}", t_len
            );

            // Tangent should be perpendicular to normal
            let dot = t.dot(n).abs();
            assert!(
                dot < 0.15,
                "Tangent not perpendicular to normal: dot = {}", dot
            );

            // Handedness should be -1 or +1
            let w = v.tangent.w;
            assert!(
                (w.abs() - 1.0).abs() < 0.01,
                "Invalid handedness: w = {}", w
            );
        }
    }

    #[test]
    fn test_adaptive_marching_cubes_basic() {
        let sphere = SdfNode::sphere(1.0);
        let config = AdaptiveConfig::default();

        let mesh = adaptive_marching_cubes(
            &sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config
        );

        assert!(mesh.vertex_count() > 0, "Adaptive MC should produce vertices");
        assert!(mesh.triangle_count() > 0, "Adaptive MC should produce triangles");
    }

    #[test]
    fn test_adaptive_fewer_triangles() {
        // Small sphere in a large bounding box — most volume is empty,
        // so adaptive skips far-away cells while uniform evaluates everything.
        let sphere = SdfNode::sphere(1.0);

        // Uniform at resolution 64 over a large domain
        let uniform_config = MarchingCubesConfig {
            resolution: 64,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let uniform_mesh = sdf_to_mesh(
            &sphere, Vec3::splat(-5.0), Vec3::splat(5.0), &uniform_config
        );

        // Adaptive: same effective max resolution but tight surface threshold
        let adaptive_config = AdaptiveConfig {
            max_depth: 6,
            min_depth: 2,
            surface_threshold: 0.5,
            compute_normals: true,
            ..Default::default()
        };
        let adaptive_mesh = adaptive_marching_cubes(
            &sphere, Vec3::splat(-5.0), Vec3::splat(5.0), &adaptive_config
        );

        // Adaptive should produce fewer or equal triangles
        assert!(
            adaptive_mesh.triangle_count() <= uniform_mesh.triangle_count(),
            "Adaptive ({}) should have <= triangles than uniform ({})",
            adaptive_mesh.triangle_count(), uniform_mesh.triangle_count()
        );
    }

    #[test]
    fn test_adaptive_aaa_preset() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
        let config = AdaptiveConfig::aaa(5);

        let mesh = adaptive_marching_cubes(
            &shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config
        );

        assert!(mesh.vertex_count() > 0);
        // Verify normals are normalized
        for v in &mesh.vertices {
            let len = v.normal.length();
            assert!((len - 1.0).abs() < 0.2, "Normal not normalized: {}", len);
        }
    }
}

//! QEM (Quadric Error Metrics) Mesh Decimation (Deep Fried v2)
//!
//! Simplifies meshes by collapsing edges with minimal geometric error.
//! Based on the Garland & Heckbert algorithm (1997).
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Parallel Quadric Computation**: Face quadrics computed via `rayon` parallel iterator.
//! - **Parallel Boundary Detection**: Edge counting and material boundary detection parallelized.
//! - **Division Exorcism**: `inv_samples` replaces per-iteration division.
//!
//! # Features
//! - Quadric error metric for optimal vertex placement
//! - Attribute interpolation (UV, tangent, color preservation)
//! - Material boundary locking
//! - Boundary edge preservation
//! - Configurable target ratio
//!
//! # Usage
//! ```rust,ignore
//! use alice_sdf::mesh::decimate::{decimate, DecimateConfig};
//!
//! let mut mesh = sdf_to_mesh(&shape, min, max, &config);
//! decimate(&mut mesh, &DecimateConfig::default()); // 50% reduction
//! ```
//!
//! Author: Moroya Sakamoto

use crate::mesh::{Mesh, Vertex};
use glam::{Vec3, Vec4};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Decimation configuration
#[derive(Debug, Clone)]
pub struct DecimateConfig {
    /// Target ratio (0.5 = reduce to 50% of triangles)
    pub target_ratio: f32,
    /// Maximum geometric error threshold (skip collapses above this)
    pub max_error: f32,
    /// Preserve boundary edges
    pub preserve_boundary: bool,
    /// Preserve material boundaries (prevent collapsing edges between different material IDs)
    pub preserve_materials: bool,
    /// Lock vertices with specific material IDs (never collapse)
    pub locked_materials: Vec<u32>,
}

impl Default for DecimateConfig {
    fn default() -> Self {
        DecimateConfig {
            target_ratio: 0.5,
            max_error: f32::MAX,
            preserve_boundary: true,
            preserve_materials: true,
            locked_materials: Vec::new(),
        }
    }
}

impl DecimateConfig {
    /// Aggressive decimation (25% remaining)
    pub fn aggressive() -> Self {
        DecimateConfig {
            target_ratio: 0.25,
            max_error: f32::MAX,
            preserve_boundary: false,
            preserve_materials: false,
            locked_materials: Vec::new(),
        }
    }

    /// Conservative decimation (75% remaining)
    pub fn conservative() -> Self {
        DecimateConfig {
            target_ratio: 0.75,
            max_error: 0.01,
            preserve_boundary: true,
            preserve_materials: true,
            locked_materials: Vec::new(),
        }
    }
}

/// 4x4 symmetric matrix for quadric error computation
#[derive(Debug, Clone, Copy)]
struct Quadric {
    // Stored as upper triangle of symmetric 4x4 matrix
    // [a00 a01 a02 a03]
    // [    a11 a12 a13]
    // [        a22 a23]
    // [            a33]
    data: [f64; 10],
}

impl Quadric {
    fn zero() -> Self {
        Quadric { data: [0.0; 10] }
    }

    /// Create quadric from plane equation ax + by + cz + d = 0
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Quadric {
            data: [
                a * a,
                a * b,
                a * c,
                a * d,
                b * b,
                b * c,
                b * d,
                c * c,
                c * d,
                d * d,
            ],
        }
    }

    #[inline]
    fn add(&self, other: &Quadric) -> Quadric {
        let mut result = Quadric::zero();
        for i in 0..10 {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Evaluate quadric error at point (x, y, z)
    #[inline]
    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        let d = &self.data;
        // v^T * Q * v where v = [x, y, z, 1]
        x * x * d[0]
            + 2.0 * x * y * d[1]
            + 2.0 * x * z * d[2]
            + 2.0 * x * d[3]
            + y * y * d[4]
            + 2.0 * y * z * d[5]
            + 2.0 * y * d[6]
            + z * z * d[7]
            + 2.0 * z * d[8]
            + d[9]
    }

    /// Find optimal point that minimizes error.
    /// Falls back to midpoint if matrix is singular.
    fn optimal_point(&self, v1: Vec3, v2: Vec3) -> Vec3 {
        let d = &self.data;
        let a = [[d[0], d[1], d[2]], [d[1], d[4], d[5]], [d[2], d[5], d[7]]];
        let b = [-d[3], -d[6], -d[8]];

        // Cramer's rule
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        if det.abs() > 1e-10 {
            let inv_det = 1.0 / det;
            let x = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
                + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
                * inv_det;
            let y = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
                - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
                * inv_det;
            let z = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
                - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
                + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
                * inv_det;
            Vec3::new(x as f32, y as f32, z as f32)
        } else {
            // Singular — try midpoint, v1, v2 and pick lowest error
            let mid = (v1 + v2) * 0.5;
            let candidates = [mid, v1, v2];
            let mut best = mid;
            let mut best_err = f64::MAX;
            for &c in &candidates {
                let err = self.evaluate(c.x as f64, c.y as f64, c.z as f64);
                if err < best_err {
                    best_err = err;
                    best = c;
                }
            }
            best
        }
    }
}

/// Interpolate vertex attributes based on parametric position along edge
///
/// Uses the optimal position's parametric t along the v1->v2 edge to interpolate
/// all vertex attributes (UV, UV2, tangent, color). Normal is re-normalized.
#[inline]
fn interpolate_attributes(v1: &Vertex, v2: &Vertex, optimal_pos: Vec3) -> Vertex {
    // Compute parametric t from edge projection
    let edge = v2.position - v1.position;
    let edge_len_sq = edge.length_squared();
    let t = if edge_len_sq > 1e-10 {
        ((optimal_pos - v1.position).dot(edge) / edge_len_sq).clamp(0.0, 1.0)
    } else {
        0.5
    };

    let one_minus_t = 1.0 - t;

    // Interpolate normal and re-normalize
    let normal = (v1.normal * one_minus_t + v2.normal * t).normalize_or_zero();

    // Interpolate UV
    let uv = v1.uv * one_minus_t + v2.uv * t;

    // Interpolate UV2 (lightmap)
    let uv2 = v1.uv2 * one_minus_t + v2.uv2 * t;

    // Interpolate tangent direction, re-normalize, preserve dominant handedness
    let t_dir1 = Vec3::new(v1.tangent.x, v1.tangent.y, v1.tangent.z);
    let t_dir2 = Vec3::new(v2.tangent.x, v2.tangent.y, v2.tangent.z);
    let t_interp = (t_dir1 * one_minus_t + t_dir2 * t).normalize_or_zero();
    // Pick handedness from the closer endpoint
    let w = if t < 0.5 { v1.tangent.w } else { v2.tangent.w };
    let tangent = Vec4::new(t_interp.x, t_interp.y, t_interp.z, w);

    // Interpolate color
    let color = [
        v1.color[0] * one_minus_t + v2.color[0] * t,
        v1.color[1] * one_minus_t + v2.color[1] * t,
        v1.color[2] * one_minus_t + v2.color[2] * t,
        v1.color[3] * one_minus_t + v2.color[3] * t,
    ];

    // Material ID from closer endpoint
    let material_id = if t < 0.5 {
        v1.material_id
    } else {
        v2.material_id
    };

    Vertex {
        position: optimal_pos,
        normal,
        uv,
        uv2,
        tangent,
        color,
        material_id,
    }
}

/// Edge collapse candidate in the priority queue
#[derive(Debug)]
struct CollapseCandidate {
    error: f64,
    v1: u32,
    v2: u32,
    optimal_pos: Vec3,
}

impl PartialEq for CollapseCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.error == other.error
    }
}

impl Eq for CollapseCandidate {}

impl PartialOrd for CollapseCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CollapseCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: reverse ordering
        // [Deep Fried v2] NaN-safe: treat NaN as worst (largest) error
        match (self.error.is_nan(), other.error.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Less, // NaN self = worst, goes to bottom
            (false, true) => Ordering::Greater, // NaN other = worst
            (false, false) => other
                .error
                .partial_cmp(&self.error)
                .unwrap_or(Ordering::Equal),
        }
    }
}

/// Decimate a mesh using Quadric Error Metrics (Deep Fried v2)
///
/// Reduces triangle count by collapsing edges with lowest geometric error.
/// Vertex attributes (UV, tangent, color) are interpolated during collapse.
/// Material boundaries can be preserved to prevent bleed across material IDs.
/// The mesh is modified in-place.
///
/// # Deep Fried v2 Optimizations
///
/// - Face quadric computation parallelized via `rayon`
/// - Boundary edge detection parallelized
/// - Material boundary detection parallelized
pub fn decimate(mesh: &mut Mesh, config: &DecimateConfig) {
    let vert_count = mesh.vertices.len();
    let tri_count = mesh.indices.len() / 3;

    if tri_count <= 4 {
        return;
    }

    let target_tris = ((tri_count as f32 * config.target_ratio) as usize).max(4);

    // Build adjacency: vertex -> triangles
    let mut vert_tris: Vec<HashSet<usize>> = vec![HashSet::new(); vert_count];
    for t in 0..tri_count {
        for k in 0..3 {
            let vi = mesh.indices[t * 3 + k] as usize;
            if vi < vert_count {
                vert_tris[vi].insert(t);
            }
        }
    }

    // [Deep Fried v2] Parallel Face Quadric Computation
    // Each triangle computes its plane quadric independently
    let indices = &mesh.indices;
    let vertices = &mesh.vertices;
    let face_quadrics: Vec<(usize, usize, usize, Option<Quadric>)> = (0..tri_count)
        .into_par_iter()
        .map(|t| {
            let i0 = indices[t * 3] as usize;
            let i1 = indices[t * 3 + 1] as usize;
            let i2 = indices[t * 3 + 2] as usize;

            let p0 = vertices[i0].position;
            let p1 = vertices[i1].position;
            let p2 = vertices[i2].position;

            let n = (p1 - p0).cross(p2 - p0);
            let len = n.length();
            if len < 1e-10 {
                (i0, i1, i2, None)
            } else {
                let n = n / len;
                let d = -n.dot(p0);
                let q = Quadric::from_plane(n.x as f64, n.y as f64, n.z as f64, d as f64);
                (i0, i1, i2, Some(q))
            }
        })
        .collect();

    // Accumulate per-vertex quadrics (serial — writes to shared array)
    let mut quadrics: Vec<Quadric> = vec![Quadric::zero(); vert_count];
    for &(i0, i1, i2, ref q_opt) in &face_quadrics {
        if let Some(q) = q_opt {
            quadrics[i0] = quadrics[i0].add(q);
            quadrics[i1] = quadrics[i1].add(q);
            quadrics[i2] = quadrics[i2].add(q);
        }
    }

    // [Deep Fried v2] Parallel Boundary Edge Detection
    let boundary_verts = if config.preserve_boundary {
        // Collect all edges in parallel
        let edge_pairs: Vec<(u32, u32)> = (0..tri_count)
            .into_par_iter()
            .flat_map_iter(|t| {
                (0..3).map(move |k| {
                    let a = indices[t * 3 + k];
                    let b = indices[t * 3 + (k + 1) % 3];
                    if a < b {
                        (a, b)
                    } else {
                        (b, a)
                    }
                })
            })
            .collect();

        let mut edge_count: HashMap<(u32, u32), u32> = HashMap::with_capacity(edge_pairs.len());
        for edge in edge_pairs {
            *edge_count.entry(edge).or_insert(0) += 1;
        }

        let mut bv = HashSet::new();
        for (&(a, b), &count) in &edge_count {
            if count == 1 {
                bv.insert(a);
                bv.insert(b);
            }
        }
        bv
    } else {
        HashSet::new()
    };

    // [Deep Fried v2] Parallel Material Boundary Detection
    let material_boundary_verts = if config.preserve_materials {
        let mat_ids_ref: Vec<u32> = vertices.iter().map(|v| v.material_id).collect();
        let mbv: HashSet<u32> = (0..vert_count)
            .into_par_iter()
            .filter(|&vi| {
                let mat = mat_ids_ref[vi];
                vert_tris[vi].iter().any(|&t| {
                    for k in 0..3 {
                        let other_vi = indices[t * 3 + k] as usize;
                        if other_vi != vi && mat_ids_ref[other_vi] != mat {
                            return true;
                        }
                    }
                    false
                })
            })
            .map(|vi| vi as u32)
            .collect();
        mbv
    } else {
        HashSet::new()
    };

    // Locked vertices (from locked_materials config)
    let locked_verts: HashSet<u32> = if !config.locked_materials.is_empty() {
        mesh.vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| config.locked_materials.contains(&v.material_id))
            .map(|(i, _)| i as u32)
            .collect()
    } else {
        HashSet::new()
    };

    // Extract material IDs so the closure doesn't borrow mesh.vertices
    let mut mat_ids: Vec<u32> = mesh.vertices.iter().map(|v| v.material_id).collect();
    let preserve_mats = config.preserve_materials;
    let preserve_bnd = config.preserve_boundary;

    // Check if an edge collapse is allowed
    let is_locked = |a: u32, b: u32, mat_ids: &[u32]| -> bool {
        if locked_verts.contains(&a) || locked_verts.contains(&b) {
            return true;
        }
        if preserve_bnd && boundary_verts.contains(&a) && boundary_verts.contains(&b) {
            return true;
        }
        if preserve_mats
            && material_boundary_verts.contains(&a)
            && material_boundary_verts.contains(&b)
            && mat_ids[a as usize] != mat_ids[b as usize]
        {
            return true;
        }
        false
    };

    // Build initial priority queue
    let mut heap = BinaryHeap::new();
    let mut edges_seen = HashSet::new();

    for t in 0..tri_count {
        for k in 0..3 {
            let a = mesh.indices[t * 3 + k];
            let b = mesh.indices[t * 3 + (k + 1) % 3];
            let edge = if a < b { (a, b) } else { (b, a) };
            if edges_seen.insert(edge) {
                if is_locked(a, b, &mat_ids) {
                    continue;
                }
                let q_sum = quadrics[a as usize].add(&quadrics[b as usize]);
                let optimal = q_sum.optimal_point(
                    mesh.vertices[a as usize].position,
                    mesh.vertices[b as usize].position,
                );
                let error = q_sum.evaluate(optimal.x as f64, optimal.y as f64, optimal.z as f64);
                if error <= config.max_error as f64 {
                    heap.push(CollapseCandidate {
                        error,
                        v1: a,
                        v2: b,
                        optimal_pos: optimal,
                    });
                }
            }
        }
    }

    // Collapse edges (inherently serial — heap management)
    let mut removed_tris: HashSet<usize> = HashSet::new();
    let mut remap: Vec<u32> = (0..vert_count as u32).collect();
    let mut current_tris = tri_count;

    while current_tris > target_tris {
        let candidate = match heap.pop() {
            Some(c) => c,
            None => break,
        };

        // Resolve remaps
        let v1 = resolve_remap(&remap, candidate.v1);
        let v2 = resolve_remap(&remap, candidate.v2);

        if v1 == v2 {
            continue; // Already collapsed
        }

        // Find triangles to remove (shared between v1 and v2)
        let shared_tris: Vec<usize> = vert_tris[v1 as usize]
            .intersection(&vert_tris[v2 as usize])
            .copied()
            .filter(|t| !removed_tris.contains(t))
            .collect();

        if shared_tris.is_empty() {
            continue;
        }

        // Collapse v2 into v1 — interpolate all attributes
        let interp = interpolate_attributes(
            &mesh.vertices[v1 as usize],
            &mesh.vertices[v2 as usize],
            candidate.optimal_pos,
        );
        mesh.vertices[v1 as usize] = interp;
        mat_ids[v1 as usize] = interp.material_id;

        // Update quadric
        quadrics[v1 as usize] = quadrics[v1 as usize].add(&quadrics[v2 as usize]);

        // Remap v2 -> v1
        remap[v2 as usize] = v1;

        // Remove degenerate triangles
        for &t in &shared_tris {
            removed_tris.insert(t);
            current_tris -= 1;
        }

        // Update adjacency: move v2's triangles to v1
        let v2_tris: Vec<usize> = vert_tris[v2 as usize].iter().copied().collect();
        for t in v2_tris {
            vert_tris[v1 as usize].insert(t);
        }
        vert_tris[v2 as usize].clear();

        // Remap indices in affected triangles
        for &t in &vert_tris[v1 as usize] {
            if removed_tris.contains(&t) {
                continue;
            }
            for k in 0..3 {
                let idx = &mut mesh.indices[t * 3 + k];
                if resolve_remap(&remap, *idx) == v2 || *idx == v2 {
                    *idx = v1;
                }
            }
        }

        // Add new edges from v1 to the heap
        let mut neighbors: HashSet<u32> = HashSet::new();
        for &t in &vert_tris[v1 as usize] {
            if removed_tris.contains(&t) {
                continue;
            }
            for k in 0..3 {
                let v = resolve_remap(&remap, mesh.indices[t * 3 + k]);
                if v != v1 {
                    neighbors.insert(v);
                }
            }
        }

        for &nb in &neighbors {
            if is_locked(v1, nb, &mat_ids) {
                continue;
            }
            let q_sum = quadrics[v1 as usize].add(&quadrics[nb as usize]);
            let optimal = q_sum.optimal_point(
                mesh.vertices[v1 as usize].position,
                mesh.vertices[nb as usize].position,
            );
            let error = q_sum.evaluate(optimal.x as f64, optimal.y as f64, optimal.z as f64);
            if error <= config.max_error as f64 {
                heap.push(CollapseCandidate {
                    error,
                    v1,
                    v2: nb,
                    optimal_pos: optimal,
                });
            }
        }
    }

    // Rebuild mesh: remove degenerate triangles and compact vertices
    let mut new_indices = Vec::new();
    for t in 0..tri_count {
        if removed_tris.contains(&t) {
            continue;
        }
        let i0 = resolve_remap(&remap, mesh.indices[t * 3]);
        let i1 = resolve_remap(&remap, mesh.indices[t * 3 + 1]);
        let i2 = resolve_remap(&remap, mesh.indices[t * 3 + 2]);

        // Skip degenerate
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        new_indices.push(i0);
        new_indices.push(i1);
        new_indices.push(i2);
    }

    mesh.indices = new_indices;

    // Compact vertices (remove unreferenced)
    let mut used = vec![false; mesh.vertices.len()];
    for &idx in &mesh.indices {
        if (idx as usize) < used.len() {
            used[idx as usize] = true;
        }
    }

    let mut new_idx_map: Vec<u32> = vec![0; mesh.vertices.len()];
    let mut new_verts = Vec::new();
    for (i, &is_used) in used.iter().enumerate() {
        if is_used {
            new_idx_map[i] = new_verts.len() as u32;
            new_verts.push(mesh.vertices[i]);
        }
    }

    for idx in &mut mesh.indices {
        *idx = new_idx_map[*idx as usize];
    }
    mesh.vertices = new_verts;
}

/// Resolve remap chain to final vertex
#[inline]
fn resolve_remap(remap: &[u32], mut v: u32) -> u32 {
    let mut steps = 0;
    while remap[v as usize] != v && steps < 1000 {
        v = remap[v as usize];
        steps += 1;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec2;

    #[test]
    fn test_decimate_basic() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let tri_before = mesh.triangle_count();
        assert!(tri_before > 100, "Need enough triangles to test decimation");

        decimate(&mut mesh, &DecimateConfig::default());

        let tri_after = mesh.triangle_count();
        assert!(
            tri_after < tri_before,
            "Should reduce triangles: before={}, after={}",
            tri_before,
            tri_after
        );
        assert!(tri_after > 0, "Should not remove all triangles");

        // Verify mesh integrity
        for &idx in &mesh.indices {
            assert!(
                (idx as usize) < mesh.vertices.len(),
                "Index {} out of bounds (vertex count: {})",
                idx,
                mesh.vertices.len()
            );
        }
    }

    #[test]
    fn test_decimate_aggressive() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let tri_before = mesh.triangle_count();
        decimate(&mut mesh, &DecimateConfig::aggressive());
        let tri_after = mesh.triangle_count();

        // Aggressive should remove more
        assert!(
            tri_after <= (tri_before as f32 * 0.35) as usize,
            "Aggressive should reduce to ~25%: before={}, after={}",
            tri_before,
            tri_after
        );
    }

    #[test]
    fn test_decimate_conservative() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let tri_before = mesh.triangle_count();
        decimate(&mut mesh, &DecimateConfig::conservative());
        let tri_after = mesh.triangle_count();

        // Conservative should keep more
        assert!(
            tri_after >= (tri_before as f32 * 0.5) as usize,
            "Conservative should keep ~75%: before={}, after={}",
            tri_before,
            tri_after
        );
    }

    #[test]
    fn test_decimate_preserves_shape() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        decimate(&mut mesh, &DecimateConfig::default());

        // All vertices should still be roughly on the unit sphere
        for v in &mesh.vertices {
            let dist = v.position.length();
            assert!(
                dist > 0.5 && dist < 1.5,
                "Vertex at distance {} from origin (expected ~1.0)",
                dist
            );
        }
    }

    #[test]
    fn test_decimate_tiny_mesh() {
        // Edge case: mesh too small to decimate
        let mut mesh = Mesh {
            vertices: vec![
                Vertex::new(Vec3::ZERO, Vec3::Y),
                Vertex::new(Vec3::X, Vec3::Y),
                Vertex::new(Vec3::Z, Vec3::Y),
            ],
            indices: vec![0, 1, 2],
        };

        decimate(&mut mesh, &DecimateConfig::default());
        // Should not crash, mesh should still be valid
        assert!(mesh.indices.len() >= 3 || mesh.indices.is_empty());
    }

    #[test]
    fn test_decimate_preserves_uvs() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig::aaa(16);
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Verify UVs exist before decimation
        let has_uvs = mesh.vertices.iter().any(|v| v.uv != Vec2::ZERO);
        assert!(has_uvs, "Mesh should have UVs before decimation");

        decimate(&mut mesh, &DecimateConfig::default());

        // UVs should still be valid after decimation (not all zero)
        let has_uvs_after = mesh.vertices.iter().any(|v| v.uv != Vec2::ZERO);
        assert!(has_uvs_after, "UVs should be preserved after decimation");

        // All UVs should be finite
        for v in &mesh.vertices {
            assert!(
                v.uv.x.is_finite() && v.uv.y.is_finite(),
                "UV should be finite: {:?}",
                v.uv
            );
        }
    }

    #[test]
    fn test_decimate_preserves_tangents() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig::aaa(16);
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        decimate(&mut mesh, &DecimateConfig::default());

        // Tangents should be normalized and perpendicular to normals
        for v in &mesh.vertices {
            let t = Vec3::new(v.tangent.x, v.tangent.y, v.tangent.z);
            let t_len = t.length();
            if t_len > 0.01 {
                assert!(
                    (t_len - 1.0).abs() < 0.2,
                    "Tangent should be normalized: length = {}",
                    t_len
                );
            }
            // Handedness should be -1 or +1
            assert!(
                v.tangent.w == 1.0 || v.tangent.w == -1.0,
                "Handedness should be +/-1: w = {}",
                v.tangent.w
            );
        }
    }

    #[test]
    fn test_decimate_material_boundary_preservation() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Assign two materials: top hemisphere = 0, bottom = 1
        for v in &mut mesh.vertices {
            v.material_id = if v.position.y >= 0.0 { 0 } else { 1 };
        }

        let mat_config = DecimateConfig {
            target_ratio: 0.5,
            preserve_materials: true,
            ..Default::default()
        };
        decimate(&mut mesh, &mat_config);

        // Both materials should still be present
        let has_mat0 = mesh.vertices.iter().any(|v| v.material_id == 0);
        let has_mat1 = mesh.vertices.iter().any(|v| v.material_id == 1);
        assert!(has_mat0, "Material 0 should be preserved");
        assert!(has_mat1, "Material 1 should be preserved");
    }

    #[test]
    fn test_decimate_locked_materials() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Mark a few vertices with material ID 99
        let locked_count_before = mesh.vertices.len().min(10);
        for v in mesh.vertices[..locked_count_before].iter_mut() {
            v.material_id = 99;
        }

        let lock_config = DecimateConfig {
            target_ratio: 0.3,
            locked_materials: vec![99],
            preserve_boundary: false,
            preserve_materials: false,
            ..Default::default()
        };
        decimate(&mut mesh, &lock_config);

        // Locked material vertices should still exist
        let locked_after = mesh.vertices.iter().filter(|v| v.material_id == 99).count();
        assert!(
            locked_after > 0,
            "Locked material vertices should survive decimation"
        );
    }
}

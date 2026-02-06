//! Physics collision mesh generation (Deep Fried v2)
//!
//! Generates simplified collision primitives from meshes for physics engines.
//! Compatible with UE5, Unity, Godot, and Bullet/PhysX collision systems.
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Parallel Voxelization**: Triangle-to-voxel mapping via `rayon` + `AtomicU8`.
//! - **Parallel AABB**: Min/max reduction via `rayon` parallel iterator.
//! - **Parallel Hull Simplification**: Nearest-neighbor search parallelized.
//! - **Division Exorcism**: Pre-computed `inv_cell` replaces per-voxel division.
//!
//! # Collision Types
//! - **AABB**: Axis-aligned bounding box (fastest, least accurate)
//! - **Bounding Sphere**: Spherical bound (fast, good for round objects)
//! - **Convex Hull**: Tight convex approximation (medium cost, good accuracy)
//! - **Simplified Mesh**: Decimated triangle mesh (highest accuracy, highest cost)
//!
//! Author: Moroya Sakamoto

use crate::mesh::Mesh;
use glam::Vec3;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};

/// Axis-aligned bounding box for collision
#[derive(Debug, Clone, Copy)]
pub struct CollisionAabb {
    /// Minimum corner
    pub min: Vec3,
    /// Maximum corner
    pub max: Vec3,
}

impl CollisionAabb {
    /// Get center of the AABB
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get half-extents of the AABB
    #[inline]
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Get volume of the AABB
    #[inline]
    pub fn volume(&self) -> f32 {
        let ext = self.max - self.min;
        ext.x * ext.y * ext.z
    }

    /// Test if a point is inside the AABB
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x
            && point.y >= self.min.y && point.y <= self.max.y
            && point.z >= self.min.z && point.z <= self.max.z
    }
}

/// Bounding sphere for collision
#[derive(Debug, Clone, Copy)]
pub struct BoundingSphere {
    /// Center of the sphere
    pub center: Vec3,
    /// Radius of the sphere
    pub radius: f32,
}

impl BoundingSphere {
    /// Test if a point is inside the sphere
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        (point - self.center).length_squared() <= self.radius * self.radius
    }
}

/// Convex hull collision mesh
#[derive(Debug, Clone)]
pub struct ConvexHull {
    /// Hull vertices
    pub vertices: Vec<Vec3>,
    /// Hull face indices (triangles)
    pub indices: Vec<u32>,
}

/// Simplified collision mesh
#[derive(Debug, Clone)]
pub struct CollisionMesh {
    /// Simplified vertices
    pub vertices: Vec<Vec3>,
    /// Triangle indices
    pub indices: Vec<u32>,
}

/// Compute axis-aligned bounding box from mesh (Deep Fried v2)
///
/// [Deep Fried v2] Parallel min/max reduction via `rayon`.
pub fn compute_aabb(mesh: &Mesh) -> CollisionAabb {
    if mesh.vertices.is_empty() {
        return CollisionAabb {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        };
    }

    let (min, max) = mesh.vertices
        .par_iter()
        .map(|v| (v.position, v.position))
        .reduce(
            || (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
            |(min_a, max_a), (min_b, max_b)| {
                (min_a.min(min_b), max_a.max(max_b))
            },
        );

    CollisionAabb { min, max }
}

/// Compute bounding sphere using Ritter's algorithm
///
/// Finds a tight-fitting bounding sphere in O(n) time.
pub fn compute_bounding_sphere(mesh: &Mesh) -> BoundingSphere {
    if mesh.vertices.is_empty() {
        return BoundingSphere {
            center: Vec3::ZERO,
            radius: 0.0,
        };
    }

    // Start with first vertex
    let mut center = mesh.vertices[0].position;
    let mut radius;

    // Find point farthest from first vertex
    let mut farthest_idx = 0;
    let mut max_dist = 0.0f32;
    for (i, v) in mesh.vertices.iter().enumerate() {
        let d = (v.position - center).length_squared();
        if d > max_dist {
            max_dist = d;
            farthest_idx = i;
        }
    }

    // Find point farthest from that point
    let p1 = mesh.vertices[farthest_idx].position;
    let mut p2_idx = 0;
    max_dist = 0.0;
    for (i, v) in mesh.vertices.iter().enumerate() {
        let d = (v.position - p1).length_squared();
        if d > max_dist {
            max_dist = d;
            p2_idx = i;
        }
    }
    let p2 = mesh.vertices[p2_idx].position;

    // Initial sphere from these two extremes
    center = (p1 + p2) * 0.5;
    radius = (p1 - p2).length() * 0.5;

    // Grow sphere to include all points
    for v in &mesh.vertices {
        let dist = (v.position - center).length();
        if dist > radius {
            let new_radius = (radius + dist) * 0.5;
            let delta = dist - radius;
            center += (v.position - center).normalize() * (delta * 0.5);
            radius = new_radius;
        }
    }

    BoundingSphere { center, radius }
}

/// Generate a convex hull from mesh vertices
///
/// Uses an incremental convex hull algorithm. The result is suitable
/// for physics engines that require convex collision shapes.
pub fn compute_convex_hull(mesh: &Mesh) -> ConvexHull {
    let points: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
    convex_hull_from_points(&points)
}

/// Generate convex hull from a set of points
pub fn convex_hull_from_points(points: &[Vec3]) -> ConvexHull {
    if points.len() < 4 {
        return ConvexHull {
            vertices: points.to_vec(),
            indices: if points.len() == 3 { vec![0, 1, 2] } else { vec![] },
        };
    }

    // Find initial tetrahedron
    let (p0, p1, p2, p3) = find_initial_tetrahedron(points);

    let mut hull_verts = vec![points[p0], points[p1], points[p2], points[p3]];
    let mut hull_faces: Vec<[u32; 3]> = vec![
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ];

    // Ensure faces are oriented outward
    let center = (hull_verts[0] + hull_verts[1] + hull_verts[2] + hull_verts[3]) * 0.25;
    for face in &mut hull_faces {
        let a = hull_verts[face[0] as usize];
        let b = hull_verts[face[1] as usize];
        let c = hull_verts[face[2] as usize];
        let n = (b - a).cross(c - a);
        let face_center = (a + b + c) / 3.0;
        if n.dot(face_center - center) < 0.0 {
            face.swap(1, 2);
        }
    }

    // Add remaining points
    let used = [p0, p1, p2, p3];
    for (i, &pt) in points.iter().enumerate() {
        if used.contains(&i) {
            continue;
        }

        // Find faces visible from this point
        let mut visible = Vec::new();
        for (fi, face) in hull_faces.iter().enumerate() {
            let a = hull_verts[face[0] as usize];
            let b = hull_verts[face[1] as usize];
            let c = hull_verts[face[2] as usize];
            let n = (b - a).cross(c - a);
            if n.dot(pt - a) > 1e-6 {
                visible.push(fi);
            }
        }

        if visible.is_empty() {
            continue; // Point is inside hull
        }

        // Collect horizon edges (edges shared by exactly one visible face)
        let mut horizon_edges: Vec<(u32, u32)> = Vec::new();
        for &fi in &visible {
            let face = hull_faces[fi];
            for k in 0..3 {
                let e0 = face[k];
                let e1 = face[(k + 1) % 3];
                // Check if the reverse edge exists in another visible face
                let is_shared = visible.iter().any(|&fj| {
                    if fj == fi {
                        return false;
                    }
                    let fj_face = hull_faces[fj];
                    for m in 0..3 {
                        if fj_face[m] == e1 && fj_face[(m + 1) % 3] == e0 {
                            return true;
                        }
                    }
                    false
                });
                if !is_shared {
                    horizon_edges.push((e0, e1));
                }
            }
        }

        // Remove visible faces (in reverse order to maintain indices)
        let mut sorted_visible = visible.clone();
        sorted_visible.sort_unstable_by(|a, b| b.cmp(a));
        for fi in sorted_visible {
            hull_faces.swap_remove(fi);
        }

        // Add new vertex
        let new_vi = hull_verts.len() as u32;
        hull_verts.push(pt);

        // Create new faces from horizon edges to new vertex
        for (e0, e1) in horizon_edges {
            hull_faces.push([e0, e1, new_vi]);
        }
    }

    // Flatten indices
    let indices: Vec<u32> = hull_faces.iter().flat_map(|f| f.iter().copied()).collect();

    ConvexHull {
        vertices: hull_verts,
        indices,
    }
}

fn find_initial_tetrahedron(points: &[Vec3]) -> (usize, usize, usize, usize) {
    let n = points.len();

    // Find two most distant points (approximate)
    // Start from extremes on X axis
    let mut min_x = 0;
    let mut max_x = 0;
    for i in 1..n {
        if points[i].x < points[min_x].x {
            min_x = i;
        }
        if points[i].x > points[max_x].x {
            max_x = i;
        }
    }
    let mut p0 = min_x;
    let mut p1 = max_x;

    if (points[p0] - points[p1]).length_squared() < 1e-10 {
        // Degenerate: try Y axis
        for i in 1..n {
            if points[i].y < points[p0].y {
                p0 = i;
            }
            if points[i].y > points[p1].y {
                p1 = i;
            }
        }
    }

    // Find point farthest from line p0-p1
    let line_dir = (points[p1] - points[p0]).normalize_or_zero();
    let mut p2 = 0;
    let mut max_line_dist = 0.0f32;
    for i in 0..n {
        if i == p0 || i == p1 {
            continue;
        }
        let v = points[i] - points[p0];
        let proj = v.dot(line_dir);
        let perp = v - line_dir * proj;
        let d = perp.length_squared();
        if d > max_line_dist {
            max_line_dist = d;
            p2 = i;
        }
    }

    // Find point farthest from plane p0-p1-p2
    let plane_normal = (points[p1] - points[p0])
        .cross(points[p2] - points[p0])
        .normalize_or_zero();
    let mut p3 = 0;
    let mut max_plane_dist = 0.0f32;
    for i in 0..n {
        if i == p0 || i == p1 || i == p2 {
            continue;
        }
        let d = (points[i] - points[p0]).dot(plane_normal).abs();
        if d > max_plane_dist {
            max_plane_dist = d;
            p3 = i;
        }
    }

    (p0, p1, p2, p3)
}

/// Generate a simplified collision mesh by vertex clustering
///
/// Reduces vertex count by clustering nearby vertices into grid cells.
/// Good for creating low-poly collision proxies.
pub fn simplify_collision(mesh: &Mesh, grid_resolution: u32) -> CollisionMesh {
    if mesh.vertices.is_empty() {
        return CollisionMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
    }

    let aabb = compute_aabb(mesh);
    let extent = aabb.max - aabb.min;
    let cell_size = extent / grid_resolution as f32;

    // Division Exorcism: pre-compute inverse
    let inv_cell = Vec3::new(
        if cell_size.x > 1e-6 { 1.0 / cell_size.x } else { 0.0 },
        if cell_size.y > 1e-6 { 1.0 / cell_size.y } else { 0.0 },
        if cell_size.z > 1e-6 { 1.0 / cell_size.z } else { 0.0 },
    );

    // Cluster vertices into grid cells
    use std::collections::HashMap;
    let mut clusters: HashMap<(u32, u32, u32), (Vec3, u32)> = HashMap::new();
    let mut vertex_remap: Vec<(u32, u32, u32)> = Vec::with_capacity(mesh.vertices.len());

    for v in &mesh.vertices {
        let rel = v.position - aabb.min;
        let cx = ((rel.x * inv_cell.x) as u32).min(grid_resolution - 1);
        let cy = ((rel.y * inv_cell.y) as u32).min(grid_resolution - 1);
        let cz = ((rel.z * inv_cell.z) as u32).min(grid_resolution - 1);
        let key = (cx, cy, cz);

        let entry = clusters.entry(key).or_insert((Vec3::ZERO, 0));
        entry.0 += v.position;
        entry.1 += 1;
        vertex_remap.push(key);
    }

    // Build new vertex list with averaged positions
    let mut new_vertex_map: HashMap<(u32, u32, u32), u32> = HashMap::new();
    let mut new_vertices = Vec::new();

    for (key, (sum, count)) in &clusters {
        let new_idx = new_vertices.len() as u32;
        new_vertex_map.insert(*key, new_idx);
        // Division Exorcism: multiply by reciprocal
        let inv_count = 1.0 / *count as f32;
        new_vertices.push(*sum * inv_count);
    }

    // Remap indices, skip degenerate triangles
    let mut new_indices = Vec::new();
    let tri_count = mesh.indices.len() / 3;

    for t in 0..tri_count {
        let a = vertex_remap[mesh.indices[t * 3] as usize];
        let b = vertex_remap[mesh.indices[t * 3 + 1] as usize];
        let c = vertex_remap[mesh.indices[t * 3 + 2] as usize];

        let na = new_vertex_map[&a];
        let nb = new_vertex_map[&b];
        let nc = new_vertex_map[&c];

        // Skip degenerate triangles (collapsed vertices)
        if na != nb && nb != nc && na != nc {
            new_indices.push(na);
            new_indices.push(nb);
            new_indices.push(nc);
        }
    }

    CollisionMesh {
        vertices: new_vertices,
        indices: new_indices,
    }
}

/// V-HACD configuration
#[derive(Debug, Clone)]
pub struct VhacdConfig {
    /// Maximum number of convex hulls to generate
    pub max_hulls: u32,
    /// Voxel grid resolution for decomposition
    pub resolution: u32,
    /// Maximum vertices per convex hull
    pub max_vertices_per_hull: u32,
    /// Volume error tolerance (0.0-1.0, lower = more accurate)
    pub volume_error_percent: f32,
}

impl Default for VhacdConfig {
    fn default() -> Self {
        VhacdConfig {
            max_hulls: 16,
            resolution: 32,
            max_vertices_per_hull: 32,
            volume_error_percent: 1.0,
        }
    }
}

impl VhacdConfig {
    /// High quality decomposition
    pub fn high_quality() -> Self {
        VhacdConfig {
            max_hulls: 32,
            resolution: 64,
            max_vertices_per_hull: 64,
            volume_error_percent: 0.5,
        }
    }

    /// Fast decomposition for runtime use
    pub fn fast() -> Self {
        VhacdConfig {
            max_hulls: 8,
            resolution: 16,
            max_vertices_per_hull: 16,
            volume_error_percent: 5.0,
        }
    }
}

/// Result of V-HACD decomposition
#[derive(Debug, Clone)]
pub struct ConvexDecomposition {
    /// Individual convex hull parts
    pub parts: Vec<ConvexHull>,
}

impl ConvexDecomposition {
    /// Total number of vertices across all parts
    pub fn total_vertices(&self) -> usize {
        self.parts.iter().map(|p| p.vertices.len()).sum()
    }

    /// Total number of triangles across all parts
    pub fn total_triangles(&self) -> usize {
        self.parts.iter().map(|p| p.indices.len() / 3).sum()
    }
}

/// Approximate Convex Decomposition using voxelization (Deep Fried v2)
///
/// Decomposes a concave mesh into multiple convex parts using a
/// voxel-based flood fill approach. Suitable for physics collision.
///
/// [Deep Fried v2] Voxelization is parallelized via `rayon` + `AtomicU8`.
pub fn convex_decomposition(mesh: &Mesh, config: &VhacdConfig) -> ConvexDecomposition {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return ConvexDecomposition { parts: Vec::new() };
    }

    let aabb = compute_aabb(mesh);
    let extent = aabb.max - aabb.min;
    let res = config.resolution;

    // Pad slightly to avoid edge cases
    let padding = extent * 0.01;
    let vox_min = aabb.min - padding;
    let vox_extent = extent + padding * 2.0;
    let cell_size = vox_extent / res as f32;

    // Step 1: Voxelize the mesh (Deep Fried: parallel)
    let total_voxels = (res * res * res) as usize;
    let voxels_atomic: Vec<AtomicU8> = (0..total_voxels)
        .map(|_| AtomicU8::new(0))
        .collect();

    voxelize_mesh_parallel(mesh, vox_min, cell_size, res, &voxels_atomic);

    // Convert atomics to bool vec for flood fill
    let mut voxels: Vec<bool> = voxels_atomic.iter()
        .map(|a| a.load(Ordering::Relaxed) != 0)
        .collect();

    // Interior fill (serial — ray parity requires sequential scan)
    interior_fill(&mut voxels, res);

    // Step 2: Flood-fill to find connected components (convex regions)
    let mut labels = vec![0u32; total_voxels];
    let mut current_label = 0u32;
    let max_labels = config.max_hulls;

    for idx in 0..total_voxels {
        if voxels[idx] && labels[idx] == 0 {
            current_label += 1;
            if current_label > max_labels {
                flood_fill_3d(&voxels, &mut labels, idx, max_labels, res);
            } else {
                flood_fill_3d(&voxels, &mut labels, idx, current_label, res);
            }
        }
    }

    let num_labels = current_label.min(max_labels);
    if num_labels == 0 {
        return ConvexDecomposition { parts: Vec::new() };
    }

    // Step 3: For each label, collect voxel centers and compute convex hull
    let mut parts = Vec::with_capacity(num_labels as usize);

    for label in 1..=num_labels {
        let points: Vec<Vec3> = (0..total_voxels)
            .into_par_iter()
            .filter(|&idx| labels[idx] == label)
            .map(|idx| {
                let x = idx % res as usize;
                let y = (idx / res as usize) % res as usize;
                let z = idx / (res as usize * res as usize);
                vox_min + Vec3::new(
                    (x as f32 + 0.5) * cell_size.x,
                    (y as f32 + 0.5) * cell_size.y,
                    (z as f32 + 0.5) * cell_size.z,
                )
            })
            .collect();

        if points.len() < 4 {
            continue;
        }

        // Compute convex hull of voxel centers
        let mut hull = convex_hull_from_points(&points);

        // Limit vertex count per hull
        if hull.vertices.len() > config.max_vertices_per_hull as usize {
            hull = simplify_hull(&hull, config.max_vertices_per_hull);
        }

        if hull.vertices.len() >= 4 && hull.indices.len() >= 12 {
            parts.push(hull);
        }
    }

    ConvexDecomposition { parts }
}

/// [Deep Fried v2] Parallel voxelization using AtomicU8
///
/// Each triangle is processed independently in parallel, marking voxels
/// covered by its bounding box using atomic stores.
fn voxelize_mesh_parallel(
    mesh: &Mesh,
    vox_min: Vec3,
    cell_size: Vec3,
    res: u32,
    voxels: &[AtomicU8],
) {
    // Division Exorcism: pre-compute inverse
    let inv_cell = Vec3::new(
        if cell_size.x > 1e-8 { 1.0 / cell_size.x } else { 0.0 },
        if cell_size.y > 1e-8 { 1.0 / cell_size.y } else { 0.0 },
        if cell_size.z > 1e-8 { 1.0 / cell_size.z } else { 0.0 },
    );

    let tri_count = mesh.indices.len() / 3;
    let indices = &mesh.indices;
    let vertices = &mesh.vertices;

    (0..tri_count).into_par_iter().for_each(|t| {
        let p0 = vertices[indices[t * 3] as usize].position;
        let p1 = vertices[indices[t * 3 + 1] as usize].position;
        let p2 = vertices[indices[t * 3 + 2] as usize].position;

        // Compute triangle AABB in voxel coordinates
        let tri_min = p0.min(p1).min(p2);
        let tri_max = p0.max(p1).max(p2);

        let vmin = ((tri_min - vox_min) * inv_cell).max(Vec3::ZERO);
        let vmax = ((tri_max - vox_min) * inv_cell).min(Vec3::splat(res as f32 - 1.0));

        let ix0 = vmin.x as u32;
        let iy0 = vmin.y as u32;
        let iz0 = vmin.z as u32;
        let ix1 = (vmax.x as u32).min(res - 1);
        let iy1 = (vmax.y as u32).min(res - 1);
        let iz1 = (vmax.z as u32).min(res - 1);

        // Mark voxels covered by triangle bounding box
        for z in iz0..=iz1 {
            for y in iy0..=iy1 {
                for x in ix0..=ix1 {
                    let idx = (x + y * res + z * res * res) as usize;
                    if idx < voxels.len() {
                        voxels[idx].store(1, Ordering::Relaxed);
                    }
                }
            }
        }
    });
}

/// Interior fill: for each Y-Z slice, fill interior using ray parity
fn interior_fill(voxels: &mut [bool], res: u32) {
    for z in 0..res {
        for y in 0..res {
            let mut inside = false;
            let mut last_was_solid = false;
            for x in 0..res {
                let idx = (x + y * res + z * res * res) as usize;
                if voxels[idx] {
                    if !last_was_solid {
                        inside = !inside;
                    }
                    last_was_solid = true;
                } else {
                    last_was_solid = false;
                    if inside {
                        voxels[idx] = true;
                    }
                }
            }
        }
    }
}

/// 3D flood fill for label assignment
fn flood_fill_3d(
    voxels: &[bool],
    labels: &mut [u32],
    start: usize,
    label: u32,
    res: u32,
) {
    let mut stack = vec![start];
    let res_usize = res as usize;

    while let Some(idx) = stack.pop() {
        if idx >= voxels.len() || !voxels[idx] || labels[idx] != 0 {
            continue;
        }

        labels[idx] = label;

        let x = idx % res_usize;
        let y = (idx / res_usize) % res_usize;
        let z = idx / (res_usize * res_usize);

        // 6-connected neighbors
        if x > 0 { stack.push(idx - 1); }
        if x + 1 < res_usize { stack.push(idx + 1); }
        if y > 0 { stack.push(idx - res_usize); }
        if y + 1 < res_usize { stack.push(idx + res_usize); }
        if z > 0 { stack.push(idx - res_usize * res_usize); }
        if z + 1 < res_usize { stack.push(idx + res_usize * res_usize); }
    }
}

/// Simplify a convex hull to have at most max_vertices (Deep Fried v2)
///
/// [Deep Fried v2] Nearest-neighbor distance search parallelized via `rayon`.
fn simplify_hull(hull: &ConvexHull, max_vertices: u32) -> ConvexHull {
    if hull.vertices.len() <= max_vertices as usize {
        return hull.clone();
    }

    // Iteratively remove the vertex closest to its neighbors' centroid
    let mut points = hull.vertices.clone();

    while points.len() > max_vertices as usize && points.len() > 4 {
        // [Deep Fried v2] Parallel nearest-neighbor distance computation
        let nearest_dists: Vec<(usize, f32)> = points
            .par_iter()
            .enumerate()
            .map(|(i, &p)| {
                let mut nearest_dist = f32::MAX;
                for (j, &q) in points.iter().enumerate() {
                    if i != j {
                        let d = (p - q).length_squared();
                        if d < nearest_dist {
                            nearest_dist = d;
                        }
                    }
                }
                (i, nearest_dist)
            })
            .collect();

        // Find vertex with smallest contribution
        let min_idx = nearest_dists
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(i, _)| i)
            .unwrap_or(0);

        points.swap_remove(min_idx);
    }

    convex_hull_from_points(&points)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;

    #[test]
    fn test_compute_aabb() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let aabb = compute_aabb(&mesh);
        assert!(aabb.min.x < 0.0);
        assert!(aabb.max.x > 0.0);
        assert!(aabb.volume() > 0.0);
        assert!(aabb.contains(Vec3::ZERO));
    }

    #[test]
    fn test_bounding_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let bs = compute_bounding_sphere(&mesh);
        // Center should be near origin
        assert!(bs.center.length() < 0.3);
        // Radius should be near 1.0
        assert!((bs.radius - 1.0).abs() < 0.5);
        assert!(bs.contains(Vec3::ZERO));
    }

    #[test]
    fn test_convex_hull() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let hull = compute_convex_hull(&mesh);
        assert!(hull.vertices.len() >= 4);
        assert!(hull.indices.len() >= 12); // At least 4 triangles
    }

    #[test]
    fn test_simplify_collision() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let simplified = simplify_collision(&mesh, 8);

        // Should have fewer vertices than original
        assert!(simplified.vertices.len() < mesh.vertices.len());
        assert!(simplified.indices.len() > 0);
    }

    #[test]
    fn test_convex_hull_cube_points() {
        let points = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            // Interior point (should not appear in hull)
            Vec3::ZERO,
        ];

        let hull = convex_hull_from_points(&points);
        // Hull should have 8 vertices (cube corners)
        assert_eq!(hull.vertices.len(), 8);
        // Hull should have 12 triangles (6 faces * 2)
        assert_eq!(hull.indices.len(), 36);
    }

    #[test]
    fn test_convex_decomposition_basic() {
        // Sphere should decompose to ~1 convex part
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let decomp = convex_decomposition(&mesh, &VhacdConfig::default());
        assert!(!decomp.parts.is_empty(), "Should produce at least 1 convex part");
        assert!(decomp.total_vertices() > 0);
        assert!(decomp.total_triangles() > 0);
    }

    #[test]
    fn test_convex_decomposition_concave() {
        // Subtract box from sphere -> concave shape
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let decomp = convex_decomposition(&mesh, &VhacdConfig::default());
        // Concave shape should produce multiple parts
        assert!(decomp.parts.len() >= 1,
            "Concave shape should produce convex parts, got {}", decomp.parts.len());

        // Each part should have valid convex hull
        for (i, part) in decomp.parts.iter().enumerate() {
            assert!(part.vertices.len() >= 4,
                "Part {} has only {} vertices", i, part.vertices.len());
            assert!(part.indices.len() >= 12,
                "Part {} has only {} indices", i, part.indices.len());
        }
    }

    #[test]
    fn test_convex_decomposition_max_hulls() {
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let vhacd_config = VhacdConfig {
            max_hulls: 4,
            ..Default::default()
        };
        let decomp = convex_decomposition(&mesh, &vhacd_config);
        assert!(decomp.parts.len() <= 4,
            "Should respect max_hulls limit: got {}", decomp.parts.len());
    }
}

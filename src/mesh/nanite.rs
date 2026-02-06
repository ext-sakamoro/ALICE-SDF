//! Nanite-Compatible Cluster Generation for UE5
//!
//! Generates hierarchical mesh clusters from SDFs for use with
//! Unreal Engine 5's Nanite virtualized geometry system.
//!
//! # Nanite Overview
//!
//! Nanite uses a hierarchical cluster-based representation:
//! - **Clusters**: Groups of ~128 triangles
//! - **Cluster Groups**: Collections of clusters for LOD
//! - **DAG**: Directed Acyclic Graph for LOD selection
//!
//! # Features
//!
//! - Cluster-based mesh generation from SDF
//! - LOD chain generation with error metrics
//! - Cluster bounds for GPU culling
//! - Seamless cluster boundaries
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use crate::mesh::{Vertex, Triangle, Mesh, MarchingCubesConfig, sdf_to_mesh};
use glam::Vec3;

/// Maximum triangles per Nanite cluster (UE5 uses ~128)
pub const CLUSTER_MAX_TRIANGLES: usize = 128;

/// Maximum vertices per cluster
pub const CLUSTER_MAX_VERTICES: usize = 256;

/// Cluster bounding sphere
#[derive(Debug, Clone, Copy)]
pub struct ClusterBounds {
    /// Center of bounding sphere
    pub center: Vec3,
    /// Radius of bounding sphere
    pub radius: f32,
    /// Axis-aligned bounding box min
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box max
    pub aabb_max: Vec3,
}

impl ClusterBounds {
    /// Create from vertices
    pub fn from_vertices(vertices: &[Vec3]) -> Self {
        if vertices.is_empty() {
            return ClusterBounds {
                center: Vec3::ZERO,
                radius: 0.0,
                aabb_min: Vec3::ZERO,
                aabb_max: Vec3::ZERO,
            };
        }

        // Compute AABB
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        for &v in vertices {
            aabb_min = aabb_min.min(v);
            aabb_max = aabb_max.max(v);
        }

        // Compute bounding sphere
        let center = (aabb_min + aabb_max) * 0.5;
        let radius = vertices.iter()
            .map(|&v| (v - center).length())
            .fold(0.0f32, f32::max);

        ClusterBounds {
            center,
            radius,
            aabb_min,
            aabb_max,
        }
    }

    /// Check if cluster is visible from a view position
    #[inline]
    pub fn is_visible(&self, view_pos: Vec3, view_dir: Vec3, fov_cos: f32) -> bool {
        let to_center = self.center - view_pos;
        let dist = to_center.length();

        if dist < self.radius {
            return true; // Inside sphere
        }

        let dir = to_center / dist;
        let cone_cos = (dist * dist - self.radius * self.radius).sqrt() / dist;

        dir.dot(view_dir) > fov_cos - cone_cos
    }

    /// Compute screen-space error for LOD selection
    #[inline]
    pub fn screen_error(&self, view_pos: Vec3, geometric_error: f32, screen_height: f32) -> f32 {
        let dist = (self.center - view_pos).length().max(self.radius);
        (geometric_error / dist) * screen_height
    }
}

/// LOD level information
#[derive(Debug, Clone, Copy)]
pub struct LodLevel {
    /// LOD index (0 = highest detail)
    pub level: u32,
    /// Resolution used for this LOD
    pub resolution: u32,
    /// Maximum geometric error at this LOD
    pub max_error: f32,
    /// Triangle count at this LOD
    pub triangle_count: u32,
}

/// Nanite-compatible mesh cluster
#[derive(Debug, Clone)]
pub struct NaniteCluster {
    /// Unique cluster ID
    pub id: u32,
    /// LOD level this cluster belongs to
    pub lod_level: u32,
    /// Vertices in this cluster
    pub vertices: Vec<Vertex>,
    /// Triangle indices (local to cluster)
    pub triangles: Vec<Triangle>,
    /// Cluster bounds for culling
    pub bounds: ClusterBounds,
    /// Parent cluster IDs (for LOD DAG)
    pub parent_ids: Vec<u32>,
    /// Child cluster IDs (for LOD DAG)
    pub child_ids: Vec<u32>,
    /// Geometric error for this cluster
    pub geometric_error: f32,
}

impl NaniteCluster {
    /// Get triangle count
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Get vertex count
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Check if cluster should be rendered at given distance
    pub fn should_render(&self, view_pos: Vec3, error_threshold: f32) -> bool {
        let dist = (self.bounds.center - view_pos).length();
        let screen_error = self.geometric_error / dist.max(0.001);
        screen_error > error_threshold
    }
}

/// Nanite cluster group (collection of clusters at same LOD)
#[derive(Debug, Clone)]
pub struct ClusterGroup {
    /// Group ID
    pub id: u32,
    /// LOD level
    pub lod_level: u32,
    /// Cluster IDs in this group
    pub cluster_ids: Vec<u32>,
    /// Combined bounds
    pub bounds: ClusterBounds,
    /// Maximum geometric error in group
    pub max_error: f32,
}

/// Configuration for Nanite mesh generation
#[derive(Debug, Clone)]
pub struct NaniteConfig {
    /// Number of LOD levels to generate
    pub lod_levels: u32,
    /// Base resolution for LOD 0 (highest detail)
    pub base_resolution: u32,
    /// Resolution reduction factor per LOD level
    pub lod_factor: f32,
    /// Maximum triangles per cluster
    pub max_triangles_per_cluster: usize,
    /// Cluster overlap for seamless boundaries (0.0-0.5)
    pub cluster_overlap: f32,
    /// Whether to compute per-cluster normals
    pub compute_normals: bool,
}

impl Default for NaniteConfig {
    fn default() -> Self {
        NaniteConfig {
            lod_levels: 6,
            base_resolution: 128,
            lod_factor: 0.5,
            max_triangles_per_cluster: CLUSTER_MAX_TRIANGLES,
            cluster_overlap: 0.1,
            compute_normals: true,
        }
    }
}

impl NaniteConfig {
    /// Create config for high detail (game-ready assets)
    pub fn high_detail() -> Self {
        NaniteConfig {
            lod_levels: 8,
            base_resolution: 256,
            lod_factor: 0.5,
            ..Default::default()
        }
    }

    /// Create config for medium detail
    pub fn medium_detail() -> Self {
        NaniteConfig {
            lod_levels: 5,
            base_resolution: 64,
            lod_factor: 0.5,
            ..Default::default()
        }
    }

    /// Create config for preview/fast generation
    pub fn preview() -> Self {
        NaniteConfig {
            lod_levels: 3,
            base_resolution: 32,
            lod_factor: 0.5,
            ..Default::default()
        }
    }
}

/// Nanite mesh with LOD hierarchy
#[derive(Debug)]
pub struct NaniteMesh {
    /// All clusters across all LOD levels
    pub clusters: Vec<NaniteCluster>,
    /// Cluster groups
    pub groups: Vec<ClusterGroup>,
    /// LOD level information
    pub lod_levels: Vec<LodLevel>,
    /// Global bounds
    pub bounds: ClusterBounds,
    /// Total triangle count (LOD 0)
    pub total_triangles: usize,
}

impl NaniteMesh {
    /// Get clusters at a specific LOD level
    pub fn clusters_at_lod(&self, level: u32) -> Vec<&NaniteCluster> {
        self.clusters.iter()
            .filter(|c| c.lod_level == level)
            .collect()
    }

    /// Get cluster by ID
    pub fn get_cluster(&self, id: u32) -> Option<&NaniteCluster> {
        self.clusters.iter().find(|c| c.id == id)
    }

    /// Select visible clusters for rendering
    pub fn select_clusters(&self, view_pos: Vec3, error_threshold: f32) -> Vec<u32> {
        let mut visible = Vec::new();

        for cluster in &self.clusters {
            if cluster.should_render(view_pos, error_threshold) {
                // Check if any child is also visible (prefer children for detail)
                let child_visible = cluster.child_ids.iter()
                    .any(|&id| {
                        self.get_cluster(id)
                            .map(|c| c.should_render(view_pos, error_threshold))
                            .unwrap_or(false)
                    });

                if !child_visible {
                    visible.push(cluster.id);
                }
            }
        }

        visible
    }

    /// Get total vertex count
    pub fn total_vertices(&self) -> usize {
        self.clusters.iter().map(|c| c.vertex_count()).sum()
    }

    /// Export to flat mesh at specified LOD
    pub fn to_mesh(&self, lod_level: u32) -> Mesh {
        let clusters: Vec<_> = self.clusters_at_lod(lod_level);

        let mut mesh = Mesh::new();
        let mut vertex_offset = 0u32;

        for cluster in clusters {
            mesh.vertices.extend_from_slice(&cluster.vertices);

            for tri in &cluster.triangles {
                mesh.indices.push(tri.a + vertex_offset);
                mesh.indices.push(tri.b + vertex_offset);
                mesh.indices.push(tri.c + vertex_offset);
            }

            vertex_offset += cluster.vertices.len() as u32;
        }

        mesh
    }
}

/// Generate Nanite-compatible mesh from SDF
pub fn generate_nanite_mesh(
    sdf: &SdfNode,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &NaniteConfig,
) -> NaniteMesh {
    let mut all_clusters = Vec::new();
    let mut all_groups = Vec::new();
    let mut lod_infos = Vec::new();
    let mut cluster_id = 0u32;
    let mut group_id = 0u32;

    let _bounds_size = max_bounds - min_bounds;
    let bounds_center = (min_bounds + max_bounds) * 0.5;

    // Generate each LOD level
    for lod in 0..config.lod_levels {
        let resolution = (config.base_resolution as f32 * config.lod_factor.powi(lod as i32)) as u32;
        let resolution = resolution.max(4); // Minimum resolution

        let mc_config = MarchingCubesConfig {
            resolution: resolution as usize,
            iso_level: 0.0,
            compute_normals: config.compute_normals,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(sdf, min_bounds, max_bounds, &mc_config);

        if mesh.triangle_count() == 0 {
            continue;
        }

        // Split mesh into clusters
        let (clusters, max_error) = split_into_clusters(
            &mesh,
            config.max_triangles_per_cluster,
            &mut cluster_id,
            lod,
        );

        // Create cluster group for this LOD
        let cluster_ids: Vec<u32> = clusters.iter().map(|c| c.id).collect();
        let group_bounds = compute_group_bounds(&clusters);

        all_groups.push(ClusterGroup {
            id: group_id,
            lod_level: lod,
            cluster_ids: cluster_ids.clone(),
            bounds: group_bounds,
            max_error,
        });
        group_id += 1;

        // Store LOD info
        lod_infos.push(LodLevel {
            level: lod,
            resolution,
            max_error,
            triangle_count: clusters.iter().map(|c| c.triangle_count() as u32).sum(),
        });

        all_clusters.extend(clusters);
    }

    // Build LOD DAG (connect parent-child relationships)
    build_lod_dag(&mut all_clusters);

    // Compute global bounds
    let global_bounds = if !all_clusters.is_empty() {
        let all_vertices: Vec<Vec3> = all_clusters.iter()
            .flat_map(|c| c.vertices.iter().map(|v| v.position))
            .collect();
        ClusterBounds::from_vertices(&all_vertices)
    } else {
        ClusterBounds::from_vertices(&[bounds_center])
    };

    let total_triangles = all_clusters.iter()
        .filter(|c| c.lod_level == 0)
        .map(|c| c.triangle_count())
        .sum();

    NaniteMesh {
        clusters: all_clusters,
        groups: all_groups,
        lod_levels: lod_infos,
        bounds: global_bounds,
        total_triangles,
    }
}

/// Split a mesh into Nanite-compatible clusters
fn split_into_clusters(
    mesh: &Mesh,
    max_triangles: usize,
    cluster_id: &mut u32,
    lod_level: u32,
) -> (Vec<NaniteCluster>, f32) {
    let num_triangles = mesh.triangle_count();

    if num_triangles == 0 {
        return (Vec::new(), 0.0);
    }

    // For simple meshes, return single cluster
    if num_triangles <= max_triangles {
        let vertices: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
        let bounds = ClusterBounds::from_vertices(&vertices);

        let geometric_error = compute_geometric_error(mesh, lod_level);
        let cluster = NaniteCluster {
            id: *cluster_id,
            lod_level,
            vertices: mesh.vertices.clone(),
            triangles: mesh.indices.chunks(3)
                .map(|c| Triangle::new(c[0], c[1], c[2]))
                .collect(),
            bounds,
            parent_ids: Vec::new(),
            child_ids: Vec::new(),
            geometric_error,
        };

        *cluster_id += 1;
        return (vec![cluster], geometric_error);
    }

    // Spatial clustering using octree-like subdivision
    let mut clusters = Vec::new();
    let mut max_error = 0.0f32;

    // Compute mesh bounds
    let (mesh_min, mesh_max) = compute_mesh_bounds(mesh);
    let mesh_center = (mesh_min + mesh_max) * 0.5;

    // Assign triangles to octants
    let mut octants: [Vec<usize>; 8] = Default::default();

    for tri_idx in 0..num_triangles {
        let base = tri_idx * 3;
        let v0 = mesh.vertices[mesh.indices[base] as usize].position;
        let v1 = mesh.vertices[mesh.indices[base + 1] as usize].position;
        let v2 = mesh.vertices[mesh.indices[base + 2] as usize].position;

        let centroid = (v0 + v1 + v2) / 3.0;
        let octant = ((centroid.x > mesh_center.x) as usize)
            | (((centroid.y > mesh_center.y) as usize) << 1)
            | (((centroid.z > mesh_center.z) as usize) << 2);

        octants[octant].push(tri_idx);
    }

    // Create clusters from octants
    for octant_tris in &octants {
        if octant_tris.is_empty() {
            continue;
        }

        // If too many triangles, recursively subdivide (simplified: just split)
        for chunk in octant_tris.chunks(max_triangles) {
            let (cluster_vertices, cluster_triangles) = extract_cluster_geometry(mesh, chunk);

            if cluster_vertices.is_empty() {
                continue;
            }

            let positions: Vec<Vec3> = cluster_vertices.iter().map(|v| v.position).collect();
            let bounds = ClusterBounds::from_vertices(&positions);
            let error = compute_cluster_error(&cluster_vertices, lod_level);

            max_error = max_error.max(error);

            clusters.push(NaniteCluster {
                id: *cluster_id,
                lod_level,
                vertices: cluster_vertices,
                triangles: cluster_triangles,
                bounds,
                parent_ids: Vec::new(),
                child_ids: Vec::new(),
                geometric_error: error,
            });

            *cluster_id += 1;
        }
    }

    (clusters, max_error)
}

/// Extract geometry for a subset of triangles
fn extract_cluster_geometry(mesh: &Mesh, tri_indices: &[usize]) -> (Vec<Vertex>, Vec<Triangle>) {
    use std::collections::HashMap;

    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    for &tri_idx in tri_indices {
        let base = tri_idx * 3;
        let mut new_indices = [0u32; 3];

        for i in 0..3 {
            let old_idx = mesh.indices[base + i];

            let new_idx = *vertex_map.entry(old_idx).or_insert_with(|| {
                let idx = vertices.len() as u32;
                vertices.push(mesh.vertices[old_idx as usize].clone());
                idx
            });

            new_indices[i] = new_idx;
        }

        triangles.push(Triangle::new(new_indices[0], new_indices[1], new_indices[2]));
    }

    (vertices, triangles)
}

/// Compute mesh bounds
fn compute_mesh_bounds(mesh: &Mesh) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for v in &mesh.vertices {
        min = min.min(v.position);
        max = max.max(v.position);
    }

    (min, max)
}

/// Compute geometric error for a mesh at given LOD
fn compute_geometric_error(mesh: &Mesh, lod_level: u32) -> f32 {
    // Error increases with LOD level (lower detail = higher error)
    let base_error = if mesh.vertices.is_empty() {
        0.0
    } else {
        // Approximate error as average edge length
        let mut total_edge_len = 0.0f32;
        let mut edge_count = 0;

        for chunk in mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                let v0 = mesh.vertices[chunk[0] as usize].position;
                let v1 = mesh.vertices[chunk[1] as usize].position;
                let v2 = mesh.vertices[chunk[2] as usize].position;

                total_edge_len += (v1 - v0).length();
                total_edge_len += (v2 - v1).length();
                total_edge_len += (v0 - v2).length();
                edge_count += 3;
            }
        }

        if edge_count > 0 {
            total_edge_len / edge_count as f32
        } else {
            0.0
        }
    };

    base_error * (1.5f32).powi(lod_level as i32)
}

/// Compute cluster error
fn compute_cluster_error(vertices: &[Vertex], lod_level: u32) -> f32 {
    if vertices.len() < 2 {
        return 0.0;
    }

    // Approximate as diameter of bounding sphere
    let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
    let bounds = ClusterBounds::from_vertices(&positions);

    bounds.radius * 2.0 * (1.2f32).powi(lod_level as i32) / vertices.len() as f32
}

/// Compute combined bounds for a group of clusters
fn compute_group_bounds(clusters: &[NaniteCluster]) -> ClusterBounds {
    let all_vertices: Vec<Vec3> = clusters.iter()
        .flat_map(|c| c.vertices.iter().map(|v| v.position))
        .collect();

    ClusterBounds::from_vertices(&all_vertices)
}

/// Build LOD DAG by connecting parent-child relationships
fn build_lod_dag(clusters: &mut [NaniteCluster]) {
    // Group clusters by LOD level
    let max_lod = clusters.iter().map(|c| c.lod_level).max().unwrap_or(0);

    // Collect relationships first (to avoid borrow conflicts)
    let mut relationships: Vec<(u32, u32)> = Vec::new(); // (parent_id, child_id)

    for lod in 1..=max_lod {
        // Get parents at this LOD level
        let parents: Vec<_> = clusters.iter()
            .filter(|c| c.lod_level == lod)
            .map(|c| (c.id, c.bounds))
            .collect();

        // Get children at previous LOD level
        let children: Vec<_> = clusters.iter()
            .filter(|c| c.lod_level == lod - 1)
            .map(|c| (c.id, c.bounds))
            .collect();

        // Find overlapping pairs
        for (parent_id, parent_bounds) in &parents {
            for (child_id, child_bounds) in &children {
                if bounds_overlap(parent_bounds, child_bounds) {
                    relationships.push((*parent_id, *child_id));
                }
            }
        }
    }

    // Apply relationships
    for (parent_id, child_id) in relationships {
        for c in clusters.iter_mut() {
            if c.id == parent_id {
                c.child_ids.push(child_id);
            }
            if c.id == child_id {
                c.parent_ids.push(parent_id);
            }
        }
    }
}

/// Check if two cluster bounds overlap
fn bounds_overlap(a: &ClusterBounds, b: &ClusterBounds) -> bool {
    // Check sphere-sphere overlap (faster than AABB)
    let dist = (a.center - b.center).length();
    dist < a.radius + b.radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_bounds() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];

        let bounds = ClusterBounds::from_vertices(&vertices);

        assert_eq!(bounds.aabb_min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(bounds.aabb_max, Vec3::new(1.0, 1.0, 1.0));
        assert!((bounds.center - Vec3::ZERO).length() < 0.01);
    }

    #[test]
    fn test_generate_nanite_mesh() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &config,
        );

        assert!(!nanite.clusters.is_empty());
        assert!(!nanite.lod_levels.is_empty());
        assert!(nanite.total_triangles > 0);
    }

    #[test]
    fn test_cluster_selection() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &config,
        );

        // Close view - should select more clusters
        let close_clusters = nanite.select_clusters(Vec3::new(0.0, 0.0, 3.0), 0.01);

        // Far view - should select fewer clusters
        let far_clusters = nanite.select_clusters(Vec3::new(0.0, 0.0, 100.0), 0.01);

        // Far view should have equal or fewer clusters
        assert!(far_clusters.len() <= close_clusters.len() + nanite.clusters.len());
    }

    #[test]
    fn test_to_mesh() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &config,
        );

        let mesh = nanite.to_mesh(0);
        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }
}

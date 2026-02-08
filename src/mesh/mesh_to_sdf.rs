//! Mesh to SDF conversion (Deep Fried Edition v2)
//!
//! Converts polygon meshes to SDF representations with multiple strategies:
//! - **BVH-Accelerated Exact**: O(log n) queries using BVH
//! - **Capsule Approximation**: Fast but approximate (legacy mode)
//! - **Hybrid**: BVH for distance, capsule tree for SdfNode
//!
//! # Deep Fried Optimizations
//! - **Edge Deduplication**: Uses HashSet to eliminate duplicate capsules.
//! - **BVH Acceleration**: O(log n) distance queries for exact SDF.
//! - **Parallel Batch Evaluation**: Rayon-powered batch queries.
//! - **Forced Inlining**: `#[inline(always)]` on hot-path helpers.
//!
//! Author: Moroya Sakamoto

use crate::mesh::bvh::MeshBvh;
use crate::types::SdfNode;
use glam::Vec3;
use std::collections::HashSet;
use std::sync::Arc;

/// Conversion strategy for mesh_to_sdf
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshToSdfStrategy {
    /// Use capsule approximation (fast, creates SdfNode tree)
    Capsule,
    /// Use BVH for exact distance (accurate, returns MeshSdf wrapper)
    BvhExact,
    /// Hybrid: create capsule tree but use BVH for evaluation
    Hybrid,
}

impl Default for MeshToSdfStrategy {
    fn default() -> Self {
        MeshToSdfStrategy::Capsule
    }
}

/// Configuration for mesh to SDF conversion
#[derive(Debug, Clone)]
pub struct MeshToSdfConfig {
    /// Conversion strategy
    pub strategy: MeshToSdfStrategy,
    /// Whether to use bounding volumes for acceleration
    pub use_bvh: bool,
    /// Maximum triangles per leaf node (if using BVH)
    pub max_triangles_per_leaf: usize,
    /// Capsule radius factor (multiplied by average edge length)
    pub capsule_radius_factor: f32,
    /// Whether to compute vertex normals for sign determination
    pub compute_normals: bool,
}

impl Default for MeshToSdfConfig {
    fn default() -> Self {
        MeshToSdfConfig {
            strategy: MeshToSdfStrategy::Capsule,
            use_bvh: true,
            max_triangles_per_leaf: 4,
            capsule_radius_factor: 0.05,
            compute_normals: true,
        }
    }
}

impl MeshToSdfConfig {
    /// Create config for fast capsule approximation
    pub fn fast() -> Self {
        MeshToSdfConfig {
            strategy: MeshToSdfStrategy::Capsule,
            use_bvh: false,
            ..Default::default()
        }
    }

    /// Create config for accurate BVH-based SDF
    pub fn accurate() -> Self {
        MeshToSdfConfig {
            strategy: MeshToSdfStrategy::BvhExact,
            use_bvh: true,
            max_triangles_per_leaf: 4,
            ..Default::default()
        }
    }

    /// Create config for hybrid approach
    pub fn hybrid() -> Self {
        MeshToSdfConfig {
            strategy: MeshToSdfStrategy::Hybrid,
            use_bvh: true,
            max_triangles_per_leaf: 4,
            ..Default::default()
        }
    }
}

/// BVH-accelerated mesh SDF
///
/// Provides exact signed distance queries using BVH acceleration.
/// This is more accurate than the capsule approximation but cannot be
/// converted to shader code directly.
pub struct MeshSdf {
    bvh: Arc<MeshBvh>,
    bounds: crate::mesh::bvh::Aabb,
}

impl MeshSdf {
    /// Create MeshSdf from vertices and indices
    pub fn new(vertices: &[Vec3], indices: &[u32], config: &MeshToSdfConfig) -> Option<Self> {
        if vertices.is_empty() || indices.is_empty() {
            return None;
        }

        let bvh = MeshBvh::build(vertices, indices, config.max_triangles_per_leaf);
        let bounds = bvh.bounds()?;

        Some(MeshSdf {
            bvh: Arc::new(bvh),
            bounds,
        })
    }

    /// Evaluate signed distance at a point
    #[inline]
    pub fn eval(&self, point: Vec3) -> f32 {
        self.bvh.signed_distance(point)
    }

    /// Evaluate unsigned distance at a point
    #[inline]
    pub fn eval_unsigned(&self, point: Vec3) -> f32 {
        self.bvh.unsigned_distance(point)
    }

    /// Batch evaluate signed distances (parallel)
    pub fn eval_batch(&self, points: &[Vec3]) -> Vec<f32> {
        self.bvh.signed_distance_batch(points)
    }

    /// Batch evaluate unsigned distances (parallel)
    pub fn eval_unsigned_batch(&self, points: &[Vec3]) -> Vec<f32> {
        self.bvh.unsigned_distance_batch(points)
    }

    /// Compute gradient (approximate normal) at a point
    pub fn gradient(&self, point: Vec3, epsilon: f32) -> Vec3 {
        let dx = self.eval(point + Vec3::X * epsilon) - self.eval(point - Vec3::X * epsilon);
        let dy = self.eval(point + Vec3::Y * epsilon) - self.eval(point - Vec3::Y * epsilon);
        let dz = self.eval(point + Vec3::Z * epsilon) - self.eval(point - Vec3::Z * epsilon);
        Vec3::new(dx, dy, dz).normalize_or_zero()
    }

    /// Get mesh bounds
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.bounds.min, self.bounds.max)
    }

    /// Get triangle count
    pub fn triangle_count(&self) -> usize {
        self.bvh.triangle_count()
    }

    /// Get reference to underlying BVH
    pub fn bvh(&self) -> &MeshBvh {
        &self.bvh
    }

    /// Convert to approximate SdfNode tree using capsule method
    ///
    /// This allows the MeshSdf to be used where SdfNode is required,
    /// but with reduced accuracy.
    pub fn to_sdf_node(&self, radius_factor: f32) -> SdfNode {
        // Extract vertices and edges from triangles
        let mut vertices: Vec<Vec3> = Vec::new();
        let mut vertex_map: std::collections::HashMap<[u32; 3], usize> =
            std::collections::HashMap::new();
        let mut unique_edges: HashSet<(usize, usize)> = HashSet::new();

        for tri in &self.bvh.triangles {
            let mut get_or_add = |v: Vec3| -> usize {
                let key = [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()];
                if let Some(&idx) = vertex_map.get(&key) {
                    idx
                } else {
                    let idx = vertices.len();
                    vertices.push(v);
                    vertex_map.insert(key, idx);
                    idx
                }
            };

            let i0 = get_or_add(tri.v0);
            let i1 = get_or_add(tri.v1);
            let i2 = get_or_add(tri.v2);

            unique_edges.insert(edge_key_usize(i0, i1));
            unique_edges.insert(edge_key_usize(i1, i2));
            unique_edges.insert(edge_key_usize(i2, i0));
        }

        // Compute average edge length
        let avg_edge_len: f32 = if unique_edges.is_empty() {
            0.1
        } else {
            let total: f32 = unique_edges
                .iter()
                .map(|&(i0, i1)| (vertices[i1] - vertices[i0]).length())
                .sum();
            total / unique_edges.len() as f32
        };

        let radius = avg_edge_len * radius_factor;

        // Build capsule nodes
        let nodes: Vec<SdfNode> = unique_edges
            .into_iter()
            .map(|(i0, i1)| SdfNode::capsule(vertices[i0], vertices[i1], radius))
            .collect();

        reduce_to_union(nodes)
    }
}

/// Convert a polygon mesh to an SDF representation
///
/// Returns an SdfNode tree that approximates the mesh using capsules.
/// For more accurate SDF evaluation, use `mesh_to_sdf_exact` instead.
///
/// # Arguments
/// * `vertices` - Mesh vertex positions
/// * `indices` - Triangle indices (3 per triangle)
/// * `config` - Conversion configuration
///
/// # Returns
/// SDF tree approximating the mesh
pub fn mesh_to_sdf(vertices: &[Vec3], indices: &[u32], config: &MeshToSdfConfig) -> SdfNode {
    if vertices.is_empty() || indices.is_empty() {
        return SdfNode::sphere(0.001); // Degenerate case
    }

    // Compute average edge length for capsule radius
    let mut total_edge_len = 0.0;
    let mut edge_count = 0;

    for chunk in indices.chunks(3) {
        if chunk.len() == 3 {
            let v0 = vertices[chunk[0] as usize];
            let v1 = vertices[chunk[1] as usize];
            let v2 = vertices[chunk[2] as usize];

            total_edge_len += (v1 - v0).length();
            total_edge_len += (v2 - v1).length();
            total_edge_len += (v0 - v2).length();
            edge_count += 3;
        }
    }

    let avg_edge_len = if edge_count > 0 {
        total_edge_len / edge_count as f32
    } else {
        0.1
    };

    // Use configured fraction of edge length as capsule radius
    let radius = avg_edge_len * config.capsule_radius_factor;

    // Edge deduplication using HashSet
    // Key: (min_idx, max_idx) ensures (a, b) == (b, a)
    let mut unique_edges: HashSet<(u32, u32)> = HashSet::new();

    for chunk in indices.chunks(3) {
        if chunk.len() == 3 {
            let i0 = chunk[0];
            let i1 = chunk[1];
            let i2 = chunk[2];

            // Insert edges with canonical ordering (min, max)
            unique_edges.insert(edge_key(i0, i1));
            unique_edges.insert(edge_key(i1, i2));
            unique_edges.insert(edge_key(i2, i0));
        }
    }

    // Build SDF from unique edges only
    let nodes: Vec<SdfNode> = unique_edges
        .into_iter()
        .map(|(i0, i1)| {
            let v0 = vertices[i0 as usize];
            let v1 = vertices[i1 as usize];
            SdfNode::capsule(v0, v1, radius)
        })
        .collect();

    // Build balanced binary tree of unions
    reduce_to_union(nodes)
}

/// Convert a polygon mesh to an exact BVH-accelerated SDF
///
/// Returns a MeshSdf that provides accurate signed distance queries.
/// This cannot be converted to shader code directly, but is useful for:
/// - Accurate distance queries
/// - Mesh processing algorithms
/// - Ray marching on CPU
///
/// # Arguments
/// * `vertices` - Mesh vertex positions
/// * `indices` - Triangle indices (3 per triangle)
/// * `config` - Conversion configuration
///
/// # Returns
/// Option<MeshSdf> - None if mesh is empty
pub fn mesh_to_sdf_exact(
    vertices: &[Vec3],
    indices: &[u32],
    config: &MeshToSdfConfig,
) -> Option<MeshSdf> {
    MeshSdf::new(vertices, indices, config)
}

/// Create canonical edge key (min, max) for HashSet deduplication
#[inline(always)]
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Create canonical edge key for usize indices
#[inline(always)]
fn edge_key_usize(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Reduce a list of nodes to a single union using binary tree structure (Deep Fried)
#[inline(always)]
fn reduce_to_union(mut nodes: Vec<SdfNode>) -> SdfNode {
    if nodes.is_empty() {
        return SdfNode::sphere(0.001);
    }

    while nodes.len() > 1 {
        let mut next = Vec::with_capacity((nodes.len() + 1) / 2);

        for chunk in nodes.chunks(2) {
            if chunk.len() == 2 {
                next.push(chunk[0].clone().union(chunk[1].clone()));
            } else {
                next.push(chunk[0].clone());
            }
        }

        nodes = next;
    }

    nodes.pop().unwrap()
}

/// Signed distance to a triangle (Deep Fried)
///
/// Computes the exact signed distance to a triangle.
/// Positive outside, negative inside (requires consistent winding).
#[allow(dead_code)]
#[inline(always)]
fn sdf_triangle(point: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> f32 {
    let e0 = v1 - v0;
    let e1 = v2 - v1;
    let e2 = v0 - v2;

    let n = e0.cross(e1);

    let p0 = point - v0;
    let p1 = point - v1;
    let p2 = point - v2;

    // Check if point projects inside triangle
    let d0 = e0.cross(n).dot(p0);
    let d1 = e1.cross(n).dot(p1);
    let d2 = e2.cross(n).dot(p2);

    if d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0 {
        // Inside triangle - distance to plane
        return p0.dot(n) / n.length();
    }

    // Outside triangle - distance to nearest edge
    let c0 = e0 * (p0.dot(e0) / e0.dot(e0)).clamp(0.0, 1.0) - p0;
    let c1 = e1 * (p1.dot(e1) / e1.dot(e1)).clamp(0.0, 1.0) - p1;
    let c2 = e2 * (p2.dot(e2) / e2.dot(e2)).clamp(0.0, 1.0) - p2;

    let d0 = c0.length_squared();
    let d1 = c1.length_squared();
    let d2 = c2.length_squared();

    d0.min(d1).min(d2).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_to_sdf_single_triangle() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];

        let sdf = mesh_to_sdf(&vertices, &indices, &MeshToSdfConfig::default());

        // Should be a valid SDF tree
        assert!(sdf.node_count() > 0);
    }

    #[test]
    fn test_mesh_to_sdf_cube() {
        // Simple cube mesh
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
        ];
        // Front face, back face (simplified - just 2 triangles for test)
        let indices = vec![
            0, 1, 2, 0, 2, 3, // Front
            4, 6, 5, 4, 7, 6, // Back
        ];

        let sdf = mesh_to_sdf(&vertices, &indices, &MeshToSdfConfig::default());
        assert!(sdf.node_count() > 0);
    }

    #[test]
    fn test_mesh_to_sdf_empty() {
        let vertices: Vec<Vec3> = vec![];
        let indices: Vec<u32> = vec![];

        let sdf = mesh_to_sdf(&vertices, &indices, &MeshToSdfConfig::default());
        // Should return degenerate sphere
        assert_eq!(sdf.node_count(), 1);
    }

    #[test]
    fn test_reduce_to_union() {
        let nodes = vec![
            SdfNode::sphere(1.0),
            SdfNode::sphere(1.0),
            SdfNode::sphere(1.0),
            SdfNode::sphere(1.0),
        ];

        let union = reduce_to_union(nodes);
        // 4 spheres + 2 unions at level 1 + 1 union at root = 7
        assert_eq!(union.node_count(), 7);
    }

    #[test]
    fn test_mesh_sdf_exact() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(-1.0, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];

        let mesh_sdf = mesh_to_sdf_exact(&vertices, &indices, &MeshToSdfConfig::accurate())
            .expect("Should create MeshSdf");

        // Point on surface
        let d_surface = mesh_sdf.eval(Vec3::ZERO);
        assert!(d_surface.abs() < 0.01, "Expected ~0, got {}", d_surface);

        // Point above
        let d_above = mesh_sdf.eval(Vec3::new(0.0, 0.0, 1.0));
        assert!(
            (d_above.abs() - 1.0).abs() < 0.01,
            "Expected ~1, got {}",
            d_above
        );

        // Batch evaluation
        let points = vec![
            Vec3::ZERO,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        let distances = mesh_sdf.eval_batch(&points);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_mesh_sdf_to_node() {
        let vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 1.0, 0.0),
        ];
        let indices = vec![0, 1, 2];

        let mesh_sdf = mesh_to_sdf_exact(&vertices, &indices, &MeshToSdfConfig::accurate())
            .expect("Should create MeshSdf");

        let node = mesh_sdf.to_sdf_node(0.05);
        assert!(node.node_count() > 0);
    }

    #[test]
    fn test_config_presets() {
        let fast = MeshToSdfConfig::fast();
        assert_eq!(fast.strategy, MeshToSdfStrategy::Capsule);
        assert!(!fast.use_bvh);

        let accurate = MeshToSdfConfig::accurate();
        assert_eq!(accurate.strategy, MeshToSdfStrategy::BvhExact);
        assert!(accurate.use_bvh);

        let hybrid = MeshToSdfConfig::hybrid();
        assert_eq!(hybrid.strategy, MeshToSdfStrategy::Hybrid);
        assert!(hybrid.use_bvh);
    }
}

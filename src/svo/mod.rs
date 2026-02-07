//! Sparse Voxel Octree (Deep Fried Edition)
//!
//! Hierarchical spatial structure for large SDF scenes. GPU-friendly,
//! supports streaming, and enables O(log N) point/ray queries.
//!
//! # Architecture
//!
//! - **SvoNode**: 32-byte aligned node with distance, normal, child_mask
//! - **Linearized**: Breadth-first flat array for GPU-friendly traversal
//! - **Adaptive**: Subdivide near surface, leaf when far from surface
//! - **Streaming**: Chunk-based loading with LRU cache
//!
//! # Usage
//!
//! ```rust,ignore
//! use alice_sdf::prelude::*;
//! use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};
//!
//! let shape = SdfNode::sphere(1.0)
//!     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
//!
//! let config = SvoBuildConfig::default();
//! let svo = SparseVoxelOctree::build(&shape, &config);
//!
//! // Point query
//! let dist = svo.query_point(Vec3::new(0.5, 0.0, 0.0));
//!
//! // Ray query
//! let hit = svo.ray_query(Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::X), 10.0);
//! ```
//!
//! Author: Moroya Sakamoto

pub mod build;
pub mod query;
pub mod linearize;
pub mod streaming;

use glam::Vec3;
use crate::compiled::AabbPacked;

pub use build::{build_svo, build_svo_compiled};
pub use query::{svo_query_point, svo_ray_query, svo_nearest_surface, SvoRayHit};
pub use linearize::{linearize_svo, LinearizedSvo};
pub use streaming::{SvoStreamingCache, SvoChunk};

/// Sparse Voxel Octree node (32 bytes, cache-aligned)
///
/// Each node represents an octant of 3D space. Interior nodes have children;
/// leaf nodes store the SDF distance and normal at their center.
///
/// Navigation: child index = `first_child + popcount(child_mask & ((1 << octant) - 1))`
#[repr(C, align(32))]
#[derive(Copy, Clone, Debug)]
pub struct SvoNode {
    /// Signed distance at node center
    pub distance: f32,
    /// Surface normal X (valid near surface)
    pub nx: f32,
    /// Surface normal Y
    pub ny: f32,
    /// Surface normal Z
    pub nz: f32,
    /// Bitmask of which octant children exist (8 bits used)
    pub child_mask: u8,
    /// Whether this node is a leaf (1) or interior (0)
    pub is_leaf: u8,
    /// Material ID for this region
    pub material_id: u16,
    /// Index of the first child in the node array
    pub first_child: u32,
    /// Padding for 32-byte alignment
    pub _pad: [f32; 2],
}

impl Default for SvoNode {
    fn default() -> Self {
        SvoNode {
            distance: f32::MAX,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            child_mask: 0,
            is_leaf: 1,
            material_id: 0,
            first_child: 0,
            _pad: [0.0; 2],
        }
    }
}

impl SvoNode {
    /// Create a leaf node with distance and normal
    #[inline]
    pub fn leaf(distance: f32, normal: Vec3) -> Self {
        SvoNode {
            distance,
            nx: normal.x,
            ny: normal.y,
            nz: normal.z,
            child_mask: 0,
            is_leaf: 1,
            material_id: 0,
            first_child: 0,
            _pad: [0.0; 2],
        }
    }

    /// Create an interior node
    #[inline]
    pub fn interior(distance: f32, child_mask: u8, first_child: u32) -> Self {
        SvoNode {
            distance,
            nx: 0.0,
            ny: 0.0,
            nz: 0.0,
            child_mask,
            is_leaf: 0,
            material_id: 0,
            first_child,
            _pad: [0.0; 2],
        }
    }

    /// Get the normal as a Vec3
    #[inline]
    pub fn normal(&self) -> Vec3 {
        Vec3::new(self.nx, self.ny, self.nz)
    }

    /// Number of active children
    #[inline]
    pub fn child_count(&self) -> u32 {
        self.child_mask.count_ones()
    }

    /// Get the index of a specific octant's child in the node array
    ///
    /// Returns `None` if the octant has no child.
    #[inline]
    pub fn child_index(&self, octant: u8) -> Option<u32> {
        if self.child_mask & (1 << octant) == 0 {
            return None;
        }
        let bits_before = self.child_mask & ((1 << octant) - 1);
        Some(self.first_child + bits_before.count_ones())
    }
}

/// Configuration for SVO construction
#[derive(Debug, Clone)]
pub struct SvoBuildConfig {
    /// Maximum octree depth (10 = 1024^3 effective resolution)
    pub max_depth: u32,
    /// Stop subdividing if |distance| > threshold * node_size
    pub distance_threshold: f32,
    /// Surface detail level (lower = more nodes near surface)
    pub surface_detail: f32,
    /// Use compiled evaluator for faster build
    pub use_compiled: bool,
    /// World-space bounds of the SVO
    pub bounds_min: Vec3,
    /// World-space bounds of the SVO
    pub bounds_max: Vec3,
}

impl Default for SvoBuildConfig {
    fn default() -> Self {
        SvoBuildConfig {
            max_depth: 8,
            distance_threshold: 1.5,
            surface_detail: 1.0,
            use_compiled: true,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
        }
    }
}

/// Sparse Voxel Octree
///
/// Stores an SDF as a hierarchical octree with adaptive resolution.
/// Interior nodes near the surface are subdivided; far regions are stored
/// as large leaf nodes.
pub struct SparseVoxelOctree {
    /// Linearized breadth-first node array
    pub nodes: Vec<SvoNode>,
    /// World-space bounding box
    pub bounds: AabbPacked,
    /// Maximum depth in the tree
    pub max_depth: u32,
    /// Number of leaf nodes
    pub leaf_count: u32,
    /// Number of interior nodes
    pub interior_count: u32,
}

impl SparseVoxelOctree {
    /// Build an SVO from an SDF node
    pub fn build(node: &crate::types::SdfNode, config: &SvoBuildConfig) -> Self {
        build_svo(node, config)
    }

    /// Build an SVO using the compiled evaluator (faster)
    pub fn build_compiled(
        compiled: &crate::compiled::CompiledSdf,
        config: &SvoBuildConfig,
    ) -> Self {
        build_svo_compiled(compiled, config)
    }

    /// Total number of nodes
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Memory usage in bytes
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<SvoNode>()
    }

    /// Query signed distance at a point
    #[inline]
    pub fn query_point(&self, point: Vec3) -> f32 {
        svo_query_point(self, point)
    }

    /// Ray query: find first intersection with the SDF surface
    #[inline]
    pub fn ray_query(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<SvoRayHit> {
        svo_ray_query(self, origin, direction, max_distance)
    }

    /// Find the nearest surface point and distance
    #[inline]
    pub fn nearest_surface(&self, point: Vec3) -> (f32, Vec3) {
        svo_nearest_surface(self, point)
    }

    /// Linearize for GPU upload (breadth-first, contiguous)
    pub fn linearize(&self) -> LinearizedSvo {
        linearize_svo(self)
    }
}

/// Compute octant index for a point relative to a center
///
/// Returns 0-7 based on which octant the point falls in:
/// ```text
/// bit 0 = x > center.x
/// bit 1 = y > center.y
/// bit 2 = z > center.z
/// ```
#[inline]
pub fn octant_for_point(point: Vec3, center: Vec3) -> u8 {
    let mut octant = 0u8;
    if point.x > center.x { octant |= 1; }
    if point.y > center.y { octant |= 2; }
    if point.z > center.z { octant |= 4; }
    octant
}

/// Compute the center of a child octant
#[inline]
pub fn child_center(parent_center: Vec3, parent_half_size: Vec3, octant: u8) -> Vec3 {
    let quarter = parent_half_size * 0.5;
    Vec3::new(
        parent_center.x + if octant & 1 != 0 { quarter.x } else { -quarter.x },
        parent_center.y + if octant & 2 != 0 { quarter.y } else { -quarter.y },
        parent_center.z + if octant & 4 != 0 { quarter.z } else { -quarter.z },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svo_node_size() {
        assert_eq!(std::mem::size_of::<SvoNode>(), 32);
    }

    #[test]
    fn test_svo_node_default() {
        let node = SvoNode::default();
        assert_eq!(node.distance, f32::MAX);
        assert_eq!(node.is_leaf, 1);
        assert_eq!(node.child_mask, 0);
    }

    #[test]
    fn test_svo_node_leaf() {
        let node = SvoNode::leaf(1.5, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(node.distance, 1.5);
        assert_eq!(node.ny, 1.0);
        assert_eq!(node.is_leaf, 1);
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn test_svo_node_interior() {
        let node = SvoNode::interior(0.5, 0b10110101, 42);
        assert_eq!(node.is_leaf, 0);
        assert_eq!(node.child_mask, 0b10110101);
        assert_eq!(node.first_child, 42);
        assert_eq!(node.child_count(), 5);
    }

    #[test]
    fn test_child_index() {
        // child_mask = 0b00110101 (children at octants 0, 2, 4, 5)
        // first_child = 10
        let node = SvoNode::interior(0.0, 0b00110101, 10);

        assert_eq!(node.child_index(0), Some(10)); // first child
        assert_eq!(node.child_index(1), None);     // no child
        assert_eq!(node.child_index(2), Some(11)); // second child
        assert_eq!(node.child_index(3), None);     // no child
        assert_eq!(node.child_index(4), Some(12)); // third child
        assert_eq!(node.child_index(5), Some(13)); // fourth child
        assert_eq!(node.child_index(6), None);     // no child
        assert_eq!(node.child_index(7), None);     // no child
    }

    #[test]
    fn test_octant_for_point() {
        let center = Vec3::ZERO;
        assert_eq!(octant_for_point(Vec3::new(-1.0, -1.0, -1.0), center), 0);
        assert_eq!(octant_for_point(Vec3::new(1.0, -1.0, -1.0), center), 1);
        assert_eq!(octant_for_point(Vec3::new(-1.0, 1.0, -1.0), center), 2);
        assert_eq!(octant_for_point(Vec3::new(1.0, 1.0, -1.0), center), 3);
        assert_eq!(octant_for_point(Vec3::new(-1.0, -1.0, 1.0), center), 4);
        assert_eq!(octant_for_point(Vec3::new(1.0, 1.0, 1.0), center), 7);
    }

    #[test]
    fn test_child_center() {
        let parent = Vec3::ZERO;
        let half = Vec3::splat(1.0);

        let c0 = child_center(parent, half, 0);
        assert_eq!(c0, Vec3::new(-0.5, -0.5, -0.5));

        let c7 = child_center(parent, half, 7);
        assert_eq!(c7, Vec3::new(0.5, 0.5, 0.5));

        let c3 = child_center(parent, half, 3); // +X, +Y, -Z
        assert_eq!(c3, Vec3::new(0.5, 0.5, -0.5));
    }
}

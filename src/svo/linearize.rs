//! SVO Linearization: Pointer tree → GPU-friendly flat array (Deep Fried Edition)
//!
//! Converts an SVO into a compact, GPU-uploadable format with:
//! - Breadth-first node ordering (cache-friendly traversal)
//! - Packed node data for minimal memory bandwidth
//! - AABB bounds per level for hierarchical culling
//!
//! Author: Moroya Sakamoto

use super::{SparseVoxelOctree, SvoNode};
use crate::compiled::AabbPacked;

/// GPU-friendly linearized SVO
///
/// Nodes are stored in breadth-first order for cache-coherent traversal.
/// Each node is 32 bytes, suitable for direct GPU buffer upload.
pub struct LinearizedSvo {
    /// Nodes in breadth-first order (same as SparseVoxelOctree.nodes)
    pub nodes: Vec<SvoNode>,
    /// Per-level node count for hierarchical processing
    pub level_counts: Vec<u32>,
    /// Per-level cumulative offset into nodes array
    pub level_offsets: Vec<u32>,
    /// Total number of levels
    pub depth: u32,
    /// World-space bounds
    pub bounds: AabbPacked,
}

impl LinearizedSvo {
    /// Total node count
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Memory size in bytes (nodes only)
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<SvoNode>()
    }

    /// Get raw bytes for GPU upload
    ///
    /// Returns the node array as a byte slice, suitable for
    /// creating a wgpu storage buffer.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.nodes.as_ptr() as *const u8,
                self.nodes.len() * std::mem::size_of::<SvoNode>(),
            )
        }
    }

    /// Get nodes at a specific depth level
    pub fn nodes_at_level(&self, level: u32) -> &[SvoNode] {
        if level as usize >= self.level_offsets.len() {
            return &[];
        }
        let start = self.level_offsets[level as usize] as usize;
        let count = self.level_counts[level as usize] as usize;
        let end = (start + count).min(self.nodes.len());
        &self.nodes[start..end]
    }
}

/// Linearize an SVO into GPU-friendly breadth-first format
///
/// The SVO is already stored in BFS order from construction.
/// This function computes per-level metadata and validates the structure.
pub fn linearize_svo(svo: &SparseVoxelOctree) -> LinearizedSvo {
    if svo.nodes.is_empty() {
        return LinearizedSvo {
            nodes: Vec::new(),
            level_counts: Vec::new(),
            level_offsets: Vec::new(),
            depth: 0,
            bounds: svo.bounds,
        };
    }

    // Compute per-level counts by BFS traversal
    let mut level_counts: Vec<u32> = Vec::new();
    let mut level_offsets: Vec<u32> = Vec::new();

    // BFS: track which nodes are at each level
    let mut current_level_start = 0usize;
    let mut current_level_count = 1usize; // root is level 0
    let mut depth = 0u32;

    loop {
        level_offsets.push(current_level_start as u32);
        level_counts.push(current_level_count as u32);

        // Count children of all nodes at this level
        let mut next_level_count = 0usize;
        for i in current_level_start..(current_level_start + current_level_count) {
            if i >= svo.nodes.len() {
                break;
            }
            let node = &svo.nodes[i];
            if node.is_leaf == 0 {
                next_level_count += node.child_mask.count_ones() as usize;
            }
        }

        if next_level_count == 0 {
            break;
        }

        current_level_start += current_level_count;
        current_level_count = next_level_count;
        depth += 1;
    }

    LinearizedSvo {
        nodes: svo.nodes.clone(),
        level_counts,
        level_offsets,
        depth,
        bounds: svo.bounds,
    }
}

/// Validate the integrity of a linearized SVO
///
/// Checks that all child indices are valid and within bounds.
pub fn validate_linearized(lin: &LinearizedSvo) -> Result<(), String> {
    for (i, node) in lin.nodes.iter().enumerate() {
        if node.is_leaf == 0 {
            let num_children = node.child_mask.count_ones();
            if num_children == 0 {
                return Err(format!("Interior node {} has no children", i));
            }

            let first = node.first_child as usize;
            let last = first + num_children as usize - 1;

            if last >= lin.nodes.len() {
                return Err(format!(
                    "Node {} references children {}-{} but only {} nodes exist",
                    i, first, last, lin.nodes.len()
                ));
            }
        }
    }

    Ok(())
}

/// Compact an SVO by removing unreachable nodes
///
/// After modifications (e.g., destruction), some nodes may become
/// unreachable. This function rebuilds the flat array with only
/// reachable nodes and fixes up child indices.
pub fn compact_svo(svo: &SparseVoxelOctree) -> SparseVoxelOctree {
    if svo.nodes.is_empty() {
        return SparseVoxelOctree {
            nodes: Vec::new(),
            bounds: svo.bounds,
            max_depth: 0,
            leaf_count: 0,
            interior_count: 0,
        };
    }

    // Mark reachable nodes via BFS
    let mut reachable = vec![false; svo.nodes.len()];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    reachable[0] = true;
    queue.push_back(0);

    while let Some(idx) = queue.pop_front() {
        let node = &svo.nodes[idx];
        if node.is_leaf == 0 {
            for octant in 0..8u8 {
                if let Some(child_idx) = node.child_index(octant) {
                    let ci = child_idx as usize;
                    if ci < svo.nodes.len() && !reachable[ci] {
                        reachable[ci] = true;
                        queue.push_back(ci);
                    }
                }
            }
        }
    }

    // Build old→new index mapping
    let mut old_to_new = vec![0u32; svo.nodes.len()];
    let mut new_idx = 0u32;
    for (i, &r) in reachable.iter().enumerate() {
        if r {
            old_to_new[i] = new_idx;
            new_idx += 1;
        }
    }

    // Build compacted array
    let mut new_nodes = Vec::with_capacity(new_idx as usize);
    let mut leaf_count = 0u32;
    let mut interior_count = 0u32;

    for (i, &r) in reachable.iter().enumerate() {
        if !r {
            continue;
        }

        let mut node = svo.nodes[i];

        if node.is_leaf == 1 {
            leaf_count += 1;
        } else {
            interior_count += 1;
            // Fix up first_child index
            let old_first = node.first_child as usize;
            if old_first < svo.nodes.len() {
                node.first_child = old_to_new[old_first];
            }
        }

        new_nodes.push(node);
    }

    SparseVoxelOctree {
        nodes: new_nodes,
        bounds: svo.bounds,
        max_depth: svo.max_depth,
        leaf_count,
        interior_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use crate::types::SdfNode;
    use crate::svo::SvoBuildConfig;

    fn make_test_svo() -> SparseVoxelOctree {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 3,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };
        super::super::build::build_svo(&sphere, &config)
    }

    #[test]
    fn test_linearize() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);

        assert_eq!(lin.node_count(), svo.node_count());
        assert!(lin.level_counts.len() > 1, "Should have multiple levels");
        assert_eq!(lin.level_counts[0], 1, "Level 0 should have 1 node (root)");
    }

    #[test]
    fn test_linearize_level_offsets() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);

        // Level offsets should be cumulative
        assert_eq!(lin.level_offsets[0], 0);
        for i in 1..lin.level_offsets.len() {
            assert_eq!(
                lin.level_offsets[i],
                lin.level_offsets[i - 1] + lin.level_counts[i - 1],
                "Level offsets should be cumulative at level {}", i
            );
        }
    }

    #[test]
    fn test_validate() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);
        assert!(validate_linearized(&lin).is_ok());
    }

    #[test]
    fn test_nodes_at_level() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);

        let root_level = lin.nodes_at_level(0);
        assert_eq!(root_level.len(), 1);

        // Level 1 should have up to 8 nodes
        let level1 = lin.nodes_at_level(1);
        assert!(level1.len() <= 8);
    }

    #[test]
    fn test_as_bytes() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);

        let bytes = lin.as_bytes();
        assert_eq!(bytes.len(), lin.node_count() * 32);
    }

    #[test]
    fn test_compact() {
        let svo = make_test_svo();
        let compacted = compact_svo(&svo);

        // Compacted should have same or fewer nodes (no unreachable nodes initially)
        assert!(compacted.node_count() <= svo.node_count());
        assert_eq!(
            compacted.leaf_count + compacted.interior_count,
            compacted.node_count() as u32
        );
    }

    #[test]
    fn test_linearize_empty() {
        let svo = SparseVoxelOctree {
            nodes: Vec::new(),
            bounds: AabbPacked::empty(),
            max_depth: 0,
            leaf_count: 0,
            interior_count: 0,
        };
        let lin = linearize_svo(&svo);
        assert_eq!(lin.node_count(), 0);
        assert_eq!(lin.depth, 0);
    }

    #[test]
    fn test_memory_bytes() {
        let svo = make_test_svo();
        let lin = linearize_svo(&svo);
        assert_eq!(lin.memory_bytes(), lin.node_count() * 32);
    }
}

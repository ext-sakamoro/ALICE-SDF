//! SVO Construction: Top-down adaptive build (Deep Fried Edition)
//!
//! Builds a Sparse Voxel Octree by recursively subdividing space.
//! Nodes near the SDF surface are subdivided to max_depth; nodes far
//! from the surface become leaves early.
//!
//! # Algorithm
//!
//! 1. Start with root AABB covering the scene
//! 2. Evaluate SDF at node center
//! 3. If |distance| > threshold * node_size → leaf (far from surface)
//! 4. If depth == max_depth → leaf (maximum resolution)
//! 5. Otherwise → subdivide into 8 children, recurse
//! 6. Linearize pointer tree to breadth-first flat array
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use crate::compiled::{CompiledSdf, AabbPacked};
use crate::eval::{eval, normal};
use crate::compiled::{eval_compiled, eval_compiled_normal};
use crate::types::SdfNode;
use super::{SvoNode, SvoBuildConfig, SparseVoxelOctree, child_center};

/// Temporary tree node during construction (pointer-based)
struct BuildNode {
    distance: f32,
    normal: Vec3,
    depth: u32,
    children: [Option<Box<BuildNode>>; 8],
}

impl BuildNode {
    fn is_leaf(&self) -> bool {
        self.children.iter().all(|c| c.is_none())
    }

    fn child_mask(&self) -> u8 {
        let mut mask = 0u8;
        for (i, c) in self.children.iter().enumerate() {
            if c.is_some() {
                mask |= 1 << i;
            }
        }
        mask
    }
}

/// Build SVO from an interpreted SDF node
pub fn build_svo(node: &SdfNode, config: &SvoBuildConfig) -> SparseVoxelOctree {
    let center = (config.bounds_min + config.bounds_max) * 0.5;
    let half_size = (config.bounds_max - config.bounds_min) * 0.5;

    let evaluator = |p: Vec3| eval(node, p);
    let normal_fn = |p: Vec3| normal(node, p, 0.001);

    let root = build_recursive(
        center,
        half_size,
        0,
        config,
        &evaluator,
        &normal_fn,
    );

    flatten_tree(root, config, center, half_size)
}

/// Build SVO from a compiled SDF (faster)
pub fn build_svo_compiled(compiled: &CompiledSdf, config: &SvoBuildConfig) -> SparseVoxelOctree {
    let center = (config.bounds_min + config.bounds_max) * 0.5;
    let half_size = (config.bounds_max - config.bounds_min) * 0.5;

    let evaluator = |p: Vec3| eval_compiled(compiled, p);
    let normal_fn = |p: Vec3| eval_compiled_normal(compiled, p, 0.001);

    let root = build_recursive(
        center,
        half_size,
        0,
        config,
        &evaluator,
        &normal_fn,
    );

    flatten_tree(root, config, center, half_size)
}

/// Recursive top-down SVO build
fn build_recursive(
    center: Vec3,
    half_size: Vec3,
    depth: u32,
    config: &SvoBuildConfig,
    eval_fn: &dyn Fn(Vec3) -> f32,
    normal_fn: &dyn Fn(Vec3) -> Vec3,
) -> BuildNode {
    let d = eval_fn(center);
    let node_size = half_size.max_element() * 2.0;

    // Leaf conditions:
    // 1. Maximum depth reached
    // 2. Far from surface (conservative: multiply by threshold)
    let is_leaf = depth >= config.max_depth
        || d.abs() > config.distance_threshold * node_size * config.surface_detail;

    if is_leaf {
        let n = if d.abs() < node_size * 2.0 {
            normal_fn(center)
        } else {
            Vec3::ZERO
        };

        return BuildNode {
            distance: d,
            normal: n,
            depth,
            children: Default::default(),
        };
    }

    // Subdivide into 8 children
    let child_half = half_size * 0.5;
    let children: [Option<Box<BuildNode>>; 8] = std::array::from_fn(|i| {
        let octant = i as u8;
        let child_c = child_center(center, half_size, octant);
        Some(Box::new(build_recursive(
            child_c,
            child_half,
            depth + 1,
            config,
            eval_fn,
            normal_fn,
        )))
    });

    BuildNode {
        distance: d,
        normal: Vec3::ZERO,
        depth,
        children,
    }
}

/// Flatten pointer-based tree to breadth-first flat array
fn flatten_tree(
    root: BuildNode,
    _config: &SvoBuildConfig,
    center: Vec3,
    half_size: Vec3,
) -> SparseVoxelOctree {
    let mut nodes = Vec::new();
    let mut leaf_count = 0u32;
    let mut interior_count = 0u32;
    let mut max_depth_seen = 0u32;

    // Count total nodes for pre-allocation
    fn count_nodes(node: &BuildNode) -> usize {
        if node.is_leaf() {
            1
        } else {
            1 + node.children.iter()
                .filter_map(|c| c.as_ref())
                .map(|c| count_nodes(c))
                .sum::<usize>()
        }
    }

    let total_estimate = count_nodes(&root);
    nodes.reserve(total_estimate);

    // BFS flattening with forward child index allocation
    let mut bfs_queue: std::collections::VecDeque<BuildNode> = std::collections::VecDeque::new();
    nodes.push(SvoNode::default()); // placeholder for root
    bfs_queue.push_back(root);

    let mut write_idx = 0usize;
    let mut next_free = 1usize;

    while let Some(build_node) = bfs_queue.pop_front() {
        max_depth_seen = max_depth_seen.max(build_node.depth);

        if build_node.is_leaf() {
            nodes[write_idx] = SvoNode::leaf(build_node.distance, build_node.normal);
            leaf_count += 1;
        } else {
            let mask = build_node.child_mask();
            let num_children = mask.count_ones() as usize;

            // Allocate slots for children
            let first_child_idx = next_free;
            next_free += num_children;

            // Ensure nodes vec is large enough
            while nodes.len() < next_free {
                nodes.push(SvoNode::default());
            }

            nodes[write_idx] = SvoNode::interior(
                build_node.distance,
                mask,
                first_child_idx as u32,
            );
            interior_count += 1;

            // Enqueue children in order
            for child in build_node.children.into_iter().flatten() {
                bfs_queue.push_back(*child);
            }
        }

        write_idx += 1;
    }

    // Trim to actual size
    nodes.truncate(write_idx);

    SparseVoxelOctree {
        nodes,
        bounds: AabbPacked::new(center - half_size, center + half_size),
        max_depth: max_depth_seen,
        leaf_count,
        interior_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 4,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };

        let svo = build_svo(&sphere, &config);

        assert!(svo.nodes.len() > 1, "SVO should have multiple nodes");
        assert!(svo.leaf_count > 0, "SVO should have leaves");
        assert!(svo.interior_count > 0, "SVO should have interior nodes");
        assert_eq!(svo.node_count(), (svo.leaf_count + svo.interior_count) as usize);
    }

    #[test]
    fn test_build_compiled() {
        let sphere = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&sphere);
        let config = SvoBuildConfig {
            max_depth: 4,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };

        let svo = build_svo_compiled(&compiled, &config);

        assert!(svo.nodes.len() > 1);
        assert!(svo.leaf_count > 0);
    }

    #[test]
    fn test_build_depth_limit() {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 2,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };

        let svo = build_svo(&sphere, &config);
        assert!(svo.max_depth <= 2, "SVO depth should not exceed max_depth");
    }

    #[test]
    fn test_build_far_object() {
        // Object far from bounds center → mostly leaves
        let sphere = SdfNode::sphere(0.1).translate(100.0, 100.0, 100.0);
        let config = SvoBuildConfig {
            max_depth: 6,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };

        let svo = build_svo(&sphere, &config);
        // Should be very compact since everything is far from surface
        assert!(svo.interior_count <= 1, "Far object should produce mostly root-level leaf");
    }

    #[test]
    fn test_root_node_is_first() {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 3,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };

        let svo = build_svo(&sphere, &config);

        // Root should be at index 0 and be interior (sphere surface exists within bounds)
        let root = &svo.nodes[0];
        assert_eq!(root.is_leaf, 0, "Root should be interior for sphere in bounds");
        assert!(root.child_mask > 0, "Root should have children");
    }
}

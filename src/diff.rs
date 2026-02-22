//! CSG Tree structural diff and patch
//!
//! Computes a minimal set of operations to transform one SDF tree into another.
//! Useful for undo/redo systems, network synchronization, and collaborative editing.
//!
//! Author: Moroya Sakamoto

use crate::crispy::fnv1a_hash;
use crate::types::SdfNode;
use std::sync::Arc;

// ── Types ────────────────────────────────────────────────────

/// A path from the root to a specific node in the SDF tree.
///
/// Each element is the child index at that depth level.
/// Example: `[0, 2, 1]` means root → 0th child → 2nd child → 1st child.
pub type TreePath = Vec<usize>;

/// A single diff operation.
#[derive(Debug, Clone)]
pub enum DiffOp {
    /// Replace the node at `path` with a new node.
    Replace {
        /// Path to the node to replace.
        path: TreePath,
        /// Hash of the old node (for verification).
        old_hash: u64,
        /// Replacement node.
        new_node: Arc<SdfNode>,
    },
}

/// Error type for patch application.
#[derive(Debug, Clone)]
pub enum DiffError {
    /// The path does not lead to a valid node.
    InvalidPath(TreePath),
    /// Hash mismatch: the tree has changed since the diff was computed.
    HashMismatch {
        /// Path where the mismatch occurred.
        path: TreePath,
        /// Expected hash.
        expected: u64,
        /// Actual hash.
        actual: u64,
    },
}

impl std::fmt::Display for DiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiffError::InvalidPath(p) => write!(f, "Invalid path: {:?}", p),
            DiffError::HashMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "Hash mismatch at {:?}: expected {:#x}, got {:#x}",
                path, expected, actual
            ),
        }
    }
}

/// A patch containing a set of diff operations.
#[derive(Debug, Clone)]
pub struct TreePatch {
    /// Ordered list of diff operations.
    pub ops: Vec<DiffOp>,
    /// Content hash of this patch (for deduplication).
    pub content_hash: u64,
}

// ── Tree Hashing ─────────────────────────────────────────────

/// Compute a deterministic structural hash of an SDF node.
///
/// Two structurally identical trees produce the same hash.
/// Uses FNV-1a over a serialized representation.
pub fn tree_hash(node: &SdfNode) -> u64 {
    let serialized = format!("{:?}", node);
    fnv1a_hash(serialized.as_bytes())
}

// ── Diff Computation ─────────────────────────────────────────

/// Compute the structural diff between two SDF trees.
///
/// Returns a `TreePatch` describing the minimal set of `Replace` operations
/// needed to transform `old` into `new`.
pub fn tree_diff(old: &SdfNode, new: &SdfNode) -> TreePatch {
    let mut ops = Vec::new();
    diff_recursive(old, new, &mut vec![], &mut ops);

    let mut hash_data = Vec::with_capacity(ops.len() * 8);
    for (i, op) in ops.iter().enumerate() {
        hash_data.extend_from_slice(&(i as u64).to_le_bytes());
        match op {
            DiffOp::Replace { old_hash, .. } => {
                hash_data.extend_from_slice(&old_hash.to_le_bytes());
            }
        }
    }
    let content_hash = if hash_data.is_empty() {
        0
    } else {
        fnv1a_hash(&hash_data)
    };

    TreePatch { ops, content_hash }
}

/// Recursive diff: compare old and new subtrees at the current path.
fn diff_recursive(
    old: &SdfNode,
    new: &SdfNode,
    path: &mut Vec<usize>,
    ops: &mut Vec<DiffOp>,
) {
    let old_hash = tree_hash(old);
    let new_hash = tree_hash(new);

    // Identical subtrees: skip
    if old_hash == new_hash {
        return;
    }

    // Try to recurse into matching structure
    let old_children = get_children(old);
    let new_children = get_children(new);

    if same_variant(old, new) && old_children.len() == new_children.len() && !old_children.is_empty()
    {
        // Same node type with same child count: recurse into children
        let ops_before = ops.len();
        for i in 0..old_children.len() {
            path.push(i);
            diff_recursive(old_children[i], new_children[i], path, ops);
            path.pop();
        }
        // If no child diffs were emitted but node hashes differ,
        // the node's own parameters changed (e.g. offset, k, radius) → Replace
        if ops.len() == ops_before {
            ops.push(DiffOp::Replace {
                path: path.clone(),
                old_hash,
                new_node: Arc::new(new.clone()),
            });
        }
    } else {
        // Different structure: emit Replace
        ops.push(DiffOp::Replace {
            path: path.clone(),
            old_hash,
            new_node: Arc::new(new.clone()),
        });
    }
}

/// Check if two SdfNode variants have the same discriminant.
fn same_variant(a: &SdfNode, b: &SdfNode) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

/// Extract children of an SdfNode.
fn get_children(node: &SdfNode) -> Vec<&SdfNode> {
    match node {
        // Operations with two children
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. } => vec![a.as_ref(), b.as_ref()],

        // Single-child transforms and modifiers
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Scale { child, .. }
        | SdfNode::ScaleNonUniform { child, .. }
        | SdfNode::ProjectiveTransform { child, .. }
        | SdfNode::LatticeDeform { child, .. }
        | SdfNode::SdfSkinning { child, .. }
        | SdfNode::Twist { child, .. }
        | SdfNode::Bend { child, .. }
        | SdfNode::RepeatInfinite { child, .. }
        | SdfNode::RepeatFinite { child, .. }
        | SdfNode::Noise { child, .. }
        | SdfNode::Round { child, .. }
        | SdfNode::Onion { child, .. }
        | SdfNode::Elongate { child, .. }
        | SdfNode::Mirror { child, .. }
        | SdfNode::OctantMirror { child }
        | SdfNode::Revolution { child, .. }
        | SdfNode::Extrude { child, .. }
        | SdfNode::Taper { child, .. }
        | SdfNode::Displacement { child, .. }
        | SdfNode::PolarRepeat { child, .. }
        | SdfNode::SweepBezier { child, .. }
        | SdfNode::IcosahedralSymmetry { child }
        | SdfNode::IFS { child, .. }
        | SdfNode::HeightmapDisplacement { child, .. }
        | SdfNode::SurfaceRoughness { child, .. }
        | SdfNode::Shear { child, .. }
        | SdfNode::Animated { child, .. }
        | SdfNode::WithMaterial { child, .. } => vec![child.as_ref()],

        // Leaf nodes: no children
        _ => vec![],
    }
}

// ── Patch Application ────────────────────────────────────────

/// Apply a patch to an SDF tree, producing a new tree.
pub fn apply_patch(tree: &SdfNode, patch: &TreePatch) -> Result<SdfNode, DiffError> {
    let mut result = tree.clone();
    for op in &patch.ops {
        match op {
            DiffOp::Replace {
                path,
                old_hash,
                new_node,
            } => {
                result = replace_at_path(&result, path, *old_hash, new_node)?;
            }
        }
    }
    Ok(result)
}

/// Replace the node at the given path.
fn replace_at_path(
    node: &SdfNode,
    path: &[usize],
    expected_hash: u64,
    replacement: &Arc<SdfNode>,
) -> Result<SdfNode, DiffError> {
    if path.is_empty() {
        let actual_hash = tree_hash(node);
        if actual_hash != expected_hash {
            return Err(DiffError::HashMismatch {
                path: vec![],
                expected: expected_hash,
                actual: actual_hash,
            });
        }
        return Ok(replacement.as_ref().clone());
    }

    let idx = path[0];
    let rest = &path[1..];
    let children = get_children(node);

    if idx >= children.len() {
        return Err(DiffError::InvalidPath(path.to_vec()));
    }

    let new_child = Arc::new(replace_at_path(children[idx], rest, expected_hash, replacement)?);
    Ok(replace_child(node, idx, new_child))
}

/// Create a clone of the node with one child replaced.
fn replace_child(node: &SdfNode, child_idx: usize, new_child: Arc<SdfNode>) -> SdfNode {
    match node {
        // ── Two-child operations ──────────────────────────────────
        SdfNode::Union { a, b } => {
            if child_idx == 0 {
                SdfNode::Union { a: new_child, b: b.clone() }
            } else {
                SdfNode::Union { a: a.clone(), b: new_child }
            }
        }
        SdfNode::Intersection { a, b } => {
            if child_idx == 0 {
                SdfNode::Intersection { a: new_child, b: b.clone() }
            } else {
                SdfNode::Intersection { a: a.clone(), b: new_child }
            }
        }
        SdfNode::Subtraction { a, b } => {
            if child_idx == 0 {
                SdfNode::Subtraction { a: new_child, b: b.clone() }
            } else {
                SdfNode::Subtraction { a: a.clone(), b: new_child }
            }
        }
        SdfNode::SmoothUnion { a, b, k } => {
            if child_idx == 0 {
                SdfNode::SmoothUnion { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::SmoothUnion { a: a.clone(), b: new_child, k: *k }
            }
        }
        SdfNode::SmoothIntersection { a, b, k } => {
            if child_idx == 0 {
                SdfNode::SmoothIntersection { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::SmoothIntersection { a: a.clone(), b: new_child, k: *k }
            }
        }
        SdfNode::SmoothSubtraction { a, b, k } => {
            if child_idx == 0 {
                SdfNode::SmoothSubtraction { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::SmoothSubtraction { a: a.clone(), b: new_child, k: *k }
            }
        }
        SdfNode::ChamferUnion { a, b, r } => {
            if child_idx == 0 {
                SdfNode::ChamferUnion { a: new_child, b: b.clone(), r: *r }
            } else {
                SdfNode::ChamferUnion { a: a.clone(), b: new_child, r: *r }
            }
        }
        SdfNode::ChamferIntersection { a, b, r } => {
            if child_idx == 0 {
                SdfNode::ChamferIntersection { a: new_child, b: b.clone(), r: *r }
            } else {
                SdfNode::ChamferIntersection { a: a.clone(), b: new_child, r: *r }
            }
        }
        SdfNode::ChamferSubtraction { a, b, r } => {
            if child_idx == 0 {
                SdfNode::ChamferSubtraction { a: new_child, b: b.clone(), r: *r }
            } else {
                SdfNode::ChamferSubtraction { a: a.clone(), b: new_child, r: *r }
            }
        }
        SdfNode::StairsUnion { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::StairsUnion { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::StairsUnion { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::StairsIntersection { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::StairsIntersection { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::StairsIntersection { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::StairsSubtraction { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::StairsSubtraction { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::StairsSubtraction { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::XOR { a, b } => {
            if child_idx == 0 {
                SdfNode::XOR { a: new_child, b: b.clone() }
            } else {
                SdfNode::XOR { a: a.clone(), b: new_child }
            }
        }
        SdfNode::Morph { a, b, t } => {
            if child_idx == 0 {
                SdfNode::Morph { a: new_child, b: b.clone(), t: *t }
            } else {
                SdfNode::Morph { a: a.clone(), b: new_child, t: *t }
            }
        }
        SdfNode::ColumnsUnion { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::ColumnsUnion { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::ColumnsUnion { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::ColumnsIntersection { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::ColumnsIntersection { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::ColumnsIntersection { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::ColumnsSubtraction { a, b, r, n } => {
            if child_idx == 0 {
                SdfNode::ColumnsSubtraction { a: new_child, b: b.clone(), r: *r, n: *n }
            } else {
                SdfNode::ColumnsSubtraction { a: a.clone(), b: new_child, r: *r, n: *n }
            }
        }
        SdfNode::Pipe { a, b, r } => {
            if child_idx == 0 {
                SdfNode::Pipe { a: new_child, b: b.clone(), r: *r }
            } else {
                SdfNode::Pipe { a: a.clone(), b: new_child, r: *r }
            }
        }
        SdfNode::Engrave { a, b, r } => {
            if child_idx == 0 {
                SdfNode::Engrave { a: new_child, b: b.clone(), r: *r }
            } else {
                SdfNode::Engrave { a: a.clone(), b: new_child, r: *r }
            }
        }
        SdfNode::Groove { a, b, ra, rb } => {
            if child_idx == 0 {
                SdfNode::Groove { a: new_child, b: b.clone(), ra: *ra, rb: *rb }
            } else {
                SdfNode::Groove { a: a.clone(), b: new_child, ra: *ra, rb: *rb }
            }
        }
        SdfNode::Tongue { a, b, ra, rb } => {
            if child_idx == 0 {
                SdfNode::Tongue { a: new_child, b: b.clone(), ra: *ra, rb: *rb }
            } else {
                SdfNode::Tongue { a: a.clone(), b: new_child, ra: *ra, rb: *rb }
            }
        }
        SdfNode::ExpSmoothUnion { a, b, k } => {
            if child_idx == 0 {
                SdfNode::ExpSmoothUnion { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::ExpSmoothUnion { a: a.clone(), b: new_child, k: *k }
            }
        }
        SdfNode::ExpSmoothIntersection { a, b, k } => {
            if child_idx == 0 {
                SdfNode::ExpSmoothIntersection { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::ExpSmoothIntersection { a: a.clone(), b: new_child, k: *k }
            }
        }
        SdfNode::ExpSmoothSubtraction { a, b, k } => {
            if child_idx == 0 {
                SdfNode::ExpSmoothSubtraction { a: new_child, b: b.clone(), k: *k }
            } else {
                SdfNode::ExpSmoothSubtraction { a: a.clone(), b: new_child, k: *k }
            }
        }

        // ── Single-child transforms ───────────────────────────────
        SdfNode::Translate { offset, .. } => SdfNode::Translate {
            child: new_child,
            offset: *offset,
        },
        SdfNode::Rotate { rotation, .. } => SdfNode::Rotate {
            child: new_child,
            rotation: *rotation,
        },
        SdfNode::Scale { factor, .. } => SdfNode::Scale {
            child: new_child,
            factor: *factor,
        },
        SdfNode::ScaleNonUniform { factors, .. } => SdfNode::ScaleNonUniform {
            child: new_child,
            factors: *factors,
        },
        SdfNode::ProjectiveTransform { inv_matrix, lipschitz_bound, .. } => {
            SdfNode::ProjectiveTransform {
                child: new_child,
                inv_matrix: *inv_matrix,
                lipschitz_bound: *lipschitz_bound,
            }
        }
        SdfNode::LatticeDeform { control_points, nx, ny, nz, bbox_min, bbox_max, .. } => {
            SdfNode::LatticeDeform {
                child: new_child,
                control_points: control_points.clone(),
                nx: *nx,
                ny: *ny,
                nz: *nz,
                bbox_min: *bbox_min,
                bbox_max: *bbox_max,
            }
        }
        SdfNode::SdfSkinning { bones, .. } => SdfNode::SdfSkinning {
            child: new_child,
            bones: bones.clone(),
        },

        // ── Single-child modifiers ────────────────────────────────
        SdfNode::Twist { strength, .. } => SdfNode::Twist {
            child: new_child,
            strength: *strength,
        },
        SdfNode::Bend { curvature, .. } => SdfNode::Bend {
            child: new_child,
            curvature: *curvature,
        },
        SdfNode::RepeatInfinite { spacing, .. } => SdfNode::RepeatInfinite {
            child: new_child,
            spacing: *spacing,
        },
        SdfNode::RepeatFinite { count, spacing, .. } => SdfNode::RepeatFinite {
            child: new_child,
            count: *count,
            spacing: *spacing,
        },
        SdfNode::Noise { amplitude, frequency, seed, .. } => SdfNode::Noise {
            child: new_child,
            amplitude: *amplitude,
            frequency: *frequency,
            seed: *seed,
        },
        SdfNode::Round { radius, .. } => SdfNode::Round {
            child: new_child,
            radius: *radius,
        },
        SdfNode::Onion { thickness, .. } => SdfNode::Onion {
            child: new_child,
            thickness: *thickness,
        },
        SdfNode::Elongate { amount, .. } => SdfNode::Elongate {
            child: new_child,
            amount: *amount,
        },
        SdfNode::Mirror { axes, .. } => SdfNode::Mirror {
            child: new_child,
            axes: *axes,
        },
        SdfNode::OctantMirror { .. } => SdfNode::OctantMirror { child: new_child },
        SdfNode::Revolution { offset, .. } => SdfNode::Revolution {
            child: new_child,
            offset: *offset,
        },
        SdfNode::Extrude { half_height, .. } => SdfNode::Extrude {
            child: new_child,
            half_height: *half_height,
        },
        SdfNode::Taper { factor, .. } => SdfNode::Taper {
            child: new_child,
            factor: *factor,
        },
        SdfNode::Displacement { strength, .. } => SdfNode::Displacement {
            child: new_child,
            strength: *strength,
        },
        SdfNode::PolarRepeat { count, .. } => SdfNode::PolarRepeat {
            child: new_child,
            count: *count,
        },
        SdfNode::SweepBezier { p0, p1, p2, .. } => SdfNode::SweepBezier {
            child: new_child,
            p0: *p0,
            p1: *p1,
            p2: *p2,
        },
        SdfNode::IcosahedralSymmetry { .. } => SdfNode::IcosahedralSymmetry { child: new_child },
        SdfNode::IFS { transforms, iterations, .. } => SdfNode::IFS {
            child: new_child,
            transforms: transforms.clone(),
            iterations: *iterations,
        },
        SdfNode::HeightmapDisplacement { heightmap, width, height, amplitude, scale, .. } => {
            SdfNode::HeightmapDisplacement {
                child: new_child,
                heightmap: heightmap.clone(),
                width: *width,
                height: *height,
                amplitude: *amplitude,
                scale: *scale,
            }
        }
        SdfNode::SurfaceRoughness { frequency, amplitude, octaves, .. } => {
            SdfNode::SurfaceRoughness {
                child: new_child,
                frequency: *frequency,
                amplitude: *amplitude,
                octaves: *octaves,
            }
        }
        SdfNode::Shear { shear, .. } => SdfNode::Shear {
            child: new_child,
            shear: *shear,
        },
        SdfNode::Animated { speed, amplitude, .. } => SdfNode::Animated {
            child: new_child,
            speed: *speed,
            amplitude: *amplitude,
        },
        SdfNode::WithMaterial { material_id, .. } => SdfNode::WithMaterial {
            child: new_child,
            material_id: *material_id,
        },

        // Leaf nodes have no children; reaching here indicates a bug in replace_at_path
        _ => unreachable!("Leaf nodes have no children"),
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_trees_empty_diff() {
        let tree = SdfNode::sphere(1.0);
        let patch = tree_diff(&tree, &tree);
        assert!(patch.ops.is_empty());
        assert_eq!(patch.content_hash, 0);
    }

    #[test]
    fn different_primitives() {
        let old = SdfNode::sphere(1.0);
        let new = SdfNode::sphere(2.0);
        let patch = tree_diff(&old, &new);
        assert_eq!(patch.ops.len(), 1);
        match &patch.ops[0] {
            DiffOp::Replace { path, .. } => assert!(path.is_empty()),
        }
    }

    #[test]
    fn different_primitive_types() {
        let old = SdfNode::sphere(1.0);
        let new = SdfNode::box3d(1.0, 1.0, 1.0);
        let patch = tree_diff(&old, &new);
        assert_eq!(patch.ops.len(), 1);
    }

    #[test]
    fn child_changed_in_union() {
        let old = SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        let new = SdfNode::sphere(2.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        let patch = tree_diff(&old, &new);
        // Only the first child changed
        assert_eq!(patch.ops.len(), 1);
        match &patch.ops[0] {
            DiffOp::Replace { path, .. } => assert_eq!(path, &vec![0]),
        }
    }

    #[test]
    fn apply_patch_roundtrip() {
        let old = SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        let new = SdfNode::sphere(2.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        let patch = tree_diff(&old, &new);
        let result = apply_patch(&old, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new));
    }

    #[test]
    fn tree_hash_deterministic() {
        let a = SdfNode::sphere(1.0);
        let h1 = tree_hash(&a);
        let h2 = tree_hash(&a);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    #[test]
    fn tree_hash_differs() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(2.0);
        assert_ne!(tree_hash(&a), tree_hash(&b));
    }

    #[test]
    fn patch_content_hash_nonempty() {
        let old = SdfNode::sphere(1.0);
        let new = SdfNode::sphere(2.0);
        let patch = tree_diff(&old, &new);
        assert_ne!(patch.content_hash, 0);
    }

    #[test]
    fn apply_patch_invalid_path() {
        let tree = SdfNode::sphere(1.0);
        let patch = TreePatch {
            ops: vec![DiffOp::Replace {
                path: vec![0], // sphere has no children
                old_hash: 0,
                new_node: Arc::new(SdfNode::sphere(2.0)),
            }],
            content_hash: 1,
        };
        assert!(apply_patch(&tree, &patch).is_err());
    }

    #[test]
    fn apply_patch_hash_mismatch() {
        let tree = SdfNode::sphere(1.0);
        let patch = TreePatch {
            ops: vec![DiffOp::Replace {
                path: vec![],
                old_hash: 9999, // wrong hash
                new_node: Arc::new(SdfNode::sphere(2.0)),
            }],
            content_hash: 1,
        };
        match apply_patch(&tree, &patch) {
            Err(DiffError::HashMismatch { .. }) => {}
            _ => panic!("Expected HashMismatch error"),
        }
    }

    #[test]
    fn nested_tree_diff() {
        let old = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .union(SdfNode::box3d(1.0, 1.0, 1.0));
        let new = SdfNode::sphere(1.0)
            .translate(2.0, 0.0, 0.0) // changed offset
            .union(SdfNode::box3d(1.0, 1.0, 1.0));
        let patch = tree_diff(&old, &new);
        assert!(!patch.ops.is_empty());
    }

    #[test]
    fn test_replace_child_smooth_subtraction() {
        let old = SdfNode::SmoothSubtraction {
            a: Arc::new(SdfNode::sphere(1.0)),
            b: Arc::new(SdfNode::box3d(0.5, 0.5, 0.5)),
            k: 0.1,
        };
        let new = SdfNode::SmoothSubtraction {
            a: Arc::new(SdfNode::sphere(2.0)), // changed
            b: Arc::new(SdfNode::box3d(0.5, 0.5, 0.5)),
            k: 0.1,
        };
        let patch = tree_diff(&old, &new);
        assert_eq!(patch.ops.len(), 1);
        let result = apply_patch(&old, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new));
    }

    #[test]
    fn test_replace_child_morph() {
        let old = SdfNode::Morph {
            a: Arc::new(SdfNode::sphere(1.0)),
            b: Arc::new(SdfNode::box3d(1.0, 1.0, 1.0)),
            t: 0.5,
        };
        let new = SdfNode::Morph {
            a: Arc::new(SdfNode::sphere(1.0)),
            b: Arc::new(SdfNode::box3d(2.0, 2.0, 2.0)), // changed second child
            t: 0.5,
        };
        let patch = tree_diff(&old, &new);
        assert_eq!(patch.ops.len(), 1);
        match &patch.ops[0] {
            DiffOp::Replace { path, .. } => assert_eq!(path, &vec![1]),
        }
        let result = apply_patch(&old, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new));
    }

    #[test]
    fn test_replace_child_xor() {
        let old = SdfNode::XOR {
            a: Arc::new(SdfNode::sphere(1.0)),
            b: Arc::new(SdfNode::sphere(1.5)),
        };
        let new = SdfNode::XOR {
            a: Arc::new(SdfNode::sphere(1.0)),
            b: Arc::new(SdfNode::sphere(3.0)), // changed
        };
        let patch = tree_diff(&old, &new);
        assert_eq!(patch.ops.len(), 1);
        let result = apply_patch(&old, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new));
    }

    #[test]
    fn test_replace_child_complex_tree() {
        // SmoothUnion( Groove(sphere, box), StairsUnion(cylinder, torus) )
        use glam::Vec3;
        let old = SdfNode::SmoothUnion {
            a: Arc::new(SdfNode::Groove {
                a: Arc::new(SdfNode::sphere(1.0)),
                b: Arc::new(SdfNode::box3d(0.5, 0.5, 0.5)),
                ra: 0.1,
                rb: 0.2,
            }),
            b: Arc::new(SdfNode::StairsUnion {
                a: Arc::new(SdfNode::Cylinder { radius: 0.5, half_height: 1.0 }),
                b: Arc::new(SdfNode::Torus { major_radius: 1.0, minor_radius: 0.2 }),
                r: 0.3,
                n: 4.0,
            }),
            k: 0.05,
        };
        // Change the torus minor_radius inside the StairsUnion (path: [1, 1])
        let new = SdfNode::SmoothUnion {
            a: Arc::new(SdfNode::Groove {
                a: Arc::new(SdfNode::sphere(1.0)),
                b: Arc::new(SdfNode::box3d(0.5, 0.5, 0.5)),
                ra: 0.1,
                rb: 0.2,
            }),
            b: Arc::new(SdfNode::StairsUnion {
                a: Arc::new(SdfNode::Cylinder { radius: 0.5, half_height: 1.0 }),
                b: Arc::new(SdfNode::Torus { major_radius: 1.0, minor_radius: 0.5 }), // changed
                r: 0.3,
                n: 4.0,
            }),
            k: 0.05,
        };
        let _ = Vec3::ZERO; // suppress unused import warning
        let patch = tree_diff(&old, &new);
        assert!(!patch.ops.is_empty());
        let result = apply_patch(&old, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new));
    }

    #[test]
    fn test_replace_child_single_child_modifiers() {
        // Test Bend, Elongate, Mirror roundtrips
        use glam::Vec3;

        // Bend
        let old_bend = SdfNode::Bend {
            child: Arc::new(SdfNode::sphere(1.0)),
            curvature: 0.5,
        };
        let new_bend = SdfNode::Bend {
            child: Arc::new(SdfNode::sphere(2.0)),
            curvature: 0.5,
        };
        let patch = tree_diff(&old_bend, &new_bend);
        assert_eq!(patch.ops.len(), 1);
        let result = apply_patch(&old_bend, &patch).unwrap();
        assert_eq!(tree_hash(&result), tree_hash(&new_bend));

        // Elongate
        let old_elongate = SdfNode::Elongate {
            child: Arc::new(SdfNode::sphere(1.0)),
            amount: Vec3::new(0.5, 0.5, 0.5),
        };
        let new_elongate = SdfNode::Elongate {
            child: Arc::new(SdfNode::sphere(3.0)),
            amount: Vec3::new(0.5, 0.5, 0.5),
        };
        let patch2 = tree_diff(&old_elongate, &new_elongate);
        assert_eq!(patch2.ops.len(), 1);
        let result2 = apply_patch(&old_elongate, &patch2).unwrap();
        assert_eq!(tree_hash(&result2), tree_hash(&new_elongate));

        // Mirror
        let old_mirror = SdfNode::Mirror {
            child: Arc::new(SdfNode::box3d(1.0, 1.0, 1.0)),
            axes: Vec3::new(1.0, 0.0, 0.0),
        };
        let new_mirror = SdfNode::Mirror {
            child: Arc::new(SdfNode::box3d(2.0, 2.0, 2.0)),
            axes: Vec3::new(1.0, 0.0, 0.0),
        };
        let patch3 = tree_diff(&old_mirror, &new_mirror);
        assert_eq!(patch3.ops.len(), 1);
        let result3 = apply_patch(&old_mirror, &patch3).unwrap();
        assert_eq!(tree_hash(&result3), tree_hash(&new_mirror));
    }
}

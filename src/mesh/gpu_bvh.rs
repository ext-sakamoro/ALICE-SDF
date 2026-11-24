//! A built BVH flattened for a GPU kernel: storage-buffer words walked by parent links (no stack).
//!
//! Node layout (8 `u32` words): `min.xyz` (f32 bits), `a`, `max.xyz` (f32 bits), `b`. A leaf has the top bit of
//! `a` set, the rest of `a` its first entry in the leaf list and `b` its entry count; an internal node has its
//! left child in `a` and its right in `b`. The root is node 0, children follow their parent in depth-first
//! order, and `parents[i]` is node `i`'s parent (the root's is itself).
//!
//! [`flatten_nodes`] flattens any [`BvhNode`] tree (leaf entries are primitive indices, e.g. the sheet boxes of
//! `sheet_field`).

use super::bvh::BvhNode;

/// The flag on word `a` that marks a leaf.
pub const LEAF: u32 = 0x8000_0000;

/// A flattened tree: nodes, leaf entries and parent links.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FlatNodes {
    /// 8 words a node (see the module docs).
    pub nodes: Vec<u32>,
    /// Primitive indices of the leaves, leaf by leaf.
    pub leaf_entries: Vec<u32>,
    /// The parent of each node (the root's is itself).
    pub parents: Vec<u32>,
}

fn push(node: &BvhNode, out: &mut FlatNodes) -> u32 {
    let at = (out.nodes.len() / 8) as u32;
    out.parents.push(at);
    let a = node.aabb();
    out.nodes.extend_from_slice(&[
        a.min.x.to_bits(),
        a.min.y.to_bits(),
        a.min.z.to_bits(),
        0,
        a.max.x.to_bits(),
        a.max.y.to_bits(),
        a.max.z.to_bits(),
        0,
    ]);
    let (w3, w7) = match node {
        BvhNode::Leaf { triangles, .. } => {
            let first = out.leaf_entries.len() as u32;
            out.leaf_entries.extend(triangles.iter().map(|&t| t as u32));
            (first | LEAF, triangles.len() as u32)
        }
        BvhNode::Internal { left, right, .. } => {
            let l = push(left, out);
            let r = push(right, out);
            out.parents[l as usize] = at;
            out.parents[r as usize] = at;
            (l, r)
        }
    };
    out.nodes[at as usize * 8 + 3] = w3;
    out.nodes[at as usize * 8 + 7] = w7;
    at
}

/// Flatten a tree; `None` gives no nodes.
#[must_use]
pub fn flatten_nodes(root: Option<&BvhNode>) -> FlatNodes {
    let mut out = FlatNodes::default();
    if let Some(r) = root {
        push(r, &mut out);
    }
    out
}

//! The generalized winding number of a triangle mesh (Jacobson, Kavan, Sorkine-Hornung 2013)
//!
//! ```text
//! w(p) = Σ Ω_t(p) / 4π        Ω_t = the signed solid angle triangle t subtends at p
//! ```
//!
//! For a closed, consistently outward-wound mesh `w` is 1 inside and 0 outside. A hole, a slit
//! or a fold gap only removes the solid angle of the missing piece, so a point well inside
//! still reads close to 1, and `w ≥ ½` keeps the inside where a flood fill leaks through the
//! opening. Two closed parts that touch or leave a small gap each keep their own inside.
//!
//! What it does not tolerate is inconsistent winding: a flipped triangle subtracts its solid
//! angle instead of adding it. [`winding_report`] counts the edges two triangles run the same
//! way (the mark of a flip), and the winding sign of
//! [`MeshSdf`](crate::mesh::MeshSdf) refuses a mesh with any.
//!
//! # The solid angle
//!
//! Van Oosterom and Strackee (1983): with `a, b, c` the vertices relative to `p` and `|·|` their
//! lengths,
//!
//! ```text
//! Ω = 2 · atan2( a · (b × c),  |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b| )
//! ```
//!
//! A point in the triangle's plane (`a · (b × c) = 0`, which includes every zero-area triangle)
//! contributes 0: off the triangle that is its solid angle, and on it the distance is 0, so the
//! sign does not matter. Without this rule a denominator that rounds to a negative value or −0
//! would give `atan2(0, ·) = ±π`, a phantom half sphere.
//!
//! # Determinism
//!
//! Computed in `f64` from the `f32` inputs (exact conversions), with `alice_det_math::sqrt64` and
//! `atan2_64`, in a fixed operation order, and summed in triangle index order: the same bits on
//! every target. No fused multiply-add.
//!
//! Author: Moroya Sakamoto

use super::bvh::BvhTriangle;
use glam::Vec3;

/// `4π`, the solid angle of the whole sphere
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

#[inline]
fn d3(v: Vec3) -> [f64; 3] {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The signed solid angle the triangle `v0 v1 v2` subtends at `p` (Van Oosterom–Strackee)
///
/// Positive when `p` lies behind the triangle, on the side its normal `(v1 − v0) × (v2 − v0)`
/// points away from (the inside of an outward-wound mesh); 0 for a point in the triangle's
/// plane.
#[must_use]
pub fn solid_angle(v0: Vec3, v1: Vec3, v2: Vec3, p: Vec3) -> f64 {
    let p = d3(p);
    let (a, b, c) = (sub(d3(v0), p), sub(d3(v1), p), sub(d3(v2), p));
    let det = dot(a, cross(b, c));
    if det == 0.0 {
        return 0.0;
    }
    let (la, lb, lc) = (
        alice_det_math::sqrt64(dot(a, a)),
        alice_det_math::sqrt64(dot(b, b)),
        alice_det_math::sqrt64(dot(c, c)),
    );
    let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    2.0 * alice_det_math::atan2_64(det, den)
}

/// The generalized winding number of `triangles` at `p`: `Σ Ω / 4π`, summed in slice order
///
/// 1 inside and 0 outside a closed, consistently outward-wound mesh; close to 1 inside a mesh
/// with holes, slits or fold gaps. A [`MeshBvh`](crate::mesh::MeshBvh) keeps its `triangles`
/// in input order.
#[must_use]
pub fn winding_number(triangles: &[BvhTriangle], p: Vec3) -> f64 {
    let mut sum = 0.0f64;
    for t in triangles {
        sum += solid_angle(t.v0, t.v1, t.v2, p);
    }
    sum / FOUR_PI
}

/// How consistently a triangle list is wound, counted on its index edges
///
/// An edge is an unordered pair of distinct vertex indices; each triangle `(a, b, c)` runs its
/// edges `a→b`, `b→c`, `c→a`. Vertices are matched by index, so weld coincident vertices first
/// if the mesh repeats positions along its seams (unwelded seams read as boundary edges, which
/// the winding number tolerates).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindingReport {
    /// Edges of exactly one triangle (holes, slits, the rim of an open sheet)
    pub boundary_edges: usize,
    /// Edges of two triangles that both run them the same way: one of the two is flipped
    pub inconsistent_edges: usize,
    /// Edges of three or more triangles
    pub non_manifold_edges: usize,
    /// Triangles with a repeated index (no area), whose edges are not counted
    pub degenerate_triangles: usize,
}

/// Count the boundary, inconsistently wound and non-manifold edges of `indices` (triples; a
/// trailing partial triple is ignored, as [`MeshBvh::build`](crate::mesh::MeshBvh::build) does)
#[must_use]
pub fn winding_report(indices: &[u32]) -> WindingReport {
    let mut report = WindingReport::default();
    // (low, high, runs low→high): sorted, so equal edges are adjacent and the count does not
    // depend on a hash order
    let mut edges: Vec<(u32, u32, bool)> = Vec::with_capacity(indices.len());
    for t in indices.chunks_exact(3) {
        let (a, b, c) = (t[0], t[1], t[2]);
        if a == b || b == c || c == a {
            report.degenerate_triangles += 1;
            continue;
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            edges.extend([(u.min(v), u.max(v), u < v)]);
        }
    }
    edges.sort_unstable();
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        match j - i {
            1 => report.boundary_edges += 1,
            2 if edges[i].2 == edges[i + 1].2 => report.inconsistent_edges += 1,
            2 => {}
            _ => report.non_manifold_edges += 1,
        }
        i = j;
    }
    report
}

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
use crate::compiled::real::{law_max_f64, law_min_f64};
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

/// How the closed components of a triangle mesh are oriented
///
/// A consistently wound closed component is either outward (positive signed volume, winding
/// number +1 inside it) or inward (−1 inside it). An inward component is a cavity when the rest
/// of the mesh encloses it (the winding number of the other components at it is at least 1, so
/// the total stays at 0 in the hole); otherwise the winding number goes negative inside it and
/// the sign `w ≥ ½` reads that inside as outside. [`MeshSdf`](crate::mesh::MeshSdf) with the
/// winding sign refuses a mesh with an inward or undetermined component.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OrientationReport {
    /// Components (triangles joined through shared index edges) whose every edge has exactly two
    /// triangles
    pub closed_components: usize,
    /// Components with a boundary or non-manifold edge: no volume, not judged
    pub open_components: usize,
    /// Closed components wound inward that nothing encloses: the winding number is −1 inside
    pub inward_components: usize,
    /// Closed components wound inward inside another shell: holes in a solid
    pub cavities: usize,
    /// Closed components whose orientation could not be decided: a signed volume within
    /// [`VOLUME_TOLERANCE`] of 0 (relative to the cube of the component's extent), or an inward
    /// component at which no sample gave the others' winding number within ¼ of an integer
    pub undetermined_components: usize,
}

/// The relative tolerance of a closed component's signed volume: `|V| ≤ VOLUME_TOLERANCE · L³`
/// (`L` the largest side of the component's bounding box) is undetermined
pub const VOLUME_TOLERANCE: f64 = 1e-9;

/// How many of an inward component's triangles (in index order) are tried as the sample point of
/// the others' winding number
pub const ORIENTATION_SAMPLES: usize = 16;

/// Classify the closed components of `vertices` / `indices` by orientation
///
/// Components are the triangles (degenerate ones excluded, as in [`winding_report`]) joined
/// through shared index edges, numbered by their first triangle. A component is closed when
/// every edge it has belongs to exactly two of its triangles.
///
/// For a closed component the signed volume `Σ (a − o) · ((b − o) × (c − o)) / 6` is summed in
/// `f64` in triangle index order, relative to the component's first vertex `o` (so the terms are
/// of the size of the component, not of its distance from the origin); the operations are plain
/// `f64` multiplications and additions in a fixed order with no fused multiply-add, so the value
/// is the same on every target. A volume within [`VOLUME_TOLERANCE`] is undetermined.
///
/// For a closed component with a negative volume, the winding number of all other triangles is
/// taken at the centroid of its triangles in index order, up to [`ORIENTATION_SAMPLES`] of them,
/// until one is within ¼ of an integer (a centroid on another shell reads about ½ and is skipped).
/// An integer of at least 1 makes the component a cavity, below 1 inward; no usable sample leaves
/// it undetermined.
///
/// A triangle with an index out of range of `vertices` belongs to no component.
#[must_use]
pub fn orientation_report(vertices: &[Vec3], indices: &[u32]) -> OrientationReport {
    const fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let mut report = OrientationReport::default();
    let tris: Vec<[u32; 3]> = indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .filter(|t| {
            t[0] != t[1]
                && t[1] != t[2]
                && t[2] != t[0]
                && t.iter().all(|&i| (i as usize) < vertices.len())
        })
        .collect();

    // triangles sharing an edge, through the sorted edge list; an edge without exactly two
    // triangles opens every triangle on it
    let mut parent: Vec<usize> = (0..tris.len()).collect();
    let mut edges: Vec<(u32, u32, usize)> = Vec::with_capacity(3 * tris.len());
    for (k, t) in tris.iter().enumerate() {
        for (u, v) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.extend([(u.min(v), u.max(v), k)]);
        }
    }
    edges.sort_unstable();
    let mut open_edge = vec![false; tris.len()];
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j].0 == edges[i].0 && edges[j].1 == edges[i].1 {
            j += 1;
        }
        if j - i != 2 {
            for e in &edges[i..j] {
                open_edge[e.2] = true;
            }
        }
        for e in &edges[i + 1..j] {
            let (a, b) = (root(&mut parent, edges[i].2), root(&mut parent, e.2));
            if a != b {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                parent[hi] = lo;
            }
        }
        i = j;
    }

    // components in the order of their first triangle
    let label: Vec<usize> = (0..tris.len()).map(|k| root(&mut parent, k)).collect();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut slot = vec![usize::MAX; tris.len()];
    for (k, &l) in label.iter().enumerate() {
        if slot[l] == usize::MAX {
            slot[l] = members.len();
            members.extend([Vec::new()]);
        }
        members[slot[l]].extend([k]);
    }

    let pos = |i: u32| d3(vertices[i as usize]);
    for m in &members {
        if m.iter().any(|&k| open_edge[k]) {
            report.open_components += 1;
            continue;
        }
        report.closed_components += 1;
        let o = pos(tris[m[0]][0]);
        let (mut lo, mut hi) = (o, o);
        let mut six_v = 0.0f64;
        for &k in m {
            let t = tris[k];
            let (a, b, c) = (pos(t[0]), pos(t[1]), pos(t[2]));
            six_v += dot(sub(a, o), cross(sub(b, o), sub(c, o)));
            for p in [a, b, c] {
                for d in 0..3 {
                    lo[d] = law_min_f64(lo[d], p[d]);
                    hi[d] = law_max_f64(hi[d], p[d]);
                }
            }
        }
        let extent = law_max_f64(law_max_f64(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]);
        let volume = six_v / 6.0;
        if volume.abs() <= VOLUME_TOLERANCE * extent * extent * extent {
            report.undetermined_components += 1;
            continue;
        }
        if volume > 0.0 {
            continue;
        }
        let own = label[m[0]];
        let mut enclosing = None;
        for &k in m.iter().take(ORIENTATION_SAMPLES) {
            let t = tris[k];
            let p =
                (vertices[t[0] as usize] + vertices[t[1] as usize] + vertices[t[2] as usize]) / 3.0;
            let mut sum = 0.0f64;
            for (q, s) in tris.iter().enumerate() {
                if label[q] != own {
                    sum += solid_angle(
                        vertices[s[0] as usize],
                        vertices[s[1] as usize],
                        vertices[s[2] as usize],
                        p,
                    );
                }
            }
            let w = sum / FOUR_PI;
            let r = w.round();
            if (w - r).abs() < 0.25 {
                enclosing = Some(r);
                break;
            }
        }
        match enclosing {
            Some(r) if r >= 1.0 => report.cavities += 1,
            Some(_) => report.inward_components += 1,
            None => report.undetermined_components += 1,
        }
    }
    report
}

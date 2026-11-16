//! Coplanar, overlapping faces between the operands of a CSG node — `coplanar_faces`.
//!
//! # MODEL
//!
//! Two exact SDF solids that touch along a shared face give `min(a, b) = 0` on that face instead of a
//! negative value (a **union seam**); a cutter whose face lies exactly on the material's face gives
//! `max(a, −b) = 0` over the cut's mouth (a **flush cut**); two solids that only touch give
//! `max(a, b) = 0` on the contact (an **intersection contact**). In each case the field touches zero
//! with no material on one or both sides: a zero-thickness sheet that a mesher can turn into a membrane
//! and a gap measurement reads as a wall. Sampling finds such sheets only where a sample lands near them;
//! this analysis finds them from the tree, exactly where the operands' planar faces coincide.
//!
//! **Faces followed**: the six faces of `Box3d`, the two caps of `Cylinder` (as an inscribed 64-gon,
//! area within 0.16 % of the disk), the six flat faces of `RoundedBox` (its field puts them at
//! `±(h + r)`, flat over `±h` in the other two axes), the one face of `Plane` (a half-space, followed as a
//! square of half-size [`PLANE_HALF_EXTENT`]), and `Elongate` over a `Box3d` or `RoundedBox` (exactly the
//! same primitive with half extents `h + amount`; over any other child it is not analysed). A primitive
//! with a non-positive size is an empty shape and has no faces (it is analysed, with none), through `Translate`, `Rotate` (any quaternion) and `Scale` (positive
//! uniform). `Sphere`, `Torus` and `Capsule` have no planar face and are analysed (they contribute
//! none). A rotation is followed exactly whatever its axis (the face normals and vertices are turned by
//! the quaternion, so a pair that touches after any rotation is found; a pair turned by different
//! rotations is compared exactly and is not parallel). A **non-uniform scale** is not followed: it maps a
//! cylinder's cap to an ellipse and changes which faces stay parallel only through the stretch, so it is
//! counted as not analysed rather than approximated. Every other node is **not analysed**: it is counted (`ops_not_analysed` for an inner node,
//! `leaves_not_analysed` for a leaf), the faces it does have are not known, and a pair through it can
//! be missed. A report with any not-analysed node is not a clean bill ([`CoplanarReport::is_fully_analysed`]).
//! Faces already known on the other operand are still compared, so a seam inside a partly analysed tree
//! is still found.
//!
//! **Smooth operators**: `SmoothUnion` / `SmoothSubtraction` / `SmoothIntersection` move the field at a
//! coincident face by exactly `±k/4` (negative for a union: the seam becomes material; positive for a
//! subtraction or an intersection: the mouth or contact opens to air), measured for `k` from 1 down to
//! `1e-7`. With `k > SMOOTH_OPENS_K` (`4 × SURFACE_HIT_EPS`) that clears every default surface band, so
//! no pair is reported and the operands' faces pass up as for the plain operator; with a smaller `k` the
//! blend is followed exactly like the plain operator and its pairs are reported (a consumer would still
//! read the face as surface). A non-finite `k` is not analysed.
//!
//! **Pairs reported**, at each `Union` / `Subtraction` / `Intersection`, between a face of one operand
//! and a face of the other:
//!
//! * `Union`: opposite normals, same plane, overlapping extent ⇒ [`CoplanarKind::UnionSeam`]
//! * `Subtraction a − b`: a face of `b` with the same normal as a face of `a`, same plane, overlapping
//!   ⇒ [`CoplanarKind::FlushCut`]
//! * `Intersection`: opposite normals, same plane, overlapping ⇒ [`CoplanarKind::IntersectionContact`]
//!
//! Same-normal faces in a union (blocks flush side by side) are an ordinary flat exterior and are not
//! reported; opposite-normal faces in a subtraction (a cutter touching the material from outside) remove
//! nothing and are not reported.
//!
//! **Same plane**: unit normals within `1 − |n_a · n_b| ≤ COPLANAR_REL_TOL` and plane offsets within
//! `COPLANAR_REL_TOL · max(1, |d_a|, |d_b|)` (relative to the distance of the plane from the origin, so
//! the float rounding of a transformed face does not split a real coincidence). **Overlapping**: the two
//! convex faces, projected into the common plane, intersect in an area above `COPLANAR_REL_TOL` times
//! the smaller face's area (faces meeting only along an edge do not overlap).
//!
//! **Faces of a result** (what the next operator up compares): a union and an intersection keep both
//! operands' faces; a subtraction keeps `a`'s faces and `b`'s faces with their normals reversed (the
//! cutter's surface becomes the result's boundary). This over-approximates (a face hidden inside the
//! other operand is kept), so a pair can be reported where the sampled field shows no sheet; the
//! agreement with a field scan is measured by the consumer over its templates.
//!
//! Tests: `tests/test_coplanar_faces.rs` (seam, flush cavity, flush counterbore pocket, the tolerance
//! from both sides, a rotated pair and a non-parallel one, a pair turned about a general axis, flush
//! exteriors and disjoint faces, the not-analysed count, a non-uniform scale counted as not analysed).

use crate::compiled::real::LawMinMax;
use crate::types::SdfNode;
use glam::{Quat, Vec2, Vec3};
use std::collections::BTreeMap;

/// Relative tolerance for "same plane" and "overlapping" (see the module's MODEL section).
pub const COPLANAR_REL_TOL: f32 = 1e-6;

/// Smallest blend radius a smooth operator needs before it counts as opening a seam or flush cut:
/// the polynomial blend moves the field at the coincident face by exactly `k / 4` (measured for union,
/// subtraction and intersection from `k = 1` down to `1e-7`), and that has to clear the widest default
/// surface band, [`crate::raycast::SURFACE_HIT_EPS`], or a consumer still reads the face as surface.
pub const SMOOTH_OPENS_K: f32 = 4.0 * crate::raycast::SURFACE_HIT_EPS;

/// Sides of the polygon a cylinder's cap is followed as.
const CAP_SIDES: usize = 64;

/// Half-size of the square a `Plane`'s face is followed as (the half-space is unbounded; every shape
/// compared against it is far inside this square).
pub const PLANE_HALF_EXTENT: f32 = 1.0e4;

/// What kind of coincidence a [`CoplanarPair`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoplanarKind {
    /// Two solids of a union touching back to back along a face.
    UnionSeam,
    /// A cutter's face lying exactly on the material's face.
    FlushCut,
    /// Two solids of an intersection touching only along a face.
    IntersectionContact,
}

/// One coplanar, overlapping pair of faces across a CSG operator.
#[derive(Debug, Clone, PartialEq)]
pub struct CoplanarPair {
    /// What the coincidence is.
    pub kind: CoplanarKind,
    /// Unit normal of the first operand's face, in world coordinates.
    pub normal: Vec3,
    /// Centroid of the overlap, in world coordinates.
    pub point: Vec3,
    /// Area of the overlap.
    pub overlap_area: f32,
    /// The overlap itself, a convex polygon in world coordinates (counter-clockwise seen along
    /// `normal`): where the zero-thickness sheet or seam is, for a consumer that clips by it.
    pub overlap_polygon: Vec<Vec3>,
}

/// Result of [`coplanar_faces`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoplanarReport {
    /// Every coplanar, overlapping pair found.
    pub pairs: Vec<CoplanarPair>,
    /// CSG operators whose operands were both fully analysed.
    pub ops_analysed: usize,
    /// Inner nodes the analysis could not follow, and CSG operators with an operand that was not
    /// fully analysed.
    pub ops_not_analysed: usize,
    /// Leaves whose planar faces are not known.
    pub leaves_not_analysed: usize,
    /// The nodes counted in `ops_not_analysed` (those the analysis cannot follow, not the CSG operators
    /// above them) and `leaves_not_analysed`, by variant name: what to extend first, and what a
    /// per-kind ratchet freezes.
    pub not_analysed_kinds: BTreeMap<String, usize>,
}

impl CoplanarReport {
    /// Whether every node was analysed (no pair can have been missed through an unknown node).
    #[must_use]
    pub const fn is_fully_analysed(&self) -> bool {
        self.ops_not_analysed == 0 && self.leaves_not_analysed == 0
    }
}

/// A planar convex face: unit outward normal `n`, plane `n · x = d`, vertices in order.
#[derive(Clone)]
struct Face {
    n: Vec3,
    d: f32,
    verts: Vec<Vec3>,
}

/// World transform of a subtree: `x ↦ rot · (scale · x) + offset`.
#[derive(Clone, Copy)]
struct Xf {
    rot: Quat,
    scale: f32,
    offset: Vec3,
}

impl Xf {
    const IDENTITY: Self = Self {
        rot: Quat::IDENTITY,
        scale: 1.0,
        offset: Vec3::ZERO,
    };

    fn point(self, p: Vec3) -> Vec3 {
        self.rot * (p * self.scale) + self.offset
    }

    fn face(self, n_local: Vec3, verts_local: &[Vec3]) -> Face {
        let n = (self.rot * n_local).normalize();
        let verts: Vec<Vec3> = verts_local.iter().map(|&v| self.point(v)).collect();
        let d = n.dot(verts[0]);
        Face { n, d, verts }
    }
}

/// Faces found under a node, and whether every node below it was analysed.
struct Faces {
    faces: Vec<Face>,
    complete: bool,
}

/// Every coplanar, overlapping pair of faces across the CSG operators of `node`.
#[must_use]
pub fn coplanar_faces(node: &SdfNode) -> CoplanarReport {
    let mut report = CoplanarReport::default();
    let _ = collect(node, Xf::IDENTITY, &mut report);
    report
}

fn box_faces(h: Vec3, xf: Xf) -> Vec<Face> {
    flat_faces(h, Vec3::ZERO, xf)
}

/// The flat faces of a box of half extents `h` grown by `r` along each face normal (a rounded box: the
/// face of axis `i` lies at `±(h_i + r)` and is flat over `±h` in the other axes). A face whose flat
/// part has no area is left out; an empty shape (a non-positive or non-finite size) has none.
fn flat_faces(h: Vec3, r: Vec3, xf: Xf) -> Vec<Face> {
    let mut out = Vec::with_capacity(6);
    let sizes_ok = h.is_finite() && r.is_finite() && !r.cmplt(Vec3::ZERO).any();
    if !sizes_ok || !(h + r).cmpgt(Vec3::ZERO).all() {
        return out;
    }
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            if !(h[u] > 0.0 && h[v] > 0.0) {
                continue;
            }
            let mut n = Vec3::ZERO;
            n[axis] = sign;
            let corner = |su: f32, sv: f32| {
                let mut p = Vec3::ZERO;
                p[axis] = sign * (h[axis] + r[axis]);
                p[u] = su * h[u];
                p[v] = sv * h[v];
                p
            };
            let verts = [
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ];
            out.push(xf.face(n, &verts));
        }
    }
    out
}

fn cylinder_caps(radius: f32, half_height: f32, xf: Xf) -> Vec<Face> {
    if !(radius > 0.0 && half_height > 0.0 && radius.is_finite() && half_height.is_finite()) {
        return Vec::new();
    }
    [-1.0_f32, 1.0]
        .iter()
        .map(|&sign| {
            let verts: Vec<Vec3> = (0..CAP_SIDES)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss)] // k < 64
                    let a = core::f32::consts::TAU * k as f32 / CAP_SIDES as f32;
                    let (s, c) = alice_det_math::sin_cos(a);
                    Vec3::new(radius * c, sign * half_height, radius * s)
                })
                .collect();
            xf.face(Vec3::Y * sign, &verts)
        })
        .collect()
}

/// The one face of the half-space `normal · p < distance`, as a square of half-size
/// [`PLANE_HALF_EXTENT`]; `None` for a normal of zero or non-finite length.
fn plane_face(normal: Vec3, distance: f32, xf: Xf) -> Option<Face> {
    let len = normal.length();
    if !(len > 0.0 && len.is_finite() && distance.is_finite()) {
        return None;
    }
    let n = normal / len;
    let c = n * (distance / len);
    let u = n.any_orthonormal_vector() * PLANE_HALF_EXTENT;
    let v = n.cross(u.normalize()) * PLANE_HALF_EXTENT;
    Some(xf.face(n, &[c - u - v, c + u - v, c + u + v, c - u + v]))
}

fn collect(node: &SdfNode, xf: Xf, report: &mut CoplanarReport) -> Faces {
    match node {
        SdfNode::Box3d { half_extents } => Faces {
            faces: box_faces(*half_extents, xf),
            complete: true,
        },
        SdfNode::Cylinder {
            radius,
            half_height,
        } => Faces {
            faces: cylinder_caps(*radius, *half_height, xf),
            complete: true,
        },
        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => Faces {
            faces: flat_faces(*half_extents, Vec3::splat(*round_radius), xf),
            complete: true,
        },
        SdfNode::Plane { normal, distance } => match plane_face(*normal, *distance, xf) {
            Some(face) => Faces {
                faces: vec![face],
                complete: true,
            },
            None => not_analysed(node, report),
        },
        SdfNode::Elongate { child, amount } if amount.is_finite() && !amount.cmplt(Vec3::ZERO).any() => {
            match child.as_ref() {
                SdfNode::Box3d { half_extents } => Faces {
                    faces: flat_faces(*half_extents + *amount, Vec3::ZERO, xf),
                    complete: true,
                },
                SdfNode::RoundedBox {
                    half_extents,
                    round_radius,
                } => Faces {
                    faces: flat_faces(*half_extents + *amount, Vec3::splat(*round_radius), xf),
                    complete: true,
                },
                _ => not_analysed(node, report),
            }
        }
        SdfNode::Sphere { .. } | SdfNode::Torus { .. } | SdfNode::Capsule { .. } => Faces {
            faces: Vec::new(),
            complete: true,
        },
        SdfNode::Translate { child, offset } => collect(
            child,
            Xf {
                offset: xf.point(*offset),
                ..xf
            },
            report,
        ),
        SdfNode::Rotate { child, rotation } => collect(
            child,
            Xf {
                rot: xf.rot * *rotation,
                ..xf
            },
            report,
        ),
        SdfNode::Scale { child, factor } if factor.is_finite() && *factor > 0.0 => collect(
            child,
            Xf {
                scale: xf.scale * *factor,
                ..xf
            },
            report,
        ),
        SdfNode::Union { a, b } => binary(a, b, Op::Union, true, xf, report),
        SdfNode::Subtraction { a, b } => binary(a, b, Op::Subtraction, true, xf, report),
        SdfNode::Intersection { a, b } => binary(a, b, Op::Intersection, true, xf, report),
        SdfNode::SmoothUnion { a, b, k } if k.is_finite() => {
            binary(a, b, Op::Union, *k <= SMOOTH_OPENS_K, xf, report)
        }
        SdfNode::SmoothSubtraction { a, b, k } if k.is_finite() => {
            binary(a, b, Op::Subtraction, *k <= SMOOTH_OPENS_K, xf, report)
        }
        SdfNode::SmoothIntersection { a, b, k } if k.is_finite() => {
            binary(a, b, Op::Intersection, *k <= SMOOTH_OPENS_K, xf, report)
        }
        other => not_analysed(other, report),
    }
}

/// A node the analysis does not follow: its own faces are unknown and its children are not visited
/// (their faces would be reshaped by it); counted by kind.
fn not_analysed(node: &SdfNode, report: &mut CoplanarReport) -> Faces {
    *report
        .not_analysed_kinds
        .entry(variant_name(node))
        .or_insert(0) += 1;
    if crate::diff::node_children(node).is_empty() {
        report.leaves_not_analysed += 1;
    } else {
        report.ops_not_analysed += 1;
    }
    Faces {
        faces: Vec::new(),
        complete: false,
    }
}

#[derive(Clone, Copy)]
enum Op {
    Union,
    Subtraction,
    Intersection,
}

/// `report_pairs` is false for a smooth operator whose blend opens any coincidence (its faces still pass
/// up, over-approximated as for the plain operator).
fn binary(
    a: &SdfNode,
    b: &SdfNode,
    op: Op,
    report_pairs: bool,
    xf: Xf,
    report: &mut CoplanarReport,
) -> Faces {
    let fa = collect(a, xf, report);
    let fb = collect(b, xf, report);
    let complete = fa.complete && fb.complete;
    if complete {
        report.ops_analysed += 1;
    } else {
        report.ops_not_analysed += 1;
    }
    for x in fa.faces.iter().filter(|_| report_pairs) {
        for y in &fb.faces {
            let dot = x.n.dot(y.n);
            let (kind, same_sign) = match op {
                Op::Union => (CoplanarKind::UnionSeam, false),
                Op::Subtraction => (CoplanarKind::FlushCut, true),
                Op::Intersection => (CoplanarKind::IntersectionContact, false),
            };
            let parallel = if same_sign {
                1.0 - dot <= COPLANAR_REL_TOL
            } else {
                1.0 + dot <= COPLANAR_REL_TOL
            };
            if !parallel {
                continue;
            }
            // y's plane as an offset along x's normal
            let dy = if same_sign { y.d } else { -y.d };
            let scale = 1.0_f32.law_max(x.d.abs()).law_max(y.d.abs());
            if (x.d - dy).abs() > COPLANAR_REL_TOL * scale {
                continue;
            }
            if let Some((area, point, polygon)) = overlap(x, y) {
                report.pairs.push(CoplanarPair {
                    kind,
                    normal: x.n,
                    point,
                    overlap_area: area,
                    overlap_polygon: polygon,
                });
            }
        }
    }
    let mut faces = fa.faces;
    match op {
        Op::Union | Op::Intersection => faces.extend(fb.faces),
        Op::Subtraction => faces.extend(fb.faces.into_iter().map(|f| Face {
            n: -f.n,
            d: -f.d,
            verts: f.verts,
        })),
    }
    Faces { faces, complete }
}

/// Signed area of a 2D polygon (positive when counter-clockwise).
fn signed_area(p: &[Vec2]) -> f32 {
    let mut s = 0.0;
    for i in 0..p.len() {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        s += a.x.mul_add(b.y, -(b.x * a.y));
    }
    0.5 * s
}

/// The polygon `subject` clipped to the convex counter-clockwise polygon `clip` (Sutherland–Hodgman).
fn clip_convex(subject: &[Vec2], clip: &[Vec2]) -> Vec<Vec2> {
    let mut out = subject.to_vec();
    for i in 0..clip.len() {
        if out.is_empty() {
            break;
        }
        let (c0, c1) = (clip[i], clip[(i + 1) % clip.len()]);
        let edge = c1 - c0;
        let inside = |p: Vec2| edge.perp_dot(p - c0) >= 0.0;
        let input = std::mem::take(&mut out);
        for j in 0..input.len() {
            let (p, q) = (input[j], input[(j + 1) % input.len()]);
            let (pi, qi) = (inside(p), inside(q));
            if pi {
                out.push(p);
            }
            if pi != qi {
                let denom = edge.perp_dot(q - p);
                if denom != 0.0 {
                    let t = edge.perp_dot(c0 - p) / denom;
                    out.push(p + (q - p) * t);
                }
            }
        }
    }
    out
}

/// Area, world centroid and world polygon of the overlap of two coplanar faces, `None` when it is not
/// above the tolerance (disjoint, or meeting along an edge).
fn overlap(x: &Face, y: &Face) -> Option<(f32, Vec3, Vec<Vec3>)> {
    let n = x.n;
    let u = n.any_orthonormal_vector();
    let v = n.cross(u);
    let origin = n * x.d;
    let to2 = |p: Vec3| Vec2::new((p - origin).dot(u), (p - origin).dot(v));
    let ccw = |verts: &[Vec3]| {
        let mut p: Vec<Vec2> = verts.iter().map(|&q| to2(q)).collect();
        if signed_area(&p) < 0.0 {
            p.reverse();
        }
        p
    };
    let (px, py) = (ccw(&x.verts), ccw(&y.verts));
    let ax = signed_area(&px);
    let ay = signed_area(&py);
    // clip the smaller face against the larger: the result's vertices then come from the small face's
    // edges, not from far-away edges of a large one (a plane's square), which keeps the f32 precision
    let inter = if ax <= ay {
        clip_convex(&px, &py)
    } else {
        clip_convex(&py, &px)
    };
    if inter.len() < 3 {
        return None;
    }
    let area = signed_area(&inter);
    if !(area > COPLANAR_REL_TOL * ax.law_min(ay)) {
        return None;
    }
    // area centroid of the overlap polygon
    let mut c = Vec2::ZERO;
    for i in 0..inter.len() {
        let (a, b) = (inter[i], inter[(i + 1) % inter.len()]);
        let w = a.perp_dot(b);
        c += (a + b) * w;
    }
    c /= 6.0 * area;
    let to3 = |q: Vec2| origin + u * q.x + v * q.y;
    Some((area, to3(c), inter.iter().map(|&q| to3(q)).collect()))
}

/// The variant name of a node (`"SmoothUnion"`, `"RoundedBox"`, ...), read from its `Debug` output
/// without formatting the subtree: the writer stops at the first character after the name.
fn variant_name(node: &SdfNode) -> String {
    struct Name(String);
    impl core::fmt::Write for Name {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for c in s.chars() {
                if c.is_alphanumeric() || c == '_' {
                    self.0.push(c);
                } else {
                    return Err(core::fmt::Error);
                }
            }
            Ok(())
        }
    }
    let mut name = Name(String::new());
    let _ = core::fmt::write(&mut name, format_args!("{node:?}"));
    name.0
}

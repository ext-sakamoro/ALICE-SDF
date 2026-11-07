//! First hit of a ray on a mesh thickened by a radius `r`
//!
//! The thickened triangle is the set of points within `r` of the triangle:
//! a prism over the face (two offset planes), a cylinder around each edge and a
//! sphere at each vertex. The first hit of a ray is the smallest `t >= 0` on any
//! of these pieces that lies on the boundary of the union.
//!
//! - A ray that starts inside the shell (`dist(o) <= r`) hits at `t = 0` with
//!   [`ShellFeature::Inside`].
//! - A tangent (a ray that touches a cylinder or sphere at one point) counts as a hit.
//! - Ties inside one triangle keep the earlier piece in the order face, edge,
//!   vertex; ties between triangles keep the lower triangle index, so the BVH walk
//!   and a brute-force scan return the same triangle.
//! - A degenerate triangle (zero area, collinear, a point) has no face; its edges
//!   and vertices cover it.
//!
//! Quadratic roots are solved about the parameter of closest approach
//! (`t0 = -(w0.g)/(g.g)`), so the discriminant is `(g.g)(rr - w.w)` with a small
//! `w`; the textbook `b^2 - ac` cancels in f32 when a ray runs nearly along an
//! edge far from its origin (see the grazing cases in the tests).
//!
//! Author: Moroya Sakamoto

use super::bvh::{BvhNode, BvhTriangle, MeshBvh};
use glam::Vec3;

/// Which piece of the thickened triangle the ray met first
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ShellFeature {
    /// One of the two planes offset from the face by `r`
    Face = 1,
    /// The cylinder of radius `r` around an edge
    Edge = 2,
    /// The sphere of radius `r` around a vertex
    Vertex = 3,
    /// The ray starts within `r` of the triangle
    Inside = 4,
}

/// First hit of a ray on the thickened mesh
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellHit {
    /// Ray parameter of the hit (the hit point is `origin + t * dir`): `t_base + t_local`,
    /// rounded once; the pair keeps the precision that one f32 far from the origin loses
    pub t: f32,
    /// The part of `t` that moves the ray to its frame (see [`ShellFrame`]); 0 in the world frame
    pub t_base: f32,
    /// The part of `t` solved in the frame, from the moved origin
    pub t_local: f32,
    /// Index into [`MeshBvh::triangles`]
    pub triangle: u32,
    /// Piece of that triangle the ray met
    pub feature: ShellFeature,
}

/// A ray `origin + t * dir`, `0 <= t <= tmax`; `dir` need not be unit length
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShellRay {
    /// Start
    pub origin: Vec3,
    /// Direction
    pub dir: Vec3,
    /// Largest `t` a hit may have
    pub tmax: f32,
}

/// Smallest root `t >= 0` of `|w0 + t g|^2 = rr`, solved about the closest approach
#[inline]
fn shifted_root(w0: Vec3, g: Vec3, rr: f32) -> Option<f32> {
    let a = g.dot(g);
    if a == 0.0 {
        return None;
    }
    let t0 = -w0.dot(g) / a;
    let w = w0 + g * t0;
    let h = rr - w.dot(w);
    if h < 0.0 {
        return None;
    }
    let dt = (h / a).sqrt();
    let (lo, hi) = (t0 - dt, t0 + dt);
    if lo >= 0.0 {
        Some(lo)
    } else if hi >= 0.0 {
        Some(hi)
    } else {
        None
    }
}

#[inline]
fn sphere_t(o: Vec3, d: Vec3, c: Vec3, r: f32) -> Option<f32> {
    shifted_root(o - c, d, r * r)
}

/// Hit on the finite cylinder around the edge `u -> v`
#[inline]
fn cylinder_t(o: Vec3, d: Vec3, u: Vec3, v: Vec3, r: f32, root: RootFn) -> Option<f32> {
    // distance to the edge's line is |(m + t d) x e| / |e|, so solve
    // |m x e + t (d x e)|^2 = r^2 |e|^2
    // one order for an edge, whichever triangle names it: a shared edge gives the same bits in both
    let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
    let (u, v) = if key(v) < key(u) { (v, u) } else { (u, v) };
    let e = v - u;
    let l2 = e.dot(e);
    if l2 == 0.0 {
        return None;
    }
    let m = o - u;
    let t = root(m.cross(e), d.cross(e), r * r * l2)?;
    let ax = (m + d * t).dot(e);
    if !(0.0..=l2).contains(&ax) {
        return None;
    }
    Some(t)
}

// Error-free transforms. A product's rounding error comes from Dekker's split (Veltkamp,
// 4097 = 2^12 + 1 for f32), not from a fused multiply-add: `+ - *` are correctly rounded in WGSL
// and the shader path keeps them unfused, so the CPU and the GPU form the same bits. They are
// exact while every intermediate value is zero or a normal f32, since a GPU may flush subnormals
// to zero.
//
// The plane distance keeps them there by construction: a difference component or its remainder
// below `FLUSH_W` (2^-40), and a normal component below `FLUSH_N` (2^-38), is set to exactly 0, on
// the CPU and in WGSL alike. A nonzero lo part of Dekker's split has |lo(x)| >= ulp(x) >= 2^-24 |x|,
// so every partial product is at least 2^-48 |w_i nu_i| >= 2^-126 once every nonzero
// |w_i nu_i| >= 2^-78, and every TwoSum remainder is a multiple of an operand's ulp above 2^-102:
// all normal, for any mesh and any ray. The flush moves the distance by at most
// (|w| 2^-38 + |nu| 2^-40) / |nu|, about 2e-10 mm over a 60 mm frame.
// Measured on a figure (4 views, r .002 and .004 mm, world and leaf frames), without the flush,
// the smallest intermediate was 5.0e-16 (a lo * lo), |anchor - o| 7.66e-3 and a normal component
// 1.37e-6 mm: the flush does not change those results.

/// `a + b = s + e` exactly
#[inline]
pub(crate) fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a - b = s + e` exactly
#[inline]
pub(crate) fn two_diff(a: f32, b: f32) -> (f32, f32) {
    let s = a - b;
    let bb = s - a;
    (s, (a - (s - bb)) + (-b - bb))
}

#[inline]
fn split(a: f32) -> (f32, f32) {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}

/// `a * b = p + e` exactly (Dekker)
#[inline]
pub(crate) fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// Below this a difference component or its remainder is 0 (see the comment above `two_sum`)
const FLUSH_W: f32 = 1.0 / 1_099_511_627_776.0; // 2^-40
/// Below this a normal component is 0
const FLUSH_N: f32 = 1.0 / 274_877_906_944.0; // 2^-38

#[inline]
fn flush(x: f32, below: f32) -> f32 {
    if x.abs() < below {
        0.0
    } else {
        x
    }
}

/// `(a - o) . n` for an unnormalised `n`: the difference exact per component, then Dot2 (Ogita,
/// Rump, Oishi) of it with `n`, plus the differences' remainders against `n`; the flush keeps every
/// intermediate normal
#[inline]
fn plane_distance(a: Vec3, o: Vec3, n: Vec3) -> f32 {
    let w = |x: f32, y: f32| {
        let (s, e) = two_diff(x, y);
        (flush(s, FLUSH_W), flush(e, FLUSH_W))
    };
    let (wx, ex) = w(a.x, o.x);
    let (wy, ey) = w(a.y, o.y);
    let (wz, ez) = w(a.z, o.z);
    let n = Vec3::new(
        flush(n.x, FLUSH_N),
        flush(n.y, FLUSH_N),
        flush(n.z, FLUSH_N),
    );
    let (p0, q0) = two_prod(wx, n.x);
    let (h1, r1) = two_prod(wy, n.y);
    let (p1, s1) = two_sum(p0, h1);
    let (h2, r2) = two_prod(wz, n.z);
    let (p2, s2) = two_sum(p1, h2);
    p2 + (q0 + r1 + s1 + r2 + s2 + (ex * n.x + ey * n.y + ez * n.z))
}

/// Whether every intermediate value of `plane_distance(a, o, n)` (after its flush) is zero or a
/// normal f32, so a GPU that flushes subnormals forms the same bits as the CPU
#[cfg(test)]
pub(crate) fn plane_distance_in_domain(a: Vec3, o: Vec3, n: Vec3) -> bool {
    plane_distance_in_domain_with(a, o, n, true)
}

/// The same check with the flush on or off (off shows the inputs it protects against)
#[cfg(test)]
pub(crate) fn plane_distance_in_domain_with(a: Vec3, o: Vec3, n: Vec3, flushed: bool) -> bool {
    let ok = |v: f32| v == 0.0 || v.abs() >= f32::MIN_POSITIVE;
    let (fw, fnn) = if flushed {
        (FLUSH_W, FLUSH_N)
    } else {
        (0.0, 0.0)
    };
    let mut vals = vec![a.x, a.y, a.z, o.x, o.y, o.z];
    let diff = |x: f32, y: f32, vals: &mut Vec<f32>| {
        let s = x - y;
        let bb = s - x;
        vals.extend([s, bb, s - bb, x - (s - bb), -y - bb]);
        let (s, e) = two_diff(x, y);
        (flush(s, fw), flush(e, fw))
    };
    let (wx, ex) = diff(a.x, o.x, &mut vals);
    let (wy, ey) = diff(a.y, o.y, &mut vals);
    let (wz, ez) = diff(a.z, o.z, &mut vals);
    let n = Vec3::new(flush(n.x, fnn), flush(n.y, fnn), flush(n.z, fnn));
    vals.extend([n.x, n.y, n.z, wx, wy, wz, ex, ey, ez]);
    let prod = |x: f32, y: f32, vals: &mut Vec<f32>| {
        let p = x * y;
        let (xh, xl) = split(x);
        let (yh, yl) = split(y);
        vals.extend([
            4097.0 * x,
            4097.0 * y,
            xh,
            xl,
            yh,
            yl,
            p,
            xh * yh,
            xh * yh - p,
            xh * yl,
            xl * yh,
            xl * yl,
        ]);
        vals.push((xh * yh - p) + xh * yl);
        vals.push(((xh * yh - p) + xh * yl) + xl * yh);
        two_prod(x, y)
    };
    let (p0, q0) = prod(wx, n.x, &mut vals);
    let (h1, r1) = prod(wy, n.y, &mut vals);
    let (p1, s1) = two_sum(p0, h1);
    let (h2, r2) = prod(wz, n.z, &mut vals);
    let (p2, s2) = two_sum(p1, h2);
    vals.extend([p0, q0, h1, r1, p1, s1, h2, r2, p2, s2, p1 - p0, p2 - p1]);
    vals.extend([ex * n.x, ey * n.y, ez * n.z, ex * n.x + ey * n.y]);
    vals.extend([
        q0 + r1,
        q0 + r1 + s1,
        q0 + r1 + s1 + r2,
        q0 + r1 + s1 + r2 + s2,
    ]);
    vals.into_iter().all(ok)
}

/// The anchor vertex and the unnormalised normal of a triangle: the cross of its two shortest edges,
/// at the vertex opposite the longest (the largest angle), which keeps the normal well conditioned
/// on a long thin triangle. The longest edge is chosen by squared lengths in a fixed order, a tie
/// going to the lower bits of the opposite vertex; each pair has the orientation of `(b-a) x (c-a)`.
#[inline]
fn face_frame(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
    // edge k is opposite vertex (c, a, b)[k]
    let cand = [
        ((b - a).dot(b - a), key(c), 0u8),
        ((c - b).dot(c - b), key(a), 1),
        ((a - c).dot(a - c), key(b), 2),
    ];
    let longest = cand[1..].iter().fold(cand[0], |m, x| {
        if x.0 > m.0 || (x.0 == m.0 && x.1 < m.1) {
            *x
        } else {
            m
        }
    });
    match longest.2 {
        0 => (c, (a - c).cross(b - c)),
        1 => (a, (b - a).cross(c - a)),
        _ => (b, (c - b).cross(a - b)),
    }
}

/// The edge function of `u -> v` at `q` against `nu`, evaluated in one direction per edge (lower
/// vertex bits first, the sign turned back), so a shared edge gives the same value from both
/// triangles
#[inline]
fn edge_function(u: Vec3, v: Vec3, q: Vec3, nu: Vec3) -> f32 {
    let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
    if key(u) <= key(v) {
        (v - u).cross(q - u).dot(nu)
    } else {
        -((u - v).cross(q - v).dot(nu))
    }
}

/// Hit on either plane offset from the face by `r`, inside the triangle's prism
///
/// The plane: the shortest-edge normal at its anchor (`face_frame`), and the signed distance of `o`
/// to it in compensated arithmetic (`plane_distance`), `r` added after on that small value. Over the
/// triangle: the three edge functions against the unnormalised normal, no division, so a sliver or
/// near-collinear triangle (whose barycentric weights would divide by a Gram determinant near 0)
/// accepts only points over itself.
#[inline]
fn face_t(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let (anchor, nu) = face_frame(a, b, c);
    let nl = nu.length();
    if nl == 0.0 || !nl.is_finite() {
        return None;
    }
    // t = (P +- r |nu|) / D with P = (anchor - o) . nu and D = d . nu, both from + - * alone (the
    // same bits on a GPU): t carries one square root (in |nu|) and one division
    let dd = d.dot(nu);
    if dd == 0.0 {
        return None;
    }
    let p = plane_distance(anchor, o, nu);
    let rn = r * nl;
    // the offset of the plane along nu, one division shared by the three components
    let off = r / nl;
    let mut best: Option<f32> = None;
    for sg in [1.0f32, -1.0] {
        let t = (p + sg * rn) / dd;
        if t < 0.0 {
            continue;
        }
        let q = o + d * t - nu * (sg * off);
        let over = edge_function(a, b, q, nu) >= 0.0
            && edge_function(b, c, q, nu) >= 0.0
            && edge_function(c, a, q, nu) >= 0.0;
        if over {
            best = Some(best.map_or(t, |x| x.min(t)));
        }
    }
    best
}

type RootFn = fn(Vec3, Vec3, f32) -> Option<f32>;

fn triangle_hit_with(
    o: Vec3,
    d: Vec3,
    tri: &BvhTriangle,
    r: f32,
    root: RootFn,
) -> Option<(f32, ShellFeature)> {
    if tri.unsigned_distance(o) <= r {
        return Some((0.0, ShellFeature::Inside));
    }
    let mut best: Option<(f32, ShellFeature)> = None;
    let mut put = |t: Option<f32>, f: ShellFeature| {
        if let Some(t) = t {
            // strict: on a tie the earlier piece (face, edge, vertex) stays
            if best.is_none_or(|(b, _)| t < b) {
                best = Some((t, f));
            }
        }
    };
    put(face_t(o, d, tri.v0, tri.v1, tri.v2, r), ShellFeature::Face);
    for (u, v) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)] {
        put(cylinder_t(o, d, u, v, r, root), ShellFeature::Edge);
    }
    for c in [tri.v0, tri.v1, tri.v2] {
        put(sphere_t(o, d, c, r), ShellFeature::Vertex);
    }
    best
}

/// First hit of the ray `o + t d` on one triangle thickened by `r`
///
/// `d` need not be unit length; `t` is in units of `d`.
pub fn triangle_shell_hit(
    o: Vec3,
    d: Vec3,
    tri: &BvhTriangle,
    r: f32,
) -> Option<(f32, ShellFeature)> {
    triangle_hit_with(o, d, tri, r, shifted_root)
}

/// Entry parameter of the ray into the box `[lo, hi]`, clamped to 0
#[inline]
fn slab_t(lo: Vec3, hi: Vec3, o: Vec3, inv: Vec3) -> Option<f32> {
    let a = (lo - o) * inv;
    let b = (hi - o) * inv;
    let t0 = a.min(b).max_element().max(0.0);
    let t1 = a.max(b).min_element();
    if t1 >= t0 {
        Some(t0)
    } else {
        None
    }
}

#[inline]
fn take(
    best: &mut Option<ShellHit>,
    t_base: f32,
    t_local: f32,
    triangle: u32,
    feature: ShellFeature,
) {
    let t = t_base + t_local;
    let better = match best {
        None => true,
        Some(b) => t < b.t || (t == b.t && triangle < b.triangle),
    };
    if better {
        *best = Some(ShellHit {
            t,
            t_base,
            t_local,
            triangle,
            feature,
        });
    }
}

/// Where a triangle's hit is solved
///
/// Far from the world origin an f32 coordinate carries an ulp of that distance, larger than a thin
/// shell's tolerance. The triangle frame moves the triangle and the ray near the centre `c` of the
/// triangle's box: the triangle as `v - c` and the ray's origin advanced to `t_base`, the entry of
/// that box inflated by `r`, with every rounding error of `o + d t_base - c` carried
/// (`frame_origin`). The hit is then `t_base + t_local`.
///
/// Measured on 60 thin triangles about 1400 mm out (`far_thin`, rays from 900-1400 mm back), the
/// worst distance of a hit from the shell was 8.3 % of `r = 0.002` unframed and 0.18 % in the
/// triangle frame; a BVH leaf's box as the frame gave 1.1 % and was slower on the GPU, where the
/// triangle box's entry also skips a triangle the ray misses.
///
/// Only the triangle frame is public (through [`MeshBvh::shell_first_hit`]); the unframed one is
/// kept inside the crate as the reference the tests compare against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShellFrame {
    /// No frame: the triangle and the ray as given (the unframed reference, not public)
    World,
    /// The triangle's own box: its centre, and the entry of that box (the default)
    Triangle,
}

/// `o + d t - c` with every rounding error carried: the difference exact (TwoSum), the product
/// exact (Dekker), their sum exact, the remainders added last. Each component's difference,
/// product and remainders below `FLUSH_W` are 0, so every intermediate stays normal.
#[inline]
pub(crate) fn frame_origin(o: Vec3, d: Vec3, c: Vec3, t: f32) -> Vec3 {
    let comp = |o: f32, c: f32, d: f32| {
        let (s, e) = two_diff(o, c);
        let (s, e) = (flush(s, FLUSH_W), flush(e, FLUSH_W));
        let (p, pe) = two_prod(flush(d, FLUSH_W), t);
        let (p, pe) = (flush(p, FLUSH_W), flush(pe, FLUSH_W));
        let (x, xe) = two_sum(p, s);
        x + (xe + pe + e)
    };
    Vec3::new(
        comp(o.x, c.x, d.x),
        comp(o.y, c.y, d.y),
        comp(o.z, c.z, d.z),
    )
}

/// The centre of a box, as the GPU's flattened BVH stores it
#[inline]
fn centre(min: Vec3, max: Vec3) -> Vec3 {
    (min + max) * 0.5
}

struct Walk<'a> {
    tris: &'a [BvhTriangle],
    frame: ShellFrame,
    o: Vec3,
    d: Vec3,
    inv: Vec3,
    r: f32,
    tmax: f32,
}

impl Walk<'_> {
    fn visit(&self, n: &BvhNode, best: &mut Option<ShellHit>) {
        match n {
            BvhNode::Leaf { triangles, .. } => {
                let ra = Vec3::splat(self.r);
                for &k in triangles {
                    let tri = &self.tris[k];
                    let (c, t_base) = match self.frame {
                        ShellFrame::World => (Vec3::ZERO, 0.0),
                        ShellFrame::Triangle => (
                            centre(tri.aabb.min, tri.aabb.max),
                            match slab_t(tri.aabb.min - ra, tri.aabb.max + ra, self.o, self.inv) {
                                Some(t) => t,
                                // the ray misses this triangle's inflated box: no hit on it
                                None => continue,
                            },
                        ),
                    };
                    let hit = if self.frame == ShellFrame::World {
                        triangle_shell_hit(self.o, self.d, tri, self.r)
                    } else {
                        let local = BvhTriangle::new(tri.v0 - c, tri.v1 - c, tri.v2 - c);
                        let oa = frame_origin(self.o, self.d, c, t_base);
                        triangle_shell_hit(oa, self.d, &local, self.r)
                    };
                    if let Some((t, f)) = hit {
                        if t_base + t <= self.tmax {
                            take(best, t_base, t, k as u32, f);
                        }
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                let ra = Vec3::splat(self.r);
                let enter =
                    |c: &BvhNode| slab_t(c.aabb().min - ra, c.aabb().max + ra, self.o, self.inv);
                let (tl, tr) = (enter(left), enter(right));
                let mut order = [(tl, &**left), (tr, &**right)];
                // nearer box first; a box the ray misses goes last
                match (tl, tr) {
                    (Some(x), Some(y)) if y < x => order.swap(0, 1),
                    (None, _) => order.swap(0, 1),
                    _ => {}
                }
                for (t, c) in order {
                    if let Some(t) = t {
                        // `<=`, not `<`: a box entered at the current best t can still
                        // hold a lower-index triangle at the same t
                        let limit = best.map_or(self.tmax, |b| b.t);
                        if t <= limit {
                            self.visit(c, best);
                        }
                    }
                }
            }
        }
    }
}

impl MeshBvh {
    /// First hit of the ray `origin + t * dir` (`0 <= t <= tmax`) on the mesh
    /// thickened by `r`: the boundary of the set of points within `r` of a triangle
    ///
    /// Ties at the same `t` go to the lower triangle index, as a brute-force scan
    /// over [`Self::triangles`] would give. `dir` need not be unit length.
    ///
    /// Each triangle's hit is solved near the triangle (see [`ShellFrame`]): the answer is
    /// [`ShellHit::t_base`] plus [`ShellHit::t_local`], and that pair keeps the precision one f32
    /// `t` far from the origin loses.
    pub fn shell_first_hit(&self, origin: Vec3, dir: Vec3, r: f32, tmax: f32) -> Option<ShellHit> {
        self.shell_first_hit_in(ShellFrame::Triangle, origin, dir, r, tmax)
    }

    /// [`Self::shell_first_hit`], solved in the given frame
    pub(crate) fn shell_first_hit_in(
        &self,
        frame: ShellFrame,
        origin: Vec3,
        dir: Vec3,
        r: f32,
        tmax: f32,
    ) -> Option<ShellHit> {
        let root = self.root.as_ref()?;
        let walk = Walk {
            tris: &self.triangles,
            frame,
            o: origin,
            d: dir,
            inv: dir.recip(),
            r,
            tmax,
        };
        let ra = Vec3::splat(r);
        slab_t(root.aabb().min - ra, root.aabb().max + ra, origin, walk.inv)?;
        let mut best = None;
        walk.visit(root, &mut best);
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The textbook `b^2 - ac`, kept only to show the grazing cases have teeth
    #[allow(clippy::suspicious_operation_groupings)]
    fn naive_root(w0: Vec3, g: Vec3, rr: f32) -> Option<f32> {
        let a = g.dot(g);
        if a == 0.0 {
            return None;
        }
        let b = w0.dot(g);
        let c = w0.dot(w0) - rr;
        let disc = b * b - a * c;
        if disc < 0.0 {
            return None;
        }
        let q = disc.sqrt();
        let (lo, hi) = ((-b - q) / a, (-b + q) / a);
        if lo >= 0.0 {
            Some(lo)
        } else if hi >= 0.0 {
            Some(hi)
        } else {
            None
        }
    }

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z)
    }

    type Case<'a> = (
        &'a str,
        Vec3,
        Vec3,
        &'a BvhTriangle,
        Option<(f32, ShellFeature)>,
    );

    fn close(
        got: Option<(f32, ShellFeature)>,
        want: Option<(f32, ShellFeature)>,
        tol: f32,
    ) -> bool {
        match (got, want) {
            (Some((t, f)), Some((w, g))) => (t - w).abs() < tol && f == g,
            (None, None) => true,
            _ => false,
        }
    }

    #[test]
    fn closed_form_cases() {
        use ShellFeature::*;
        let r = 0.5;
        let tri = BvhTriangle::new(v(0., 0., 0.), v(4., 0., 0.), v(0., 4., 0.));
        let col = BvhTriangle::new(v(0., 0., 0.), v(2., 0., 0.), v(4., 0., 0.));
        let pt = BvhTriangle::new(v(1., 1., 1.), v(1., 1., 1.), v(1., 1., 1.));
        let zl = BvhTriangle::new(v(0., 0., 0.), v(0., 0., 0.), v(4., 0., 0.));
        let cyl = 5.0 - (0.25f32 - 0.09).sqrt();
        #[rustfmt::skip]
        let cases: [Case<'_>; 15] = [
            ("face from above", v(1., 1., 5.), v(0., 0., -1.), &tri, Some((4.5, Face))),
            ("face from below", v(1., 1., -5.), v(0., 0., 1.), &tri, Some((4.5, Face))),
            ("edge cylinder", v(2., -5., 0.3), v(0., 1., 0.), &tri, Some((cyl, Edge))),
            ("vertex sphere", v(-5., -5., 0.), v(1., 1., 0.).normalize(), &tri, Some((50f32.sqrt() - 0.5, Vertex))),
            ("miss", v(10., 10., 5.), v(0., 0., -1.), &tri, None),
            ("inside start", v(1., 1., 0.2), v(0., 0., 1.), &tri, Some((0.0, Inside))),
            ("collinear (capsule)", v(3., 0., 5.), v(0., 0., -1.), &col, Some((4.5, Edge))),
            ("point (sphere)", v(1., 1., 6.), v(0., 0., -1.), &pt, Some((4.5, Vertex))),
            ("tangent to an edge cylinder", v(2., -0.5, 5.), v(0., 0., -1.), &tri, Some((5.0, Edge))),
            ("exactly over an edge", v(2., 0., 5.), v(0., 0., -1.), &tri, Some((4.5, Face))),
            // the crossing edge cylinder and the vertex sphere tie at 4.5; the edge is earlier
            ("along an edge", v(-5., 0., 0.), v(1., 0., 0.), &tri, Some((4.5, Edge))),
            ("parallel to the face, meets an edge", v(-5., 1., 0.3), v(1., 0., 0.), &tri, Some((cyl, Edge))),
            ("parallel to the face, too high", v(-5., 1., 0.6), v(1., 0., 0.), &tri, None),
            ("zero-length edge", v(2., 0., 5.), v(0., 0., -1.), &zl, Some((4.5, Edge))),
            ("unnormalized direction", v(1., 1., 5.), v(0., 0., -2.), &tri, Some((2.25, Face))),
        ];
        let mut bad = Vec::new();
        for (name, o, d, t, want) in cases {
            let got = triangle_shell_hit(o, d, t, r);
            if !close(got, want, 1e-4) {
                bad.push(format!("{name}: got {got:?} want {want:?}"));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }

    /// A ray at slope `s` past a unit edge at height `h`, closest approach at the
    /// edge's middle, origin far back. Answer in f64.
    fn grazing_cases() -> Vec<(Vec3, Vec3, Option<f64>)> {
        let rg = 0.002f64;
        [
            (-50.0f64, 0.01f64, 0.0019f64),
            (-50.0, 0.01, 0.00201),
            (-200.0, 0.005, 0.00199),
            (-200.0, 0.005, 0.002005),
            (-500.0, 0.004, 0.0015),
            (-500.0, 0.004, 0.0020002),
        ]
        .iter()
        .map(|&(x0, s, h)| {
            let nn = (1.0 + s * s).sqrt();
            let y0 = -s * (0.5 - x0);
            let o = v(x0 as f32, y0 as f32, h as f32);
            let d = v((1.0 / nn) as f32, (s / nn) as f32, 0.0);
            let want = (h < rg).then(|| (-(rg * rg - h * h).sqrt() - y0) * nn / s);
            (o, d, want)
        })
        .collect()
    }

    fn grazing_failures(root: RootFn) -> usize {
        let seg = BvhTriangle::new(v(0., 0., 0.), v(0.5, 0., 0.), v(1., 0., 0.));
        grazing_cases()
            .into_iter()
            .filter(|&(o, d, want)| {
                let got = triangle_hit_with(o, d, &seg, 0.002, root);
                let ok = match (got, want) {
                    (Some((t, f)), Some(w)) => {
                        (t as f64 - w).abs() < 2e-3 && f == ShellFeature::Edge
                    }
                    (None, None) => true,
                    _ => false,
                };
                !ok
            })
            .count()
    }

    #[test]
    fn grazing_along_an_edge_far_from_it() {
        assert_eq!(grazing_failures(shifted_root), 0);
        // the textbook discriminant gets these wrong in f32: the cases have teeth
        assert!(
            grazing_failures(naive_root) >= 2,
            "naive root passed the grazing cases"
        );
    }

    fn brute(tris: &[BvhTriangle], o: Vec3, d: Vec3, r: f32, tmax: f32) -> Option<ShellHit> {
        let mut best = None;
        for (k, t) in tris.iter().enumerate() {
            if let Some((s, f)) = triangle_shell_hit(o, d, t, r) {
                if s <= tmax {
                    take(&mut best, 0.0, s, k as u32, f);
                }
            }
        }
        best
    }

    #[test]
    fn a_tie_goes_to_the_lower_triangle_in_bvh_and_brute_force() {
        let p = [v(0., 0., 0.), v(4., 0., 0.), v(0., 4., 0.), v(4., 4., 0.)];
        let b = MeshBvh::build(&p, &[1, 3, 2, 0, 1, 2], 1);
        let (o, d) = (v(2., 2., 5.), v(0., 0., -1.));
        let g = b.shell_first_hit_in(ShellFrame::World, o, d, 0.5, 100.0);
        assert_eq!(g, brute(&b.triangles, o, d, 0.5, 100.0));
        assert_eq!(g.map(|h| h.triangle), Some(0));
    }

    #[test]
    fn bvh_walk_matches_brute_force_on_random_meshes() {
        let mut x = 0x2545F4914F6CDD1Du64;
        let mut u = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut hits = 0;
        for _ in 0..20 {
            let pts: Vec<Vec3> = (0..90)
                .map(|_| v(u() * 4. - 2., u() * 4. - 2., u() * 4. - 2.))
                .collect();
            let idx: Vec<u32> = (0..90).collect();
            let b = MeshBvh::build(&pts, &idx, 4);
            for _ in 0..200 {
                let o = v(u() * 12. - 6., u() * 12. - 6., u() * 12. - 6.);
                let d = v(u() - 0.5, u() - 0.5, u() - 0.5).normalize();
                let r = 0.01 + u() * 0.2;
                let g = b.shell_first_hit_in(ShellFrame::World, o, d, r, 50.0);
                assert_eq!(
                    g,
                    brute(&b.triangles, o, d, r, 50.0),
                    "o {o:?} d {d:?} r {r}"
                );
                hits += g.is_some() as usize;
            }
        }
        assert!(hits > 100, "too few hits to compare: {hits}");
    }

    #[test]
    fn hit_lies_on_the_level_set_and_nothing_before_it_is_inside() {
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut u = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        let r = 0.5;
        let (mut hits, mut off, mut early) = (0, 0, 0);
        for k in 0..20000 {
            let mut rv = || v(u() * 4. - 2., u() * 4. - 2., u() * 4. - 2.);
            let (a, b) = (rv(), rv());
            let t = if k % 10 == 0 {
                BvhTriangle::new(a, b, (a + b) * 0.5)
            } else {
                BvhTriangle::new(a, b, rv())
            };
            let o = rv() * 3.0;
            let d = rv().normalize();
            if t.unsigned_distance(o) <= r {
                continue;
            }
            if let Some((s, _)) = triangle_shell_hit(o, d, &t, r) {
                hits += 1;
                if (t.unsigned_distance(o + d * s) - r).abs() > 2e-4 {
                    off += 1;
                }
                if (1..200)
                    .any(|j| t.unsigned_distance(o + d * (s * j as f32 / 200.0 - 1e-3)) < r - 2e-4)
                {
                    early += 1;
                }
            } else if (0..400).any(|j| t.unsigned_distance(o + d * (j as f32 * 0.05)) < r - 2e-4) {
                early += 1;
            }
        }
        // 409 hits with this seed; the floor only rules out an empty comparison
        assert!(hits > 300, "too few hits: {hits}");
        assert_eq!((off, early), (0, 0), "of {hits} hits");
    }

    #[test]
    fn a_shared_edge_gives_the_same_bits_in_both_triangles() {
        // the edge p-q, named q-p by the second triangle (as a consistently wound mesh does)
        let (p, q) = (v(0.3, -0.7, 0.2), v(1.9, 0.45, -0.35));
        let one = BvhTriangle::new(p, q, v(0.1, 1.2, 0.4));
        let two = BvhTriangle::new(q, p, v(1.4, -1.1, -0.9));
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut u = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        };
        let (mut both, mut differ) = (0, 0);
        for _ in 0..20000 {
            let aim = p + (q - p) * u();
            let o = aim + v(u() - 0.5, u() - 0.5, u() - 0.5).normalize() * (1.0 + u() * 5.0);
            let d = (aim - o + v(u() - 0.5, u() - 0.5, u() - 0.5) * 0.02).normalize();
            let r = 0.05;
            if let (Some((s1, ShellFeature::Edge)), Some((s2, ShellFeature::Edge))) = (
                triangle_shell_hit(o, d, &one, r),
                triangle_shell_hit(o, d, &two, r),
            ) {
                // only rays whose first hit on both is the shared edge's cylinder
                let h = o + d * s1;
                let on_shared = |t: &BvhTriangle| {
                    let e = q - p;
                    let s = ((h - p).dot(e) / e.dot(e)).clamp(0.0, 1.0);
                    (h - (p + e * s)).length() <= t.unsigned_distance(h) + 1e-5
                };
                if on_shared(&one) && on_shared(&two) {
                    both += 1;
                    differ += (s1.to_bits() != s2.to_bits()) as usize;
                }
            }
        }
        assert!(both > 500, "too few rays on the shared edge: {both}");
        assert_eq!(differ, 0, "of {both} rays on the shared edge");
    }

    fn xorshift(seed: u64) -> impl FnMut() -> f32 {
        let mut x = seed;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// f64 distance from a point to a triangle (no shared code with the method)
    fn dist64(p: glam::DVec3, a: glam::DVec3, b: glam::DVec3, c: glam::DVec3) -> f64 {
        let seg = |u: glam::DVec3, v: glam::DVec3| {
            let e = v - u;
            let l = e.dot(e);
            let t = if l > 0.0 {
                ((p - u).dot(e) / l).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (u + e * t - p).length()
        };
        let m = seg(a, b).min(seg(b, c)).min(seg(c, a));
        let n = (b - a).cross(c - a);
        let nn = n.dot(n);
        if nn <= 0.0 {
            return m;
        }
        let q = p - n * ((p - a).dot(n) / nn);
        let inside = |u: glam::DVec3, v: glam::DVec3| (v - u).cross(q - u).dot(n) >= 0.0;
        if inside(a, b) && inside(b, c) && inside(c, a) {
            (q - p).length().min(m)
        } else {
            m
        }
    }

    #[test]
    fn dekker_products_are_exact() {
        let mut u = xorshift(0x51A7);
        let mut checked = 0;
        for _ in 0..20000 {
            let scale = [1e-6f32, 1e-4, 1e-2, 1.0, 1e2, 1e4];
            let a = (u() - 0.5) * scale[(u() * 6.0) as usize % 6];
            let b = (u() - 0.5) * scale[(u() * 6.0) as usize % 6];
            let (p, e) = two_prod(a, b);
            // f64 holds an f32 product exactly
            assert_eq!(
                f64::from(p) + f64::from(e),
                f64::from(a) * f64::from(b),
                "{a} * {b}"
            );
            checked += 1;
        }
        assert_eq!(checked, 20000);
    }

    #[test]
    fn the_plane_distance_is_within_two_ulp_of_f64() {
        // oracle: (a - o) . n in f64 on the same f32 inputs; limit 2 ulp of the result + 1e-9
        let mut u = xorshift(0xD1B54A32D192ED03);
        let mut over = 0;
        for _ in 0..20000 {
            let a = v(
                u() * 800.0 - 400.0,
                u() * 800.0 - 400.0,
                u() * 800.0 - 400.0,
            );
            let o = a + v(u() - 0.5, u() - 0.5, u() - 0.5) * (u() * 600.0);
            let n = v(u() - 0.5, u() - 0.5, u() - 0.5).normalize();
            let got = plane_distance(a, o, n);
            let want = (a.as_dvec3() - o.as_dvec3()).dot(n.as_dvec3());
            let lim = 2.0 * f64::from(f32::EPSILON) * want.abs() + 1e-9;
            if (f64::from(got) - want).abs() > lim {
                over += 1;
            }
            assert!(plane_distance_in_domain(a, o, n), "a {a:?} o {o:?} n {n:?}");
        }
        assert_eq!(over, 0, "of 20000 over 2 ulp");
    }

    /// Seeded long thin triangles (40-80 x 1-4 mm) about 1400 mm out, 300 oblique rays each aimed at a
    /// point of the triangle, solved in a 60 x 50 x 50 mm leaf frame: the origin advanced to the
    /// leaf's entry, the difference to the leaf centre exact. Returns the worst |dist - r| against
    /// the f64 distance (infinite for a miss), and checks the inputs stay in the exact domain.
    fn thin_case(seed: u64) -> f64 {
        let mut u = xorshift(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
        let r = 0.002f32;
        let base = v(u() * 1200.0 - 600.0, u() * 500.0 - 170.0, u() * 1400.0);
        let ax = v(u() - 0.5, u() - 0.5, u() - 0.5).normalize();
        let pp = v(u() - 0.5, u() - 0.5, u() - 0.5).cross(ax).normalize();
        let (l, w) = (40.0 + 40.0 * u(), 1.0 + 3.0 * u());
        let tri = BvhTriangle::new(
            base,
            base + ax * l,
            base + ax * (l - w * (u() - 0.5)) + pp * w,
        );
        let nrm = (tri.v1 - tri.v0).cross(tri.v2 - tri.v0).normalize();
        let c0 = (tri.aabb.min + tri.aabb.max) * 0.5;
        let half = v(30.0, 25.0, 25.0);
        let lo = tri.aabb.min.min(c0 - half) - Vec3::ONE;
        let hi = tri.aabb.max.max(c0 + half) + Vec3::ONE;
        let c = (lo + hi) * 0.5;
        let local = BvhTriangle::new(tri.v0 - c, tri.v1 - c, tri.v2 - c);
        let mut worst = 0f64;
        for _ in 0..300 {
            let (wa, wb0) = (u(), u());
            let wb = wb0 * (1.0 - wa);
            let p = tri.v0 + (tri.v1 - tri.v0) * wa + (tri.v2 - tri.v0) * wb;
            let tilt = 0.1 + 0.8 * u();
            let side = v(u() - 0.5, u() - 0.5, u() - 0.5).cross(nrm).normalize();
            let d = (-nrm * tilt + side * (1.0 - tilt * tilt).sqrt()).normalize();
            let o = p - d * (900.0 + 500.0 * u());
            let ra = Vec3::splat(r);
            let t_adv = slab_t(lo - ra, hi + ra, o, d.recip()).unwrap_or(0.0);
            // o_a = d t_adv + (o - c), every rounding error carried
            let comp = |o: f32, c: f32, d: f32| {
                let (s, e) = two_diff(o, c);
                let (pr, pe) = two_prod(d, t_adv);
                let (x, xe) = two_sum(pr, s);
                x + (xe + pe + e)
            };
            let oa = v(
                comp(o.x, c.x, d.x),
                comp(o.y, c.y, d.y),
                comp(o.z, c.z, d.z),
            );
            let (anchor, nu) = face_frame(local.v0, local.v1, local.v2);
            assert!(plane_distance_in_domain(anchor, oa, nu), "seed {seed}");
            let e = match triangle_shell_hit(oa, d, &local, r) {
                Some((tl, _)) => {
                    let at = o.as_dvec3() + d.as_dvec3() * (f64::from(t_adv) + f64::from(tl));
                    (dist64(at, tri.v0.as_dvec3(), tri.v1.as_dvec3(), tri.v2.as_dvec3())
                        - f64::from(r))
                    .abs()
                }
                None => f64::INFINITY,
            };
            worst = worst.max(e);
        }
        worst
    }

    #[test]
    fn thin_far_triangles_hit_on_the_level_set() {
        let mut worst = 0f64;
        for seed in [
            1277u64, 53, 2531, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
        ] {
            worst = worst.max(thin_case(seed));
        }
        assert!(worst <= 2e-5, "worst |d - r| {worst:e} (a miss is inf)");
    }

    #[test]
    fn the_smallest_magnitudes_stay_in_the_exact_domain() {
        // the smallest magnitudes measured on the figure (4 views, r .002 and .004, world and leaf
        // frames, every piece the walk reached), in mm: |anchor - o| 7.66e-3 in a leaf frame, a
        // component of anchor - o of 2.77e-5, a normal component of 1.37e-6; at those sizes the
        // smallest intermediate (a Dekker lo * lo) was 5.0e-16, far above the f32 normal minimum
        let a = v(12.5, -7.25, 3.0);
        let o = a - v(2.77e-5, 7.66e-3, 1.0e-4);
        let n = v(1.37e-6, 0.6, 0.8).normalize();
        assert!(n.x > 1.0e-6);
        assert!(plane_distance_in_domain(a, o, n));
        // and in world coordinates, |anchor - o| 1.10 and up
        assert!(plane_distance_in_domain(
            a + v(800.0, 600.0, 1400.0),
            o + v(800.0, 600.0, 1400.0) - v(0.0, 1.1, 0.0),
            n
        ));
        assert!(plane_distance_in_domain(a, a, n));
        // and a case outside it, so the check has teeth
        assert!(!plane_distance_in_domain_with(
            v(1e-20, 0.0, 0.0),
            v(-1e-20, 0.0, 0.0),
            v(1e-20, 0.0, 1.0),
            false
        ));
    }

    #[test]
    fn a_face_hit_matches_the_f64_plane_solve_for_oblique_rays() {
        // oracle: t = ((anchor - o) . n + r) / dn in f64 on the same f32 anchor, normal and dn; the
        // rays start far along the plane (|anchor - o| much larger than the plane distance), where a
        // plain f32 dot of anchor - o loses the plane distance's low digits
        let mut u = xorshift(0x000B_110E);
        let (mut over, mut checked) = (0, 0);
        for _ in 0..20000 {
            let a = v(u() * 40.0 - 20.0, u() * 40.0 - 20.0, u() * 40.0 - 20.0);
            let b = a + v(u() * 30.0 - 15.0, u() * 30.0 - 15.0, u() * 30.0 - 15.0);
            let c = a + v(u() * 30.0 - 15.0, u() * 30.0 - 15.0, u() * 30.0 - 15.0);
            let (anchor, nu) = face_frame(a, b, c);
            let n = nu / nu.length();
            let target = a + (b - a) * 0.3 + (c - a) * 0.3;
            let along = (b - a).normalize();
            let o = target - along * (300.0 + 100.0 * u()) + n * (0.05 + u() * 0.05);
            let d = (target - o).normalize();
            let r = 0.002;
            let Some(t) = face_t(o, d, a, b, c, r) else {
                continue;
            };
            // t = (P +- r |nu|) / D: P = (anchor - o) . nu in f64, with the method's own f32 |nu|
            // and D = d . nu, so this test isolates the plane distance P
            let p6 = (anchor.as_dvec3() - o.as_dvec3()).dot(nu.as_dvec3());
            let nl6 = f64::from(nu.length());
            let dd = f64::from(d.dot(nu));
            let n6 = nu.as_dvec3() / nl6;
            // each side's hit point, kept when it lies over the triangle (f64 edge tests)
            let (a6, b6, c6) = (a.as_dvec3(), b.as_dvec3(), c.as_dvec3());
            let nn6 = (b6 - a6).cross(c6 - a6);
            let want = [0.002f64, -0.002]
                .into_iter()
                .map(|off| ((p6 + off * nl6) / dd, off))
                .filter(|(t, off)| {
                    let q = o.as_dvec3() + d.as_dvec3() * *t - n6 * *off;
                    let e = |u: glam::DVec3, w: glam::DVec3| (w - u).cross(q - u).dot(nn6) >= 0.0;
                    *t >= 0.0 && e(a6, b6) && e(b6, c6) && e(c6, a6)
                })
                .map(|(t, _)| t)
                .fold(f64::INFINITY, f64::min);
            if !want.is_finite() {
                // on the edge of the triangle: the edge cylinders decide, not this test
                continue;
            }
            checked += 1;
            if (f64::from(t) - want).abs() > 8.0 * f64::from(f32::EPSILON) * want.abs() {
                over += 1;
            }
        }
        assert!(checked > 1000, "checked {checked}");
        assert_eq!(
            over, 0,
            "of {checked} face hits over 8 ulp of the f64 plane solve"
        );
    }

    #[test]
    fn two_coplanar_triangles_leave_no_crack_on_their_shared_edge() {
        // a quad split along p-q: a point of the plane next to the diagonal is over one triangle or
        // both, never neither
        let mut u = xorshift(0xC0FFEE);
        let mut crack = 0;
        for _ in 0..200 {
            let p = v(u() * 10.0, u() * 10.0, u() * 10.0);
            let q = p + v(u() * 7.0 + 1.0, u() * 7.0 - 3.0, u() * 7.0 - 3.0);
            let side = (q - p).cross(v(0.3, 1.0, 0.2)).normalize();
            let (one, two) = (
                [p, q, p + side * 3.0 + (q - p) * 0.5],
                [q, p, p - side * 3.0 + (q - p) * 0.5],
            );
            let (_, n1) = face_frame(one[0], one[1], one[2]);
            let (_, n2) = face_frame(two[0], two[1], two[2]);
            for _ in 0..500 {
                let s = p + (q - p) * (0.05 + 0.9 * u()) + side * ((u() - 0.5) * 1e-5);
                let over = |t: [Vec3; 3], nu: Vec3| {
                    edge_function(t[0], t[1], s, nu) >= 0.0
                        && edge_function(t[1], t[2], s, nu) >= 0.0
                        && edge_function(t[2], t[0], s, nu) >= 0.0
                };
                crack += (!over(one, n1) && !over(two, n2)) as usize;
            }
        }
        assert_eq!(crack, 0, "points over neither triangle");
    }

    #[test]
    fn the_flush_keeps_every_intermediate_normal_for_any_input() {
        // a proof check of the construction: random inputs over 80 decades, components near zero
        // included, are all in the exact domain once flushed
        let mut u = xorshift(0xF1_05);
        let mut tiny = 0;
        for _ in 0..50000 {
            let mut c = || {
                let x = (u() - 0.5) * 2.0;
                let e = (u() * 80.0) as i32 - 60;
                let s = f32::from_bits(((e + 127) as u32) << 23);
                if u() < 0.1 {
                    0.0
                } else {
                    x * s
                }
            };
            let (a, o, n) = (v(c(), c(), c()), v(c(), c(), c()), v(c(), c(), c()));
            tiny += (!plane_distance_in_domain_with(a, o, n, false)) as usize;
            assert!(plane_distance_in_domain(a, o, n), "a {a:?} o {o:?} n {n:?}");
        }
        // inputs the flush is there for did occur
        assert!(
            tiny > 100,
            "only {tiny} inputs left the domain without the flush"
        );
    }

    #[test]
    fn the_triangle_frame_is_closer_to_the_shell_far_from_the_origin() {
        // thin triangles about 1400 mm out, oblique rays from 900-1400 mm back: the worst
        // |d - r| of the hits, in f64 (no shared code), unframed and in the triangle frame
        let mut u = xorshift(0x5EED_F00D);
        let r = 0.002f32;
        let (mut v, mut ix) = (Vec::new(), Vec::new());
        for k in 0..40u32 {
            let b0 = Vec3::new(
                u() * 1200.0 - 600.0,
                u() * 500.0 - 170.0,
                900.0 + u() * 500.0,
            );
            let ax = Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5).normalize();
            let pp = Vec3::new(u() - 0.5, u() - 0.5, u() - 0.5)
                .cross(ax)
                .normalize();
            let (l, w) = (40.0 + 40.0 * u(), 1.0 + 3.0 * u());
            v.extend([b0, b0 + ax * l, b0 + ax * (l - w * (u() - 0.5)) + pp * w]);
            ix.extend([3 * k, 3 * k + 1, 3 * k + 2]);
        }
        let bvh = MeshBvh::build(&v, &ix, 4);
        let worst = |frame| {
            let mut uu = xorshift(0xA11CE);
            let mut w = 0f64;
            let mut hits = 0;
            for t in &bvh.triangles {
                for _ in 0..20 {
                    let (a, b0) = (uu(), uu());
                    let p = t.v0 + (t.v1 - t.v0) * a + (t.v2 - t.v0) * (b0 * (1.0 - a));
                    let n = (t.v1 - t.v0).cross(t.v2 - t.v0).normalize();
                    let tilt = 0.1 + 0.8 * uu();
                    let side = Vec3::new(uu() - 0.5, uu() - 0.5, uu() - 0.5)
                        .cross(n)
                        .normalize();
                    let d = (-n * tilt + side * (1.0 - tilt * tilt).sqrt()).normalize();
                    let o = p - d * (900.0 + 500.0 * uu());
                    if let Some(h) = bvh.shell_first_hit_in(frame, o, d, r, 3000.0) {
                        hits += 1;
                        let at = o.as_dvec3()
                            + d.as_dvec3() * (f64::from(h.t_base) + f64::from(h.t_local));
                        let tri = &bvh.triangles[h.triangle as usize];
                        let e =
                            (dist64(at, tri.v0.as_dvec3(), tri.v1.as_dvec3(), tri.v2.as_dvec3())
                                - f64::from(r))
                            .abs();
                        w = f64::max(w, e);
                    }
                }
            }
            assert!(hits > 700, "{frame:?}: {hits} hits");
            w
        };
        let (world, framed) = (worst(ShellFrame::World), worst(ShellFrame::Triangle));
        // the frame must gain at least an order of magnitude, and stay within the 2e-5 limit the
        // thin-triangle cases use
        assert!(
            framed * 10.0 < world,
            "world {world:e}, triangle {framed:e}"
        );
        assert!(framed <= 2e-5, "triangle {framed:e}");
    }
}

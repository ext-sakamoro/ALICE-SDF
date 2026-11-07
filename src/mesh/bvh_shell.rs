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
    /// Ray parameter of the hit (the hit point is `origin + t * dir`)
    pub t: f32,
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

/// Hit on either plane offset from the face by `r`, inside the triangle's prism
/// Hit on either plane offset from the face by `r`, inside the triangle's prism
///
/// Whether the point lies over the triangle is decided by the three edge functions against the
/// unnormalised normal, `((v1 - v0) x (q - v0)) . nn >= 0` for each edge: no division, so a sliver
/// or near-collinear triangle (whose barycentric weights would divide by a Gram determinant near 0)
/// accepts only points over itself.
#[inline]
fn face_t(o: Vec3, d: Vec3, a: Vec3, b: Vec3, c: Vec3, r: f32) -> Option<f32> {
    let nn = (b - a).cross(c - a);
    let nl = nn.length();
    if nl == 0.0 || !nl.is_finite() {
        return None;
    }
    let n = nn / nl;
    let dn = d.dot(n);
    if dn == 0.0 {
        return None;
    }
    let mut best: Option<f32> = None;
    for sg in [1.0f32, -1.0] {
        let t = ((a + n * (sg * r)) - o).dot(n) / dn;
        if t < 0.0 {
            continue;
        }
        let q = o + d * t - n * (sg * r);
        let over = (b - a).cross(q - a).dot(nn) >= 0.0
            && (c - b).cross(q - b).dot(nn) >= 0.0
            && (a - c).cross(q - c).dot(nn) >= 0.0;
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
fn take(best: &mut Option<ShellHit>, t: f32, triangle: u32, feature: ShellFeature) {
    let better = match best {
        None => true,
        Some(b) => t < b.t || (t == b.t && triangle < b.triangle),
    };
    if better {
        *best = Some(ShellHit {
            t,
            triangle,
            feature,
        });
    }
}

struct Walk<'a> {
    tris: &'a [BvhTriangle],
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
                for &k in triangles {
                    if let Some((t, f)) = triangle_shell_hit(self.o, self.d, &self.tris[k], self.r)
                    {
                        if t <= self.tmax {
                            take(best, t, k as u32, f);
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
    pub fn shell_first_hit(&self, origin: Vec3, dir: Vec3, r: f32, tmax: f32) -> Option<ShellHit> {
        let root = self.root.as_ref()?;
        let walk = Walk {
            tris: &self.triangles,
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
                    take(&mut best, s, k as u32, f);
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
        let g = b.shell_first_hit(o, d, 0.5, 100.0);
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
                let g = b.shell_first_hit(o, d, r, 50.0);
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
}

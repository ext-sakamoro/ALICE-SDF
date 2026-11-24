//! Exact nearest face of an open triangle mesh, in f64, for the sheet builder.
//!
//! The builder evaluates its channels (side-signed distance, distance past the rim, UV, author normal) from the nearest face of every
//! brick node and from the distance to it, so both must be exact and reproducible: [`nearest_faces`] returns, per query point, the
//! face with the smallest per-face distance and that distance, ties going to the lowest face index.
//!
//! # Per-face distance
//!
//! Two formulas, chosen by the mesh size ([`formula_for`]): below [`BRUTE_FACES`] faces the plane-or-edge form ([`Formula::PlaneSeg`]:
//! the distance to the plane when the projection falls inside the triangle, else the smallest distance to the three edges), from
//! [`BRUTE_FACES`] up the Ericson closest point ([`Formula::Ericson`], Real-Time Collision Detection 5.1.5) and the length of the
//! offset. Both are written operation for operation as the reference float64 implementation (dot products summed left to right, no
//! fused multiply-add), so the distance bits are equal, not only close. They agree up to rounding; keeping both is what makes the
//! fields byte-identical to the earlier builder.
//!
//! # Search
//!
//! A bounding volume hierarchy over the face boxes (median split on the longest axis, leaves of up to [`LEAF`] faces), walked nearest
//! box first. A box is skipped only when the distance from the point to it exceeds the best distance so far by more than the rounding
//! tolerance `tol = 1e-9 (1 + |p|_inf + best)`: the box distance is a lower bound of the true distance to every face in it, and each
//! formula is within far less than `tol` of the true distance, so no face that could be the minimum, or tie with it, is skipped.
//! Faces that are degenerate or nearly so ([`COND`]) skip the hierarchy and are evaluated for every query. Among the faces evaluated the smallest `(distance, face index)` wins, which is the exhaustive minimum with the lowest index on
//! ties. Work per query depends on the faces near the point, not on the mesh size.

use super::SheetFieldError;
use crate::compiled::real::LawMinMax;

/// Meshes with fewer faces use [`Formula::PlaneSeg`], larger ones [`Formula::Ericson`].
pub const BRUTE_FACES: usize = 400;
/// Faces per leaf of the hierarchy.
pub const LEAF: usize = 4;

/// Per-face distance formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formula {
    /// Distance to the plane when the projection is inside the triangle, else the smallest edge distance.
    PlaneSeg,
    /// Length of the offset to the Ericson closest point.
    Ericson,
}

/// The formula the builder uses for a mesh of `n_faces` faces.
#[must_use]
pub const fn formula_for(n_faces: usize) -> Formula {
    if n_faces < BRUTE_FACES {
        Formula::PlaneSeg
    } else {
        Formula::Ericson
    }
}

/// Nearest face of one query point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NearestFace {
    /// Distance by the mesh's formula.
    pub dist: f64,
    /// Face index (lowest on ties).
    pub face: u32,
}

type V3 = [f64; 3];

#[inline]
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn norm(a: V3) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}
/// the reference's `np.maximum(x, m)` (x is never NaN for finite input)
#[inline]
fn maxf(x: f64, m: f64) -> f64 {
    if x > m {
        x
    } else {
        m
    }
}
/// the reference's `np.where(abs(x) > 1e-30, x, 1e-30)`
#[inline]
fn guard(x: f64) -> f64 {
    if x.abs() > 1e-30 {
        x
    } else {
        1e-30
    }
}
#[inline]
fn seg(p: V3, p0: V3, p1: V3) -> f64 {
    let e = sub(p1, p0);
    let t = (dot(sub(p, p0), e) / maxf(dot(e, e), 1e-30)).clamp(0.0, 1.0);
    norm(sub(
        p,
        [p0[0] + t * e[0], p0[1] + t * e[1], p0[2] + t * e[2]],
    ))
}

fn plane_seg(p: V3, a: V3, b: V3, c: V3) -> f64 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let n = cross(ab, ac);
    let nn = dot(n, n);
    let (ab2, ac2, abac) = (dot(ab, ab), dot(ac, ac), dot(ab, ac));
    let abac_sq = abac * abac; // (ab . ac)^2
    let det = maxf(ab2 * ac2 - abac_sq, 1e-30);
    let u = (ac2 * d1 - abac * d2) / det;
    let v = (ab2 * d2 - abac * d1) / det;
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        dot(ap, n).abs() / maxf(nn, 1e-30).sqrt()
    } else {
        seg(p, a, b).law_min(seg(p, b, c)).law_min(seg(p, c, a))
    }
}

fn ericson(p: V3, a: V3, b: V3, c: V3) -> f64 {
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    let vc = d1 * d4 - d3 * d2;
    let vb = d5 * d2 - d1 * d6;
    let va = d3 * d6 - d5 * d4;
    // the reference evaluates the regions in this order and the first that holds wins
    let w3: V3 = if d1 <= 0.0 && d2 <= 0.0 {
        [1.0, 0.0, 0.0]
    } else if d3 >= 0.0 && d4 <= d3 {
        [0.0, 1.0, 0.0]
    } else if d6 >= 0.0 && d5 <= d6 {
        [0.0, 0.0, 1.0]
    } else if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / guard(d1 - d3);
        [1.0 - t, t, 0.0]
    } else if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / guard(d2 - d6);
        [1.0 - t, 0.0, t]
    } else if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / guard((d4 - d3) + (d5 - d6));
        [0.0, 1.0 - t, t]
    } else {
        let den = guard(va + vb + vc);
        let (v, w) = (vb / den, vc / den);
        [1.0 - v - w, v, w]
    };
    let q = [
        w3[0] * a[0] + w3[1] * b[0] + w3[2] * c[0],
        w3[0] * a[1] + w3[1] * b[1] + w3[2] * c[1],
        w3[0] * a[2] + w3[1] * b[2] + w3[2] * c[2],
    ];
    norm(sub(p, q))
}

/// Distance from `p` to the triangle `a b c` by `formula`.
#[must_use]
pub fn face_distance(formula: Formula, p: V3, a: V3, b: V3, c: V3) -> f64 {
    match formula {
        Formula::PlaneSeg => plane_seg(p, a, b, c),
        Formula::Ericson => ericson(p, a, b, c),
    }
}

/// Faces with `|ab x ac|^2 <= COND |ab|^2 |ac|^2` (sin^2 of the corner angle at `a`) skip the hierarchy and are evaluated for every query.
///
/// On such faces the formulas can stray from the true distance by more than the pruning tolerance
/// (a zero-area face gives a plane-or-edge distance of 0 from anywhere, the clamped determinant making every projection "inside"), so
/// box bounds cannot exclude them; the exhaustive minimum still includes them.
pub const COND: f64 = 1e-6;

fn ill_conditioned(a: V3, b: V3, c: V3) -> bool {
    let (ab, ac) = (sub(b, a), sub(c, a));
    let (ab2, ac2, abac) = (dot(ab, ab), dot(ac, ac), dot(ab, ac));
    let abac_sq = abac * abac; // (ab . ac)^2
    let det = ab2 * ac2 - abac_sq;
    // a non-finite det (overflow) also counts as ill-conditioned
    !det.is_finite() || det <= COND * ab2 * ac2 || det <= 1e-30
}

struct Node {
    lo: V3,
    hi: V3,
    /// leaf: faces `order[start..start + count]`; inner: children `left`, `left + 1` (count == 0)
    start: u32,
    count: u32,
    left: u32,
}

struct Bvh {
    nodes: Vec<Node>,
    order: Vec<u32>,
}

fn box_of(lo: &[V3], hi: &[V3], ids: &[u32]) -> (V3, V3) {
    let mut l = [f64::INFINITY; 3];
    let mut h = [f64::NEG_INFINITY; 3];
    for &i in ids {
        for k in 0..3 {
            l[k] = l[k].law_min(lo[i as usize][k]);
            h[k] = h[k].law_max(hi[i as usize][k]);
        }
    }
    (l, h)
}

impl Bvh {
    /// over the faces `ids` (not empty)
    fn new(v: &[V3], f: &[[u32; 3]], ids: Vec<u32>) -> Self {
        let mut lo = Vec::with_capacity(f.len());
        let mut hi = Vec::with_capacity(f.len());
        let mut cen = Vec::with_capacity(f.len());
        for t in f {
            let (a, b, c) = (v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            let l = [
                a[0].law_min(b[0]).law_min(c[0]),
                a[1].law_min(b[1]).law_min(c[1]),
                a[2].law_min(b[2]).law_min(c[2]),
            ];
            let h = [
                a[0].law_max(b[0]).law_max(c[0]),
                a[1].law_max(b[1]).law_max(c[1]),
                a[2].law_max(b[2]).law_max(c[2]),
            ];
            cen.push([
                f64::midpoint(l[0], h[0]),
                f64::midpoint(l[1], h[1]),
                f64::midpoint(l[2], h[2]),
            ]);
            lo.push(l);
            hi.push(h);
        }
        let mut order = ids;
        let mut nodes = Vec::with_capacity(2 * order.len() / LEAF + 1);
        let (l, h) = box_of(&lo, &hi, &order);
        nodes.push(Node {
            lo: l,
            hi: h,
            start: 0,
            count: order.len() as u32,
            left: 0,
        });
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            let (s, n) = (nodes[ni].start as usize, nodes[ni].count as usize);
            if n <= LEAF {
                continue;
            }
            let (l, h) = (nodes[ni].lo, nodes[ni].hi);
            let ax = (0..3)
                .max_by(|&x, &y| (h[x] - l[x]).total_cmp(&(h[y] - l[y])))
                .unwrap_or(0);
            let ids = &mut order[s..s + n];
            let mid = n / 2;
            ids.select_nth_unstable_by(mid, |&x, &y| {
                cen[x as usize][ax]
                    .total_cmp(&cen[y as usize][ax])
                    .then(x.cmp(&y))
            });
            let left = nodes.len();
            for (cs, cn) in [(s, mid), (s + mid, n - mid)] {
                let (cl, ch) = box_of(&lo, &hi, &order[cs..cs + cn]);
                nodes.push(Node {
                    lo: cl,
                    hi: ch,
                    start: cs as u32,
                    count: cn as u32,
                    left: 0,
                });
            }
            nodes[ni].count = 0;
            nodes[ni].left = left as u32;
            stack.push(left);
            stack.push(left + 1);
        }
        Self { nodes, order }
    }
}

/// distance from p to an axis-aligned box (0 inside): a lower bound of the distance to anything in the box
#[inline]
fn box_dist(p: V3, lo: V3, hi: V3) -> f64 {
    let mut s = 0.0;
    for k in 0..3 {
        let d = if p[k] < lo[k] {
            lo[k] - p[k]
        } else if p[k] > hi[k] {
            p[k] - hi[k]
        } else {
            0.0
        };
        s += d * d;
    }
    s.sqrt()
}

fn validate(v: &[V3], f: &[[u32; 3]], q: &[V3]) -> Result<(), SheetFieldError> {
    if f.is_empty() {
        return Err(SheetFieldError::Shape("nearest_faces: no faces".into()));
    }
    if f.len() > u32::MAX as usize {
        return Err(SheetFieldError::Shape(
            "nearest_faces: more faces than u32 indices".into(),
        ));
    }
    if let Some((k, t)) = f
        .iter()
        .enumerate()
        .find(|(_, t)| t.iter().any(|&i| i as usize >= v.len()))
    {
        return Err(SheetFieldError::Shape(format!(
            "nearest_faces: face {k} {t:?} indexes past {} vertices",
            v.len()
        )));
    }
    if let Some(i) = v.iter().position(|x| x.iter().any(|c| !c.is_finite())) {
        return Err(SheetFieldError::Shape(format!(
            "nearest_faces: vertex {i} not finite"
        )));
    }
    if let Some(i) = q.iter().position(|x| x.iter().any(|c| !c.is_finite())) {
        return Err(SheetFieldError::Shape(format!(
            "nearest_faces: query {i} not finite"
        )));
    }
    Ok(())
}

/// Nearest face and its distance for every query point (see the module doc), plus the number of per-face evaluations.
///
/// # Errors
/// [`SheetFieldError::Shape`] for an empty face list, a face index past the vertices, or a non-finite vertex or query.
pub fn nearest_faces_stats(
    v: &[V3],
    f: &[[u32; 3]],
    q: &[V3],
) -> Result<(Vec<NearestFace>, u64), SheetFieldError> {
    validate(v, f, q)?;
    let fm = formula_for(f.len());
    let (always, rest): (Vec<u32>, Vec<u32>) = (0..f.len() as u32).partition(|&k| {
        let t = f[k as usize];
        ill_conditioned(v[t[0] as usize], v[t[1] as usize], v[t[2] as usize])
    });
    let bvh = if rest.is_empty() {
        None
    } else {
        Some(Bvh::new(v, f, rest))
    };
    let mut out = Vec::with_capacity(q.len());
    let mut evals = 0u64;
    let mut stack: Vec<(f64, u32)> = Vec::with_capacity(64);
    for &p in q {
        let scale = 1.0 + p[0].abs().law_max(p[1].abs()).law_max(p[2].abs());
        let (mut bd, mut bf) = (f64::INFINITY, u32::MAX);
        for &fi in &always {
            let t = f[fi as usize];
            let d = face_distance(fm, p, v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
            evals += 1;
            if d < bd || (d == bd && fi < bf) {
                bd = d;
                bf = fi;
            }
        }
        stack.clear();
        let Some(bvh) = &bvh else {
            out.push(NearestFace { dist: bd, face: bf });
            continue;
        };
        stack.push((box_dist(p, bvh.nodes[0].lo, bvh.nodes[0].hi), 0));
        while let Some((lb, ni)) = stack.pop() {
            if lb > bd + 1e-9 * (scale + bd) {
                continue;
            }
            let nd = &bvh.nodes[ni as usize];
            if nd.count > 0 {
                for &fi in &bvh.order[nd.start as usize..(nd.start + nd.count) as usize] {
                    let t = f[fi as usize];
                    let d =
                        face_distance(fm, p, v[t[0] as usize], v[t[1] as usize], v[t[2] as usize]);
                    evals += 1;
                    if d < bd || (d == bd && fi < bf) {
                        bd = d;
                        bf = fi;
                    }
                }
            } else {
                let (l, r) = (nd.left, nd.left + 1);
                let dl = box_dist(p, bvh.nodes[l as usize].lo, bvh.nodes[l as usize].hi);
                let dr = box_dist(p, bvh.nodes[r as usize].lo, bvh.nodes[r as usize].hi);
                // push the farther child first so the nearer one is walked first
                if dl <= dr {
                    stack.push((dr, r));
                    stack.push((dl, l));
                } else {
                    stack.push((dl, l));
                    stack.push((dr, r));
                }
            }
        }
        out.push(NearestFace { dist: bd, face: bf });
    }
    Ok((out, evals))
}

/// Nearest face and its distance for every query point (see the module doc).
///
/// # Errors
/// As [`nearest_faces_stats`].
pub fn nearest_faces(
    v: &[V3],
    f: &[[u32; 3]],
    q: &[V3],
) -> Result<Vec<NearestFace>, SheetFieldError> {
    nearest_faces_stats(v, f, q).map(|(r, _)| r)
}

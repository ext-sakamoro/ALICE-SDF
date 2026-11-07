//! Which rays may answer differently on a GPU than on the CPU, from the WGSL accuracy rules
//!
//! The shell first hit uses `+ - *` (correctly rounded in WGSL, and the shader path forbids
//! contraction), `/` and `sqrt`. WGSL allows `x / y` within 2.5 ULP and `sqrt` as `1 / inverseSqrt`
//! (2 ULP, then a 2.5 ULP division): against the CPU's correctly rounded results (0.5 ULP each) a
//! division may differ by 3 ULP and a square root by 5 (`DIV`, `SQRT`, in units of `E`, an ULP at
//! most `E` times the value). Everything before the first division or root is bit-identical.
//!
//! Each piece of each triangle near the ray is recomputed here with the CPU's f32 arithmetic, and
//! alongside it (in f64) how far each value may move on the GPU: an operation with moved inputs
//! moves by its sensitivity to them plus one ULP of its own rounding. A decision (`h < 0`, `t < 0`,
//! an edge function's sign, the cylinder's axial range, the inside test) is *ambiguous* when its
//! value lies within its bound of the threshold; two candidates are *tied* when their `t` lie within
//! the sum of their bounds. A ray is in the ambiguous band when an ambiguous decision or a tie could
//! change its answer. The rule set mirrors `src/mesh/bvh_shell.rs` operation by operation.
//!
//! Author: Moroya Sakamoto
#![allow(dead_code)]

use alice_sdf::mesh::{BvhNode, BvhTriangle, MeshBvh, ShellRay};
use glam::{DVec3, Vec3};

/// An ULP is at most `E` times the value (f32, `2^-23`)
const E: f64 = 1.192_092_895_507_812_5e-7;
/// GPU against CPU, in `E`: a division
const DIV: f64 = 3.0;
/// GPU against CPU, in `E`: a square root
const SQRT: f64 = 5.0;

fn a(v: Vec3) -> DVec3 {
    v.as_dvec3().abs()
}

/// A value with how far it may move on the GPU
#[derive(Clone, Copy, Debug)]
struct M {
    v: f32,
    d: f64,
}

/// One piece's hit: `t` with its bound, and whether a decision on the way was ambiguous
#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub tri: u32,
    pub feature: u32,
    pub t: f32,
    pub dt: f64,
    /// a decision that let this piece hit (or would have let it hit) is within its bound
    pub ambiguous: bool,
    /// this piece hits on the CPU
    pub hits: bool,
    /// which decision was ambiguous ("" when none)
    pub why: &'static str,
}

/// `|w0 + t g|^2 = rr` about the closest approach; returns (lo or hi, its bound, ambiguous, hits)
fn root(w0: Vec3, g: Vec3, rr: f32) -> Option<(M, bool, bool)> {
    let aa = g.dot(g);
    if aa == 0.0 {
        return None;
    }
    let t0 = -w0.dot(g) / aa;
    let dt0 = DIV * E * f64::from(t0.abs());
    let w = w0 + g * t0;
    // each component: g t0 moves by |g| dt0 plus its rounding, the sum by one more rounding
    let dw = a(g) * dt0 + (a(g * t0) + a(w)) * E;
    let ww = w.dot(w);
    let dww = 2.0 * a(w).dot(dw) + dw.dot(dw) + 3.0 * E * f64::from(ww);
    let h = rr - ww;
    let dh = dww + E * f64::from(h.abs());
    let amb_h = f64::from(h.abs()) <= dh;
    if h < 0.0 {
        // a miss on the CPU; ambiguous only if the GPU could see h >= 0
        return amb_h.then_some((
            M {
                v: t0,
                d: dt0 + (dh / f64::from(aa)).sqrt(),
            },
            true,
            false,
        ));
    }
    let q = h / aa;
    let dq = dh / f64::from(aa) + DIV * E * f64::from(q);
    let dt = q.sqrt();
    let ddt = f64::min(dq.sqrt(), dq / (2.0 * f64::from(dt).max(f64::MIN_POSITIVE)))
        + SQRT * E * f64::from(dt);
    let (lo, hi) = (t0 - dt, t0 + dt);
    let dlo = dt0 + ddt + E * f64::from(lo.abs());
    let dhi = dt0 + ddt + E * f64::from(hi.abs());
    let amb_lo = f64::from(lo.abs()) <= dlo;
    let amb = amb_h || amb_lo;
    if lo >= 0.0 {
        Some((M { v: lo, d: dlo }, amb, true))
    } else if hi >= 0.0 {
        Some((M { v: hi, d: dhi }, amb || f64::from(hi.abs()) <= dhi, true))
    } else {
        (amb || f64::from(hi.abs()) <= dhi).then_some((M { v: hi, d: dhi }, true, false))
    }
}

fn cylinder(o: Vec3, d: Vec3, u0: Vec3, v0: Vec3, r: f32) -> Option<(M, bool, bool)> {
    let key = |p: Vec3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
    let (u, v) = if key(v0) < key(u0) {
        (v0, u0)
    } else {
        (u0, v0)
    };
    let e = v - u;
    let l2 = e.dot(e);
    if l2 == 0.0 {
        return None;
    }
    let m = o - u;
    let (t, amb, hits) = root(m.cross(e), d.cross(e), r * r * l2)?;
    let p = m + d * t.v;
    let dp = a(d) * t.d + a(p) * E;
    let ax = p.dot(e);
    let dax = a(e).dot(dp) + 3.0 * E * a(p).dot(a(e));
    let amb_ax = f64::from(ax.abs()) <= dax || f64::from((ax - l2).abs()) <= dax;
    let inside = (0.0..=l2).contains(&ax);
    let hits = hits && inside;
    (hits || amb_ax || amb).then_some((t, amb || amb_ax, hits))
}

fn key(p: Vec3) -> (u32, u32, u32) {
    (p.x.to_bits(), p.y.to_bits(), p.z.to_bits())
}

// the method's plane: the shortest-edge normal at its anchor, and (anchor - o) . n in compensated
// arithmetic (Dekker products), mirrored from src/mesh/bvh_shell.rs
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}
fn two_diff(a: f32, b: f32) -> (f32, f32) {
    let s = a - b;
    let bb = s - a;
    (s, (a - (s - bb)) + (-b - bb))
}
fn split(a: f32) -> (f32, f32) {
    let c = 4097.0 * a;
    let hi = c - (c - a);
    (hi, a - hi)
}
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}
fn plane_distance(a: Vec3, o: Vec3, n: Vec3) -> f32 {
    let (wx, ex) = two_diff(a.x, o.x);
    let (wy, ey) = two_diff(a.y, o.y);
    let (wz, ez) = two_diff(a.z, o.z);
    let (p0, q0) = two_prod(wx, n.x);
    let (h1, r1) = two_prod(wy, n.y);
    let (p1, s1) = two_sum(p0, h1);
    let (h2, r2) = two_prod(wz, n.z);
    let (p2, s2) = two_sum(p1, h2);
    p2 + (q0 + r1 + s1 + r2 + s2 + (ex * n.x + ey * n.y + ez * n.z))
}
fn face_frame(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
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
fn edge_function(u: Vec3, v: Vec3, q: Vec3, nu: Vec3) -> f32 {
    if key(u) <= key(v) {
        (v - u).cross(q - u).dot(nu)
    } else {
        -((u - v).cross(q - v).dot(nu))
    }
}

/// Both offset planes; each side is its own piece
fn face(
    o: Vec3,
    d: Vec3,
    va: Vec3,
    vb: Vec3,
    vc: Vec3,
    r: f32,
) -> Vec<(M, bool, bool, &'static str)> {
    let (anchor, nu) = face_frame(va, vb, vc);
    let nl = nu.length();
    if nl == 0.0 || !nl.is_finite() {
        return Vec::new();
    }
    let n = nu / nl;
    // n: a square root, then a division per component
    let dn_vec = a(n) * ((SQRT + DIV + 1.0) * E);
    let dn = d.dot(n);
    let ddn = a(d).dot(dn_vec) + 3.0 * E * a(d).dot(a(n));
    let amb_dn = f64::from(dn.abs()) <= ddn;
    if dn == 0.0 {
        return Vec::new();
    }
    // h is exact up to its last rounding for given n, and moves with n. n_i = nu_i / nl: the square
    // root moves every component by the same factor, which moves h by SQRT E |h|; each division
    // adds its own DIV E |n_i|, which moves h by DIV E sum |w_i n_i| (w = anchor - o)
    let h = plane_distance(anchor, o, n);
    let w = (anchor.as_dvec3() - o.as_dvec3()).abs();
    let dh = SQRT * E * f64::from(h.abs()) + DIV * E * w.dot(a(n)) + 2.0 * E * f64::from(h.abs());
    let mut out = Vec::new();
    for sg in [1.0f32, -1.0] {
        let off = sg * r;
        let num = h + off;
        let dnum = dh + E * f64::from(num.abs());
        let t = num / dn;
        let dt =
            (dnum + f64::from(t.abs()) * ddn) / f64::from(dn.abs()) + DIV * E * f64::from(t.abs());
        let amb_t = f64::from(t.abs()) <= dt;
        if t < 0.0 && !amb_t {
            continue;
        }
        let q = o + d * t - n * off;
        let dq = a(d) * dt + dn_vec * f64::from(off.abs()) + (a(o + d * t) + a(q)) * (2.0 * E);
        let mut over = true;
        let mut amb_e = false;
        for (p0, p1) in [(va, vb), (vb, vc), (vc, va)] {
            let ef = edge_function(p0, p1, q, nu);
            let s = (p1 - p0).cross(q - p0);
            // |((p1 - p0) x dq) . nu| <= |p1 - p0| |dq| |nu|, plus the roundings of the cross and dot
            let de = (p1 - p0).as_dvec3().length() * dq.length() * f64::from(nl)
                + 6.0 * E * a(s).dot(a(nu))
                + 4.0 * E * a(p1 - p0).dot(a(q - p0)) * f64::from(nl);
            amb_e |= f64::from(ef.abs()) <= de;
            over &= ef >= 0.0;
        }
        let hits = t >= 0.0 && over;
        let amb = amb_dn || amb_t || amb_e;
        let why = if amb_dn {
            "face dn"
        } else if amb_t {
            "face t"
        } else if amb_e {
            "face edge"
        } else {
            ""
        };
        if hits || amb {
            out.push((M { v: t, d: dt }, amb, hits, why));
        }
    }
    out
}

/// Every piece of `tri` that hits on the CPU or has an ambiguous decision
pub fn pieces(o: Vec3, d: Vec3, tri: &BvhTriangle, index: u32, r: f32) -> Vec<Piece> {
    let mut out = Vec::new();
    // the inside test: dist = |o - closest(o)| against r; the closest point divides (its region
    // weights), so allow it DIV plus rounding on the scale of the triangle and the point
    let dist = tri.unsigned_distance(o);
    let scale =
        f64::from((o - tri.v0).length() + (tri.v1 - tri.v0).length() + (tri.v2 - tri.v0).length());
    let ddist = (DIV + 8.0) * E * scale + SQRT * E * f64::from(dist);
    let amb_in = f64::from((dist - r).abs()) <= ddist;
    if dist <= r || amb_in {
        out.push(Piece {
            tri: index,
            feature: 4,
            t: 0.0,
            dt: 0.0,
            ambiguous: amb_in,
            hits: dist <= r,
            why: if amb_in { "inside" } else { "" },
        });
        if dist <= r && !amb_in {
            return out;
        }
    }
    for (m, amb, hits, why) in face(o, d, tri.v0, tri.v1, tri.v2, r) {
        out.push(Piece {
            tri: index,
            feature: 1,
            t: m.v,
            dt: m.d,
            ambiguous: amb,
            hits,
            why,
        });
    }
    let mut add = |m: Option<(M, bool, bool)>, f: u32| {
        if let Some((t, amb, hits)) = m {
            let why = if !amb {
                ""
            } else if f == 2 {
                "edge"
            } else {
                "vertex"
            };
            out.push(Piece {
                tri: index,
                feature: f,
                t: t.v,
                dt: t.d,
                ambiguous: amb,
                hits,
                why,
            });
        }
    };
    for (u, v) in [(tri.v0, tri.v1), (tri.v1, tri.v2), (tri.v2, tri.v0)] {
        add(cylinder(o, d, u, v, r), 2);
    }
    for c in [tri.v0, tri.v1, tri.v2] {
        add(root(o - c, d, r * r), 3);
    }
    out
}

fn near(n: &BvhNode, o: DVec3, inv: DVec3, pad: f64, tmax: f64, out: &mut Vec<usize>) {
    let b = n.aabb();
    let lo = (b.min.as_dvec3() - DVec3::splat(pad) - o) * inv;
    let hi = (b.max.as_dvec3() + DVec3::splat(pad) - o) * inv;
    let t0 = lo.min(hi).max_element().max(0.0);
    let t1 = lo.max(hi).min_element();
    if !(t1 >= t0 && t0 <= tmax) {
        return;
    }
    match n {
        BvhNode::Leaf { triangles, .. } => out.extend_from_slice(triangles),
        BvhNode::Internal { left, right, .. } => {
            near(left, o, inv, pad, tmax, out);
            near(right, o, inv, pad, tmax, out);
        }
    }
}

/// Whether the ray lies in the ambiguous band, and why
pub fn band(bvh: &MeshBvh, ray: &ShellRay, r: f32) -> Option<String> {
    let root = bvh.root.as_ref()?;
    let (o, d) = (ray.origin.as_dvec3(), ray.dir.as_dvec3());
    let mut tris = Vec::new();
    // a padded box: any triangle with a decision near its threshold is within r (+ slack) of the ray
    near(
        root,
        o,
        d.recip(),
        f64::from(r) * 1.01 + 1e-4,
        f64::from(ray.tmax) * 1.01 + 1e-3,
        &mut tris,
    );
    let mut all = Vec::new();
    for k in tris {
        all.extend(pieces(ray.origin, ray.dir, &bvh.triangles[k], k as u32, r));
    }
    let within = |p: &Piece| f64::from(p.t) - p.dt <= f64::from(ray.tmax);
    let best = all
        .iter()
        .filter(|p| p.hits && p.t <= ray.tmax)
        .min_by(|x, y| x.t.total_cmp(&y.t).then(x.tri.cmp(&y.tri)));
    let reach = best.map_or(f64::from(ray.tmax), |b| f64::from(b.t) + b.dt);
    // an ambiguous decision at or before the answer
    if let Some(p) = all
        .iter()
        .find(|p| p.ambiguous && within(p) && f64::from(p.t) - p.dt <= reach)
    {
        return Some(format!(
            "ambiguous decision: {} tri {} feature {} t {} dt {:.3e}",
            p.why, p.tri, p.feature, p.t, p.dt
        ));
    }
    // a different piece whose t may cross the answer's (tmax counts when the answer is near it)
    if let Some(b) = best {
        if f64::from((ray.tmax - b.t).abs()) <= b.dt {
            return Some(format!("hit at {} within {:.3e} of tmax", b.t, b.dt));
        }
        if let Some(p) = all.iter().find(|p| {
            p.hits
                && (p.tri, p.feature) != (b.tri, b.feature)
                && f64::from((p.t - b.t).abs()) <= p.dt + b.dt
        }) {
            return Some(format!(
                "tie: tri {} f {} t {} against tri {} f {} t {}",
                p.tri, p.feature, p.t, b.tri, b.feature, b.t
            ));
        }
    }
    None
}

/// The bound on `t` for the CPU's answer `(tri, feature)` on this ray
pub fn t_bound(
    bvh: &MeshBvh,
    ray: &ShellRay,
    r: f32,
    tri: u32,
    feature: u32,
    t: f32,
) -> Option<f64> {
    pieces(ray.origin, ray.dir, &bvh.triangles[tri as usize], tri, r)
        .into_iter()
        .filter(|p| p.feature == feature && p.hits && p.t.to_bits() == t.to_bits())
        .map(|p| p.dt)
        .next()
}

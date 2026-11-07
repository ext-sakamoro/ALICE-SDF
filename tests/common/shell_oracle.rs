//! An independent truth for the first hit of a ray on a mesh thickened by `r`
//!
//! It shares nothing with `src/mesh/bvh_shell.rs`: the distance from a point to a triangle is
//! Ericson's closest point (*Real-Time Collision Detection* §5.1.5) in f64, the minimum is taken
//! over every triangle (no BVH), and the first hit is found by sphere tracing that distance in f64.
//!
//! A method's answer is judged against two shells, `r - eta` and `r + eta`: a hit at `t` must have
//! `dist(o + t d) <= r + eta` (it reached the grown shell) and the shrunk shell must not be entered
//! before `t`; a miss must never enter the shrunk shell. `eta` covers the method's f32 rounding:
//! every point the method forms has coordinates up to `M = max(|o|, |o + t d|)` and each of its
//! roundings moves it by at most `2^-24 M`; the root about the closest approach chains about a dozen
//! of them, so `eta = ETA + 16 * 2^-24 * M`. It stays far below `r` in these scenes.
//!
//! Author: Moroya Sakamoto

use glam::{DVec3, Vec3};

/// Allowance between the method's f32 shell and the f64 truth at unit scale
pub const ETA: f64 = 2e-5;

/// The allowance for points of magnitude up to `m` (see the module doc)
fn eta(m: f64) -> f64 {
    ETA + 16.0 * f64::powi(2.0, -24) * m
}

fn closest_on_segment(p: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let ab = b - a;
    let l = ab.dot(ab);
    if l == 0.0 {
        return a;
    }
    a + ab * ((p - a).dot(ab) / l).clamp(0.0, 1.0)
}

/// Ericson's closest point on triangle `abc` to `p`, in f64; a zero-area triangle takes the nearest
/// of its edges
pub fn closest(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 && d1 - d3 != 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 && d2 - d6 != 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 && (d4 - d3) + (d5 - d6) != 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let den = va + vb + vc;
    if den == 0.0 || !den.is_finite() {
        // zero area: the nearest edge point
        let qs = [
            closest_on_segment(p, a, b),
            closest_on_segment(p, b, c),
            closest_on_segment(p, c, a),
        ];
        return qs
            .into_iter()
            .min_by(|x, y| {
                (p - *x)
                    .length_squared()
                    .total_cmp(&(p - *y).length_squared())
            })
            .unwrap_or(a);
    }
    a + ab * (vb / den) + ac * (vc / den)
}

/// Triangles in f64
pub fn triangles(v: &[Vec3], ix: &[u32]) -> Vec<[DVec3; 3]> {
    ix.chunks_exact(3)
        .map(|t| [0, 1, 2].map(|k| v[t[k] as usize].as_dvec3()))
        .collect()
}

/// Distance from `p` to the nearest triangle
pub fn distance(tris: &[[DVec3; 3]], p: DVec3) -> f64 {
    tris.iter()
        .map(|[a, b, c]| (p - closest(p, *a, *b, *c)).length())
        .fold(f64::INFINITY, f64::min)
}

/// The first `t` in `[0, tmax]` with `distance <= level`, by sphere tracing in f64 (the distance
/// is 1-Lipschitz along a unit direction, so a step of `distance - level` never passes the shell)
pub fn first_entry(tris: &[[DVec3; 3]], o: DVec3, d: DVec3, tmax: f64, level: f64) -> Option<f64> {
    let d = d.normalize();
    let mut t = 0.0;
    for _ in 0..200_000 {
        let s = distance(tris, o + d * t) - level;
        if s <= 1e-9 {
            return Some(t);
        }
        t += s;
        if t > tmax {
            return None;
        }
    }
    panic!("the f64 trace did not settle: o {o:?} d {d:?}");
}

/// Why a method's answer is wrong, or `None` when the truth accepts it. `t` is in units of `dir`.
pub fn judge(
    tris: &[[DVec3; 3]],
    o: Vec3,
    dir: Vec3,
    tmax: f32,
    r: f32,
    got: Option<f32>,
) -> Option<String> {
    let (o, d) = (o.as_dvec3(), dir.as_dvec3());
    let len = d.length();
    let (r, tmax) = (f64::from(r), f64::from(tmax) * len);
    let far = (o + d * tmax).length().max(o.length());
    let shrunk = first_entry(tris, o, d, tmax, r - eta(far));
    match got {
        Some(t) => {
            let t = f64::from(t) * len;
            let p = o + d / len * t;
            let e = eta(o.length().max(p.length()));
            let at = distance(tris, p);
            if at > r + e {
                return Some(format!(
                    "hit at t {t} is {at} from the mesh, outside r + eta ({})",
                    r + e
                ));
            }
            if let Some(s) = shrunk {
                if s < t - e {
                    return Some(format!(
                        "the shrunk shell is entered at {s}, before the hit at {t}"
                    ));
                }
            }
            None
        }
        None => shrunk.map(|s| format!("missed, but the shrunk shell is entered at {s}")),
    }
}

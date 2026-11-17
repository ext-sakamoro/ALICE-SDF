//! Interval bounds on the gradient of the distance field over a box (`eval_interval_gradient`).
//!
//! [`crate::interval::eval_interval`] encloses the field's value over a box; this module encloses its
//! direction: an interval for each component of `∇d` that contains the gradient at every point of the
//! box where the field is differentiable (and, where two branches meet, both branches' gradients).
//!
//! # MODEL
//!
//! - Quantity: `∇d(p)` for `p` in a box `B`, one interval per axis, each within `[-1, 1]` (a distance
//!   field is 1-Lipschitz).
//! - Leaves, exact enclosures of the closed-form gradient:
//!   - sphere `|p| − r`: `p / |p|`, the box's coordinates times the reciprocal of its norm interval;
//!   - plane `n·p − h`: `n`;
//!   - box, rounded box (the box field minus a constant) and cylinder, the "box-like" form `|max(q, 0)| + min(max_k q_k, 0)` over component
//!     distances `q_k` with unit directions `u_k`: outside, the normalised non-negative combination
//!     `Σ max(q_k, 0) u_k / |max(q, 0)|`; inside, `u_k` of a component that can be the largest; a box
//!     that can lie on both sides takes the hull of both.
//! - Transforms: translate (same gradient), rotate (`∇_world = R ∇_local`, the interval vector times
//!   the rotation matrix), uniform scale (same gradient at the scaled point).
//! - Combinations: `min` / `max` take the gradient of the child that is the minimum (maximum) over the
//!   whole box when the children's value intervals do not overlap, and the hull of both otherwise,
//!   which is where two walls meet (the ridge between them). The smooth forms blend the two gradients
//!   with weights in `[0, 1]`, so the hull of both always encloses them.
//! - Value band ([`eval_interval_gradient_in_band`]): the enclosure may be asked only for the points
//!   whose field value lies in a band (a proof about the air asks for `d ≥ 0`, so the direction inside
//!   the material, which flips across its inner medial axis, does not count). Each rule keeps every
//!   point of the band:
//!   - a node whose value interval over the box misses the band contributes nothing;
//!   - min / max / `max(a, −b)`: where a child is the extreme, the node's value is that child's value, so
//!     each child gets the same band, and the cutter `b` of a subtraction gets its negation;
//!   - uniform scale `s·child(p/s)`: the child's value is the node's divided by `s`;
//!   - rounded box `box(p) − r`: the box's value is the node's plus `r`;
//!   - box-like leaves: the outside branch has values above 0, the inside branch the values
//!     `max_k q_k ≤ 0`, and each is kept only when its values can lie in the band;
//!   - smooth union lies at or below the minimum, so both children are at least the node's value (band
//!     `[lo, ∞)`); smooth intersection at or above the maximum (`(−∞, hi]`); smooth subtraction
//!     `smax(a, −b)` gives `a ≤ hi` and `b ≥ −hi`.
//! - Every other node kind returns [`GradientInterval::unknown`]: `[-1, 1]` on every axis and
//!   `known == false`. A caller must not read a narrow direction from it.
//! - The field's value at each node comes from `eval_interval` (the same enclosure the rest of the crate
//!   uses), so the choice between two children is as sound as that enclosure.
//! - Verified by `tests/test_interval_gradient.rs`: at random points of random boxes, for every covered
//!   kind, the point gradient lies inside the box's intervals; uncovered kinds report unknown.

use crate::compiled::real::{law_max_f32, law_min_f32};
use crate::interval::{eval_interval, next_down, next_up, Interval, Vec3Interval};
use crate::types::SdfNode;
use glam::{Mat3, Vec3};

/// An interval per component of the gradient of the distance field over a box.
#[derive(Debug, Clone, Copy)]
pub struct GradientInterval {
    /// `∂d/∂x` over the box.
    pub x: Interval,
    /// `∂d/∂y` over the box.
    pub y: Interval,
    /// `∂d/∂z` over the box.
    pub z: Interval,
    /// `false` when the node kind (or one below it) has no gradient enclosure here: the components are
    /// then `[-1, 1]` and say nothing about the direction.
    pub known: bool,
}

impl GradientInterval {
    /// Nothing is known about the direction: `[-1, 1]` on every axis.
    #[must_use]
    pub const fn unknown() -> Self {
        let all = Interval { lo: -1.0, hi: 1.0 };
        Self {
            x: all,
            y: all,
            z: all,
            known: false,
        }
    }

    /// A single direction (an exact gradient that does not vary over the box).
    #[must_use]
    pub const fn point(v: Vec3) -> Self {
        Self {
            x: Interval::point(v.x),
            y: Interval::point(v.y),
            z: Interval::point(v.z),
            known: true,
        }
    }

    /// Whether `v` lies within every component interval.
    #[must_use]
    pub fn contains(&self, v: Vec3) -> bool {
        self.x.contains(v.x) && self.y.contains(v.y) && self.z.contains(v.z)
    }

    fn hull(self, other: Self) -> Self {
        Self {
            x: self.x.hull(other.x),
            y: self.y.hull(other.y),
            z: self.z.hull(other.z),
            known: self.known && other.known,
        }
    }

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            known: self.known,
        }
    }

    /// Each component clamped to `[-1, 1]` (a distance field is 1-Lipschitz).
    fn unit_bounded(self) -> Self {
        let all = Interval { lo: -1.0, hi: 1.0 };
        let clip = |i: Interval| {
            let c = i.intersect(all);
            if c.lo > c.hi {
                all
            } else {
                c
            }
        };
        Self {
            x: clip(self.x),
            y: clip(self.y),
            z: clip(self.z),
            known: self.known,
        }
    }
}

/// `1 / i` for an interval strictly above zero, rounded outward; `None` otherwise.
fn recip(i: Interval) -> Option<Interval> {
    (i.lo > 0.0 && i.hi.is_finite()).then(|| Interval {
        lo: next_down(1.0 / i.hi),
        hi: next_up(1.0 / i.lo),
    })
}

/// The sign of `i` as an interval: `1` above zero, `-1` below, `[-1, 1]` when it can be either.
fn sign(i: Interval) -> Interval {
    if i.lo > 0.0 {
        Interval::point(1.0)
    } else if i.hi < 0.0 {
        Interval::point(-1.0)
    } else {
        Interval { lo: -1.0, hi: 1.0 }
    }
}

/// `a / sqrt(a² + b²)` over the box: as `sign(a) / sqrt(1 + (b/a)²)` when `a` stays away from zero
/// (monotone in the one ratio `|b| / |a|`, so the interval stays tight), else `a · inv_rho`.
fn radial_component(a: Interval, b: Interval, inv_rho: Interval) -> Interval {
    let abs_a = a.abs();
    match recip(abs_a) {
        Some(inv_a) if a.lo > 0.0 || a.hi < 0.0 => {
            let t = b.abs() * inv_a;
            let one_plus = Interval::point(1.0) + t.sqr();
            match recip(one_plus.sqrt()) {
                Some(mag) => sign(a) * mag,
                None => a * inv_rho,
            }
        }
        _ => a * inv_rho,
    }
}

/// Gradient enclosure of `|max(q, 0)| + min(max_k q_k, 0)` from the component distances `q` and their
/// unit directions `u` (see the module's MODEL), at the points whose value lies in `band`: the outside
/// branch (value > 0) only when the band reaches above zero, the inside branch (value `max_k q_k ≤ 0`)
/// only when that value can lie in the band. `None` when neither branch can.
fn box_like(q: &[Interval], u: &[[Interval; 3]], band: Interval) -> Option<GradientInterval> {
    let mut out: Option<GradientInterval> = None;
    let mut add = |g: GradientInterval| {
        out = Some(match out {
            Some(o) => o.hull(g),
            None => g,
        });
    };
    let positive: Vec<usize> = (0..q.len()).filter(|&k| q[k].hi > 0.0).collect();
    let outside_allowed = band.hi > 0.0;
    if outside_allowed {
        if let [k] = positive[..] {
            // only one component can be positive: outside, the direction is exactly that component's
            // (its weight is m_k / |m_k| = 1; dividing the two intervals would lose that)
            add(GradientInterval {
                x: u[k][0],
                y: u[k][1],
                z: u[k][2],
                known: true,
            });
        } else if !positive.is_empty() {
            let m: Vec<Interval> = q.iter().map(|q| q.max(Interval::ZERO)).collect();
            let norm = m
                .iter()
                .fold(Interval::ZERO, |acc, mk| acc + mk.sqr())
                .sqrt();
            let mut c = [Interval::ZERO; 3];
            if let Some(inv) = recip(norm) {
                for (mk, uk) in m.iter().zip(u) {
                    let w = *mk * inv;
                    for a in 0..3 {
                        c[a] = c[a] + w * uk[a];
                    }
                }
            } else {
                // the box touches the place where every component is zero: the weights are any
                // non-negative unit combination, each within [0, 1], of the components that can be
                // positive (the others weigh 0)
                for &k in &positive {
                    let uk = &u[k];
                    for a in 0..3 {
                        c[a] = c[a]
                            + Interval {
                                lo: law_min_f32(uk[a].lo, 0.0),
                                hi: law_max_f32(uk[a].hi, 0.0),
                            };
                    }
                }
            }
            add(GradientInterval {
                x: c[0],
                y: c[1],
                z: c[2],
                known: true,
            });
        }
    }
    if q.iter().all(|q| q.lo <= 0.0) {
        let floor = q.iter().map(|q| q.lo).fold(f32::NEG_INFINITY, law_max_f32);
        let top = q
            .iter()
            .map(|q| q.hi)
            .fold(f32::NEG_INFINITY, law_max_f32)
            .min(0.0);
        // the inside value is max_k q_k, within [floor, min(top, 0)]
        if band.lo <= top && band.hi >= floor {
            for (qk, uk) in q.iter().zip(u) {
                if qk.hi >= floor {
                    add(GradientInterval {
                        x: uk[0],
                        y: uk[1],
                        z: uk[2],
                        known: true,
                    });
                }
            }
        }
    }
    out.map(GradientInterval::unit_bounded)
}

/// Encloses the gradient of `node`'s distance field over `bounds` (see the module's MODEL).
#[must_use]
pub fn eval_interval_gradient(node: &SdfNode, bounds: Vec3Interval) -> GradientInterval {
    eval_interval_gradient_in_band(node, bounds, Interval::EVERYTHING)
        .unwrap_or_else(GradientInterval::unknown)
}

/// Encloses the gradient of `node`'s distance field at the points of `bounds` whose field value lies in
/// `band` (a proof about the air passes `[0, ∞)`, so the direction inside the material does not count).
/// `None`: no point of the box can have a value in the band.
#[must_use]
pub fn eval_interval_gradient_in_band(
    node: &SdfNode,
    bounds: Vec3Interval,
    band: Interval,
) -> Option<GradientInterval> {
    if !eval_interval(node, bounds).overlaps(band) {
        return None;
    }
    let zero = Interval::ZERO;
    let any = Interval::EVERYTHING;
    match node {
        SdfNode::Sphere { .. } => Some(match recip(bounds.length()) {
            Some(inv) => GradientInterval {
                x: bounds.x * inv,
                y: bounds.y * inv,
                z: bounds.z * inv,
                known: true,
            }
            .unit_bounded(),
            None => GradientInterval {
                known: true,
                ..GradientInterval::unknown()
            },
        }),
        SdfNode::Plane { normal, .. } => Some(GradientInterval::point(*normal)),
        SdfNode::Box3d { half_extents } => box_gradient(*half_extents, bounds, band),
        // a rounded box is the box field minus `r`: the same gradient, the box's value is `band + r`
        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => box_gradient(
            *half_extents,
            bounds,
            Interval {
                lo: band.lo + *round_radius,
                hi: band.hi + *round_radius,
            },
        ),
        SdfNode::Cylinder {
            radius,
            half_height,
        } => {
            let rho = bounds.length_xz();
            let radial = match recip(rho) {
                Some(inv) => [
                    radial_component(bounds.x, bounds.z, inv),
                    zero,
                    radial_component(bounds.z, bounds.x, inv),
                ],
                None => [
                    Interval { lo: -1.0, hi: 1.0 },
                    zero,
                    Interval { lo: -1.0, hi: 1.0 },
                ],
            };
            let q = [
                rho - Interval::point(*radius),
                bounds.y.abs() - Interval::point(*half_height),
            ];
            let u = [radial, [zero, sign(bounds.y), zero]];
            box_like(&q, &u, band)
        }
        SdfNode::Translate { child, offset } => {
            eval_interval_gradient_in_band(child, bounds.translate(*offset), band)
        }
        SdfNode::Rotate { child, rotation } => {
            let g = eval_interval_gradient_in_band(child, bounds.rotate(*rotation), band)?;
            if !g.known {
                return Some(GradientInterval::unknown());
            }
            let m = Mat3::from_quat(*rotation);
            let row = |r: Vec3| g.x * r.x + g.y * r.y + g.z * r.z;
            let t = m.transpose();
            Some(
                GradientInterval {
                    x: row(t.x_axis),
                    y: row(t.y_axis),
                    z: row(t.z_axis),
                    known: true,
                }
                .unit_bounded(),
            )
        }
        SdfNode::Scale { child, factor } => {
            let inv = 1.0 / *factor;
            if !(factor.is_finite() && *factor > 0.0 && inv.is_finite()) {
                return Some(GradientInterval::unknown());
            }
            let scaled = Vec3Interval {
                x: bounds.x * inv,
                y: bounds.y * inv,
                z: bounds.z * inv,
            };
            // d = s · child(p / s): the child's value is the node's divided by s
            let child_band = Interval {
                lo: band.lo * inv,
                hi: band.hi * inv,
            };
            eval_interval_gradient_in_band(child, scaled, child_band)
        }
        SdfNode::Union { a, b } => pick(a, b, bounds, band, false, false),
        SdfNode::Intersection { a, b } => pick(a, b, bounds, band, true, false),
        SdfNode::Subtraction { a, b } => pick(a, b, bounds, band, true, true),
        // a smooth minimum lies at or below the minimum, so both children are at least the node's
        // value; a smooth maximum lies at or above the maximum, so both are at most it
        SdfNode::SmoothUnion { a, b, .. } => {
            let child = Interval {
                lo: band.lo,
                hi: any.hi,
            };
            hull_opt(
                eval_interval_gradient_in_band(a, bounds, child),
                eval_interval_gradient_in_band(b, bounds, child),
            )
        }
        SdfNode::SmoothIntersection { a, b, .. } => {
            let child = Interval {
                lo: any.lo,
                hi: band.hi,
            };
            hull_opt(
                eval_interval_gradient_in_band(a, bounds, child),
                eval_interval_gradient_in_band(b, bounds, child),
            )
        }
        SdfNode::SmoothSubtraction { a, b, .. } => hull_opt(
            eval_interval_gradient_in_band(
                a,
                bounds,
                Interval {
                    lo: any.lo,
                    hi: band.hi,
                },
            ),
            eval_interval_gradient_in_band(
                b,
                bounds,
                Interval {
                    lo: -band.hi,
                    hi: any.hi,
                },
            )
            .map(GradientInterval::neg),
        ),
        _ => Some(GradientInterval::unknown()),
    }
}

fn box_gradient(h: Vec3, bounds: Vec3Interval, band: Interval) -> Option<GradientInterval> {
    let zero = Interval::ZERO;
    let q = [
        bounds.x.abs() - Interval::point(h.x),
        bounds.y.abs() - Interval::point(h.y),
        bounds.z.abs() - Interval::point(h.z),
    ];
    let u = [
        [sign(bounds.x), zero, zero],
        [zero, sign(bounds.y), zero],
        [zero, zero, sign(bounds.z)],
    ];
    box_like(&q, &u, band)
}

fn hull_opt(a: Option<GradientInterval>, b: Option<GradientInterval>) -> Option<GradientInterval> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.hull(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

/// `min(a, b)` (or `max`, or `max(a, −b)`) at the points whose value lies in `band`: where a child is
/// the extreme the node's value is that child's, so each child is asked for the same band (the cutter of
/// a subtraction for its negation); a child that is never the extreme over the box is left out.
fn pick(
    a: &SdfNode,
    b: &SdfNode,
    bounds: Vec3Interval,
    band: Interval,
    max: bool,
    negate_b: bool,
) -> Option<GradientInterval> {
    let da = eval_interval(a, bounds);
    let mut db = eval_interval(b, bounds);
    if negate_b {
        db = -db;
    }
    let ga = || eval_interval_gradient_in_band(a, bounds, band);
    let gb = || {
        if negate_b {
            eval_interval_gradient_in_band(b, bounds, -band).map(GradientInterval::neg)
        } else {
            eval_interval_gradient_in_band(b, bounds, band)
        }
    };
    let (a_wins, b_wins) = if max {
        (da.lo > db.hi, db.lo > da.hi)
    } else {
        (da.hi < db.lo, db.hi < da.lo)
    };
    if a_wins {
        ga()
    } else if b_wins {
        gb()
    } else {
        hull_opt(ga(), gb())
    }
}

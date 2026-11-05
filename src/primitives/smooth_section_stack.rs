//! Smooth section stack: closed spline sections stacked along local Y, any number of loops per
//! level, no point correspondence between levels. A branch (one loop splitting into several) is held
//! because the levels' 2D signed distance fields are interpolated, not their points.
//!
//! # The shape
//!
//! - **Levels** `y₀ < y₁ < … < y_{n−1}` (n ≥ 4, any spacing). The field is interpolated on
//!   `[y₁, y_{n−2}]`; the two outer levels only serve as tangent neighbours.
//! - **Loops**: per level one or more closed uniform cubic B-splines in local (x, z), each given by
//!   `m ≥ 4` control points (knots `0, 1, …, m`, periodic).
//! - **Per level** `d_i(x, z)`: the distance to the nearest loop curve (the loft's curve-section
//!   closest point, [`closed_curve_distance`]: a bounded span scan and Newton steps, the same on every
//!   backend), negative inside an odd number of loops (even-odd over the level's loops; every loop is stored
//!   counter-clockwise, reversed at construction when its curve's area is negative). It is
//!   1-Lipschitz in (x, z).
//! - **Along the axis**: on `[y_i, y_{i+1}]`, `t = (y − y_i)/D`, a cubic Hermite with finite-difference
//!   tangents `m_i = (d_{i+1} − d_{i−1})/(y_{i+1} − y_{i−1})`. That is Catmull-Rom on uniform levels,
//!   C1 across levels: `f = Σ_{k=−1..2} w_k(t) d_{i+k}`.
//! - **Ends**: past `y₁` / `y_{n−2}` the end field is combined with the distance past the end plane like
//!   an extrusion.
//!
//! # The bound
//!
//! - In the plane `|∇_xz f| ≤ S_i = max_t Σ|w_k|`. Along the axis `f_y = Σ_j c_j(t)(d_{j+1} − d_j)` with
//!   `c_j = −Σ_{k≤j} w'_k` (the weights' derivatives sum to 0), and `|d_{j+1} − d_j| ≤ H_j`, the Hausdorff
//!   distance of the two levels' curves (sampled, widened by half the sampling chord). So
//!   `|f_y| ≤ A_i = max_t Σ_j |c_j| H_j` and `L_i = √(S_i² + A_i²)` per interval.
//! - Knots `K_i = max(L_{i−1}, L_i)`, raised so neighbouring knots differ by at most
//!   `ε · min(K) · D / r₀`; `L(y)` is linear between knots and used as `(1 + 2ε) L(y)`. That covers the
//!   `f · L'/L²` term within `|f| ≤ r₀ L(y)`. Beyond it the field continues with slope `1/((1 + 2ε) L_max)`
//!   (a continuous clamp), so the output is 1-Lipschitz everywhere.
//!
//! Author: Moroya Sakamoto

use super::loft::closed_curve_distance;
use crate::compiled::real::LawMinMax;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Samples per span when bounding the Hausdorff distance between two levels' curves
pub const SMOOTH_STACK_HAUSDORFF_SAMPLES: usize = 32;
/// Samples of the interval parameter when bounding the interpolation weights
const WEIGHT_SAMPLES: usize = 400;

/// A stack of closed spline sections along local Y
///
/// Loading rebuilds the stack from its levels, loops, `r0` and `eps` (the checks of
/// [`SmoothSectionStack::new`], loops re-oriented counter-clockwise) and refuses stored knots or
/// `lmax` that differ from the recomputed ones bit for bit: a smaller stored bound would overstate
/// the distance and let sphere tracing step through the surface.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SmoothSectionStack {
    /// Heights of the levels (strictly increasing, at least 4)
    pub levels: Vec<f32>,
    /// Per level, its loops, each the control points of a closed uniform cubic B-spline in (x, z)
    pub loops: Vec<Vec<Vec<Vec2>>>,
    /// The bound's knots at the levels (see the module doc), computed at construction
    knots: Vec<f32>,
    /// The largest per-interval bound
    lmax: f32,
    /// The near-field radius r₀ and the knot slope limit ε
    r0: f32,
    eps: f32,
    /// Local bound (0 = the knot-smoothing bound above): each knot is the largest per-interval bound
    /// within `reach` of its two intervals and the value is `clamp(f / L(y), −reach, reach)`
    #[serde(default, skip_serializing_if = "is_zero")]
    reach: f32,
    /// The field's slope claim (1 with the knot-smoothing bound, else
    /// `1 + reach · max |L'| / min L` per interval)
    #[serde(skip)]
    slope: f32,
}

#[allow(
    clippy::trivially_copy_pass_by_ref,
    reason = "the signature serde asks for"
)]
fn is_zero(x: &f32) -> bool {
    x.to_bits() == 0
}

/// The stored form of a [`SmoothSectionStack`], checked on load
#[derive(Deserialize)]
struct SmoothSectionStackStored {
    levels: Vec<f32>,
    loops: Vec<Vec<Vec<Vec2>>>,
    knots: Vec<f32>,
    lmax: f32,
    r0: f32,
    eps: f32,
    #[serde(default)]
    reach: f32,
}

/// Read a stack through its checks ([`SmoothSectionStack::new`]; the stored bound must match)
///
/// # Errors
///
/// The data does not parse, a check refuses it, or the stored bound differs
#[allow(
    clippy::disallowed_methods,
    reason = "stored form other-type: the stack is read as a field of an SdfNode, whose stored tree goes through the checked loader"
)]
// ALLOW-UNWIRED: called by the derived Deserialize of SdfNode through #[serde(deserialize_with = "crate::primitives::smooth_section_stack::deserialize_stack")]
pub fn deserialize_stack<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<SmoothSectionStack, D::Error> {
    SmoothSectionStack::try_from(SmoothSectionStackStored::deserialize(d)?)
        .map_err(serde::de::Error::custom)
}

impl TryFrom<SmoothSectionStackStored> for SmoothSectionStack {
    type Error = String;
    fn try_from(v: SmoothSectionStackStored) -> Result<Self, String> {
        let s = if v.reach == 0.0 {
            Self::new(v.levels, v.loops, v.r0, v.eps)?
        } else {
            Self::new_local(v.levels, v.loops, v.reach)?
        };
        let same = s.lmax.to_bits() == v.lmax.to_bits()
            && s.knots.len() == v.knots.len()
            && s.knots
                .iter()
                .zip(&v.knots)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        if !same {
            return Err("smooth section stack: stored knots or lmax differ from the bound recomputed from its levels and loops; rebuild the stack".into());
        }
        Ok(s)
    }
}

#[inline]
fn hermite(levels: &[f32], i: usize, t: f32) -> ([f32; 4], [f32; 4]) {
    let d = levels[i + 1] - levels[i];
    let a1 = d / (levels[i + 1] - levels[i - 1]);
    let a2 = d / (levels[i + 2] - levels[i]);
    let (t2, t3) = (t * t, t * t * t);
    let (h00, h10, h01, h11) = (
        2.0 * t3 - 3.0 * t2 + 1.0,
        t3 - 2.0 * t2 + t,
        -2.0 * t3 + 3.0 * t2,
        t3 - t2,
    );
    let (g00, g10, g01, g11) = (
        6.0 * t2 - 6.0 * t,
        3.0 * t2 - 4.0 * t + 1.0,
        -6.0 * t2 + 6.0 * t,
        3.0 * t2 - 2.0 * t,
    );
    (
        [-h10 * a1, h00 - h11 * a2, h01 + h10 * a1, h11 * a2],
        [
            -g10 * a1 / d,
            (g00 - g11 * a2) / d,
            (g01 + g10 * a1) / d,
            g11 * a2 / d,
        ],
    )
}

/// The signed distance from `q` to a level's loops: the nearest curve, negative inside an odd number
pub fn sdf_spline_loops_2d(q: Vec2, loops: &[Vec<Vec2>]) -> f32 {
    let mut best = f32::INFINITY;
    let mut inside = false;
    for lp in loops {
        let m = lp.len() as isize;
        let d = closed_curve_distance(
            m,
            &|j: isize| lp[j.rem_euclid(m) as usize],
            &|i: isize| i as f32,
            q,
        );
        if d < 0.0 {
            inside = !inside;
        }
        best = best.law_min(d.abs());
    }
    if inside {
        -best
    } else {
        best
    }
}

/// A point of a closed uniform cubic B-spline at parameter `u ∈ [0, m)`
fn spline_point(lp: &[Vec2], u: f32) -> Vec2 {
    let m = lp.len() as isize;
    let i = u.floor() as isize;
    let f = u - i as f32;
    let g = 1.0 - f;
    let w = [
        g * g * g / 6.0,
        (3.0 * f * f * f - 6.0 * f * f + 4.0) / 6.0,
        (-3.0 * f * f * f + 3.0 * f * f + 3.0 * f + 1.0) / 6.0,
        f * f * f / 6.0,
    ];
    (0..4)
        .map(|k| w[k] * lp[(i - 1 + k as isize).rem_euclid(m) as usize])
        .fold(Vec2::ZERO, |a, b| a + b)
}

/// The signed area of a loop's curve (shoelace over [`SMOOTH_STACK_HAUSDORFF_SAMPLES`] points per span):
/// positive when the curve runs counter-clockwise in (x, z)
/// The loop's curve sampled at [`SMOOTH_STACK_HAUSDORFF_SAMPLES`] per span (the simplicity check)
fn sampled_curve(lp: &[Vec2]) -> Vec<Vec2> {
    let n = lp.len() * SMOOTH_STACK_HAUSDORFF_SAMPLES;
    (0..n)
        .map(|s| spline_point(lp, s as f32 / SMOOTH_STACK_HAUSDORFF_SAMPLES as f32))
        .collect()
}

fn curve_area(lp: &[Vec2]) -> f32 {
    let n = lp.len() * SMOOTH_STACK_HAUSDORFF_SAMPLES;
    let mut a = 0.0f32;
    let mut prev = spline_point(lp, 0.0);
    for s in 1..=n {
        let p = spline_point(
            lp,
            s as f32 / SMOOTH_STACK_HAUSDORFF_SAMPLES as f32 % lp.len() as f32,
        );
        a += prev.x * p.y - p.x * prev.y;
        prev = p;
    }
    0.5 * a
}

/// An upper bound of the Hausdorff distance between two levels' curves
///
/// Each curve sampled at
/// [`SMOOTH_STACK_HAUSDORFF_SAMPLES`] per span, each sample's distance to the other level, the largest
/// plus half the largest sampling chord
pub fn levels_hausdorff_bound(a: &[Vec<Vec2>], b: &[Vec<Vec2>]) -> f32 {
    let one = |x: &[Vec<Vec2>], y: &[Vec<Vec2>]| -> f32 {
        let (mut h, mut chord) = (0.0f32, 0.0f32);
        for lp in x {
            let n = lp.len() * SMOOTH_STACK_HAUSDORFF_SAMPLES;
            let mut prev = spline_point(lp, 0.0);
            for s in 1..=n {
                let p = spline_point(
                    lp,
                    s as f32 / SMOOTH_STACK_HAUSDORFF_SAMPLES as f32 % lp.len() as f32,
                );
                chord = chord.law_max((p - prev).length());
                h = h.law_max(sdf_spline_loops_2d(p, y).abs());
                prev = p;
            }
        }
        h + 0.5 * chord
    };
    one(a, b).law_max(one(b, a))
}

/// Per-interval bounds `L_i` on `[y_i, y_{i+1}]` (index 1 ..= n−3, the others 0)
fn interval_bounds(levels: &[f32], loops: &[Vec<Vec<Vec2>>]) -> Vec<f32> {
    let n = levels.len();
    let hs: Vec<f32> = (0..n - 1)
        .map(|j| levels_hausdorff_bound(&loops[j], &loops[j + 1]))
        .collect();
    // per-interval bounds on [y_i, y_{i+1}], i = 1 .. n−3
    let mut li = vec![0.0f32; n - 1];
    for i in 1..n - 2 {
        let mut worst = 0.0f32;
        for k in 0..=WEIGHT_SAMPLES {
            let (w, wd) = hermite(levels, i, k as f32 / WEIGHT_SAMPLES as f32);
            let s: f32 = w.iter().map(|x| x.abs()).sum();
            let (c0, c1, c2) = (-wd[0], -(wd[0] + wd[1]), -(wd[0] + wd[1] + wd[2]));
            let a = c0.abs() * hs[i - 1] + c1.abs() * hs[i] + c2.abs() * hs[i + 1];
            worst = worst.law_max((s * s + a * a).sqrt());
        }
        li[i] = worst;
    }
    li
}

impl SmoothSectionStack {
    /// A stack from its levels and loops, with the near-field radius `r0` (> 0) and the knot slope
    /// limit `eps` (0 < ε < 1); Err on degenerate input
    pub fn new(
        levels: Vec<f32>,
        mut loops: Vec<Vec<Vec<Vec2>>>,
        r0: f32,
        eps: f32,
    ) -> Result<Self, String> {
        let n = levels.len();
        if n < 4 || loops.len() != n {
            return Err(format!(
                "smooth section stack: {} levels and {} loop sets (at least 4, equal)",
                n,
                loops.len()
            ));
        }
        if levels.iter().any(|y| !y.is_finite()) || levels.windows(2).any(|w| w[1] <= w[0]) {
            return Err(
                "smooth section stack: levels must be finite and strictly increasing".into(),
            );
        }
        if !(r0.is_finite() && r0 > 0.0 && eps.is_finite() && eps > 0.0 && eps < 1.0) {
            return Err("smooth section stack: r0 must be > 0 and eps in (0, 1)".into());
        }
        for (i, lv) in loops.iter().enumerate() {
            if lv.is_empty() {
                return Err(format!("smooth section stack: level {i} has no loop"));
            }
            for lp in lv {
                if lp.len() < 4 {
                    return Err(format!(
                        "smooth section stack: a loop at level {i} has fewer than 4 control points"
                    ));
                }
                if lp.iter().any(|p| !p.is_finite()) {
                    return Err(format!(
                        "smooth section stack: a non-finite control point at level {i}"
                    ));
                }
                if super::simple_loop::closed_polyline_crosses(&sampled_curve(lp)) {
                    return Err(format!(
                        "smooth section stack: a loop at level {i} crosses itself"
                    ));
                }
            }
        }
        // every loop stored counter-clockwise: the per-loop distance is signed by the side of the
        // tangent, so a clockwise loop would read its inside as outside and the even-odd count would be
        // wrong. The orientation is the sign of the sampled curve's area (not the control polygon's);
        // a reversed control-point order is the same curve run backwards
        for (i, lv) in loops.iter_mut().enumerate() {
            for lp in lv.iter_mut() {
                let a = curve_area(lp);
                if !(a.is_finite() && a != 0.0) {
                    return Err(format!(
                        "smooth section stack: a loop at level {i} encloses no area"
                    ));
                }
                if a < 0.0 {
                    lp.reverse();
                }
            }
        }
        let li = interval_bounds(&levels, &loops);
        let lmax = li.iter().fold(0.0f32, |a, &b| a.law_max(b));
        let mut knots: Vec<f32> = (0..n)
            .map(|i| li[i.saturating_sub(1).max(1).min(n - 3)].law_max(li[i.max(1).min(n - 3)]))
            .collect();
        for _ in 0..2 * n {
            for i in 0..n - 1 {
                let lim = eps * knots[i].law_min(knots[i + 1]) * (levels[i + 1] - levels[i]) / r0;
                if knots[i + 1] < knots[i] - lim {
                    knots[i + 1] = knots[i] - lim;
                }
                if knots[i] < knots[i + 1] - lim {
                    knots[i] = knots[i + 1] - lim;
                }
            }
        }
        Ok(Self {
            levels,
            loops,
            knots,
            lmax,
            r0,
            eps,
            reach: 0.0,
            slope: 1.0,
        })
    }

    /// A stack with the local bound: knot `i` is the largest per-interval bound over the intervals
    /// that meet `[y_{i−1} − reach, y_{i+1} + reach]`, so for `y` in `[y_i, y_{i+1}]` both knots bound
    /// the field's slope on the ball of radius `reach` around the point; the value is
    /// `clamp(f / L(y), −reach, reach)` (never more than the distance to the surface, the distance's
    /// own scale where the levels change slowly nearby). Err on degenerate input or `reach` ≤ 0
    pub fn new_local(
        levels: Vec<f32>,
        loops: Vec<Vec<Vec<Vec2>>>,
        reach: f32,
    ) -> Result<Self, String> {
        if !(reach.is_finite() && reach > 0.0) {
            return Err("smooth section stack: reach must be > 0".into());
        }
        let base = Self::new(levels, loops, reach, 0.5)?;
        let n = base.levels.len();
        let y = &base.levels;
        let li = interval_bounds(y, &base.loops);
        let mut knots = vec![0.0f32; n];
        for (i, kn) in knots.iter_mut().enumerate() {
            let lo = y[i.saturating_sub(1)] - reach;
            let hi = y[(i + 1).min(n - 1)] + reach;
            for k in 1..n - 2 {
                if y[k + 1] >= lo && y[k] <= hi {
                    *kn = kn.law_max(li[k]);
                }
            }
        }
        // neighbouring knots differ by at most K · D / s (s = reach / 3): |L'| / L stays ≤ 1 / s, so the
        // f · L' / L² part of the gradient is at most |f / L| / s near the surface (no visible step in
        // the shading where the levels' bounds jump), while a large bound spreads only ~s per e-fold
        let sm = reach / 3.0;
        for _ in 0..2 * n {
            for i in 0..n - 1 {
                let lim = knots[i].law_min(knots[i + 1]) * (y[i + 1] - y[i]) / sm;
                if knots[i + 1] < knots[i] - lim {
                    knots[i + 1] = knots[i] - lim;
                }
                if knots[i] < knots[i + 1] - lim {
                    knots[i] = knots[i + 1] - lim;
                }
            }
        }
        let mut slope = 1.0f32;
        for i in 1..n - 2 {
            let dk = (knots[i + 1] - knots[i]).abs() / (y[i + 1] - y[i]);
            slope = slope.law_max(1.0 + reach * dk / knots[i].law_min(knots[i + 1]));
        }
        Ok(Self {
            knots,
            reach,
            slope,
            ..base
        })
    }

    /// The local bound's reach (0 for the knot-smoothing bound)
    #[must_use]
    pub fn reach(&self) -> f32 {
        self.reach
    }

    /// The field's slope claim (1 for the knot-smoothing bound)
    #[must_use]
    pub fn slope_claim(&self) -> f32 {
        self.slope
    }

    /// The bound's knots at the levels
    #[must_use]
    pub fn knots(&self) -> &[f32] {
        &self.knots
    }

    /// `(lmax, r0, eps)` for the shader emit
    #[must_use]
    pub const fn shader_params(&self) -> (f32, f32, f32) {
        (self.lmax, self.r0, self.eps)
    }

    /// The largest per-interval bound
    #[must_use]
    pub const fn lipschitz(&self) -> f32 {
        self.lmax
    }

    /// The interpolated, unnormalized field and the bound at height `y` (clamped to the interpolated range)
    fn raw(&self, y: f32, q: Vec2) -> (f32, f32) {
        let n = self.levels.len();
        let yc = y.law_max(self.levels[1]).law_min(self.levels[n - 2]);
        let mut i = 1;
        while i < n - 3 && yc > self.levels[i + 1] {
            i += 1;
        }
        let t = ((yc - self.levels[i]) / (self.levels[i + 1] - self.levels[i]))
            .law_max(0.0)
            .law_min(1.0);
        let (w, _) = hermite(&self.levels, i, t);
        let mut f = 0.0f32;
        for (k, wk) in w.iter().enumerate() {
            f += wk * sdf_spline_loops_2d(q, &self.loops[i - 1 + k]);
        }
        let e = if self.reach > 0.0 {
            1.0
        } else {
            1.0 + 2.0 * self.eps
        };
        let l = ((1.0 - t) * self.knots[i] + t * self.knots[i + 1]) * e;
        (f, l)
    }

    /// The field at p (local coordinates, the stack's axis along Y)
    #[must_use]
    pub fn eval(&self, p: Vec3) -> f32 {
        let n = self.levels.len();
        let q = Vec2::new(p.x, p.z);
        let (f, l) = self.raw(p.y, q);
        let lfar = self.lmax * (1.0 + 2.0 * self.eps);
        let d2 = if self.reach > 0.0 {
            (f / l).law_max(-self.reach).law_min(self.reach)
        } else if f.abs() <= self.r0 * l {
            f / l
        } else {
            f.signum() * (self.r0 + (f.abs() - self.r0 * l) / lfar)
        };
        // past the end levels: an extrusion with the distance past the end plane
        let dw = (self.levels[1] - p.y).law_max(p.y - self.levels[n - 2]);
        let o = Vec2::new(d2.law_max(0.0), dw.law_max(0.0));
        d2.law_max(dw).law_min(0.0) + o.length()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
        // control points of a closed uniform cubic B-spline close to a circle of radius r: the control polygon
        // scaled by 3 / (2 + cos(2π/m)) puts the curve's points on the circle at the knots
        let k = 3.0 / (2.0 + alice_det_math::sin_cos(std::f32::consts::TAU / m as f32).1);
        (0..m)
            .map(|j| {
                let a = std::f32::consts::TAU * j as f32 / m as f32;
                {
                    let (s, co) = alice_det_math::sin_cos(a);
                    c + Vec2::new(co, s) * r * k
                }
            })
            .collect()
    }
    fn lcg(s: &mut u64) -> f32 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*s >> 40) as f32) / ((1u64 << 24) as f32)
    }

    // oracle (closed form): a constant stack of one circle is the circle extruded over [y₁, y_{n−2}] = [1, 3]
    // and capped there. Near the wall (|q| − r within r₀ L, the unclamped field) the in-plane field is
    // a = (|q| − r) / L, and the output is the capped extrusion of it in eval's units,
    // min(max(a, dw), 0) + |(max(a, 0), max(dw, 0))| with dw the distance past the end planes, up to the
    // curve's own approximation of the circle (< 2e-3 r for m = 32, so 4e-3 / L); compared at heights
    // inside, on the end planes and past the caps. The sign is the capped extrusion's everywhere (on the
    // end planes inside the circle the capped distance is 0)
    #[test]
    fn a_constant_stack_is_the_extruded_circle() {
        let lp = circle(1.0, Vec2::ZERO, 32);
        let st =
            SmoothSectionStack::new(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![vec![lp]; 5], 0.5, 0.1)
                .unwrap();
        let mut n = 0;
        for &(x, z) in &[
            (0.0f32, 0.0f32),
            (0.5, 0.2),
            (0.9, 0.1),
            (1.0, 0.0),
            (1.1, -0.1),
            (1.4, -0.3),
            (0.0, 1.2),
        ] {
            for &y in &[0.5f32, 1.0, 1.5, 2.3, 3.0, 3.6] {
                let dc = (x * x + z * z).sqrt() - 1.0;
                let dw = (1.0 - y).max(y - 3.0);
                let want = dc.max(dw).min(0.0) + Vec2::new(dc.max(0.0), dw.max(0.0)).length();
                let l = st.knots()[1] * (1.0 + 0.2);
                if dc.abs() < 0.5 * l {
                    let a = dc / l;
                    let capped = a.max(dw).min(0.0) + Vec2::new(a.max(0.0), dw.max(0.0)).length();
                    let got = st.eval(Vec3::new(x, y, z));
                    assert!(
                        (got - capped).abs() < 4e-3 / l,
                        "({x}, {y}, {z}): got {got} want {capped}"
                    );
                    n += 1;
                }
                if want.abs() > 1e-2 {
                    assert_eq!(
                        st.eval(Vec3::new(x, y, z)) < 0.0,
                        want < 0.0,
                        "sign at ({x}, {y}, {z})"
                    );
                }
            }
        }
        println!("near-wall values compared: {n}");
        assert!(n >= 36, "only {n} near-wall values compared");
    }

    // oracle (property): the output is 1-Lipschitz near and far, on a stack where one loop splits into two
    // (the topology change the node exists for) and the bound varies along the axis
    #[test]
    fn a_split_keeps_the_sign_and_is_one_lipschitz_near_and_far() {
        let one = vec![circle(1.0, Vec2::ZERO, 16)];
        let two = vec![
            circle(0.45, Vec2::new(-0.55, 0.0), 16),
            circle(0.45, Vec2::new(0.55, 0.0), 16),
        ];
        let lv = vec![one.clone(), one.clone(), one, two.clone(), two.clone(), two];
        let st = SmoothSectionStack::new(vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5], lv, 0.3, 0.1).unwrap();
        let mut seed = 7u64;
        let mut worst = 0.0f32;
        for _ in 0..20000 {
            let p = Vec3::new(
                lcg(&mut seed) * 4.0 - 2.0,
                lcg(&mut seed) * 3.5 - 0.5,
                lcg(&mut seed) * 4.0 - 2.0,
            );
            let dv = Vec3::new(
                lcg(&mut seed) - 0.5,
                lcg(&mut seed) - 0.5,
                lcg(&mut seed) - 0.5,
            )
            .normalize()
                * 1e-2;
            let r = (st.eval(p + dv) - st.eval(p)).abs() / 1e-2;
            worst = worst.max(r);
        }
        assert!(worst <= 1.0 + 1e-3, "Lipschitz ratio {worst}");
        // the sign: deep inside each lobe and between the lobes past the split, inside [y₁, y_{n−2}] = [0.5, 2]
        // (on the end planes the capped field is 0)
        assert!(
            st.eval(Vec3::new(0.0, 0.75, 0.0)) < 0.0,
            "inside the single loop"
        );
        assert!(
            st.eval(Vec3::new(-0.55, 1.9, 0.0)) < 0.0 && st.eval(Vec3::new(0.55, 1.9, 0.0)) < 0.0,
            "inside both lobes"
        );
        assert!(
            st.eval(Vec3::new(0.0, 1.9, 0.0)) > 0.0,
            "the gap between the lobes is outside"
        );
        assert!(st.eval(Vec3::new(0.0, 0.5, 3.0)) > 0.0, "outside");
    }

    // oracle (property): C1 across a level: the y-derivative from either side of an interior level agrees
    #[test]
    fn the_field_is_c1_across_levels() {
        let lv: Vec<Vec<Vec<Vec2>>> = [1.0f32, 1.2, 0.9, 1.3, 1.1, 1.0]
            .iter()
            .map(|&r| vec![circle(r, Vec2::ZERO, 16)])
            .collect();
        let st =
            SmoothSectionStack::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], lv, 10.0, 0.5).unwrap();
        let q = Vec2::new(1.6, 0.2);
        for &y in &[2.0f32, 3.0] {
            let h = 1e-3;
            let (f0, _) = st.raw(y, q);
            let (fm, _) = st.raw(y - h, q);
            let (fp, _) = st.raw(y + h, q);
            let (dl, dr) = ((f0 - fm) / h, (fp - f0) / h);
            assert!((dl - dr).abs() < 2e-2, "y {y}: left {dl} right {dr}");
        }
    }

    // degenerate input is refused
    #[test]
    fn degenerate_input_is_refused() {
        let c = vec![circle(1.0, Vec2::ZERO, 8)];
        assert!(
            SmoothSectionStack::new(vec![0.0, 1.0, 2.0], vec![c.clone(); 3], 1.0, 0.1).is_err()
        );
        assert!(
            SmoothSectionStack::new(vec![0.0, 1.0, 1.0, 2.0], vec![c.clone(); 4], 1.0, 0.1)
                .is_err()
        );
        assert!(SmoothSectionStack::new(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![c.clone(), vec![], c.clone(), c.clone()],
            1.0,
            0.1
        )
        .is_err());
        assert!(SmoothSectionStack::new(
            vec![0.0, 1.0, 2.0, 3.0],
            vec![c.clone(), vec![vec![Vec2::ZERO; 3]], c.clone(), c.clone()],
            1.0,
            0.1
        )
        .is_err());
        assert!(SmoothSectionStack::new(
            vec![0.0, 1.0, 2.0, f32::NAN],
            vec![c.clone(); 4],
            1.0,
            0.1
        )
        .is_err());
        assert!(
            SmoothSectionStack::new(vec![0.0, 1.0, 2.0, 3.0], vec![c.clone(); 4], 0.0, 0.1)
                .is_err()
        );
        assert!(SmoothSectionStack::new(vec![0.0, 1.0, 2.0, 3.0], vec![c; 4], 1.0, 1.5).is_err());
    }

    // oracle (invariance): the field does not depend on the loops' orientation. The same levels with every
    // loop given clockwise (control points reversed), an outer loop with a hole and a separate island,
    // give the bit-equal field; and the sign is even-odd (inside the hole is outside)
    #[test]
    fn a_clockwise_loop_gives_the_same_field() {
        let lv = |r: f32| {
            vec![
                circle(1.0 * r, Vec2::ZERO, 16),
                circle(0.4 * r, Vec2::new(0.1, 0.0), 12),
                circle(0.3, Vec2::new(2.2, 0.0), 8),
            ]
        };
        let ccw: Vec<Vec<Vec<Vec2>>> = [1.0f32, 1.05, 1.1, 1.0, 0.95]
            .iter()
            .map(|&r| lv(r))
            .collect();
        let cw: Vec<Vec<Vec<Vec2>>> = ccw
            .iter()
            .map(|l| {
                l.iter()
                    .map(|lp| lp.iter().rev().copied().collect())
                    .collect()
            })
            .collect();
        let mixed: Vec<Vec<Vec<Vec2>>> = ccw
            .iter()
            .enumerate()
            .map(|(i, l)| if i % 2 == 0 { cw[i].clone() } else { l.clone() })
            .collect();
        let a = SmoothSectionStack::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], ccw, 0.3, 0.1).unwrap();
        let b = SmoothSectionStack::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], cw, 0.3, 0.1).unwrap();
        let c = SmoothSectionStack::new(vec![0.0, 0.5, 1.0, 1.5, 2.0], mixed, 0.3, 0.1).unwrap();
        let mut seed = 3u64;
        let mut n = 0;
        for _ in 0..4000 {
            let p = Vec3::new(
                lcg(&mut seed) * 6.0 - 2.5,
                lcg(&mut seed) * 2.4 - 0.2,
                lcg(&mut seed) * 4.0 - 2.0,
            );
            let fa = a.eval(p);
            assert_eq!(
                fa.to_bits(),
                b.eval(p).to_bits(),
                "clockwise loops changed the field at {p:?}"
            );
            assert_eq!(
                fa.to_bits(),
                c.eval(p).to_bits(),
                "mixed orientations changed the field at {p:?}"
            );
            n += 1;
        }
        assert_eq!(n, 4000);
        // even-odd: inside the outer loop, inside the hole (outside), inside the island
        assert!(
            b.eval(Vec3::new(-0.75, 1.0, 0.0)) < 0.0,
            "between the outer loop and the hole is inside"
        );
        assert!(
            b.eval(Vec3::new(0.1, 1.0, 0.0)) > 0.0,
            "inside the hole is outside"
        );
        assert!(
            b.eval(Vec3::new(2.2, 1.0, 0.0)) < 0.0,
            "inside the island is inside"
        );
    }

    // determinism: the same input gives the same bits
    #[test]
    fn evaluation_is_deterministic() {
        let lv: Vec<Vec<Vec<Vec2>>> = (0..6)
            .map(|i| vec![circle(1.0 + 0.1 * i as f32, Vec2::ZERO, 12)])
            .collect();
        let a = SmoothSectionStack::new((0..6).map(|i| i as f32).collect(), lv.clone(), 0.5, 0.1)
            .unwrap();
        let b = SmoothSectionStack::new((0..6).map(|i| i as f32).collect(), lv, 0.5, 0.1).unwrap();
        for k in 0..200 {
            let p = Vec3::new(
                alice_det_math::sin_cos(k as f32 * 0.37).0 * 2.0,
                1.0 + (k % 30) as f32 * 0.1,
                alice_det_math::sin_cos(k as f32 * 0.11).1 * 2.0,
            );
            assert_eq!(a.eval(p).to_bits(), b.eval(p).to_bits());
        }
    }

    // oracle (closed form): a constant stack of one circle with the local bound is the circle extruded
    // and capped, in units of the knot L (constant here), clamped to ±reach: compared where the
    // unclamped and the clamped values are both seen
    #[test]
    fn a_local_constant_stack_is_the_clamped_extruded_circle() {
        let lp = circle(1.0, Vec2::ZERO, 32);
        let reach = 0.3;
        let st =
            SmoothSectionStack::new_local(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![vec![lp]; 5], reach)
                .unwrap();
        let l = st.knots()[2];
        assert!(
            st.knots().iter().all(|&k| k == l),
            "constant stack, constant knots {:?}",
            st.knots()
        );
        let (mut n, mut clamped) = (0, 0);
        for &(x, z) in &[
            (0.0f32, 0.0f32),
            (0.5, 0.2),
            (0.9, 0.1),
            (1.1, -0.1),
            (1.4, -0.3),
            (0.0, 1.9),
        ] {
            for &y in &[0.5f32, 1.5, 2.3, 3.6] {
                let dc = (x * x + z * z).sqrt() - 1.0;
                let dw = (1.0 - y).max(y - 3.0);
                let a = (dc / l).clamp(-reach, reach);
                let want = a.max(dw).min(0.0) + Vec2::new(a.max(0.0), dw.max(0.0)).length();
                let got = st.eval(Vec3::new(x, y, z));
                assert!(
                    (got - want).abs() < 4e-3 / l,
                    "({x}, {y}, {z}): got {got} want {want}"
                );
                if (dc / l).abs() > reach {
                    clamped += 1;
                }
                n += 1;
            }
        }
        assert!(n == 24 && clamped >= 4, "compared {n}, clamped {clamped}");
    }

    /// Zero-set samples of `f`: bisection on every sign change along lines in x and lines in y (steps
    /// of `h` across, `h / 4` along), over the box `xs` × `ys` × `zs`
    fn zero_set(
        f: &dyn Fn(Vec3) -> f32,
        ys: (f32, f32),
        xs: (f32, f32),
        zs: (f32, f32),
        h: f32,
    ) -> Vec<Vec3> {
        let mut out = Vec::new();
        let steps = |a: f32, b: f32| ((b - a) / h).round() as usize;
        // one line from p0 along unit axis `dir` over `len`
        let mut line = |p0: Vec3, dir: Vec3, len: f32| {
            let n = (len / h).round() as usize * 4;
            let mut prev = (0.0f32, f(p0));
            for k in 1..=n {
                let t = k as f32 * h / 4.0;
                let v = f(p0 + dir * t);
                if (v < 0.0) != (prev.1 < 0.0) {
                    let (mut a, mut b) = (prev.0, t);
                    for _ in 0..30 {
                        let m = 0.5 * (a + b);
                        if (f(p0 + dir * m) < 0.0) == (prev.1 < 0.0) {
                            a = m
                        } else {
                            b = m
                        }
                    }
                    out.push(p0 + dir * (0.5 * (a + b)));
                }
                prev = (t, v);
            }
        };
        for iz in 0..=steps(zs.0, zs.1) {
            let z = zs.0 + iz as f32 * h;
            for iy in 0..=steps(ys.0, ys.1) {
                line(
                    Vec3::new(xs.0, ys.0 + iy as f32 * h, z),
                    Vec3::X,
                    xs.1 - xs.0,
                );
            }
            for ix in 0..=steps(xs.0, xs.1) {
                line(
                    Vec3::new(xs.0 + ix as f32 * h, ys.0, z),
                    Vec3::Y,
                    ys.1 - ys.0,
                );
            }
        }
        out
    }

    // oracle (the sphere-tracing contract, independent of the bound's formula): |value| never exceeds
    // the distance to the field's own zero set (sampled by bisection; the sampling gap is added). The
    // stack flares at one level (radius 0.5 → 1.5 → 0.5 over one level spacing of 0.1, a thin disc at
    // y ≈ 1.2), so below and above the disc, within the reach but outside the knots' own intervals, the plain
    // cylinder's value would overshoot the distance up to the disc: the knots must take the disc's
    // interval bounds over the reach
    #[test]
    fn a_local_stack_never_gives_more_than_the_distance_beside_a_bulge() {
        let levels: Vec<f32> = (0..25).map(|i| 0.1 * i as f32).collect();
        let loops: Vec<Vec<Vec<Vec2>>> = (0..25)
            .map(|i| vec![circle(if i == 12 { 1.5 } else { 0.5 }, Vec2::ZERO, 24)])
            .collect();
        let reach = 0.5;
        let st = SmoothSectionStack::new_local(levels, loops, reach).unwrap();
        let f = |p: Vec3| st.eval(p);
        let h = 0.01;
        let zs = zero_set(&f, (0.1, 2.3), (-2.0, 2.0), (-0.02, 0.02), h);
        assert!(zs.len() > 300, "zero-set samples {}", zs.len());
        let gap = h * 1.5;
        let (mut n, mut tight) = (0, 0);
        // two bands: below the disc (y 0.62 ..= 1.1) and above it (1.3 ..= 1.78)
        for iy in 0..30 {
            for ix in 0..30 {
                let y = if iy < 15 {
                    0.62 + 0.48 * iy as f32 / 14.0
                } else {
                    1.3 + 0.48 * (iy - 15) as f32 / 14.0
                };
                let p = Vec3::new(0.6 + 1.0 * ix as f32 / 29.0, y, 0.0);
                let v = f(p);
                let d = zs
                    .iter()
                    .map(|z| (*z - p).length())
                    .fold(f32::INFINITY, f32::min);
                assert!(
                    v.abs() <= d + gap,
                    "p {p}: value {v}, distance to the zero set {d}"
                );
                if v.abs() > 0.5 * d && d > 0.05 {
                    tight += 1;
                }
                n += 1;
            }
        }
        assert!(n == 900 && tight > 50, "compared {n}, tight {tight}");
    }
}

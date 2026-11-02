//! Ramp blend: a partition of unity between two fields across a slab, confined to the band where both
//! fields are near zero and to a box.
//!
//! # The field
//!
//! With `m = min(f_a, f_b)`, `s = n·p − d` (`n` a unit normal) and `t = smoothstep(−w/2, w/2, s)`, the
//! blend is `L = f_a + (f_b − f_a)·t`. It applies only where both fields are near zero and inside the
//! box: `B = B_q·G`, with `q = √(f_a² + f_b²)`, `B_q = 1 − smoothstep(r/2, r, q)` and `G` the product of
//! one smoothstep per box face, rising from 0 at the face to 1 at `e` inside it. The result is
//! `(B·L + (1 − B)·m) / Λ`.
//!
//! - Where both fields describe the same surface (`f_a = f_b`), `L = m`: the zero set is that surface,
//!   with nothing added (a smooth union would move it outward by up to `k/4`).
//! - Inside the band (`B = 1`) the field is `L / Λ`, continuously differentiable when both fields are;
//!   so is the zero set there. Near one field's own surface far from the other's (`q ≥ r`), and
//!   outside the box, the field is `min(f_a, f_b) / Λ`: the union, unchanged up to the scale.
//!
//! # The bound
//!
//! `∇(B·L + (1 − B)·m) = B·∇L + (1 − B)·∇m + (L − m)·∇B`, and
//! `∇L = (1 − t)·∇f_a + t·∇f_b + (f_b − f_a)·∇t`. With `L_a`, `L_b` the fields' Lipschitz bounds and
//! `D ≥ |f_a − f_b|` wherever `B > 0` (so also `|L − m| ≤ D` there):
//!
//! `Λ = max(L_a, L_b, 1) + D·(1.5/w + (3/r)·√(L_a² + L_b²) + 1.5·√3/e)`
//!
//! (`1.5/w` the steepest slope of `t`, `3/r` that of `B_q` times `|∇q| ≤ √(L_a² + L_b²)`, and
//! `1.5·√3/e` the steepest slope of `G`, at a corner where three faces taper together). The field is
//! divided by `Λ`, so it never grows faster than 1 per unit length and sphere tracing stays safe
//! without special cases.
//!
//! `D` is measured at construction, rigorously: on a grid of spacing `h` over the box,
//! `|f_a − f_b|` is within `(L_a + L_b)·h·√3/2` of its value at the nearest grid point, and every point
//! with `B > 0` (so `q < r`) has a grid point within `h·√3/2` whose `q < r + √(L_a² + L_b²)·h·√3/2`. So
//! `D = max |f_a − f_b|` over those grid points `+ (L_a + L_b)·h·√3/2` bounds it everywhere `B > 0`.
//! When no grid point qualifies, `D = 0` (the band does not meet the box). `D ≤ 2r` always holds
//! (both `|f| < r` where `B > 0`) and caps the measured value.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::LawMinMax;
use alice_det_math::metric::{lerp, smoothstep};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// The parameters of a ramp blend and the bound computed from them
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RampBlendParams {
    /// Unit normal of the ramp's plane
    pub normal: Vec3,
    /// Plane offset: `s = normal·p − offset`
    pub offset: f32,
    /// Ramp width `w` (the blend runs over `|s| < w/2`)
    pub width: f32,
    /// Band `r`: the blend applies where `√(f_a² + f_b²) < r`
    pub band: f32,
    /// Box corner (low)
    pub lo: Vec3,
    /// Box corner (high)
    pub hi: Vec3,
    /// Taper `e` inside each box face
    pub margin: f32,
    /// `D`, the bound of `|f_a − f_b|` where the blend applies
    pub diff_bound: f32,
    /// `Λ`, the bound the field is divided by
    pub lipschitz: f32,
}

impl RampBlendParams {
    /// The weights `(t, B)` at `p` for field values `fa`, `fb`
    #[inline]
    #[must_use]
    pub fn weights(&self, p: Vec3, fa: f32, fb: f32) -> (f32, f32) {
        let half = 0.5 * self.width;
        let t = smoothstep(-half, half, self.normal.dot(p) - self.offset);
        let q = (fa * fa + fb * fb).sqrt();
        let bq = 1.0 - smoothstep(0.5 * self.band, self.band, q);
        let e = self.margin;
        let mut g = 1.0_f32;
        for (x, (l, h)) in [
            (p.x, (self.lo.x, self.hi.x)),
            (p.y, (self.lo.y, self.hi.y)),
            (p.z, (self.lo.z, self.hi.z)),
        ] {
            // the high face as 1 − smoothstep (det-math's smoothstep with edge1 ≤ edge0 is a hard step)
            g *= smoothstep(l, l + e, x) * (1.0 - smoothstep(h - e, h, x));
        }
        (t, bq * g)
    }

    /// The blended value at `p` from the two fields' values (divided by the bound)
    #[inline]
    #[must_use]
    pub fn combine(&self, p: Vec3, fa: f32, fb: f32) -> f32 {
        let (t, b) = self.weights(p, fa, fb);
        let l = lerp(fa, fb, t);
        let m = fa.law_min(fb);
        lerp(m, l, b) / self.lipschitz
    }

    /// The bound `Λ` for children bounds `la`, `lb` and a difference bound `d`
    #[must_use]
    pub fn bound(width: f32, band: f32, margin: f32, la: f32, lb: f32, d: f32) -> f32 {
        let base = la.max(lb).max(1.0);
        base + d
            * (1.5 / width
                + (3.0 / band) * (la * la + lb * lb).sqrt()
                + 1.5 * 3.0_f32.sqrt() / margin)
    }

    /// Checks the parameters and measures `D` and `Λ` from the two fields (`fa`, `fb` evaluate them,
    /// `la`, `lb` their Lipschitz bounds) on a grid of spacing `h` over the box
    ///
    /// # Errors
    /// A normal that is not unit length (within 1e-4), a non-positive width, band, margin or grid
    /// spacing, a margin wider than half the box, a box with `lo ≥ hi` on an axis, a non-finite
    /// value, or a non-positive child bound
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        normal: Vec3,
        offset: f32,
        width: f32,
        band: f32,
        lo: Vec3,
        hi: Vec3,
        margin: f32,
        h: f32,
        la: f32,
        lb: f32,
        fa: impl Fn(Vec3) -> f32,
        fb: impl Fn(Vec3) -> f32,
    ) -> Result<Self, String> {
        let finite = [offset, width, band, margin, h, la, lb]
            .iter()
            .all(|x| x.is_finite())
            && normal.is_finite()
            && lo.is_finite()
            && hi.is_finite();
        if !finite {
            return Err("ramp blend: a non-finite parameter".into());
        }
        if (normal.length() - 1.0).abs() > 1e-4 {
            return Err(format!(
                "ramp blend: the normal must be unit length (got {})",
                normal.length()
            ));
        }
        if width <= 0.0 || band <= 0.0 || margin <= 0.0 || h <= 0.0 || la <= 0.0 || lb <= 0.0 {
            return Err("ramp blend: width, band, margin, grid spacing and the children bounds must be positive".into());
        }
        let size = hi - lo;
        if size.min_element() <= 0.0 {
            return Err("ramp blend: the box must have lo < hi on every axis".into());
        }
        if 2.0 * margin > size.min_element() {
            return Err(
                "ramp blend: the margin must be at most half the box's smallest side".into(),
            );
        }
        let n = (size / h).ceil().as_uvec3() + glam::UVec3::ONE;
        if (n.x as u64) * (n.y as u64) * (n.z as u64) > 50_000_000 {
            return Err(
                "ramp blend: the grid over the box has more than 5e7 points (raise h)".into(),
            );
        }
        let reach = 0.5 * 3.0_f32.sqrt() * h;
        let lq = (la * la + lb * lb).sqrt();
        let mut dmax = f32::NEG_INFINITY;
        for i in 0..n.x {
            for j in 0..n.y {
                for k in 0..n.z {
                    let p = (lo + Vec3::new(i as f32, j as f32, k as f32) * h).min(hi);
                    let (a, b) = (fa(p), fb(p));
                    if (a * a + b * b).sqrt() < band + lq * reach {
                        dmax = dmax.max((a - b).abs());
                    }
                }
            }
        }
        let d = if dmax == f32::NEG_INFINITY {
            0.0
        } else {
            (dmax + (la + lb) * reach).min(2.0 * band)
        };
        Ok(Self {
            normal,
            offset,
            width,
            band,
            lo,
            hi,
            margin,
            diff_bound: d,
            lipschitz: Self::bound(width, band, margin, la, lb, d),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(c: Vec3, r: f32) -> impl Fn(Vec3) -> f32 {
        move |p: Vec3| (p - c).length() - r
    }

    fn params(fa: impl Fn(Vec3) -> f32, fb: impl Fn(Vec3) -> f32) -> RampBlendParams {
        RampBlendParams::new(
            Vec3::Y,
            0.0,
            0.4,
            0.3,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            0.3,
            0.05,
            1.0,
            1.0,
            fa,
            fb,
        )
        .unwrap()
    }

    // tiny deterministic points
    fn pts(n: usize, scale: f32) -> Vec<Vec3> {
        let mut s = 0x2545_f491_u32;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        (0..n)
            .map(|_| Vec3::new(next(), next(), next()) * scale)
            .collect()
    }

    // oracle (closed form): outside the box the field is min/Λ; inside the band past the slab it is
    // exactly one side's field /Λ
    #[test]
    fn endpoints_are_exact() {
        let (fa, fb) = (
            sphere(Vec3::new(0.0, -0.01, 0.0), 1.0),
            sphere(Vec3::new(0.0, 0.01, 0.0), 1.004),
        );
        let pr = params(&fa, &fb);
        let out = Vec3::new(3.0, 0.5, 0.0);
        assert_eq!(
            pr.combine(out, fa(out), fb(out)),
            fa(out).min(fb(out)) / pr.lipschitz
        );
        // well inside the box, near the shared surface, below and above the slab (|s| > w/2 = 0.2)
        for (p, want) in [
            (Vec3::new(0.0, -1.0, 0.3).normalize() * 1.0, None),
            (Vec3::new(0.0, 1.0, 0.3).normalize() * 1.0, Some(())),
        ] {
            let (a, b) = (fa(p), fb(p));
            assert!(
                (a * a + b * b).sqrt() < 0.15,
                "test point not in the full band"
            );
            let want = if want.is_none() { a } else { b };
            // liveness: deep inside the box and the band the blend is fully on (B = 1)
            assert_eq!(pr.weights(p, a, b).1, 1.0, "the blend is not on at {p}");
            assert_eq!(pr.combine(p, a, b), want / pr.lipschitz, "{p}");
        }
    }

    // oracle: two fields describing the same surface give exactly that surface (no inflation); a
    // smooth union of the same pair moves it out by k/4
    #[test]
    fn no_inflation_on_a_shared_surface() {
        let f = sphere(Vec3::ZERO, 1.0);
        let pr = params(&f, &f);
        assert_eq!(pr.diff_bound, (1.0 + 1.0) * 0.5 * 3.0_f32.sqrt() * 0.05);
        for d in pts(500, 1.0) {
            let dir = d.normalize_or_zero();
            if dir == Vec3::ZERO {
                continue;
            }
            let p = dir; // on the unit sphere
            assert!(pr.combine(p, f(p), f(p)).abs() < 1e-6, "{p}");
        }
    }

    // the measured |f_a − f_b| over the band never exceeds D (a finer grid than construction)
    #[test]
    fn diff_bound_holds_on_a_finer_grid() {
        let (fa, fb) = (
            sphere(Vec3::new(0.0, 0.0, 0.0), 1.0),
            sphere(Vec3::new(0.07, 0.0, 0.03), 1.05),
        );
        let pr = params(&fa, &fb);
        let mut worst = 0.0_f32;
        for p in pts(200_000, 1.6) {
            let (a, b) = (fa(p), fb(p));
            if (a * a + b * b).sqrt() < pr.band {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(
            worst > 0.0 && worst <= pr.diff_bound,
            "measured {worst} vs bound {}",
            pr.diff_bound
        );
    }

    // the field divided by Λ grows at most 1 per unit length, on random close pairs over the
    // slab, the band switch and the box taper (the pair differs enough that every term matters)
    // pinned from the measured run (Λ/raw = 3.95, ceiled to 2 significant digits)
    const RATCHET: f32 = 4.0;

    #[test]
    fn lipschitz_at_most_one() {
        let (fa, fb) = (
            sphere(Vec3::new(0.0, -0.05, 0.0), 1.0),
            sphere(Vec3::new(0.1, 0.05, 0.0), 1.1),
        );
        let pr = RampBlendParams::new(
            Vec3::Y,
            0.0,
            0.2,
            0.25,
            Vec3::new(-1.5, -0.6, -1.5),
            Vec3::new(1.5, 0.6, 1.5),
            0.15,
            0.02,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        let mut worst = 0.0_f32;
        for (i, p) in pts(400_000, 1.4).into_iter().enumerate() {
            let d = Vec3::new(
                ((i * 7) % 11) as f32 - 5.0,
                ((i * 3) % 7) as f32 - 3.0,
                ((i * 5) % 13) as f32 - 6.0,
            )
            .normalize_or_zero()
                * 1e-3;
            let q = p + d;
            worst = worst.max((f(q) - f(p)).abs() / d.length().max(1e-9));
        }
        assert!(
            worst <= 1.0 + 1e-3,
            "measured slope {worst} (bound divided out: Λ = {})",
            pr.lipschitz
        );
        let raw = worst * pr.lipschitz;
        println!(
            "lipschitz_at_most_one: Λ = {}, D = {}, measured raw slope {raw}, Λ/raw = {}",
            pr.lipschitz,
            pr.diff_bound,
            pr.lipschitz / raw
        );
        // tightness ratchet (pinned from the run): the bound may not be more conservative than this
        assert!(
            pr.lipschitz <= RATCHET * raw,
            "Λ {} exceeds {RATCHET}·measured {raw}",
            pr.lipschitz
        );
        // the min region grows at exactly 1/Λ: the pairs must have seen at least that
        assert!(
            worst >= 0.99 / pr.lipschitz,
            "the test never exercised the field ({worst}, Λ {})",
            pr.lipschitz
        );
    }

    // continuously differentiable across the slab edge and the band switch: the one-sided
    // differences agree as the step shrinks
    #[test]
    fn c1_across_slab_and_switch() {
        let (fa, fb) = (
            sphere(Vec3::new(0.0, -0.05, 0.0), 1.0),
            sphere(Vec3::new(0.0, 0.05, 0.0), 1.04),
        );
        let pr = params(&fa, &fb);
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        // along y across the slab edge s = w/2 at the surface (x ≈ 0.98), and along x across the band edge
        for (p0, dir) in [
            (Vec3::new(0.98, 0.2, 0.0), Vec3::Y),
            (Vec3::new(1.0 + 0.2, 0.0, 0.0), Vec3::X),
            (Vec3::new(1.0 + 0.15, 0.0, 0.0), Vec3::X),
        ] {
            for hs in [1e-3_f32, 2e-4] {
                let left = (f(p0) - f(p0 - dir * hs)) / hs;
                let right = (f(p0 + dir * hs) - f(p0)) / hs;
                assert!(
                    (left - right).abs() < 50.0 * hs + 2e-3,
                    "{p0} {dir}: {left} vs {right} at h {hs}"
                );
            }
        }
    }

    #[test]
    fn refuses_bad_input() {
        let f = sphere(Vec3::ZERO, 1.0);
        let ok = |n: Vec3, w: f32, lo: Vec3, hi: Vec3, e: f32| {
            RampBlendParams::new(n, 0.0, w, 0.2, lo, hi, e, 0.1, 1.0, 1.0, &f, &f).is_ok()
        };
        assert!(ok(Vec3::Y, 0.2, Vec3::splat(-1.0), Vec3::ONE, 0.1));
        assert!(!ok(
            Vec3::new(0.0, 2.0, 0.0),
            0.2,
            Vec3::splat(-1.0),
            Vec3::ONE,
            0.1
        ));
        assert!(!ok(Vec3::Y, 0.0, Vec3::splat(-1.0), Vec3::ONE, 0.1));
        assert!(!ok(Vec3::Y, 0.2, Vec3::ONE, Vec3::splat(-1.0), 0.1));
        assert!(!ok(Vec3::Y, 0.2, Vec3::splat(-1.0), Vec3::ONE, 1.5));
    }

    // oracle: near one field's own surface, far from the other's, inside the box and past the slab,
    // the band gate keeps the union: the field is min/Λ (here f_a/Λ), not the ramp's side (f_b)
    #[test]
    fn band_gate_keeps_the_union_near_one_surface() {
        let (fa, fb) = (
            sphere(Vec3::ZERO, 1.0),
            sphere(Vec3::new(2.5, 0.0, 0.0), 0.5),
        );
        let pr = RampBlendParams::new(
            Vec3::Y,
            0.0,
            0.4,
            0.3,
            Vec3::splat(-3.5),
            Vec3::splat(3.5),
            0.3,
            0.1,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        for p in [
            Vec3::new(0.0, 0.98, 0.0),
            Vec3::new(0.3, 0.95, 0.0),
            Vec3::new(0.0, 1.02, 0.1),
        ] {
            let (a, b) = (fa(p), fb(p));
            assert!(a.abs() < 0.05 && b > 1.0, "test point setup {p}: {a} {b}");
            assert!(p.y > 0.2, "past the slab");
            assert_eq!(pr.combine(p, a, b), a / pr.lipschitz, "{p}");
        }
    }

    // the band term of the bound is needed: two concentric spheres 0.02 apart, a narrow band and a
    // wide ramp (t ≈ 1/2 near s = 0, so L − m ≈ (f_b − f_a)/2), probed radially across the band switch
    #[test]
    fn lipschitz_across_the_band_switch() {
        let (fa, fb) = (sphere(Vec3::ZERO, 1.0), sphere(Vec3::ZERO, 1.02));
        let pr = RampBlendParams::new(
            Vec3::Y,
            0.0,
            4.0,
            0.03,
            Vec3::splat(-1.5),
            Vec3::splat(1.5),
            0.2,
            0.01,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        let mut worst = 0.0_f32;
        for k in 0..64 {
            let a = std::f32::consts::TAU * k as f32 / 64.0;
            let dir = Vec3::new(a.cos(), 0.0, a.sin());
            let mut r = 0.95_f32;
            while r < 1.08 {
                let (p, q) = (dir * r, dir * (r + 1e-4));
                worst = worst.max((f(q) - f(p)).abs() / 1e-4);
                r += 1e-4;
            }
        }
        assert!(
            worst <= 1.0 + 1e-3,
            "measured slope {worst} (Λ = {})",
            pr.lipschitz
        );
    }

    // continuously differentiable at the slab edge, against the size of the kink a non-smooth ramp
    // would leave there: one-sided derivatives along the normal agree to a tenth of (f_b − f_a)/(w Λ)
    #[test]
    fn ramp_has_no_kink_at_the_slab_edge() {
        let (fa, fb) = (sphere(Vec3::ZERO, 1.0), sphere(Vec3::ZERO, 1.04));
        let pr = RampBlendParams::new(
            Vec3::Y,
            0.0,
            0.4,
            0.3,
            Vec3::splat(-1.5),
            Vec3::splat(1.5),
            0.3,
            0.02,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        for x in [0.97_f32, 0.99, 1.01] {
            let p0 = Vec3::new(x, 0.2, 0.0); // s = w/2
            let kink = (fb(p0) - fa(p0)).abs() / (0.4 * pr.lipschitz);
            let h = 1e-4;
            let left = (f(p0) - f(p0 - Vec3::Y * h)) / h;
            let right = (f(p0 + Vec3::Y * h) - f(p0)) / h;
            assert!(
                (left - right).abs() < 0.1 * kink,
                "x {x}: {left} vs {right}, a linear ramp's kink {kink}"
            );
        }
    }

    // the taper term of the bound is needed: a narrow taper (e = 0.02) with a wide band, and the shared
    // surface crossing the taper side-on (the spheres' tops, normal ±y, at x ≈ 1.49 inside the taper
    // 1.48 < x < 1.5; the ramp along z so t = 1/2 there), probed along x, where the fields barely change
    // and the taper's slope times (L − m) is the whole gradient
    #[test]
    fn lipschitz_across_the_box_taper() {
        let c = Vec3::new(1.49, 0.0, 0.0);
        let (fa, fb) = (sphere(c, 1.0), sphere(c, 1.05));
        let pr = RampBlendParams::new(
            Vec3::Z,
            0.0,
            4.0,
            1.0,
            Vec3::splat(-1.5),
            Vec3::splat(1.5),
            0.02,
            0.01,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        let mut worst = 0.0_f32;
        for k in 0..25 {
            let y = 0.95 + 0.005 * k as f32;
            let mut x = 1.46_f32;
            while x < 1.5 {
                let (p, q) = (Vec3::new(x, y, 0.0), Vec3::new(x + 1e-5, y, 0.0));
                worst = worst.max((f(q) - f(p)).abs() / 1e-5);
                x += 1e-5;
            }
        }
        println!(
            "taper: worst {worst}, Λ {}, D {}",
            pr.lipschitz, pr.diff_bound
        );
        assert!(
            worst <= 1.0 + 1e-3,
            "measured slope {worst} (Λ = {})",
            pr.lipschitz
        );
        // liveness floor from the run (0.19 measured): a dead blend or a dead taper gives about 0
        assert!(
            worst >= 0.15,
            "the taper probe never saw the blend ({worst})"
        );
    }

    // the realistic case: two fits of one surface (D small), where Λ must stay close to 1 and the
    // ratchet is tight (pinned from the run)
    #[test]
    fn lipschitz_small_difference_is_tight() {
        let (fa, fb) = (
            sphere(Vec3::new(0.0, 0.0, 0.0), 1.0),
            sphere(Vec3::new(0.0005, 0.0, 0.0), 1.001),
        );
        let pr = RampBlendParams::new(
            Vec3::Y,
            0.0,
            0.4,
            0.2,
            Vec3::splat(-1.1),
            Vec3::splat(1.1),
            0.3,
            0.008,
            1.0,
            1.0,
            &fa,
            &fb,
        )
        .unwrap();
        let f = |p: Vec3| pr.combine(p, fa(p), fb(p));
        let mut worst = 0.0_f32;
        for (i, p) in pts(300_000, 1.2).into_iter().enumerate() {
            let p = p.normalize_or_zero() * (0.97 + 0.06 * ((i % 61) as f32 / 60.0));
            let d = Vec3::new(
                ((i * 7) % 11) as f32 - 5.0,
                ((i * 3) % 7) as f32 - 3.0,
                ((i * 5) % 13) as f32 - 6.0,
            )
            .normalize_or_zero()
                * 1e-4;
            worst = worst.max((f(p + d) - f(p)).abs() / d.length().max(1e-12));
        }
        let raw = worst * pr.lipschitz;
        println!("lipschitz_small_difference_is_tight: Λ = {}, D = {}, measured raw slope {raw}, Λ/raw = {}", pr.lipschitz, pr.diff_bound, pr.lipschitz / raw);
        assert!(worst <= 1.0 + 1e-3, "measured slope {worst}");
        assert!(
            pr.lipschitz < 3.0,
            "Λ {} for a small difference",
            pr.lipschitz
        );
        assert!(
            pr.lipschitz <= RATCHET_SMALL * raw,
            "Λ {} exceeds {RATCHET_SMALL}·measured {raw}",
            pr.lipschitz
        );
    }
    // pinned from the measured run (Λ/raw = 1.515, ceiled to 2 significant digits)
    const RATCHET_SMALL: f32 = 1.6;

    // the tree node: eval equals the parameters' law at every point, a JSON round trip keeps it bit for
    // bit, and the bytecode compiler refuses it (tree eval only until its shader entries exist)
    #[test]
    fn node_eval_roundtrip_and_compile_refusal() {
        use crate::types::SdfNode;
        let a = SdfNode::sphere(1.0).translate(0.0, -0.01, 0.0);
        let b = SdfNode::sphere(1.004).translate(0.0, 0.01, 0.0);
        let node = a
            .clone()
            .ramp_blend(
                b.clone(),
                Vec3::Y,
                0.0,
                0.4,
                0.3,
                Vec3::splat(-2.0),
                Vec3::splat(2.0),
                0.3,
                0.05,
            )
            .unwrap();
        let SdfNode::RampBlend { params, .. } = &node else {
            panic!("not a ramp blend")
        };
        for p in pts(2000, 1.5) {
            let want = params.combine(p, crate::eval::eval(&a, p), crate::eval::eval(&b, p));
            assert_eq!(crate::eval::eval(&node, p).to_bits(), want.to_bits(), "{p}");
        }
        let json = serde_json::to_string(&node).unwrap();
        let back: SdfNode = serde_json::from_str(&json).unwrap();
        for p in pts(500, 1.5) {
            assert_eq!(
                crate::eval::eval(&back, p).to_bits(),
                crate::eval::eval(&node, p).to_bits()
            );
        }
        assert!(crate::compiled::CompiledSdf::try_compile(&node).is_err());
    }
}

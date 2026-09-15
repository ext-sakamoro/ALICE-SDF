//! `Real` — the scalar abstraction that lets one evaluator serve `f32` and
//! `wide::f32x8`.
//!
//! Every law in the compiled evaluator (transforms, modifiers, post-processing,
//! CSG blends) is written once against this trait and instantiated twice.
//! Branchy scalar code becomes branch-free `select`; data-dependent loops
//! (noise, heightmaps, lattices, polygons, …) escape through [`Real::map`] /
//! [`Vec3R::map`], which run the scalar law per lane.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec2, Vec3};
use std::ops::{Add, Div, Mul, Neg, Sub};
use wide::{f32x8, CmpGe, CmpGt, CmpLe, CmpLt};

/// Scalar (or SIMD lane bundle) with the operations the evaluator needs.
pub trait Real:
    Copy
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Comparison result (`bool` for `f32`, lane mask for `f32x8`).
    type Mask: Copy;
    /// Number of lanes (1 for `f32`, 8 for `f32x8`).
    const LANES: usize;

    /// Broadcast a scalar.
    fn splat(v: f32) -> Self;
    /// All-zero value.
    fn zero() -> Self {
        Self::splat(0.0)
    }
    /// All-one value.
    fn one() -> Self {
        Self::splat(1.0)
    }

    /// Square root.
    fn sqrt(self) -> Self;
    /// Absolute value.
    fn abs(self) -> Self;
    /// Round toward negative infinity.
    fn floor(self) -> Self;
    /// Round to nearest (ties away from zero for `f32`; the `f32x8` impl
    /// follows the hardware and ties to even on AVX / NEON).
    ///
    /// Not path-safe at exact `.5` inputs — laws must use
    /// [`Real::round_half_up`] instead.
    fn round(self) -> Self;
    /// Lane-wise minimum.
    fn min(self, other: Self) -> Self;
    /// Lane-wise maximum.
    fn max(self, other: Self) -> Self;
    /// Simultaneous sine and cosine.
    fn sin_cos(self) -> (Self, Self);
    /// Four-quadrant arctangent `atan2(self, x)`.
    fn atan2(self, x: Self) -> Self;
    /// Natural exponential.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Sign: `-1` for negative lanes, `+1` otherwise (`±0.0` and NaN give `+1`).
    ///
    /// This is the branchless `x < 0 ? -1 : 1` that the SIMD instantiation,
    /// both JITs and the shader transpilers use — *not* `f32::signum`, whose
    /// `-0.0 → -1` made the tree / compiled scalar path disagree with SIMD by
    /// a sign flip at the pyramid base centre (found by `fuzz_eval_parity`).
    fn signum(self) -> Self;

    /// `self < other`
    fn lt(self, other: Self) -> Self::Mask;
    /// `self > other`
    fn gt(self, other: Self) -> Self::Mask;
    /// `self <= other`
    fn le(self, other: Self) -> Self::Mask;
    /// `self >= other`
    fn ge(self, other: Self) -> Self::Mask;
    /// `mask ? a : b`
    fn select(mask: Self::Mask, a: Self, b: Self) -> Self;
    /// Logical AND of two masks.
    fn mask_and(a: Self::Mask, b: Self::Mask) -> Self::Mask;
    /// Logical OR of two masks.
    fn mask_or(a: Self::Mask, b: Self::Mask) -> Self::Mask;

    /// Apply a scalar function to every lane (per-lane escape hatch).
    fn map(self, f: impl Fn(f32) -> f32) -> Self;
    /// Apply a scalar binary function to every lane pair.
    fn map2(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self;
    /// Apply `f` per lane to `(x, y, z)` returning one scalar per lane.
    fn map3(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> f32) -> Self;
    /// Apply `f` per lane to `(x, y, z)` returning a vector per lane.
    fn map3v(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> Vec3) -> Vec3R<Self>;
    /// Apply `f` per lane to `(x, y, z)` returning a vector and a scalar per lane.
    fn map3vs(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> (Vec3, f32)) -> (Vec3R<Self>, Self);
    /// Apply `f` per lane to a distance and a point, returning a distance per lane.
    fn map_dp(d: Self, p: Vec3R<Self>, f: impl Fn(f32, Vec3) -> f32) -> Self;

    /// `self.max(lo).min(hi)`
    #[inline(always)]
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// `self * m + a`
    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        self * m + a
    }
    /// Round to nearest with ties toward `+∞`: `floor(self + 0.5)`.
    ///
    /// The canonical rounding rule of the repeat / polar / helix laws, shared
    /// with the tree evaluator ([`crate::crispy::round_half_up`]), the JIT and
    /// the three shader transpilers so that every path agrees at cell
    /// boundaries. See [`crate::crispy::round_half_up`] for why `round` is
    /// not usable here.
    #[inline(always)]
    fn round_half_up(self) -> Self {
        (self + Self::splat(0.5)).floor()
    }
    /// Linear interpolation `a + (b - a) * t`
    #[inline(always)]
    fn lerp(a: Self, b: Self, t: Self) -> Self {
        a + (b - a) * t
    }

    /// Twist around Y (hookable so `f32` can use the fused canonical law).
    #[inline(always)]
    fn twist(p: Vec3R<Self>, strength: f32) -> Vec3R<Self> {
        let (s, c) = (p.y * Self::splat(strength)).sin_cos();
        Vec3R::new(p.x * c - p.z * s, p.y, p.x * s + p.z * c)
    }

    /// Rotate `p` by the inverse of unit quaternion `q`.
    ///
    /// One formula for every instantiation (two cross products). `f32` used to
    /// go through `glam` here, but `glam`'s quaternion multiply rounds
    /// differently by an ulp, and laws with a sign discontinuity (pyramid base)
    /// flip on that ulp — the tree evaluator calls this too so all Rust paths
    /// rotate bit-identically (found by `fuzz_eval_parity`).
    #[inline(always)]
    fn rotate_inverse(q: Quat, p: Vec3R<Self>) -> Vec3R<Self> {
        // inverse of unit quaternion = conjugate: (-v, w)
        let qv = Vec3R::<Self>::splat(-Vec3::new(q.x, q.y, q.z));
        let w = Self::splat(q.w);
        // p' = p + 2w (qv × p) + 2 qv × (qv × p)
        let t = qv.cross(p) * Self::splat(2.0);
        p + t * w + qv.cross(t)
    }
}

impl Real for f32 {
    type Mask = bool;
    const LANES: usize = 1;

    #[inline(always)]
    fn splat(v: f32) -> Self {
        v
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32::floor(self)
    }
    #[inline(always)]
    fn round(self) -> Self {
        f32::round(self)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        f32::sin_cos(self)
    }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        f32::atan2(self, x)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[inline(always)]
    fn signum(self) -> Self {
        if self < 0.0 {
            -1.0
        } else {
            1.0
        }
    }
    #[inline(always)]
    fn lt(self, other: Self) -> bool {
        self < other
    }
    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }
    #[inline(always)]
    fn le(self, other: Self) -> bool {
        self <= other
    }
    #[inline(always)]
    fn ge(self, other: Self) -> bool {
        self >= other
    }
    #[inline(always)]
    fn select(mask: bool, a: Self, b: Self) -> Self {
        if mask {
            a
        } else {
            b
        }
    }
    #[inline(always)]
    fn mask_and(a: bool, b: bool) -> bool {
        a && b
    }
    #[inline(always)]
    fn mask_or(a: bool, b: bool) -> bool {
        a || b
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        f(self)
    }
    #[inline(always)]
    fn map2(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        f(self, other)
    }
    #[inline(always)]
    fn map3(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> f32) -> Self {
        f(Vec3::new(x, y, z))
    }
    #[inline(always)]
    fn map3v(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> Vec3) -> Vec3R<Self> {
        let v = f(Vec3::new(x, y, z));
        Vec3R::new(v.x, v.y, v.z)
    }
    #[inline(always)]
    fn map3vs(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> (Vec3, f32)) -> (Vec3R<Self>, Self) {
        let (v, s) = f(Vec3::new(x, y, z));
        (Vec3R::new(v.x, v.y, v.z), s)
    }
    #[inline(always)]
    fn map_dp(d: Self, p: Vec3R<Self>, f: impl Fn(f32, Vec3) -> f32) -> Self {
        f(d, Vec3::new(p.x, p.y, p.z))
    }
    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        f32::mul_add(self, m, a)
    }
    #[inline(always)]
    fn twist(p: Vec3R<Self>, strength: f32) -> Vec3R<Self> {
        crate::modifiers::modifier_twist(Vec3::from(p), strength).into()
    }
}

impl Real for f32x8 {
    type Mask = f32x8;
    const LANES: usize = 8;

    #[inline(always)]
    fn splat(v: f32) -> Self {
        f32x8::splat(v)
    }
    #[inline(always)]
    fn sqrt(self) -> Self {
        f32x8::sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> Self {
        f32x8::abs(self)
    }
    #[inline(always)]
    fn floor(self) -> Self {
        f32x8::floor(self)
    }
    #[inline(always)]
    fn round(self) -> Self {
        f32x8::round(self)
    }
    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32x8::min(self, other)
    }
    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32x8::max(self, other)
    }
    #[inline(always)]
    fn sin_cos(self) -> (Self, Self) {
        f32x8::sin_cos(self)
    }
    #[inline(always)]
    fn atan2(self, x: Self) -> Self {
        // Per-lane libm, not `f32x8::atan2`: the polar-repeat / helix laws snap
        // the angle to a sector, and the SIMD polynomial is a few ulp off libm,
        // which is enough to cross an exact sector boundary (atan2(0, -x) = π
        // with an odd sector count lands on k + 0.5) and disagree with the
        // scalar path by a whole sector (found by `fuzz_eval_parity`).
        self.map2(x, f32::atan2)
    }
    #[inline(always)]
    fn exp(self) -> Self {
        f32x8::exp(self)
    }
    #[inline(always)]
    fn ln(self) -> Self {
        f32x8::ln(self)
    }
    #[inline(always)]
    fn signum(self) -> Self {
        self.cmp_lt(f32x8::ZERO)
            .blend(f32x8::splat(-1.0), f32x8::ONE)
    }
    #[inline(always)]
    fn lt(self, other: Self) -> f32x8 {
        self.cmp_lt(other)
    }
    #[inline(always)]
    fn gt(self, other: Self) -> f32x8 {
        self.cmp_gt(other)
    }
    #[inline(always)]
    fn le(self, other: Self) -> f32x8 {
        self.cmp_le(other)
    }
    #[inline(always)]
    fn ge(self, other: Self) -> f32x8 {
        self.cmp_ge(other)
    }
    #[inline(always)]
    fn select(mask: f32x8, a: Self, b: Self) -> Self {
        mask.blend(a, b)
    }
    #[inline(always)]
    fn mask_and(a: f32x8, b: f32x8) -> f32x8 {
        a & b
    }
    #[inline(always)]
    fn mask_or(a: f32x8, b: f32x8) -> f32x8 {
        a | b
    }
    #[inline(always)]
    fn map(self, f: impl Fn(f32) -> f32) -> Self {
        let a = self.as_array_ref();
        f32x8::new([
            f(a[0]),
            f(a[1]),
            f(a[2]),
            f(a[3]),
            f(a[4]),
            f(a[5]),
            f(a[6]),
            f(a[7]),
        ])
    }
    #[inline(always)]
    fn map2(self, other: Self, f: impl Fn(f32, f32) -> f32) -> Self {
        let a = self.as_array_ref();
        let b = other.as_array_ref();
        f32x8::new([
            f(a[0], b[0]),
            f(a[1], b[1]),
            f(a[2], b[2]),
            f(a[3], b[3]),
            f(a[4], b[4]),
            f(a[5], b[5]),
            f(a[6], b[6]),
            f(a[7], b[7]),
        ])
    }
    #[inline(always)]
    fn map3(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> f32) -> Self {
        let (xa, ya, za) = (x.as_array_ref(), y.as_array_ref(), z.as_array_ref());
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = f(Vec3::new(xa[i], ya[i], za[i]));
        }
        f32x8::new(out)
    }
    #[inline(always)]
    fn map3v(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> Vec3) -> Vec3R<Self> {
        let (xa, ya, za) = (x.as_array_ref(), y.as_array_ref(), z.as_array_ref());
        let (mut ox, mut oy, mut oz) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        for i in 0..8 {
            let v = f(Vec3::new(xa[i], ya[i], za[i]));
            ox[i] = v.x;
            oy[i] = v.y;
            oz[i] = v.z;
        }
        Vec3R::new(f32x8::new(ox), f32x8::new(oy), f32x8::new(oz))
    }
    #[inline(always)]
    fn map3vs(x: Self, y: Self, z: Self, f: impl Fn(Vec3) -> (Vec3, f32)) -> (Vec3R<Self>, Self) {
        let (xa, ya, za) = (x.as_array_ref(), y.as_array_ref(), z.as_array_ref());
        let (mut ox, mut oy, mut oz, mut os) = ([0.0f32; 8], [0.0f32; 8], [0.0f32; 8], [0.0f32; 8]);
        for i in 0..8 {
            let (v, s) = f(Vec3::new(xa[i], ya[i], za[i]));
            ox[i] = v.x;
            oy[i] = v.y;
            oz[i] = v.z;
            os[i] = s;
        }
        (
            Vec3R::new(f32x8::new(ox), f32x8::new(oy), f32x8::new(oz)),
            f32x8::new(os),
        )
    }
    #[inline(always)]
    fn map_dp(d: Self, p: Vec3R<Self>, f: impl Fn(f32, Vec3) -> f32) -> Self {
        let (da, xa, ya, za) = (
            d.as_array_ref(),
            p.x.as_array_ref(),
            p.y.as_array_ref(),
            p.z.as_array_ref(),
        );
        let mut out = [0.0f32; 8];
        for i in 0..8 {
            out[i] = f(da[i], Vec3::new(xa[i], ya[i], za[i]));
        }
        f32x8::new(out)
    }
    #[inline(always)]
    fn mul_add(self, m: Self, a: Self) -> Self {
        f32x8::mul_add(self, m, a)
    }
}

/// 3-vector over a [`Real`].
#[derive(Clone, Copy, Debug)]
pub struct Vec3R<R: Real> {
    /// X component
    pub x: R,
    /// Y component
    pub y: R,
    /// Z component
    pub z: R,
}

impl<R: Real> Vec3R<R> {
    /// Construct from components.
    #[inline(always)]
    pub fn new(x: R, y: R, z: R) -> Self {
        Self { x, y, z }
    }
    /// Broadcast a `glam::Vec3`.
    #[inline(always)]
    pub fn splat(v: Vec3) -> Self {
        Self {
            x: R::splat(v.x),
            y: R::splat(v.y),
            z: R::splat(v.z),
        }
    }
    /// All-zero vector.
    #[inline(always)]
    pub fn zero() -> Self {
        Self::new(R::zero(), R::zero(), R::zero())
    }
    /// Dot product.
    #[inline(always)]
    pub fn dot(self, o: Self) -> R {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    /// Cross product.
    #[inline(always)]
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    /// Squared length.
    #[inline(always)]
    pub fn length_squared(self) -> R {
        self.dot(self)
    }
    /// Length.
    #[inline(always)]
    pub fn length(self) -> R {
        self.length_squared().sqrt()
    }
    /// Component-wise absolute value.
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs(), self.z.abs())
    }
    /// Component-wise minimum.
    #[inline(always)]
    pub fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    /// Component-wise maximum.
    #[inline(always)]
    pub fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    /// Component-wise clamp.
    #[inline(always)]
    pub fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
    /// Component-wise round (see [`Real::round`] for the tie caveat).
    #[inline(always)]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round(), self.z.round())
    }
    /// Component-wise [`Real::round_half_up`] (the law-canonical rounding).
    #[inline(always)]
    pub fn round_half_up(self) -> Self {
        Self::new(
            self.x.round_half_up(),
            self.y.round_half_up(),
            self.z.round_half_up(),
        )
    }
    /// Largest component.
    #[inline(always)]
    pub fn max_element(self) -> R {
        self.x.max(self.y).max(self.z)
    }
    /// Component-wise multiply.
    #[inline(always)]
    pub fn mul_vec(self, o: Self) -> Self {
        Self::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
    /// Apply a scalar `Vec3 -> Vec3` law to every lane (per-lane escape hatch).
    #[inline(always)]
    pub fn map(self, f: impl Fn(Vec3) -> Vec3) -> Self {
        R::map3v(self.x, self.y, self.z, f)
    }
    /// Apply a scalar `Vec3 -> f32` law to every lane.
    #[inline(always)]
    pub fn map_scalar(self, f: impl Fn(Vec3) -> f32) -> R {
        R::map3(self.x, self.y, self.z, f)
    }
}

impl<R: Real> Add for Vec3R<R> {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl<R: Real> Sub for Vec3R<R> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl<R: Real> Mul<R> for Vec3R<R> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, s: R) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}
impl<R: Real> Div<R> for Vec3R<R> {
    type Output = Self;
    #[inline(always)]
    fn div(self, s: R) -> Self {
        Self::new(self.x / s, self.y / s, self.z / s)
    }
}
impl<R: Real> Neg for Vec3R<R> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl From<Vec3> for Vec3R<f32> {
    #[inline(always)]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}
impl From<Vec3R<f32>> for Vec3 {
    #[inline(always)]
    fn from(v: Vec3R<f32>) -> Self {
        Vec3::new(v.x, v.y, v.z)
    }
}
impl From<super::simd::Vec3x8> for Vec3R<f32x8> {
    #[inline(always)]
    fn from(v: super::simd::Vec3x8) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}
impl From<Vec3R<f32x8>> for super::simd::Vec3x8 {
    #[inline(always)]
    fn from(v: Vec3R<f32x8>) -> Self {
        super::simd::Vec3x8 {
            x: v.x,
            y: v.y,
            z: v.z,
        }
    }
}

// ===========================================================================
// Generic laws shared by every instantiation of the evaluator
// ===========================================================================

/// Rotate `p` by the inverse of `q` (dispatches to [`Real::rotate_inverse`]).
#[inline(always)]
pub fn rotate_inverse<R: Real>(q: Quat, p: Vec3R<R>) -> Vec3R<R> {
    R::rotate_inverse(q, p)
}

/// Twist around Y: rotate XZ by `strength * y` (dispatches to [`Real::twist`]).
#[inline(always)]
pub fn twist<R: Real>(p: Vec3R<R>, strength: f32) -> Vec3R<R> {
    R::twist(p, strength)
}

/// Bend: rotate XY by `curvature * x`.
#[inline(always)]
pub fn bend<R: Real>(p: Vec3R<R>, curvature: f32) -> Vec3R<R> {
    let (s, c) = (p.x * R::splat(curvature)).sin_cos();
    Vec3R::new(p.x * c - p.y * s, p.x * s + p.y * c, p.z)
}

/// Infinite repetition with precomputed reciprocal spacing.
#[inline(always)]
pub fn repeat_infinite<R: Real>(p: Vec3R<R>, spacing: Vec3, recip: Vec3) -> Vec3R<R> {
    let cell = p.mul_vec(Vec3R::splat(recip)).round_half_up();
    p - cell.mul_vec(Vec3R::splat(spacing))
}

/// Finite repetition: `count` cells clamped to `±count/2`.
#[inline(always)]
pub fn repeat_finite<R: Real>(p: Vec3R<R>, count: Vec3, spacing: Vec3) -> Vec3R<R> {
    let limit = Vec3R::splat(count * 0.5);
    let inv = Vec3R::splat(Vec3::new(1.0 / spacing.x, 1.0 / spacing.y, 1.0 / spacing.z));
    let cell = p.mul_vec(inv).round_half_up().clamp(-limit, limit);
    p - cell.mul_vec(Vec3R::splat(spacing))
}

/// Elongate: `p - clamp(p, -amount, amount)`.
#[inline(always)]
pub fn elongate<R: Real>(p: Vec3R<R>, amount: Vec3) -> Vec3R<R> {
    let a = Vec3R::splat(amount);
    p - p.clamp(-a, a)
}

/// Mirror the axes whose mask component is non-zero.
#[inline(always)]
pub fn mirror<R: Real>(p: Vec3R<R>, axes: Vec3) -> Vec3R<R> {
    Vec3R::new(
        if axes.x != 0.0 { p.x.abs() } else { p.x },
        if axes.y != 0.0 { p.y.abs() } else { p.y },
        if axes.z != 0.0 { p.z.abs() } else { p.z },
    )
}

/// Octant mirror: `abs` then sort so `x >= y >= z`.
#[inline(always)]
pub fn octant_mirror<R: Real>(p: Vec3R<R>) -> Vec3R<R> {
    let (x, y, z) = (p.x.abs(), p.y.abs(), p.z.abs());
    // Three compare-swaps (same network as the scalar law: xy, yz, xy)
    let (x, y) = (x.max(y), x.min(y));
    let (y, z) = (y.max(z), y.min(z));
    let (x, y) = (x.max(y), x.min(y));
    Vec3R::new(x, y, z)
}

/// Revolution: `(length(xz) - offset, y, 0)`.
#[inline(always)]
pub fn revolution<R: Real>(p: Vec3R<R>, offset: f32) -> Vec3R<R> {
    let r = (p.x * p.x + p.z * p.z).sqrt() - R::splat(offset);
    Vec3R::new(r, p.y, R::zero())
}

/// Extrude (point part): evaluate the child in the XY plane.
#[inline(always)]
pub fn extrude_point<R: Real>(p: Vec3R<R>) -> Vec3R<R> {
    Vec3R::new(p.x, p.y, R::zero())
}

/// Extrude (distance part): combine the 2D child distance with the Z slab.
#[inline(always)]
pub fn extrude_distance<R: Real>(d: R, original_z: R, half_height: f32) -> R {
    let dz = original_z.abs() - R::splat(half_height);
    let wx = d.max(R::zero());
    let wy = dz.max(R::zero());
    d.max(dz).min(R::zero()) + (wx * wx + wy * wy).sqrt()
}

/// Taper: scale XZ by `1 / (1 - factor * y)` (same law as `modifier_taper`).
#[inline(always)]
pub fn taper<R: Real>(p: Vec3R<R>, factor: f32) -> Vec3R<R> {
    // `1 - f * y` reaches 0 on the plane y = 1 / f; keep the denominator away
    // from it (same sign, |den| ≥ 1e-6) so no path produces inf / NaN there
    // (the tree evaluator used to give NaN and the SIMD one a finite value,
    // found by `fuzz_eval_parity`). Shared by the tree evaluator, the compiled
    // scalar / SIMD paths and mirrored by the shader transpilers.
    let den = R::one() - p.y * R::splat(factor);
    let eps = R::splat(1e-6);
    let mag = den.abs().max(eps);
    let den = R::select(den.lt(R::zero()), -mag, mag);
    let s = R::one() / den;
    Vec3R::new(p.x * s, p.y, p.z * s)
}

/// Polar repeat around Y with precomputed sector / reciprocal.
#[inline(always)]
pub fn polar_repeat<R: Real>(p: Vec3R<R>, sector: f32, recip_sector: f32) -> Vec3R<R> {
    let a = p.z.atan2(p.x);
    let r = (p.x * p.x + p.z * p.z).sqrt();
    let sector_angle = a - (a * R::splat(recip_sector)).round_half_up() * R::splat(sector);
    let (s, c) = sector_angle.sin_cos();
    Vec3R::new(r * c, p.y, r * s)
}

/// Inverse shear: `(x, y - xy*x, z - xz*x - yz*y)`.
#[inline(always)]
pub fn shear<R: Real>(p: Vec3R<R>, sh: Vec3) -> Vec3R<R> {
    Vec3R::new(
        p.x,
        p.y - R::splat(sh.x) * p.x,
        p.z - R::splat(sh.y) * p.x - R::splat(sh.z) * p.y,
    )
}

/// Sweep along a quadratic Bézier in the XZ plane: `(perp distance, y, 0)`.
///
/// Coarse 5-sample search + 5 Newton steps (same law as `modifiers::sweep`).
#[inline]
pub fn sweep_bezier<R: Real>(p: Vec3R<R>, p0: Vec2, p1: Vec2, p2: Vec2) -> Vec3R<R> {
    let (p0x, p0z) = (R::splat(p0.x), R::splat(p0.y));
    let (p1x, p1z) = (R::splat(p1.x), R::splat(p1.y));
    let (p2x, p2z) = (R::splat(p2.x), R::splat(p2.y));
    let two = R::splat(2.0);
    let (qx, qz) = (p.x, p.z);

    let eval = |t: R| {
        let omt = R::one() - t;
        let (a, b, c) = (omt * omt, two * omt * t, t * t);
        (p0x * a + p1x * b + p2x * c, p0z * a + p1z * b + p2z * c)
    };

    let mut best_t = R::zero();
    let mut best_d2 = R::splat(f32::MAX);
    for i in 0..5u32 {
        let t = R::splat(i as f32 * 0.25);
        let (bx, bz) = eval(t);
        let (dx, dz) = (qx - bx, qz - bz);
        let d2 = dx * dx + dz * dz;
        let better = d2.lt(best_d2);
        best_d2 = R::select(better, d2, best_d2);
        best_t = R::select(better, t, best_t);
    }

    let bddx = two * (p0x - two * p1x + p2x);
    let bddz = two * (p0z - two * p1z + p2z);
    let eps = R::splat(1e-10);
    let mut t = best_t;
    for _ in 0..5 {
        let omt = R::one() - t;
        let (bx, bz) = eval(t);
        let tdx = (p1x - p0x) * (two * omt) + (p2x - p1x) * (two * t);
        let tdz = (p1z - p0z) * (two * omt) + (p2z - p1z) * (two * t);
        let (diffx, diffz) = (bx - qx, bz - qz);
        let num = diffx * tdx + diffz * tdz;
        let den = tdx * tdx + tdz * tdz + diffx * bddx + diffz * bddz;
        let tiny = den.abs().lt(eps);
        let stepped = (t - num / R::select(tiny, R::one(), den)).clamp(R::zero(), R::one());
        // scalar law breaks out of the loop on a tiny denominator: keep t unchanged
        t = R::select(tiny, t, stepped);
    }
    let (cx, cz) = eval(t);
    let (dx, dz) = (qx - cx, qz - cz);
    Vec3R::new((dx * dx + dz * dz).sqrt(), p.y, R::zero())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modifiers;

    fn lanes(v: Vec3) -> Vec3R<f32x8> {
        Vec3R::splat(v)
    }
    fn lane0(v: Vec3R<f32x8>) -> Vec3 {
        Vec3::new(
            v.x.as_array_ref()[0],
            v.y.as_array_ref()[0],
            v.z.as_array_ref()[0],
        )
    }
    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-4
    }

    const PTS: [Vec3; 5] = [
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(-2.0, 0.4, -0.9),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.25, 0.0, 0.0),
    ];

    #[test]
    fn generic_laws_match_scalar_modifiers_on_both_instantiations() {
        for p in PTS {
            let checks: Vec<(&str, Vec3, Vec3, Vec3)> = vec![
                (
                    "twist",
                    modifiers::modifier_twist(p, 1.5),
                    twist::<f32>(p.into(), 1.5).into(),
                    lane0(twist(lanes(p), 1.5)),
                ),
                (
                    "bend",
                    modifiers::modifier_bend(p, 0.8),
                    bend::<f32>(p.into(), 0.8).into(),
                    lane0(bend(lanes(p), 0.8)),
                ),
                (
                    "repeat_finite",
                    modifiers::modifier_repeat_finite(p, [2, 1, 2], Vec3::splat(1.5)),
                    repeat_finite::<f32>(p.into(), Vec3::new(2.0, 1.0, 2.0), Vec3::splat(1.5))
                        .into(),
                    lane0(repeat_finite(
                        lanes(p),
                        Vec3::new(2.0, 1.0, 2.0),
                        Vec3::splat(1.5),
                    )),
                ),
                (
                    "octant_mirror",
                    modifiers::modifier_octant_mirror(p),
                    octant_mirror::<f32>(p.into()).into(),
                    lane0(octant_mirror(lanes(p))),
                ),
                (
                    "polar_repeat",
                    modifiers::modifier_polar_repeat_rk(
                        p,
                        std::f32::consts::TAU / 6.0,
                        6.0 / std::f32::consts::TAU,
                    ),
                    polar_repeat::<f32>(
                        p.into(),
                        std::f32::consts::TAU / 6.0,
                        6.0 / std::f32::consts::TAU,
                    )
                    .into(),
                    lane0(polar_repeat(
                        lanes(p),
                        std::f32::consts::TAU / 6.0,
                        6.0 / std::f32::consts::TAU,
                    )),
                ),
                (
                    "sweep_bezier",
                    modifiers::modifier_sweep_bezier(
                        p,
                        Vec2::new(-1.0, 0.0),
                        Vec2::new(0.0, 1.0),
                        Vec2::new(1.0, 0.0),
                    ),
                    sweep_bezier::<f32>(
                        p.into(),
                        Vec2::new(-1.0, 0.0),
                        Vec2::new(0.0, 1.0),
                        Vec2::new(1.0, 0.0),
                    )
                    .into(),
                    lane0(sweep_bezier(
                        lanes(p),
                        Vec2::new(-1.0, 0.0),
                        Vec2::new(0.0, 1.0),
                        Vec2::new(1.0, 0.0),
                    )),
                ),
                (
                    "shear",
                    modifiers::modifier_shear(p, Vec3::new(1.0, 0.3, 0.0)),
                    shear::<f32>(p.into(), Vec3::new(1.0, 0.3, 0.0)).into(),
                    lane0(shear(lanes(p), Vec3::new(1.0, 0.3, 0.0))),
                ),
                (
                    "taper",
                    modifiers::modifier_taper(p, 0.5),
                    taper::<f32>(p.into(), 0.5).into(),
                    lane0(taper(lanes(p), 0.5)),
                ),
                (
                    "revolution",
                    modifiers::modifier_revolution(p, 0.6),
                    revolution::<f32>(p.into(), 0.6).into(),
                    lane0(revolution(lanes(p), 0.6)),
                ),
                (
                    "rotate_inverse",
                    Quat::from_rotation_y(0.7).inverse() * p,
                    rotate_inverse::<f32>(Quat::from_rotation_y(0.7), p.into()).into(),
                    lane0(rotate_inverse(Quat::from_rotation_y(0.7), lanes(p))),
                ),
            ];
            for (name, reference, scalar, simd) in checks {
                assert!(
                    close(reference, scalar),
                    "{name} scalar @ {p:?}: {reference:?} vs {scalar:?}"
                );
                assert!(
                    close(reference, simd),
                    "{name} simd @ {p:?}: {reference:?} vs {simd:?}"
                );
            }
        }
    }
}

//! Smooth CSG operations for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Branchless**: Removed `if k <= 0.0` checks. Assumes `k > 0`.
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::Real;

/// Polynomial smooth minimum (Deep Fried)
///
/// Branchless k=0 safety: clamps k to epsilon via max(),
/// which compiles to a single maxss instruction on x86.
#[inline(always)]
pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    // Branchless k=0 guard: maxss on x86, fmax on ARM
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    (h * h * k) * -0.25 + a.min(b)
}

/// Polynomial smooth minimum — Division Exorcism edition.
///
/// Takes precomputed `rk = 1.0 / k` to eliminate division from the hot path.
/// Mathematically equivalent to `smooth_min` but uses `(1.0 - abs_diff * rk)`
/// instead of `(k - abs_diff) / k`.
#[inline(always)]
pub fn smooth_min_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    smooth_min_rk_r::<f32>(a, b, k, rk)
}

/// Polynomial smooth maximum (Deep Fried)
///
/// Branchless k=0 safety via max().
#[inline(always)]
pub fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    (h * h * k) * 0.25 + a.max(b)
}

/// Polynomial smooth maximum — Division Exorcism edition.
///
/// Takes precomputed `rk = 1.0 / k` to eliminate division from the hot path.
#[inline(always)]
pub fn smooth_max_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    smooth_max_rk_r::<f32>(a, b, k, rk)
}

/// Smooth union of two SDFs
#[inline(always)]
pub fn sdf_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    smooth_min(d1, d2, k)
}

/// Smooth intersection of two SDFs
#[inline(always)]
pub fn sdf_smooth_intersection(d1: f32, d2: f32, k: f32) -> f32 {
    smooth_max(d1, d2, k)
}

/// Smooth subtraction of B from A
#[inline(always)]
pub fn sdf_smooth_subtraction(d1: f32, d2: f32, k: f32) -> f32 {
    smooth_max(d1, -d2, k)
}

/// Smooth union — Division Exorcism edition. Takes precomputed `rk = 1.0/k`.
#[inline(always)]
pub fn sdf_smooth_union_rk(d1: f32, d2: f32, k: f32, rk: f32) -> f32 {
    smooth_min_rk(d1, d2, k, rk)
}

/// Smooth intersection — Division Exorcism edition. Takes precomputed `rk = 1.0/k`.
#[inline(always)]
pub fn sdf_smooth_intersection_rk(d1: f32, d2: f32, k: f32, rk: f32) -> f32 {
    smooth_max_rk(d1, d2, k, rk)
}

/// Smooth subtraction — Division Exorcism edition. Takes precomputed `rk = 1.0/k`.
#[inline(always)]
pub fn sdf_smooth_subtraction_rk(d1: f32, d2: f32, k: f32, rk: f32) -> f32 {
    smooth_max_rk(d1, -d2, k, rk)
}

/// Exponential smooth minimum (Deep Fried)
#[inline(always)]
pub fn smooth_min_exp(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-10);
    let res = alice_det_math::exp(-k * a) + alice_det_math::exp(-k * b);
    -alice_det_math::ln(res) / k
}

/// Exponential smooth minimum — precomputed reciprocal edition.
/// Takes `rk = 1.0 / k` to eliminate division.
#[allow(dead_code)] // reserved for future SIMD optimization path
#[inline(always)]
pub fn smooth_min_exp_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    let res = alice_det_math::exp(-k * a) + alice_det_math::exp(-k * b);
    -alice_det_math::ln(res) * rk
}

/// Exponential smooth union with blend *width* `k` (`SdfNode::ExpSmoothUnion` law).
///
/// `-k * ln(exp(-d1/k) + exp(-d2/k))` — note the `d/k` convention (width),
/// unlike [`smooth_min_exp`] which takes `k` as a rate (`k*d`).
/// Single source of truth for tree / compiled scalar / SIMD per-lane paths.
#[inline(always)]
pub fn sdf_exp_smooth_union(d1: f32, d2: f32, k: f32) -> f32 {
    sdf_exp_smooth_union_r::<f32>(d1, d2, k)
}

/// Exponential smooth intersection with blend width `k` (`SdfNode::ExpSmoothIntersection` law).
#[inline(always)]
pub fn sdf_exp_smooth_intersection(d1: f32, d2: f32, k: f32) -> f32 {
    sdf_exp_smooth_intersection_r::<f32>(d1, d2, k)
}

/// Exponential smooth subtraction of B from A with blend width `k` (`SdfNode::ExpSmoothSubtraction` law).
#[inline(always)]
pub fn sdf_exp_smooth_subtraction(d1: f32, d2: f32, k: f32) -> f32 {
    sdf_exp_smooth_subtraction_r::<f32>(d1, d2, k)
}

/// Cubic smooth minimum (Deep Fried)
#[inline(always)]
pub fn smooth_min_cubic(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    (h * h * h * k) * -(1.0 / 6.0) + a.min(b)
}

/// Cubic smooth minimum — precomputed reciprocal edition.
/// Takes `rk = 1.0 / k` to eliminate division.
#[allow(dead_code)] // reserved for future SIMD optimization path
#[inline(always)]
pub fn smooth_min_cubic_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    let h = ((a - b).abs() * -rk + 1.0).max(0.0);
    (h * h * h * k) * -(1.0 / 6.0) + a.min(b)
}

/// Square root smooth minimum (IQ)
///
/// `0.5 * (a + b - sqrt((b-a)^2 + k^2))`
///
/// Unlike polynomial smooth_min, this variant is C-infinity smooth
/// and has a constant blend region width (doesn't depend on a/b separation).
/// The blend radius is always k, regardless of input values.
#[inline(always)]
pub fn smooth_min_root(a: f32, b: f32, k: f32) -> f32 {
    let x = b - a;
    0.5 * (a + b - alice_det_math::hypot(x, k))
}

// ---------------------------------------------------------------------------
// Generic ([`Real`]) forms — the single law behind the scalar and SIMD evaluators
// ---------------------------------------------------------------------------

/// Polynomial smooth minimum with precomputed `rk = 1/k` (generic over [`Real`]).
#[inline(always)]
pub fn smooth_min_rk_r<R: Real>(a: R, b: R, k: f32, rk: f32) -> R {
    let h = (R::one() - (a - b).abs() * R::splat(rk)).max(R::zero());
    a.min(b) - h * h * R::splat(k * 0.25)
}

/// Polynomial smooth maximum with precomputed `rk = 1/k` (generic over [`Real`]).
#[inline(always)]
pub fn smooth_max_rk_r<R: Real>(a: R, b: R, k: f32, rk: f32) -> R {
    let h = (R::one() - (a - b).abs() * R::splat(rk)).max(R::zero());
    a.max(b) + h * h * R::splat(k * 0.25)
}

/// Smooth union (generic, `rk = 1/k`).
#[inline(always)]
pub fn sdf_smooth_union_rk_r<R: Real>(d1: R, d2: R, k: f32, rk: f32) -> R {
    smooth_min_rk_r(d1, d2, k, rk)
}

/// Smooth intersection (generic, `rk = 1/k`).
#[inline(always)]
pub fn sdf_smooth_intersection_rk_r<R: Real>(d1: R, d2: R, k: f32, rk: f32) -> R {
    smooth_max_rk_r(d1, d2, k, rk)
}

/// Smooth subtraction of B from A (generic, `rk = 1/k`).
#[inline(always)]
pub fn sdf_smooth_subtraction_rk_r<R: Real>(d1: R, d2: R, k: f32, rk: f32) -> R {
    smooth_max_rk_r(d1, -d2, k, rk)
}

/// Exponential smooth union, blend width `k` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_exp_smooth_union_r<R: Real>(d1: R, d2: R, k: f32) -> R {
    // Stable form of `-k ln(e^{-a/k} + e^{-b/k})`: factor out the minimum so
    // the remaining exponent is ≤ 0. The textbook form underflows both
    // exponentials to 0 when d ≫ k and then `ln(0)` is -inf on libm but NaN
    // on the SIMD polynomial (found by `fuzz_eval_parity`); this form is
    // finite for every finite input and identical elsewhere.
    let k = R::splat(k.max(1e-6));
    let m = d1.min(d2);
    let delta = (d1 - d2).abs();
    m - (R::one() + (-delta / k).exp()).ln() * k
}

/// Exponential smooth intersection, blend width `k` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_exp_smooth_intersection_r<R: Real>(d1: R, d2: R, k: f32) -> R {
    // `k ln(e^{a/k} + e^{b/k})` with the maximum factored out (see union).
    let k = R::splat(k.max(1e-6));
    let m = d1.max(d2);
    let delta = (d1 - d2).abs();
    m + (R::one() + (-delta / k).exp()).ln() * k
}

/// Exponential smooth subtraction, blend width `k` (generic over [`Real`]).
#[inline(always)]
pub fn sdf_exp_smooth_subtraction_r<R: Real>(d1: R, d2: R, k: f32) -> R {
    // Intersection of `d1` with `-d2` (same stable form).
    sdf_exp_smooth_intersection_r(d1, -d2, k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_min_basic() {
        let a = 1.0;
        let b = 3.0;
        let k = 0.5;
        let result = smooth_min(a, b, k);
        assert!(result <= a);
    }

    #[test]
    fn test_smooth_union_symmetry() {
        let d1 = 0.5;
        let d2 = 0.8;
        let k = 0.3;
        assert!((sdf_smooth_union(d1, d2, k) - sdf_smooth_union(d2, d1, k)).abs() < 0.0001);
    }

    #[test]
    fn test_smooth_intersection() {
        let d1 = -0.5;
        let d2 = -0.3;
        let k = 0.2;
        let result = sdf_smooth_intersection(d1, d2, k);
        assert!(result >= d1.max(d2));
    }

    #[test]
    fn test_smooth_min_exp() {
        let result = smooth_min_exp(1.0, 1.0, 10.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_smooth_min_cubic() {
        let result = smooth_min_cubic(1.0, 1.0, 0.5);
        assert!(result < 1.0);
    }

    #[test]
    fn test_smooth_min_root_basic() {
        let result = smooth_min_root(1.0, 3.0, 0.5);
        assert!(result <= 1.0, "Should be <= min(a,b), got {}", result);
    }

    #[test]
    fn test_smooth_min_root_symmetry() {
        let k = 0.3;
        let r1 = smooth_min_root(0.5, 0.8, k);
        let r2 = smooth_min_root(0.8, 0.5, k);
        assert!(
            (r1 - r2).abs() < 1e-6,
            "Should be symmetric: {} vs {}",
            r1,
            r2
        );
    }

    #[test]
    fn test_smooth_min_root_k_zero() {
        // k=0 should degenerate to min
        let result = smooth_min_root(2.0, 5.0, 0.0);
        assert!(
            (result - 2.0).abs() < 1e-6,
            "k=0 should be min, got {}",
            result
        );
    }

    #[test]
    fn test_smooth_min_root_equal() {
        let result = smooth_min_root(1.0, 1.0, 0.5);
        assert!(
            result < 1.0,
            "Equal inputs with k>0 should blend below, got {}",
            result
        );
    }
}

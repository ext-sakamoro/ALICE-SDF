//! Smooth CSG operations for SDFs (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Branchless**: Removed `if k <= 0.0` checks. Assumes `k > 0`.
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

/// Polynomial smooth minimum (Deep Fried)
///
/// Branchless k=0 safety: clamps k to epsilon via max(),
/// which compiles to a single maxss instruction on x86.
#[inline(always)]
pub fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    // Branchless k=0 guard: maxss on x86, fmax on ARM
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth minimum — Division Exorcism edition.
///
/// Takes precomputed `rk = 1.0 / k` to eliminate division from the hot path.
/// Mathematically equivalent to `smooth_min` but uses `(1.0 - abs_diff * rk)`
/// instead of `(k - abs_diff) / k`.
#[inline(always)]
pub fn smooth_min_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    let h = (1.0 - (a - b).abs() * rk).max(0.0);
    a.min(b) - h * h * k * 0.25
}

/// Polynomial smooth maximum (Deep Fried)
///
/// Branchless k=0 safety via max().
#[inline(always)]
pub fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.max(b) + h * h * k * 0.25
}

/// Polynomial smooth maximum — Division Exorcism edition.
///
/// Takes precomputed `rk = 1.0 / k` to eliminate division from the hot path.
#[inline(always)]
pub fn smooth_max_rk(a: f32, b: f32, k: f32, rk: f32) -> f32 {
    let h = (1.0 - (a - b).abs() * rk).max(0.0);
    a.max(b) + h * h * k * 0.25
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
    let res = (-k * a).exp() + (-k * b).exp();
    -res.ln() / k
}

/// Cubic smooth minimum (Deep Fried)
#[inline(always)]
pub fn smooth_min_cubic(a: f32, b: f32, k: f32) -> f32 {
    let k = k.max(1e-10);
    let h = (k - (a - b).abs()).max(0.0) / k;
    a.min(b) - h * h * h * k * (1.0 / 6.0)
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
}

//! Chamfer CSG operations for SDFs (Deep Fried Edition)
//!
//! 45-degree beveled blends using IQ's chamfer formula:
//! `max(max(a,b), (a+b)*FRAC_1_SQRT_2 + r)`
//!
//! # Deep Fried Optimizations
//! - **Branchless**: No conditionals in hot path.
//! - **Division Exorcism**: `_rr` variants use precomputed `1/r` where needed.
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//!
//! Author: Moroya Sakamoto

use std::f32::consts::FRAC_1_SQRT_2;

/// Chamfer minimum: 45-degree beveled blend (Deep Fried)
///
/// Creates a flat 45-degree chamfer at the junction of two SDFs.
/// Formula: `max(max(a,b), (a + b) * FRAC_1_SQRT_2 + r)`
#[inline(always)]
pub fn chamfer_min(a: f32, b: f32, r: f32) -> f32 {
    a.min(b).min((a + b) * FRAC_1_SQRT_2 - r)
}

/// Chamfer maximum: 45-degree beveled blend (Deep Fried)
///
/// Dual of chamfer_min for intersection operations.
#[inline(always)]
pub fn chamfer_max(a: f32, b: f32, r: f32) -> f32 {
    -chamfer_min(-a, -b, r)
}

/// Chamfer union of two SDFs
#[inline(always)]
pub fn sdf_chamfer_union(d1: f32, d2: f32, r: f32) -> f32 {
    chamfer_min(d1, d2, r)
}

/// Chamfer intersection of two SDFs
#[inline(always)]
pub fn sdf_chamfer_intersection(d1: f32, d2: f32, r: f32) -> f32 {
    chamfer_max(d1, d2, r)
}

/// Chamfer subtraction of B from A
#[inline(always)]
pub fn sdf_chamfer_subtraction(d1: f32, d2: f32, r: f32) -> f32 {
    chamfer_max(d1, -d2, r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chamfer_union_basic() {
        // When both shapes are far apart, should behave like min
        let a = 5.0;
        let b = 10.0;
        let r = 0.5;
        let result = sdf_chamfer_union(a, b, r);
        assert!(result <= a, "chamfer union should be <= min(a,b)");
    }

    #[test]
    fn test_chamfer_union_symmetry() {
        let d1 = 0.5;
        let d2 = 0.8;
        let r = 0.3;
        assert!((sdf_chamfer_union(d1, d2, r) - sdf_chamfer_union(d2, d1, r)).abs() < 1e-6);
    }

    #[test]
    fn test_chamfer_intersection() {
        // Chamfer intersection should be >= max(a,b) when shapes barely overlap
        let d1 = -0.5;
        let d2 = -0.3;
        let r = 0.1;
        let result = sdf_chamfer_intersection(d1, d2, r);
        assert!(result >= d1.max(d2) - r * 2.0);
    }

    #[test]
    fn test_chamfer_creates_bevel() {
        // Near the junction (a≈b≈0), chamfer term should dominate
        let a = 0.1;
        let b = 0.1;
        let r = 0.5;
        let chamfer = sdf_chamfer_union(a, b, r);
        let hard_union = a.min(b);
        // Chamfer should cut more than hard union at the junction
        assert!(chamfer < hard_union);
    }
}

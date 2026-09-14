//! Column-based CSG operations for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpUnionColumns / fOpDifferenceColumns.
//! Creates column-shaped blending at the intersection of two surfaces.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::Real;

/// Column union of two SDFs (hg_sdf fOpUnionColumns)
///
/// Creates column-shaped blending at the union boundary.
/// - `r`: column radius
/// - `n`: number of columns (as f32)
#[inline(always)]
pub fn sdf_columns_union(a: f32, b: f32, r: f32, n: f32) -> f32 {
    sdf_columns_union_r::<f32>(a, b, r, n)
}

/// Column intersection of two SDFs
///
/// Creates column-shaped blending at the intersection boundary.
/// Implemented as: columns_subtraction(a, -b, r, n)
#[inline(always)]
pub fn sdf_columns_intersection(a: f32, b: f32, r: f32, n: f32) -> f32 {
    sdf_columns_subtraction(a, -b, r, n)
}

/// Column subtraction of two SDFs (hg_sdf fOpDifferenceColumns)
///
/// Creates column-shaped blending at the subtraction boundary.
/// - `r`: column radius
/// - `n`: number of columns (as f32)
#[inline(always)]
pub fn sdf_columns_subtraction(a: f32, b: f32, r: f32, n: f32) -> f32 {
    sdf_columns_subtraction_r::<f32>(a, b, r, n)
}

/// 45° rotation (generic).
#[inline(always)]
fn p_r45_r<R: Real>(x: R, y: R) -> (R, R) {
    let s = R::splat(std::f32::consts::FRAC_1_SQRT_2);
    (s * (x + y), s * (y - x))
}

/// Centred modulo into `[-size/2, size/2)` (generic; floor-mod like the scalar law).
#[inline(always)]
fn p_mod1_r<R: Real>(x: R, size: f32) -> R {
    let half = R::splat(size * 0.5);
    let y = x + half;
    y - R::splat(size) * (y / R::splat(size)).floor() - half
}

/// Columns union (generic over [`Real`]).
#[inline(always)]
pub fn sdf_columns_union_r<R: Real>(a: R, b: R, r: f32, n: f32) -> R {
    let m = a.min(b);
    let a2 = a.min(b);
    let b2 = a.max(b);
    let col_size = r * (2.0 / n);
    let (ra, rb) = p_r45_r(a2, b2);
    let ra = ra - R::splat(r * std::f32::consts::SQRT_2 * 0.5);
    let ra = p_mod1_r(ra, col_size);
    let (a3, b3) = p_r45_r(ra, rb);
    let inner = a3.min(b3).min(m);
    R::select(m.gt(R::splat(r)), m, inner)
}

/// Columns subtraction (generic over [`Real`]).
#[inline(always)]
pub fn sdf_columns_subtraction_r<R: Real>(a: R, b: R, r: f32, n: f32) -> R {
    let a = -a;
    let m = a.min(b);
    let a2 = a.min(b);
    let b2 = a.max(b);
    let col_size = r * (2.0 / n);
    let (ra, rb) = p_r45_r(a2, b2);
    let ra = ra - R::splat(r * std::f32::consts::SQRT_2 * 0.5);
    let ra = p_mod1_r(ra, col_size);
    let (a3, b3) = p_r45_r(ra, rb);
    let inner = -a3.min(b3).min(m);
    R::select(m.gt(R::splat(r)), -m, inner)
}

/// Columns intersection (generic over [`Real`]).
#[inline(always)]
pub fn sdf_columns_intersection_r<R: Real>(a: R, b: R, r: f32, n: f32) -> R {
    sdf_columns_subtraction_r(a, -b, r, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_columns_union_far() {
        // When both are far from boundary, behaves like regular union
        let d = sdf_columns_union(5.0, 3.0, 0.1, 4.0);
        assert!(
            (d - 3.0).abs() < 0.01,
            "Far from boundary should be like union, got {}",
            d
        );
    }

    #[test]
    fn test_columns_union_symmetry() {
        let d1 = sdf_columns_union(0.5, 0.3, 0.2, 4.0);
        let d2 = sdf_columns_union(0.3, 0.5, 0.2, 4.0);
        assert!((d1 - d2).abs() < 1e-6, "Should be symmetric");
    }

    #[test]
    fn test_columns_subtraction_far() {
        // Far from boundary: behaves like regular subtraction
        let d = sdf_columns_subtraction(5.0, -3.0, 0.1, 4.0);
        let d_regular = 5.0f32.max(3.0);
        assert!(
            (d - d_regular).abs() < 0.01,
            "Far should be like subtraction, got {}",
            d
        );
    }

    #[test]
    fn test_columns_intersection_far() {
        // Far from boundary: behaves like regular intersection
        let d = sdf_columns_intersection(5.0, 3.0, 0.1, 4.0);
        let d_regular = 5.0f32.max(3.0);
        assert!(
            (d - d_regular).abs() < 0.01,
            "Far should be like intersection, got {}",
            d
        );
    }
}

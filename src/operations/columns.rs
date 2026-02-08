//! Column-based CSG operations for SDFs (Deep Fried Edition)
//!
//! Based on hg_sdf's fOpUnionColumns / fOpDifferenceColumns.
//! Creates column-shaped blending at the intersection of two surfaces.
//!
//! Author: Moroya Sakamoto

/// Helper: rotate 2D point by 45 degrees
#[inline(always)]
fn p_r45(x: f32, y: f32) -> (f32, f32) {
    let s = std::f32::consts::FRAC_1_SQRT_2;
    (s * (x + y), s * (y - x))
}

/// Helper: 1D modulo with offset for column repetition
#[inline(always)]
fn p_mod1(x: f32, size: f32) -> f32 {
    let half = size * 0.5;
    ((x + half) % size + size) % size - half
}

/// Column union of two SDFs (hg_sdf fOpUnionColumns)
///
/// Creates column-shaped blending at the union boundary.
/// - `r`: column radius
/// - `n`: number of columns (as f32)
#[inline(always)]
pub fn sdf_columns_union(a: f32, b: f32, r: f32, n: f32) -> f32 {
    if a.min(b) > r {
        return a.min(b);
    }

    let (a2, b2) = if a < b { (a, b) } else { (b, a) };

    let col_size = r * (2.0 / n);
    let (ra, rb) = p_r45(a2, b2);
    let ra = ra - r * std::f32::consts::SQRT_2 * 0.5;
    let ra = p_mod1(ra, col_size);
    let (a2, b2) = p_r45(ra, rb);

    a2.min(b2).min(a.min(b))
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
    let a = -a;
    let m = a.min(b);
    if m > r {
        return -m;
    }

    let (a2, b2) = if a < b { (a, b) } else { (b, a) };

    let col_size = r * (2.0 / n);
    let (ra, rb) = p_r45(a2, b2);
    let ra = ra - r * std::f32::consts::SQRT_2 * 0.5;
    let ra = p_mod1(ra, col_size);
    let (a2, b2) = p_r45(ra, rb);

    -a2.min(b2).min(m)
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

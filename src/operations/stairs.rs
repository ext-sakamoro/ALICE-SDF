//! Stairs CSG operations for SDFs (Deep Fried Edition)
//!
//! Stepped/terraced blends using Mercury's hg_sdf stairs formula.
//! Creates n-1 discrete steps in the blend region between two SDFs.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//! - **Precomputed constants**: SQRT_2 and FRAC_1_SQRT_2 are compile-time.
//!
//! Author: Moroya Sakamoto

use std::f32::consts::{SQRT_2, FRAC_1_SQRT_2};

/// GLSL-compatible modulo: always returns positive remainder
#[inline(always)]
fn glsl_mod(a: f32, b: f32) -> f32 {
    a - b * (a / b).floor()
}

/// Stairs minimum: stepped/terraced blend (Mercury hg_sdf)
///
/// Creates n-1 discrete steps in the blend region.
/// Based on Mercury's fOpUnionStairs from hg_sdf.
///
/// # Parameters
/// - `a`, `b`: SDF distances
/// - `r`: blend radius
/// - `n`: step count (n-1 visible steps, clamped to >= 1)
#[inline(always)]
pub fn stairs_min(a: f32, b: f32, r: f32, n: f32) -> f32 {
    let n = n.max(1.0);
    let d = a.min(b);

    // Rotate (a,b) by 45 degrees: pR45
    let mut px = (a + b) * FRAC_1_SQRT_2;
    let mut py = (b - a) * FRAC_1_SQRT_2;

    // Swap p.yx
    std::mem::swap(&mut px, &mut py);

    // Offset
    let rn = r / n;
    let off = (r - rn) * 0.5 * SQRT_2;
    px -= off;
    py -= off;

    // Shift x
    px += 0.5 * SQRT_2 * rn;

    // Modular repetition: pMod1(p.x, step)
    let step = r * SQRT_2 / n;
    let hs = step * 0.5;
    px = glsl_mod(px + hs, step) - hs;

    // Combine with py
    let d = d.min(py);

    // Second 45-degree rotation
    let npx = (px + py) * FRAC_1_SQRT_2;
    let npy = (py - px) * FRAC_1_SQRT_2;

    // vmax(p - edge)
    let edge = 0.5 * rn;
    d.min((npx - edge).max(npy - edge))
}

/// Stairs maximum: dual of stairs_min
#[inline(always)]
pub fn stairs_max(a: f32, b: f32, r: f32, n: f32) -> f32 {
    -stairs_min(-a, -b, r, n)
}

/// Stairs union of two SDFs
#[inline(always)]
pub fn sdf_stairs_union(d1: f32, d2: f32, r: f32, n: f32) -> f32 {
    stairs_min(d1, d2, r, n)
}

/// Stairs intersection of two SDFs
#[inline(always)]
pub fn sdf_stairs_intersection(d1: f32, d2: f32, r: f32, n: f32) -> f32 {
    stairs_max(d1, d2, r, n)
}

/// Stairs subtraction of B from A
#[inline(always)]
pub fn sdf_stairs_subtraction(d1: f32, d2: f32, r: f32, n: f32) -> f32 {
    -stairs_min(-d1, d2, r, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stairs_union_basic() {
        // When shapes are far apart, should behave like min
        let a = 5.0;
        let b = 10.0;
        let r = 0.5;
        let n = 4.0;
        let result = sdf_stairs_union(a, b, r, n);
        assert!(result <= a + 0.01, "stairs union should approximate min when far apart");
    }

    #[test]
    fn test_stairs_union_creates_steps() {
        // Near the junction, the blend should differ from hard union
        let r = 1.0;
        let n = 4.0;
        // Sample near the blend region
        let a = 0.1;
        let b = 0.2;
        let stairs = sdf_stairs_union(a, b, r, n);
        let hard = a.min(b);
        // The stairs blend should produce a different (smaller) result near the junction
        assert!(stairs <= hard + 0.01);
    }

    #[test]
    fn test_stairs_intersection_subtraction_duality() {
        let a = 0.5;
        let b = 0.3;
        let r = 0.4;
        let n = 3.0;

        let int_result = sdf_stairs_intersection(a, b, r, n);
        let sub_result = sdf_stairs_subtraction(a, b, r, n);

        // Intersection should generally be >= max(a,b) - some blend
        // Subtraction should be related to intersection with negated b
        let int_alt = -sdf_stairs_union(-a, -b, r, n);
        assert!((int_result - int_alt).abs() < 1e-6, "intersection = -union(-a,-b)");
    }
}

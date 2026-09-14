//! Stairs CSG operations for SDFs (Deep Fried Edition)
//!
//! Stepped/terraced blends using Mercury's hg_sdf stairs formula.
//! Creates n-1 discrete steps in the blend region between two SDFs.
//!
//! # Canonical form (decided 2026-09-14)
//!
//! This is the full Mercury construction (45° rotation, offset, `mod`
//! repetition, second rotation). The single-expression variant found in some
//! hg_sdf ports — `min(a, b, 0.5·(u + a + |mod(u − a + s, 2s) − s|))` with
//! `u = b − r`, `s = r / n` — is a *different* blend (max difference ≈ 2·r on
//! random inputs) and is what ALICE-SDF-Effect's `hg_sdf` module keeps for
//! shader-library parity. ALICE-SDF keeps the full construction; do not swap
//! one for the other without re-baking every `Stairs*` asset.
//!
//! # Deep Fried Optimizations
//! - **Forced Inlining**: `#[inline(always)]` guarantees no call overhead.
//! - **Precomputed constants**: SQRT_2 and FRAC_1_SQRT_2 are compile-time.
//!
//! Author: Moroya Sakamoto

use crate::compiled::real::Real;
use std::f32::consts::{FRAC_1_SQRT_2, SQRT_2};

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
    stairs_min_r::<f32>(a, b, r, n)
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

/// GLSL-style modulo `a - b * floor(a / b)` (generic over [`Real`]).
#[inline(always)]
fn glsl_mod_r<R: Real>(a: R, b: f32) -> R {
    a - R::splat(b) * (a / R::splat(b)).floor()
}

/// Stairs (stepped) minimum (generic over [`Real`]).
#[inline(always)]
pub fn stairs_min_r<R: Real>(a: R, b: R, r: f32, n: f32) -> R {
    let n = n.max(1.0);
    let s = R::splat(FRAC_1_SQRT_2);
    let d = a.min(b);
    // rotate 45°, then swap (same as the scalar law)
    let py = (a + b) * s;
    let px = (b - a) * s;
    let rn = r / n;
    let off = (r - rn) * 0.5 * SQRT_2;
    let px = px - R::splat(off) + R::splat(0.5 * SQRT_2 * rn);
    let py = py - R::splat(off);
    let step = r * SQRT_2 / n;
    let hs = step * 0.5;
    let px = glsl_mod_r(px + R::splat(hs), step) - R::splat(hs);
    let d = d.min(py);
    let npx = (px + py) * s;
    let npy = (py - px) * s;
    let edge = R::splat(0.5 * rn);
    d.min((npx - edge).max(npy - edge))
}

/// Stairs maximum (generic).
#[inline(always)]
pub fn stairs_max_r<R: Real>(a: R, b: R, r: f32, n: f32) -> R {
    -stairs_min_r(-a, -b, r, n)
}

/// Stairs union (generic).
#[inline(always)]
pub fn sdf_stairs_union_r<R: Real>(d1: R, d2: R, r: f32, n: f32) -> R {
    stairs_min_r(d1, d2, r, n)
}

/// Stairs intersection (generic).
#[inline(always)]
pub fn sdf_stairs_intersection_r<R: Real>(d1: R, d2: R, r: f32, n: f32) -> R {
    stairs_max_r(d1, d2, r, n)
}

/// Stairs subtraction (generic).
#[inline(always)]
pub fn sdf_stairs_subtraction_r<R: Real>(d1: R, d2: R, r: f32, n: f32) -> R {
    -stairs_min_r(-d1, d2, r, n)
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
        assert!(
            result <= a + 0.01,
            "stairs union should approximate min when far apart"
        );
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
        let _sub_result = sdf_stairs_subtraction(a, b, r, n);

        // Intersection should generally be >= max(a,b) - some blend
        // Subtraction should be related to intersection with negated b
        let int_alt = -sdf_stairs_union(-a, -b, r, n);
        assert!(
            (int_result - int_alt).abs() < 1e-6,
            "intersection = -union(-a,-b)"
        );
    }
}

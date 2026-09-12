//! Morphological operations on signed distance fields.
//!
//! Complements the [`crate::shell`] module (variable-thickness offset
//! surface) with the two operations most useful for CAD / 3-D-print
//! tolerance workflows:
//!
//! - **Signed offset** — the canonical dilate (`r > 0`) / erode (`r < 0`)
//!   transformation on an SDF. Exact for the set-level operation `A ⊕ B_r`
//!   /  `A ⊖ B_r`.
//! - **Tolerance fit check** — sample-based test that the `inner` shape
//!   is contained inside `outer` inflated by `tolerance`. Reports the
//!   worst-case violation so callers can auto-adjust print clearance.
//!
//! # Set-morphology identities used
//!
//! Given the signed distance field `d_A(p)` for a set `A`:
//!
//! ```text
//! d_{A ⊕ B_r}(p) = d_A(p) - r      (dilate by ball of radius r)
//! d_{A ⊖ B_r}(p) = d_A(p) + r      (erode by ball of radius r)
//! ```
//!
//! These identities hold exactly for the SDF representation of the
//! morphologically transformed set. In contrast, the "close" and "open"
//! compositions (`dilate ∘ erode` / `erode ∘ dilate`) do not reduce to
//! closed-form SDF expressions for arbitrary shapes because they are
//! topology-changing operations that fill holes / prune protrusions;
//! implementing them properly requires reifying the intermediate level
//! set. This module therefore ships offset + tolerance only, with the
//! composed morphological operations flagged as future work.
//!
//! # Usage
//!
//! ```rust
//! use alice_sdf::morphology::{eval_offset, tolerance_fits};
//! use alice_sdf::types::SdfNode;
//! use glam::Vec3;
//!
//! let sphere = SdfNode::sphere(1.0);
//! let d = eval_offset(&sphere, Vec3::new(1.05, 0.0, 0.0), 0.1);
//! assert!(d < 0.0); // inside the offset (dilated) shape
//!
//! let inner = SdfNode::sphere(0.9);
//! let outer = SdfNode::sphere(1.0);
//! assert!(tolerance_fits(&inner, &outer, 0.05, 512, 2.0));
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use glam::Vec3;
use rayon::prelude::*;

use crate::eval::eval;
use crate::types::SdfNode;

/// Signed morphological offset applied to a SDF field.
///
/// `radius > 0` expands the surface outward (`dilate`), rounding convex
/// features and closing off narrow protrusions on the resulting level
/// set. `radius < 0` shrinks the surface inward (`erode`).
///
/// The returned value is a valid SDF for the transformed set: it is
/// negative inside the offset region and positive outside, with
/// gradient magnitude 1 wherever the underlying `node` provides a
/// well-behaved distance field.
#[inline]
#[must_use]
pub fn eval_offset(node: &SdfNode, point: Vec3, radius: f32) -> f32 {
    eval(node, point) - radius
}

/// Batched offset evaluation.
#[must_use]
pub fn eval_offset_batch(node: &SdfNode, points: &[Vec3], radius: f32) -> Vec<f32> {
    points
        .iter()
        .map(|&p| eval_offset(node, p, radius))
        .collect()
}

/// Parallel batched offset evaluation.
#[must_use]
pub fn eval_offset_batch_parallel(node: &SdfNode, points: &[Vec3], radius: f32) -> Vec<f32> {
    points
        .par_iter()
        .map(|&p| eval_offset(node, p, radius))
        .collect()
}

/// Check that `inner` fits inside `outer` inflated by `tolerance`.
///
/// Samples an axis-aligned cube centred on the origin of half-extent
/// `bounds_half_extent` at a per-axis grid resolution chosen so the
/// total sample count is `≥ sample_count`. For every sample lying on
/// or inside the `inner` surface, the check verifies that the same
/// point sits inside `outer` offset outward by `tolerance`.
///
/// A sample count of `≈ 128` — grid ≈ 5 × 5 × 5 — is adequate for
/// small-part 3-D-print clearance validation; `1024` yields tighter
/// guarantees at proportionally higher cost.
///
/// `bounds_half_extent` should enclose the `inner` shape. Callers who
/// track an AABB elsewhere can pass it directly.
///
/// # Panics
///
/// Panics if `sample_count == 0` or `bounds_half_extent <= 0.0`.
#[must_use]
pub fn tolerance_fits(
    inner: &SdfNode,
    outer: &SdfNode,
    tolerance: f32,
    sample_count: usize,
    bounds_half_extent: f32,
) -> bool {
    tolerance_max_violation(inner, outer, tolerance, sample_count, bounds_half_extent).is_none()
}

/// Sample-based worst-case tolerance violation.
///
/// Returns `Some(violation)` when at least one interior point of `inner`
/// falls outside the tolerance-inflated `outer`. The reported value is
/// the maximum (positive) SDF depth by which the inner-interior sample
/// pierced the outer offset surface. `None` indicates the fit is
/// satisfied.
///
/// # Panics
///
/// Panics if `sample_count == 0` or `bounds_half_extent <= 0.0`.
#[must_use]
pub fn tolerance_max_violation(
    inner: &SdfNode,
    outer: &SdfNode,
    tolerance: f32,
    sample_count: usize,
    bounds_half_extent: f32,
) -> Option<f32> {
    assert!(sample_count > 0, "sample_count must be > 0");
    assert!(
        bounds_half_extent > 0.0 && bounds_half_extent.is_finite(),
        "bounds_half_extent must be positive and finite"
    );
    let per_axis = ((sample_count as f32).cbrt().ceil() as usize).max(2);
    let step = (bounds_half_extent * 2.0) / (per_axis as f32 - 1.0);
    let worst = AtomicU64::new(f32::NEG_INFINITY.to_bits() as u64);

    (0..per_axis).into_par_iter().for_each(|i| {
        let x = -bounds_half_extent + (i as f32) * step;
        for j in 0..per_axis {
            let y = -bounds_half_extent + (j as f32) * step;
            for k in 0..per_axis {
                let z = -bounds_half_extent + (k as f32) * step;
                let p = Vec3::new(x, y, z);
                let d_inner = eval(inner, p);
                if d_inner <= 0.0 {
                    let d_outer_offset = eval_offset(outer, p, tolerance);
                    if d_outer_offset > 0.0 {
                        atomic_max_f32(&worst, d_outer_offset);
                    }
                }
            }
        }
    });

    let best_bits = worst.load(Ordering::Relaxed);
    let value = f32::from_bits(best_bits as u32);
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

fn atomic_max_f32(cell: &AtomicU64, candidate: f32) {
    let candidate_bits = u64::from(candidate.to_bits());
    let mut current = cell.load(Ordering::Relaxed);
    loop {
        let current_val = f32::from_bits(current as u32);
        if candidate <= current_val {
            return;
        }
        match cell.compare_exchange_weak(
            current,
            candidate_bits,
            Ordering::AcqRel,
            Ordering::Relaxed,
        ) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(r: f32) -> SdfNode {
        SdfNode::sphere(r)
    }

    #[test]
    fn offset_dilates_sphere() {
        let s = sphere(1.0);
        // Point on the offset boundary of an expanded sphere.
        let d = eval_offset(&s, Vec3::new(1.2, 0.0, 0.0), 0.2);
        assert!(d.abs() < 1.0e-4, "expected ≈ 0, got {d}");
    }

    #[test]
    fn offset_erodes_sphere() {
        let s = sphere(1.0);
        // Negative offset shrinks the sphere.
        let d = eval_offset(&s, Vec3::new(0.9, 0.0, 0.0), -0.1);
        // Point (0.9, 0, 0) has original sdf = -0.1, offset by -0.1 => 0.
        assert!(d.abs() < 1.0e-4, "expected ≈ 0, got {d}");
    }

    #[test]
    fn offset_zero_is_identity() {
        let s = sphere(1.0);
        let p = Vec3::new(0.7, 0.3, 0.2);
        let base = eval(&s, p);
        let offset_zero = eval_offset(&s, p, 0.0);
        assert!((base - offset_zero).abs() < 1.0e-6);
    }

    #[test]
    fn offset_batch_matches_scalar() {
        let s = sphere(1.0);
        let points = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.5, 0.0, 0.0),
        ];
        let batch = eval_offset_batch(&s, &points, 0.1);
        for (i, &p) in points.iter().enumerate() {
            let scalar = eval_offset(&s, p, 0.1);
            assert!((batch[i] - scalar).abs() < 1.0e-6);
        }
    }

    #[test]
    fn offset_batch_parallel_matches_scalar() {
        let s = sphere(1.0);
        let points: Vec<Vec3> = (0..64)
            .map(|i| Vec3::new((i as f32) * 0.05 - 1.5, 0.0, 0.0))
            .collect();
        let parallel = eval_offset_batch_parallel(&s, &points, 0.1);
        for (i, &p) in points.iter().enumerate() {
            let scalar = eval_offset(&s, p, 0.1);
            assert!(
                (parallel[i] - scalar).abs() < 1.0e-6,
                "mismatch at index {i}: parallel={}, scalar={}",
                parallel[i],
                scalar
            );
        }
    }

    #[test]
    fn tolerance_fits_inner_smaller_sphere() {
        let inner = sphere(0.9);
        let outer = sphere(1.0);
        // Inner is smaller — should fit inside outer even with 0 tolerance.
        assert!(tolerance_fits(&inner, &outer, 0.0, 4096, 2.0));
    }

    #[test]
    fn tolerance_fits_requires_extra_room_when_inner_too_large() {
        let inner = sphere(1.1);
        let outer = sphere(1.0);
        // Inner is 0.1 larger than outer — needs at least 0.1 tolerance to fit.
        assert!(!tolerance_fits(&inner, &outer, 0.05, 4096, 2.0));
        assert!(tolerance_fits(&inner, &outer, 0.15, 4096, 2.0));
    }

    #[test]
    fn tolerance_max_violation_reports_shortfall() {
        let inner = sphere(1.2);
        let outer = sphere(1.0);
        let violation = tolerance_max_violation(&inner, &outer, 0.05, 4096, 2.0);
        let v = violation.expect("expected a violation");
        // Inner surface is at radius 1.2, offset outer boundary at 1.05,
        // so worst-case penetration ≈ 0.15 (subject to sampling granularity).
        assert!(v > 0.05);
        assert!(v < 0.3);
    }

    #[test]
    #[should_panic(expected = "sample_count must be > 0")]
    fn tolerance_zero_samples_panics() {
        let s = sphere(1.0);
        let _ = tolerance_max_violation(&s, &s, 0.0, 0, 2.0);
    }

    #[test]
    #[should_panic(expected = "bounds_half_extent must be positive and finite")]
    fn tolerance_zero_bounds_panics() {
        let s = sphere(1.0);
        let _ = tolerance_max_violation(&s, &s, 0.0, 32, 0.0);
    }
}

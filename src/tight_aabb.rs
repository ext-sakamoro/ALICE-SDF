//! Auto tight AABB computation for SDF trees
//!
//! Uses interval arithmetic evaluation + binary search to find the
//! minimal axis-aligned bounding box that contains the SDF surface.
//!
//! # Algorithm
//!
//! For each of the 6 axis-aligned faces of a conservative initial AABB:
//! 1. Slice the volume perpendicular to the axis
//! 2. Use `eval_interval` to check if the surface can exist in the slice
//! 3. Binary search for the tightest plane where the surface still exists
//!
//! # Deep Fried Optimizations
//! - **Parallel axis search**: All 6 faces searched with Rayon.
//! - **Early-out**: Coarse scan before binary search to skip empty space fast.
//!
//! Author: Moroya Sakamoto

use crate::interval::{eval_interval, Vec3Interval};
use crate::types::{Aabb, SdfNode};
use glam::Vec3;
use rayon::prelude::*;

/// Configuration for tight AABB computation
#[derive(Debug, Clone, Copy)]
pub struct TightAabbConfig {
    /// Initial conservative half-size to start the search from (default: 10.0)
    ///
    /// The search starts with a cube `[-initial_half_size, +initial_half_size]³`
    /// and shrinks inward. Must be large enough to contain the shape.
    pub initial_half_size: f32,

    /// Number of binary search iterations per axis (default: 20)
    ///
    /// Each iteration halves the remaining uncertainty, so 20 iterations
    /// gives ~1e-6 precision on a size-10 initial box.
    pub bisection_iterations: u32,

    /// Number of coarse scan subdivisions for early-out (default: 8)
    ///
    /// Before binary search, the axis is divided into this many slabs.
    /// Empty slabs are skipped entirely, reducing the search range.
    pub coarse_subdivisions: u32,
}

impl Default for TightAabbConfig {
    fn default() -> Self {
        TightAabbConfig {
            initial_half_size: 10.0,
            bisection_iterations: 20,
            coarse_subdivisions: 8,
        }
    }
}

/// Compute a tight axis-aligned bounding box for an SDF tree.
///
/// Uses interval arithmetic to conservatively determine the smallest AABB
/// that contains the entire zero-level surface of the SDF.
///
/// # Arguments
/// * `node` - The SDF tree
///
/// # Returns
/// A tight AABB containing the SDF surface, or a zero-size AABB at origin
/// if no surface is found within the search range.
///
/// # Example
///
/// ```
/// use alice_sdf::prelude::*;
/// use alice_sdf::tight_aabb::compute_tight_aabb;
///
/// let shape = SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0);
/// let aabb = compute_tight_aabb(&shape);
///
/// // Sphere at (2,0,0) with radius 1 → AABB ~(1,-1,-1) to (3,1,1)
/// assert!(aabb.min.x > 0.5 && aabb.min.x < 1.1);
/// assert!(aabb.max.x > 2.9 && aabb.max.x < 3.5);
/// ```
pub fn compute_tight_aabb(node: &SdfNode) -> Aabb {
    compute_tight_aabb_with_config(node, &TightAabbConfig::default())
}

/// Compute a tight AABB with custom configuration.
pub fn compute_tight_aabb_with_config(node: &SdfNode, config: &TightAabbConfig) -> Aabb {
    let h = config.initial_half_size;
    let initial_min = Vec3::splat(-h);
    let initial_max = Vec3::splat(h);

    // First check: does the surface even exist in the initial box?
    let full_bounds = Vec3Interval::from_bounds(initial_min, initial_max);
    let full_interval = eval_interval(node, full_bounds);
    if full_interval.lo > 0.0 || full_interval.hi < 0.0 {
        // No surface crossing in the entire initial box
        return Aabb::new(Vec3::ZERO, Vec3::ZERO);
    }

    // Search all 6 faces in parallel (min_x, max_x, min_y, max_y, min_z, max_z)
    let results: Vec<f32> = (0..6u8)
        .into_par_iter()
        .map(|face| {
            let axis = (face / 2) as usize; // 0=X, 1=Y, 2=Z
            let is_max = face % 2 == 1;

            find_tight_bound(node, axis, is_max, initial_min, initial_max, config)
        })
        .collect();

    Aabb::new(
        Vec3::new(results[0], results[2], results[4]),
        Vec3::new(results[1], results[3], results[5]),
    )
}

/// Find the tight bound for one face of the AABB.
///
/// For min faces: search inward from initial_min[axis] toward center.
/// For max faces: search inward from initial_max[axis] toward center.
fn find_tight_bound(
    node: &SdfNode,
    axis: usize,
    is_max: bool,
    initial_min: Vec3,
    initial_max: Vec3,
    config: &TightAabbConfig,
) -> f32 {
    let lo = get_axis(initial_min, axis);
    let hi = get_axis(initial_max, axis);

    // Phase 1: Coarse scan to narrow the search range
    let (search_lo, search_hi) = coarse_scan(
        node,
        axis,
        is_max,
        lo,
        hi,
        initial_min,
        initial_max,
        config.coarse_subdivisions,
    );

    // Phase 2: Binary search within the narrowed range
    bisect_bound(
        node,
        axis,
        is_max,
        search_lo,
        search_hi,
        initial_min,
        initial_max,
        config.bisection_iterations,
    )
}

/// Coarse scan: divide the axis into subdivisions and find the outermost
/// slab that can contain the surface.
fn coarse_scan(
    node: &SdfNode,
    axis: usize,
    is_max: bool,
    lo: f32,
    hi: f32,
    initial_min: Vec3,
    initial_max: Vec3,
    subdivisions: u32,
) -> (f32, f32) {
    let step = (hi - lo) / subdivisions as f32;

    if is_max {
        // Scan from hi toward lo, find outermost slab that may contain surface
        for i in 0..subdivisions {
            let slab_hi = hi - i as f32 * step;
            let slab_lo = slab_hi - step;

            let bounds = make_slab_bounds(axis, slab_lo, slab_hi, initial_min, initial_max);
            let interval = eval_interval(node, bounds);

            if may_contain_surface(&interval) {
                // Surface may exist in [slab_lo, hi]
                // Return range for binary search: [slab_lo, slab_hi]
                return (slab_lo, slab_hi);
            }
        }
        // Nothing found, return lo
        (lo, lo)
    } else {
        // Scan from lo toward hi, find outermost slab that may contain surface
        for i in 0..subdivisions {
            let slab_lo = lo + i as f32 * step;
            let slab_hi = slab_lo + step;

            let bounds = make_slab_bounds(axis, slab_lo, slab_hi, initial_min, initial_max);
            let interval = eval_interval(node, bounds);

            if may_contain_surface(&interval) {
                return (slab_lo, slab_hi);
            }
        }
        (hi, hi)
    }
}

/// Binary search for the tight bound within [search_lo, search_hi].
fn bisect_bound(
    node: &SdfNode,
    axis: usize,
    is_max: bool,
    search_lo: f32,
    search_hi: f32,
    initial_min: Vec3,
    initial_max: Vec3,
    iterations: u32,
) -> f32 {
    let mut lo = search_lo;
    let mut hi = search_hi;

    for _ in 0..iterations {
        let mid = (lo + hi) * 0.5;

        if is_max {
            // For max bound: test slab [mid, hi] of full cross-section
            // If surface can exist in [mid, current_max], then max >= mid
            let bounds = make_slab_bounds(axis, mid, hi, initial_min, initial_max);
            let interval = eval_interval(node, bounds);

            if may_contain_surface(&interval) {
                // Surface exists above mid → keep searching higher
                lo = mid;
            } else {
                // No surface above mid → max is below mid
                hi = mid;
            }
        } else {
            // For min bound: test slab [lo, mid] of full cross-section
            let bounds = make_slab_bounds(axis, lo, mid, initial_min, initial_max);
            let interval = eval_interval(node, bounds);

            if may_contain_surface(&interval) {
                // Surface exists below mid → keep searching lower
                hi = mid;
            } else {
                // No surface below mid → min is above mid
                lo = mid;
            }
        }
    }

    if is_max {
        lo
    } else {
        hi
    }
}

/// Create a Vec3Interval slab: full extent on two axes, restricted on one axis.
#[inline(always)]
fn make_slab_bounds(
    axis: usize,
    slab_lo: f32,
    slab_hi: f32,
    initial_min: Vec3,
    initial_max: Vec3,
) -> Vec3Interval {
    match axis {
        0 => Vec3Interval::from_bounds(
            Vec3::new(slab_lo, initial_min.y, initial_min.z),
            Vec3::new(slab_hi, initial_max.y, initial_max.z),
        ),
        1 => Vec3Interval::from_bounds(
            Vec3::new(initial_min.x, slab_lo, initial_min.z),
            Vec3::new(initial_max.x, slab_hi, initial_max.z),
        ),
        _ => Vec3Interval::from_bounds(
            Vec3::new(initial_min.x, initial_min.y, slab_lo),
            Vec3::new(initial_max.x, initial_max.y, slab_hi),
        ),
    }
}

/// Check if an interval may contain the zero-level surface.
/// The surface exists where SDF transitions from negative to positive,
/// so we need the interval to span zero.
#[inline(always)]
fn may_contain_surface(interval: &crate::interval::Interval) -> bool {
    interval.lo <= 0.0 && interval.hi >= 0.0
}

/// Get axis value from Vec3
#[inline(always)]
fn get_axis(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_tight_aabb() {
        let sphere = SdfNode::sphere(1.0);
        let aabb = compute_tight_aabb(&sphere);

        // Sphere of radius 1 → AABB should be close to [-1, 1]³
        for i in 0..3 {
            let lo = get_axis(aabb.min, i);
            let hi = get_axis(aabb.max, i);
            assert!(
                lo < -0.9 && lo > -1.2,
                "min[{}] = {} should be near -1.0",
                i,
                lo
            );
            assert!(
                hi > 0.9 && hi < 1.2,
                "max[{}] = {} should be near 1.0",
                i,
                hi
            );
        }
    }

    #[test]
    fn test_box_tight_aabb() {
        // box3d(w,h,d) stores half_extents = (w/2, h/2, d/2)
        // So box3d(1.0, 2.0, 0.5) → half_extents (0.5, 1.0, 0.25)
        // → AABB [-0.5,0.5] x [-1.0,1.0] x [-0.25,0.25]
        let box3 = SdfNode::box3d(1.0, 2.0, 0.5);
        let aabb = compute_tight_aabb(&box3);

        assert!(
            aabb.min.x < -0.4 && aabb.min.x > -0.7,
            "min.x = {}",
            aabb.min.x
        );
        assert!(
            aabb.max.x > 0.4 && aabb.max.x < 0.7,
            "max.x = {}",
            aabb.max.x
        );
        assert!(
            aabb.min.y < -0.9 && aabb.min.y > -1.2,
            "min.y = {}",
            aabb.min.y
        );
        assert!(
            aabb.max.y > 0.9 && aabb.max.y < 1.2,
            "max.y = {}",
            aabb.max.y
        );
        assert!(
            aabb.min.z < -0.15 && aabb.min.z > -0.4,
            "min.z = {}",
            aabb.min.z
        );
        assert!(
            aabb.max.z > 0.15 && aabb.max.z < 0.4,
            "max.z = {}",
            aabb.max.z
        );
    }

    #[test]
    fn test_translated_sphere() {
        let shape = SdfNode::sphere(1.0).translate(3.0, -2.0, 1.0);
        let aabb = compute_tight_aabb(&shape);

        // Sphere at (3, -2, 1) → AABB ~(2, -3, 0) to (4, -1, 2)
        assert!(
            aabb.min.x > 1.5 && aabb.min.x < 2.2,
            "min.x = {}",
            aabb.min.x
        );
        assert!(
            aabb.max.x > 3.8 && aabb.max.x < 4.5,
            "max.x = {}",
            aabb.max.x
        );
        assert!(
            aabb.min.y > -3.2 && aabb.min.y < -2.5,
            "min.y = {}",
            aabb.min.y
        );
        assert!(
            aabb.max.y > -1.5 && aabb.max.y < -0.8,
            "max.y = {}",
            aabb.max.y
        );
    }

    #[test]
    fn test_scaled_sphere() {
        let shape = SdfNode::sphere(1.0).scale(3.0);
        let aabb = compute_tight_aabb(&shape);

        // Scaled sphere radius 3 → AABB ~[-3, 3]³
        for i in 0..3 {
            let lo = get_axis(aabb.min, i);
            let hi = get_axis(aabb.max, i);
            assert!(lo < -2.7 && lo > -3.5, "min[{}] = {}", i, lo);
            assert!(hi > 2.7 && hi < 3.5, "max[{}] = {}", i, hi);
        }
    }

    #[test]
    fn test_union_aabb() {
        let shape = SdfNode::sphere(1.0)
            .translate(-3.0, 0.0, 0.0)
            .union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0));

        let aabb = compute_tight_aabb(&shape);

        // Two spheres at (-3,0,0) and (3,0,0) → AABB ~(-4,-1,-1) to (4,1,1)
        assert!(aabb.min.x < -3.5, "min.x = {}", aabb.min.x);
        assert!(aabb.max.x > 3.5, "max.x = {}", aabb.max.x);
        assert!(
            aabb.min.y > -1.5 && aabb.min.y < -0.5,
            "min.y = {}",
            aabb.min.y
        );
    }

    #[test]
    fn test_subtraction_aabb() {
        // Sphere minus small box → outer surface is the sphere
        let shape = SdfNode::sphere(2.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));

        let aabb = compute_tight_aabb(&shape);

        // Should be close to sphere bounds [-2, 2]³
        assert!(aabb.min.x < -1.5, "min.x = {}", aabb.min.x);
        assert!(aabb.max.x > 1.5, "max.x = {}", aabb.max.x);
    }

    #[test]
    fn test_no_surface_returns_zero() {
        // A shape that's entirely positive (empty) within the search range
        // A plane at y=100 — no surface in [-10, 10]³
        let shape = SdfNode::Plane {
            normal: Vec3::Y,
            distance: 100.0,
        };

        let aabb = compute_tight_aabb(&shape);

        // Should return zero-size AABB
        assert_eq!(aabb.min, Vec3::ZERO);
        assert_eq!(aabb.max, Vec3::ZERO);
    }

    #[test]
    fn test_custom_config() {
        let sphere = SdfNode::sphere(5.0);

        // Default initial_half_size=10 should not be enough for larger shapes
        // but radius 5 fits in [-10, 10]³
        let config = TightAabbConfig {
            initial_half_size: 8.0,
            bisection_iterations: 15,
            coarse_subdivisions: 4,
        };

        let aabb = compute_tight_aabb_with_config(&sphere, &config);

        for i in 0..3 {
            let lo = get_axis(aabb.min, i);
            let hi = get_axis(aabb.max, i);
            assert!(lo < -4.5 && lo > -5.5, "min[{}] = {}", i, lo);
            assert!(hi > 4.5 && hi < 5.5, "max[{}] = {}", i, hi);
        }
    }

    #[test]
    fn test_torus_tight_aabb() {
        let torus = SdfNode::torus(2.0, 0.5);
        let aabb = compute_tight_aabb(&torus);

        // Torus major=2, minor=0.5 → AABB ~(-2.5, -0.5, -2.5) to (2.5, 0.5, 2.5)
        assert!(
            aabb.min.x < -2.3 && aabb.min.x > -2.8,
            "min.x = {}",
            aabb.min.x
        );
        assert!(
            aabb.max.x > 2.3 && aabb.max.x < 2.8,
            "max.x = {}",
            aabb.max.x
        );
        assert!(
            aabb.min.y > -0.8 && aabb.min.y < -0.3,
            "min.y = {}",
            aabb.min.y
        );
        assert!(
            aabb.max.y > 0.3 && aabb.max.y < 0.8,
            "max.y = {}",
            aabb.max.y
        );
    }
}

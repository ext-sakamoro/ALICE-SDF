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
use glam::{Mat3, Vec3};
use rayon::prelude::*;

/// Configuration for tight AABB computation.
///
/// Construct via [`TightAabbConfig::preset_small`] / [`preset_medium`](Self::preset_medium) /
/// [`preset_large`](Self::preset_large) for canonical use cases, or
/// [`try_new`](Self::try_new) for validated arbitrary values.
///
/// `impl Default` was intentionally removed in 1.7.7 to prevent silent
/// load-bearing use of the historical default (`initial_half_size: 10.0`) which
/// caused large patterns (SKADIS panel / gridfinity 200mm+ bbox) to fall outside
/// the search range and produce empty AABB. See `rust-config-struct-guard` skill.
#[derive(Debug, Clone, Copy)]
pub struct TightAabbConfig {
    /// Initial conservative half-size to start the search from.
    ///
    /// The search starts with a cube `[-initial_half_size, +initial_half_size]³`
    /// and shrinks inward. Must be large enough to contain the shape.
    initial_half_size: f32,

    /// Number of binary search iterations per axis.
    ///
    /// Each iteration halves the remaining uncertainty, so 20 iterations
    /// gives ~1e-6 precision on a size-10 initial box.
    bisection_iterations: u32,

    /// Number of coarse scan subdivisions for early-out.
    ///
    /// Before binary search, the axis is divided into this many slabs.
    /// Empty slabs are skipped entirely, reducing the search range.
    coarse_subdivisions: u32,
}

/// Error variants for [`TightAabbConfig::try_new`].
#[derive(Debug, thiserror::Error)]
pub enum TightAabbConfigError {
    /// `initial_half_size` must be strictly positive.
    #[error("initial_half_size must be > 0, got {0}")]
    InvalidHalfSize(f32),
    /// `bisection_iterations` must be strictly positive.
    #[error("bisection_iterations must be > 0")]
    ZeroIterations,
    /// `coarse_subdivisions` must be strictly positive.
    #[error("coarse_subdivisions must be > 0")]
    ZeroSubdivisions,
}

impl TightAabbConfig {
    /// Preset for shapes ≤ 20mm bbox (`initial_half_size: 10.0, iters: 20, subdivisions: 8`).
    ///
    /// Historical `Default::default()` values. Coin / badge / small mechanical parts.
    #[must_use]
    pub const fn preset_small() -> Self {
        Self {
            initial_half_size: 10.0,
            bisection_iterations: 20,
            coarse_subdivisions: 8,
        }
    }

    /// Preset for shapes 20-200mm bbox (`initial_half_size: 100.0, iters: 22, subdivisions: 12`).
    ///
    /// Intermediate: bracket / holder / desktop organizer.
    #[must_use]
    pub const fn preset_medium() -> Self {
        Self {
            initial_half_size: 100.0,
            bisection_iterations: 22,
            coarse_subdivisions: 12,
        }
    }

    /// Preset for shapes 200mm+ bbox (`initial_half_size: 500.0, iters: 24, subdivisions: 16`).
    ///
    /// Bamboo canonical (matches `alice_bamboo::sdf_to_bambu_3mf` internal config).
    /// SKADIS panel / gridfinity / large wall organizer.
    #[must_use]
    pub const fn preset_large() -> Self {
        Self {
            initial_half_size: 500.0,
            bisection_iterations: 24,
            coarse_subdivisions: 16,
        }
    }

    /// Construct with custom values, validating each field.
    ///
    /// # Errors
    /// - [`TightAabbConfigError::InvalidHalfSize`] if `initial_half_size <= 0`
    /// - [`TightAabbConfigError::ZeroIterations`] if `bisection_iterations == 0`
    /// - [`TightAabbConfigError::ZeroSubdivisions`] if `coarse_subdivisions == 0`
    pub fn try_new(
        initial_half_size: f32,
        bisection_iterations: u32,
        coarse_subdivisions: u32,
    ) -> Result<Self, TightAabbConfigError> {
        if initial_half_size.is_nan() || initial_half_size <= 0.0 {
            return Err(TightAabbConfigError::InvalidHalfSize(initial_half_size));
        }
        if bisection_iterations == 0 {
            return Err(TightAabbConfigError::ZeroIterations);
        }
        if coarse_subdivisions == 0 {
            return Err(TightAabbConfigError::ZeroSubdivisions);
        }
        Ok(Self {
            initial_half_size,
            bisection_iterations,
            coarse_subdivisions,
        })
    }

    /// Initial half-size accessor.
    #[must_use]
    #[inline]
    pub const fn initial_half_size(&self) -> f32 {
        self.initial_half_size
    }

    /// Bisection iterations accessor.
    #[must_use]
    #[inline]
    pub const fn bisection_iterations(&self) -> u32 {
        self.bisection_iterations
    }

    /// Coarse subdivisions accessor.
    #[must_use]
    #[inline]
    pub const fn coarse_subdivisions(&self) -> u32 {
        self.coarse_subdivisions
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
    compute_tight_aabb_with_config(node, &TightAabbConfig::preset_small())
}

/// Compute a tight AABB with custom configuration.
pub fn compute_tight_aabb_with_config(node: &SdfNode, config: &TightAabbConfig) -> Aabb {
    let h = config.initial_half_size();
    let initial_min = Vec3::splat(-h);
    let initial_max = Vec3::splat(h);

    // First check: does the surface even exist in the initial box?
    let full_bounds = Vec3Interval::from_bounds(initial_min, initial_max);
    let full_interval = eval_interval(node, full_bounds);
    if full_interval.is_positive() || full_interval.is_negative() {
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

    let searched = Aabb::new(
        Vec3::new(results[0], results[2], results[4]),
        Vec3::new(results[1], results[3], results[5]),
    );

    // Both the interval search and the level-set bound contain the surface, so their
    // intersection does too, and it is never looser than either. The interval search
    // inflates under rotation (it drops the correlation between the rotated axes);
    // the level-set bound does not.
    match analytic_aabb(node) {
        AnalyticAabb::Bounded(analytic) => {
            let lo = searched.min.max(analytic.min);
            let hi = searched.max.min(analytic.max);
            if lo.x <= hi.x && lo.y <= hi.y && lo.z <= hi.z {
                Aabb::new(lo, hi)
            } else {
                searched
            }
        }
        AnalyticAabb::Empty | AnalyticAabb::Unsupported => searched,
    }
}

/// Result of [`analytic_aabb`], the level-set bound propagated over the tree.
#[derive(Debug, Clone, Copy)]
pub enum AnalyticAabb {
    /// The solid `{f <= 0}` is empty (no surface).
    Empty,
    /// A box that contains the solid `{f <= 0}`, and with it the surface `{f = 0}`.
    Bounded(Aabb),
    /// The tree contains a node this bound does not cover (twist, bend, repeat,
    /// approximate distance fields, ...): no analytic statement is made.
    Unsupported,
}

/// An affine map `p -> a * p + t` accumulated from the root down to a leaf.
#[derive(Clone, Copy)]
struct Affine {
    a: Mat3,
    t: Vec3,
}

/// A region in world space: empty, a box, or "not known".
#[derive(Clone, Copy)]
enum Region {
    Empty,
    Box(Vec3, Vec3),
    Unknown,
}

impl Region {
    fn hull(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Empty, r) | (r, Self::Empty) => r,
            (Self::Box(a0, a1), Self::Box(b0, b1)) => Self::Box(a0.min(b0), a1.max(b1)),
        }
    }

    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, r) | (r, Self::Unknown) => r,
            (Self::Empty, _) | (_, Self::Empty) => Self::Empty,
            (Self::Box(a0, a1), Self::Box(b0, b1)) => {
                let (lo, hi) = (a0.max(b0), a1.min(b1));
                if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
                    Self::Empty
                } else {
                    Self::Box(lo, hi)
                }
            }
        }
    }
}

/// Bound of the leaf box `center +- half` (local frame) mapped through `xf`.
///
/// A box maps to the box with half-extents `|A| * half` exactly, so a chain of
/// rigid transforms costs one evaluation at the leaf and does not accumulate.
fn leaf_box(xf: &Affine, center: Vec3, half: Vec3) -> Region {
    // NaN / inf is "not known"; a negative half-extent is an empty level set
    if !center.is_finite() || !half.is_finite() {
        return Region::Unknown;
    }
    if half.min_element() < 0.0 {
        return Region::Empty;
    }
    let abs = Mat3::from_cols(xf.a.x_axis.abs(), xf.a.y_axis.abs(), xf.a.z_axis.abs());
    let c = xf.a * center + xf.t;
    let h = abs * half;
    Region::Box(c - h, c + h)
}

/// Row `i` of `a` (`Mat3` is column-major).
fn row(a: &Mat3, i: usize) -> Vec3 {
    Vec3::new(a.x_axis[i], a.y_axis[i], a.z_axis[i])
}

/// Bound of a round leaf mapped through `xf`: the Minkowski sum of
/// - a ball of radius `ball` (image: an ellipsoid, extent `ball * |row_i(A)|`),
/// - a disc of radius `disc` in the local XZ plane (extent `disc * |(A_i0, A_i2)|`),
/// - a segment `center +- seg` (extent `|(A * seg)_i|`).
///
/// The AABB of a Minkowski sum is the sum of the AABBs, and each term is the exact
/// AABB of its image under any linear map, so a rotated sphere / cylinder / torus /
/// capsule gets its exact box rather than the box of its rotated box.
fn round_leaf(xf: &Affine, center: Vec3, seg: Vec3, disc: f32, ball: f32) -> Region {
    if !center.is_finite() || !seg.is_finite() || !disc.is_finite() || !ball.is_finite() {
        return Region::Unknown;
    }
    if disc < 0.0 || ball < 0.0 {
        return Region::Empty;
    }
    let c = xf.a * center + xf.t;
    let s = (xf.a * seg).abs();
    let mut h = s;
    for i in 0..3 {
        let r = row(&xf.a, i);
        h[i] += disc * alice_det_math::hypot(r.x, r.z) + ball * r.length();
    }
    Region::Box(c - h, c + h)
}

/// World-space region that contains `{ p : f_node(p) <= delta }` (node local frame
/// mapped through `xf`).
///
/// `delta` is the level: the surface is level 0, an offset by `r` asks the child
/// for level `delta + r`, a smooth union loosens it by the largest deviation of
/// its blend. Only nodes whose level sets are known are covered; everything else
/// is [`Region::Unknown`] so the caller falls back to the interval search.
fn level_region(node: &SdfNode, xf: &Affine, delta: f32) -> Region {
    if !delta.is_finite() {
        return Region::Unknown;
    }
    match node {
        // exact distance fields: the level set of f is the shape grown by delta
        SdfNode::Sphere { radius } => round_leaf(xf, Vec3::ZERO, Vec3::ZERO, 0.0, radius + delta),
        SdfNode::Box3d { half_extents } => {
            leaf_box(xf, Vec3::ZERO, *half_extents + Vec3::splat(delta))
        }
        // a disc of radius r in XZ swept along Y by +-h; the level-delta set of the
        // exact field lies in the cylinder (r + delta, h + delta)
        SdfNode::Cylinder {
            radius,
            half_height,
        } => {
            let (r, h) = (radius + delta, half_height + delta);
            if h < 0.0 {
                return Region::Empty;
            }
            round_leaf(xf, Vec3::ZERO, Vec3::new(0.0, h, 0.0), r, 0.0)
        }
        // the circle of radius R in XZ plus a ball of the minor radius
        SdfNode::Torus {
            major_radius,
            minor_radius,
        } => {
            let minor = minor_radius + delta;
            if minor < 0.0 || *major_radius < 0.0 {
                return Region::Empty;
            }
            round_leaf(xf, Vec3::ZERO, Vec3::ZERO, *major_radius, minor)
        }
        // the segment a-b plus a ball
        SdfNode::Capsule {
            point_a,
            point_b,
            radius,
        } => {
            let center = (*point_a + *point_b) * 0.5;
            let seg = (*point_b - *point_a) * 0.5;
            round_leaf(xf, center, seg, 0.0, radius + delta)
        }
        // f >= max_i(|p_i| - b_i) - r, so {f <= delta} lies in the box b + r + delta
        SdfNode::RoundedBox {
            half_extents,
            round_radius,
        } => leaf_box(
            xf,
            Vec3::ZERO,
            *half_extents + Vec3::splat(round_radius + delta),
        ),
        // f >= max(rho - (radius - rr), |y| - h) - rr, so rho <= radius + delta and
        // |y| <= h + rr + delta
        SdfNode::RoundedCylinder {
            radius,
            round_radius,
            half_height,
        } => {
            let h = half_height + round_radius + delta;
            if h < 0.0 {
                return Region::Empty;
            }
            round_leaf(xf, Vec3::ZERO, Vec3::new(0.0, h, 0.0), radius + delta, 0.0)
        }
        // set operations
        SdfNode::Union { a, b } => level_region(a, xf, delta).hull(level_region(b, xf, delta)),
        SdfNode::Intersection { a, b } => {
            level_region(a, xf, delta).meet(level_region(b, xf, delta))
        }
        // max(fa, -fb) <= delta  implies  fa <= delta (b only removes material)
        SdfNode::Subtraction { a, .. } => level_region(a, xf, delta),
        // polynomial smooth min is at least min - k/4
        SdfNode::SmoothUnion { a, b, k } if k.is_finite() && *k >= 0.0 => {
            let d = delta + k * 0.25;
            level_region(a, xf, d).hull(level_region(b, xf, d))
        }
        // polynomial smooth max is max(a, b) plus a non-negative term (k >= 0), so its level
        // set lies in the plain intersection / in the minuend's level set
        SdfNode::SmoothIntersection { a, b, k } if k.is_finite() && *k >= 0.0 => {
            level_region(a, xf, delta).meet(level_region(b, xf, delta))
        }
        SdfNode::SmoothSubtraction { a, k, .. } if k.is_finite() && *k >= 0.0 => {
            level_region(a, xf, delta)
        }
        // the material tag does not change the field
        SdfNode::WithMaterial { child, .. } => level_region(child, xf, delta),
        // rigid and uniform-scale transforms are folded into the accumulated map
        SdfNode::Translate { child, offset } => {
            let moved = Affine {
                a: xf.a,
                t: xf.t + xf.a * *offset,
            };
            level_region(child, &moved, delta)
        }
        SdfNode::Rotate { child, rotation } => {
            let turned = Affine {
                a: xf.a * Mat3::from_quat(rotation.normalize()),
                t: xf.t,
            };
            level_region(child, &turned, delta)
        }
        // f(p) = s * child(p / s): {f <= delta} = s * {child <= delta / s}
        SdfNode::Scale { child, factor } if factor.is_finite() && *factor > 0.0 => {
            let scaled = Affine {
                a: xf.a * *factor,
                t: xf.t,
            };
            level_region(child, &scaled, delta / factor)
        }
        // f = child - r  /  f = |child| - t  ->  child <= delta + r / delta + t
        SdfNode::Round { child, radius } => level_region(child, xf, delta + radius),
        SdfNode::Onion { child, thickness } => level_region(child, xf, delta + thickness),
        // f(p) = child(p - clamp(p, -h, h)): p = clamp(p) + q with q in the child's set and
        // clamp(p) in [-h, h], so the set lies in (child set) + [-h, h] (Minkowski sum, local
        // frame). Through `xf` that is the child's region plus the box `|A| * h`, which is the
        // exact AABB of the parallelepiped `A * [-h, h]`.
        SdfNode::Elongate { child, amount } => {
            if !amount.is_finite() || amount.min_element() < 0.0 {
                return Region::Unknown;
            }
            match level_region(child, xf, delta) {
                Region::Box(lo, hi) => {
                    let abs =
                        Mat3::from_cols(xf.a.x_axis.abs(), xf.a.y_axis.abs(), xf.a.z_axis.abs());
                    let grow = abs * *amount;
                    Region::Box(lo - grow, hi + grow)
                }
                other => other,
            }
        }
        // f(p) = child(|p| on the mirrored axes): bound the child in the local frame, then
        // |p_i| <= hi_i on a mirrored axis (empty when the child lies entirely below 0)
        SdfNode::Mirror { child, axes } => {
            let identity = Affine {
                a: Mat3::IDENTITY,
                t: Vec3::ZERO,
            };
            match level_region(child, &identity, delta) {
                Region::Box(mut lo, hi) => {
                    for i in 0..3 {
                        if axes[i] != 0.0 {
                            if hi[i] < 0.0 {
                                return Region::Empty;
                            }
                            lo[i] = -hi[i];
                        }
                    }
                    leaf_box(xf, (lo + hi) * 0.5, (hi - lo) * 0.5)
                }
                other => other,
            }
        }
        // f(p) = m * child(p / s), m = min(s): {f <= delta} = s * {child <= delta / m}
        SdfNode::ScaleNonUniform { child, factors }
            if factors.is_finite()
                && factors.min_element() > 0.0
                && factors.recip().is_finite() =>
        {
            let scaled = Affine {
                a: xf.a * Mat3::from_diagonal(*factors),
                t: xf.t,
            };
            level_region(child, &scaled, delta / factors.min_element())
        }
        _ => Region::Unknown,
    }
}

/// Bound the solid `{f <= 0}` of `node` by propagating the level `delta` down the
/// tree and applying the accumulated rigid / uniform-scale transform once, at the
/// leaf (see the module docs).
///
/// Unlike the interval search this does not inflate under rotation: a rotated
/// box is bounded by `|R| * half`, exactly, however deeply the rotations nest.
/// The result is padded outward by a few ulps so rounding cannot cut the surface.
#[must_use]
pub fn analytic_aabb(node: &SdfNode) -> AnalyticAabb {
    let identity = Affine {
        a: Mat3::IDENTITY,
        t: Vec3::ZERO,
    };
    match level_region(node, &identity, 0.0) {
        Region::Empty => AnalyticAabb::Empty,
        Region::Unknown => AnalyticAabb::Unsupported,
        Region::Box(lo, hi) => {
            let pad = (lo.abs().max(hi.abs()).max_element() * 4.0 + 1.0) * f32::EPSILON * 4.0;
            AnalyticAabb::Bounded(Aabb::new(lo - Vec3::splat(pad), hi + Vec3::splat(pad)))
        }
    }
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
        config.coarse_subdivisions(),
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
        config.bisection_iterations(),
    )
}

/// Coarse scan: divide the axis into subdivisions and find the outermost
/// slab that can contain the surface.
#[allow(clippy::too_many_arguments)]
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
            let slab_hi = (i as f32).mul_add(-step, hi);
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
            let slab_lo = (i as f32).mul_add(step, lo);
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
#[allow(clippy::too_many_arguments)]
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
        let mid = f32::midpoint(lo, hi);

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
    interval.contains(0.0)
}

/// Get axis value from Vec3
#[inline(always)]
const fn get_axis(v: Vec3, axis: usize) -> f32 {
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
        let config = TightAabbConfig::try_new(8.0, 15, 4).expect("valid config");

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

//! SDF-to-SDF collision detection.
//!
//! Grid-based contact detection between two SDF fields, with interval
//! arithmetic AABB pruning for early rejection of empty regions.
//!
//! # Example
//!
//! ```rust
//! use alice_sdf::prelude::*;
//! use alice_sdf::collision::*;
//!
//! let a = SdfNode::sphere(1.0);
//! let b = SdfNode::sphere(1.0).translate(1.5, 0.0, 0.0);
//!
//! let aabb = Aabb { min: glam::Vec3::splat(-3.0), max: glam::Vec3::splat(3.0) };
//! assert!(sdf_overlap(&a, &b, &aabb, 16));
//!
//! let contacts = sdf_collide(&a, &b, &aabb, 16);
//! assert!(!contacts.is_empty());
//! ```
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, normal};
use crate::interval::{eval_interval, Vec3Interval};
use crate::types::{Aabb, SdfNode};
use glam::Vec3;

/// A contact point between two SDF surfaces.
#[derive(Debug, Clone, Copy)]
pub struct SdfContact {
    /// World-space contact point (midpoint of the overlap region).
    pub point: Vec3,
    /// Contact normal (points from B towards A).
    pub normal: Vec3,
    /// Penetration depth (positive = overlapping).
    pub depth: f32,
}

/// Detect all contact points between two SDFs on a uniform grid.
///
/// Samples `resolution³` points in `aabb`. A contact is reported where
/// both `a(p) < 0` and `b(p) < 0` (both interiors overlap).
///
/// Uses interval arithmetic to skip sub-cells where either SDF is
/// provably positive (no surface present).
///
/// Returns contacts sorted by depth (deepest first).
pub fn sdf_collide(a: &SdfNode, b: &SdfNode, aabb: &Aabb, resolution: u32) -> Vec<SdfContact> {
    let res = resolution.max(2) as usize;
    let extent = aabb.max - aabb.min;
    let cell = extent / res as f32;

    let mut contacts = Vec::new();

    // Subdivide into cells and prune with interval arithmetic
    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                let lo = aabb.min
                    + Vec3::new(ix as f32 * cell.x, iy as f32 * cell.y, iz as f32 * cell.z);
                let hi = lo + cell;

                // Interval pruning: if either SDF is entirely positive in
                // this cell, no overlap is possible.
                let ia = eval_interval(a, Vec3Interval::from_bounds(lo, hi));
                if ia.lo > 0.0 {
                    continue;
                }
                let ib = eval_interval(b, Vec3Interval::from_bounds(lo, hi));
                if ib.lo > 0.0 {
                    continue;
                }

                // Sample cell center
                let center = (lo + hi) * 0.5;
                let da = eval(a, center);
                let db = eval(b, center);

                if da < 0.0 && db < 0.0 {
                    // Both SDFs claim this point is interior → overlap
                    let n = normal(b, center, 0.001);
                    let depth = -(da.max(db)); // penetration depth
                    contacts.push(SdfContact {
                        point: center,
                        normal: n,
                        depth,
                    });
                }
            }
        }
    }

    // Sort deepest first
    contacts.sort_by(|c1, c2| {
        c2.depth
            .partial_cmp(&c1.depth)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    contacts
}

/// Minimum separation distance between two SDFs.
///
/// Returns 0.0 if they overlap, otherwise the smallest `da + db` found
/// on the grid (an upper bound on the true separation).
pub fn sdf_distance(a: &SdfNode, b: &SdfNode, aabb: &Aabb, resolution: u32) -> f32 {
    let res = resolution.max(2) as usize;
    let extent = aabb.max - aabb.min;
    let cell = extent / res as f32;
    let mut min_sep = f32::MAX;

    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                let lo = aabb.min
                    + Vec3::new(ix as f32 * cell.x, iy as f32 * cell.y, iz as f32 * cell.z);
                let hi = lo + cell;

                // Interval pruning: skip if we can't improve
                let ia = eval_interval(a, Vec3Interval::from_bounds(lo, hi));
                let ib = eval_interval(b, Vec3Interval::from_bounds(lo, hi));
                if ia.lo + ib.lo >= min_sep {
                    continue;
                }

                let center = (lo + hi) * 0.5;
                let da = eval(a, center);
                let db = eval(b, center);

                if da < 0.0 && db < 0.0 {
                    return 0.0; // overlapping
                }

                // Sum of distances (upper bound on separation)
                let sep = da.max(0.0) + db.max(0.0);
                if sep < min_sep {
                    min_sep = sep;
                }
            }
        }
    }

    min_sep
}

/// Fast boolean overlap test with early exit.
///
/// Returns `true` if the interiors of `a` and `b` overlap anywhere
/// on the sample grid.
pub fn sdf_overlap(a: &SdfNode, b: &SdfNode, aabb: &Aabb, resolution: u32) -> bool {
    let res = resolution.max(2) as usize;
    let extent = aabb.max - aabb.min;
    let cell = extent / res as f32;

    for iz in 0..res {
        for iy in 0..res {
            for ix in 0..res {
                let lo = aabb.min
                    + Vec3::new(ix as f32 * cell.x, iy as f32 * cell.y, iz as f32 * cell.z);
                let hi = lo + cell;

                let ia = eval_interval(a, Vec3Interval::from_bounds(lo, hi));
                if ia.lo > 0.0 {
                    continue;
                }
                let ib = eval_interval(b, Vec3Interval::from_bounds(lo, hi));
                if ib.lo > 0.0 {
                    continue;
                }

                let center = (lo + hi) * 0.5;
                let da = eval(a, center);
                let db = eval(b, center);

                if da < 0.0 && db < 0.0 {
                    return true;
                }
            }
        }
    }

    false
}

/// Contact manifold: aggregated collision result.
#[derive(Debug, Clone)]
pub struct ContactManifold {
    /// Average contact point.
    pub center: Vec3,
    /// Average contact normal.
    pub normal: Vec3,
    /// Maximum penetration depth.
    pub max_depth: f32,
    /// Number of contact points in the manifold.
    pub count: usize,
}

/// Compute a contact manifold from a set of contact points.
///
/// Aggregates individual contacts into a single manifold with
/// averaged position/normal and maximum depth.
pub fn compute_manifold(contacts: &[SdfContact]) -> Option<ContactManifold> {
    if contacts.is_empty() {
        return None;
    }
    let count = contacts.len();
    let inv_n = 1.0 / count as f32;
    let mut center = Vec3::ZERO;
    let mut normal = Vec3::ZERO;
    let mut max_depth: f32 = 0.0;
    for c in contacts {
        center += c.point;
        normal += c.normal;
        max_depth = max_depth.max(c.depth);
    }
    center *= inv_n;
    let n_len = normal.length();
    if n_len > 1e-10 {
        normal /= n_len;
    }
    Some(ContactManifold {
        center,
        normal,
        max_depth,
        count,
    })
}

/// Continuous collision detection (CCD) via sphere tracing along velocity.
///
/// Returns the time of impact (0..dt) and the contact point, or None if
/// no collision occurs within the time step.
pub fn sdf_ccd(
    sdf: &SdfNode,
    start: Vec3,
    velocity: Vec3,
    dt: f32,
    radius: f32,
) -> Option<(f32, Vec3)> {
    let speed = velocity.length();
    if speed < 1e-10 || dt < 1e-10 {
        return None;
    }
    let dir = velocity / speed;
    let max_dist = speed * dt;

    let mut t = 0.0;
    let max_steps = 128;
    for _ in 0..max_steps {
        if t >= max_dist {
            return None;
        }
        let pos = start + dir * t;
        let d = eval(sdf, pos) - radius;
        if d < 1e-5 {
            let toi = t / speed;
            return Some((toi.min(dt), pos));
        }
        // Advance by the safe distance
        t += d.max(1e-4);
    }
    None
}

/// Closest point on an SDF surface from a given query point.
///
/// Uses gradient descent to find the nearest surface point.
/// Returns (surface_point, distance).
pub fn sdf_closest_point(sdf: &SdfNode, query: Vec3, max_iter: u32) -> (Vec3, f32) {
    let mut p = query;
    for _ in 0..max_iter {
        let d = eval(sdf, p);
        if d.abs() < 1e-5 {
            return (p, 0.0);
        }
        let n = normal(sdf, p, 0.001);
        p -= n * d;
    }
    let final_d = eval(sdf, p);
    (p, final_d.abs())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_aabb() -> Aabb {
        Aabb {
            min: Vec3::splat(-3.0),
            max: Vec3::splat(3.0),
        }
    }

    #[test]
    fn test_sphere_sphere_overlap() {
        // Two unit spheres, centers 1.5 apart → overlap
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.5, 0.0, 0.0);

        assert!(sdf_overlap(&a, &b, &test_aabb(), 16));
    }

    #[test]
    fn test_sphere_sphere_no_overlap() {
        // Two unit spheres, centers 3.0 apart → no overlap
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0);

        assert!(!sdf_overlap(&a, &b, &test_aabb(), 16));
    }

    #[test]
    fn test_sphere_box_overlap() {
        // Sphere at origin, box overlapping
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(0.5, 0.5, 0.5).translate(0.8, 0.0, 0.0);

        assert!(sdf_overlap(&a, &b, &test_aabb(), 16));
    }

    #[test]
    fn test_sphere_box_no_overlap() {
        // Sphere at origin, box far away
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(0.5, 0.5, 0.5).translate(3.0, 3.0, 3.0);

        assert!(!sdf_overlap(&a, &b, &test_aabb(), 16));
    }

    #[test]
    fn test_collide_returns_contacts() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);

        let contacts = sdf_collide(&a, &b, &test_aabb(), 16);
        assert!(!contacts.is_empty(), "Should find contact points");

        // All contacts should have positive depth
        for c in &contacts {
            assert!(c.depth > 0.0, "depth should be positive: {}", c.depth);
            assert!(
                c.normal.length() > 0.5,
                "normal should be roughly unit length"
            );
        }
    }

    #[test]
    fn test_collide_deepest_first() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);

        let contacts = sdf_collide(&a, &b, &test_aabb(), 16);
        for w in contacts.windows(2) {
            assert!(
                w[0].depth >= w[1].depth,
                "Contacts should be sorted deepest first"
            );
        }
    }

    #[test]
    fn test_distance_overlapping() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);

        let dist = sdf_distance(&a, &b, &test_aabb(), 16);
        assert_eq!(dist, 0.0, "Overlapping SDFs should have distance 0");
    }

    #[test]
    fn test_distance_separated() {
        let a = SdfNode::sphere(0.5);
        let b = SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0);

        let dist = sdf_distance(&a, &b, &test_aabb(), 16);
        // True separation = 3.0 - 0.5 - 0.5 = 2.0
        // Grid-based is an upper bound, so should be close but >= 2.0
        assert!(dist > 1.5, "Separation should be > 1.5, got {}", dist);
        assert!(dist < 2.5, "Separation should be < 2.5, got {}", dist);
    }

    #[test]
    fn test_overlap_empty_no_contacts() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0);

        let contacts = sdf_collide(&a, &b, &test_aabb(), 16);
        assert!(
            contacts.is_empty(),
            "Non-overlapping should have no contacts"
        );
    }

    #[test]
    fn test_compute_manifold() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);
        let contacts = sdf_collide(&a, &b, &test_aabb(), 16);
        let manifold = compute_manifold(&contacts);
        assert!(manifold.is_some());
        let m = manifold.unwrap();
        assert!(m.max_depth > 0.0);
        assert!(m.count > 0);
        assert!(m.normal.length() > 0.5);
    }

    #[test]
    fn test_manifold_empty() {
        let contacts: Vec<SdfContact> = vec![];
        assert!(compute_manifold(&contacts).is_none());
    }

    #[test]
    fn test_ccd_hit() {
        // Thin wall at x=0 (extends 0.1 in x, large in y/z)
        let wall = SdfNode::box3d(0.1, 10.0, 10.0);
        // Start well outside at x=-5, moving toward wall
        let result = sdf_ccd(
            &wall,
            Vec3::new(-5.0, 0.0, 0.0),
            Vec3::new(20.0, 0.0, 0.0),
            1.0,
            0.0,
        );
        assert!(result.is_some(), "Should hit the wall");
        let (toi, _pos) = result.unwrap();
        assert!(toi > 0.0 && toi < 1.0, "toi={}", toi);
    }

    #[test]
    fn test_ccd_miss() {
        let wall = SdfNode::box3d(0.1, 10.0, 10.0).translate(5.0, 0.0, 0.0);
        // Moving away from wall
        let result = sdf_ccd(&wall, Vec3::ZERO, Vec3::new(-10.0, 0.0, 0.0), 1.0, 0.0);
        assert!(result.is_none(), "Should miss the wall");
    }

    #[test]
    fn test_closest_point_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let (p, d) = sdf_closest_point(&sphere, Vec3::new(3.0, 0.0, 0.0), 32);
        // Should converge near (1, 0, 0)
        assert!((p.x - 1.0).abs() < 0.05, "closest x={}", p.x);
        assert!(d < 0.05, "residual distance={}", d);
    }

    #[test]
    fn test_closest_point_inside() {
        let sphere = SdfNode::sphere(1.0);
        let (p, d) = sdf_closest_point(&sphere, Vec3::new(0.3, 0.0, 0.0), 32);
        assert!(d < 0.05, "Should find surface");
        assert!(
            p.length() > 0.9,
            "Should be near surface, len={}",
            p.length()
        );
    }

    #[test]
    fn test_contact_point_location() {
        // Two spheres overlapping along x-axis
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);

        let contacts = sdf_collide(&a, &b, &test_aabb(), 24);

        // Contact points should be near x=0.5 (midpoint of overlap region)
        if let Some(deepest) = contacts.first() {
            assert!(
                deepest.point.x > 0.0 && deepest.point.x < 1.0,
                "Deepest contact x should be between 0 and 1, got {}",
                deepest.point.x
            );
        }
    }
}

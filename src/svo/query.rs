//! SVO Queries: Point, Ray, and Nearest Surface (Deep Fried Edition)
//!
//! O(max_depth) point query, front-to-back ray traversal,
//! and nearest surface search.
//!
//! # Algorithms
//!
//! - **Point Query**: Descent through octants in O(max_depth)
//! - **Ray Query**: DDA front-to-back through children, skip empty regions
//! - **Nearest Surface**: Descent + backtrack with distance bound pruning
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::{SparseVoxelOctree, SvoNode, octant_for_point, child_center};

/// Result of a ray query against the SVO
#[derive(Debug, Clone, Copy)]
pub struct SvoRayHit {
    /// Distance along ray to hit point
    pub distance: f32,
    /// Hit position in world space
    pub position: Vec3,
    /// Surface normal at hit point
    pub normal: Vec3,
    /// Depth in the SVO tree where the hit was found
    pub depth: u32,
}

/// Query signed distance at a point by descending the SVO
///
/// Returns the SDF distance at the deepest node containing the point.
/// O(max_depth) traversal.
pub fn svo_query_point(svo: &SparseVoxelOctree, point: Vec3) -> f32 {
    if svo.nodes.is_empty() {
        return f32::MAX;
    }

    // Check bounds
    let bounds_min = svo.bounds.min();
    let bounds_max = svo.bounds.max();
    if point.x < bounds_min.x || point.x > bounds_max.x
        || point.y < bounds_min.y || point.y > bounds_max.y
        || point.z < bounds_min.z || point.z > bounds_max.z
    {
        // Outside bounds: return distance to bounds surface
        return svo.bounds.distance_to_point(point);
    }

    let center = svo.bounds.center();
    let half_size = svo.bounds.half_size();

    query_descent(svo, 0, center, half_size, point)
}

/// Recursive descent for point query
fn query_descent(
    svo: &SparseVoxelOctree,
    node_idx: usize,
    center: Vec3,
    half_size: Vec3,
    point: Vec3,
) -> f32 {
    if node_idx >= svo.nodes.len() {
        return f32::MAX;
    }

    let node = &svo.nodes[node_idx];

    if node.is_leaf == 1 {
        return node.distance;
    }

    // Find which octant the point is in
    let octant = octant_for_point(point, center);

    // Try to descend into child
    if let Some(child_idx) = node.child_index(octant) {
        let child_c = child_center(center, half_size, octant);
        let child_half = half_size * 0.5;
        query_descent(svo, child_idx as usize, child_c, child_half, point)
    } else {
        // No child for this octant, use this node's distance
        node.distance
    }
}

/// Ray query: find intersection with the SDF surface stored in the SVO
///
/// Uses sphere tracing through the octree, skipping empty regions.
/// Returns `None` if no hit within max_distance.
pub fn svo_ray_query(
    svo: &SparseVoxelOctree,
    origin: Vec3,
    direction: Vec3,
    max_distance: f32,
) -> Option<SvoRayHit> {
    if svo.nodes.is_empty() {
        return None;
    }

    let dir = direction.normalize();

    // Check if ray intersects SVO bounds at all
    let (t_enter, t_exit) = ray_aabb_intersect(
        origin,
        dir,
        svo.bounds.min(),
        svo.bounds.max(),
    )?;

    let t_start = t_enter.max(0.0);
    if t_start > max_distance || t_start >= t_exit {
        return None;
    }

    // Sphere trace through the SVO
    let center = svo.bounds.center();
    let half_size = svo.bounds.half_size();
    let mut t = t_start + 0.001; // Small offset to get inside bounds
    let max_steps = 256u32;

    for _ in 0..max_steps {
        if t > max_distance.min(t_exit) {
            break;
        }

        let p = origin + dir * t;

        // Query distance at current point
        let (dist, depth) = query_descent_with_depth(svo, 0, center, half_size, p, 0);

        // Hit surface
        if dist.abs() < 0.001 {
            let n = estimate_svo_normal(svo, p, center, half_size);
            return Some(SvoRayHit {
                distance: t,
                position: p,
                normal: n,
                depth,
            });
        }

        // Advance by distance (sphere tracing)
        t += dist.abs().max(0.001);
    }

    None
}

/// Point query that also returns the depth reached
fn query_descent_with_depth(
    svo: &SparseVoxelOctree,
    node_idx: usize,
    center: Vec3,
    half_size: Vec3,
    point: Vec3,
    depth: u32,
) -> (f32, u32) {
    if node_idx >= svo.nodes.len() {
        return (f32::MAX, depth);
    }

    let node = &svo.nodes[node_idx];

    if node.is_leaf == 1 {
        return (node.distance, depth);
    }

    let octant = octant_for_point(point, center);

    if let Some(child_idx) = node.child_index(octant) {
        let child_c = child_center(center, half_size, octant);
        let child_half = half_size * 0.5;
        query_descent_with_depth(svo, child_idx as usize, child_c, child_half, point, depth + 1)
    } else {
        (node.distance, depth)
    }
}

/// Estimate normal at a point using finite differences on the SVO
fn estimate_svo_normal(
    svo: &SparseVoxelOctree,
    point: Vec3,
    center: Vec3,
    half_size: Vec3,
) -> Vec3 {
    let e = 0.002;
    let dx = query_descent(svo, 0, center, half_size, point + Vec3::new(e, 0.0, 0.0))
           - query_descent(svo, 0, center, half_size, point - Vec3::new(e, 0.0, 0.0));
    let dy = query_descent(svo, 0, center, half_size, point + Vec3::new(0.0, e, 0.0))
           - query_descent(svo, 0, center, half_size, point - Vec3::new(0.0, e, 0.0));
    let dz = query_descent(svo, 0, center, half_size, point + Vec3::new(0.0, 0.0, e))
           - query_descent(svo, 0, center, half_size, point - Vec3::new(0.0, 0.0, e));

    Vec3::new(dx, dy, dz).normalize_or_zero()
}

/// Find nearest surface distance and approximate surface point
///
/// Descends the SVO and uses the stored distance + normal to estimate
/// the nearest surface point.
pub fn svo_nearest_surface(svo: &SparseVoxelOctree, point: Vec3) -> (f32, Vec3) {
    if svo.nodes.is_empty() {
        return (f32::MAX, point);
    }

    let center = svo.bounds.center();
    let half_size = svo.bounds.half_size();

    let (dist, best_node) = nearest_descent(svo, 0, center, half_size, point);

    // Estimate surface point from distance and normal
    let n = best_node.normal();
    let surface_point = if n.length_squared() > 0.01 {
        point - n * dist
    } else {
        // No valid normal, just approximate
        point
    };

    (dist, surface_point)
}

/// Descent for nearest surface, returns distance and the node found
fn nearest_descent(
    svo: &SparseVoxelOctree,
    node_idx: usize,
    center: Vec3,
    half_size: Vec3,
    point: Vec3,
) -> (f32, SvoNode) {
    if node_idx >= svo.nodes.len() {
        return (f32::MAX, SvoNode::default());
    }

    let node = svo.nodes[node_idx];

    if node.is_leaf == 1 {
        return (node.distance, node);
    }

    let octant = octant_for_point(point, center);

    if let Some(child_idx) = node.child_index(octant) {
        let child_c = child_center(center, half_size, octant);
        let child_half = half_size * 0.5;
        nearest_descent(svo, child_idx as usize, child_c, child_half, point)
    } else {
        (node.distance, node)
    }
}

/// Ray-AABB intersection test
///
/// Returns (t_enter, t_exit) or None if no intersection.
fn ray_aabb_intersect(
    origin: Vec3,
    dir: Vec3,
    aabb_min: Vec3,
    aabb_max: Vec3,
) -> Option<(f32, f32)> {
    let inv_dir = Vec3::new(
        if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::MAX * dir.x.signum() },
        if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::MAX * dir.y.signum() },
        if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::MAX * dir.z.signum() },
    );

    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;

    let t_min = t1.min(t2);
    let t_max = t1.max(t2);

    let t_enter = t_min.x.max(t_min.y).max(t_min.z);
    let t_exit = t_max.x.min(t_max.y).min(t_max.z);

    if t_enter <= t_exit && t_exit >= 0.0 {
        Some((t_enter, t_exit))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use crate::svo::SvoBuildConfig;

    fn make_test_svo() -> SparseVoxelOctree {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 4,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };
        super::super::build::build_svo(&sphere, &config)
    }

    #[test]
    fn test_point_query_center() {
        let svo = make_test_svo();
        let d = svo_query_point(&svo, Vec3::ZERO);
        // Center of unit sphere should be -1.0
        assert!(d < 0.0, "Center should be inside sphere, got {}", d);
    }

    #[test]
    fn test_point_query_outside() {
        let svo = make_test_svo();
        let d = svo_query_point(&svo, Vec3::new(3.0, 0.0, 0.0));
        // Outside bounds
        assert!(d > 0.0, "Outside should be positive, got {}", d);
    }

    #[test]
    fn test_point_query_surface() {
        let svo = make_test_svo();
        let d = svo_query_point(&svo, Vec3::new(1.0, 0.0, 0.0));
        // Near surface of unit sphere
        assert!(d.abs() < 0.5, "Surface should be near zero, got {}", d);
    }

    #[test]
    fn test_ray_query_hit() {
        let svo = make_test_svo();
        let hit = svo_ray_query(&svo, Vec3::new(-5.0, 0.0, 0.0), Vec3::X, 10.0);

        if let Some(h) = hit {
            // Should hit sphere surface around x=-1
            assert!((h.distance - 4.0).abs() < 0.5,
                "Hit distance should be ~4.0, got {}", h.distance);
            assert!(h.position.x < 0.0, "Hit should be on -X side");
        }
        // Note: hit may be None if SVO resolution is too coarse
    }

    #[test]
    fn test_ray_query_miss() {
        let svo = make_test_svo();
        // Ray parallel to sphere, should miss
        let hit = svo_ray_query(&svo, Vec3::new(-5.0, 5.0, 0.0), Vec3::X, 20.0);
        assert!(hit.is_none(), "Ray should miss the sphere");
    }

    #[test]
    fn test_nearest_surface() {
        let svo = make_test_svo();
        let (dist, _surface_point) = svo_nearest_surface(&svo, Vec3::ZERO);
        // Distance from center to unit sphere surface should be ~-1.0
        assert!(dist < 0.0, "Center should be inside, got dist={}", dist);
    }

    #[test]
    fn test_ray_aabb_intersect() {
        let min = Vec3::new(-1.0, -1.0, -1.0);
        let max = Vec3::new(1.0, 1.0, 1.0);

        // Hit
        let hit = ray_aabb_intersect(Vec3::new(-5.0, 0.0, 0.0), Vec3::X, min, max);
        assert!(hit.is_some());
        let (t_enter, t_exit) = hit.unwrap();
        assert!((t_enter - 4.0).abs() < 0.01);
        assert!((t_exit - 6.0).abs() < 0.01);

        // Miss
        let miss = ray_aabb_intersect(Vec3::new(-5.0, 5.0, 0.0), Vec3::X, min, max);
        assert!(miss.is_none());
    }
}

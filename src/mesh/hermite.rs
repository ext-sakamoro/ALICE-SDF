//! Hermite Data extraction for Dual Contouring
//!
//! Extracts position + normal data from mesh or SDF for high-quality
//! mesh reconstruction using Dual Contouring or similar algorithms.
//!
//! Author: Moroya Sakamoto

use glam::Vec3;
use rayon::prelude::*;

/// Hermite data point (position + normal)
#[derive(Debug, Clone, Copy)]
pub struct HermitePoint {
    /// Position on the surface
    pub position: Vec3,
    /// Surface normal at this position
    pub normal: Vec3,
}

impl HermitePoint {
    /// Create a new Hermite point
    #[inline]
    pub fn new(position: Vec3, normal: Vec3) -> Self {
        HermitePoint { position, normal }
    }
}

/// Edge crossing with Hermite data
#[derive(Debug, Clone, Copy)]
pub struct EdgeCrossing {
    /// Start of the edge
    pub start: Vec3,
    /// End of the edge
    pub end: Vec3,
    /// Intersection point
    pub intersection: Vec3,
    /// Surface normal at intersection
    pub normal: Vec3,
    /// Distance values at start and end
    pub start_dist: f32,
    pub end_dist: f32,
}

impl EdgeCrossing {
    /// Compute the parametric t value of the intersection
    #[inline]
    pub fn t(&self) -> f32 {
        if (self.end_dist - self.start_dist).abs() < 1e-10 {
            0.5
        } else {
            self.start_dist / (self.start_dist - self.end_dist)
        }
    }
}

/// Configuration for Hermite data extraction
#[derive(Debug, Clone)]
pub struct HermiteConfig {
    /// Grid resolution for sampling
    pub resolution: usize,
    /// Epsilon for gradient computation
    pub gradient_epsilon: f32,
    /// Number of refinement iterations for surface finding
    pub refinement_iterations: u32,
}

impl Default for HermiteConfig {
    fn default() -> Self {
        HermiteConfig {
            resolution: 32,
            gradient_epsilon: 0.001,
            refinement_iterations: 4,
        }
    }
}

/// Hermite data extractor
pub struct HermiteExtractor<'a, F>
where
    F: Fn(Vec3) -> f32 + Sync,
{
    sdf: &'a F,
    config: HermiteConfig,
    min_bounds: Vec3,
    max_bounds: Vec3,
}

impl<'a, F> HermiteExtractor<'a, F>
where
    F: Fn(Vec3) -> f32 + Sync,
{
    /// Create a new Hermite extractor
    pub fn new(sdf: &'a F, min_bounds: Vec3, max_bounds: Vec3, config: HermiteConfig) -> Self {
        HermiteExtractor {
            sdf,
            config,
            min_bounds,
            max_bounds,
        }
    }

    /// Compute gradient (surface normal) at a point
    #[inline]
    fn gradient(&self, point: Vec3) -> Vec3 {
        let eps = self.config.gradient_epsilon;
        let sdf = self.sdf;

        let dx = sdf(point + Vec3::X * eps) - sdf(point - Vec3::X * eps);
        let dy = sdf(point + Vec3::Y * eps) - sdf(point - Vec3::Y * eps);
        let dz = sdf(point + Vec3::Z * eps) - sdf(point - Vec3::Z * eps);

        Vec3::new(dx, dy, dz).normalize_or_zero()
    }

    /// Find surface intersection along an edge using bisection
    fn find_intersection(&self, start: Vec3, end: Vec3, start_dist: f32, end_dist: f32) -> Vec3 {
        let mut a = start;
        let mut b = end;
        let mut da = start_dist;
        let mut db = end_dist;

        for _ in 0..self.config.refinement_iterations {
            let t = da / (da - db);
            let mid = a.lerp(b, t);
            let dm = (self.sdf)(mid);

            if dm.abs() < 1e-6 {
                return mid;
            }

            if (da > 0.0) == (dm > 0.0) {
                a = mid;
                da = dm;
            } else {
                b = mid;
                db = dm;
            }
        }

        // Return best estimate
        let t = da / (da - db);
        a.lerp(b, t)
    }

    /// Extract Hermite data for edge crossings on the grid
    pub fn extract_edge_crossings(&self) -> Vec<EdgeCrossing> {
        let res = self.config.resolution;
        let cell_size = (self.max_bounds - self.min_bounds) / res as f32;

        // Sample SDF at grid vertices
        let num_vertices = (res + 1) * (res + 1) * (res + 1);
        let distances: Vec<f32> = (0..num_vertices)
            .into_par_iter()
            .map(|idx| {
                let z = idx / ((res + 1) * (res + 1));
                let y = (idx / (res + 1)) % (res + 1);
                let x = idx % (res + 1);

                let pos = self.min_bounds
                    + Vec3::new(x as f32, y as f32, z as f32) * cell_size;
                (self.sdf)(pos)
            })
            .collect();

        // Find edge crossings
        let get_idx = |x: usize, y: usize, z: usize| -> usize {
            z * (res + 1) * (res + 1) + y * (res + 1) + x
        };

        let get_pos = |x: usize, y: usize, z: usize| -> Vec3 {
            self.min_bounds + Vec3::new(x as f32, y as f32, z as f32) * cell_size
        };

        // Process all edges in parallel
        let edges: Vec<(usize, usize, usize, usize)> = (0..res)
            .flat_map(|z| {
                (0..res).flat_map(move |y| {
                    (0..res).flat_map(move |x| {
                        // X edges
                        let mut e = vec![(x, y, z, 0)];
                        // Y edges
                        e.push((x, y, z, 1));
                        // Z edges
                        e.push((x, y, z, 2));
                        e
                    })
                })
            })
            .collect();

        edges
            .into_par_iter()
            .filter_map(|(x, y, z, axis)| {
                let (start, end, start_idx, end_idx) = match axis {
                    0 => {
                        // X edge
                        if x >= res {
                            return None;
                        }
                        (
                            get_pos(x, y, z),
                            get_pos(x + 1, y, z),
                            get_idx(x, y, z),
                            get_idx(x + 1, y, z),
                        )
                    }
                    1 => {
                        // Y edge
                        if y >= res {
                            return None;
                        }
                        (
                            get_pos(x, y, z),
                            get_pos(x, y + 1, z),
                            get_idx(x, y, z),
                            get_idx(x, y + 1, z),
                        )
                    }
                    2 => {
                        // Z edge
                        if z >= res {
                            return None;
                        }
                        (
                            get_pos(x, y, z),
                            get_pos(x, y, z + 1),
                            get_idx(x, y, z),
                            get_idx(x, y, z + 1),
                        )
                    }
                    _ => return None,
                };

                let start_dist = distances[start_idx];
                let end_dist = distances[end_idx];

                // Check for sign change
                if (start_dist > 0.0) == (end_dist > 0.0) {
                    return None;
                }

                // Find intersection
                let intersection = self.find_intersection(start, end, start_dist, end_dist);
                let normal = self.gradient(intersection);

                Some(EdgeCrossing {
                    start,
                    end,
                    intersection,
                    normal,
                    start_dist,
                    end_dist,
                })
            })
            .collect()
    }

    /// Extract surface points using uniform grid sampling
    pub fn extract_surface_points(&self) -> Vec<HermitePoint> {
        self.extract_edge_crossings()
            .into_iter()
            .map(|crossing| HermitePoint::new(crossing.intersection, crossing.normal))
            .collect()
    }
}

/// Extract Hermite data from an SDF function
pub fn extract_hermite<F>(
    sdf: &F,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &HermiteConfig,
) -> Vec<HermitePoint>
where
    F: Fn(Vec3) -> f32 + Sync,
{
    let extractor = HermiteExtractor::new(sdf, min_bounds, max_bounds, config.clone());
    extractor.extract_surface_points()
}

/// Extract edge crossings from an SDF function
pub fn extract_edge_crossings<F>(
    sdf: &F,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &HermiteConfig,
) -> Vec<EdgeCrossing>
where
    F: Fn(Vec3) -> f32 + Sync,
{
    let extractor = HermiteExtractor::new(sdf, min_bounds, max_bounds, config.clone());
    extractor.extract_edge_crossings()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_point() {
        let p = HermitePoint::new(Vec3::ZERO, Vec3::Y);
        assert_eq!(p.position, Vec3::ZERO);
        assert_eq!(p.normal, Vec3::Y);
    }

    #[test]
    fn test_edge_crossing_t() {
        let crossing = EdgeCrossing {
            start: Vec3::ZERO,
            end: Vec3::X,
            intersection: Vec3::new(0.5, 0.0, 0.0),
            normal: Vec3::Y,
            start_dist: -0.5,
            end_dist: 0.5,
        };

        let t = crossing.t();
        assert!((t - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_extract_hermite_sphere() {
        let sphere = |p: Vec3| p.length() - 1.0;
        let config = HermiteConfig {
            resolution: 8,
            ..Default::default()
        };

        let points = extract_hermite(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &config,
        );

        // Should find surface points
        assert!(!points.is_empty());

        // All points should be near the sphere surface
        for point in &points {
            let dist = (point.position.length() - 1.0).abs();
            assert!(dist < 0.1, "Point at distance {} from surface", dist);

            // Normal should point outward (away from center)
            let expected_normal = point.position.normalize();
            let dot = point.normal.dot(expected_normal);
            assert!(dot > 0.9, "Normal alignment: {}", dot);
        }
    }

    #[test]
    fn test_extract_edge_crossings() {
        let sphere = |p: Vec3| p.length() - 1.0;
        let config = HermiteConfig {
            resolution: 8,
            ..Default::default()
        };

        let crossings = extract_edge_crossings(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &config,
        );

        assert!(!crossings.is_empty());

        // All crossings should have opposite signs at start/end
        for crossing in &crossings {
            assert!(
                (crossing.start_dist > 0.0) != (crossing.end_dist > 0.0),
                "Should have sign change"
            );
        }
    }
}

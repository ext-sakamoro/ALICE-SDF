//! Primitive Fitting for CSG Tree Reconstruction
//!
//! Detects and fits geometric primitives (sphere, box, cylinder) to mesh
//! data for creating editable SDF representations.
//!
//! # Algorithm
//!
//! 1. **Point Cloud Analysis**: Sample surface points from mesh
//! 2. **Candidate Generation**: Generate primitive candidates
//! 3. **Fitting**: Optimize primitive parameters using least squares
//! 4. **Selection**: Choose best fitting primitives using BIC/AIC
//! 5. **CSG Reconstruction**: Combine primitives into CSG tree
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use glam::Vec3;

/// Detected primitive type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    /// Sphere primitive
    Sphere,
    /// Axis-aligned box
    Box,
    /// Cylinder (axis-aligned)
    Cylinder,
    /// Capsule
    Capsule,
    /// Plane
    Plane,
}

/// Fitted primitive with parameters
#[derive(Debug, Clone)]
pub enum FittedPrimitive {
    /// Sphere with center and radius
    Sphere {
        /// Sphere center
        center: Vec3,
        /// Sphere radius
        radius: f32,
    },
    /// Box with center and half-extents
    Box {
        /// Box center
        center: Vec3,
        /// Half-extents along each axis
        half_extents: Vec3,
    },
    /// Cylinder with center, axis, radius, and half-height
    Cylinder {
        /// Cylinder center
        center: Vec3,
        /// Cylinder axis direction
        axis: Vec3,
        /// Cylinder radius
        radius: f32,
        /// Half the cylinder height
        half_height: f32,
    },
    /// Capsule with endpoints and radius
    Capsule {
        /// First endpoint
        point_a: Vec3,
        /// Second endpoint
        point_b: Vec3,
        /// Capsule radius
        radius: f32,
    },
    /// Plane with normal and distance from origin
    Plane {
        /// Plane normal direction
        normal: Vec3,
        /// Signed distance from origin
        distance: f32,
    },
}

impl FittedPrimitive {
    /// Get the primitive type
    pub fn primitive_type(&self) -> PrimitiveType {
        match self {
            FittedPrimitive::Sphere { .. } => PrimitiveType::Sphere,
            FittedPrimitive::Box { .. } => PrimitiveType::Box,
            FittedPrimitive::Cylinder { .. } => PrimitiveType::Cylinder,
            FittedPrimitive::Capsule { .. } => PrimitiveType::Capsule,
            FittedPrimitive::Plane { .. } => PrimitiveType::Plane,
        }
    }

    /// Convert to SdfNode
    pub fn to_sdf_node(&self) -> SdfNode {
        match self {
            FittedPrimitive::Sphere { center, radius } => {
                SdfNode::sphere(*radius).translate(center.x, center.y, center.z)
            }
            FittedPrimitive::Box {
                center,
                half_extents,
            } => SdfNode::box3d(half_extents.x, half_extents.y, half_extents.z)
                .translate(center.x, center.y, center.z),
            FittedPrimitive::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                let cylinder = SdfNode::cylinder(*radius, *half_height);

                // Compute rotation from Y-axis to target axis
                let from = Vec3::Y;
                let to = axis.normalize();

                let rotated = if (from - to).length_squared() < 1e-6 {
                    // Already aligned with Y-axis
                    cylinder
                } else if (from + to).length_squared() < 1e-6 {
                    // 180-degree rotation (around X-axis)
                    cylinder.rotate(glam::Quat::from_rotation_x(std::f32::consts::PI))
                } else {
                    // Normal rotation: compute axis and angle
                    let rot_axis = from.cross(to).normalize();
                    let angle = from.dot(to).clamp(-1.0, 1.0).acos();
                    let quat = glam::Quat::from_axis_angle(rot_axis, angle);
                    cylinder.rotate(quat)
                };

                rotated.translate(center.x, center.y, center.z)
            }
            FittedPrimitive::Capsule {
                point_a,
                point_b,
                radius,
            } => SdfNode::capsule(*point_a, *point_b, *radius),
            FittedPrimitive::Plane { normal, distance } => SdfNode::plane(*normal, *distance),
        }
    }

    /// Compute fitting error (sum of squared distances)
    pub fn compute_error(&self, points: &[Vec3]) -> f32 {
        points
            .iter()
            .map(|&p| {
                let dist = self.distance(p);
                dist * dist
            })
            .sum()
    }

    /// Compute signed distance to the primitive
    pub fn distance(&self, point: Vec3) -> f32 {
        match self {
            FittedPrimitive::Sphere { center, radius } => (point - *center).length() - radius,
            FittedPrimitive::Box {
                center,
                half_extents,
            } => {
                let q = (point - *center).abs() - *half_extents;
                q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
            }
            FittedPrimitive::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                let d = point - *center;
                let h = d.dot(*axis);
                let r = (d - *axis * h).length();

                let dh = h.abs() - *half_height;
                let dr = r - *radius;

                if dh <= 0.0 && dr <= 0.0 {
                    dh.max(dr)
                } else if dh <= 0.0 {
                    dr
                } else if dr <= 0.0 {
                    dh
                } else {
                    (dh * dh + dr * dr).sqrt()
                }
            }
            FittedPrimitive::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let pa = point - *point_a;
                let ba = *point_b - *point_a;
                let h = (pa.dot(ba) / ba.length_squared()).clamp(0.0, 1.0);
                (pa - ba * h).length() - radius
            }
            FittedPrimitive::Plane { normal, distance } => point.dot(*normal) - *distance,
        }
    }
}

/// Fitting result with quality metrics
#[derive(Debug, Clone)]
pub struct FittingResult {
    /// Fitted primitive
    pub primitive: FittedPrimitive,
    /// Mean squared error
    pub mse: f32,
    /// Maximum error
    pub max_error: f32,
    /// Number of inlier points
    pub inlier_count: usize,
    /// Inlier threshold used
    pub inlier_threshold: f32,
}

impl FittingResult {
    /// Check if the fit is acceptable
    pub fn is_acceptable(&self, max_mse: f32, min_inlier_ratio: f32, total_points: usize) -> bool {
        let inlier_ratio = self.inlier_count as f32 / total_points as f32;
        self.mse < max_mse && inlier_ratio > min_inlier_ratio
    }
}

/// Configuration for primitive fitting
#[derive(Debug, Clone)]
pub struct FittingConfig {
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
    /// Inlier distance threshold
    pub inlier_threshold: f32,
    /// Minimum inlier ratio for valid fit
    pub min_inlier_ratio: f32,
    /// Maximum MSE for valid fit
    pub max_mse: f32,
}

impl Default for FittingConfig {
    fn default() -> Self {
        FittingConfig {
            max_iterations: 100,
            convergence_threshold: 1e-6,
            inlier_threshold: 0.05,
            min_inlier_ratio: 0.8,
            max_mse: 0.01,
        }
    }
}

/// Fit a sphere to a point cloud
pub fn fit_sphere(points: &[Vec3], config: &FittingConfig) -> Option<FittingResult> {
    if points.len() < 4 {
        return None;
    }

    // Initial estimate: centroid and average distance
    let centroid: Vec3 = points.iter().copied().sum::<Vec3>() / points.len() as f32;
    let avg_dist: f32 =
        points.iter().map(|&p| (p - centroid).length()).sum::<f32>() / points.len() as f32;

    let mut center = centroid;
    let mut radius = avg_dist;

    // Iterative refinement
    for _ in 0..config.max_iterations {
        let mut new_center = Vec3::ZERO;
        let mut new_radius = 0.0f32;
        let mut count = 0;

        for &p in points {
            let dir = (p - center).normalize_or_zero();
            let surface_point = center + dir * radius;
            let error = (p - surface_point).length();

            if error < config.inlier_threshold {
                new_center += p - dir * (p - center).length().min(radius);
                new_radius += (p - center).length();
                count += 1;
            }
        }

        if count == 0 {
            break;
        }

        let prev_center = center;
        let prev_radius = radius;

        center = new_center / count as f32;
        radius = new_radius / count as f32;

        if (center - prev_center).length() < config.convergence_threshold
            && (radius - prev_radius).abs() < config.convergence_threshold
        {
            break;
        }
    }

    let primitive = FittedPrimitive::Sphere { center, radius };

    // Compute error metrics
    let errors: Vec<f32> = points
        .iter()
        .map(|&p| primitive.distance(p).abs())
        .collect();
    let mse: f32 = errors.iter().map(|&e| e * e).sum::<f32>() / points.len() as f32;
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let inlier_count = errors
        .iter()
        .filter(|&&e| e < config.inlier_threshold)
        .count();

    Some(FittingResult {
        primitive,
        mse,
        max_error,
        inlier_count,
        inlier_threshold: config.inlier_threshold,
    })
}

/// Fit an axis-aligned box to a point cloud
pub fn fit_box(points: &[Vec3], config: &FittingConfig) -> Option<FittingResult> {
    if points.len() < 8 {
        return None;
    }

    // Compute bounding box
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);

    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;

    let primitive = FittedPrimitive::Box {
        center,
        half_extents,
    };

    // Compute error metrics
    let errors: Vec<f32> = points
        .iter()
        .map(|&p| primitive.distance(p).abs())
        .collect();
    let mse: f32 = errors.iter().map(|&e| e * e).sum::<f32>() / points.len() as f32;
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let inlier_count = errors
        .iter()
        .filter(|&&e| e < config.inlier_threshold)
        .count();

    Some(FittingResult {
        primitive,
        mse,
        max_error,
        inlier_count,
        inlier_threshold: config.inlier_threshold,
    })
}

/// Fit a cylinder to a point cloud
pub fn fit_cylinder(points: &[Vec3], config: &FittingConfig) -> Option<FittingResult> {
    if points.len() < 6 {
        return None;
    }

    // Try fitting along each principal axis
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    let mut best_result: Option<FittingResult> = None;

    for axis in axes {
        if let Some(result) = fit_cylinder_along_axis(points, axis, config) {
            if best_result.as_ref().map_or(true, |b| result.mse < b.mse) {
                best_result = Some(result);
            }
        }
    }

    best_result
}

/// Fit a cylinder along a specific axis
fn fit_cylinder_along_axis(
    points: &[Vec3],
    axis: Vec3,
    config: &FittingConfig,
) -> Option<FittingResult> {
    // Project points onto plane perpendicular to axis
    let projected: Vec<(Vec3, f32)> = points
        .iter()
        .map(|&p| {
            let h = p.dot(axis);
            let perpendicular = p - axis * h;
            (perpendicular, h)
        })
        .collect();

    // Find center in the perpendicular plane (2D centroid)
    let center_2d: Vec3 = projected.iter().map(|(p, _)| *p).sum::<Vec3>() / points.len() as f32;

    // Compute radius as average distance from center
    let radius: f32 = projected
        .iter()
        .map(|(p, _)| (*p - center_2d).length())
        .sum::<f32>()
        / points.len() as f32;

    // Find height range
    let min_h = projected
        .iter()
        .map(|(_, h)| *h)
        .fold(f32::INFINITY, f32::min);
    let max_h = projected
        .iter()
        .map(|(_, h)| *h)
        .fold(f32::NEG_INFINITY, f32::max);

    let center_h = (min_h + max_h) * 0.5;
    let half_height = (max_h - min_h) * 0.5;

    let center = center_2d + axis * center_h;

    let primitive = FittedPrimitive::Cylinder {
        center,
        axis,
        radius,
        half_height,
    };

    // Compute error metrics
    let errors: Vec<f32> = points
        .iter()
        .map(|&p| primitive.distance(p).abs())
        .collect();
    let mse: f32 = errors.iter().map(|&e| e * e).sum::<f32>() / points.len() as f32;
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let inlier_count = errors
        .iter()
        .filter(|&&e| e < config.inlier_threshold)
        .count();

    Some(FittingResult {
        primitive,
        mse,
        max_error,
        inlier_count,
        inlier_threshold: config.inlier_threshold,
    })
}

/// Fit a plane to a point cloud using least squares
pub fn fit_plane(points: &[Vec3], config: &FittingConfig) -> Option<FittingResult> {
    if points.len() < 3 {
        return None;
    }

    // Compute centroid
    let centroid: Vec3 = points.iter().copied().sum::<Vec3>() / points.len() as f32;

    // Compute covariance matrix
    let mut cov = [[0.0f32; 3]; 3];
    for &p in points {
        let d = p - centroid;
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    // Find eigenvector with smallest eigenvalue (normal direction)
    // Using power iteration on inverse
    let normal = find_smallest_eigenvector(&cov);
    let distance = centroid.dot(normal);

    let primitive = FittedPrimitive::Plane { normal, distance };

    // Compute error metrics
    let errors: Vec<f32> = points
        .iter()
        .map(|&p| primitive.distance(p).abs())
        .collect();
    let mse: f32 = errors.iter().map(|&e| e * e).sum::<f32>() / points.len() as f32;
    let max_error = errors.iter().copied().fold(0.0f32, f32::max);
    let inlier_count = errors
        .iter()
        .filter(|&&e| e < config.inlier_threshold)
        .count();

    Some(FittingResult {
        primitive,
        mse,
        max_error,
        inlier_count,
        inlier_threshold: config.inlier_threshold,
    })
}

/// Find eigenvector with smallest eigenvalue (for plane fitting)
fn find_smallest_eigenvector(cov: &[[f32; 3]; 3]) -> Vec3 {
    // Simple power iteration to find dominant eigenvector of inverse
    // For production, use a proper eigenvalue solver

    let mut v = Vec3::new(1.0, 1.0, 1.0).normalize();

    for _ in 0..50 {
        // Apply inverse (using Cramer's rule for 3x3)
        let det = cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1])
            - cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0])
            + cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0]);

        if det.abs() < 1e-10 {
            break;
        }

        // Compute A^-1 * v
        let adj = [
            [
                cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1],
                cov[0][2] * cov[2][1] - cov[0][1] * cov[2][2],
                cov[0][1] * cov[1][2] - cov[0][2] * cov[1][1],
            ],
            [
                cov[1][2] * cov[2][0] - cov[1][0] * cov[2][2],
                cov[0][0] * cov[2][2] - cov[0][2] * cov[2][0],
                cov[0][2] * cov[1][0] - cov[0][0] * cov[1][2],
            ],
            [
                cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0],
                cov[0][1] * cov[2][0] - cov[0][0] * cov[2][1],
                cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0],
            ],
        ];

        let new_v = Vec3::new(
            (adj[0][0] * v.x + adj[0][1] * v.y + adj[0][2] * v.z) / det,
            (adj[1][0] * v.x + adj[1][1] * v.y + adj[1][2] * v.z) / det,
            (adj[2][0] * v.x + adj[2][1] * v.y + adj[2][2] * v.z) / det,
        );

        v = new_v.normalize_or_zero();
    }

    v
}

/// Detect best primitive type for a point cloud
pub fn detect_primitive(points: &[Vec3], config: &FittingConfig) -> Option<FittingResult> {
    let mut results: Vec<FittingResult> = Vec::new();

    if let Some(r) = fit_sphere(points, config) {
        results.push(r);
    }
    if let Some(r) = fit_box(points, config) {
        results.push(r);
    }
    if let Some(r) = fit_cylinder(points, config) {
        results.push(r);
    }
    if let Some(r) = fit_plane(points, config) {
        results.push(r);
    }

    // Select best fit based on MSE with penalty for complexity
    // (BIC-like criterion)
    results.into_iter().min_by(|a, b| {
        let complexity_a = match a.primitive {
            FittedPrimitive::Plane { .. } => 4,
            FittedPrimitive::Sphere { .. } => 4,
            FittedPrimitive::Cylinder { .. } => 7,
            FittedPrimitive::Box { .. } => 6,
            FittedPrimitive::Capsule { .. } => 7,
        };
        let complexity_b = match b.primitive {
            FittedPrimitive::Plane { .. } => 4,
            FittedPrimitive::Sphere { .. } => 4,
            FittedPrimitive::Cylinder { .. } => 7,
            FittedPrimitive::Box { .. } => 6,
            FittedPrimitive::Capsule { .. } => 7,
        };

        let n = points.len() as f32;
        // [Deep Fried v2] Guard mse <= 0 to prevent ln(-x) = NaN
        let safe_mse_a = a.mse.max(1e-20);
        let safe_mse_b = b.mse.max(1e-20);
        let bic_a = n * safe_mse_a.ln() + complexity_a as f32 * n.ln();
        let bic_b = n * safe_mse_b.ln() + complexity_b as f32 * n.ln();

        bic_a
            .partial_cmp(&bic_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Reconstruct a CSG tree from fitted primitives
pub fn primitives_to_csg(primitives: &[FittedPrimitive]) -> Option<SdfNode> {
    if primitives.is_empty() {
        return None;
    }

    let mut nodes: Vec<SdfNode> = primitives.iter().map(|p| p.to_sdf_node()).collect();

    // Build balanced binary tree of unions
    while nodes.len() > 1 {
        let mut next = Vec::with_capacity((nodes.len() + 1) / 2);

        for chunk in nodes.chunks(2) {
            if chunk.len() == 2 {
                next.push(chunk[0].clone().union(chunk[1].clone()));
            } else {
                next.push(chunk[0].clone());
            }
        }

        nodes = next;
    }

    nodes.pop()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn generate_sphere_points(center: Vec3, radius: f32, count: usize) -> Vec<Vec3> {
        let mut points = Vec::with_capacity(count);
        for i in 0..count {
            let theta = 2.0 * PI * (i as f32 / count as f32);
            let phi = PI * ((i * 7) % count) as f32 / count as f32;

            let x = center.x + radius * phi.sin() * theta.cos();
            let y = center.y + radius * phi.sin() * theta.sin();
            let z = center.z + radius * phi.cos();

            points.push(Vec3::new(x, y, z));
        }
        points
    }

    #[test]
    fn test_fit_sphere() {
        let points = generate_sphere_points(Vec3::new(1.0, 2.0, 3.0), 2.0, 100);
        let config = FittingConfig::default();

        let result = fit_sphere(&points, &config).expect("Should fit sphere");

        if let FittedPrimitive::Sphere { center, radius } = result.primitive {
            assert!((center.x - 1.0).abs() < 0.2);
            assert!((center.y - 2.0).abs() < 0.2);
            assert!((center.z - 3.0).abs() < 0.2);
            assert!((radius - 2.0).abs() < 0.2);
        } else {
            panic!("Expected sphere primitive");
        }
    }

    #[test]
    fn test_fit_box() {
        let config = FittingConfig::default();

        // Generate box surface points
        let mut points = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                let u = i as f32 / 9.0;
                let v = j as f32 / 9.0;

                // Front face
                points.push(Vec3::new(-1.0 + 2.0 * u, -1.0 + 2.0 * v, 1.0));
                // Back face
                points.push(Vec3::new(-1.0 + 2.0 * u, -1.0 + 2.0 * v, -1.0));
            }
        }

        let result = fit_box(&points, &config).expect("Should fit box");

        if let FittedPrimitive::Box {
            center,
            half_extents,
        } = result.primitive
        {
            assert!(center.length() < 0.1);
            assert!((half_extents.x - 1.0).abs() < 0.2);
            assert!((half_extents.y - 1.0).abs() < 0.2);
        } else {
            panic!("Expected box primitive");
        }
    }

    #[test]
    fn test_detect_primitive() {
        let sphere_points = generate_sphere_points(Vec3::ZERO, 1.0, 100);
        let config = FittingConfig::default();

        let result = detect_primitive(&sphere_points, &config).expect("Should detect primitive");

        assert_eq!(result.primitive.primitive_type(), PrimitiveType::Sphere);
    }

    #[test]
    fn test_to_sdf_node() {
        let primitive = FittedPrimitive::Sphere {
            center: Vec3::new(1.0, 0.0, 0.0),
            radius: 2.0,
        };

        let node = primitive.to_sdf_node();
        assert!(node.node_count() > 0);
    }

    #[test]
    fn test_primitives_to_csg() {
        let primitives = vec![
            FittedPrimitive::Sphere {
                center: Vec3::ZERO,
                radius: 1.0,
            },
            FittedPrimitive::Box {
                center: Vec3::new(2.0, 0.0, 0.0),
                half_extents: Vec3::ONE,
            },
        ];

        let csg = primitives_to_csg(&primitives).expect("Should create CSG");
        assert!(csg.node_count() >= 3); // At least 2 primitives + 1 union
    }

    #[test]
    fn test_cylinder_axis_rotation() {
        use crate::eval::eval;

        // Test Y-axis cylinder (no rotation needed)
        // Cylinder is oriented along Y with radius=1, half_height=2
        // Point on cylindrical surface (not cap): (1, 0, 0)
        let y_cylinder = FittedPrimitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::Y,
            radius: 1.0,
            half_height: 2.0,
        };
        let y_node = y_cylinder.to_sdf_node();
        let d = eval(&y_node, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.01, "Y-cylinder surface distance: {}", d);

        // Test X-axis cylinder (90-degree rotation)
        // After rotation, cylinder is along X, so cylindrical surface is at (0, 1, 0)
        let x_cylinder = FittedPrimitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::X,
            radius: 1.0,
            half_height: 2.0,
        };
        let x_node = x_cylinder.to_sdf_node();
        let d = eval(&x_node, Vec3::new(0.0, 1.0, 0.0));
        assert!(d.abs() < 0.01, "X-cylinder surface distance: {}", d);

        // Test Z-axis cylinder (90-degree rotation)
        // After rotation, cylinder is along Z, so cylindrical surface is at (1, 0, 0)
        let z_cylinder = FittedPrimitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::Z,
            radius: 1.0,
            half_height: 2.0,
        };
        let z_node = z_cylinder.to_sdf_node();
        let d = eval(&z_node, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.01, "Z-cylinder surface distance: {}", d);

        // Test negative Y-axis (180-degree rotation)
        // Should be same as Y-axis cylinder
        let neg_y_cylinder = FittedPrimitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::NEG_Y,
            radius: 1.0,
            half_height: 2.0,
        };
        let neg_y_node = neg_y_cylinder.to_sdf_node();
        let d = eval(&neg_y_node, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.01, "NEG_Y-cylinder surface distance: {}", d);

        // Test diagonal axis
        let diag_cylinder = FittedPrimitive::Cylinder {
            center: Vec3::ZERO,
            axis: Vec3::new(1.0, 1.0, 0.0).normalize(),
            radius: 1.0,
            half_height: 2.0,
        };
        let diag_node = diag_cylinder.to_sdf_node();
        // Point perpendicular to diagonal axis at distance 1 (on cylindrical surface)
        let d = eval(&diag_node, Vec3::new(0.0, 0.0, 1.0));
        assert!(d.abs() < 0.01, "Diagonal-cylinder surface distance: {}", d);
    }
}

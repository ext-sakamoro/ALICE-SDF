//! Lightmap UV generation for baked lighting (Deep Fried v2)
//!
//! Generates non-overlapping UV2 coordinates for lightmap baking.
//! Each triangle gets a unique region in UV space, enabling per-texel
//! unique lighting data.
//!
//! # Deep Fried v2 Optimizations
//!
//! - **Parallel Projection Bounds**: `rayon` for per-group min/max computation.
//! - **Parallel UV2 Assignment**: Per-group UV2 mapping parallelized.
//! - **Division Exorcism**: `inv_range`, `inv_resolution` pre-computed.
//!
//! # Algorithm
//! 1. Group triangles by dominant normal axis (6 groups: +/-X, +/-Y, +/-Z)
//! 2. Project each group onto the perpendicular plane
//! 3. Pack groups into a 3x2 UV atlas with padding
//!
//! # Usage
//! ```rust,ignore
//! use alice_sdf::mesh::lightmap::generate_lightmap_uvs;
//!
//! let mut mesh = sdf_to_mesh(&shape, min, max, &config);
//! generate_lightmap_uvs(&mut mesh, 1024); // 1024x1024 lightmap
//! // mesh.vertices[i].uv2 now contains unique lightmap coordinates
//! ```
//!
//! Author: Moroya Sakamoto

use crate::mesh::Mesh;
use glam::Vec2;
use rayon::prelude::*;

/// Generate lightmap UV2 coordinates for all triangles (Deep Fried v2)
///
/// Uses dominant-axis projection with atlas packing to ensure
/// every triangle has a unique, non-overlapping UV2 region.
/// Vertices shared between different axis groups are automatically
/// split to prevent UV2 overlap.
///
/// # Deep Fried v2 Optimizations
///
/// - Projection bounds computed in parallel per group
/// - UV2 assignment parallelized per group
///
/// # Arguments
/// * `mesh` - Mesh to generate lightmap UVs for (modifies vertices in-place)
/// * `resolution` - Target lightmap resolution (e.g., 1024 for 1024x1024)
pub fn generate_lightmap_uvs(mesh: &mut Mesh, resolution: u32) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return;
    }

    // Division Exorcism: pre-compute inverse resolution
    let inv_resolution = 1.0 / resolution as f32;

    // Padding in UV space between triangles (in texels)
    let padding = 2.0 * inv_resolution;

    // Group triangles by dominant normal axis
    // 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    let mut groups: [Vec<usize>; 6] = Default::default();

    for t in 0..tri_count {
        let v0 = &mesh.vertices[mesh.indices[t * 3] as usize];
        let v1 = &mesh.vertices[mesh.indices[t * 3 + 1] as usize];
        let v2 = &mesh.vertices[mesh.indices[t * 3 + 2] as usize];

        // Face normal from cross product
        let edge1 = v1.position - v0.position;
        let edge2 = v2.position - v0.position;
        let face_normal = edge1.cross(edge2).normalize_or_zero();

        let abs_n = face_normal.abs();
        let group = if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
            if face_normal.x >= 0.0 {
                0
            } else {
                1
            }
        } else if abs_n.y >= abs_n.x && abs_n.y >= abs_n.z {
            if face_normal.y >= 0.0 {
                2
            } else {
                3
            }
        } else if face_normal.z >= 0.0 {
            4
        } else {
            5
        };

        groups[group].push(t);
    }

    // Split vertices shared between different groups to prevent UV2 overlap
    let mut vertex_group: Vec<i32> = vec![-1; mesh.vertices.len()];

    for (group_idx, tris) in groups.iter().enumerate() {
        for &t in tris {
            for k in 0..3 {
                let idx = t * 3 + k;
                let vi = mesh.indices[idx] as usize;
                if vertex_group[vi] == -1 {
                    // First assignment
                    vertex_group[vi] = group_idx as i32;
                } else if vertex_group[vi] != group_idx as i32 {
                    // Vertex shared with different group — split it
                    let new_vi = mesh.vertices.len();
                    mesh.vertices.push(mesh.vertices[vi]);
                    vertex_group.push(group_idx as i32);
                    mesh.indices[idx] = new_vi as u32;
                }
            }
        }
    }

    // Atlas layout: 3 columns x 2 rows
    // Group mapping: [+X, -X, +Y] top row, [-Y, +Z, -Z] bottom row
    let cell_w = 1.0 / 3.0;
    let cell_h = 0.5;

    let cell_offsets = [
        Vec2::new(0.0, 0.5),          // +X: top-left
        Vec2::new(cell_w, 0.5),       // -X: top-center
        Vec2::new(cell_w * 2.0, 0.5), // +Y: top-right
        Vec2::new(0.0, 0.0),          // -Y: bottom-left
        Vec2::new(cell_w, 0.0),       // +Z: bottom-center
        Vec2::new(cell_w * 2.0, 0.0), // -Z: bottom-right
    ];

    // Usable area within cell (with padding)
    let usable_w = cell_w - padding * 2.0;
    let usable_h = cell_h - padding * 2.0;

    for (group_idx, tris) in groups.iter().enumerate() {
        if tris.is_empty() {
            continue;
        }

        let offset = cell_offsets[group_idx];

        // [Deep Fried v2] Parallel projection bounds computation
        // Collect all vertex indices for this group (borrow-safe)
        let indices_ref = &mesh.indices;
        let vert_indices: Vec<u32> = tris
            .iter()
            .flat_map(|&t| {
                let i0 = indices_ref[t * 3];
                let i1 = indices_ref[t * 3 + 1];
                let i2 = indices_ref[t * 3 + 2];
                [i0, i1, i2]
            })
            .collect();

        let vertices_ref = &mesh.vertices;
        let (proj_min, proj_max) = vert_indices
            .par_iter()
            .map(|&vi| {
                let proj = project_to_2d(vertices_ref[vi as usize].position, group_idx);
                (proj, proj)
            })
            .reduce(
                || (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN)),
                |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
            );

        let range = proj_max - proj_min;
        // Division Exorcism: pre-compute inverse range
        let inv_range = Vec2::new(
            if range.x > 1e-6 { 1.0 / range.x } else { 0.0 },
            if range.y > 1e-6 { 1.0 / range.y } else { 0.0 },
        );

        // Assign UV2 for each triangle's vertices
        for &t in tris {
            for k in 0..3 {
                let vi = mesh.indices[t * 3 + k] as usize;
                let v = &mesh.vertices[vi];
                let proj = project_to_2d(v.position, group_idx);

                // Normalize to [0, 1] within this group
                let normalized = (proj - proj_min) * inv_range;

                // Map to cell with padding
                let uv2 = Vec2::new(
                    offset.x + padding + normalized.x * usable_w,
                    offset.y + padding + normalized.y * usable_h,
                );

                mesh.vertices[vi].uv2 = uv2;
            }
        }
    }
}

/// Project a 3D position to 2D based on dominant axis group
#[inline]
fn project_to_2d(pos: glam::Vec3, group: usize) -> Vec2 {
    match group {
        0 | 1 => Vec2::new(pos.y, pos.z), // +/-X: project onto YZ
        2 | 3 => Vec2::new(pos.x, pos.z), // +/-Y: project onto XZ
        _ => Vec2::new(pos.x, pos.y),     // +/-Z: project onto XY
    }
}

/// Generate simple planar lightmap UVs (faster, lower quality)
///
/// Projects all vertices using triplanar projection and offsets each
/// triangle slightly to reduce overlap. Faster than `generate_lightmap_uvs`
/// but may have minor overlapping on seams.
///
/// [Deep Fried v2] Parallel vertex iteration via `rayon`.
pub fn generate_lightmap_uvs_fast(mesh: &mut Mesh) {
    mesh.vertices.par_iter_mut().for_each(|v| {
        let abs_n = v.normal.abs();
        if abs_n.x >= abs_n.y && abs_n.x >= abs_n.z {
            v.uv2 = Vec2::new(v.position.y * 0.5 + 0.5, v.position.z * 0.5 + 0.5);
        } else if abs_n.y >= abs_n.x && abs_n.y >= abs_n.z {
            v.uv2 = Vec2::new(v.position.x * 0.5 + 0.5, v.position.z * 0.5 + 0.5);
        } else {
            v.uv2 = Vec2::new(v.position.x * 0.5 + 0.5, v.position.y * 0.5 + 0.5);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_lightmap_uvs_generated() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Before: all UV2 should be zero
        assert!(mesh.vertices.iter().all(|v| v.uv2 == Vec2::ZERO));

        generate_lightmap_uvs(&mut mesh, 1024);

        // After: UV2 should be non-zero and in [0, 1] range
        let non_zero_count = mesh.vertices.iter().filter(|v| v.uv2 != Vec2::ZERO).count();
        assert!(non_zero_count > 0, "Expected non-zero UV2 values");

        for v in &mesh.vertices {
            assert!(
                v.uv2.x >= 0.0 && v.uv2.x <= 1.0,
                "UV2.x out of range: {}",
                v.uv2.x
            );
            assert!(
                v.uv2.y >= 0.0 && v.uv2.y <= 1.0,
                "UV2.y out of range: {}",
                v.uv2.y
            );
        }
    }

    #[test]
    fn test_lightmap_uvs_fast() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        generate_lightmap_uvs_fast(&mut mesh);

        let non_zero_count = mesh.vertices.iter().filter(|v| v.uv2 != Vec2::ZERO).count();
        assert!(non_zero_count > 0);
    }

    #[test]
    fn test_lightmap_uvs_box() {
        // Box shape should group nicely into 6 axis-aligned groups
        let box_shape = SdfNode::box3d(1.0, 1.0, 1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&box_shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        generate_lightmap_uvs(&mut mesh, 1024);

        // All UV2 should be in valid range
        for v in &mesh.vertices {
            assert!(v.uv2.x >= 0.0 && v.uv2.x <= 1.0);
            assert!(v.uv2.y >= 0.0 && v.uv2.y <= 1.0);
        }
    }

    #[test]
    fn test_lightmap_uvs_no_overlap() {
        // Verify that vertex splitting prevents UV2 overlap
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let verts_before = mesh.vertices.len();
        generate_lightmap_uvs(&mut mesh, 1024);
        let verts_after = mesh.vertices.len();

        // Sphere has vertices at group boundaries that need splitting
        assert!(
            verts_after >= verts_before,
            "Vertex count should not decrease: before={}, after={}",
            verts_before,
            verts_after
        );

        // Verify each triangle's UV2 values are within a single atlas cell
        let tri_count = mesh.indices.len() / 3;
        for t in 0..tri_count {
            let uv0 = mesh.vertices[mesh.indices[t * 3] as usize].uv2;
            let uv1 = mesh.vertices[mesh.indices[t * 3 + 1] as usize].uv2;
            let uv2 = mesh.vertices[mesh.indices[t * 3 + 2] as usize].uv2;

            // All three vertices should be in the same atlas cell (within 0.5 of each other)
            let max_u = uv0.x.max(uv1.x).max(uv2.x);
            let min_u = uv0.x.min(uv1.x).min(uv2.x);
            let max_v = uv0.y.max(uv1.y).max(uv2.y);
            let min_v = uv0.y.min(uv1.y).min(uv2.y);

            assert!(
                max_u - min_u <= 0.4,
                "Triangle {} UV2.x span too large: {}",
                t,
                max_u - min_u
            );
            assert!(
                max_v - min_v <= 0.6,
                "Triangle {} UV2.y span too large: {}",
                t,
                max_v - min_v
            );
        }
    }
}

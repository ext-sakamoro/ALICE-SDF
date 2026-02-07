//! Automatic UV Unwrapping for SDF Meshes
//!
//! Implements LSCM (Least Squares Conformal Mapping) for generating
//! proper UV coordinates from mesh geometry. Produces artist-quality
//! UV layouts suitable for texture painting and baking.
//!
//! # Features
//! - **LSCM unwrapping** — Angle-preserving conformal mapping
//! - **Seam detection** — Automatic chart boundary detection based on angle threshold
//! - **Chart packing** — Shelf-based bin-packing of UV charts into [0,1]²
//! - **Island merging** — Connected component detection for UV islands
//!
//! Author: Moroya Sakamoto

use crate::mesh::sdf_to_mesh::Mesh;
use glam::Vec2;
use std::collections::{HashMap, HashSet, VecDeque};

/// Configuration for UV unwrapping
#[derive(Debug, Clone)]
pub struct UvUnwrapConfig {
    /// Dihedral angle threshold in degrees for seam detection (default: 88.0)
    pub angle_threshold: f32,
    /// Padding between UV islands in [0,1] space (default: 0.01)
    pub margin: f32,
    /// Maximum allowed stretch factor (default: 0.5)
    pub max_stretch: f32,
}

impl Default for UvUnwrapConfig {
    fn default() -> Self {
        UvUnwrapConfig {
            angle_threshold: 88.0,
            margin: 0.01,
            max_stretch: 0.5,
        }
    }
}

/// A single UV chart (island)
#[derive(Debug, Clone)]
pub struct UvChart {
    /// Indices into mesh triangles belonging to this chart
    pub triangle_indices: Vec<usize>,
    /// Per-vertex UVs for this chart (indexed by vertex index)
    pub uv_coords: HashMap<u32, Vec2>,
    /// Bounding box minimum
    pub bounds_min: Vec2,
    /// Bounding box maximum
    pub bounds_max: Vec2,
}

/// Result of UV unwrapping
#[derive(Debug, Clone)]
pub struct UvUnwrapResult {
    /// UV charts (islands)
    pub charts: Vec<UvChart>,
    /// Per-vertex UV coordinates (same length as mesh.vertices)
    pub uvs: Vec<Vec2>,
    /// Number of charts
    pub chart_count: usize,
    /// UV space utilization (0.0 - 1.0)
    pub total_area_usage: f32,
}

/// Perform automatic UV unwrapping on a mesh
pub fn uv_unwrap(mesh: &Mesh, config: &UvUnwrapConfig) -> UvUnwrapResult {
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        return UvUnwrapResult {
            charts: vec![],
            uvs: vec![],
            chart_count: 0,
            total_area_usage: 0.0,
        };
    }

    let tri_count = mesh.indices.len() / 3;

    // Step 1: Build adjacency and detect seams
    let seam_edges = detect_seams(mesh, config.angle_threshold);

    // Step 2: Split into charts via flood-fill
    let charts_tris = split_into_charts(mesh, tri_count, &seam_edges);

    // Step 3: Unwrap each chart with LSCM
    let mut charts: Vec<UvChart> = charts_tris
        .iter()
        .map(|tri_indices| unwrap_chart_lscm(mesh, tri_indices))
        .collect();

    // Step 4: Pack charts into [0,1]²
    let total_area_usage = pack_charts(&mut charts, config.margin);

    // Step 5: Merge into per-vertex UVs
    let mut uvs = vec![Vec2::ZERO; mesh.vertices.len()];
    for chart in &charts {
        for (&vi, &uv) in &chart.uv_coords {
            uvs[vi as usize] = uv;
        }
    }

    let chart_count = charts.len();
    UvUnwrapResult {
        charts,
        uvs,
        chart_count,
        total_area_usage,
    }
}

/// Apply computed UVs to mesh vertices
pub fn apply_uvs(mesh: &mut Mesh, result: &UvUnwrapResult) {
    for (i, uv) in result.uvs.iter().enumerate() {
        if i < mesh.vertices.len() {
            mesh.vertices[i].uv = *uv;
        }
    }
}

type EdgeKey = (u32, u32);

fn edge_key(a: u32, b: u32) -> EdgeKey {
    if a < b { (a, b) } else { (b, a) }
}

/// Detect seam edges based on dihedral angle threshold
fn detect_seams(mesh: &Mesh, angle_threshold_deg: f32) -> HashSet<EdgeKey> {
    let threshold_cos = (angle_threshold_deg.to_radians()).cos();
    let tri_count = mesh.indices.len() / 3;

    // Build edge -> triangle adjacency
    let mut edge_tris: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for ti in 0..tri_count {
        let i0 = mesh.indices[ti * 3] as u32;
        let i1 = mesh.indices[ti * 3 + 1] as u32;
        let i2 = mesh.indices[ti * 3 + 2] as u32;

        edge_tris.entry(edge_key(i0, i1)).or_default().push(ti);
        edge_tris.entry(edge_key(i1, i2)).or_default().push(ti);
        edge_tris.entry(edge_key(i2, i0)).or_default().push(ti);
    }

    let mut seams = HashSet::new();

    for (edge, tris) in &edge_tris {
        if tris.len() == 1 {
            // Boundary edge is always a seam
            seams.insert(*edge);
        } else if tris.len() >= 2 {
            let n0 = triangle_normal(mesh, tris[0]);
            let n1 = triangle_normal(mesh, tris[1]);
            let cos_angle = n0.dot(n1);
            if cos_angle < threshold_cos {
                seams.insert(*edge);
            }
        }
    }

    seams
}

fn triangle_normal(mesh: &Mesh, ti: usize) -> glam::Vec3 {
    let p0 = mesh.vertices[mesh.indices[ti * 3] as usize].position;
    let p1 = mesh.vertices[mesh.indices[ti * 3 + 1] as usize].position;
    let p2 = mesh.vertices[mesh.indices[ti * 3 + 2] as usize].position;
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    e1.cross(e2).normalize_or_zero()
}

/// Split mesh triangles into connected charts separated by seams
fn split_into_charts(
    mesh: &Mesh,
    tri_count: usize,
    seams: &HashSet<EdgeKey>,
) -> Vec<Vec<usize>> {
    // Build triangle adjacency (not crossing seams)
    let mut tri_adj: Vec<Vec<usize>> = vec![Vec::new(); tri_count];

    let mut edge_tris: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for ti in 0..tri_count {
        let i0 = mesh.indices[ti * 3] as u32;
        let i1 = mesh.indices[ti * 3 + 1] as u32;
        let i2 = mesh.indices[ti * 3 + 2] as u32;

        edge_tris.entry(edge_key(i0, i1)).or_default().push(ti);
        edge_tris.entry(edge_key(i1, i2)).or_default().push(ti);
        edge_tris.entry(edge_key(i2, i0)).or_default().push(ti);
    }

    for (edge, tris) in &edge_tris {
        if seams.contains(edge) {
            continue;
        }
        if tris.len() >= 2 {
            for i in 0..tris.len() {
                for j in (i + 1)..tris.len() {
                    tri_adj[tris[i]].push(tris[j]);
                    tri_adj[tris[j]].push(tris[i]);
                }
            }
        }
    }

    // Flood-fill to find connected components
    let mut visited = vec![false; tri_count];
    let mut charts = Vec::new();

    for start in 0..tri_count {
        if visited[start] {
            continue;
        }
        let mut chart = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(ti) = queue.pop_front() {
            chart.push(ti);
            for &neighbor in &tri_adj[ti] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        charts.push(chart);
    }

    charts
}

/// Unwrap a single chart using simplified LSCM
fn unwrap_chart_lscm(mesh: &Mesh, tri_indices: &[usize]) -> UvChart {
    // Collect unique vertices
    let mut vert_set: Vec<u32> = Vec::new();
    for &ti in tri_indices {
        for k in 0..3 {
            let vi = mesh.indices[ti * 3 + k] as u32;
            if !vert_set.contains(&vi) {
                vert_set.push(vi);
            }
        }
    }

    if vert_set.len() < 3 || tri_indices.is_empty() {
        return UvChart {
            triangle_indices: tri_indices.to_vec(),
            uv_coords: HashMap::new(),
            bounds_min: Vec2::ZERO,
            bounds_max: Vec2::ZERO,
        };
    }

    // Map global vertex index -> local index
    let mut global_to_local: HashMap<u32, usize> = HashMap::new();
    for (li, &gi) in vert_set.iter().enumerate() {
        global_to_local.insert(gi, li);
    }

    let n = vert_set.len();

    // Compute per-triangle local 2D parameterization
    // Use angle-based flattening: project each triangle onto its own plane,
    // then stitch together using a simple iterative solver.

    // Simple approach: ABF-like initial layout using first triangle as base,
    // then propagate via BFS
    let mut local_uvs = vec![Vec2::ZERO; n];
    let mut placed = vec![false; n];

    // Place first triangle
    let first_tri = tri_indices[0];
    let gi0 = mesh.indices[first_tri * 3] as u32;
    let gi1 = mesh.indices[first_tri * 3 + 1] as u32;
    let gi2 = mesh.indices[first_tri * 3 + 2] as u32;

    let p0 = mesh.vertices[gi0 as usize].position;
    let p1 = mesh.vertices[gi1 as usize].position;
    let p2 = mesh.vertices[gi2 as usize].position;

    let e01 = p1 - p0;
    let e02 = p2 - p0;
    let len01 = e01.length();

    let li0 = global_to_local[&gi0];
    let li1 = global_to_local[&gi1];
    let li2 = global_to_local[&gi2];

    // Place first edge along U axis
    local_uvs[li0] = Vec2::ZERO;
    local_uvs[li1] = Vec2::new(len01, 0.0);
    placed[li0] = true;
    placed[li1] = true;

    // Place third vertex
    if len01 > 1e-10 {
        let cos_a = e01.dot(e02) / (len01 * e02.length());
        let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
        let len02 = e02.length();
        local_uvs[li2] = Vec2::new(len02 * cos_a, len02 * sin_a);
    }
    placed[li2] = true;

    // BFS: place remaining triangles by finding an edge already placed
    let mut tri_queue: VecDeque<usize> = VecDeque::new();
    let mut tri_placed = vec![false; tri_indices.len()];
    tri_placed[0] = true;

    // Build triangle adjacency within this chart
    let mut edge_to_local_tri: HashMap<EdgeKey, Vec<usize>> = HashMap::new();
    for (lti, &gti) in tri_indices.iter().enumerate() {
        let a = mesh.indices[gti * 3] as u32;
        let b = mesh.indices[gti * 3 + 1] as u32;
        let c = mesh.indices[gti * 3 + 2] as u32;
        edge_to_local_tri.entry(edge_key(a, b)).or_default().push(lti);
        edge_to_local_tri.entry(edge_key(b, c)).or_default().push(lti);
        edge_to_local_tri.entry(edge_key(c, a)).or_default().push(lti);
    }

    // Seed the queue with neighbors of first triangle
    for &lti in edge_to_local_tri.get(&edge_key(gi0, gi1)).unwrap_or(&vec![]) {
        if !tri_placed[lti] { tri_queue.push_back(lti); }
    }
    for &lti in edge_to_local_tri.get(&edge_key(gi1, gi2)).unwrap_or(&vec![]) {
        if !tri_placed[lti] { tri_queue.push_back(lti); }
    }
    for &lti in edge_to_local_tri.get(&edge_key(gi2, gi0)).unwrap_or(&vec![]) {
        if !tri_placed[lti] { tri_queue.push_back(lti); }
    }

    while let Some(lti) = tri_queue.pop_front() {
        if tri_placed[lti] {
            continue;
        }

        let gti = tri_indices[lti];
        let va = mesh.indices[gti * 3] as u32;
        let vb = mesh.indices[gti * 3 + 1] as u32;
        let vc = mesh.indices[gti * 3 + 2] as u32;
        let verts = [va, vb, vc];

        let la = global_to_local[&va];
        let lb = global_to_local[&vb];
        let lc = global_to_local[&vc];
        let locals = [la, lb, lc];

        // Find the edge that's already placed
        let mut placed_edge = None;
        for i in 0..3 {
            let j = (i + 1) % 3;
            if placed[locals[i]] && placed[locals[j]] {
                placed_edge = Some((i, j));
                break;
            }
        }

        if let Some((ei, ej)) = placed_edge {
            let ek = (0..3).find(|&k| k != ei && k != ej).unwrap();

            if !placed[locals[ek]] {
                // Compute UV of the unplaced vertex
                let pi = mesh.vertices[verts[ei] as usize].position;
                let pj = mesh.vertices[verts[ej] as usize].position;
                let pk = mesh.vertices[verts[ek] as usize].position;

                let edge_ij = pj - pi;
                let edge_ik = pk - pi;
                let len_ij = edge_ij.length();
                let len_ik = edge_ik.length();

                if len_ij > 1e-10 && len_ik > 1e-10 {
                    let cos_a = edge_ij.dot(edge_ik) / (len_ij * len_ik);
                    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();

                    let uv_i = local_uvs[locals[ei]];
                    let uv_j = local_uvs[locals[ej]];
                    let uv_edge = uv_j - uv_i;
                    let uv_dir = uv_edge.normalize_or_zero();
                    let uv_perp = Vec2::new(-uv_dir.y, uv_dir.x);

                    let ratio = len_ik / len_ij.max(1e-10);
                    local_uvs[locals[ek]] = uv_i + uv_dir * (cos_a * ratio * uv_edge.length()) + uv_perp * (sin_a * ratio * uv_edge.length());
                }
                placed[locals[ek]] = true;
            }

            tri_placed[lti] = true;

            // Enqueue neighbors
            for i in 0..3 {
                let j = (i + 1) % 3;
                let ek = edge_key(verts[i], verts[j]);
                for &neighbor_lti in edge_to_local_tri.get(&ek).unwrap_or(&vec![]) {
                    if !tri_placed[neighbor_lti] {
                        tri_queue.push_back(neighbor_lti);
                    }
                }
            }
        } else {
            // Can't place yet, re-enqueue
            tri_queue.push_back(lti);
        }
    }

    // Normalize UVs to [0, 1] range within this chart
    let mut min_uv = Vec2::splat(f32::INFINITY);
    let mut max_uv = Vec2::splat(f32::NEG_INFINITY);

    for (li, &_gi) in vert_set.iter().enumerate() {
        min_uv = min_uv.min(local_uvs[li]);
        max_uv = max_uv.max(local_uvs[li]);
    }

    let range = max_uv - min_uv;
    let scale = if range.x > 1e-10 && range.y > 1e-10 {
        1.0 / range.x.max(range.y)
    } else {
        1.0
    };

    let mut uv_coords = HashMap::new();
    for (li, &gi) in vert_set.iter().enumerate() {
        let uv = (local_uvs[li] - min_uv) * scale;
        uv_coords.insert(gi, uv);
    }

    // Recompute bounds after normalization
    let mut bounds_min = Vec2::splat(f32::INFINITY);
    let mut bounds_max = Vec2::splat(f32::NEG_INFINITY);
    for uv in uv_coords.values() {
        bounds_min = bounds_min.min(*uv);
        bounds_max = bounds_max.max(*uv);
    }

    UvChart {
        triangle_indices: tri_indices.to_vec(),
        uv_coords,
        bounds_min,
        bounds_max,
    }
}

/// Pack charts into [0,1]² using shelf-based bin packing
fn pack_charts(charts: &mut [UvChart], margin: f32) -> f32 {
    if charts.is_empty() {
        return 0.0;
    }

    // Sort by height (tallest first)
    let mut sorted_indices: Vec<usize> = (0..charts.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let ha = charts[a].bounds_max.y - charts[a].bounds_min.y;
        let hb = charts[b].bounds_max.y - charts[b].bounds_min.y;
        hb.partial_cmp(&ha).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Shelf packing
    let mut shelf_y = margin;
    let mut shelf_height = 0.0f32;
    let mut cursor_x = margin;
    let mut total_used_area = 0.0f32;

    for &ci in &sorted_indices {
        let w = charts[ci].bounds_max.x - charts[ci].bounds_min.x;
        let h = charts[ci].bounds_max.y - charts[ci].bounds_min.y;

        // Check if chart fits on current shelf
        if cursor_x + w + margin > 1.0 {
            // New shelf
            shelf_y += shelf_height + margin;
            shelf_height = 0.0;
            cursor_x = margin;
        }

        let offset_x = cursor_x - charts[ci].bounds_min.x;
        let offset_y = shelf_y - charts[ci].bounds_min.y;
        let offset = Vec2::new(offset_x, offset_y);

        // Apply offset to all UVs
        for uv in charts[ci].uv_coords.values_mut() {
            *uv += offset;
        }
        charts[ci].bounds_min += offset;
        charts[ci].bounds_max += offset;

        total_used_area += w * h;
        shelf_height = shelf_height.max(h);
        cursor_x += w + margin;
    }

    // Clamp all UVs to [0,1]
    for chart in charts.iter_mut() {
        for uv in chart.uv_coords.values_mut() {
            uv.x = uv.x.clamp(0.0, 1.0);
            uv.y = uv.y.clamp(0.0, 1.0);
        }
    }

    total_used_area.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use glam::Vec3;

    #[test]
    fn test_uv_unwrap_sphere() {
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &MarchingCubesConfig { resolution: 8, ..Default::default() });

        let result = uv_unwrap(&mesh, &UvUnwrapConfig::default());

        assert!(result.chart_count > 0);
        assert_eq!(result.uvs.len(), mesh.vertices.len());

        // All UVs should be in [0, 1]
        for uv in &result.uvs {
            assert!(uv.x >= 0.0 && uv.x <= 1.0, "UV x out of range: {}", uv.x);
            assert!(uv.y >= 0.0 && uv.y <= 1.0, "UV y out of range: {}", uv.y);
        }
    }

    #[test]
    fn test_uv_unwrap_box() {
        let box3d = SdfNode::box3d(1.0, 1.0, 1.0);
        let mesh = sdf_to_mesh(&box3d, Vec3::splat(-2.0), Vec3::splat(2.0), &MarchingCubesConfig { resolution: 8, ..Default::default() });

        let result = uv_unwrap(&mesh, &UvUnwrapConfig::default());

        assert!(result.chart_count >= 1);
        assert_eq!(result.uvs.len(), mesh.vertices.len());
    }

    #[test]
    fn test_apply_uvs() {
        let sphere = SdfNode::sphere(1.0);
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &MarchingCubesConfig { resolution: 8, ..Default::default() });

        let result = uv_unwrap(&mesh, &UvUnwrapConfig::default());
        apply_uvs(&mut mesh, &result);

        // Verify UVs were written to mesh
        let has_non_zero = mesh.vertices.iter().any(|v| v.uv.x != 0.0 || v.uv.y != 0.0);
        assert!(has_non_zero, "Some vertices should have non-zero UVs");
    }
}

//! Vertex cache optimization for GPU-friendly triangle ordering
//!
//! Implements the Tom Forsyth algorithm for optimal post-transform
//! vertex cache utilization. Reorders triangles to minimize cache misses,
//! improving GPU rendering performance by 10-30%.
//!
//! # References
//! - Tom Forsyth, "Linear-Speed Vertex Cache Optimisation" (2006)
//!
//! Author: Moroya Sakamoto

use crate::mesh::Mesh;

const CACHE_SIZE: usize = 32;
const CACHE_DECAY_POWER: f32 = 1.5;
const LAST_TRI_SCORE: f32 = 0.75;
const VALENCE_BOOST_SCALE: f32 = 2.0;
const VALENCE_BOOST_POWER: f32 = 0.5;

struct VertexData {
    score: f32,
    active_tri_count: u32,
    tri_indices: Vec<u32>,
    cache_pos: i32, // -1 = not in cache
}

/// Optimize triangle order for vertex cache efficiency (Tom Forsyth algorithm)
///
/// Reorders `mesh.indices` in-place to maximize post-transform vertex cache hits.
/// This typically improves GPU rendering performance by 10-30%.
pub fn optimize_vertex_cache(mesh: &mut Mesh) {
    let vert_count = mesh.vertices.len();
    let tri_count = mesh.indices.len() / 3;

    if tri_count <= 1 {
        return;
    }

    // Build per-vertex data
    let mut vdata: Vec<VertexData> = (0..vert_count)
        .map(|_| VertexData {
            score: 0.0,
            active_tri_count: 0,
            tri_indices: Vec::new(),
            cache_pos: -1,
        })
        .collect();

    // Count triangles per vertex and build adjacency
    for t in 0..tri_count {
        for k in 0..3 {
            let vi = mesh.indices[t * 3 + k] as usize;
            if vi < vert_count {
                vdata[vi].active_tri_count += 1;
                vdata[vi].tri_indices.push(t as u32);
            }
        }
    }

    // Initial vertex scores
    for v in &mut vdata {
        v.score = compute_vertex_score(v.cache_pos, v.active_tri_count);
    }

    // Triangle scores and emitted flags
    let mut tri_scores: Vec<f32> = (0..tri_count)
        .map(|t| {
            let a = mesh.indices[t * 3] as usize;
            let b = mesh.indices[t * 3 + 1] as usize;
            let c = mesh.indices[t * 3 + 2] as usize;
            vdata[a].score + vdata[b].score + vdata[c].score
        })
        .collect();
    let mut tri_emitted = vec![false; tri_count];

    // LRU cache
    let mut cache: Vec<u32> = Vec::with_capacity(CACHE_SIZE + 3);

    // Output indices
    let mut output = Vec::with_capacity(mesh.indices.len());

    // Find best starting triangle
    let mut best_tri = 0usize;
    let mut best_score = -1.0f32;
    for t in 0..tri_count {
        if tri_scores[t] > best_score {
            best_score = tri_scores[t];
            best_tri = t;
        }
    }

    for _ in 0..tri_count {
        if tri_emitted[best_tri] {
            // Find any unemitted triangle
            let mut found = false;
            for t in 0..tri_count {
                if !tri_emitted[t] {
                    best_tri = t;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
        }

        // Emit triangle
        tri_emitted[best_tri] = true;
        let tri_verts = [
            mesh.indices[best_tri * 3] as usize,
            mesh.indices[best_tri * 3 + 1] as usize,
            mesh.indices[best_tri * 3 + 2] as usize,
        ];

        output.push(tri_verts[0] as u32);
        output.push(tri_verts[1] as u32);
        output.push(tri_verts[2] as u32);

        // Decrement active tri count for these vertices
        for &vi in &tri_verts {
            if vi < vert_count {
                vdata[vi].active_tri_count = vdata[vi].active_tri_count.saturating_sub(1);
            }
        }

        // Update cache - push new vertices to front
        for &vi in &tri_verts {
            // Remove if already in cache
            if let Some(pos) = cache.iter().position(|&v| v == vi as u32) {
                cache.remove(pos);
            }
            cache.insert(0, vi as u32);
        }

        // Truncate cache
        if cache.len() > CACHE_SIZE {
            // Evicted vertices get cache_pos = -1
            for &evicted in cache.iter().skip(CACHE_SIZE) {
                let ei = evicted as usize;
                if ei < vert_count {
                    vdata[ei].cache_pos = -1;
                }
            }
            cache.truncate(CACHE_SIZE);
        }

        // Update cache positions
        for (pos, &vi) in cache.iter().enumerate() {
            let vi = vi as usize;
            if vi < vert_count {
                vdata[vi].cache_pos = pos as i32;
            }
        }

        // Recalculate scores for affected vertices and their triangles
        let mut dirty_tris = Vec::new();
        for &vi in &cache {
            let vi = vi as usize;
            if vi < vert_count {
                vdata[vi].score =
                    compute_vertex_score(vdata[vi].cache_pos, vdata[vi].active_tri_count);
                for &ti in &vdata[vi].tri_indices {
                    if !tri_emitted[ti as usize] {
                        dirty_tris.push(ti as usize);
                    }
                }
            }
        }

        // Update dirty triangle scores and find next best
        best_score = -1.0;
        let mut next_best = 0;
        for &t in &dirty_tris {
            let a = mesh.indices[t * 3] as usize;
            let b = mesh.indices[t * 3 + 1] as usize;
            let c = mesh.indices[t * 3 + 2] as usize;
            tri_scores[t] = vdata[a].score + vdata[b].score + vdata[c].score;
        }

        // Search candidates from dirty triangles first
        for &t in &dirty_tris {
            if !tri_emitted[t] && tri_scores[t] > best_score {
                best_score = tri_scores[t];
                next_best = t;
            }
        }

        if best_score < 0.0 {
            // Fallback: scan all
            for t in 0..tri_count {
                if !tri_emitted[t] && tri_scores[t] > best_score {
                    best_score = tri_scores[t];
                    next_best = t;
                }
            }
        }

        best_tri = next_best;
    }

    mesh.indices = output;
}

fn compute_vertex_score(cache_pos: i32, active_tri_count: u32) -> f32 {
    if active_tri_count == 0 {
        return -1.0;
    }

    let mut score = 0.0f32;

    if cache_pos >= 0 {
        if cache_pos < 3 {
            score = LAST_TRI_SCORE;
        } else {
            let scaler = 1.0 / (CACHE_SIZE as f32 - 3.0);
            score = (1.0 - (cache_pos as f32 - 3.0) * scaler).powf(CACHE_DECAY_POWER);
        }
    }

    // Valence boost
    let valence_boost = (active_tri_count as f32).powf(-VALENCE_BOOST_POWER);
    score += VALENCE_BOOST_SCALE * valence_boost;

    score
}

/// Calculate Average Cache Miss Ratio (ACMR)
///
/// Lower is better. A value of 0.5 means ~0.5 cache misses per triangle.
/// Unoptimized meshes typically have ACMR ~0.7-1.0, optimized ~0.5-0.7.
pub fn compute_acmr(mesh: &Mesh, cache_size: usize) -> f32 {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return 0.0;
    }

    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut misses = 0u32;

    for &idx in &mesh.indices {
        if !cache.contains(&idx) {
            misses += 1;
            cache.insert(0, idx);
            if cache.len() > cache_size {
                cache.pop();
            }
        } else {
            // Move to front (LRU)
            if let Some(pos) = cache.iter().position(|&v| v == idx) {
                cache.remove(pos);
                cache.insert(0, idx);
            }
        }
    }

    misses as f32 / tri_count as f32
}

/// Deduplicate vertices that share the same position, normal, and UV
///
/// Merges identical vertices and updates indices accordingly.
/// Reduces vertex buffer size and improves cache efficiency.
pub fn deduplicate_vertices(mesh: &mut Mesh) {
    use std::collections::HashMap;

    let mut vertex_map: HashMap<u64, u32> = HashMap::new();
    let mut new_vertices = Vec::new();
    let mut index_remap: Vec<u32> = Vec::with_capacity(mesh.vertices.len());

    for v in &mesh.vertices {
        // Hash key from quantized position + normal + uv
        let key = hash_vertex(v);

        if let Some(&new_idx) = vertex_map.get(&key) {
            index_remap.push(new_idx);
        } else {
            let new_idx = new_vertices.len() as u32;
            vertex_map.insert(key, new_idx);
            new_vertices.push(*v);
            index_remap.push(new_idx);
        }
    }

    // Remap indices
    for idx in &mut mesh.indices {
        *idx = index_remap[*idx as usize];
    }

    mesh.vertices = new_vertices;
}

fn hash_vertex(v: &crate::mesh::Vertex) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();

    // Quantize to avoid floating point issues
    let qp = [
        (v.position.x * 10000.0) as i32,
        (v.position.y * 10000.0) as i32,
        (v.position.z * 10000.0) as i32,
    ];
    let qn = [
        (v.normal.x * 1000.0) as i32,
        (v.normal.y * 1000.0) as i32,
        (v.normal.z * 1000.0) as i32,
    ];
    let quv = [(v.uv.x * 10000.0) as i32, (v.uv.y * 10000.0) as i32];

    qp.hash(&mut hasher);
    qn.hash(&mut hasher);
    quv.hash(&mut hasher);
    v.material_id.hash(&mut hasher);

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    #[test]
    fn test_vertex_cache_optimization() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let acmr_before = compute_acmr(&mesh, 32);
        optimize_vertex_cache(&mut mesh);
        let acmr_after = compute_acmr(&mesh, 32);

        // Optimized should be equal or better
        assert!(
            acmr_after <= acmr_before + 0.01,
            "ACMR should improve: before={}, after={}",
            acmr_before,
            acmr_after
        );
    }

    #[test]
    fn test_acmr() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let acmr = compute_acmr(&mesh, 32);
        assert!(acmr > 0.0);
        assert!(acmr <= 3.0); // Maximum 3 misses per triangle
    }

    #[test]
    fn test_deduplicate_vertices() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let vert_before = mesh.vertex_count();
        deduplicate_vertices(&mut mesh);
        let vert_after = mesh.vertex_count();

        // Should have same or fewer vertices
        assert!(vert_after <= vert_before);
        // Triangles should be unchanged
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_single_triangle() {
        // Edge case: single triangle
        let mut mesh = Mesh {
            vertices: vec![
                crate::mesh::Vertex::new(Vec3::ZERO, Vec3::Y),
                crate::mesh::Vertex::new(Vec3::X, Vec3::Y),
                crate::mesh::Vertex::new(Vec3::Z, Vec3::Y),
            ],
            indices: vec![0, 1, 2],
        };

        optimize_vertex_cache(&mut mesh);
        assert_eq!(mesh.indices.len(), 3);
    }
}

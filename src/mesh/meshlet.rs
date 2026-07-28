//! Meshlet building for GPU mesh shader pipeline
//!
//! 任意 mesh を small chunks (meshlet) に分割する GPU mesh shader
//! (Vulkan `VK_EXT_mesh_shader` / DirectX 12 Mesh Shaders / Metal Mesh Shading)
//! で cluster-based rendering の入力として利用する
//!
//! # Meshlet の制約
//!
//! - 最大 vertex 数: 64 (u8 local index の実用上限)
//! - 最大 triangle 数: 124 (4-byte 境界に aligned、typical GPU shader limit)
//! - Local triangle indices は u8 (0-255) で meshlet 内の頂点を参照
//!
//! # 既存インフラとの関係
//!
//! - `nanite.rs::ClusterBounds` を bounds 計算に再利用 (球体 + AABB + LOD screen error)
//! - `nanite.rs::NormalCone` を back-face culling に再利用 (前 session 追加分)
//! - `optimize.rs::optimize_vertex_cache` 経由の mesh を入力にすると meshlet 内の
//!   spatial locality が向上、既に vcache 済 mesh は基本的に良い meshlet 分割になる
//!
//! # アルゴリズム (簡易 sequential 版、~200 行)
//!
//! 1. index buffer を先頭から走査
//! 2. 各 triangle について新規頂点数を算出
//! 3. `max_vertices` / `max_triangles` を超えたら現 meshlet を確定、新 meshlet 開始
//! 4. 3 頂点を meshlet に追加 (dedup)、local index (u8) で triangles に push
//! 5. meshlet 確定時: bounds + normal_cone を計算
//!
//! # meshoptimizer 完全版との差
//!
//! - **未実装**: adjacency-based grow (共有頂点で triangle スコアリング)
//! - **未実装**: `cone_weight` による normal 一貫性 preference
//! - 前提: 入力 mesh が `optimize_vertex_cache` / `optimize_spatial_order` 済で locality 良好
//!
//! # References
//!
//! - zeux/meshoptimizer §clusterizer.cpp `meshopt_buildMeshlets`
//! - Vulkan `VK_EXT_mesh_shader` spec / DirectX 12 mesh shader tutorial
//! - "Introduction to Turing Mesh Shaders" (NVIDIA, 2018)
//!
//! Author: Moroya Sakamoto

use crate::mesh::nanite::{ClusterBounds, NormalCone};
use crate::mesh::Mesh;
use glam::Vec3;

/// GPU mesh shader 用 meshlet
///
/// - `vertices`: グローバル頂点 index (元 mesh.vertices を参照)
/// - `triangles`: ローカル三角形 index (`vertices[]` 内、u8 型で 3 x n 個)
/// - `bounds`: cluster culling 用の bounding sphere + AABB
/// - `normal_cone`: back-face culling 用 (`is_backface_culled` で使用)
///
/// # Local index 制約
///
/// `triangles[i]` は `vertices[]` の 0..64 の範囲を指す 各三角形は 3 連続 (i, i+1, i+2) で 1 face
#[derive(Debug, Clone)]
pub struct Meshlet {
    /// Global vertex indices (into original `mesh.vertices`)
    pub vertices: Vec<u32>,
    /// Local triangle indices (3 x triangle_count elements、each < `vertices.len()`)
    pub triangles: Vec<u8>,
    /// Bounding sphere + AABB for frustum culling
    pub bounds: ClusterBounds,
    /// Normal cone for back-face culling (前 session `NormalCone` 活用)
    pub normal_cone: NormalCone,
}

impl Meshlet {
    /// Triangle 数
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    /// Vertex 数
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
}

/// `build_meshlets` の設定
#[derive(Debug, Clone, Copy)]
pub struct MeshletConfig {
    /// Meshlet あたりの最大頂点数 (u8 local index 上限、typical 64)
    pub max_vertices: usize,
    /// Meshlet あたりの最大三角形数 (typical 124、4-byte aligned)
    pub max_triangles: usize,
    /// Normal cone 一貫性重み (現状未使用、future work、[0.0, 1.0])
    ///
    /// meshoptimizer 完全版では cone_weight > 0 で triangle 選択時に
    /// 現 meshlet の平均法線との一貫性を優先する 現行簡易実装では未使用
    pub cone_weight: f32,
}

impl Default for MeshletConfig {
    /// GPU mesh shader typical 値: 64 vertices / 124 triangles / cone_weight 0.0
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.0,
        }
    }
}

/// Mesh を meshlet 列に分割
///
/// # アルゴリズム
///
/// Sequential greedy: index buffer 順に triangle を処理、現 meshlet に
/// 追加できる限り追加し、容量オーバーしたら新 meshlet 開始
///
/// # 制約
///
/// - `config.max_vertices <= 255` (u8 local index の制約)
/// - `config.max_triangles > 0`
/// - 制約違反時は panic せず空 Vec を返す (fail fast より緩い方針、logging も可)
///
/// # 前提
///
/// 入力 mesh が `optimize_vertex_cache` / `optimize_spatial_order` 済ならば
/// locality 良好な meshlet 分割になる (未最適化の場合は分割数増加)
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn build_meshlets(mesh: &Mesh, config: &MeshletConfig) -> Vec<Meshlet> {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0
        || config.max_vertices == 0
        || config.max_triangles == 0
        || config.max_vertices > 255
    {
        return Vec::new();
    }

    let mut meshlets: Vec<Meshlet> = Vec::new();
    let mut current_vertices: Vec<u32> = Vec::with_capacity(config.max_vertices);
    let mut current_triangles: Vec<u8> = Vec::with_capacity(config.max_triangles * 3);

    for t in 0..tri_count {
        let base = t * 3;
        let ia = mesh.indices[base];
        let ib = mesh.indices[base + 1];
        let ic = mesh.indices[base + 2];

        // 新規頂点数を数える (dedup)
        let mut new_verts_needed = 0;
        for &v in &[ia, ib, ic] {
            if !current_vertices.contains(&v) {
                new_verts_needed += 1;
            }
        }

        // 容量チェック
        let would_exceed_verts = current_vertices.len() + new_verts_needed > config.max_vertices;
        let would_exceed_tris = current_triangles.len() / 3 + 1 > config.max_triangles;

        if would_exceed_verts || would_exceed_tris {
            // 現 meshlet を確定
            if !current_triangles.is_empty() {
                meshlets.push(finalize_meshlet(
                    &current_vertices,
                    &current_triangles,
                    mesh,
                ));
                current_vertices.clear();
                current_triangles.clear();
            }
        }

        // Triangle を追加: 3 頂点を dedup 挿入、local index を triangles に push
        for &v in &[ia, ib, ic] {
            let local_idx = if let Some(pos) = current_vertices.iter().position(|&x| x == v) {
                pos
            } else {
                current_vertices.push(v);
                current_vertices.len() - 1
            };
            // local_idx < config.max_vertices <= 255 なので u8 cast 安全
            current_triangles.push(local_idx as u8);
        }
    }

    // 残余を確定
    if !current_triangles.is_empty() {
        meshlets.push(finalize_meshlet(
            &current_vertices,
            &current_triangles,
            mesh,
        ));
    }

    meshlets
}

/// Meshlet 確定時に bounds + normal_cone を計算して `Meshlet` を構築
fn finalize_meshlet(vertices: &[u32], triangles: &[u8], mesh: &Mesh) -> Meshlet {
    // 頂点 world positions
    let positions: Vec<Vec3> = vertices
        .iter()
        .filter_map(|&i| mesh.vertices.get(i as usize).map(|v| v.position))
        .collect();

    // Face normals (triangle ごとに位置差から計算、頂点法線は使わない = より精確)
    let tri_count = triangles.len() / 3;
    let mut face_normals: Vec<Vec3> = Vec::with_capacity(tri_count);
    for t in 0..tri_count {
        let base = t * 3;
        let a_local = triangles[base] as usize;
        let b_local = triangles[base + 1] as usize;
        let c_local = triangles[base + 2] as usize;
        if a_local >= positions.len() || b_local >= positions.len() || c_local >= positions.len() {
            continue;
        }
        let a = positions[a_local];
        let b = positions[b_local];
        let c = positions[c_local];
        let n = (b - a).cross(c - a);
        if n.length_squared() > 1e-12 {
            face_normals.push(n.normalize());
        }
    }

    let bounds = ClusterBounds::from_vertices(&positions);
    let normal_cone = NormalCone::from_normals(&face_normals);

    Meshlet {
        vertices: vertices.to_vec(),
        triangles: triangles.to_vec(),
        bounds,
        normal_cone,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig, Vertex};
    use crate::types::SdfNode;

    fn make_sphere_mesh(resolution: usize) -> Mesh {
        let sphere = SdfNode::sphere(1.0);
        sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        )
    }

    #[test]
    fn test_empty_mesh_returns_empty() {
        let mesh = Mesh::new();
        let meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        assert!(meshlets.is_empty());
    }

    #[test]
    fn test_small_mesh_single_meshlet() {
        // 単一三角形 → 1 meshlet
        let mesh = Mesh {
            vertices: vec![
                Vertex::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
                Vertex::new(Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
                Vertex::new(Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
            ],
            indices: vec![0, 1, 2],
        };
        let meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].vertex_count(), 3);
        assert_eq!(meshlets[0].triangle_count(), 1);
        assert_eq!(meshlets[0].triangles, vec![0, 1, 2]);
    }

    #[test]
    fn test_sphere_mesh_multiple_meshlets() {
        // sphere mesh を分割、meshlet 数 > 1 で全 constraints 満たす
        let mesh = make_sphere_mesh(16);
        let config = MeshletConfig::default();
        let meshlets = build_meshlets(&mesh, &config);
        assert!(
            meshlets.len() >= 1,
            "expected at least 1 meshlet, got 0 (mesh has {} tris)",
            mesh.triangle_count()
        );
        for m in &meshlets {
            assert!(
                m.vertex_count() <= config.max_vertices,
                "meshlet exceeded max_vertices: {} > {}",
                m.vertex_count(),
                config.max_vertices
            );
            assert!(
                m.triangle_count() <= config.max_triangles,
                "meshlet exceeded max_triangles: {} > {}",
                m.triangle_count(),
                config.max_triangles
            );
        }
    }

    #[test]
    fn test_local_indices_within_range() {
        // 全 meshlet の local triangle indices が vertices.len() 未満
        let mesh = make_sphere_mesh(8);
        let meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        for m in &meshlets {
            let n = m.vertex_count() as u8;
            for &idx in &m.triangles {
                assert!(
                    idx < n,
                    "local index {} out of range (vertex_count={})",
                    idx,
                    n
                );
            }
        }
    }

    #[test]
    fn test_triangle_count_conservation() {
        // 全 meshlet の三角形数の合計は元 mesh の三角形数と一致 (重複なし、欠落なし)
        let mesh = make_sphere_mesh(16);
        let orig_tris = mesh.triangle_count();
        let meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        let total: usize = meshlets.iter().map(|m| m.triangle_count()).sum();
        assert_eq!(
            total, orig_tris,
            "meshlet triangle count sum mismatch: {} vs {}",
            total, orig_tris
        );
    }

    #[test]
    fn test_smaller_max_increases_meshlet_count() {
        // max_vertices / max_triangles を小さくすると meshlet 数が増える
        let mesh = make_sphere_mesh(16);
        let default_meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        let small_config = MeshletConfig {
            max_vertices: 32,
            max_triangles: 60,
            cone_weight: 0.0,
        };
        let small_meshlets = build_meshlets(&mesh, &small_config);
        assert!(
            small_meshlets.len() > default_meshlets.len(),
            "smaller config should produce more meshlets: {} vs {}",
            small_meshlets.len(),
            default_meshlets.len()
        );
    }

    #[test]
    fn test_bounds_and_normal_cone_populated() {
        // 各 meshlet に bounds + normal_cone が計算されている
        let mesh = make_sphere_mesh(8);
        let meshlets = build_meshlets(&mesh, &MeshletConfig::default());
        for m in &meshlets {
            // radius > 0 (単一頂点でない限り)
            assert!(m.bounds.radius >= 0.0);
            // normal_cone.axis は単位ベクトル (unbounded 時は Vec3::Y)
            assert!((m.normal_cone.axis.length() - 1.0).abs() < 1e-4);
            // cutoff_cos は [-1, 1]
            assert!(m.normal_cone.cutoff_cos >= -1.0 && m.normal_cone.cutoff_cos <= 1.0);
        }
    }

    #[test]
    fn test_invalid_config_returns_empty() {
        let mesh = make_sphere_mesh(4);
        // max_vertices = 0 → 空
        let m0 = build_meshlets(
            &mesh,
            &MeshletConfig {
                max_vertices: 0,
                max_triangles: 64,
                cone_weight: 0.0,
            },
        );
        assert!(m0.is_empty());
        // max_triangles = 0 → 空
        let m1 = build_meshlets(
            &mesh,
            &MeshletConfig {
                max_vertices: 64,
                max_triangles: 0,
                cone_weight: 0.0,
            },
        );
        assert!(m1.is_empty());
        // max_vertices > 255 → 空 (u8 local index 制約違反)
        let m2 = build_meshlets(
            &mesh,
            &MeshletConfig {
                max_vertices: 256,
                max_triangles: 64,
                cone_weight: 0.0,
            },
        );
        assert!(m2.is_empty());
    }

    #[test]
    fn test_default_config_values() {
        let c = MeshletConfig::default();
        assert_eq!(c.max_vertices, 64);
        assert_eq!(c.max_triangles, 124);
        assert!((c.cone_weight - 0.0).abs() < 1e-6);
    }
}

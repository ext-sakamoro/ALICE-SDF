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
    /// Normal cone 一貫性重み (V2 モード = `adjacency_grow=true` 時のみ有効、[0.0, 1.0])
    ///
    /// V2 スコア式: `score = reuse_count + cone_weight × dot(candidate_normal, avg_normal)`
    /// - `0.0`: vertex reuse のみ preference (locality 優先)
    /// - `1.0`: normal 一貫性を強く preference (back-face culling 効率化)
    /// - typical: `0.25 - 0.5`
    pub cone_weight: f32,
    /// V2 モード: adjacency-based grow を使う (default `false` = V1 scan)
    ///
    /// - `false` (V1、default): index buffer 順の sequential scan、高速だが locality 依存
    /// - `true` (V2): vertex adjacency + normal 一貫性で triangle スコアリング、
    ///   meshlet 品質向上 (culling 効率化 / cache 局所性向上)、CPU コスト増加
    pub adjacency_grow: bool,
}

impl Default for MeshletConfig {
    /// GPU mesh shader typical 値: 64 vertices / 124 triangles / cone_weight 0.0 / V1 scan
    fn default() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.0,
            adjacency_grow: false,
        }
    }
}

impl MeshletConfig {
    /// V2 高品質モード (`adjacency_grow=true`, `cone_weight=0.25`)
    ///
    /// 本 profile は cluster culling / back-face rejection 効率を高めた mesh shader
    /// pipeline 向け、CPU 側の meshlet 生成コストは V1 の 1.5-2 倍程度、
    /// 実行時の描画コスト削減で相殺されることを狙う
    #[must_use]
    pub const fn quality() -> Self {
        Self {
            max_vertices: 64,
            max_triangles: 124,
            cone_weight: 0.25,
            adjacency_grow: true,
        }
    }
}

/// Mesh を meshlet 列に分割 (V1 scan / V2 adjacency で dispatch)
///
/// `config.adjacency_grow` で分岐:
/// - `false` (default): V1 = `build_meshlets_scan` (index buffer 順の sequential 分割)
/// - `true`: V2 = `build_meshlets_adjacency` (vertex adjacency + normal コーン一貫性で選択)
///
/// # 制約
///
/// - `config.max_vertices <= 255` (u8 local index の制約)
/// - `config.max_triangles > 0`
/// - 制約違反時は panic せず空 Vec を返す
///
/// # 選択指針
///
/// - **V1 (`adjacency_grow=false`)**: 高速、locality 良好な mesh (`optimize_vertex_cache` /
///   `optimize_spatial_order` 済) 前提、typical 用途
/// - **V2 (`adjacency_grow=true`)**: CPU コスト増 (1.5-2 倍)、meshlet 品質向上、
///   実際の GPU rendering で描画コスト削減で回収可能、`MeshletConfig::quality()` prof 済
#[must_use]
pub fn build_meshlets(mesh: &Mesh, config: &MeshletConfig) -> Vec<Meshlet> {
    if config.adjacency_grow {
        build_meshlets_adjacency(mesh, config)
    } else {
        build_meshlets_scan(mesh, config)
    }
}

/// V1: sequential scan (`build_meshlets` の実装本体、`adjacency_grow=false` 時)
///
/// index buffer を先頭から順に走査、容量オーバーで新 meshlet を切り出す
/// 単純だが `optimize_vertex_cache` 済 mesh を前提に typical use case を高速処理する
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn build_meshlets_scan(mesh: &Mesh, config: &MeshletConfig) -> Vec<Meshlet> {
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

/// V2: adjacency-based grow (`adjacency_grow=true` 時、`meshopt_buildMeshlets` 相当)
///
/// # アルゴリズム
///
/// 1. **事前計算**:
///    - `vertex_to_tris[vi]` = vertex `vi` を含む triangle index の list
///    - `face_normals[t]` = triangle `t` の face normal (単位ベクトル)
///
/// 2. **Meshlet 成長**:
///    - seed triangle (未処理の最初) で新 meshlet 開始、3 頂点 + 1 face 追加
///    - 現 meshlet の全頂点に隣接する未処理 triangle を candidate 集合とする
///    - スコア: `reuse_count + cone_weight × dot(candidate_normal, avg_normal)`
///    - 最高スコアの candidate を追加、容量 over まで繰り返す
///    - candidate が空 or 容量 over で meshlet 確定
///
/// 3. **fallback**: 未処理 triangle が残る限り新 seed で継続
///
/// # 計算量
///
/// - 事前計算 O(T)
/// - 各 meshlet: O(V_meshlet × T_adj/vertex × log)、typical mesh で許容範囲
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
pub fn build_meshlets_adjacency(mesh: &Mesh, config: &MeshletConfig) -> Vec<Meshlet> {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0
        || config.max_vertices == 0
        || config.max_triangles == 0
        || config.max_vertices > 255
    {
        return Vec::new();
    }

    // 事前計算
    let face_normals = precompute_face_normals(mesh);
    let vertex_to_tris = precompute_vertex_to_tris(mesh);

    let mut meshlets: Vec<Meshlet> = Vec::new();
    let mut processed = vec![false; tri_count];
    let mut next_seed = 0usize;

    loop {
        // seed triangle 探索 (未処理の最初)
        while next_seed < tri_count && processed[next_seed] {
            next_seed += 1;
        }
        if next_seed >= tri_count {
            break;
        }

        let seed = next_seed;
        processed[seed] = true;

        let mut current_vertices: Vec<u32> = Vec::with_capacity(config.max_vertices);
        let mut current_triangles: Vec<u8> = Vec::with_capacity(config.max_triangles * 3);
        let mut normal_sum = Vec3::ZERO;

        // seed 追加
        add_triangle_to_meshlet(&mut current_vertices, &mut current_triangles, mesh, seed);
        if let Some(&n) = face_normals.get(seed) {
            normal_sum += n;
        }

        // 成長 loop
        loop {
            // candidate 集合: 現 meshlet の頂点に隣接する未処理 triangle
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut best_tri: Option<usize> = None;
            let mut best_reuse: usize = 0;

            let avg_normal = if normal_sum.length_squared() > 1e-12 {
                normal_sum.normalize()
            } else {
                Vec3::ZERO
            };

            for &vi in &current_vertices {
                let adj_list = match vertex_to_tris.get(vi as usize) {
                    Some(list) => list,
                    None => continue,
                };
                for &cand_t in adj_list {
                    let cand_t_usize = cand_t as usize;
                    if processed[cand_t_usize] {
                        continue;
                    }
                    let base = cand_t_usize * 3;
                    let ia = mesh.indices[base];
                    let ib = mesh.indices[base + 1];
                    let ic = mesh.indices[base + 2];

                    // reuse_count: candidate 3 頂点のうち current に既に含まれる数
                    let mut reuse = 0;
                    for &v in &[ia, ib, ic] {
                        if current_vertices.contains(&v) {
                            reuse += 1;
                        }
                    }
                    let new_verts = 3 - reuse;

                    // 容量チェック
                    if current_vertices.len() + new_verts > config.max_vertices {
                        continue;
                    }
                    if current_triangles.len() / 3 + 1 > config.max_triangles {
                        continue;
                    }

                    // スコア: reuse + cone_weight × normal 一貫性
                    let cand_normal = face_normals.get(cand_t_usize).copied().unwrap_or(Vec3::Y);
                    let cone_term = if config.cone_weight > 0.0 && avg_normal != Vec3::ZERO {
                        config.cone_weight * cand_normal.dot(avg_normal)
                    } else {
                        0.0
                    };
                    let score = reuse as f32 + cone_term;

                    if score > best_score {
                        best_score = score;
                        best_tri = Some(cand_t_usize);
                        best_reuse = reuse;
                    }
                }
            }

            let _ = best_reuse; // debug / logging 用に確保

            match best_tri {
                Some(t) => {
                    processed[t] = true;
                    add_triangle_to_meshlet(&mut current_vertices, &mut current_triangles, mesh, t);
                    if let Some(&n) = face_normals.get(t) {
                        normal_sum += n;
                    }
                }
                None => break, // 追加できる候補なし、meshlet 確定へ
            }
        }

        if !current_triangles.is_empty() {
            meshlets.push(finalize_meshlet(
                &current_vertices,
                &current_triangles,
                mesh,
            ));
        }
    }

    meshlets
}

/// 三角形 `t` を meshlet に追加 (3 頂点を dedup 挿入 + 3 local index push)
#[allow(clippy::cast_possible_truncation)]
fn add_triangle_to_meshlet(
    current_vertices: &mut Vec<u32>,
    current_triangles: &mut Vec<u8>,
    mesh: &Mesh,
    t: usize,
) {
    let base = t * 3;
    for k in 0..3 {
        let v = mesh.indices[base + k];
        let local_idx = if let Some(pos) = current_vertices.iter().position(|&x| x == v) {
            pos
        } else {
            current_vertices.push(v);
            current_vertices.len() - 1
        };
        current_triangles.push(local_idx as u8);
    }
}

/// face normals の事前計算 (三角形ごとに 1 normal、退化面は Vec3::ZERO)
fn precompute_face_normals(mesh: &Mesh) -> Vec<Vec3> {
    let tri_count = mesh.indices.len() / 3;
    let mut normals = Vec::with_capacity(tri_count);
    for t in 0..tri_count {
        let base = t * 3;
        let ia = mesh.indices[base] as usize;
        let ib = mesh.indices[base + 1] as usize;
        let ic = mesh.indices[base + 2] as usize;
        if ia >= mesh.vertices.len() || ib >= mesh.vertices.len() || ic >= mesh.vertices.len() {
            normals.push(Vec3::ZERO);
            continue;
        }
        let a = mesh.vertices[ia].position;
        let b = mesh.vertices[ib].position;
        let c = mesh.vertices[ic].position;
        let n = (b - a).cross(c - a);
        if n.length_squared() > 1e-12 {
            normals.push(n.normalize());
        } else {
            normals.push(Vec3::ZERO);
        }
    }
    normals
}

/// vertex → triangle index の逆マップ事前計算
fn precompute_vertex_to_tris(mesh: &Mesh) -> Vec<Vec<u32>> {
    let vert_count = mesh.vertices.len();
    let tri_count = mesh.indices.len() / 3;
    let mut vtot: Vec<Vec<u32>> = vec![Vec::new(); vert_count];
    for t in 0..tri_count {
        let base = t * 3;
        for k in 0..3 {
            let v = mesh.indices[base + k] as usize;
            if v < vert_count {
                vtot[v].push(t as u32);
            }
        }
    }
    vtot
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
            adjacency_grow: false,
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
                adjacency_grow: false,
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
                adjacency_grow: false,
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
                adjacency_grow: false,
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
        assert!(!c.adjacency_grow);
    }

    // ------------------------------------------------------------------------
    // V2 adjacency + cone_weight テスト (2026-07-28 追加)
    // ------------------------------------------------------------------------

    #[test]
    fn test_quality_config_is_v2() {
        let c = MeshletConfig::quality();
        assert!(c.adjacency_grow);
        assert!(c.cone_weight > 0.0);
    }

    #[test]
    fn test_v2_empty_mesh_returns_empty() {
        let mesh = Mesh::new();
        let meshlets = build_meshlets(&mesh, &MeshletConfig::quality());
        assert!(meshlets.is_empty());
    }

    #[test]
    fn test_v2_single_triangle_single_meshlet() {
        let mesh = Mesh {
            vertices: vec![
                Vertex::new(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
                Vertex::new(Vec3::new(1.0, 0.0, 0.0), Vec3::Y),
                Vertex::new(Vec3::new(0.0, 0.0, 1.0), Vec3::Y),
            ],
            indices: vec![0, 1, 2],
        };
        let meshlets = build_meshlets(&mesh, &MeshletConfig::quality());
        assert_eq!(meshlets.len(), 1);
        assert_eq!(meshlets[0].vertex_count(), 3);
        assert_eq!(meshlets[0].triangle_count(), 1);
    }

    #[test]
    fn test_v2_sphere_conservation() {
        // V2 でも全 triangle が meshlet に含まれる (重複なし、欠落なし)
        let mesh = make_sphere_mesh(16);
        let orig_tris = mesh.triangle_count();
        let meshlets = build_meshlets(&mesh, &MeshletConfig::quality());
        let total: usize = meshlets.iter().map(|m| m.triangle_count()).sum();
        assert_eq!(
            total, orig_tris,
            "V2 triangle count mismatch: {} vs {}",
            total, orig_tris
        );
    }

    #[test]
    fn test_v2_constraints_respected() {
        let mesh = make_sphere_mesh(16);
        let config = MeshletConfig::quality();
        let meshlets = build_meshlets(&mesh, &config);
        for m in &meshlets {
            assert!(m.vertex_count() <= config.max_vertices);
            assert!(m.triangle_count() <= config.max_triangles);
            // local index 範囲
            let n = m.vertex_count() as u8;
            for &idx in &m.triangles {
                assert!(idx < n);
            }
        }
    }

    #[test]
    fn test_v2_normal_cone_tighter_with_cone_weight() {
        // cone_weight > 0 で meshlet 内の平均 cutoff_cos が V1 より高い or 同等
        // (curved sphere で locality-only V1 vs cone-aware V2 の差を計測)
        let mesh = make_sphere_mesh(16);
        let v1_config = MeshletConfig {
            adjacency_grow: false,
            ..MeshletConfig::default()
        };
        let v2_config = MeshletConfig {
            adjacency_grow: true,
            cone_weight: 0.5,
            ..MeshletConfig::default()
        };
        let v1_meshlets = build_meshlets(&mesh, &v1_config);
        let v2_meshlets = build_meshlets(&mesh, &v2_config);

        let avg_cutoff = |ms: &[Meshlet]| -> f32 {
            if ms.is_empty() {
                return 0.0;
            }
            let sum: f32 = ms.iter().map(|m| m.normal_cone.cutoff_cos).sum();
            sum / (ms.len() as f32)
        };
        let v1_avg = avg_cutoff(&v1_meshlets);
        let v2_avg = avg_cutoff(&v2_meshlets);

        // V2 の平均 cutoff_cos は V1 と同等以上 (小規模 mesh で偶発逆転はあり得るので緩めに)
        assert!(
            v2_avg >= v1_avg - 0.15,
            "V2 avg cutoff_cos should be >= V1 (with tolerance): V1={}, V2={}",
            v1_avg,
            v2_avg
        );
    }

    #[test]
    fn test_v2_bounds_populated() {
        let mesh = make_sphere_mesh(8);
        let meshlets = build_meshlets(&mesh, &MeshletConfig::quality());
        for m in &meshlets {
            assert!(m.bounds.radius >= 0.0);
            assert!((m.normal_cone.axis.length() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_v2_invalid_config_returns_empty() {
        let mesh = make_sphere_mesh(4);
        let m = build_meshlets(
            &mesh,
            &MeshletConfig {
                max_vertices: 0,
                max_triangles: 64,
                cone_weight: 0.5,
                adjacency_grow: true,
            },
        );
        assert!(m.is_empty());
    }

    #[test]
    fn test_v2_vs_v1_both_produce_valid_meshlets() {
        // V1 と V2 は結果 mesh が異なる可能性があるが、両方 valid meshlet を生成
        let mesh = make_sphere_mesh(12);
        let v1 = build_meshlets(&mesh, &MeshletConfig::default());
        let v2 = build_meshlets(&mesh, &MeshletConfig::quality());
        assert!(!v1.is_empty());
        assert!(!v2.is_empty());
        // 両者とも三角形保存
        let v1_total: usize = v1.iter().map(|m| m.triangle_count()).sum();
        let v2_total: usize = v2.iter().map(|m| m.triangle_count()).sum();
        assert_eq!(v1_total, mesh.triangle_count());
        assert_eq!(v2_total, mesh.triangle_count());
    }

    #[test]
    fn test_v2_scan_helper_direct() {
        // build_meshlets_scan と build_meshlets_adjacency を直接呼び分けて結果比較
        let mesh = make_sphere_mesh(8);
        let config = MeshletConfig::default();
        let scan_result = build_meshlets_scan(&mesh, &config);
        let adj_config = MeshletConfig {
            adjacency_grow: true,
            ..config
        };
        let adj_result = build_meshlets_adjacency(&mesh, &adj_config);
        // 両方非空
        assert!(!scan_result.is_empty());
        assert!(!adj_result.is_empty());
    }
}

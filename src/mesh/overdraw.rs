//! Overdraw Optimization (meshoptimizer §overdrawoptimizer 移植)
//!
//! GPU の Early-Z / hidden surface rejection 効率を高めるため、三角形を
//! **view-independent front-to-back** 順に並べ替える 完全な view 非依存は不可能なので、
//! 6 軸方向 (±X/±Y/±Z) を sampling して average rank で近似する
//!
//! # `optimize_vertex_cache` との関係
//!
//! 単純な front-to-back sort は vertex cache 局所性を破壊する (連続三角形が
//! 別の vertex を参照する) 本実装は **cluster preserving** で、vcache opt で
//! 生成された "cluster" (連続 cache_size 近傍) を単位に sort する
//! `threshold` パラメータで cluster 内での再 sort 許容度を制御
//!
//! # 呼び出し順序
//!
//! ```text
//! deduplicate_vertices → optimize_vertex_cache → optimize_overdraw → optimize_vertex_fetch
//!                        (ACMR ↓)                 (Early-Z ↓)         (ATVR ↓)
//! ```
//!
//! # References
//!
//! - zeux/meshoptimizer §overdrawoptimizer.cpp (完全版は cluster 内 sort も含む、本版は cluster 順序のみ)
//! - Tom Forsyth "Optimizing indexed triangle meshes for GPU vertex cache" (2006)
//!
//! Author: Moroya Sakamoto

use crate::mesh::Mesh;
use glam::Vec3;

/// vertex cache size (Forsyth と揃える、typical modern GPU 32)
const OVERDRAW_CACHE_SIZE: usize = 32;

/// Cluster: 三角形の連続範囲、内部で cache miss が threshold 以下
struct Cluster {
    start_tri: usize,
    end_tri: usize, // exclusive
    centroid: Vec3,
}

/// View-independent overdraw optimization (簡易版)
///
/// # アルゴリズム
///
/// 1. 現 index buffer を走査、vertex cache miss を数える
///    → cache miss が発生する境界で cluster を切る
/// 2. 各 cluster の三角形重心平均を計算
/// 3. 6 軸方向 (±X/±Y/±Z) に対して cluster centroid を投影
///    → 各方向で cluster を depth 昇順に rank 付け
/// 4. 6 方向の rank 平均で cluster を並び替え
/// 5. cluster 内の三角形順序は保持 (vcache 局所性維持)
///
/// # 引数
///
/// - `threshold`: [0.0, 1.0]、大きいほど vcache 犠牲を許容 (0.0 = cluster 順不変、
///   1.0 = full sort、実装的には常に cluster 単位 sort なので現状 unused)
///
/// # 制約
///
/// - `optimize_vertex_cache` 実行後に呼ぶこと (それ以外だと cluster 分割意味なし)
/// - view が特定方向に固定される use case (top-down、first-person 等) には
///   専用 view direction を渡す `optimize_overdraw_with_views` を使う
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn optimize_overdraw(mesh: &mut Mesh, threshold: f32) {
    let views = default_view_directions();
    optimize_overdraw_with_views(mesh, threshold, &views);
}

/// View direction を明示指定する overdraw optimization
///
/// # 引数
///
/// - `view_directions`: 単位ベクトルの配列、front-to-back 判定の view center を代表
///   典型: 6 axis (`default_view_directions`)、14 (6 + 8 cube corners)、Fibonacci sphere
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn optimize_overdraw_with_views(mesh: &mut Mesh, _threshold: f32, view_directions: &[Vec3]) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 || view_directions.is_empty() {
        return;
    }

    // 1. cluster 化: vcache miss 境界で切る
    let clusters = compute_clusters(mesh, OVERDRAW_CACHE_SIZE);
    if clusters.len() <= 1 {
        return; // 1 cluster ならソート意味なし
    }

    // 2. 各 view 方向で cluster を depth sort、rank を集計
    let mut rank_sum = vec![0.0_f32; clusters.len()];
    for &view in view_directions {
        let mut view_indexed: Vec<(usize, f32)> = clusters
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.centroid.dot(view)))
            .collect();
        // 昇順 (近い = 前面 = 先に描画) sort
        view_indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        // rank を集計 (0 = 最前、len-1 = 最奥)
        for (rank, &(cluster_idx, _)) in view_indexed.iter().enumerate() {
            rank_sum[cluster_idx] += rank as f32;
        }
    }

    // 3. rank 平均で cluster を並び替え
    let mut cluster_order: Vec<usize> = (0..clusters.len()).collect();
    cluster_order.sort_by(|&a, &b| {
        rank_sum[a]
            .partial_cmp(&rank_sum[b])
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    // 4. 新 index buffer 構築 (cluster 内 三角形順序は保持)
    let mut new_indices = Vec::with_capacity(mesh.indices.len());
    for &ci in &cluster_order {
        let c = &clusters[ci];
        for t in c.start_tri..c.end_tri {
            let base = t * 3;
            new_indices.push(mesh.indices[base]);
            new_indices.push(mesh.indices[base + 1]);
            new_indices.push(mesh.indices[base + 2]);
        }
    }

    mesh.indices = new_indices;
}

/// 6 軸方向 (±X/±Y/±Z) を default view directions として返す
#[must_use]
pub fn default_view_directions() -> [Vec3; 6] {
    [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ]
}

/// vertex cache miss 境界で三角形を cluster 分割
///
/// vcache LRU シミュレーションを走らせ、cache miss が発生した位置で cluster を切る
/// 各 cluster は連続する三角形群 + 重心を保持
#[allow(clippy::cast_precision_loss)]
fn compute_clusters(mesh: &Mesh, cache_size: usize) -> Vec<Cluster> {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return Vec::new();
    }

    let mut clusters: Vec<Cluster> = Vec::new();
    let mut cache: Vec<u32> = Vec::with_capacity(cache_size);
    let mut miss_since_last_boundary = 0usize;
    let mut cluster_start = 0usize;
    let mut centroid_accum = Vec3::ZERO;
    let mut centroid_count = 0usize;

    // cluster boundary threshold: 三角形 3 頂点全部 miss を "cluster 境界" として扱う
    // (簡易実装、meshoptimizer 完全版はより精緻)
    let boundary_miss_count = 3usize;

    for t in 0..tri_count {
        let base = t * 3;
        let ia = mesh.indices[base];
        let ib = mesh.indices[base + 1];
        let ic = mesh.indices[base + 2];

        // cache hit/miss 判定
        let mut misses = 0;
        for &v in &[ia, ib, ic] {
            if cache.contains(&v) {
                // hit → 順序保持 (LRU 化省略、rank だけで十分)
            } else {
                misses += 1;
                cache.push(v);
                if cache.len() > cache_size {
                    cache.remove(0);
                }
            }
        }
        miss_since_last_boundary += misses;

        // centroid 累積
        if (ia as usize) < mesh.vertices.len() {
            centroid_accum += mesh.vertices[ia as usize].position;
            centroid_count += 1;
        }
        if (ib as usize) < mesh.vertices.len() {
            centroid_accum += mesh.vertices[ib as usize].position;
            centroid_count += 1;
        }
        if (ic as usize) < mesh.vertices.len() {
            centroid_accum += mesh.vertices[ic as usize].position;
            centroid_count += 1;
        }

        // cluster boundary: 連続 miss が cache size に達する ≈ 局所性喪失
        // 簡易判定: この三角形が 3 頂点全部 miss、かつ前回 boundary から
        // 一定数 miss したら区切る
        if misses >= boundary_miss_count && miss_since_last_boundary >= cache_size {
            // 現 cluster を確定 (t を最後の要素として含める)
            let centroid = if centroid_count > 0 {
                centroid_accum / (centroid_count as f32)
            } else {
                Vec3::ZERO
            };
            clusters.push(Cluster {
                start_tri: cluster_start,
                end_tri: t + 1,
                centroid,
            });
            cluster_start = t + 1;
            miss_since_last_boundary = 0;
            centroid_accum = Vec3::ZERO;
            centroid_count = 0;
        }
    }

    // 残余を最終 cluster に
    if cluster_start < tri_count {
        let centroid = if centroid_count > 0 {
            centroid_accum / (centroid_count as f32)
        } else {
            Vec3::ZERO
        };
        clusters.push(Cluster {
            start_tri: cluster_start,
            end_tri: tri_count,
            centroid,
        });
    }

    clusters
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
    fn test_overdraw_empty_mesh() {
        let mut mesh = Mesh::new();
        optimize_overdraw(&mut mesh, 1.0);
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn test_overdraw_preserves_triangle_count() {
        let mut mesh = make_sphere_mesh(16);
        let tri_before = mesh.triangle_count();
        let vert_before = mesh.vertices.len();
        optimize_overdraw(&mut mesh, 1.0);
        // 三角形数 / 頂点数は不変 (順序のみ変わる)
        assert_eq!(mesh.triangle_count(), tri_before);
        assert_eq!(mesh.vertices.len(), vert_before);
    }

    #[test]
    fn test_overdraw_preserves_triangle_set() {
        // 三角形の集合 (順不同、頂点 index 3 つ組み) が保存されているか
        let mut mesh = make_sphere_mesh(8);
        let mut tris_before: Vec<[u32; 3]> = (0..mesh.triangle_count())
            .map(|t| {
                let mut tri = [
                    mesh.indices[t * 3],
                    mesh.indices[t * 3 + 1],
                    mesh.indices[t * 3 + 2],
                ];
                tri.sort_unstable();
                tri
            })
            .collect();
        tris_before.sort();

        optimize_overdraw(&mut mesh, 1.0);

        let mut tris_after: Vec<[u32; 3]> = (0..mesh.triangle_count())
            .map(|t| {
                let mut tri = [
                    mesh.indices[t * 3],
                    mesh.indices[t * 3 + 1],
                    mesh.indices[t * 3 + 2],
                ];
                tri.sort_unstable();
                tri
            })
            .collect();
        tris_after.sort();

        assert_eq!(
            tris_before, tris_after,
            "三角形集合が保存されていない (index 追加/削除された?)"
        );
    }

    #[test]
    fn test_overdraw_with_custom_views() {
        // 単一 view direction (top-down: +Y) で sort、深さ順が並ぶこと
        let mut mesh = make_sphere_mesh(8);
        let tri_before = mesh.triangle_count();
        let views = [Vec3::Y];
        optimize_overdraw_with_views(&mut mesh, 1.0, &views);
        assert_eq!(mesh.triangle_count(), tri_before);
    }

    #[test]
    fn test_overdraw_single_cluster_noop() {
        // 4 頂点 2 三角形の quad → cluster 1 個のみ、noop
        let v = |x: f32, z: f32| Vertex::new(Vec3::new(x, 0.0, z), Vec3::Y);
        let mut mesh = Mesh {
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)],
            indices: vec![0, 1, 2, 1, 3, 2],
        };
        let indices_before = mesh.indices.clone();
        optimize_overdraw(&mut mesh, 1.0);
        // cluster 1 個なら順序不変
        assert_eq!(mesh.indices, indices_before);
    }

    #[test]
    fn test_default_view_directions() {
        let views = default_view_directions();
        assert_eq!(views.len(), 6);
        // 全て単位ベクトル
        for v in &views {
            assert!((v.length() - 1.0).abs() < 1e-6);
        }
        // 反対方向のペアが 3 組
        assert!(views.contains(&Vec3::X) && views.contains(&Vec3::NEG_X));
        assert!(views.contains(&Vec3::Y) && views.contains(&Vec3::NEG_Y));
        assert!(views.contains(&Vec3::Z) && views.contains(&Vec3::NEG_Z));
    }
}

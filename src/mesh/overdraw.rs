//! Overdraw Optimization (meshoptimizer §overdrawoptimizer 移植)
//!
//! GPU の Early-Z / hidden surface rejection 効率を高めるため、三角形を
//! cluster 単位で並べ替える
//!
//! - [`optimize_overdraw`]: 視点に依存しない並べ替え (meshoptimizer
//!   `meshopt_optimizeOverdraw` と同じ方式) cluster の平均法線と、mesh 重心から
//!   cluster 重心への方向の内積が大きい cluster (外側を向き、他を隠しやすい) から描く
//! - [`optimize_overdraw_with_views`]: 指定した view 方向での深さ順位による並べ替え
//!
//! # `optimize_vertex_cache` との関係
//!
//! 単純な front-to-back sort は vertex cache 局所性を破壊する (連続三角形が
//! 別の vertex を参照する) 本実装は **cluster preserving** で、vcache opt で
//! 生成された "cluster" を単位に sort し、cluster 内の順序は保つ
//! `optimize_overdraw` の `threshold` は cluster をさらに細かく切る際の
//! ACMR 悪化の許容率 (meshoptimizer と同じ意味)
//!
//! # 呼び出し順序
//!
//! ```text
//! deduplicate_vertices → optimize_vertex_cache → optimize_overdraw → optimize_vertex_fetch
//!                        (ACMR ↓)                 (Early-Z ↓)         (vertex buffer の参照が連続に)
//! ```
//!
//! # References
//!
//! - zeux/meshoptimizer `src/overdrawoptimizer.cpp`
//! - Sander, Nehab, Barczak, "Fast Triangle Reordering for Vertex Locality and
//!   Reduced Overdraw" (SIGGRAPH 2007)
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

/// View-independent overdraw optimization (meshoptimizer `meshopt_optimizeOverdraw` 準拠)
///
/// # アルゴリズム
///
/// 1. **hard boundary**: 16 entry の FIFO vertex cache を走らせ、3 頂点すべてが
///    miss した三角形で cluster を切る (vcache 最適化後の "patch" の境界)
/// 2. **soft boundary**: 各 hard cluster の ACMR (cluster 先頭で cache を空にして測る)
///    に `threshold` を掛けた値を目標とし、cluster 内を前から走査して
///    累積 ACMR が目標以下になった時点で切る 最後の未達分は直前の cluster に併合する
/// 3. 各 cluster の **面積重み付き重心** `c` と **平均法線** `n` (面法線の和を正規化) を求め、
///    mesh 全体の重心 `m` (index 参照で数えた頂点平均) に対して
///    `key = n · (c − m)` を計算する
/// 4. `key` の大きい cluster から順に並べる (外側を向いた、中心から遠い cluster ほど
///    他の部分を隠す可能性が高いので先に描く) 同値は元の順序を保つ (stable)
/// 5. cluster 内の三角形順序は保持 (vcache 局所性維持)
///
/// # 引数
///
/// - `threshold`: vertex cache 効率の悪化の許容率 (meshoptimizer と同じ意味)
///   各 soft cluster の ACMR は、それを含む hard cluster の ACMR の `threshold` 倍以下を
///   目標に切られる `1.05` で最大 5% の悪化を許容、`1.0` は悪化を許容しない分割、
///   `0.0` 以下は soft 分割をせず hard cluster 単位でだけ並べ替える
///
/// # 制約
///
/// - `optimize_vertex_cache` 実行後に呼ぶこと (それ以外だと cluster 分割意味なし)
/// - view が特定方向に固定される use case (top-down、first-person 等) には
///   専用 view direction を渡す `optimize_overdraw_with_views` を使う
///
/// # References
///
/// - zeux/meshoptimizer `src/overdrawoptimizer.cpp`
///   (`generateHardBoundaries` / `generateSoftBoundaries` / `calculateSortData`)
/// - Sander, Nehab, Barczak, "Fast Triangle Reordering for Vertex Locality and
///   Reduced Overdraw" (SIGGRAPH 2007)
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn optimize_overdraw(mesh: &mut Mesh, threshold: f32) {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return;
    }
    let vertex_count = mesh.vertices.len();
    if mesh.indices.iter().any(|&i| i as usize >= vertex_count) {
        return; // 範囲外 index を含む mesh は並べ替えない
    }

    let hard = hard_boundaries(&mesh.indices, vertex_count, MESHOPT_OVERDRAW_CACHE_SIZE);
    let soft = soft_boundaries(
        &mesh.indices,
        vertex_count,
        &hard,
        MESHOPT_OVERDRAW_CACHE_SIZE,
        threshold,
    );
    if soft.len() <= 1 {
        return;
    }

    let keys = cluster_sort_keys(mesh, &soft);
    let mut order: Vec<usize> = (0..soft.len()).collect();
    // key の降順、同値は元の順序 (sort_by は stable)
    order.sort_by(|&a, &b| keys[b].total_cmp(&keys[a]));

    let mut new_indices = Vec::with_capacity(mesh.indices.len());
    for &ci in &order {
        let start = soft[ci];
        let end = soft.get(ci + 1).copied().unwrap_or(tri_count);
        new_indices.extend_from_slice(&mesh.indices[start * 3..end * 3]);
    }
    mesh.indices = new_indices;
}

/// meshoptimizer の overdraw optimizer が使う cache size
const MESHOPT_OVERDRAW_CACHE_SIZE: u32 = 16;

/// FIFO cache の更新 (meshoptimizer `updateCache`)、miss 数を返す
fn update_fifo(tri: &[u32], cache_size: u32, stamps: &mut [u32], timestamp: &mut u32) -> u32 {
    let mut misses = 0;
    for &v in tri {
        let s = &mut stamps[v as usize];
        if timestamp.wrapping_sub(*s) > cache_size {
            *s = *timestamp;
            *timestamp = timestamp.wrapping_add(1);
            misses += 1;
        }
    }
    misses
}

/// 3 頂点すべてが miss した三角形を cluster の先頭とする (先頭三角形は常に先頭)
fn hard_boundaries(indices: &[u32], vertex_count: usize, cache_size: u32) -> Vec<usize> {
    let mut stamps = vec![0u32; vertex_count];
    let mut timestamp = cache_size + 1;
    let mut out = Vec::new();
    for (t, tri) in indices.chunks_exact(3).enumerate() {
        let m = update_fifo(tri, cache_size, &mut stamps, &mut timestamp);
        if t == 0 || m == 3 {
            out.push(t);
        }
    }
    out
}

/// hard cluster を、累積 ACMR が `threshold × (hard cluster の ACMR)` 以下になる位置で分割
#[allow(clippy::cast_precision_loss)]
fn soft_boundaries(
    indices: &[u32],
    vertex_count: usize,
    hard: &[usize],
    cache_size: u32,
    threshold: f32,
) -> Vec<usize> {
    let tri_count = indices.len() / 3;
    let mut stamps = vec![0u32; vertex_count];
    let mut timestamp = 0u32;
    let mut out = Vec::with_capacity(hard.len());
    for (h, &start) in hard.iter().enumerate() {
        let end = hard.get(h + 1).copied().unwrap_or(tri_count);

        // cluster の ACMR を空の cache から測る
        timestamp = timestamp.wrapping_add(cache_size + 1);
        let mut cluster_misses = 0u32;
        for t in start..end {
            cluster_misses += update_fifo(
                &indices[t * 3..t * 3 + 3],
                cache_size,
                &mut stamps,
                &mut timestamp,
            );
        }
        let cluster_threshold = threshold * (cluster_misses as f32 / (end - start) as f32);

        out.push(start);
        timestamp = timestamp.wrapping_add(cache_size + 1);
        let mut running_misses = 0u32;
        let mut running_faces = 0u32;
        for t in start..end {
            running_misses += update_fifo(
                &indices[t * 3..t * 3 + 3],
                cache_size,
                &mut stamps,
                &mut timestamp,
            );
            running_faces += 1;
            if running_misses as f32 / running_faces as f32 <= cluster_threshold {
                // 目標 ACMR に達した: 次の三角形から新しい cluster
                out.push(t + 1);
                timestamp = timestamp.wrapping_add(cache_size + 1);
                running_misses = 0;
                running_faces = 0;
            }
        }
        // 最後の境界は捨てる (未達の残りを直前の cluster に併合、`end` を押した場合もこれで消える)
        if out.last() != Some(&start) {
            out.pop();
        }
    }
    out
}

/// 各 cluster の `n · (c − m)` (n: 平均法線、c: 面積重み付き重心、m: mesh 重心)
fn cluster_sort_keys(mesh: &Mesh, clusters: &[usize]) -> Vec<f32> {
    let tri_count = mesh.indices.len() / 3;
    let pos = |i: u32| mesh.vertices[i as usize].position;

    let mut mesh_centroid = Vec3::ZERO;
    for &i in &mesh.indices {
        mesh_centroid += pos(i);
    }
    #[allow(clippy::cast_precision_loss)]
    let mesh_centroid = mesh_centroid / mesh.indices.len() as f32;

    clusters
        .iter()
        .enumerate()
        .map(|(ci, &start)| {
            let end = clusters.get(ci + 1).copied().unwrap_or(tri_count);
            let mut area_sum = 0.0f32;
            let mut centroid = Vec3::ZERO;
            let mut normal = Vec3::ZERO;
            for tri in mesh.indices[start * 3..end * 3].chunks_exact(3) {
                let (p0, p1, p2) = (pos(tri[0]), pos(tri[1]), pos(tri[2]));
                let n = (p1 - p0).cross(p2 - p0);
                let area = n.length();
                centroid += (p0 + p1 + p2) * (area / 3.0);
                normal += n;
                area_sum += area;
            }
            let centroid = if area_sum == 0.0 {
                Vec3::ZERO
            } else {
                centroid / area_sum
            };
            let normal = normal.normalize_or_zero();
            normal.dot(centroid - mesh_centroid)
        })
        .collect()
}

/// View direction を明示指定する overdraw optimization
///
/// 各 view 方向で cluster 重心の深さ順位を取り、順位の和の小さい cluster から並べる
/// (cluster は vcache miss 境界で切る)
///
/// ⚠️ 向きが逆の 2 方向 (`v` と `-v`) を両方渡すと、各 cluster の 2 つの順位の和が
/// 全 cluster で同じ (`cluster 数 - 1`) になり打ち消し合う `default_view_directions`
/// (±X/±Y/±Z) をそのまま渡すと並べ替えは起きない 視点に依存しない並べ替えには
/// `optimize_overdraw` を使う
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
pub const fn default_view_directions() -> [Vec3; 6] {
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
        tris_before.sort_unstable();

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
        tris_after.sort_unstable();

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

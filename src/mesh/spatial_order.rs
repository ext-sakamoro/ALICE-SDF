//! Spatial Order Optimization — Morton (Z-order curve) code による頂点空間 sort
//!
//! 頂点を 3D 空間で近い順に並べ替えることで:
//! - BVH refit の cache miss 削減 (spatially local な頂点は同じ cache line に)
//! - Rasterizer bin culling 効率化 (画面空間で近い primitive がまとまる)
//! - GPU vertex fetch prefetch 効率化 (`optimize_vertex_fetch` と補完関係)
//!
//! # Morton code (Z-order curve)
//!
//! 3D 座標 `(x, y, z)` を各軸 10-bit に量子化し、bit interleave で 30-bit
//! Morton code を得る 空間的に近い点は Morton code も近い値になり、
//! Morton 昇順 sort で spatial locality を持った頂点列に変換できる
//!
//! ## bit interleave 例 (10-bit)
//!
//! ```text
//! x = abcdefghij (10 bit)
//! y = klmnopqrst
//! z = uvwxyz0123
//! →  ukaLvIbmwjcnxKdoyLep zM fq0N gr1 hs2 it3 ju (30 bit interleaved)
//! ```
//!
//! # 呼び出し順序
//!
//! ```text
//! optimize_spatial_order → deduplicate_vertices → optimize_vertex_cache → optimize_vertex_fetch
//!  (空間 locality ↑)       (dedup ↓)             (ACMR ↓)               (ATVR ↓)
//! ```
//!
//! # References
//!
//! - zeux/meshoptimizer §spatialorder.cpp (`meshopt_spatialSortRemap`)
//! - Karras 2012 "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees"
//!
//! Author: Moroya Sakamoto

use crate::mesh::{Mesh, Vertex};
use glam::Vec3;

/// Vertex を Morton order に並べ替え、index を remap する
///
/// # アルゴリズム
///
/// 1. 全頂点の bbox を計算
/// 2. 各頂点位置を bbox 内で 10-bit `[0, 1023]` に量子化
/// 3. `morton_3d` で 30-bit Morton code
/// 4. `(morton_code, original_index)` の list を Morton 昇順 sort
/// 5. 新 vertex buffer 構築 + index remap 適用
///
/// # 効果
///
/// - `optimize_vertex_cache` 前に呼ぶと、mesh の spatial locality が向上、
///   `optimize_vertex_cache` の効きが良くなる (連続処理する三角形群が空間的にまとまる)
/// - `mesh/bvh.rs` の BVH build 時、頂点配列走査の cache miss 削減
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn optimize_spatial_order(mesh: &mut Mesh) {
    let n = mesh.vertices.len();
    if n == 0 {
        return;
    }

    // 1. bbox 計算
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for v in &mesh.vertices {
        min = min.min(v.position);
        max = max.max(v.position);
    }
    let extent = max - min;
    let extent_safe = Vec3::new(
        if extent.x < 1e-10 { 1.0 } else { extent.x },
        if extent.y < 1e-10 { 1.0 } else { extent.y },
        if extent.z < 1e-10 { 1.0 } else { extent.z },
    );

    // 2. 頂点を Morton code + 元 index の tuple に
    let mut coded: Vec<(u32, u32)> = Vec::with_capacity(n);
    for (i, v) in mesh.vertices.iter().enumerate() {
        let normalized = (v.position - min) / extent_safe;
        // 10-bit 量子化 [0, 1023]
        let qx = (normalized.x * 1023.0).clamp(0.0, 1023.0) as u32;
        let qy = (normalized.y * 1023.0).clamp(0.0, 1023.0) as u32;
        let qz = (normalized.z * 1023.0).clamp(0.0, 1023.0) as u32;
        let code = morton_3d(qx, qy, qz);
        coded.push((code, i as u32));
    }

    // 3. Morton 昇順 sort
    coded.sort_by_key(|&(code, _)| code);

    // 4. remap 構築: old_index → new_index
    let mut remap: Vec<u32> = vec![0; n];
    for (new_idx, &(_, old_idx)) in coded.iter().enumerate() {
        remap[old_idx as usize] = new_idx as u32;
    }

    // 5. 新 vertex buffer
    let mut new_vertices: Vec<Vertex> = vec![mesh.vertices[0]; n];
    for (i, &(_, old_idx)) in coded.iter().enumerate() {
        new_vertices[i] = mesh.vertices[old_idx as usize];
    }
    mesh.vertices = new_vertices;

    // 6. index remap
    for idx in &mut mesh.indices {
        if (*idx as usize) < remap.len() {
            *idx = remap[*idx as usize];
        }
    }
}

/// 3D Morton code (30-bit): x, y, z 各 10-bit を interleave
///
/// 入力は各軸 `[0, 1023]` (10-bit)、超えた bit は truncate される
#[must_use]
#[inline]
pub const fn morton_3d(x: u32, y: u32, z: u32) -> u32 {
    expand_bits_10(x) | (expand_bits_10(y) << 1) | (expand_bits_10(z) << 2)
}

/// 10-bit 数値の bit を 3-bit 間隔で expand する
///
/// 入力 `abcdefghij` (10 bit) → 出力 `..a..b..c..d..e..f..g..h..i..j` (30 bit)
#[inline]
const fn expand_bits_10(x: u32) -> u32 {
    let x = x & 0x0000_03FF; // mask 下位 10 bit
    let x = (x | (x << 16)) & 0x030000FF;
    let x = (x | (x << 8)) & 0x0300F00F;
    let x = (x | (x << 4)) & 0x030C30C3;
    (x | (x << 2)) & 0x09249249
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;

    #[test]
    fn test_morton_3d_zero() {
        assert_eq!(morton_3d(0, 0, 0), 0);
    }

    #[test]
    fn test_morton_3d_axes() {
        // x=1, y=0, z=0 → bit 0 が set → morton = 0b001 = 1
        assert_eq!(morton_3d(1, 0, 0), 0b001);
        // x=0, y=1, z=0 → bit 1 が set → morton = 0b010 = 2
        assert_eq!(morton_3d(0, 1, 0), 0b010);
        // x=0, y=0, z=1 → bit 2 が set → morton = 0b100 = 4
        assert_eq!(morton_3d(0, 0, 1), 0b100);
    }

    #[test]
    fn test_morton_3d_max() {
        // 全 bit set (10 bit) → morton 30-bit 全 set = 0x3FFFFFFF
        let code = morton_3d(1023, 1023, 1023);
        assert_eq!(code, 0x3FFF_FFFF);
    }

    #[test]
    fn test_morton_3d_monotonic() {
        // 隣接 quantized 座標 (x=0 vs x=1) は近い Morton code に (連続 or +1)
        let a = morton_3d(0, 0, 0);
        let b = morton_3d(1, 0, 0);
        // 1 bit しか変わらない、絶対差は小さい
        assert!(b > a);
        assert!(b - a <= 8);
    }

    #[test]
    fn test_expand_bits_10_zero() {
        assert_eq!(expand_bits_10(0), 0);
    }

    #[test]
    fn test_expand_bits_10_lsb() {
        // 最下位 bit だけ → 3-bit interleave で bit 0 だけ set
        assert_eq!(expand_bits_10(1), 1);
    }

    #[test]
    fn test_expand_bits_10_msb() {
        // 最上位 (bit 9) のみ set → interleave 後 bit 27 に来る
        assert_eq!(expand_bits_10(1 << 9), 1 << 27);
    }

    #[test]
    fn test_spatial_order_empty_mesh() {
        let mut mesh = Mesh::new();
        optimize_spatial_order(&mut mesh);
        assert!(mesh.vertices.is_empty());
    }

    #[test]
    fn test_spatial_order_preserves_indexed_geometry() {
        // 頂点を並べ替えても index 経由でアクセスした position は保存される
        let sphere = SdfNode::sphere(1.0);
        let mut mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );

        // 元 mesh の三角形の position を記録 (三角形ごとに 3 頂点 position の hash 集合)
        let tri_count = mesh.triangle_count();
        let mut before_tri_positions: Vec<[[i32; 3]; 3]> = (0..tri_count)
            .map(|t| {
                let mut positions = [[0i32; 3]; 3];
                for k in 0..3 {
                    let v = &mesh.vertices[mesh.indices[t * 3 + k] as usize];
                    positions[k] = [
                        (v.position.x * 1000.0) as i32,
                        (v.position.y * 1000.0) as i32,
                        (v.position.z * 1000.0) as i32,
                    ];
                }
                positions.sort();
                positions
            })
            .collect();
        before_tri_positions.sort();

        optimize_spatial_order(&mut mesh);

        // spatial sort 後も三角形数保存
        assert_eq!(mesh.triangle_count(), tri_count);

        // 三角形の頂点 position 集合が保存されているか
        let mut after_tri_positions: Vec<[[i32; 3]; 3]> = (0..tri_count)
            .map(|t| {
                let mut positions = [[0i32; 3]; 3];
                for k in 0..3 {
                    let v = &mesh.vertices[mesh.indices[t * 3 + k] as usize];
                    positions[k] = [
                        (v.position.x * 1000.0) as i32,
                        (v.position.y * 1000.0) as i32,
                        (v.position.z * 1000.0) as i32,
                    ];
                }
                positions.sort();
                positions
            })
            .collect();
        after_tri_positions.sort();

        assert_eq!(
            before_tri_positions, after_tri_positions,
            "spatial sort で三角形集合 (position ベース) が変わってはいけない"
        );
    }

    #[test]
    fn test_spatial_order_morton_ascending() {
        // sort 後、隣接頂点の Morton code は昇順 (非厳密増加、同値許容)
        let sphere = SdfNode::sphere(1.0);
        let mut mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );

        optimize_spatial_order(&mut mesh);

        // 新 mesh の bbox 再計算
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &mesh.vertices {
            min = min.min(v.position);
            max = max.max(v.position);
        }
        let extent = max - min;
        let extent_safe = Vec3::new(
            if extent.x < 1e-10 { 1.0 } else { extent.x },
            if extent.y < 1e-10 { 1.0 } else { extent.y },
            if extent.z < 1e-10 { 1.0 } else { extent.z },
        );

        // 隣接頂点の Morton code を計算、昇順であること
        for i in 1..mesh.vertices.len() {
            let prev = &mesh.vertices[i - 1].position;
            let curr = &mesh.vertices[i].position;
            let p_norm = (*prev - min) / extent_safe;
            let c_norm = (*curr - min) / extent_safe;
            let p_code = morton_3d(
                (p_norm.x * 1023.0).clamp(0.0, 1023.0) as u32,
                (p_norm.y * 1023.0).clamp(0.0, 1023.0) as u32,
                (p_norm.z * 1023.0).clamp(0.0, 1023.0) as u32,
            );
            let c_code = morton_3d(
                (c_norm.x * 1023.0).clamp(0.0, 1023.0) as u32,
                (c_norm.y * 1023.0).clamp(0.0, 1023.0) as u32,
                (c_norm.z * 1023.0).clamp(0.0, 1023.0) as u32,
            );
            assert!(
                c_code >= p_code,
                "Morton code 非昇順: vertex {}: prev={}, curr={}",
                i,
                p_code,
                c_code
            );
        }
    }

    #[test]
    fn test_spatial_order_single_vertex() {
        let mut mesh = Mesh {
            vertices: vec![Vertex::new(Vec3::ZERO, Vec3::Y)],
            indices: vec![],
        };
        optimize_spatial_order(&mut mesh);
        assert_eq!(mesh.vertices.len(), 1);
    }

    #[test]
    fn test_spatial_order_flat_mesh() {
        // Y 方向 extent = 0 の平面 mesh (bbox 潰れ) でも panic せず
        let v = |x: f32, z: f32| Vertex::new(Vec3::new(x, 0.5, z), Vec3::Y);
        let mut mesh = Mesh {
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.5, 1.0)],
            indices: vec![0, 1, 2],
        };
        optimize_spatial_order(&mut mesh);
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.indices.len(), 3);
    }
}

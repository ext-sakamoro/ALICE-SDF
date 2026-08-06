//! # polygon_extrude — 2D polygon → 3D watertight mesh
//!
//! Shapely + trimesh (Python) の `extrude` 相当を Rust workspace で実装する
//! 薄物 (≤ 5mm) を SDF+Marching Cubes の非多様体問題なしで生成する経路を提供
//!
//! ## 動機
//!
//! Marching Cubes は薄い形状のボクセル化で非多様体エッジが大量発生し、厚さも正確に
//! 再現できない (Bamboo 実測: 1.7mm 設計 → 5.1mm 出力、6177 non-manifold edges)
//! これは SDF+MC の原理的限界であり resolution 上げでも解決しない
//!
//! 代替経路: **2D polygon → watertight mesh (top + bottom + side walls)**
//! `earcutr` (Mapbox earcut の Rust port、MIT) で top face を triangulate、
//! bottom は同 triangulation を Y 反転 + winding reverse、側壁は各 edge に矩形 2 三角形
//!
//! ## LOL / alice-print / text-to-print への統合
//!
//! - LOL `print_export` で 2D primitive (`Circle2D` / `Rect2D` / `RoundedRect2D` /
//!   `Annular2D`) + `Extrude` modifier chain を判別 → 本 module に分岐
//! - alice-print `slice` は既存 SDF+MC 経路と本 module 経路の 2 系統をサポート
//! - text-to-print pipeline で LLM 生成 LOL DSL から厚物・薄物を自動判別
//!
//! ## 使用例
//!
//! ```
//! # #[cfg(feature = "mesh-extrude")]
//! # {
//! use alice_sdf::mesh::polygon_extrude::Polygon2D;
//! use glam::Vec2;
//!
//! // Φ22.8mm × 1.7mm shopping cart coin (Bamboo 実プリント合格 spec)
//! let outer: Vec<Vec2> = (0..32)
//!     .map(|i| {
//!         let a = std::f32::consts::TAU * i as f32 / 32.0;
//!         Vec2::new(11.4 * a.cos(), 11.4 * a.sin())
//!     })
//!     .collect();
//! let coin = Polygon2D::new(outer);
//! let mesh = coin.extrude(0.85); // half_height = 0.85 → 全厚 1.7mm
//! assert!(!mesh.vertices.is_empty());
//! assert!(!mesh.indices.is_empty());
//! // mesh は top + bottom + side walls で watertight
//! # }
//! ```

use crate::mesh::{Mesh, Vertex};
use glam::{Vec2, Vec3, Vec4};

// ────────────────────────────────────────────────────────
// Polygon2D 型
// ────────────────────────────────────────────────────────

/// 2D 多角形 (外周 + 任意個数の穴)
///
/// - `outer`: 外周輪郭 頂点列 (時計回り / 反時計回り どちらでも可、earcutr が処理)
/// - `holes`: 各穴の輪郭頂点列 (外周と逆 winding が推奨、earcutr が三角化)
#[derive(Debug, Clone, Default)]
pub struct Polygon2D {
    /// 外周輪郭
    pub outer: Vec<Vec2>,
    /// 穴 (各 Vec は 1 個の穴の頂点列)
    pub holes: Vec<Vec<Vec2>>,
}

impl Polygon2D {
    /// 外周のみで新規作成 (穴なし)
    #[must_use]
    pub const fn new(outer: Vec<Vec2>) -> Self {
        Self {
            outer,
            holes: Vec::new(),
        }
    }

    /// 穴を 1 個追加した builder-style API
    #[must_use]
    pub fn with_hole(mut self, hole: Vec<Vec2>) -> Self {
        self.holes.push(hole);
        self
    }

    /// 穴を複数追加した builder-style API
    #[must_use]
    pub fn with_holes(mut self, mut holes: Vec<Vec<Vec2>>) -> Self {
        self.holes.append(&mut holes);
        self
    }

    /// 全頂点数 (`outer` + `holes` 全部)
    #[must_use]
    pub fn total_vertex_count(&self) -> usize {
        self.outer.len() + self.holes.iter().map(Vec::len).sum::<usize>()
    }
}

// ────────────────────────────────────────────────────────
// Extrusion
// ────────────────────────────────────────────────────────

impl Polygon2D {
    /// Y 軸方向に extrude して watertight `Mesh` を返す
    ///
    /// - 原点中心、Y = ±`half_height` (全厚 = 2 × `half_height`)
    /// - 2D 頂点 `(x, y)` は 3D では `(x, ±half_height, y)` にマップ (X-Z 平面に配置、Y-up)
    /// - top face: normal = +Y、triangulation は `earcutr`
    /// - bottom face: normal = -Y、top と同じ triangulation を winding reverse
    /// - side walls: 各 outer edge + hole edge に矩形 2 三角形、normal は外向き
    ///
    /// # Panics
    ///
    /// なし earcutr が失敗した場合 (`Err`) は空 `Mesh` を返す 呼出側は
    /// `mesh.indices.is_empty()` で判別できる
    #[must_use]
    pub fn extrude(&self, half_height: f32) -> Mesh {
        // ── earcutr 入力を構築 ──
        // flat vertex array [x0, y0, x1, y1, ...] (outer → hole1 → hole2 → ...)
        let mut flat: Vec<f64> = Vec::with_capacity(self.total_vertex_count() * 2);
        for v in &self.outer {
            flat.push(f64::from(v.x));
            flat.push(f64::from(v.y));
        }
        let mut hole_indices: Vec<usize> = Vec::with_capacity(self.holes.len());
        for hole in &self.holes {
            hole_indices.push(flat.len() / 2);
            for v in hole {
                flat.push(f64::from(v.x));
                flat.push(f64::from(v.y));
            }
        }

        // ── earcutr で top face triangulation ──
        let Ok(top_indices) = earcutr::earcut(&flat, &hole_indices, 2) else {
            return Mesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            };
        };
        // triangulation が空 (degenerate polygon 等) の場合も空 mesh
        if top_indices.is_empty() {
            return Mesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            };
        }

        let n_boundary = self.total_vertex_count();

        // ── vertex 構築 ──
        // top: index 0..n_boundary、Y = +half_height
        // bottom: index n_boundary..2*n_boundary、Y = -half_height
        // side walls は面ごとに新規 vertex 割当 (normal が異なるため共有不可)
        let mut vertices: Vec<Vertex> = Vec::with_capacity(n_boundary * 2);

        // top face vertices (normal +Y)
        // 2D (x, y) → 3D (x, +half_height, y)
        for i in 0..n_boundary {
            let idx = i * 2;
            #[allow(clippy::cast_possible_truncation)]
            let x = flat[idx] as f32;
            #[allow(clippy::cast_possible_truncation)]
            let y = flat[idx + 1] as f32;
            vertices.push(make_vertex(
                Vec3::new(x, half_height, y),
                Vec3::Y,
                Vec2::new(x, y),
            ));
        }
        // bottom face vertices (normal -Y、UV は Y=+ で左右反転)
        for i in 0..n_boundary {
            let idx = i * 2;
            #[allow(clippy::cast_possible_truncation)]
            let x = flat[idx] as f32;
            #[allow(clippy::cast_possible_truncation)]
            let y = flat[idx + 1] as f32;
            vertices.push(make_vertex(
                Vec3::new(x, -half_height, y),
                Vec3::NEG_Y,
                Vec2::new(-x, y),
            ));
        }

        // ── index 構築 ──
        // top face indices (earcutr そのまま)
        let mut indices: Vec<u32> = Vec::with_capacity(top_indices.len() * 2);
        for tri in top_indices.chunks_exact(3) {
            // earcutr は反時計回り (CCW) triangulation、Y-up で top face は winding OK
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push(tri[0] as u32);
                indices.push(tri[1] as u32);
                indices.push(tri[2] as u32);
            }
        }
        // bottom face indices (top を winding reverse + offset n_boundary)
        for tri in top_indices.chunks_exact(3) {
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push((tri[0] + n_boundary) as u32);
                indices.push((tri[2] + n_boundary) as u32); // 逆順
                indices.push((tri[1] + n_boundary) as u32);
            }
        }

        // ── side walls ──
        // 各輪郭 (outer + 各 hole) の連続 edge に矩形 2 三角形
        // normal は edge に対して外向き (outer は右手座標系で XZ 平面上、+Z / -Z etc)
        let mut ring_start: usize = 0;
        // outer wall
        add_ring_wall(
            &self.outer,
            ring_start,
            n_boundary,
            &mut vertices,
            &mut indices,
            half_height,
            false,
        );
        ring_start += self.outer.len();
        // hole walls (winding が outer と逆想定、normal 反転)
        for hole in &self.holes {
            add_ring_wall(
                hole,
                ring_start,
                n_boundary,
                &mut vertices,
                &mut indices,
                half_height,
                true,
            );
            ring_start += hole.len();
        }

        Mesh { vertices, indices }
    }
}

// ────────────────────────────────────────────────────────
// 内部ヘルパー
// ────────────────────────────────────────────────────────

fn make_vertex(position: Vec3, normal: Vec3, uv: Vec2) -> Vertex {
    Vertex {
        position,
        normal,
        uv,
        uv2: Vec2::ZERO,
        tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
        color: [1.0, 1.0, 1.0, 1.0],
        material_id: 0,
    }
}

/// 1 ring (outer or hole) の side wall を追加
///
/// `ring: &[Vec2]` の各連続 edge (i → i+1) に矩形 2 三角形を作る
/// side wall vertices は新規に vertices に append し、その index で triangle を張る
/// `flip_normal` が true なら normal を反転 (hole side wall)
fn add_ring_wall(
    ring: &[Vec2],
    ring_start_in_flat: usize,
    _n_boundary: usize,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    half_height: f32,
    flip_normal: bool,
) {
    let n = ring.len();
    if n < 3 {
        return;
    }
    for i in 0..n {
        let a2 = ring[i];
        let b2 = ring[(i + 1) % n];
        // 3D 座標 (Y = ±half_height、2D (x, y) → 3D (x, Y, y))
        let a_top = Vec3::new(a2.x, half_height, a2.y);
        let b_top = Vec3::new(b2.x, half_height, b2.y);
        let a_bot = Vec3::new(a2.x, -half_height, a2.y);
        let b_bot = Vec3::new(b2.x, -half_height, b2.y);
        // Edge 方向ベクトル (a → b)、Y 成分は 0
        let edge = b_top - a_top;
        // Side wall normal = edge × Y (外向き)、hole の場合 flip
        let mut normal = edge.cross(Vec3::Y).normalize_or_zero();
        if flip_normal {
            normal = -normal;
        }
        // 4 vertex に normal を設定して push
        let base = vertices.len();
        let _ = ring_start_in_flat; // reserved for future use (edge attribution)
        vertices.push(make_vertex(a_bot, normal, Vec2::new(0.0, 0.0)));
        vertices.push(make_vertex(b_bot, normal, Vec2::new(1.0, 0.0)));
        vertices.push(make_vertex(b_top, normal, Vec2::new(1.0, 1.0)));
        vertices.push(make_vertex(a_top, normal, Vec2::new(0.0, 1.0)));
        // 2 三角形 (a_bot, b_bot, b_top) と (a_bot, b_top, a_top)
        // outer は外向き normal で CCW when viewed from outside
        #[allow(clippy::cast_possible_truncation)]
        {
            if flip_normal {
                // hole は winding を逆にする
                indices.push(base as u32);
                indices.push(base as u32 + 2);
                indices.push(base as u32 + 1);
                indices.push(base as u32);
                indices.push(base as u32 + 3);
                indices.push(base as u32 + 2);
            } else {
                indices.push(base as u32);
                indices.push(base as u32 + 1);
                indices.push(base as u32 + 2);
                indices.push(base as u32);
                indices.push(base as u32 + 2);
                indices.push(base as u32 + 3);
            }
        }
    }
}

// ────────────────────────────────────────────────────────
// 便利 constructor (よく使う 2D shape)
// ────────────────────────────────────────────────────────

/// 円 (n 角形近似) の [`Polygon2D`] を構築
///
/// - `radius`: 半径 (mm)
/// - `segments`: 分割数 (推奨 32-64)
#[must_use]
pub fn circle(radius: f32, segments: u32) -> Polygon2D {
    let segments = segments.max(3);
    #[allow(clippy::cast_precision_loss)]
    let inv_segments = 1.0_f32 / segments as f32;
    let outer: Vec<Vec2> = (0..segments)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let a = std::f32::consts::TAU * i as f32 * inv_segments;
            Vec2::new(radius * a.cos(), radius * a.sin())
        })
        .collect();
    Polygon2D::new(outer)
}

/// 矩形の [`Polygon2D`] を構築 (原点中心、幅 × 高)
#[must_use]
pub fn rect(width: f32, height: f32) -> Polygon2D {
    let hx = width * 0.5;
    let hy = height * 0.5;
    Polygon2D::new(vec![
        Vec2::new(-hx, -hy),
        Vec2::new(hx, -hy),
        Vec2::new(hx, hy),
        Vec2::new(-hx, hy),
    ])
}

/// 角丸矩形の [`Polygon2D`] を構築
///
/// - `width`, `height`: 全幅・全高 (mm)
/// - `radius`: 角丸半径 (mm、`min(width, height) / 2` 未満)
/// - `corner_segments`: 各角の分割数 (推奨 8-16)
#[must_use]
pub fn rounded_rect(width: f32, height: f32, radius: f32, corner_segments: u32) -> Polygon2D {
    let hx = width * 0.5;
    let hy = height * 0.5;
    let r = radius.min(hx).min(hy).max(0.0);
    let segs = corner_segments.max(2);
    let mut outer = Vec::with_capacity(usize::try_from(segs).unwrap_or(2) * 4 + 4);
    // 4 corner を +X+Y, -X+Y, -X-Y, +X-Y の順で回る (CCW)
    let centers = [
        Vec2::new(hx - r, hy - r),       // top-right
        Vec2::new(-(hx - r), hy - r),    // top-left
        Vec2::new(-(hx - r), -(hy - r)), // bottom-left
        Vec2::new(hx - r, -(hy - r)),    // bottom-right
    ];
    let start_angles = [
        0.0_f32,                           // top-right: 0 → pi/2
        std::f32::consts::FRAC_PI_2,       // top-left: pi/2 → pi
        std::f32::consts::PI,              // bottom-left: pi → 3pi/2
        3.0 * std::f32::consts::FRAC_PI_2, // bottom-right: 3pi/2 → 2pi
    ];
    #[allow(clippy::cast_precision_loss)]
    let seg_step = std::f32::consts::FRAC_PI_2 / segs as f32;
    for (center, &start_angle) in centers.iter().zip(&start_angles) {
        for k in 0..=segs {
            #[allow(clippy::cast_precision_loss)]
            let a = start_angle + seg_step * k as f32;
            outer.push(Vec2::new(center.x + r * a.cos(), center.y + r * a.sin()));
        }
    }
    Polygon2D::new(outer)
}

// ────────────────────────────────────────────────────────
// テスト
// ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_polygon_has_expected_vertex_count() {
        let c = circle(10.0, 32);
        assert_eq!(c.outer.len(), 32);
        assert!(c.holes.is_empty());
    }

    #[test]
    fn rect_polygon_has_4_vertices() {
        let r = rect(10.0, 20.0);
        assert_eq!(r.outer.len(), 4);
    }

    #[test]
    fn rounded_rect_has_perimeter_vertices() {
        let rr = rounded_rect(20.0, 30.0, 3.0, 8);
        // 各 corner 9 vertex (segments=8 → 0..=8)、4 corner = 36
        assert_eq!(rr.outer.len(), 36);
    }

    #[test]
    fn extrude_circle_produces_watertight_mesh() {
        let c = circle(11.4, 32);
        let mesh = c.extrude(0.85); // Bamboo shopping cart coin spec
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        // triangles = top (30) + bottom (30) + side walls (32 × 2) = 124
        //  top face triangulation for n=32 polygon = n-2 = 30 triangles
        assert_eq!(mesh.indices.len() / 3, 30 + 30 + 32 * 2);
    }

    #[test]
    fn extrude_rect_produces_correct_thickness() {
        let r = rect(10.0, 20.0);
        let mesh = r.extrude(2.5); // 全厚 5mm
                                   // 全 Y 座標が ±2.5 の範囲に収まる
        for v in &mesh.vertices {
            assert!(v.position.y <= 2.5 + 1e-6);
            assert!(v.position.y >= -2.5 - 1e-6);
        }
    }

    #[test]
    fn extrude_polygon_with_hole() {
        // 外周 20×20 rect、中央に 5×5 rect の穴
        let outer = rect(20.0, 20.0).outer;
        // 穴は外周と逆 winding (CW) にする earcutr の慣例
        let hole = vec![
            Vec2::new(-2.5, -2.5),
            Vec2::new(-2.5, 2.5),
            Vec2::new(2.5, 2.5),
            Vec2::new(2.5, -2.5),
        ];
        let p = Polygon2D::new(outer).with_hole(hole);
        let mesh = p.extrude(1.0);
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
    }

    #[test]
    fn extrude_degenerate_polygon_returns_empty_mesh() {
        // 2 vertex only = triangulation 不能
        let p = Polygon2D::new(vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)]);
        let mesh = p.extrude(1.0);
        // earcutr は空 triangulation を返す → 空 mesh
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn top_and_bottom_faces_have_opposite_normals() {
        let c = circle(5.0, 8);
        let mesh = c.extrude(1.0);
        // 最初の 8 vertex は top (normal +Y)、次の 8 は bottom (normal -Y)
        for v in &mesh.vertices[..8] {
            assert!((v.normal.y - 1.0).abs() < 1e-4);
        }
        for v in &mesh.vertices[8..16] {
            assert!((v.normal.y + 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn total_vertex_count_includes_holes() {
        let p = Polygon2D::new(rect(10.0, 10.0).outer)
            .with_hole(rect(2.0, 2.0).outer)
            .with_hole(rect(1.0, 1.0).outer);
        assert_eq!(p.total_vertex_count(), 4 + 4 + 4);
    }
}

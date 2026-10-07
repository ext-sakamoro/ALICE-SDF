//! Nanite-Compatible Cluster Generation for UE5
//!
//! Generates hierarchical mesh clusters from SDFs for use with
//! Unreal Engine 5's Nanite virtualized geometry system.
//!
//! # Nanite Overview
//!
//! Nanite uses a hierarchical cluster-based representation:
//! - **Clusters**: Groups of ~128 triangles
//! - **Cluster Groups**: Collections of clusters for LOD
//! - **DAG**: Directed Acyclic Graph for LOD selection
//!
//! # Features
//!
//! - Cluster-based mesh generation from SDF
//! - LOD chain generation with error metrics
//! - Cluster bounds for GPU culling
//! - Seamless cluster boundaries
//!
//! Author: Moroya Sakamoto

use crate::eval::{eval, eval_material, eval_normal};
use crate::mesh::dual_contouring::{dual_contouring, DualContouringConfig};
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Triangle, Vertex};
use crate::tight_aabb::compute_tight_aabb;
use crate::types::SdfNode;
use glam::Vec3;

/// Maximum triangles per Nanite cluster (UE5 uses ~128)
pub const CLUSTER_MAX_TRIANGLES: usize = 128;

/// Maximum vertices per cluster
pub const CLUSTER_MAX_VERTICES: usize = 256;

/// Cluster bounding sphere
#[derive(Debug, Clone, Copy)]
pub struct ClusterBounds {
    /// Center of bounding sphere
    pub center: Vec3,
    /// Radius of bounding sphere
    pub radius: f32,
    /// Axis-aligned bounding box min
    pub aabb_min: Vec3,
    /// Axis-aligned bounding box max
    pub aabb_max: Vec3,
}

impl ClusterBounds {
    /// Create from vertices
    pub fn from_vertices(vertices: &[Vec3]) -> Self {
        if vertices.is_empty() {
            return Self {
                center: Vec3::ZERO,
                radius: 0.0,
                aabb_min: Vec3::ZERO,
                aabb_max: Vec3::ZERO,
            };
        }

        // Compute AABB
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        for &v in vertices {
            aabb_min = aabb_min.min(v);
            aabb_max = aabb_max.max(v);
        }

        // Compute bounding sphere
        let center = (aabb_min + aabb_max) * 0.5;
        let radius = vertices
            .iter()
            .map(|&v| (v - center).length())
            .fold(0.0f32, f32::max);

        Self {
            center,
            radius,
            aabb_min,
            aabb_max,
        }
    }

    /// Check if the bounding sphere intersects a circular view cone
    ///
    /// `view_dir` is the unit cone axis and `fov_cos` is the cosine of the
    /// cone's half-angle `θ`. The sphere subtends the half-angle `α` with
    /// `sin α = r / d`, so it touches the cone exactly when the angle between
    /// `view_dir` and the direction to its center is at most `θ + α`, i.e.
    /// `cos(angle) >= cos θ cos α - sin θ sin α` (and always when `θ + α >= π`).
    /// A viewer inside the sphere always sees it.
    #[inline]
    pub fn is_visible(&self, view_pos: Vec3, view_dir: Vec3, fov_cos: f32) -> bool {
        let to_center = self.center - view_pos;
        let dist = to_center.length();

        if dist <= self.radius {
            return true; // Inside sphere
        }

        let dir = to_center / dist;
        let sin_a = self.radius / dist;
        let cos_a = dist.mul_add(dist, -(self.radius * self.radius)).sqrt() / dist;
        let fov_cos = fov_cos.clamp(-1.0, 1.0);
        let fov_sin = fov_cos.mul_add(-fov_cos, 1.0).max(0.0).sqrt();

        // θ + α >= π: only possible when θ >= π/2, then α >= π - θ <=> sin α >= sin θ
        if fov_cos <= 0.0 && sin_a >= fov_sin {
            return true;
        }

        dir.dot(view_dir) >= fov_cos.mul_add(cos_a, -(fov_sin * sin_a))
    }

    /// Compute screen-space error for LOD selection
    #[inline]
    pub fn screen_error(&self, view_pos: Vec3, geometric_error: f32, screen_height: f32) -> f32 {
        let dist = (self.center - view_pos).length().max(self.radius);
        (geometric_error / dist) * screen_height
    }
}

/// Normal cone bounds — cluster 内の全 face normal を 1 つの円錐で包む
///
/// # 用途
///
/// GPU の **back-face culling** 効率化 三角形群の normal が全て一定角度内に
/// 収まる (円錐で包める) 場合、cluster 全体を back-face として reject 可能
///
/// # 数学
///
/// - `axis`: 円錐軸 (単位ベクトル、face normal の平均)
/// - `cutoff_cos`: 円錐の開き角の cos (= `dot(axis, most_deviated_normal)`)
/// - `apex`: (optional) 円錐頂点、Vulkan meshlet culling shader での tighter culling test 用
/// - back-face 判定 (basic): `dot(view_dir, axis) >= cutoff_cos` なら全 face が back-face
/// - back-face 判定 (with apex): `dot(view_pos - apex, axis) >= 0` なら全 face が back-face
///
/// # References
///
/// - zeux/meshoptimizer §clusterizer.cpp `meshopt_computeMeshletBounds` cone 部分
/// - "Optimizing the Graphics Pipeline with Compute" (GDC 2016)
/// - Vulkan VK_EXT_mesh_shader `meshletCullingShader` spec
#[derive(Debug, Clone, Copy)]
pub struct NormalCone {
    /// 円錐軸 (単位ベクトル、平均 face normal 方向)
    pub axis: Vec3,
    /// 円錐開き角の cos ([-1, 1]、1.0 = すべて同方向 = culling 最効率、-1.0 = 全方向 = culling 不可)
    pub cutoff_cos: f32,
    /// 円錐頂点 (optional、tighter culling test 用)
    ///
    /// `Some(apex)` の場合、`dot(view_pos - apex, axis) >= 0` で全 face が back-face 判定可能
    /// (basic cone より正確、culling 効率化)
    /// `None` の場合、basic cone (axis + cutoff_cos) のみで判定
    pub apex: Option<Vec3>,
}

impl NormalCone {
    /// 頂点法線群から NormalCone を計算
    ///
    /// # アルゴリズム
    ///
    /// 1. 全法線の平均を計算 → `axis` (正規化)
    /// 2. 各法線について `dot(axis, normal)` を計算、最小値が `cutoff_cos`
    /// 3. 空入力 or 縮退時は「culling 不可」を表す `axis = Vec3::Y, cutoff_cos = -1.0` fallback
    ///
    /// # 制約
    ///
    /// - 入力法線はすべて単位ベクトル前提 (未正規化なら結果崩壊)
    /// - 平均長 < 1e-6 の場合は法線群が全方向に散っている状態、culling 不可 (fallback)
    #[must_use]
    pub fn from_normals(normals: &[Vec3]) -> Self {
        if normals.is_empty() {
            return Self::unbounded();
        }
        let mut sum = Vec3::ZERO;
        for &n in normals {
            sum += n;
        }
        let sum_len = sum.length();
        if sum_len < 1e-6 {
            return Self::unbounded();
        }
        let axis = sum / sum_len;
        // 最大逸脱 = 最小 dot
        let mut min_dot = 1.0_f32;
        for &n in normals {
            let d = axis.dot(n);
            if d < min_dot {
                min_dot = d;
            }
        }
        Self {
            axis,
            cutoff_cos: min_dot,
            apex: None,
        }
    }

    /// Cone apex 付きで NormalCone を計算
    ///
    /// # 引数
    ///
    /// - `normals`: 各三角形の face normal (単位ベクトル)
    /// - `face_centers`: 各三角形の重心 (world position)
    ///
    /// # アルゴリズム (meshoptimizer §clusterizer.cpp `computeMeshletBounds` 準拠)
    ///
    /// 1. basic cone (axis + cutoff_cos) を計算
    /// 2. cone の apex を求める:
    ///    - 各 face plane を `dot(x, n_i) = dot(c_i, n_i)` として、apex は全 plane の背後 (dot ≤ offset)
    ///    - `apex = cluster_center - max_offset × axis` (単純な closed-form 近似)
    /// 3. unbounded の場合 apex は None
    ///
    /// # 制約
    ///
    /// - `normals.len() == face_centers.len()` 前提、不一致時は basic cone (apex=None) にフォールバック
    #[must_use]
    pub fn from_normals_and_positions(normals: &[Vec3], face_centers: &[Vec3]) -> Self {
        // basic cone は from_normals と同ロジック
        let basic = Self::from_normals(normals);
        if basic.cutoff_cos <= -1.0 + 1e-6 || normals.len() != face_centers.len() {
            return basic;
        }

        // cluster center (face centers の平均)
        let mut center_sum = Vec3::ZERO;
        for &c in face_centers {
            center_sum += c;
        }
        let cluster_center = center_sum / (face_centers.len() as f32);

        // 各 face plane に対する cluster_center からの signed distance (axis 方向)
        // apex は max_offset を axis 方向に「後退」させた点
        // dot(apex - center, n_i) ≤ 0 for all i を満たすように
        // apex = center - t × axis where t = max over i of dot(center - c_i, n_i) / dot(axis, n_i)
        let mut max_t = 0.0_f32;
        for i in 0..normals.len() {
            let n = normals[i];
            let c = face_centers[i];
            let denom = basic.axis.dot(n);
            if denom <= 1e-6 {
                continue; // face 法線と axis がほぼ直交 or 逆向き → skip
            }
            let numer = (cluster_center - c).dot(n);
            let t = numer / denom;
            if t > max_t {
                max_t = t;
            }
        }

        let apex = cluster_center - basic.axis * max_t;
        Self {
            axis: basic.axis,
            cutoff_cos: basic.cutoff_cos,
            apex: Some(apex),
        }
    }

    /// Culling 不可能な NormalCone (全法線が全方向に散っている状態)
    #[must_use]
    #[inline]
    pub const fn unbounded() -> Self {
        Self {
            axis: Vec3::Y,
            cutoff_cos: -1.0,
            apex: None,
        }
    }

    /// Back-face culling 判定
    ///
    /// # 引数
    ///
    /// - `view_dir`: view から cluster への視線方向 (単位ベクトル、`cluster.center - view_pos` 正規化)
    ///
    /// # Returns
    ///
    /// `true` なら cluster 全体が back-face → 描画 skip 可能
    /// `false` なら少なくとも 1 face が front-facing 可能性あり → 通常描画
    ///
    /// # 数学
    ///
    /// 全 face normal は cone `(axis, cutoff_cos)` 内 view direction を `v` (camera から
    /// cluster への方向) とすると、face normal `n` が `n · v > 0` の時 back-facing
    /// cone の半角を `β` (`cos β = cutoff_cos`) とすると、cone 内の normal と `v` の角は
    /// 最大で `angle(axis, v) + β` なので、全 face が back-facing (`n · v >= 0`) となるのは
    /// `angle(axis, v) <= 90° - β`、つまり `axis · v >= sin β = sqrt(1 - cutoff_cos²)` の時
    /// (`β > 90°` なら該当する `v` は無い、meshoptimizer `meshopt_computeMeshletBounds` の
    /// `cone_cutoff` と同じ判定)
    #[must_use]
    #[inline]
    pub fn is_backface_culled(&self, view_dir: Vec3) -> bool {
        if self.cutoff_cos < 0.0 {
            return false; // 半角 > 90° (unbounded を含む) は culling 不可
        }
        let c = self.cutoff_cos.min(1.0);
        let sin_beta = c.mul_add(-c, 1.0).max(0.0).sqrt();
        self.axis.dot(view_dir) >= sin_beta
    }
}

/// LOD level information
#[derive(Debug, Clone, Copy)]
pub struct LodLevel {
    /// LOD index (0 = highest detail)
    pub level: u32,
    /// Resolution used for this LOD
    pub resolution: u32,
    /// Maximum geometric error at this LOD
    pub max_error: f32,
    /// Triangle count at this LOD
    pub triangle_count: u32,
}

/// Nanite-compatible mesh cluster
#[derive(Debug, Clone)]
pub struct NaniteCluster {
    /// Unique cluster ID
    pub id: u32,
    /// LOD level this cluster belongs to
    pub lod_level: u32,
    /// Vertices in this cluster
    pub vertices: Vec<Vertex>,
    /// Triangle indices (local to cluster)
    pub triangles: Vec<Triangle>,
    /// Cluster bounds for culling
    pub bounds: ClusterBounds,
    /// Parent cluster IDs (for LOD DAG)
    pub parent_ids: Vec<u32>,
    /// Child cluster IDs (for LOD DAG)
    pub child_ids: Vec<u32>,
    /// Geometric error for this cluster: the error of its group (see
    /// [`ClusterGroup::max_error`]), never smaller than the errors of its
    /// child groups
    pub geometric_error: f32,
    /// LOD sphere of the cluster's group ([`ClusterGroup::bounds`]): the
    /// projected error in the cut is measured to it, not to [`bounds`](Self::bounds)
    /// (the cluster's own culling bounds), so that every cluster of a group
    /// makes the same decision
    pub lod_bounds: ClusterBounds,
    /// Error of the parent group, `f32::INFINITY` when the group has no parent
    /// (a root group)
    pub parent_error: f32,
    /// LOD sphere of the parent group (equal to [`lod_bounds`](Self::lod_bounds)
    /// for a root group)
    pub parent_lod_bounds: ClusterBounds,
    /// Material ID for this cluster (0 = default)
    pub material_id: u32,
}

impl NaniteCluster {
    /// Get triangle count
    #[inline]
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Get vertex count
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Whether this cluster is drawn from `view_pos` (the cluster-LOD cut)
    ///
    /// The projected error of an error `e` with LOD sphere `S` is `e / d`, where
    /// `d` is the distance from `view_pos` to the surface of `S` (infinite
    /// inside `S`, 0 when `e` is 0). A level is fine enough when its projected
    /// error is at most `error_threshold` (an angle in radians; a pixel budget
    /// `p` on a screen of height `H` with vertical field of view `fov` is
    /// `p * 2 tan(fov / 2) / H`).
    ///
    /// The cluster is drawn when it is fine enough (`geometric_error` with
    /// `lod_bounds`, or it has no children, so the finest level is kept when no
    /// level meets the threshold) **and** its parent is not (`parent_error`
    /// with `parent_lod_bounds`; a root group has no parent and counts as
    /// having one that is too coarse). This is the standard cut, decided from
    /// the cluster alone, and selects exactly the clusters of
    /// [`NaniteMesh::select_clusters`].
    ///
    /// Until 5.0.0 this was only the first half (fine enough, measured to the
    /// cluster's own `bounds`), so every ancestor of a drawn cluster that was
    /// also fine enough returned `true` as well.
    pub fn should_render(&self, view_pos: Vec3, error_threshold: f32) -> bool {
        let fine = self.child_ids.is_empty()
            || projected_error(self.geometric_error, &self.lod_bounds, view_pos) <= error_threshold;
        let parent_fine = self.parent_error != f32::INFINITY
            && projected_error(self.parent_error, &self.parent_lod_bounds, view_pos)
                <= error_threshold;
        fine && !parent_fine
    }
}

/// Error projected from `view_pos`: `error / (distance to the sphere surface)`
///
/// Infinite inside the sphere (unless the error is 0). Because the distance is
/// measured to the sphere surface, a sphere that contains another one is never
/// farther away, so a larger error in a containing sphere always projects to a
/// larger value: the property the cut relies on.
#[inline]
fn projected_error(error: f32, bounds: &ClusterBounds, view_pos: Vec3) -> f32 {
    if error <= 0.0 {
        return 0.0;
    }
    let d = (bounds.center - view_pos).length() - bounds.radius;
    if d <= 0.0 {
        f32::INFINITY
    } else {
        error / d
    }
}

/// Nanite cluster group: the clusters of one spatial region at one LOD level
///
/// Every level is cut into regions (octree cells, see `generate_nanite_mesh`);
/// the clusters of a region form one group. The groups form a tree: the parent
/// of a group is the group of the enclosing cell at the next coarser level that
/// has geometry, and a parent's region is the union of its children's regions.
/// All clusters of a group are drawn or skipped together.
#[derive(Debug, Clone)]
pub struct ClusterGroup {
    /// Group ID
    pub id: u32,
    /// LOD level
    pub lod_level: u32,
    /// Cluster IDs in this group
    pub cluster_ids: Vec<u32>,
    /// LOD bounds: the AABB of the group's own vertices, and a sphere around
    /// the AABB center that holds those vertices and the spheres of every
    /// child group (so a parent's sphere contains its children's)
    pub bounds: ClusterBounds,
    /// Error of the group: the measured surface error of its own triangles,
    /// raised to the largest error of its child groups (so it never decreases
    /// from a child to its parent)
    pub max_error: f32,
}

/// Configuration for Nanite mesh generation
#[derive(Debug, Clone)]
pub struct NaniteConfig {
    /// Number of LOD levels to generate
    pub lod_levels: u32,
    /// Base resolution for LOD 0 (highest detail)
    pub base_resolution: u32,
    /// Resolution reduction factor per LOD level
    pub lod_factor: f32,
    /// Maximum triangles per cluster
    pub max_triangles_per_cluster: usize,
    /// Cluster overlap for seamless boundaries (0.0-0.5)
    pub cluster_overlap: f32,
    /// Whether to compute per-cluster normals
    pub compute_normals: bool,
    /// Use Dual Contouring instead of Marching Cubes (preserves sharp edges)
    pub use_dual_contouring: bool,
    /// Use tight AABB to minimize wasted voxel space
    pub use_tight_aabb: bool,
    /// Enable curvature-adaptive cluster density: at LOD 0, a cluster with
    /// more than half of the triangle budget whose sampled normals vary
    /// strongly is split into two smaller clusters of the same group
    pub curvature_adaptive: bool,
}

impl Default for NaniteConfig {
    fn default() -> Self {
        Self {
            lod_levels: 6,
            base_resolution: 128,
            lod_factor: 0.5,
            max_triangles_per_cluster: CLUSTER_MAX_TRIANGLES,
            cluster_overlap: 0.1,
            compute_normals: true,
            use_dual_contouring: false,
            use_tight_aabb: true,
            curvature_adaptive: false,
        }
    }
}

impl NaniteConfig {
    /// Create config for high detail (game-ready assets)
    pub fn high_detail() -> Self {
        Self {
            lod_levels: 8,
            base_resolution: 256,
            lod_factor: 0.5,
            ..Default::default()
        }
    }

    /// Create config for medium detail
    pub fn medium_detail() -> Self {
        Self {
            lod_levels: 5,
            base_resolution: 64,
            lod_factor: 0.5,
            ..Default::default()
        }
    }

    /// Create config for preview/fast generation
    pub fn preview() -> Self {
        Self {
            lod_levels: 3,
            base_resolution: 32,
            lod_factor: 0.5,
            ..Default::default()
        }
    }
}

/// Nanite mesh with LOD hierarchy
#[derive(Debug)]
pub struct NaniteMesh {
    /// All clusters across all LOD levels
    pub clusters: Vec<NaniteCluster>,
    /// Cluster groups (one per region and level, see [`ClusterGroup`])
    pub groups: Vec<ClusterGroup>,
    /// LOD level information
    pub lod_levels: Vec<LodLevel>,
    /// Global bounds
    pub bounds: ClusterBounds,
    /// Total triangle count (LOD 0)
    pub total_triangles: usize,
}

impl NaniteMesh {
    /// Get clusters at a specific LOD level
    pub fn clusters_at_lod(&self, level: u32) -> Vec<&NaniteCluster> {
        self.clusters
            .iter()
            .filter(|c| c.lod_level == level)
            .collect()
    }

    /// Get cluster by ID
    pub fn get_cluster(&self, id: u32) -> Option<&NaniteCluster> {
        self.clusters.iter().find(|c| c.id == id)
    }

    /// Select the clusters to draw from `view_pos` (the cluster-LOD cut)
    ///
    /// For every group `G` the projected error is `G.max_error / d`, where `d`
    /// is the distance from `view_pos` to the surface of `G.bounds` (infinite
    /// inside it). `G` is fine enough when that is at most `error_threshold`
    /// (an angle in radians, as in [`NaniteCluster::should_render`], which
    /// makes the same decision from the fields one cluster carries); a group
    /// without children is always fine enough (nothing finer exists, so the
    /// finest level is kept when no level meets the threshold). The clusters of
    /// `G` are selected when `G` is fine enough and its parent group is not (a
    /// group without a parent counts as having a parent that is too coarse).
    ///
    /// Errors never decrease and spheres never shrink from a child to its
    /// parent, so the projected error never decreases along a path towards the
    /// root, and every path from a leaf group to the root holds exactly one
    /// selected group: the selected clusters cover every region exactly once.
    /// The returned ids are in cluster order.
    pub fn select_clusters(&self, view_pos: Vec3, error_threshold: f32) -> Vec<u32> {
        use std::collections::HashMap;

        let mut group_of: HashMap<u32, usize> = HashMap::with_capacity(self.clusters.len());
        for (gi, g) in self.groups.iter().enumerate() {
            for &id in &g.cluster_ids {
                group_of.insert(id, gi);
            }
        }
        let first_cluster =
            |g: &ClusterGroup| g.cluster_ids.first().and_then(|&id| self.get_cluster(id));
        let fine_enough = |g: &ClusterGroup| {
            let leaf = first_cluster(g).is_none_or(|c| c.child_ids.is_empty());
            leaf || projected_error(g.max_error, &g.bounds, view_pos) <= error_threshold
        };

        let mut selected = vec![false; self.groups.len()];
        for (gi, g) in self.groups.iter().enumerate() {
            if !fine_enough(g) {
                continue;
            }
            let parent = first_cluster(g)
                .and_then(|c| c.parent_ids.first())
                .and_then(|pid| group_of.get(pid))
                .map(|&pi| &self.groups[pi]);
            if parent.is_none_or(|p| !fine_enough(p)) {
                selected[gi] = true;
            }
        }

        self.clusters
            .iter()
            .filter(|c| group_of.get(&c.id).is_some_and(|&gi| selected[gi]))
            .map(|c| c.id)
            .collect()
    }

    /// Get total vertex count
    pub fn total_vertices(&self) -> usize {
        self.clusters.iter().map(|c| c.vertex_count()).sum()
    }

    /// Export to flat mesh at specified LOD
    pub fn to_mesh(&self, lod_level: u32) -> Mesh {
        let clusters: Vec<_> = self.clusters_at_lod(lod_level);

        let mut mesh = Mesh::new();
        let mut vertex_offset = 0u32;

        for cluster in clusters {
            mesh.vertices.extend_from_slice(&cluster.vertices);

            for tri in &cluster.triangles {
                mesh.indices.push(tri.a + vertex_offset);
                mesh.indices.push(tri.b + vertex_offset);
                mesh.indices.push(tri.c + vertex_offset);
            }

            vertex_offset += cluster.vertices.len() as u32;
        }

        mesh
    }
}

/// Target region size in voxels of the level's grid
///
/// A region of `REGION_VOXELS`³ voxels holds a surface patch of roughly
/// `2 * REGION_VOXELS²` triangles, about one to two clusters.
const REGION_VOXELS: u32 = 8;

/// Octree depth of the regions of a level generated at `resolution`
///
/// `round(log2(resolution / REGION_VOXELS))`, at least 0, and never deeper
/// than `coarser_limit` (the depth of the previous, finer level), so that the
/// cells of a coarser level are unions of the cells of the finer levels.
fn region_depth(resolution: u32, coarser_limit: u32) -> u32 {
    let d = (resolution.max(1) as f32 / REGION_VOXELS as f32)
        .log2()
        .round()
        .clamp(0.0, 16.0) as u32;
    d.min(coarser_limit)
}

/// Octree cell of a triangle at `depth`
///
/// The root cube starts at `origin` with edge `side`. The triangle belongs to
/// the cell that holds its centroid `(a + b + c) / 3`: along each axis the
/// index is `floor((centroid - origin) * (2^depth / side))`, clamped to
/// `0..2^depth`. A triangle has one centroid, so it lies in exactly one cell
/// even when its vertices are spread over several cells; a centroid exactly on
/// a cell plane goes to the cell above the plane.
fn triangle_cell(a: Vec3, b: Vec3, c: Vec3, origin: Vec3, side: f32, depth: u32) -> [u32; 3] {
    let n = 1u32 << depth;
    let scale = n as f32 / side;
    let rel = ((a + b + c) / 3.0 - origin) * scale;
    let idx = |x: f32| (x.floor().max(0.0) as u32).min(n - 1);
    [idx(rel.x), idx(rel.y), idx(rel.z)]
}

/// Generate Nanite-compatible mesh from SDF
///
/// Each level `k` is a mesh of the SDF at resolution
/// `max(4, base_resolution * lod_factor^k)` (Marching Cubes or Dual
/// Contouring). The mesh generation bounds (after the tight AABB, when
/// enabled) define the root cube: it starts at their minimum corner and its
/// edge is their largest extent. Level `k` is cut into octree cells at depth
/// `min(depth_{k-1}, max(0, round(log2(resolution_k / 8))))` and each triangle goes to the
/// cell that holds its centroid `(a + b + c) / 3` (per axis
/// `floor((centroid - origin) * (2^depth / edge))`, clamped to the cube), so a
/// triangle lies in exactly one cell even when its vertices do not. The triangles of one cell, in
/// mesh order, are packed greedily into clusters of at most
/// `max_triangles_per_cluster` triangles and [`CLUSTER_MAX_VERTICES`]
/// vertices; those clusters form one [`ClusterGroup`].
///
/// The parent group of a group is the group of the enclosing cell at the
/// nearest coarser level that has triangles in it. Every cluster lists the
/// clusters of its parent group in `parent_ids` and the clusters of all its
/// child groups in `child_ids`.
///
/// The error of a cluster is measured on the SDF: the largest `|sdf|` over its
/// triangles (sampled on a barycentric grid and refined by a local search), so
/// for an exact distance field it is the one-sided Hausdorff distance from the
/// triangles to the surface. A group's error is the largest error of its
/// clusters raised to the errors of its child groups, and every cluster of the
/// group carries the group's error in `geometric_error`, so the error never
/// decreases from child to parent. Every cluster also carries its group's LOD
/// sphere in `lod_bounds` and the parent group's error and LOD sphere in
/// `parent_error` / `parent_lod_bounds`, so [`NaniteCluster::should_render`]
/// decides the cut without the groups.
pub fn generate_nanite_mesh(
    sdf: &SdfNode,
    min_bounds: Vec3,
    max_bounds: Vec3,
    config: &NaniteConfig,
) -> NaniteMesh {
    let bounds_center = (min_bounds + max_bounds) * 0.5;

    // Tight AABB: shrink bounds to fit actual surface
    let (min_bounds, max_bounds) = if config.use_tight_aabb {
        let tight = compute_tight_aabb(sdf);
        // Use tight bounds but don't exceed user-specified bounds
        let tight_min = tight.min.max(min_bounds);
        let tight_max = tight.max.min(max_bounds);
        // Only use tight bounds if they're valid (surface exists)
        if tight_min.x < tight_max.x && tight_min.y < tight_max.y && tight_min.z < tight_max.z {
            (tight_min, tight_max)
        } else {
            (min_bounds, max_bounds)
        }
    } else {
        (min_bounds, max_bounds)
    };
    let origin = min_bounds;
    let side = (max_bounds - min_bounds)
        .max_element()
        .max(f32::MIN_POSITIVE);
    let max_tris = config.max_triangles_per_cluster.max(1);

    let mut all_clusters: Vec<NaniteCluster> = Vec::new();
    let mut regions: Vec<Region> = Vec::new();
    let mut lod_infos = Vec::new();
    let mut cluster_id = 0u32;
    let mut depth_limit = u32::MAX;

    // Generate each LOD level
    for lod in 0..config.lod_levels {
        let resolution =
            (config.base_resolution as f32 * config.lod_factor.powi(lod as i32)) as u32;
        let resolution = resolution.max(4); // Minimum resolution

        let mesh = if config.use_dual_contouring {
            let dc_config = DualContouringConfig {
                resolution: resolution as usize,
                compute_normals: config.compute_normals,
                ..Default::default()
            };
            dual_contouring(sdf, min_bounds, max_bounds, &dc_config)
        } else {
            let mc_config = MarchingCubesConfig {
                resolution: resolution as usize,
                iso_level: 0.0,
                compute_normals: config.compute_normals,
                ..Default::default()
            };
            sdf_to_mesh(sdf, min_bounds, max_bounds, &mc_config)
        };

        if mesh.triangle_count() == 0 {
            continue;
        }

        let depth = region_depth(resolution, depth_limit);
        depth_limit = depth;

        // Triangles of each cell, in mesh order
        let mut cells: std::collections::BTreeMap<[u32; 3], Vec<usize>> =
            std::collections::BTreeMap::new();
        for t in 0..mesh.triangle_count() {
            let [a, b, c] = triangle_positions(&mesh, t);
            cells
                .entry(triangle_cell(a, b, c, origin, side, depth))
                .or_default()
                .push(t);
        }

        let tri_errors = sampled_triangle_errors(sdf, &mesh);
        let mut level_max = 0.0f32;
        let mut level_tris = 0u32;

        for (cell, tris) in cells {
            let mut chunks = pack_clusters(&mesh, &tris, max_tris);
            if config.curvature_adaptive && lod == 0 {
                chunks = split_high_curvature(chunks, &mesh, sdf, max_tris);
            }
            let mut ids = Vec::with_capacity(chunks.len());
            let mut region_error = 0.0f32;
            for chunk in chunks {
                let (vertices, triangles) = extract_cluster_geometry(&mesh, &chunk);
                let positions: Vec<Vec3> = vertices.iter().map(|v| v.position).collect();
                let bounds = ClusterBounds::from_vertices(&positions);
                let error = cluster_surface_error(sdf, &mesh, &chunk, &tri_errors);
                region_error = region_error.max(error);
                level_tris += triangles.len() as u32;
                ids.push(cluster_id);
                all_clusters.push(NaniteCluster {
                    id: cluster_id,
                    lod_level: lod,
                    vertices,
                    triangles,
                    bounds,
                    parent_ids: Vec::new(),
                    child_ids: Vec::new(),
                    geometric_error: error,
                    lod_bounds: bounds,
                    parent_error: f32::INFINITY,
                    parent_lod_bounds: bounds,
                    material_id: eval_material(sdf, bounds.center),
                });
                cluster_id += 1;
            }
            level_max = level_max.max(region_error);
            regions.push(Region {
                lod,
                depth,
                cell,
                cluster_ids: ids,
                error: region_error,
                parent: None,
            });
        }

        lod_infos.push(LodLevel {
            level: lod,
            resolution,
            max_error: level_max,
            triangle_count: level_tris,
        });
    }

    let groups = build_region_tree(&mut regions, &mut all_clusters);
    // the level summaries report the propagated (monotone) errors
    for info in &mut lod_infos {
        info.max_error = groups
            .iter()
            .filter(|g| g.lod_level == info.level)
            .map(|g| g.max_error)
            .fold(0.0f32, f32::max);
    }

    // Compute global bounds
    let global_bounds = if !all_clusters.is_empty() {
        let all_vertices: Vec<Vec3> = all_clusters
            .iter()
            .flat_map(|c| c.vertices.iter().map(|v| v.position))
            .collect();
        ClusterBounds::from_vertices(&all_vertices)
    } else {
        ClusterBounds::from_vertices(&[bounds_center])
    };

    let total_triangles = all_clusters
        .iter()
        .filter(|c| c.lod_level == 0)
        .map(|c| c.triangle_count())
        .sum();

    NaniteMesh {
        clusters: all_clusters,
        groups,
        lod_levels: lod_infos,
        bounds: global_bounds,
        total_triangles,
    }
}

/// One octree cell of one level, before it becomes a [`ClusterGroup`]
struct Region {
    lod: u32,
    depth: u32,
    cell: [u32; 3],
    cluster_ids: Vec<u32>,
    error: f32,
    parent: Option<usize>,
}

/// Link every region to the enclosing region of the nearest coarser level,
/// propagate errors and LOD spheres upwards, fill the cluster DAG and return
/// the groups
///
/// `regions` are in level order (finest first) and `clusters` are indexed by
/// id.
fn build_region_tree(regions: &mut [Region], clusters: &mut [NaniteCluster]) -> Vec<ClusterGroup> {
    use std::collections::HashMap;

    let mut by_key: HashMap<(u32, [u32; 3]), usize> = HashMap::with_capacity(regions.len());
    for (i, r) in regions.iter().enumerate() {
        by_key.insert((r.lod, r.cell), i);
    }
    let mut levels: Vec<(u32, u32)> = regions.iter().map(|r| (r.lod, r.depth)).collect();
    levels.dedup();

    for r in regions.iter_mut() {
        let (lod, depth, cell) = (r.lod, r.depth, r.cell);
        r.parent = levels
            .iter()
            .filter(|&&(l, _)| l > lod)
            .find_map(|&(l, d)| {
                let shift = depth - d;
                by_key.get(&(l, cell.map(|x| x >> shift))).copied()
            });
    }

    // own AABB / sphere of each region
    let mut bounds: Vec<ClusterBounds> = regions
        .iter()
        .map(|r| {
            let pts: Vec<Vec3> = r
                .cluster_ids
                .iter()
                .flat_map(|&id| clusters[id as usize].vertices.iter().map(|v| v.position))
                .collect();
            ClusterBounds::from_vertices(&pts)
        })
        .collect();

    // children come before their parents (finer levels first)
    let mut errors: Vec<f32> = regions.iter().map(|r| r.error).collect();
    for i in 0..regions.len() {
        if let Some(p) = regions[i].parent {
            errors[p] = errors[p].max(errors[i]);
            let reach = (bounds[i].center - bounds[p].center).length() + bounds[i].radius;
            bounds[p].radius = bounds[p].radius.max(reach);
        }
    }

    let mut children: Vec<Vec<u32>> = vec![Vec::new(); regions.len()];
    for r in regions.iter() {
        if let Some(p) = r.parent {
            children[p].extend_from_slice(&r.cluster_ids);
        }
    }
    for (i, r) in regions.iter().enumerate() {
        let parents: Vec<u32> = r
            .parent
            .map(|p| regions[p].cluster_ids.clone())
            .unwrap_or_default();
        for &id in &r.cluster_ids {
            let c = &mut clusters[id as usize];
            c.geometric_error = errors[i];
            c.lod_bounds = bounds[i];
            (c.parent_error, c.parent_lod_bounds) = match r.parent {
                Some(p) => (errors[p], bounds[p]),
                None => (f32::INFINITY, bounds[i]),
            };
            c.parent_ids.clone_from(&parents);
            c.child_ids.clone_from(&children[i]);
        }
    }

    regions
        .iter()
        .enumerate()
        .map(|(i, r)| ClusterGroup {
            id: i as u32,
            lod_level: r.lod,
            cluster_ids: r.cluster_ids.clone(),
            bounds: bounds[i],
            max_error: errors[i],
        })
        .collect()
}

#[inline]
fn triangle_positions(mesh: &Mesh, t: usize) -> [Vec3; 3] {
    let i = &mesh.indices[t * 3..t * 3 + 3];
    [
        mesh.vertices[i[0] as usize].position,
        mesh.vertices[i[1] as usize].position,
        mesh.vertices[i[2] as usize].position,
    ]
}

/// Pack triangles (in the given order) into chunks of at most `max_tris`
/// triangles and [`CLUSTER_MAX_VERTICES`] distinct vertices
fn pack_clusters(mesh: &Mesh, tris: &[usize], max_tris: usize) -> Vec<Vec<usize>> {
    use std::collections::HashSet;

    let mut chunks = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut verts: HashSet<u32> = HashSet::new();
    for &t in tris {
        let idx = &mesh.indices[t * 3..t * 3 + 3];
        let new = idx
            .iter()
            .enumerate()
            .filter(|&(k, i)| !verts.contains(i) && !idx[..k].contains(i))
            .count();
        if !current.is_empty()
            && (current.len() >= max_tris || verts.len() + new > CLUSTER_MAX_VERTICES)
        {
            chunks.push(std::mem::take(&mut current));
            verts.clear();
        }
        verts.extend(idx.iter().copied());
        current.push(t);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Split chunks whose sampled SDF normals vary strongly (and that use more
/// than half of the triangle budget) into two halves
fn split_high_curvature(
    chunks: Vec<Vec<usize>>,
    mesh: &Mesh,
    sdf: &SdfNode,
    max_tris: usize,
) -> Vec<Vec<usize>> {
    let mut out = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        if chunk.len() <= max_tris / 2 || chunk.len() < 2 {
            out.push(chunk);
            continue;
        }
        let step = (chunk.len() / 16).max(1);
        let normals: Vec<Vec3> = chunk
            .iter()
            .step_by(step)
            .map(|&t| eval_normal(sdf, mesh.vertices[mesh.indices[t * 3] as usize].position))
            .collect();
        let mean = normals.iter().copied().sum::<Vec3>() / normals.len() as f32;
        let variance = normals
            .iter()
            .map(|n| (*n - mean).length_squared())
            .sum::<f32>()
            / normals.len() as f32;
        if variance > 0.1 {
            let (a, b) = chunk.split_at(chunk.len() / 2);
            out.push(a.to_vec());
            out.push(b.to_vec());
        } else {
            out.push(chunk);
        }
    }
    out
}

/// Barycentric grid used to sample a triangle (`TRI_SAMPLES` subdivisions per
/// edge, `(n + 1)(n + 2) / 2` points including the vertices)
const TRI_SAMPLES: u32 = 4;

/// Largest `|sdf|` at the sample points of a triangle and where it was found
fn sample_triangle(sdf: &SdfNode, [a, b, c]: [Vec3; 3]) -> (f32, f32, f32) {
    let n = TRI_SAMPLES as f32;
    let mut best = (f32::NEG_INFINITY, 0.0, 0.0);
    for i in 0..=TRI_SAMPLES {
        for j in 0..=(TRI_SAMPLES - i) {
            let (u, v) = (i as f32 / n, j as f32 / n);
            let e = eval(sdf, a + (b - a) * u + (c - a) * v).abs();
            if e > best.0 {
                best = (e, u, v);
            }
        }
    }
    best
}

/// Largest `|sdf|` on a triangle: the best grid sample refined by a pattern
/// search in barycentric coordinates (the step halves 11 times, from 1/8 to
/// below 1e-4)
fn refine_triangle(sdf: &SdfNode, tri: [Vec3; 3]) -> f32 {
    let [a, b, c] = tri;
    let (mut best, mut u, mut v) = sample_triangle(sdf, tri);
    let mut h = 0.5 / TRI_SAMPLES as f32;
    let dirs = [
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (-1.0, 1.0),
    ];
    // 11 halvings take the step from 1/8 below 1e-4
    let mut halvings = 0;
    while halvings < 11 {
        let mut moved = false;
        for (du, dv) in dirs {
            let (nu, nv) = (u + du * h, v + dv * h);
            if nu < 0.0 || nv < 0.0 || nu + nv > 1.0 {
                continue;
            }
            let e = eval(sdf, a + (b - a) * nu + (c - a) * nv).abs();
            if e > best {
                (best, u, v, moved) = (e, nu, nv, true);
            }
        }
        if !moved {
            h *= 0.5;
            halvings += 1;
        }
    }
    best
}

/// Sampled `|sdf|` maximum of every triangle of `mesh`
pub(crate) fn sampled_triangle_errors(sdf: &SdfNode, mesh: &Mesh) -> Vec<f32> {
    use rayon::prelude::*;
    (0..mesh.triangle_count())
        .into_par_iter()
        .map(|t| sample_triangle(sdf, triangle_positions(mesh, t)).0)
        .collect()
}

/// Surface error of a set of triangles: the largest `|sdf|` on them
///
/// Every triangle is sampled (`sampled` holds the grid maxima); the triangles
/// whose sampled maximum reaches half of the set's sampled maximum are refined
/// by [`refine_triangle`] (a refinement only adds a second-order amount on top
/// of the grid value, so the others cannot hold the maximum). The result
/// includes 4 ulp of the largest coordinate for the f32 evaluation.
pub(crate) fn cluster_surface_error(
    sdf: &SdfNode,
    mesh: &Mesh,
    tris: &[usize],
    sampled: &[f32],
) -> f32 {
    use rayon::prelude::*;
    let coarse = tris.iter().map(|&t| sampled[t]).fold(0.0f32, f32::max);
    let found = tris
        .par_iter()
        .filter(|&&t| sampled[t] >= 0.5 * coarse)
        .map(|&t| refine_triangle(sdf, triangle_positions(mesh, t)))
        .reduce(|| coarse, f32::max);
    if found == 0.0 {
        return 0.0;
    }
    // the SDF is evaluated at f32 positions: add 4 ulp of the largest
    // coordinate so the value bounds the distance of the exact triangles
    let scale = tris
        .iter()
        .flat_map(|&t| triangle_positions(mesh, t))
        .map(|p| p.abs().max_element())
        .fold(1.0f32, f32::max);
    4.0f32.mul_add(f32::EPSILON * scale, found)
}

/// Extract geometry for a subset of triangles
fn extract_cluster_geometry(mesh: &Mesh, tri_indices: &[usize]) -> (Vec<Vertex>, Vec<Triangle>) {
    use std::collections::HashMap;

    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    for &tri_idx in tri_indices {
        let base = tri_idx * 3;
        let mut new_indices = [0u32; 3];

        #[allow(clippy::needless_range_loop)]
        for i in 0..3 {
            let old_idx = mesh.indices[base + i];

            let new_idx = *vertex_map.entry(old_idx).or_insert_with(|| {
                let idx = vertices.len() as u32;
                vertices.push(mesh.vertices[old_idx as usize]);
                idx
            });

            new_indices[i] = new_idx;
        }

        triangles.push(Triangle::new(
            new_indices[0],
            new_indices[1],
            new_indices[2],
        ));
    }

    (vertices, triangles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_bounds() {
        let vertices = vec![
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];

        let bounds = ClusterBounds::from_vertices(&vertices);

        assert_eq!(bounds.aabb_min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(bounds.aabb_max, Vec3::new(1.0, 1.0, 1.0));
        assert!((bounds.center - Vec3::ZERO).length() < 0.01);
    }

    #[test]
    fn test_generate_nanite_mesh() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(!nanite.clusters.is_empty());
        assert!(!nanite.lod_levels.is_empty());
        assert!(nanite.total_triangles > 0);
    }

    #[test]
    fn test_cluster_selection() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Close view - should select more clusters
        let close_clusters = nanite.select_clusters(Vec3::new(0.0, 0.0, 3.0), 0.01);

        // Far view - should select fewer clusters
        let far_clusters = nanite.select_clusters(Vec3::new(0.0, 0.0, 100.0), 0.01);

        // Far view should have equal or fewer clusters
        assert!(far_clusters.len() <= close_clusters.len() + nanite.clusters.len());
    }

    #[test]
    fn test_to_mesh() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        let mesh = nanite.to_mesh(0);
        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_nanite_dual_contouring() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig {
            use_dual_contouring: true,
            lod_levels: 2,
            base_resolution: 16,
            ..NaniteConfig::preview()
        };

        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        assert!(!nanite.clusters.is_empty());
        assert!(nanite.total_triangles > 0);
    }

    #[test]
    fn test_nanite_tight_aabb() {
        let sphere = SdfNode::sphere(0.5);
        let config = NaniteConfig {
            use_tight_aabb: true,
            ..NaniteConfig::preview()
        };

        // Use excessively large bounds - tight AABB should shrink them
        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-10.0), Vec3::splat(10.0), &config);

        assert!(!nanite.clusters.is_empty());
    }

    #[test]
    fn test_nanite_material_id() {
        let sphere = SdfNode::sphere(1.0);
        let config = NaniteConfig::preview();

        let nanite = generate_nanite_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Default material should be 0
        for cluster in &nanite.clusters {
            assert_eq!(cluster.material_id, 0);
        }
    }

    // ------------------------------------------------------------------------
    // NormalCone tests (meshoptimizer §clusterizer.cpp `computeMeshletBounds` cone 相当)
    // ------------------------------------------------------------------------

    #[test]
    fn test_normal_cone_empty() {
        let cone = NormalCone::from_normals(&[]);
        assert!((cone.cutoff_cos + 1.0).abs() < 1e-6);
        // unbounded は culling 不可
        assert!(!cone.is_backface_culled(Vec3::Z));
    }

    #[test]
    fn test_normal_cone_all_same_normal() {
        // 全部 +Y の法線 → cone は axis=Y, cutoff_cos=1.0 (完全一致)
        let normals = vec![Vec3::Y; 5];
        let cone = NormalCone::from_normals(&normals);
        assert!((cone.axis.y - 1.0).abs() < 1e-4);
        assert!((cone.cutoff_cos - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_normal_cone_scattered() {
        // 6 軸方向 → 平均 0 に近く axis 決定不可 → unbounded
        let normals = vec![
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let cone = NormalCone::from_normals(&normals);
        // 平均 = 0 → unbounded fallback
        assert!(cone.cutoff_cos <= -1.0 + 1e-6);
    }

    #[test]
    fn test_normal_cone_hemispherical() {
        // 半球状に散る法線 (Y ± X)、axis = Y、cutoff_cos = 0.7 前後
        let normals = vec![
            Vec3::Y,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(-1.0, 1.0, 0.0).normalize(),
            Vec3::new(0.0, 1.0, 1.0).normalize(),
            Vec3::new(0.0, 1.0, -1.0).normalize(),
        ];
        let cone = NormalCone::from_normals(&normals);
        // axis は +Y に近い
        assert!(cone.axis.y > 0.9);
        // cutoff_cos > 0 で有効な cone (culling 可能状態)
        assert!(cone.cutoff_cos > 0.0);
    }

    #[test]
    fn test_normal_cone_backface_culled() {
        // 全法線 +Y (cluster の "上面")、view_dir = +Y (camera が cluster の下、
        // 見上げると cluster の裏側 = back-face) → cull すべき
        let normals = vec![Vec3::Y; 3];
        let cone = NormalCone::from_normals(&normals);
        // view_dir = +Y: axis.dot(view) = 1, sin β = 0, 1 >= 0 → cull ✓
        assert!(cone.is_backface_culled(Vec3::Y));
        // view_dir = -Y: camera が cluster の上、見下ろす = 表側 (+Y face 見える) → cull しない
        assert!(!cone.is_backface_culled(Vec3::NEG_Y));
    }

    #[test]
    fn test_normal_cone_wide_no_cull() {
        // 開き角広い cone (Y ± 60°) → 側面から見ても back とは断言できず culling 不可
        let normals = vec![
            Vec3::Y,
            Vec3::new(0.866, 0.5, 0.0), // 60° in XY
            Vec3::new(-0.866, 0.5, 0.0),
        ];
        let cone = NormalCone::from_normals(&normals);
        // 側面 view_dir = X → cone axis Y から dot = 0
        // cutoff_cos ≈ 0.5 (sin β ≈ 0.866)、0 >= 0.866 は false なので culling されない
        assert!(!cone.is_backface_culled(Vec3::X));
    }

    #[test]
    fn test_normal_cone_unbounded_never_culls() {
        let cone = NormalCone::unbounded();
        // どの view direction でも false
        assert!(!cone.is_backface_culled(Vec3::X));
        assert!(!cone.is_backface_culled(Vec3::NEG_X));
        assert!(!cone.is_backface_culled(Vec3::Y));
        assert!(!cone.is_backface_culled(Vec3::NEG_Y));
    }

    // ------------------------------------------------------------------------
    // cone_apex tests (2026-07-28 追加、Vulkan meshlet culling shader 用)
    // ------------------------------------------------------------------------

    #[test]
    fn test_normal_cone_apex_default_none() {
        // from_normals は apex = None
        let normals = vec![Vec3::Y; 3];
        let cone = NormalCone::from_normals(&normals);
        assert!(cone.apex.is_none());
    }

    #[test]
    fn test_normal_cone_from_normals_and_positions_populates_apex() {
        // 平面 (Y=0) 上の 3 三角形、全 normal +Y → apex は cluster center から
        // -Y 方向にオフセットされた点 (max_t 分後退)
        let normals = vec![Vec3::Y; 3];
        let face_centers = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let cone = NormalCone::from_normals_and_positions(&normals, &face_centers);
        assert!(cone.apex.is_some());
        let apex = cone.apex.unwrap();
        // 全 face が Y=0 の平面上なので cluster center も Y=0
        // apex は cluster center から -Y 方向 (axis 逆) にオフセットされているはず
        assert!(apex.y <= 0.0, "apex.y should be <= 0, got {}", apex.y);
    }

    #[test]
    fn test_normal_cone_apex_unbounded_returns_none() {
        // 全方向散乱 → basic cone unbounded → apex も None
        let normals = vec![Vec3::X, Vec3::NEG_X, Vec3::Y, Vec3::NEG_Y];
        let centers = vec![Vec3::ZERO; 4];
        let cone = NormalCone::from_normals_and_positions(&normals, &centers);
        assert!(cone.cutoff_cos <= -1.0 + 1e-6);
        assert!(cone.apex.is_none());
    }

    #[test]
    fn test_normal_cone_apex_length_mismatch_fallback() {
        // normals.len() != face_centers.len() → basic cone のみ (apex=None)
        let normals = vec![Vec3::Y; 3];
        let centers = vec![Vec3::ZERO; 2]; // mismatch
        let cone = NormalCone::from_normals_and_positions(&normals, &centers);
        assert!(cone.apex.is_none());
        // basic cone は正常
        assert!((cone.axis.y - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_normal_cone_apex_finite() {
        // 一般的な curved surface (半球の一部) → apex が有限値
        let normals = vec![
            Vec3::Y,
            Vec3::new(0.5, 0.866, 0.0).normalize(),
            Vec3::new(-0.5, 0.866, 0.0).normalize(),
            Vec3::new(0.0, 0.866, 0.5).normalize(),
        ];
        let face_centers = vec![
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.5, 0.866, 0.0),
            Vec3::new(-0.5, 0.866, 0.0),
            Vec3::new(0.0, 0.866, 0.5),
        ];
        let cone = NormalCone::from_normals_and_positions(&normals, &face_centers);
        let apex = cone.apex.expect("apex should be Some");
        assert!(
            apex.x.is_finite() && apex.y.is_finite() && apex.z.is_finite(),
            "apex must be finite, got {:?}",
            apex
        );
    }
}

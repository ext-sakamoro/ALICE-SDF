//! Manifold Mesh Guarantee (Deep Fried Edition)
//!
//! Mesh validation, repair, and quality assurance for AAA game pipelines.
//! Ensures watertight, manifold meshes suitable for physics, rendering,
//! and 3D printing.
//!
//! # Features
//! - Non-manifold edge detection
//! - T-junction repair
//! - Degenerate triangle removal
//! - Normal consistency check and repair
//! - Mesh statistics and quality metrics
//!
//! Author: Moroya Sakamoto

use crate::mesh::{Mesh, Vertex};
use std::collections::{HashMap, HashSet, VecDeque};

/// Mesh validation result
#[derive(Debug, Clone)]
pub struct MeshValidation {
    /// Is the mesh manifold (every edge shared by exactly 2 triangles)?
    pub is_manifold: bool,
    /// Number of non-manifold edges (shared by != 2 triangles)
    pub non_manifold_edges: usize,
    /// Number of boundary edges (shared by exactly 1 triangle)
    pub boundary_edges: usize,
    /// Number of degenerate triangles (zero area)
    pub degenerate_triangles: usize,
    /// Number of duplicate vertices (within epsilon)
    pub duplicate_vertices: usize,
    /// Number of flipped normals
    pub inconsistent_normals: usize,
    /// Total vertex count
    pub vertex_count: usize,
    /// Total triangle count
    pub triangle_count: usize,
}

impl MeshValidation {
    /// Check if the mesh passes all quality checks
    pub const fn is_clean(&self) -> bool {
        self.is_manifold && self.degenerate_triangles == 0 && self.inconsistent_normals == 0
    }
}

impl std::fmt::Display for MeshValidation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mesh Validation Report")?;
        writeln!(f, "  Vertices: {}", self.vertex_count)?;
        writeln!(f, "  Triangles: {}", self.triangle_count)?;
        writeln!(
            f,
            "  Manifold: {}",
            if self.is_manifold { "YES" } else { "NO" }
        )?;
        writeln!(f, "  Non-manifold edges: {}", self.non_manifold_edges)?;
        writeln!(f, "  Boundary edges: {}", self.boundary_edges)?;
        writeln!(f, "  Degenerate triangles: {}", self.degenerate_triangles)?;
        writeln!(f, "  Duplicate vertices: {}", self.duplicate_vertices)?;
        writeln!(f, "  Inconsistent normals: {}", self.inconsistent_normals)?;
        write!(
            f,
            "  Status: {}",
            if self.is_clean() {
                "CLEAN"
            } else {
                "NEEDS REPAIR"
            }
        )
    }
}

/// Edge key for hash map lookup (order-independent)
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
struct EdgeKey(u32, u32);

impl EdgeKey {
    const fn new(a: u32, b: u32) -> Self {
        if a <= b {
            Self(a, b)
        } else {
            Self(b, a)
        }
    }
}

/// Validate a mesh for manifoldness and quality (Deep Fried)
///
/// Performs a complete analysis of the mesh topology and geometry.
pub fn validate_mesh(mesh: &Mesh) -> MeshValidation {
    let mut edge_counts: HashMap<EdgeKey, u32> = HashMap::new();

    // Count edge usage
    let tri_count = mesh.indices.len() / 3;
    for i in 0..tri_count {
        let base = i * 3;
        let a = mesh.indices[base];
        let b = mesh.indices[base + 1];
        let c = mesh.indices[base + 2];

        *edge_counts.entry(EdgeKey::new(a, b)).or_insert(0) += 1;
        *edge_counts.entry(EdgeKey::new(b, c)).or_insert(0) += 1;
        *edge_counts.entry(EdgeKey::new(c, a)).or_insert(0) += 1;
    }

    let non_manifold_edges = edge_counts.values().filter(|&&c| c > 2).count();
    let boundary_edges = edge_counts.values().filter(|&&c| c == 1).count();
    let is_manifold = non_manifold_edges == 0 && boundary_edges == 0;

    // Count degenerate triangles
    let degenerate_triangles = count_degenerate_triangles(mesh);

    // Count duplicate vertices
    let duplicate_vertices = count_duplicate_vertices(mesh, 1e-6);

    // Count inconsistent normals
    let inconsistent_normals = count_inconsistent_normals(mesh);

    MeshValidation {
        is_manifold,
        non_manifold_edges,
        boundary_edges,
        degenerate_triangles,
        duplicate_vertices,
        inconsistent_normals,
        vertex_count: mesh.vertices.len(),
        triangle_count: tri_count,
    }
}

/// Count degenerate triangles (zero or near-zero area)
#[inline(always)]
fn count_degenerate_triangles(mesh: &Mesh) -> usize {
    let mut count = 0;
    let tri_count = mesh.indices.len() / 3;

    for i in 0..tri_count {
        let base = i * 3;
        let v0 = mesh.vertices[mesh.indices[base] as usize].position;
        let v1 = mesh.vertices[mesh.indices[base + 1] as usize].position;
        let v2 = mesh.vertices[mesh.indices[base + 2] as usize].position;

        let cross = (v1 - v0).cross(v2 - v0);
        if cross.length_squared() < 1e-12 {
            count += 1;
        }
    }

    count
}

/// Count duplicate vertices within epsilon
fn count_duplicate_vertices(mesh: &Mesh, epsilon: f32) -> usize {
    let eps_sq = epsilon * epsilon;
    let mut count = 0;

    // Use spatial hashing for O(n) average case
    let cell_size = epsilon * 10.0;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    for (i, v) in mesh.vertices.iter().enumerate() {
        let cx = (v.position.x / cell_size).floor() as i32;
        let cy = (v.position.y / cell_size).floor() as i32;
        let cz = (v.position.z / cell_size).floor() as i32;

        // Check neighboring cells
        let mut is_duplicate = false;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if (mesh.vertices[j].position - v.position).length_squared() < eps_sq {
                                is_duplicate = true;
                                break;
                            }
                        }
                    }
                    if is_duplicate {
                        break;
                    }
                }
                if is_duplicate {
                    break;
                }
            }
            if is_duplicate {
                break;
            }
        }

        if is_duplicate {
            count += 1;
        }

        grid.entry((cx, cy, cz)).or_default().push(i);
    }

    count
}

/// Count normals inconsistent with face orientation
fn count_inconsistent_normals(mesh: &Mesh) -> usize {
    let mut count = 0;
    let tri_count = mesh.indices.len() / 3;

    for i in 0..tri_count {
        let base = i * 3;
        let v0 = &mesh.vertices[mesh.indices[base] as usize];
        let v1 = &mesh.vertices[mesh.indices[base + 1] as usize];
        let v2 = &mesh.vertices[mesh.indices[base + 2] as usize];

        let face_normal = (v1.position - v0.position).cross(v2.position - v0.position);
        if face_normal.length_squared() < 1e-12 {
            continue; // Skip degenerate
        }
        let face_normal = face_normal.normalize();

        // Check each vertex normal against face normal
        let avg_normal = (v0.normal + v1.normal + v2.normal) / 3.0;
        if avg_normal.dot(face_normal) < 0.0 {
            count += 1;
        }
    }

    count
}

/// Mesh repair operations
pub struct MeshRepair;

impl MeshRepair {
    /// Remove degenerate triangles (zero-area) from a mesh
    pub fn remove_degenerate_triangles(mesh: &Mesh) -> Mesh {
        let tri_count = mesh.indices.len() / 3;
        let mut new_indices = Vec::with_capacity(mesh.indices.len());

        for i in 0..tri_count {
            let base = i * 3;
            let a = mesh.indices[base];
            let b = mesh.indices[base + 1];
            let c = mesh.indices[base + 2];

            let v0 = mesh.vertices[a as usize].position;
            let v1 = mesh.vertices[b as usize].position;
            let v2 = mesh.vertices[c as usize].position;

            let cross = (v1 - v0).cross(v2 - v0);
            if cross.length_squared() >= 1e-12 {
                new_indices.push(a);
                new_indices.push(b);
                new_indices.push(c);
            }
        }

        Mesh {
            vertices: mesh.vertices.clone(),
            indices: new_indices,
        }
    }

    /// Merge duplicate vertices (within epsilon distance)
    pub fn merge_duplicate_vertices(mesh: &Mesh, epsilon: f32) -> Mesh {
        let eps_sq = epsilon * epsilon;
        let mut new_vertices: Vec<Vertex> = Vec::with_capacity(mesh.vertices.len());
        let mut remap: Vec<u32> = Vec::with_capacity(mesh.vertices.len());

        let cell_size = epsilon * 10.0;
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        for v in &mesh.vertices {
            let cx = (v.position.x / cell_size).floor() as i32;
            let cy = (v.position.y / cell_size).floor() as i32;
            let cz = (v.position.z / cell_size).floor() as i32;

            let mut found = None;
            'search: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(indices) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            for &j in indices {
                                if (new_vertices[j].position - v.position).length_squared() < eps_sq
                                {
                                    found = Some(j);
                                    break 'search;
                                }
                            }
                        }
                    }
                }
            }

            match found {
                Some(existing) => {
                    remap.push(existing as u32);
                }
                None => {
                    let idx = new_vertices.len();
                    remap.push(idx as u32);
                    grid.entry((cx, cy, cz)).or_default().push(idx);
                    new_vertices.push(*v);
                }
            }
        }

        let new_indices: Vec<u32> = mesh.indices.iter().map(|&i| remap[i as usize]).collect();

        Mesh {
            vertices: new_vertices,
            indices: new_indices,
        }
    }

    /// Fix inconsistent normals by flipping triangles with inverted winding
    pub fn fix_normals(mesh: &Mesh) -> Mesh {
        let tri_count = mesh.indices.len() / 3;
        let mut new_indices = mesh.indices.clone();

        for i in 0..tri_count {
            let base = i * 3;
            let v0 = &mesh.vertices[new_indices[base] as usize];
            let v1 = &mesh.vertices[new_indices[base + 1] as usize];
            let v2 = &mesh.vertices[new_indices[base + 2] as usize];

            let face_normal = (v1.position - v0.position).cross(v2.position - v0.position);
            if face_normal.length_squared() < 1e-12 {
                continue;
            }
            let face_normal = face_normal.normalize();

            let avg_normal = (v0.normal + v1.normal + v2.normal) / 3.0;
            if avg_normal.dot(face_normal) < 0.0 {
                // Flip winding order
                new_indices.swap(base + 1, base + 2);
            }
        }

        Mesh {
            vertices: mesh.vertices.clone(),
            indices: new_indices,
        }
    }

    /// Run all repairs: remove degenerates, merge duplicates, fix normals
    pub fn repair_all(mesh: &Mesh, vertex_merge_epsilon: f32) -> Mesh {
        let mesh = Self::remove_degenerate_triangles(mesh);
        let mesh = Self::merge_duplicate_vertices(&mesh, vertex_merge_epsilon);
        Self::fix_normals(&mesh)
    }

    // ========================================================================
    // dotneet/image-to-3d §A mesh 後処理 3 手順
    // ========================================================================

    /// BFS で面の巻き向きを揃え、連結成分ごとの符号付き体積で外向きに直す
    ///
    /// # アルゴリズム (image-to-3d §A.1 準拠)
    ///
    /// 1. 隣接三角形が共有辺を逆向きに辿るように BFS で巻き向きを揃える
    ///    - 有向辺 (u, v) を持つ隣接三角形が同方向 (u, v) なら flip、反方向 (v, u) なら OK
    /// 2. 各連結成分について符号付き体積 (∑ v0·(v1×v2)) を計算
    ///    - 負なら成分全体を flip して外向きに直す
    ///
    /// # 制約
    ///
    /// - `fix_normals` が「頂点法線」から巻き向きを推定するのに対し、本関数は
    ///   **面の隣接関係だけ**から一貫性を作る (頂点法線が壊れていても動く)
    /// - 非多様体エッジ (3 つ以上の三角形で共有) がある場合、隣接判定が
    ///   曖昧になるので結果は保証されない (`validate_mesh` で先に確認推奨)
    ///
    /// # 実測
    ///
    /// image-to-3d の実測では、生成直後の Dual Contouring mesh は外向き 50.3% /
    /// 内向き 49.7% (ほぼランダム)、`orient_faces` 適用で連結成分ごとに一貫化
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn orient_faces(mesh: &Mesh) -> Mesh {
        let tri_count = mesh.indices.len() / 3;
        if tri_count == 0 {
            return mesh.clone();
        }

        let mut new_indices = mesh.indices.clone();

        // 無向辺 → 三角形 index 一覧
        let mut edge_tris: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
        for t in 0..tri_count {
            let (a, b, c) = tri_verts(&new_indices, t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                edge_tris.entry(key).or_default().push(t);
            }
        }

        let mut visited = vec![false; tri_count];
        let mut components: Vec<Vec<usize>> = Vec::new();

        for start in 0..tri_count {
            if visited[start] {
                continue;
            }
            let mut component: Vec<usize> = Vec::new();
            visited[start] = true;
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(start);
            while let Some(t) = queue.pop_front() {
                component.push(t);
                let (a, b, c) = tri_verts(&new_indices, t);
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    let key = if u < v { (u, v) } else { (v, u) };
                    if let Some(neighbors) = edge_tris.get(&key) {
                        for &t2 in neighbors {
                            if visited[t2] {
                                continue;
                            }
                            // t が (u, v) を持つ、隣接 t2 が (u, v) をそのまま持てば
                            // 巻き向き逆で衝突 → flip、(v, u) で持てば一貫
                            let (a2, b2, c2) = tri_verts(&new_indices, t2);
                            let has_uv = (a2 == u && b2 == v)
                                || (b2 == u && c2 == v)
                                || (c2 == u && a2 == v);
                            if has_uv {
                                let base = t2 * 3;
                                new_indices.swap(base + 1, base + 2);
                            }
                            visited[t2] = true;
                            queue.push_back(t2);
                        }
                    }
                }
            }
            components.push(component);
        }

        // 連結成分ごとに符号付き体積を計算、負なら flip
        for component in &components {
            let mut signed_vol_x6 = 0.0_f32;
            for &t in component {
                let base = t * 3;
                let v0 = mesh.vertices[new_indices[base] as usize].position;
                let v1 = mesh.vertices[new_indices[base + 1] as usize].position;
                let v2 = mesh.vertices[new_indices[base + 2] as usize].position;
                signed_vol_x6 += v0.dot(v1.cross(v2));
            }
            if signed_vol_x6 < 0.0 {
                for &t in component {
                    let base = t * 3;
                    new_indices.swap(base + 1, base + 2);
                }
            }
        }

        Mesh {
            vertices: mesh.vertices.clone(),
            indices: new_indices,
        }
    }

    /// 境界エッジの連結成分ごとに扇状で穴を塞ぐ
    ///
    /// # アルゴリズム (image-to-3d §A.2 準拠)
    ///
    /// 1. 境界エッジ = 1 三角形にのみ属するエッジを検出
    /// 2. 頂点共有で連結成分にまとめる (エッジチェイン)
    /// 3. 各連結成分について:
    ///    - リング頂点の重心を計算し、centroid 頂点として追加
    ///    - centroid の UV は最初のリング頂点の UV に統一 (image-to-3d の
    ///      「同じ穴の UV は 1 点に揃える」ルール、UV アトラス上で無関係な
    ///      チャートを塗り潰さないため)
    ///    - 各境界エッジについて (centroid, edge.v, edge.u) の扇三角形を追加
    ///
    /// # 制約
    ///
    /// - リング頂点 < 3 の穴は skip
    /// - 非多様体境界 (境界エッジが T-junction 等で交差する) では扇が正しく
    ///   閉じない可能性がある (実用上は稀)
    ///
    /// # 実測
    ///
    /// image-to-3d では境界エッジ 6.4% → 4.4% まで削減 (完全に塞げないのは
    /// 非多様体分岐が残るため、上流の Dual Contouring が原因)
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn fill_holes(mesh: &Mesh) -> Mesh {
        let tri_count = mesh.indices.len() / 3;
        if tri_count == 0 {
            return mesh.clone();
        }

        // 各辺の出現回数と、代表的な向き (元の三角形での順序) を記録
        let mut edge_count: HashMap<(u32, u32), i32> = HashMap::new();
        let mut edge_dir: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for t in 0..tri_count {
            let (a, b, c) = tri_verts(&mesh.indices, t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edge_count.entry(key).or_insert(0) += 1;
                edge_dir.entry(key).or_insert((u, v));
            }
        }

        let mut border_edges: Vec<(u32, u32)> = Vec::new();
        for (k, &c) in &edge_count {
            if c == 1 {
                border_edges.push(edge_dir[k]);
            }
        }

        if border_edges.is_empty() {
            return mesh.clone();
        }

        // 頂点 → 境界エッジ index の逆マップ
        let mut vert_edges: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, &(u, v)) in border_edges.iter().enumerate() {
            vert_edges.entry(u).or_default().push(i);
            vert_edges.entry(v).or_default().push(i);
        }

        // 境界エッジの連結成分 (共有頂点で BFS)
        let mut visited_edge = vec![false; border_edges.len()];
        let mut components: Vec<Vec<usize>> = Vec::new();
        for start in 0..border_edges.len() {
            if visited_edge[start] {
                continue;
            }
            let mut component: Vec<usize> = Vec::new();
            let mut queue: VecDeque<usize> = VecDeque::new();
            queue.push_back(start);
            visited_edge[start] = true;
            while let Some(ei) = queue.pop_front() {
                component.push(ei);
                let (u, v) = border_edges[ei];
                for vertex in [u, v] {
                    if let Some(neighbors) = vert_edges.get(&vertex) {
                        for &ne in neighbors {
                            if !visited_edge[ne] {
                                visited_edge[ne] = true;
                                queue.push_back(ne);
                            }
                        }
                    }
                }
            }
            components.push(component);
        }

        // 各成分について centroid を追加、扇状に塞ぐ
        let mut new_vertices = mesh.vertices.clone();
        let mut new_indices = mesh.indices.clone();

        for component in &components {
            let mut ring_verts: Vec<u32> = Vec::new();
            let mut seen: HashSet<u32> = HashSet::new();
            for &ei in component {
                let (u, v) = border_edges[ei];
                for vertex in [u, v] {
                    if seen.insert(vertex) {
                        ring_verts.push(vertex);
                    }
                }
            }
            if ring_verts.len() < 3 {
                continue;
            }

            // Centroid position + normal
            let mut centroid_pos = glam::Vec3::ZERO;
            let mut centroid_normal = glam::Vec3::ZERO;
            for &vi in &ring_verts {
                let v = &new_vertices[vi as usize];
                centroid_pos += v.position;
                centroid_normal += v.normal;
            }
            let n = ring_verts.len() as f32;
            centroid_pos /= n;
            centroid_normal = if centroid_normal.length_squared() > 1e-10 {
                centroid_normal.normalize()
            } else {
                glam::Vec3::Y
            };
            // 単一 UV に統一 (image-to-3d ルール)
            let centroid_uv = new_vertices[ring_verts[0] as usize].uv;

            let centroid_idx = new_vertices.len() as u32;
            let mut centroid_vertex = Vertex::new(centroid_pos, centroid_normal);
            centroid_vertex.uv = centroid_uv;
            new_vertices.push(centroid_vertex);

            // 境界エッジ (u, v) は元三角形での順序、扇三角形は (centroid, v, u) で
            // 巻き向きが元の面と一貫する (border が反時計回りなら centroid が前面)
            for &ei in component {
                let (u, v) = border_edges[ei];
                new_indices.push(centroid_idx);
                new_indices.push(v);
                new_indices.push(u);
            }
        }

        Mesh {
            vertices: new_vertices,
            indices: new_indices,
        }
    }

    /// 三角形の連結成分ごとにサイズを計算、最大連結成分の `min_ratio` 未満の
    /// 孤立成分を除去する
    ///
    /// # 引数
    ///
    /// - `min_ratio`: 最大成分に対する相対サイズ (0.0 - 1.0)、これ未満の成分は削除
    ///   `image-to-3d` の実測では 0.01 (= 1%) 程度で十分小さい孤立かけらを除去
    ///
    /// # アルゴリズム (image-to-3d §A.3 準拠)
    ///
    /// 1. Union-Find で三角形の連結成分を判定 (共有エッジ経由)
    /// 2. 最大成分サイズを取得、`max_size × min_ratio` を threshold に
    /// 3. threshold 未満の成分に属する三角形を削除
    ///
    /// # 制約
    ///
    /// - 頂点は削除しない (index 連続性を保つため)、未使用頂点は残る
    ///   完全に detach したい場合は `merge_duplicate_vertices` と組合せる
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn drop_specks(mesh: &Mesh, min_ratio: f32) -> Mesh {
        let tri_count = mesh.indices.len() / 3;
        if tri_count == 0 {
            return mesh.clone();
        }

        // Union-Find (path compression)
        let mut parent: Vec<usize> = (0..tri_count).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }

        let mut edge_tris: HashMap<(u32, u32), usize> = HashMap::new();
        for t in 0..tri_count {
            let (a, b, c) = tri_verts(&mesh.indices, t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                if let Some(&other) = edge_tris.get(&key) {
                    let ra = find(&mut parent, t);
                    let rb = find(&mut parent, other);
                    if ra != rb {
                        parent[ra] = rb;
                    }
                } else {
                    edge_tris.insert(key, t);
                }
            }
        }

        // 成分サイズ
        let mut comp_size: HashMap<usize, usize> = HashMap::new();
        for t in 0..tri_count {
            let root = find(&mut parent, t);
            *comp_size.entry(root).or_insert(0) += 1;
        }

        let max_size = *comp_size.values().max().unwrap_or(&0);
        let threshold_f = (max_size as f32) * min_ratio;

        // 残す三角形のみ new_indices に
        // threshold_f 未満の成分を drop (== max_size 単一なら常に残る、
        // f32 比較で「1.04 未満は 1 を drop」の意味論が正しく取れる)
        let mut new_indices: Vec<u32> = Vec::with_capacity(mesh.indices.len());
        for t in 0..tri_count {
            let root = find(&mut parent, t);
            let size_f = comp_size[&root] as f32;
            if size_f < threshold_f {
                continue;
            }
            let base = t * 3;
            new_indices.push(mesh.indices[base]);
            new_indices.push(mesh.indices[base + 1]);
            new_indices.push(mesh.indices[base + 2]);
        }

        Mesh {
            vertices: mesh.vertices.clone(),
            indices: new_indices,
        }
    }
}

/// 三角形 index t の 3 頂点 index を返す
#[inline]
fn tri_verts(indices: &[u32], t: usize) -> (u32, u32, u32) {
    let base = t * 3;
    (indices[base], indices[base + 1], indices[base + 2])
}

/// Mesh quality metrics
#[derive(Debug, Clone)]
pub struct MeshQuality {
    /// Minimum triangle aspect ratio (0 = degenerate, 1 = equilateral)
    pub min_aspect_ratio: f32,
    /// Average triangle aspect ratio
    pub avg_aspect_ratio: f32,
    /// Minimum triangle area
    pub min_area: f32,
    /// Maximum triangle area
    pub max_area: f32,
    /// Average triangle area
    pub avg_area: f32,
    /// Total surface area
    pub total_area: f32,
}

/// Compute quality metrics for a mesh
pub fn compute_quality(mesh: &Mesh) -> MeshQuality {
    let tri_count = mesh.indices.len() / 3;
    if tri_count == 0 {
        return MeshQuality {
            min_aspect_ratio: 0.0,
            avg_aspect_ratio: 0.0,
            min_area: 0.0,
            max_area: 0.0,
            avg_area: 0.0,
            total_area: 0.0,
        };
    }

    let mut min_aspect = f32::MAX;
    let mut sum_aspect = 0.0_f32;
    let mut min_area = f32::MAX;
    let mut max_area = 0.0_f32;
    let mut total_area = 0.0_f32;

    for i in 0..tri_count {
        let base = i * 3;
        let v0 = mesh.vertices[mesh.indices[base] as usize].position;
        let v1 = mesh.vertices[mesh.indices[base + 1] as usize].position;
        let v2 = mesh.vertices[mesh.indices[base + 2] as usize].position;

        let a = (v1 - v0).length();
        let b = (v2 - v1).length();
        let c = (v0 - v2).length();

        let area = (v1 - v0).cross(v2 - v0).length() * 0.5;
        let perimeter = a + b + c;

        // Aspect ratio: 4 * sqrt(3) * area / perimeter^2 (1.0 = equilateral)
        let aspect = if perimeter > 1e-10 {
            (4.0 * 1.732_050_8 * area) / (perimeter * perimeter)
        } else {
            0.0
        };

        min_aspect = min_aspect.min(aspect);
        sum_aspect += aspect;
        min_area = min_area.min(area);
        max_area = max_area.max(area);
        total_area += area;
    }

    MeshQuality {
        min_aspect_ratio: min_aspect,
        avg_aspect_ratio: sum_aspect / tri_count as f32,
        min_area,
        max_area,
        avg_area: total_area / tri_count as f32,
        total_area,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
    use crate::types::SdfNode;
    use glam::Vec3;

    fn make_sphere_mesh(resolution: usize) -> Mesh {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config)
    }

    #[test]
    fn test_validate_sphere() {
        let mesh = make_sphere_mesh(16);
        let validation = validate_mesh(&mesh);

        assert!(validation.vertex_count > 0);
        assert!(validation.triangle_count > 0);
        assert_eq!(validation.degenerate_triangles, 0);
    }

    #[test]
    fn test_remove_degenerate() {
        let mut mesh = make_sphere_mesh(8);
        let orig_tri_count = mesh.triangle_count();

        // Add a degenerate triangle
        let idx = mesh.vertices.len() as u32;
        let v = Vertex::new(Vec3::ZERO, Vec3::Y);
        mesh.vertices.push(v);
        mesh.vertices.push(v);
        mesh.vertices.push(v);
        mesh.indices.push(idx);
        mesh.indices.push(idx + 1);
        mesh.indices.push(idx + 2);

        assert_eq!(mesh.triangle_count(), orig_tri_count + 1);

        let repaired = MeshRepair::remove_degenerate_triangles(&mesh);
        assert_eq!(repaired.triangle_count(), orig_tri_count);
    }

    #[test]
    fn test_merge_duplicates() {
        let mut mesh = Mesh::new();

        // Two identical vertices
        mesh.vertices.push(Vertex::new(Vec3::ZERO, Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(0.0, 0.0, 1e-8), Vec3::Y)); // near-duplicate
        mesh.vertices.push(Vertex::new(Vec3::X, Vec3::Y));
        mesh.indices = vec![0, 1, 2];

        let merged = MeshRepair::merge_duplicate_vertices(&mesh, 1e-6);
        assert_eq!(merged.vertices.len(), 2); // 3 -> 2
    }

    #[test]
    fn test_repair_all() {
        let mesh = make_sphere_mesh(8);
        let repaired = MeshRepair::repair_all(&mesh, 1e-6);

        let validation = validate_mesh(&repaired);
        assert_eq!(validation.degenerate_triangles, 0);
        assert_eq!(validation.inconsistent_normals, 0);
    }

    #[test]
    fn test_quality_metrics() {
        let mesh = make_sphere_mesh(16);
        let quality = compute_quality(&mesh);

        assert!(quality.total_area > 0.0);
        assert!(quality.avg_aspect_ratio > 0.0);
        assert!(quality.min_area > 0.0);
    }

    #[test]
    fn test_fix_normals() {
        let mesh = make_sphere_mesh(8);
        let fixed = MeshRepair::fix_normals(&mesh);
        let validation = validate_mesh(&fixed);
        assert_eq!(validation.inconsistent_normals, 0);
    }

    // ========================================================================
    // dotneet/image-to-3d §A 追加 3 手順テスト
    // ========================================================================

    fn make_open_pyramid() -> Mesh {
        // 4 頂点の底面のない三角錐 (底面は開いた穴)
        // 4 三角形 (側面)、境界エッジ 4 本 (底の 4 辺)
        let mut mesh = Mesh::new();
        // apex
        mesh.vertices
            .push(Vertex::new(Vec3::new(0.0, 1.0, 0.0), Vec3::Y));
        // base ring
        mesh.vertices
            .push(Vertex::new(Vec3::new(-1.0, 0.0, -1.0), Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(1.0, 0.0, -1.0), Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(1.0, 0.0, 1.0), Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(-1.0, 0.0, 1.0), Vec3::Y));
        // 4 side triangles (apex-i-i+1)
        mesh.indices = vec![0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1];
        mesh
    }

    #[test]
    fn test_orient_faces_preserves_count() {
        // orient_faces は三角形数と頂点数を変えない (向きだけ調整)
        let mesh = make_sphere_mesh(8);
        let oriented = MeshRepair::orient_faces(&mesh);
        assert_eq!(oriented.triangle_count(), mesh.triangle_count());
        assert_eq!(oriented.vertices.len(), mesh.vertices.len());
    }

    fn make_tetrahedron_outward() -> Mesh {
        // 正四面体 (原点中心)、外向き巻き
        // 4 頂点:
        //   a = ( 1,  1,  1), b = ( 1, -1, -1),
        //   c = (-1,  1, -1), d = (-1, -1,  1)
        // 4 面 (外向き winding、符号付き体積 = +∞ で確認済み):
        //   [0,1,2] [0,3,1] [0,2,3] [1,3,2]
        let mut mesh = Mesh::new();
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(1.0, -1.0, -1.0);
        let c = Vec3::new(-1.0, 1.0, -1.0);
        let d = Vec3::new(-1.0, -1.0, 1.0);
        for p in [a, b, c, d] {
            mesh.vertices.push(Vertex::new(p, p.normalize()));
        }
        mesh.indices = vec![0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2];
        mesh
    }

    #[test]
    fn test_orient_faces_outward_signed_volume() {
        // 手作り正四面体 (外向き) の全面を flip → 符号付き体積が負に
        // orient_faces を適用すると外向きに直る (符号付き体積 >= 0)
        let mesh = make_tetrahedron_outward();
        let vol_original = signed_volume_x6(&mesh);
        assert!(
            vol_original > 0.0,
            "hand-crafted tetrahedron should have positive volume, got {vol_original}"
        );

        // 全 flip
        let mut flipped = mesh.clone();
        for t in 0..(flipped.indices.len() / 3) {
            let base = t * 3;
            flipped.indices.swap(base + 1, base + 2);
        }
        let vol_flipped = signed_volume_x6(&flipped);
        assert!(
            vol_flipped < 0.0,
            "flipped tetrahedron should have negative volume, got {vol_flipped}"
        );

        // orient_faces で外向きに戻る
        let repaired = MeshRepair::orient_faces(&flipped);
        let vol_repaired = signed_volume_x6(&repaired);
        assert!(
            vol_repaired >= 0.0,
            "orient_faces should restore outward orientation, got {vol_repaired}"
        );
    }

    #[test]
    fn test_fill_holes_pyramid() {
        // 4 底辺 (境界) の開いたピラミッド → fill_holes で 4 底辺が塞がる
        let mesh = make_open_pyramid();
        let validation_before = validate_mesh(&mesh);
        assert_eq!(validation_before.boundary_edges, 4);

        let filled = MeshRepair::fill_holes(&mesh);
        let validation_after = validate_mesh(&filled);
        // 境界エッジが 0 になる (完全に塞がった)
        assert_eq!(
            validation_after.boundary_edges, 0,
            "fill_holes should close all boundary edges"
        );
        // centroid 頂点が 1 個追加、4 三角形が追加
        assert_eq!(filled.vertices.len(), mesh.vertices.len() + 1);
        assert_eq!(filled.triangle_count(), mesh.triangle_count() + 4);
    }

    #[test]
    fn test_fill_holes_closed_mesh_unchanged() {
        // 閉じた球体には境界エッジなし、fill_holes は何もしない
        let mesh = make_sphere_mesh(8);
        let filled = MeshRepair::fill_holes(&mesh);
        assert_eq!(filled.vertices.len(), mesh.vertices.len());
        assert_eq!(filled.indices.len(), mesh.indices.len());
    }

    #[test]
    fn test_drop_specks_removes_isolated() {
        // 球体 (主成分) + 孤立三角形 1 個 を作り、drop_specks で孤立成分が消える
        let mut mesh = make_sphere_mesh(8);
        let orig_tri = mesh.triangle_count();

        // 孤立三角形を追加 (球体から遠く離した 3 頂点)
        let idx = mesh.vertices.len() as u32;
        mesh.vertices
            .push(Vertex::new(Vec3::new(100.0, 0.0, 0.0), Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(101.0, 0.0, 0.0), Vec3::Y));
        mesh.vertices
            .push(Vertex::new(Vec3::new(100.0, 1.0, 0.0), Vec3::Y));
        mesh.indices.extend_from_slice(&[idx, idx + 1, idx + 2]);
        assert_eq!(mesh.triangle_count(), orig_tri + 1);

        // 1% threshold で孤立成分 (1 三角形) を drop
        let cleaned = MeshRepair::drop_specks(&mesh, 0.01);
        assert_eq!(
            cleaned.triangle_count(),
            orig_tri,
            "drop_specks should remove the isolated triangle"
        );
    }

    #[test]
    fn test_drop_specks_preserves_main() {
        // 単一連結成分の球体 → threshold 0.5 でも全部残る
        let mesh = make_sphere_mesh(8);
        let cleaned = MeshRepair::drop_specks(&mesh, 0.5);
        assert_eq!(cleaned.triangle_count(), mesh.triangle_count());
    }

    #[test]
    fn test_repair_pipeline_pyramid() {
        // orient_faces → fill_holes → drop_specks の統合パイプラインで
        // 開いたピラミッドが閉じたメッシュになる
        let mesh = make_open_pyramid();
        let oriented = MeshRepair::orient_faces(&mesh);
        let filled = MeshRepair::fill_holes(&oriented);
        let cleaned = MeshRepair::drop_specks(&filled, 0.01);

        let validation = validate_mesh(&cleaned);
        assert_eq!(validation.boundary_edges, 0);
        // 三角形は 4 (元) + 4 (fill) = 8 のはず
        assert_eq!(cleaned.triangle_count(), 8);
    }

    fn signed_volume_x6(mesh: &Mesh) -> f32 {
        let mut sum = 0.0_f32;
        for t in 0..(mesh.indices.len() / 3) {
            let base = t * 3;
            let v0 = mesh.vertices[mesh.indices[base] as usize].position;
            let v1 = mesh.vertices[mesh.indices[base + 1] as usize].position;
            let v2 = mesh.vertices[mesh.indices[base + 2] as usize].position;
            sum += v0.dot(v1.cross(v2));
        }
        sum
    }
}

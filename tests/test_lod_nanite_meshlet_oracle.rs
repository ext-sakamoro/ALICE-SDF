//! Oracles for LOD chains, Nanite clusters and meshlets
//!
//! The expected values are built independently of the code under test:
//!
//! - partitions: the clusters / meshlets of a level, put back together, are
//!   exactly the triangle multiset of the marching-cubes mesh they were cut
//!   from (the mesh is regenerated here with the same resolution)
//! - capacity limits, local index ranges and the bounding spheres / AABBs are
//!   recomputed from the vertex positions
//! - the Nanite groups and DAG are compared with octree regions rebuilt from
//!   the documented assignment rule (centroid cell, depth per level, nearest
//!   enclosing coarser region), and the cut with the selection rule evaluated
//!   in f64; cluster errors with the exact distance of their triangles to the
//!   unit sphere (closed form) and a chord bound
//! - the LOD error bound is checked with the two-sided Hausdorff distance to
//!   the exact unit sphere (mesh → sphere in closed form, sphere → mesh with a
//!   brute-force point-triangle distance)
//! - view-cone and back-face tests are compared with the angle formulas
//!   evaluated in f64 (`acos` / `asin`)
//!
//! Every test counts its comparisons and fails when the count is 0.
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::mesh::lod::{
    generate_lod_chain, generate_lod_chain_decimated, ContinuousLod, DecimationLodConfig, LodChain,
    LodConfig, LodSelector,
};
use alice_sdf::mesh::lod_persist::{LodChainConfig, LodChainPersist};
use alice_sdf::mesh::meshlet::{
    build_meshlets, build_meshlets_adjacency, build_meshlets_scan, Meshlet, MeshletConfig,
};
use alice_sdf::mesh::nanite::{
    generate_nanite_mesh, ClusterBounds, NaniteConfig, NaniteMesh, NormalCone,
    CLUSTER_MAX_TRIANGLES, CLUSTER_MAX_VERTICES,
};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn mc(sdf: &SdfNode, res: u32, lo: Vec3, hi: Vec3) -> Mesh {
    sdf_to_mesh(
        sdf,
        lo,
        hi,
        &MarchingCubesConfig {
            resolution: res as usize,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        },
    )
}

const fn key(p: Vec3) -> [u32; 3] {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

/// Triangle multiset of a mesh as sorted position triples (orientation kept)
fn tri_multiset(tris: impl Iterator<Item = [Vec3; 3]>) -> Vec<[[u32; 3]; 3]> {
    let mut v: Vec<_> = tris.map(|t| [key(t[0]), key(t[1]), key(t[2])]).collect();
    v.sort_unstable();
    v
}

fn mesh_tris(m: &Mesh) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    m.indices.chunks(3).map(|c| {
        [
            m.vertices[c[0] as usize].position,
            m.vertices[c[1] as usize].position,
            m.vertices[c[2] as usize].position,
        ]
    })
}

/// Independent AABB + "sphere contains every point" check
fn check_bounds(b: &ClusterBounds, pts: &[Vec3]) -> usize {
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for &p in pts {
        lo = lo.min(p);
        hi = hi.max(p);
    }
    assert_eq!(b.aabb_min, lo);
    assert_eq!(b.aabb_max, hi);
    let mut n = 0;
    for &p in pts {
        let d = (p - b.center).length();
        assert!(
            d <= b.radius * (1.0 + 1e-6) + 1e-7,
            "point {p} at {d} outside sphere r={}",
            b.radius
        );
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------------
// Meshlets
// ---------------------------------------------------------------------------

fn meshlet_global_tris(m: &Meshlet) -> impl Iterator<Item = [u32; 3]> + '_ {
    m.triangles.chunks(3).map(|t| {
        [
            m.vertices[t[0] as usize],
            m.vertices[t[1] as usize],
            m.vertices[t[2] as usize],
        ]
    })
}

fn check_meshlets(mesh: &Mesh, cfg: &MeshletConfig, ms: &[Meshlet]) -> usize {
    let mut checks = 0;
    // partition by global index triple (orientation kept)
    let mut want: Vec<[u32; 3]> = mesh.indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
    let mut got: Vec<[u32; 3]> = ms.iter().flat_map(meshlet_global_tris).collect();
    want.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, want, "meshlets are not a partition of the triangles");
    checks += want.len();

    for m in ms {
        assert!(m.vertex_count() <= cfg.max_vertices, "vertex cap");
        assert!(m.triangle_count() <= cfg.max_triangles, "triangle cap");
        assert!(m.triangle_count() > 0);
        assert_eq!(m.triangles.len(), 3 * m.triangle_count());
        assert_eq!(m.vertex_count(), m.vertices.len());
        // vertices are distinct and every one of them is referenced
        let mut seen = vec![false; m.vertices.len()];
        for &l in &m.triangles {
            assert!((l as usize) < m.vertices.len(), "local index out of range");
            seen[l as usize] = true;
        }
        assert!(seen.iter().all(|s| *s), "unreferenced meshlet vertex");
        let mut dedup = m.vertices.clone();
        dedup.sort_unstable();
        dedup.dedup();
        assert_eq!(dedup.len(), m.vertices.len(), "duplicate meshlet vertex");

        let pts: Vec<Vec3> = m
            .vertices
            .iter()
            .map(|&i| mesh.vertices[i as usize].position)
            .collect();
        checks += check_bounds(&m.bounds, &pts);

        // every non-degenerate face normal lies inside the normal cone
        for t in m.triangles.chunks(3) {
            let (a, b, c) = (pts[t[0] as usize], pts[t[1] as usize], pts[t[2] as usize]);
            let n = (b - a).cross(c - a);
            if n.length_squared() > 1e-12 {
                let n = n.normalize();
                assert!(
                    m.normal_cone.axis.dot(n) >= m.normal_cone.cutoff_cos - 1e-5,
                    "face normal outside the cone"
                );
                checks += 1;
            }
        }
    }
    checks
}

#[test]
fn meshlets_partition_the_mesh_within_the_caps() {
    let mesh = mc(
        &SdfNode::sphere(1.0),
        24,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
    );
    let shape = SdfNode::box3d(2.0, 1.2, 1.6).smooth_union(SdfNode::sphere(0.7), 0.2);
    let mesh2 = mc(&shape, 20, Vec3::splat(-2.0), Vec3::splat(2.0));
    assert!(mesh.triangle_count() > 500 && mesh2.triangle_count() > 500);

    let mut checks = 0;
    for m in [&mesh, &mesh2] {
        for (mv, mt) in [(3, 1), (3, 7), (16, 7), (64, 124), (255, 512)] {
            for grow in [false, true] {
                let cfg = MeshletConfig {
                    max_vertices: mv,
                    max_triangles: mt,
                    cone_weight: if grow { 0.5 } else { 0.0 },
                    adjacency_grow: grow,
                };
                let ms = build_meshlets(m, &cfg);
                checks += check_meshlets(m, &cfg, &ms);
                let direct = if grow {
                    build_meshlets_adjacency(m, &cfg)
                } else {
                    build_meshlets_scan(m, &cfg)
                };
                assert_eq!(direct.len(), ms.len());
            }
        }
        let q = MeshletConfig::quality();
        assert!(q.adjacency_grow && q.cone_weight > 0.0);
        checks += check_meshlets(m, &q, &build_meshlets(m, &q));
    }
    assert!(checks > 0);
}

/// V1 is documented as a sequential greedy scan: the meshlets are the index
/// buffer in order, and every meshlet but the last is closed only because the
/// next triangle would break a cap.
#[test]
fn meshlet_scan_is_the_sequential_greedy_cut() {
    let mesh = mc(
        &SdfNode::torus(1.0, 0.35),
        24,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
    );
    let mut checks = 0;
    for (mv, mt) in [(3, 2), (10, 6), (64, 124)] {
        let cfg = MeshletConfig {
            max_vertices: mv,
            max_triangles: mt,
            ..Default::default()
        };
        let ms = build_meshlets_scan(&mesh, &cfg);
        let order: Vec<[u32; 3]> = ms.iter().flat_map(meshlet_global_tris).collect();
        let want: Vec<[u32; 3]> = mesh.indices.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        assert_eq!(order, want);
        let mut next = 0usize;
        for (k, m) in ms.iter().enumerate() {
            next += m.triangle_count();
            if k + 1 < ms.len() {
                // distinct vertices of the next triangle that the meshlet lacks
                let mut distinct: Vec<u32> = want[next]
                    .iter()
                    .copied()
                    .filter(|v| !m.vertices.contains(v))
                    .collect();
                distinct.sort_unstable();
                distinct.dedup();
                assert!(
                    m.vertex_count() + distinct.len() > mv || m.triangle_count() == mt,
                    "meshlet {k} closed while the next triangle still fit"
                );
                checks += 1;
            }
        }
    }
    assert!(checks > 0);
}

/// A meshlet that holds one triangle needs three vertex slots, so a cap below
/// 3 is an invalid config and returns no meshlets (as documented)
#[test]
fn meshlet_vertex_cap_below_three_is_rejected() {
    let mesh = mc(
        &SdfNode::sphere(1.0),
        8,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
    );
    let mut checks = 0;
    for mv in [1, 2] {
        for grow in [false, true] {
            let cfg = MeshletConfig {
                max_vertices: mv,
                adjacency_grow: grow,
                ..Default::default()
            };
            assert!(build_meshlets(&mesh, &cfg).is_empty());
            checks += 1;
        }
    }
    assert!(build_meshlets(&Mesh::new(), &MeshletConfig::default()).is_empty());
    assert!(checks > 0);
}

// ---------------------------------------------------------------------------
// Normal cone and cluster bounds
// ---------------------------------------------------------------------------

fn unit(theta: f64, phi: f64) -> Vec3 {
    Vec3::new(
        (theta.sin() * phi.cos()) as f32,
        (theta.sin() * phi.sin()) as f32,
        theta.cos() as f32,
    )
}

/// Normals on a circle at polar angle `β` around +Z: the cone axis is +Z and
/// the cutoff is cos β (closed form). The cluster is back-facing for view
/// direction `v` exactly when every normal has `n · v >= 0`, i.e. when
/// `angle(z, v) <= 90° - β`.
#[test]
fn normal_cone_matches_the_closed_form() {
    let mut checks = 0;
    for beta_deg in [5.0f64, 20.0, 44.0, 46.0, 60.0, 80.0] {
        let beta = beta_deg.to_radians();
        let normals: Vec<Vec3> = (0..12)
            .map(|k| unit(beta, k as f64 * std::f64::consts::TAU / 12.0))
            .collect();
        let cone = NormalCone::from_normals(&normals);
        assert!((cone.axis - Vec3::Z).length() < 1e-5);
        assert!((cone.cutoff_cos as f64 - beta.cos()).abs() < 1e-5);
        assert!(cone.apex.is_none());
        for n in &normals {
            assert!(cone.axis.dot(*n) >= cone.cutoff_cos - 1e-6);
        }
        // views away from the boundary 90° - β by more than 1°
        for view_deg in (0..=180).step_by(3) {
            let gamma = (view_deg as f64).to_radians();
            let limit = std::f64::consts::FRAC_PI_2 - beta;
            if (gamma - limit).abs() < 1f64.to_radians() {
                continue;
            }
            let v = unit(gamma, 0.3);
            let want = gamma <= limit;
            assert_eq!(
                cone.is_backface_culled(v),
                want,
                "β={beta_deg} view={view_deg}"
            );
            // independent: every normal of the dense ring is back-facing
            if want {
                for k in 0..360 {
                    let n = unit(beta, k as f64 * std::f64::consts::TAU / 360.0);
                    assert!(n.dot(v) >= -1e-5);
                }
            }
            checks += 1;
        }
    }
    // a 60° ring plus one normal 120° from +Z: half-angle > 90°, nothing culls
    let mut ring: Vec<Vec3> = (0..12)
        .map(|k| unit(60f64.to_radians(), k as f64 * std::f64::consts::TAU / 12.0))
        .collect();
    ring.push(unit(120f64.to_radians(), 0.0));
    let wide = NormalCone::from_normals(&ring);
    assert!(wide.cutoff_cos < 0.0);
    for v in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::NEG_Z] {
        assert!(!wide.is_backface_culled(v));
        checks += 1;
    }
    let u = NormalCone::unbounded();
    assert_eq!(u.cutoff_cos, -1.0);
    assert!(!u.is_backface_culled(u.axis));
    assert_eq!(NormalCone::from_normals(&[]).cutoff_cos, -1.0);
    assert!(checks > 0);
}

/// The apex lies behind every face plane whose normal points along the axis
/// (`dot(apex - c_i, n_i) <= 0`), as documented
#[test]
fn normal_cone_apex_is_behind_every_face() {
    let mesh = mc(
        &SdfNode::sphere(1.0),
        16,
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
    );
    let ms = build_meshlets(&mesh, &MeshletConfig::default());
    let mut checks = 0;
    for m in &ms {
        let mut normals = Vec::new();
        let mut centers = Vec::new();
        for t in meshlet_global_tris(m) {
            let [a, b, c] = t.map(|i| mesh.vertices[i as usize].position);
            let n = (b - a).cross(c - a);
            if n.length_squared() > 1e-12 {
                normals.push(n.normalize());
                centers.push((a + b + c) / 3.0);
            }
        }
        // the outside of the sphere (convex) and the inside (concave: the
        // patch centroid is in front of the faces, so the apex must move back)
        for sign in [1.0f32, -1.0] {
            let ns: Vec<Vec3> = normals.iter().map(|n| *n * sign).collect();
            let cone = NormalCone::from_normals_and_positions(&ns, &centers);
            let apex = cone.apex.expect("bounded cone carries an apex");
            for (n, c) in ns.iter().zip(&centers) {
                if cone.axis.dot(*n) > 1e-6 {
                    assert!((apex - *c).dot(*n) <= 1e-4, "apex in front of a face");
                    checks += 1;
                }
            }
        }
    }
    assert!(checks > 0);
}

/// Sphere vs circular view cone: visible iff the angle between the cone axis
/// and the direction to the center is at most `θ + asin(r / d)`
#[test]
fn cluster_bounds_visibility_matches_the_cone_angle() {
    let mut checks = 0;
    let view_pos = Vec3::new(0.3, -0.2, 0.1);
    let view_dir = Vec3::new(0.0, 0.0, 1.0);
    for half_deg in [10.0f64, 30.0, 60.0, 89.0, 100.0, 150.0] {
        let fov_cos = half_deg.to_radians().cos() as f32;
        for r in [0.0f32, 0.5, 2.0, 9.0] {
            for ang_deg in (0..=180).step_by(5) {
                let dist = 10.0f32;
                let dir = unit((ang_deg as f64).to_radians(), 1.1);
                let center = view_pos + dir * dist;
                let b = ClusterBounds {
                    center,
                    radius: r,
                    aabb_min: center - Vec3::splat(r),
                    aabb_max: center + Vec3::splat(r),
                };
                let alpha = (r as f64 / dist as f64).asin();
                let margin = (ang_deg as f64).to_radians() - (half_deg.to_radians() + alpha);
                if margin.abs() < 0.5f64.to_radians() {
                    continue;
                }
                assert_eq!(
                    b.is_visible(view_pos, view_dir, fov_cos),
                    margin <= 0.0,
                    "half={half_deg} r={r} angle={ang_deg}"
                );
                checks += 1;
            }
        }
    }
    // a viewer inside the sphere sees it in every direction
    let b = ClusterBounds::from_vertices(&[Vec3::splat(-1.0), Vec3::splat(1.0)]);
    assert!(b.is_visible(Vec3::ZERO, Vec3::NEG_X, 0.99));
    // screen error: geometric error over max(distance, radius), times the height
    let se = b.screen_error(Vec3::new(10.0, 0.0, 0.0), 0.02, 1080.0);
    assert!((se - 0.02 / 10.0 * 1080.0).abs() < 1e-4);
    let inside = b.screen_error(Vec3::ZERO, 0.02, 1080.0);
    assert!((inside - 0.02 / b.radius * 1080.0).abs() < 1e-4);
    assert!(checks > 500);
}

// ---------------------------------------------------------------------------
// Nanite clusters
// ---------------------------------------------------------------------------

fn nanite(sdf: &SdfNode, cfg: &NaniteConfig) -> NaniteMesh {
    generate_nanite_mesh(sdf, Vec3::splat(-2.0), Vec3::splat(2.0), cfg)
}

#[test]
fn nanite_levels_are_partitions_of_the_marching_cubes_mesh() {
    let shapes = [
        SdfNode::sphere(1.0),
        SdfNode::box3d(2.0, 1.0, 1.4).subtract(SdfNode::sphere(0.6)),
    ];
    let mut checks = 0;
    for sdf in &shapes {
        for max_tris in [CLUSTER_MAX_TRIANGLES, 32] {
            let cfg = NaniteConfig {
                lod_levels: 3,
                base_resolution: 32,
                max_triangles_per_cluster: max_tris,
                use_tight_aabb: false,
                ..NaniteConfig::default()
            };
            let n = nanite(sdf, &cfg);
            assert_eq!(n.lod_levels.len(), 3);
            for lod in 0..3u32 {
                let res = ((32.0 * 0.5f32.powi(lod as i32)) as u32).max(4);
                assert_eq!(n.lod_levels[lod as usize].resolution, res);
                let reference = mc(sdf, res, Vec3::splat(-2.0), Vec3::splat(2.0));
                let want = tri_multiset(mesh_tris(&reference));
                let clusters = n.clusters_at_lod(lod);
                let got = tri_multiset(clusters.iter().flat_map(|c| {
                    c.triangles.iter().map(move |t| {
                        [
                            c.vertices[t.a as usize].position,
                            c.vertices[t.b as usize].position,
                            c.vertices[t.c as usize].position,
                        ]
                    })
                }));
                assert_eq!(got, want, "lod {lod}: clusters differ from the mesh");
                assert_eq!(tri_multiset(mesh_tris(&n.to_mesh(lod))), want);
                assert_eq!(
                    n.lod_levels[lod as usize].triangle_count as usize,
                    want.len()
                );
                checks += want.len();
                for c in &clusters {
                    assert!(c.triangle_count() <= max_tris);
                    assert!(c.vertex_count() <= CLUSTER_MAX_VERTICES);
                    let mut used = vec![false; c.vertex_count()];
                    for t in &c.triangles {
                        for i in [t.a, t.b, t.c] {
                            used[i as usize] = true;
                        }
                    }
                    assert!(used.iter().all(|u| *u), "unreferenced cluster vertex");
                    let pts: Vec<Vec3> = c.vertices.iter().map(|v| v.position).collect();
                    checks += check_bounds(&c.bounds, &pts);
                }
            }
            // triangle counts drop level by level
            for w in n.lod_levels.windows(2) {
                assert!(w[1].triangle_count < w[0].triangle_count);
            }
            // bookkeeping
            let ids: Vec<u32> = n.clusters.iter().map(|c| c.id).collect();
            let mut uniq = ids.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), ids.len());
            for &id in &ids {
                assert_eq!(n.get_cluster(id).map(|c| c.id), Some(id));
            }
            assert!(n.get_cluster(u32::MAX).is_none());
            assert_eq!(
                n.total_vertices(),
                n.clusters.iter().map(|c| c.vertices.len()).sum::<usize>()
            );
            assert_eq!(
                n.total_triangles,
                n.clusters_at_lod(0)
                    .iter()
                    .map(|c| c.triangle_count())
                    .sum::<usize>()
            );
            let all: Vec<Vec3> = n
                .clusters
                .iter()
                .flat_map(|c| c.vertices.iter().map(|v| v.position))
                .collect();
            checks += check_bounds(&n.bounds, &all);
            for g in &n.groups {
                let pts: Vec<Vec3> = g
                    .cluster_ids
                    .iter()
                    .flat_map(|&id| {
                        n.get_cluster(id)
                            .unwrap()
                            .vertices
                            .iter()
                            .map(|v| v.position)
                    })
                    .collect();
                checks += check_bounds(&g.bounds, &pts);
            }
        }
    }
    assert!(checks > 0);
}

#[test]
fn nanite_presets_increase_in_detail() {
    let (p, m, h) = (
        NaniteConfig::preview(),
        NaniteConfig::medium_detail(),
        NaniteConfig::high_detail(),
    );
    assert!(p.base_resolution < m.base_resolution && m.base_resolution < h.base_resolution);
    assert!(p.lod_levels < m.lod_levels && m.lod_levels < h.lod_levels);
    // the finest level of each preset keeps at least the 4-cell floor
    let mut checks = 0;
    for c in [&p, &m, &h] {
        let coarsest =
            (c.base_resolution as f32 * c.lod_factor.powi(c.lod_levels as i32 - 1)) as u32;
        assert!(coarsest >= 1);
        checks += 1;
    }
    let n = nanite(&SdfNode::sphere(1.0), &m);
    assert_eq!(n.lod_levels.first().map(|l| l.resolution), Some(64));
    assert!(checks > 0);
}

// --- the region tree, recomputed independently ------------------------------
//
// The documented rule (`generate_nanite_mesh`): the root cube starts at the
// minimum corner of the meshing bounds and its edge is their largest extent;
// level k uses octree depth min(depth_{k-1}, max(0, round(log2(res_k / 8))));
// a triangle belongs to the cell holding its centroid (a + b + c) / 3, index
// floor((centroid - origin) * (2^depth / edge)) per axis, clamped. The parent
// of a region is the enclosing cell at the nearest coarser level that has
// triangles. Everything below is rebuilt from that rule and the marching-cubes
// meshes, without reading the groups or the DAG of the generated mesh.

fn expected_depths(resolutions: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut limit = u32::MAX;
    for &r in resolutions {
        let d = ((f64::from(r) / 8.0).log2().round().max(0.0) as u32).min(limit);
        out.push(d);
        limit = d;
    }
    out
}

fn cell_of(t: [Vec3; 3], origin: Vec3, side: f32, depth: u32) -> [u32; 3] {
    let n = 1u32 << depth;
    let rel = ((t[0] + t[1] + t[2]) / 3.0 - origin) * (n as f32 / side);
    let idx = |x: f32| (x.floor().max(0.0) as u32).min(n - 1);
    [idx(rel.x), idx(rel.y), idx(rel.z)]
}

fn cluster_tris(
    c: &alice_sdf::mesh::nanite::NaniteCluster,
) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    c.triangles.iter().map(move |t| {
        [
            c.vertices[t.a as usize].position,
            c.vertices[t.b as usize].position,
            c.vertices[t.c as usize].position,
        ]
    })
}

type Cell = (u32, [u32; 3]);

/// Expected regions: (level, cell) -> triangle multiset, and the parent map
#[allow(clippy::type_complexity)]
fn expected_regions(
    sdf: &SdfNode,
    n: &NaniteMesh,
    lo: Vec3,
    hi: Vec3,
) -> (
    std::collections::BTreeMap<Cell, Vec<[[u32; 3]; 3]>>,
    std::collections::BTreeMap<Cell, Option<Cell>>,
    usize,
) {
    use std::collections::BTreeMap;
    let side = (hi - lo).max_element();
    let res: Vec<u32> = n.lod_levels.iter().map(|l| l.resolution).collect();
    let depths = expected_depths(&res);
    let mut regions: BTreeMap<Cell, Vec<[Vec3; 3]>> = BTreeMap::new();
    let mut straddling = 0;
    for (l, &r) in n.lod_levels.iter().zip(&res) {
        let d = depths[l.level as usize];
        let m = mc(sdf, r, lo, hi);
        for t in mesh_tris(&m) {
            let c = cell_of(t, lo, side, d);
            // a triangle whose vertices lie in different cells sits on a cell
            // boundary; it must still land in exactly one region
            let vc: Vec<[u32; 3]> = t.iter().map(|&p| cell_of([p, p, p], lo, side, d)).collect();
            if vc.iter().any(|x| *x != vc[0]) {
                straddling += 1;
            }
            regions.entry((l.level, c)).or_default().push(t);
        }
    }
    let mut parent: BTreeMap<Cell, Option<Cell>> = BTreeMap::new();
    for &(lvl, cell) in regions.keys() {
        let d = depths[lvl as usize];
        let p = n
            .lod_levels
            .iter()
            .filter(|l| l.level > lvl)
            .map(|l| {
                let s = d - depths[l.level as usize];
                (l.level, cell.map(|x| x >> s))
            })
            .find(|k| regions.contains_key(k));
        parent.insert((lvl, cell), p);
    }
    let regions = regions
        .into_iter()
        .map(|(k, v)| (k, tri_multiset(v.into_iter())))
        .collect();
    (regions, parent, straddling)
}

/// The groups are exactly the octree cells of each level (every triangle in
/// exactly one region, triangles on cell boundaries included), the DAG links
/// each region to its enclosing region, errors never decrease towards the
/// parent and a parent's sphere contains its children's.
///
/// This replaces the previous oracle that compared the DAG with a brute-force
/// scan of overlapping bounding spheres between adjacent levels: that was the
/// old definition of parent and child, under which a child could belong to
/// several parents with different errors and no cut could cover the surface
/// exactly once.
#[test]
fn nanite_groups_are_the_octree_regions_of_each_level() {
    use std::collections::{BTreeMap, BTreeSet};
    let (lo, hi) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    let mut checks = 0;
    let mut straddling_total = 0;
    // the last two have regions whose own measured error is below that of a
    // child region, so the propagation of the error to the parent is exercised
    for sdf in [
        SdfNode::sphere(1.0),
        SdfNode::torus(1.0, 0.3),
        SdfNode::box3d(2.0, 1.0, 1.4).subtract(SdfNode::sphere(0.6)),
        SdfNode::sphere(1.0).union(SdfNode::sphere(0.15).translate(1.1, 0.0, 0.0)),
        SdfNode::gyroid(5.0, 0.05).intersection(SdfNode::sphere(1.6)),
    ] {
        let n = nanite(
            &sdf,
            &NaniteConfig {
                lod_levels: 4,
                base_resolution: 48,
                max_triangles_per_cluster: 48,
                use_tight_aabb: false,
                ..NaniteConfig::default()
            },
        );
        let (want, want_parent, straddling) = expected_regions(&sdf, &n, lo, hi);
        straddling_total += straddling;
        let side = (hi - lo).max_element();
        let res: Vec<u32> = n.lod_levels.iter().map(|l| l.resolution).collect();
        let depths = expected_depths(&res);

        // group -> cell, from the triangles of its clusters
        let mut cell_of_group: BTreeMap<u32, Cell> = BTreeMap::new();
        let mut got: BTreeMap<Cell, Vec<[[u32; 3]; 3]>> = BTreeMap::new();
        for g in &n.groups {
            let d = depths[g.lod_level as usize];
            let tris: Vec<[Vec3; 3]> = g
                .cluster_ids
                .iter()
                .flat_map(|&id| cluster_tris(n.get_cluster(id).unwrap()).collect::<Vec<_>>())
                .collect();
            assert!(!tris.is_empty());
            let cells: BTreeSet<[u32; 3]> = tris.iter().map(|&t| cell_of(t, lo, side, d)).collect();
            assert_eq!(cells.len(), 1, "group {} spans {} cells", g.id, cells.len());
            let key = (g.lod_level, *cells.iter().next().unwrap());
            assert!(
                got.insert(key, tri_multiset(tris.into_iter())).is_none(),
                "two groups share {key:?}"
            );
            cell_of_group.insert(g.id, key);
            for &id in &g.cluster_ids {
                assert_eq!(n.get_cluster(id).unwrap().lod_level, g.lod_level);
            }
        }
        assert_eq!(got.len(), want.len());
        for (k, tris) in &want {
            assert_eq!(got.get(k), Some(tris), "region {k:?}");
            checks += tris.len();
        }

        // every cluster in exactly one group
        let mut seen = BTreeSet::new();
        for g in &n.groups {
            for &id in &g.cluster_ids {
                assert!(seen.insert(id));
            }
        }
        assert_eq!(seen.len(), n.clusters.len());

        // DAG: parent_ids = clusters of the expected parent region, child_ids =
        // clusters of every region whose expected parent is this one
        let group_at: BTreeMap<Cell, &alice_sdf::mesh::nanite::ClusterGroup> =
            n.groups.iter().map(|g| (cell_of_group[&g.id], g)).collect();
        for g in &n.groups {
            let key = cell_of_group[&g.id];
            let mut want_parents: Vec<u32> = want_parent[&key]
                .map(|p| group_at[&p].cluster_ids.clone())
                .unwrap_or_default();
            want_parents.sort_unstable();
            let mut want_children: Vec<u32> = want_parent
                .iter()
                .filter(|(_, p)| **p == Some(key))
                .flat_map(|(c, _)| group_at[c].cluster_ids.clone())
                .collect();
            want_children.sort_unstable();
            for &id in &g.cluster_ids {
                let c = n.get_cluster(id).unwrap();
                let mut p = c.parent_ids.clone();
                p.sort_unstable();
                let mut ch = c.child_ids.clone();
                ch.sort_unstable();
                assert_eq!(p, want_parents, "parents of cluster {id}");
                assert_eq!(ch, want_children, "children of cluster {id}");
                assert_eq!(c.geometric_error, g.max_error);
                checks += 1;
            }
            if let Some(pk) = want_parent[&key] {
                let pg = group_at[&pk];
                assert!(
                    pg.max_error >= g.max_error,
                    "error drops towards the parent"
                );
                let reach = (g.bounds.center - pg.bounds.center).length() + g.bounds.radius;
                assert!(
                    reach <= pg.bounds.radius * (1.0 + 1e-6) + 1e-6,
                    "child sphere sticks out"
                );
            }
        }
        // the parent relation holds at least one link per non-root level
        assert!(want_parent.values().filter(|p| p.is_some()).count() > 0);
    }
    assert!(
        straddling_total > 0,
        "no triangle on a cell boundary was exercised"
    );
    assert!(checks > 0);
}

/// A centroid exactly on a cell plane goes to the cell above it, and a
/// triangle whose vertices touch several cells is never duplicated: checked on
/// meshes whose triangles straddle the planes of every octree depth
#[test]
fn nanite_boundary_triangles_land_in_one_region() {
    let (lo, hi) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    // a box whose faces lie on the cell planes x = 0 and y = 0 of every depth
    // and a sphere centred on a cell corner
    let mut checks = 0;
    for sdf in [SdfNode::box3d(1.0, 1.0, 0.75), SdfNode::sphere(0.9)] {
        let n = nanite(
            &sdf,
            &NaniteConfig {
                lod_levels: 3,
                base_resolution: 32,
                use_tight_aabb: false,
                ..NaniteConfig::default()
            },
        );
        let (want, _, straddling) = expected_regions(&sdf, &n, lo, hi);
        assert!(straddling > 0);
        // total triangles over all groups == total over the expected regions
        let total_want: usize = want.values().map(|v| v.len()).sum();
        let total_got: usize = n.clusters.iter().map(|c| c.triangle_count()).sum();
        assert_eq!(
            total_got, total_want,
            "a boundary triangle was duplicated or lost"
        );
        // each triangle key occurs in exactly one group of its level
        for l in &n.lod_levels {
            let mut keys: Vec<[[u32; 3]; 3]> = n
                .groups
                .iter()
                .filter(|g| g.lod_level == l.level)
                .flat_map(|g| {
                    tri_multiset(
                        g.cluster_ids
                            .iter()
                            .flat_map(|&id| {
                                cluster_tris(n.get_cluster(id).unwrap()).collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
                })
                .collect();
            let len = keys.len();
            keys.sort_unstable();
            let reference = tri_multiset(mesh_tris(&mc(&sdf, l.resolution, lo, hi)));
            assert_eq!(keys, reference);
            checks += len;
        }
    }
    assert!(checks > 0);
}

/// Exact one-sided distance from a triangle to the unit sphere: |p| is convex,
/// so its maximum is at a vertex and its minimum at the closest point to O
fn tri_err_unit_sphere([a, b, c]: [Vec3; 3]) -> f32 {
    let far = a.length().max(b.length()).max(c.length());
    let near = closest_on_triangle(Vec3::ZERO, a, b, c).length();
    (far - 1.0).abs().max((1.0 - near).abs())
}

/// Chord bound: on a segment whose ends are at radius >= rho,
/// |p| >= sqrt(rho² − (len/2)²); applying it twice (vertex to a point on the
/// opposite edge) gives |p| >= sqrt(rho² − l²/2) on a triangle with longest
/// edge l
fn tri_chord_bound([a, b, c]: [Vec3; 3]) -> f32 {
    let far = a.length().max(b.length()).max(c.length());
    let rho = a.length().min(b.length()).min(c.length());
    let l = (b - a).length().max((c - b).length()).max((a - c).length());
    let inner = (rho * rho - 0.5 * l * l).max(0.0).sqrt();
    (far - 1.0).max(1.0 - inner)
}

fn group_tris<'a>(
    n: &'a NaniteMesh,
    g: &'a alice_sdf::mesh::nanite::ClusterGroup,
) -> Vec<[Vec3; 3]> {
    g.cluster_ids
        .iter()
        .flat_map(|&id| cluster_tris(n.get_cluster(id).unwrap()).collect::<Vec<_>>())
        .collect()
}

/// The cluster error is the measured surface error: for the unit sphere every
/// group's error equals the exact one-sided Hausdorff distance of its own
/// triangles to the sphere (closed form, per triangle), raised to the errors
/// of its children, and stays under the chord bound
#[test]
fn nanite_cluster_error_is_the_measured_distance_to_the_surface() {
    let n = nanite(&SdfNode::sphere(1.0), &NaniteConfig::medium_detail());
    let mut by_cluster = std::collections::HashMap::new();
    for g in &n.groups {
        for &id in &g.cluster_ids {
            by_cluster.insert(id, g);
        }
    }
    // own exact errors, then propagate through the child links (finest first)
    let mut want: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
    let mut checks = 0;
    let mut groups: Vec<_> = n.groups.iter().collect();
    groups.sort_by_key(|g| g.lod_level);
    for g in groups {
        let tris = group_tris(&n, g);
        let own = tris
            .iter()
            .map(|&t| tri_err_unit_sphere(t))
            .fold(0.0f32, f32::max);
        let chord = tris
            .iter()
            .map(|&t| tri_chord_bound(t))
            .fold(0.0f32, f32::max);
        assert!(
            own <= chord * (1.0 + 1e-5) + 1e-6,
            "own {own} above the chord bound {chord}"
        );
        let first = n.get_cluster(g.cluster_ids[0]).unwrap();
        let kids = first
            .child_ids
            .iter()
            .map(|id| want[&by_cluster[id].id])
            .fold(0.0f64, f64::max);
        let e = f64::from(own).max(kids);
        want.insert(g.id, e);
        let got = f64::from(g.max_error);
        assert!(
            (got - e).abs() <= 2e-3 * e + 1e-6,
            "group {} at LOD {}: error {got} vs exact {e}",
            g.id,
            g.lod_level
        );
        // the old size heuristic was about 100 times the real error
        assert!(got <= 1.01 * e + 1e-6);
        checks += 1;
    }
    for c in &n.clusters {
        assert_eq!(c.geometric_error, by_cluster[&c.id].max_error);
    }
    // the level summaries hold the largest group error of the level
    for l in &n.lod_levels {
        let m = n
            .groups
            .iter()
            .filter(|g| g.lod_level == l.level)
            .map(|g| g.max_error)
            .fold(0.0f32, f32::max);
        assert_eq!(l.max_error, m);
    }
    assert!(checks > 0);
}

/// Clusters respect the vertex cap even when the triangle budget alone would
/// let them exceed it
#[test]
fn nanite_clusters_respect_the_vertex_cap() {
    let mut checks = 0;
    let mut largest = 0;
    for sdf in vertex_heavy_shapes() {
        let n = nanite(
            &sdf,
            &NaniteConfig {
                lod_levels: 2,
                base_resolution: 48,
                max_triangles_per_cluster: 4096,
                use_tight_aabb: false,
                ..NaniteConfig::default()
            },
        );
        // some region holds more vertices than one cluster may
        let region_max = n
            .groups
            .iter()
            .map(|g| {
                g.cluster_ids
                    .iter()
                    .map(|&id| n.get_cluster(id).unwrap().vertex_count())
                    .sum::<usize>()
            })
            .max()
            .unwrap();
        assert!(region_max > CLUSTER_MAX_VERTICES, "region {region_max}");
        for c in &n.clusters {
            assert!(
                c.vertex_count() <= CLUSTER_MAX_VERTICES,
                "{} vertices",
                c.vertex_count()
            );
            largest = largest.max(c.vertex_count());
            checks += 1;
        }
    }
    // the cap was binding: some cluster is filled close to it
    assert!(
        largest > CLUSTER_MAX_VERTICES - 16,
        "largest cluster {largest}"
    );
    assert!(checks > 0);
}

fn area_of(n: &NaniteMesh, ids: &[u32]) -> f64 {
    ids.iter()
        .flat_map(|&id| cluster_tris(n.get_cluster(id).unwrap()).collect::<Vec<_>>())
        .map(|[a, b, c]| f64::from((b - a).cross(c - a).length()) * 0.5)
        .sum()
}

/// Groups selected from the given cluster ids (asserting whole groups)
fn selected_groups(n: &NaniteMesh, sel: &[u32]) -> std::collections::BTreeSet<u32> {
    let set: std::collections::BTreeSet<u32> = sel.iter().copied().collect();
    let mut out = std::collections::BTreeSet::new();
    for g in &n.groups {
        let k = g.cluster_ids.iter().filter(|id| set.contains(id)).count();
        assert!(
            k == 0 || k == g.cluster_ids.len(),
            "group {} partly selected",
            g.id
        );
        if k > 0 {
            out.insert(g.id);
        }
    }
    out
}

fn parent_group(n: &NaniteMesh, g: &alice_sdf::mesh::nanite::ClusterGroup) -> Option<u32> {
    let pid = *n
        .get_cluster(g.cluster_ids[0])
        .unwrap()
        .parent_ids
        .first()?;
    n.groups
        .iter()
        .find(|h| h.cluster_ids.contains(&pid))
        .map(|h| h.id)
}

/// Projected error of a group in f64: error over the distance to its sphere
fn projected(g: &alice_sdf::mesh::nanite::ClusterGroup, eye: Vec3) -> f64 {
    let e = f64::from(g.max_error);
    if e <= 0.0 {
        return 0.0;
    }
    let d = f64::from((g.bounds.center - eye).length()) - f64::from(g.bounds.radius);
    if d <= 0.0 {
        f64::INFINITY
    } else {
        e / d
    }
}

/// The cut covers the surface exactly once at every distance and threshold
///
/// - every path from a leaf group to the root holds exactly one selected group
/// - a group is selected exactly when it is fine enough (projected error within
///   the threshold, or no children) and its parent is not
/// - the selected area lies between the areas of the spheres of radius 1 - e
///   and 1 + e (e = largest selected error), and within 20 % of 4π
/// - parent errors are never below the errors of their children
#[test]
fn nanite_cut_covers_the_surface_at_every_distance() {
    let n = nanite(&SdfNode::sphere(1.0), &NaniteConfig::medium_detail());
    for p in &n.clusters {
        for &c in &p.child_ids {
            assert!(p.geometric_error >= n.get_cluster(c).unwrap().geometric_error);
        }
    }
    let leaf = |g: &alice_sdf::mesh::nanite::ClusterGroup| {
        n.get_cluster(g.cluster_ids[0])
            .unwrap()
            .child_ids
            .is_empty()
    };
    let by_id: std::collections::HashMap<u32, &alice_sdf::mesh::nanite::ClusterGroup> =
        n.groups.iter().map(|g| (g.id, g)).collect();
    let mut checks = 0;
    let mut levels_used = std::collections::BTreeSet::new();
    for d in [1.5f32, 3.0, 10.0, 30.0, 100.0, 1000.0] {
        for th in [1e-4f32, 1e-3, 0.01, 0.1] {
            let eye = Vec3::new(0.0, 0.0, d);
            let sel = n.select_clusters(eye, th);
            assert!(!sel.is_empty(), "d={d} th={th}: nothing selected");
            let groups = selected_groups(&n, &sel);
            // exactly one selected group on every leaf-to-root path
            for g in n.groups.iter().filter(|g| leaf(g)) {
                let mut count = 0;
                let mut cur = Some(g.id);
                while let Some(id) = cur {
                    count += usize::from(groups.contains(&id));
                    cur = parent_group(&n, by_id[&id]);
                }
                assert_eq!(
                    count, 1,
                    "d={d} th={th}: path from group {} selects {count}",
                    g.id
                );
                checks += 1;
            }
            // the selection rule, evaluated independently in f64
            for g in &n.groups {
                let fine = |h: &alice_sdf::mesh::nanite::ClusterGroup| {
                    leaf(h) || projected(h, eye) <= f64::from(th)
                };
                let parent_fine = parent_group(&n, g).is_some_and(|p| fine(by_id[&p]));
                assert_eq!(
                    groups.contains(&g.id),
                    fine(g) && !parent_fine,
                    "group {}",
                    g.id
                );
                if groups.contains(&g.id) {
                    levels_used.insert(g.lod_level);
                }
            }
            // area between the nested spheres of the selected error
            let e = groups
                .iter()
                .map(|id| f64::from(by_id[id].max_error))
                .fold(0.0f64, f64::max);
            let area = area_of(&n, &sel);
            let sphere = 4.0 * std::f64::consts::PI;
            let (lo, hi) = (sphere * (1.0 - e).powi(2), sphere * (1.0 + e).powi(2));
            assert!(
                area >= lo * 0.98 && area <= hi * 1.02,
                "d={d} th={th}: area {area} outside [{lo}, {hi}] (e={e})"
            );
            if e <= 0.1 {
                assert!(
                    (area - sphere).abs() < 0.2 * sphere,
                    "d={d} th={th}: selected area {area} vs {sphere}"
                );
            }
            checks += 1;
        }
    }
    // the sweep reaches both the finest and the coarsest level
    assert!(levels_used.contains(&0));
    assert!(levels_used.contains(&n.lod_levels.last().unwrap().level));
    assert!(checks > 0);
}

/// Lowering the threshold refines the cut and never selects fewer triangles;
/// moving away coarsens it and never selects more
#[test]
fn nanite_cut_refines_with_the_threshold_and_coarsens_with_distance() {
    let n = nanite(&SdfNode::sphere(1.0), &NaniteConfig::medium_detail());
    let tris = |sel: &[u32]| -> usize {
        sel.iter()
            .map(|&id| n.get_cluster(id).unwrap().triangle_count())
            .sum()
    };
    let by_id: std::collections::HashMap<u32, &alice_sdf::mesh::nanite::ClusterGroup> =
        n.groups.iter().map(|g| (g.id, g)).collect();
    // ancestors of a group, itself included
    let ancestors = |id: u32| {
        let mut out = vec![id];
        let mut cur = parent_group(&n, by_id[&id]);
        while let Some(p) = cur {
            out.push(p);
            cur = parent_group(&n, by_id[&p]);
        }
        out
    };
    let mut checks = 0;
    let mut grew = 0;
    for d in [2.0f32, 6.0, 40.0] {
        let eye = Vec3::new(0.3, -0.2, d);
        let mut prev: Option<(usize, std::collections::BTreeSet<u32>)> = None;
        for th in [
            1.0f32, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.0,
        ] {
            let sel = n.select_clusters(eye, th);
            let groups = selected_groups(&n, &sel);
            let t = tris(&sel);
            if let Some((pt, pg)) = &prev {
                assert!(t >= *pt, "d={d} th={th}: {t} < {pt} triangles");
                grew += usize::from(t > *pt);
                // refinement: every newly selected group descends from a group
                // selected at the larger threshold (or is that group)
                for &g in &groups {
                    assert!(
                        ancestors(g).iter().any(|a| pg.contains(a)),
                        "group {g} not under the coarser cut"
                    );
                }
            }
            prev = Some((t, groups));
            checks += 1;
        }
        // threshold 0 keeps only the finest level
        assert!(sel_levels(&n, &n.select_clusters(eye, 0.0))
            .iter()
            .all(|&l| l == 0));
    }
    assert!(grew > 0);
    for th in [0.001f32, 0.01, 0.05] {
        let mut prev = usize::MAX;
        for d in [1.5f32, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0, 4096.0] {
            let t = tris(&n.select_clusters(Vec3::new(0.0, 0.0, d), th));
            assert!(t <= prev, "th={th} d={d}: {t} > {prev} triangles");
            prev = t;
            checks += 1;
        }
    }
    assert!(checks > 0);
}

fn sel_levels(n: &NaniteMesh, sel: &[u32]) -> Vec<u32> {
    sel.iter()
        .map(|&id| n.get_cluster(id).unwrap().lod_level)
        .collect()
}

/// `should_render` decides the whole cut from the fields of one cluster: each
/// cluster carries its group's error and LOD sphere and its parent group's
/// error and LOD sphere (`+inf` / its own sphere for a root group), and the
/// clusters it accepts are exactly the clusters of `select_clusters` and of the
/// selection rule evaluated independently over the groups in `f64`
#[test]
fn nanite_should_render_is_the_complete_cut() {
    let mut checks = 0;
    let (mut yes, mut no) = (0, 0);
    for cfg in [NaniteConfig::preview(), NaniteConfig::medium_detail()] {
        let n = nanite(&SdfNode::sphere(1.0), &cfg);
        let by_id: std::collections::HashMap<u32, &alice_sdf::mesh::nanite::ClusterGroup> =
            n.groups.iter().map(|g| (g.id, g)).collect();
        let group_of = |id: u32| {
            n.groups
                .iter()
                .find(|g| g.cluster_ids.contains(&id))
                .unwrap()
        };
        let same_sphere = |a: &ClusterBounds, b: &ClusterBounds| {
            a.center.to_array() == b.center.to_array() && a.radius.to_bits() == b.radius.to_bits()
        };
        // the per-cluster copies are the group's and the parent group's values
        let mut roots = 0;
        for c in &n.clusters {
            let g = group_of(c.id);
            assert_eq!(c.geometric_error.to_bits(), g.max_error.to_bits());
            assert!(same_sphere(&c.lod_bounds, &g.bounds), "cluster {}", c.id);
            match parent_group(&n, g) {
                Some(p) => {
                    let p = by_id[&p];
                    assert_eq!(c.parent_error.to_bits(), p.max_error.to_bits());
                    assert!(same_sphere(&c.parent_lod_bounds, &p.bounds));
                }
                None => {
                    assert_eq!(c.parent_error, f32::INFINITY);
                    assert!(same_sphere(&c.parent_lod_bounds, &c.lod_bounds));
                    roots += 1;
                }
            }
            checks += 1;
        }
        assert!(roots > 0);
        let leaf = |g: &alice_sdf::mesh::nanite::ClusterGroup| {
            n.get_cluster(g.cluster_ids[0])
                .unwrap()
                .child_ids
                .is_empty()
        };
        for eye in [
            Vec3::new(0.0, 1.2, 0.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(0.4, -0.3, 10.0),
            Vec3::new(30.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1000.0),
            Vec3::ZERO,
        ] {
            for th in [0.0f32, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, f32::INFINITY] {
                let by_cluster: Vec<u32> = n
                    .clusters
                    .iter()
                    .filter(|c| c.should_render(eye, th))
                    .map(|c| c.id)
                    .collect();
                assert_eq!(by_cluster, n.select_clusters(eye, th), "eye={eye} th={th}");
                // the rule over the groups in f64
                let fine = |h: &alice_sdf::mesh::nanite::ClusterGroup| {
                    leaf(h) || projected(h, eye) <= f64::from(th)
                };
                let mut want = Vec::new();
                for g in &n.groups {
                    let parent_fine = parent_group(&n, g).is_some_and(|p| fine(by_id[&p]));
                    if fine(g) && !parent_fine {
                        want.extend_from_slice(&g.cluster_ids);
                    }
                }
                want.sort_unstable();
                assert_eq!(by_cluster, want, "eye={eye} th={th}");
                assert!(!by_cluster.is_empty(), "eye={eye} th={th}");
                for c in &n.clusters {
                    if c.should_render(eye, th) {
                        yes += 1;
                    } else {
                        no += 1;
                    }
                }
                checks += 1;
            }
        }
    }
    assert!(yes > 0 && no > 0);
    assert!(checks > 0);
}

/// Version 3 of `.nanite` adds the LOD spheres and the parent error per cluster
#[test]
fn nanite_format_version_marks_the_lod_spheres() {
    assert_eq!(alice_sdf::io::nanite::NANITE_VERSION, 3);
}

/// Shapes with several surface sheets per region, so that a region holds far
/// more than [`CLUSTER_MAX_VERTICES`] vertices
fn vertex_heavy_shapes() -> [SdfNode; 2] {
    [
        SdfNode::sphere(1.2).onion(0.12).onion(0.05).onion(0.02),
        SdfNode::gyroid(8.0, 0.05).intersection(SdfNode::sphere(1.6)),
    ]
}

// ---------------------------------------------------------------------------
// LOD chains
// ---------------------------------------------------------------------------

fn closest_on_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    // Ericson, Real-Time Collision Detection 5.1.5
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

/// Two-sided Hausdorff distance between a mesh and the unit sphere
fn hausdorff_to_unit_sphere(m: &Mesh) -> f32 {
    let mut h = 0.0f32;
    // mesh -> sphere, exact: |p| is convex, so its max on a triangle is at a
    // vertex and its min is the distance from the origin to the triangle
    for [a, b, c] in mesh_tris(m) {
        let far = a.length().max(b.length()).max(c.length());
        let near = closest_on_triangle(Vec3::ZERO, a, b, c).length();
        h = h.max((far - 1.0).abs()).max((1.0 - near).abs());
    }
    // sphere -> mesh, sampled on a Fibonacci sphere
    let n = 400;
    let golden = std::f32::consts::PI * (3.0 - 5f32.sqrt());
    for i in 0..n {
        let y = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
        let r = (1.0 - y * y).sqrt();
        let th = golden * i as f32;
        let p = Vec3::new(r * th.cos(), y, r * th.sin());
        let d = mesh_tris(m)
            .map(|[a, b, c]| (closest_on_triangle(p, a, b, c) - p).length())
            .fold(f32::INFINITY, f32::min);
        h = h.max(d);
    }
    h
}

fn check_chain_geometry(chain: &LodChain, strict: bool) -> usize {
    let mut checks = 0;
    for w in chain.levels.windows(2) {
        let (a, b) = (w[0].mesh.triangle_count(), w[1].mesh.triangle_count());
        assert!(if strict { b < a } else { b <= a }, "{b} !< {a}");
        assert_eq!(
            w[0].max_distance, w[1].min_distance,
            "ranges not contiguous"
        );
        assert!(w[1].max_error >= w[0].max_error);
    }
    assert_eq!(chain.levels[0].min_distance, 0.0);
    assert_eq!(chain.levels.last().unwrap().max_distance, f32::INFINITY);
    for l in &chain.levels {
        if l.mesh.triangle_count() == 0 {
            continue;
        }
        let h = hausdorff_to_unit_sphere(&l.mesh);
        assert!(
            h <= l.max_error,
            "level {}: Hausdorff {h} exceeds the error bound {}",
            l.level,
            l.max_error
        );
        checks += 1;
    }
    let mem: usize = chain
        .levels
        .iter()
        .map(|l| {
            l.mesh.vertices.len() * std::mem::size_of::<Vertex>()
                + l.mesh.indices.len() * std::mem::size_of::<u32>()
        })
        .sum();
    assert_eq!(chain.memory_usage(), mem);
    assert_eq!(
        chain.base_triangle_count(),
        chain.levels[0].mesh.triangle_count()
    );
    checks
}

#[test]
fn lod_chain_levels_shrink_and_stay_within_their_error() {
    let s = SdfNode::sphere(1.0);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let mut checks = 0;
    for cfg in [
        LodConfig::default(),
        LodConfig::balanced(),
        LodConfig::fast(),
    ] {
        let chain = generate_lod_chain(&s, lo, hi, &cfg);
        assert_eq!(chain.levels.len(), cfg.num_levels as usize);
        for (i, l) in chain.levels.iter().enumerate() {
            let want = ((cfg.base_resolution as f64 * (cfg.reduction_factor as f64).powi(i as i32))
                as u32)
                .max(4);
            assert_eq!(l.resolution, want);
            assert_eq!(cfg.resolution_at_level(i as u32), want);
            let (mn, mx) = cfg.distance_range(i as u32);
            assert_eq!((l.min_distance, l.max_distance), (mn, mx));
        }
        checks += check_chain_geometry(&chain, false);
    }
    let hq = LodConfig::high_quality();
    assert!(hq.base_resolution > LodConfig::balanced().base_resolution);
    let dcfg = DecimationLodConfig {
        num_levels: 4,
        base_resolution: 32,
        ..DecimationLodConfig::default()
    };
    let dchain = generate_lod_chain_decimated(&s, lo, hi, &dcfg);
    for (i, l) in dchain.levels.iter().enumerate() {
        let (mn, mx) = dcfg.distance_range(i as u32);
        assert_eq!((l.min_distance, l.max_distance), (mn, mx));
    }
    checks += check_chain_geometry(&dchain, true);
    let dhq = DecimationLodConfig::high_quality();
    let dfast = DecimationLodConfig::fast();
    assert!(dhq.base_resolution > dfast.base_resolution && dhq.num_levels > dfast.num_levels);
    assert!(checks > 0);
}

#[test]
fn lod_chain_queries_follow_the_distance_ranges() {
    let cfg = LodConfig::fast();
    let chain = generate_lod_chain(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &cfg,
    );
    let mut checks = 0;
    for i in 0..400 {
        let d = i as f32 * 0.37;
        // the level whose [min, max) holds d
        let want = chain
            .levels
            .iter()
            .position(|l| d >= l.min_distance && d < l.max_distance)
            .unwrap();
        let lod = chain.get_lod(d).unwrap();
        assert_eq!(lod.level as usize, want);
        assert!(lod.is_active(d));
        assert_eq!(chain.get_level(want as u32).unwrap().level as usize, want);
        match chain.get_blend_pair(d) {
            Some((a, b, t)) => {
                assert_eq!(a.level as usize, want);
                assert_eq!(b.level as usize, want + 1);
                let lm = &chain.levels[want];
                let tw =
                    ((d - lm.min_distance) / (lm.max_distance - lm.min_distance)).clamp(0.0, 1.0);
                assert_eq!(t, tw);
                assert_eq!(a.blend_factor(d), tw);
            }
            None => assert_eq!(want, chain.levels.len() - 1),
        }
        // select_by_error: the coarsest level whose error over distance meets
        // the threshold (only cases where at least one level meets it)
        let thr = 0.02;
        if let Some(w) = chain
            .levels
            .iter()
            .rposition(|l| l.max_error / d.max(0.001) <= thr)
        {
            assert_eq!(chain.select_by_error(d, thr).unwrap().level as usize, w);
        }
        checks += 1;
    }
    assert!(chain.get_level(99).is_none());
    // LodSelector: projected error = e / d * H / (2 tan(fov / 2))
    for sel in [LodSelector::default(), LodSelector::high_res()] {
        let proj = sel.screen_height as f64 / (2.0 * (sel.fov_y as f64 * 0.5).tan());
        for &(e, d) in &[(0.1f32, 10.0f32), (0.01, 1.0), (0.5, 250.0)] {
            let want = e as f64 / d as f64 * proj;
            assert!(((sel.screen_error(e, d) as f64) - want).abs() <= want * 1e-5);
            checks += 1;
        }
        for i in 1..200 {
            let d = i as f32 * 0.5;
            let want = chain
                .levels
                .iter()
                .rposition(|l| (l.max_error as f64 / d as f64) * proj <= sel.pixel_threshold as f64)
                .unwrap_or(0);
            assert_eq!(sel.select(&chain, d).unwrap().level as usize, want);
            for l in &chain.levels {
                let ok = (l.max_error as f64 / d as f64) * proj <= sel.pixel_threshold as f64;
                assert_eq!(sel.is_acceptable(l, d), ok);
            }
            checks += 1;
        }
    }
    assert_eq!(LodSelector::high_res().screen_height, 2160.0);
    assert!(checks > 0);
}

/// `max_error` is the measured distance of the level to the surface: for the
/// unit sphere it equals the exact one-sided Hausdorff distance of the mesh
/// (closed form per triangle), raised to the error of the finer levels; the
/// old estimate (half a cell of the bounding-box diagonal) did not look at the
/// mesh.
/// When no level meets the threshold, both selectors fall back to the finest
/// level.
#[test]
fn lod_chain_error_is_the_measured_distance_and_falls_back_to_the_finest() {
    let s = SdfNode::sphere(1.0);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let mut checks = 0;
    let chains = [
        generate_lod_chain(&s, lo, hi, &LodConfig::balanced()),
        generate_lod_chain(&s, lo, hi, &LodConfig::fast()),
        generate_lod_chain_decimated(
            &s,
            lo,
            hi,
            &DecimationLodConfig {
                num_levels: 4,
                base_resolution: 32,
                ..DecimationLodConfig::default()
            },
        ),
    ];
    for chain in &chains {
        let mut floor = 0.0f64;
        for l in &chain.levels {
            let own = mesh_tris(&l.mesh)
                .map(tri_err_unit_sphere)
                .fold(0.0f32, f32::max);
            floor = floor.max(f64::from(own));
            let got = f64::from(l.max_error);
            assert!(
                (got - floor).abs() <= 2e-3 * floor + 1e-6,
                "level {}: max_error {got} vs measured {floor}",
                l.level
            );
            checks += 1;
        }
        for d in [0.5f32, 5.0, 50.0] {
            for thr in [0.0f32, -1.0] {
                assert_eq!(chain.select_by_error(d, thr).unwrap().level, 0);
                checks += 1;
            }
        }
        let strict = LodSelector {
            pixel_threshold: 0.0,
            ..LodSelector::default()
        };
        assert_eq!(strict.select(chain, 5.0).unwrap().level, 0);
    }
    assert!(checks > 0);
}

/// ContinuousLod moves towards the target level by at most speed·dt per
/// update and stops on it
#[test]
fn continuous_lod_reaches_the_target_without_overshoot() {
    let cfg = LodConfig::fast();
    let mk = || {
        generate_lod_chain(
            &SdfNode::sphere(1.0),
            Vec3::splat(-1.5),
            Vec3::splat(1.5),
            &cfg,
        )
    };
    let chain = mk();
    let mut checks = 0;
    for d in [0.5f32, 4.0, 7.0, 12.5, 30.0] {
        // target = index of the active level + its blend factor
        let i = chain.levels.iter().position(|l| l.is_active(d)).unwrap();
        let target = i as f32 + chain.levels[i].blend_factor(d);
        let mut clod = ContinuousLod::new(mk(), 2.0);
        let mut prev = 0.0f32;
        for _ in 0..100 {
            clod.update(d, 0.1);
            let (base, blend) = clod.get_render_meshes();
            let cur = match blend {
                Some((_, t)) => {
                    let lvl = chain
                        .levels
                        .iter()
                        .position(|l| l.mesh.triangle_count() == base.triangle_count())
                        .unwrap();
                    lvl as f32 + t
                }
                None => chain
                    .levels
                    .iter()
                    .position(|l| l.mesh.triangle_count() == base.triangle_count())
                    .unwrap() as f32,
            };
            assert!(
                (cur - prev).abs() <= 0.2 + 1e-5,
                "step larger than speed·dt"
            );
            assert!(
                cur <= target + 0.0101,
                "stepped past the target {target}: {cur}"
            );
            prev = cur;
        }
        assert!(
            (prev - target).abs() <= 0.0101,
            "d={d}: settled at {prev}, target {target}"
        );
        checks += 1;
    }
    assert!(checks > 0);
}

// ---------------------------------------------------------------------------
// LOD persistence summary
// ---------------------------------------------------------------------------

#[test]
fn lod_persist_queries_follow_the_levels() {
    let chain = generate_lod_chain(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &LodConfig::default(),
    );
    let meshes: Vec<Mesh> = chain.levels.iter().map(|l| l.mesh.clone()).collect();
    let dists: Vec<f32> = chain.levels.iter().map(|l| l.min_distance).collect();
    let p = LodChainPersist::new(meshes.clone(), dists.clone(), 7, LodChainConfig::default());
    assert_eq!(p.level_count(), meshes.len());
    let mut checks = 0;
    for (i, m) in meshes.iter().enumerate() {
        let got = p.mesh(i).unwrap();
        assert_eq!(got.indices, m.indices);
        assert_eq!(got.vertices.len(), m.vertices.len());
        checks += 1;
    }
    assert!(p.mesh(meshes.len()).is_none());
    for i in 0..300 {
        let d = i as f32 * 0.21 - 1.0;
        let want = dists.iter().rposition(|&t| d >= t).unwrap_or(0);
        assert_eq!(p.select_lod(d), want);
        checks += 1;
    }
    let mem: usize = meshes
        .iter()
        .map(|m| m.vertices.len() * std::mem::size_of::<Vertex>() + m.indices.len() * 4)
        .sum();
    assert_eq!(p.total_memory_bytes(), mem);
    let s = p.summary();
    assert_eq!(s.level_count, meshes.len());
    assert_eq!(
        s.total_vertices,
        meshes.iter().map(|m| m.vertices.len()).sum::<usize>()
    );
    assert_eq!(
        s.total_triangles,
        meshes.iter().map(|m| m.indices.len() / 3).sum::<usize>()
    );
    assert_eq!(s.total_memory_bytes, mem);
    assert_eq!(s.lod0_vertices, meshes[0].vertices.len());
    assert_eq!(s.lod0_triangles, meshes[0].indices.len() / 3);
    let empty = LodChainPersist::new(vec![], vec![], 0, LodChainConfig::default()).summary();
    assert_eq!(
        (
            empty.level_count,
            empty.lod0_triangles,
            empty.total_memory_bytes
        ),
        (0, 0, 0)
    );
    assert!(checks > 0);
}

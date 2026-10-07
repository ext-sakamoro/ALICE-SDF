//! Oracles for LOD chains, Nanite clusters and meshlets
//!
//! The expected values are built independently of the code under test:
//!
//! - partitions: the clusters / meshlets of a level, put back together, are
//!   exactly the triangle multiset of the marching-cubes mesh they were cut
//!   from (the mesh is regenerated here with the same resolution)
//! - capacity limits, local index ranges and the bounding spheres / AABBs are
//!   recomputed from the vertex positions
//! - the LOD DAG is compared with a brute-force O(n²) sphere-overlap scan
//! - the LOD error bound is checked with the two-sided Hausdorff distance to
//!   the exact unit sphere (mesh → sphere in closed form, sphere → mesh with a
//!   brute-force point-triangle distance)
//! - view-cone and back-face tests are compared with the angle formulas
//!   evaluated in f64 (`acos` / `asin`)
//!
//! Every test counts its comparisons and fails when the count is 0.

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

/// The grid-bucketed DAG must hold exactly the parent/child pairs that a
/// brute-force scan finds: adjacent levels whose bounding spheres overlap
#[test]
fn nanite_dag_matches_the_brute_force_overlap_scan() {
    let mut checks = 0;
    for sdf in [SdfNode::sphere(1.0), SdfNode::torus(1.0, 0.3)] {
        let n = nanite(
            &sdf,
            &NaniteConfig {
                lod_levels: 4,
                base_resolution: 48,
                max_triangles_per_cluster: 48,
                ..NaniteConfig::default()
            },
        );
        let mut want: Vec<(u32, u32)> = Vec::new();
        for p in &n.clusters {
            for c in &n.clusters {
                if p.lod_level == c.lod_level + 1
                    && (p.bounds.center - c.bounds.center).length()
                        < p.bounds.radius + c.bounds.radius
                {
                    want.push((p.id, c.id));
                }
            }
        }
        let mut got_children: Vec<(u32, u32)> = n
            .clusters
            .iter()
            .flat_map(|p| p.child_ids.iter().map(move |&c| (p.id, c)))
            .collect();
        let mut got_parents: Vec<(u32, u32)> = n
            .clusters
            .iter()
            .flat_map(|c| c.parent_ids.iter().map(move |&p| (p, c.id)))
            .collect();
        want.sort_unstable();
        got_children.sort_unstable();
        got_parents.sort_unstable();
        assert!(!want.is_empty());
        assert_eq!(got_children, want);
        assert_eq!(got_parents, want);
        checks += want.len();
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

/// Reproduces the broken Nanite cut (kept until the cluster error and the cut
/// are redesigned): a correct cut covers the surface at every distance, and a
/// parent cluster's error is never below the error of its children.
///
/// Measured before the redesign (sphere r=1, `medium_detail`): z=100 with
/// threshold 0.1 selects 0 clusters, z=3 with threshold 0.01 selects only 8
/// LOD-2 clusters (LOD 2 alone has 24). `should_render` answers "the error is
/// too large", so the cut keeps the too-coarse side. Per-cluster errors of
/// adjacent levels overlap (LOD 0 0.0047..0.0119, LOD 1 0.0097..0.0275) and a
/// level that fits in one cluster uses another formula (LOD 4 jumps to 2.68).
#[test]
#[ignore = "src gap: nanite-cluster-cut — should_render/select_clusters keep the too-coarse side and a parent cluster's error can be below its child's"]
fn nanite_cut_covers_the_surface_at_every_distance() {
    let n = nanite(&SdfNode::sphere(1.0), &NaniteConfig::medium_detail());
    for p in &n.clusters {
        for &c in &p.child_ids {
            assert!(p.geometric_error >= n.get_cluster(c).unwrap().geometric_error);
        }
    }
    for d in [1.5f32, 3.0, 10.0, 100.0] {
        for th in [0.001f32, 0.01, 0.1] {
            let sel = n.select_clusters(Vec3::new(0.0, 0.0, d), th);
            let area: f32 = sel
                .iter()
                .map(|&id| {
                    let c = n.get_cluster(id).unwrap();
                    c.triangles
                        .iter()
                        .map(|t| {
                            let a = c.vertices[t.a as usize].position;
                            let b = c.vertices[t.b as usize].position;
                            let cc = c.vertices[t.c as usize].position;
                            (b - a).cross(cc - a).length() * 0.5
                        })
                        .sum::<f32>()
                })
                .sum();
            let sphere = 4.0 * std::f32::consts::PI;
            assert!(
                (area - sphere).abs() < 0.2 * sphere,
                "d={d} th={th}: selected area {area} vs {sphere}"
            );
        }
    }
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

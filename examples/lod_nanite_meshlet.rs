//! LOD chains, Nanite-style clusters and GPU meshlets from one SDF
//!
//! Builds a resolution LOD chain and a decimation LOD chain, picks levels by
//! distance and by screen-space error, blends between them, cuts the mesh into
//! Nanite clusters and mesh-shader meshlets, and culls with the bounding
//! sphere and the normal cone. Every step prints its numbers and checks them.
//!
//! Run: `cargo run --release --example lod_nanite_meshlet`

use alice_sdf::mesh::lod::{
    generate_lod_chain, generate_lod_chain_decimated, ContinuousLod, DecimationLodConfig,
    LodConfig, LodSelector,
};
use alice_sdf::mesh::lod_persist::{LodChainConfig, LodChainPersist, LodChainSummary};
use alice_sdf::mesh::meshlet::{
    build_meshlets, build_meshlets_adjacency, build_meshlets_scan, MeshletConfig,
};
use alice_sdf::mesh::nanite::{
    generate_nanite_mesh, ClusterBounds, ClusterGroup, LodLevel, NaniteConfig, NormalCone,
    CLUSTER_MAX_TRIANGLES, CLUSTER_MAX_VERTICES,
};
use alice_sdf::mesh::Mesh;
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn main() {
    let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(2.6, 0.5, 0.5), 0.15);
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));

    // --- LOD chains --------------------------------------------------------
    let fast = LodConfig::fast();
    let balanced = LodConfig::balanced();
    let hq = LodConfig::high_quality();
    println!(
        "LodConfig base resolution: fast {} / balanced {} / high_quality {}",
        fast.base_resolution, balanced.base_resolution, hq.base_resolution
    );
    assert!(fast.base_resolution < balanced.base_resolution);
    assert!(balanced.base_resolution < hq.base_resolution);
    for level in 0..balanced.num_levels {
        let (near, far) = balanced.distance_range(level);
        println!(
            "  balanced level {level}: resolution {} for distance [{near}, {far})",
            balanced.resolution_at_level(level)
        );
    }

    let chain = generate_lod_chain(&shape, lo, hi, &balanced);
    assert_eq!(chain.levels.len(), balanced.num_levels as usize);
    for l in &chain.levels {
        println!(
            "  level {}: {} triangles, error bound {:.4}",
            l.level,
            l.mesh.triangle_count(),
            l.max_error
        );
    }
    println!(
        "  base triangles {}, memory {} bytes",
        chain.base_triangle_count(),
        chain.memory_usage()
    );

    for d in [0.5f32, 3.0, 9.0, 40.0] {
        let lod = chain.get_lod(d).expect("every distance has a level");
        assert!(lod.is_active(d));
        let by_error = chain.select_by_error(d, 0.01).expect("non-empty chain");
        match chain.get_blend_pair(d) {
            Some((a, b, t)) => {
                assert!((0.0..=1.0).contains(&t));
                assert_eq!(t, a.blend_factor(d));
                println!(
                    "  d={d}: level {} (blend {t:.3} towards {}), by error {}",
                    lod.level, b.level, by_error.level
                );
            }
            None => println!(
                "  d={d}: level {} (last), by error {}",
                lod.level, by_error.level
            ),
        }
    }
    assert_eq!(chain.get_level(1).map(|l| l.level), Some(1));

    let selector = LodSelector::default();
    let selector_4k = LodSelector::high_res();
    for d in [2.0f32, 20.0] {
        let a = selector.select(&chain, d).unwrap();
        let b = selector_4k.select(&chain, d).unwrap();
        println!(
            "  d={d}: 1080p picks level {} ({:.2} px), 4K picks level {} ({:.2} px)",
            a.level,
            selector.screen_error(a.max_error, d),
            b.level,
            selector_4k.screen_error(b.max_error, d)
        );
        // the 4K screen needs at least as much detail as 1080p
        assert!(b.level <= a.level);
        assert!(selector.is_acceptable(a, d) || a.level == 0);
    }

    let dec_fast = DecimationLodConfig::fast();
    let dec_hq = DecimationLodConfig::high_quality();
    assert!(dec_fast.num_levels < dec_hq.num_levels);
    let dcfg = DecimationLodConfig {
        num_levels: 4,
        base_resolution: 32,
        ..DecimationLodConfig::default()
    };
    let dchain = generate_lod_chain_decimated(&shape, lo, hi, &dcfg);
    let tris: Vec<usize> = dchain
        .levels
        .iter()
        .map(|l| l.mesh.triangle_count())
        .collect();
    println!(
        "decimated chain triangles {tris:?}, level 3 range {:?}",
        dcfg.distance_range(3)
    );
    assert!(tris.windows(2).all(|w| w[1] < w[0]));

    let mut clod = ContinuousLod::new(generate_lod_chain(&shape, lo, hi, &fast), 2.0);
    for _ in 0..30 {
        clod.update(10.0, 0.1);
    }
    let (base, blend) = clod.get_render_meshes();
    println!(
        "continuous LOD at d=10: base {} triangles, blend {:?}",
        base.triangle_count(),
        blend.map(|(_, t)| t)
    );

    // --- persistence metadata ---------------------------------------------
    let meshes: Vec<Mesh> = chain.levels.iter().map(|l| l.mesh.clone()).collect();
    let dists: Vec<f32> = chain.levels.iter().map(|l| l.min_distance).collect();
    let persist = LodChainPersist::new(meshes, dists, 0x5eed, LodChainConfig::default());
    let summary: LodChainSummary = persist.summary();
    println!(
        "persisted chain: {} levels, {} triangles total, {} bytes, level for d=9 is {}",
        persist.level_count(),
        summary.total_triangles,
        persist.total_memory_bytes(),
        persist.select_lod(9.0)
    );
    assert_eq!(summary.total_memory_bytes, persist.total_memory_bytes());
    assert_eq!(
        persist.mesh(0).map(|m| m.triangle_count()),
        Some(summary.lod0_triangles)
    );

    // --- Nanite clusters ---------------------------------------------------
    let (preview, medium, high) = (
        NaniteConfig::preview(),
        NaniteConfig::medium_detail(),
        NaniteConfig::high_detail(),
    );
    println!(
        "NaniteConfig base resolution: preview {} / medium {} / high {}",
        preview.base_resolution, medium.base_resolution, high.base_resolution
    );
    let nanite = generate_nanite_mesh(&shape, lo, hi, &preview);
    let levels: &[LodLevel] = &nanite.lod_levels;
    let groups: &[ClusterGroup] = &nanite.groups;
    println!(
        "nanite: {} clusters in {} groups, {} vertices, levels {:?}",
        nanite.clusters.len(),
        groups.len(),
        nanite.total_vertices(),
        levels.iter().map(|l| l.triangle_count).collect::<Vec<_>>()
    );
    for l in levels {
        let clusters = nanite.clusters_at_lod(l.level);
        for c in &clusters {
            assert!(c.triangle_count() <= CLUSTER_MAX_TRIANGLES);
            assert!(c.vertex_count() <= CLUSTER_MAX_VERTICES);
            assert_eq!(nanite.get_cluster(c.id).map(|x| x.id), Some(c.id));
        }
        let flat = nanite.to_mesh(l.level);
        assert_eq!(flat.triangle_count() as u32, l.triangle_count);
    }

    // frustum test: a 30° half-angle cone looking down -Z from z=5
    let eye = Vec3::new(0.0, 0.0, 5.0);
    let fov_cos = 30f32.to_radians().cos();
    let visible = nanite
        .clusters_at_lod(0)
        .iter()
        .filter(|c| c.bounds.is_visible(eye, Vec3::NEG_Z, fov_cos))
        .count();
    let behind = ClusterBounds::from_vertices(&[Vec3::new(0.0, 0.0, 9.0)]);
    assert!(!behind.is_visible(eye, Vec3::NEG_Z, fov_cos));
    println!(
        "  {visible} LOD-0 clusters in view, screen error of the whole mesh {:.3} px",
        nanite.bounds.screen_error(eye, levels[0].max_error, 1080.0)
    );

    // --- meshlets and back-face culling ------------------------------------
    let mesh = chain.levels[1].mesh.clone();
    let v1 = build_meshlets(&mesh, &MeshletConfig::default());
    let v1_direct = build_meshlets_scan(&mesh, &MeshletConfig::default());
    let quality = MeshletConfig::quality();
    let v2 = build_meshlets(&mesh, &quality);
    let v2_direct = build_meshlets_adjacency(&mesh, &quality);
    assert_eq!(v1.len(), v1_direct.len());
    assert_eq!(v2.len(), v2_direct.len());
    let total: usize = v1.iter().map(|m| m.triangle_count()).sum();
    assert_eq!(total, mesh.triangle_count());
    assert!(v1.iter().all(|m| m.vertex_count() <= 64));

    let view_dir = Vec3::NEG_Z; // looking down -Z: the +Z side faces the camera
    let culled = |ms: &[alice_sdf::mesh::meshlet::Meshlet]| {
        ms.iter()
            .filter(|m| m.normal_cone.is_backface_culled(view_dir))
            .count()
    };
    println!(
        "meshlets: scan {} ({} back-facing), adjacency {} ({} back-facing)",
        v1.len(),
        culled(&v1),
        v2.len(),
        culled(&v2)
    );

    let normals = [Vec3::Z, Vec3::new(0.0, 0.2, 1.0).normalize()];
    let centers = [Vec3::ZERO, Vec3::Y];
    let cone = NormalCone::from_normals(&normals);
    let with_apex = NormalCone::from_normals_and_positions(&normals, &centers);
    println!(
        "normal cone: cutoff {:.4}, apex {:?}",
        cone.cutoff_cos, with_apex.apex
    );
    assert!(cone.is_backface_culled(Vec3::Z));
    assert!(!cone.is_backface_culled(Vec3::NEG_Z));
    assert!(!NormalCone::unbounded().is_backface_culled(Vec3::Z));
}

//! # ALICE-SDF
//!
//! **A.L.I.C.E. - Adaptive Lightweight Implicit Compression Engine**
//!
//! A 3D/spatial data specialist that transmits mathematical descriptions
//! of shapes (Signed Distance Functions) instead of polygon meshes.
//!
//! ## Features
//!
//! - **Primitives**: Sphere, Box, Cylinder, Torus, Plane, Capsule
//! - **Operations**: Union, Intersection, Subtraction (smooth variants)
//! - **Transforms**: Translate, Rotate, Scale
//! - **Modifiers**: Twist, Bend, Repeat, Noise
//! - **Conversion**: Mesh ↔ SDF
//! - **Raymarching**: Real-time rendering
//! - **File I/O**: Binary (.asdf) and JSON (.asdf.json) formats
//!
//! ## Example
//!
//! ```rust
//! use alice_sdf::prelude::*;
//!
//! // Create a sphere with radius 1
//! let sphere = SdfNode::sphere(1.0);
//!
//! // Subtract a box from it
//! let result = sphere.subtract(SdfNode::box3d(1.5, 1.5, 1.5));
//!
//! // Evaluate distance at a point
//! let distance = eval(&result, glam::Vec3::ZERO);
//!
//! // Convert to mesh
//! let mesh = sdf_to_mesh(
//!     &result,
//!     glam::Vec3::splat(-2.0),
//!     glam::Vec3::splat(2.0),
//!     &MarchingCubesConfig::default()
//! );
//! ```
//!
//! ## Author
//!
//! Moroya Sakamoto

#![warn(missing_docs)]

pub mod animation;
pub mod cache;
pub mod collision;
pub mod compiled;
pub mod crispy;
pub mod eval;
pub mod interval;
pub mod io;
pub mod material;
pub mod mesh;
pub mod modifiers;
pub mod neural;
pub mod operations;
pub mod optimize;
pub mod primitives;
pub mod raycast;
pub mod soa;
pub mod tight_aabb;
pub mod transforms;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "godot")]
pub mod godot;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "codec")]
pub mod codec_bridge;

#[cfg(feature = "physics")]
pub mod physics_bridge;

#[cfg(feature = "physics")]
pub mod sim_bridge;

#[cfg(feature = "texture-fit")]
pub mod texture;

#[cfg(feature = "volume")]
pub mod volume;

#[cfg(feature = "svo")]
pub mod svo;

#[cfg(feature = "destruction")]
pub mod destruction;

#[cfg(feature = "terrain")]
pub mod terrain;

#[cfg(feature = "gi")]
pub mod gi;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude - commonly used types and functions
pub mod prelude {
    pub use crate::animation::{
        morph, AnimatedSdf, Interpolation, Keyframe, LoopMode, Timeline, Track,
    };
    pub use crate::collision::{sdf_collide, sdf_distance, sdf_overlap, SdfContact};
    pub use crate::compiled::{
        eval_compiled, eval_compiled_batch_simd, eval_compiled_batch_simd_parallel,
        eval_compiled_batch_soa, eval_compiled_batch_soa_parallel, eval_compiled_bvh,
        eval_compiled_distance_and_normal, eval_compiled_normal, eval_compiled_simd,
        get_scene_aabb, AabbPacked, CompiledSdf, CompiledSdfBvh, Vec3x8,
    };
    #[cfg(feature = "destruction")]
    pub use crate::destruction::{
        carve, carve_batch, generate_debris, voronoi_fracture, CarveShape, ChunkMesh, DebrisConfig,
        DebrisPiece, DestructionResult, FractureConfig, FracturePiece, MutableVoxelGrid,
    };
    pub use crate::eval::{
        eval, eval_batch, eval_batch_parallel, eval_gradient, eval_grid, eval_material,
        eval_normal, gradient, normal,
    };
    #[cfg(feature = "gi")]
    pub use crate::gi::{
        bake_irradiance_grid, cone_trace, direct_lighting, sky_color, trace_hemisphere,
        BakeGiConfig, ConeTraceConfig, ConeTraceResult, DirectionalLight, IrradianceGrid,
        IrradianceProbe, PointLight,
    };
    pub use crate::io::{
        export_alembic, export_fbx, export_glb, export_glb_bytes, export_gltf_json, export_nanite,
        export_nanite_json, export_nanite_with_config, export_obj, export_usda, get_info,
        import_fbx, import_glb, import_glb_bytes, import_obj, load, load_asdf, load_asdf_json,
        save, save_asdf, save_asdf_json, AlembicConfig, FbxConfig, FbxFormat, FbxUpAxis,
        GltfConfig, NaniteExportConfig, ObjConfig, UsdConfig, UsdUpAxis,
    };
    pub use crate::material::{Material, MaterialLibrary, TextureSlot};
    pub use crate::mesh::{
        // Adaptive marching cubes
        adaptive_marching_cubes,
        apply_uvs,
        compute_aabb,
        compute_acmr,
        compute_bounding_sphere,
        compute_convex_hull,
        compute_quality,
        convex_decomposition,
        // Mesh decimation
        decimate,
        deduplicate_vertices,
        detect_primitive,
        // Dual Contouring
        dual_contouring,
        dual_contouring_compiled,
        extract_hermite,
        // Lightmap UV generation
        generate_lightmap_uvs,
        generate_lightmap_uvs_fast,
        generate_lod_chain,
        generate_lod_chain_decimated,
        generate_nanite_mesh,
        marching_cubes,
        mesh_to_sdf,
        mesh_to_sdf_exact,
        // Vertex cache optimization
        optimize_vertex_cache,
        point_cloud_to_sdf,
        primitives_to_csg,
        sdf_to_mesh,
        simplify_collision,
        // UV unwrapping (LSCM)
        uv_unwrap,
        validate_mesh,
        AdaptiveConfig,
        BoundingSphere,
        // Physics collision primitives
        CollisionAabb,
        CollisionMesh,
        ConvexDecomposition,
        ConvexHull,
        DecimateConfig,
        DecimationLodConfig,
        DualContouringConfig,
        // Primitive fitting
        FittedPrimitive,
        FittingConfig,
        HermiteConfig,
        // Hermite data extraction
        HermitePoint,
        // LOD generation
        LodChain,
        LodConfig,
        LodSelector,
        MarchingCubesConfig,
        Mesh,
        MeshBvh,
        MeshQuality,
        MeshRepair,
        MeshSdf,
        MeshToSdfConfig,
        MeshToSdfStrategy,
        // Manifold mesh validation & repair
        MeshValidation,
        // Nanite clusters
        NaniteCluster,
        NaniteConfig,
        NaniteMesh,
        // Point cloud SDF
        PointCloudSdf,
        PointCloudSdfConfig,
        Triangle,
        UvChart,
        UvUnwrapConfig,
        UvUnwrapResult,
        Vertex,
        // Convex decomposition (V-HACD)
        VhacdConfig,
    };
    #[cfg(feature = "gpu")]
    pub use crate::mesh::{
        gpu_marching_cubes, gpu_marching_cubes_from_shader, GpuMarchingCubesConfig,
    };
    pub use crate::modifiers::*;
    pub use crate::operations::*;
    pub use crate::primitives::*;
    pub use crate::raycast::{
        ambient_occlusion, raycast, raycast_batch, raymarch, raymarch_with_config, soft_shadow,
        RaymarchConfig, RaymarchResult,
    };
    #[cfg(feature = "physics")]
    pub use crate::sim_bridge::{simulate_sdf, SimulatedSdf};
    pub use crate::soa::{SoADistances, SoAPoints};
    #[cfg(feature = "svo")]
    pub use crate::svo::{
        build_svo, build_svo_compiled, linearize_svo, svo_nearest_surface, svo_query_point,
        svo_ray_query, LinearizedSvo, SparseVoxelOctree, SvoBuildConfig, SvoChunk, SvoNode,
        SvoRayHit, SvoStreamingCache,
    };
    #[cfg(all(feature = "terrain", feature = "image"))]
    pub use crate::terrain::HeightmapImageConfig;
    #[cfg(feature = "terrain")]
    pub use crate::terrain::{
        erode, generate_cave_sdf, terrain_sdf, CaveConfig, ClipmapLevel, ClipmapMesh,
        ClipmapTerrain, ErosionConfig, Heightmap, SplatLayer, Splatmap, TerrainConfig,
    };
    pub use crate::transforms::*;
    pub use crate::types::{Aabb, Hit, Ray, SdfCategory, SdfMetadata, SdfNode, SdfTree};
    #[cfg(feature = "volume")]
    pub use crate::volume::{
        bake_volume, bake_volume_with_normals, export_dds_3d, export_raw, generate_mip_chain,
        BakeChannels, BakeConfig, Volume3D, VoxelDistGrad,
    };
    pub use glam::{Quat, Vec3};
}

// Re-exports for convenience
pub use cache::{
    compute_cache_key, hash_sdf_node, CacheConfig, ChunkCoord, ChunkedCacheConfig,
    ChunkedMeshCache, MeshCache, MeshCacheKey,
};
pub use eval::eval;
pub use io::{load, save};
pub use mesh::sdf_to_mesh;
pub use types::{SdfNode, SdfTree};

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_workflow() {
        // Create a shape: sphere with hole carved out
        let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));

        // Evaluate at origin - box is inside sphere, so origin is carved out
        let d = eval(&shape, Vec3::ZERO);
        assert!(d > 0.0); // Outside (carved by box)

        // Evaluate at surface - should be near zero
        let d_surface = eval(&shape, Vec3::new(1.0, 0.0, 0.0));
        assert!(d_surface.abs() < 0.01);

        // Create tree
        let tree = SdfTree::new(shape);
        assert!(tree.node_count() > 0);
    }

    #[test]
    fn test_complex_tree() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::cylinder(0.5, 2.0), 0.2)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5).repeat_infinite(1.0, 1.0, 1.0))
            .twist(0.5)
            .scale(2.0);

        assert!(shape.node_count() >= 6);
    }

    #[test]
    fn test_mesh_generation() {
        let sphere = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 8,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&sphere, Vec3::splat(-2.0), Vec3::splat(2.0), &config);
        assert!(mesh.vertex_count() > 0);
        assert!(mesh.triangle_count() > 0);
    }

    #[test]
    fn test_raymarching() {
        let sphere = SdfNode::sphere(1.0);
        let ray = Ray::new(Vec3::new(-5.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));

        let hit = raycast(&sphere, ray, 10.0);
        assert!(hit.is_some());

        let hit = hit.unwrap();
        assert!((hit.distance - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_production_stress_high_res() {
        // High-resolution mesh generation (production scale)
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
        let config = MarchingCubesConfig {
            resolution: 64,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Should produce a substantial mesh
        assert!(
            mesh.vertex_count() > 1000,
            "Expected >1000 vertices, got {}",
            mesh.vertex_count()
        );
        assert!(
            mesh.triangle_count() > 500,
            "Expected >500 triangles, got {}",
            mesh.triangle_count()
        );
    }

    #[test]
    fn test_production_multi_material_pipeline() {
        // Full pipeline: SDF → mesh → optimize → glTF with multi-material
        let shape = SdfNode::sphere(1.0);
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mut mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Assign materials to half the triangles
        let half = mesh.vertices.len() / 2;
        for v in mesh.vertices[half..].iter_mut() {
            v.material_id = 1;
        }

        // Optimize vertex cache
        let acmr_before = compute_acmr(&mesh, 32);
        optimize_vertex_cache(&mut mesh);
        let acmr_after = compute_acmr(&mesh, 32);
        assert!(acmr_after <= acmr_before + 0.01);

        // Create materials
        let mut mat_lib = MaterialLibrary::new();
        mat_lib.add(
            Material::metal("Chrome", 0.9, 0.9, 0.9, 0.2)
                .with_albedo_map("textures/chrome_albedo.png")
                .with_normal_map("textures/chrome_normal.png"),
        );

        // Export glTF with multi-material
        let path = std::env::temp_dir().join("alice_stress_multi_mat.glb");
        export_glb(&mesh, &path, &GltfConfig::aaa(), Some(&mat_lib)).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() > 100);
        // Verify it's valid GLB
        assert_eq!(&bytes[0..4], &0x46546C67u32.to_le_bytes());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_production_collision_pipeline() {
        // Full pipeline: SDF → mesh → collision primitives
        let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let config = MarchingCubesConfig {
            resolution: 16,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // AABB
        let aabb = compute_aabb(&mesh);
        assert!(aabb.contains(Vec3::new(0.5, 0.0, 0.0)));

        // Bounding sphere
        let bs = compute_bounding_sphere(&mesh);
        assert!(bs.radius > 0.5);

        // Convex hull
        let hull = compute_convex_hull(&mesh);
        assert!(hull.vertices.len() >= 4);
        assert!(hull.indices.len() >= 12);

        // Simplified collision
        let simplified = simplify_collision(&mesh, 8);
        assert!(simplified.vertices.len() < mesh.vertices.len());
    }

    #[test]
    fn test_compiled_vs_interpreted() {
        // Complex shape to test
        let shape = SdfNode::sphere(1.0)
            .smooth_union(
                SdfNode::cylinder(0.3, 1.5).rotate_euler(1.57, 0.0, 0.0),
                0.2,
            )
            .subtract(SdfNode::box3d(0.4, 0.4, 0.4))
            .translate(0.5, 0.0, 0.0);

        let compiled = CompiledSdf::compile(&shape);

        // Test at many points
        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(0.25, 0.25, 0.25),
            Vec3::new(2.0, 0.0, 0.0),
        ];

        for p in test_points {
            let d_interp = eval(&shape, p);
            let d_compiled = eval_compiled(&compiled, p);
            assert!(
                (d_interp - d_compiled).abs() < 0.001,
                "Mismatch at {:?}: interp={}, compiled={}",
                p,
                d_interp,
                d_compiled
            );
        }
    }
}

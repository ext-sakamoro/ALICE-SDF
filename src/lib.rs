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

pub mod types;
pub mod material;
pub mod animation;
pub mod primitives;
pub mod operations;
pub mod transforms;
pub mod modifiers;
pub mod eval;
pub mod raycast;
pub mod mesh;
pub mod io;
pub mod compiled;
pub mod soa;
pub mod crispy;
pub mod interval;
pub mod neural;
pub mod collision;
pub mod optimize;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "ffi")]
pub mod ffi;

#[cfg(feature = "physics")]
pub mod physics_bridge;

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
    pub use crate::types::{SdfNode, SdfTree, SdfMetadata, Aabb, Ray, Hit};
    pub use crate::material::{Material, MaterialLibrary, TextureSlot};
    pub use crate::animation::{Keyframe, Track, Timeline, AnimatedSdf, Interpolation, LoopMode, morph};
    pub use crate::eval::{eval, eval_material, normal, gradient, eval_gradient, eval_normal, eval_batch, eval_batch_parallel, eval_grid};
    pub use crate::raycast::{
        raycast, raycast_batch, raymarch, raymarch_with_config,
        RaymarchConfig, RaymarchResult, ambient_occlusion, soft_shadow,
    };
    pub use crate::mesh::{
        sdf_to_mesh, mesh_to_sdf, mesh_to_sdf_exact, marching_cubes,
        MarchingCubesConfig, MeshToSdfConfig, MeshToSdfStrategy, MeshSdf,
        Mesh, Vertex, Triangle, MeshBvh,
        // Adaptive marching cubes
        adaptive_marching_cubes, AdaptiveConfig,
        // Dual Contouring
        dual_contouring, dual_contouring_compiled, DualContouringConfig,
        // Hermite data extraction
        HermitePoint, HermiteConfig, extract_hermite,
        // Primitive fitting
        FittedPrimitive, FittingConfig, detect_primitive, primitives_to_csg,
        // Nanite clusters
        NaniteCluster, NaniteMesh, NaniteConfig, generate_nanite_mesh,
        // LOD generation
        LodChain, LodConfig, LodSelector, generate_lod_chain,
        DecimationLodConfig, generate_lod_chain_decimated,
        // Manifold mesh validation & repair
        MeshValidation, MeshRepair, MeshQuality, validate_mesh, compute_quality,
        // Mesh decimation
        decimate, DecimateConfig,
        // Lightmap UV generation
        generate_lightmap_uvs, generate_lightmap_uvs_fast,
        // Vertex cache optimization
        optimize_vertex_cache, compute_acmr, deduplicate_vertices,
        // Physics collision primitives
        CollisionAabb, BoundingSphere, ConvexHull, CollisionMesh,
        compute_aabb, compute_bounding_sphere, compute_convex_hull, simplify_collision,
        // Convex decomposition (V-HACD)
        VhacdConfig, ConvexDecomposition, convex_decomposition,
        // UV unwrapping (LSCM)
        uv_unwrap, apply_uvs, UvUnwrapConfig, UvUnwrapResult, UvChart,
    };
    pub use crate::io::{
        save, load, save_asdf, load_asdf, save_asdf_json, load_asdf_json, get_info,
        export_obj, import_obj, ObjConfig,
        export_glb, export_gltf_json, GltfConfig,
        export_fbx, FbxConfig, FbxFormat, FbxUpAxis,
        export_usda, UsdConfig, UsdUpAxis,
        export_alembic, AlembicConfig,
        export_nanite, export_nanite_with_config, export_nanite_json, NaniteExportConfig,
    };
    pub use crate::compiled::{
        CompiledSdf, eval_compiled, eval_compiled_normal, eval_compiled_distance_and_normal,
        eval_compiled_simd, eval_compiled_batch_simd, eval_compiled_batch_simd_parallel,
        eval_compiled_batch_soa, eval_compiled_batch_soa_parallel,
        Vec3x8,
        CompiledSdfBvh, eval_compiled_bvh, get_scene_aabb, AabbPacked,
    };
    pub use crate::soa::{SoAPoints, SoADistances};
    #[cfg(feature = "volume")]
    pub use crate::volume::{
        Volume3D, BakeConfig, BakeChannels, VoxelDistGrad,
        bake_volume, bake_volume_with_normals,
        export_raw, export_dds_3d,
        generate_mip_chain,
    };
    #[cfg(feature = "gpu")]
    pub use crate::mesh::{
        gpu_marching_cubes, gpu_marching_cubes_from_shader, GpuMarchingCubesConfig,
    };
    #[cfg(feature = "terrain")]
    pub use crate::terrain::{
        Heightmap, TerrainConfig, terrain_sdf,
        ClipmapTerrain, ClipmapLevel, ClipmapMesh,
        Splatmap, SplatLayer,
        ErosionConfig, erode,
        CaveConfig, generate_cave_sdf,
    };
    #[cfg(feature = "destruction")]
    pub use crate::destruction::{
        MutableVoxelGrid, ChunkMesh,
        CarveShape, DestructionResult, carve, carve_batch,
        DebrisConfig, DebrisPiece, generate_debris,
        FractureConfig, FracturePiece, voronoi_fracture,
    };
    #[cfg(feature = "gi")]
    pub use crate::gi::{
        DirectionalLight, PointLight, sky_color, direct_lighting,
        ConeTraceConfig, ConeTraceResult, cone_trace, trace_hemisphere,
        IrradianceGrid, IrradianceProbe,
        BakeGiConfig, bake_irradiance_grid,
    };
    #[cfg(feature = "svo")]
    pub use crate::svo::{
        SparseVoxelOctree, SvoNode, SvoBuildConfig, SvoRayHit,
        build_svo, build_svo_compiled,
        svo_query_point, svo_ray_query, svo_nearest_surface,
        LinearizedSvo, linearize_svo,
        SvoStreamingCache, SvoChunk,
    };
    pub use crate::collision::{SdfContact, sdf_collide, sdf_distance, sdf_overlap};
    pub use crate::primitives::*;
    pub use crate::operations::*;
    pub use crate::transforms::*;
    pub use crate::modifiers::*;
    pub use glam::{Vec3, Quat};
}

// Re-exports for convenience
pub use types::{SdfNode, SdfTree};
pub use eval::eval;
pub use mesh::sdf_to_mesh;
pub use io::{save, load};

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_workflow() {
        // Create a shape: sphere with hole carved out
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5));

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
            .subtract(
                SdfNode::box3d(0.5, 0.5, 0.5)
                    .repeat_infinite(1.0, 1.0, 1.0)
            )
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
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
        let config = MarchingCubesConfig {
            resolution: 64,
            iso_level: 0.0,
            compute_normals: true,
            ..Default::default()
        };
        let mesh = sdf_to_mesh(&shape, Vec3::splat(-2.0), Vec3::splat(2.0), &config);

        // Should produce a substantial mesh
        assert!(mesh.vertex_count() > 1000, "Expected >1000 vertices, got {}", mesh.vertex_count());
        assert!(mesh.triangle_count() > 500, "Expected >500 triangles, got {}", mesh.triangle_count());
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
                .with_normal_map("textures/chrome_normal.png")
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
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5));
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
            .smooth_union(SdfNode::cylinder(0.3, 1.5).rotate_euler(1.57, 0.0, 0.0), 0.2)
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
                p, d_interp, d_compiled
            );
        }
    }
}

//! Mesh conversion for SDFs (Deep Fried Edition v2)
//!
//! Convert between polygon meshes and SDF representations.
//!
//! # Deep Fried Optimizations
//! - **Z-Slab Parallelization**: Marching Cubes processes Z-layers in parallel.
//! - **Edge Deduplication**: Mesh-to-SDF uses HashSet to halve capsule count.
//! - **BVH Acceleration**: O(log n) distance queries for exact mesh SDF.
//! - **Forced Inlining**: All hot-path functions use `#[inline(always)]`.
//!
//! # Conversion Strategies
//!
//! - **Capsule Approximation** (`mesh_to_sdf`): Fast but approximate. Creates an
//!   SdfNode tree by representing edges as capsules.
//!
//! - **BVH Exact** (`mesh_to_sdf_exact`): Accurate signed distance using BVH.
//!   Cannot be converted to shaders but provides exact mesh distance queries.
//!
//! # Advanced Features
//!
//! - **Hermite Data** (`hermite`): Extract position + normal data for Dual Contouring.
//!
//! - **Primitive Fitting** (`primitive_fitting`): Detect and fit geometric primitives
//!   (sphere, box, cylinder) to mesh data for CSG reconstruction.
//!
//! - **Nanite Clusters** (`nanite`): Generate UE5 Nanite-compatible hierarchical clusters.
//!
//! - **LOD Generation** (`lod`): Generate level-of-detail chains for efficient rendering.
//!
//! Author: Moroya Sakamoto

pub mod bvh;
pub mod collision;
pub mod decimate;
pub mod dual_contouring;
pub mod hermite;
pub mod lightmap;
pub mod lod;
pub mod manifold;
pub mod nanite;
pub mod optimize;
pub mod primitive_fitting;
pub mod point_cloud_sdf;
pub mod uv_unwrap;
mod mesh_to_sdf;
mod sdf_to_mesh;

#[cfg(feature = "gpu")]
pub mod gpu_mc_shaders;
#[cfg(feature = "gpu")]
pub mod gpu_marching_cubes;

pub use bvh::{Aabb, MeshBvh, BvhNode, Triangle as BvhTriangle};
pub use hermite::{
    HermitePoint, EdgeCrossing, HermiteConfig, HermiteExtractor,
    extract_hermite, extract_edge_crossings,
};
pub use lod::{
    LodMesh, LodChain, LodConfig, LodSelector, ContinuousLod,
    generate_lod_chain,
    DecimationLodConfig, generate_lod_chain_decimated,
};
pub use mesh_to_sdf::{
    mesh_to_sdf, mesh_to_sdf_exact,
    MeshToSdfConfig, MeshToSdfStrategy, MeshSdf
};
pub use nanite::{
    NaniteCluster, NaniteMesh, NaniteConfig, ClusterBounds, ClusterGroup, LodLevel,
    generate_nanite_mesh, CLUSTER_MAX_TRIANGLES, CLUSTER_MAX_VERTICES,
};
pub use primitive_fitting::{
    PrimitiveType, FittedPrimitive, FittingResult, FittingConfig,
    fit_sphere, fit_box, fit_cylinder, fit_plane,
    detect_primitive, primitives_to_csg,
};
pub use manifold::{
    MeshValidation, MeshRepair, MeshQuality,
    validate_mesh, compute_quality,
};
pub use decimate::{decimate, DecimateConfig};
pub use lightmap::{generate_lightmap_uvs, generate_lightmap_uvs_fast};
pub use optimize::{optimize_vertex_cache, compute_acmr, deduplicate_vertices};
pub use uv_unwrap::{uv_unwrap, apply_uvs, UvUnwrapConfig, UvUnwrapResult, UvChart};
pub use point_cloud_sdf::{PointCloudSdf, PointCloudSdfConfig, point_cloud_to_sdf};
pub use collision::{
    CollisionAabb, BoundingSphere, ConvexHull, CollisionMesh,
    compute_aabb, compute_bounding_sphere, compute_convex_hull,
    convex_hull_from_points, simplify_collision,
    VhacdConfig, ConvexDecomposition, convex_decomposition,
};
pub use sdf_to_mesh::{
    sdf_to_mesh, sdf_to_mesh_compiled, marching_cubes, marching_cubes_compiled,
    MarchingCubesConfig, Mesh,
    adaptive_marching_cubes, adaptive_marching_cubes_compiled, AdaptiveConfig,
};
pub use dual_contouring::{dual_contouring, dual_contouring_compiled, DualContouringConfig};
#[cfg(feature = "gpu")]
pub use gpu_marching_cubes::{gpu_marching_cubes, gpu_marching_cubes_from_shader, GpuMarchingCubesConfig};

/// Vertex with position, normal, UV, tangent, color, and material ID
///
/// Compatible with standard game engine vertex formats (UE5, Unity, Godot).
/// Includes UV2 for lightmap coordinates.
#[derive(Debug, Clone, Copy, Default)]
pub struct Vertex {
    /// Position in 3D space
    pub position: glam::Vec3,
    /// Surface normal (normalized)
    pub normal: glam::Vec3,
    /// Texture coordinates (triplanar projected)
    pub uv: glam::Vec2,
    /// Lightmap UV coordinates (UV channel 1)
    pub uv2: glam::Vec2,
    /// Tangent vector (xyz = direction, w = handedness sign)
    pub tangent: glam::Vec4,
    /// Vertex color (RGBA, linear space)
    pub color: [f32; 4],
    /// Material ID (indexes into MaterialLibrary)
    pub material_id: u32,
}

impl Vertex {
    /// Create a new vertex with position and normal (UV/tangent/color default)
    pub fn new(position: glam::Vec3, normal: glam::Vec3) -> Self {
        Vertex {
            position,
            normal,
            uv: glam::Vec2::ZERO,
            uv2: glam::Vec2::ZERO,
            tangent: glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            color: [1.0, 1.0, 1.0, 1.0],
            material_id: 0,
        }
    }

    /// Create a vertex with all fields specified
    pub fn with_all(
        position: glam::Vec3,
        normal: glam::Vec3,
        uv: glam::Vec2,
        uv2: glam::Vec2,
        tangent: glam::Vec4,
        color: [f32; 4],
        material_id: u32,
    ) -> Self {
        Vertex { position, normal, uv, uv2, tangent, color, material_id }
    }
}

/// Triangle face indices
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    /// First vertex index
    pub a: u32,
    /// Second vertex index
    pub b: u32,
    /// Third vertex index
    pub c: u32,
}

impl Triangle {
    /// Create a new triangle
    pub fn new(a: u32, b: u32, c: u32) -> Self {
        Triangle { a, b, c }
    }
}

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
pub mod hermite;
pub mod lod;
pub mod nanite;
pub mod primitive_fitting;
mod mesh_to_sdf;
mod sdf_to_mesh;

pub use bvh::{Aabb, MeshBvh, BvhNode, Triangle as BvhTriangle};
pub use hermite::{
    HermitePoint, EdgeCrossing, HermiteConfig, HermiteExtractor,
    extract_hermite, extract_edge_crossings,
};
pub use lod::{
    LodMesh, LodChain, LodConfig, LodSelector, ContinuousLod,
    generate_lod_chain,
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
pub use sdf_to_mesh::{sdf_to_mesh, marching_cubes, MarchingCubesConfig, Mesh};

/// Vertex with position and normal
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    /// Position in 3D space
    pub position: glam::Vec3,
    /// Surface normal
    pub normal: glam::Vec3,
}

impl Vertex {
    /// Create a new vertex
    pub fn new(position: glam::Vec3, normal: glam::Vec3) -> Self {
        Vertex { position, normal }
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

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

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "ffi")]
pub mod ffi;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Prelude - commonly used types and functions
pub mod prelude {
    pub use crate::types::{SdfNode, SdfTree, SdfMetadata, Aabb, Ray, Hit};
    pub use crate::eval::{eval, normal, gradient, eval_batch, eval_batch_parallel, eval_grid};
    pub use crate::raycast::{
        raycast, raycast_batch, raymarch, raymarch_with_config,
        RaymarchConfig, RaymarchResult, ambient_occlusion, soft_shadow,
    };
    pub use crate::mesh::{
        sdf_to_mesh, mesh_to_sdf, mesh_to_sdf_exact, marching_cubes,
        MarchingCubesConfig, MeshToSdfConfig, MeshToSdfStrategy, MeshSdf,
        Mesh, Vertex, Triangle, MeshBvh,
        // Hermite data extraction
        HermitePoint, HermiteConfig, extract_hermite,
        // Primitive fitting
        FittedPrimitive, FittingConfig, detect_primitive, primitives_to_csg,
        // Nanite clusters
        NaniteCluster, NaniteMesh, NaniteConfig, generate_nanite_mesh,
        // LOD generation
        LodChain, LodConfig, LodSelector, generate_lod_chain,
    };
    pub use crate::io::{save, load, save_asdf, load_asdf, save_asdf_json, load_asdf_json, get_info};
    pub use crate::compiled::{
        CompiledSdf, eval_compiled, eval_compiled_normal,
        eval_compiled_simd, eval_compiled_batch_simd, eval_compiled_batch_simd_parallel,
        eval_compiled_batch_soa, eval_compiled_batch_soa_parallel,
        Vec3x8,
        CompiledSdfBvh, eval_compiled_bvh, get_scene_aabb, AabbPacked,
    };
    pub use crate::soa::{SoAPoints, SoADistances};
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

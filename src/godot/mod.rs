//! Godot GDExtension integration for ALICE-SDF
//!
//! Provides native Godot 4.x nodes for SDF modeling and mesh generation.
//!
//! # Nodes
//! - `AliceSdfNode`: SDF tree builder exposed as a Godot Node3D
//! - `AliceSdfMeshInstance`: Auto-generates mesh from SDF with LOD support
//! - `AliceSdfResource`: Serializable SDF resource for the Godot inspector
//!
//! # Usage
//! ```gdscript
//! var sdf = AliceSdfNode.new()
//! sdf.set_shape("sphere", {"radius": 1.0})
//! sdf.boolean_union(other_sdf)
//! var mesh_inst = AliceSdfMeshInstance.new()
//! mesh_inst.set_sdf(sdf)
//! mesh_inst.resolution = 64
//! add_child(mesh_inst)
//! ```
//!
//! Author: Moroya Sakamoto

use godot::classes::{
    ArrayMesh, IMeshInstance3D, INode3D, IResource, MeshInstance3D, Node3D, Resource,
};
use godot::prelude::*;
use std::sync::Arc;

use crate::compiled::CompiledSdf;
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig, Vertex};
use crate::types::SdfNode;

/// GDExtension entry point
struct AliceSdfExtension;

#[gdextension]
unsafe impl ExtensionLibrary for AliceSdfExtension {}

// ============================================================
// AliceSdfResource — Serializable SDF for Godot inspector
// ============================================================

/// A serializable SDF resource that can be saved/loaded in Godot
#[derive(GodotClass)]
#[class(base=Resource, tool)]
pub struct AliceSdfResource {
    base: Base<Resource>,

    /// Internal SDF node (not directly exposed to GDScript)
    #[var]
    shape_type: GString,

    /// Shape parameters as JSON string (for Godot serialization)
    #[var]
    shape_params_json: GString,

    /// Cached SDF node
    #[init(val = None)]
    sdf_cache: Option<SdfNode>,
}

#[godot_api]
impl IResource for AliceSdfResource {
    fn init(base: Base<Resource>) -> Self {
        Self {
            base,
            shape_type: "sphere".into(),
            shape_params_json: r#"{"radius":1.0}"#.into(),
            sdf_cache: None,
        }
    }
}

#[godot_api]
impl AliceSdfResource {
    /// Set shape type and parameters
    #[func]
    fn set_shape(&mut self, shape_type: GString, params_json: GString) {
        self.shape_type = shape_type;
        self.shape_params_json = params_json;
        self.sdf_cache = None; // invalidate cache
    }

    /// Build the SdfNode from current parameters
    #[func]
    fn build_sdf(&mut self) -> bool {
        let shape = self.shape_type.to_string();
        let params = self.shape_params_json.to_string();

        let node = match build_sdf_from_params(&shape, &params) {
            Some(n) => n,
            None => return false,
        };

        self.sdf_cache = Some(node);
        true
    }

    /// Get available shape types
    #[func]
    fn get_available_shapes() -> PackedStringArray {
        let shapes = vec![
            "sphere",
            "box",
            "cylinder",
            "torus",
            "capsule",
            "cone",
            "ellipsoid",
            "rounded_box",
            "gyroid",
            "schwarz_p",
        ];
        let mut arr = PackedStringArray::new();
        for s in shapes {
            arr.push(&GString::from(s));
        }
        arr
    }
}

// ============================================================
// AliceSdfNode — SDF tree builder as Node3D
// ============================================================

/// A Node3D that represents an SDF shape in the scene tree
#[derive(GodotClass)]
#[class(base=Node3D, tool)]
pub struct AliceSdfNode {
    base: Base<Node3D>,

    /// Shape type
    #[var]
    shape: GString,

    /// Radius (for sphere, torus, etc.)
    #[var]
    radius: f32,

    /// Half extents (for box)
    #[var]
    half_extents: Vector3,

    /// Height (for cylinder, cone)
    #[var]
    height: f32,

    /// Smoothing factor for boolean operations
    #[var]
    smooth_k: f32,

    /// Built SDF node
    #[init(val = None)]
    sdf_node: Option<SdfNode>,
}

#[godot_api]
impl INode3D for AliceSdfNode {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            shape: "sphere".into(),
            radius: 1.0,
            half_extents: Vector3::new(0.5, 0.5, 0.5),
            height: 2.0,
            smooth_k: 0.0,
            sdf_node: None,
        }
    }
}

#[godot_api]
impl AliceSdfNode {
    /// Build the SDF from current node properties
    #[func]
    fn build(&mut self) -> bool {
        let node = match self.shape.to_string().as_str() {
            "sphere" => SdfNode::sphere(self.radius),
            "box" => SdfNode::box3d(
                self.half_extents.x * 2.0,
                self.half_extents.y * 2.0,
                self.half_extents.z * 2.0,
            ),
            "cylinder" => SdfNode::cylinder(self.radius, self.height),
            "torus" => SdfNode::torus(self.radius, self.radius * 0.3),
            "capsule" => SdfNode::capsule(self.height, self.radius),
            _ => return false,
        };
        self.sdf_node = Some(node);
        true
    }

    /// Perform smooth union with another AliceSdfNode
    #[func]
    fn boolean_union(&mut self, other: Gd<AliceSdfNode>) {
        let other_sdf = {
            let other_ref = other.bind();
            other_ref.sdf_node.clone()
        };
        if let (Some(a), Some(b)) = (self.sdf_node.take(), other_sdf) {
            self.sdf_node = Some(if self.smooth_k > 0.0 {
                a.smooth_union(b, self.smooth_k)
            } else {
                a.union(b)
            });
        }
    }

    /// Perform smooth subtraction with another AliceSdfNode
    #[func]
    fn boolean_subtract(&mut self, other: Gd<AliceSdfNode>) {
        let other_sdf = {
            let other_ref = other.bind();
            other_ref.sdf_node.clone()
        };
        if let (Some(a), Some(b)) = (self.sdf_node.take(), other_sdf) {
            self.sdf_node = Some(if self.smooth_k > 0.0 {
                a.smooth_subtraction(b, self.smooth_k)
            } else {
                a.subtraction(b)
            });
        }
    }

    /// Perform smooth intersection with another AliceSdfNode
    #[func]
    fn boolean_intersect(&mut self, other: Gd<AliceSdfNode>) {
        let other_sdf = {
            let other_ref = other.bind();
            other_ref.sdf_node.clone()
        };
        if let (Some(a), Some(b)) = (self.sdf_node.take(), other_sdf) {
            self.sdf_node = Some(if self.smooth_k > 0.0 {
                a.smooth_intersection(b, self.smooth_k)
            } else {
                a.intersection(b)
            });
        }
    }

    /// Evaluate the SDF distance at a point
    #[func]
    fn eval_distance(&self, point: Vector3) -> f32 {
        match &self.sdf_node {
            Some(node) => crate::eval::eval(node, glam::Vec3::new(point.x, point.y, point.z)),
            None => f32::MAX,
        }
    }

    /// Export SDF to GLSL shader code
    #[func]
    fn to_glsl(&self) -> GString {
        #[cfg(feature = "glsl")]
        {
            match &self.sdf_node {
                Some(node) => {
                    let compiled = CompiledSdf::compile(node);
                    match crate::compiled::glsl::transpile_glsl(&compiled) {
                        Ok(code) => GString::from(code),
                        Err(_) => GString::new(),
                    }
                }
                None => GString::new(),
            }
        }
        #[cfg(not(feature = "glsl"))]
        {
            GString::from("GLSL feature not enabled")
        }
    }
}

// ============================================================
// AliceSdfMeshInstance — Auto mesh generation from SDF
// ============================================================

/// A MeshInstance3D that automatically generates a mesh from an SDF
#[derive(GodotClass)]
#[class(base=MeshInstance3D, tool)]
pub struct AliceSdfMeshInstance {
    base: Base<MeshInstance3D>,

    /// Marching Cubes resolution
    #[var]
    resolution: i32,

    /// Bounds size (half-extent of the evaluation volume)
    #[var]
    bounds_size: f32,

    /// Auto-rebuild mesh when SDF changes
    #[var]
    auto_rebuild: bool,

    /// Cached SDF for mesh generation
    #[init(val = None)]
    cached_sdf: Option<SdfNode>,
}

#[godot_api]
impl IMeshInstance3D for AliceSdfMeshInstance {
    fn init(base: Base<MeshInstance3D>) -> Self {
        Self {
            base,
            resolution: 32,
            bounds_size: 2.0,
            auto_rebuild: true,
            cached_sdf: None,
        }
    }
}

#[godot_api]
impl AliceSdfMeshInstance {
    /// Set the SDF source from an AliceSdfNode
    #[func]
    fn set_sdf_from_node(&mut self, node: Gd<AliceSdfNode>) {
        let sdf = {
            let node_ref = node.bind();
            node_ref.sdf_node.clone()
        };
        self.cached_sdf = sdf;
        if self.auto_rebuild {
            self.rebuild_mesh();
        }
    }

    /// Set the SDF source from an AliceSdfResource
    #[func]
    fn set_sdf_from_resource(&mut self, mut resource: Gd<AliceSdfResource>) {
        {
            let mut res = resource.bind_mut();
            res.build_sdf();
        }
        let sdf = {
            let res = resource.bind();
            res.sdf_cache.clone()
        };
        self.cached_sdf = sdf;
        if self.auto_rebuild {
            self.rebuild_mesh();
        }
    }

    /// Rebuild the mesh from the current SDF
    #[func]
    fn rebuild_mesh(&mut self) {
        let sdf = match &self.cached_sdf {
            Some(s) => s,
            None => return,
        };

        let bounds = self.bounds_size;
        let min_bounds = glam::Vec3::splat(-bounds);
        let max_bounds = glam::Vec3::splat(bounds);

        let config = MarchingCubesConfig {
            resolution: self.resolution.max(4) as usize,
            ..Default::default()
        };

        let mesh = sdf_to_mesh(sdf, min_bounds, max_bounds, &config);

        if mesh.vertices.is_empty() {
            return;
        }

        // Convert to Godot ArrayMesh
        let godot_mesh = alice_mesh_to_godot(&mesh);
        self.base_mut().set_mesh(&godot_mesh.upcast());
    }

    /// Get vertex count of the current mesh
    #[func]
    fn get_vertex_count(&self) -> i32 {
        // Approximate from Godot mesh
        if let Some(mesh) = self.base().get_mesh() {
            if let Ok(array_mesh) = mesh.try_cast::<ArrayMesh>() {
                return array_mesh.get_surface_count();
            }
        }
        0
    }
}

// ============================================================
// Helper functions
// ============================================================

/// Convert ALICE Mesh to Godot ArrayMesh
fn alice_mesh_to_godot(mesh: &crate::mesh::Mesh) -> Gd<ArrayMesh> {
    use godot::classes::mesh::PrimitiveType;
    use godot::classes::rendering_server::ArrayType;

    let mut array_mesh = ArrayMesh::new_gd();

    let vert_count = mesh.vertices.len();
    let mut positions = PackedVector3Array::new();
    let mut normals = PackedVector3Array::new();
    let mut uvs = PackedVector2Array::new();
    let mut indices = PackedInt32Array::new();

    positions.resize(vert_count);
    normals.resize(vert_count);
    uvs.resize(vert_count);

    for (i, v) in mesh.vertices.iter().enumerate() {
        positions.set(i, Vector3::new(v.position.x, v.position.y, v.position.z));
        normals.set(i, Vector3::new(v.normal.x, v.normal.y, v.normal.z));
        uvs.set(i, Vector2::new(v.uv.x, v.uv.y));
    }

    indices.resize(mesh.indices.len());
    for (i, &idx) in mesh.indices.iter().enumerate() {
        indices.set(i, idx as i32);
    }

    let mut arrays = VariantArray::new();
    arrays.resize(ArrayType::ARRAY_MAX.ord() as usize);
    arrays.set(
        ArrayType::ARRAY_VERTEX.ord() as usize,
        &positions.to_variant(),
    );
    arrays.set(
        ArrayType::ARRAY_NORMAL.ord() as usize,
        &normals.to_variant(),
    );
    arrays.set(ArrayType::ARRAY_TEX_UV.ord() as usize, &uvs.to_variant());
    arrays.set(ArrayType::ARRAY_INDEX.ord() as usize, &indices.to_variant());

    array_mesh.add_surface_from_arrays(PrimitiveType::PRIMITIVE_TRIANGLES, &arrays);

    array_mesh
}

/// Build SdfNode from shape name and JSON params
fn build_sdf_from_params(shape: &str, params_json: &str) -> Option<SdfNode> {
    let params: serde_json::Value = serde_json::from_str(params_json).ok()?;

    match shape {
        "sphere" => {
            let radius = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::sphere(radius))
        }
        "box" => {
            let w = params.get("width")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::box3d(w, h, d))
        }
        "cylinder" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::cylinder(r, h))
        }
        "torus" => {
            let major = params.get("major_radius")?.as_f64()? as f32;
            let minor = params.get("minor_radius")?.as_f64()? as f32;
            Some(SdfNode::torus(major, minor))
        }
        "capsule" => {
            let h = params.get("height")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::capsule(h, r))
        }
        "gyroid" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::gyroid(scale, thickness))
        }
        _ => None,
    }
}

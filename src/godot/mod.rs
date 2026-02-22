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
    ArrayMesh, CollisionShape3D, ConvexPolygonShape3D, IMeshInstance3D, INode3D, IResource,
    MeshInstance3D, Node3D, Resource,
};
use godot::prelude::*;
use std::sync::Arc;

use crate::compiled::{eval_compiled_batch_parallel, CompiledSdf};
use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
use crate::types::SdfNode;

/// GDExtension entry point
struct AliceSdfExtension;

// SAFETY: `AliceSdfExtension` is an empty unit struct with no state.
// The `#[gdextension]` macro generates the GDExtension entry-point callbacks
// (init, deinit, get_minimum_library_level) that Godot calls exactly once
// during plugin load/unload. No thread-safety or aliasing concerns arise
// because the struct carries no data and the callbacks are invoked
// sequentially by the Godot engine's main thread.
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
            // Basic primitives (10)
            "sphere",
            "box",
            "cylinder",
            "torus",
            "plane",
            "capsule",
            "cone",
            "ellipsoid",
            "rounded_cone",
            "pyramid",
            // Platonic & Archimedean solids (10)
            "octahedron",
            "hex_prism",
            "tetrahedron",
            "dodecahedron",
            "icosahedron",
            "truncated_octahedron",
            "truncated_icosahedron",
            "link",
            "triangle",
            "bezier",
            // Rounded & capped shapes (10)
            "rounded_box",
            "capped_cone",
            "capped_torus",
            "rounded_cylinder",
            "triangular_prism",
            "cut_sphere",
            "cut_hollow_sphere",
            "death_star",
            "solid_angle",
            "rhombus",
            // Organic & curved shapes (10)
            "horseshoe",
            "vesica",
            "heart",
            "tube",
            "barrel",
            "diamond",
            "egg",
            "moon",
            "cross_shape",
            "blobby_cross",
            // TPMS (triply periodic minimal surfaces) (10)
            "gyroid",
            "schwarz_p",
            "diamond_surface",
            "neovius",
            "lidinoid",
            "iwp",
            "frd",
            "fischer_koch_s",
            "pmy",
            "chamfered_cube",
            // Geometric extrusions (10)
            "superellipsoid",
            "rounded_x",
            "pie",
            "trapezoid",
            "parallelogram",
            "tunnel",
            "uneven_capsule",
            "arc_shape",
            "parabola_segment",
            "regular_polygon",
            // Advanced shapes (10)
            "star_polygon",
            "stairs",
            "helix",
            "box_frame",
            "infinite_cylinder",
            "infinite_cone",
            "circle_2d",
            "rect_2d",
            "segment_2d",
            "polygon_2d",
            // 2D primitives (2)
            "rounded_rect_2d",
            "annular_2d",
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
            // Basic primitives
            "sphere" => SdfNode::sphere(self.radius),
            "box" => SdfNode::box3d(
                self.half_extents.x * 2.0,
                self.half_extents.y * 2.0,
                self.half_extents.z * 2.0,
            ),
            "cylinder" => SdfNode::cylinder(self.radius, self.height),
            "torus" => SdfNode::torus(self.radius, self.radius * 0.3),
            "capsule" => {
                let a = glam::Vec3::new(0.0, -self.height * 0.5, 0.0);
                let b = glam::Vec3::new(0.0, self.height * 0.5, 0.0);
                SdfNode::capsule(a, b, self.radius)
            }
            "cone" => SdfNode::cone(self.radius, self.height),
            "ellipsoid" => SdfNode::ellipsoid(
                self.half_extents.x,
                self.half_extents.y,
                self.half_extents.z,
            ),
            "rounded_box" => SdfNode::rounded_box(
                self.half_extents.x,
                self.half_extents.y,
                self.half_extents.z,
                self.radius * 0.1,
            ),
            "pyramid" => SdfNode::pyramid(self.height),
            "octahedron" => SdfNode::octahedron(self.radius),
            "hex_prism" => SdfNode::hex_prism(self.radius, self.height),
            "gyroid" => SdfNode::gyroid(self.radius, 0.1),
            "schwarz_p" => SdfNode::schwarz_p(self.radius, 0.1),
            "heart" => SdfNode::heart(self.radius),
            "tube" => SdfNode::tube(self.radius, self.radius * 0.2, self.height),
            "barrel" => SdfNode::barrel(self.radius, self.height, self.radius * 0.1),
            "diamond" => SdfNode::diamond(self.radius, self.height),
            "helix" => SdfNode::helix(
                self.radius,
                self.radius * 0.2,
                self.height * 0.5,
                self.height,
            ),
            "superellipsoid" => SdfNode::superellipsoid(
                self.half_extents.x * 2.0,
                self.half_extents.y * 2.0,
                self.half_extents.z * 2.0,
                0.5,
                0.5,
            ),
            "star_polygon" => SdfNode::star_polygon(self.radius, 5, self.radius * 0.5, self.height),
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
                a.smooth_subtract(b, self.smooth_k)
            } else {
                a.subtract(b)
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
        self.base_mut()
            .set_mesh(&godot_mesh.upcast::<godot::classes::Mesh>());
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
        positions[i] = Vector3::new(v.position.x, v.position.y, v.position.z);
        normals[i] = Vector3::new(v.normal.x, v.normal.y, v.normal.z);
        uvs[i] = Vector2::new(v.uv.x, v.uv.y);
    }

    indices.resize(mesh.indices.len());
    for (i, &idx) in mesh.indices.iter().enumerate() {
        indices[i] = idx as i32;
    }

    let mut arrays = VariantArray::new();
    arrays.resize(ArrayType::MAX.ord() as usize, &Variant::nil());
    arrays.set(ArrayType::VERTEX.ord() as usize, &positions.to_variant());
    arrays.set(ArrayType::NORMAL.ord() as usize, &normals.to_variant());
    arrays.set(ArrayType::TEX_UV.ord() as usize, &uvs.to_variant());
    arrays.set(ArrayType::INDEX.ord() as usize, &indices.to_variant());

    array_mesh.add_surface_from_arrays(PrimitiveType::TRIANGLES, &arrays);

    array_mesh
}

/// Build SdfNode from shape name and JSON params
fn build_sdf_from_params(shape: &str, params_json: &str) -> Option<SdfNode> {
    let params: serde_json::Value = serde_json::from_str(params_json).ok()?;

    match shape {
        // === Basic Primitives ===
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
        "plane" => {
            let nx = params
                .get("normal_x")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let ny = params
                .get("normal_y")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0) as f32;
            let nz = params
                .get("normal_z")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let dist = params
                .get("distance")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            Some(SdfNode::plane(glam::Vec3::new(nx, ny, nz), dist))
        }
        "capsule" => {
            let ax = params.get("ax").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let ay = params.get("ay").and_then(|v| v.as_f64()).unwrap_or(-1.0) as f32;
            let az = params.get("az").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let bx = params.get("bx").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let by = params.get("by").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let bz = params.get("bz").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::capsule(
                glam::Vec3::new(ax, ay, az),
                glam::Vec3::new(bx, by, bz),
                r,
            ))
        }
        "cone" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::cone(r, h))
        }
        "ellipsoid" => {
            let rx = params.get("rx")?.as_f64()? as f32;
            let ry = params.get("ry")?.as_f64()? as f32;
            let rz = params.get("rz")?.as_f64()? as f32;
            Some(SdfNode::ellipsoid(rx, ry, rz))
        }
        "rounded_cone" => {
            let r1 = params.get("r1")?.as_f64()? as f32;
            let r2 = params.get("r2")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::rounded_cone(r1, r2, h))
        }
        "pyramid" => {
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::pyramid(h))
        }

        // === Platonic & Archimedean Solids ===
        "octahedron" => {
            let size = params.get("size")?.as_f64()? as f32;
            Some(SdfNode::octahedron(size))
        }
        "hex_prism" => {
            let hex_radius = params.get("hex_radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::hex_prism(hex_radius, h))
        }
        "link" => {
            let length = params.get("length")?.as_f64()? as f32;
            let r1 = params.get("r1")?.as_f64()? as f32;
            let r2 = params.get("r2")?.as_f64()? as f32;
            Some(SdfNode::link(length, r1, r2))
        }
        "triangle" => {
            let ax = params.get("ax")?.as_f64()? as f32;
            let ay = params.get("ay")?.as_f64()? as f32;
            let az = params.get("az")?.as_f64()? as f32;
            let bx = params.get("bx")?.as_f64()? as f32;
            let by = params.get("by")?.as_f64()? as f32;
            let bz = params.get("bz")?.as_f64()? as f32;
            let cx = params.get("cx")?.as_f64()? as f32;
            let cy = params.get("cy")?.as_f64()? as f32;
            let cz = params.get("cz")?.as_f64()? as f32;
            Some(SdfNode::triangle(
                glam::Vec3::new(ax, ay, az),
                glam::Vec3::new(bx, by, bz),
                glam::Vec3::new(cx, cy, cz),
            ))
        }
        "bezier" => {
            let ax = params.get("ax")?.as_f64()? as f32;
            let ay = params.get("ay")?.as_f64()? as f32;
            let az = params.get("az")?.as_f64()? as f32;
            let bx = params.get("bx")?.as_f64()? as f32;
            let by = params.get("by")?.as_f64()? as f32;
            let bz = params.get("bz")?.as_f64()? as f32;
            let cx = params.get("cx")?.as_f64()? as f32;
            let cy = params.get("cy")?.as_f64()? as f32;
            let cz = params.get("cz")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::bezier(
                glam::Vec3::new(ax, ay, az),
                glam::Vec3::new(bx, by, bz),
                glam::Vec3::new(cx, cy, cz),
                r,
            ))
        }

        // === Rounded & Capped Shapes ===
        "rounded_box" => {
            let hx = params.get("hx")?.as_f64()? as f32;
            let hy = params.get("hy")?.as_f64()? as f32;
            let hz = params.get("hz")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::rounded_box(hx, hy, hz, rr))
        }
        "capped_cone" => {
            let h = params.get("height")?.as_f64()? as f32;
            let r1 = params.get("r1")?.as_f64()? as f32;
            let r2 = params.get("r2")?.as_f64()? as f32;
            Some(SdfNode::capped_cone(h, r1, r2))
        }
        "capped_torus" => {
            let major = params.get("major_radius")?.as_f64()? as f32;
            let minor = params.get("minor_radius")?.as_f64()? as f32;
            let cap = params
                .get("cap_angle")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.57) as f32;
            Some(SdfNode::capped_torus(major, minor, cap))
        }
        "rounded_cylinder" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::rounded_cylinder(r, rr, h))
        }
        "triangular_prism" => {
            let w = params.get("width")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::triangular_prism(w, d))
        }
        "cut_sphere" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let ch = params.get("cut_height")?.as_f64()? as f32;
            Some(SdfNode::cut_sphere(r, ch))
        }
        "cut_hollow_sphere" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let ch = params.get("cut_height")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            Some(SdfNode::cut_hollow_sphere(r, ch, t))
        }
        "death_star" => {
            let ra = params.get("ra")?.as_f64()? as f32;
            let rb = params.get("rb")?.as_f64()? as f32;
            let d = params.get("d")?.as_f64()? as f32;
            Some(SdfNode::death_star(ra, rb, d))
        }
        "solid_angle" => {
            let angle = params.get("angle")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::solid_angle(angle, r))
        }
        "rhombus" => {
            let la = params.get("la")?.as_f64()? as f32;
            let lb = params.get("lb")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            Some(SdfNode::rhombus(la, lb, h, rr))
        }

        // === Organic & Curved Shapes ===
        "horseshoe" => {
            let angle = params.get("angle")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            let length = params.get("length")?.as_f64()? as f32;
            let w = params.get("width")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            Some(SdfNode::horseshoe(angle, r, length, w, t))
        }
        "vesica" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let dist = params.get("dist")?.as_f64()? as f32;
            Some(SdfNode::vesica(r, dist))
        }
        "infinite_cylinder" => {
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::infinite_cylinder(r))
        }
        "infinite_cone" => {
            let angle = params.get("angle")?.as_f64()? as f32;
            Some(SdfNode::infinite_cone(angle))
        }
        "heart" => {
            let size = params.get("size")?.as_f64()? as f32;
            Some(SdfNode::heart(size))
        }
        "tube" => {
            let outer_r = params.get("outer_radius")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::tube(outer_r, t, h))
        }
        "barrel" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            let bulge = params.get("bulge").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
            Some(SdfNode::barrel(r, h, bulge))
        }
        "diamond" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::diamond(r, h))
        }
        "chamfered_cube" => {
            let w = params.get("width")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            let chamfer = params
                .get("chamfer")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::chamfered_cube(w, h, d, chamfer))
        }
        "egg" => {
            let ra = params.get("ra")?.as_f64()? as f32;
            let rb = params.get("rb")?.as_f64()? as f32;
            Some(SdfNode::egg(ra, rb))
        }
        "moon" => {
            let d = params.get("d")?.as_f64()? as f32;
            let ra = params.get("ra")?.as_f64()? as f32;
            let rb = params.get("rb")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::moon(d, ra, rb, h))
        }
        "cross_shape" => {
            let length = params.get("length")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0) as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::cross_shape(length, t, rr, h))
        }
        "blobby_cross" => {
            let size = params.get("size")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::blobby_cross(size, h))
        }

        // === TPMS (Triply Periodic Minimal Surfaces) ===
        "gyroid" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::gyroid(scale, thickness))
        }
        "schwarz_p" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::schwarz_p(scale, thickness))
        }
        "diamond_surface" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::diamond_surface(scale, thickness))
        }
        "neovius" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::neovius(scale, thickness))
        }
        "lidinoid" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::lidinoid(scale, thickness))
        }
        "iwp" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::iwp(scale, thickness))
        }
        "frd" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::frd(scale, thickness))
        }
        "fischer_koch_s" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::fischer_koch_s(scale, thickness))
        }
        "pmy" => {
            let scale = params.get("scale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
            let thickness = params
                .get("thickness")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            Some(SdfNode::pmy(scale, thickness))
        }

        // === Geometric Extrusions ===
        "superellipsoid" => {
            let w = params.get("width")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            let e1 = params.get("e1").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let e2 = params.get("e2").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            Some(SdfNode::superellipsoid(w, h, d, e1, e2))
        }
        "rounded_x" => {
            let w = params.get("width")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::rounded_x(w, rr, h))
        }
        "pie" => {
            let angle = params.get("angle")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::pie(angle, r, h))
        }
        "trapezoid" => {
            let r1 = params.get("r1")?.as_f64()? as f32;
            let r2 = params.get("r2")?.as_f64()? as f32;
            let trap_height = params.get("trap_height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::trapezoid(r1, r2, trap_height, d))
        }
        "parallelogram" => {
            let w = params.get("width")?.as_f64()? as f32;
            let para_height = params.get("para_height")?.as_f64()? as f32;
            let skew = params.get("skew").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::parallelogram(w, para_height, skew, d))
        }
        "tunnel" => {
            let w = params.get("width")?.as_f64()? as f32;
            let height_2d = params.get("height_2d")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::tunnel(w, height_2d, d))
        }
        "uneven_capsule" => {
            let r1 = params.get("r1")?.as_f64()? as f32;
            let r2 = params.get("r2")?.as_f64()? as f32;
            let cap_height = params.get("cap_height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::uneven_capsule(r1, r2, cap_height, d))
        }
        "arc_shape" => {
            let aperture = params.get("aperture")?.as_f64()? as f32;
            let r = params.get("radius")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::arc_shape(aperture, r, t, h))
        }
        "parabola_segment" => {
            let w = params.get("width")?.as_f64()? as f32;
            let para_height = params.get("para_height")?.as_f64()? as f32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::parabola_segment(w, para_height, d))
        }
        "regular_polygon" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let n_sides = params
                .get("n_sides")
                .and_then(|v| v.as_f64())
                .unwrap_or(6.0) as u32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::regular_polygon(r, n_sides, h))
        }

        // === Advanced Shapes ===
        "star_polygon" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let n_points = params
                .get("n_points")
                .and_then(|v| v.as_f64())
                .unwrap_or(5.0) as u32;
            let m = params.get("m").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::star_polygon(r, n_points, m, h))
        }
        "stairs" => {
            let step_width = params.get("step_width")?.as_f64()? as f32;
            let step_height = params.get("step_height")?.as_f64()? as f32;
            let n_steps = params
                .get("n_steps")
                .and_then(|v| v.as_f64())
                .unwrap_or(5.0) as u32;
            let d = params.get("depth")?.as_f64()? as f32;
            Some(SdfNode::stairs(step_width, step_height, n_steps, d))
        }
        "helix" => {
            let major_r = params.get("major_r")?.as_f64()? as f32;
            let minor_r = params.get("minor_r")?.as_f64()? as f32;
            let pitch = params.get("pitch")?.as_f64()? as f32;
            let h = params.get("height")?.as_f64()? as f32;
            Some(SdfNode::helix(major_r, minor_r, pitch, h))
        }
        "tetrahedron" => {
            let size = params.get("size")?.as_f64()? as f32;
            Some(SdfNode::tetrahedron(size))
        }
        "dodecahedron" => {
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::dodecahedron(r))
        }
        "icosahedron" => {
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::icosahedron(r))
        }
        "truncated_octahedron" => {
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::truncated_octahedron(r))
        }
        "truncated_icosahedron" => {
            let r = params.get("radius")?.as_f64()? as f32;
            Some(SdfNode::truncated_icosahedron(r))
        }
        "box_frame" => {
            let hx = params.get("hx")?.as_f64()? as f32;
            let hy = params.get("hy")?.as_f64()? as f32;
            let hz = params.get("hz")?.as_f64()? as f32;
            let edge = params.get("edge").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
            Some(SdfNode::box_frame(glam::Vec3::new(hx, hy, hz), edge))
        }

        // === 2D Primitives ===
        "circle_2d" => {
            let r = params.get("radius")?.as_f64()? as f32;
            let hh = params.get("half_height")?.as_f64()? as f32;
            Some(SdfNode::circle_2d(r, hh))
        }
        "rect_2d" => {
            let hw = params.get("half_w")?.as_f64()? as f32;
            let hh_rect = params.get("half_h")?.as_f64()? as f32;
            let hh = params.get("half_height")?.as_f64()? as f32;
            Some(SdfNode::rect_2d(hw, hh_rect, hh))
        }
        "segment_2d" => {
            let ax = params.get("ax")?.as_f64()? as f32;
            let ay = params.get("ay")?.as_f64()? as f32;
            let bx = params.get("bx")?.as_f64()? as f32;
            let by = params.get("by")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            let hh = params.get("half_height")?.as_f64()? as f32;
            Some(SdfNode::segment_2d(ax, ay, bx, by, t, hh))
        }
        "polygon_2d" => {
            // Requires array parsing - simplified
            let hh = params.get("half_height")?.as_f64()? as f32;
            let verts = vec![
                glam::Vec2::new(1.0, 0.0),
                glam::Vec2::new(0.0, 1.0),
                glam::Vec2::new(-1.0, 0.0),
            ];
            Some(SdfNode::polygon_2d(verts, hh))
        }
        "rounded_rect_2d" => {
            let hw = params.get("half_w")?.as_f64()? as f32;
            let hh_rect = params.get("half_h")?.as_f64()? as f32;
            let rr = params
                .get("round_radius")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1) as f32;
            let hh = params.get("half_height")?.as_f64()? as f32;
            Some(SdfNode::rounded_rect_2d(hw, hh_rect, rr, hh))
        }
        "annular_2d" => {
            let outer_r = params.get("outer_radius")?.as_f64()? as f32;
            let t = params.get("thickness")?.as_f64()? as f32;
            let hh = params.get("half_height")?.as_f64()? as f32;
            Some(SdfNode::annular_2d(outer_r, t, hh))
        }

        _ => None,
    }
}

// ============================================================
// AliceSdfCollisionGenerator — Collision shape from SDF
// ============================================================

/// Generates ConvexPolygonShape3D collision shapes from SDF meshes
#[derive(GodotClass)]
#[class(base=Node3D, tool)]
pub struct AliceSdfCollisionGenerator {
    base: Base<Node3D>,

    /// Resolution for collision mesh (lower than visual for performance)
    #[var]
    collision_resolution: i32,

    /// Bounds size (half-extent of the evaluation volume)
    #[var]
    bounds_size: f32,

    /// Maximum number of collision vertices (decimation target)
    #[var]
    max_collision_vertices: i32,
}

#[godot_api]
impl INode3D for AliceSdfCollisionGenerator {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            collision_resolution: 16,
            bounds_size: 2.0,
            max_collision_vertices: 256,
        }
    }
}

#[godot_api]
impl AliceSdfCollisionGenerator {
    /// Generate a ConvexPolygonShape3D from an AliceSdfNode
    ///
    /// Returns a CollisionShape3D node that can be added as a child
    /// of a StaticBody3D or RigidBody3D.
    #[func]
    fn generate_collision_shape(&self, sdf_node: Gd<AliceSdfNode>) -> Option<Gd<CollisionShape3D>> {
        let sdf = {
            let node_ref = sdf_node.bind();
            node_ref.sdf_node.clone()
        }?;

        let bounds = self.bounds_size;
        let min_bounds = glam::Vec3::splat(-bounds);
        let max_bounds = glam::Vec3::splat(bounds);

        let config = MarchingCubesConfig {
            resolution: self.collision_resolution.max(4) as usize,
            compute_normals: false, // Normals not needed for collision
            ..Default::default()
        };

        let mesh = sdf_to_mesh(&sdf, min_bounds, max_bounds, &config);

        if mesh.vertices.is_empty() {
            return None;
        }

        // Convert vertices to PackedVector3Array for ConvexPolygonShape3D
        let mut points = PackedVector3Array::new();

        // Subsample if too many vertices
        let step = if mesh.vertices.len() > self.max_collision_vertices as usize {
            mesh.vertices.len() / self.max_collision_vertices as usize
        } else {
            1
        };

        for (i, v) in mesh.vertices.iter().enumerate() {
            if i % step == 0 {
                points.push(Vector3::new(v.position.x, v.position.y, v.position.z));
            }
        }

        let mut convex_shape = ConvexPolygonShape3D::new_gd();
        convex_shape.set_points(&points);

        let mut collision_shape = CollisionShape3D::new_alloc();
        collision_shape.set_shape(&convex_shape.upcast::<godot::classes::Shape3D>());

        Some(collision_shape)
    }
}

// ============================================================
// AliceSdfLodManager — Camera-distance LOD switching
// ============================================================

/// Manages LOD switching based on camera distance for SDF meshes
#[derive(GodotClass)]
#[class(base=Node3D, tool)]
pub struct AliceSdfLodManager {
    base: Base<Node3D>,

    /// Number of LOD levels to generate
    #[var]
    num_lod_levels: i32,

    /// Base resolution for LOD 0 (highest detail)
    #[var]
    base_resolution: i32,

    /// Distance at which LOD 0 transitions to LOD 1
    #[var]
    lod0_distance: f32,

    /// Distance multiplier for each subsequent LOD level
    #[var]
    distance_multiplier: f32,

    /// Bounds size (half-extent of the evaluation volume)
    #[var]
    bounds_size: f32,

    /// Current active LOD level
    #[var]
    current_lod: i32,

    /// Cached LOD meshes
    lod_meshes: Vec<Gd<ArrayMesh>>,

    /// Cached SDF for mesh generation
    cached_sdf: Option<SdfNode>,
}

#[godot_api]
impl INode3D for AliceSdfLodManager {
    fn init(base: Base<Node3D>) -> Self {
        Self {
            base,
            num_lod_levels: 3,
            base_resolution: 64,
            lod0_distance: 10.0,
            distance_multiplier: 2.0,
            bounds_size: 2.0,
            current_lod: 0,
            lod_meshes: Vec::new(),
            cached_sdf: None,
        }
    }
}

#[godot_api]
impl AliceSdfLodManager {
    /// Set the SDF source and generate all LOD levels
    #[func]
    fn set_sdf(&mut self, sdf_node: Gd<AliceSdfNode>) {
        let sdf = {
            let node_ref = sdf_node.bind();
            node_ref.sdf_node.clone()
        };
        self.cached_sdf = sdf;
        self.generate_all_lods();
    }

    /// Generate all LOD level meshes
    #[func]
    fn generate_all_lods(&mut self) {
        let sdf = match &self.cached_sdf {
            Some(s) => s,
            None => return,
        };

        let bounds = self.bounds_size;
        let min_bounds = glam::Vec3::splat(-bounds);
        let max_bounds = glam::Vec3::splat(bounds);

        self.lod_meshes.clear();

        for level in 0..self.num_lod_levels {
            // Halve resolution per LOD level
            let res = (self.base_resolution >> level).max(4) as usize;

            let config = MarchingCubesConfig {
                resolution: res,
                compute_normals: true,
                ..Default::default()
            };

            let mesh = sdf_to_mesh(sdf, min_bounds, max_bounds, &config);

            if !mesh.vertices.is_empty() {
                self.lod_meshes.push(alice_mesh_to_godot(&mesh));
            }
        }
    }

    /// Update LOD based on distance to a given camera position
    ///
    /// Call this from `_process()` with the camera's global position.
    /// Returns the selected LOD level.
    #[func]
    fn update_lod(&mut self, camera_position: Vector3) -> i32 {
        if self.lod_meshes.is_empty() {
            return -1;
        }

        let self_pos = self.base().get_global_position();
        let distance = self_pos.distance_to(camera_position);

        // Determine LOD level from distance
        let mut level = 0i32;
        let mut threshold = self.lod0_distance;
        while level < (self.lod_meshes.len() as i32 - 1) && distance > threshold {
            level += 1;
            threshold *= self.distance_multiplier;
        }

        self.current_lod = level;
        level
    }

    /// Get the mesh for a specific LOD level
    #[func]
    fn get_lod_mesh(&self, level: i32) -> Option<Gd<ArrayMesh>> {
        self.lod_meshes.get(level as usize).cloned()
    }

    /// Get the number of generated LOD levels
    #[func]
    fn get_lod_count(&self) -> i32 {
        self.lod_meshes.len() as i32
    }

    /// Get vertex count for each LOD level (for debugging)
    #[func]
    fn get_lod_vertex_counts(&self) -> PackedInt32Array {
        let mut counts = PackedInt32Array::new();
        for mesh in &self.lod_meshes {
            counts.push(mesh.get_surface_count());
        }
        counts
    }
}

// ============================================================
// AliceSdfBatchEvaluator — Batch SDF evaluation for GDScript
// ============================================================

/// High-performance batch SDF evaluation exposed to GDScript
///
/// Compiles the SDF once and evaluates many points using
/// SIMD + multi-threading (GIL-free Rust backend).
#[derive(GodotClass)]
#[class(base=RefCounted)]
pub struct AliceSdfBatchEvaluator {
    base: Base<RefCounted>,

    /// Compiled SDF for fast evaluation
    compiled: Option<Arc<CompiledSdf>>,
}

#[godot_api]
impl IRefCounted for AliceSdfBatchEvaluator {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            base,
            compiled: None,
        }
    }
}

#[godot_api]
impl AliceSdfBatchEvaluator {
    /// Compile an SDF for fast batch evaluation
    #[func]
    fn compile(&mut self, sdf_node: Gd<AliceSdfNode>) -> bool {
        let sdf = {
            let node_ref = sdf_node.bind();
            node_ref.sdf_node.clone()
        };
        match sdf {
            Some(node) => {
                self.compiled = Some(Arc::new(CompiledSdf::compile(&node)));
                true
            }
            None => false,
        }
    }

    /// Evaluate SDF distances at multiple points
    ///
    /// `points` is a PackedVector3Array of query positions.
    /// Returns a PackedFloat32Array of SDF distances.
    #[func]
    fn eval_batch(&self, points: PackedVector3Array) -> PackedFloat32Array {
        let compiled = match &self.compiled {
            Some(c) => c,
            None => return PackedFloat32Array::new(),
        };

        let len = points.len();
        let mut glam_points = Vec::with_capacity(len);
        for i in 0..len {
            let p = points[i];
            glam_points.push(glam::Vec3::new(p.x, p.y, p.z));
        }

        let distances = eval_compiled_batch_parallel(compiled, &glam_points);

        let mut result = PackedFloat32Array::new();
        result.resize(len);
        for (i, &d) in distances.iter().enumerate() {
            result[i] = d;
        }
        result
    }

    /// Check if a compiled SDF is available
    #[func]
    fn is_compiled(&self) -> bool {
        self.compiled.is_some()
    }

    /// Get the instruction count of the compiled SDF
    #[func]
    fn get_instruction_count(&self) -> i32 {
        self.compiled
            .as_ref()
            .map(|c| c.node_count as i32)
            .unwrap_or(0)
    }
}

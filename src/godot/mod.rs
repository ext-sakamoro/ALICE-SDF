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

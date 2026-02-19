//! WGSL Transpiler: SDF to WebGPU Shading Language (Dynamic Edition)
//!
//! Supports "Hardcoded" (max speed) and "Dynamic" (animation) modes.
//!
//! - **Hardcoded**: Constants baked into shader (fastest execution, requires recompile on change)
//! - **Dynamic**: Constants read from uniform buffer (fast update via `write_buffer`, good execution)
//!
//! # Deep Fried Optimizations
//!
//! - **Division Exorcism**: Replaces `/k` with `* inv_k` in smooth ops
//! - **Inline Smooth Ops**: Pre-computes reciprocals at transpile time
//! - **FMA-friendly**: Structures operations for GPU FMA units
//!
//! Author: Moroya Sakamoto

use super::super::transpiler_common::ShaderLang;
use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
const FOLD_EPSILON: f32 = 1e-6;

// ============================================================================
// WgslLang: ShaderLang trait implementation for WGSL
// ============================================================================

/// Marker type for WGSL shader language syntax
pub struct WgslLang;

impl ShaderLang for WgslLang {
    fn vec2_ctor(x: &str, y: &str) -> String {
        format!("vec2<f32>({}, {})", x, y)
    }
    fn vec3_ctor(x: &str, y: &str, z: &str) -> String {
        format!("vec3<f32>({}, {}, {})", x, y, z)
    }
    fn vec4_ctor(x: &str, y: &str, z: &str, w: &str) -> String {
        format!("vec4<f32>({}, {}, {}, {})", x, y, z, w)
    }
    fn vec2_zero() -> &'static str {
        "vec2<f32>(0.0)"
    }
    fn vec3_zero() -> &'static str {
        "vec3<f32>(0.0)"
    }
    fn vec2_splat(v: &str) -> String {
        format!("vec2<f32>({})", v)
    }
    fn vec3_splat(v: &str) -> String {
        format!("vec3<f32>({})", v)
    }
    fn decl_float(name: &str, expr: &str) -> String {
        format!("    let {} = {};\n", name, expr)
    }
    fn decl_vec2(name: &str, expr: &str) -> String {
        format!("    let {} = {};\n", name, expr)
    }
    fn decl_vec3(name: &str, expr: &str) -> String {
        format!("    let {} = {};\n", name, expr)
    }
    fn decl_mut_float(name: &str, expr: &str) -> String {
        format!("    var {} = {};\n", name, expr)
    }
    fn decl_mut_vec2(name: &str, expr: &str) -> String {
        format!("    var {} = {};\n", name, expr)
    }
    fn decl_mut_vec3(name: &str, expr: &str) -> String {
        format!("    var {} = {};\n", name, expr)
    }
    fn decl_mut_float_typed(name: &str) -> String {
        format!("    var {}: f32;\n", name)
    }
    fn decl_mut_vec3_typed(name: &str) -> String {
        format!("    var {}: vec3<f32>;\n", name)
    }
    fn select_expr(cond: &str, true_val: &str, false_val: &str) -> String {
        format!("select({}, {}, {})", false_val, true_val, cond)
    }
    fn modulo_expr(a: &str, b: &str) -> String {
        format!("({} % {})", a, b)
    }
    fn cast_float(expr: &str) -> String {
        format!("f32({})", expr)
    }
    fn for_loop_int(name: &str, init: i32, cond: &str, incr: &str) -> String {
        format!(
            "    var {} = {}i;\n    loop {{\n        if !({}) {{ break; }}\n",
            name, init, cond
        )
    }
    fn param_dynamic(vec_idx: usize, comp: &str) -> String {
        format!("sdf_params.data[{}].{}", vec_idx, comp)
    }
    fn func_signature() -> &'static str {
        "fn sdf_eval(p: vec3<f32>) -> f32 {"
    }
    const CAPSULE_DEGENERATE_GUARD: bool = true;
    const CAPSULE_RE_EMIT_PARAMS: bool = false;
    fn helper_source(name: &str) -> Option<&'static str> {
        match name {
            "smooth_min" => Some(HELPER_SMOOTH_MIN),
            "smooth_max" => Some(HELPER_SMOOTH_MAX),
            "quat_rotate" => Some(HELPER_QUAT_ROTATE),
            "hash_noise" => Some(HELPER_HASH_NOISE),
            "sdf_rounded_cone" => Some(HELPER_SDF_ROUNDED_CONE),
            "sdf_pyramid" => Some(HELPER_SDF_PYRAMID),
            "sdf_octahedron" => Some(HELPER_SDF_OCTAHEDRON),
            "sdf_hex_prism" => Some(HELPER_SDF_HEX_PRISM),
            "sdf_link" => Some(HELPER_SDF_LINK),
            "sdf_triangle" => Some(HELPER_SDF_TRIANGLE),
            "sdf_bezier" => Some(HELPER_SDF_BEZIER),
            "sdf_capped_cone" => Some(HELPER_SDF_CAPPED_CONE),
            "sdf_capped_torus" => Some(HELPER_SDF_CAPPED_TORUS),
            "sdf_rounded_cylinder" => Some(HELPER_SDF_ROUNDED_CYLINDER),
            "sdf_triangular_prism" => Some(HELPER_SDF_TRIANGULAR_PRISM),
            "sdf_cut_sphere" => Some(HELPER_SDF_CUT_SPHERE),
            "sdf_cut_hollow_sphere" => Some(HELPER_SDF_CUT_HOLLOW_SPHERE),
            "sdf_death_star" => Some(HELPER_SDF_DEATH_STAR),
            "sdf_solid_angle" => Some(HELPER_SDF_SOLID_ANGLE),
            "sdf_rhombus" => Some(HELPER_SDF_RHOMBUS),
            "sdf_horseshoe" => Some(HELPER_SDF_HORSESHOE),
            "sdf_vesica" => Some(HELPER_SDF_VESICA),
            "sdf_infinite_cone" => Some(HELPER_SDF_INFINITE_CONE),
            "sdf_heart" => Some(HELPER_SDF_HEART),
            "sdf_tube" => Some(HELPER_SDF_TUBE),
            "sdf_barrel" => Some(HELPER_SDF_BARREL),
            "sdf_diamond" => Some(HELPER_SDF_DIAMOND),
            "sdf_chamfered_cube" => Some(HELPER_SDF_CHAMFERED_CUBE),
            "sdf_superellipsoid" => Some(HELPER_SDF_SUPERELLIPSOID),
            "sdf_rounded_x" => Some(HELPER_SDF_ROUNDED_X),
            "sdf_pie" => Some(HELPER_SDF_PIE),
            "sdf_trapezoid" => Some(HELPER_SDF_TRAPEZOID),
            "sdf_parallelogram" => Some(HELPER_SDF_PARALLELOGRAM),
            "sdf_tunnel" => Some(HELPER_SDF_TUNNEL),
            "sdf_uneven_capsule" => Some(HELPER_SDF_UNEVEN_CAPSULE),
            "sdf_egg" => Some(HELPER_SDF_EGG),
            "sdf_arc_shape" => Some(HELPER_SDF_ARC_SHAPE),
            "sdf_moon" => Some(HELPER_SDF_MOON),
            "sdf_cross_shape" => Some(HELPER_SDF_CROSS_SHAPE),
            "sdf_blobby_cross" => Some(HELPER_SDF_BLOBBY_CROSS),
            "sdf_parabola_segment" => Some(HELPER_SDF_PARABOLA_SEGMENT),
            "sdf_regular_polygon" => Some(HELPER_SDF_REGULAR_POLYGON),
            "sdf_star_polygon" => Some(HELPER_SDF_STAR_POLYGON),
            "sdf_stairs" => Some(HELPER_SDF_STAIRS),
            "sdf_helix" => Some(HELPER_SDF_HELIX),
            _ => None,
        }
    }
}

/// Transpilation mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranspileMode {
    /// Constants are baked into the shader (Faster execution, slow update)
    Hardcoded,
    /// Constants are read from uniform buffer (Fast update, good execution)
    Dynamic,
}

/// Generated WGSL shader code with metadata
#[derive(Debug, Clone)]
pub struct WgslShader {
    /// The generated WGSL source code (sdf_eval function + helpers)
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
    /// Initial parameter values (empty for Hardcoded mode)
    pub param_layout: Vec<f32>,
    /// Transpilation mode used
    pub mode: TranspileMode,
    /// [Deep Fried v2] Workgroup size for compute shaders (default 256).
    /// Set via `with_workgroup_size()` after querying device limits.
    pub workgroup_size: u32,
}

impl WgslShader {
    /// Transpile an SDF node tree to WGSL with the specified mode
    pub fn transpile(node: &SdfNode, mode: TranspileMode) -> Self {
        let mut transpiler = WgslTranspiler::new(mode);
        let body = transpiler.transpile_node(node, "p");
        let source = transpiler.generate_shader(&body);

        WgslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
            param_layout: transpiler.params,
            mode,
            workgroup_size: 256, // Default; override with with_workgroup_size()
        }
    }

    /// Extract parameter values from an SDF node tree
    ///
    /// Returns the same parameter layout as produced by `transpile()` in Dynamic mode.
    /// Use this with `GpuEvaluator::update_params()` for zero-latency parameter updates.
    pub fn extract_params(node: &SdfNode) -> Vec<f32> {
        let mut transpiler = WgslTranspiler::new(TranspileMode::Dynamic);
        let _ = transpiler.transpile_node(node, "p");
        transpiler.params
    }

    /// [Deep Fried v2] Set workgroup size based on device limits.
    ///
    /// Call with `device.limits().max_compute_workgroup_size_x` to adapt
    /// to the GPU's optimal workgroup size. Clamps to power-of-2 between 64 and 1024.
    pub fn with_workgroup_size(mut self, max_workgroup_x: u32) -> Self {
        // Clamp to power-of-2 <= device limit, between 64..=1024
        let clamped = max_workgroup_x.min(1024).max(64);
        // Round down to nearest power of 2
        self.workgroup_size = 1 << (31 - clamped.leading_zeros());
        self
    }

    /// Generate a complete compute shader for batch distance evaluation
    pub fn to_compute_shader(&self) -> String {
        let params_decl = if self.mode == TranspileMode::Dynamic {
            "\nstruct Params {\n    data: array<vec4<f32>, 1024>,\n}\n\n@group(0) @binding(3) var<uniform> sdf_params: Params;\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Generated Compute Shader ({mode:?} Mode)

struct InputPoint {{
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}}

struct OutputDistance {{
    distance: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}}

@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
{params_decl}
{sdf_func}

@compute @workgroup_size({wg_size})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
    let idx = global_id.x;
    if (idx >= point_count) {{
        return;
    }}

    let point = input_points[idx];
    let p = vec3<f32>(point.x, point.y, point.z);
    let distance = sdf_eval(p);
    output_distances[idx].distance = distance;
}}
"#,
            mode = self.mode,
            params_decl = params_decl,
            sdf_func = self.source,
            wg_size = self.workgroup_size,
        )
    }

    /// Generate a compute shader that outputs both distances and normals
    ///
    /// Uses the Tetrahedral Method (4 SDF evaluations) for GPU-side normal estimation.
    /// Output is packed as `(dist, nx, ny, nz)` per point.
    pub fn to_compute_shader_with_normals(&self) -> String {
        let params_decl = if self.mode == TranspileMode::Dynamic {
            "\nstruct Params {\n    data: array<vec4<f32>, 1024>,\n}\n\n@group(0) @binding(3) var<uniform> sdf_params: Params;\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Generated Compute Shader - Distance + Normals ({mode:?} Mode)

struct InputPoint {{
    x: f32,
    y: f32,
    z: f32,
    _pad: f32,
}}

struct Output {{
    dist: f32,
    nx: f32,
    ny: f32,
    nz: f32,
}}

@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output: array<Output>;
@group(0) @binding(2) var<uniform> point_count: u32;
{params_decl}
{sdf_func}

fn estimate_normal(p: vec3<f32>) -> vec3<f32> {{
    let e = 0.001;
    let k0 = vec3<f32>(1.0, -1.0, -1.0);
    let k1 = vec3<f32>(-1.0, -1.0, 1.0);
    let k2 = vec3<f32>(-1.0, 1.0, -1.0);
    let k3 = vec3<f32>(1.0, 1.0, 1.0);

    let d0 = sdf_eval(p + k0 * e);
    let d1 = sdf_eval(p + k1 * e);
    let d2 = sdf_eval(p + k2 * e);
    let d3 = sdf_eval(p + k3 * e);

    return normalize(k0 * d0 + k1 * d1 + k2 * d2 + k3 * d3);
}}

@compute @workgroup_size({wg_size})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= point_count) {{ return; }}

    let pt = input_points[idx];
    let p = vec3<f32>(pt.x, pt.y, pt.z);

    let d = sdf_eval(p);
    let n = estimate_normal(p);

    output[idx].dist = d;
    output[idx].nx = n.x;
    output[idx].ny = n.y;
    output[idx].nz = n.z;
}}
"#,
            mode = self.mode,
            params_decl = params_decl,
            sdf_func = self.source,
            wg_size = self.workgroup_size,
        )
    }

    /// Get the SDF evaluation function only (for embedding in custom shaders)
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }

    /// Generate a 3D volume bake compute shader
    ///
    /// Dispatches with `@workgroup_size(4, 4, 4)` over the volume grid.
    /// Each thread evaluates one voxel and writes to a flat storage buffer.
    #[cfg(feature = "volume")]
    pub fn to_volume_shader(&self) -> String {
        format!(
            r#"// ALICE-SDF Volume Bake Shader (3D Dispatch)

struct VolumeUniforms {{
    resolution: vec4<u32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}}

@group(0) @binding(0) var<storage, read_write> output_volume: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: VolumeUniforms;

{sdf_func}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let res = uniforms.resolution.xyz;

    if (gid.x >= res.x || gid.y >= res.y || gid.z >= res.z) {{
        return;
    }}

    let fres = vec3<f32>(f32(res.x) - 1.0, f32(res.y) - 1.0, f32(res.z) - 1.0);
    let t = vec3<f32>(f32(gid.x), f32(gid.y), f32(gid.z)) / max(fres, vec3<f32>(1.0));
    let p = mix(uniforms.bounds_min.xyz, uniforms.bounds_max.xyz, t);

    let distance = sdf_eval(p);

    let idx = gid.x + gid.y * res.x + gid.z * res.x * res.y;
    output_volume[idx] = distance;
}}
"#,
            sdf_func = self.source,
        )
    }
}

/// Internal transpiler state
struct WgslTranspiler {
    /// Counter for generating unique variable names
    var_counter: usize,
    /// Helper functions that need to be included
    helper_functions: Vec<&'static str>,
    /// Transpilation mode
    mode: TranspileMode,
    /// Collected parameter values (Dynamic mode)
    params: Vec<f32>,
}

impl WgslTranspiler {
    fn new(mode: TranspileMode) -> Self {
        WgslTranspiler {
            var_counter: 0,
            helper_functions: Vec::new(),
            mode,
            params: Vec::new(),
        }
    }

    fn next_var(&mut self) -> String {
        let var = format!("d{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    fn ensure_helper(&mut self, name: &'static str) {
        if !self.helper_functions.contains(&name) {
            self.helper_functions.push(name);
        }
    }

    /// Register a float parameter and return its WGSL expression string.
    ///
    /// - Hardcoded: returns a literal like `"1.000000"`
    /// - Dynamic: pushes to param buffer and returns `"sdf_params.data[i].comp"`
    fn param(&mut self, value: f32) -> String {
        match self.mode {
            TranspileMode::Hardcoded => format!("{:.6}", value),
            TranspileMode::Dynamic => {
                let idx = self.params.len();
                self.params.push(value);
                let vec_idx = idx / 4;
                let comp = match idx % 4 {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => "w",
                };
                format!("sdf_params.data[{}].{}", vec_idx, comp)
            }
        }
    }

    fn transpile_node(&mut self, node: &SdfNode, point_var: &str) -> String {
        use super::super::transpiler_common::{GenericTranspiler, TranspileModeLang};
        let mode = match self.mode {
            TranspileMode::Hardcoded => TranspileModeLang::Hardcoded,
            TranspileMode::Dynamic => TranspileModeLang::Dynamic,
        };
        let mut generic: GenericTranspiler<WgslLang> = GenericTranspiler::new(mode);
        generic.var_counter = self.var_counter;
        generic.params = std::mem::take(&mut self.params);
        generic.helper_functions = std::mem::take(&mut self.helper_functions);
        let body = generic.transpile_node(node, point_var);
        self.var_counter = generic.var_counter;
        self.params = generic.params;
        self.helper_functions = generic.helper_functions;
        body
    }

    fn generate_shader(&self, body: &str) -> String {
        let mut shader = String::new();

        // Add helper functions
        for helper in &self.helper_functions {
            match *helper {
                "smooth_min" => {
                    shader.push_str(HELPER_SMOOTH_MIN);
                    shader.push('\n');
                }
                "smooth_max" => {
                    shader.push_str(HELPER_SMOOTH_MAX);
                    shader.push('\n');
                }
                "quat_rotate" => {
                    shader.push_str(HELPER_QUAT_ROTATE);
                    shader.push('\n');
                }
                "hash_noise" => {
                    shader.push_str(HELPER_HASH_NOISE);
                    shader.push('\n');
                }
                "sdf_rounded_cone" => {
                    shader.push_str(HELPER_SDF_ROUNDED_CONE);
                    shader.push('\n');
                }
                "sdf_pyramid" => {
                    shader.push_str(HELPER_SDF_PYRAMID);
                    shader.push('\n');
                }
                "sdf_octahedron" => {
                    shader.push_str(HELPER_SDF_OCTAHEDRON);
                    shader.push('\n');
                }
                "sdf_hex_prism" => {
                    shader.push_str(HELPER_SDF_HEX_PRISM);
                    shader.push('\n');
                }
                "sdf_link" => {
                    shader.push_str(HELPER_SDF_LINK);
                    shader.push('\n');
                }
                "sdf_triangle" => {
                    shader.push_str(HELPER_SDF_TRIANGLE);
                    shader.push('\n');
                }
                "sdf_bezier" => {
                    shader.push_str(HELPER_SDF_BEZIER);
                    shader.push('\n');
                }
                "sdf_capped_cone" => {
                    shader.push_str(HELPER_SDF_CAPPED_CONE);
                    shader.push('\n');
                }
                "sdf_capped_torus" => {
                    shader.push_str(HELPER_SDF_CAPPED_TORUS);
                    shader.push('\n');
                }
                "sdf_rounded_cylinder" => {
                    shader.push_str(HELPER_SDF_ROUNDED_CYLINDER);
                    shader.push('\n');
                }
                "sdf_triangular_prism" => {
                    shader.push_str(HELPER_SDF_TRIANGULAR_PRISM);
                    shader.push('\n');
                }
                "sdf_cut_sphere" => {
                    shader.push_str(HELPER_SDF_CUT_SPHERE);
                    shader.push('\n');
                }
                "sdf_cut_hollow_sphere" => {
                    shader.push_str(HELPER_SDF_CUT_HOLLOW_SPHERE);
                    shader.push('\n');
                }
                "sdf_death_star" => {
                    shader.push_str(HELPER_SDF_DEATH_STAR);
                    shader.push('\n');
                }
                "sdf_solid_angle" => {
                    shader.push_str(HELPER_SDF_SOLID_ANGLE);
                    shader.push('\n');
                }
                "sdf_rhombus" => {
                    shader.push_str(HELPER_SDF_RHOMBUS);
                    shader.push('\n');
                }
                "sdf_horseshoe" => {
                    shader.push_str(HELPER_SDF_HORSESHOE);
                    shader.push('\n');
                }
                "sdf_vesica" => {
                    shader.push_str(HELPER_SDF_VESICA);
                    shader.push('\n');
                }
                "sdf_infinite_cone" => {
                    shader.push_str(HELPER_SDF_INFINITE_CONE);
                    shader.push('\n');
                }
                "sdf_heart" => {
                    shader.push_str(HELPER_SDF_HEART);
                    shader.push('\n');
                }
                "sdf_tube" => {
                    shader.push_str(HELPER_SDF_TUBE);
                    shader.push('\n');
                }
                "sdf_barrel" => {
                    shader.push_str(HELPER_SDF_BARREL);
                    shader.push('\n');
                }
                "sdf_diamond" => {
                    shader.push_str(HELPER_SDF_DIAMOND);
                    shader.push('\n');
                }
                "sdf_chamfered_cube" => {
                    shader.push_str(HELPER_SDF_CHAMFERED_CUBE);
                    shader.push('\n');
                }
                "sdf_superellipsoid" => {
                    shader.push_str(HELPER_SDF_SUPERELLIPSOID);
                    shader.push('\n');
                }
                "sdf_rounded_x" => {
                    shader.push_str(HELPER_SDF_ROUNDED_X);
                    shader.push('\n');
                }
                "sdf_pie" => {
                    shader.push_str(HELPER_SDF_PIE);
                    shader.push('\n');
                }
                "sdf_trapezoid" => {
                    shader.push_str(HELPER_SDF_TRAPEZOID);
                    shader.push('\n');
                }
                "sdf_parallelogram" => {
                    shader.push_str(HELPER_SDF_PARALLELOGRAM);
                    shader.push('\n');
                }
                "sdf_tunnel" => {
                    shader.push_str(HELPER_SDF_TUNNEL);
                    shader.push('\n');
                }
                "sdf_uneven_capsule" => {
                    shader.push_str(HELPER_SDF_UNEVEN_CAPSULE);
                    shader.push('\n');
                }
                "sdf_egg" => {
                    shader.push_str(HELPER_SDF_EGG);
                    shader.push('\n');
                }
                "sdf_arc_shape" => {
                    shader.push_str(HELPER_SDF_ARC_SHAPE);
                    shader.push('\n');
                }
                "sdf_moon" => {
                    shader.push_str(HELPER_SDF_MOON);
                    shader.push('\n');
                }
                "sdf_cross_shape" => {
                    shader.push_str(HELPER_SDF_CROSS_SHAPE);
                    shader.push('\n');
                }
                "sdf_blobby_cross" => {
                    shader.push_str(HELPER_SDF_BLOBBY_CROSS);
                    shader.push('\n');
                }
                "sdf_parabola_segment" => {
                    shader.push_str(HELPER_SDF_PARABOLA_SEGMENT);
                    shader.push('\n');
                }
                "sdf_regular_polygon" => {
                    shader.push_str(HELPER_SDF_REGULAR_POLYGON);
                    shader.push('\n');
                }
                "sdf_star_polygon" => {
                    shader.push_str(HELPER_SDF_STAR_POLYGON);
                    shader.push('\n');
                }
                "sdf_stairs" => {
                    shader.push_str(HELPER_SDF_STAIRS);
                    shader.push('\n');
                }
                "sdf_helix" => {
                    shader.push_str(HELPER_SDF_HELIX);
                    shader.push('\n');
                }
                _ => {}
            }
        }

        // Add main SDF function
        writeln!(shader, "fn sdf_eval(p: vec3<f32>) -> f32 {{").unwrap();
        shader.push_str(body);
        shader.push_str("}\n");

        shader
    }
}

// Helper function definitions
const HELPER_SMOOTH_MIN: &str = r#"// Deep Fried Edition: Division-free smooth_min
// inv_k should be pre-computed as 1.0/k
fn smooth_min_fast(a: f32, b: f32, k: f32, inv_k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) * inv_k;  // MUL instead of DIV!
    return min(a, b) - h * h * k * 0.25;
}

// Legacy smooth_min (slower, uses division)
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    return smooth_min_fast(a, b, k, 1.0 / k);
}"#;

const HELPER_SMOOTH_MAX: &str = r#"// Deep Fried Edition: Division-free smooth_max
fn smooth_max_fast(a: f32, b: f32, k: f32, inv_k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) * inv_k;
    return max(a, b) + h * h * k * 0.25;
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    return smooth_max_fast(a, b, k, 1.0 / k);
}"#;

const HELPER_HASH_NOISE: &str = r#"fn hash_noise_3d(p: vec3<f32>, seed: u32) -> f32 {
    let f = fract(p);
    let i = floor(p);
    let u = f * f * (3.0 - 2.0 * f);
    let s = f32(seed);
    let n000 = fract(sin(dot(i, vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n100 = fract(sin(dot(i + vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n010 = fract(sin(dot(i + vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n110 = fract(sin(dot(i + vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n001 = fract(sin(dot(i + vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n101 = fract(sin(dot(i + vec3<f32>(1.0, 0.0, 1.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n011 = fract(sin(dot(i + vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let n111 = fract(sin(dot(i + vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(127.1, 311.7, 74.7)) + s) * 43758.5453);
    let c00 = mix(n000, n100, u.x);
    let c10 = mix(n010, n110, u.x);
    let c01 = mix(n001, n101, u.x);
    let c11 = mix(n011, n111, u.x);
    let c0 = mix(c00, c10, u.y);
    let c1 = mix(c01, c11, u.y);
    return mix(c0, c1, u.z) * 2.0 - 1.0;
}"#;

const HELPER_QUAT_ROTATE: &str = r#"fn quat_rotate(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

const HELPER_SDF_ROUNDED_CONE: &str = r#"fn sdf_rounded_cone(p: vec3<f32>, r1: f32, r2: f32, h: f32) -> f32 {
    let qx = length(p.xz);
    let qy = p.y + h;
    let ht = h * 2.0;
    let b = (r1 - r2) / ht;
    let a = sqrt(1.0 - b * b);
    let k = qx * (-b) + qy * a;
    if (k < 0.0) { return length(vec2<f32>(qx, qy)) - r1; }
    if (k > a * ht) { return length(vec2<f32>(qx, qy - ht)) - r2; }
    return qx * a + qy * b - r1;
}
"#;

const HELPER_SDF_PYRAMID: &str = r#"fn sdf_pyramid(p: vec3<f32>, h: f32) -> f32 {
    let ht = h * 2.0;
    let m2 = ht * ht + 0.25;
    let py = p.y + h;
    var px = abs(p.x);
    var pz = abs(p.z);
    if (pz > px) { let tmp = px; px = pz; pz = tmp; }
    px -= 0.5;
    pz -= 0.5;
    let qx = pz;
    let qy = ht * py - 0.5 * px;
    let qz = ht * px + 0.5 * py;
    let s = max(-qx, 0.0);
    let t = clamp((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    let a = m2 * (qx + s) * (qx + s) + qy * qy;
    let b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
    var d2: f32;
    if (min(-qx * m2 - qy * 0.5, qy) > 0.0) { d2 = 0.0; } else { d2 = min(a, b); }
    return sqrt((d2 + qz * qz) / m2) * sign(max(qz, -py));
}
"#;

const HELPER_SDF_OCTAHEDRON: &str = r#"fn sdf_octahedron(p: vec3<f32>, s: f32) -> f32 {
    let ap = abs(p);
    let m = ap.x + ap.y + ap.z - s;
    var q: vec3<f32>;
    if (3.0 * ap.x < m) { q = ap; }
    else if (3.0 * ap.y < m) { q = vec3<f32>(ap.y, ap.z, ap.x); }
    else if (3.0 * ap.z < m) { q = vec3<f32>(ap.z, ap.x, ap.y); }
    else { return m * 0.57735027; }
    let k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3<f32>(q.x, q.y - s + k, q.z - k));
}
"#;

const HELPER_SDF_HEX_PRISM: &str = r#"fn sdf_hex_prism(p: vec3<f32>, hex_r: f32, h: f32) -> f32 {
    let kx: f32 = -0.8660254;
    let ky: f32 = 0.5;
    let kz: f32 = 0.57735027;
    var px = abs(p.x);
    var py = abs(p.y);
    let pz = abs(p.z);
    let dot_kxy = kx * px + ky * py;
    let refl = 2.0 * min(dot_kxy, 0.0);
    px -= refl * kx;
    py -= refl * ky;
    let clamped_x = clamp(px, -kz * hex_r, kz * hex_r);
    let dx = px - clamped_x;
    let dy = py - hex_r;
    let d_xy = sqrt(dx * dx + dy * dy) * sign(dy);
    let d_z = pz - h;
    return min(max(d_xy, d_z), 0.0) + length(max(vec2<f32>(d_xy, d_z), vec2<f32>(0.0)));
}
"#;

const HELPER_SDF_TRIANGLE: &str = r#"fn sdf_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> f32 {
    let ba = b - a; let pa = p - a;
    let cb = c - b; let pb = p - b;
    let ac = a - c; let pc = p - c;
    let nor = cross(ba, ac);
    let sign_check = sign(dot(cross(ba, nor), pa)) +
                     sign(dot(cross(cb, nor), pb)) +
                     sign(dot(cross(ac, nor), pc));
    if (sign_check < 2.0) {
        let d1 = dot(ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa,
                      ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa);
        let d2 = dot(cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb,
                      cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb);
        let d3 = dot(ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc,
                      ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc);
        return sqrt(min(min(d1, d2), d3));
    } else {
        return sqrt(dot(nor, pa) * dot(nor, pa) / dot(nor, nor));
    }
}
"#;

const HELPER_SDF_BEZIER: &str = r#"fn sdf_bezier(pos: vec3<f32>, A: vec3<f32>, B: vec3<f32>, C: vec3<f32>, rad: f32) -> f32 {
    let a = B - A;
    let b = A - 2.0 * B + C;
    let c = a * 2.0;
    let d = A - pos;
    let kk = 1.0 / max(dot(b, b), 1e-10);
    let kx = kk * dot(a, b);
    let ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    let kz = kk * dot(d, a);
    let p = ky - kx * kx;
    let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    let p3 = p * p * p;
    let q2 = q * q;
    let h = q2 + 4.0 * p3;
    var res: f32;
    if (h >= 0.0) {
        let sh = sqrt(h);
        let x = (vec2<f32>(sh, -sh) - q) * 0.5;
        let uv = sign(x) * pow(abs(x), vec2<f32>(1.0 / 3.0));
        let t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        let qp = d + (c + b * t) * t;
        res = dot(qp, qp);
    } else {
        let z = sqrt(-p);
        let v = acos(q / (p * z * 2.0)) / 3.0;
        let m = cos(v);
        let n = sin(v) * 1.732050808;
        let t0 = clamp(( m + m) * z - kx, 0.0, 1.0);
        let t1 = clamp((-n - m) * z - kx, 0.0, 1.0);
        let qp0 = d + (c + b * t0) * t0;
        let qp1 = d + (c + b * t1) * t1;
        res = min(dot(qp0, qp0), dot(qp1, qp1));
    }
    return sqrt(res) - rad;
}
"#;

const HELPER_SDF_LINK: &str = r#"fn sdf_link(p: vec3<f32>, le: f32, r1: f32, r2: f32) -> f32 {
    let qx = p.x;
    let qy = max(abs(p.y) - le, 0.0);
    let qz = p.z;
    let xy_len = sqrt(qx * qx + qy * qy) - r1;
    return sqrt(xy_len * xy_len + qz * qz) - r2;
}
"#;

const HELPER_SDF_CAPPED_CONE: &str = r#"fn sdf_capped_cone(p: vec3<f32>, h: f32, r1: f32, r2: f32) -> f32 {
    let qx = length(p.xz);
    let qy = p.y;
    let k1 = vec2<f32>(r2, h);
    let k2 = vec2<f32>(r2 - r1, 2.0 * h);
    var min_r: f32;
    if (qy < 0.0) { min_r = r1; } else { min_r = r2; }
    let ca = vec2<f32>(qx - min(qx, min_r), abs(qy) - h);
    let q = vec2<f32>(qx, qy);
    let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    var s: f32;
    if (cb.x < 0.0 && ca.y < 0.0) { s = -1.0; } else { s = 1.0; }
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
"#;

const HELPER_SDF_CAPPED_TORUS: &str = r#"fn sdf_capped_torus(p: vec3<f32>, ra: f32, rb: f32, an: f32) -> f32 {
    let sc = vec2<f32>(sin(an), cos(an));
    let px = abs(p.x);
    var k: f32;
    if (sc.y * px > sc.x * p.y) {
        k = dot(vec2<f32>(px, p.y), sc);
    } else {
        k = length(vec2<f32>(px, p.y));
    }
    return sqrt(px * px + p.y * p.y + p.z * p.z + ra * ra - 2.0 * ra * k) - rb;
}
"#;

const HELPER_SDF_ROUNDED_CYLINDER: &str = r#"fn sdf_rounded_cylinder(p: vec3<f32>, r: f32, rr: f32, h: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - r + rr, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0))) - rr;
}
"#;

const HELPER_SDF_TRIANGULAR_PRISM: &str = r#"fn sdf_triangular_prism(p: vec3<f32>, w: f32, h: f32) -> f32 {
    let q = abs(p);
    return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - w * 0.5);
}
"#;

const HELPER_SDF_CUT_SPHERE: &str = r#"fn sdf_cut_sphere(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let w = sqrt(max(r * r - h * h, 0.0));
    let q = vec2<f32>(length(p.xz), p.y);
    let s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
    if (s < 0.0) { return length(q) - r; }
    if (q.x < w) { return h - q.y; }
    return length(q - vec2<f32>(w, h));
}
"#;

const HELPER_SDF_CUT_HOLLOW_SPHERE: &str = r#"fn sdf_cut_hollow_sphere(p: vec3<f32>, r: f32, h: f32, t: f32) -> f32 {
    let w = sqrt(max(r * r - h * h, 0.0));
    let q = vec2<f32>(length(p.xz), p.y);
    if (h * q.x < w * q.y) {
        return length(q - vec2<f32>(w, h)) - t;
    }
    return abs(length(q) - r) - t;
}
"#;

const HELPER_SDF_DEATH_STAR: &str = r#"fn sdf_death_star(p: vec3<f32>, ra: f32, rb: f32, d: f32) -> f32 {
    let a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    let b = sqrt(max(ra * ra - a * a, 0.0));
    let q = vec2<f32>(p.x, length(p.yz));
    if (q.x * b - q.y * a > d * max(b - q.y, 0.0)) {
        return length(q - vec2<f32>(a, b));
    }
    return max(length(q) - ra, -(length(q - vec2<f32>(d, 0.0)) - rb));
}
"#;

const HELPER_SDF_SOLID_ANGLE: &str = r#"fn sdf_solid_angle(p: vec3<f32>, an: f32, ra: f32) -> f32 {
    let c = vec2<f32>(sin(an), cos(an));
    let q = vec2<f32>(length(p.xz), p.y);
    let l = length(q) - ra;
    let m = length(q - c * clamp(dot(q, c), 0.0, ra));
    return max(l, m * sign(c.y * q.x - c.x * q.y));
}
"#;

const HELPER_SDF_RHOMBUS: &str = r#"fn ndot_rh(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.x - a.y * b.y;
}
fn sdf_rhombus(p: vec3<f32>, la: f32, lb: f32, h: f32, ra: f32) -> f32 {
    let ap = abs(p);
    let b = vec2<f32>(la, lb);
    let f = clamp(ndot_rh(b, b - 2.0 * ap.xz) / dot(b, b), -1.0, 1.0);
    let dxz = length(ap.xz - 0.5 * b * vec2<f32>(1.0 - f, 1.0 + f)) * sign(ap.x * b.y + ap.z * b.x - b.x * b.y) - ra;
    let q = vec2<f32>(dxz, ap.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0)));
}
"#;

const HELPER_SDF_HORSESHOE: &str = r#"fn sdf_horseshoe(pos: vec3<f32>, an: f32, r: f32, le: f32, w: f32, t: f32) -> f32 {
    let c = vec2<f32>(cos(an), sin(an));
    let px = abs(pos.x);
    let l = length(vec2<f32>(px, pos.y));
    var qx = -c.x * px + c.y * pos.y;
    var qy = c.y * px + c.x * pos.y;
    if (!(qy > 0.0 || qx > 0.0)) { qx = l * sign(-c.x); }
    if (qx <= 0.0) { qy = l; }
    qx = abs(qx) - le;
    qy = abs(qy - r);
    let e = length(max(vec2<f32>(qx, qy), vec2<f32>(0.0))) + min(max(qx, qy), 0.0);
    let d = abs(vec2<f32>(e, pos.z)) - vec2<f32>(w, t);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}
"#;

const HELPER_SDF_VESICA: &str = r#"fn sdf_vesica(p: vec3<f32>, r: f32, d: f32) -> f32 {
    let px = abs(p.x);
    let py = length(p.yz);
    let b = sqrt(max(r * r - d * d, 0.0));
    if ((py - b) * d > px * b) {
        return length(vec2<f32>(px, py - b));
    }
    return length(vec2<f32>(px - d, py)) - r;
}
"#;

const HELPER_SDF_INFINITE_CONE: &str = r#"fn sdf_infinite_cone(p: vec3<f32>, an: f32) -> f32 {
    let c = vec2<f32>(sin(an), cos(an));
    let q = vec2<f32>(length(p.xz), -p.y);
    let d = length(q - c * max(dot(q, c), 0.0));
    if (q.x * c.y - q.y * c.x < 0.0) { return -d; }
    return d;
}
"#;

const HELPER_SDF_HEART: &str = r#"fn sdf_heart(p: vec3<f32>, s: f32) -> f32 {
    let q = p / s;
    let x = length(q.xz);
    let y = -(q.y - 0.5);
    let x2 = x * x;
    let y2 = y * y;
    let y3 = y2 * y;
    let cubic = x2 + y2 - 1.0;
    let iv = cubic * cubic * cubic - x2 * y3;
    if (iv <= 0.0) { return -0.02 * s; }
    return (pow(iv, 1.0 / 6.0) * 0.5 - 0.02) * s;
}
"#;

const HELPER_SDF_TUBE: &str = r#"fn sdf_tube(p: vec3<f32>, outer_r: f32, thick: f32, h: f32) -> f32 {
    let r = length(p.xz);
    let dr = abs(r - outer_r) - thick;
    let dy = abs(p.y) - h;
    let w = max(vec2<f32>(dr, dy), vec2<f32>(0.0));
    return min(max(dr, dy), 0.0) + length(w);
}
"#;

const HELPER_SDF_BARREL: &str = r#"fn sdf_barrel(p: vec3<f32>, radius: f32, h: f32, bulge: f32) -> f32 {
    let r = length(p.xz);
    let yn = clamp(p.y / h, -1.0, 1.0);
    let er = radius + bulge * (1.0 - yn * yn);
    let dr = r - er;
    let dy = abs(p.y) - h;
    let w = max(vec2<f32>(dr, dy), vec2<f32>(0.0));
    return min(max(dr, dy), 0.0) + length(w);
}
"#;

const HELPER_SDF_DIAMOND: &str = r#"fn sdf_diamond(p: vec3<f32>, r: f32, h: f32) -> f32 {
    let q = vec2<f32>(length(p.xz), abs(p.y));
    let ba = vec2<f32>(-r, h);
    let qa = q - vec2<f32>(r, 0.0);
    let t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    let closest = vec2<f32>(r, 0.0) + ba * t;
    let dist = length(q - closest);
    if (q.x * h + q.y * r < r * h) { return -dist; }
    return dist;
}
"#;

const HELPER_SDF_CHAMFERED_CUBE: &str = r#"fn sdf_chamfered_cube(p: vec3<f32>, hx: f32, hy: f32, hz: f32, ch: f32) -> f32 {
    let ap = abs(p);
    let q = ap - vec3<f32>(hx, hy, hz);
    let d_box = min(max(q.x, max(q.y, q.z)), 0.0) + length(max(q, vec3<f32>(0.0)));
    let s = hx + hy + hz;
    let d_ch = (ap.x + ap.y + ap.z - s + ch) * 0.57735;
    return max(d_box, d_ch);
}
"#;

const HELPER_SDF_SUPERELLIPSOID: &str = r#"fn sdf_superellipsoid(p: vec3<f32>, hx: f32, hy: f32, hz: f32, e1: f32, e2: f32) -> f32 {
    let qx = max(abs(p.x / hx), 0.00001);
    let qy = max(abs(p.y / hy), 0.00001);
    let qz = max(abs(p.z / hz), 0.00001);
    let ee1 = max(e1, 0.02);
    let ee2 = max(e2, 0.02);
    let m1 = 2.0 / ee2;
    let m2 = 2.0 / ee1;
    let w = pow(qx, m1) + pow(qz, m1);
    let v = pow(w, ee2 / ee1) + pow(qy, m2);
    let f = pow(v, ee1 * 0.5);
    return (f - 1.0) * min(hx, min(hy, hz)) * 0.5;
}
"#;

const HELPER_SDF_ROUNDED_X: &str = r#"fn sdf_rounded_x(p: vec3<f32>, w: f32, r: f32, h: f32) -> f32 {
    let q = abs(p.xz);
    let s = min(q.x + q.y, w) * 0.5;
    let d2d = length(q - vec2<f32>(s)) - r;
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_PIE: &str = r#"fn sdf_pie(p: vec3<f32>, angle: f32, radius: f32, h: f32) -> f32 {
    let q = vec2<f32>(p.x, p.z);
    let l = length(q) - radius;
    let sc = vec2<f32>(sin(angle), cos(angle));
    let m = length(q) * clamp(sc.y * abs(q.x) - sc.x * q.y, -radius, 0.0);
    let d2d = max(l, m / max(radius, 1e-10));
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_TRAPEZOID: &str = r#"fn _trap_ds(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    var t = 0.0;
    if (len_sq > 0.0) { t = ((px - ax) * dx + (py - ay) * dy) / len_sq; }
    let tc = clamp(t, 0.0, 1.0);
    let cx = ax + dx * tc;
    let cy = ay + dy * tc;
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}
fn sdf_trapezoid(p: vec3<f32>, r1: f32, r2: f32, th: f32, hd: f32) -> f32 {
    let px = abs(p.x);
    let py = p.y;
    let he = th;
    let d_bot = _trap_ds(px, py, 0.0, -he, r1, -he);
    let d_slant = _trap_ds(px, py, r1, -he, r2, he);
    let d_top = _trap_ds(px, py, r2, he, 0.0, he);
    let d_unsigned = min(d_bot, min(d_slant, d_top));
    let nx = 2.0 * he;
    let ny = r1 - r2;
    let d_slant_plane = (px - r1) * nx + (py + he) * ny;
    var d2d = d_unsigned;
    if (py >= -he && py <= he && d_slant_plane <= 0.0) { d2d = -d_unsigned; }
    let dz = abs(p.z) - hd;
    let ww = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
"#;

const HELPER_SDF_PARALLELOGRAM: &str = r#"fn _para_ds(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;
    var t = 0.0;
    if (len_sq > 0.0) { t = ((px - ax) * dx + (py - ay) * dy) / len_sq; }
    let tc = clamp(t, 0.0, 1.0);
    let cx = ax + dx * tc;
    let cy = ay + dy * tc;
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}
fn _para_c2d(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}
fn sdf_parallelogram(p: vec3<f32>, w: f32, ph: f32, sk: f32, hd: f32) -> f32 {
    let px = p.x;
    let py = p.y;
    let vdx = w - sk;  let vdy = -ph;
    let vax = w + sk;  let vay = ph;
    let vbx = -w + sk; let vby = ph;
    let vcx = -w - sk; let vcy = -ph;
    let d1 = _para_ds(px, py, vdx, vdy, vax, vay);
    let d2 = _para_ds(px, py, vax, vay, vbx, vby);
    let d3 = _para_ds(px, py, vbx, vby, vcx, vcy);
    let d4 = _para_ds(px, py, vcx, vcy, vdx, vdy);
    let d_unsigned = min(d1, min(d2, min(d3, d4)));
    let c1 = _para_c2d(px, py, vdx, vdy, vax, vay);
    let c2 = _para_c2d(px, py, vax, vay, vbx, vby);
    let c3 = _para_c2d(px, py, vbx, vby, vcx, vcy);
    let c4 = _para_c2d(px, py, vcx, vcy, vdx, vdy);
    var d2d = d_unsigned;
    if (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0 && c4 <= 0.0) { d2d = -d_unsigned; }
    let dz = abs(p.z) - hd;
    let ww = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
"#;

const HELPER_SDF_TUNNEL: &str = r#"fn sdf_tunnel(p: vec3<f32>, w: f32, h2d: f32, hd: f32) -> f32 {
    let px = abs(p.x);
    let py = p.y;
    let dx = px - w;
    let dy_rect = abs(py) - h2d;
    let d_rect = length(max(vec2<f32>(dx, dy_rect), vec2<f32>(0.0))) + min(max(dx, dy_rect), 0.0);
    let d_circle = length(vec2<f32>(px, py - h2d)) - w;
    var d2d = d_rect;
    if (py > h2d) { d2d = min(d_rect, d_circle); }
    let dz = abs(p.z) - hd;
    let ww = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
"#;

const HELPER_SDF_UNEVEN_CAPSULE: &str = r#"fn sdf_uneven_capsule(p: vec3<f32>, r1: f32, r2: f32, ch: f32, hd: f32) -> f32 {
    let px = abs(p.x);
    let hh = ch * 2.0;
    let b = (r1 - r2) / hh;
    let a = sqrt(max(1.0 - b * b, 0.0));
    let k = dot(vec2<f32>(-b, a), vec2<f32>(px, p.y));
    var d2d: f32;
    if (k < 0.0) {
        d2d = length(vec2<f32>(px, p.y)) - r1;
    } else if (k > a * hh) {
        d2d = length(vec2<f32>(px, p.y - hh)) - r2;
    } else {
        d2d = dot(vec2<f32>(px, p.y), vec2<f32>(a, b)) - r1;
    }
    let dz = abs(p.z) - hd;
    let ww = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
"#;

const HELPER_SDF_EGG: &str = r#"fn sdf_egg(p: vec3<f32>, ra: f32, rb: f32) -> f32 {
    let px = length(p.xz);
    let py = p.y;
    let r = ra - rb;
    if (py < 0.0) {
        return length(vec2<f32>(px, py)) - r;
    } else if (px * ra < py * rb) {
        return length(vec2<f32>(px, py - ra));
    } else {
        return length(vec2<f32>(px + rb, py)) - ra;
    }
}
"#;

const HELPER_SDF_ARC_SHAPE: &str = r#"fn sdf_arc_shape(p: vec3<f32>, aperture: f32, radius: f32, thickness: f32, h: f32) -> f32 {
    let qx = abs(p.x);
    let qz = p.z;
    let sc = vec2<f32>(sin(aperture), cos(aperture));
    var d2d: f32;
    if (sc.y * qx > sc.x * qz) {
        d2d = length(vec2<f32>(qx, qz) - sc * radius) - thickness;
    } else {
        d2d = abs(length(vec2<f32>(qx, qz)) - radius) - thickness;
    }
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_MOON: &str = r#"fn sdf_moon(p: vec3<f32>, d: f32, ra: f32, rb: f32, h: f32) -> f32 {
    let qx = abs(p.x);
    let qz = p.z;
    let d_outer = length(vec2<f32>(qx, qz)) - ra;
    let d_inner = length(vec2<f32>(qx - d, qz)) - rb;
    let d2d = max(d_outer, -d_inner);
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_CROSS_SHAPE: &str = r#"fn sdf_cross_shape(p: vec3<f32>, len: f32, th: f32, rr: f32, h: f32) -> f32 {
    let qx = abs(p.x);
    let qz = abs(p.z);
    let dh = vec2<f32>(qx - len, qz - th);
    let dv = vec2<f32>(qx - th, qz - len);
    let dh_sdf = length(max(dh, vec2<f32>(0.0))) + min(max(dh.x, dh.y), 0.0);
    let dv_sdf = length(max(dv, vec2<f32>(0.0))) + min(max(dv.x, dv.y), 0.0);
    let d2d = min(dh_sdf, dv_sdf) - rr;
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_BLOBBY_CROSS: &str = r#"fn sdf_blobby_cross(p: vec3<f32>, size: f32, h: f32) -> f32 {
    let qx = abs(p.x) / size;
    let qz = abs(p.z) / size;
    let n = qx + qz;
    var d2d: f32;
    if (n < 1.0) {
        let t = 1.0 - n;
        let b = qx * qz;
        d2d = (-(max(t * t - 2.0 * b, 0.0))) * sqrt(0.5) * size;
        d2d = d2d + (n - 1.0) * sqrt(0.5) * size;
    } else {
        let dx = vec2<f32>(qx - 1.0, qz);
        let dz = vec2<f32>(qx, qz - 1.0);
        let d1 = max(qx - 1.0, 0.0);
        let d22 = max(qz - 1.0, 0.0);
        d2d = min(length(dx), min(length(dz), sqrt(d1 * d1 + d22 * d22))) * size;
    }
    let dy = abs(p.y) - h;
    let ww = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
"#;

const HELPER_SDF_PARABOLA_SEGMENT: &str = r#"fn sdf_parabola_segment(p: vec3<f32>, w: f32, ph: f32, hd: f32) -> f32 {
    let px = abs(p.x);
    let py = p.y;
    let ww_sq = w * w;
    let y_arch = ph * (1.0 - px * px / ww_sq);
    var is_in = false;
    if (px <= w && py >= 0.0 && py <= y_arch) { is_in = true; }
    var t = clamp(px, 0.0, w);
    for (var i = 0; i < 8; i = i + 1) {
        let ft = ph * (1.0 - t * t / ww_sq);
        let dft = -2.0 * ph * t / ww_sq;
        let ex = px - t;
        let ey = py - ft;
        let f = -ex + ey * dft;
        let df = 1.0 + dft * dft + ey * (-2.0 * ph / ww_sq);
        if (abs(df) > 1e-10) { t = clamp(t - f / df, 0.0, w); }
    }
    let cy = ph * (1.0 - t * t / ww_sq);
    let d_para = length(vec2<f32>(px - t, py - cy));
    var d_base: f32;
    if (px <= w) { d_base = abs(py); } else { d_base = length(vec2<f32>(px - w, py)); }
    let d_unsigned = min(d_para, d_base);
    var d2d = d_unsigned;
    if (is_in) { d2d = -d_unsigned; }
    let dz = abs(p.z) - hd;
    let ext = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(ext);
}
"#;

const HELPER_SDF_REGULAR_POLYGON: &str = r#"fn sdf_regular_polygon(p: vec3<f32>, radius: f32, n: f32, hh: f32) -> f32 {
    let qx = abs(p.x);
    let qz = p.z;
    let nn = max(n, 3.0);
    let an = 3.14159265358979 / nn;
    let he = radius * cos(an);
    let angle = atan2(qx, qz);
    let bn = an * floor((angle + an) / (2.0 * an));
    let rx = cos(bn) * qx + sin(bn) * qz;
    let d2d = rx - he;
    let dy = abs(p.y) - hh;
    let w = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(w);
}
"#;

const HELPER_SDF_STAR_POLYGON: &str = r#"fn sdf_star_polygon(p: vec3<f32>, radius: f32, np: f32, m: f32, hh: f32) -> f32 {
    let qx = abs(p.x);
    let qz = p.z;
    let n = max(np, 3.0);
    let an = 3.14159265358979 / n;
    let r = length(vec2<f32>(qx, qz));
    var angle = atan2(qx, qz);
    angle = ((angle % (2.0 * an)) + 2.0 * an) % (2.0 * an);
    if (angle > an) { angle = 2.0 * an - angle; }
    let pt = vec2<f32>(r * cos(angle), r * sin(angle));
    let a = vec2<f32>(radius, 0.0);
    let b = vec2<f32>(m * cos(an), m * sin(an));
    let ab = b - a;
    let ap = pt - a;
    let t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    let closest = a + ab * t;
    let dist = length(pt - closest);
    let cross_val = ab.x * ap.y - ab.y * ap.x;
    var d2d = dist;
    if (cross_val > 0.0) { d2d = -dist; }
    let dy = abs(p.y) - hh;
    let w = max(vec2<f32>(d2d, dy), vec2<f32>(0.0));
    return min(max(d2d, dy), 0.0) + length(w);
}
"#;

const HELPER_SDF_STAIRS: &str = r#"fn _stair_box(lx: f32, ly: f32, s: f32, sw: f32, sh: f32) -> f32 {
    let cx = s * sw + sw * 0.5;
    let hy = (s + 1.0) * sh * 0.5;
    let dx = abs(lx - cx) - sw * 0.5;
    let dy = abs(ly - hy) - hy;
    return length(max(vec2<f32>(dx, dy), vec2<f32>(0.0))) + min(max(dx, dy), 0.0);
}
fn sdf_stairs(p: vec3<f32>, sw: f32, sh: f32, ns: f32, hd: f32) -> f32 {
    let n = max(ns, 1.0);
    let tw = n * sw;
    let th = n * sh;
    let lx = p.x + tw * 0.5;
    let ly = p.y + th * 0.5;
    let si = clamp(floor(lx / sw), 0.0, n - 1.0);
    let sj = clamp(ceil(ly / sh) - 1.0, 0.0, n - 1.0);
    var d2d = _stair_box(lx, ly, si, sw, sh);
    if (si > 0.0) { d2d = min(d2d, _stair_box(lx, ly, si - 1.0, sw, sh)); }
    if (si < n - 1.0) { d2d = min(d2d, _stair_box(lx, ly, si + 1.0, sw, sh)); }
    if (sj != si && sj != si - 1.0 && sj != si + 1.0) { d2d = min(d2d, _stair_box(lx, ly, sj, sw, sh)); }
    let dz = abs(p.z) - hd;
    let w = max(vec2<f32>(d2d, dz), vec2<f32>(0.0));
    return min(max(d2d, dz), 0.0) + length(w);
}
"#;

const HELPER_SDF_HELIX: &str = r#"fn sdf_helix(p: vec3<f32>, major_r: f32, minor_r: f32, pitch: f32, hh: f32) -> f32 {
    let r_xz = length(vec2<f32>(p.x, p.z));
    let theta = atan2(p.z, p.x);
    let py = p.y;
    let tau = 6.28318530717959;
    let d_radial = r_xz - major_r;
    let y_at_theta = theta * pitch / tau;
    let k = round((py - y_at_theta) / pitch);
    var d_tube = 1e20;
    for (var dk = -1.0; dk <= 1.0; dk = dk + 1.0) {
        let kk = k + dk;
        let y_helix = y_at_theta + kk * pitch;
        let dy = py - y_helix;
        let d = length(vec2<f32>(d_radial, dy)) - minor_r;
        d_tube = min(d_tube, d);
    }
    let d_cap = abs(py) - hh;
    return max(d_tube, d_cap);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_transpile_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn test_transpile_sphere_dynamic() {
        let sphere = SdfNode::Sphere { radius: 1.5 };
        let shader = WgslShader::transpile(&sphere, TranspileMode::Dynamic);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("sdf_params.data[0].x"));
        assert_eq!(shader.param_layout, vec![1.5]);
        assert_eq!(shader.mode, TranspileMode::Dynamic);
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = WgslShader::transpile(&box3d, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("abs(p)"));
    }

    #[test]
    fn test_transpile_box_dynamic() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let shader = WgslShader::transpile(&box3d, TranspileMode::Dynamic);

        assert!(shader.source.contains("sdf_params.data[0].x")); // 1.0
        assert!(shader.source.contains("sdf_params.data[0].y")); // 2.0
        assert!(shader.source.contains("sdf_params.data[0].z")); // 3.0
        assert_eq!(shader.param_layout, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_transpile_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(2.0, 0.0, 0.0),
        );
        let shader = WgslShader::transpile(&shape, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("min("));
    }

    #[test]
    fn test_transpile_smooth_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Cylinder {
                radius: 0.5,
                half_height: 1.0,
            },
            0.2,
        );
        let shader = WgslShader::transpile(&shape, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("* 5.0")); // inv_k = 1/0.2 = 5.0
        assert!(shader.source.contains("* 0.25"));
        assert_eq!(shader.helper_count, 0);
    }

    #[test]
    fn test_transpile_smooth_union_dynamic() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Cylinder {
                radius: 0.5,
                half_height: 1.0,
            },
            0.2,
        );
        let shader = WgslShader::transpile(&shape, TranspileMode::Dynamic);

        assert!(shader.source.contains("sdf_params.data["));
        // Should have params: sphere_radius, cylinder_radius, cylinder_half_height, k, inv_k
        assert!(shader.param_layout.len() >= 5);
    }

    #[test]
    fn test_transpile_rotate() {
        let shape =
            SdfNode::Sphere { radius: 1.0 }.rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let shader = WgslShader::transpile(&shape, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn quat_rotate"));
        assert!(shader.source.contains("quat_rotate("));
    }

    #[test]
    fn test_compute_shader_generation() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere, TranspileMode::Hardcoded);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("@compute"));
        assert!(compute.contains("@workgroup_size("));
        assert!(compute.contains("input_points"));
        assert!(compute.contains("output_distances"));
        // Hardcoded mode: no sdf_params
        assert!(!compute.contains("sdf_params"));
    }

    #[test]
    fn test_compute_shader_dynamic() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere, TranspileMode::Dynamic);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("@compute"));
        assert!(compute.contains("sdf_params"));
        assert!(compute.contains("@group(0) @binding(3)"));
        assert!(compute.contains("array<vec4<f32>, 1024>"));
    }

    #[test]
    fn test_compute_shader_with_normals() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere, TranspileMode::Dynamic);
        let compute = shader.to_compute_shader_with_normals();

        assert!(compute.contains("estimate_normal"));
        assert!(compute.contains("output[idx].dist"));
        assert!(compute.contains("output[idx].nx"));
        assert!(compute.contains("sdf_params"));
    }

    #[test]
    fn test_extract_params() {
        let shape = SdfNode::Sphere { radius: 1.5 }.translate(1.0, 2.0, 3.0);

        let params = WgslShader::extract_params(&shape);
        // Translate: 3 params (offset.x, y, z), then Sphere: 1 param (radius)
        // Translate is outer, processes first in transpile_node_inner
        assert_eq!(params.len(), 4);
        assert_eq!(params[0], 1.0); // offset.x
        assert_eq!(params[1], 2.0); // offset.y
        assert_eq!(params[2], 3.0); // offset.z
        assert_eq!(params[3], 1.5); // radius
    }

    #[test]
    fn test_extract_params_matches_transpile() {
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Box3d {
                half_extents: Vec3::new(0.5, 0.5, 0.5),
            },
            0.3,
        );

        let shader = WgslShader::transpile(&shape, TranspileMode::Dynamic);
        let extracted = WgslShader::extract_params(&shape);

        assert_eq!(shader.param_layout, extracted);
    }

    #[test]
    fn test_transpile_complex() {
        let shape = SdfNode::Sphere { radius: 1.0 }
            .smooth_union(
                SdfNode::Cylinder {
                    radius: 0.3,
                    half_height: 0.75,
                }
                .rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
                0.2,
            )
            .subtract(SdfNode::Box3d {
                half_extents: Vec3::splat(0.4),
            })
            .translate(0.5, 0.0, 0.0);

        let shader = WgslShader::transpile(&shape, TranspileMode::Hardcoded);

        assert!(shader.source.contains("fn quat_rotate"));
        assert!(shader.source.contains("quat_rotate("));
        assert!(shader.source.contains("* 5.0")); // inv_k = 1/0.2 = 5.0
        assert_eq!(shader.helper_count, 1); // Only quat_rotate helper

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("return"));
    }

    #[test]
    fn test_dynamic_no_constant_fold() {
        // In Dynamic mode, smooth ops should NOT constant-fold even when k ≈ 0
        let shape = SdfNode::Sphere { radius: 1.0 }.smooth_union(
            SdfNode::Sphere { radius: 0.5 },
            0.0, // k = 0, would fold to min() in Hardcoded
        );

        // Hardcoded: should fold to min()
        let hardcoded = WgslShader::transpile(&shape, TranspileMode::Hardcoded);
        assert!(hardcoded.source.contains("min("));
        assert!(!hardcoded.source.contains("0.25")); // no smooth op

        // Dynamic: should NOT fold (k might change later)
        let dynamic = WgslShader::transpile(&shape, TranspileMode::Dynamic);
        assert!(dynamic.source.contains("0.25")); // smooth op present
        assert!(dynamic.source.contains("sdf_params.data["));
    }

    /// Exhaustive test: every SdfNode variant transpiles without panic
    /// and produces valid WGSL containing `sdf_eval`.
    #[test]
    fn test_transpile_all_variants_exhaustive() {
        use glam::Vec2;

        let primitives: Vec<(&str, SdfNode)> = vec![
            ("sphere", SdfNode::sphere(1.0)),
            ("box3d", SdfNode::box3d(1.0, 0.5, 0.5)),
            ("cylinder", SdfNode::cylinder(0.5, 1.0)),
            ("torus", SdfNode::torus(1.0, 0.3)),
            (
                "plane",
                SdfNode::Plane {
                    normal: Vec3::Y,
                    distance: 0.0,
                },
            ),
            ("capsule", SdfNode::capsule(Vec3::ZERO, Vec3::Y, 0.3)),
            ("cone", SdfNode::cone(0.5, 1.0)),
            ("ellipsoid", SdfNode::ellipsoid(1.0, 0.5, 0.3)),
            ("rounded_cone", SdfNode::rounded_cone(0.5, 0.2, 1.0)),
            ("pyramid", SdfNode::pyramid(1.0)),
            ("octahedron", SdfNode::octahedron(1.0)),
            ("hex_prism", SdfNode::hex_prism(0.5, 1.0)),
            ("link", SdfNode::link(0.5, 0.3, 0.1)),
            ("rounded_box", SdfNode::rounded_box(1.0, 0.5, 0.5, 0.1)),
            ("capped_cone", SdfNode::capped_cone(1.0, 0.5, 0.2)),
            ("capped_torus", SdfNode::capped_torus(1.0, 0.3, 1.0)),
            ("rounded_cylinder", SdfNode::rounded_cylinder(0.5, 0.1, 1.0)),
            ("triangular_prism", SdfNode::triangular_prism(0.5, 1.0)),
            ("cut_sphere", SdfNode::cut_sphere(1.0, 0.3)),
            (
                "cut_hollow_sphere",
                SdfNode::cut_hollow_sphere(1.0, 0.3, 0.1),
            ),
            ("death_star", SdfNode::death_star(1.0, 0.8, 0.5)),
            ("solid_angle", SdfNode::solid_angle(0.5, 1.0)),
            ("rhombus", SdfNode::rhombus(0.5, 0.3, 1.0, 0.05)),
            ("horseshoe", SdfNode::horseshoe(1.0, 0.5, 0.3, 0.1, 0.05)),
            ("vesica", SdfNode::vesica(1.0, 0.5)),
            ("infinite_cylinder", SdfNode::infinite_cylinder(0.5)),
            ("infinite_cone", SdfNode::infinite_cone(0.5)),
            ("gyroid", SdfNode::gyroid(1.0, 0.1)),
            ("schwarz_p", SdfNode::schwarz_p(1.0, 0.1)),
            ("heart", SdfNode::heart(1.0)),
            ("tube", SdfNode::tube(0.5, 0.1, 1.0)),
            ("barrel", SdfNode::barrel(0.5, 1.0, 0.2)),
            ("diamond", SdfNode::diamond(0.5, 1.0)),
            ("egg", SdfNode::egg(1.0, 0.5)),
            (
                "superellipsoid",
                SdfNode::superellipsoid(1.0, 0.5, 0.5, 0.5, 0.5),
            ),
            ("rounded_x", SdfNode::rounded_x(0.5, 0.1, 1.0)),
            ("pie", SdfNode::pie(1.0, 0.5, 1.0)),
            ("trapezoid", SdfNode::trapezoid(0.5, 0.3, 1.0, 0.5)),
            ("parallelogram", SdfNode::parallelogram(0.5, 1.0, 0.2, 0.5)),
            ("tunnel", SdfNode::tunnel(0.5, 1.0, 0.5)),
            (
                "uneven_capsule",
                SdfNode::uneven_capsule(0.3, 0.5, 1.0, 0.5),
            ),
            ("arc_shape", SdfNode::arc_shape(1.0, 0.5, 0.1, 1.0)),
            ("moon", SdfNode::moon(0.5, 1.0, 0.8, 1.0)),
            ("cross_shape", SdfNode::cross_shape(0.5, 0.1, 0.05, 1.0)),
            ("blobby_cross", SdfNode::blobby_cross(0.5, 1.0)),
            ("parabola_segment", SdfNode::parabola_segment(0.5, 1.0, 0.5)),
            ("regular_polygon", SdfNode::regular_polygon(0.5, 6, 1.0)),
            ("star_polygon", SdfNode::star_polygon(0.5, 5, 2.0, 1.0)),
            (
                "chamfered_cube",
                SdfNode::chamfered_cube(0.5, 0.5, 0.5, 0.1),
            ),
            ("stairs", SdfNode::stairs(0.3, 0.2, 5, 0.5)),
            ("helix", SdfNode::helix(1.0, 0.1, 0.5, 2.0)),
            ("tetrahedron", SdfNode::tetrahedron(1.0)),
            ("dodecahedron", SdfNode::dodecahedron(1.0)),
            ("icosahedron", SdfNode::icosahedron(1.0)),
            ("truncated_octahedron", SdfNode::truncated_octahedron(1.0)),
            ("truncated_icosahedron", SdfNode::truncated_icosahedron(1.0)),
            ("box_frame", SdfNode::box_frame(Vec3::splat(1.0), 0.1)),
            ("diamond_surface", SdfNode::diamond_surface(1.0, 0.1)),
            ("neovius", SdfNode::neovius(1.0, 0.1)),
            ("lidinoid", SdfNode::lidinoid(1.0, 0.1)),
            ("iwp", SdfNode::iwp(1.0, 0.1)),
            ("frd", SdfNode::frd(1.0, 0.1)),
            ("fischer_koch_s", SdfNode::fischer_koch_s(1.0, 0.1)),
            ("pmy", SdfNode::pmy(1.0, 0.1)),
            ("triangle", SdfNode::triangle(Vec3::X, Vec3::Y, Vec3::Z)),
            (
                "bezier",
                SdfNode::bezier(Vec3::ZERO, Vec3::new(0.5, 1.0, 0.0), Vec3::X, 0.1),
            ),
            ("circle_2d", SdfNode::circle_2d(0.5, 1.0)),
            ("rect_2d", SdfNode::rect_2d(0.5, 0.3, 1.0)),
            (
                "segment_2d",
                SdfNode::segment_2d(0.0, 0.0, 1.0, 1.0, 0.1, 1.0),
            ),
            (
                "polygon_2d",
                SdfNode::polygon_2d(
                    vec![
                        Vec2::new(0.0, 0.0),
                        Vec2::new(1.0, 0.0),
                        Vec2::new(0.5, 1.0),
                    ],
                    1.0,
                ),
            ),
            (
                "rounded_rect_2d",
                SdfNode::rounded_rect_2d(0.5, 0.3, 0.1, 1.0),
            ),
            ("annular_2d", SdfNode::annular_2d(0.5, 0.1, 1.0)),
        ];

        for (name, prim) in &primitives {
            let shader = WgslShader::transpile(prim, TranspileMode::Hardcoded);
            assert!(
                shader.source.contains("fn sdf_eval"),
                "WGSL transpile failed for primitive '{}': missing sdf_eval",
                name,
            );
        }

        // Operations
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(0.5, 0.5, 0.5);
        let operations: Vec<(&str, SdfNode)> = vec![
            ("union", a.clone().union(b.clone())),
            ("intersection", a.clone().intersection(b.clone())),
            ("subtract", a.clone().subtract(b.clone())),
            ("smooth_union", a.clone().smooth_union(b.clone(), 0.2)),
            (
                "smooth_intersection",
                a.clone().smooth_intersection(b.clone(), 0.2),
            ),
            ("smooth_subtract", a.clone().smooth_subtract(b.clone(), 0.2)),
            ("chamfer_union", a.clone().chamfer_union(b.clone(), 0.2)),
            (
                "chamfer_intersection",
                a.clone().chamfer_intersection(b.clone(), 0.2),
            ),
            (
                "chamfer_subtract",
                a.clone().chamfer_subtract(b.clone(), 0.2),
            ),
            ("stairs_union", a.clone().stairs_union(b.clone(), 0.2, 4.0)),
            (
                "stairs_intersection",
                a.clone().stairs_intersection(b.clone(), 0.2, 4.0),
            ),
            (
                "stairs_subtract",
                a.clone().stairs_subtract(b.clone(), 0.2, 4.0),
            ),
            (
                "columns_union",
                a.clone().columns_union(b.clone(), 0.2, 4.0),
            ),
            (
                "columns_intersection",
                a.clone().columns_intersection(b.clone(), 0.2, 4.0),
            ),
            (
                "columns_subtract",
                a.clone().columns_subtract(b.clone(), 0.2, 4.0),
            ),
            ("xor", a.clone().xor(b.clone())),
            ("morph", a.clone().morph(b.clone(), 0.5)),
            ("pipe", a.clone().pipe(b.clone(), 0.2)),
            ("engrave", a.clone().engrave(b.clone(), 0.1)),
            ("groove", a.clone().groove(b.clone(), 0.2, 0.1)),
            ("tongue", a.clone().tongue(b.clone(), 0.2, 0.1)),
            (
                "exp_smooth_union",
                a.clone().exp_smooth_union(b.clone(), 0.2),
            ),
            (
                "exp_smooth_intersection",
                a.clone().exp_smooth_intersection(b.clone(), 0.2),
            ),
            (
                "exp_smooth_subtract",
                a.clone().exp_smooth_subtract(b.clone(), 0.2),
            ),
        ];

        for (name, op) in &operations {
            let shader = WgslShader::transpile(op, TranspileMode::Hardcoded);
            assert!(
                shader.source.contains("fn sdf_eval"),
                "WGSL transpile failed for operation '{}': missing sdf_eval",
                name,
            );
        }

        // Transforms & Modifiers
        let s = SdfNode::sphere(1.0);
        let modifiers: Vec<(&str, SdfNode)> = vec![
            ("translate", s.clone().translate(1.0, 0.0, 0.0)),
            ("rotate", s.clone().rotate_euler(0.5, 0.0, 0.0)),
            ("scale", s.clone().scale(2.0)),
            ("scale_xyz", s.clone().scale_xyz(1.0, 2.0, 0.5)),
            ("twist", s.clone().twist(1.0)),
            ("bend", s.clone().bend(0.5)),
            ("round", s.clone().round(0.1)),
            ("onion", s.clone().onion(0.1)),
            ("elongate", s.clone().elongate(0.5, 0.0, 0.0)),
            ("repeat_infinite", s.clone().repeat_infinite(3.0, 3.0, 3.0)),
            ("mirror", s.clone().mirror(true, false, false)),
            ("octant_mirror", s.clone().octant_mirror()),
            ("revolution", s.clone().revolution(0.5)),
            ("extrude", s.clone().extrude(0.5)),
            ("noise", s.clone().noise(0.1, 2.0, 42)),
            ("taper", s.clone().taper(0.5)),
            ("displacement", s.clone().displacement(0.1)),
            ("polar_repeat", s.clone().polar_repeat(6)),
            ("shear", s.clone().shear(0.1, 0.0, 0.0)),
            ("animated", s.clone().animated(1.0, 0.5)),
            ("with_material", s.clone().with_material(1)),
        ];

        for (name, m) in &modifiers {
            let shader = WgslShader::transpile(m, TranspileMode::Hardcoded);
            assert!(
                shader.source.contains("fn sdf_eval"),
                "WGSL transpile failed for modifier '{}': missing sdf_eval",
                name,
            );
        }
    }
}

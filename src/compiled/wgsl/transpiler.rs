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

use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
const FOLD_EPSILON: f32 = 1e-6;

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

    /// Emit inline WGSL code for stairs_union(a, b, r, n) → result stored in `out_var`.
    /// Mercury hg_sdf fOpUnionStairs formula, fully inlined.
    fn emit_stairs_union_inline(
        &mut self, code: &mut String,
        d_a: &str, d_b: &str, r_s: &str, n_s: &str, out_var: &str,
    ) {
        let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
        let s2_str = self.param(std::f32::consts::SQRT_2);
        let rn = self.next_var();
        let off = self.next_var();
        let step = self.next_var();
        let px = self.next_var();
        let py = self.next_var();
        let px2 = self.next_var();
        let t = self.next_var();
        let px3 = self.next_var();
        let d2 = self.next_var();
        let npx = self.next_var();
        let npy = self.next_var();
        let edge = self.next_var();

        writeln!(code, "    let {} = {} / {};", rn, r_s, n_s).unwrap();
        writeln!(code, "    let {} = ({} - {}) * 0.5 * {};", off, r_s, rn, s2_str).unwrap();
        writeln!(code, "    let {} = {} * {} / {};", step, r_s, s2_str, n_s).unwrap();
        writeln!(code, "    let {} = ({} - {}) * {} - {};", px, d_b, d_a, s_str, off).unwrap();
        writeln!(code, "    let {} = ({} + {}) * {} - {};", py, d_a, d_b, s_str, off).unwrap();
        writeln!(code, "    let {} = {} + 0.5 * {} * {};", px2, px, s2_str, rn).unwrap();
        writeln!(code, "    let {} = {} + {} * 0.5;", t, px2, step).unwrap();
        writeln!(code, "    let {} = {} - {} * floor({} / {}) - {} * 0.5;", px3, t, step, t, step, step).unwrap();
        writeln!(code, "    let {} = min(min({}, {}), {});", d2, d_a, d_b, py).unwrap();
        writeln!(code, "    let {} = ({} + {}) * {};", npx, px3, py, s_str).unwrap();
        writeln!(code, "    let {} = ({} - {}) * {};", npy, py, px3, s_str).unwrap();
        writeln!(code, "    let {} = 0.5 * {};", edge, rn).unwrap();
        writeln!(code, "    let {} = min({}, max({} - {}, {} - {}));", out_var, d2, npx, edge, npy, edge).unwrap();
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

    fn transpile_node(&mut self, node: &SdfNode, point_var: &str) -> String {
        let mut code = String::new();
        let result_var = self.transpile_node_inner(node, point_var, &mut code);
        writeln!(code, "    return {};", result_var).unwrap();
        code
    }

    fn transpile_node_inner(
        &mut self,
        node: &SdfNode,
        point_var: &str,
        code: &mut String,
    ) -> String {
        match node {
            // ============ Primitives ============

            SdfNode::Sphere { radius } => {
                let var = self.next_var();
                let r = self.param(*radius);
                writeln!(code, "    let {} = length({}) - {};", var, point_var, r).unwrap();
                var
            }

            SdfNode::Box3d { half_extents } => {
                let q_var = self.next_var();
                let var = self.next_var();
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                writeln!(
                    code,
                    "    let {} = abs({}) - vec3<f32>({}, {}, {});",
                    q_var, point_var, hx, hy, hz
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = length(max({}, vec3<f32>(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);",
                    var, q_var, q_var, q_var, q_var
                )
                .unwrap();
                var
            }

            SdfNode::Cylinder { radius, half_height } => {
                let d_var = self.next_var();
                let var = self.next_var();
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                writeln!(
                    code,
                    "    let {} = vec2<f32>(length({}.xz) - {}, abs({}.y) - {});",
                    d_var, point_var, r, point_var, hh
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = min(max({}.x, {}.y), 0.0) + length(max({}, vec2<f32>(0.0)));",
                    var, d_var, d_var, d_var
                )
                .unwrap();
                var
            }

            SdfNode::Torus {
                major_radius,
                minor_radius,
            } => {
                let q_var = self.next_var();
                let var = self.next_var();
                let mr = self.param(*major_radius);
                let mnr = self.param(*minor_radius);
                writeln!(
                    code,
                    "    let {} = vec2<f32>(length({}.xz) - {}, {}.y);",
                    q_var, point_var, mr, point_var
                )
                .unwrap();
                writeln!(code, "    let {} = length({}) - {};", var, q_var, mnr).unwrap();
                var
            }

            SdfNode::Plane { normal, distance } => {
                let var = self.next_var();
                let nx = self.param(normal.x);
                let ny = self.param(normal.y);
                let nz = self.param(normal.z);
                let d = self.param(*distance);
                writeln!(
                    code,
                    "    let {} = dot({}, vec3<f32>({}, {}, {})) + {};",
                    var, point_var, nx, ny, nz, d
                )
                .unwrap();
                var
            }

            SdfNode::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let pa_var = self.next_var();
                let ba_var = self.next_var();
                let h_var = self.next_var();
                let var = self.next_var();

                let pax = self.param(point_a.x);
                let pay = self.param(point_a.y);
                let paz = self.param(point_a.z);
                let pbx = self.param(point_b.x);
                let pby = self.param(point_b.y);
                let pbz = self.param(point_b.z);
                let r = self.param(*radius);

                writeln!(
                    code,
                    "    let {} = {} - vec3<f32>({}, {}, {});",
                    pa_var, point_var, pax, pay, paz
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = vec3<f32>({}, {}, {}) - vec3<f32>({}, {}, {});",
                    ba_var, pbx, pby, pbz, pax, pay, paz
                )
                .unwrap();
                // ★ Deep Fried: Zero-safe guard for degenerate capsule (a == b)
                writeln!(
                    code,
                    "    let {} = clamp(dot({}, {}) / max(dot({}, {}), 1e-10), 0.0, 1.0);",
                    h_var, pa_var, ba_var, ba_var, ba_var
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = length({} - {} * {}) - {};",
                    var, pa_var, ba_var, h_var, r
                )
                .unwrap();
                var
            }

            SdfNode::Cone { radius, half_height } => {
                let qx_var = self.next_var();
                let h_var = self.next_var();
                let k2x = -radius;
                let k2y = 2.0 * half_height;
                let ca_var = self.next_var();
                let t_var = self.next_var();
                let cb_var = self.next_var();
                let s_var = self.next_var();
                let d2_var = self.next_var();
                let var = self.next_var();

                let p_hh = self.param(*half_height);
                let p_r = self.param(*radius);
                let p_k2x = self.param(k2x);
                let p_k2y = self.param(k2y);
                let p_k2sq = self.param(k2x * k2x + k2y * k2y);

                writeln!(code, "    let {} = length({}.xz);", qx_var, point_var).unwrap();
                writeln!(code, "    let {} = {};", h_var, p_hh).unwrap();
                writeln!(code,
                    "    let {} = vec2<f32>({} - min({}, select(0.0, {}, {}.y < 0.0)), abs({}.y) - {});",
                    ca_var, qx_var, qx_var, p_r, point_var, point_var, h_var
                ).unwrap();
                writeln!(code,
                    "    let {} = clamp((-{} * {} + ({} - {}.y) * {}) / {}, 0.0, 1.0);",
                    t_var, qx_var, p_k2x, h_var, point_var, p_k2y, p_k2sq
                ).unwrap();
                writeln!(code,
                    "    let {} = vec2<f32>({} + {} * {}, {}.y - {} + {} * {});",
                    cb_var, qx_var, p_k2x, t_var, point_var, h_var, p_k2y, t_var
                ).unwrap();
                writeln!(code,
                    "    let {} = select(1.0, -1.0, {}.x < 0.0 && {}.y < 0.0);",
                    s_var, cb_var, ca_var
                ).unwrap();
                writeln!(code,
                    "    let {} = min(dot({}, {}), dot({}, {}));",
                    d2_var, ca_var, ca_var, cb_var, cb_var
                ).unwrap();
                writeln!(code,
                    "    let {} = {} * sqrt({});",
                    var, s_var, d2_var
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for Ellipsoid
            SdfNode::Ellipsoid { radii } => {
                let var = self.next_var();
                let k0_var = self.next_var();
                let k1_var = self.next_var();
                let inv_rx = self.param(1.0 / radii.x.max(1e-10));
                let inv_ry = self.param(1.0 / radii.y.max(1e-10));
                let inv_rz = self.param(1.0 / radii.z.max(1e-10));
                let inv_rx2 = self.param(1.0 / (radii.x * radii.x).max(1e-10));
                let inv_ry2 = self.param(1.0 / (radii.y * radii.y).max(1e-10));
                let inv_rz2 = self.param(1.0 / (radii.z * radii.z).max(1e-10));
                writeln!(code,
                    "    let {} = length({} * vec3<f32>({}, {}, {}));",
                    k0_var, point_var, inv_rx, inv_ry, inv_rz
                ).unwrap();
                writeln!(code,
                    "    let {} = length({} * vec3<f32>({}, {}, {}));",
                    k1_var, point_var, inv_rx2, inv_ry2, inv_rz2
                ).unwrap();
                writeln!(code,
                    "    let {} = {} * ({} - 1.0) / max({}, 1e-10);",
                    var, k0_var, k0_var, k1_var
                ).unwrap();
                var
            }

            SdfNode::RoundedCone { r1, r2, half_height } => {
                self.ensure_helper("sdf_rounded_cone");
                let var = self.next_var();
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let p_hh = self.param(*half_height);
                writeln!(code,
                    "    let {} = sdf_rounded_cone({}, {}, {}, {});",
                    var, point_var, p_r1, p_r2, p_hh
                ).unwrap();
                var
            }

            SdfNode::Pyramid { half_height } => {
                self.ensure_helper("sdf_pyramid");
                let var = self.next_var();
                let p_hh = self.param(*half_height);
                writeln!(code,
                    "    let {} = sdf_pyramid({}, {});",
                    var, point_var, p_hh
                ).unwrap();
                var
            }

            SdfNode::Octahedron { size } => {
                self.ensure_helper("sdf_octahedron");
                let var = self.next_var();
                let p_s = self.param(*size);
                writeln!(code,
                    "    let {} = sdf_octahedron({}, {});",
                    var, point_var, p_s
                ).unwrap();
                var
            }

            SdfNode::HexPrism { hex_radius, half_height } => {
                self.ensure_helper("sdf_hex_prism");
                let var = self.next_var();
                let p_hr = self.param(*hex_radius);
                let p_hh = self.param(*half_height);
                writeln!(code,
                    "    let {} = sdf_hex_prism({}, {}, {});",
                    var, point_var, p_hr, p_hh
                ).unwrap();
                var
            }

            SdfNode::Link { half_length, r1, r2 } => {
                self.ensure_helper("sdf_link");
                let var = self.next_var();
                let p_hl = self.param(*half_length);
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                writeln!(code,
                    "    let {} = sdf_link({}, {}, {}, {});",
                    var, point_var, p_hl, p_r1, p_r2
                ).unwrap();
                var
            }

            SdfNode::Triangle { point_a, point_b, point_c } => {
                self.ensure_helper("sdf_triangle");
                let var = self.next_var();
                let ax = self.param(point_a.x);
                let ay = self.param(point_a.y);
                let az = self.param(point_a.z);
                let bx = self.param(point_b.x);
                let by = self.param(point_b.y);
                let bz = self.param(point_b.z);
                let cx = self.param(point_c.x);
                let cy = self.param(point_c.y);
                let cz = self.param(point_c.z);
                writeln!(code,
                    "    let {} = sdf_triangle({}, vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}));",
                    var, point_var, ax, ay, az, bx, by, bz, cx, cy, cz
                ).unwrap();
                var
            }

            SdfNode::Bezier { point_a, point_b, point_c, radius } => {
                self.ensure_helper("sdf_bezier");
                let var = self.next_var();
                let ax = self.param(point_a.x);
                let ay = self.param(point_a.y);
                let az = self.param(point_a.z);
                let bx = self.param(point_b.x);
                let by = self.param(point_b.y);
                let bz = self.param(point_b.z);
                let cx = self.param(point_c.x);
                let cy = self.param(point_c.y);
                let cz = self.param(point_c.z);
                let r = self.param(*radius);
                writeln!(code,
                    "    let {} = sdf_bezier({}, vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}), {});",
                    var, point_var, ax, ay, az, bx, by, bz, cx, cy, cz, r
                ).unwrap();
                var
            }

            // --- New Primitives (16) ---

            SdfNode::RoundedBox { half_extents, round_radius } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let rr = self.param(*round_radius);
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = abs({}) - vec3<f32>({}, {}, {});", q_var, point_var, hx, hy, hz).unwrap();
                writeln!(code, "    let {} = length(max({}, vec3<f32>(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0) - {};", var, q_var, q_var, q_var, q_var, rr).unwrap();
                var
            }

            SdfNode::CappedCone { half_height, r1, r2 } => {
                self.ensure_helper("sdf_capped_cone");
                let h = self.param(*half_height);
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_capped_cone({}, {}, {}, {});", var, point_var, h, p_r1, p_r2).unwrap();
                var
            }

            SdfNode::CappedTorus { major_radius, minor_radius, cap_angle } => {
                self.ensure_helper("sdf_capped_torus");
                let ra = self.param(*major_radius);
                let rb = self.param(*minor_radius);
                let an = self.param(*cap_angle);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_capped_torus({}, {}, {}, {});", var, point_var, ra, rb, an).unwrap();
                var
            }

            SdfNode::RoundedCylinder { radius, round_radius, half_height } => {
                self.ensure_helper("sdf_rounded_cylinder");
                let r = self.param(*radius);
                let rr = self.param(*round_radius);
                let h = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_rounded_cylinder({}, {}, {}, {});", var, point_var, r, rr, h).unwrap();
                var
            }

            SdfNode::TriangularPrism { width, half_depth } => {
                self.ensure_helper("sdf_triangular_prism");
                let w = self.param(*width);
                let d = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_triangular_prism({}, {}, {});", var, point_var, w, d).unwrap();
                var
            }

            SdfNode::CutSphere { radius, cut_height } => {
                self.ensure_helper("sdf_cut_sphere");
                let r = self.param(*radius);
                let h = self.param(*cut_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_cut_sphere({}, {}, {});", var, point_var, r, h).unwrap();
                var
            }

            SdfNode::CutHollowSphere { radius, cut_height, thickness } => {
                self.ensure_helper("sdf_cut_hollow_sphere");
                let r = self.param(*radius);
                let h = self.param(*cut_height);
                let t = self.param(*thickness);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_cut_hollow_sphere({}, {}, {}, {});", var, point_var, r, h, t).unwrap();
                var
            }

            SdfNode::DeathStar { ra, rb, d } => {
                self.ensure_helper("sdf_death_star");
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                let p_d = self.param(*d);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_death_star({}, {}, {}, {});", var, point_var, p_ra, p_rb, p_d).unwrap();
                var
            }

            SdfNode::SolidAngle { angle, radius } => {
                self.ensure_helper("sdf_solid_angle");
                let an = self.param(*angle);
                let r = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_solid_angle({}, {}, {});", var, point_var, an, r).unwrap();
                var
            }

            SdfNode::Rhombus { la, lb, half_height, round_radius } => {
                self.ensure_helper("sdf_rhombus");
                let p_la = self.param(*la);
                let p_lb = self.param(*lb);
                let h = self.param(*half_height);
                let rr = self.param(*round_radius);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_rhombus({}, {}, {}, {}, {});", var, point_var, p_la, p_lb, h, rr).unwrap();
                var
            }

            SdfNode::Horseshoe { angle, radius, half_length, width, thickness } => {
                self.ensure_helper("sdf_horseshoe");
                let an = self.param(*angle);
                let r = self.param(*radius);
                let hl = self.param(*half_length);
                let w = self.param(*width);
                let t = self.param(*thickness);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_horseshoe({}, {}, {}, {}, {}, {});", var, point_var, an, r, hl, w, t).unwrap();
                var
            }

            SdfNode::Vesica { radius, half_dist } => {
                self.ensure_helper("sdf_vesica");
                let r = self.param(*radius);
                let d = self.param(*half_dist);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_vesica({}, {}, {});", var, point_var, r, d).unwrap();
                var
            }

            SdfNode::InfiniteCylinder { radius } => {
                let r = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    let {} = length({}.xz) - {};", var, point_var, r).unwrap();
                var
            }

            SdfNode::InfiniteCone { angle } => {
                self.ensure_helper("sdf_infinite_cone");
                let an = self.param(*angle);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_infinite_cone({}, {});", var, point_var, an).unwrap();
                var
            }

            SdfNode::Gyroid { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp_var = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp_var, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(sin({}.x) * cos({}.y) + sin({}.y) * cos({}.z) + sin({}.z) * cos({}.x)) / {} - {};", var, sp_var, sp_var, sp_var, sp_var, sp_var, sp_var, sc, th).unwrap();
                var
            }

            SdfNode::Heart { size } => {
                self.ensure_helper("sdf_heart");
                let s = self.param(*size);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_heart({}, {});", var, point_var, s).unwrap();
                var
            }

            SdfNode::Tube { outer_radius, thickness, half_height } => {
                self.ensure_helper("sdf_tube");
                let or = self.param(*outer_radius);
                let th = self.param(*thickness);
                let hh = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_tube({}, {}, {}, {});", var, point_var, or, th, hh).unwrap();
                var
            }

            SdfNode::Barrel { radius, half_height, bulge } => {
                self.ensure_helper("sdf_barrel");
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                let b = self.param(*bulge);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_barrel({}, {}, {}, {});", var, point_var, r, hh, b).unwrap();
                var
            }

            SdfNode::Diamond { radius, half_height } => {
                self.ensure_helper("sdf_diamond");
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_diamond({}, {}, {});", var, point_var, r, hh).unwrap();
                var
            }

            SdfNode::ChamferedCube { half_extents, chamfer } => {
                self.ensure_helper("sdf_chamfered_cube");
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let ch = self.param(*chamfer);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_chamfered_cube({}, {}, {}, {}, {});", var, point_var, hx, hy, hz, ch).unwrap();
                var
            }

            SdfNode::SchwarzP { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp_var = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp_var, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(cos({}.x) + cos({}.y) + cos({}.z)) / {} - {};", var, sp_var, sp_var, sp_var, sc, th).unwrap();
                var
            }

            SdfNode::Superellipsoid { half_extents, e1, e2 } => {
                self.ensure_helper("sdf_superellipsoid");
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let e1_s = self.param(*e1);
                let e2_s = self.param(*e2);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_superellipsoid({}, {}, {}, {}, {}, {});", var, point_var, hx, hy, hz, e1_s, e2_s).unwrap();
                var
            }

            SdfNode::RoundedX { width, round_radius, half_height } => {
                self.ensure_helper("sdf_rounded_x");
                let w = self.param(*width);
                let r = self.param(*round_radius);
                let hh = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_rounded_x({}, {}, {}, {});", var, point_var, w, r, hh).unwrap();
                var
            }

            SdfNode::Pie { angle, radius, half_height } => {
                self.ensure_helper("sdf_pie");
                let a = self.param(*angle);
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_pie({}, {}, {}, {});", var, point_var, a, r, hh).unwrap();
                var
            }

            SdfNode::Trapezoid { r1, r2, trap_height, half_depth } => {
                self.ensure_helper("sdf_trapezoid");
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let th = self.param(*trap_height);
                let hd = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_trapezoid({}, {}, {}, {}, {});", var, point_var, p_r1, p_r2, th, hd).unwrap();
                var
            }

            SdfNode::Parallelogram { width, para_height, skew, half_depth } => {
                self.ensure_helper("sdf_parallelogram");
                let w = self.param(*width);
                let ph = self.param(*para_height);
                let sk = self.param(*skew);
                let hd = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_parallelogram({}, {}, {}, {}, {});", var, point_var, w, ph, sk, hd).unwrap();
                var
            }

            SdfNode::Tunnel { width, height_2d, half_depth } => {
                self.ensure_helper("sdf_tunnel");
                let w = self.param(*width);
                let h2d = self.param(*height_2d);
                let hd = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_tunnel({}, {}, {}, {});", var, point_var, w, h2d, hd).unwrap();
                var
            }

            SdfNode::UnevenCapsule { r1, r2, cap_height, half_depth } => {
                self.ensure_helper("sdf_uneven_capsule");
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let ch = self.param(*cap_height);
                let hd = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_uneven_capsule({}, {}, {}, {}, {});", var, point_var, p_r1, p_r2, ch, hd).unwrap();
                var
            }

            SdfNode::Egg { ra, rb } => {
                self.ensure_helper("sdf_egg");
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_egg({}, {}, {});", var, point_var, p_ra, p_rb).unwrap();
                var
            }

            SdfNode::ArcShape { aperture, radius, thickness, half_height } => {
                self.ensure_helper("sdf_arc_shape");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_arc_shape({}, {}, {}, {}, {});", var, point_var,
                    self.param(*aperture), self.param(*radius), self.param(*thickness), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::Moon { d, ra, rb, half_height } => {
                self.ensure_helper("sdf_moon");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_moon({}, {}, {}, {}, {});", var, point_var,
                    self.param(*d), self.param(*ra), self.param(*rb), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::CrossShape { length, thickness, round_radius, half_height } => {
                self.ensure_helper("sdf_cross_shape");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_cross_shape({}, {}, {}, {}, {});", var, point_var,
                    self.param(*length), self.param(*thickness), self.param(*round_radius), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::BlobbyCross { size, half_height } => {
                self.ensure_helper("sdf_blobby_cross");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_blobby_cross({}, {}, {});", var, point_var,
                    self.param(*size), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::ParabolaSegment { width, para_height, half_depth } => {
                self.ensure_helper("sdf_parabola_segment");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_parabola_segment({}, {}, {}, {});", var, point_var,
                    self.param(*width), self.param(*para_height), self.param(*half_depth)).unwrap();
                var
            }

            SdfNode::RegularPolygon { radius, n_sides, half_height } => {
                self.ensure_helper("sdf_regular_polygon");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_regular_polygon({}, {}, {}, {});", var, point_var,
                    self.param(*radius), self.param(*n_sides), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::StarPolygon { radius, n_points, m, half_height } => {
                self.ensure_helper("sdf_star_polygon");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_star_polygon({}, {}, {}, {}, {});", var, point_var,
                    self.param(*radius), self.param(*n_points), self.param(*m), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::Stairs { step_width, step_height, n_steps, half_depth } => {
                self.ensure_helper("sdf_stairs");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_stairs({}, {}, {}, {}, {});", var, point_var,
                    self.param(*step_width), self.param(*step_height), self.param(*n_steps), self.param(*half_depth)).unwrap();
                var
            }

            SdfNode::Helix { major_r, minor_r, pitch, half_height } => {
                self.ensure_helper("sdf_helix");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_helix({}, {}, {}, {}, {});", var, point_var,
                    self.param(*major_r), self.param(*minor_r), self.param(*pitch), self.param(*half_height)).unwrap();
                var
            }

            SdfNode::Tetrahedron { size } => {
                self.ensure_helper("sdf_tetrahedron");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_tetrahedron({}, {});", var, point_var, self.param(*size)).unwrap();
                var
            }

            SdfNode::Dodecahedron { radius } => {
                self.ensure_helper("sdf_dodecahedron");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_dodecahedron({}, {});", var, point_var, self.param(*radius)).unwrap();
                var
            }

            SdfNode::Icosahedron { radius } => {
                self.ensure_helper("sdf_icosahedron");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_icosahedron({}, {});", var, point_var, self.param(*radius)).unwrap();
                var
            }

            SdfNode::TruncatedOctahedron { radius } => {
                self.ensure_helper("sdf_truncated_octahedron");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_truncated_octahedron({}, {});", var, point_var, self.param(*radius)).unwrap();
                var
            }

            SdfNode::TruncatedIcosahedron { radius } => {
                self.ensure_helper("sdf_truncated_icosahedron");
                let var = self.next_var();
                writeln!(code, "    let {} = sdf_truncated_icosahedron({}, {});", var, point_var, self.param(*radius)).unwrap();
                var
            }

            SdfNode::BoxFrame { half_extents, edge } => {
                let bx = self.param(half_extents.x);
                let by = self.param(half_extents.y);
                let bz = self.param(half_extents.z);
                let e = self.param(*edge);
                let pv = self.next_var();
                let qv = self.next_var();
                let d1 = self.next_var();
                let d2 = self.next_var();
                let d3 = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = abs({}) - vec3<f32>({}, {}, {});", pv, point_var, bx, by, bz).unwrap();
                writeln!(code, "    let {} = abs({} + {}) - {};", qv, pv, e, e).unwrap();
                writeln!(code, "    let {} = length(max(vec3<f32>({}.x, {}.y, {}.z), vec3<f32>(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);", d1, pv, qv, qv, pv, qv, qv).unwrap();
                writeln!(code, "    let {} = length(max(vec3<f32>({}.x, {}.y, {}.z), vec3<f32>(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);", d2, qv, pv, qv, qv, pv, qv).unwrap();
                writeln!(code, "    let {} = length(max(vec3<f32>({}.x, {}.y, {}.z), vec3<f32>(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);", d3, qv, qv, pv, qv, qv, pv).unwrap();
                writeln!(code, "    let {} = min({}, min({}, {}));", var, d1, d2, d3).unwrap();
                var
            }

            SdfNode::DiamondSurface { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(sin({}.x)*sin({}.y)*sin({}.z) + sin({}.x)*cos({}.y)*cos({}.z) + cos({}.x)*sin({}.y)*cos({}.z) + cos({}.x)*cos({}.y)*sin({}.z)) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::Neovius { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(3.0*(cos({}.x)+cos({}.y)+cos({}.z)) + 4.0*cos({}.x)*cos({}.y)*cos({}.z)) / {} - {};", var, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::Lidinoid { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(0.5*(sin(2.0*{}.x)*cos({}.y)*sin({}.z)+sin({}.x)*sin(2.0*{}.y)*cos({}.z)+cos({}.x)*sin({}.y)*sin(2.0*{}.z)) - 0.5*(cos(2.0*{}.x)*cos(2.0*{}.y)+cos(2.0*{}.y)*cos(2.0*{}.z)+cos(2.0*{}.z)*cos(2.0*{}.x)) + 0.15) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::IWP { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(2.0*(cos({}.x)*cos({}.y)+cos({}.y)*cos({}.z)+cos({}.z)*cos({}.x)) - (cos(2.0*{}.x)+cos(2.0*{}.y)+cos(2.0*{}.z))) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::FRD { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(cos(2.0*{}.x)*sin({}.y)*cos({}.z) + cos({}.x)*cos(2.0*{}.y)*sin({}.z) + sin({}.x)*cos({}.y)*cos(2.0*{}.z)) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::FischerKochS { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(cos(2.0*{}.x)*sin({}.y)*cos({}.z) + cos({}.x)*cos(2.0*{}.y)*sin({}.z) + sin({}.x)*cos({}.y)*cos(2.0*{}.z) - 0.4) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            SdfNode::PMY { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp = self.next_var();
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", sp, point_var, sc).unwrap();
                writeln!(code, "    let {} = abs(2.0*cos({}.x)*cos({}.y)*cos({}.z) + sin(2.0*{}.x)*sin({}.y) + sin({}.x)*sin(2.0*{}.z) + sin(2.0*{}.y)*sin({}.z)) / {} - {};", var, sp, sp, sp, sp, sp, sp, sp, sp, sp, sc, th).unwrap();
                var
            }

            // ============ Operations ============

            SdfNode::Union { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    let {} = min({}, {});", var, d_a, d_b).unwrap();
                var
            }

            SdfNode::Intersection { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    let {} = max({}, {});", var, d_a, d_b).unwrap();
                var
            }

            SdfNode::Subtraction { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    let {} = max({}, -{});", var, d_a, d_b).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothUnion
            // In Dynamic mode, never constant-fold (k might change at runtime)
            SdfNode::SmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                // Constant folding: only in Hardcoded mode
                if self.mode == TranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    let {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    let {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_str, d_a, d_b, inv_k_str
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = min({}, {}) - {} * {} * {} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k_str
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothIntersection
            SdfNode::SmoothIntersection { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    let {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_str, d_a, d_b, inv_k_str
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = max({}, {}) + {} * {} * {} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k_str
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothSubtraction
            SdfNode::SmoothSubtraction { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                let k_str = self.param(*k);
                let inv_k_str = self.param(1.0 / k);
                let h_var = self.next_var();
                let neg_b = self.next_var();

                writeln!(code, "    let {} = -{};", neg_b, d_b).unwrap();
                writeln!(
                    code,
                    "    let {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_str, d_a, neg_b, inv_k_str
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = max({}, {}) + {} * {} * {} * 0.25;",
                    var, d_a, neg_b, h_var, h_var, k_str
                ).unwrap();
                var
            }

            // ★ Chamfer blends: 45-degree beveled edge
            SdfNode::ChamferUnion { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);

                writeln!(
                    code,
                    "    let {} = min(min({}, {}), ({} + {}) * {} - {});",
                    var, d_a, d_b, d_a, d_b, s_str, r_str
                ).unwrap();
                var
            }

            SdfNode::ChamferIntersection { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);

                writeln!(
                    code,
                    "    let {} = max(max({}, {}), ({} + {}) * {} + {});",
                    var, d_a, d_b, d_a, d_b, s_str, r_str
                ).unwrap();
                var
            }

            SdfNode::ChamferSubtraction { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                let r_str = self.param(*r);
                let s_str = self.param(std::f32::consts::FRAC_1_SQRT_2);
                let neg_b = self.next_var();

                writeln!(code, "    let {} = -{};", neg_b, d_b).unwrap();
                writeln!(
                    code,
                    "    let {} = max(max({}, {}), ({} + {}) * {} + {});",
                    var, d_a, neg_b, d_a, neg_b, s_str, r_str
                ).unwrap();
                var
            }

            // ★ Stairs blends: stepped/terraced edge (Mercury hg_sdf)
            SdfNode::StairsUnion { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let r_s = self.param(*r);
                let n_s = self.param(*n);
                self.emit_stairs_union_inline(code, &d_a, &d_b, &r_s, &n_s, &var);
                var
            }

            SdfNode::StairsIntersection { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                // stairs_intersection(a,b,r,n) = -stairs_union(-a, -b, r, n)
                let na = self.next_var();
                let nb = self.next_var();
                writeln!(code, "    let {} = -{};", na, d_a).unwrap();
                writeln!(code, "    let {} = -{};", nb, d_b).unwrap();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let su = self.next_var();
                self.emit_stairs_union_inline(code, &na, &nb, &r_s, &n_s, &su);
                writeln!(code, "    let {} = -{};", var, su).unwrap();
                var
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == TranspileMode::Hardcoded && *r < FOLD_EPSILON {
                    writeln!(code, "    let {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                // stairs_subtraction(a,b,r,n) = -stairs_union(-a, b, r, n)
                let na = self.next_var();
                writeln!(code, "    let {} = -{};", na, d_a).unwrap();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                let su = self.next_var();
                self.emit_stairs_union_inline(code, &na, &d_b, &r_s, &n_s, &su);
                writeln!(code, "    let {} = -{};", var, su).unwrap();
                var
            }

            SdfNode::XOR { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    let {} = max(min({}, {}), -max({}, {}));", var, d_a, d_b, d_a, d_b).unwrap();
                var
            }

            SdfNode::Morph { a, b, t } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let t_s = self.param(*t);
                writeln!(code, "    let {} = mix({}, {}, {});", var, d_a, d_b, t_s).unwrap();
                var
            }

            SdfNode::ColumnsUnion { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                writeln!(code, "    var {v}_m = min({a}, {b});", v=var, a=d_a, b=d_b).unwrap();
                writeln!(code, "    var {v}_a2 = min({a}, {b}); var {v}_b2 = max({a}, {b});", v=var, a=d_a, b=d_b).unwrap();
                writeln!(code, "    let {v}_cs = {r} * 2.0 / {n};", v=var, r=r_s, n=n_s).unwrap();
                writeln!(code, "    var {v}_ra = 0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678;", v=var, r=r_s).unwrap();
                writeln!(code, "    let {v}_rb = 0.70710678 * ({v}_b2 - {v}_a2);", v=var).unwrap();
                writeln!(code, "    {v}_ra = ({v}_ra + {v}_cs * 0.5) % {v}_cs - {v}_cs * 0.5;", v=var).unwrap();
                writeln!(code, "    let {} = select(min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m), {v}_m, {v}_m > {r});", var, v=var, r=r_s).unwrap();
                var
            }

            SdfNode::ColumnsIntersection { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                writeln!(code, "    let {v}_na = -({a}); var {v}_m = min({v}_na, {b});", v=var, a=d_a, b=d_b).unwrap();
                writeln!(code, "    var {v}_a2 = min({v}_na, {b}); var {v}_b2 = max({v}_na, {b});", v=var, b=d_b).unwrap();
                writeln!(code, "    let {v}_cs = {r} * 2.0 / {n};", v=var, r=r_s, n=n_s).unwrap();
                writeln!(code, "    var {v}_ra = 0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678;", v=var, r=r_s).unwrap();
                writeln!(code, "    let {v}_rb = 0.70710678 * ({v}_b2 - {v}_a2);", v=var).unwrap();
                writeln!(code, "    {v}_ra = ({v}_ra + {v}_cs * 0.5) % {v}_cs - {v}_cs * 0.5;", v=var).unwrap();
                writeln!(code, "    let {} = select(-min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m), -{v}_m, {v}_m > {r});", var, v=var, r=r_s).unwrap();
                var
            }

            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                let n_s = self.param(*n);
                writeln!(code, "    let {v}_na = -({a}); var {v}_m = min({v}_na, {b});", v=var, a=d_a, b=d_b).unwrap();
                writeln!(code, "    var {v}_a2 = min({v}_na, {b}); var {v}_b2 = max({v}_na, {b});", v=var, b=d_b).unwrap();
                writeln!(code, "    let {v}_cs = {r} * 2.0 / {n};", v=var, r=r_s, n=n_s).unwrap();
                writeln!(code, "    var {v}_ra = 0.70710678 * ({v}_a2 + {v}_b2) - {r} * 0.70710678;", v=var, r=r_s).unwrap();
                writeln!(code, "    let {v}_rb = 0.70710678 * ({v}_b2 - {v}_a2);", v=var).unwrap();
                writeln!(code, "    {v}_ra = ({v}_ra + {v}_cs * 0.5) % {v}_cs - {v}_cs * 0.5;", v=var).unwrap();
                writeln!(code, "    let {} = select(-min(min(0.70710678 * ({v}_ra + {v}_rb), 0.70710678 * ({v}_rb - {v}_ra)), {v}_m), -{v}_m, {v}_m > {r});", var, v=var, r=r_s).unwrap();
                var
            }

            SdfNode::Pipe { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                writeln!(code, "    let {} = length(vec2<f32>({}, {})) - {};", var, d_a, d_b, r_s).unwrap();
                var
            }

            SdfNode::Engrave { a, b, r } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let r_s = self.param(*r);
                writeln!(code, "    let {} = max({}, ({} + {} - abs({})) * 0.70710678);", var, d_a, d_a, r_s, d_b).unwrap();
                var
            }

            SdfNode::Groove { a, b, ra, rb } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let ra_s = self.param(*ra);
                let rb_s = self.param(*rb);
                writeln!(code, "    let {} = max({}, min({} + {}, {} - abs({})));", var, d_a, d_a, ra_s, rb_s, d_b).unwrap();
                var
            }

            SdfNode::Tongue { a, b, ra, rb } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                let ra_s = self.param(*ra);
                let rb_s = self.param(*rb);
                writeln!(code, "    let {} = min({}, max({} - {}, abs({}) - {}));", var, d_a, d_a, ra_s, d_b, rb_s).unwrap();
                var
            }

            // ============ Transforms ============

            SdfNode::Translate { child, offset } => {
                let new_p = self.next_var();
                let ox = self.param(offset.x);
                let oy = self.param(offset.y);
                let oz = self.param(offset.z);
                writeln!(
                    code,
                    "    let {} = {} - vec3<f32>({}, {}, {});",
                    new_p, point_var, ox, oy, oz
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Rotate { child, rotation } => {
                self.ensure_helper("quat_rotate");
                let inv_rot = rotation.inverse();
                let new_p = self.next_var();
                let qx = self.param(inv_rot.x);
                let qy = self.param(inv_rot.y);
                let qz = self.param(inv_rot.z);
                let qw = self.param(inv_rot.w);
                writeln!(
                    code,
                    "    let {} = quat_rotate({}, vec4<f32>({}, {}, {}, {}));",
                    new_p, point_var, qx, qy, qz, qw
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Scale { child, factor } => {
                let new_p = self.next_var();
                let inv_factor = 1.0 / factor;
                let p_inv = self.param(inv_factor);
                let p_factor = self.param(*factor);
                writeln!(
                    code,
                    "    let {} = {} * {};",
                    new_p, point_var, p_inv
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", var, d, p_factor).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for ScaleNonUniform
            SdfNode::ScaleNonUniform { child, factors } => {
                let new_p = self.next_var();
                let inv_x = self.param(1.0 / factors.x);
                let inv_y = self.param(1.0 / factors.y);
                let inv_z = self.param(1.0 / factors.z);
                let min_scale = factors.x.min(factors.y).min(factors.z);
                let p_min = self.param(min_scale);
                writeln!(
                    code,
                    "    let {} = {} * vec3<f32>({}, {}, {});",
                    new_p, point_var, inv_x, inv_y, inv_z
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {};", var, d, p_min).unwrap();
                var
            }

            // ============ Modifiers ============

            SdfNode::Twist { child, strength } => {
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let str_val = self.param(*strength);

                writeln!(
                    code,
                    "    let {} = {} * {}.y;",
                    angle_var, str_val, point_var
                )
                .unwrap();
                writeln!(code, "    let {} = cos({});", c_var, angle_var).unwrap();
                writeln!(code, "    let {} = sin({});", s_var, angle_var).unwrap();
                writeln!(
                    code,
                    "    let {} = vec3<f32>({} * {}.x - {} * {}.z, {}.y, {} * {}.x + {} * {}.z);",
                    new_p, c_var, point_var, s_var, point_var, point_var, s_var, point_var, c_var, point_var
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Bend { child, curvature } => {
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();
                let curv = self.param(*curvature);

                writeln!(
                    code,
                    "    let {} = {} * {}.x;",
                    angle_var, curv, point_var
                )
                .unwrap();
                writeln!(code, "    let {} = cos({});", c_var, angle_var).unwrap();
                writeln!(code, "    let {} = sin({});", s_var, angle_var).unwrap();
                writeln!(
                    code,
                    "    let {} = vec3<f32>({} * {}.x + {} * {}.y, {} * {}.y - {} * {}.x, {}.z);",
                    new_p, c_var, point_var, s_var, point_var, c_var, point_var, s_var, point_var, point_var
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Round { child, radius } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let r = self.param(*radius);
                writeln!(code, "    let {} = {} - {};", var, d, r).unwrap();
                var
            }

            SdfNode::Onion { child, thickness } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                let t = self.param(*thickness);
                writeln!(code, "    let {} = abs({}) - {};", var, d, t).unwrap();
                var
            }

            SdfNode::Elongate { child, amount } => {
                let q_var = self.next_var();
                let neg_ax = self.param(-amount.x);
                let neg_ay = self.param(-amount.y);
                let neg_az = self.param(-amount.z);
                let ax = self.param(amount.x);
                let ay = self.param(amount.y);
                let az = self.param(amount.z);
                writeln!(
                    code,
                    "    let {} = {} - clamp({}, vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}));",
                    q_var, point_var, point_var,
                    neg_ax, neg_ay, neg_az,
                    ax, ay, az
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let q_var = self.next_var();
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                writeln!(
                    code,
                    "    let {} = ({} + vec3<f32>({}, {}, {}) * 0.5) % vec3<f32>({}, {}, {}) - vec3<f32>({}, {}, {}) * 0.5;",
                    q_var, point_var,
                    sx, sy, sz,
                    sx, sy, sz,
                    sx, sy, sz
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatFinite {
                child,
                count,
                spacing,
            } => {
                let r_var = self.next_var();
                let q_var = self.next_var();
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                let neg_cx = self.param(-(count[0] as f32));
                let neg_cy = self.param(-(count[1] as f32));
                let neg_cz = self.param(-(count[2] as f32));
                let cx = self.param(count[0] as f32);
                let cy = self.param(count[1] as f32);
                let cz = self.param(count[2] as f32);
                writeln!(
                    code,
                    "    let {} = clamp(round({} / vec3<f32>({}, {}, {})), vec3<f32>({}, {}, {}), vec3<f32>({}, {}, {}));",
                    r_var, point_var,
                    sx, sy, sz,
                    neg_cx, neg_cy, neg_cz,
                    cx, cy, cz
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = {} - vec3<f32>({}, {}, {}) * {};",
                    q_var, point_var, sx, sy, sz, r_var
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::Noise { child, amplitude, frequency, seed } => {
                self.ensure_helper("hash_noise");
                let d = self.transpile_node_inner(child, point_var, code);
                let n_var = self.next_var();
                let var = self.next_var();
                let freq = self.param(*frequency);
                let amp = self.param(*amplitude);
                writeln!(code,
                    "    let {} = hash_noise_3d({} * {}, {}u);",
                    n_var, point_var, freq, seed
                ).unwrap();
                writeln!(code,
                    "    let {} = {} + {} * {};",
                    var, d, n_var, amp
                ).unwrap();
                var
            }

            SdfNode::Mirror { child, axes } => {
                let new_p = self.next_var();
                writeln!(code,
                    "    let {} = vec3<f32>({}, {}, {});",
                    new_p,
                    if axes.x != 0.0 { format!("abs({}.x)", point_var) } else { format!("{}.x", point_var) },
                    if axes.y != 0.0 { format!("abs({}.y)", point_var) } else { format!("{}.y", point_var) },
                    if axes.z != 0.0 { format!("abs({}.z)", point_var) } else { format!("{}.z", point_var) },
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::OctantMirror { child } => {
                let abs_p = self.next_var();
                let mx = self.next_var();
                let mn = self.next_var();
                let sorted_p = self.next_var();
                writeln!(code, "    let {} = abs({});", abs_p, point_var).unwrap();
                writeln!(code, "    let {} = max({}.x, max({}.y, {}.z));", mx, abs_p, abs_p, abs_p).unwrap();
                writeln!(code, "    let {} = min({}.x, min({}.y, {}.z));", mn, abs_p, abs_p, abs_p).unwrap();
                writeln!(code, "    let {} = vec3<f32>({}, {}.x + {}.y + {}.z - {} - {}, {});",
                    sorted_p, mx, abs_p, abs_p, abs_p, mx, mn, mn).unwrap();
                self.transpile_node_inner(child, &sorted_p, code)
            }

            SdfNode::Revolution { child, offset } => {
                let q_var = self.next_var();
                let new_p = self.next_var();
                let off = self.param(*offset);
                writeln!(code,
                    "    let {} = length({}.xz) - {};",
                    q_var, point_var, off
                ).unwrap();
                writeln!(code,
                    "    let {} = vec3<f32>({}, {}.y, 0.0);",
                    new_p, q_var, point_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Extrude { child, half_height } => {
                let flat_p = self.next_var();
                writeln!(code,
                    "    let {} = vec3<f32>({}.x, {}.y, 0.0);",
                    flat_p, point_var, point_var
                ).unwrap();
                let d = self.transpile_node_inner(child, &flat_p, code);
                let w_var = self.next_var();
                let var = self.next_var();
                let hh = self.param(*half_height);
                writeln!(code,
                    "    let {} = vec2<f32>({}, abs({}.z) - {});",
                    w_var, d, point_var, hh
                ).unwrap();
                writeln!(code,
                    "    let {} = min(max({}.x, {}.y), 0.0) + length(max({}, vec2<f32>(0.0)));",
                    var, w_var, w_var, w_var
                ).unwrap();
                var
            }

            SdfNode::Taper { child, factor } => {
                let new_p = self.next_var();
                let f = self.param(*factor);
                writeln!(code,
                    "    let {np}_s = 1.0 / (1.0 - {p}.y * {f});",
                    np = new_p, p = point_var, f = f
                ).unwrap();
                writeln!(code,
                    "    let {} = vec3<f32>({}.x * {np}_s, {}.y, {}.z * {np}_s);",
                    new_p, point_var, point_var, point_var, np = new_p
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Displacement { child, strength } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let disp_var = self.next_var();
                let var = self.next_var();
                let s = self.param(*strength);
                writeln!(code,
                    "    let {} = sin({}.x * 5.0) * sin({}.y * 5.0) * sin({}.z * 5.0);",
                    disp_var, point_var, point_var, point_var
                ).unwrap();
                writeln!(code,
                    "    let {} = {} + {} * {};",
                    var, d, disp_var, s
                ).unwrap();
                var
            }

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                let new_p = self.next_var();
                let p0x = self.param(p0.x); let p0z = self.param(p0.y);
                let p1x = self.param(p1.x); let p1z = self.param(p1.y);
                let p2x = self.param(p2.x); let p2z = self.param(p2.y);
                // Query point in XZ
                writeln!(code, "    let {np}_qx = {p}.x;", np = new_p, p = point_var).unwrap();
                writeln!(code, "    let {np}_qz = {p}.z;", np = new_p, p = point_var).unwrap();
                // Coarse search: 5 samples
                writeln!(code, "    var {np}_bt: f32 = 0.0;", np = new_p).unwrap();
                writeln!(code, "    var {np}_bd: f32 = 1e38;", np = new_p).unwrap();
                for i in 0..5u32 {
                    let t_val = i as f32 * 0.25;
                    let omt = 1.0 - t_val;
                    let omt2 = omt * omt;
                    let t2 = t_val * t_val;
                    let omt_t_2 = 2.0 * omt * t_val;
                    writeln!(code, "    {{ let bx = {p0x}*{omt2} + {p1x}*{omt_t_2} + {p2x}*{t2}; let bz = {p0z}*{omt2} + {p1z}*{omt_t_2} + {p2z}*{t2}; let dx = {np}_qx - bx; let dz = {np}_qz - bz; let d2 = dx*dx + dz*dz; if (d2 < {np}_bd) {{ {np}_bd = d2; {np}_bt = {t}; }} }}",
                        np = new_p, p0x = p0x, p1x = p1x, p2x = p2x, p0z = p0z, p1z = p1z, p2z = p2z,
                        omt2 = self.param(omt2), t2 = self.param(t2), omt_t_2 = self.param(omt_t_2), t = self.param(t_val)
                    ).unwrap();
                }
                // B''(t) constant
                let bddx_val = 2.0 * (p0.x - 2.0 * p1.x + p2.x);
                let bddz_val = 2.0 * (p0.y - 2.0 * p1.y + p2.y);
                let bddx = self.param(bddx_val);
                let bddz = self.param(bddz_val);
                // Newton iterations (unrolled)
                writeln!(code, "    var {np}_t: f32 = {np}_bt;", np = new_p).unwrap();
                for _ in 0..5 {
                    writeln!(code, "    {{ let omt = 1.0 - {np}_t; let bx = {p0x}*omt*omt + {p1x}*2.0*omt*{np}_t + {p2x}*{np}_t*{np}_t; let bz = {p0z}*omt*omt + {p1z}*2.0*omt*{np}_t + {p2z}*{np}_t*{np}_t; let tdx = ({p1x}-{p0x})*2.0*omt + ({p2x}-{p1x})*2.0*{np}_t; let tdz = ({p1z}-{p0z})*2.0*omt + ({p2z}-{p1z})*2.0*{np}_t; let dx = bx - {np}_qx; let dz = bz - {np}_qz; let num = dx*tdx + dz*tdz; let den = tdx*tdx + tdz*tdz + dx*{bddx} + dz*{bddz}; if (abs(den) > 1e-10) {{ {np}_t = clamp({np}_t - num/den, 0.0, 1.0); }} }}",
                        np = new_p, p0x = p0x, p1x = p1x, p2x = p2x, p0z = p0z, p1z = p1z, p2z = p2z, bddx = bddx, bddz = bddz
                    ).unwrap();
                }
                // Distance
                writeln!(code, "    let {np}_omt = 1.0 - {np}_t;", np = new_p).unwrap();
                writeln!(code, "    let {np}_cx = {p0x}*{np}_omt*{np}_omt + {p1x}*2.0*{np}_omt*{np}_t + {p2x}*{np}_t*{np}_t;",
                    np = new_p, p0x = p0x, p1x = p1x, p2x = p2x).unwrap();
                writeln!(code, "    let {np}_cz = {p0z}*{np}_omt*{np}_omt + {p1z}*2.0*{np}_omt*{np}_t + {p2z}*{np}_t*{np}_t;",
                    np = new_p, p0z = p0z, p1z = p1z, p2z = p2z).unwrap();
                writeln!(code, "    let {np}_dx = {np}_qx - {np}_cx;", np = new_p).unwrap();
                writeln!(code, "    let {np}_dz = {np}_qz - {np}_cz;", np = new_p).unwrap();
                writeln!(code, "    let {} = vec3<f32>(sqrt({np}_dx*{np}_dx + {np}_dz*{np}_dz), {p}.y, 0.0);",
                    new_p, np = new_p, p = point_var).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            // ★ Deep Fried: Division Exorcism for PolarRepeat
            SdfNode::PolarRepeat { child, count } => {
                let new_p = self.next_var();
                let count_f = *count as f32;
                let sector_angle = std::f32::consts::TAU / count_f;
                let sa = self.param(sector_angle);
                writeln!(code,
                    "    let {np}_a = atan2({p}.z, {p}.x) + {sa} * 0.5;",
                    np = new_p, p = point_var, sa = sa
                ).unwrap();
                writeln!(code,
                    "    let {np}_r = length({p}.xz);",
                    np = new_p, p = point_var
                ).unwrap();
                writeln!(code,
                    "    let {np}_am = (({np}_a + 100.0 * {sa}) % {sa}) - {sa} * 0.5;",
                    np = new_p, sa = sa
                ).unwrap();
                writeln!(code,
                    "    let {} = vec3<f32>({np}_r * cos({np}_am), {p}.y, {np}_r * sin({np}_am));",
                    new_p, np = new_p, p = point_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            // WithMaterial is transparent for shader evaluation
            SdfNode::WithMaterial { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
            }

            // === 2D Primitives (extruded to 3D) ===
            SdfNode::Circle2D { radius, half_height } => {
                let r = self.param(*radius);
                let hh = self.param(*half_height);
                let v = self.next_var();
                writeln!(code, "    let {v}_d2d = length({p}.xy) - {r};", v=v, p=point_var, r=r).unwrap();
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            SdfNode::Rect2D { half_extents, half_height } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hh = self.param(*half_height);
                let v = self.next_var();
                writeln!(code, "    let {v}_dx = abs({p}.x) - {hx};", v=v, p=point_var, hx=hx).unwrap();
                writeln!(code, "    let {v}_dy = abs({p}.y) - {hy};", v=v, p=point_var, hy=hy).unwrap();
                writeln!(code, "    let {v}_d2d = length(max(vec2<f32>({v}_dx, {v}_dy), vec2<f32>(0.0))) + min(max({v}_dx, {v}_dy), 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            SdfNode::Segment2D { a, b, thickness, half_height } => {
                let ax = self.param(a.x);
                let ay = self.param(a.y);
                let bx = self.param(b.x);
                let by = self.param(b.y);
                let th = self.param(*thickness);
                let hh = self.param(*half_height);
                let v = self.next_var();
                writeln!(code, "    let {v}_pa = {p}.xy - vec2<f32>({ax}, {ay});", v=v, p=point_var, ax=ax, ay=ay).unwrap();
                writeln!(code, "    let {v}_ba = vec2<f32>({bx}, {by}) - vec2<f32>({ax}, {ay});", v=v, bx=bx, by=by, ax=ax, ay=ay).unwrap();
                writeln!(code, "    let {v}_h = clamp(dot({v}_pa, {v}_ba) / dot({v}_ba, {v}_ba), 0.0, 1.0);", v=v).unwrap();
                writeln!(code, "    let {v}_d2d = length({v}_pa - {v}_ba * {v}_h) - {th};", v=v, th=th).unwrap();
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            SdfNode::Polygon2D { vertices, half_height } => {
                let hh = self.param(*half_height);
                let v = self.next_var();
                if vertices.len() >= 2 {
                    let max_r = vertices.iter().map(|v| (v.x*v.x + v.y*v.y).sqrt()).fold(0.0f32, f32::max);
                    let r = self.param(max_r);
                    writeln!(code, "    let {v}_d2d = length({p}.xy) - {r};", v=v, p=point_var, r=r).unwrap();
                } else {
                    writeln!(code, "    let {v}_d2d = length({p}.xy);", v=v, p=point_var).unwrap();
                }
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            SdfNode::RoundedRect2D { half_extents, round_radius, half_height } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let rr = self.param(*round_radius);
                let hh = self.param(*half_height);
                let v = self.next_var();
                writeln!(code, "    let {v}_dx = abs({p}.x) - {hx} + {rr};", v=v, p=point_var, hx=hx, rr=rr).unwrap();
                writeln!(code, "    let {v}_dy = abs({p}.y) - {hy} + {rr};", v=v, p=point_var, hy=hy, rr=rr).unwrap();
                writeln!(code, "    let {v}_d2d = length(max(vec2<f32>({v}_dx, {v}_dy), vec2<f32>(0.0))) + min(max({v}_dx, {v}_dy), 0.0) - {rr};", v=v, rr=rr).unwrap();
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            SdfNode::Annular2D { outer_radius, thickness, half_height } => {
                let or = self.param(*outer_radius);
                let th = self.param(*thickness);
                let hh = self.param(*half_height);
                let v = self.next_var();
                writeln!(code, "    let {v}_d2d = abs(length({p}.xy) - {or}) - {th};", v=v, p=point_var, or=or, th=th).unwrap();
                writeln!(code, "    let {v}_dz = abs({p}.z) - {hh};", v=v, p=point_var, hh=hh).unwrap();
                writeln!(code, "    let {v}_wx = max({v}_d2d, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v}_wy = max({v}_dz, 0.0);", v=v).unwrap();
                writeln!(code, "    let {v} = sqrt({v}_wx*{v}_wx + {v}_wy*{v}_wy) + min(max({v}_d2d, {v}_dz), 0.0);", v=v).unwrap();
                v
            }

            // === Exponential Smooth Operations ===
            SdfNode::ExpSmoothUnion { a, b, k } => {
                let da = self.transpile_node_inner(a, point_var, code);
                let db = self.transpile_node_inner(b, point_var, code);
                let k_val = self.param(*k);
                let v = self.next_var();
                writeln!(code, "    let {v}_ea = exp(-{da} / {k});", v=v, da=da, k=k_val).unwrap();
                writeln!(code, "    let {v}_eb = exp(-{db} / {k});", v=v, db=db, k=k_val).unwrap();
                writeln!(code, "    let {v} = -log(max({v}_ea + {v}_eb, 1e-10)) * {k};", v=v, k=k_val).unwrap();
                v
            }

            SdfNode::ExpSmoothIntersection { a, b, k } => {
                let da = self.transpile_node_inner(a, point_var, code);
                let db = self.transpile_node_inner(b, point_var, code);
                let k_val = self.param(*k);
                let v = self.next_var();
                writeln!(code, "    let {v}_ea = exp({da} / {k});", v=v, da=da, k=k_val).unwrap();
                writeln!(code, "    let {v}_eb = exp({db} / {k});", v=v, db=db, k=k_val).unwrap();
                writeln!(code, "    let {v} = log(max({v}_ea + {v}_eb, 1e-10)) * {k};", v=v, k=k_val).unwrap();
                v
            }

            SdfNode::ExpSmoothSubtraction { a, b, k } => {
                let da = self.transpile_node_inner(a, point_var, code);
                let db = self.transpile_node_inner(b, point_var, code);
                let k_val = self.param(*k);
                let v = self.next_var();
                writeln!(code, "    let {v}_ea = exp({da} / {k});", v=v, da=da, k=k_val).unwrap();
                writeln!(code, "    let {v}_enb = exp(-{db} / {k});", v=v, db=db, k=k_val).unwrap();
                writeln!(code, "    let {v} = log(max({v}_ea + {v}_enb, 1e-10)) * {k};", v=v, k=k_val).unwrap();
                v
            }

            // === New Modifiers ===
            SdfNode::Shear { child, shear } => {
                let sx = self.param(shear.x);
                let sy = self.param(shear.y);
                let sz = self.param(shear.z);
                let new_p = self.next_var();
                writeln!(code, "    let {} = vec3<f32>({p}.x, {p}.y - {sx} * {p}.x, {p}.z - {sy} * {p}.x - {sz} * {p}.y);",
                    new_p, p=point_var, sx=sx, sy=sy, sz=sz).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Animated { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
            }

        }
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
        let shape = SdfNode::Sphere { radius: 1.0 }
            .rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
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
        let shape = SdfNode::Sphere { radius: 1.5 }
            .translate(1.0, 2.0, 3.0);

        let params = WgslShader::extract_params(&shape);
        // Translate: 3 params (offset.x, y, z), then Sphere: 1 param (radius)
        // Translate is outer, processes first in transpile_node_inner
        assert_eq!(params.len(), 4);
        assert_eq!(params[0], 1.0);  // offset.x
        assert_eq!(params[1], 2.0);  // offset.y
        assert_eq!(params[2], 3.0);  // offset.z
        assert_eq!(params[3], 1.5);  // radius
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
            ("plane", SdfNode::Plane { normal: Vec3::Y, distance: 0.0 }),
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
            ("cut_hollow_sphere", SdfNode::cut_hollow_sphere(1.0, 0.3, 0.1)),
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
            ("superellipsoid", SdfNode::superellipsoid(1.0, 0.5, 0.5, 0.5, 0.5)),
            ("rounded_x", SdfNode::rounded_x(0.5, 0.1, 1.0)),
            ("pie", SdfNode::pie(1.0, 0.5, 1.0)),
            ("trapezoid", SdfNode::trapezoid(0.5, 0.3, 1.0, 0.5)),
            ("parallelogram", SdfNode::parallelogram(0.5, 1.0, 0.2, 0.5)),
            ("tunnel", SdfNode::tunnel(0.5, 1.0, 0.5)),
            ("uneven_capsule", SdfNode::uneven_capsule(0.3, 0.5, 1.0, 0.5)),
            ("arc_shape", SdfNode::arc_shape(1.0, 0.5, 0.1, 1.0)),
            ("moon", SdfNode::moon(0.5, 1.0, 0.8, 1.0)),
            ("cross_shape", SdfNode::cross_shape(0.5, 0.1, 0.05, 1.0)),
            ("blobby_cross", SdfNode::blobby_cross(0.5, 1.0)),
            ("parabola_segment", SdfNode::parabola_segment(0.5, 1.0, 0.5)),
            ("regular_polygon", SdfNode::regular_polygon(0.5, 6, 1.0)),
            ("star_polygon", SdfNode::star_polygon(0.5, 5, 2.0, 1.0)),
            ("chamfered_cube", SdfNode::chamfered_cube(0.5, 0.5, 0.5, 0.1)),
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
            ("bezier", SdfNode::bezier(Vec3::ZERO, Vec3::new(0.5, 1.0, 0.0), Vec3::X, 0.1)),
            ("circle_2d", SdfNode::circle_2d(0.5, 1.0)),
            ("rect_2d", SdfNode::rect_2d(0.5, 0.3, 1.0)),
            ("segment_2d", SdfNode::segment_2d(0.0, 0.0, 1.0, 1.0, 0.1, 1.0)),
            ("polygon_2d", SdfNode::polygon_2d(vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), Vec2::new(0.5, 1.0)], 1.0)),
            ("rounded_rect_2d", SdfNode::rounded_rect_2d(0.5, 0.3, 0.1, 1.0)),
            ("annular_2d", SdfNode::annular_2d(0.5, 0.1, 1.0)),
        ];

        for (name, prim) in &primitives {
            let shader = WgslShader::transpile(prim, TranspileMode::Hardcoded);
            assert!(
                shader.source.contains("fn sdf_eval"),
                "WGSL transpile failed for primitive '{}': missing sdf_eval", name,
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
            ("smooth_intersection", a.clone().smooth_intersection(b.clone(), 0.2)),
            ("smooth_subtract", a.clone().smooth_subtract(b.clone(), 0.2)),
            ("chamfer_union", a.clone().chamfer_union(b.clone(), 0.2)),
            ("chamfer_intersection", a.clone().chamfer_intersection(b.clone(), 0.2)),
            ("chamfer_subtract", a.clone().chamfer_subtract(b.clone(), 0.2)),
            ("stairs_union", a.clone().stairs_union(b.clone(), 0.2, 4.0)),
            ("stairs_intersection", a.clone().stairs_intersection(b.clone(), 0.2, 4.0)),
            ("stairs_subtract", a.clone().stairs_subtract(b.clone(), 0.2, 4.0)),
            ("columns_union", a.clone().columns_union(b.clone(), 0.2, 4.0)),
            ("columns_intersection", a.clone().columns_intersection(b.clone(), 0.2, 4.0)),
            ("columns_subtract", a.clone().columns_subtract(b.clone(), 0.2, 4.0)),
            ("xor", a.clone().xor(b.clone())),
            ("morph", a.clone().morph(b.clone(), 0.5)),
            ("pipe", a.clone().pipe(b.clone(), 0.2)),
            ("engrave", a.clone().engrave(b.clone(), 0.1)),
            ("groove", a.clone().groove(b.clone(), 0.2, 0.1)),
            ("tongue", a.clone().tongue(b.clone(), 0.2, 0.1)),
            ("exp_smooth_union", a.clone().exp_smooth_union(b.clone(), 0.2)),
            ("exp_smooth_intersection", a.clone().exp_smooth_intersection(b.clone(), 0.2)),
            ("exp_smooth_subtract", a.clone().exp_smooth_subtract(b.clone(), 0.2)),
        ];

        for (name, op) in &operations {
            let shader = WgslShader::transpile(op, TranspileMode::Hardcoded);
            assert!(
                shader.source.contains("fn sdf_eval"),
                "WGSL transpile failed for operation '{}': missing sdf_eval", name,
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
                "WGSL transpile failed for modifier '{}': missing sdf_eval", name,
            );
        }
    }
}

//! GLSL Transpiler: SDF to OpenGL Shading Language (Deep Fried Edition)
//!
//! This module converts SDF node trees to GLSL code for OpenGL, Vulkan,
//! Unity, and other GLSL-compatible environments.
//!
//! # Deep Fried Optimizations
//!
//! - **Division Exorcism**: Replaces `/k` with `* inv_k` in smooth ops
//! - **Inline Smooth Ops**: Pre-computes reciprocals at transpile time
//! - **FMA-friendly**: Structures operations for GPU FMA units
//! - **Dynamic Parameters**: Constants read from uniform block for zero-latency updates
//!
//! # Usage in Unity
//!
//! 1. Create a Custom Shader or Shader Graph Custom Function
//! 2. Paste the generated GLSL code
//! 3. Connect world position to `p` input
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
const FOLD_EPSILON: f32 = 1e-6;

/// Transpilation mode for GLSL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlslTranspileMode {
    /// Constants are baked into the shader (Fastest execution, requires recompile on change)
    Hardcoded,
    /// Constants are read from a uniform block (Fast update via glBufferSubData, good execution)
    Dynamic,
}

/// Generated GLSL shader code
#[derive(Debug, Clone)]
pub struct GlslShader {
    /// The generated GLSL source code
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
    /// Initial parameter values (empty for Hardcoded mode)
    pub param_layout: Vec<f32>,
    /// Transpilation mode used
    pub mode: GlslTranspileMode,
    /// GLSL version used (default: 450)
    pub version: u32,
}

impl GlslShader {
    /// Transpile an SDF node tree to GLSL with the specified mode
    pub fn transpile(node: &SdfNode, mode: GlslTranspileMode) -> Self {
        Self::transpile_with_version(node, mode, 450)
    }

    /// Transpile an SDF node tree to GLSL with specific version and mode
    pub fn transpile_with_version(node: &SdfNode, mode: GlslTranspileMode, version: u32) -> Self {
        let mut transpiler = GlslTranspiler::new(mode);
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        GlslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
            param_layout: transpiler.params,
            mode,
            version,
        }
    }

    /// Extract parameter values from an SDF node tree
    ///
    /// Returns the same parameter layout as produced by `transpile()` in Dynamic mode.
    /// Use this to update the uniform buffer without recompiling the shader.
    pub fn extract_params(node: &SdfNode) -> Vec<f32> {
        let mut transpiler = GlslTranspiler::new(GlslTranspileMode::Dynamic);
        let _ = transpiler.transpile_node(node, "p");
        transpiler.params
    }

    /// Generate GLSL for Unity Shader Graph Custom Function
    ///
    /// Returns code suitable for pasting into a Custom Function node in Unity.
    pub fn to_unity_custom_function(&self) -> String {
        let dynamic_note = if self.mode == GlslTranspileMode::Dynamic {
            "// NOTE: Dynamic mode - parameters are read from uniform block SdfParams.\n// Set up a UBO to update params at runtime.\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Generated GLSL for Unity Custom Function
// Input: float3 p (World Position)
// Output: float (SDF Distance)
{dynamic_note}
{source}

void SdfEval_float(float3 p, out float distance) {{
    distance = sdf_eval(p);
}}
"#,
            dynamic_note = dynamic_note,
            source = self.source
        )
    }

    /// Generate a complete GLSL compute shader for batch evaluation
    pub fn to_compute_shader(&self) -> String {
        let params_decl = if self.mode == GlslTranspileMode::Dynamic {
            "\nlayout(std140, binding = 2) uniform SdfParams {\n    vec4 params[1024]; // 4096 scalar floats\n};\n"
        } else {
            ""
        };

        format!(
            r#"#version {}

// ALICE-SDF Generated GLSL Compute Shader ({mode:?} Mode)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct InputPoint {{
    float x, y, z, _pad;
}};

struct OutputDistance {{
    float distance, _pad1, _pad2, _pad3;
}};

layout(std430, binding = 0) readonly buffer InputBuffer {{
    InputPoint input_points[];
}};

layout(std430, binding = 1) writeonly buffer OutputBuffer {{
    OutputDistance output_distances[];
}};

layout(location = 0) uniform uint point_count;
{params_decl}
{source}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) return;

    InputPoint pt = input_points[idx];
    vec3 p = vec3(pt.x, pt.y, pt.z);
    output_distances[idx].distance = sdf_eval(p);
}}
"#,
            self.version,
            mode = self.mode,
            params_decl = params_decl,
            source = self.source
        )
    }

    /// Generate a fragment shader for raymarching (Shadertoy-compatible)
    pub fn to_fragment_shader(&self) -> String {
        let params_decl = if self.mode == GlslTranspileMode::Dynamic {
            "\nlayout(std140, binding = 2) uniform SdfParams {\n    vec4 params[1024];\n};\n"
        } else {
            ""
        };

        format!(
            r#"#version {}

// ALICE-SDF Generated GLSL Fragment Shader
// Compatible with Shadertoy-style rendering

precision highp float;

uniform vec2 iResolution;
uniform float iTime;

out vec4 fragColor;
{params_decl}
{source}

vec3 calcNormal(vec3 p) {{
    const float h = 0.0001;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * sdf_eval(p + k.xyy * h) +
        k.yyx * sdf_eval(p + k.yyx * h) +
        k.yxy * sdf_eval(p + k.yxy * h) +
        k.xxx * sdf_eval(p + k.xxx * h)
    );
}}

void main() {{
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

    // Camera
    vec3 ro = vec3(0.0, 0.0, 5.0);
    vec3 rd = normalize(vec3(uv, -1.5));

    // Raymarching
    float t = 0.0;
    for (int i = 0; i < 128; i++) {{
        vec3 p = ro + rd * t;
        float d = sdf_eval(p);
        if (d < 0.001 || t > 100.0) break;
        t += d;
    }}

    // Shading
    vec3 col = vec3(0.0);
    if (t < 100.0) {{
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 light = normalize(vec3(1.0, 1.0, 1.0));
        float diff = max(dot(n, light), 0.0);
        col = vec3(0.8, 0.7, 0.6) * (0.2 + 0.8 * diff);
    }}

    fragColor = vec4(col, 1.0);
}}
"#,
            self.version,
            params_decl = params_decl,
            source = self.source
        )
    }

    /// Get the SDF evaluation function only (for embedding in custom shaders)
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }
}

/// Internal transpiler state
struct GlslTranspiler {
    /// Counter for generating unique variable names
    var_counter: usize,
    /// Helper functions that need to be included
    helper_functions: Vec<&'static str>,
    /// Transpilation mode
    mode: GlslTranspileMode,
    /// Collected parameter values (Dynamic mode)
    params: Vec<f32>,
}

impl GlslTranspiler {
    fn new(mode: GlslTranspileMode) -> Self {
        GlslTranspiler {
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

    /// Register a float parameter and return its GLSL expression string.
    ///
    /// - Hardcoded: returns a literal like `"1.000000"`
    /// - Dynamic: pushes to param buffer and returns `"params[i].comp"` (std140 vec4 packing)
    fn param(&mut self, value: f32) -> String {
        match self.mode {
            GlslTranspileMode::Hardcoded => format!("{:.6}", value),
            GlslTranspileMode::Dynamic => {
                let idx = self.params.len();
                self.params.push(value);
                let vec_idx = idx / 4;
                let comp = match idx % 4 {
                    0 => "x",
                    1 => "y",
                    2 => "z",
                    _ => "w",
                };
                format!("params[{}].{}", vec_idx, comp)
            }
        }
    }

    fn generate_shader(&self, body: &str) -> String {
        let mut shader = String::new();

        // Add helper functions
        for helper in &self.helper_functions {
            match *helper {
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
                _ => {}
            }
        }

        // Add main SDF function
        writeln!(shader, "float sdf_eval(vec3 p) {{").unwrap();
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
                let r_s = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    float {} = length({}) - {};", var, point_var, r_s).unwrap();
                var
            }

            SdfNode::Box3d { half_extents } => {
                let hx = self.param(half_extents.x);
                let hy = self.param(half_extents.y);
                let hz = self.param(half_extents.z);
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = abs({}) - vec3({}, {}, {});",
                    q_var, point_var, hx, hy, hz
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = length(max({}, vec3(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);",
                    var, q_var, q_var, q_var, q_var
                )
                .unwrap();
                var
            }

            SdfNode::Cylinder { radius, half_height } => {
                let r_s = self.param(*radius);
                let h_s = self.param(*half_height);
                let d_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec2 {} = vec2(length({}.xz) - {}, abs({}.y) - {});",
                    d_var, point_var, r_s, point_var, h_s
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = min(max({}.x, {}.y), 0.0) + length(max({}, vec2(0.0)));",
                    var, d_var, d_var, d_var
                )
                .unwrap();
                var
            }

            SdfNode::Torus {
                major_radius,
                minor_radius,
            } => {
                let maj_s = self.param(*major_radius);
                let min_s = self.param(*minor_radius);
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec2 {} = vec2(length({}.xz) - {}, {}.y);",
                    q_var, point_var, maj_s, point_var
                )
                .unwrap();
                writeln!(code, "    float {} = length({}) - {};", var, q_var, min_s).unwrap();
                var
            }

            SdfNode::Plane { normal, distance } => {
                let nx = self.param(normal.x);
                let ny = self.param(normal.y);
                let nz = self.param(normal.z);
                let dist = self.param(*distance);
                let var = self.next_var();
                writeln!(
                    code,
                    "    float {} = dot({}, vec3({}, {}, {})) + {};",
                    var, point_var, nx, ny, nz, dist
                )
                .unwrap();
                var
            }

            SdfNode::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let pax = self.param(point_a.x);
                let pay = self.param(point_a.y);
                let paz = self.param(point_a.z);
                let pbx = self.param(point_b.x);
                let pby = self.param(point_b.y);
                let pbz = self.param(point_b.z);
                // ba = point_b - point_a (re-emit point_a for subtraction)
                let pax2 = self.param(point_a.x);
                let pay2 = self.param(point_a.y);
                let paz2 = self.param(point_a.z);
                let r_s = self.param(*radius);

                let pa_var = self.next_var();
                let ba_var = self.next_var();
                let h_var = self.next_var();
                let var = self.next_var();

                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({}, {}, {});",
                    pa_var, point_var, pax, pay, paz
                )
                .unwrap();
                writeln!(
                    code,
                    "    vec3 {} = vec3({}, {}, {}) - vec3({}, {}, {});",
                    ba_var, pbx, pby, pbz, pax2, pay2, paz2
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = clamp(dot({}, {}) / dot({}, {}), 0.0, 1.0);",
                    h_var, pa_var, ba_var, ba_var, ba_var
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = length({} - {} * {}) - {};",
                    var, pa_var, ba_var, h_var, r_s
                )
                .unwrap();
                var
            }

            SdfNode::Cone { radius, half_height } => {
                let k2x = -radius;
                let k2y = 2.0 * half_height;
                let denom = k2x * k2x + k2y * k2y;

                let r_s = self.param(*radius);
                let hh_s = self.param(*half_height);
                let k2x_s = self.param(k2x);
                let k2y_s = self.param(k2y);
                let denom_s = self.param(denom);

                let qx_var = self.next_var();
                let h_var = self.next_var();
                let ca_var = self.next_var();
                let t_var = self.next_var();
                let cb_var = self.next_var();
                let s_var = self.next_var();
                let d2_var = self.next_var();
                let var = self.next_var();

                writeln!(code, "    float {} = length({}.xz);", qx_var, point_var).unwrap();
                writeln!(code, "    float {} = {};", h_var, hh_s).unwrap();
                writeln!(code,
                    "    vec2 {} = vec2({} - min({}, ({}.y < 0.0) ? {} : 0.0), abs({}.y) - {});",
                    ca_var, qx_var, qx_var, point_var, r_s, point_var, h_var
                ).unwrap();
                writeln!(code,
                    "    float {} = clamp((-{} * {} + ({} - {}.y) * {}) / {}, 0.0, 1.0);",
                    t_var, qx_var, k2x_s, h_var, point_var, k2y_s, denom_s
                ).unwrap();
                writeln!(code,
                    "    vec2 {} = vec2({} + {} * {}, {}.y - {} + {} * {});",
                    cb_var, qx_var, k2x_s, t_var, point_var, h_var, k2y_s, t_var
                ).unwrap();
                writeln!(code,
                    "    float {} = ({}.x < 0.0 && {}.y < 0.0) ? -1.0 : 1.0;",
                    s_var, cb_var, ca_var
                ).unwrap();
                writeln!(code,
                    "    float {} = min(dot({}, {}), dot({}, {}));",
                    d2_var, ca_var, ca_var, cb_var, cb_var
                ).unwrap();
                writeln!(code,
                    "    float {} = {} * sqrt({});",
                    var, s_var, d2_var
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for Ellipsoid
            SdfNode::Ellipsoid { radii } => {
                let inv_rx = self.param(1.0 / radii.x.max(1e-10));
                let inv_ry = self.param(1.0 / radii.y.max(1e-10));
                let inv_rz = self.param(1.0 / radii.z.max(1e-10));
                let inv_rx2 = self.param(1.0 / (radii.x * radii.x).max(1e-10));
                let inv_ry2 = self.param(1.0 / (radii.y * radii.y).max(1e-10));
                let inv_rz2 = self.param(1.0 / (radii.z * radii.z).max(1e-10));
                let var = self.next_var();
                let k0_var = self.next_var();
                let k1_var = self.next_var();
                writeln!(code,
                    "    float {} = length({} * vec3({}, {}, {}));",
                    k0_var, point_var, inv_rx, inv_ry, inv_rz
                ).unwrap();
                writeln!(code,
                    "    float {} = length({} * vec3({}, {}, {}));",
                    k1_var, point_var, inv_rx2, inv_ry2, inv_rz2
                ).unwrap();
                writeln!(code,
                    "    float {} = {} * ({} - 1.0) / max({}, 1e-10);",
                    var, k0_var, k0_var, k1_var
                ).unwrap();
                var
            }

            SdfNode::RoundedCone { r1, r2, half_height } => {
                self.ensure_helper("sdf_rounded_cone");
                let r1_s = self.param(*r1);
                let r2_s = self.param(*r2);
                let hh_s = self.param(*half_height);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_rounded_cone({}, {}, {}, {});",
                    var, point_var, r1_s, r2_s, hh_s
                ).unwrap();
                var
            }

            SdfNode::Pyramid { half_height } => {
                self.ensure_helper("sdf_pyramid");
                let hh_s = self.param(*half_height);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_pyramid({}, {});",
                    var, point_var, hh_s
                ).unwrap();
                var
            }

            SdfNode::Octahedron { size } => {
                self.ensure_helper("sdf_octahedron");
                let s_s = self.param(*size);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_octahedron({}, {});",
                    var, point_var, s_s
                ).unwrap();
                var
            }

            SdfNode::HexPrism { hex_radius, half_height } => {
                self.ensure_helper("sdf_hex_prism");
                let hr_s = self.param(*hex_radius);
                let hh_s = self.param(*half_height);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_hex_prism({}, {}, {});",
                    var, point_var, hr_s, hh_s
                ).unwrap();
                var
            }

            SdfNode::Link { half_length, r1, r2 } => {
                self.ensure_helper("sdf_link");
                let hl_s = self.param(*half_length);
                let r1_s = self.param(*r1);
                let r2_s = self.param(*r2);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_link({}, {}, {}, {});",
                    var, point_var, hl_s, r1_s, r2_s
                ).unwrap();
                var
            }

            SdfNode::Triangle { point_a, point_b, point_c } => {
                self.ensure_helper("sdf_triangle");
                let ax = self.param(point_a.x);
                let ay = self.param(point_a.y);
                let az = self.param(point_a.z);
                let bx = self.param(point_b.x);
                let by = self.param(point_b.y);
                let bz = self.param(point_b.z);
                let cx = self.param(point_c.x);
                let cy = self.param(point_c.y);
                let cz = self.param(point_c.z);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_triangle({}, vec3({}, {}, {}), vec3({}, {}, {}), vec3({}, {}, {}));",
                    var, point_var, ax, ay, az, bx, by, bz, cx, cy, cz
                ).unwrap();
                var
            }

            SdfNode::Bezier { point_a, point_b, point_c, radius } => {
                self.ensure_helper("sdf_bezier");
                let ax = self.param(point_a.x);
                let ay = self.param(point_a.y);
                let az = self.param(point_a.z);
                let bx = self.param(point_b.x);
                let by = self.param(point_b.y);
                let bz = self.param(point_b.z);
                let cx = self.param(point_c.x);
                let cy = self.param(point_c.y);
                let cz = self.param(point_c.z);
                let r_s = self.param(*radius);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = sdf_bezier({}, vec3({}, {}, {}), vec3({}, {}, {}), vec3({}, {}, {}), {});",
                    var, point_var, ax, ay, az, bx, by, bz, cx, cy, cz, r_s
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
                writeln!(code, "    vec3 {} = abs({}) - vec3({}, {}, {});", q_var, point_var, hx, hy, hz).unwrap();
                writeln!(code, "    float {} = length(max({}, vec3(0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0) - {};", var, q_var, q_var, q_var, q_var, rr).unwrap();
                var
            }

            SdfNode::CappedCone { half_height, r1, r2 } => {
                self.ensure_helper("sdf_capped_cone");
                let h = self.param(*half_height);
                let p_r1 = self.param(*r1);
                let p_r2 = self.param(*r2);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_capped_cone({}, {}, {}, {});", var, point_var, h, p_r1, p_r2).unwrap();
                var
            }

            SdfNode::CappedTorus { major_radius, minor_radius, cap_angle } => {
                self.ensure_helper("sdf_capped_torus");
                let ra = self.param(*major_radius);
                let rb = self.param(*minor_radius);
                let an = self.param(*cap_angle);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_capped_torus({}, {}, {}, {});", var, point_var, ra, rb, an).unwrap();
                var
            }

            SdfNode::RoundedCylinder { radius, round_radius, half_height } => {
                self.ensure_helper("sdf_rounded_cylinder");
                let r = self.param(*radius);
                let rr = self.param(*round_radius);
                let h = self.param(*half_height);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_rounded_cylinder({}, {}, {}, {});", var, point_var, r, rr, h).unwrap();
                var
            }

            SdfNode::TriangularPrism { width, half_depth } => {
                self.ensure_helper("sdf_triangular_prism");
                let w = self.param(*width);
                let d = self.param(*half_depth);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_triangular_prism({}, {}, {});", var, point_var, w, d).unwrap();
                var
            }

            SdfNode::CutSphere { radius, cut_height } => {
                self.ensure_helper("sdf_cut_sphere");
                let r = self.param(*radius);
                let h = self.param(*cut_height);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_cut_sphere({}, {}, {});", var, point_var, r, h).unwrap();
                var
            }

            SdfNode::CutHollowSphere { radius, cut_height, thickness } => {
                self.ensure_helper("sdf_cut_hollow_sphere");
                let r = self.param(*radius);
                let h = self.param(*cut_height);
                let t = self.param(*thickness);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_cut_hollow_sphere({}, {}, {}, {});", var, point_var, r, h, t).unwrap();
                var
            }

            SdfNode::DeathStar { ra, rb, d } => {
                self.ensure_helper("sdf_death_star");
                let p_ra = self.param(*ra);
                let p_rb = self.param(*rb);
                let p_d = self.param(*d);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_death_star({}, {}, {}, {});", var, point_var, p_ra, p_rb, p_d).unwrap();
                var
            }

            SdfNode::SolidAngle { angle, radius } => {
                self.ensure_helper("sdf_solid_angle");
                let an = self.param(*angle);
                let r = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_solid_angle({}, {}, {});", var, point_var, an, r).unwrap();
                var
            }

            SdfNode::Rhombus { la, lb, half_height, round_radius } => {
                self.ensure_helper("sdf_rhombus");
                let p_la = self.param(*la);
                let p_lb = self.param(*lb);
                let h = self.param(*half_height);
                let rr = self.param(*round_radius);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_rhombus({}, {}, {}, {}, {});", var, point_var, p_la, p_lb, h, rr).unwrap();
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
                writeln!(code, "    float {} = sdf_horseshoe({}, {}, {}, {}, {}, {});", var, point_var, an, r, hl, w, t).unwrap();
                var
            }

            SdfNode::Vesica { radius, half_dist } => {
                self.ensure_helper("sdf_vesica");
                let r = self.param(*radius);
                let d = self.param(*half_dist);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_vesica({}, {}, {});", var, point_var, r, d).unwrap();
                var
            }

            SdfNode::InfiniteCylinder { radius } => {
                let r = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    float {} = length({}.xz) - {};", var, point_var, r).unwrap();
                var
            }

            SdfNode::InfiniteCone { angle } => {
                self.ensure_helper("sdf_infinite_cone");
                let an = self.param(*angle);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_infinite_cone({}, {});", var, point_var, an).unwrap();
                var
            }

            SdfNode::Gyroid { scale, thickness } => {
                let sc = self.param(*scale);
                let th = self.param(*thickness);
                let sp_var = self.next_var();
                let var = self.next_var();
                writeln!(code, "    vec3 {} = {} * {};", sp_var, point_var, sc).unwrap();
                writeln!(code, "    float {} = abs(sin({}.x) * cos({}.y) + sin({}.y) * cos({}.z) + sin({}.z) * cos({}.x)) / {} - {};", var, sp_var, sp_var, sp_var, sp_var, sp_var, sp_var, sc, th).unwrap();
                var
            }

            SdfNode::Heart { size } => {
                self.ensure_helper("sdf_heart");
                let s = self.param(*size);
                let var = self.next_var();
                writeln!(code, "    float {} = sdf_heart({}, {});", var, point_var, s).unwrap();
                var
            }

            // ============ Operations ============

            SdfNode::Union { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    float {} = min({}, {});", var, d_a, d_b).unwrap();
                var
            }

            SdfNode::Intersection { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    float {} = max({}, {});", var, d_a, d_b).unwrap();
                var
            }

            SdfNode::Subtraction { a, b } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();
                writeln!(code, "    float {} = max({}, -{});", var, d_a, d_b).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothUnion
            SdfNode::SmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                // Constant folding only in Hardcoded mode
                if self.mode == GlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / k };
                let k_s = self.param(*k);
                let inv_k_s = self.param(inv_k);
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    float {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_s, d_a, d_b, inv_k_s
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = min({}, {}) - {} * {} * {} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k_s
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothIntersection
            SdfNode::SmoothIntersection { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == GlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / k };
                let k_s = self.param(*k);
                let inv_k_s = self.param(inv_k);
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    float {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_s, d_a, d_b, inv_k_s
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = max({}, {}) + {} * {} * {} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k_s
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothSubtraction
            SdfNode::SmoothSubtraction { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if self.mode == GlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = if k.abs() < 1e-10 { 1.0 } else { 1.0 / k };
                let k_s = self.param(*k);
                let inv_k_s = self.param(inv_k);
                let h_var = self.next_var();
                let neg_b = self.next_var();

                writeln!(code, "    float {} = -{};", neg_b, d_b).unwrap();
                writeln!(
                    code,
                    "    float {} = max({} - abs({} - {}), 0.0) * {};",
                    h_var, k_s, d_a, neg_b, inv_k_s
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = max({}, {}) + {} * {} * {} * 0.25;",
                    var, d_a, neg_b, h_var, h_var, k_s
                ).unwrap();
                var
            }

            // ============ Transforms ============

            SdfNode::Translate { child, offset } => {
                let ox = self.param(offset.x);
                let oy = self.param(offset.y);
                let oz = self.param(offset.z);
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({}, {}, {});",
                    new_p, point_var, ox, oy, oz
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Rotate { child, rotation } => {
                self.ensure_helper("quat_rotate");
                let inv_rot = rotation.inverse();
                let qx = self.param(inv_rot.x);
                let qy = self.param(inv_rot.y);
                let qz = self.param(inv_rot.z);
                let qw = self.param(inv_rot.w);
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = quat_rotate({}, vec4({}, {}, {}, {}));",
                    new_p, point_var, qx, qy, qz, qw
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            // ★ Deep Fried: Division Exorcism for Scale
            SdfNode::Scale { child, factor } => {
                let inv_factor = if factor.abs() < 1e-10 { 1.0 } else { 1.0 / factor };
                let inv_f_s = self.param(inv_factor);
                let f_s = self.param(*factor);
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} * {};",
                    new_p, point_var, inv_f_s
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    float {} = {} * {};", var, d, f_s).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for ScaleNonUniform
            SdfNode::ScaleNonUniform { child, factors } => {
                let inv_x = if factors.x.abs() < 1e-10 { 1.0 } else { 1.0 / factors.x };
                let inv_y = if factors.y.abs() < 1e-10 { 1.0 } else { 1.0 / factors.y };
                let inv_z = if factors.z.abs() < 1e-10 { 1.0 } else { 1.0 / factors.z };
                let inv_x_s = self.param(inv_x);
                let inv_y_s = self.param(inv_y);
                let inv_z_s = self.param(inv_z);
                let min_scale = factors.x.min(factors.y).min(factors.z);
                let ms_s = self.param(min_scale);
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} * vec3({}, {}, {});",
                    new_p, point_var, inv_x_s, inv_y_s, inv_z_s
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    float {} = {} * {};", var, d, ms_s).unwrap();
                var
            }

            // ============ Modifiers ============

            SdfNode::Twist { child, strength } => {
                let str_s = self.param(*strength);
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();

                writeln!(
                    code,
                    "    float {} = {} * {}.y;",
                    angle_var, str_s, point_var
                )
                .unwrap();
                writeln!(code, "    float {} = cos({});", c_var, angle_var).unwrap();
                writeln!(code, "    float {} = sin({});", s_var, angle_var).unwrap();
                writeln!(
                    code,
                    "    vec3 {} = vec3({} * {}.x - {} * {}.z, {}.y, {} * {}.x + {} * {}.z);",
                    new_p, c_var, point_var, s_var, point_var, point_var, s_var, point_var, c_var, point_var
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Bend { child, curvature } => {
                let curv_s = self.param(*curvature);
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();

                writeln!(
                    code,
                    "    float {} = {} * {}.x;",
                    angle_var, curv_s, point_var
                )
                .unwrap();
                writeln!(code, "    float {} = cos({});", c_var, angle_var).unwrap();
                writeln!(code, "    float {} = sin({});", s_var, angle_var).unwrap();
                writeln!(
                    code,
                    "    vec3 {} = vec3({} * {}.x + {} * {}.y, {} * {}.y - {} * {}.x, {}.z);",
                    new_p, c_var, point_var, s_var, point_var, c_var, point_var, s_var, point_var, point_var
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Round { child, radius } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let r_s = self.param(*radius);
                let var = self.next_var();
                writeln!(code, "    float {} = {} - {};", var, d, r_s).unwrap();
                var
            }

            SdfNode::Onion { child, thickness } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let t_s = self.param(*thickness);
                let var = self.next_var();
                writeln!(code, "    float {} = abs({}) - {};", var, d, t_s).unwrap();
                var
            }

            SdfNode::Elongate { child, amount } => {
                let ax = self.param(amount.x);
                let ay = self.param(amount.y);
                let az = self.param(amount.z);
                let nax = self.param(-amount.x);
                let nay = self.param(-amount.y);
                let naz = self.param(-amount.z);
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} - clamp({}, vec3({}, {}, {}), vec3({}, {}, {}));",
                    q_var, point_var, point_var, nax, nay, naz, ax, ay, az
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = mod({} + vec3({}, {}, {}) * 0.5, vec3({}, {}, {})) - vec3({}, {}, {}) * 0.5;",
                    q_var, point_var, sx, sy, sz, sx, sy, sz, sx, sy, sz
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatFinite {
                child,
                count,
                spacing,
            } => {
                let sx = self.param(spacing.x);
                let sy = self.param(spacing.y);
                let sz = self.param(spacing.z);
                let ncx = self.param(-(count[0] as f32));
                let ncy = self.param(-(count[1] as f32));
                let ncz = self.param(-(count[2] as f32));
                let pcx = self.param(count[0] as f32);
                let pcy = self.param(count[1] as f32);
                let pcz = self.param(count[2] as f32);
                let r_var = self.next_var();
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = clamp(round({} / vec3({}, {}, {})), vec3({}, {}, {}), vec3({}, {}, {}));",
                    r_var, point_var, sx, sy, sz, ncx, ncy, ncz, pcx, pcy, pcz
                )
                .unwrap();
                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({}, {}, {}) * {};",
                    q_var, point_var, sx, sy, sz, r_var
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::Noise { child, amplitude, frequency, seed } => {
                self.ensure_helper("hash_noise");
                let d = self.transpile_node_inner(child, point_var, code);
                let freq_s = self.param(*frequency);
                let amp_s = self.param(*amplitude);
                let seed_s = self.param(*seed as f32);
                let n_var = self.next_var();
                let var = self.next_var();
                writeln!(code,
                    "    float {} = hash_noise_3d({} * {}, uint({}));",
                    n_var, point_var, freq_s, seed_s
                ).unwrap();
                writeln!(code,
                    "    float {} = {} + {} * {};",
                    var, d, n_var, amp_s
                ).unwrap();
                var
            }

            SdfNode::Mirror { child, axes } => {
                // Mirror axes are structural (which axes to mirror), not parameterizable
                let new_p = self.next_var();
                writeln!(code,
                    "    vec3 {} = vec3({}, {}, {});",
                    new_p,
                    if axes.x != 0.0 { format!("abs({}.x)", point_var) } else { format!("{}.x", point_var) },
                    if axes.y != 0.0 { format!("abs({}.y)", point_var) } else { format!("{}.y", point_var) },
                    if axes.z != 0.0 { format!("abs({}.z)", point_var) } else { format!("{}.z", point_var) },
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Revolution { child, offset } => {
                let off_s = self.param(*offset);
                let q_var = self.next_var();
                let new_p = self.next_var();
                writeln!(code,
                    "    float {} = length({}.xz) - {};",
                    q_var, point_var, off_s
                ).unwrap();
                writeln!(code,
                    "    vec3 {} = vec3({}, {}.y, 0.0);",
                    new_p, q_var, point_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Extrude { child, half_height } => {
                let hh_s = self.param(*half_height);
                let flat_p = self.next_var();
                writeln!(code,
                    "    vec3 {} = vec3({}.x, {}.y, 0.0);",
                    flat_p, point_var, point_var
                ).unwrap();
                let d = self.transpile_node_inner(child, &flat_p, code);
                let w_var = self.next_var();
                let var = self.next_var();
                writeln!(code,
                    "    vec2 {} = vec2({}, abs({}.z) - {});",
                    w_var, d, point_var, hh_s
                ).unwrap();
                writeln!(code,
                    "    float {} = min(max({}.x, {}.y), 0.0) + length(max({}, vec2(0.0)));",
                    var, w_var, w_var, w_var
                ).unwrap();
                var
            }

            SdfNode::Taper { child, factor } => {
                let f_s = self.param(*factor);
                let s_var = self.next_var();
                let new_p = self.next_var();
                writeln!(code,
                    "    float {} = 1.0 / (1.0 - {}.y * {});",
                    s_var, point_var, f_s
                ).unwrap();
                writeln!(code,
                    "    vec3 {} = vec3({}.x * {}, {}.y, {}.z * {});",
                    new_p, point_var, s_var, point_var, point_var, s_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Displacement { child, strength } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let str_s = self.param(*strength);
                let var = self.next_var();
                writeln!(code,
                    "    float {} = {} + sin(5.0 * {}.x) * sin(5.0 * {}.y) * sin(5.0 * {}.z) * {};",
                    var, d, point_var, point_var, point_var, str_s
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for PolarRepeat
            SdfNode::PolarRepeat { child, count } => {
                let count_f = *count as f32;
                let sector_angle = std::f32::consts::TAU / count_f;
                let sa = self.param(sector_angle);
                let angle_var = self.next_var();
                let new_angle_var = self.next_var();
                let new_p = self.next_var();
                writeln!(code,
                    "    float {} = atan({}.z, {}.x);",
                    angle_var, point_var, point_var
                ).unwrap();
                writeln!(code,
                    "    float {} = mod({} + {} * 0.5, {}) - {} * 0.5;",
                    new_angle_var, angle_var, sa, sa, sa
                ).unwrap();
                writeln!(code,
                    "    vec3 {} = vec3(length({}.xz) * cos({}), {}.y, length({}.xz) * sin({}));",
                    new_p, point_var, new_angle_var, point_var, point_var, new_angle_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            // WithMaterial is transparent for shader evaluation
            SdfNode::WithMaterial { child, .. } => {
                self.transpile_node_inner(child, point_var, code)
            }
        }
    }
}

// Helper function definitions for GLSL
const HELPER_HASH_NOISE: &str = r#"float hash_noise_3d(vec3 p, uint seed) {
    vec3 f = fract(p);
    vec3 i = floor(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float n000 = fract(sin(dot(i, vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n100 = fract(sin(dot(i + vec3(1,0,0), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n010 = fract(sin(dot(i + vec3(0,1,0), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n110 = fract(sin(dot(i + vec3(1,1,0), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n001 = fract(sin(dot(i + vec3(0,0,1), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n101 = fract(sin(dot(i + vec3(1,0,1), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n011 = fract(sin(dot(i + vec3(0,1,1), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float n111 = fract(sin(dot(i + vec3(1,1,1), vec3(127.1, 311.7, 74.7)) + float(seed)) * 43758.5453);
    float c00 = mix(n000, n100, u.x);
    float c10 = mix(n010, n110, u.x);
    float c01 = mix(n001, n101, u.x);
    float c11 = mix(n011, n111, u.x);
    float c0 = mix(c00, c10, u.y);
    float c1 = mix(c01, c11, u.y);
    return mix(c0, c1, u.z) * 2.0 - 1.0;
}"#;

const HELPER_QUAT_ROTATE: &str = r#"vec3 quat_rotate(vec3 v, vec4 q) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

const HELPER_SDF_ROUNDED_CONE: &str = r#"float sdf_rounded_cone(vec3 p, float r1, float r2, float h) {
    float qx = length(p.xz);
    float qy = p.y + h;
    float ht = h * 2.0;
    float b = (r1 - r2) / ht;
    float a = sqrt(1.0 - b * b);
    float k = qx * (-b) + qy * a;
    if (k < 0.0) return length(vec2(qx, qy)) - r1;
    if (k > a * ht) return length(vec2(qx, qy - ht)) - r2;
    return qx * a + qy * b - r1;
}
"#;

const HELPER_SDF_PYRAMID: &str = r#"float sdf_pyramid(vec3 p, float h) {
    float ht = h * 2.0;
    float m2 = ht * ht + 0.25;
    float py = p.y + h;
    float px = abs(p.x);
    float pz = abs(p.z);
    if (pz > px) { float tmp = px; px = pz; pz = tmp; }
    px -= 0.5;
    pz -= 0.5;
    float qx = pz;
    float qy = ht * py - 0.5 * px;
    float qz = ht * px + 0.5 * py;
    float s = max(-qx, 0.0);
    float t = clamp((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    float a = m2 * (qx + s) * (qx + s) + qy * qy;
    float b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
    float d2 = (min(-qx * m2 - qy * 0.5, qy) > 0.0) ? 0.0 : min(a, b);
    return sqrt((d2 + qz * qz) / m2) * sign(max(qz, -py));
}
"#;

const HELPER_SDF_OCTAHEDRON: &str = r#"float sdf_octahedron(vec3 p, float s) {
    vec3 ap = abs(p);
    float m = ap.x + ap.y + ap.z - s;
    vec3 q;
    if (3.0 * ap.x < m) q = ap;
    else if (3.0 * ap.y < m) q = vec3(ap.y, ap.z, ap.x);
    else if (3.0 * ap.z < m) q = vec3(ap.z, ap.x, ap.y);
    else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}
"#;

const HELPER_SDF_HEX_PRISM: &str = r#"float sdf_hex_prism(vec3 p, float hex_r, float h) {
    float kx = -0.8660254;
    float ky = 0.5;
    float kz = 0.57735027;
    float px = abs(p.x);
    float py = abs(p.y);
    float pz = abs(p.z);
    float dot_kxy = kx * px + ky * py;
    float refl = 2.0 * min(dot_kxy, 0.0);
    px -= refl * kx;
    py -= refl * ky;
    float clamped_x = clamp(px, -kz * hex_r, kz * hex_r);
    float dx = px - clamped_x;
    float dy = py - hex_r;
    float d_xy = sqrt(dx * dx + dy * dy) * sign(dy);
    float d_z = pz - h;
    return min(max(d_xy, d_z), 0.0) + length(max(vec2(d_xy, d_z), vec2(0.0)));
}
"#;

const HELPER_SDF_TRIANGLE: &str = r#"float sdf_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ba = b - a; vec3 pa = p - a;
    vec3 cb = c - b; vec3 pb = p - b;
    vec3 ac = a - c; vec3 pc = p - c;
    vec3 nor = cross(ba, ac);
    return sqrt(
        (sign(dot(cross(ba, nor), pa)) +
         sign(dot(cross(cb, nor), pb)) +
         sign(dot(cross(ac, nor), pc)) < 2.0)
        ?
        min(min(
            dot(ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa,
                ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa),
            dot(cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb,
                cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb)),
            dot(ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc,
                ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc))
        :
        dot(nor, pa) * dot(nor, pa) / dot(nor, nor)
    );
}
"#;

const HELPER_SDF_BEZIER: &str = r#"float sdf_bezier(vec3 pos, vec3 A, vec3 B, vec3 C, float r) {
    vec3 a = B - A;
    vec3 b = A - 2.0 * B + C;
    vec3 c = a * 2.0;
    vec3 d = A - pos;
    float kk = 1.0 / max(dot(b, b), 1e-10);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p2 = ky - kx * kx;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float p3 = p2 * p2 * p2;
    float h = q * q + 4.0 * p3;
    float res;
    if (h >= 0.0) {
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) * 0.5;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = dot(d + (c + b * t) * t, d + (c + b * t) * t);
    } else {
        float z = sqrt(-p2);
        float v2 = acos(q / (p2 * z * 2.0)) / 3.0;
        float m = cos(v2);
        float n = sin(v2) * 1.732050808;
        float t1 = clamp(( m + m) * z - kx, 0.0, 1.0);
        float t2 = clamp((-n - m) * z - kx, 0.0, 1.0);
        float d1 = dot(d + (c + b * t1) * t1, d + (c + b * t1) * t1);
        float d2 = dot(d + (c + b * t2) * t2, d + (c + b * t2) * t2);
        res = min(d1, d2);
    }
    return sqrt(res) - r;
}
"#;

const HELPER_SDF_LINK: &str = r#"float sdf_link(vec3 p, float le, float r1, float r2) {
    float qx = p.x;
    float qy = max(abs(p.y) - le, 0.0);
    float qz = p.z;
    float xy_len = sqrt(qx * qx + qy * qy) - r1;
    return sqrt(xy_len * xy_len + qz * qz) - r2;
}
"#;

const HELPER_SDF_CAPPED_CONE: &str = r#"float sdf_capped_cone(vec3 p, float h, float r1, float r2) {
    float qx = length(p.xz);
    float qy = p.y;
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    float min_r = (qy < 0.0) ? r1 : r2;
    vec2 ca = vec2(qx - min(qx, min_r), abs(qy) - h);
    vec2 q = vec2(qx, qy);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
"#;

const HELPER_SDF_CAPPED_TORUS: &str = r#"float sdf_capped_torus(vec3 p, float ra, float rb, float an) {
    vec2 sc = vec2(sin(an), cos(an));
    float px = abs(p.x);
    float k = (sc.y * px > sc.x * p.y) ? dot(vec2(px, p.y), sc) : length(vec2(px, p.y));
    return sqrt(px * px + p.y * p.y + p.z * p.z + ra * ra - 2.0 * ra * k) - rb;
}
"#;

const HELPER_SDF_ROUNDED_CYLINDER: &str = r#"float sdf_rounded_cylinder(vec3 p, float r, float rr, float h) {
    vec2 d = vec2(length(p.xz) - r + rr, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0))) - rr;
}
"#;

const HELPER_SDF_TRIANGULAR_PRISM: &str = r#"float sdf_triangular_prism(vec3 p, float w, float h) {
    vec3 q = abs(p);
    return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - w * 0.5);
}
"#;

const HELPER_SDF_CUT_SPHERE: &str = r#"float sdf_cut_sphere(vec3 p, float r, float h) {
    float w = sqrt(max(r * r - h * h, 0.0));
    vec2 q = vec2(length(p.xz), p.y);
    float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
    if (s < 0.0) return length(q) - r;
    if (q.x < w) return h - q.y;
    return length(q - vec2(w, h));
}
"#;

const HELPER_SDF_CUT_HOLLOW_SPHERE: &str = r#"float sdf_cut_hollow_sphere(vec3 p, float r, float h, float t) {
    float w = sqrt(max(r * r - h * h, 0.0));
    vec2 q = vec2(length(p.xz), p.y);
    if (h * q.x < w * q.y) return length(q - vec2(w, h)) - t;
    return abs(length(q) - r) - t;
}
"#;

const HELPER_SDF_DEATH_STAR: &str = r#"float sdf_death_star(vec3 p, float ra, float rb, float d) {
    float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    float b = sqrt(max(ra * ra - a * a, 0.0));
    vec2 q = vec2(p.x, length(p.yz));
    if (q.x * b - q.y * a > d * max(b - q.y, 0.0))
        return length(q - vec2(a, b));
    return max(length(q) - ra, -(length(q - vec2(d, 0.0)) - rb));
}
"#;

const HELPER_SDF_SOLID_ANGLE: &str = r#"float sdf_solid_angle(vec3 p, float an, float ra) {
    vec2 c = vec2(sin(an), cos(an));
    vec2 q = vec2(length(p.xz), p.y);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q, c), 0.0, ra));
    return max(l, m * sign(c.y * q.x - c.x * q.y));
}
"#;

const HELPER_SDF_RHOMBUS: &str = r#"float ndot_rh(vec2 a, vec2 b) {
    return a.x * b.x - a.y * b.y;
}
float sdf_rhombus(vec3 p, float la, float lb, float h, float ra) {
    vec3 ap = abs(p);
    vec2 b = vec2(la, lb);
    float f = clamp(ndot_rh(b, b - 2.0 * ap.xz) / dot(b, b), -1.0, 1.0);
    float dxz = length(ap.xz - 0.5 * b * vec2(1.0 - f, 1.0 + f)) * sign(ap.x * b.y + ap.z * b.x - b.x * b.y) - ra;
    vec2 q = vec2(dxz, ap.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0)));
}
"#;

const HELPER_SDF_HORSESHOE: &str = r#"float sdf_horseshoe(vec3 pos, float an, float r, float le, float w, float t) {
    vec2 c = vec2(cos(an), sin(an));
    float px = abs(pos.x);
    float l = length(vec2(px, pos.y));
    float qx = -c.x * px + c.y * pos.y;
    float qy = c.y * px + c.x * pos.y;
    if (!(qy > 0.0 || qx > 0.0)) qx = l * sign(-c.x);
    if (qx <= 0.0) qy = l;
    qx = abs(qx) - le;
    qy = abs(qy - r);
    float e = length(max(vec2(qx, qy), vec2(0.0))) + min(max(qx, qy), 0.0);
    vec2 d = abs(vec2(e, pos.z)) - vec2(w, t);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}
"#;

const HELPER_SDF_VESICA: &str = r#"float sdf_vesica(vec3 p, float r, float d) {
    float px = abs(p.x);
    float py = length(p.yz);
    float b = sqrt(max(r * r - d * d, 0.0));
    if ((py - b) * d > px * b) return length(vec2(px, py - b));
    return length(vec2(px - d, py)) - r;
}
"#;

const HELPER_SDF_INFINITE_CONE: &str = r#"float sdf_infinite_cone(vec3 p, float an) {
    vec2 c = vec2(sin(an), cos(an));
    vec2 q = vec2(length(p.xz), -p.y);
    float d = length(q - c * max(dot(q, c), 0.0));
    return d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
}
"#;

const HELPER_SDF_HEART: &str = r#"float sdf_heart(vec3 p, float s) {
    vec3 q = p / s;
    float x = length(q.xz);
    float y = -(q.y - 0.5);
    float x2 = x * x;
    float y2 = y * y;
    float y3 = y2 * y;
    float cubic = x2 + y2 - 1.0;
    float iv = cubic * cubic * cubic - x2 * y3;
    if (iv <= 0.0) return -0.02 * s;
    return (pow(iv, 1.0 / 6.0) * 0.5 - 0.02) * s;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_transpile_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Hardcoded);

        assert!(shader.source.contains("float sdf_eval(vec3 p)"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = GlslShader::transpile(&box3d, GlslTranspileMode::Hardcoded);

        assert!(shader.source.contains("float sdf_eval(vec3 p)"));
        assert!(shader.source.contains("abs(p)"));
        assert!(shader.source.contains("vec3("));
    }

    #[test]
    fn test_unity_custom_function() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Hardcoded);
        let unity_code = shader.to_unity_custom_function();

        assert!(unity_code.contains("Unity Custom Function"));
        assert!(unity_code.contains("SdfEval_float"));
    }

    #[test]
    fn test_compute_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Hardcoded);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("#version 450"));
        assert!(compute.contains("layout(local_size_x = 256"));
        assert!(compute.contains("layout(std430"));
    }

    #[test]
    fn test_fragment_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Hardcoded);
        let fragment = shader.to_fragment_shader();

        assert!(fragment.contains("#version 450"));
        assert!(fragment.contains("calcNormal"));
        assert!(fragment.contains("fragColor"));
    }

    #[test]
    fn test_version_override() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile_with_version(&sphere, GlslTranspileMode::Hardcoded, 330);

        assert_eq!(shader.version, 330);
        let compute = shader.to_compute_shader();
        assert!(compute.contains("#version 330"));
    }

    // ============ Dynamic Mode Tests ============

    #[test]
    fn test_dynamic_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.5 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Dynamic);

        assert!(shader.source.contains("params["));
        assert_eq!(shader.param_layout.len(), 1);
        assert!((shader.param_layout[0] - 1.5).abs() < 1e-6);
        assert_eq!(shader.mode, GlslTranspileMode::Dynamic);
    }

    #[test]
    fn test_dynamic_smooth_union() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.3);
        let shader = GlslShader::transpile(&shape, GlslTranspileMode::Dynamic);

        // Should contain uniform block references
        assert!(shader.source.contains("params["));
        // Smooth union emits k and inv_k (Division Exorcism)
        let k_idx = shader.param_layout.iter().position(|&v| (v - 0.3).abs() < 1e-6);
        assert!(k_idx.is_some(), "k=0.3 should be in param_layout");
    }

    #[test]
    fn test_dynamic_extract_params() {
        let shape = SdfNode::sphere(1.5).translate(2.0, 3.0, 4.0);

        let shader = GlslShader::transpile(&shape, GlslTranspileMode::Dynamic);
        let extracted = GlslShader::extract_params(&shape);

        assert_eq!(shader.param_layout, extracted);
    }

    #[test]
    fn test_dynamic_compute_shader_has_ubo() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Dynamic);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("uniform SdfParams"));
        assert!(compute.contains("vec4 params[1024]"));
    }

    #[test]
    fn test_hardcoded_no_params() {
        let shape = SdfNode::sphere(1.0).translate(1.0, 2.0, 3.0);
        let shader = GlslShader::transpile(&shape, GlslTranspileMode::Hardcoded);

        assert!(!shader.source.contains("params["));
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn test_dynamic_fragment_shader_has_ubo() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere, GlslTranspileMode::Dynamic);
        let fragment = shader.to_fragment_shader();

        assert!(fragment.contains("uniform SdfParams"));
        assert!(fragment.contains("vec4 params[1024]"));
    }
}

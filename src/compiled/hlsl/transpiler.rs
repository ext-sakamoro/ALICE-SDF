//! HLSL Transpiler: SDF to High-Level Shading Language (Deep Fried Edition)
//!
//! This module converts SDF node trees to HLSL code for Unreal Engine 5,
//! DirectX, and other HLSL-compatible environments.
//!
//! # Deep Fried Optimizations
//!
//! - **Division Exorcism**: Replaces `/k` with `* inv_k` in smooth ops
//! - **Inline Smooth Ops**: Pre-computes reciprocals at transpile time
//! - **FMA-friendly**: Structures operations for GPU FMA units
//! - **Dynamic Parameters**: Constants read from cbuffer for zero-latency updates
//!
//! # Usage in Unreal Engine 5
//!
//! 1. Create a Custom Material Expression node
//! 2. Paste the generated HLSL code
//! 3. Connect the world position to `p` input
//!
//! Author: Moroya Sakamoto

use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
const FOLD_EPSILON: f32 = 1e-6;

/// Transpilation mode for HLSL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HlslTranspileMode {
    /// Constants are baked into the shader (Fastest execution, requires recompile on change)
    Hardcoded,
    /// Constants are read from a constant buffer (Fast update via cbuffer, good execution)
    Dynamic,
}

/// Generated HLSL shader code
#[derive(Debug, Clone)]
pub struct HlslShader {
    /// The generated HLSL source code
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
    /// Initial parameter values (empty for Hardcoded mode)
    pub param_layout: Vec<f32>,
    /// Transpilation mode used
    pub mode: HlslTranspileMode,
}

impl HlslShader {
    /// Transpile an SDF node tree to HLSL with the specified mode
    pub fn transpile(node: &SdfNode, mode: HlslTranspileMode) -> Self {
        let mut transpiler = HlslTranspiler::new(mode);
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        HlslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
            param_layout: transpiler.params,
            mode,
        }
    }

    /// Extract parameter values from an SDF node tree
    ///
    /// Returns the same parameter layout as produced by `transpile()` in Dynamic mode.
    /// Use this to update the constant buffer without recompiling the shader.
    pub fn extract_params(node: &SdfNode) -> Vec<f32> {
        let mut transpiler = HlslTranspiler::new(HlslTranspileMode::Dynamic);
        let _ = transpiler.transpile_node(node, "p");
        transpiler.params
    }

    /// Generate HLSL for UE5 Custom Material Expression
    ///
    /// Returns code suitable for pasting into a Custom node in UE5's Material Editor.
    pub fn to_ue5_custom_node(&self) -> String {
        let dynamic_note = if self.mode == HlslTranspileMode::Dynamic {
            "// NOTE: Dynamic mode - parameters are read from cbuffer SdfParams.\n// Set up a Material Parameter Collection or custom cbuffer to update params at runtime.\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Generated HLSL for UE5 Custom Node
// Input: float3 p (World Position)
// Output: float (SDF Distance)
{dynamic_note}
{source}
return sdf_eval(p);
"#,
            dynamic_note = dynamic_note,
            source = self.source
        )
    }

    /// Generate a complete HLSL compute shader for batch evaluation
    pub fn to_compute_shader(&self) -> String {
        let params_decl = if self.mode == HlslTranspileMode::Dynamic {
            "\ncbuffer SdfParams : register(b1) {\n    float4 params[1024]; // 4096 scalar floats\n};\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Generated HLSL Compute Shader ({mode:?} Mode)

struct InputPoint {{
    float x, y, z, _pad;
}};

struct OutputDistance {{
    float distance, _pad1, _pad2, _pad3;
}};

StructuredBuffer<InputPoint> input_points : register(t0);
RWStructuredBuffer<OutputDistance> output_distances : register(u0);
cbuffer Constants : register(b0) {{
    uint point_count;
}};
{params_decl}
{source}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    if (id.x >= point_count) return;

    InputPoint pt = input_points[id.x];
    float3 p = float3(pt.x, pt.y, pt.z);
    output_distances[id.x].distance = sdf_eval(p);
}}
"#,
            mode = self.mode,
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
struct HlslTranspiler {
    /// Counter for generating unique variable names
    var_counter: usize,
    /// Helper functions that need to be included
    helper_functions: Vec<&'static str>,
    /// Transpilation mode
    mode: HlslTranspileMode,
    /// Collected parameter values (Dynamic mode)
    params: Vec<f32>,
}

impl HlslTranspiler {
    fn new(mode: HlslTranspileMode) -> Self {
        HlslTranspiler {
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

    /// Register a float parameter and return its HLSL expression string.
    ///
    /// - Hardcoded: returns a literal like `"1.000000"`
    /// - Dynamic: pushes to param buffer and returns `"params[i].comp"`
    fn param(&mut self, value: f32) -> String {
        match self.mode {
            HlslTranspileMode::Hardcoded => format!("{:.6}", value),
            HlslTranspileMode::Dynamic => {
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
        writeln!(shader, "float sdf_eval(float3 p) {{").unwrap();
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
                    "    float3 {} = abs({}) - float3({}, {}, {});",
                    q_var, point_var, hx, hy, hz
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = length(max({}, float3(0.0, 0.0, 0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0);",
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
                    "    float2 {} = float2(length({}.xz) - {}, abs({}.y) - {});",
                    d_var, point_var, r_s, point_var, h_s
                )
                .unwrap();
                writeln!(
                    code,
                    "    float {} = min(max({}.x, {}.y), 0.0) + length(max({}, float2(0.0, 0.0)));",
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
                    "    float2 {} = float2(length({}.xz) - {}, {}.y);",
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
                    "    float {} = dot({}, float3({}, {}, {})) + {};",
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
                    "    float3 {} = {} - float3({}, {}, {});",
                    pa_var, point_var, pax, pay, paz
                )
                .unwrap();
                writeln!(
                    code,
                    "    float3 {} = float3({}, {}, {}) - float3({}, {}, {});",
                    ba_var, pbx, pby, pbz, pax2, pay2, paz2
                )
                .unwrap();
                // ★ Deep Fried: Zero-safe guard for degenerate capsule (a == b)
                writeln!(
                    code,
                    "    float {} = clamp(dot({}, {}) / max(dot({}, {}), 1e-10), 0.0, 1.0);",
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
                    "    float2 {} = float2({} - min({}, ({}.y < 0.0) ? {} : 0.0), abs({}.y) - {});",
                    ca_var, qx_var, qx_var, point_var, r_s, point_var, h_var
                ).unwrap();
                writeln!(code,
                    "    float {} = clamp((-{} * {} + ({} - {}.y) * {}) / {}, 0.0, 1.0);",
                    t_var, qx_var, k2x_s, h_var, point_var, k2y_s, denom_s
                ).unwrap();
                writeln!(code,
                    "    float2 {} = float2({} + {} * {}, {}.y - {} + {} * {});",
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
                    "    float {} = length({} * float3({}, {}, {}));",
                    k0_var, point_var, inv_rx, inv_ry, inv_rz
                ).unwrap();
                writeln!(code,
                    "    float {} = length({} * float3({}, {}, {}));",
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
                    "    float {} = sdf_triangle({}, float3({}, {}, {}), float3({}, {}, {}), float3({}, {}, {}));",
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
                    "    float {} = sdf_bezier({}, float3({}, {}, {}), float3({}, {}, {}), float3({}, {}, {}), {});",
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
                writeln!(code, "    float3 {} = abs({}) - float3({}, {}, {});", q_var, point_var, hx, hy, hz).unwrap();
                writeln!(code, "    float {} = length(max({}, float3(0.0, 0.0, 0.0))) + min(max({}.x, max({}.y, {}.z)), 0.0) - {};", var, q_var, q_var, q_var, q_var, rr).unwrap();
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
                writeln!(code, "    float3 {} = {} * {};", sp_var, point_var, sc).unwrap();
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
                if self.mode == HlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
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

                if self.mode == HlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
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

                if self.mode == HlslTranspileMode::Hardcoded && k.abs() < FOLD_EPSILON {
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
                    "    float3 {} = {} - float3({}, {}, {});",
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
                    "    float3 {} = quat_rotate({}, float4({}, {}, {}, {}));",
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
                    "    float3 {} = {} * {};",
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
                    "    float3 {} = {} * float3({}, {}, {});",
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
                    "    float3 {} = float3({} * {}.x - {} * {}.z, {}.y, {} * {}.x + {} * {}.z);",
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
                    "    float3 {} = float3({} * {}.x + {} * {}.y, {} * {}.y - {} * {}.x, {}.z);",
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
                    "    float3 {} = {} - clamp({}, float3({}, {}, {}), float3({}, {}, {}));",
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
                    "    float3 {} = fmod({} + float3({}, {}, {}) * 0.5, float3({}, {}, {})) - float3({}, {}, {}) * 0.5;",
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
                    "    float3 {} = clamp(round({} / float3({}, {}, {})), float3({}, {}, {}), float3({}, {}, {}));",
                    r_var, point_var, sx, sy, sz, ncx, ncy, ncz, pcx, pcy, pcz
                )
                .unwrap();
                writeln!(
                    code,
                    "    float3 {} = {} - float3({}, {}, {}) * {};",
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
                    "    float {} = hash_noise_3d({} * {}, (uint){});",
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
                    "    float3 {} = float3({}, {}, {});",
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
                    "    float3 {} = float3({}, {}.y, 0.0);",
                    new_p, q_var, point_var
                ).unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Extrude { child, half_height } => {
                let hh_s = self.param(*half_height);
                let flat_p = self.next_var();
                writeln!(code,
                    "    float3 {} = float3({}.x, {}.y, 0.0);",
                    flat_p, point_var, point_var
                ).unwrap();
                let d = self.transpile_node_inner(child, &flat_p, code);
                let w_var = self.next_var();
                let var = self.next_var();
                writeln!(code,
                    "    float2 {} = float2({}, abs({}.z) - {});",
                    w_var, d, point_var, hh_s
                ).unwrap();
                writeln!(code,
                    "    float {} = min(max({}.x, {}.y), 0.0) + length(max({}, float2(0.0, 0.0)));",
                    var, w_var, w_var, w_var
                ).unwrap();
                var
            }

            SdfNode::Taper { child, factor } => {
                let new_p = self.next_var();
                let f = self.param(*factor);
                writeln!(code,
                    "    float {np}_s = 1.0 / (1.0 - {p}.y * {f});",
                    np = new_p, p = point_var, f = f
                ).unwrap();
                writeln!(code,
                    "    float3 {} = float3({}.x * {np}_s, {}.y, {}.z * {np}_s);",
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
                    "    float {} = sin({}.x * 5.0) * sin({}.y * 5.0) * sin({}.z * 5.0);",
                    disp_var, point_var, point_var, point_var
                ).unwrap();
                writeln!(code,
                    "    float {} = {} + {} * {};",
                    var, d, disp_var, s
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for PolarRepeat
            SdfNode::PolarRepeat { child, count } => {
                let new_p = self.next_var();
                let count_f = *count as f32;
                let sector_angle = std::f32::consts::TAU / count_f;
                let sa = self.param(sector_angle);
                writeln!(code,
                    "    float {np}_a = atan2({p}.z, {p}.x) + {sa} * 0.5;",
                    np = new_p, p = point_var, sa = sa
                ).unwrap();
                writeln!(code,
                    "    float {np}_r = length({p}.xz);",
                    np = new_p, p = point_var
                ).unwrap();
                writeln!(code,
                    "    float {np}_am = fmod({np}_a + 100.0 * {sa}, {sa}) - {sa} * 0.5;",
                    np = new_p, sa = sa
                ).unwrap();
                writeln!(code,
                    "    float3 {} = float3({np}_r * cos({np}_am), {p}.y, {np}_r * sin({np}_am));",
                    new_p, np = new_p, p = point_var
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

// Helper function definitions for HLSL
const HELPER_HASH_NOISE: &str = r#"float hash_noise_3d(float3 p, uint seed) {
    float3 f = frac(p);
    float3 i = floor(p);
    float3 u = f * f * (3.0 - 2.0 * f);
    float s = (float)seed;
    float n000 = frac(sin(dot(i, float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n100 = frac(sin(dot(i + float3(1,0,0), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n010 = frac(sin(dot(i + float3(0,1,0), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n110 = frac(sin(dot(i + float3(1,1,0), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n001 = frac(sin(dot(i + float3(0,0,1), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n101 = frac(sin(dot(i + float3(1,0,1), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n011 = frac(sin(dot(i + float3(0,1,1), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float n111 = frac(sin(dot(i + float3(1,1,1), float3(127.1, 311.7, 74.7)) + s) * 43758.5453);
    float c00 = lerp(n000, n100, u.x);
    float c10 = lerp(n010, n110, u.x);
    float c01 = lerp(n001, n101, u.x);
    float c11 = lerp(n011, n111, u.x);
    float c0 = lerp(c00, c10, u.y);
    float c1 = lerp(c01, c11, u.y);
    return lerp(c0, c1, u.z) * 2.0 - 1.0;
}"#;

const HELPER_QUAT_ROTATE: &str = r#"float3 quat_rotate(float3 v, float4 q) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

const HELPER_SDF_ROUNDED_CONE: &str = r#"float sdf_rounded_cone(float3 p, float r1, float r2, float h) {
    float qx = length(p.xz);
    float qy = p.y + h;
    float ht = h * 2.0;
    float b = (r1 - r2) / ht;
    float a = sqrt(1.0 - b * b);
    float k = qx * (-b) + qy * a;
    if (k < 0.0) return length(float2(qx, qy)) - r1;
    if (k > a * ht) return length(float2(qx, qy - ht)) - r2;
    return qx * a + qy * b - r1;
}
"#;

const HELPER_SDF_PYRAMID: &str = r#"float sdf_pyramid(float3 p, float h) {
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

const HELPER_SDF_OCTAHEDRON: &str = r#"float sdf_octahedron(float3 p, float s) {
    float3 ap = abs(p);
    float m = ap.x + ap.y + ap.z - s;
    float3 q;
    if (3.0 * ap.x < m) q = ap;
    else if (3.0 * ap.y < m) q = float3(ap.y, ap.z, ap.x);
    else if (3.0 * ap.z < m) q = float3(ap.z, ap.x, ap.y);
    else return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(float3(q.x, q.y - s + k, q.z - k));
}
"#;

const HELPER_SDF_HEX_PRISM: &str = r#"float sdf_hex_prism(float3 p, float hex_r, float h) {
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
    return min(max(d_xy, d_z), 0.0) + length(max(float2(d_xy, d_z), float2(0.0, 0.0)));
}
"#;

const HELPER_SDF_TRIANGLE: &str = r#"float sdf_triangle(float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a; float3 pa = p - a;
    float3 cb = c - b; float3 pb = p - b;
    float3 ac = a - c; float3 pc = p - c;
    float3 nor = cross(ba, ac);
    float sign_check = sign(dot(cross(ba, nor), pa)) +
                       sign(dot(cross(cb, nor), pb)) +
                       sign(dot(cross(ac, nor), pc));
    if (sign_check < 2.0) {
        float d1 = dot(ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa,
                       ba * clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0) - pa);
        float d2 = dot(cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb,
                       cb * clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0) - pb);
        float d3 = dot(ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc,
                       ac * clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0) - pc);
        return sqrt(min(min(d1, d2), d3));
    } else {
        return sqrt(dot(nor, pa) * dot(nor, pa) / dot(nor, nor));
    }
}
"#;

const HELPER_SDF_BEZIER: &str = r#"float sdf_bezier(float3 pos, float3 A, float3 B, float3 C, float rad) {
    float3 a = B - A;
    float3 b = A - 2.0 * B + C;
    float3 c = a * 2.0;
    float3 d = A - pos;
    float kk = 1.0 / max(dot(b, b), 1e-10);
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p = ky - kx * kx;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float p3 = p * p * p;
    float q2 = q * q;
    float h = q2 + 4.0 * p3;
    float res;
    if (h >= 0.0) {
        float sh = sqrt(h);
        float2 x = (float2(sh, -sh) - q) * 0.5;
        float2 uv = sign(x) * pow(abs(x), float2(1.0 / 3.0, 1.0 / 3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        float3 qp = d + (c + b * t) * t;
        res = dot(qp, qp);
    } else {
        float z = sqrt(-p);
        float v = acos(q / (p * z * 2.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        float t0 = clamp(( m + m) * z - kx, 0.0, 1.0);
        float t1 = clamp((-n - m) * z - kx, 0.0, 1.0);
        float3 qp0 = d + (c + b * t0) * t0;
        float3 qp1 = d + (c + b * t1) * t1;
        res = min(dot(qp0, qp0), dot(qp1, qp1));
    }
    return sqrt(res) - rad;
}
"#;

const HELPER_SDF_LINK: &str = r#"float sdf_link(float3 p, float le, float r1, float r2) {
    float qx = p.x;
    float qy = max(abs(p.y) - le, 0.0);
    float qz = p.z;
    float xy_len = sqrt(qx * qx + qy * qy) - r1;
    return sqrt(xy_len * xy_len + qz * qz) - r2;
}
"#;

const HELPER_SDF_CAPPED_CONE: &str = r#"float sdf_capped_cone(float3 p, float h, float r1, float r2) {
    float qx = length(p.xz);
    float qy = p.y;
    float2 k1 = float2(r2, h);
    float2 k2 = float2(r2 - r1, 2.0 * h);
    float min_r = (qy < 0.0) ? r1 : r2;
    float2 ca = float2(qx - min(qx, min_r), abs(qy) - h);
    float2 q = float2(qx, qy);
    float2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}
"#;

const HELPER_SDF_CAPPED_TORUS: &str = r#"float sdf_capped_torus(float3 p, float ra, float rb, float an) {
    float2 sc = float2(sin(an), cos(an));
    float px = abs(p.x);
    float k = (sc.y * px > sc.x * p.y) ? dot(float2(px, p.y), sc) : length(float2(px, p.y));
    return sqrt(px * px + p.y * p.y + p.z * p.z + ra * ra - 2.0 * ra * k) - rb;
}
"#;

const HELPER_SDF_ROUNDED_CYLINDER: &str = r#"float sdf_rounded_cylinder(float3 p, float r, float rr, float h) {
    float2 d = float2(length(p.xz) - r + rr, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, float2(0.0, 0.0))) - rr;
}
"#;

const HELPER_SDF_TRIANGULAR_PRISM: &str = r#"float sdf_triangular_prism(float3 p, float w, float h) {
    float3 q = abs(p);
    return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - w * 0.5);
}
"#;

const HELPER_SDF_CUT_SPHERE: &str = r#"float sdf_cut_sphere(float3 p, float r, float h) {
    float w = sqrt(max(r * r - h * h, 0.0));
    float2 q = float2(length(p.xz), p.y);
    float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
    if (s < 0.0) return length(q) - r;
    if (q.x < w) return h - q.y;
    return length(q - float2(w, h));
}
"#;

const HELPER_SDF_CUT_HOLLOW_SPHERE: &str = r#"float sdf_cut_hollow_sphere(float3 p, float r, float h, float t) {
    float w = sqrt(max(r * r - h * h, 0.0));
    float2 q = float2(length(p.xz), p.y);
    if (h * q.x < w * q.y) return length(q - float2(w, h)) - t;
    return abs(length(q) - r) - t;
}
"#;

const HELPER_SDF_DEATH_STAR: &str = r#"float sdf_death_star(float3 p, float ra, float rb, float d) {
    float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    float b = sqrt(max(ra * ra - a * a, 0.0));
    float2 q = float2(p.x, length(p.yz));
    if (q.x * b - q.y * a > d * max(b - q.y, 0.0))
        return length(q - float2(a, b));
    return max(length(q) - ra, -(length(q - float2(d, 0.0)) - rb));
}
"#;

const HELPER_SDF_SOLID_ANGLE: &str = r#"float sdf_solid_angle(float3 p, float an, float ra) {
    float2 c = float2(sin(an), cos(an));
    float2 q = float2(length(p.xz), p.y);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q, c), 0.0, ra));
    return max(l, m * sign(c.y * q.x - c.x * q.y));
}
"#;

const HELPER_SDF_RHOMBUS: &str = r#"float ndot_rh(float2 a, float2 b) {
    return a.x * b.x - a.y * b.y;
}
float sdf_rhombus(float3 p, float la, float lb, float h, float ra) {
    float3 ap = abs(p);
    float2 b = float2(la, lb);
    float f = clamp(ndot_rh(b, b - 2.0 * ap.xz) / dot(b, b), -1.0, 1.0);
    float dxz = length(ap.xz - 0.5 * b * float2(1.0 - f, 1.0 + f)) * sign(ap.x * b.y + ap.z * b.x - b.x * b.y) - ra;
    float2 q = float2(dxz, ap.y - h);
    return min(max(q.x, q.y), 0.0) + length(max(q, float2(0.0, 0.0)));
}
"#;

const HELPER_SDF_HORSESHOE: &str = r#"float sdf_horseshoe(float3 pos, float an, float r, float le, float w, float t) {
    float2 c = float2(cos(an), sin(an));
    float px = abs(pos.x);
    float l = length(float2(px, pos.y));
    float qx = -c.x * px + c.y * pos.y;
    float qy = c.y * px + c.x * pos.y;
    if (!(qy > 0.0 || qx > 0.0)) qx = l * sign(-c.x);
    if (qx <= 0.0) qy = l;
    qx = abs(qx) - le;
    qy = abs(qy - r);
    float e = length(max(float2(qx, qy), float2(0.0, 0.0))) + min(max(qx, qy), 0.0);
    float2 d = abs(float2(e, pos.z)) - float2(w, t);
    return min(max(d.x, d.y), 0.0) + length(max(d, float2(0.0, 0.0)));
}
"#;

const HELPER_SDF_VESICA: &str = r#"float sdf_vesica(float3 p, float r, float d) {
    float px = abs(p.x);
    float py = length(p.yz);
    float b = sqrt(max(r * r - d * d, 0.0));
    if ((py - b) * d > px * b) return length(float2(px, py - b));
    return length(float2(px - d, py)) - r;
}
"#;

const HELPER_SDF_INFINITE_CONE: &str = r#"float sdf_infinite_cone(float3 p, float an) {
    float2 c = float2(sin(an), cos(an));
    float2 q = float2(length(p.xz), -p.y);
    float d = length(q - c * max(dot(q, c), 0.0));
    return d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
}
"#;

const HELPER_SDF_HEART: &str = r#"float sdf_heart(float3 p, float s) {
    float3 q = p / s;
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
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Hardcoded);

        assert!(shader.source.contains("float sdf_eval(float3 p)"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = HlslShader::transpile(&box3d, HlslTranspileMode::Hardcoded);

        assert!(shader.source.contains("float sdf_eval(float3 p)"));
        assert!(shader.source.contains("abs(p)"));
        assert!(shader.source.contains("float3("));
    }

    #[test]
    fn test_ue5_custom_node() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Hardcoded);
        let ue5_code = shader.to_ue5_custom_node();

        assert!(ue5_code.contains("UE5 Custom Node"));
        assert!(ue5_code.contains("return sdf_eval(p);"));
    }

    #[test]
    fn test_compute_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Hardcoded);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("[numthreads(256, 1, 1)]"));
        assert!(compute.contains("StructuredBuffer"));
        assert!(compute.contains("RWStructuredBuffer"));
    }

    // ============ Dynamic Mode Tests ============

    #[test]
    fn test_dynamic_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.5 };
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Dynamic);

        assert!(shader.source.contains("params["));
        assert_eq!(shader.param_layout.len(), 1);
        assert!((shader.param_layout[0] - 1.5).abs() < 1e-6);
        assert_eq!(shader.mode, HlslTranspileMode::Dynamic);
    }

    #[test]
    fn test_dynamic_smooth_union() {
        let shape = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.3);
        let shader = HlslShader::transpile(&shape, HlslTranspileMode::Dynamic);

        // Should contain cbuffer references
        assert!(shader.source.contains("params["));
        // Smooth union emits k and inv_k (Division Exorcism)
        let k_idx = shader.param_layout.iter().position(|&v| (v - 0.3).abs() < 1e-6);
        assert!(k_idx.is_some(), "k=0.3 should be in param_layout");
    }

    #[test]
    fn test_dynamic_extract_params() {
        let shape = SdfNode::sphere(1.5).translate(2.0, 3.0, 4.0);

        let shader = HlslShader::transpile(&shape, HlslTranspileMode::Dynamic);
        let extracted = HlslShader::extract_params(&shape);

        assert_eq!(shader.param_layout, extracted);
    }

    #[test]
    fn test_dynamic_compute_shader_has_cbuffer() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Dynamic);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("cbuffer SdfParams"));
        assert!(compute.contains("float4 params[1024]"));
    }

    #[test]
    fn test_hardcoded_no_params() {
        let shape = SdfNode::sphere(1.0).translate(1.0, 2.0, 3.0);
        let shader = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);

        assert!(!shader.source.contains("params["));
        assert!(shader.param_layout.is_empty());
    }

    #[test]
    fn test_dynamic_ue5_has_note() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere, HlslTranspileMode::Dynamic);
        let ue5_code = shader.to_ue5_custom_node();

        assert!(ue5_code.contains("Dynamic mode"));
    }
}

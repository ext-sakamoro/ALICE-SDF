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

/// Generated GLSL shader code
#[derive(Debug, Clone)]
pub struct GlslShader {
    /// The generated GLSL source code
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
    /// GLSL version used (default: 450)
    pub version: u32,
}

impl GlslShader {
    /// Transpile an SDF node tree to GLSL
    pub fn transpile(node: &SdfNode) -> Self {
        Self::transpile_with_version(node, 450)
    }

    /// Transpile an SDF node tree to GLSL with specific version
    pub fn transpile_with_version(node: &SdfNode, version: u32) -> Self {
        let mut transpiler = GlslTranspiler::new();
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        GlslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
            version,
        }
    }

    /// Generate GLSL for Unity Shader Graph Custom Function
    ///
    /// Returns code suitable for pasting into a Custom Function node in Unity.
    pub fn to_unity_custom_function(&self) -> String {
        format!(
            r#"// ALICE-SDF Generated GLSL for Unity Custom Function
// Input: float3 p (World Position)
// Output: float (SDF Distance)

{}

void SdfEval_float(float3 p, out float distance) {{
    distance = sdf_eval(p);
}}
"#,
            self.source
        )
    }

    /// Generate a complete GLSL compute shader for batch evaluation
    pub fn to_compute_shader(&self) -> String {
        format!(
            r#"#version {}

// ALICE-SDF Generated GLSL Compute Shader
// Evaluates SDF at multiple points in parallel

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

{}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) return;

    InputPoint pt = input_points[idx];
    vec3 p = vec3(pt.x, pt.y, pt.z);
    output_distances[idx].distance = sdf_eval(p);
}}
"#,
            self.version, self.source
        )
    }

    /// Generate a fragment shader for raymarching (Shadertoy-compatible)
    pub fn to_fragment_shader(&self) -> String {
        format!(
            r#"#version {}

// ALICE-SDF Generated GLSL Fragment Shader
// Compatible with Shadertoy-style rendering

precision highp float;

uniform vec2 iResolution;
uniform float iTime;

out vec4 fragColor;

{}

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
            self.version, self.source
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
}

impl GlslTranspiler {
    fn new() -> Self {
        GlslTranspiler {
            var_counter: 0,
            helper_functions: Vec::new(),
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

    fn generate_shader(&self, body: &str) -> String {
        let mut shader = String::new();

        // Add helper functions
        for helper in &self.helper_functions {
            match *helper {
                "quat_rotate" => {
                    shader.push_str(HELPER_QUAT_ROTATE);
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
                let var = self.next_var();
                writeln!(
                    code,
                    "    float {} = length({}) - {:.6};",
                    var, point_var, radius
                )
                .unwrap();
                var
            }

            SdfNode::Box3d { half_extents } => {
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = abs({}) - vec3({:.6}, {:.6}, {:.6});",
                    q_var, point_var, half_extents.x, half_extents.y, half_extents.z
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
                let d_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec2 {} = vec2(length({}.xz) - {:.6}, abs({}.y) - {:.6});",
                    d_var, point_var, radius, point_var, half_height
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
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    vec2 {} = vec2(length({}.xz) - {:.6}, {}.y);",
                    q_var, point_var, major_radius, point_var
                )
                .unwrap();
                writeln!(code, "    float {} = length({}) - {:.6};", var, q_var, minor_radius).unwrap();
                var
            }

            SdfNode::Plane { normal, distance } => {
                let var = self.next_var();
                writeln!(
                    code,
                    "    float {} = dot({}, vec3({:.6}, {:.6}, {:.6})) + {:.6};",
                    var, point_var, normal.x, normal.y, normal.z, distance
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

                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({:.6}, {:.6}, {:.6});",
                    pa_var, point_var, point_a.x, point_a.y, point_a.z
                )
                .unwrap();
                writeln!(
                    code,
                    "    vec3 {} = vec3({:.6}, {:.6}, {:.6}) - vec3({:.6}, {:.6}, {:.6});",
                    ba_var,
                    point_b.x, point_b.y, point_b.z,
                    point_a.x, point_a.y, point_a.z
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
                    "    float {} = length({} - {} * {}) - {:.6};",
                    var, pa_var, ba_var, h_var, radius
                )
                .unwrap();
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

                if k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = 1.0 / k;
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    float {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, d_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = min({}, {}) - {} * {} * {:.6} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothIntersection
            SdfNode::SmoothIntersection { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = 1.0 / k;
                let h_var = self.next_var();

                writeln!(
                    code,
                    "    float {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, d_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = max({}, {}) + {} * {} * {:.6} * 0.25;",
                    var, d_a, d_b, h_var, h_var, k
                ).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for SmoothSubtraction
            SdfNode::SmoothSubtraction { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                if k.abs() < FOLD_EPSILON {
                    writeln!(code, "    float {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = 1.0 / k;
                let h_var = self.next_var();
                let neg_b = self.next_var();

                writeln!(code, "    float {} = -{};", neg_b, d_b).unwrap();
                writeln!(
                    code,
                    "    float {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, neg_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    float {} = max({}, {}) + {} * {} * {:.6} * 0.25;",
                    var, d_a, neg_b, h_var, h_var, k
                ).unwrap();
                var
            }

            // ============ Transforms ============

            SdfNode::Translate { child, offset } => {
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({:.6}, {:.6}, {:.6});",
                    new_p, point_var, offset.x, offset.y, offset.z
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Rotate { child, rotation } => {
                self.ensure_helper("quat_rotate");
                let inv_rot = rotation.inverse();
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = quat_rotate({}, vec4({:.6}, {:.6}, {:.6}, {:.6}));",
                    new_p, point_var, inv_rot.x, inv_rot.y, inv_rot.z, inv_rot.w
                )
                .unwrap();
                self.transpile_node_inner(child, &new_p, code)
            }

            SdfNode::Scale { child, factor } => {
                let new_p = self.next_var();
                let inv_factor = 1.0 / factor;
                writeln!(
                    code,
                    "    vec3 {} = {} * {:.6};",
                    new_p, point_var, inv_factor
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    float {} = {} * {:.6};", var, d, factor).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for ScaleNonUniform
            SdfNode::ScaleNonUniform { child, factors } => {
                let new_p = self.next_var();
                let inv_x = 1.0 / factors.x;
                let inv_y = 1.0 / factors.y;
                let inv_z = 1.0 / factors.z;
                writeln!(
                    code,
                    "    vec3 {} = {} * vec3({:.6}, {:.6}, {:.6});",
                    new_p, point_var, inv_x, inv_y, inv_z
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let min_scale = factors.x.min(factors.y).min(factors.z);
                let var = self.next_var();
                writeln!(code, "    float {} = {} * {:.6};", var, d, min_scale).unwrap();
                var
            }

            // ============ Modifiers ============

            SdfNode::Twist { child, strength } => {
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();

                writeln!(
                    code,
                    "    float {} = {:.6} * {}.y;",
                    angle_var, strength, point_var
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
                let angle_var = self.next_var();
                let c_var = self.next_var();
                let s_var = self.next_var();
                let new_p = self.next_var();

                writeln!(
                    code,
                    "    float {} = {:.6} * {}.x;",
                    angle_var, curvature, point_var
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
                let var = self.next_var();
                writeln!(code, "    float {} = {} - {:.6};", var, d, radius).unwrap();
                var
            }

            SdfNode::Onion { child, thickness } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                writeln!(code, "    float {} = abs({}) - {:.6};", var, d, thickness).unwrap();
                var
            }

            SdfNode::Elongate { child, amount } => {
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = {} - clamp({}, vec3({:.6}, {:.6}, {:.6}), vec3({:.6}, {:.6}, {:.6}));",
                    q_var, point_var, point_var,
                    -amount.x, -amount.y, -amount.z,
                    amount.x, amount.y, amount.z
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    vec3 {} = mod({} + vec3({:.6}, {:.6}, {:.6}) * 0.5, vec3({:.6}, {:.6}, {:.6})) - vec3({:.6}, {:.6}, {:.6}) * 0.5;",
                    q_var, point_var,
                    spacing.x, spacing.y, spacing.z,
                    spacing.x, spacing.y, spacing.z,
                    spacing.x, spacing.y, spacing.z
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
                writeln!(
                    code,
                    "    vec3 {} = clamp(round({} / vec3({:.6}, {:.6}, {:.6})), vec3({:.6}, {:.6}, {:.6}), vec3({:.6}, {:.6}, {:.6}));",
                    r_var, point_var,
                    spacing.x, spacing.y, spacing.z,
                    -(count[0] as f32), -(count[1] as f32), -(count[2] as f32),
                    count[0] as f32, count[1] as f32, count[2] as f32
                )
                .unwrap();
                writeln!(
                    code,
                    "    vec3 {} = {} - vec3({:.6}, {:.6}, {:.6}) * {};",
                    q_var, point_var, spacing.x, spacing.y, spacing.z, r_var
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::Noise { child, .. } => {
                // Noise is not supported in GLSL transpiler
                self.transpile_node_inner(child, point_var, code)
            }
        }
    }
}

// Helper function definitions for GLSL
const HELPER_QUAT_ROTATE: &str = r#"vec3 quat_rotate(vec3 v, vec4 q) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_transpile_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere);

        assert!(shader.source.contains("float sdf_eval(vec3 p)"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = GlslShader::transpile(&box3d);

        assert!(shader.source.contains("float sdf_eval(vec3 p)"));
        assert!(shader.source.contains("abs(p)"));
        assert!(shader.source.contains("vec3("));
    }

    #[test]
    fn test_unity_custom_function() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere);
        let unity_code = shader.to_unity_custom_function();

        assert!(unity_code.contains("Unity Custom Function"));
        assert!(unity_code.contains("SdfEval_float"));
    }

    #[test]
    fn test_compute_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("#version 450"));
        assert!(compute.contains("layout(local_size_x = 256"));
        assert!(compute.contains("layout(std430"));
    }

    #[test]
    fn test_fragment_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile(&sphere);
        let fragment = shader.to_fragment_shader();

        assert!(fragment.contains("#version 450"));
        assert!(fragment.contains("calcNormal"));
        assert!(fragment.contains("fragColor"));
    }

    #[test]
    fn test_version_override() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = GlslShader::transpile_with_version(&sphere, 330);

        assert_eq!(shader.version, 330);
        let compute = shader.to_compute_shader();
        assert!(compute.contains("#version 330"));
    }
}

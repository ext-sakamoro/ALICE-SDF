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

/// Generated HLSL shader code
#[derive(Debug, Clone)]
pub struct HlslShader {
    /// The generated HLSL source code
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
}

impl HlslShader {
    /// Transpile an SDF node tree to HLSL
    pub fn transpile(node: &SdfNode) -> Self {
        let mut transpiler = HlslTranspiler::new();
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        HlslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
        }
    }

    /// Generate HLSL for UE5 Custom Material Expression
    ///
    /// Returns code suitable for pasting into a Custom node in UE5's Material Editor.
    pub fn to_ue5_custom_node(&self) -> String {
        format!(
            r#"// ALICE-SDF Generated HLSL for UE5 Custom Node
// Input: float3 p (World Position)
// Output: float (SDF Distance)

{}
return sdf_eval(p);
"#,
            self.source
        )
    }

    /// Generate a complete HLSL compute shader for batch evaluation
    pub fn to_compute_shader(&self) -> String {
        format!(
            r#"// ALICE-SDF Generated HLSL Compute Shader
// Evaluates SDF at multiple points in parallel

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

{}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {{
    if (id.x >= point_count) return;

    InputPoint pt = input_points[id.x];
    float3 p = float3(pt.x, pt.y, pt.z);
    output_distances[id.x].distance = sdf_eval(p);
}}
"#,
            self.source
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
}

impl HlslTranspiler {
    fn new() -> Self {
        HlslTranspiler {
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
                    "    float3 {} = abs({}) - float3({:.6}, {:.6}, {:.6});",
                    q_var, point_var, half_extents.x, half_extents.y, half_extents.z
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
                let d_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    float2 {} = float2(length({}.xz) - {:.6}, abs({}.y) - {:.6});",
                    d_var, point_var, radius, point_var, half_height
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
                let q_var = self.next_var();
                let var = self.next_var();
                writeln!(
                    code,
                    "    float2 {} = float2(length({}.xz) - {:.6}, {}.y);",
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
                    "    float {} = dot({}, float3({:.6}, {:.6}, {:.6})) + {:.6};",
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
                    "    float3 {} = {} - float3({:.6}, {:.6}, {:.6});",
                    pa_var, point_var, point_a.x, point_a.y, point_a.z
                )
                .unwrap();
                writeln!(
                    code,
                    "    float3 {} = float3({:.6}, {:.6}, {:.6}) - float3({:.6}, {:.6}, {:.6});",
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
                    "    float3 {} = {} - float3({:.6}, {:.6}, {:.6});",
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
                    "    float3 {} = quat_rotate({}, float4({:.6}, {:.6}, {:.6}, {:.6}));",
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
                    "    float3 {} = {} * {:.6};",
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
                    "    float3 {} = {} * float3({:.6}, {:.6}, {:.6});",
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
                    "    float3 {} = float3({} * {}.x - {} * {}.z, {}.y, {} * {}.x + {} * {}.z);",
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
                    "    float3 {} = float3({} * {}.x + {} * {}.y, {} * {}.y - {} * {}.x, {}.z);",
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
                    "    float3 {} = {} - clamp({}, float3({:.6}, {:.6}, {:.6}), float3({:.6}, {:.6}, {:.6}));",
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
                    "    float3 {} = fmod({} + float3({:.6}, {:.6}, {:.6}) * 0.5, float3({:.6}, {:.6}, {:.6})) - float3({:.6}, {:.6}, {:.6}) * 0.5;",
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
                    "    float3 {} = clamp(round({} / float3({:.6}, {:.6}, {:.6})), float3({:.6}, {:.6}, {:.6}), float3({:.6}, {:.6}, {:.6}));",
                    r_var, point_var,
                    spacing.x, spacing.y, spacing.z,
                    -(count[0] as f32), -(count[1] as f32), -(count[2] as f32),
                    count[0] as f32, count[1] as f32, count[2] as f32
                )
                .unwrap();
                writeln!(
                    code,
                    "    float3 {} = {} - float3({:.6}, {:.6}, {:.6}) * {};",
                    q_var, point_var, spacing.x, spacing.y, spacing.z, r_var
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::Noise { child, .. } => {
                // Noise is not supported in HLSL transpiler
                self.transpile_node_inner(child, point_var, code)
            }
        }
    }
}

// Helper function definitions for HLSL
const HELPER_QUAT_ROTATE: &str = r#"float3 quat_rotate(float3 v, float4 q) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_transpile_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere);

        assert!(shader.source.contains("float sdf_eval(float3 p)"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = HlslShader::transpile(&box3d);

        assert!(shader.source.contains("float sdf_eval(float3 p)"));
        assert!(shader.source.contains("abs(p)"));
        assert!(shader.source.contains("float3("));
    }

    #[test]
    fn test_ue5_custom_node() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere);
        let ue5_code = shader.to_ue5_custom_node();

        assert!(ue5_code.contains("UE5 Custom Node"));
        assert!(ue5_code.contains("return sdf_eval(p);"));
    }

    #[test]
    fn test_compute_shader() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = HlslShader::transpile(&sphere);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("[numthreads(256, 1, 1)]"));
        assert!(compute.contains("StructuredBuffer"));
        assert!(compute.contains("RWStructuredBuffer"));
    }
}

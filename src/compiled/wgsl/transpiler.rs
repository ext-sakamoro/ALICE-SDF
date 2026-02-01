//! WGSL Transpiler: SDF to WebGPU Shading Language (Deep Fried Edition)
//!
//! This module converts SDF node trees to WGSL code that can be executed
//! on the GPU for massively parallel evaluation.
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

/// Generated WGSL shader code
#[derive(Debug, Clone)]
pub struct WgslShader {
    /// The generated WGSL source code
    pub source: String,
    /// Number of helper functions generated
    pub helper_count: usize,
}

impl WgslShader {
    /// Transpile an SDF node tree to WGSL
    pub fn transpile(node: &SdfNode) -> Self {
        let mut transpiler = WgslTranspiler::new();
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        WgslShader {
            source,
            helper_count: transpiler.helper_functions.len(),
        }
    }

    /// Generate a complete compute shader for batch evaluation
    pub fn to_compute_shader(&self) -> String {
        format!(
            r#"// ALICE-SDF Generated Compute Shader
// Evaluates SDF at multiple points in parallel

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

{}

@compute @workgroup_size(256)
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
            self.source
        )
    }

    /// Get the SDF evaluation function only (for embedding in custom shaders)
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }
}

/// Internal transpiler state
struct WgslTranspiler {
    /// Counter for generating unique variable names
    var_counter: usize,
    /// Helper functions that need to be included
    helper_functions: Vec<&'static str>,
}

impl WgslTranspiler {
    fn new() -> Self {
        WgslTranspiler {
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
                writeln!(
                    code,
                    "    let {} = length({}) - {:.6};",
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
                    "    let {} = abs({}) - vec3<f32>({:.6}, {:.6}, {:.6});",
                    q_var, point_var, half_extents.x, half_extents.y, half_extents.z
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
                writeln!(
                    code,
                    "    let {} = vec2<f32>(length({}.xz) - {:.6}, abs({}.y) - {:.6});",
                    d_var, point_var, radius, point_var, half_height
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
                writeln!(
                    code,
                    "    let {} = vec2<f32>(length({}.xz) - {:.6}, {}.y);",
                    q_var, point_var, major_radius, point_var
                )
                .unwrap();
                writeln!(code, "    let {} = length({}) - {:.6};", var, q_var, minor_radius).unwrap();
                var
            }

            SdfNode::Plane { normal, distance } => {
                let var = self.next_var();
                writeln!(
                    code,
                    "    let {} = dot({}, vec3<f32>({:.6}, {:.6}, {:.6})) + {:.6};",
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
                    "    let {} = {} - vec3<f32>({:.6}, {:.6}, {:.6});",
                    pa_var, point_var, point_a.x, point_a.y, point_a.z
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = vec3<f32>({:.6}, {:.6}, {:.6}) - vec3<f32>({:.6}, {:.6}, {:.6});",
                    ba_var,
                    point_b.x, point_b.y, point_b.z,
                    point_a.x, point_a.y, point_a.z
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = clamp(dot({}, {}) / dot({}, {}), 0.0, 1.0);",
                    h_var, pa_var, ba_var, ba_var, ba_var
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = length({} - {} * {}) - {:.6};",
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
            // Pre-compute inv_k at transpile time, use * inv_k instead of / k
            SdfNode::SmoothUnion { a, b, k } => {
                let d_a = self.transpile_node_inner(a, point_var, code);
                let d_b = self.transpile_node_inner(b, point_var, code);
                let var = self.next_var();

                // Constant folding: if k ≈ 0, just use min()
                if k.abs() < FOLD_EPSILON {
                    writeln!(code, "    let {} = min({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                // Division Exorcism: compute 1/k at Rust compile time
                let inv_k = 1.0 / k;
                let h_var = self.next_var();

                // Inline smooth_min with multiplication:
                // h = max(k - abs(a - b), 0.0) * inv_k  (MUL instead of DIV!)
                // result = min(a, b) - h * h * k * 0.25
                writeln!(
                    code,
                    "    let {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, d_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = min({}, {}) - {} * {} * {:.6} * 0.25;",
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
                    writeln!(code, "    let {} = max({}, {});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = 1.0 / k;
                let h_var = self.next_var();

                // smooth_max(a, b, k) = -smooth_min(-a, -b, k)
                // Inline with division exorcism:
                writeln!(
                    code,
                    "    let {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, d_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = max({}, {}) + {} * {} * {:.6} * 0.25;",
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
                    writeln!(code, "    let {} = max({}, -{});", var, d_a, d_b).unwrap();
                    return var;
                }

                let inv_k = 1.0 / k;
                let h_var = self.next_var();
                let neg_b = self.next_var();

                // smooth_max(a, -b, k) with division exorcism:
                writeln!(code, "    let {} = -{};", neg_b, d_b).unwrap();
                writeln!(
                    code,
                    "    let {} = max({:.6} - abs({} - {}), 0.0) * {:.6};",
                    h_var, k, d_a, neg_b, inv_k
                ).unwrap();
                writeln!(
                    code,
                    "    let {} = max({}, {}) + {} * {} * {:.6} * 0.25;",
                    var, d_a, neg_b, h_var, h_var, k
                ).unwrap();
                var
            }

            // ============ Transforms ============

            SdfNode::Translate { child, offset } => {
                let new_p = self.next_var();
                writeln!(
                    code,
                    "    let {} = {} - vec3<f32>({:.6}, {:.6}, {:.6});",
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
                    "    let {} = quat_rotate({}, vec4<f32>({:.6}, {:.6}, {:.6}, {:.6}));",
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
                    "    let {} = {} * {:.6};",
                    new_p, point_var, inv_factor
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {:.6};", var, d, factor).unwrap();
                var
            }

            // ★ Deep Fried: Division Exorcism for ScaleNonUniform
            SdfNode::ScaleNonUniform { child, factors } => {
                let new_p = self.next_var();
                // Pre-compute reciprocals at transpile time
                let inv_x = 1.0 / factors.x;
                let inv_y = 1.0 / factors.y;
                let inv_z = 1.0 / factors.z;
                writeln!(
                    code,
                    "    let {} = {} * vec3<f32>({:.6}, {:.6}, {:.6});",
                    new_p, point_var, inv_x, inv_y, inv_z
                )
                .unwrap();
                let d = self.transpile_node_inner(child, &new_p, code);
                let min_scale = factors.x.min(factors.y).min(factors.z);
                let var = self.next_var();
                writeln!(code, "    let {} = {} * {:.6};", var, d, min_scale).unwrap();
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
                    "    let {} = {:.6} * {}.y;",
                    angle_var, strength, point_var
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

                writeln!(
                    code,
                    "    let {} = {:.6} * {}.x;",
                    angle_var, curvature, point_var
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
                writeln!(code, "    let {} = {} - {:.6};", var, d, radius).unwrap();
                var
            }

            SdfNode::Onion { child, thickness } => {
                let d = self.transpile_node_inner(child, point_var, code);
                let var = self.next_var();
                writeln!(code, "    let {} = abs({}) - {:.6};", var, d, thickness).unwrap();
                var
            }

            SdfNode::Elongate { child, amount } => {
                let q_var = self.next_var();
                writeln!(
                    code,
                    "    let {} = {} - clamp({}, vec3<f32>({:.6}, {:.6}, {:.6}), vec3<f32>({:.6}, {:.6}, {:.6}));",
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
                    "    let {} = ({} + vec3<f32>({:.6}, {:.6}, {:.6}) * 0.5) % vec3<f32>({:.6}, {:.6}, {:.6}) - vec3<f32>({:.6}, {:.6}, {:.6}) * 0.5;",
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
                    "    let {} = clamp(round({} / vec3<f32>({:.6}, {:.6}, {:.6})), vec3<f32>({:.6}, {:.6}, {:.6}), vec3<f32>({:.6}, {:.6}, {:.6}));",
                    r_var, point_var,
                    spacing.x, spacing.y, spacing.z,
                    -(count[0] as f32), -(count[1] as f32), -(count[2] as f32),
                    count[0] as f32, count[1] as f32, count[2] as f32
                )
                .unwrap();
                writeln!(
                    code,
                    "    let {} = {} - vec3<f32>({:.6}, {:.6}, {:.6}) * {};",
                    q_var, point_var, spacing.x, spacing.y, spacing.z, r_var
                )
                .unwrap();
                self.transpile_node_inner(child, &q_var, code)
            }

            SdfNode::Noise { child, .. } => {
                // Noise is not supported in WGSL transpiler (would require texture sampling)
                // Fall back to evaluating just the child
                self.transpile_node_inner(child, point_var, code)
            }
        }
    }
}

// Helper function definitions
// Note: These are kept for backwards compatibility but the transpiler now
// inlines smooth operations with pre-computed reciprocals for better performance.
// If you need a helper function, use smooth_min_fast which accepts inv_k directly.

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

const HELPER_QUAT_ROTATE: &str = r#"fn quat_rotate(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}"#;

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    #[test]
    fn test_transpile_sphere() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("length(p)"));
        assert!(shader.source.contains("1.0"));
    }

    #[test]
    fn test_transpile_box() {
        let box3d = SdfNode::Box3d {
            half_extents: Vec3::new(1.0, 0.5, 0.5),
        };
        let shader = WgslShader::transpile(&box3d);

        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("abs(p)"));
    }

    #[test]
    fn test_transpile_union() {
        let shape = SdfNode::Sphere { radius: 1.0 }.union(
            SdfNode::Box3d {
                half_extents: Vec3::splat(0.5),
            }
            .translate(2.0, 0.0, 0.0),
        );
        let shader = WgslShader::transpile(&shape);

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
        let shader = WgslShader::transpile(&shape);

        // Deep Fried Edition: smooth ops are now inlined with division exorcism
        // Check for the inline pattern: max(k - abs(a - b), 0.0) * inv_k
        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("* 5.0")); // inv_k = 1/0.2 = 5.0
        assert!(shader.source.contains("* 0.25")); // The 0.25 constant
        // No helper function needed anymore
        assert_eq!(shader.helper_count, 0);
    }

    #[test]
    fn test_transpile_rotate() {
        let shape = SdfNode::Sphere { radius: 1.0 }
            .rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let shader = WgslShader::transpile(&shape);

        assert!(shader.source.contains("fn quat_rotate"));
        assert!(shader.source.contains("quat_rotate("));
    }

    #[test]
    fn test_compute_shader_generation() {
        let sphere = SdfNode::Sphere { radius: 1.0 };
        let shader = WgslShader::transpile(&sphere);
        let compute = shader.to_compute_shader();

        assert!(compute.contains("@compute"));
        assert!(compute.contains("@workgroup_size(256)"));
        assert!(compute.contains("input_points"));
        assert!(compute.contains("output_distances"));
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

        let shader = WgslShader::transpile(&shape);

        // Deep Fried Edition: smooth ops are inlined, only quat_rotate helper remains
        assert!(shader.source.contains("fn quat_rotate"));
        assert!(shader.source.contains("quat_rotate("));
        // Smooth ops are inlined with division exorcism (no helper function)
        assert!(shader.source.contains("* 5.0")); // inv_k = 1/0.2 = 5.0
        assert_eq!(shader.helper_count, 1); // Only quat_rotate helper

        // Should compile to valid WGSL (basic syntax check)
        assert!(shader.source.contains("fn sdf_eval"));
        assert!(shader.source.contains("return"));
    }
}

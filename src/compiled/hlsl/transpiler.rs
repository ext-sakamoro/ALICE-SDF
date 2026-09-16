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

use super::super::transpiler_common::ShaderLang;
use crate::types::SdfNode;
use std::fmt::Write;

/// Epsilon for constant folding (skip operations that are no-ops)
#[allow(dead_code)]
const FOLD_EPSILON: f32 = 1e-6;

// ============================================================================
// HlslLang: ShaderLang trait implementation for HLSL
// ============================================================================

/// Marker type for HLSL shader language syntax
pub struct HlslLang;

impl ShaderLang for HlslLang {
    fn vec2_ctor(x: &str, y: &str) -> String {
        format!("float2({}, {})", x, y)
    }
    fn vec3_ctor(x: &str, y: &str, z: &str) -> String {
        format!("float3({}, {}, {})", x, y, z)
    }
    fn vec4_ctor(x: &str, y: &str, z: &str, w: &str) -> String {
        format!("float4({}, {}, {}, {})", x, y, z, w)
    }
    fn vec2_zero() -> &'static str {
        "float2(0.0, 0.0)"
    }
    fn vec3_zero() -> &'static str {
        "float3(0.0, 0.0, 0.0)"
    }
    fn vec2_splat(v: &str) -> String {
        format!("float2({0}, {0})", v)
    }
    fn vec3_splat(v: &str) -> String {
        format!("float3({0}, {0}, {0})", v)
    }
    fn decl_float(name: &str, expr: &str) -> String {
        format!("    float {} = {};\n", name, expr)
    }
    fn decl_vec2(name: &str, expr: &str) -> String {
        format!("    float2 {} = {};\n", name, expr)
    }
    fn decl_vec3(name: &str, expr: &str) -> String {
        format!("    float3 {} = {};\n", name, expr)
    }
    fn decl_mut_float(name: &str, expr: &str) -> String {
        format!("    float {} = {};\n", name, expr)
    }
    fn decl_mut_vec2(name: &str, expr: &str) -> String {
        format!("    float2 {} = {};\n", name, expr)
    }
    fn decl_mut_vec3(name: &str, expr: &str) -> String {
        format!("    float3 {} = {};\n", name, expr)
    }
    fn decl_mut_float_typed(name: &str) -> String {
        format!("    float {};\n", name)
    }
    fn decl_mut_vec3_typed(name: &str) -> String {
        format!("    float3 {};\n", name)
    }
    fn select_expr(cond: &str, true_val: &str, false_val: &str) -> String {
        format!("(({}) ? {} : {})", cond, true_val, false_val)
    }
    fn modulo_expr(a: &str, b: &str) -> String {
        // Floor modulo like the CPU law (`p_mod1_r`) and GLSL `mod()`; HLSL
        // `fmod` is truncated (sign of the dividend).
        format!("(({a}) - ({b}) * floor(({a}) / ({b})))")
    }
    fn cast_float(expr: &str) -> String {
        format!("(float)({})", expr)
    }
    fn for_loop_int(name: &str, init: i32, cond: &str, incr: &str) -> String {
        format!("    for (int {} = {}; {}; {}) {{\n", name, init, cond, incr)
    }
    fn param_dynamic(vec_idx: usize, comp: &str) -> String {
        format!("params[{}].{}", vec_idx, comp)
    }
    fn func_signature() -> &'static str {
        "float sdf_eval(float3 p) {"
    }
    const CAPSULE_DEGENERATE_GUARD: bool = true;
    const CAPSULE_RE_EMIT_PARAMS: bool = true;
    fn helper_source(name: &str) -> Option<&'static str> {
        match name {
            "quat_rotate" => Some(HELPER_QUAT_ROTATE),
            "hash_noise" => Some(HELPER_HASH_NOISE),
            "taper_bound" => Some(HELPER_TAPER_BOUND),
            "perlin_noise" => Some(HELPER_PERLIN_NOISE),
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
            "sdf_ellipsoid" => Some(HELPER_SDF_ELLIPSOID),
            "bezier_distance_2d" => Some(HELPER_BEZIER_DISTANCE_2D),
            "sdf_tetrahedron" => Some(HELPER_SDF_TETRAHEDRON),
            "sdf_dodecahedron" => Some(HELPER_SDF_DODECAHEDRON),
            "sdf_icosahedron" => Some(HELPER_SDF_ICOSAHEDRON),
            "sdf_truncated_octahedron" => Some(HELPER_SDF_TRUNCATED_OCTAHEDRON),
            "sdf_truncated_icosahedron" => Some(HELPER_SDF_TRUNCATED_ICOSAHEDRON),
            _ => None,
        }
    }
}

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

        Self {
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
            r"// ALICE-SDF Generated HLSL for UE5 Custom Node
// Input: float3 p (World Position)
// Output: float (SDF Distance)
{dynamic_note}
{source}
return sdf_eval(p);
",
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
            r"// ALICE-SDF Generated HLSL Compute Shader ({mode:?} Mode)

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
",
            mode = self.mode,
            params_decl = params_decl,
            source = self.source
        )
    }

    /// Get the SDF evaluation function only (for embedding in custom shaders)
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }

    /// Export as UE5 Material Function (.ush include file)
    ///
    /// Generates an HLSL file that can be imported as a Material Function in UE5.
    /// Place the output in `Engine/Shaders/Private/` or your project's shader directory,
    /// then reference it via a Custom Expression or Material Function Library.
    pub fn export_ue5_material_function(&self) -> String {
        let params_section = if self.mode == HlslTranspileMode::Dynamic {
            "// Dynamic parameters - bind via Material Parameter Collection\ncbuffer SdfParams : register(b1) {\n    float4 params[1024];\n};\n\n"
        } else {
            ""
        };

        format!(
            r#"// ALICE-SDF Material Function (UE5)
// Generated by ALICE-SDF Compiler
// Usage: #include "/Project/AliceSdf/MF_AliceSdf.ush"
//
// float dist = AliceSdf_Eval(WorldPosition);
// float3 normal = AliceSdf_Normal(WorldPosition);

#pragma once

{params_section}{source}

// Material Function entry point
float AliceSdf_Eval(float3 WorldPosition) {{
    return sdf_eval(WorldPosition);
}}

// Analytic normal via central differences
float3 AliceSdf_Normal(float3 p) {{
    const float e = 0.001;
    return normalize(float3(
        sdf_eval(p + float3(e,0,0)) - sdf_eval(p - float3(e,0,0)),
        sdf_eval(p + float3(0,e,0)) - sdf_eval(p - float3(0,e,0)),
        sdf_eval(p + float3(0,0,e)) - sdf_eval(p - float3(0,0,e))
    ));
}}
"#,
            params_section = params_section,
            source = self.source
        )
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

#[allow(dead_code)]
impl HlslTranspiler {
    const fn new(mode: HlslTranspileMode) -> Self {
        Self {
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

    fn transpile_node(&mut self, node: &SdfNode, point_var: &str) -> String {
        use super::super::transpiler_common::{GenericTranspiler, TranspileModeLang};
        let mode = match self.mode {
            HlslTranspileMode::Hardcoded => TranspileModeLang::Hardcoded,
            HlslTranspileMode::Dynamic => TranspileModeLang::Dynamic,
        };
        let mut generic: GenericTranspiler<HlslLang> = GenericTranspiler::new(mode);
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

        // Helper sources come from `ShaderLang::helper_source` — the single
        // table per language. A helper the walker referenced but the table
        // does not know is a transpiler bug that would surface as a shader
        // compile error at runtime: fail here instead (until 1.10.3 the
        // five GDF polyhedra were skipped silently by a duplicated table).
        for helper in &self.helper_functions {
            let src = <HlslLang as ShaderLang>::helper_source(helper).unwrap_or_else(|| {
                panic!("hlsl transpiler: no helper source registered for `{helper}`")
            });
            shader.push_str(src);
            shader.push('\n');
        }

        // Add main SDF function
        writeln!(shader, "float sdf_eval(float3 p) {{").unwrap();
        shader.push_str(body);
        shader.push_str("}\n");

        shader
    }
}

// Helper function definitions for HLSL
const HELPER_TAPER_BOUND: &str = r"// Same law as `alice_sdf::compiled::real::taper_bound` (Jacobian ball + cone ∩ slab).
float alice_taper_bound(float d, float3 p, float factor, float rx, float ry) {
    float fa = abs(factor);
    if (fa == 0.0) { return d; }
    float den_abs = max(abs(1.0 - p.y * factor), 1e-6);
    float s = 1.0 / den_abs;
    float rho = sqrt(p.x * p.x + p.z * p.z);
    float d_abs = abs(d);
    float j0 = max(s, 1.0) + fa * s * s * rho;
    float r = min(d_abs / j0, den_abs / (fa + fa));
    float den1 = max(den_abs - fa * r, 1e-6);
    float s1 = 1.0 / den1;
    float j1 = max(s1, 1.0) + fa * s1 * s1 * (rho + r);
    float mag = min(d_abs / j1, r);
    float d_j = (d < 0.0) ? -mag : mag;
    if (rx >= 1e30 || ry >= 1e30) { return d_j; }
    float k = rx * fa;
    float inv_n = 1.0 / sqrt(1.0 + k * k);
    float big_y = p.y - 1.0 / factor;
    float d_cone = (rho - k * abs(big_y)) * inv_n;
    float d_slab = abs(p.y) - ry;
    float d_region = max(d_cone, d_slab);
    return (d < 0.0) ? d_j : max(d_j, d_region);
}";

const HELPER_HASH_NOISE: &str = crate::modifiers::HASH_NOISE_HLSL;

const HELPER_PERLIN_NOISE: &str = r"uint alice_perlin_hash(int x, int y, int z, uint seed) {
    uint h = seed;
    h ^= asuint(x);
    h *= 0x85EBCA6Bu;
    h ^= asuint(y);
    h *= 0xC2B2AE35u;
    h ^= asuint(z);
    h *= 0x27D4EB2Du;
    h ^= h >> 16u;
    return h;
}
float alice_perlin_grad(uint h, float3 c) {
    const uint LUT[16] = { 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 24u, 25u, 26u, 27u, 16u, 10u, 18u, 9u };
    uint e = LUT[h & 15u];
    uint u_src = (e >> 4u) & 3u;
    uint v_src = (e >> 2u) & 3u;
    float u_sign = 1.0 - 2.0 * (float)((e >> 1u) & 1u);
    float v_sign = 1.0 - 2.0 * (float)(e & 1u);
    return c[u_src] * u_sign + c[v_src] * v_sign;
}
float alice_perlin_fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}
// Same law as `alice_sdf::modifiers::perlin_noise_3d` (integer lattice hash + 16-gradient LUT + quintic fade).
float perlin_noise_3d(float3 p, uint seed) {
    float3 fl = floor(p);
    int xi = (int)fl.x;
    int yi = (int)fl.y;
    int zi = (int)fl.z;
    float3 f = p - fl;
    float u = alice_perlin_fade(f.x);
    float v = alice_perlin_fade(f.y);
    float w = alice_perlin_fade(f.z);
    float aaa = alice_perlin_grad(alice_perlin_hash(xi, yi, zi, seed), f);
    float aba = alice_perlin_grad(alice_perlin_hash(xi, yi + 1, zi, seed), float3(f.x, f.y - 1.0, f.z));
    float aab = alice_perlin_grad(alice_perlin_hash(xi, yi, zi + 1, seed), float3(f.x, f.y, f.z - 1.0));
    float abb = alice_perlin_grad(alice_perlin_hash(xi, yi + 1, zi + 1, seed), float3(f.x, f.y - 1.0, f.z - 1.0));
    float baa = alice_perlin_grad(alice_perlin_hash(xi + 1, yi, zi, seed), float3(f.x - 1.0, f.y, f.z));
    float bba = alice_perlin_grad(alice_perlin_hash(xi + 1, yi + 1, zi, seed), float3(f.x - 1.0, f.y - 1.0, f.z));
    float bab = alice_perlin_grad(alice_perlin_hash(xi + 1, yi, zi + 1, seed), float3(f.x - 1.0, f.y, f.z - 1.0));
    float bbb = alice_perlin_grad(alice_perlin_hash(xi + 1, yi + 1, zi + 1, seed), float3(f.x - 1.0, f.y - 1.0, f.z - 1.0));
    float x0 = aaa + u * (baa - aaa);
    float x1 = aba + u * (bba - aba);
    float x2 = aab + u * (bab - aab);
    float x3 = abb + u * (bbb - abb);
    float y0 = x0 + v * (x1 - x0);
    float y1 = x2 + v * (x3 - x2);
    return y0 + w * (y1 - y0);
}";

const HELPER_QUAT_ROTATE: &str = r"float3 quat_rotate(float3 v, float4 q) {
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}";

const HELPER_SDF_ROUNDED_CONE: &str = r"float sdf_rounded_cone(float3 p, float r1, float r2, float h) {
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
";

const HELPER_SDF_PYRAMID: &str = r"float sdf_pyramid(float3 p, float h) {
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
    return sqrt((d2 + qz * qz) / m2) * (max(qz, -py) < 0.0 ? -1.0 : 1.0);
}
";

const HELPER_SDF_OCTAHEDRON: &str = r"float sdf_octahedron(float3 p, float s) {
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
";

const HELPER_SDF_HEX_PRISM: &str = r"float sdf_hex_prism(float3 p, float hex_r, float h) {
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
    float d_xy = sqrt(dx * dx + dy * dy) * (dy < 0.0 ? -1.0 : 1.0);
    float d_z = pz - h;
    return min(max(d_xy, d_z), 0.0) + length(max(float2(d_xy, d_z), float2(0.0, 0.0)));
}
";

const HELPER_SDF_TRIANGLE: &str = r"float sdf_triangle(float3 p, float3 a, float3 b, float3 c) {
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
";

const HELPER_SDF_BEZIER: &str = r"float sdf_bezier(float3 pos, float3 A, float3 B, float3 C, float rad) {
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
";

const HELPER_SDF_LINK: &str = r"float sdf_link(float3 p, float le, float r1, float r2) {
    float qx = p.x;
    float qy = max(abs(p.y) - le, 0.0);
    float qz = p.z;
    float xy_len = sqrt(qx * qx + qy * qy) - r1;
    return sqrt(xy_len * xy_len + qz * qz) - r2;
}
";

const HELPER_SDF_CAPPED_CONE: &str = r"float sdf_capped_cone(float3 p, float h, float r1, float r2) {
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
";

const HELPER_SDF_CAPPED_TORUS: &str = r"float sdf_capped_torus(float3 p, float ra, float rb, float an) {
    float2 sc = float2(sin(an), cos(an));
    float px = abs(p.x);
    float k = (sc.y * px > sc.x * p.y) ? dot(float2(px, p.y), sc) : length(float2(px, p.y));
    return sqrt(px * px + p.y * p.y + p.z * p.z + ra * ra - 2.0 * ra * k) - rb;
}
";

const HELPER_SDF_ROUNDED_CYLINDER: &str = r"float sdf_rounded_cylinder(float3 p, float r, float rr, float h) {
    float2 d = float2(length(p.xz) - r + rr, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, float2(0.0, 0.0))) - rr;
}
";

const HELPER_SDF_TRIANGULAR_PRISM: &str = r"float sdf_triangular_prism(float3 p, float w, float h) {
    float3 q = abs(p);
    return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - w * 0.5);
}
";

const HELPER_SDF_CUT_SPHERE: &str = r"float sdf_cut_sphere(float3 p, float r, float h) {
    float w = sqrt(max(r * r - h * h, 0.0));
    float2 q = float2(length(p.xz), p.y);
    float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
    if (s < 0.0) return length(q) - r;
    if (q.x < w) return h - q.y;
    return length(q - float2(w, h));
}
";

const HELPER_SDF_CUT_HOLLOW_SPHERE: &str = r"float sdf_cut_hollow_sphere(float3 p, float r, float h, float t) {
    float w = sqrt(max(r * r - h * h, 0.0));
    float2 q = float2(length(p.xz), p.y);
    if (h * q.x < w * q.y) return length(q - float2(w, h)) - t;
    return abs(length(q) - r) - t;
}
";

const HELPER_SDF_DEATH_STAR: &str = r"float sdf_death_star(float3 p, float ra, float rb, float d) {
    float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    float b = sqrt(max(ra * ra - a * a, 0.0));
    float2 q = float2(p.x, length(p.yz));
    if (q.x * b - q.y * a > d * max(b - q.y, 0.0))
        return length(q - float2(a, b));
    return max(length(q) - ra, -(length(q - float2(d, 0.0)) - rb));
}
";

const HELPER_SDF_SOLID_ANGLE: &str = r"float sdf_solid_angle(float3 p, float an, float ra) {
    float2 c = float2(sin(an), cos(an));
    float2 q = float2(length(p.xz), p.y);
    float l = length(q) - ra;
    float m = length(q - c * clamp(dot(q, c), 0.0, ra));
    return max(l, m * sign(c.y * q.x - c.x * q.y));
}
";

const HELPER_SDF_RHOMBUS: &str = r"float ndot_rh(float2 a, float2 b) {
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
";

const HELPER_SDF_HORSESHOE: &str = r"float sdf_horseshoe(float3 pos, float an, float r, float le, float w, float t) {
    float2 c = float2(cos(an), sin(an));
    float px = abs(pos.x);
    float l = length(float2(px, pos.y));
    float qx = -c.x * px + c.y * pos.y;
    float qy = c.y * px + c.x * pos.y;
    if (!(qy > 0.0 || qx > 0.0)) { qx = l * sign(-c.x); }
    if (!(qx > 0.0)) { qy = l; }
    float bx = qx - le;
    float by = abs(qy - r) - w;
    float d2 = length(max(float2(bx, by), float2(0.0, 0.0))) + min(max(bx, by), 0.0);
    float dz = abs(pos.z) - t;
    return length(max(float2(d2, dz), float2(0.0, 0.0))) + min(max(d2, dz), 0.0);
}
";

const HELPER_SDF_VESICA: &str = r"float sdf_vesica(float3 p, float r, float d) {
    float px = abs(p.x);
    float py = length(p.yz);
    float b = sqrt(max(r * r - d * d, 0.0));
    if ((py - b) * d > px * b) return length(float2(px, py - b));
    return length(float2(px - d, py)) - r;
}
";

const HELPER_SDF_INFINITE_CONE: &str = r"float sdf_infinite_cone(float3 p, float an) {
    float2 c = float2(sin(an), cos(an));
    float2 q = float2(length(p.xz), -p.y);
    float d = length(q - c * max(dot(q, c), 0.0));
    return d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
}
";

const HELPER_SDF_HEART: &str = r"float sdf_heart(float3 p, float s) {
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
";

const HELPER_SDF_TUBE: &str = r"float sdf_tube(float3 p, float outer_r, float thick, float h) {
    float r = length(p.xz);
    float dr = abs(r - outer_r) - thick;
    float dy = abs(p.y) - h;
    float2 w = max(float2(dr, dy), float2(0.0, 0.0));
    return min(max(dr, dy), 0.0) + length(w);
}
";

const HELPER_SDF_BARREL: &str = r"float sdf_barrel(float3 p, float radius, float h, float bulge) {
    float r = length(p.xz);
    float yn = clamp(p.y / h, -1.0, 1.0);
    float er = radius + bulge * (1.0 - yn * yn);
    float dr = r - er;
    float dy = abs(p.y) - h;
    float2 w = max(float2(dr, dy), float2(0.0, 0.0));
    return min(max(dr, dy), 0.0) + length(w);
}
";

const HELPER_SDF_DIAMOND: &str = r"float sdf_diamond(float3 p, float r, float h) {
    float2 q = float2(length(p.xz), abs(p.y));
    float2 ba = float2(-r, h);
    float2 qa = q - float2(r, 0.0);
    float t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    float2 closest = float2(r, 0.0) + ba * t;
    float dist = length(q - closest);
    if (q.x * h + q.y * r < r * h) return -dist;
    return dist;
}
";

const HELPER_SDF_CHAMFERED_CUBE: &str = r"float sdf_chamfered_cube(float3 p, float hx, float hy, float hz, float ch) {
    float3 ap = abs(p);
    float3 q = ap - float3(hx, hy, hz);
    float d_box = min(max(q.x, max(q.y, q.z)), 0.0) + length(max(q, float3(0.0, 0.0, 0.0)));
    float s = hx + hy + hz;
    float d_ch = (ap.x + ap.y + ap.z - s + ch) * 0.57735;
    return max(d_box, d_ch);
}
";

const HELPER_SDF_SUPERELLIPSOID: &str = r"float sdf_superellipsoid(float3 p, float hx, float hy, float hz, float e1, float e2) {
    float qx = max(abs(p.x / hx), 0.00001);
    float qy = max(abs(p.y / hy), 0.00001);
    float qz = max(abs(p.z / hz), 0.00001);
    float ee1 = max(e1, 0.02);
    float ee2 = max(e2, 0.02);
    float m1 = 2.0 / ee2;
    float m2 = 2.0 / ee1;
    float w = pow(qx, m1) + pow(qz, m1);
    float v = pow(w, ee2 / ee1) + pow(qy, m2);
    float f = pow(v, ee1 * 0.5);
    return (f - 1.0) * min(hx, min(hy, hz)) * 0.5;
}
";

const HELPER_SDF_ROUNDED_X: &str = r"float sdf_rounded_x(float3 p, float w, float r, float h) {
    float2 q = abs(p.xz);
    float s = min(q.x + q.y, w) * 0.5;
    float d2d = length(q - float2(s, s)) - r;
    float dy = abs(p.y) - h;
    float2 ww = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
";

const HELPER_SDF_PIE: &str = r"float sdf_pie(float3 p, float angle, float radius, float h) {
    float2 q = float2(p.x, p.z);
    float l = length(q) - radius;
    float2 sc = float2(sin(angle), cos(angle));
    float m = length(q) * clamp(sc.y * abs(q.x) - sc.x * q.y, -radius, 0.0);
    float d2d = max(l, m / max(radius, 1e-10));
    float dy = abs(p.y) - h;
    float2 ww = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
";

const HELPER_SDF_TRAPEZOID: &str = r"float _trap_ds(float px, float py, float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    float len_sq = dx * dx + dy * dy;
    float t = len_sq > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / len_sq : 0.0;
    t = clamp(t, 0.0, 1.0);
    float cx = ax + dx * t;
    float cy = ay + dy * t;
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}
float sdf_trapezoid(float3 p, float r1, float r2, float th, float hd) {
    float px = abs(p.x);
    float py = p.y;
    float he = th;
    float d_bot = _trap_ds(px, py, 0.0, -he, r1, -he);
    float d_slant = _trap_ds(px, py, r1, -he, r2, he);
    float d_top = _trap_ds(px, py, r2, he, 0.0, he);
    float d_unsigned = min(d_bot, min(d_slant, d_top));
    float nx = 2.0 * he;
    float ny = r1 - r2;
    float d_slant_plane = (px - r1) * nx + (py + he) * ny;
    float d2d = (py >= -he && py <= he && d_slant_plane <= 0.0) ? -d_unsigned : d_unsigned;
    float dz = abs(p.z) - hd;
    float2 ww = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
";

const HELPER_SDF_PARALLELOGRAM: &str = r"float _para_ds(float px, float py, float ax, float ay, float bx, float by) {
    float dx = bx - ax;
    float dy = by - ay;
    float len_sq = dx * dx + dy * dy;
    float t = len_sq > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / len_sq : 0.0;
    t = clamp(t, 0.0, 1.0);
    float cx = ax + dx * t;
    float cy = ay + dy * t;
    return sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
}
float _para_c2d(float px, float py, float ax, float ay, float bx, float by) {
    return (px - ax) * (by - ay) - (py - ay) * (bx - ax);
}
float sdf_parallelogram(float3 p, float w, float ph, float sk, float hd) {
    float px = p.x;
    float py = p.y;
    float vdx = w - sk;  float vdy = -ph;
    float vax = w + sk;  float vay = ph;
    float vbx = -w + sk; float vby = ph;
    float vcx = -w - sk; float vcy = -ph;
    float d1 = _para_ds(px, py, vdx, vdy, vax, vay);
    float d2 = _para_ds(px, py, vax, vay, vbx, vby);
    float d3 = _para_ds(px, py, vbx, vby, vcx, vcy);
    float d4 = _para_ds(px, py, vcx, vcy, vdx, vdy);
    float d_unsigned = min(d1, min(d2, min(d3, d4)));
    float c1 = _para_c2d(px, py, vdx, vdy, vax, vay);
    float c2 = _para_c2d(px, py, vax, vay, vbx, vby);
    float c3 = _para_c2d(px, py, vbx, vby, vcx, vcy);
    float c4 = _para_c2d(px, py, vcx, vcy, vdx, vdy);
    float d2d = (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0 && c4 <= 0.0) ? -d_unsigned : d_unsigned;
    float dz = abs(p.z) - hd;
    float2 ww = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
";

const HELPER_SDF_TUNNEL: &str = r"float sdf_tunnel(float3 p, float w, float h2d, float hd) {
    float px = abs(p.x);
    float py = p.y;
    float dx = px - w;
    float dy_rect = abs(py) - h2d;
    float d_rect = length(max(float2(dx, dy_rect), float2(0.0, 0.0))) + min(max(dx, dy_rect), 0.0);
    float d_circle = length(float2(px, py - h2d)) - w;
    float d2d = py > h2d ? min(d_rect, d_circle) : d_rect;
    float dz = abs(p.z) - hd;
    float2 ww = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
";

const HELPER_SDF_UNEVEN_CAPSULE: &str = r"float sdf_uneven_capsule(float3 p, float r1, float r2, float ch, float hd) {
    float px = abs(p.x);
    float hh = ch * 2.0;
    float b = (r1 - r2) / hh;
    float a = sqrt(max(1.0 - b * b, 0.0));
    float k = dot(float2(-b, a), float2(px, p.y));
    float d2d;
    if (k < 0.0) {
        d2d = length(float2(px, p.y)) - r1;
    } else if (k > a * hh) {
        d2d = length(float2(px, p.y - hh)) - r2;
    } else {
        d2d = dot(float2(px, p.y), float2(a, b)) - r1;
    }
    float dz = abs(p.z) - hd;
    float2 ww = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(ww);
}
";

const HELPER_SDF_EGG: &str = r"float sdf_egg(float3 p, float ra, float rb) {
    const float k = 1.7320508;
    float px = length(p.xz);
    float py = p.y;
    float r = ra - rb;
    float d;
    if (py < 0.0) {
        d = length(float2(px, py)) - r;
    } else if (k * (px + r) < py) {
        d = length(float2(px, py - k * r)) - r;
    } else {
        d = length(float2(px + r, py)) - 2.0 * r;
    }
    return d - rb;
}
";

const HELPER_SDF_ARC_SHAPE: &str = r"float sdf_arc_shape(float3 p, float aperture, float radius, float thickness, float h) {
    float qx = abs(p.x);
    float qz = p.z;
    float2 sc = float2(sin(aperture), cos(aperture));
    float d2d;
    if (sc.y * qx > sc.x * qz) {
        d2d = length(float2(qx, qz) - sc * radius) - thickness;
    } else {
        d2d = abs(length(float2(qx, qz)) - radius) - thickness;
    }
    float dy = abs(p.y) - h;
    float2 ww = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
";

const HELPER_SDF_MOON: &str = r"float sdf_moon(float3 p, float d, float ra, float rb, float h) {
    float qx = abs(p.x);
    float qz = p.z;
    float d_outer = length(float2(qx, qz)) - ra;
    float d_inner = length(float2(qx - d, qz)) - rb;
    float d2d = max(d_outer, -d_inner);
    float dy = abs(p.y) - h;
    float2 ww = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
";

const HELPER_SDF_CROSS_SHAPE: &str = r"float sdf_cross_shape(float3 p, float len, float th, float rr, float h) {
    float qx = abs(p.x);
    float qz = abs(p.z);
    float2 dh = float2(qx - len, qz - th);
    float2 dv = float2(qx - th, qz - len);
    float dh_sdf = length(max(dh, float2(0.0, 0.0))) + min(max(dh.x, dh.y), 0.0);
    float dv_sdf = length(max(dv, float2(0.0, 0.0))) + min(max(dv.x, dv.y), 0.0);
    float d2d = min(dh_sdf, dv_sdf) - rr;
    float dy = abs(p.y) - h;
    float2 ww = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(ww);
}
";

const HELPER_SDF_BLOBBY_CROSS: &str = r"float sdf_blobby_cross(float3 p, float size, float hh) {
    const float he = 0.5;
    float2 pos = abs(float2(p.x, p.z) / size);
    pos = float2(abs(pos.x - pos.y), 1.0 - pos.x - pos.y) * 0.70710678;
    float pp = (he - pos.y - 0.25 / he) / (6.0 * he);
    float q = pos.x / (he * he * 16.0);
    float h = q * q - pp * pp * pp;
    float x;
    if (h >= 0.0) {
        float r = sqrt(h);
        x = pow(q + r, 1.0 / 3.0) - pow(abs(q - r), 1.0 / 3.0) * sign(r - q);
    } else {
        float r = sqrt(pp);
        x = 2.0 * r * cos(acos(q / (pp * r)) / 3.0);
    }
    x = min(x, 0.70710678);
    float2 z = float2(x, he * (1.0 - 2.0 * x * x)) - pos;
    float d_2d = length(z) * sign(z.y) * size;
    float d_y = abs(p.y) - hh;
    float2 w = float2(max(d_2d, 0.0), max(d_y, 0.0));
    return min(max(d_2d, d_y), 0.0) + length(w);
}
";

const HELPER_SDF_PARABOLA_SEGMENT: &str = r"float sdf_parabola_segment(float3 p, float w, float ph, float hd) {
    float px = abs(p.x);
    float py = p.y;
    float ww_sq = w * w;
    float y_arch = ph * (1.0 - px * px / ww_sq);
    bool is_in = (px <= w && py >= 0.0 && py <= y_arch);
    float t = clamp(px, 0.0, w);
    for (int i = 0; i < 8; i++) {
        float ft = ph * (1.0 - t * t / ww_sq);
        float dft = -2.0 * ph * t / ww_sq;
        float ex = px - t;
        float ey = py - ft;
        float f = -ex + ey * dft;
        float df = 1.0 + dft * dft + ey * (-2.0 * ph / ww_sq);
        if (abs(df) > 1e-10) { t = clamp(t - f / df, 0.0, w); }
    }
    float cy = ph * (1.0 - t * t / ww_sq);
    float d_para = length(float2(px - t, py - cy));
    float d_base = (px <= w) ? abs(py) : length(float2(px - w, py));
    float d_unsigned = min(d_para, d_base);
    float d2d = is_in ? -d_unsigned : d_unsigned;
    float dz = abs(p.z) - hd;
    float2 ext = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(ext);
}
";

const HELPER_SDF_REGULAR_POLYGON: &str = r"float sdf_regular_polygon(float3 p, float radius, float n, float hh) {
    // IQ regular polygon (exact), XZ plane, radius = circumradius (1.9.2)
    float nn = trunc(max(n, 3.0));
    float an = 3.14159265358979 / nn;
    float2 acs = float2(cos(an), sin(an));
    float a0 = atan2(p.x, p.z);
    float bn = a0 - 2.0 * an * floor(a0 / (2.0 * an)) - an;
    float r = length(p.xz);
    float2 q = float2(r * cos(bn), abs(r * sin(bn)));
    q -= radius * acs;
    q.y += clamp(-q.y, 0.0, radius * acs.y);
    float d2d = length(q) * sign(q.x);
    float dy = abs(p.y) - hh;
    float2 w = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(w);
}
";

const HELPER_SDF_STAR_POLYGON: &str = r"float sdf_star_polygon(float3 p, float radius, float np, float m, float hh) {
    float qx = abs(p.x);
    float qz = p.z;
    float n = max(np, 3.0);
    float an = 3.14159265358979 / n;
    float r = length(float2(qx, qz));
    float angle = atan2(qx, qz);
    angle = fmod(fmod(angle, 2.0 * an) + 2.0 * an, 2.0 * an);
    if (angle > an) angle = 2.0 * an - angle;
    float2 pt = float2(r * cos(angle), r * sin(angle));
    float2 a = float2(radius, 0.0);
    float2 b = float2(m * cos(an), m * sin(an));
    float2 ab = b - a;
    float2 ap = pt - a;
    float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    float2 closest = a + ab * t;
    float dist = length(pt - closest);
    float cross_val = ab.x * ap.y - ab.y * ap.x;
    float d2d = (cross_val > 0.0) ? -dist : dist;
    float dy = abs(p.y) - hh;
    float2 w = max(float2(d2d, dy), float2(0.0, 0.0));
    return min(max(d2d, dy), 0.0) + length(w);
}
";

const HELPER_SDF_STAIRS: &str = r"float _stair_box(float lx, float ly, float s, float sw, float sh) {
    float cx = s * sw + sw * 0.5;
    float hy = (s + 1.0) * sh * 0.5;
    float dx = abs(lx - cx) - sw * 0.5;
    float dy = abs(ly - hy) - hy;
    return length(max(float2(dx, dy), float2(0.0, 0.0))) + min(max(dx, dy), 0.0);
}
float sdf_stairs(float3 p, float sw, float sh, float ns, float hd) {
    float n = max(ns, 1.0);
    float tw = n * sw;
    float th = n * sh;
    float lx = p.x + tw * 0.5;
    float ly = p.y + th * 0.5;
    float d2d = 1e30;
    for (float si = 0.0; si < n; si += 1.0) {
        d2d = min(d2d, _stair_box(lx, ly, si, sw, sh));
    }
    float dz = abs(p.z) - hd;
    float2 w = max(float2(d2d, dz), float2(0.0, 0.0));
    return min(max(d2d, dz), 0.0) + length(w);
}
";

/// Regular tetrahedron: max of 4 face-plane distances (mirrors `primitives::tetrahedron`).
const HELPER_SDF_TETRAHEDRON: &str = r"float sdf_tetrahedron(float3 p, float s) {
    float d = dot(p, float3(0.57735026, 0.57735026, 0.57735026));
    d = max(d, dot(p, float3(-0.57735026, -0.57735026, 0.57735026)));
    d = max(d, dot(p, float3(-0.57735026, 0.57735026, -0.57735026)));
    d = max(d, dot(p, float3(0.57735026, -0.57735026, -0.57735026)));
    return d - s;
}
";

/// Dodecahedron: GDF over the 6 dodecahedron normals (mirrors `gdf_eval`).
const HELPER_SDF_DODECAHEDRON: &str = r"float sdf_dodecahedron(float3 p, float r) {
    float d = abs(dot(p, float3(0.0, 0.8506508, 0.5257311)));
    d = max(d, abs(dot(p, float3(0.0, 0.8506508, -0.5257311))));
    d = max(d, abs(dot(p, float3(0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, float3(-0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.0, 0.8506508))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.0, -0.8506508))));
    return d - r;
}
";

/// Icosahedron: GDF over octahedron + icosahedron normals (mirrors `primitives::icosahedron`).
const HELPER_SDF_ICOSAHEDRON: &str = r"float sdf_icosahedron(float3 p, float r) {
    float d = abs(dot(p, float3(0.57735026, 0.57735026, 0.57735026)));
    d = max(d, abs(dot(p, float3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, 0.57735026, -0.57735026))));
    d = max(d, abs(dot(p, float3(0.0, 0.5257311, 0.8506508))));
    d = max(d, abs(dot(p, float3(0.0, 0.5257311, -0.8506508))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.8506508, 0.0))));
    d = max(d, abs(dot(p, float3(0.5257311, -0.8506508, 0.0))));
    d = max(d, abs(dot(p, float3(0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, float3(-0.8506508, 0.0, 0.5257311))));
    return d - r;
}
";

/// Truncated octahedron: GDF over cube + octahedron normals.
const HELPER_SDF_TRUNCATED_OCTAHEDRON: &str = r"float sdf_truncated_octahedron(float3 p, float r) {
    float d = abs(dot(p, float3(1.0, 0.0, 0.0)));
    d = max(d, abs(dot(p, float3(0.0, 1.0, 0.0))));
    d = max(d, abs(dot(p, float3(0.0, 0.0, 1.0))));
    d = max(d, abs(dot(p, float3(0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, 0.57735026, -0.57735026))));
    return d - r;
}
";

/// Truncated icosahedron: GDF over octahedron + icosahedron + dodecahedron normals.
const HELPER_SDF_TRUNCATED_ICOSAHEDRON: &str = r"float sdf_truncated_icosahedron(float3 p, float r) {
    float d = abs(dot(p, float3(0.57735026, 0.57735026, 0.57735026)));
    d = max(d, abs(dot(p, float3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, float3(0.57735026, 0.57735026, -0.57735026))));
    d = max(d, abs(dot(p, float3(0.0, 0.5257311, 0.8506508))));
    d = max(d, abs(dot(p, float3(0.0, 0.5257311, -0.8506508))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.8506508, 0.0))));
    d = max(d, abs(dot(p, float3(0.5257311, -0.8506508, 0.0))));
    d = max(d, abs(dot(p, float3(0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, float3(-0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, float3(0.0, 0.8506508, 0.5257311))));
    d = max(d, abs(dot(p, float3(0.0, 0.8506508, -0.5257311))));
    d = max(d, abs(dot(p, float3(0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, float3(-0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.0, 0.8506508))));
    d = max(d, abs(dot(p, float3(0.5257311, 0.0, -0.8506508))));
    return d - r;
}
";

/// IQ `sdBezier`: closed-form distance to a quadratic Bézier (mirrors
/// `modifiers::sweep::bezier_distance_2d`, degenerate curve → segment).
const HELPER_BEZIER_DISTANCE_2D: &str = r"float bezier_distance_2d(float2 q, float2 p0, float2 p1, float2 p2) {
    float2 a = p1 - p0;
    float2 b = p0 - p1 * 2.0 + p2;
    float2 c = a * 2.0;
    float2 d = p0 - q;
    float bb = dot(b, b);
    if (bb < 1e-12) {
        float2 ac = p2 - p0;
        float t = clamp(-dot(d, ac) / max(dot(ac, ac), 1e-12), 0.0, 1.0);
        return length(d + ac * t);
    }
    float kk = 1.0 / bb;
    float kx = kk * dot(a, b);
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    float kz = kk * dot(d, a);
    float p = ky - kx * kx;
    float p3 = p * p * p;
    float qq = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = qq * qq + 4.0 * p3;
    float res;
    if (h >= 0.0) {
        float hs = sqrt(h);
        float x0 = (hs - qq) * 0.5;
        float x1 = (-hs - qq) * 0.5;
        float u = sign(x0) * pow(abs(x0), 1.0 / 3.0);
        float v = sign(x1) * pow(abs(x1), 1.0 / 3.0);
        float t = clamp(u + v - kx, 0.0, 1.0);
        float2 e = d + (c + b * t) * t;
        res = dot(e, e);
    } else {
        float z = sqrt(-p);
        float v = acos(clamp(qq / (p * z * 2.0), -1.0, 1.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.7320508;
        float t0 = clamp((m + m) * z - kx, 0.0, 1.0);
        float t1 = clamp((-n - m) * z - kx, 0.0, 1.0);
        float2 e0 = d + (c + b * t0) * t0;
        float2 e1 = d + (c + b * t1) * t1;
        res = min(dot(e0, e0), dot(e1, e1));
    }
    return sqrt(res);
}
";

/// Exact ellipsoid signed distance (Eberly nearest point; mirrors
/// `primitives::ellipsoid::sdf_ellipsoid_exact`, dimension reductions unrolled).
const HELPER_SDF_ELLIPSOID: &str = r"float _ell_bisect(int n, float r0, float r1, float r2, float z0, float z1, float z2, float g) {
    float n0 = r0 * z0;
    float n1 = (n > 1) ? r1 * z1 : 0.0;
    float n2 = (n > 2) ? r2 * z2 : 0.0;
    float zl = (n == 3) ? z2 : ((n == 2) ? z1 : z0);
    float s0 = zl - 1.0;
    float s1 = (g < 0.0) ? 0.0 : sqrt(n0 * n0 + n1 * n1 + n2 * n2) - 1.0;
    float s = 0.0;
    for (int it = 0; it < 64; it++) {
        s = 0.5 * (s0 + s1);
        if (s == s0 || s == s1) { break; }
        float gs = -1.0;
        float q0 = n0 / (s + r0);
        gs += q0 * q0;
        if (n > 1) { float q1 = n1 / (s + r1); gs += q1 * q1; }
        if (n > 2) { float q2 = n2 / (s + r2); gs += q2 * q2; }
        if (gs > 0.0) { s0 = s; } else if (gs < 0.0) { s1 = s; } else { break; }
    }
    return s;
}
float _ell_sqr_dist_n(int n, float e0, float e1, float e2, float y0, float y1, float y2) {
    float el = (n == 3) ? e2 : ((n == 2) ? e1 : e0);
    float z0 = y0 / e0;
    float z1 = (n > 1) ? y1 / e1 : 0.0;
    float z2 = (n > 2) ? y2 / e2 : 0.0;
    float g = z0 * z0 + z1 * z1 + z2 * z2 - 1.0;
    if (g == 0.0) { return 0.0; }
    float r0 = (e0 / el) * (e0 / el);
    float r1 = (e1 / el) * (e1 / el);
    float r2 = (e2 / el) * (e2 / el);
    float sb = _ell_bisect(n, r0, r1, r2, z0, z1, z2, g);
    float x0 = r0 * y0 / (sb + r0) - y0;
    float x1 = (n > 1) ? r1 * y1 / (sb + r1) - y1 : 0.0;
    float x2 = (n > 2) ? r2 * y2 / (sb + r2) - y2 : 0.0;
    return x0 * x0 + x1 * x1 + x2 * x2;
}
float sdf_ellipsoid(float3 p, float3 radii) {
    float3 e = max(radii, float3(1e-10, 1e-10, 1e-10));
    float3 y = abs(p);
    if (e.x < e.y) { float t = e.x; e.x = e.y; e.y = t; float u = y.x; y.x = y.y; y.y = u; }
    if (e.y < e.z) { float t = e.y; e.y = e.z; e.z = t; float u = y.y; y.y = y.z; y.z = u; }
    if (e.x < e.y) { float t = e.x; e.x = e.y; e.y = t; float u = y.x; y.x = y.y; y.y = u; }
    float d2 = 0.0;
    if (y.z > 0.0) {
        d2 = _ell_sqr_dist_n(3, e.x, e.y, e.z, y.x, y.y, y.z);
    } else {
        float n0 = e.x * y.x; float n1 = e.y * y.y;
        float d0 = e.x * e.x - e.z * e.z; float d1 = e.y * e.y - e.z * e.z;
        bool done = false;
        if (n0 < d0 && n1 < d1) {
            float x0 = n0 / d0; float x1 = n1 / d1;
            float discr = 1.0 - x0 * x0 - x1 * x1;
            if (discr > 0.0) {
                float a = e.x * x0 - y.x; float b = e.y * x1 - y.y; float c = e.z * sqrt(discr);
                d2 = a * a + b * b + c * c;
                done = true;
            }
        }
        if (!done) {
            if (y.y > 0.0) {
                d2 = _ell_sqr_dist_n(2, e.x, e.y, e.z, y.x, y.y, 0.0);
            } else {
                float m0 = e.x * y.x;
                float q0 = e.x * e.x - e.y * e.y;
                bool done2 = false;
                if (m0 < q0) {
                    float x0 = m0 / q0;
                    float discr = 1.0 - x0 * x0;
                    if (discr > 0.0) {
                        float a = e.x * x0 - y.x; float b = e.y * sqrt(discr);
                        d2 = a * a + b * b;
                        done2 = true;
                    }
                }
                if (!done2) { float a = y.x - e.x; d2 = a * a; }
            }
        }
    }
    bool inside = (y.x / e.x) * (y.x / e.x) + (y.y / e.y) * (y.y / e.y) + (y.z / e.z) * (y.z / e.z) < 1.0;
    float d = sqrt(max(d2, 0.0));
    return inside ? -d : d;
}
";

const HELPER_SDF_HELIX: &str = r"float sdf_helix(float3 p, float major_r, float minor_r, float pitch, float hh) {
    const float tau = 6.28318530717959;
    float r = length(float2(p.x, p.z));
    float theta = (r > 0.0) ? atan2(p.z, p.x) : 0.0;
    float py = p.y;
    float c = pitch / tau;
    float two_rr = 2.0 * r * major_r;
    float k = floor((py - theta * c) / pitch + 0.5);
    float best = 1e30;
    for (float dk = -1.0; dk <= 1.0; dk += 1.0) {
        float phi = theta + (k + dk) * tau;
        for (int it = 0; it < 6; it++) {
            float s = sin(phi - theta);
            float co = cos(phi - theta);
            float dy = py - c * phi;
            float f1 = two_rr * s - 2.0 * c * dy;
            float f2 = max(two_rr * co + 2.0 * c * c, 1e-6);
            phi = phi - clamp(f1 / f2, -1.5707963, 1.5707963);
        }
        float co = cos(phi - theta);
        float dy = py - c * phi;
        float d2 = r * r + major_r * major_r - two_rr * co + dy * dy;
        best = min(best, d2);
    }
    float d_tube = sqrt(best) - minor_r;
    float d_cap = abs(py) - hh;
    return max(d_tube, d_cap);
}
";

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
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.3);
        let shader = HlslShader::transpile(&shape, HlslTranspileMode::Dynamic);

        // Should contain cbuffer references
        assert!(shader.source.contains("params["));
        // Smooth union emits k and inv_k (Division Exorcism)
        let k_idx = shader
            .param_layout
            .iter()
            .position(|&v| (v - 0.3).abs() < 1e-6);
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

    #[test]
    fn test_ue5_material_function_export() {
        let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.5, 0.5, 0.5));
        let shader = HlslShader::transpile(&shape, HlslTranspileMode::Hardcoded);
        let mf = shader.export_ue5_material_function();

        assert!(mf.contains("#pragma once"));
        assert!(mf.contains("AliceSdf_Eval"));
        assert!(mf.contains("AliceSdf_Normal"));
        assert!(mf.contains("sdf_eval"));
        assert!(mf.contains("MF_AliceSdf.ush"));
        assert!(!mf.contains("cbuffer SdfParams"));

        // Dynamic mode should include params
        let shader_dyn = HlslShader::transpile(&shape, HlslTranspileMode::Dynamic);
        let mf_dyn = shader_dyn.export_ue5_material_function();
        assert!(mf_dyn.contains("cbuffer SdfParams"));
    }

    /// Exhaustive test: every SdfNode variant transpiles without panic
    /// and produces valid HLSL containing `sdf_eval`.
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
            let shader = HlslShader::transpile(prim, HlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(float3 p)"),
                "HLSL transpile failed for primitive '{}': missing sdf_eval",
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
            ("exp_smooth_subtract", a.exp_smooth_subtract(b, 0.2)),
        ];

        for (name, op) in &operations {
            let shader = HlslShader::transpile(op, HlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(float3 p)"),
                "HLSL transpile failed for operation '{}': missing sdf_eval",
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
            ("with_material", s.with_material(1)),
        ];

        for (name, m) in &modifiers {
            let shader = HlslShader::transpile(m, HlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(float3 p)"),
                "HLSL transpile failed for modifier '{}': missing sdf_eval",
                name,
            );
        }
    }
}

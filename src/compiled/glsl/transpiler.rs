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

use super::super::transpiler_common::ShaderLang;
use crate::types::SdfNode;

// ============================================================================
// GlslLang: ShaderLang trait implementation for GLSL
// ============================================================================

/// Marker type for GLSL shader language syntax
pub struct GlslLang;

impl crate::compiled::transpiler_common::private::Sealed for GlslLang {}

impl ShaderLang for GlslLang {
    const NAME: &'static str = "glsl";

    fn vec2_ctor(x: &str, y: &str) -> String {
        format!("vec2({}, {})", x, y)
    }
    fn vec3_ctor(x: &str, y: &str, z: &str) -> String {
        format!("vec3({}, {}, {})", x, y, z)
    }
    fn vec4_ctor(x: &str, y: &str, z: &str, w: &str) -> String {
        format!("vec4({}, {}, {}, {})", x, y, z, w)
    }
    fn vec2_zero() -> &'static str {
        "vec2(0.0)"
    }
    fn vec3_zero() -> &'static str {
        "vec3(0.0)"
    }
    fn vec2_splat(v: &str) -> String {
        format!("vec2({})", v)
    }
    fn vec3_splat(v: &str) -> String {
        format!("vec3({})", v)
    }
    fn decl_float(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE float {} = {};\n", name, expr)
    }
    fn decl_vec2(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE vec2 {} = {};\n", name, expr)
    }
    fn decl_vec3(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE vec3 {} = {};\n", name, expr)
    }
    fn decl_mut_float(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE float {} = {};\n", name, expr)
    }
    fn decl_mut_vec2(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE vec2 {} = {};\n", name, expr)
    }
    fn decl_mut_vec3(name: &str, expr: &str) -> String {
        format!("    ALICE_PRECISE vec3 {} = {};\n", name, expr)
    }
    fn decl_mut_float_typed(name: &str) -> String {
        format!("    ALICE_PRECISE float {};\n", name)
    }
    fn decl_mut_vec3_typed(name: &str) -> String {
        format!("    ALICE_PRECISE vec3 {};\n", name)
    }
    fn select_expr(cond: &str, true_val: &str, false_val: &str) -> String {
        format!("(({}) ? {} : {})", cond, true_val, false_val)
    }
    fn modulo_expr(a: &str, b: &str) -> String {
        format!("mod({}, {})", a, b)
    }
    fn atan2_expr(y: &str, x: &str) -> String {
        format!("alice_atan2({y}, {x})")
    }
    fn cast_float(expr: &str) -> String {
        format!("float({})", expr)
    }
    fn cast_int(expr: &str) -> String {
        format!("int({})", expr)
    }
    fn decl_int(name: &str, expr: &str) -> String {
        format!("    int {name} = {expr};\n")
    }
    fn global_float_array(name: &str, values: &[f32]) -> String {
        let n = values.len();
        let items: Vec<String> = values
            .iter()
            .map(|v| super::super::transpiler_common::lit(*v))
            .collect();
        format!(
            "const float {name}[{n}] = float[{n}]({});\n",
            items.join(", ")
        )
    }
    fn global_vec3_fn(name: &str, body: &str) -> String {
        format!("vec3 {name}(vec3 p) {{\n{body}}}\n")
    }
    fn for_loop_int(name: &str, init: i32, cond: &str, incr: &str) -> String {
        format!("    for (int {} = {}; {}; {}) {{\n", name, init, cond, incr)
    }
    fn param_dynamic(vec_idx: usize, comp: &str) -> String {
        format!("params[{}].{}", vec_idx, comp)
    }
    fn func_signature() -> &'static str {
        "float sdf_eval(vec3 p) {"
    }
    const CAPSULE_DEGENERATE_GUARD: bool = false;
    const CAPSULE_RE_EMIT_PARAMS: bool = true;
    fn helper_source(name: &str) -> Option<&'static str> {
        match name {
            "quat_rotate" => Some(HELPER_QUAT_ROTATE),
            "alice_atan2" => Some(HELPER_ALICE_ATAN2),
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
            "columns" => Some(HELPER_COLUMNS),
            "icosahedral_fold" => Some(HELPER_ICOSAHEDRAL_FOLD),
            "octant_mirror" => Some(HELPER_OCTANT_MIRROR),
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
    ///
    /// Every floating-point local is declared `ALICE_PRECISE` (ALICE-Law G3: no
    /// contraction of `a * b + c` into one rounding). The qualifier is `precise`
    /// from GLSL 4.00 and GLSL ES 3.20; with an older `version` (330, or GLSL ES
    /// 3.00 as in `render_pipeline`) it is empty and the driver may contract.
    pub fn transpile_with_version(node: &SdfNode, mode: GlslTranspileMode, version: u32) -> Self {
        let mut transpiler = GlslTranspiler::new(mode);
        let body = transpiler.transpile_node(node, "p");

        let source = transpiler.generate_shader(&body);

        Self {
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

    /// GLSL text wrapped in a Unity Shader Graph Custom Function signature
    ///
    /// **The output is GLSL syntax (`vec3`, `mix`, ...) and does not compile in
    /// Unity Shader Graph, which takes HLSL.** A `GlslShader` holds only the
    /// GLSL source, so it cannot produce the HLSL; transpile the same node with
    /// `HlslShader` (`hlsl` feature) and call its `to_unity_custom_function`,
    /// whose output is compiled in the test suite. Kept unchanged for existing
    /// callers; deprecated since 5.0.0.
    #[deprecated(
        since = "5.0.0",
        note = "the output is GLSL and does not compile in Unity Shader Graph; transpile with HlslShader and use HlslShader::to_unity_custom_function"
    )]
    pub fn to_unity_custom_function(&self) -> String {
        let dynamic_note = if self.mode == GlslTranspileMode::Dynamic {
            "// NOTE: Dynamic mode - parameters are read from uniform block SdfParams.\n// Set up a UBO to update params at runtime.\n"
        } else {
            ""
        };

        format!(
            r"// ALICE-SDF Generated GLSL for Unity Custom Function
// Input: float3 p (World Position)
// Output: float (SDF Distance)
{dynamic_note}
{source}

void SdfEval_float(float3 p, out float distance) {{
    ALICE_PRECISE float alice_d = sdf_eval(p);
    distance = alice_d;
}}
",
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
            r"#version {}

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
    ALICE_PRECISE vec3 p = vec3(pt.x, pt.y, pt.z);
    output_distances[idx].distance = sdf_eval(p);
}}
",
            self.version,
            mode = self.mode,
            params_decl = params_decl,
            source = self.source
        )
    }

    /// Generate a Vulkan GLSL compute shader for batch evaluation
    ///
    /// [`Self::to_compute_shader`] targets OpenGL: its loose
    /// `layout(location = 0) uniform uint point_count` and its blocks without a
    /// descriptor set are not valid Vulkan GLSL (GL_KHR_vulkan_glsl). This
    /// output is the same kernel with a Vulkan host interface, always
    /// `#version 450` (ignoring [`Self::version`]), descriptor set 0:
    ///
    /// | binding | resource |
    /// |---|---|
    /// | 0 | `readonly buffer InputBuffer { InputPoint input_points[]; }` (`x, y, z, pad` as 4 floats) |
    /// | 1 | `buffer OutputBuffer { OutputDistance output_distances[]; }` (`distance, pad, pad, pad`) |
    /// | 2 | `uniform Constants { uint point_count; }` |
    /// | 3 | Dynamic mode only: `uniform SdfParams { vec4 params[1024]; }`, laid out as [`Self::extract_params`] |
    ///
    /// Bindings 0-2 are the layout `GpuEvaluator` (`gpu` feature) binds, so
    /// a Hardcoded shader runs through `GpuEvaluator::from_glsl_compute`.
    pub fn to_vulkan_compute_shader(&self) -> String {
        let params_decl = if self.mode == GlslTranspileMode::Dynamic {
            "\nlayout(std140, set = 0, binding = 3) uniform SdfParams {\n    vec4 params[1024]; // 4096 scalar floats\n};\n"
        } else {
            ""
        };

        format!(
            r"#version 450

// ALICE-SDF Generated Vulkan GLSL Compute Shader ({mode:?} Mode)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

struct InputPoint {{
    float x, y, z, _pad;
}};

struct OutputDistance {{
    float distance, _pad1, _pad2, _pad3;
}};

layout(std430, set = 0, binding = 0) readonly buffer InputBuffer {{
    InputPoint input_points[];
}};

layout(std430, set = 0, binding = 1) buffer OutputBuffer {{
    OutputDistance output_distances[];
}};

layout(std140, set = 0, binding = 2) uniform Constants {{
    uint point_count;
}};
{params_decl}
{source}

void main() {{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= point_count) return;

    InputPoint pt = input_points[idx];
    ALICE_PRECISE vec3 p = vec3(pt.x, pt.y, pt.z);
    output_distances[idx].distance = sdf_eval(p);
}}
",
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
            r"#version {}

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
    // the sample points are precise locals: glslang also keeps the call arguments of
    // a precise expression uncontracted, a local does not depend on that
    ALICE_PRECISE vec3 p0 = p + k.xyy * h;
    ALICE_PRECISE vec3 p1 = p + k.yyx * h;
    ALICE_PRECISE vec3 p2 = p + k.yxy * h;
    ALICE_PRECISE vec3 p3 = p + k.xxx * h;
    ALICE_PRECISE vec3 n = normalize(
        k.xyy * sdf_eval(p0) +
        k.yyx * sdf_eval(p1) +
        k.yxy * sdf_eval(p2) +
        k.xxx * sdf_eval(p3)
    );
    return n;
}}

void main() {{
    ALICE_PRECISE vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

    // Camera
    ALICE_PRECISE vec3 ro = vec3(0.0, 0.0, 5.0);
    ALICE_PRECISE vec3 rd = normalize(vec3(uv, -1.5));

    // Raymarching
    ALICE_PRECISE float t = 0.0;
    for (int i = 0; i < 128; i++) {{
        ALICE_PRECISE vec3 p = ro + rd * t;
        ALICE_PRECISE float d = sdf_eval(p);
        if (d < 0.001 || t > 100.0) break;
        t += d;
    }}

    // Shading
    ALICE_PRECISE vec3 col = vec3(0.0);
    if (t < 100.0) {{
        ALICE_PRECISE vec3 p = ro + rd * t;
        ALICE_PRECISE vec3 n = calcNormal(p);
        ALICE_PRECISE vec3 light = normalize(vec3(1.0, 1.0, 1.0));
        ALICE_PRECISE float diff = max(dot(n, light), 0.0);
        col = vec3(0.8, 0.7, 0.6) * (0.2 + 0.8 * diff);
    }}

    fragColor = vec4(col, 1.0);
}}
",
            self.version,
            params_decl = params_decl,
            source = self.source
        )
    }

    /// Get the SDF evaluation function only (for embedding in custom shaders)
    pub fn get_eval_function(&self) -> &str {
        &self.source
    }

    /// フルレンダリングパイプライン付きフラグメントシェーダーを生成
    ///
    /// PBR (Cook-Torrance GGX), 大気散乱 (Rayleigh/Mie), 昼夜サイクル,
    /// 天候 (霧/雨/雷), ボリュメトリックライト, ポストプロセス (ACES/ブルーム/CA)
    /// を含むフルシェーダーを出力する。
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use alice_sdf::prelude::*;
    /// use alice_sdf::compiled::glsl::{GlslShader, GlslTranspileMode, RenderConfig};
    ///
    /// let shape = SdfNode::sphere(1.0)
    ///     .smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.2);
    ///
    /// let shader = GlslShader::transpile(&shape, GlslTranspileMode::Hardcoded);
    /// let full = shader.to_fragment_shader_full(&RenderConfig::default());
    /// ```
    ///
    /// With `config.dual_sdf` the light-weight `sdf_eval_lite` used for AO /
    /// shadow / rain is the scene itself (a transpiled tree has no separate
    /// simplified version).
    ///
    /// # Panics
    ///
    /// Panics if `config.material_slots > 1`. A transpiled tree has a single
    /// material and its `sdf_eval` returns a `float`, while the multi-material
    /// pipeline needs `vec2 sdf_eval` and a `getMat`; the shader would not
    /// compile. For a multi-material scene write that source and call
    /// [`super::render_pipeline::build_full_shader`].
    pub fn to_fragment_shader_full(&self, config: &super::render_pipeline::RenderConfig) -> String {
        assert!(
            config.material_slots <= 1,
            "GlslShader::to_fragment_shader_full: material_slots = {} needs a multi-material \
             source (vec2 sdf_eval + getMat); a transpiled tree has one material. \
             Use render_pipeline::build_full_shader with your own source.",
            config.material_slots
        );
        if config.dual_sdf {
            let source = format!(
                "{}\nfloat sdf_eval_lite(vec3 p) {{ return sdf_eval(p); }}\n",
                self.source
            );
            super::render_pipeline::build_full_shader(&source, config)
        } else {
            super::render_pipeline::build_full_shader(&self.source, config)
        }
    }

    /// GLSL text wrapped in a Unity Shader Graph Custom Function file layout
    ///
    /// **The embedded SDF functions are GLSL syntax (`vec3`, `mix`, ...) inside
    /// an `.hlsl` file, so the output does not compile in Unity Shader Graph.**
    /// A `GlslShader` holds only the GLSL source, so it cannot produce the
    /// HLSL; transpile the same node with `HlslShader` (`hlsl` feature) and
    /// call its `export_unity_shader_graph`, whose output is compiled in the
    /// test suite. Kept unchanged for existing callers; deprecated since
    /// 5.0.0.
    #[deprecated(
        since = "5.0.0",
        note = "the output embeds GLSL and does not compile in Unity Shader Graph; transpile with HlslShader and use HlslShader::export_unity_shader_graph"
    )]
    pub fn export_unity_shader_graph(&self) -> String {
        let params_section = if self.mode == GlslTranspileMode::Dynamic {
            "// Dynamic parameters - bind via MaterialPropertyBlock or UBO\nuniform float4 _SdfParams[1024];\n#define params _SdfParams\n\n"
        } else {
            ""
        };

        // Unity Shader Graph Custom Function files use HLSL syntax
        format!(
            r"// ALICE-SDF Custom Function for Unity Shader Graph
// Generated by ALICE-SDF Compiler
// Usage: Custom Function node → Mode: File → Source: AliceSdf.hlsl
//        Function Name: AliceSdf
//        Inputs: float3 Position
//        Outputs: float Distance, float3 Normal

#ifndef ALICE_SDF_INCLUDED
#define ALICE_SDF_INCLUDED

{params_section}{source}

void AliceSdf_float(float3 Position, out float Distance, out float3 Normal) {{
    ALICE_PRECISE float alice_d = sdf_eval(Position);
    Distance = alice_d;

    // Central difference normal
    ALICE_PRECISE float e = 0.001;
    ALICE_PRECISE float3 n = normalize(float3(
        sdf_eval(Position + float3(e,0,0)) - sdf_eval(Position - float3(e,0,0)),
        sdf_eval(Position + float3(0,e,0)) - sdf_eval(Position - float3(0,e,0)),
        sdf_eval(Position + float3(0,0,e)) - sdf_eval(Position - float3(0,0,e))
    ));
    Normal = n;
}}

void AliceSdf_half(float3 Position, out float Distance, out float3 Normal) {{
    AliceSdf_float(Position, Distance, Normal);
}}

#endif // ALICE_SDF_INCLUDED
",
            params_section = params_section,
            source = self.source
        )
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
    /// Module-scope data / functions emitted before the entry point
    globals: String,
}

impl GlslTranspiler {
    const fn new(mode: GlslTranspileMode) -> Self {
        Self {
            var_counter: 0,
            helper_functions: Vec::new(),
            mode,
            params: Vec::new(),
            globals: String::new(),
        }
    }

    fn transpile_node(&mut self, node: &SdfNode, point_var: &str) -> String {
        use super::super::transpiler_common::{GenericTranspiler, TranspileModeLang};
        let mode = match self.mode {
            GlslTranspileMode::Hardcoded => TranspileModeLang::Hardcoded,
            GlslTranspileMode::Dynamic => TranspileModeLang::Dynamic,
        };
        let mut generic: GenericTranspiler<GlslLang> = GenericTranspiler::new(mode);
        generic.var_counter = self.var_counter;
        generic.params = std::mem::take(&mut self.params);
        generic.helper_functions = std::mem::take(&mut self.helper_functions);
        generic.globals = std::mem::take(&mut self.globals);
        let body = generic.transpile_node(node, point_var);
        self.globals = generic.globals;
        self.var_counter = generic.var_counter;
        self.params = generic.params;
        self.helper_functions = generic.helper_functions;
        body
    }

    fn generate_shader(&self, body: &str) -> String {
        use super::super::transpiler_common::{GenericTranspiler, TranspileModeLang};
        // Assembled by the language-generic `GenericTranspiler::generate_shader`
        // (helpers from `ShaderLang::helper_source`, then the module-scope
        // globals, then `sdf_eval`) so the three languages share one layout.
        let mut generic: GenericTranspiler<GlslLang> =
            GenericTranspiler::new(TranspileModeLang::Hardcoded);
        generic.helper_functions.clone_from(&self.helper_functions);
        generic.globals.clone_from(&self.globals);
        format!("{PRECISE_MACRO}{}", generic.generate_shader(body))
    }

    // transpile_node_inner has been moved to GenericTranspiler<GlslLang>
    // in transpiler_common.rs — see the transpile_node delegation method above.
}

/// The `ALICE_PRECISE` definition every GLSL source starts with (ALICE-Law G3,
/// see `glsl_precise_macro`).
const PRECISE_MACRO: &str = concat!(crate::modifiers::glsl_precise_macro!(), "\n");

// Helper function definitions for GLSL
const HELPER_TAPER_BOUND: &str = r"// Same law as `alice_sdf::compiled::real::taper_bound` (Jacobian ball + cone ∩ slab).
float alice_taper_bound(float d, vec3 p, float factor, float rx, float ry) {
    ALICE_PRECISE float fa = abs(factor);
    if (fa == 0.0) { return d; }
    ALICE_PRECISE float den_abs = max(abs(1.0 - p.y * factor), 1e-6);
    ALICE_PRECISE float s = 1.0 / den_abs;
    ALICE_PRECISE float rho = sqrt(p.x * p.x + p.z * p.z);
    ALICE_PRECISE float d_abs = abs(d);
    ALICE_PRECISE float j0 = max(s, 1.0) + fa * s * s * rho;
    ALICE_PRECISE float r = min(d_abs / j0, den_abs / (fa + fa));
    ALICE_PRECISE float den1 = max(den_abs - fa * r, 1e-6);
    ALICE_PRECISE float s1 = 1.0 / den1;
    ALICE_PRECISE float j1 = max(s1, 1.0) + fa * s1 * s1 * (rho + r);
    ALICE_PRECISE float mag = min(d_abs / j1, r);
    ALICE_PRECISE float d_j = (d < 0.0) ? -mag : mag;
    if (rx >= 1e30 || ry >= 1e30) { return d_j; }
    ALICE_PRECISE float k = rx * fa;
    ALICE_PRECISE float inv_n = 1.0 / sqrt(1.0 + k * k);
    ALICE_PRECISE float big_y = p.y - 1.0 / factor;
    ALICE_PRECISE float d_cone = (rho - k * abs(big_y)) * inv_n;
    ALICE_PRECISE float d_slab = abs(p.y) - ry;
    ALICE_PRECISE float d_region = max(d_cone, d_slab);
    ALICE_PRECISE float alice_r = (d < 0.0) ? d_j : max(d_j, d_region);
    return alice_r;
}";

const HELPER_HASH_NOISE: &str = crate::modifiers::HASH_NOISE_GLSL;

const HELPER_PERLIN_NOISE: &str = r"uint alice_perlin_hash(int x, int y, int z, uint seed) {
    uint h = seed;
    h ^= uint(x);
    h *= 0x85EBCA6Bu;
    h ^= uint(y);
    h *= 0xC2B2AE35u;
    h ^= uint(z);
    h *= 0x27D4EB2Du;
    h ^= h >> 16u;
    return h;
}
float alice_perlin_grad(uint h, vec3 c) {
    const uint LUT[16] = uint[16](4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 24u, 25u, 26u, 27u, 16u, 10u, 18u, 9u);
    uint e = LUT[h & 15u];
    uint u_src = (e >> 4u) & 3u;
    uint v_src = (e >> 2u) & 3u;
    ALICE_PRECISE float u_sign = 1.0 - 2.0 * float((e >> 1u) & 1u);
    ALICE_PRECISE float v_sign = 1.0 - 2.0 * float(e & 1u);
    ALICE_PRECISE float alice_r = c[u_src] * u_sign + c[v_src] * v_sign;
    return alice_r;
}
float alice_perlin_fade(float t) {
    ALICE_PRECISE float alice_r = t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    return alice_r;
}
// Same law as `alice_sdf::modifiers::perlin_noise_3d` (integer lattice hash + 16-gradient LUT + quintic fade).
float perlin_noise_3d(vec3 p, uint seed) {
    ALICE_PRECISE vec3 fl = floor(p);
    int xi = int(fl.x);
    int yi = int(fl.y);
    int zi = int(fl.z);
    ALICE_PRECISE vec3 f = p - fl;
    ALICE_PRECISE float u = alice_perlin_fade(f.x);
    ALICE_PRECISE float v = alice_perlin_fade(f.y);
    ALICE_PRECISE float w = alice_perlin_fade(f.z);
    ALICE_PRECISE float aaa = alice_perlin_grad(alice_perlin_hash(xi, yi, zi, seed), f);
    ALICE_PRECISE float aba = alice_perlin_grad(alice_perlin_hash(xi, yi + 1, zi, seed), vec3(f.x, f.y - 1.0, f.z));
    ALICE_PRECISE float aab = alice_perlin_grad(alice_perlin_hash(xi, yi, zi + 1, seed), vec3(f.x, f.y, f.z - 1.0));
    ALICE_PRECISE float abb = alice_perlin_grad(alice_perlin_hash(xi, yi + 1, zi + 1, seed), vec3(f.x, f.y - 1.0, f.z - 1.0));
    ALICE_PRECISE float baa = alice_perlin_grad(alice_perlin_hash(xi + 1, yi, zi, seed), vec3(f.x - 1.0, f.y, f.z));
    ALICE_PRECISE float bba = alice_perlin_grad(alice_perlin_hash(xi + 1, yi + 1, zi, seed), vec3(f.x - 1.0, f.y - 1.0, f.z));
    ALICE_PRECISE float bab = alice_perlin_grad(alice_perlin_hash(xi + 1, yi, zi + 1, seed), vec3(f.x - 1.0, f.y, f.z - 1.0));
    ALICE_PRECISE float bbb = alice_perlin_grad(alice_perlin_hash(xi + 1, yi + 1, zi + 1, seed), vec3(f.x - 1.0, f.y - 1.0, f.z - 1.0));
    ALICE_PRECISE float x0 = aaa + u * (baa - aaa);
    ALICE_PRECISE float x1 = aba + u * (bba - aba);
    ALICE_PRECISE float x2 = aab + u * (bab - aab);
    ALICE_PRECISE float x3 = abb + u * (bbb - abb);
    ALICE_PRECISE float y0 = x0 + v * (x1 - x0);
    ALICE_PRECISE float y1 = x2 + v * (x3 - x2);
    ALICE_PRECISE float alice_r = y0 + w * (y1 - y0);
    return alice_r;
}";

/// `atan2` with the CPU law's exact axis cases (alice-det-math `atan2`,
/// fdlibm rules): the GPU builtin is the tolerance domain, but an exact tie on
/// an axis (`atan2(±0, x < 0) = ±π`, `atan2(y, ±0) = ±π/2`) must snap to the
/// same sector as the CPU, so those return the same `f32` constants. The sign
/// of zero is read from the bits (`sign(-0.0)` is 0 in GLSL). Single exit,
/// no early `return`: naga's GLSL front end miscompiled the nested-return
/// form (the function returned 0 for every negative `y`).
const HELPER_ALICE_ATAN2: &str = r"float alice_atan2(float y, float x) {
    bool sy = floatBitsToInt(y) < 0;
    bool sx = floatBitsToInt(x) < 0;
    ALICE_PRECISE float r = atan(y, x);
    r = (x == 0.0) ? (sy ? -1.5707964 : 1.5707964) : r;
    r = (y == 0.0) ? (sx ? (sy ? -3.1415927 : 3.1415927) : y) : r;
    return r;
}
";

const HELPER_QUAT_ROTATE: &str = r"vec3 quat_rotate(vec3 v, vec4 q) {
    ALICE_PRECISE vec3 t = 2.0 * cross(q.xyz, v);
    ALICE_PRECISE vec3 alice_r = v + q.w * t + cross(q.xyz, t);
    return alice_r;
}";

const HELPER_SDF_ROUNDED_CONE: &str = r"float sdf_rounded_cone(vec3 p, float r1, float r2, float h) {
    ALICE_PRECISE float qx = length(p.xz);
    ALICE_PRECISE float qy = p.y + h;
    ALICE_PRECISE float ht = h * 2.0;
    ALICE_PRECISE float b = (r1 - r2) / ht;
    ALICE_PRECISE float a = sqrt(1.0 - b * b);
    ALICE_PRECISE float k = qx * (-b) + qy * a;
    if (k < 0.0) { ALICE_PRECISE float alice_r = length(vec2(qx, qy)) - r1; return alice_r; }
    if (k > a * ht) { ALICE_PRECISE float alice_r = length(vec2(qx, qy - ht)) - r2; return alice_r; }
    ALICE_PRECISE float alice_r = qx * a + qy * b - r1;
    return alice_r;
}
";

const HELPER_SDF_PYRAMID: &str = r"float sdf_pyramid(vec3 p, float h) {
    ALICE_PRECISE float ht = h * 2.0;
    ALICE_PRECISE float m2 = ht * ht + 0.25;
    ALICE_PRECISE float py = p.y + h;
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float pz = abs(p.z);
    if (pz > px) { ALICE_PRECISE float tmp = px; px = pz; pz = tmp; }
    px -= 0.5;
    pz -= 0.5;
    ALICE_PRECISE float qx = pz;
    ALICE_PRECISE float qy = ht * py - 0.5 * px;
    ALICE_PRECISE float qz = ht * px + 0.5 * py;
    ALICE_PRECISE float s = max(-qx, 0.0);
    ALICE_PRECISE float t = clamp((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    ALICE_PRECISE float a = m2 * (qx + s) * (qx + s) + qy * qy;
    ALICE_PRECISE float b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
    ALICE_PRECISE float d2 = (min(-qx * m2 - qy * 0.5, qy) > 0.0) ? 0.0 : min(a, b);
    ALICE_PRECISE float alice_r = sqrt((d2 + qz * qz) / m2) * (max(qz, -py) < 0.0 ? -1.0 : 1.0);
    return alice_r;
}
";

const HELPER_SDF_OCTAHEDRON: &str = r"float sdf_octahedron(vec3 p, float s) {
    ALICE_PRECISE vec3 ap = abs(p);
    ALICE_PRECISE float m = ap.x + ap.y + ap.z - s;
    ALICE_PRECISE vec3 q;
    if (3.0 * ap.x < m) q = ap;
    else if (3.0 * ap.y < m) q = vec3(ap.y, ap.z, ap.x);
    else if (3.0 * ap.z < m) q = vec3(ap.z, ap.x, ap.y);
    else { ALICE_PRECISE float alice_r = m * 0.57735027; return alice_r; }
    ALICE_PRECISE float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    ALICE_PRECISE float alice_r = length(vec3(q.x, q.y - s + k, q.z - k));
    return alice_r;
}
";

const HELPER_SDF_HEX_PRISM: &str = r"float sdf_hex_prism(vec3 p, float hex_r, float h) {
    ALICE_PRECISE float kx = -0.8660254;
    ALICE_PRECISE float ky = 0.5;
    ALICE_PRECISE float kz = 0.57735027;
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float py = abs(p.y);
    ALICE_PRECISE float pz = abs(p.z);
    ALICE_PRECISE float dot_kxy = kx * px + ky * py;
    ALICE_PRECISE float refl = 2.0 * min(dot_kxy, 0.0);
    px -= refl * kx;
    py -= refl * ky;
    ALICE_PRECISE float clamped_x = clamp(px, -kz * hex_r, kz * hex_r);
    ALICE_PRECISE float dx = px - clamped_x;
    ALICE_PRECISE float dy = py - hex_r;
    ALICE_PRECISE float d_xy = sqrt(dx * dx + dy * dy) * (dy < 0.0 ? -1.0 : 1.0);
    ALICE_PRECISE float d_z = pz - h;
    ALICE_PRECISE float alice_r = min(max(d_xy, d_z), 0.0) + length(max(vec2(d_xy, d_z), vec2(0.0)));
    return alice_r;
}
";

const HELPER_SDF_TRIANGLE: &str = r"float sdf_triangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    ALICE_PRECISE vec3 ba = b - a; ALICE_PRECISE vec3 pa = p - a;
    ALICE_PRECISE vec3 cb = c - b; ALICE_PRECISE vec3 pb = p - b;
    ALICE_PRECISE vec3 ac = a - c; ALICE_PRECISE vec3 pc = p - c;
    ALICE_PRECISE vec3 nor = cross(ba, ac);
    ALICE_PRECISE float alice_r = sqrt(
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
    return alice_r;
}
";

const HELPER_SDF_BEZIER: &str = r"float sdf_bezier(vec3 pos, vec3 A, vec3 B, vec3 C, float r) {
    ALICE_PRECISE vec3 a = B - A;
    ALICE_PRECISE vec3 b = A - 2.0 * B + C;
    ALICE_PRECISE vec3 c = a * 2.0;
    ALICE_PRECISE vec3 d = A - pos;
    ALICE_PRECISE float kk = 1.0 / max(dot(b, b), 1e-10);
    ALICE_PRECISE float kx = kk * dot(a, b);
    ALICE_PRECISE float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    ALICE_PRECISE float kz = kk * dot(d, a);
    ALICE_PRECISE float p2 = ky - kx * kx;
    ALICE_PRECISE float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    ALICE_PRECISE float p3 = p2 * p2 * p2;
    ALICE_PRECISE float h = q * q + 4.0 * p3;
    ALICE_PRECISE float res;
    if (h >= 0.0) {
        h = sqrt(h);
        ALICE_PRECISE vec2 x = (vec2(h, -h) - q) * 0.5;
        ALICE_PRECISE vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        ALICE_PRECISE float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        res = dot(d + (c + b * t) * t, d + (c + b * t) * t);
    } else {
        ALICE_PRECISE float z = sqrt(-p2);
        ALICE_PRECISE float v2 = acos(q / (p2 * z * 2.0)) / 3.0;
        ALICE_PRECISE float m = cos(v2);
        ALICE_PRECISE float n = sin(v2) * 1.732050808;
        ALICE_PRECISE float t1 = clamp(( m + m) * z - kx, 0.0, 1.0);
        ALICE_PRECISE float t2 = clamp((-n - m) * z - kx, 0.0, 1.0);
        ALICE_PRECISE float d1 = dot(d + (c + b * t1) * t1, d + (c + b * t1) * t1);
        ALICE_PRECISE float d2 = dot(d + (c + b * t2) * t2, d + (c + b * t2) * t2);
        res = min(d1, d2);
    }
    ALICE_PRECISE float alice_r = sqrt(res) - r;
    return alice_r;
}
";

const HELPER_SDF_LINK: &str = r"float sdf_link(vec3 p, float le, float r1, float r2) {
    ALICE_PRECISE float qx = p.x;
    ALICE_PRECISE float qy = max(abs(p.y) - le, 0.0);
    ALICE_PRECISE float qz = p.z;
    ALICE_PRECISE float xy_len = sqrt(qx * qx + qy * qy) - r1;
    ALICE_PRECISE float alice_r = sqrt(xy_len * xy_len + qz * qz) - r2;
    return alice_r;
}
";

const HELPER_SDF_CAPPED_CONE: &str = r"float sdf_capped_cone(vec3 p, float h, float r1, float r2) {
    ALICE_PRECISE float qx = length(p.xz);
    ALICE_PRECISE float qy = p.y;
    ALICE_PRECISE vec2 k1 = vec2(r2, h);
    ALICE_PRECISE vec2 k2 = vec2(r2 - r1, 2.0 * h);
    ALICE_PRECISE float min_r = (qy < 0.0) ? r1 : r2;
    ALICE_PRECISE vec2 ca = vec2(qx - min(qx, min_r), abs(qy) - h);
    ALICE_PRECISE vec2 q = vec2(qx, qy);
    ALICE_PRECISE vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    ALICE_PRECISE float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    ALICE_PRECISE float alice_r = s * sqrt(min(dot(ca, ca), dot(cb, cb)));
    return alice_r;
}
";

const HELPER_SDF_CAPPED_TORUS: &str = r"float sdf_capped_torus(vec3 p, float ra, float rb, float an) {
    ALICE_PRECISE vec2 sc = vec2(sin(an), cos(an));
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float k = (sc.y * px > sc.x * p.y) ? dot(vec2(px, p.y), sc) : length(vec2(px, p.y));
    ALICE_PRECISE float alice_r = sqrt(px * px + p.y * p.y + p.z * p.z + ra * ra - 2.0 * ra * k) - rb;
    return alice_r;
}
";

const HELPER_SDF_ROUNDED_CYLINDER: &str = r"float sdf_rounded_cylinder(vec3 p, float r, float rr, float h) {
    ALICE_PRECISE vec2 d = vec2(length(p.xz) - r + rr, abs(p.y) - h);
    ALICE_PRECISE float alice_r = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0))) - rr;
    return alice_r;
}
";

const HELPER_SDF_TRIANGULAR_PRISM: &str = r"float sdf_triangular_prism(vec3 p, float w, float h) {
    ALICE_PRECISE vec3 q = abs(p);
    ALICE_PRECISE float alice_r = max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - w * 0.5);
    return alice_r;
}
";

const HELPER_SDF_CUT_SPHERE: &str = r"float sdf_cut_sphere(vec3 p, float r, float h) {
    ALICE_PRECISE float w = sqrt(max(r * r - h * h, 0.0));
    ALICE_PRECISE vec2 q = vec2(length(p.xz), p.y);
    ALICE_PRECISE float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
    if (s < 0.0) { ALICE_PRECISE float alice_r = length(q) - r; return alice_r; }
    if (q.x < w) { ALICE_PRECISE float alice_r = h - q.y; return alice_r; }
    ALICE_PRECISE float alice_r = length(q - vec2(w, h));
    return alice_r;
}
";

const HELPER_SDF_CUT_HOLLOW_SPHERE: &str = r"float sdf_cut_hollow_sphere(vec3 p, float r, float h, float t) {
    ALICE_PRECISE float w = sqrt(max(r * r - h * h, 0.0));
    ALICE_PRECISE vec2 q = vec2(length(p.xz), p.y);
    if (h * q.x < w * q.y) { ALICE_PRECISE float alice_r = length(q - vec2(w, h)) - t; return alice_r; }
    ALICE_PRECISE float alice_r = abs(length(q) - r) - t;
    return alice_r;
}
";

const HELPER_SDF_DEATH_STAR: &str = r"float sdf_death_star(vec3 p, float ra, float rb, float d) {
    ALICE_PRECISE float a = (ra * ra - rb * rb + d * d) / (2.0 * d);
    ALICE_PRECISE float b = sqrt(max(ra * ra - a * a, 0.0));
    ALICE_PRECISE vec2 q = vec2(p.x, length(p.yz));
    ALICE_PRECISE float alice_lhs = q.x * b - q.y * a;
    ALICE_PRECISE float alice_rhs = d * max(b - q.y, 0.0);
    if (alice_lhs > alice_rhs)
        { ALICE_PRECISE float alice_r = length(q - vec2(a, b)); return alice_r; }
    ALICE_PRECISE float alice_r = max(length(q) - ra, -(length(q - vec2(d, 0.0)) - rb));
    return alice_r;
}
";

const HELPER_SDF_SOLID_ANGLE: &str = r"float sdf_solid_angle(vec3 p, float an, float ra) {
    ALICE_PRECISE vec2 c = vec2(sin(an), cos(an));
    ALICE_PRECISE vec2 q = vec2(length(p.xz), p.y);
    ALICE_PRECISE float l = length(q) - ra;
    ALICE_PRECISE float m = length(q - c * clamp(dot(q, c), 0.0, ra));
    ALICE_PRECISE float alice_r = max(l, m * sign(c.y * q.x - c.x * q.y));
    return alice_r;
}
";

const HELPER_SDF_RHOMBUS: &str = r"float ndot_rh(vec2 a, vec2 b) {
    ALICE_PRECISE float alice_r = a.x * b.x - a.y * b.y;
    return alice_r;
}
float sdf_rhombus(vec3 p, float la, float lb, float h, float ra) {
    ALICE_PRECISE vec3 ap = abs(p);
    ALICE_PRECISE vec2 b = vec2(la, lb);
    ALICE_PRECISE float f = clamp(ndot_rh(b, b - 2.0 * ap.xz) / dot(b, b), -1.0, 1.0);
    ALICE_PRECISE float dxz = length(ap.xz - 0.5 * b * vec2(1.0 - f, 1.0 + f)) * sign(ap.x * b.y + ap.z * b.x - b.x * b.y) - ra;
    ALICE_PRECISE vec2 q = vec2(dxz, ap.y - h);
    ALICE_PRECISE float alice_r = min(max(q.x, q.y), 0.0) + length(max(q, vec2(0.0)));
    return alice_r;
}
";

const HELPER_SDF_HORSESHOE: &str = r"float sdf_horseshoe(vec3 pos, float an, float r, float le, float w, float t) {
    ALICE_PRECISE vec2 c = vec2(cos(an), sin(an));
    ALICE_PRECISE float px = abs(pos.x);
    ALICE_PRECISE float l = length(vec2(px, pos.y));
    ALICE_PRECISE float qx = -c.x * px + c.y * pos.y;
    ALICE_PRECISE float qy = c.y * px + c.x * pos.y;
    if (!(qy > 0.0 || qx > 0.0)) { qx = l * sign(-c.x); }
    if (!(qx > 0.0)) { qy = l; }
    ALICE_PRECISE float bx = qx - le;
    ALICE_PRECISE float by = abs(qy - r) - w;
    ALICE_PRECISE float d2 = length(max(vec2(bx, by), vec2(0.0, 0.0))) + min(max(bx, by), 0.0);
    ALICE_PRECISE float dz = abs(pos.z) - t;
    ALICE_PRECISE float alice_r = length(max(vec2(d2, dz), vec2(0.0, 0.0))) + min(max(d2, dz), 0.0);
    return alice_r;
}
";

const HELPER_SDF_VESICA: &str = r"float sdf_vesica(vec3 p, float r, float d) {
    // Lens of revolution about Y: 2-D vesica in (length(p.xz), |y|) — same law as `sdf_vesica` on the CPU.
    ALICE_PRECISE vec2 q = vec2(length(p.xz), abs(p.y));
    ALICE_PRECISE float b = sqrt(max(r * r - d * d, 0.0));
    if ((q.y - b) * d > q.x * b) { ALICE_PRECISE float alice_r = length(vec2(q.x, q.y - b)); return alice_r; }
    ALICE_PRECISE float alice_r = length(vec2(q.x + d, q.y)) - r;
    return alice_r;
}
";

const HELPER_SDF_INFINITE_CONE: &str = r"float sdf_infinite_cone(vec3 p, float an) {
    ALICE_PRECISE vec2 c = vec2(sin(an), cos(an));
    ALICE_PRECISE vec2 q = vec2(length(p.xz), -p.y);
    ALICE_PRECISE float d = length(q - c * max(dot(q, c), 0.0));
    ALICE_PRECISE float alice_r = d * ((q.x * c.y - q.y * c.x < 0.0) ? -1.0 : 1.0);
    return alice_r;
}
";

const HELPER_SDF_HEART: &str = r"float sdf_heart(vec3 p, float s) {
    // IQ sdHeart (2-D) revolved about Y, shifted so the centre is at the origin — same law as `sdf_heart` on the CPU.
    ALICE_PRECISE vec3 sp = p / s;
    ALICE_PRECISE float qx = length(sp.xz);
    ALICE_PRECISE float qy = sp.y - 0.5;
    if (qx + qy > 1.0) {
        ALICE_PRECISE vec2 e = vec2(qx - 0.25, qy - 0.75);
        ALICE_PRECISE float alice_r = (sqrt(dot(e, e)) - 0.35355339) * s;
        return alice_r;
    }
    ALICE_PRECISE vec2 e1 = vec2(qx, qy - 1.0);
    ALICE_PRECISE float t = max(qx + qy, 0.0) * 0.5;
    ALICE_PRECISE vec2 e2 = vec2(qx - t, qy - t);
    ALICE_PRECISE float d = sqrt(min(dot(e1, e1), dot(e2, e2)));
    ALICE_PRECISE float alice_r = (qx > qy ? d : -d) * s;
    return alice_r;
}
";

const HELPER_SDF_TUBE: &str = r"float sdf_tube(vec3 p, float outer_r, float thick, float h) {
    ALICE_PRECISE float r = length(p.xz);
    ALICE_PRECISE float dr = abs(r - outer_r) - thick;
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 w = max(vec2(dr, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(dr, dy), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_SDF_BARREL: &str = r"float sdf_barrel(vec3 p, float radius, float h, float bulge) {
    ALICE_PRECISE float r = length(p.xz);
    ALICE_PRECISE float yn = clamp(p.y / h, -1.0, 1.0);
    ALICE_PRECISE float er = radius + bulge * (1.0 - yn * yn);
    ALICE_PRECISE float dr = r - er;
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 w = max(vec2(dr, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(dr, dy), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_SDF_DIAMOND: &str = r"float sdf_diamond(vec3 p, float r, float h) {
    ALICE_PRECISE vec2 q = vec2(length(p.xz), abs(p.y));
    ALICE_PRECISE vec2 ba = vec2(-r, h);
    ALICE_PRECISE vec2 qa = q - vec2(r, 0.0);
    ALICE_PRECISE float t = clamp(dot(qa, ba) / dot(ba, ba), 0.0, 1.0);
    ALICE_PRECISE vec2 closest = vec2(r, 0.0) + ba * t;
    ALICE_PRECISE float dist = length(q - closest);
    ALICE_PRECISE float alice_lhs = q.x * h + q.y * r;
    ALICE_PRECISE float alice_rhs = r * h;
    if (alice_lhs < alice_rhs) return -dist;
    return dist;
}
";

const HELPER_SDF_CHAMFERED_CUBE: &str = r"float sdf_chamfered_cube(vec3 p, float hx, float hy, float hz, float ch) {
    ALICE_PRECISE vec3 ap = abs(p);
    ALICE_PRECISE vec3 q = ap - vec3(hx, hy, hz);
    ALICE_PRECISE float d_box = min(max(q.x, max(q.y, q.z)), 0.0) + length(max(q, vec3(0.0)));
    ALICE_PRECISE float s = hx + hy + hz;
    ALICE_PRECISE float d_ch = (ap.x + ap.y + ap.z - s + ch) * 0.57735;
    ALICE_PRECISE float alice_r = max(d_box, d_ch);
    return alice_r;
}
";

const HELPER_SDF_SUPERELLIPSOID: &str = r"float sdf_superellipsoid(vec3 p, float hx, float hy, float hz, float e1, float e2) {
    ALICE_PRECISE float qx = max(abs(p.x / hx), 0.00001);
    ALICE_PRECISE float qy = max(abs(p.y / hy), 0.00001);
    ALICE_PRECISE float qz = max(abs(p.z / hz), 0.00001);
    ALICE_PRECISE float ee1 = max(e1, 0.02);
    ALICE_PRECISE float ee2 = max(e2, 0.02);
    ALICE_PRECISE float m1 = 2.0 / ee2;
    ALICE_PRECISE float m2 = 2.0 / ee1;
    ALICE_PRECISE float w = pow(qx, m1) + pow(qz, m1);
    ALICE_PRECISE float v = pow(w, ee2 / ee1) + pow(qy, m2);
    ALICE_PRECISE float f = pow(v, ee1 * 0.5);
    ALICE_PRECISE float alice_r = (f - 1.0) * min(hx, min(hy, hz)) * 0.5;
    return alice_r;
}
";

const HELPER_SDF_ROUNDED_X: &str = r"float sdf_rounded_x(vec3 p, float w, float r, float h) {
    ALICE_PRECISE vec2 q = abs(p.xz);
    ALICE_PRECISE float s = min(q.x + q.y, w) * 0.5;
    ALICE_PRECISE float d2d = length(q - vec2(s)) - r;
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_PIE: &str = r"float sdf_pie(vec3 p, float angle, float radius, float h) {
    // IQ sdPie in the XZ plane (|x|, z), extruded along Y — same law as `sdf_pie` on the CPU.
    ALICE_PRECISE vec2 q = vec2(abs(p.x), p.z);
    ALICE_PRECISE vec2 sc = vec2(sin(angle), cos(angle));
    ALICE_PRECISE float l = length(q) - radius;
    ALICE_PRECISE float m = length(q - sc * clamp(dot(q, sc), 0.0, radius));
    ALICE_PRECISE float cr = sc.y * q.x - sc.x * q.y;
    ALICE_PRECISE float sgn = cr > 0.0 ? 1.0 : (cr < 0.0 ? -1.0 : 0.0);
    ALICE_PRECISE float d2d = max(l, m * sgn);
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 w = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_COLUMNS: &str = r"// Columns (hg_sdf fOpUnionColumns family) — same law as `sdf_columns_*_r` on the CPU: 45° rotate, shift by r/√2, wrap into r·2/n cells, rotate back.
float alice_columns_union(float a, float b, float r, float n) {
    ALICE_PRECISE float m = min(a, b);
    ALICE_PRECISE float a2 = min(a, b);
    ALICE_PRECISE float b2 = max(a, b);
    ALICE_PRECISE float cs = r * (2.0 / n);
    ALICE_PRECISE float s = 0.70710678;
    ALICE_PRECISE float ra = s * (a2 + b2) - r * 0.70710678;
    ALICE_PRECISE float rb = s * (b2 - a2);
    ALICE_PRECISE float hc = cs * 0.5;
    ALICE_PRECISE float y = ra + hc;
    ra = y - cs * floor(y / cs) - hc;
    ALICE_PRECISE float a3 = s * (ra + rb);
    ALICE_PRECISE float b3 = s * (rb - ra);
    ALICE_PRECISE float inner = min(min(a3, b3), m);
    ALICE_PRECISE float alice_r = m > r ? m : inner;
    return alice_r;
}
float alice_columns_subtraction(float a0, float b, float r, float n) {
    ALICE_PRECISE float a = -a0;
    ALICE_PRECISE float m = min(a, b);
    ALICE_PRECISE float a2 = min(a, b);
    ALICE_PRECISE float b2 = max(a, b);
    ALICE_PRECISE float cs = r * (2.0 / n);
    ALICE_PRECISE float s = 0.70710678;
    ALICE_PRECISE float ra = s * (a2 + b2) - r * 0.70710678;
    ALICE_PRECISE float rb = s * (b2 - a2);
    ALICE_PRECISE float hc = cs * 0.5;
    ALICE_PRECISE float y = ra + hc;
    ra = y - cs * floor(y / cs) - hc;
    ALICE_PRECISE float a3 = s * (ra + rb);
    ALICE_PRECISE float b3 = s * (rb - ra);
    ALICE_PRECISE float inner = -min(min(a3, b3), m);
    ALICE_PRECISE float alice_r = m > r ? -m : inner;
    return alice_r;
}
float alice_columns_intersection(float a, float b, float r, float n) {
    ALICE_PRECISE float alice_r = alice_columns_subtraction(a, -b, r, n);
    return alice_r;
}
";

const HELPER_ICOSAHEDRAL_FOLD: &str = r"// Icosahedral fold — same law as `icosahedral_fold` on the CPU (octant fold, 8 × sort + 3 plane reflections).
vec3 alice_icosahedral_fold(vec3 p0) {
    ALICE_PRECISE vec3 p = abs(p0);
    ALICE_PRECISE vec3 n0 = normalize(vec3(0.0, 1.0, 1.618034));
    ALICE_PRECISE vec3 n1 = normalize(vec3(1.0, 1.618034, 0.0));
    ALICE_PRECISE vec3 n2 = normalize(vec3(1.618034, 0.0, 1.0));
    for (int i = 0; i < 8; i++) {
        if (p.y > p.x) p = vec3(p.y, p.x, p.z);
        if (p.z > p.y) p = vec3(p.x, p.z, p.y);
        if (p.y > p.x) p = vec3(p.y, p.x, p.z);
        ALICE_PRECISE float d0 = dot(p, n0);
        if (d0 < 0.0) p -= n0 * (2.0 * d0);
        ALICE_PRECISE float d1 = dot(p, n1);
        if (d1 < 0.0) p -= n1 * (2.0 * d1);
        ALICE_PRECISE float d2 = dot(p, n2);
        if (d2 < 0.0) p -= n2 * (2.0 * d2);
    }
    return p;
}
";

const HELPER_OCTANT_MIRROR: &str = r"// Octant mirror — same law as `octant_mirror` on the CPU: abs then sort descending (x ≥ y ≥ z).
vec3 alice_octant_mirror(vec3 p0) {
    ALICE_PRECISE vec3 a = abs(p0);
    ALICE_PRECISE float x = max(a.x, a.y);
    ALICE_PRECISE float y = min(a.x, a.y);
    ALICE_PRECISE float y1 = max(y, a.z);
    ALICE_PRECISE float z = min(y, a.z);
    ALICE_PRECISE vec3 alice_r = vec3(max(x, y1), min(x, y1), z);
    return alice_r;
}
";

const HELPER_SDF_TRAPEZOID: &str = r"float _trap_ds(float px, float py, float ax, float ay, float bx, float by) {
    ALICE_PRECISE float dx = bx - ax;
    ALICE_PRECISE float dy = by - ay;
    ALICE_PRECISE float len_sq = dx * dx + dy * dy;
    ALICE_PRECISE float t = len_sq > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / len_sq : 0.0;
    t = clamp(t, 0.0, 1.0);
    ALICE_PRECISE float cx = ax + dx * t;
    ALICE_PRECISE float cy = ay + dy * t;
    ALICE_PRECISE float alice_r = sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
    return alice_r;
}
float sdf_trapezoid(vec3 p, float r1, float r2, float th, float hd) {
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float he = th;
    ALICE_PRECISE float d_bot = _trap_ds(px, py, 0.0, -he, r1, -he);
    ALICE_PRECISE float d_slant = _trap_ds(px, py, r1, -he, r2, he);
    ALICE_PRECISE float d_top = _trap_ds(px, py, r2, he, 0.0, he);
    ALICE_PRECISE float d_unsigned = min(d_bot, min(d_slant, d_top));
    ALICE_PRECISE float nx = 2.0 * he;
    ALICE_PRECISE float ny = r1 - r2;
    ALICE_PRECISE float d_slant_plane = (px - r1) * nx + (py + he) * ny;
    ALICE_PRECISE float d2d = (py >= -he && py <= he && d_slant_plane <= 0.0) ? -d_unsigned : d_unsigned;
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dz), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_PARALLELOGRAM: &str = r"float _para_ds(float px, float py, float ax, float ay, float bx, float by) {
    ALICE_PRECISE float dx = bx - ax;
    ALICE_PRECISE float dy = by - ay;
    ALICE_PRECISE float len_sq = dx * dx + dy * dy;
    ALICE_PRECISE float t = len_sq > 0.0 ? ((px - ax) * dx + (py - ay) * dy) / len_sq : 0.0;
    t = clamp(t, 0.0, 1.0);
    ALICE_PRECISE float cx = ax + dx * t;
    ALICE_PRECISE float cy = ay + dy * t;
    ALICE_PRECISE float alice_r = sqrt((px - cx) * (px - cx) + (py - cy) * (py - cy));
    return alice_r;
}
float _para_c2d(float px, float py, float ax, float ay, float bx, float by) {
    ALICE_PRECISE float alice_r = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    return alice_r;
}
float sdf_parallelogram(vec3 p, float w, float ph, float sk, float hd) {
    ALICE_PRECISE float px = p.x;
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float vdx = w - sk;  ALICE_PRECISE float vdy = -ph;
    ALICE_PRECISE float vax = w + sk;  ALICE_PRECISE float vay = ph;
    ALICE_PRECISE float vbx = -w + sk; ALICE_PRECISE float vby = ph;
    ALICE_PRECISE float vcx = -w - sk; ALICE_PRECISE float vcy = -ph;
    ALICE_PRECISE float d1 = _para_ds(px, py, vdx, vdy, vax, vay);
    ALICE_PRECISE float d2 = _para_ds(px, py, vax, vay, vbx, vby);
    ALICE_PRECISE float d3 = _para_ds(px, py, vbx, vby, vcx, vcy);
    ALICE_PRECISE float d4 = _para_ds(px, py, vcx, vcy, vdx, vdy);
    ALICE_PRECISE float d_unsigned = min(d1, min(d2, min(d3, d4)));
    ALICE_PRECISE float c1 = _para_c2d(px, py, vdx, vdy, vax, vay);
    ALICE_PRECISE float c2 = _para_c2d(px, py, vax, vay, vbx, vby);
    ALICE_PRECISE float c3 = _para_c2d(px, py, vbx, vby, vcx, vcy);
    ALICE_PRECISE float c4 = _para_c2d(px, py, vcx, vcy, vdx, vdy);
    ALICE_PRECISE float d2d = (c1 <= 0.0 && c2 <= 0.0 && c3 <= 0.0 && c4 <= 0.0) ? -d_unsigned : d_unsigned;
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dz), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_TUNNEL: &str = r"float sdf_tunnel(vec3 p, float w, float h2d, float hd) {
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float dx = px - w;
    ALICE_PRECISE float dy_rect = abs(py) - h2d;
    ALICE_PRECISE float d_rect = length(max(vec2(dx, dy_rect), vec2(0.0))) + min(max(dx, dy_rect), 0.0);
    ALICE_PRECISE float d_circle = length(vec2(px, py - h2d)) - w;
    ALICE_PRECISE float d2d = py > h2d ? min(d_rect, d_circle) : d_rect;
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dz), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_UNEVEN_CAPSULE: &str = r"float sdf_uneven_capsule(vec3 p, float r1, float r2, float ch, float hd) {
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float hh = ch * 2.0;
    ALICE_PRECISE float b = (r1 - r2) / hh;
    ALICE_PRECISE float a = sqrt(max(1.0 - b * b, 0.0));
    ALICE_PRECISE float k = dot(vec2(-b, a), vec2(px, p.y));
    ALICE_PRECISE float d2d;
    if (k < 0.0) {
        d2d = length(vec2(px, p.y)) - r1;
    } else if (k > a * hh) {
        d2d = length(vec2(px, p.y - hh)) - r2;
    } else {
        d2d = dot(vec2(px, p.y), vec2(a, b)) - r1;
    }
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dz), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_EGG: &str = r"float sdf_egg(vec3 p, float ra, float rb) {
    const float k = 1.7320508;
    ALICE_PRECISE float px = length(p.xz);
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float r = ra - rb;
    ALICE_PRECISE float d;
    ALICE_PRECISE float alice_kpr = k * (px + r);
    if (py < 0.0) {
        d = length(vec2(px, py)) - r;
    } else if (alice_kpr < py) {
        d = length(vec2(px, py - k * r)) - r;
    } else {
        d = length(vec2(px + r, py)) - 2.0 * r;
    }
    ALICE_PRECISE float alice_r = d - rb;
    return alice_r;
}
";

const HELPER_SDF_ARC_SHAPE: &str = r"float sdf_arc_shape(vec3 p, float aperture, float radius, float thickness, float h) {
    ALICE_PRECISE float qx = abs(p.x);
    ALICE_PRECISE float qz = p.z;
    ALICE_PRECISE vec2 sc = vec2(sin(aperture), cos(aperture));
    ALICE_PRECISE float d2d;
    if (sc.y * qx > sc.x * qz) {
        d2d = length(vec2(qx, qz) - sc * radius) - thickness;
    } else {
        d2d = abs(length(vec2(qx, qz)) - radius) - thickness;
    }
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_MOON: &str = r"float sdf_moon(vec3 p, float d, float ra, float rb, float h) {
    ALICE_PRECISE float qx = abs(p.x);
    ALICE_PRECISE float qz = p.z;
    ALICE_PRECISE float d_outer = length(vec2(qx, qz)) - ra;
    ALICE_PRECISE float d_inner = length(vec2(qx - d, qz)) - rb;
    ALICE_PRECISE float d2d = max(d_outer, -d_inner);
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_CROSS_SHAPE: &str = r"float sdf_cross_shape(vec3 p, float len, float th, float rr, float h) {
    ALICE_PRECISE float qx = abs(p.x);
    ALICE_PRECISE float qz = abs(p.z);
    ALICE_PRECISE vec2 dh = vec2(qx - len, qz - th);
    ALICE_PRECISE vec2 dv = vec2(qx - th, qz - len);
    ALICE_PRECISE float dh_sdf = length(max(dh, vec2(0.0))) + min(max(dh.x, dh.y), 0.0);
    ALICE_PRECISE float dv_sdf = length(max(dv, vec2(0.0))) + min(max(dv.x, dv.y), 0.0);
    ALICE_PRECISE float d2d = min(dh_sdf, dv_sdf) - rr;
    ALICE_PRECISE float dy = abs(p.y) - h;
    ALICE_PRECISE vec2 ww = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(ww);
    return alice_r;
}
";

const HELPER_SDF_BLOBBY_CROSS: &str = r"float sdf_blobby_cross(vec3 p, float size, float hh) {
    const float he = 0.5;
    ALICE_PRECISE vec2 pos = abs(vec2(p.x, p.z) / size);
    pos = vec2(abs(pos.x - pos.y), 1.0 - pos.x - pos.y) * 0.70710678;
    ALICE_PRECISE float pp = (he - pos.y - 0.25 / he) / (6.0 * he);
    ALICE_PRECISE float q = pos.x / (he * he * 16.0);
    ALICE_PRECISE float h = q * q - pp * pp * pp;
    ALICE_PRECISE float x;
    if (h >= 0.0) {
        ALICE_PRECISE float r = sqrt(h);
        x = pow(q + r, 1.0 / 3.0) - pow(abs(q - r), 1.0 / 3.0) * sign(r - q);
    } else {
        ALICE_PRECISE float r = sqrt(pp);
        x = 2.0 * r * cos(acos(q / (pp * r)) / 3.0);
    }
    x = min(x, 0.70710678);
    ALICE_PRECISE vec2 z = vec2(x, he * (1.0 - 2.0 * x * x)) - pos;
    ALICE_PRECISE float d_2d = length(z) * sign(z.y) * size;
    ALICE_PRECISE float d_y = abs(p.y) - hh;
    ALICE_PRECISE vec2 w = vec2(max(d_2d, 0.0), max(d_y, 0.0));
    ALICE_PRECISE float alice_r = min(max(d_2d, d_y), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_SDF_PARABOLA_SEGMENT: &str = r"float sdf_parabola_segment(vec3 p, float w, float ph, float hd) {
    ALICE_PRECISE float px = abs(p.x);
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float ww_sq = w * w;
    ALICE_PRECISE float y_arch = ph * (1.0 - px * px / ww_sq);
    bool is_in = (px <= w && py >= 0.0 && py <= y_arch);
    ALICE_PRECISE float t = clamp(px, 0.0, w);
    for (int i = 0; i < 8; i++) {
        ALICE_PRECISE float ft = ph * (1.0 - t * t / ww_sq);
        ALICE_PRECISE float dft = -2.0 * ph * t / ww_sq;
        ALICE_PRECISE float ex = px - t;
        ALICE_PRECISE float ey = py - ft;
        ALICE_PRECISE float f = -ex + ey * dft;
        ALICE_PRECISE float df = 1.0 + dft * dft + ey * (-2.0 * ph / ww_sq);
        if (abs(df) > 1e-10) { t = clamp(t - f / df, 0.0, w); }
    }
    ALICE_PRECISE float cy = ph * (1.0 - t * t / ww_sq);
    ALICE_PRECISE float d_para = length(vec2(px - t, py - cy));
    ALICE_PRECISE float d_base = (px <= w) ? abs(py) : length(vec2(px - w, py));
    ALICE_PRECISE float d_unsigned = min(d_para, d_base);
    ALICE_PRECISE float d2d = is_in ? -d_unsigned : d_unsigned;
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 ext = max(vec2(d2d, dz), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(ext);
    return alice_r;
}
";

const HELPER_SDF_REGULAR_POLYGON: &str = r"float sdf_regular_polygon(vec3 p, float radius, float n, float hh) {
    // IQ regular polygon (exact), XZ plane, radius = circumradius (1.9.2)
    ALICE_PRECISE float nn = trunc(max(n, 3.0));
    ALICE_PRECISE float an = 3.14159265358979 / nn;
    ALICE_PRECISE vec2 acs = vec2(cos(an), sin(an));
    ALICE_PRECISE float bn = mod(alice_atan2(p.x, p.z), 2.0 * an) - an;
    ALICE_PRECISE float r = length(p.xz);
    ALICE_PRECISE vec2 q = vec2(r * cos(bn), abs(r * sin(bn)));
    q -= radius * acs;
    q.y += clamp(-q.y, 0.0, radius * acs.y);
    ALICE_PRECISE float d2d = length(q) * sign(q.x);
    ALICE_PRECISE float dy = abs(p.y) - hh;
    ALICE_PRECISE vec2 w = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_SDF_STAR_POLYGON: &str = r"float sdf_star_polygon(vec3 p, float radius, float np, float m, float hh) {
    ALICE_PRECISE float qx = abs(p.x);
    ALICE_PRECISE float qz = p.z;
    ALICE_PRECISE float n = max(np, 3.0);
    ALICE_PRECISE float an = 3.14159265358979 / n;
    ALICE_PRECISE float r = length(vec2(qx, qz));
    ALICE_PRECISE float angle = alice_atan2(qx, qz);
    angle = mod(mod(angle, 2.0 * an) + 2.0 * an, 2.0 * an);
    if (angle > an) angle = 2.0 * an - angle;
    ALICE_PRECISE vec2 pt = vec2(r * cos(angle), r * sin(angle));
    ALICE_PRECISE vec2 a = vec2(radius, 0.0);
    ALICE_PRECISE vec2 b = vec2(m * cos(an), m * sin(an));
    ALICE_PRECISE vec2 ab = b - a;
    ALICE_PRECISE vec2 ap = pt - a;
    ALICE_PRECISE float t = clamp(dot(ap, ab) / dot(ab, ab), 0.0, 1.0);
    ALICE_PRECISE vec2 closest = a + ab * t;
    ALICE_PRECISE float dist = length(pt - closest);
    ALICE_PRECISE float cross_val = ab.x * ap.y - ab.y * ap.x;
    ALICE_PRECISE float d2d = (cross_val > 0.0) ? -dist : dist;
    ALICE_PRECISE float dy = abs(p.y) - hh;
    ALICE_PRECISE vec2 w = max(vec2(d2d, dy), vec2(0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dy), 0.0) + length(w);
    return alice_r;
}
";

const HELPER_SDF_STAIRS: &str = r"float _stair_box(float lx, float ly, float s, float sw, float sh) {
    ALICE_PRECISE float cx = s * sw + sw * 0.5;
    ALICE_PRECISE float hy = (s + 1.0) * sh * 0.5;
    ALICE_PRECISE float dx = abs(lx - cx) - sw * 0.5;
    ALICE_PRECISE float dy = abs(ly - hy) - hy;
    ALICE_PRECISE float alice_r = length(max(vec2(dx, dy), vec2(0.0, 0.0))) + min(max(dx, dy), 0.0);
    return alice_r;
}
float sdf_stairs(vec3 p, float sw, float sh, float ns, float hd) {
    ALICE_PRECISE float n = max(ns, 1.0);
    ALICE_PRECISE float tw = n * sw;
    ALICE_PRECISE float th = n * sh;
    ALICE_PRECISE float lx = p.x + tw * 0.5;
    ALICE_PRECISE float ly = p.y + th * 0.5;
    ALICE_PRECISE float d2d = 1e30;
    for (ALICE_PRECISE float si = 0.0; si < n; si += 1.0) {
        d2d = min(d2d, _stair_box(lx, ly, si, sw, sh));
    }
    ALICE_PRECISE float dz = abs(p.z) - hd;
    ALICE_PRECISE vec2 w = max(vec2(d2d, dz), vec2(0.0, 0.0));
    ALICE_PRECISE float alice_r = min(max(d2d, dz), 0.0) + length(w);
    return alice_r;
}
";

/// Regular tetrahedron: max of 4 face-plane distances (mirrors `primitives::tetrahedron`).
const HELPER_SDF_TETRAHEDRON: &str = r"float sdf_tetrahedron(vec3 p, float s) {
    ALICE_PRECISE float d = dot(p, vec3(0.57735026, 0.57735026, 0.57735026));
    d = max(d, dot(p, vec3(-0.57735026, -0.57735026, 0.57735026)));
    d = max(d, dot(p, vec3(-0.57735026, 0.57735026, -0.57735026)));
    d = max(d, dot(p, vec3(0.57735026, -0.57735026, -0.57735026)));
    ALICE_PRECISE float alice_r = d - s;
    return alice_r;
}
";

/// Dodecahedron: GDF over the 6 dodecahedron normals (mirrors `gdf_eval`).
const HELPER_SDF_DODECAHEDRON: &str = r"float sdf_dodecahedron(vec3 p, float r) {
    ALICE_PRECISE float d = abs(dot(p, vec3(0.0, 0.8506508, 0.5257311)));
    d = max(d, abs(dot(p, vec3(0.0, 0.8506508, -0.5257311))));
    d = max(d, abs(dot(p, vec3(0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, vec3(-0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.0, 0.8506508))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.0, -0.8506508))));
    ALICE_PRECISE float alice_r = d - r;
    return alice_r;
}
";

/// Icosahedron: GDF over octahedron + icosahedron normals (mirrors `primitives::icosahedron`).
const HELPER_SDF_ICOSAHEDRON: &str = r"float sdf_icosahedron(vec3 p, float r) {
    ALICE_PRECISE float d = abs(dot(p, vec3(0.57735026, 0.57735026, 0.57735026)));
    d = max(d, abs(dot(p, vec3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, 0.57735026, -0.57735026))));
    d = max(d, abs(dot(p, vec3(0.0, 0.5257311, 0.8506508))));
    d = max(d, abs(dot(p, vec3(0.0, 0.5257311, -0.8506508))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.8506508, 0.0))));
    d = max(d, abs(dot(p, vec3(0.5257311, -0.8506508, 0.0))));
    d = max(d, abs(dot(p, vec3(0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, vec3(-0.8506508, 0.0, 0.5257311))));
    ALICE_PRECISE float alice_r = d - r;
    return alice_r;
}
";

/// Truncated octahedron: GDF over cube + octahedron normals.
const HELPER_SDF_TRUNCATED_OCTAHEDRON: &str = r"float sdf_truncated_octahedron(vec3 p, float r) {
    ALICE_PRECISE float d = abs(dot(p, vec3(1.0, 0.0, 0.0)));
    d = max(d, abs(dot(p, vec3(0.0, 1.0, 0.0))));
    d = max(d, abs(dot(p, vec3(0.0, 0.0, 1.0))));
    d = max(d, abs(dot(p, vec3(0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, 0.57735026, -0.57735026))));
    ALICE_PRECISE float alice_r = d - r;
    return alice_r;
}
";

/// Truncated icosahedron: GDF over octahedron + icosahedron + dodecahedron normals.
const HELPER_SDF_TRUNCATED_ICOSAHEDRON: &str = r"float sdf_truncated_icosahedron(vec3 p, float r) {
    ALICE_PRECISE float d = abs(dot(p, vec3(0.57735026, 0.57735026, 0.57735026)));
    d = max(d, abs(dot(p, vec3(-0.57735026, 0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, -0.57735026, 0.57735026))));
    d = max(d, abs(dot(p, vec3(0.57735026, 0.57735026, -0.57735026))));
    d = max(d, abs(dot(p, vec3(0.0, 0.5257311, 0.8506508))));
    d = max(d, abs(dot(p, vec3(0.0, 0.5257311, -0.8506508))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.8506508, 0.0))));
    d = max(d, abs(dot(p, vec3(0.5257311, -0.8506508, 0.0))));
    d = max(d, abs(dot(p, vec3(0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, vec3(-0.8506508, 0.0, 0.5257311))));
    d = max(d, abs(dot(p, vec3(0.0, 0.8506508, 0.5257311))));
    d = max(d, abs(dot(p, vec3(0.0, 0.8506508, -0.5257311))));
    d = max(d, abs(dot(p, vec3(0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, vec3(-0.8506508, 0.5257311, 0.0))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.0, 0.8506508))));
    d = max(d, abs(dot(p, vec3(0.5257311, 0.0, -0.8506508))));
    ALICE_PRECISE float alice_r = d - r;
    return alice_r;
}
";

/// IQ `sdBezier`: closed-form distance to a quadratic Bézier (mirrors
/// `modifiers::sweep::bezier_distance_2d`, degenerate curve → segment).
const HELPER_BEZIER_DISTANCE_2D: &str = r"float bezier_distance_2d(vec2 q, vec2 p0, vec2 p1, vec2 p2) {
    ALICE_PRECISE vec2 a = p1 - p0;
    ALICE_PRECISE vec2 b = p0 - p1 * 2.0 + p2;
    ALICE_PRECISE vec2 c = a * 2.0;
    ALICE_PRECISE vec2 d = p0 - q;
    ALICE_PRECISE float bb = dot(b, b);
    if (bb < 1e-12) {
        ALICE_PRECISE vec2 ac = p2 - p0;
        ALICE_PRECISE float t = clamp(-dot(d, ac) / max(dot(ac, ac), 1e-12), 0.0, 1.0);
        ALICE_PRECISE float alice_r = length(d + ac * t);
        return alice_r;
    }
    ALICE_PRECISE float kk = 1.0 / bb;
    ALICE_PRECISE float kx = kk * dot(a, b);
    ALICE_PRECISE float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
    ALICE_PRECISE float kz = kk * dot(d, a);
    ALICE_PRECISE float p = ky - kx * kx;
    ALICE_PRECISE float p3 = p * p * p;
    ALICE_PRECISE float qq = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    ALICE_PRECISE float h = qq * qq + 4.0 * p3;
    ALICE_PRECISE float res;
    if (h >= 0.0) {
        ALICE_PRECISE float hs = sqrt(h);
        ALICE_PRECISE float x0 = (hs - qq) * 0.5;
        ALICE_PRECISE float x1 = (-hs - qq) * 0.5;
        ALICE_PRECISE float u = sign(x0) * pow(abs(x0), 1.0 / 3.0);
        ALICE_PRECISE float v = sign(x1) * pow(abs(x1), 1.0 / 3.0);
        ALICE_PRECISE float t = clamp(u + v - kx, 0.0, 1.0);
        ALICE_PRECISE vec2 e = d + (c + b * t) * t;
        res = dot(e, e);
    } else {
        ALICE_PRECISE float z = sqrt(-p);
        ALICE_PRECISE float v = acos(clamp(qq / (p * z * 2.0), -1.0, 1.0)) / 3.0;
        ALICE_PRECISE float m = cos(v);
        ALICE_PRECISE float n = sin(v) * 1.7320508;
        ALICE_PRECISE float t0 = clamp((m + m) * z - kx, 0.0, 1.0);
        ALICE_PRECISE float t1 = clamp((-n - m) * z - kx, 0.0, 1.0);
        ALICE_PRECISE vec2 e0 = d + (c + b * t0) * t0;
        ALICE_PRECISE vec2 e1 = d + (c + b * t1) * t1;
        res = min(dot(e0, e0), dot(e1, e1));
    }
    ALICE_PRECISE float alice_r = sqrt(res);
    return alice_r;
}
";

/// Exact ellipsoid signed distance (Eberly nearest point; mirrors
/// `primitives::ellipsoid::sdf_ellipsoid_exact`, dimension reductions unrolled).
const HELPER_SDF_ELLIPSOID: &str = r"float _ell_bisect(int n, float r0, float r1, float r2, float z0, float z1, float z2, float g) {
    ALICE_PRECISE float n0 = r0 * z0;
    ALICE_PRECISE float n1 = (n > 1) ? r1 * z1 : 0.0;
    ALICE_PRECISE float n2 = (n > 2) ? r2 * z2 : 0.0;
    ALICE_PRECISE float zl = (n == 3) ? z2 : ((n == 2) ? z1 : z0);
    ALICE_PRECISE float s0 = zl - 1.0;
    ALICE_PRECISE float s1 = (g < 0.0) ? 0.0 : sqrt(n0 * n0 + n1 * n1 + n2 * n2) - 1.0;
    ALICE_PRECISE float s = 0.0;
    for (int it = 0; it < 64; it++) {
        s = 0.5 * (s0 + s1);
        if (s == s0 || s == s1) { break; }
        ALICE_PRECISE float gs = -1.0;
        ALICE_PRECISE float q0 = n0 / (s + r0);
        gs += q0 * q0;
        if (n > 1) { ALICE_PRECISE float q1 = n1 / (s + r1); gs += q1 * q1; }
        if (n > 2) { ALICE_PRECISE float q2 = n2 / (s + r2); gs += q2 * q2; }
        if (gs > 0.0) { s0 = s; } else if (gs < 0.0) { s1 = s; } else { break; }
    }
    return s;
}
float _ell_sqr_dist_n(int n, float e0, float e1, float e2, float y0, float y1, float y2) {
    ALICE_PRECISE float el = (n == 3) ? e2 : ((n == 2) ? e1 : e0);
    ALICE_PRECISE float z0 = y0 / e0;
    ALICE_PRECISE float z1 = (n > 1) ? y1 / e1 : 0.0;
    ALICE_PRECISE float z2 = (n > 2) ? y2 / e2 : 0.0;
    ALICE_PRECISE float g = z0 * z0 + z1 * z1 + z2 * z2 - 1.0;
    if (g == 0.0) { return 0.0; }
    ALICE_PRECISE float r0 = (e0 / el) * (e0 / el);
    ALICE_PRECISE float r1 = (e1 / el) * (e1 / el);
    ALICE_PRECISE float r2 = (e2 / el) * (e2 / el);
    ALICE_PRECISE float sb = _ell_bisect(n, r0, r1, r2, z0, z1, z2, g);
    ALICE_PRECISE float x0 = r0 * y0 / (sb + r0) - y0;
    ALICE_PRECISE float x1 = (n > 1) ? r1 * y1 / (sb + r1) - y1 : 0.0;
    ALICE_PRECISE float x2 = (n > 2) ? r2 * y2 / (sb + r2) - y2 : 0.0;
    ALICE_PRECISE float alice_r = x0 * x0 + x1 * x1 + x2 * x2;
    return alice_r;
}
float sdf_ellipsoid(vec3 p, vec3 radii) {
    ALICE_PRECISE vec3 e = max(radii, vec3(1e-10, 1e-10, 1e-10));
    ALICE_PRECISE vec3 y = abs(p);
    if (e.x < e.y) { ALICE_PRECISE float t = e.x; e.x = e.y; e.y = t; ALICE_PRECISE float u = y.x; y.x = y.y; y.y = u; }
    if (e.y < e.z) { ALICE_PRECISE float t = e.y; e.y = e.z; e.z = t; ALICE_PRECISE float u = y.y; y.y = y.z; y.z = u; }
    if (e.x < e.y) { ALICE_PRECISE float t = e.x; e.x = e.y; e.y = t; ALICE_PRECISE float u = y.x; y.x = y.y; y.y = u; }
    ALICE_PRECISE float d2 = 0.0;
    if (y.z > 0.0) {
        d2 = _ell_sqr_dist_n(3, e.x, e.y, e.z, y.x, y.y, y.z);
    } else {
        ALICE_PRECISE float n0 = e.x * y.x; ALICE_PRECISE float n1 = e.y * y.y;
        ALICE_PRECISE float d0 = e.x * e.x - e.z * e.z; ALICE_PRECISE float d1 = e.y * e.y - e.z * e.z;
        bool done = false;
        if (n0 < d0 && n1 < d1) {
            ALICE_PRECISE float x0 = n0 / d0; ALICE_PRECISE float x1 = n1 / d1;
            ALICE_PRECISE float discr = 1.0 - x0 * x0 - x1 * x1;
            if (discr > 0.0) {
                ALICE_PRECISE float a = e.x * x0 - y.x; ALICE_PRECISE float b = e.y * x1 - y.y; ALICE_PRECISE float c = e.z * sqrt(discr);
                d2 = a * a + b * b + c * c;
                done = true;
            }
        }
        if (!done) {
            if (y.y > 0.0) {
                d2 = _ell_sqr_dist_n(2, e.x, e.y, e.z, y.x, y.y, 0.0);
            } else {
                ALICE_PRECISE float m0 = e.x * y.x;
                ALICE_PRECISE float q0 = e.x * e.x - e.y * e.y;
                bool done2 = false;
                if (m0 < q0) {
                    ALICE_PRECISE float x0 = m0 / q0;
                    ALICE_PRECISE float discr = 1.0 - x0 * x0;
                    if (discr > 0.0) {
                        ALICE_PRECISE float a = e.x * x0 - y.x; ALICE_PRECISE float b = e.y * sqrt(discr);
                        d2 = a * a + b * b;
                        done2 = true;
                    }
                }
                if (!done2) { ALICE_PRECISE float a = y.x - e.x; d2 = a * a; }
            }
        }
    }
    bool inside = (y.x / e.x) * (y.x / e.x) + (y.y / e.y) * (y.y / e.y) + (y.z / e.z) * (y.z / e.z) < 1.0;
    ALICE_PRECISE float d = sqrt(max(d2, 0.0));
    ALICE_PRECISE float alice_r = inside ? -d : d;
    return alice_r;
}
";

const HELPER_SDF_HELIX: &str = r"float sdf_helix(vec3 p, float major_r, float minor_r, float pitch, float hh) {
    const float tau = 6.28318530717959;
    ALICE_PRECISE float r = length(vec2(p.x, p.z));
    ALICE_PRECISE float theta = (r > 0.0) ? alice_atan2(p.z, p.x) : 0.0;
    ALICE_PRECISE float py = p.y;
    ALICE_PRECISE float c = pitch / tau;
    ALICE_PRECISE float two_rr = 2.0 * r * major_r;
    ALICE_PRECISE float k = floor((py - theta * c) / pitch + 0.5);
    ALICE_PRECISE float best = 1e30;
    for (ALICE_PRECISE float dk = -1.0; dk <= 1.0; dk += 1.0) {
        ALICE_PRECISE float phi = theta + (k + dk) * tau;
        for (int it = 0; it < 6; it++) {
            ALICE_PRECISE float s = sin(phi - theta);
            ALICE_PRECISE float co = cos(phi - theta);
            ALICE_PRECISE float dy = py - c * phi;
            ALICE_PRECISE float f1 = two_rr * s - 2.0 * c * dy;
            ALICE_PRECISE float f2 = max(two_rr * co + 2.0 * c * c, 1e-6);
            phi = phi - clamp(f1 / f2, -1.5707963, 1.5707963);
        }
        ALICE_PRECISE float co = cos(phi - theta);
        ALICE_PRECISE float dy = py - c * phi;
        ALICE_PRECISE float d2 = dy * dy + max(r * r + major_r * major_r - two_rr * co, 0.0);
        best = min(best, d2);
    }
    ALICE_PRECISE float d_tube = sqrt(best) - minor_r;
    ALICE_PRECISE float d_cap = abs(py) - hh;
    ALICE_PRECISE float alice_r = max(d_tube, d_cap);
    return alice_r;
}
";

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
    #[allow(deprecated)] // pins the legacy output kept for existing callers
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
        let shape = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5), 0.3);
        let shader = GlslShader::transpile(&shape, GlslTranspileMode::Dynamic);

        // Should contain uniform block references
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

    #[test]
    #[allow(deprecated)] // pins the legacy output kept for existing callers
    fn test_unity_shader_graph_export() {
        let shape = SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5));
        let shader = GlslShader::transpile(&shape, GlslTranspileMode::Hardcoded);
        let sg = shader.export_unity_shader_graph();

        assert!(sg.contains("ALICE_SDF_INCLUDED"));
        assert!(sg.contains("AliceSdf_float"));
        assert!(sg.contains("AliceSdf_half"));
        assert!(sg.contains("sdf_eval"));
        assert!(sg.contains("out float Distance"));
        assert!(sg.contains("out float3 Normal"));
        assert!(!sg.contains("_SdfParams"));

        // Dynamic mode should include params
        let shader_dyn = GlslShader::transpile(&shape, GlslTranspileMode::Dynamic);
        let sg_dyn = shader_dyn.export_unity_shader_graph();
        assert!(sg_dyn.contains("_SdfParams"));
    }

    /// Exhaustive test: every SdfNode variant transpiles without panic
    /// and produces valid GLSL containing `sdf_eval`.
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
            let shader = GlslShader::transpile(prim, GlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(vec3 p)"),
                "GLSL transpile failed for primitive '{}': missing sdf_eval",
                name,
            );
        }

        // Operations: test each with sphere+box
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
            let shader = GlslShader::transpile(op, GlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(vec3 p)"),
                "GLSL transpile failed for operation '{}': missing sdf_eval",
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
            let shader = GlslShader::transpile(m, GlslTranspileMode::Hardcoded);
            assert!(
                shader.source.contains("float sdf_eval(vec3 p)"),
                "GLSL transpile failed for modifier '{}': missing sdf_eval",
                name,
            );
        }
    }
}

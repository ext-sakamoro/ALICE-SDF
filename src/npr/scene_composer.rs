//! Builder that composes a complete NPR raymarching shader source
//!
//! Bundles the NPR helper library, the transpiled SDF evaluator, and a
//! canonical raymarching `main()` per shader language into a single
//! shader source string. Callers configure sky, sun, shading, and outline
//! parameters via a builder API; `build()` returns a `String` that can be
//! uploaded directly to the corresponding graphics API.
//!
//! # Example
//! ```
//! use alice_sdf::prelude::*;
//! let node = SdfNode::sphere(1.0);
//! let shader = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl).build();
//! assert!(shader.contains("void main"));
//! assert!(shader.contains("sdf_eval"));
//! assert!(shader.contains("alice_soft_toon_ramp"));
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(feature = "glsl")]
use crate::compiled::glsl::{GlslShader, GlslTranspileMode};
#[cfg(feature = "hlsl")]
use crate::compiled::hlsl::{HlslShader, HlslTranspileMode};
#[cfg(feature = "gpu")]
use crate::compiled::wgsl::{TranspileMode as WgslTranspileMode, WgslShader};
use crate::npr::dsl::NprColorNode;
use crate::npr::dsl_shader::{transpile_npr_color_node, NprShaderContext};
use crate::npr::shader_glue::{full_helpers_for, ShaderLanguage};
use crate::types::SdfNode;
use glam::Vec3;

/// Builder for a fully-composed NPR raymarching shader source
#[derive(Debug, Clone)]
pub struct SceneShaderBuilder<'a> {
    node: &'a SdfNode,
    language: ShaderLanguage,
    sky_horizon: Vec3,
    sky_mid: Vec3,
    sky_zenith: Vec3,
    sun_direction: Vec3,
    sun_color: Vec3,
    shadow_color: Vec3,
    light_color: Vec3,
    outline_color: Vec3,
    toon_bands: u32,
    toon_smoothness: f32,
    outline_inner: f32,
    outline_outer: f32,
    vignette_radius: f32,
    vignette_softness: f32,
    max_steps: u32,
    max_dist: f32,
    surface_eps: f32,
    camera_position: Vec3,
    sun_disc_radius: f32,
    sun_disc_softness: f32,
    pipeline: Option<NprColorNode>,
}

impl<'a> SceneShaderBuilder<'a> {
    /// Create a builder with canonical defaults
    #[must_use]
    pub fn new(node: &'a SdfNode, language: ShaderLanguage) -> Self {
        Self {
            node,
            language,
            sky_horizon: Vec3::new(0.90, 0.60, 0.40),
            sky_mid: Vec3::new(0.65, 0.70, 0.85),
            sky_zenith: Vec3::new(0.20, 0.30, 0.60),
            sun_direction: Vec3::new(0.4, 0.7, -0.6).normalize(),
            sun_color: Vec3::new(1.0, 0.95, 0.8),
            shadow_color: Vec3::new(0.20, 0.18, 0.35),
            light_color: Vec3::new(0.95, 0.88, 0.75),
            outline_color: Vec3::new(0.02, 0.02, 0.05),
            toon_bands: 3,
            toon_smoothness: 0.05,
            outline_inner: 0.005,
            outline_outer: 0.03,
            vignette_radius: 0.42,
            vignette_softness: 0.35,
            max_steps: 64,
            max_dist: 20.0,
            surface_eps: 1e-3,
            camera_position: Vec3::new(0.0, 0.0, -3.0),
            sun_disc_radius: 0.03,
            sun_disc_softness: 0.05,
            pipeline: None,
        }
    }

    /// Replace the built-in soft-toon + outline colour block with the
    /// caller's [`NprColorNode`] pipeline
    ///
    /// The pipeline sees the shading context via [`NprShaderContext::canonical`]
    /// (`ndl` scalar and `d` scalar). When set, `with_shading` and
    /// `with_outline` parameters are ignored on the hit branch.
    #[must_use]
    pub fn with_pipeline(mut self, pipeline: NprColorNode) -> Self {
        self.pipeline = Some(pipeline);
        self
    }

    /// Override the 3-anchor sky gradient (horizon / mid / zenith)
    #[must_use]
    pub fn with_sky(mut self, horizon: Vec3, mid: Vec3, zenith: Vec3) -> Self {
        self.sky_horizon = horizon;
        self.sky_mid = mid;
        self.sky_zenith = zenith;
        self
    }

    /// Override the sun direction and colour
    #[must_use]
    pub fn with_sun(mut self, direction: Vec3, color: Vec3) -> Self {
        self.sun_direction = direction.normalize_or_zero();
        self.sun_color = color;
        self
    }

    /// Override the shadow and light colours plus the toon band count
    #[must_use]
    pub fn with_shading(mut self, shadow: Vec3, light: Vec3, bands: u32) -> Self {
        self.shadow_color = shadow;
        self.light_color = light;
        self.toon_bands = bands.max(1);
        self
    }

    /// Override the outline colour and width band
    #[must_use]
    pub fn with_outline(mut self, color: Vec3, inner: f32, outer: f32) -> Self {
        self.outline_color = color;
        self.outline_inner = inner.max(0.0);
        self.outline_outer = outer.max(inner + 1e-4);
        self
    }

    /// Override the raymarching budget
    #[must_use]
    pub fn with_raymarch(mut self, max_steps: u32, max_dist: f32, surface_eps: f32) -> Self {
        self.max_steps = max_steps.max(1);
        self.max_dist = max_dist.max(1.0);
        self.surface_eps = surface_eps.max(1e-6);
        self
    }

    /// Override the camera position (image plane sits at world origin)
    #[must_use]
    pub fn with_camera(mut self, position: Vec3) -> Self {
        self.camera_position = position;
        self
    }

    /// Compose and return the final shader source, or `None` if the target
    /// language's transpiler feature is not enabled at compile time
    #[must_use]
    pub fn try_build(&self) -> Option<String> {
        let helpers = full_helpers_for(self.language);
        let sdf_fn = self.transpile_sdf()?;
        let main_fn = self.main_source();
        let header = match self.language {
            ShaderLanguage::Glsl => "#version 460 core\n\n",
            ShaderLanguage::Wgsl => "",
            ShaderLanguage::Hlsl => "",
        };
        Some(format!("{header}{helpers}\n{sdf_fn}\n{main_fn}"))
    }

    /// Compose and return the final shader source
    ///
    /// # Panics
    /// Panics if the target language's transpiler feature is not enabled
    /// at compile time. Use [`Self::try_build`] to handle this cleanly.
    #[must_use]
    pub fn build(&self) -> String {
        self.try_build().unwrap_or_else(|| {
            panic!(
                "scene_composer: shader language {:?} transpiler feature not enabled",
                self.language
            )
        })
    }

    fn transpile_sdf(&self) -> Option<String> {
        match self.language {
            ShaderLanguage::Glsl => {
                #[cfg(feature = "glsl")]
                {
                    let shader = GlslShader::transpile(self.node, GlslTranspileMode::Hardcoded);
                    Some(shader.get_eval_function().to_string())
                }
                #[cfg(not(feature = "glsl"))]
                {
                    None
                }
            }
            ShaderLanguage::Wgsl => {
                #[cfg(feature = "gpu")]
                {
                    let shader = WgslShader::transpile(self.node, WgslTranspileMode::Hardcoded);
                    Some(shader.get_eval_function().to_string())
                }
                #[cfg(not(feature = "gpu"))]
                {
                    None
                }
            }
            ShaderLanguage::Hlsl => {
                #[cfg(feature = "hlsl")]
                {
                    let shader = HlslShader::transpile(self.node, HlslTranspileMode::Hardcoded);
                    Some(shader.get_eval_function().to_string())
                }
                #[cfg(not(feature = "hlsl"))]
                {
                    None
                }
            }
        }
    }

    fn main_source(&self) -> String {
        match self.language {
            ShaderLanguage::Glsl => self.main_glsl(),
            ShaderLanguage::Wgsl => self.main_wgsl(),
            ShaderLanguage::Hlsl => self.main_hlsl(),
        }
    }

    fn hit_block_glsl(&self) -> String {
        if let Some(pipeline) = &self.pipeline {
            let snip = transpile_npr_color_node(
                pipeline,
                ShaderLanguage::Glsl,
                NprShaderContext::canonical(),
            );
            format!(
                "{statements}        color = {expr};",
                statements = snip.statements,
                expr = snip.color_expression
            )
        } else {
            let shadow = format_vec3_glsl(self.shadow_color);
            let light = format_vec3_glsl(self.light_color);
            let outline = format_vec3_glsl(self.outline_color);
            format!(
                "        float brightness = alice_soft_toon_ramp(ndl, {bands}.0, {smooth});\n\
                 \x20       vec3 shaded = mix({shadow}, {light}, brightness);\n\
                 \x20       float outline_mask = alice_distance_field_outline_soft(d, {out_i}, {out_o});\n\
                 \x20       color = alice_composite_outline(shaded, {outline}, outline_mask);",
                bands = self.toon_bands,
                smooth = format_f32(self.toon_smoothness),
                shadow = shadow,
                light = light,
                out_i = format_f32(self.outline_inner),
                out_o = format_f32(self.outline_outer),
                outline = outline,
            )
        }
    }

    fn hit_block_wgsl(&self) -> String {
        if let Some(pipeline) = &self.pipeline {
            let snip = transpile_npr_color_node(
                pipeline,
                ShaderLanguage::Wgsl,
                NprShaderContext::canonical(),
            );
            format!(
                "{statements}        color = {expr};",
                statements = snip.statements,
                expr = snip.color_expression
            )
        } else {
            let shadow = format_vec3_wgsl(self.shadow_color);
            let light = format_vec3_wgsl(self.light_color);
            let outline = format_vec3_wgsl(self.outline_color);
            format!(
                "        let brightness = alice_soft_toon_ramp(ndl, f32({bands}), {smooth});\n\
                 \x20       let shaded = mix({shadow}, {light}, vec3<f32>(brightness));\n\
                 \x20       let outline_mask = alice_distance_field_outline_soft(d, {out_i}, {out_o});\n\
                 \x20       color = alice_composite_outline(shaded, {outline}, outline_mask);",
                bands = self.toon_bands,
                smooth = format_f32(self.toon_smoothness),
                shadow = shadow,
                light = light,
                out_i = format_f32(self.outline_inner),
                out_o = format_f32(self.outline_outer),
                outline = outline,
            )
        }
    }

    fn hit_block_hlsl(&self) -> String {
        if let Some(pipeline) = &self.pipeline {
            let snip = transpile_npr_color_node(
                pipeline,
                ShaderLanguage::Hlsl,
                NprShaderContext::canonical(),
            );
            format!(
                "{statements}        color = {expr};",
                statements = snip.statements,
                expr = snip.color_expression
            )
        } else {
            let shadow = format_vec3_hlsl(self.shadow_color);
            let light = format_vec3_hlsl(self.light_color);
            let outline = format_vec3_hlsl(self.outline_color);
            format!(
                "        float brightness = alice_soft_toon_ramp(ndl, (float){bands}, {smooth});\n\
                 \x20       float3 shaded = lerp({shadow}, {light}, brightness);\n\
                 \x20       float outline_mask = alice_distance_field_outline_soft(d, {out_i}, {out_o});\n\
                 \x20       color = alice_composite_outline(shaded, {outline}, outline_mask);",
                bands = self.toon_bands,
                smooth = format_f32(self.toon_smoothness),
                shadow = shadow,
                light = light,
                out_i = format_f32(self.outline_inner),
                out_o = format_f32(self.outline_outer),
                outline = outline,
            )
        }
    }

    fn main_glsl(&self) -> String {
        let sky_h = format_vec3_glsl(self.sky_horizon);
        let sky_m = format_vec3_glsl(self.sky_mid);
        let sky_z = format_vec3_glsl(self.sky_zenith);
        let sun = format_vec3_glsl(self.sun_direction);
        let sun_c = format_vec3_glsl(self.sun_color);
        let cam = format_vec3_glsl(self.camera_position);
        format!(
            r"
layout(binding = 0) uniform SceneUniforms {{
    vec2 iResolution;
    float iTime;
}};
layout(location = 0) out vec4 alice_out_color;

vec3 alice_scene_normal(vec3 p) {{
    float e = 1e-3;
    return normalize(vec3(
        sdf_eval(p + vec3(e, 0.0, 0.0)) - sdf_eval(p - vec3(e, 0.0, 0.0)),
        sdf_eval(p + vec3(0.0, e, 0.0)) - sdf_eval(p - vec3(0.0, e, 0.0)),
        sdf_eval(p + vec3(0.0, 0.0, e)) - sdf_eval(p - vec3(0.0, 0.0, e))
    ));
}}

void main() {{
    vec2 uv = gl_FragCoord.xy / iResolution.xy;
    float aspect = iResolution.x / max(iResolution.y, 1.0);
    vec2 sxy = (uv * 2.0 - 1.0);
    sxy.x *= aspect;

    vec3 ray_origin = {cam};
    vec3 ray_dir = normalize(vec3(sxy.x, sxy.y, 1.0));
    vec3 to_sun = {sun};

    float t = 0.0;
    bool hit = false;
    vec3 hit_point = ray_origin;
    for (int i = 0; i < {max_steps}; i++) {{
        hit_point = ray_origin + ray_dir * t;
        float d = sdf_eval(hit_point);
        if (d < {surface_eps}) {{ hit = true; break; }}
        t += max(d, {surface_eps});
        if (t > {max_dist}) break;
    }}

    vec3 color;
    if (hit) {{
        vec3 n = alice_scene_normal(hit_point);
        float ndl = dot(n, to_sun);
        float ndv = -dot(n, ray_dir);
        float d = sdf_eval(hit_point);
{hit_block}
    }} else {{
        vec3 sky = alice_sky_gradient_bands_3(ray_dir, {sky_h}, {sky_m}, {sky_z});
        float sun_i = alice_sun_disc(ray_dir, to_sun, {sd_r}, {sd_s});
        color = sky + {sun_c} * sun_i;
    }}

    color *= alice_vignette(uv, {vig_r}, {vig_s});
    alice_out_color = vec4(color, 1.0);
}}
",
            hit_block = self.hit_block_glsl(),
            cam = cam,
            sun = sun,
            max_steps = self.max_steps,
            surface_eps = format_f32(self.surface_eps),
            max_dist = format_f32(self.max_dist),
            sky_h = sky_h,
            sky_m = sky_m,
            sky_z = sky_z,
            sd_r = format_f32(self.sun_disc_radius),
            sd_s = format_f32(self.sun_disc_softness),
            sun_c = sun_c,
            vig_r = format_f32(self.vignette_radius),
            vig_s = format_f32(self.vignette_softness),
        )
    }

    fn main_wgsl(&self) -> String {
        let sky_h = format_vec3_wgsl(self.sky_horizon);
        let sky_m = format_vec3_wgsl(self.sky_mid);
        let sky_z = format_vec3_wgsl(self.sky_zenith);
        let sun = format_vec3_wgsl(self.sun_direction);
        let sun_c = format_vec3_wgsl(self.sun_color);
        let cam = format_vec3_wgsl(self.camera_position);
        format!(
            r"
struct SceneUniforms {{
    resolution: vec2<f32>,
    iTime: f32,
}}
@group(0) @binding(0) var<uniform> u_scene: SceneUniforms;

fn alice_scene_normal(p: vec3<f32>) -> vec3<f32> {{
    let e = 1e-3;
    return normalize(vec3<f32>(
        sdf_eval(p + vec3<f32>(e, 0.0, 0.0)) - sdf_eval(p - vec3<f32>(e, 0.0, 0.0)),
        sdf_eval(p + vec3<f32>(0.0, e, 0.0)) - sdf_eval(p - vec3<f32>(0.0, e, 0.0)),
        sdf_eval(p + vec3<f32>(0.0, 0.0, e)) - sdf_eval(p - vec3<f32>(0.0, 0.0, e))
    ));
}}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {{
    let uv = vec2<f32>(frag_coord.x / u_scene.resolution.x, frag_coord.y / u_scene.resolution.y);
    let iTime = u_scene.iTime;
    let aspect = u_scene.resolution.x / max(u_scene.resolution.y, 1.0);
    var sxy = uv * 2.0 - vec2<f32>(1.0, 1.0);
    sxy.x = sxy.x * aspect;

    let ray_origin = {cam};
    let ray_dir = normalize(vec3<f32>(sxy.x, sxy.y, 1.0));
    let to_sun = {sun};

    var t = 0.0;
    var hit = false;
    var hit_point = ray_origin;
    for (var i: i32 = 0; i < {max_steps}; i = i + 1) {{
        hit_point = ray_origin + ray_dir * t;
        let d = sdf_eval(hit_point);
        if (d < {surface_eps}) {{
            hit = true;
            break;
        }}
        t = t + max(d, {surface_eps});
        if (t > {max_dist}) {{ break; }}
    }}

    var color: vec3<f32>;
    if (hit) {{
        let n = alice_scene_normal(hit_point);
        let ndl = dot(n, to_sun);
        let ndv = -dot(n, ray_dir);
        let d = sdf_eval(hit_point);
{hit_block}
    }} else {{
        let sky = alice_sky_gradient_bands_3(ray_dir, {sky_h}, {sky_m}, {sky_z});
        let sun_i = alice_sun_disc(ray_dir, to_sun, {sd_r}, {sd_s});
        color = sky + {sun_c} * sun_i;
    }}

    color = color * alice_vignette(uv, {vig_r}, {vig_s});
    return vec4<f32>(color, 1.0);
}}
",
            hit_block = self.hit_block_wgsl(),
            cam = cam,
            sun = sun,
            max_steps = self.max_steps,
            surface_eps = format_f32(self.surface_eps),
            max_dist = format_f32(self.max_dist),
            sky_h = sky_h,
            sky_m = sky_m,
            sky_z = sky_z,
            sd_r = format_f32(self.sun_disc_radius),
            sd_s = format_f32(self.sun_disc_softness),
            sun_c = sun_c,
            vig_r = format_f32(self.vignette_radius),
            vig_s = format_f32(self.vignette_softness),
        )
    }

    fn main_hlsl(&self) -> String {
        let sky_h = format_vec3_hlsl(self.sky_horizon);
        let sky_m = format_vec3_hlsl(self.sky_mid);
        let sky_z = format_vec3_hlsl(self.sky_zenith);
        let sun = format_vec3_hlsl(self.sun_direction);
        let sun_c = format_vec3_hlsl(self.sun_color);
        let cam = format_vec3_hlsl(self.camera_position);
        format!(
            r"
cbuffer SceneCB : register(b0) {{
    float2 resolution;
    float iTime;
}};

float3 alice_scene_normal(float3 p) {{
    float e = 1e-3;
    return normalize(float3(
        sdf_eval(p + float3(e, 0.0, 0.0)) - sdf_eval(p - float3(e, 0.0, 0.0)),
        sdf_eval(p + float3(0.0, e, 0.0)) - sdf_eval(p - float3(0.0, e, 0.0)),
        sdf_eval(p + float3(0.0, 0.0, e)) - sdf_eval(p - float3(0.0, 0.0, e))
    ));
}}

float4 PS(float4 pos : SV_POSITION) : SV_TARGET {{
    float2 uv = pos.xy / resolution;
    float aspect = resolution.x / max(resolution.y, 1.0);
    float2 sxy = uv * 2.0 - float2(1.0, 1.0);
    sxy.x *= aspect;

    float3 ray_origin = {cam};
    float3 ray_dir = normalize(float3(sxy.x, sxy.y, 1.0));
    float3 to_sun = {sun};

    float t = 0.0;
    bool hit = false;
    float3 hit_point = ray_origin;
    [loop] for (int i = 0; i < {max_steps}; i++) {{
        hit_point = ray_origin + ray_dir * t;
        float d = sdf_eval(hit_point);
        if (d < {surface_eps}) {{ hit = true; break; }}
        t += max(d, {surface_eps});
        if (t > {max_dist}) break;
    }}

    float3 color;
    if (hit) {{
        float3 n = alice_scene_normal(hit_point);
        float ndl = dot(n, to_sun);
        float ndv = -dot(n, ray_dir);
        float d = sdf_eval(hit_point);
{hit_block}
    }} else {{
        float3 sky = alice_sky_gradient_bands_3(ray_dir, {sky_h}, {sky_m}, {sky_z});
        float sun_i = alice_sun_disc(ray_dir, to_sun, {sd_r}, {sd_s});
        color = sky + {sun_c} * sun_i;
    }}

    color *= alice_vignette(uv, {vig_r}, {vig_s});
    return float4(color, 1.0);
}}
",
            hit_block = self.hit_block_hlsl(),
            cam = cam,
            sun = sun,
            max_steps = self.max_steps,
            surface_eps = format_f32(self.surface_eps),
            max_dist = format_f32(self.max_dist),
            sky_h = sky_h,
            sky_m = sky_m,
            sky_z = sky_z,
            sd_r = format_f32(self.sun_disc_radius),
            sd_s = format_f32(self.sun_disc_softness),
            sun_c = sun_c,
            vig_r = format_f32(self.vignette_radius),
            vig_s = format_f32(self.vignette_softness),
        )
    }
}

#[inline]
fn format_f32(value: f32) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        format!("{value:.6}")
    }
}

#[inline]
fn format_vec3_glsl(v: Vec3) -> String {
    format!(
        "vec3({}, {}, {})",
        format_f32(v.x),
        format_f32(v.y),
        format_f32(v.z)
    )
}

#[inline]
fn format_vec3_wgsl(v: Vec3) -> String {
    format!(
        "vec3<f32>({}, {}, {})",
        format_f32(v.x),
        format_f32(v.y),
        format_f32(v.z)
    )
}

#[inline]
fn format_vec3_hlsl(v: Vec3) -> String {
    format!(
        "float3({}, {}, {})",
        format_f32(v.x),
        format_f32(v.y),
        format_f32(v.z)
    )
}

// Each test is gated on the transpiler feature it builds with: the module
// exists for any of glsl / hlsl / gpu, and `build()` panics for a language
// whose transpiler is compiled out (found by making the `aaa` CI step a
// hard gate: gpu without glsl / hlsl).
#[cfg(test)]
mod tests {
    use super::*;

    fn unit_sphere() -> SdfNode {
        SdfNode::sphere(1.0)
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn glsl_build_contains_expected_sections() {
        let node = unit_sphere();
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl).build();
        assert!(source.contains("alice_toon_ramp"), "missing NPR helpers");
        assert!(
            source.contains("alice_sky_gradient_bands_3"),
            "missing palette helpers"
        );
        assert!(source.contains("sdf_eval"), "missing sdf function");
        assert!(source.contains("void main"), "missing main entry");
        assert!(
            source.contains("alice_out_color"),
            "missing GLSL output binding"
        );
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn wgsl_build_contains_expected_sections() {
        let node = unit_sphere();
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Wgsl).build();
        assert!(source.contains("alice_toon_ramp"));
        assert!(source.contains("sdf_eval"));
        assert!(source.contains("@fragment"), "missing WGSL fragment stage");
        assert!(source.contains("fs_main"), "missing WGSL entry");
    }

    #[cfg(feature = "hlsl")]
    #[test]
    fn hlsl_build_contains_expected_sections() {
        let node = unit_sphere();
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Hlsl).build();
        assert!(source.contains("alice_toon_ramp"));
        assert!(source.contains("sdf_eval"));
        assert!(source.contains("SV_TARGET"), "missing HLSL semantic");
        assert!(source.contains("float4 PS"), "missing HLSL PS entry");
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn build_reflects_configured_toon_bands() {
        let node = unit_sphere();
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl)
            .with_shading(Vec3::ZERO, Vec3::ONE, 5)
            .build();
        assert!(
            source.contains("5.0"),
            "toon_bands 5 should appear as 5.0 literal"
        );
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn build_reflects_configured_sun_direction() {
        let node = unit_sphere();
        let sun = Vec3::new(0.0, 1.0, 0.0);
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl)
            .with_sun(sun, Vec3::ONE)
            .build();
        // Sun (0, 1, 0) should show up literally in the vec3 constructor
        assert!(
            source.contains("vec3(0.0, 1.0, 0.0)"),
            "sun direction not baked"
        );
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn build_reflects_raymarch_budget() {
        let node = unit_sphere();
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl)
            .with_raymarch(128, 50.0, 1e-4)
            .build();
        assert!(source.contains("128"), "max_steps 128 should appear");
        assert!(source.contains("50.0"), "max_dist 50.0 should appear");
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn shader_source_is_deterministic() {
        let node = unit_sphere();
        let a = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl).build();
        let b = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl).build();
        assert_eq!(a, b);
    }

    #[cfg(feature = "glsl")]
    #[test]
    fn with_pipeline_replaces_hardcoded_hit_block() {
        use crate::npr::dsl::NprColorNode;
        let node = unit_sphere();
        let pipeline = NprColorNode::TwoTone {
            shadow: Vec3::new(0.1, 0.1, 0.3),
            light: Vec3::new(0.9, 0.9, 0.7),
            threshold: 0.5,
        };
        let with = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl)
            .with_pipeline(pipeline)
            .build();
        let without = SceneShaderBuilder::new(&node, ShaderLanguage::Glsl).build();
        // Both shaders always contain the helper function definitions;
        // the pipeline swap only affects which helpers `main()` invokes.
        // Look inside `if (hit)` blocks for the invocation contrast.
        let with_hit = extract_hit_block_glsl(&with);
        let without_hit = extract_hit_block_glsl(&without);
        assert!(
            with_hit.contains("alice_two_tone("),
            "with_pipeline hit block should call alice_two_tone, got:\n{with_hit}"
        );
        assert!(
            !with_hit.contains("alice_soft_toon_ramp("),
            "with_pipeline hit block should not call alice_soft_toon_ramp, got:\n{with_hit}"
        );
        assert!(
            without_hit.contains("alice_soft_toon_ramp("),
            "default hit block should call alice_soft_toon_ramp, got:\n{without_hit}"
        );
    }

    #[cfg(feature = "glsl")]
    fn extract_hit_block_glsl(source: &str) -> String {
        let start = source
            .find("if (hit) {")
            .expect("scene shader missing `if (hit) {` marker");
        let rest = &source[start..];
        let end = rest
            .find("} else {")
            .expect("scene shader missing `} else {` marker");
        rest[..end].to_string()
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn with_pipeline_supports_wgsl() {
        use crate::npr::dsl::NprColorNode;
        let node = unit_sphere();
        let pipeline = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 4,
        };
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Wgsl)
            .with_pipeline(pipeline)
            .build();
        assert!(source.contains("alice_toon_ramp(ndl, 4.0)"));
        assert!(source.contains("vec3<f32>("));
    }

    #[cfg(feature = "hlsl")]
    #[test]
    fn with_pipeline_supports_hlsl_outline_over() {
        use crate::npr::dsl::NprColorNode;
        let node = unit_sphere();
        let base = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let pipeline = base.with_outline(Vec3::ZERO, 0.7);
        let source = SceneShaderBuilder::new(&node, ShaderLanguage::Hlsl)
            .with_pipeline(pipeline)
            .build();
        assert!(source.contains("alice_composite_outline"));
        assert!(source.contains("alice_col_0"));
        assert!(source.contains("float3 alice_col_0"));
    }
}

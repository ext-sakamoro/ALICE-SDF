//! Shader-language helper strings for the NPR primitives
//!
//! Static source snippets for GLSL, WGSL, and HLSL that reproduce the
//! Rust NPR primitives on the GPU. Callers concatenate the appropriate
//! constant with their own shader source to make the helpers available.
//!
//! Coverage in this module is the subset of primitives whose signatures
//! translate directly to shader code (no `&[Vec3]` palette arrays).
//! Palette-driven primitives (`palette_gradient`, `time_of_day`,
//! `season_palette`, `sky_gradient_bands`) can be recreated in shader
//! code by the caller using the fixed-arity mix / step patterns shown
//! for the other primitives.
//!
//! Author: Moroya Sakamoto

// ============================================================================
// GLSL (OpenGL / Vulkan)
// ============================================================================

/// GLSL 330+ helpers for the NPR primitives with fixed-arity signatures
pub const NPR_GLSL_HELPERS: &str = r#"
// ALICE-SDF NPR helpers (GLSL 330+)

float alice_toon_ramp(float n_dot_l, float bands) {
    float c = clamp(n_dot_l, 0.0, 1.0);
    float idx = min(floor(c * bands), bands - 1.0);
    return idx / max(bands - 1.0, 1.0);
}

float alice_soft_toon_ramp(float n_dot_l, float bands, float smoothness) {
    float c = clamp(n_dot_l, 0.0, 1.0);
    float scaled = c * bands;
    float idx = min(floor(scaled), bands - 1.0);
    float frac_ = clamp(scaled - idx, 0.0, 1.0);
    float s = clamp(smoothness, 0.001, 0.5);
    float t = smoothstep(0.5 - s, 0.5 + s, frac_);
    return clamp((idx + t) / bands, 0.0, 1.0);
}

vec3 alice_two_tone(float n_dot_l, vec3 shadow, vec3 light, float threshold) {
    return mix(shadow, light, step(threshold, clamp(n_dot_l, 0.0, 1.0)));
}

vec3 alice_posterize_color(vec3 color, float levels) {
    vec3 c = clamp(color, vec3(0.0), vec3(1.0));
    float denom = max(levels - 1.0, 1.0);
    return clamp(floor(c * levels) / denom, vec3(0.0), vec3(1.0));
}

float alice_distance_field_outline(float sdf_v, float width) {
    return step(abs(sdf_v), max(width, 0.0));
}

float alice_distance_field_outline_soft(float sdf_v, float width_inner, float width_outer) {
    float d = abs(sdf_v);
    float inner = max(width_inner, 0.0);
    float outer = max(width_outer, inner + 1e-6);
    if (d <= inner) return 1.0;
    if (d >= outer) return 0.0;
    float t = clamp((d - inner) / (outer - inner), 0.0, 1.0);
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

vec3 alice_composite_outline(vec3 base, vec3 outline_col, float alpha) {
    return mix(base, outline_col, clamp(alpha, 0.0, 1.0));
}

float alice_fresnel_rim(float n_dot_v, float power, float intensity) {
    float base = max(1.0 - clamp(n_dot_v, 0.0, 1.0), 0.0);
    return pow(base, max(power, 0.0)) * intensity;
}

vec3 alice_procedural_matcap(vec3 normal_view, vec3 bl, vec3 br, vec3 tl, vec3 tr) {
    float u = clamp(normal_view.x * 0.5 + 0.5, 0.0, 1.0);
    float v = clamp(normal_view.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 bottom = mix(bl, br, u);
    vec3 top = mix(tl, tr, u);
    return mix(bottom, top, v);
}

float alice_stylized_specular(float n_dot_h, float sharpness, float soft_edge) {
    float ndh = clamp(n_dot_h, 0.0, 1.0);
    float raw = pow(ndh, max(sharpness, 1.0));
    float s = clamp(soft_edge, 0.0, 0.5);
    if (s < 1e-6) return step(0.5, raw);
    return smoothstep(0.5 - s, 0.5 + s, raw);
}

float alice_vignette(vec2 uv, float radius, float softness) {
    vec2 d = uv - vec2(0.5);
    float len = length(d);
    float inner = max(radius, 0.0);
    float outer = max(inner + max(softness, 0.0), inner + 1e-6);
    if (len <= inner) return 1.0;
    if (len >= outer) return 0.0;
    float t = clamp((len - inner) / (outer - inner), 0.0, 1.0);
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

vec3 alice_bloom_toon(vec3 color, float threshold, float intensity) {
    float m = max(max(color.r, color.g), color.b);
    return (m > threshold) ? color * intensity : vec3(0.0);
}

float alice_puffy_cloud_layer(float noise_sample, float coverage, float softness) {
    float n = clamp(noise_sample, 0.0, 1.0);
    float c = clamp(coverage, 0.0, 1.0);
    float s = max(softness, 1e-4);
    return clamp((n - (1.0 - c)) / s, 0.0, 1.0);
}

float alice_light_shaft_beam(vec3 view_dir, vec3 to_sun, float density) {
    vec3 v = normalize(view_dir + vec3(1e-8));
    vec3 s = normalize(to_sun + vec3(1e-8));
    float cos_theta = max(dot(v, s), 0.0);
    return pow(cos_theta, max(density, 1.0));
}

float alice_impact_flash(float t, float decay, float intensity) {
    if (t < 0.0) return 0.0;
    return intensity * exp(-t / max(decay, 1e-4));
}
"#;

// ============================================================================
// WGSL (WebGPU)
// ============================================================================

/// WGSL helpers matching the GLSL set above
pub const NPR_WGSL_HELPERS: &str = r#"
// ALICE-SDF NPR helpers (WGSL)

fn alice_toon_ramp(n_dot_l: f32, bands: f32) -> f32 {
    let c = clamp(n_dot_l, 0.0, 1.0);
    let idx = min(floor(c * bands), bands - 1.0);
    return idx / max(bands - 1.0, 1.0);
}

fn alice_soft_toon_ramp(n_dot_l: f32, bands: f32, smoothness: f32) -> f32 {
    let c = clamp(n_dot_l, 0.0, 1.0);
    let scaled = c * bands;
    let idx = min(floor(scaled), bands - 1.0);
    let frac_val = clamp(scaled - idx, 0.0, 1.0);
    let s = clamp(smoothness, 0.001, 0.5);
    let t = smoothstep(0.5 - s, 0.5 + s, frac_val);
    return clamp((idx + t) / bands, 0.0, 1.0);
}

fn alice_two_tone(n_dot_l: f32, shadow: vec3<f32>, light: vec3<f32>, threshold: f32) -> vec3<f32> {
    return mix(shadow, light, vec3<f32>(step(threshold, clamp(n_dot_l, 0.0, 1.0))));
}

fn alice_posterize_color(color: vec3<f32>, levels: f32) -> vec3<f32> {
    let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
    let denom = max(levels - 1.0, 1.0);
    return clamp(floor(c * levels) / denom, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn alice_distance_field_outline(sdf_v: f32, width: f32) -> f32 {
    return step(abs(sdf_v), max(width, 0.0));
}

fn alice_distance_field_outline_soft(sdf_v: f32, width_inner: f32, width_outer: f32) -> f32 {
    let d = abs(sdf_v);
    let inner = max(width_inner, 0.0);
    let outer = max(width_outer, inner + 1e-6);
    if (d <= inner) { return 1.0; }
    if (d >= outer) { return 0.0; }
    let t = clamp((d - inner) / (outer - inner), 0.0, 1.0);
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

fn alice_composite_outline(base: vec3<f32>, outline_col: vec3<f32>, alpha: f32) -> vec3<f32> {
    return mix(base, outline_col, vec3<f32>(clamp(alpha, 0.0, 1.0)));
}

fn alice_fresnel_rim(n_dot_v: f32, power: f32, intensity: f32) -> f32 {
    let base = max(1.0 - clamp(n_dot_v, 0.0, 1.0), 0.0);
    return pow(base, max(power, 0.0)) * intensity;
}

fn alice_procedural_matcap(
    normal_view: vec3<f32>,
    bl: vec3<f32>, br: vec3<f32>,
    tl: vec3<f32>, tr: vec3<f32>,
) -> vec3<f32> {
    let u = clamp(normal_view.x * 0.5 + 0.5, 0.0, 1.0);
    let v = clamp(normal_view.y * 0.5 + 0.5, 0.0, 1.0);
    let bottom = mix(bl, br, vec3<f32>(u));
    let top = mix(tl, tr, vec3<f32>(u));
    return mix(bottom, top, vec3<f32>(v));
}

fn alice_stylized_specular(n_dot_h: f32, sharpness: f32, soft_edge: f32) -> f32 {
    let ndh = clamp(n_dot_h, 0.0, 1.0);
    let raw = pow(ndh, max(sharpness, 1.0));
    let s = clamp(soft_edge, 0.0, 0.5);
    if (s < 1e-6) { return step(0.5, raw); }
    return smoothstep(0.5 - s, 0.5 + s, raw);
}

fn alice_vignette(uv: vec2<f32>, radius: f32, softness: f32) -> f32 {
    let d = uv - vec2<f32>(0.5, 0.5);
    let len = length(d);
    let inner = max(radius, 0.0);
    let outer = max(inner + max(softness, 0.0), inner + 1e-6);
    if (len <= inner) { return 1.0; }
    if (len >= outer) { return 0.0; }
    let t = clamp((len - inner) / (outer - inner), 0.0, 1.0);
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

fn alice_bloom_toon(color: vec3<f32>, threshold: f32, intensity: f32) -> vec3<f32> {
    let m = max(max(color.r, color.g), color.b);
    if (m > threshold) { return color * intensity; }
    return vec3<f32>(0.0, 0.0, 0.0);
}

fn alice_puffy_cloud_layer(noise_sample: f32, coverage: f32, softness: f32) -> f32 {
    let n = clamp(noise_sample, 0.0, 1.0);
    let c = clamp(coverage, 0.0, 1.0);
    let s = max(softness, 1e-4);
    return clamp((n - (1.0 - c)) / s, 0.0, 1.0);
}

fn alice_light_shaft_beam(view_dir: vec3<f32>, to_sun: vec3<f32>, density: f32) -> f32 {
    let v = normalize(view_dir + vec3<f32>(1e-8));
    let s = normalize(to_sun + vec3<f32>(1e-8));
    let cos_theta = max(dot(v, s), 0.0);
    return pow(cos_theta, max(density, 1.0));
}

fn alice_impact_flash(t: f32, decay: f32, intensity: f32) -> f32 {
    if (t < 0.0) { return 0.0; }
    return intensity * exp(-t / max(decay, 1e-4));
}
"#;

// ============================================================================
// HLSL (Direct3D)
// ============================================================================

/// HLSL Shader Model 5.0+ helpers matching the GLSL set
pub const NPR_HLSL_HELPERS: &str = r#"
// ALICE-SDF NPR helpers (HLSL SM 5.0+)

float alice_toon_ramp(float n_dot_l, float bands) {
    float c = saturate(n_dot_l);
    float idx = min(floor(c * bands), bands - 1.0);
    return idx / max(bands - 1.0, 1.0);
}

float alice_soft_toon_ramp(float n_dot_l, float bands, float smoothness) {
    float c = saturate(n_dot_l);
    float scaled = c * bands;
    float idx = min(floor(scaled), bands - 1.0);
    float frac_val = saturate(scaled - idx);
    float s = clamp(smoothness, 0.001, 0.5);
    float t = smoothstep(0.5 - s, 0.5 + s, frac_val);
    return saturate((idx + t) / bands);
}

float3 alice_two_tone(float n_dot_l, float3 shadow, float3 light, float threshold) {
    return lerp(shadow, light, step(threshold, saturate(n_dot_l)));
}

float3 alice_posterize_color(float3 color, float levels) {
    float3 c = saturate(color);
    float denom = max(levels - 1.0, 1.0);
    return saturate(floor(c * levels) / denom);
}

float alice_distance_field_outline(float sdf_v, float width) {
    return step(abs(sdf_v), max(width, 0.0));
}

float alice_distance_field_outline_soft(float sdf_v, float width_inner, float width_outer) {
    float d = abs(sdf_v);
    float inner = max(width_inner, 0.0);
    float outer = max(width_outer, inner + 1e-6);
    if (d <= inner) return 1.0;
    if (d >= outer) return 0.0;
    float t = saturate((d - inner) / (outer - inner));
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

float3 alice_composite_outline(float3 base, float3 outline_col, float alpha) {
    return lerp(base, outline_col, saturate(alpha));
}

float alice_fresnel_rim(float n_dot_v, float power, float intensity) {
    float base = max(1.0 - saturate(n_dot_v), 0.0);
    return pow(base, max(power, 0.0)) * intensity;
}

float3 alice_procedural_matcap(float3 normal_view, float3 bl, float3 br, float3 tl, float3 tr) {
    float u = saturate(normal_view.x * 0.5 + 0.5);
    float v = saturate(normal_view.y * 0.5 + 0.5);
    float3 bottom = lerp(bl, br, u);
    float3 top = lerp(tl, tr, u);
    return lerp(bottom, top, v);
}

float alice_stylized_specular(float n_dot_h, float sharpness, float soft_edge) {
    float ndh = saturate(n_dot_h);
    float raw = pow(ndh, max(sharpness, 1.0));
    float s = clamp(soft_edge, 0.0, 0.5);
    if (s < 1e-6) return step(0.5, raw);
    return smoothstep(0.5 - s, 0.5 + s, raw);
}

float alice_vignette(float2 uv, float radius, float softness) {
    float2 d = uv - float2(0.5, 0.5);
    float len = length(d);
    float inner = max(radius, 0.0);
    float outer = max(inner + max(softness, 0.0), inner + 1e-6);
    if (len <= inner) return 1.0;
    if (len >= outer) return 0.0;
    float t = saturate((len - inner) / (outer - inner));
    return 1.0 - (t * t * (3.0 - 2.0 * t));
}

float3 alice_bloom_toon(float3 color, float threshold, float intensity) {
    float m = max(max(color.r, color.g), color.b);
    return (m > threshold) ? color * intensity : float3(0.0, 0.0, 0.0);
}

float alice_puffy_cloud_layer(float noise_sample, float coverage, float softness) {
    float n = saturate(noise_sample);
    float c = saturate(coverage);
    float s = max(softness, 1e-4);
    return saturate((n - (1.0 - c)) / s);
}

float alice_light_shaft_beam(float3 view_dir, float3 to_sun, float density) {
    float3 v = normalize(view_dir + float3(1e-8, 1e-8, 1e-8));
    float3 s = normalize(to_sun + float3(1e-8, 1e-8, 1e-8));
    float cos_theta = max(dot(v, s), 0.0);
    return pow(cos_theta, max(density, 1.0));
}

float alice_impact_flash(float t, float decay, float intensity) {
    if (t < 0.0) return 0.0;
    return intensity * exp(-t / max(decay, 1e-4));
}
"#;

/// Return the helper snippet for the requested shader language, or `None`
/// if the language is not supported
#[must_use]
pub const fn helpers_for(language: ShaderLanguage) -> &'static str {
    match language {
        ShaderLanguage::Glsl => NPR_GLSL_HELPERS,
        ShaderLanguage::Wgsl => NPR_WGSL_HELPERS,
        ShaderLanguage::Hlsl => NPR_HLSL_HELPERS,
    }
}

/// Selector for the shader language dispatched by [`helpers_for`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderLanguage {
    /// OpenGL / Vulkan Shading Language 330+
    Glsl,
    /// WebGPU Shading Language
    Wgsl,
    /// Direct3D High-Level Shading Language 5.0+
    Hlsl,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glsl_helpers_have_expected_functions() {
        for name in [
            "alice_toon_ramp",
            "alice_soft_toon_ramp",
            "alice_two_tone",
            "alice_posterize_color",
            "alice_distance_field_outline",
            "alice_distance_field_outline_soft",
            "alice_fresnel_rim",
            "alice_procedural_matcap",
            "alice_stylized_specular",
            "alice_vignette",
            "alice_bloom_toon",
            "alice_puffy_cloud_layer",
            "alice_light_shaft_beam",
            "alice_impact_flash",
        ] {
            assert!(
                NPR_GLSL_HELPERS.contains(name),
                "missing GLSL helper: {name}"
            );
            assert!(
                NPR_WGSL_HELPERS.contains(name),
                "missing WGSL helper: {name}"
            );
            assert!(
                NPR_HLSL_HELPERS.contains(name),
                "missing HLSL helper: {name}"
            );
        }
    }

    #[test]
    fn helpers_for_dispatch_returns_expected_source() {
        assert_eq!(helpers_for(ShaderLanguage::Glsl), NPR_GLSL_HELPERS);
        assert_eq!(helpers_for(ShaderLanguage::Wgsl), NPR_WGSL_HELPERS);
        assert_eq!(helpers_for(ShaderLanguage::Hlsl), NPR_HLSL_HELPERS);
    }

    #[test]
    fn snippets_are_nontrivial() {
        assert!(NPR_GLSL_HELPERS.len() > 200);
        assert!(NPR_WGSL_HELPERS.len() > 200);
        assert!(NPR_HLSL_HELPERS.len() > 200);
    }
}

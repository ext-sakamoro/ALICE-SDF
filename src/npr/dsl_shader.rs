//! Transpile [`NprColorNode`] expression trees to shader-language source
//!
//! Produces a `NprShaderSnippet` containing:
//! - a list of variable declaration statements
//! - a final expression that evaluates to the composed colour
//!
//! Callers (typically `crate::npr::scene_composer::SceneShaderBuilder`, feature-gated)
//! embed the statements at the call site and use the expression as the
//! resulting `vec3` / `float3` colour.
//!
//! Emitted code depends on the NPR helpers exported by
//! [`crate::npr::shader_glue`]: `alice_toon_ramp`, `alice_soft_toon_ramp`,
//! `alice_two_tone`, and `alice_composite_outline`; `RegionNoise` also needs
//! the crate's noise law (`alice_hash3` / `hash_noise_3d` from
//! `crate::modifiers::HASH_NOISE_{GLSL, WGSL, HLSL}`) in the same shader.
//!
//! Author: Moroya Sakamoto

use crate::npr::dsl::{NprColorNode, PaletteSource};
use crate::npr::shader_glue::ShaderLanguage;
use glam::Vec3;

/// Names of the shader-side variables holding the shading context
///
/// Callers set these to whatever variable names their emitted `main()`
/// uses for the diffuse dot, view dot, and signed distance at the
/// shading point.
#[derive(Debug, Clone, Copy)]
pub struct NprShaderContext<'a> {
    /// Name of the `n . l` scalar variable
    pub n_dot_l: &'a str,
    /// Name of the `n . v` scalar variable (view direction from surface to camera)
    pub n_dot_v: &'a str,
    /// Name of the signed-distance scalar variable at the hit point
    pub sdf: &'a str,
    /// Name of the 2D UV variable (`vec2` / `vec2<f32>` / `float2`)
    pub uv: &'a str,
    /// Name of the animation time scalar variable (typically `iTime`)
    pub time: &'a str,
    /// Name of the unit surface normal variable (`vec3` / `vec3<f32>` / `float3`)
    pub normal: &'a str,
    /// Name of the shading position variable in the law's frame (`vec3` / `vec3<f32>` / `float3`)
    pub position: &'a str,
}

impl NprShaderContext<'_> {
    /// Canonical context matching `SceneShaderBuilder`'s emitted `main()`
    #[must_use]
    pub const fn canonical() -> NprShaderContext<'static> {
        NprShaderContext {
            n_dot_l: "ndl",
            n_dot_v: "ndv",
            sdf: "d",
            uv: "uv",
            time: "iTime",
            normal: "n",
            position: "p",
        }
    }
}

/// A compiled color-pipeline snippet ready to embed in a shader `main()`
#[derive(Debug, Clone)]
pub struct NprShaderSnippet {
    /// Statements declaring intermediate colour variables (may be empty)
    pub statements: String,
    /// Final expression producing a `vec3` / `float3` colour
    pub color_expression: String,
}

/// Transpile an [`NprColorNode`] to a shader snippet in the target language
#[must_use]
pub fn transpile_npr_color_node(
    node: &NprColorNode,
    language: ShaderLanguage,
    ctx: NprShaderContext,
) -> NprShaderSnippet {
    let mut walker = Walker::new(language);
    let expr = walker.walk(node, ctx);
    NprShaderSnippet {
        statements: walker.statements,
        color_expression: expr,
    }
}

struct Walker {
    language: ShaderLanguage,
    var_counter: usize,
    statements: String,
}

impl Walker {
    const fn new(language: ShaderLanguage) -> Self {
        Self {
            language,
            var_counter: 0,
            statements: String::new(),
        }
    }

    fn next_var(&mut self) -> String {
        let name = format!("alice_col_{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    fn vec3_literal(&self, v: Vec3) -> String {
        format_vec3(self.language, v)
    }

    fn decl_vec3(&self, name: &str, expr: &str) -> String {
        match self.language {
            ShaderLanguage::Glsl => format!("    vec3 {name} = {expr};\n"),
            ShaderLanguage::Wgsl => format!("    let {name} = {expr};\n"),
            ShaderLanguage::Hlsl => format!("    float3 {name} = {expr};\n"),
        }
    }

    /// An immutable scalar declaration
    fn decl_f32(&self, name: &str, expr: &str) -> String {
        match self.language {
            ShaderLanguage::Glsl | ShaderLanguage::Hlsl => format!("    float {name} = {expr};\n"),
            ShaderLanguage::Wgsl => format!("    let {name} = {expr};\n"),
        }
    }

    /// A mutable scalar declaration (assigned again below it)
    fn decl_var_f32(&self, name: &str, expr: &str) -> String {
        match self.language {
            ShaderLanguage::Glsl | ShaderLanguage::Hlsl => format!("    float {name} = {expr};\n"),
            ShaderLanguage::Wgsl => format!("    var {name}: f32 = {expr};\n"),
        }
    }

    /// A mutable colour declaration (assigned again below it)
    fn decl_var_vec3(&self, name: &str, expr: &str) -> String {
        match self.language {
            ShaderLanguage::Glsl => format!("    vec3 {name} = {expr};\n"),
            ShaderLanguage::Wgsl => format!("    var {name}: vec3<f32> = {expr};\n"),
            ShaderLanguage::Hlsl => format!("    float3 {name} = {expr};\n"),
        }
    }

    fn mix_call(&self, a: &str, b: &str, t: &str) -> String {
        match self.language {
            ShaderLanguage::Glsl => format!("mix({a}, {b}, {t})"),
            // WGSL requires the third argument to be a vec3 when a, b are vec3
            ShaderLanguage::Wgsl => format!("mix({a}, {b}, vec3<f32>({t}))"),
            ShaderLanguage::Hlsl => format!("lerp({a}, {b}, {t})"),
        }
    }

    fn walk(&mut self, node: &NprColorNode, ctx: NprShaderContext) -> String {
        match node {
            NprColorNode::Constant(color) => self.vec3_literal(*color),
            NprColorNode::Toon {
                shadow,
                light,
                bands,
            } => {
                let shadow_expr = self.vec3_literal(*shadow);
                let light_expr = self.vec3_literal(*light);
                let ramp = format!("alice_toon_ramp({}, {}.0)", ctx.n_dot_l, bands);
                self.mix_call(&shadow_expr, &light_expr, &ramp)
            }
            NprColorNode::SoftToon {
                shadow,
                light,
                bands,
                smoothness,
            } => {
                let shadow_expr = self.vec3_literal(*shadow);
                let light_expr = self.vec3_literal(*light);
                let ramp = format!(
                    "alice_soft_toon_ramp({}, {}.0, {})",
                    ctx.n_dot_l,
                    bands,
                    format_f32(*smoothness)
                );
                self.mix_call(&shadow_expr, &light_expr, &ramp)
            }
            NprColorNode::TwoTone {
                shadow,
                light,
                threshold,
            } => {
                let shadow_expr = self.vec3_literal(*shadow);
                let light_expr = self.vec3_literal(*light);
                format!(
                    "alice_two_tone({}, {}, {}, {})",
                    ctx.n_dot_l,
                    shadow_expr,
                    light_expr,
                    format_f32(*threshold)
                )
            }
            NprColorNode::OutlineOver {
                base,
                outline,
                alpha,
            } => {
                let base_expr = self.walk(base, ctx);
                let base_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&base_var, &base_expr));
                let outline_expr = self.vec3_literal(*outline);
                format!(
                    "alice_composite_outline({}, {}, {})",
                    base_var,
                    outline_expr,
                    format_f32(*alpha)
                )
            }
            NprColorNode::Multiply { a, b } => {
                // Materialize both sides so nested expressions are only evaluated once
                let a_expr = self.walk(a, ctx);
                let a_var = self.next_var();
                self.statements.push_str(&self.decl_vec3(&a_var, &a_expr));
                let b_expr = self.walk(b, ctx);
                let b_var = self.next_var();
                self.statements.push_str(&self.decl_vec3(&b_var, &b_expr));
                format!("({a_var} * {b_var})")
            }
            NprColorNode::Add { a, b } => {
                let a_expr = self.walk(a, ctx);
                let a_var = self.next_var();
                self.statements.push_str(&self.decl_vec3(&a_var, &a_expr));
                let b_expr = self.walk(b, ctx);
                let b_var = self.next_var();
                self.statements.push_str(&self.decl_vec3(&b_var, &b_expr));
                format!("({a_var} + {b_var})")
            }
            NprColorNode::Scale { child, factor } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!("({child_var} * {})", format_f32(*factor))
            }
            NprColorNode::Fresnel { base, edge, power } => {
                let base_expr = self.walk(base, ctx);
                let base_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&base_var, &base_expr));
                let edge_expr = self.vec3_literal(*edge);
                let mask = format!(
                    "alice_fresnel_rim({}, {}, 1.0)",
                    ctx.n_dot_v,
                    format_f32(*power)
                );
                self.mix_call(&base_var, &edge_expr, &mask)
            }
            NprColorNode::Saturate { child, factor } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!("alice_saturate({child_var}, {})", format_f32(*factor))
            }
            NprColorNode::Bloom {
                child,
                threshold,
                intensity,
            } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!(
                    "alice_bloom_toon({child_var}, {}, {})",
                    format_f32(*threshold),
                    format_f32(*intensity)
                )
            }
            NprColorNode::PosterizeColor { child, levels } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!("alice_posterize_color({child_var}, {}.0)", (*levels).max(2))
            }
            NprColorNode::Vignette {
                child,
                radius,
                softness,
            } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!(
                    "({child_var} * alice_vignette({}, {}, {}))",
                    ctx.uv,
                    format_f32(*radius),
                    format_f32(*softness)
                )
            }
            NprColorNode::FamilyPalette { families } => {
                // the family law unrolled: one compare per family, the lowest index wins a tie (strict `>`)
                let out = self.next_var();
                let zero = self.vec3_literal(Vec3::ZERO);
                self.statements.push_str(&self.decl_var_vec3(&out, &zero));
                let best = format!("{out}_best");
                self.statements
                    .push_str(&self.decl_var_f32(&best, "-3.402823466e38"));
                for (i, (n, c)) in families.iter().enumerate() {
                    let d = format!("{out}_d{i}");
                    let dot = format!(
                        "({nrm}.x * {} + {nrm}.y * {} + {nrm}.z * {})",
                        format_f32(n.x),
                        format_f32(n.y),
                        format_f32(n.z),
                        nrm = ctx.normal
                    );
                    self.statements.push_str(&self.decl_f32(&d, &dot));
                    let colour = self.vec3_literal(*c);
                    self.statements.push_str(&format!(
                        "    if ({d} > {best}) {{\n        {best} = {d};\n        {out} = {colour};\n    }}\n"
                    ));
                }
                out
            }
            NprColorNode::RegionNoise {
                base,
                cell,
                region_amp,
                noise_amp,
                seed,
            } => {
                let base_expr = self.walk(base, ctx);
                let base_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&base_var, &base_expr));
                let q = format!("{base_var}_q");
                let q_expr = format!("{} / {}", ctx.position, format_f32(*cell));
                self.statements.push_str(&self.decl_vec3(&q, &q_expr));
                let h = format!("{base_var}_h");
                self.statements
                    .push_str(&self.decl_f32(&h, &format!("alice_hash3(floor({q}), {seed}u)")));
                let v = format!("{base_var}_v");
                self.statements.push_str(&self.decl_f32(
                    &v,
                    &format!("hash_noise_3d({q}, {}u)", seed.wrapping_add(1)),
                ));
                format!(
                    "({base_var} + {} * (2.0 * {h} - 1.0) + {} * {v})",
                    self.vec3_literal(*region_amp),
                    self.vec3_literal(*noise_amp)
                )
            }
            NprColorNode::Palette3 { source, c0, c1, c2 } => {
                let c0_expr = self.vec3_literal(*c0);
                let c1_expr = self.vec3_literal(*c1);
                let c2_expr = self.vec3_literal(*c2);
                let t = palette_source_expression(*source, ctx);
                format!("alice_palette_gradient_3({t}, {c0_expr}, {c1_expr}, {c2_expr})")
            }
            NprColorNode::Hatch {
                base,
                angle_rad,
                density,
                thickness,
                ink,
            } => {
                let base_expr = self.walk(base, ctx);
                let base_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&base_var, &base_expr));
                let ink_expr = self.vec3_literal(*ink);
                let mask = format!(
                    "alice_hatch_lines({}, {}, {}, {})",
                    ctx.uv,
                    format_f32(*angle_rad),
                    format_f32(*density),
                    format_f32(*thickness)
                );
                self.mix_call(&base_var, &ink_expr, &mask)
            }
            NprColorNode::Palette5 {
                source,
                c0,
                c1,
                c2,
                c3,
                c4,
            } => {
                let c0_expr = self.vec3_literal(*c0);
                let c1_expr = self.vec3_literal(*c1);
                let c2_expr = self.vec3_literal(*c2);
                let c3_expr = self.vec3_literal(*c3);
                let c4_expr = self.vec3_literal(*c4);
                let t = palette_source_expression(*source, ctx);
                format!(
                    "alice_palette_gradient_5({t}, {c0_expr}, {c1_expr}, {c2_expr}, {c3_expr}, {c4_expr})"
                )
            }
            NprColorNode::Tonemap { child, exposure } => {
                let child_expr = self.walk(child, ctx);
                let child_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&child_var, &child_expr));
                format!(
                    "alice_tonemap_reinhard({child_var}, {})",
                    format_f32(*exposure)
                )
            }
            NprColorNode::SpeedLine {
                base,
                focus,
                count,
                thickness,
                ink,
            } => {
                let base_expr = self.walk(base, ctx);
                let base_var = self.next_var();
                self.statements
                    .push_str(&self.decl_vec3(&base_var, &base_expr));
                let ink_expr = self.vec3_literal(*ink);
                let focus_expr = format_vec2(self.language, focus.x, focus.y);
                let mask = format!(
                    "alice_speed_line({}, {}, {}.0, {})",
                    ctx.uv,
                    focus_expr,
                    count,
                    format_f32(*thickness)
                );
                self.mix_call(&base_var, &ink_expr, &mask)
            }
        }
    }
}

#[inline]
fn format_vec2(language: ShaderLanguage, x: f32, y: f32) -> String {
    match language {
        ShaderLanguage::Glsl => format!("vec2({}, {})", format_f32(x), format_f32(y)),
        ShaderLanguage::Wgsl => format!("vec2<f32>({}, {})", format_f32(x), format_f32(y)),
        ShaderLanguage::Hlsl => format!("float2({}, {})", format_f32(x), format_f32(y)),
    }
}

/// Emit the shader expression for a [`PaletteSource`] against a context
#[inline]
fn palette_source_expression(source: PaletteSource, ctx: NprShaderContext) -> String {
    match source {
        PaletteSource::NDotL => format!("clamp({}, 0.0, 1.0)", ctx.n_dot_l),
        PaletteSource::NDotV => format!("clamp({}, 0.0, 1.0)", ctx.n_dot_v),
        PaletteSource::Sdf => format!("clamp(abs({}), 0.0, 1.0)", ctx.sdf),
        PaletteSource::UvY => format!("clamp({}.y, 0.0, 1.0)", ctx.uv),
        PaletteSource::TimeCycle => format!("({time} - floor({time}))", time = ctx.time),
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
fn format_vec3(language: ShaderLanguage, v: Vec3) -> String {
    match language {
        ShaderLanguage::Glsl => format!(
            "vec3({}, {}, {})",
            format_f32(v.x),
            format_f32(v.y),
            format_f32(v.z)
        ),
        ShaderLanguage::Wgsl => format!(
            "vec3<f32>({}, {}, {})",
            format_f32(v.x),
            format_f32(v.y),
            format_f32(v.z)
        ),
        ShaderLanguage::Hlsl => format!(
            "float3({}, {}, {})",
            format_f32(v.x),
            format_f32(v.y),
            format_f32(v.z)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> NprShaderContext<'static> {
        NprShaderContext::canonical()
    }

    #[test]
    fn constant_emits_vec3_literal_glsl() {
        let node = NprColorNode::Constant(Vec3::new(0.1, 0.2, 0.3));
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.statements.is_empty());
        assert_eq!(snip.color_expression, "vec3(0.100000, 0.200000, 0.300000)");
    }

    #[test]
    fn constant_emits_vec3_literal_wgsl() {
        let node = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert_eq!(
            snip.color_expression,
            "vec3<f32>(0.500000, 0.500000, 0.500000)"
        );
    }

    #[test]
    fn constant_emits_vec3_literal_hlsl() {
        let node = NprColorNode::Constant(Vec3::new(1.0, 0.0, 0.0));
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert_eq!(snip.color_expression, "float3(1.0, 0.0, 0.0)");
    }

    #[test]
    fn toon_emits_mix_with_ramp_glsl() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 3,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.statements.is_empty());
        assert!(snip.color_expression.contains("alice_toon_ramp(ndl, 3.0)"));
        assert!(snip.color_expression.starts_with("mix("));
    }

    #[test]
    fn soft_toon_emits_mix_with_soft_ramp_wgsl() {
        let node = NprColorNode::SoftToon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 4,
            smoothness: 0.05,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert!(snip
            .color_expression
            .contains("alice_soft_toon_ramp(ndl, 4.0, 0.050000)"));
        assert!(snip
            .color_expression
            .contains("vec3<f32>(alice_soft_toon_ramp"));
    }

    #[test]
    fn two_tone_emits_direct_call_hlsl() {
        let node = NprColorNode::TwoTone {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            threshold: 0.5,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.starts_with("alice_two_tone(ndl,"));
        assert!(snip.color_expression.contains("0.5"));
    }

    #[test]
    fn outline_over_uses_temp_var() {
        let base = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let node = base.with_outline(Vec3::ZERO, 0.7);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        // Should have declared a temp var and used it in composite_outline
        assert!(snip.statements.contains("alice_col_0"));
        assert!(snip.statements.contains("vec3"));
        assert!(snip
            .color_expression
            .starts_with("alice_composite_outline(alice_col_0,"));
        assert!(snip.color_expression.contains("0.700000"));
    }

    #[test]
    fn nested_outline_produces_two_temp_vars() {
        // OutlineOver(OutlineOver(Constant, ...), ...)
        let inner = NprColorNode::Constant(Vec3::ZERO).with_outline(Vec3::X, 0.3);
        let outer = inner.with_outline(Vec3::Y, 0.6);
        let snip = transpile_npr_color_node(&outer, ShaderLanguage::Glsl, ctx());
        assert!(snip.statements.contains("alice_col_0"));
        assert!(snip.statements.contains("alice_col_1"));
    }

    #[test]
    fn wgsl_mix_wraps_scalar_in_vec3() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 2,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        // WGSL mix(vec3, vec3, vec3) form
        assert!(snip.color_expression.contains("vec3<f32>(alice_toon_ramp"));
    }

    #[test]
    fn hlsl_uses_lerp_and_float3() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 2,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.starts_with("lerp(float3("));
    }

    #[test]
    fn canonical_context_uses_expected_names() {
        let c = NprShaderContext::canonical();
        assert_eq!(c.n_dot_l, "ndl");
        assert_eq!(c.n_dot_v, "ndv");
        assert_eq!(c.sdf, "d");
        assert_eq!(c.uv, "uv");
        assert_eq!(c.time, "iTime");
    }

    #[test]
    fn multiply_emits_product() {
        let node = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5))
            .multiply(NprColorNode::Constant(Vec3::new(0.4, 0.4, 0.4)));
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.statements.contains("alice_col_0"));
        assert!(snip.statements.contains("alice_col_1"));
        assert!(snip.color_expression.contains("alice_col_0 * alice_col_1"));
    }

    #[test]
    fn add_emits_sum() {
        let node = NprColorNode::Constant(Vec3::new(0.3, 0.1, 0.0))
            .plus(NprColorNode::Constant(Vec3::new(0.2, 0.4, 0.5)));
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.contains("alice_col_0 + alice_col_1"));
    }

    #[test]
    fn scale_emits_scalar_multiply() {
        let node = NprColorNode::Constant(Vec3::new(0.6, 0.6, 0.6)).scale(0.5);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.contains("* 0.500000"));
    }

    #[test]
    fn fresnel_emits_mix_with_fresnel_rim() {
        let base = NprColorNode::Constant(Vec3::ZERO);
        let node = base.with_fresnel(Vec3::ONE, 2.0);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip
            .color_expression
            .contains("alice_fresnel_rim(ndv, 2.0, 1.0)"));
        assert!(snip.color_expression.starts_with("mix("));
    }

    #[test]
    fn fresnel_in_wgsl_wraps_mask_as_vec3() {
        let base = NprColorNode::Constant(Vec3::ZERO);
        let node = base.with_fresnel(Vec3::ONE, 2.0);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        // WGSL mix needs vec3<f32>(scalar_mask)
        assert!(snip
            .color_expression
            .contains("vec3<f32>(alice_fresnel_rim(ndv"));
    }

    #[test]
    fn saturate_emits_alice_saturate() {
        let node = NprColorNode::Constant(Vec3::new(0.7, 0.4, 0.2)).saturate(0.5);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.contains("alice_saturate("));
        assert!(snip.color_expression.contains("0.5"));
    }

    #[test]
    fn bloom_emits_alice_bloom_toon() {
        let node = NprColorNode::Constant(Vec3::new(0.9, 0.9, 0.9)).bloom(0.5, 1.0);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert!(snip.color_expression.contains("alice_bloom_toon("));
        assert!(snip.color_expression.contains("0.500000, 1.0"));
    }

    #[test]
    fn posterize_emits_alice_posterize_color() {
        let node = NprColorNode::Constant(Vec3::new(0.25, 0.5, 0.75)).posterize(4);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.contains("alice_posterize_color("));
        assert!(snip.color_expression.contains("4.0"));
    }

    #[test]
    fn vignette_emits_alice_vignette_scalar_multiply() {
        let node = NprColorNode::Constant(Vec3::ONE).vignetted(0.3, 0.2);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.contains("alice_vignette(uv,"));
        assert!(snip.color_expression.contains("0.300000"));
        assert!(snip.color_expression.contains(" * alice_vignette"));
    }

    #[test]
    fn palette3_ndotl_emits_gradient_call() {
        use crate::npr::dsl::PaletteSource;
        let node = NprColorNode::Palette3 {
            source: PaletteSource::NDotL,
            c0: Vec3::new(1.0, 0.0, 0.0),
            c1: Vec3::new(0.0, 1.0, 0.0),
            c2: Vec3::new(0.0, 0.0, 1.0),
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip
            .color_expression
            .starts_with("alice_palette_gradient_3("));
        assert!(snip.color_expression.contains("clamp(ndl, 0.0, 1.0)"));
    }

    #[test]
    fn palette3_uv_y_emits_uv_dot_y_source() {
        use crate::npr::dsl::PaletteSource;
        let node = NprColorNode::Palette3 {
            source: PaletteSource::UvY,
            c0: Vec3::ZERO,
            c1: Vec3::ZERO,
            c2: Vec3::ONE,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert!(snip.color_expression.contains("clamp(uv.y, 0.0, 1.0)"));
    }

    #[test]
    fn hatch_emits_alice_hatch_lines_mask_glsl() {
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_hatch(0.4, 12.0, 0.1, Vec3::ZERO);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.starts_with("mix("));
        assert!(snip.color_expression.contains("alice_hatch_lines(uv,"));
        assert!(snip.color_expression.contains("12.0"));
    }

    #[test]
    fn hatch_wraps_mask_as_vec3_in_wgsl() {
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_hatch(0.4, 12.0, 0.1, Vec3::ZERO);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert!(snip
            .color_expression
            .contains("vec3<f32>(alice_hatch_lines"));
    }

    #[test]
    fn palette5_emits_gradient_call() {
        use crate::npr::dsl::PaletteSource;
        let node = NprColorNode::Palette5 {
            source: PaletteSource::UvY,
            c0: Vec3::ZERO,
            c1: Vec3::new(0.25, 0.25, 0.25),
            c2: Vec3::new(0.5, 0.5, 0.5),
            c3: Vec3::new(0.75, 0.75, 0.75),
            c4: Vec3::ONE,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip
            .color_expression
            .starts_with("alice_palette_gradient_5("));
        assert!(snip.color_expression.contains("clamp(uv.y, 0.0, 1.0)"));
    }

    #[test]
    fn tonemap_emits_reinhard_call() {
        let node = NprColorNode::Constant(Vec3::splat(5.0)).tonemap_reinhard(1.5);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert!(snip.color_expression.contains("alice_tonemap_reinhard("));
        assert!(snip.color_expression.contains("1.500000"));
    }

    #[test]
    fn speed_line_emits_alice_speed_line_glsl() {
        let node = NprColorNode::Constant(Vec3::ONE).with_speed_lines(
            glam::Vec2::new(0.5, 0.5),
            16,
            0.05,
            Vec3::ZERO,
        );
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(snip.color_expression.starts_with("mix("));
        assert!(snip.color_expression.contains("alice_speed_line(uv,"));
        assert!(snip.color_expression.contains("vec2(0.500000, 0.500000)"));
        assert!(snip.color_expression.contains("16.0"));
    }

    #[test]
    fn speed_line_uses_float2_in_hlsl() {
        let node = NprColorNode::Constant(Vec3::ONE).with_speed_lines(
            glam::Vec2::new(0.5, 0.5),
            8,
            0.05,
            Vec3::ZERO,
        );
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.contains("float2(0.500000, 0.500000)"));
        assert!(snip.color_expression.starts_with("lerp("));
    }

    #[test]
    fn hatch_uses_lerp_in_hlsl() {
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_hatch(0.4, 12.0, 0.1, Vec3::ZERO);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.starts_with("lerp("));
        assert!(snip.color_expression.contains("alice_hatch_lines(uv,"));
    }

    #[test]
    fn palette3_sdf_emits_abs_source() {
        use crate::npr::dsl::PaletteSource;
        let node = NprColorNode::Palette3 {
            source: PaletteSource::Sdf,
            c0: Vec3::ZERO,
            c1: Vec3::ZERO,
            c2: Vec3::ONE,
        };
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.contains("clamp(abs(d), 0.0, 1.0)"));
    }

    #[test]
    fn posterize_clamps_low_levels_in_shader() {
        let node = NprColorNode::Constant(Vec3::ZERO).posterize(1);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        // levels < 2 clamped to 2
        assert!(snip.color_expression.contains("2.0"));
    }

    #[test]
    fn fresnel_in_hlsl_uses_lerp() {
        let base = NprColorNode::Constant(Vec3::ZERO);
        let node = base.with_fresnel(Vec3::ONE, 2.0);
        let snip = transpile_npr_color_node(&node, ShaderLanguage::Hlsl, ctx());
        assert!(snip.color_expression.starts_with("lerp("));
        assert!(snip.color_expression.contains("alice_fresnel_rim(ndv"));
    }

    #[test]
    fn family_palette_unrolls_one_compare_per_family_wgsl() {
        let node = NprColorNode::FamilyPalette {
            families: vec![
                (Vec3::Y, Vec3::new(0.9, 0.8, 0.7)),
                (Vec3::X, Vec3::new(0.5, 0.3, 0.2)),
                (Vec3::Z, Vec3::new(0.2, 0.3, 0.5)),
            ],
        };
        let s = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        assert_eq!(s.color_expression, "alice_col_0");
        assert!(s
            .statements
            .contains("var alice_col_0: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);"));
        assert!(s
            .statements
            .contains("var alice_col_0_best: f32 = -3.402823466e38;"));
        assert_eq!(
            s.statements.matches("if (alice_col_0_d").count(),
            3,
            "{}",
            s.statements
        );
        assert!(s
            .statements
            .contains("let alice_col_0_d1 = (n.x * 1.0 + n.y * 0.0 + n.z * 0.0);"));
        assert!(s
            .statements
            .contains("alice_col_0 = vec3<f32>(0.200000, 0.300000, 0.500000);"));
    }

    #[test]
    fn region_noise_calls_the_crate_hash_and_noise_glsl() {
        let node = NprColorNode::RegionNoise {
            base: Box::new(NprColorNode::Constant(Vec3::new(0.5, 0.4, 0.3))),
            cell: 0.04,
            region_amp: Vec3::new(0.1, 0.05, 0.02),
            noise_amp: Vec3::splat(0.03),
            seed: 11,
        };
        let s = transpile_npr_color_node(&node, ShaderLanguage::Glsl, ctx());
        assert!(
            s.statements.contains("vec3 alice_col_0_q = p / 0.040000;"),
            "{}",
            s.statements
        );
        assert!(s
            .statements
            .contains("float alice_col_0_h = alice_hash3(floor(alice_col_0_q), 11u);"));
        assert!(s
            .statements
            .contains("float alice_col_0_v = hash_noise_3d(alice_col_0_q, 12u);"));
        assert_eq!(
            s.color_expression,
            "(alice_col_0 + vec3(0.100000, 0.050000, 0.020000) * (2.0 * alice_col_0_h - 1.0) + vec3(0.030000, 0.030000, 0.030000) * alice_col_0_v)"
        );
    }

    #[test]
    fn family_and_region_snippets_validate_in_naga_wgsl() {
        let node = NprColorNode::RegionNoise {
            base: Box::new(NprColorNode::FamilyPalette {
                families: vec![
                    (Vec3::Y, Vec3::new(0.9, 0.8, 0.7)),
                    (Vec3::X, Vec3::new(0.5, 0.3, 0.2)),
                ],
            }),
            cell: 0.04,
            region_amp: Vec3::new(0.1, 0.05, 0.02),
            noise_amp: Vec3::splat(0.03),
            seed: 11,
        };
        let s = transpile_npr_color_node(&node, ShaderLanguage::Wgsl, ctx());
        let source = format!(
            "{}\n@group(0) @binding(0) var<storage, read_write> out: array<vec3<f32>>;\n@compute @workgroup_size(1)\nfn main() {{\n    let ndl = 0.5;\n    let ndv = 0.5;\n    let d = 0.1;\n    let uv = vec2<f32>(0.5, 0.5);\n    let iTime = 0.0;\n    let n = vec3<f32>(0.0, 1.0, 0.0);\n    let p = vec3<f32>(0.01, 0.02, 0.03);\n{}    out[0] = {};\n}}\n",
            crate::modifiers::HASH_NOISE_WGSL, s.statements, s.color_expression
        );
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("the snippet validates");
    }
}

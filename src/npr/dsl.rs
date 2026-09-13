//! Color-pipeline DSL for composing NPR primitives
//!
//! `SdfNode` produces signed distances; NPR primitives produce colors.
//! The two pipelines are distinct, so this module introduces a small
//! `NprColorNode` tree that describes how NPR primitives compose into a
//! final `NprColor`. The tree is evaluated against an [`NprInput`] and
//! yields a color; it is deliberately kept small (no bytecode, no
//! transpile) so callers can experiment before committing to a fuller
//! integration with the `SdfNode` bytecode compiler.
//!
//! # Extension paths (future work)
//!
//! - Bytecode compilation into the existing `CompiledSdf` machinery
//! - Shader transpile via [`crate::npr::shader_glue`]
//! - A dedicated `SdfNode::WithColor(...)` variant that pairs a distance
//!   sub-tree with an `NprColorNode` at each surface point
//!
//! Author: Moroya Sakamoto

use crate::npr::composition::{bloom_toon, vignette};
use crate::npr::hatch::hatch_lines;
use crate::npr::motion::speed_line;
use crate::npr::outline::composite_outline;
use crate::npr::palette::palette_gradient;
use crate::npr::toon::{posterize_color, soft_toon_ramp, toon_ramp, two_tone};
use crate::npr::NprColor;
use glam::{Vec2, Vec3};

/// Inputs available to every color node during evaluation
#[derive(Debug, Clone, Copy)]
pub struct NprColorContext {
    /// Signed distance at the point being shaded
    pub sdf: f32,
    /// Surface normal (assumed unit)
    pub normal: Vec3,
    /// View direction from surface to camera (assumed unit)
    pub view: Vec3,
    /// Direction from surface to light (assumed unit)
    pub light: Vec3,
    /// UV coordinate in `[0, 1]` for the current pixel
    pub uv: Vec2,
    /// Elapsed time in seconds since the animation started
    pub time: f32,
}

impl NprColorContext {
    /// Diffuse `n . l`
    #[inline]
    #[must_use]
    pub fn n_dot_l(&self) -> f32 {
        self.normal.dot(self.light)
    }

    /// View-normal `n . v`
    #[inline]
    #[must_use]
    pub fn n_dot_v(&self) -> f32 {
        self.normal.dot(self.view)
    }
}

/// A composable expression tree that evaluates to an `NprColor`
///
/// Kept intentionally small; extend as new composition patterns become
/// load-bearing.
#[derive(Debug, Clone)]
pub enum NprColorNode {
    /// A constant color, ignoring context
    Constant(NprColor),
    /// Toon-ramp lookup between shadow and light using `n . l`
    Toon {
        /// Color returned when the toon ramp is 0
        shadow: NprColor,
        /// Color returned when the toon ramp is 1
        light: NprColor,
        /// Number of discrete bands
        bands: u32,
    },
    /// Soft toon ramp with smoothstep transitions
    SoftToon {
        /// Color returned when the ramp is 0
        shadow: NprColor,
        /// Color returned when the ramp is 1
        light: NprColor,
        /// Number of discrete bands
        bands: u32,
        /// Half-width of the smoothstep transition
        smoothness: f32,
    },
    /// Two-tone shading with a sharp cutoff at `threshold`
    TwoTone {
        /// Shadow color returned below the threshold
        shadow: NprColor,
        /// Light color returned above the threshold
        light: NprColor,
        /// Cutoff on `clamp(n . l, 0, 1)`
        threshold: f32,
    },
    /// Blend a base child with an outline color by an alpha mask
    OutlineOver {
        /// Base child evaluated for the interior color
        base: Box<NprColorNode>,
        /// Outline color to paint when the mask is high
        outline: NprColor,
        /// Precomputed outline mask (typically from `distance_field_outline_soft`)
        alpha: f32,
    },
    /// Component-wise multiply two colour subtrees
    Multiply {
        /// Left-hand subtree
        a: Box<NprColorNode>,
        /// Right-hand subtree
        b: Box<NprColorNode>,
    },
    /// Component-wise add two colour subtrees
    Add {
        /// Left-hand subtree
        a: Box<NprColorNode>,
        /// Right-hand subtree
        b: Box<NprColorNode>,
    },
    /// Uniformly scale a subtree by a scalar factor
    Scale {
        /// Subtree to scale
        child: Box<NprColorNode>,
        /// Scalar multiplier applied to every channel
        factor: f32,
    },
    /// Overlay an edge colour on top of a base subtree using a Fresnel mask
    Fresnel {
        /// Base child evaluated for the interior color
        base: Box<NprColorNode>,
        /// Edge colour blended in at grazing view angles
        edge: NprColor,
        /// Fresnel exponent (higher = tighter rim)
        power: f32,
    },
    /// Adjust saturation of a subtree by blending toward luminance grey
    ///
    /// `factor == 0` collapses the colour to greyscale, `factor == 1` keeps
    /// the original saturation, `factor > 1` over-saturates.
    Saturate {
        /// Subtree whose saturation is adjusted
        child: Box<NprColorNode>,
        /// Saturation blend factor
        factor: f32,
    },
    /// Toon-style bloom: pass a subtree's colour through only when its
    /// brightest channel exceeds `threshold`, then scale by `intensity`
    Bloom {
        /// Subtree evaluated for the source colour
        child: Box<NprColorNode>,
        /// Threshold on the maximum channel
        threshold: f32,
        /// Multiplier applied to the passing colour
        intensity: f32,
    },
    /// Posterise a subtree's colour into `levels` discrete steps per channel
    PosterizeColor {
        /// Subtree evaluated for the source colour
        child: Box<NprColorNode>,
        /// Number of discrete levels per channel (>= 2)
        levels: u32,
    },
    /// Multiply a subtree by a UV-centred vignette mask
    Vignette {
        /// Subtree evaluated for the source colour
        child: Box<NprColorNode>,
        /// Radius (in UV units) at which the mask starts to fall off
        radius: f32,
        /// Half-width of the falloff transition
        softness: f32,
    },
    /// Three-anchor palette gradient driven by a context scalar
    Palette3 {
        /// Which context scalar to use as the interpolation parameter
        source: PaletteSource,
        /// Colour at `t = 0`
        c0: NprColor,
        /// Colour at `t = 0.5`
        c1: NprColor,
        /// Colour at `t = 1`
        c2: NprColor,
    },
    /// Overlay hatch line ink on top of a base subtree using UV
    Hatch {
        /// Base child evaluated for the interior colour
        base: Box<NprColorNode>,
        /// Line direction in radians (0 = horizontal)
        angle_rad: f32,
        /// Lines per UV unit
        density: f32,
        /// Line half-width in normalised phase (`[0, 0.5]`)
        thickness: f32,
        /// Ink colour blended in where the mask is high
        ink: NprColor,
    },
    /// Five-anchor palette gradient driven by a context scalar
    Palette5 {
        /// Which context scalar to use as the interpolation parameter
        source: PaletteSource,
        /// Colour at `t = 0`
        c0: NprColor,
        /// Colour at `t = 0.25`
        c1: NprColor,
        /// Colour at `t = 0.5`
        c2: NprColor,
        /// Colour at `t = 0.75`
        c3: NprColor,
        /// Colour at `t = 1`
        c4: NprColor,
    },
    /// Reinhard tone-mapping applied to a subtree
    Tonemap {
        /// Subtree evaluated for the source colour
        child: Box<NprColorNode>,
        /// Exposure multiplier applied before tone-mapping
        exposure: f32,
    },
    /// Radial speed-line ink overlay from a focal UV
    SpeedLine {
        /// Base child evaluated for the interior colour
        base: Box<NprColorNode>,
        /// Focus UV (typical: `Vec2::new(0.5, 0.5)`)
        focus: Vec2,
        /// Number of radial lines around the full circle
        count: u32,
        /// Line half-width in normalised phase (`[0, 0.5]`)
        thickness: f32,
        /// Ink colour blended in where the mask is high
        ink: NprColor,
    },
}

/// Which context scalar drives a palette-style variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaletteSource {
    /// `clamp(n . l, 0, 1)`
    NDotL,
    /// `clamp(n . v, 0, 1)`
    NDotV,
    /// Absolute signed-distance value clamped to `[0, 1]`
    Sdf,
    /// UV-Y coordinate (`uv.y`)
    UvY,
    /// Repeating time cycle: `fract(time)`, useful for animated palettes
    TimeCycle,
}

impl NprColorNode {
    /// Evaluate the node against a shading context
    #[must_use]
    pub fn eval(&self, ctx: &NprColorContext) -> NprColor {
        match self {
            Self::Constant(c) => *c,
            Self::Toon {
                shadow,
                light,
                bands,
            } => {
                let t = toon_ramp(ctx.n_dot_l(), *bands);
                shadow.lerp(*light, t)
            }
            Self::SoftToon {
                shadow,
                light,
                bands,
                smoothness,
            } => {
                let t = soft_toon_ramp(ctx.n_dot_l(), *bands, *smoothness);
                shadow.lerp(*light, t)
            }
            Self::TwoTone {
                shadow,
                light,
                threshold,
            } => two_tone(ctx.n_dot_l(), *shadow, *light, *threshold),
            Self::OutlineOver {
                base,
                outline,
                alpha,
            } => {
                let base_color = base.eval(ctx);
                composite_outline(base_color, *outline, *alpha)
            }
            Self::Multiply { a, b } => a.eval(ctx) * b.eval(ctx),
            Self::Add { a, b } => a.eval(ctx) + b.eval(ctx),
            Self::Scale { child, factor } => child.eval(ctx) * *factor,
            Self::Fresnel { base, edge, power } => {
                let base_color = base.eval(ctx);
                let ndv = ctx.n_dot_v();
                let fresnel = (1.0 - ndv.clamp(0.0, 1.0)).max(0.0).powf(power.max(0.0));
                base_color.lerp(*edge, fresnel.clamp(0.0, 1.0))
            }
            Self::Saturate { child, factor } => {
                let color = child.eval(ctx);
                let lum = color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                Vec3::splat(lum).lerp(color, *factor)
            }
            Self::Bloom {
                child,
                threshold,
                intensity,
            } => bloom_toon(child.eval(ctx), *threshold, *intensity),
            Self::PosterizeColor { child, levels } => {
                posterize_color(child.eval(ctx), (*levels).max(2))
            }
            Self::Vignette {
                child,
                radius,
                softness,
            } => {
                let mask = vignette(ctx.uv.x, ctx.uv.y, *radius, *softness);
                child.eval(ctx) * mask
            }
            Self::Palette3 { source, c0, c1, c2 } => {
                let t = palette_source_scalar(*source, ctx).clamp(0.0, 1.0);
                let palette = [*c0, *c1, *c2];
                palette_gradient(t, &palette)
            }
            Self::Hatch {
                base,
                angle_rad,
                density,
                thickness,
                ink,
            } => {
                let base_color = base.eval(ctx);
                let mask = hatch_lines(ctx.uv.x, ctx.uv.y, *angle_rad, *density, *thickness);
                base_color.lerp(*ink, mask)
            }
            Self::Palette5 {
                source,
                c0,
                c1,
                c2,
                c3,
                c4,
            } => {
                let t = palette_source_scalar(*source, ctx).clamp(0.0, 1.0);
                let palette = [*c0, *c1, *c2, *c3, *c4];
                palette_gradient(t, &palette)
            }
            Self::Tonemap { child, exposure } => {
                let color = child.eval(ctx);
                let scaled = color * exposure.max(0.0);
                scaled / (Vec3::ONE + scaled)
            }
            Self::SpeedLine {
                base,
                focus,
                count,
                thickness,
                ink,
            } => {
                let base_color = base.eval(ctx);
                let mask = speed_line(ctx.uv.x, ctx.uv.y, focus.x, focus.y, *count, *thickness);
                base_color.lerp(*ink, mask)
            }
        }
    }

    /// Builder helper: wrap `self` in an outline pass
    #[must_use]
    pub fn with_outline(self, outline: NprColor, alpha: f32) -> Self {
        Self::OutlineOver {
            base: Box::new(self),
            outline,
            alpha,
        }
    }

    /// Builder helper: component-wise multiply `self` by another subtree
    #[must_use]
    pub fn multiply(self, other: NprColorNode) -> Self {
        Self::Multiply {
            a: Box::new(self),
            b: Box::new(other),
        }
    }

    /// Builder helper: component-wise add another subtree to `self`
    #[must_use]
    pub fn plus(self, other: NprColorNode) -> Self {
        Self::Add {
            a: Box::new(self),
            b: Box::new(other),
        }
    }

    /// Builder helper: uniformly scale `self` by `factor`
    #[must_use]
    pub fn scale(self, factor: f32) -> Self {
        Self::Scale {
            child: Box::new(self),
            factor,
        }
    }

    /// Builder helper: overlay a Fresnel edge colour on `self`
    #[must_use]
    pub fn with_fresnel(self, edge: NprColor, power: f32) -> Self {
        Self::Fresnel {
            base: Box::new(self),
            edge,
            power,
        }
    }

    /// Builder helper: adjust saturation by blending toward luminance grey
    #[must_use]
    pub fn saturate(self, factor: f32) -> Self {
        Self::Saturate {
            child: Box::new(self),
            factor,
        }
    }

    /// Builder helper: pass through only bright pixels for a bloom pass
    #[must_use]
    pub fn bloom(self, threshold: f32, intensity: f32) -> Self {
        Self::Bloom {
            child: Box::new(self),
            threshold,
            intensity,
        }
    }

    /// Builder helper: posterise into `levels` discrete steps per channel
    #[must_use]
    pub fn posterize(self, levels: u32) -> Self {
        Self::PosterizeColor {
            child: Box::new(self),
            levels: levels.max(2),
        }
    }

    /// Builder helper: multiply by a UV vignette mask
    #[must_use]
    pub fn vignetted(self, radius: f32, softness: f32) -> Self {
        Self::Vignette {
            child: Box::new(self),
            radius,
            softness,
        }
    }

    /// Builder helper: overlay hatch lines using UV
    #[must_use]
    pub fn with_hatch(self, angle_rad: f32, density: f32, thickness: f32, ink: NprColor) -> Self {
        Self::Hatch {
            base: Box::new(self),
            angle_rad,
            density,
            thickness,
            ink,
        }
    }

    /// Builder helper: apply Reinhard tone-mapping
    #[must_use]
    pub fn tonemap_reinhard(self, exposure: f32) -> Self {
        Self::Tonemap {
            child: Box::new(self),
            exposure,
        }
    }

    /// Builder helper: overlay radial speed lines from a focal UV
    #[must_use]
    pub fn with_speed_lines(self, focus: Vec2, count: u32, thickness: f32, ink: NprColor) -> Self {
        Self::SpeedLine {
            base: Box::new(self),
            focus,
            count,
            thickness,
            ink,
        }
    }
}

/// Compute the scalar interpolation parameter for a `PaletteSource`
#[must_use]
pub(crate) fn palette_source_scalar(source: PaletteSource, ctx: &NprColorContext) -> f32 {
    match source {
        PaletteSource::NDotL => ctx.n_dot_l().clamp(0.0, 1.0),
        PaletteSource::NDotV => ctx.n_dot_v().clamp(0.0, 1.0),
        PaletteSource::Sdf => ctx.sdf.abs().clamp(0.0, 1.0),
        PaletteSource::UvY => ctx.uv.y.clamp(0.0, 1.0),
        PaletteSource::TimeCycle => ctx.time - ctx.time.floor(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lit_context() -> NprColorContext {
        NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 0.5),
            time: 0.0,
        }
    }

    fn dark_context() -> NprColorContext {
        NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, -1.0, 0.0),
            uv: Vec2::new(0.5, 0.5),
            time: 0.0,
        }
    }

    #[test]
    fn constant_returns_stored_color() {
        let node = NprColorNode::Constant(Vec3::new(0.3, 0.6, 0.9));
        let c = node.eval(&lit_context());
        assert_eq!(c, Vec3::new(0.3, 0.6, 0.9));
    }

    #[test]
    fn toon_lit_side_returns_light() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 3,
        };
        let c = node.eval(&lit_context());
        assert!((c - Vec3::ONE).length() < 1e-3);
    }

    #[test]
    fn toon_dark_side_returns_shadow() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 3,
        };
        let c = node.eval(&dark_context());
        assert!((c - Vec3::ZERO).length() < 1e-3);
    }

    #[test]
    fn two_tone_dispatches_by_threshold() {
        let node = NprColorNode::TwoTone {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            threshold: 0.5,
        };
        assert_eq!(node.eval(&lit_context()), Vec3::ONE);
        assert_eq!(node.eval(&dark_context()), Vec3::ZERO);
    }

    #[test]
    fn outline_over_full_alpha_returns_outline() {
        let base = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let node = base.with_outline(Vec3::ZERO, 1.0);
        let c = node.eval(&lit_context());
        assert_eq!(c, Vec3::ZERO);
    }

    #[test]
    fn outline_over_zero_alpha_returns_base() {
        let base = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let node = base.with_outline(Vec3::ONE, 0.0);
        let c = node.eval(&lit_context());
        assert_eq!(c, Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn multiply_combines_children() {
        let a = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let b = NprColorNode::Constant(Vec3::new(0.4, 0.4, 0.4));
        let c = a.multiply(b).eval(&lit_context());
        assert!((c - Vec3::splat(0.2)).length() < 1e-4);
    }

    #[test]
    fn add_sums_children() {
        let a = NprColorNode::Constant(Vec3::new(0.3, 0.1, 0.0));
        let b = NprColorNode::Constant(Vec3::new(0.2, 0.4, 0.5));
        let c = a.plus(b).eval(&lit_context());
        assert!((c - Vec3::new(0.5, 0.5, 0.5)).length() < 1e-4);
    }

    #[test]
    fn scale_multiplies_by_factor() {
        let node = NprColorNode::Constant(Vec3::new(0.4, 0.4, 0.4)).scale(0.5);
        let c = node.eval(&lit_context());
        assert!((c - Vec3::splat(0.2)).length() < 1e-4);
    }

    #[test]
    fn fresnel_head_on_returns_base() {
        // Normal towards view: n.v = 1, fresnel = 0, returns base
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 0.0, 1.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 0.5),
            time: 0.0,
        };
        let base = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let node = base.with_fresnel(Vec3::ZERO, 2.0);
        let c = node.eval(&ctx);
        assert!((c - Vec3::splat(0.5)).length() < 1e-4);
    }

    #[test]
    fn fresnel_grazing_returns_edge() {
        // Normal perpendicular to view: n.v = 0, fresnel = 1, returns edge
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(1.0, 0.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 0.5),
            time: 0.0,
        };
        let base = NprColorNode::Constant(Vec3::ZERO);
        let node = base.with_fresnel(Vec3::ONE, 2.0);
        let c = node.eval(&ctx);
        assert!((c - Vec3::ONE).length() < 1e-4);
    }

    #[test]
    fn saturate_zero_returns_grey() {
        let color = Vec3::new(1.0, 0.0, 0.0);
        let node = NprColorNode::Constant(color).saturate(0.0);
        let c = node.eval(&lit_context());
        // Red channel weight is 0.2126, so grey = 0.2126
        let expected = Vec3::splat(0.2126);
        assert!(
            (c - expected).length() < 1e-4,
            "expected grey ~ {expected:?}, got {c:?}"
        );
    }

    #[test]
    fn saturate_one_returns_original() {
        let color = Vec3::new(0.7, 0.4, 0.2);
        let node = NprColorNode::Constant(color).saturate(1.0);
        let c = node.eval(&lit_context());
        assert!((c - color).length() < 1e-4);
    }

    #[test]
    fn bloom_passes_bright_colors() {
        let color = Vec3::new(0.9, 0.9, 0.9);
        let node = NprColorNode::Constant(color).bloom(0.5, 0.5);
        let c = node.eval(&lit_context());
        assert!((c - color * 0.5).length() < 1e-4);
    }

    #[test]
    fn bloom_rejects_dim_colors() {
        let color = Vec3::new(0.3, 0.3, 0.3);
        let node = NprColorNode::Constant(color).bloom(0.5, 1.0);
        let c = node.eval(&lit_context());
        assert!((c - Vec3::ZERO).length() < 1e-4);
    }

    #[test]
    fn posterize_reduces_levels() {
        let node = NprColorNode::Constant(Vec3::new(0.25, 0.5, 0.75)).posterize(2);
        let c = node.eval(&lit_context());
        // 2-level posterize -> binary per channel
        for ch in [c.x, c.y, c.z] {
            assert!(ch == 0.0 || ch == 1.0, "expected binary, got {ch}");
        }
    }

    #[test]
    fn vignette_at_center_returns_full_child() {
        let child = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5));
        let node = child.vignetted(0.5, 0.3);
        let c = node.eval(&lit_context()); // uv = (0.5, 0.5)
        assert!((c - Vec3::splat(0.5)).length() < 1e-4);
    }

    #[test]
    fn vignette_at_corner_returns_dark() {
        let child = NprColorNode::Constant(Vec3::ONE);
        let node = child.vignetted(0.3, 0.2);
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.0, 0.0),
            time: 0.0,
        };
        let c = node.eval(&ctx);
        assert!((c - Vec3::ZERO).length() < 1e-4);
    }

    #[test]
    fn palette3_ndotl_ends_pick_endpoints() {
        let node = NprColorNode::Palette3 {
            source: PaletteSource::NDotL,
            c0: Vec3::new(1.0, 0.0, 0.0),
            c1: Vec3::new(0.0, 1.0, 0.0),
            c2: Vec3::new(0.0, 0.0, 1.0),
        };
        // Fully lit (n.l = 1) -> c2
        assert!((node.eval(&lit_context()) - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-3);
        // Fully dark (n.l = -1 clamped to 0) -> c0
        assert!((node.eval(&dark_context()) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn hatch_off_line_returns_base() {
        // At angle=0 the projected coord is uv_y; density=4 puts lines at
        // multiples of 0.25 (phase boundaries). uv_y=0.375 sits mid-cell.
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_hatch(0.0, 4.0, 0.02, Vec3::ZERO);
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 0.375),
            time: 0.0,
        };
        let c = node.eval(&ctx);
        assert!((c - Vec3::ONE).length() < 1e-4);
    }

    #[test]
    fn palette5_endpoints_pick_c0_c4() {
        let node = NprColorNode::Palette5 {
            source: PaletteSource::NDotL,
            c0: Vec3::new(1.0, 0.0, 0.0),
            c1: Vec3::new(0.0, 1.0, 0.0),
            c2: Vec3::new(0.0, 0.0, 1.0),
            c3: Vec3::new(1.0, 1.0, 0.0),
            c4: Vec3::new(1.0, 0.0, 1.0),
        };
        // Fully lit (n.l = 1) picks c4
        assert!((node.eval(&lit_context()) - Vec3::new(1.0, 0.0, 1.0)).length() < 1e-3);
        // Fully dark (n.l clamped to 0) picks c0
        assert!((node.eval(&dark_context()) - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn tonemap_reinhard_maps_hdr_into_unit() {
        let child = NprColorNode::Constant(Vec3::splat(10.0));
        let node = child.tonemap_reinhard(1.0);
        let c = node.eval(&lit_context());
        for ch in [c.x, c.y, c.z] {
            assert!(ch < 1.0 && ch > 0.9, "expected close-to-1, got {ch}");
        }
    }

    #[test]
    fn tonemap_reinhard_black_stays_black() {
        let child = NprColorNode::Constant(Vec3::ZERO);
        let node = child.tonemap_reinhard(2.0);
        let c = node.eval(&lit_context());
        assert!((c - Vec3::ZERO).length() < 1e-6);
    }

    #[test]
    fn speed_line_at_focus_returns_base() {
        // uv == focus -> mask 0 -> returns base
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_speed_lines(Vec2::new(0.5, 0.5), 12, 0.05, Vec3::ZERO);
        let c = node.eval(&lit_context()); // uv = (0.5, 0.5)
        assert!((c - Vec3::ONE).length() < 1e-4);
    }

    #[test]
    fn hatch_on_line_returns_ink() {
        // uv_y=0.5 with density=4 gives phase=0, dist=0.5, thick=0.2 -> mask=1
        let base = NprColorNode::Constant(Vec3::ONE);
        let node = base.with_hatch(0.0, 4.0, 0.2, Vec3::ZERO);
        let c = node.eval(&lit_context()); // uv = (0.5, 0.5)
        assert!((c - Vec3::ZERO).length() < 1e-4);
    }

    #[test]
    fn palette3_uv_y_selects_by_uv() {
        let node = NprColorNode::Palette3 {
            source: PaletteSource::UvY,
            c0: Vec3::new(1.0, 0.0, 0.0),
            c1: Vec3::new(0.0, 1.0, 0.0),
            c2: Vec3::new(0.0, 0.0, 1.0),
        };
        let ctx_top = NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 1.0),
            time: 0.0,
        };
        assert!((node.eval(&ctx_top) - Vec3::new(0.0, 0.0, 1.0)).length() < 1e-3);
    }

    #[test]
    fn posterize_clamps_low_levels() {
        // levels < 2 clamped to 2, so 1 should behave same as 2
        let a = NprColorNode::Constant(Vec3::new(0.6, 0.6, 0.6)).posterize(1);
        let b = NprColorNode::Constant(Vec3::new(0.6, 0.6, 0.6)).posterize(2);
        assert!((a.eval(&lit_context()) - b.eval(&lit_context())).length() < 1e-4);
    }

    #[test]
    fn soft_toon_bounded() {
        let node = NprColorNode::SoftToon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 4,
            smoothness: 0.05,
        };
        for i in 0..12 {
            let a = i as f32 * std::f32::consts::TAU / 12.0;
            let ctx = NprColorContext {
                sdf: 0.0,
                normal: Vec3::new(a.cos(), a.sin(), 0.0),
                view: Vec3::new(0.0, 0.0, 1.0),
                light: Vec3::new(0.0, 1.0, 0.0),
                uv: Vec2::new(0.5, 0.5),
                time: 0.0,
            };
            let c = node.eval(&ctx);
            for ch in [c.x, c.y, c.z] {
                assert!((0.0..=1.0).contains(&ch), "out of range at a={a}: {ch}");
            }
        }
    }
}

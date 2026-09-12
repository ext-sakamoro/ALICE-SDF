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

use crate::npr::outline::composite_outline;
use crate::npr::toon::{soft_toon_ramp, toon_ramp, two_tone};
use crate::npr::NprColor;
use glam::Vec3;

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
        }
    }

    fn dark_context() -> NprColorContext {
        NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, -1.0, 0.0),
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
            };
            let c = node.eval(&ctx);
            for ch in [c.x, c.y, c.z] {
                assert!((0.0..=1.0).contains(&ch), "out of range at a={a}: {ch}");
            }
        }
    }
}

//! Bytecode compilation of `NprColorNode` trees for host-side evaluation
//!
//! This module compiles an [`NprColorNode`] expression tree into a flat
//! opcode stream ([`CompiledColorPipeline`]) that a small stack machine
//! evaluates. The compiled form is a stepping stone toward full integration
//! with the existing `SdfNode` bytecode compiler in `src/compiled/` and
//! the future SIMD / GPU offload paths.
//!
//! # Coverage
//!
//! As of Phase 12-D all 17 `NprColorNode` variants have a native opcode.
//! The [`ColorOp::Fallback`] path is preserved for forward compatibility:
//! any future variant added to [`NprColorNode`] compiles down to a
//! `Fallback` opcode until an explicit native encoding is added, so the
//! pipeline never blocks new DSL variants.
//!
//! # Current performance
//!
//! On single-point scalar evaluation the bytecode evaluator is still
//! **slower** than the recursive tree walker for shallow trees, because
//! opcode `Vec` allocation and per-instruction stack push/pop dominate
//! the tiny math they replace. The bytecode form starts to pay off once
//! integrated with:
//!
//! - **SIMD batch evaluation** — one bytecode fetch amortised across
//!   a `Vec3x8` lane group
//! - **GPU offload** — bytecode serialised into a uniform / storage
//!   buffer and executed on the shader side
//! - **Deep composition trees** — where per-node function-call cost
//!   in the tree walker starts to dominate
//!
//! Callers that only need scalar host-side evaluation of a shallow tree
//! should stay on [`NprColorNode::eval`]. This module ships now so the
//! pattern is locked in ahead of the SIMD / GPU work.
//!
//! # Integration with `compiled::CompiledSdf`
//!
//! The host-side bytecode here is intentionally decoupled from the
//! `SdfNode` bytecode compiler in `src/compiled/`. Bringing colour
//! opcodes into that machine would require adding colour variants to
//! the `Instruction` / `Opcode` types and extending the SIMD /
//! stack-based interpreters. That work is deferred to a follow-up
//! phase; the DSL exposed here (`compile` / `eval`) is the seam that
//! the future integration will preserve.
//!
//! Author: Moroya Sakamoto

use crate::npr::composition::{bloom_toon, vignette};
use crate::npr::dsl::{palette_source_scalar, NprColorContext, NprColorNode, PaletteSource};
use crate::npr::hatch::hatch_lines;
use crate::npr::motion::speed_line;
use crate::npr::outline::composite_outline;
use crate::npr::palette::palette_gradient;
use crate::npr::toon::{posterize_color, soft_toon_ramp, toon_ramp, two_tone};
use glam::{Vec2, Vec3};

/// A single instruction in the compiled colour pipeline
///
/// Opcodes are grouped by arity:
///
/// - **0-input, 1-output**: [`PushConstant`](Self::PushConstant),
///   [`Palette3`](Self::Palette3), [`Palette5`](Self::Palette5)
/// - **1-input, 1-output**: [`Scale`](Self::Scale),
///   [`Saturate`](Self::Saturate), [`Bloom`](Self::Bloom),
///   [`PosterizeColor`](Self::PosterizeColor),
///   [`Vignette`](Self::Vignette), [`Tonemap`](Self::Tonemap),
///   [`OutlineOver`](Self::OutlineOver), [`Fresnel`](Self::Fresnel),
///   [`Hatch`](Self::Hatch), [`SpeedLine`](Self::SpeedLine)
/// - **2-input, 1-output**: [`Toon`](Self::Toon),
///   [`SoftToon`](Self::SoftToon), [`TwoTone`](Self::TwoTone),
///   [`Multiply`](Self::Multiply), [`Add`](Self::Add)
/// - **1-input, 1-output (delegating)**: [`Fallback`](Self::Fallback)
#[derive(Debug, Clone)]
pub enum ColorOp {
    /// Push a constant colour onto the stack (0-in, 1-out)
    PushConstant(Vec3),
    /// Toon-ramp lookup (2-in, 1-out): pop `shadow`, `light`; push
    /// `mix(shadow, light, toon_ramp(n_dot_l, bands))`
    Toon {
        /// Number of discrete bands
        bands: u32,
    },
    /// Soft toon ramp with smoothstep transitions (2-in, 1-out)
    SoftToon {
        /// Number of discrete bands
        bands: u32,
        /// Smoothstep half-width per boundary
        smoothness: f32,
    },
    /// Two-tone shading (2-in, 1-out): pop `shadow`, `light`; push
    /// `two_tone(n_dot_l, shadow, light, threshold)`
    TwoTone {
        /// Cutoff on `clamp(n_dot_l, 0, 1)`
        threshold: f32,
    },
    /// Componentwise multiply (2-in, 1-out)
    Multiply,
    /// Componentwise add (2-in, 1-out)
    Add,
    /// Scale the top-of-stack colour by a scalar (1-in, 1-out)
    Scale {
        /// Scalar multiplier applied to every channel
        factor: f32,
    },
    /// Composite an outline colour over the top-of-stack base (1-in, 1-out)
    OutlineOver {
        /// Outline colour blended in where the mask is high
        outline: Vec3,
        /// Precomputed outline alpha mask
        alpha: f32,
    },
    /// Blend edge colour into the top-of-stack base using a Fresnel mask (1-in, 1-out)
    Fresnel {
        /// Edge colour blended in at grazing view angles
        edge: Vec3,
        /// Fresnel exponent (higher = tighter rim)
        power: f32,
    },
    /// Adjust saturation by blending toward luminance grey (1-in, 1-out)
    Saturate {
        /// `0` = greyscale, `1` = identity, `>1` = over-saturate
        factor: f32,
    },
    /// Toon-style bloom filter (1-in, 1-out)
    Bloom {
        /// Threshold on the maximum channel
        threshold: f32,
        /// Multiplier applied to the passing colour
        intensity: f32,
    },
    /// Posterise into `levels` discrete steps per channel (1-in, 1-out)
    PosterizeColor {
        /// Number of discrete levels per channel (clamped to `>= 2`)
        levels: u32,
    },
    /// Multiply the top-of-stack colour by a UV-centred vignette mask (1-in, 1-out)
    Vignette {
        /// Radius (in UV units) at which the mask starts to fall off
        radius: f32,
        /// Half-width of the falloff transition
        softness: f32,
    },
    /// Three-anchor palette gradient driven by a context scalar (0-in, 1-out)
    Palette3 {
        /// Which context scalar drives the interpolation parameter
        source: PaletteSource,
        /// Colour at `t = 0`
        c0: Vec3,
        /// Colour at `t = 0.5`
        c1: Vec3,
        /// Colour at `t = 1`
        c2: Vec3,
    },
    /// Five-anchor palette gradient driven by a context scalar (0-in, 1-out)
    Palette5 {
        /// Which context scalar drives the interpolation parameter
        source: PaletteSource,
        /// Colour at `t = 0`
        c0: Vec3,
        /// Colour at `t = 0.25`
        c1: Vec3,
        /// Colour at `t = 0.5`
        c2: Vec3,
        /// Colour at `t = 0.75`
        c3: Vec3,
        /// Colour at `t = 1`
        c4: Vec3,
    },
    /// Overlay hatch line ink on top of the top-of-stack base using UV (1-in, 1-out)
    Hatch {
        /// Line direction in radians (0 = horizontal)
        angle_rad: f32,
        /// Lines per UV unit
        density: f32,
        /// Line half-width in normalised phase (`[0, 0.5]`)
        thickness: f32,
        /// Ink colour blended in where the mask is high
        ink: Vec3,
    },
    /// Reinhard tone-mapping applied to the top-of-stack colour (1-in, 1-out)
    Tonemap {
        /// Exposure multiplier applied before tone-mapping
        exposure: f32,
    },
    /// Radial speed-line ink overlay from a focal UV (1-in, 1-out)
    SpeedLine {
        /// Focus UV (typical: `Vec2::new(0.5, 0.5)`)
        focus: Vec2,
        /// Number of radial lines around the full circle
        count: u32,
        /// Line half-width in normalised phase (`[0, 0.5]`)
        thickness: f32,
        /// Ink colour blended in where the mask is high
        ink: Vec3,
    },
    /// Forward-compat fallback (1-out, delegating): evaluate the enclosed
    /// tree via the recursive walker
    ///
    /// Reserved for future `NprColorNode` variants that have not yet
    /// received a native opcode. As of Phase 12-D all existing variants
    /// compile to native opcodes, so a well-formed pipeline produced by
    /// [`CompiledColorPipeline::compile`] on today's DSL surface contains
    /// no `Fallback`. New DSL variants will emit `Fallback` transparently
    /// until an explicit native encoding is added.
    Fallback(Box<NprColorNode>),
}

/// A compiled colour pipeline ready for repeated evaluation
#[derive(Debug, Clone, Default)]
pub struct CompiledColorPipeline {
    /// Instructions executed in order
    pub ops: Vec<ColorOp>,
}

impl CompiledColorPipeline {
    /// Compile an [`NprColorNode`] tree into a bytecode pipeline
    #[must_use]
    pub fn compile(node: &NprColorNode) -> Self {
        let mut ops = Vec::new();
        Self::compile_node(node, &mut ops);
        Self { ops }
    }

    /// Execute the pipeline against a shading context
    ///
    /// # Panics
    /// Panics if the pipeline is malformed (empty result stack or an
    /// opcode's stack pop underflows). A well-formed pipeline produced
    /// by [`Self::compile`] never panics.
    #[must_use]
    pub fn eval(&self, ctx: &NprColorContext) -> Vec3 {
        let mut stack: Vec<Vec3> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            match op {
                ColorOp::PushConstant(c) => stack.push(*c),
                ColorOp::Toon { bands } => {
                    let light = stack.pop().expect("Toon: missing light on stack");
                    let shadow = stack.pop().expect("Toon: missing shadow on stack");
                    let t = toon_ramp(ctx.n_dot_l(), *bands);
                    stack.push(shadow.lerp(light, t));
                }
                ColorOp::SoftToon { bands, smoothness } => {
                    let light = stack.pop().expect("SoftToon: missing light on stack");
                    let shadow = stack.pop().expect("SoftToon: missing shadow on stack");
                    let t = soft_toon_ramp(ctx.n_dot_l(), *bands, *smoothness);
                    stack.push(shadow.lerp(light, t));
                }
                ColorOp::TwoTone { threshold } => {
                    let light = stack.pop().expect("TwoTone: missing light on stack");
                    let shadow = stack.pop().expect("TwoTone: missing shadow on stack");
                    stack.push(two_tone(ctx.n_dot_l(), shadow, light, *threshold));
                }
                ColorOp::Multiply => {
                    let b = stack.pop().expect("Multiply: missing rhs on stack");
                    let a = stack.pop().expect("Multiply: missing lhs on stack");
                    stack.push(a * b);
                }
                ColorOp::Add => {
                    let b = stack.pop().expect("Add: missing rhs on stack");
                    let a = stack.pop().expect("Add: missing lhs on stack");
                    stack.push(a + b);
                }
                ColorOp::Scale { factor } => {
                    let top = stack.pop().expect("Scale: empty stack");
                    stack.push(top * *factor);
                }
                ColorOp::OutlineOver { outline, alpha } => {
                    let base = stack.pop().expect("OutlineOver: missing base");
                    stack.push(composite_outline(base, *outline, *alpha));
                }
                ColorOp::Fresnel { edge, power } => {
                    let base = stack.pop().expect("Fresnel: missing base");
                    let ndv = ctx.n_dot_v();
                    let fresnel = (1.0 - ndv.clamp(0.0, 1.0)).max(0.0).powf(power.max(0.0));
                    stack.push(base.lerp(*edge, fresnel.clamp(0.0, 1.0)));
                }
                ColorOp::Saturate { factor } => {
                    let color = stack.pop().expect("Saturate: empty stack");
                    let lum = color.dot(Vec3::new(0.2126, 0.7152, 0.0722));
                    stack.push(Vec3::splat(lum).lerp(color, *factor));
                }
                ColorOp::Bloom {
                    threshold,
                    intensity,
                } => {
                    let color = stack.pop().expect("Bloom: empty stack");
                    stack.push(bloom_toon(color, *threshold, *intensity));
                }
                ColorOp::PosterizeColor { levels } => {
                    let color = stack.pop().expect("PosterizeColor: empty stack");
                    stack.push(posterize_color(color, (*levels).max(2)));
                }
                ColorOp::Vignette { radius, softness } => {
                    let color = stack.pop().expect("Vignette: empty stack");
                    let mask = vignette(ctx.uv.x, ctx.uv.y, *radius, *softness);
                    stack.push(color * mask);
                }
                ColorOp::Palette3 { source, c0, c1, c2 } => {
                    let t = palette_source_scalar(*source, ctx).clamp(0.0, 1.0);
                    let palette = [*c0, *c1, *c2];
                    stack.push(palette_gradient(t, &palette));
                }
                ColorOp::Palette5 {
                    source,
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                } => {
                    let t = palette_source_scalar(*source, ctx).clamp(0.0, 1.0);
                    let palette = [*c0, *c1, *c2, *c3, *c4];
                    stack.push(palette_gradient(t, &palette));
                }
                ColorOp::Hatch {
                    angle_rad,
                    density,
                    thickness,
                    ink,
                } => {
                    let base = stack.pop().expect("Hatch: missing base");
                    let mask = hatch_lines(ctx.uv.x, ctx.uv.y, *angle_rad, *density, *thickness);
                    stack.push(base.lerp(*ink, mask));
                }
                ColorOp::Tonemap { exposure } => {
                    let color = stack.pop().expect("Tonemap: empty stack");
                    let scaled = color * exposure.max(0.0);
                    stack.push(scaled / (Vec3::ONE + scaled));
                }
                ColorOp::SpeedLine {
                    focus,
                    count,
                    thickness,
                    ink,
                } => {
                    let base = stack.pop().expect("SpeedLine: missing base");
                    let mask = speed_line(ctx.uv.x, ctx.uv.y, focus.x, focus.y, *count, *thickness);
                    stack.push(base.lerp(*ink, mask));
                }
                ColorOp::Fallback(node) => {
                    // Reserved forward-compat path for future DSL variants
                    stack.push(node.eval(ctx));
                }
            }
        }
        stack.pop().expect("pipeline produced no result")
    }

    fn compile_node(node: &NprColorNode, ops: &mut Vec<ColorOp>) {
        match node {
            NprColorNode::Constant(c) => ops.push(ColorOp::PushConstant(*c)),
            NprColorNode::Toon {
                shadow,
                light,
                bands,
            } => {
                ops.push(ColorOp::PushConstant(*shadow));
                ops.push(ColorOp::PushConstant(*light));
                ops.push(ColorOp::Toon { bands: *bands });
            }
            NprColorNode::SoftToon {
                shadow,
                light,
                bands,
                smoothness,
            } => {
                ops.push(ColorOp::PushConstant(*shadow));
                ops.push(ColorOp::PushConstant(*light));
                ops.push(ColorOp::SoftToon {
                    bands: *bands,
                    smoothness: *smoothness,
                });
            }
            NprColorNode::TwoTone {
                shadow,
                light,
                threshold,
            } => {
                ops.push(ColorOp::PushConstant(*shadow));
                ops.push(ColorOp::PushConstant(*light));
                ops.push(ColorOp::TwoTone {
                    threshold: *threshold,
                });
            }
            NprColorNode::OutlineOver {
                base,
                outline,
                alpha,
            } => {
                Self::compile_node(base, ops);
                ops.push(ColorOp::OutlineOver {
                    outline: *outline,
                    alpha: *alpha,
                });
            }
            NprColorNode::Multiply { a, b } => {
                Self::compile_node(a, ops);
                Self::compile_node(b, ops);
                ops.push(ColorOp::Multiply);
            }
            NprColorNode::Add { a, b } => {
                Self::compile_node(a, ops);
                Self::compile_node(b, ops);
                ops.push(ColorOp::Add);
            }
            NprColorNode::Scale { child, factor } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Scale { factor: *factor });
            }
            NprColorNode::Fresnel { base, edge, power } => {
                Self::compile_node(base, ops);
                ops.push(ColorOp::Fresnel {
                    edge: *edge,
                    power: *power,
                });
            }
            NprColorNode::Saturate { child, factor } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Saturate { factor: *factor });
            }
            NprColorNode::Bloom {
                child,
                threshold,
                intensity,
            } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Bloom {
                    threshold: *threshold,
                    intensity: *intensity,
                });
            }
            NprColorNode::PosterizeColor { child, levels } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::PosterizeColor { levels: *levels });
            }
            NprColorNode::Vignette {
                child,
                radius,
                softness,
            } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Vignette {
                    radius: *radius,
                    softness: *softness,
                });
            }
            NprColorNode::Palette3 { source, c0, c1, c2 } => {
                ops.push(ColorOp::Palette3 {
                    source: *source,
                    c0: *c0,
                    c1: *c1,
                    c2: *c2,
                });
            }
            NprColorNode::Palette5 {
                source,
                c0,
                c1,
                c2,
                c3,
                c4,
            } => {
                ops.push(ColorOp::Palette5 {
                    source: *source,
                    c0: *c0,
                    c1: *c1,
                    c2: *c2,
                    c3: *c3,
                    c4: *c4,
                });
            }
            NprColorNode::Hatch {
                base,
                angle_rad,
                density,
                thickness,
                ink,
            } => {
                Self::compile_node(base, ops);
                ops.push(ColorOp::Hatch {
                    angle_rad: *angle_rad,
                    density: *density,
                    thickness: *thickness,
                    ink: *ink,
                });
            }
            NprColorNode::Tonemap { child, exposure } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Tonemap {
                    exposure: *exposure,
                });
            }
            NprColorNode::SpeedLine {
                base,
                focus,
                count,
                thickness,
                ink,
            } => {
                Self::compile_node(base, ops);
                ops.push(ColorOp::SpeedLine {
                    focus: *focus,
                    count: *count,
                    thickness: *thickness,
                    ink: *ink,
                });
            }
        }
    }

    /// Number of native (non-fallback) opcodes in the pipeline
    #[must_use]
    pub fn native_op_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| !matches!(op, ColorOp::Fallback(_)))
            .count()
    }

    /// Number of fallback opcodes in the pipeline
    ///
    /// After Phase 12-D this is always zero for pipelines compiled from
    /// current DSL variants; it becomes non-zero only when future
    /// `NprColorNode` variants land without an accompanying native opcode.
    #[must_use]
    pub fn fallback_op_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, ColorOp::Fallback(_)))
            .count()
    }
}

impl NprColorNode {
    /// Compile this tree into a [`CompiledColorPipeline`]
    #[must_use]
    pub fn compile(&self) -> CompiledColorPipeline {
        CompiledColorPipeline::compile(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> NprColorContext {
        NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.5, 0.5),
            time: 0.0,
        }
    }

    fn ctx_grazing() -> NprColorContext {
        NprColorContext {
            sdf: 0.0,
            normal: Vec3::new(0.0, 0.0, 1.0),
            view: Vec3::new(1.0, 0.0, 0.05).normalize(),
            light: Vec3::new(0.0, 1.0, 0.0),
            uv: Vec2::new(0.1, 0.9),
            time: 0.25,
        }
    }

    fn assert_close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-5,
            "expected {expected:?}, got {actual:?}"
        );
    }

    #[test]
    fn constant_compiles_and_evaluates() {
        let node = NprColorNode::Constant(Vec3::new(0.3, 0.6, 0.9));
        let compiled = node.compile();
        assert_eq!(compiled.ops.len(), 1);
        assert_eq!(compiled.eval(&ctx()), Vec3::new(0.3, 0.6, 0.9));
    }

    #[test]
    fn toon_compiles_and_matches_tree_eval() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 3,
        };
        let compiled = node.compile();
        assert_eq!(compiled.ops.len(), 3);
        assert_eq!(compiled.native_op_count(), 3);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn soft_toon_compiles_and_matches_tree_eval() {
        let node = NprColorNode::SoftToon {
            shadow: Vec3::new(0.1, 0.1, 0.2),
            light: Vec3::new(0.9, 0.9, 0.8),
            bands: 4,
            smoothness: 0.05,
        };
        let compiled = node.compile();
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn two_tone_compiles_and_matches_tree_eval() {
        let node = NprColorNode::TwoTone {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            threshold: 0.5,
        };
        let compiled = node.compile();
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn scale_wraps_child_and_matches_tree_eval() {
        let node = NprColorNode::Constant(Vec3::splat(0.6)).scale(0.5);
        let compiled = node.compile();
        assert_eq!(compiled.native_op_count(), 2);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn outline_over_compiles_natively() {
        let node =
            NprColorNode::Constant(Vec3::splat(0.4)).with_outline(Vec3::new(0.0, 0.0, 0.0), 0.7);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        assert_eq!(compiled.native_op_count(), 2);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn multiply_compiles_natively_and_matches() {
        let a = NprColorNode::Constant(Vec3::new(0.5, 0.7, 0.9));
        let b = NprColorNode::Constant(Vec3::new(0.8, 0.6, 0.4));
        let node = a.multiply(b);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        assert_eq!(compiled.native_op_count(), 3); // push a, push b, mul
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn add_compiles_natively_and_matches() {
        let a = NprColorNode::Constant(Vec3::new(0.2, 0.3, 0.4));
        let b = NprColorNode::Constant(Vec3::new(0.1, 0.2, 0.3));
        let node = a.plus(b);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        assert_eq!(compiled.native_op_count(), 3);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn fresnel_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5))
            .with_fresnel(Vec3::new(1.0, 1.0, 0.6), 2.5);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx_grazing();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn saturate_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.8, 0.2, 0.4)).saturate(0.3);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn bloom_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.9, 0.4, 0.1)).bloom(0.6, 1.5);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn posterize_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.33, 0.66, 0.99)).posterize(4);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn vignette_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::ONE).vignetted(0.5, 0.3);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        assert_eq!(compiled.native_op_count(), 2);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
        // Off-centre UV to exercise the mask
        let c2 = ctx_grazing();
        assert_close(compiled.eval(&c2), node.eval(&c2));
    }

    #[test]
    fn palette3_compiles_natively_and_matches() {
        let node = NprColorNode::Palette3 {
            source: PaletteSource::NDotL,
            c0: Vec3::new(0.1, 0.1, 0.4),
            c1: Vec3::new(0.6, 0.4, 0.2),
            c2: Vec3::new(1.0, 0.9, 0.6),
        };
        let compiled = node.compile();
        assert_eq!(compiled.native_op_count(), 1);
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn palette5_compiles_natively_and_matches() {
        let node = NprColorNode::Palette5 {
            source: PaletteSource::UvY,
            c0: Vec3::new(0.0, 0.0, 0.1),
            c1: Vec3::new(0.2, 0.1, 0.3),
            c2: Vec3::new(0.6, 0.3, 0.4),
            c3: Vec3::new(0.9, 0.6, 0.5),
            c4: Vec3::new(1.0, 0.95, 0.9),
        };
        let compiled = node.compile();
        assert_eq!(compiled.native_op_count(), 1);
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx_grazing();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn palette3_time_cycle_matches_tree() {
        let node = NprColorNode::Palette3 {
            source: PaletteSource::TimeCycle,
            c0: Vec3::new(0.1, 0.2, 0.3),
            c1: Vec3::new(0.5, 0.6, 0.7),
            c2: Vec3::new(0.9, 0.8, 0.7),
        };
        let compiled = node.compile();
        let mut c = ctx();
        for &t in &[0.0_f32, 0.1, 0.5, 0.9, 1.7] {
            c.time = t;
            assert_close(compiled.eval(&c), node.eval(&c));
        }
    }

    #[test]
    fn hatch_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.9, 0.9, 0.9)).with_hatch(
            0.5,
            30.0,
            0.15,
            Vec3::new(0.1, 0.1, 0.1),
        );
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx_grazing();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn tonemap_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(2.5, 0.4, 1.2)).tonemap_reinhard(1.2);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn speed_line_compiles_natively_and_matches() {
        let node = NprColorNode::Constant(Vec3::new(0.8, 0.8, 0.6)).with_speed_lines(
            Vec2::new(0.5, 0.5),
            24,
            0.05,
            Vec3::new(0.05, 0.05, 0.05),
        );
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx_grazing();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn nested_compile_preserves_semantics() {
        // Scale(Toon(shadow, light, bands))
        let node = NprColorNode::Toon {
            shadow: Vec3::new(0.1, 0.1, 0.2),
            light: Vec3::new(0.9, 0.8, 0.7),
            bands: 3,
        }
        .scale(0.75);
        let compiled = node.compile();
        assert_eq!(compiled.native_op_count(), 4);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn deep_composition_matches_tree_eval() {
        // Toon + outline + fresnel + vignette + saturate + tonemap
        let node = NprColorNode::Toon {
            shadow: Vec3::new(0.1, 0.05, 0.15),
            light: Vec3::new(0.85, 0.75, 0.65),
            bands: 4,
        }
        .with_outline(Vec3::new(0.0, 0.0, 0.0), 0.15)
        .with_fresnel(Vec3::new(0.9, 0.9, 1.0), 2.0)
        .vignetted(0.6, 0.25)
        .saturate(0.9)
        .tonemap_reinhard(1.0);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx_grazing();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn multiply_of_toon_and_palette3_matches_tree() {
        // Multiply(Toon(...), Palette3(...))
        let toon = NprColorNode::Toon {
            shadow: Vec3::splat(0.2),
            light: Vec3::splat(0.9),
            bands: 3,
        };
        let palette = NprColorNode::Palette3 {
            source: PaletteSource::NDotL,
            c0: Vec3::new(0.4, 0.2, 0.7),
            c1: Vec3::new(0.7, 0.5, 0.3),
            c2: Vec3::new(0.9, 0.9, 0.6),
        };
        let node = toon.multiply(palette);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 0);
        let c = ctx();
        assert_close(compiled.eval(&c), node.eval(&c));
    }

    #[test]
    fn empty_pipeline_panics_on_eval() {
        let pipeline = CompiledColorPipeline::default();
        let result = std::panic::catch_unwind(|| pipeline.eval(&ctx()));
        assert!(result.is_err());
    }

    #[test]
    fn all_current_variants_compile_without_fallback() {
        // Compose a tree touching every current NprColorNode variant so
        // that regression in native coverage surfaces as a fallback count > 0.
        let leaf = NprColorNode::Constant(Vec3::splat(0.5));
        let node = leaf
            .clone()
            .plus(leaf.clone())
            .multiply(NprColorNode::TwoTone {
                shadow: Vec3::ZERO,
                light: Vec3::ONE,
                threshold: 0.4,
            })
            .with_outline(Vec3::ZERO, 0.2)
            .with_fresnel(Vec3::ONE, 3.0)
            .saturate(0.8)
            .bloom(0.5, 1.2)
            .posterize(5)
            .vignetted(0.5, 0.2)
            .with_hatch(0.3, 20.0, 0.1, Vec3::ZERO)
            .tonemap_reinhard(1.0)
            .with_speed_lines(Vec2::new(0.5, 0.5), 12, 0.05, Vec3::ZERO)
            .scale(0.95);
        let compiled = node.compile();
        assert_eq!(
            compiled.fallback_op_count(),
            0,
            "Phase 12-D guarantee: current DSL surface has zero fallback opcodes"
        );
    }
}

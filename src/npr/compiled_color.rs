//! Bytecode compilation of `NprColorNode` trees for host-side evaluation
//!
//! This module compiles an [`NprColorNode`] expression tree into a flat
//! opcode stream ([`CompiledColorPipeline`]) that a small stack machine
//! evaluates. The compiled form is faster than recursive tree walking
//! for repeated evaluation (e.g. per-pixel in a CPU raymarching demo)
//! and is a stepping stone toward full integration with the existing
//! `SdfNode` bytecode compiler in `src/compiled/`.
//!
//! # Scope
//!
//! The current implementation supports the five most commonly-used
//! variants (`Constant`, `Toon`, `SoftToon`, `TwoTone`, `Scale`).
//! Other variants (`OutlineOver`, `Multiply`, `Add`, `Fresnel`,
//! `Saturate`, `Bloom`, `PosterizeColor`, `Vignette`, `Palette3`,
//! `Hatch`, `Palette5`, `Tonemap`, `SpeedLine`) fall back to the tree
//! walker via [`NprColorNode::eval`] and are transparently supported
//! by [`CompiledColorPipeline::eval`]; future revisions can extend
//! the opcode set to cover all variants.
//!
//! # Current performance
//!
//! On a single-point scalar path the bytecode evaluator is **slower**
//! than the recursive tree walker for shallow trees, because the
//! opcode Vec allocation and per-instruction stack push/pop dominate
//! the tiny math they replace. Measured on an Apple Silicon host:
//!
//! - `toon` tree eval: ~3.5 ns
//! - `toon` compiled eval: ~19 ns (~5-6x slower)
//!
//! The bytecode form starts to pay off once integrated with:
//!
//! - **SIMD batch evaluation** — one bytecode fetch amortised across
//!   a Vec3x8 lane group
//! - **GPU offload** — bytecode serialised into a uniform / storage
//!   buffer and executed on the shader side
//! - **Deep composition trees** — where per-node function-call cost
//!   in the tree walker starts to dominate
//!
//! Callers that only need scalar host-side evaluation should stay on
//! [`NprColorNode::eval`]. This module ships now so the pattern is
//! locked in ahead of the SIMD / GPU work.
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

use crate::npr::dsl::{NprColorContext, NprColorNode};
use crate::npr::toon::{soft_toon_ramp, toon_ramp, two_tone};
use glam::Vec3;

/// A single instruction in the compiled colour pipeline
#[derive(Debug, Clone)]
pub enum ColorOp {
    /// Push a constant colour onto the stack
    PushConstant(Vec3),
    /// Toon-ramp lookup: pop `shadow`, `light`; push `mix(shadow, light, toon_ramp(n_dot_l, bands))`
    Toon {
        /// Number of discrete bands
        bands: u32,
    },
    /// Soft toon-ramp lookup with smoothstep
    SoftToon {
        /// Number of discrete bands
        bands: u32,
        /// Smoothstep half-width per boundary
        smoothness: f32,
    },
    /// Two-tone: pop `shadow`, `light`; push `two_tone(n_dot_l, shadow, light, threshold)`
    TwoTone {
        /// Cutoff on `clamp(n_dot_l, 0, 1)`
        threshold: f32,
    },
    /// Scale the top-of-stack colour by a scalar
    Scale {
        /// Scalar multiplier applied to every channel
        factor: f32,
    },
    /// Fallback: evaluate the enclosed tree via the recursive walker
    ///
    /// Used for variants the bytecode does not yet cover directly.
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
    /// Panics if the pipeline is malformed (empty result stack). A
    /// well-formed pipeline produced by [`Self::compile`] never panics.
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
                ColorOp::Scale { factor } => {
                    let top = stack.pop().expect("Scale: empty stack");
                    stack.push(top * *factor);
                }
                ColorOp::Fallback(node) => {
                    // Fallback path: recursive tree eval for un-covered variants
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
            NprColorNode::Scale { child, factor } => {
                Self::compile_node(child, ops);
                ops.push(ColorOp::Scale { factor: *factor });
            }
            // All other variants: fall back to recursive eval.
            // Future revisions extend the opcode set to cover them.
            other => ops.push(ColorOp::Fallback(Box::new(other.clone()))),
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
    use glam::Vec2;

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
        assert_eq!(compiled.ops.len(), 3); // push shadow, push light, toon
        assert_eq!(compiled.native_op_count(), 3);
        let c = ctx();
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
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
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
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
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
    }

    #[test]
    fn scale_wraps_child_and_matches_tree_eval() {
        let node = NprColorNode::Constant(Vec3::splat(0.6)).scale(0.5);
        let compiled = node.compile();
        assert_eq!(compiled.native_op_count(), 2);
        let c = ctx();
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
    }

    #[test]
    fn fallback_used_for_uncovered_variants() {
        // Vignette is not natively encoded; expect a single Fallback opcode
        let child = NprColorNode::Constant(Vec3::ONE);
        let node = child.vignetted(0.5, 0.3);
        let compiled = node.compile();
        assert_eq!(compiled.fallback_op_count(), 1);
        assert_eq!(compiled.native_op_count(), 0);
        // Evaluation still produces the correct result via fallback
        let c = ctx();
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
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
        assert_eq!(compiled.native_op_count(), 4); // push, push, toon, scale
        let c = ctx();
        let expected = node.eval(&c);
        let actual = compiled.eval(&c);
        assert!((expected - actual).length() < 1e-6);
    }

    #[test]
    fn empty_pipeline_panics_on_eval() {
        let pipeline = CompiledColorPipeline::default();
        let result = std::panic::catch_unwind(|| pipeline.eval(&ctx()));
        assert!(result.is_err());
    }
}

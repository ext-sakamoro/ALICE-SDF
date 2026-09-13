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
//!   a `NprColorBatch8` lane group
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

// =====================================================================
// Phase 13 — 8-lane SIMD batch evaluator
// =====================================================================
//
// `CompiledColorPipeline::eval_batch8` evaluates the same bytecode
// against 8 shading contexts packed in SoA layout, using `wide::f32x8`.
// SIMD-native opcodes (pure arithmetic, floor, sqrt, blend) run fully
// lane-parallel; opcodes that require transcendental math (`Fresnel`
// `powf`, `SpeedLine` `atan2`) or that walk a small palette (`Palette5`)
// fall back to a per-lane scalar loop *inside* the batched context so
// SoA loading is preserved and the rest of the pipeline stays SIMD.

use wide::{f32x8, CmpGe, CmpGt};

/// 8-lane RGB colour batch in SoA layout
///
/// Held as three separate `f32x8` channels so lane-parallel arithmetic
/// is a straight primitive-op on each channel. Convert to and from
/// `[Vec3; 8]` via [`Self::from_vec3s`] and [`Self::to_vec3s`].
#[derive(Debug, Clone, Copy)]
pub struct NprColorBatch8 {
    /// Red channel, 8 lanes
    pub r: f32x8,
    /// Green channel, 8 lanes
    pub g: f32x8,
    /// Blue channel, 8 lanes
    pub b: f32x8,
}

impl NprColorBatch8 {
    /// Splat a scalar `Vec3` across all 8 lanes
    #[must_use]
    pub fn splat(v: Vec3) -> Self {
        Self {
            r: f32x8::splat(v.x),
            g: f32x8::splat(v.y),
            b: f32x8::splat(v.z),
        }
    }

    /// Build from 8 scalar `Vec3` values
    #[must_use]
    pub fn from_vec3s(vs: &[Vec3; 8]) -> Self {
        Self {
            r: f32x8::new([
                vs[0].x, vs[1].x, vs[2].x, vs[3].x, vs[4].x, vs[5].x, vs[6].x, vs[7].x,
            ]),
            g: f32x8::new([
                vs[0].y, vs[1].y, vs[2].y, vs[3].y, vs[4].y, vs[5].y, vs[6].y, vs[7].y,
            ]),
            b: f32x8::new([
                vs[0].z, vs[1].z, vs[2].z, vs[3].z, vs[4].z, vs[5].z, vs[6].z, vs[7].z,
            ]),
        }
    }

    /// Extract the 8 lanes back into a scalar `Vec3` array
    #[must_use]
    pub fn to_vec3s(&self) -> [Vec3; 8] {
        let r = self.r.to_array();
        let g = self.g.to_array();
        let b = self.b.to_array();
        [
            Vec3::new(r[0], g[0], b[0]),
            Vec3::new(r[1], g[1], b[1]),
            Vec3::new(r[2], g[2], b[2]),
            Vec3::new(r[3], g[3], b[3]),
            Vec3::new(r[4], g[4], b[4]),
            Vec3::new(r[5], g[5], b[5]),
            Vec3::new(r[6], g[6], b[6]),
            Vec3::new(r[7], g[7], b[7]),
        ]
    }

    /// Lane-parallel lerp: `self + (other - self) * t`
    #[must_use]
    pub fn lerp(self, other: Self, t: f32x8) -> Self {
        Self {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
        }
    }

    /// Multiply every channel by the same 8-lane scalar
    #[must_use]
    pub fn scale(self, s: f32x8) -> Self {
        Self {
            r: self.r * s,
            g: self.g * s,
            b: self.b * s,
        }
    }

    /// Multiply every channel by a scalar factor broadcast to all 8 lanes
    #[must_use]
    pub fn scale_scalar(self, s: f32) -> Self {
        self.scale(f32x8::splat(s))
    }

    /// Componentwise add
    #[must_use]
    pub fn add_vec3x8(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }

    /// Componentwise multiply
    #[must_use]
    pub fn mul_componentwise(self, other: Self) -> Self {
        Self {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
        }
    }

    /// Dot with a constant `Vec3` (per-lane): `r*v.x + g*v.y + b*v.z`
    #[must_use]
    pub fn dot_scalar(self, v: Vec3) -> f32x8 {
        self.r * f32x8::splat(v.x) + self.g * f32x8::splat(v.y) + self.b * f32x8::splat(v.z)
    }

    /// Lane-parallel max channel: `max(r, g, b)` per lane
    #[must_use]
    pub fn max_channel(self) -> f32x8 {
        self.r.max(self.g).max(self.b)
    }
}

/// 8-lane batched shading context in SoA layout
///
/// Only the derived scalars actually consumed by the opcode set are held.
/// Full `Vec3` fields (`normal` / `view` / `light`) present on
/// [`NprColorContext`] are not batched here because every opcode reads
/// only the derived dot products.
#[derive(Debug, Clone, Copy)]
pub struct NprBatchContext8 {
    /// N · L per lane
    pub n_dot_l: f32x8,
    /// N · V per lane
    pub n_dot_v: f32x8,
    /// SDF value per lane
    pub sdf: f32x8,
    /// UV.x per lane
    pub uv_x: f32x8,
    /// UV.y per lane
    pub uv_y: f32x8,
    /// Time per lane
    pub time: f32x8,
}

impl NprBatchContext8 {
    /// Build from 8 scalar `NprColorContext` values
    #[must_use]
    pub fn from_contexts(ctxs: &[NprColorContext; 8]) -> Self {
        Self {
            n_dot_l: f32x8::new([
                ctxs[0].n_dot_l(),
                ctxs[1].n_dot_l(),
                ctxs[2].n_dot_l(),
                ctxs[3].n_dot_l(),
                ctxs[4].n_dot_l(),
                ctxs[5].n_dot_l(),
                ctxs[6].n_dot_l(),
                ctxs[7].n_dot_l(),
            ]),
            n_dot_v: f32x8::new([
                ctxs[0].n_dot_v(),
                ctxs[1].n_dot_v(),
                ctxs[2].n_dot_v(),
                ctxs[3].n_dot_v(),
                ctxs[4].n_dot_v(),
                ctxs[5].n_dot_v(),
                ctxs[6].n_dot_v(),
                ctxs[7].n_dot_v(),
            ]),
            sdf: f32x8::new([
                ctxs[0].sdf,
                ctxs[1].sdf,
                ctxs[2].sdf,
                ctxs[3].sdf,
                ctxs[4].sdf,
                ctxs[5].sdf,
                ctxs[6].sdf,
                ctxs[7].sdf,
            ]),
            uv_x: f32x8::new([
                ctxs[0].uv.x,
                ctxs[1].uv.x,
                ctxs[2].uv.x,
                ctxs[3].uv.x,
                ctxs[4].uv.x,
                ctxs[5].uv.x,
                ctxs[6].uv.x,
                ctxs[7].uv.x,
            ]),
            uv_y: f32x8::new([
                ctxs[0].uv.y,
                ctxs[1].uv.y,
                ctxs[2].uv.y,
                ctxs[3].uv.y,
                ctxs[4].uv.y,
                ctxs[5].uv.y,
                ctxs[6].uv.y,
                ctxs[7].uv.y,
            ]),
            time: f32x8::new([
                ctxs[0].time,
                ctxs[1].time,
                ctxs[2].time,
                ctxs[3].time,
                ctxs[4].time,
                ctxs[5].time,
                ctxs[6].time,
                ctxs[7].time,
            ]),
        }
    }
}

#[inline]
fn clamp01_x8(x: f32x8) -> f32x8 {
    x.max(f32x8::splat(0.0)).min(f32x8::splat(1.0))
}

#[inline]
fn smoothstep_x8(edge0: f32, edge1: f32, x: f32x8) -> f32x8 {
    let denom_s = (edge1 - edge0).abs().max(1e-6);
    let t = clamp01_x8((x - f32x8::splat(edge0)) / f32x8::splat(denom_s));
    t * t * (f32x8::splat(3.0) - f32x8::splat(2.0) * t)
}

#[inline]
fn smoothstep_x8_x_edges(edge0: f32x8, edge1: f32x8, x: f32x8) -> f32x8 {
    let raw = edge1 - edge0;
    let denom = raw.abs().max(f32x8::splat(1e-6));
    let t = clamp01_x8((x - edge0) / denom);
    t * t * (f32x8::splat(3.0) - f32x8::splat(2.0) * t)
}

#[inline]
fn toon_ramp_x8(n_dot_l: f32x8, bands: u32) -> f32x8 {
    assert!(bands > 0, "bands must be > 0");
    let bands_f = bands as f32;
    let clamped = clamp01_x8(n_dot_l);
    let idx = (clamped * f32x8::splat(bands_f))
        .floor()
        .min(f32x8::splat(bands_f - 1.0));
    idx / f32x8::splat((bands_f - 1.0).max(1.0))
}

#[inline]
fn soft_toon_ramp_x8(n_dot_l: f32x8, bands: u32, smoothness: f32) -> f32x8 {
    assert!(bands > 0, "bands must be > 0");
    let bands_f = bands as f32;
    let clamped = clamp01_x8(n_dot_l);
    let scaled = clamped * f32x8::splat(bands_f);
    let idx = scaled.floor().min(f32x8::splat(bands_f - 1.0));
    let frac = clamp01_x8(scaled - idx);
    let s = smoothness.clamp(0.001, 0.5);
    let t = smoothstep_x8(0.5 - s, 0.5 + s, frac);
    clamp01_x8((idx + t) / f32x8::splat(bands_f))
}

#[inline]
fn two_tone_x8(
    n_dot_l: f32x8,
    shadow: NprColorBatch8,
    light: NprColorBatch8,
    threshold: f32,
) -> NprColorBatch8 {
    let t = clamp01_x8(n_dot_l);
    let use_light = t.cmp_ge(f32x8::splat(threshold));
    NprColorBatch8 {
        r: use_light.blend(light.r, shadow.r),
        g: use_light.blend(light.g, shadow.g),
        b: use_light.blend(light.b, shadow.b),
    }
}

#[inline]
fn posterize_color_x8(color: NprColorBatch8, levels: u32) -> NprColorBatch8 {
    let levels = levels.max(2);
    let steps = levels as f32;
    let denom = (steps - 1.0).max(1.0);
    let steps_v = f32x8::splat(steps);
    let denom_v = f32x8::splat(denom);
    let quantize = |c: f32x8| -> f32x8 { clamp01_x8((clamp01_x8(c) * steps_v).floor() / denom_v) };
    NprColorBatch8 {
        r: quantize(color.r),
        g: quantize(color.g),
        b: quantize(color.b),
    }
}

#[inline]
fn bloom_toon_x8(color: NprColorBatch8, threshold: f32, intensity: f32) -> NprColorBatch8 {
    let max_ch = color.max_channel();
    let pass = max_ch.cmp_gt(f32x8::splat(threshold));
    let intensity_v = f32x8::splat(intensity);
    let bright = color.scale(intensity_v);
    NprColorBatch8 {
        r: pass.blend(bright.r, f32x8::splat(0.0)),
        g: pass.blend(bright.g, f32x8::splat(0.0)),
        b: pass.blend(bright.b, f32x8::splat(0.0)),
    }
}

#[inline]
fn vignette_x8(uv_x: f32x8, uv_y: f32x8, radius: f32, softness: f32) -> f32x8 {
    let dx = uv_x - f32x8::splat(0.5);
    let dy = uv_y - f32x8::splat(0.5);
    let d = (dx * dx + dy * dy).sqrt();
    let inner = radius.max(0.0);
    let inner_x8 = f32x8::splat(inner);
    let outer = f32x8::splat((inner + softness.max(0.0)).max(inner + 1e-6));
    // Branch-free equivalent of the scalar impl:
    //   inside inner: 1.0 (smoothstep = 0)
    //   outside outer: 0.0 (smoothstep = 1)
    //   between: 1 - smoothstep(inner, outer, d)
    // smoothstep_x8_x_edges clamps t into [0, 1] internally, giving the
    // same end-point behaviour as the branchy scalar version.
    f32x8::splat(1.0) - smoothstep_x8_x_edges(inner_x8, outer, d)
}

#[inline]
fn hatch_lines_x8(uv_x: f32x8, uv_y: f32x8, angle_rad: f32, density: f32, thickness: f32) -> f32x8 {
    // Mirror `hatch::hatch_lines` scalar semantics:
    //   projected = uv_x * (-sin(a)) + uv_y * cos(a)
    //   raw = projected * max(density, 1e-6)
    //   phase = fract(raw)
    //   dist = |phase - 0.5|
    //   1 where dist > 0.5 - clamp(thickness, 0, 0.5), else 0
    let (sin_a, cos_a) = angle_rad.sin_cos();
    let d = density.max(1e-6);
    let t = thickness.clamp(0.0, 0.5);
    let projected = uv_x * f32x8::splat(-sin_a) + uv_y * f32x8::splat(cos_a);
    let raw = projected * f32x8::splat(d);
    let phase = raw - raw.floor();
    let dist = (phase - f32x8::splat(0.5)).abs();
    let hit = dist.cmp_gt(f32x8::splat(0.5 - t));
    hit.blend(f32x8::splat(1.0), f32x8::splat(0.0))
}

#[inline]
fn palette_gradient_3_x8(t: f32x8, c0: Vec3, c1: Vec3, c2: Vec3) -> NprColorBatch8 {
    let clamped = clamp01_x8(t);
    let scaled = clamped * f32x8::splat(2.0);
    let seg1_mask = scaled.cmp_ge(f32x8::splat(1.0));
    let frac_low = clamp01_x8(scaled);
    let frac_high = clamp01_x8(scaled - f32x8::splat(1.0));
    let c0v = NprColorBatch8::splat(c0);
    let c1v = NprColorBatch8::splat(c1);
    let c2v = NprColorBatch8::splat(c2);
    let seg0 = c0v.lerp(c1v, frac_low);
    let seg1 = c1v.lerp(c2v, frac_high);
    NprColorBatch8 {
        r: seg1_mask.blend(seg1.r, seg0.r),
        g: seg1_mask.blend(seg1.g, seg0.g),
        b: seg1_mask.blend(seg1.b, seg0.b),
    }
}

#[inline]
fn palette_source_scalar_x8(source: PaletteSource, batch: &NprBatchContext8) -> f32x8 {
    match source {
        PaletteSource::NDotL => clamp01_x8(batch.n_dot_l),
        PaletteSource::NDotV => clamp01_x8(batch.n_dot_v),
        PaletteSource::Sdf => clamp01_x8(batch.sdf.abs()),
        PaletteSource::UvY => clamp01_x8(batch.uv_y),
        PaletteSource::TimeCycle => batch.time - batch.time.floor(),
    }
}

/// Per-lane scalar loop over transcendental / small-palette ops
///
/// `f` is called once per lane with `(lane_index, batch_component_value)`
/// and produces one scalar RGB triple. Used for Fresnel (powf),
/// SpeedLine (atan2), and Palette5 (segmented lerp).
#[inline]
fn per_lane_scalar_output<F>(
    batch: &NprBatchContext8,
    base: NprColorBatch8,
    mut f: F,
) -> NprColorBatch8
where
    F: FnMut(usize, &LaneScalarView, Vec3) -> Vec3,
{
    let n_dot_l = batch.n_dot_l.to_array();
    let n_dot_v = batch.n_dot_v.to_array();
    let sdf = batch.sdf.to_array();
    let uv_x = batch.uv_x.to_array();
    let uv_y = batch.uv_y.to_array();
    let time = batch.time.to_array();
    let base_v = base.to_vec3s();
    let mut out = [Vec3::ZERO; 8];
    for i in 0..8 {
        let view = LaneScalarView {
            n_dot_l: n_dot_l[i],
            n_dot_v: n_dot_v[i],
            sdf: sdf[i],
            uv_x: uv_x[i],
            uv_y: uv_y[i],
            time: time[i],
        };
        out[i] = f(i, &view, base_v[i]);
    }
    NprColorBatch8::from_vec3s(&out)
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Full view is exposed for future closures; today's
                    // closures use only a subset (Fresnel: n_dot_v;
                    // SpeedLine: uv_x, uv_y).
struct LaneScalarView {
    n_dot_l: f32,
    n_dot_v: f32,
    sdf: f32,
    uv_x: f32,
    uv_y: f32,
    time: f32,
}

impl CompiledColorPipeline {
    /// Evaluate the pipeline against a batched 8-lane shading context
    ///
    /// SIMD-native for pure-arithmetic opcodes (`Multiply`, `Add`, `Scale`,
    /// `Saturate`, `Bloom`, `PosterizeColor`, `Tonemap`, `OutlineOver`,
    /// `Toon`, `SoftToon`, `TwoTone`, `Vignette`, `Hatch`, `Palette3`).
    /// Opcodes that need `powf` (`Fresnel`), `atan2` (`SpeedLine`), or a
    /// small palette walk (`Palette5`) use a per-lane scalar loop over
    /// the SoA batch to keep the surrounding pipeline lane-parallel.
    ///
    /// # Panics
    /// Panics if the pipeline is malformed (empty result stack or an
    /// opcode's stack pop underflows). A well-formed pipeline produced
    /// by [`Self::compile`] never panics.
    #[must_use]
    pub fn eval_batch8(&self, batch: &NprBatchContext8) -> NprColorBatch8 {
        let mut stack: Vec<NprColorBatch8> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            match op {
                ColorOp::PushConstant(c) => stack.push(NprColorBatch8::splat(*c)),
                ColorOp::Toon { bands } => {
                    let light = stack.pop().expect("Toon: missing light");
                    let shadow = stack.pop().expect("Toon: missing shadow");
                    let t = toon_ramp_x8(batch.n_dot_l, *bands);
                    stack.push(shadow.lerp(light, t));
                }
                ColorOp::SoftToon { bands, smoothness } => {
                    let light = stack.pop().expect("SoftToon: missing light");
                    let shadow = stack.pop().expect("SoftToon: missing shadow");
                    let t = soft_toon_ramp_x8(batch.n_dot_l, *bands, *smoothness);
                    stack.push(shadow.lerp(light, t));
                }
                ColorOp::TwoTone { threshold } => {
                    let light = stack.pop().expect("TwoTone: missing light");
                    let shadow = stack.pop().expect("TwoTone: missing shadow");
                    stack.push(two_tone_x8(batch.n_dot_l, shadow, light, *threshold));
                }
                ColorOp::Multiply => {
                    let b = stack.pop().expect("Multiply: missing rhs");
                    let a = stack.pop().expect("Multiply: missing lhs");
                    stack.push(a.mul_componentwise(b));
                }
                ColorOp::Add => {
                    let b = stack.pop().expect("Add: missing rhs");
                    let a = stack.pop().expect("Add: missing lhs");
                    stack.push(a.add_vec3x8(b));
                }
                ColorOp::Scale { factor } => {
                    let top = stack.pop().expect("Scale: empty stack");
                    stack.push(top.scale_scalar(*factor));
                }
                ColorOp::OutlineOver { outline, alpha } => {
                    let base = stack.pop().expect("OutlineOver: missing base");
                    let a = alpha.clamp(0.0, 1.0);
                    let outline_v = NprColorBatch8::splat(*outline);
                    // base * (1 - a) + outline * a
                    let inv = f32x8::splat(1.0 - a);
                    let av = f32x8::splat(a);
                    stack.push(NprColorBatch8 {
                        r: base.r * inv + outline_v.r * av,
                        g: base.g * inv + outline_v.g * av,
                        b: base.b * inv + outline_v.b * av,
                    });
                }
                ColorOp::Fresnel { edge, power } => {
                    // powf isn't a first-class op in this wide version;
                    // per-lane scalar over SoA batch keeps the rest SIMD.
                    let base = stack.pop().expect("Fresnel: missing base");
                    let edge = *edge;
                    let power = *power;
                    stack.push(per_lane_scalar_output(batch, base, |_, view, base_c| {
                        let f = (1.0 - view.n_dot_v.clamp(0.0, 1.0))
                            .max(0.0)
                            .powf(power.max(0.0));
                        base_c.lerp(edge, f.clamp(0.0, 1.0))
                    }));
                }
                ColorOp::Saturate { factor } => {
                    let color = stack.pop().expect("Saturate: empty stack");
                    let lum = color.dot_scalar(Vec3::new(0.2126, 0.7152, 0.0722));
                    let grey = NprColorBatch8 {
                        r: lum,
                        g: lum,
                        b: lum,
                    };
                    stack.push(grey.lerp(color, f32x8::splat(*factor)));
                }
                ColorOp::Bloom {
                    threshold,
                    intensity,
                } => {
                    let color = stack.pop().expect("Bloom: empty stack");
                    stack.push(bloom_toon_x8(color, *threshold, *intensity));
                }
                ColorOp::PosterizeColor { levels } => {
                    let color = stack.pop().expect("PosterizeColor: empty stack");
                    stack.push(posterize_color_x8(color, *levels));
                }
                ColorOp::Vignette { radius, softness } => {
                    let color = stack.pop().expect("Vignette: empty stack");
                    let mask = vignette_x8(batch.uv_x, batch.uv_y, *radius, *softness);
                    stack.push(color.scale(mask));
                }
                ColorOp::Palette3 { source, c0, c1, c2 } => {
                    let t = palette_source_scalar_x8(*source, batch);
                    stack.push(palette_gradient_3_x8(t, *c0, *c1, *c2));
                }
                ColorOp::Palette5 {
                    source,
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                } => {
                    // 5-anchor palette has 4 segments; per-lane scalar is
                    // cheaper than 4 nested blends here.
                    let t = palette_source_scalar_x8(*source, batch);
                    let ts = t.to_array();
                    let palette = [*c0, *c1, *c2, *c3, *c4];
                    let mut out = [Vec3::ZERO; 8];
                    for (i, &t_lane) in ts.iter().enumerate() {
                        out[i] = palette_gradient(t_lane, &palette);
                    }
                    stack.push(NprColorBatch8::from_vec3s(&out));
                }
                ColorOp::Hatch {
                    angle_rad,
                    density,
                    thickness,
                    ink,
                } => {
                    let base = stack.pop().expect("Hatch: missing base");
                    let mask =
                        hatch_lines_x8(batch.uv_x, batch.uv_y, *angle_rad, *density, *thickness);
                    stack.push(base.lerp(NprColorBatch8::splat(*ink), mask));
                }
                ColorOp::Tonemap { exposure } => {
                    let color = stack.pop().expect("Tonemap: empty stack");
                    let exp_v = f32x8::splat(exposure.max(0.0));
                    let scaled = color.scale(exp_v);
                    stack.push(NprColorBatch8 {
                        r: scaled.r / (f32x8::splat(1.0) + scaled.r),
                        g: scaled.g / (f32x8::splat(1.0) + scaled.g),
                        b: scaled.b / (f32x8::splat(1.0) + scaled.b),
                    });
                }
                ColorOp::SpeedLine {
                    focus,
                    count,
                    thickness,
                    ink,
                } => {
                    // atan2 isn't SIMD-native here; per-lane over SoA batch.
                    let base = stack.pop().expect("SpeedLine: missing base");
                    let focus = *focus;
                    let count = *count;
                    let thickness = *thickness;
                    let ink = *ink;
                    stack.push(per_lane_scalar_output(batch, base, |_, view, base_c| {
                        let mask =
                            speed_line(view.uv_x, view.uv_y, focus.x, focus.y, count, thickness);
                        base_c.lerp(ink, mask)
                    }));
                }
                ColorOp::Fallback(node) => {
                    // Fallback still walks the tree per lane; kept for
                    // forward-compat with future NprColorNode variants.
                    let node = node.as_ref();
                    let base = NprColorBatch8::splat(Vec3::ZERO); // unused input
                    stack.push(per_lane_scalar_output(batch, base, |i, _view, _base_c| {
                        // Rebuild a minimal NprColorContext for the tree walker.
                        // Only the derived scalars are populated; opcodes that
                        // read raw normal/view/light are not currently reachable
                        // via Fallback (they compile natively as of P12-D).
                        let ctx = fallback_reconstruct_ctx(batch, i);
                        node.eval(&ctx)
                    }));
                }
            }
        }
        stack.pop().expect("pipeline produced no result")
    }
}

/// Reconstruct a minimal `NprColorContext` for the `Fallback` code path
///
/// `normal` / `view` / `light` are set to axis-aligned defaults that
/// reproduce `n_dot_l` and `n_dot_v` via `Vec3::dot` on the derived
/// scalars. This is enough for any pure-`ctx.n_dot_l()` / `ctx.n_dot_v()`
/// consumer inside a future `NprColorNode` variant.
#[inline]
fn fallback_reconstruct_ctx(batch: &NprBatchContext8, lane: usize) -> NprColorContext {
    let n_dot_l = batch.n_dot_l.to_array()[lane];
    let n_dot_v = batch.n_dot_v.to_array()[lane];
    let sdf = batch.sdf.to_array()[lane];
    let uv_x = batch.uv_x.to_array()[lane];
    let uv_y = batch.uv_y.to_array()[lane];
    let time = batch.time.to_array()[lane];
    NprColorContext {
        sdf,
        // Normal at +Y so that `normal.dot(light)` = light.y.
        normal: Vec3::new(0.0, 1.0, 0.0),
        // View at +Z so that `normal.dot(view)` = 0 by default; the actual
        // n_dot_v is honoured by setting view to Vec3::new(0, n_dot_v, s).
        view: Vec3::new(0.0, n_dot_v, (1.0 - n_dot_v * n_dot_v).max(0.0).sqrt()),
        // Light aligned so that `normal.dot(light) == n_dot_l`.
        light: Vec3::new(0.0, n_dot_l, (1.0 - n_dot_l * n_dot_l).max(0.0).sqrt()),
        uv: Vec2::new(uv_x, uv_y),
        time,
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

    // -----------------------------------------------------------------
    // Phase 13 — SIMD batch evaluator tests
    // -----------------------------------------------------------------

    fn eight_varied_contexts() -> [NprColorContext; 8] {
        let mkctx = |ndl: f32, uv: Vec2, time: f32, sdf: f32| NprColorContext {
            sdf,
            normal: Vec3::new(0.0, 1.0, 0.0),
            view: Vec3::new(0.0, 0.0, 1.0),
            // Direct construction from n_dot_l so light.y == n_dot_l with
            // normal fixed to +Y. Zero out other components.
            light: Vec3::new(0.0, ndl, (1.0 - ndl * ndl).max(0.0).sqrt()),
            uv,
            time,
        };
        [
            mkctx(1.0, Vec2::new(0.5, 0.5), 0.0, 0.0),
            mkctx(0.85, Vec2::new(0.3, 0.7), 0.25, 0.1),
            mkctx(0.5, Vec2::new(0.75, 0.25), 0.5, -0.2),
            mkctx(0.25, Vec2::new(0.1, 0.9), 0.75, 0.5),
            mkctx(0.0, Vec2::new(0.5, 0.5), 1.0, -1.0),
            mkctx(0.75, Vec2::new(0.9, 0.1), 1.5, 0.3),
            mkctx(0.4, Vec2::new(0.2, 0.8), 2.25, -0.7),
            mkctx(0.15, Vec2::new(0.55, 0.45), 3.14159, 0.05),
        ]
    }

    fn assert_batch_matches_scalar(node: &NprColorNode, ctxs: &[NprColorContext; 8]) {
        let compiled = node.compile();
        let batch = NprBatchContext8::from_contexts(ctxs);
        let batch_out = compiled.eval_batch8(&batch).to_vec3s();
        for (i, ctx) in ctxs.iter().enumerate() {
            let scalar = compiled.eval(ctx);
            let batched = batch_out[i];
            assert!(
                (scalar - batched).length() < 1e-4,
                "lane {i}: scalar {scalar:?} vs batched {batched:?} (op set: {node:?})"
            );
        }
    }

    #[test]
    fn vec3x8_from_vec3s_roundtrip() {
        let src = [
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.4, 0.5, 0.6),
            Vec3::new(0.7, 0.8, 0.9),
            Vec3::new(1.0, 0.0, 0.5),
            Vec3::new(0.25, 0.75, 0.125),
            Vec3::new(0.9, 0.1, 0.5),
            Vec3::new(0.15, 0.35, 0.55),
            Vec3::new(0.8, 0.2, 0.6),
        ];
        let batched = NprColorBatch8::from_vec3s(&src);
        let back = batched.to_vec3s();
        for i in 0..8 {
            assert!((back[i] - src[i]).length() < 1e-6, "roundtrip lane {i}");
        }
    }

    #[test]
    fn batch_matches_scalar_for_constant() {
        let node = NprColorNode::Constant(Vec3::new(0.42, 0.24, 0.66));
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_toon() {
        let node = NprColorNode::Toon {
            shadow: Vec3::new(0.1, 0.1, 0.2),
            light: Vec3::new(0.9, 0.85, 0.75),
            bands: 4,
        };
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_soft_toon() {
        let node = NprColorNode::SoftToon {
            shadow: Vec3::new(0.05, 0.05, 0.15),
            light: Vec3::new(0.95, 0.9, 0.8),
            bands: 3,
            smoothness: 0.08,
        };
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_two_tone() {
        let node = NprColorNode::TwoTone {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            threshold: 0.5,
        };
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_multiply_add_scale() {
        let a = NprColorNode::Constant(Vec3::new(0.6, 0.4, 0.8));
        let b = NprColorNode::Constant(Vec3::new(0.5, 0.7, 0.3));
        let node = a
            .multiply(b)
            .plus(NprColorNode::Constant(Vec3::splat(0.1)))
            .scale(0.5);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_outline_over() {
        let node = NprColorNode::Constant(Vec3::new(0.6, 0.5, 0.4))
            .with_outline(Vec3::new(0.0, 0.0, 0.0), 0.35);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_fresnel() {
        let node = NprColorNode::Constant(Vec3::new(0.5, 0.5, 0.5))
            .with_fresnel(Vec3::new(1.0, 1.0, 0.6), 2.5);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_saturate() {
        let node = NprColorNode::Constant(Vec3::new(0.8, 0.2, 0.4)).saturate(0.3);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_bloom() {
        let node = NprColorNode::Constant(Vec3::new(0.9, 0.4, 0.1)).bloom(0.6, 1.5);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_posterize() {
        let node = NprColorNode::Constant(Vec3::new(0.33, 0.66, 0.99)).posterize(4);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_vignette() {
        let node = NprColorNode::Constant(Vec3::ONE).vignetted(0.5, 0.3);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_palette3_all_sources() {
        for source in [
            PaletteSource::NDotL,
            PaletteSource::NDotV,
            PaletteSource::Sdf,
            PaletteSource::UvY,
            PaletteSource::TimeCycle,
        ] {
            let node = NprColorNode::Palette3 {
                source,
                c0: Vec3::new(0.1, 0.1, 0.4),
                c1: Vec3::new(0.6, 0.4, 0.2),
                c2: Vec3::new(1.0, 0.9, 0.6),
            };
            assert_batch_matches_scalar(&node, &eight_varied_contexts());
        }
    }

    #[test]
    fn batch_matches_scalar_for_palette5() {
        let node = NprColorNode::Palette5 {
            source: PaletteSource::UvY,
            c0: Vec3::new(0.0, 0.0, 0.1),
            c1: Vec3::new(0.2, 0.1, 0.3),
            c2: Vec3::new(0.6, 0.3, 0.4),
            c3: Vec3::new(0.9, 0.6, 0.5),
            c4: Vec3::new(1.0, 0.95, 0.9),
        };
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_hatch() {
        let node = NprColorNode::Constant(Vec3::new(0.9, 0.9, 0.9)).with_hatch(
            0.5,
            30.0,
            0.15,
            Vec3::new(0.1, 0.1, 0.1),
        );
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_tonemap() {
        let node = NprColorNode::Constant(Vec3::new(2.5, 0.4, 1.2)).tonemap_reinhard(1.2);
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_speed_line() {
        let node = NprColorNode::Constant(Vec3::new(0.8, 0.8, 0.6)).with_speed_lines(
            Vec2::new(0.5, 0.5),
            24,
            0.05,
            Vec3::new(0.05, 0.05, 0.05),
        );
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_deep_composition() {
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
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }

    #[test]
    fn batch_matches_scalar_for_full_variant_composition() {
        // Mirror the P12-D guard: touch every current variant in a single
        // pipeline and verify batch == scalar.
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
        assert_batch_matches_scalar(&node, &eight_varied_contexts());
    }
}

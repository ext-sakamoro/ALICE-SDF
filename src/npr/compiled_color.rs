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
    pub const fn from_vec3s(vs: &[Vec3; 8]) -> Self {
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
        view: Vec3::new(0.0, n_dot_v, n_dot_v.mul_add(-n_dot_v, 1.0).max(0.0).sqrt()),
        // Light aligned so that `normal.dot(light) == n_dot_l`.
        light: Vec3::new(0.0, n_dot_l, n_dot_l.mul_add(-n_dot_l, 1.0).max(0.0).sqrt()),
        uv: Vec2::new(uv_x, uv_y),
        time,
    }
}

// =====================================================================
// Phase 14 — GPU bytecode serialisation
// =====================================================================
//
// `CompiledColorPipeline::serialize` encodes the opcode stream into a
// `Vec<u32>` bytecode that a shader-side stack machine can execute.
// The GPU cannot call back into the CPU tree walker, so pipelines that
// still contain [`ColorOp::Fallback`] are rejected up front.
// Well-formed pipelines produced by [`CompiledColorPipeline::compile`]
// on today's DSL surface (Phase 12-D onward) never contain `Fallback`.
//
// Layout: a flat `[u32]` stream where each instruction is
// `[tag, ..payload_words]`. Payloads store `f32` and `Vec3` values via
// `to_bits` / `from_bits`, `u32` values directly, and `PaletteSource`
// as a small `u32` tag. Instruction sizes are fixed per opcode tag; see
// [`GPU_OPCODE_TAG`] for the tag numbering and [`opcode_word_count`]
// for the exact word count consumed after the tag.

/// Opcode tag values used by the GPU bytecode format
///
/// Kept in a dedicated `mod` so the WGSL evaluator generator can emit
/// matching constants. Values are stable across releases; adding a new
/// variant must append to the end of the range.
pub mod gpu_opcode_tag {
    /// PushConstant(Vec3) — 3 payload words
    pub const PUSH_CONSTANT: u32 = 0;
    /// Toon { bands } — 1 payload word
    pub const TOON: u32 = 1;
    /// SoftToon { bands, smoothness } — 2 payload words
    pub const SOFT_TOON: u32 = 2;
    /// TwoTone { threshold } — 1 payload word
    pub const TWO_TONE: u32 = 3;
    /// Multiply — 0 payload words
    pub const MULTIPLY: u32 = 4;
    /// Add — 0 payload words
    pub const ADD: u32 = 5;
    /// Scale { factor } — 1 payload word
    pub const SCALE: u32 = 6;
    /// OutlineOver { outline, alpha } — 4 payload words
    pub const OUTLINE_OVER: u32 = 7;
    /// Fresnel { edge, power } — 4 payload words
    pub const FRESNEL: u32 = 8;
    /// Saturate { factor } — 1 payload word
    pub const SATURATE: u32 = 9;
    /// Bloom { threshold, intensity } — 2 payload words
    pub const BLOOM: u32 = 10;
    /// PosterizeColor { levels } — 1 payload word
    pub const POSTERIZE_COLOR: u32 = 11;
    /// Vignette { radius, softness } — 2 payload words
    pub const VIGNETTE: u32 = 12;
    /// Palette3 { source, c0, c1, c2 } — 10 payload words
    pub const PALETTE3: u32 = 13;
    /// Palette5 { source, c0..c4 } — 16 payload words
    pub const PALETTE5: u32 = 14;
    /// Hatch { angle_rad, density, thickness, ink } — 6 payload words
    pub const HATCH: u32 = 15;
    /// Tonemap { exposure } — 1 payload word
    pub const TONEMAP: u32 = 16;
    /// SpeedLine { focus, count, thickness, ink } — 7 payload words
    pub const SPEED_LINE: u32 = 17;
}

/// Tag values for [`PaletteSource`] in the GPU bytecode
pub mod gpu_palette_source_tag {
    /// `PaletteSource::NDotL`
    pub const N_DOT_L: u32 = 0;
    /// `PaletteSource::NDotV`
    pub const N_DOT_V: u32 = 1;
    /// `PaletteSource::Sdf`
    pub const SDF: u32 = 2;
    /// `PaletteSource::UvY`
    pub const UV_Y: u32 = 3;
    /// `PaletteSource::TimeCycle`
    pub const TIME_CYCLE: u32 = 4;
}

/// Errors returned by [`CompiledColorPipeline::serialize`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializeError {
    /// The pipeline contains a [`ColorOp::Fallback`] opcode, which
    /// cannot be executed on the GPU (the shader has no way to call
    /// back into the CPU tree walker).
    UnsupportedFallback {
        /// Zero-based instruction index of the first Fallback encountered
        instruction_index: usize,
    },
}

impl core::fmt::Display for SerializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedFallback { instruction_index } => write!(
                f,
                "GPU serialisation not supported: pipeline contains \
                 ColorOp::Fallback at instruction {instruction_index}"
            ),
        }
    }
}

impl std::error::Error for SerializeError {}

/// Errors returned by [`GpuColorProgram::deserialize`]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeserializeError {
    /// Unknown opcode tag encountered at the given word offset
    UnknownOpcode {
        /// Word offset in the input `[u32]` stream
        word_offset: usize,
        /// The unknown tag that was read
        tag: u32,
    },
    /// The stream ended mid-instruction (payload words missing)
    Truncated {
        /// Word offset where the truncation was detected
        word_offset: usize,
        /// Number of payload words expected after the tag
        expected_payload: usize,
    },
    /// Unknown palette-source tag in a `Palette3` / `Palette5` opcode
    UnknownPaletteSource {
        /// Word offset in the input stream
        word_offset: usize,
        /// The unknown tag that was read
        tag: u32,
    },
}

impl core::fmt::Display for DeserializeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownOpcode { word_offset, tag } => write!(
                f,
                "unknown GPU bytecode opcode tag {tag} at word offset {word_offset}"
            ),
            Self::Truncated {
                word_offset,
                expected_payload,
            } => write!(
                f,
                "GPU bytecode truncated at word offset {word_offset}: \
                 expected {expected_payload} payload words"
            ),
            Self::UnknownPaletteSource { word_offset, tag } => write!(
                f,
                "unknown PaletteSource tag {tag} at word offset {word_offset}"
            ),
        }
    }
}

impl std::error::Error for DeserializeError {}

/// GPU-side bytecode program derived from a [`CompiledColorPipeline`]
///
/// The `words` buffer is the flat `[u32]` stream described in the
/// module header. Upload as a uniform or storage buffer and dispatch
/// against the WGSL evaluator emitted by
/// [`emit_wgsl_bytecode_evaluator`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuColorProgram {
    /// The instruction stream, one `[tag, ..payload]` block per opcode
    pub words: Vec<u32>,
}

impl GpuColorProgram {
    /// The full stream as a `&[u32]` slice (upload-ready)
    #[must_use]
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }

    /// Total byte length of the stream (`words.len() * 4`)
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.words.len() * core::mem::size_of::<u32>()
    }

    /// Decode a bytecode stream back into a [`CompiledColorPipeline`]
    ///
    /// Provided as a round-trip check: `pipeline.serialize()?.deserialize()?`
    /// evaluates identically to the original pipeline. `Fallback` cannot
    /// appear in a valid stream (it is rejected at serialisation time).
    ///
    /// # Errors
    /// Returns [`DeserializeError`] on unknown opcode tag, truncated
    /// payload, or unknown `PaletteSource` tag.
    pub fn deserialize(&self) -> Result<CompiledColorPipeline, DeserializeError> {
        let mut ops = Vec::new();
        let mut pc = 0usize;
        while pc < self.words.len() {
            let tag = self.words[pc];
            pc += 1;
            let (op, consumed) = decode_opcode(tag, &self.words, pc)?;
            pc += consumed;
            ops.push(op);
        }
        Ok(CompiledColorPipeline { ops })
    }
}

/// Payload word count consumed *after* the tag word for a given opcode
///
/// The tag itself is not included. Total instruction size is
/// `1 + opcode_word_count(tag)?`.
///
/// Returns `None` for unknown tags.
#[must_use]
pub const fn opcode_word_count(tag: u32) -> Option<usize> {
    use gpu_opcode_tag as t;
    let n = match tag {
        t::PUSH_CONSTANT => 3,
        t::TOON => 1,
        t::SOFT_TOON => 2,
        t::TWO_TONE => 1,
        t::MULTIPLY | t::ADD => 0,
        t::SCALE => 1,
        t::OUTLINE_OVER => 4,
        t::FRESNEL => 4,
        t::SATURATE => 1,
        t::BLOOM => 2,
        t::POSTERIZE_COLOR => 1,
        t::VIGNETTE => 2,
        t::PALETTE3 => 10,
        t::PALETTE5 => 16,
        t::HATCH => 6,
        t::TONEMAP => 1,
        t::SPEED_LINE => 7,
        _ => return None,
    };
    Some(n)
}

const fn palette_source_to_tag(src: PaletteSource) -> u32 {
    use gpu_palette_source_tag as g;
    match src {
        PaletteSource::NDotL => g::N_DOT_L,
        PaletteSource::NDotV => g::N_DOT_V,
        PaletteSource::Sdf => g::SDF,
        PaletteSource::UvY => g::UV_Y,
        PaletteSource::TimeCycle => g::TIME_CYCLE,
    }
}

const fn palette_source_from_tag(
    tag: u32,
    word_offset: usize,
) -> Result<PaletteSource, DeserializeError> {
    use gpu_palette_source_tag as g;
    match tag {
        g::N_DOT_L => Ok(PaletteSource::NDotL),
        g::N_DOT_V => Ok(PaletteSource::NDotV),
        g::SDF => Ok(PaletteSource::Sdf),
        g::UV_Y => Ok(PaletteSource::UvY),
        g::TIME_CYCLE => Ok(PaletteSource::TimeCycle),
        _ => Err(DeserializeError::UnknownPaletteSource { word_offset, tag }),
    }
}

#[inline]
fn push_f32(words: &mut Vec<u32>, v: f32) {
    words.push(v.to_bits());
}

#[inline]
fn push_vec3(words: &mut Vec<u32>, v: Vec3) {
    words.push(v.x.to_bits());
    words.push(v.y.to_bits());
    words.push(v.z.to_bits());
}

#[inline]
const fn read_f32(words: &[u32], offset: usize) -> f32 {
    f32::from_bits(words[offset])
}

#[inline]
const fn read_vec3(words: &[u32], offset: usize) -> Vec3 {
    Vec3::new(
        f32::from_bits(words[offset]),
        f32::from_bits(words[offset + 1]),
        f32::from_bits(words[offset + 2]),
    )
}

fn decode_opcode(
    tag: u32,
    words: &[u32],
    payload_start: usize,
) -> Result<(ColorOp, usize), DeserializeError> {
    use gpu_opcode_tag as t;
    let expected = opcode_word_count(tag).ok_or(DeserializeError::UnknownOpcode {
        word_offset: payload_start - 1,
        tag,
    })?;
    if payload_start + expected > words.len() {
        return Err(DeserializeError::Truncated {
            word_offset: payload_start - 1,
            expected_payload: expected,
        });
    }
    let op = match tag {
        t::PUSH_CONSTANT => ColorOp::PushConstant(read_vec3(words, payload_start)),
        t::TOON => ColorOp::Toon {
            bands: words[payload_start],
        },
        t::SOFT_TOON => ColorOp::SoftToon {
            bands: words[payload_start],
            smoothness: read_f32(words, payload_start + 1),
        },
        t::TWO_TONE => ColorOp::TwoTone {
            threshold: read_f32(words, payload_start),
        },
        t::MULTIPLY => ColorOp::Multiply,
        t::ADD => ColorOp::Add,
        t::SCALE => ColorOp::Scale {
            factor: read_f32(words, payload_start),
        },
        t::OUTLINE_OVER => ColorOp::OutlineOver {
            outline: read_vec3(words, payload_start),
            alpha: read_f32(words, payload_start + 3),
        },
        t::FRESNEL => ColorOp::Fresnel {
            edge: read_vec3(words, payload_start),
            power: read_f32(words, payload_start + 3),
        },
        t::SATURATE => ColorOp::Saturate {
            factor: read_f32(words, payload_start),
        },
        t::BLOOM => ColorOp::Bloom {
            threshold: read_f32(words, payload_start),
            intensity: read_f32(words, payload_start + 1),
        },
        t::POSTERIZE_COLOR => ColorOp::PosterizeColor {
            levels: words[payload_start],
        },
        t::VIGNETTE => ColorOp::Vignette {
            radius: read_f32(words, payload_start),
            softness: read_f32(words, payload_start + 1),
        },
        t::PALETTE3 => ColorOp::Palette3 {
            source: palette_source_from_tag(words[payload_start], payload_start)?,
            c0: read_vec3(words, payload_start + 1),
            c1: read_vec3(words, payload_start + 4),
            c2: read_vec3(words, payload_start + 7),
        },
        t::PALETTE5 => ColorOp::Palette5 {
            source: palette_source_from_tag(words[payload_start], payload_start)?,
            c0: read_vec3(words, payload_start + 1),
            c1: read_vec3(words, payload_start + 4),
            c2: read_vec3(words, payload_start + 7),
            c3: read_vec3(words, payload_start + 10),
            c4: read_vec3(words, payload_start + 13),
        },
        t::HATCH => ColorOp::Hatch {
            angle_rad: read_f32(words, payload_start),
            density: read_f32(words, payload_start + 1),
            thickness: read_f32(words, payload_start + 2),
            ink: read_vec3(words, payload_start + 3),
        },
        t::TONEMAP => ColorOp::Tonemap {
            exposure: read_f32(words, payload_start),
        },
        t::SPEED_LINE => ColorOp::SpeedLine {
            focus: Vec2::new(
                read_f32(words, payload_start),
                read_f32(words, payload_start + 1),
            ),
            count: words[payload_start + 2],
            thickness: read_f32(words, payload_start + 3),
            ink: read_vec3(words, payload_start + 4),
        },
        // opcode_word_count above already returned None for unknown tags,
        // so this arm is unreachable in practice.
        _ => {
            return Err(DeserializeError::UnknownOpcode {
                word_offset: payload_start - 1,
                tag,
            })
        }
    };
    Ok((op, expected))
}

impl CompiledColorPipeline {
    /// Serialise the opcode stream into a GPU-uploadable [`GpuColorProgram`]
    ///
    /// The resulting `Vec<u32>` is a flat instruction stream in the format
    /// documented in this module's header. Upload the buffer via `wgpu` /
    /// `naga` and dispatch against the WGSL evaluator returned by
    /// [`emit_wgsl_bytecode_evaluator`].
    ///
    /// # Errors
    /// Returns [`SerializeError::UnsupportedFallback`] if the pipeline
    /// contains a [`ColorOp::Fallback`] opcode. Well-formed pipelines
    /// produced by [`Self::compile`] on the Phase 12-D+ DSL surface
    /// never contain `Fallback`, so this is only reachable if a caller
    /// hand-builds a pipeline that wraps a future DSL variant.
    pub fn serialize(&self) -> Result<GpuColorProgram, SerializeError> {
        use gpu_opcode_tag as t;
        let mut words: Vec<u32> = Vec::with_capacity(self.ops.len() * 4);
        for (i, op) in self.ops.iter().enumerate() {
            match op {
                ColorOp::PushConstant(c) => {
                    words.push(t::PUSH_CONSTANT);
                    push_vec3(&mut words, *c);
                }
                ColorOp::Toon { bands } => {
                    words.push(t::TOON);
                    words.push(*bands);
                }
                ColorOp::SoftToon { bands, smoothness } => {
                    words.push(t::SOFT_TOON);
                    words.push(*bands);
                    push_f32(&mut words, *smoothness);
                }
                ColorOp::TwoTone { threshold } => {
                    words.push(t::TWO_TONE);
                    push_f32(&mut words, *threshold);
                }
                ColorOp::Multiply => words.push(t::MULTIPLY),
                ColorOp::Add => words.push(t::ADD),
                ColorOp::Scale { factor } => {
                    words.push(t::SCALE);
                    push_f32(&mut words, *factor);
                }
                ColorOp::OutlineOver { outline, alpha } => {
                    words.push(t::OUTLINE_OVER);
                    push_vec3(&mut words, *outline);
                    push_f32(&mut words, *alpha);
                }
                ColorOp::Fresnel { edge, power } => {
                    words.push(t::FRESNEL);
                    push_vec3(&mut words, *edge);
                    push_f32(&mut words, *power);
                }
                ColorOp::Saturate { factor } => {
                    words.push(t::SATURATE);
                    push_f32(&mut words, *factor);
                }
                ColorOp::Bloom {
                    threshold,
                    intensity,
                } => {
                    words.push(t::BLOOM);
                    push_f32(&mut words, *threshold);
                    push_f32(&mut words, *intensity);
                }
                ColorOp::PosterizeColor { levels } => {
                    words.push(t::POSTERIZE_COLOR);
                    words.push(*levels);
                }
                ColorOp::Vignette { radius, softness } => {
                    words.push(t::VIGNETTE);
                    push_f32(&mut words, *radius);
                    push_f32(&mut words, *softness);
                }
                ColorOp::Palette3 { source, c0, c1, c2 } => {
                    words.push(t::PALETTE3);
                    words.push(palette_source_to_tag(*source));
                    push_vec3(&mut words, *c0);
                    push_vec3(&mut words, *c1);
                    push_vec3(&mut words, *c2);
                }
                ColorOp::Palette5 {
                    source,
                    c0,
                    c1,
                    c2,
                    c3,
                    c4,
                } => {
                    words.push(t::PALETTE5);
                    words.push(palette_source_to_tag(*source));
                    push_vec3(&mut words, *c0);
                    push_vec3(&mut words, *c1);
                    push_vec3(&mut words, *c2);
                    push_vec3(&mut words, *c3);
                    push_vec3(&mut words, *c4);
                }
                ColorOp::Hatch {
                    angle_rad,
                    density,
                    thickness,
                    ink,
                } => {
                    words.push(t::HATCH);
                    push_f32(&mut words, *angle_rad);
                    push_f32(&mut words, *density);
                    push_f32(&mut words, *thickness);
                    push_vec3(&mut words, *ink);
                }
                ColorOp::Tonemap { exposure } => {
                    words.push(t::TONEMAP);
                    push_f32(&mut words, *exposure);
                }
                ColorOp::SpeedLine {
                    focus,
                    count,
                    thickness,
                    ink,
                } => {
                    words.push(t::SPEED_LINE);
                    push_f32(&mut words, focus.x);
                    push_f32(&mut words, focus.y);
                    words.push(*count);
                    push_f32(&mut words, *thickness);
                    push_vec3(&mut words, *ink);
                }
                ColorOp::Fallback(_) => {
                    return Err(SerializeError::UnsupportedFallback {
                        instruction_index: i,
                    });
                }
            }
        }
        Ok(GpuColorProgram { words })
    }
}

/// Emit the canonical WGSL source for the GPU bytecode evaluator
///
/// The returned string defines two host-visible entities:
///
/// - `struct AliceNprBytecodeCtx` — shading context (n_dot_l / n_dot_v /
///   sdf / uv / time)
/// - `fn alice_npr_eval_bytecode(program_len, ctx)` — stack-machine
///   evaluator returning the final `vec3<f32>` colour
///
/// # Caller contract
///
/// The evaluator is decoupled from any specific bind-group layout. The
/// caller must supply a helper function with the signature:
///
/// ```wgsl
/// fn alice_npr_load(index: u32) -> u32
/// ```
///
/// that returns the `u32` word at the given index of the bytecode
/// program. Typical implementation:
///
/// ```wgsl
/// @group(0) @binding(0) var<storage, read> alice_npr_program: array<u32>;
/// fn alice_npr_load(index: u32) -> u32 {
///     return alice_npr_program[index];
/// }
/// ```
///
/// This layer isolates the evaluator from `ptr<storage, ...>` function
/// parameters (which require the `unrestricted_pointer_parameters` WGSL
/// extension) and lets callers back the bytecode with a uniform, a
/// baked `array<u32, N>` constant, or any other source.
///
/// # Stack depth
///
/// Fixed at 32 (observed maximum for today's DSL surface is under 10;
/// 32 leaves ample headroom without wasting register pressure).
///
/// # Coverage
///
/// The evaluator implements all 17 native opcodes from
/// [`gpu_opcode_tag`]. Opcodes that need transcendental math (`Fresnel`
/// `pow`, `SpeedLine` `atan2`) invoke the corresponding WGSL builtins
/// directly. `Fallback` is not supported (rejected at serialisation
/// time) so no opcode-dispatch arm exists for it.
#[must_use]
pub fn emit_wgsl_bytecode_evaluator() -> String {
    use gpu_opcode_tag as t;
    use gpu_palette_source_tag as p;
    format!(
        r"// ALICE-SDF NPR bytecode evaluator (WGSL) — Phase 14
//
// Generated by `alice_sdf::npr::compiled_color::emit_wgsl_bytecode_evaluator`.
// Do not edit by hand; regenerate to pick up new opcodes.

const ALICE_NPR_STACK_DEPTH: u32 = 32u;

// Opcode tags (must match `gpu_opcode_tag` on the Rust side)
const ALICE_OP_PUSH_CONSTANT: u32 = {push_constant}u;
const ALICE_OP_TOON: u32 = {toon}u;
const ALICE_OP_SOFT_TOON: u32 = {soft_toon}u;
const ALICE_OP_TWO_TONE: u32 = {two_tone}u;
const ALICE_OP_MULTIPLY: u32 = {multiply}u;
const ALICE_OP_ADD: u32 = {add}u;
const ALICE_OP_SCALE: u32 = {scale}u;
const ALICE_OP_OUTLINE_OVER: u32 = {outline_over}u;
const ALICE_OP_FRESNEL: u32 = {fresnel}u;
const ALICE_OP_SATURATE: u32 = {saturate}u;
const ALICE_OP_BLOOM: u32 = {bloom}u;
const ALICE_OP_POSTERIZE_COLOR: u32 = {posterize_color}u;
const ALICE_OP_VIGNETTE: u32 = {vignette}u;
const ALICE_OP_PALETTE3: u32 = {palette3}u;
const ALICE_OP_PALETTE5: u32 = {palette5}u;
const ALICE_OP_HATCH: u32 = {hatch}u;
const ALICE_OP_TONEMAP: u32 = {tonemap}u;
const ALICE_OP_SPEED_LINE: u32 = {speed_line}u;

// PaletteSource tags
const ALICE_PS_N_DOT_L: u32 = {ps_ndotl}u;
const ALICE_PS_N_DOT_V: u32 = {ps_ndotv}u;
const ALICE_PS_SDF: u32 = {ps_sdf}u;
const ALICE_PS_UV_Y: u32 = {ps_uvy}u;
const ALICE_PS_TIME_CYCLE: u32 = {ps_time}u;

struct AliceNprBytecodeCtx {{
    n_dot_l: f32,
    n_dot_v: f32,
    sdf: f32,
    uv: vec2<f32>,
    time: f32,
}};

fn alice_npr_palette_source_scalar(source: u32, ctx: AliceNprBytecodeCtx) -> f32 {{
    if (source == ALICE_PS_N_DOT_L) {{
        return clamp(ctx.n_dot_l, 0.0, 1.0);
    }}
    if (source == ALICE_PS_N_DOT_V) {{
        return clamp(ctx.n_dot_v, 0.0, 1.0);
    }}
    if (source == ALICE_PS_SDF) {{
        return clamp(abs(ctx.sdf), 0.0, 1.0);
    }}
    if (source == ALICE_PS_UV_Y) {{
        return clamp(ctx.uv.y, 0.0, 1.0);
    }}
    // TIME_CYCLE: fract(time)
    return ctx.time - floor(ctx.time);
}}

fn alice_npr_toon_ramp(n_dot_l: f32, bands: u32) -> f32 {{
    let b = max(f32(bands), 1.0);
    let t = clamp(n_dot_l, 0.0, 1.0);
    return floor(t * b) / b;
}}

fn alice_npr_soft_toon_ramp(n_dot_l: f32, bands: u32, smoothness: f32) -> f32 {{
    let b = max(f32(bands), 1.0);
    let t = clamp(n_dot_l, 0.0, 1.0);
    let scaled = t * b;
    let idx = floor(scaled);
    let frac = scaled - idx;
    let s = clamp(smoothness, 0.0, 0.5);
    let step = smoothstep(0.5 - s, 0.5 + s, frac);
    return (idx + step) / b;
}}

fn alice_npr_palette3(t: f32, c0: vec3<f32>, c1: vec3<f32>, c2: vec3<f32>) -> vec3<f32> {{
    if (t <= 0.5) {{
        return mix(c0, c1, t * 2.0);
    }}
    return mix(c1, c2, (t - 0.5) * 2.0);
}}

fn alice_npr_palette5(
    t: f32,
    c0: vec3<f32>,
    c1: vec3<f32>,
    c2: vec3<f32>,
    c3: vec3<f32>,
    c4: vec3<f32>,
) -> vec3<f32> {{
    if (t <= 0.25) {{
        return mix(c0, c1, t * 4.0);
    }}
    if (t <= 0.5) {{
        return mix(c1, c2, (t - 0.25) * 4.0);
    }}
    if (t <= 0.75) {{
        return mix(c2, c3, (t - 0.5) * 4.0);
    }}
    return mix(c3, c4, (t - 0.75) * 4.0);
}}

fn alice_npr_vignette_mask(uv_x: f32, uv_y: f32, radius: f32, softness: f32) -> f32 {{
    let dx = uv_x - 0.5;
    let dy = uv_y - 0.5;
    let d = sqrt(dx * dx + dy * dy);
    return 1.0 - smoothstep(radius, radius + max(softness, 1e-4), d);
}}

fn alice_npr_hatch_mask(
    uv_x: f32,
    uv_y: f32,
    angle_rad: f32,
    density: f32,
    thickness: f32,
) -> f32 {{
    let c = cos(angle_rad);
    let s = sin(angle_rad);
    let u = uv_x * c + uv_y * s;
    let phase = u * density;
    let f = phase - floor(phase);
    let t = clamp(thickness, 0.0, 0.5);
    return 1.0 - smoothstep(t, t + 1e-3, abs(f - 0.5) - (0.5 - t));
}}

fn alice_npr_speed_line_mask(
    uv_x: f32,
    uv_y: f32,
    fx: f32,
    fy: f32,
    count: u32,
    thickness: f32,
) -> f32 {{
    let dx = uv_x - fx;
    let dy = uv_y - fy;
    let angle = atan2(dy, dx);
    let two_pi = 6.283185307179586;
    let phase = (angle / two_pi + 0.5) * f32(max(count, 1u));
    let f = phase - floor(phase);
    let t = clamp(thickness, 0.0, 0.5);
    return 1.0 - smoothstep(t, t + 1e-3, abs(f - 0.5) - (0.5 - t));
}}

fn alice_npr_composite_outline(base: vec3<f32>, outline: vec3<f32>, alpha: f32) -> vec3<f32> {{
    return mix(base, outline, clamp(alpha, 0.0, 1.0));
}}

fn alice_npr_saturate_toward_luma(color: vec3<f32>, factor: f32) -> vec3<f32> {{
    let luma = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(luma), color, factor);
}}

fn alice_npr_bloom(color: vec3<f32>, threshold: f32, intensity: f32) -> vec3<f32> {{
    let mx = max(max(color.x, color.y), color.z);
    if (mx > threshold) {{
        return color * intensity;
    }}
    return color;
}}

fn alice_npr_posterize(color: vec3<f32>, levels: u32) -> vec3<f32> {{
    let l = f32(max(levels, 2u));
    return floor(color * l) / l;
}}

// Stack-machine evaluator.
//
// `program_len` is the number of `u32` words available (not the number
// of opcodes). The caller must define:
//
//     fn alice_npr_load(index: u32) -> u32
//
// which returns the word at the given index from whatever binding the
// caller chose (`var<storage, read>`, `var<uniform>`, `array<u32, N>`
// baked into the shader, etc). This decouples the evaluator from any
// specific bind-group layout and avoids pointer-parameter WGSL
// extensions that are not universally enabled.
fn alice_npr_eval_bytecode(
    program_len: u32,
    ctx: AliceNprBytecodeCtx,
) -> vec3<f32> {{
    var stack: array<vec3<f32>, 32>;
    var sp: u32 = 0u;
    var pc: u32 = 0u;

    loop {{
        if (pc >= program_len) {{ break; }}
        let tag = alice_npr_load(pc);
        pc = pc + 1u;

        if (tag == ALICE_OP_PUSH_CONSTANT) {{
            let c = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc)),
                bitcast<f32>(alice_npr_load(pc + 1u)),
                bitcast<f32>(alice_npr_load(pc + 2u)),
            );
            stack[sp] = c;
            sp = sp + 1u;
            pc = pc + 3u;
        }} else if (tag == ALICE_OP_TOON) {{
            let bands = alice_npr_load(pc);
            pc = pc + 1u;
            let light = stack[sp - 1u];
            let shadow = stack[sp - 2u];
            let t = alice_npr_toon_ramp(ctx.n_dot_l, bands);
            sp = sp - 2u;
            stack[sp] = mix(shadow, light, t);
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_SOFT_TOON) {{
            let bands = alice_npr_load(pc);
            let smoothness = bitcast<f32>(alice_npr_load(pc + 1u));
            pc = pc + 2u;
            let light = stack[sp - 1u];
            let shadow = stack[sp - 2u];
            let t = alice_npr_soft_toon_ramp(ctx.n_dot_l, bands, smoothness);
            sp = sp - 2u;
            stack[sp] = mix(shadow, light, t);
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_TWO_TONE) {{
            let threshold = bitcast<f32>(alice_npr_load(pc));
            pc = pc + 1u;
            let light = stack[sp - 1u];
            let shadow = stack[sp - 2u];
            let n = clamp(ctx.n_dot_l, 0.0, 1.0);
            sp = sp - 2u;
            if (n >= threshold) {{
                stack[sp] = light;
            }} else {{
                stack[sp] = shadow;
            }}
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_MULTIPLY) {{
            let b = stack[sp - 1u];
            let a = stack[sp - 2u];
            sp = sp - 2u;
            stack[sp] = a * b;
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_ADD) {{
            let b = stack[sp - 1u];
            let a = stack[sp - 2u];
            sp = sp - 2u;
            stack[sp] = a + b;
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_SCALE) {{
            let factor = bitcast<f32>(alice_npr_load(pc));
            pc = pc + 1u;
            stack[sp - 1u] = stack[sp - 1u] * factor;
        }} else if (tag == ALICE_OP_OUTLINE_OVER) {{
            let outline = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc)),
                bitcast<f32>(alice_npr_load(pc + 1u)),
                bitcast<f32>(alice_npr_load(pc + 2u)),
            );
            let alpha = bitcast<f32>(alice_npr_load(pc + 3u));
            pc = pc + 4u;
            stack[sp - 1u] = alice_npr_composite_outline(stack[sp - 1u], outline, alpha);
        }} else if (tag == ALICE_OP_FRESNEL) {{
            let edge = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc)),
                bitcast<f32>(alice_npr_load(pc + 1u)),
                bitcast<f32>(alice_npr_load(pc + 2u)),
            );
            let power = bitcast<f32>(alice_npr_load(pc + 3u));
            pc = pc + 4u;
            let ndv = clamp(ctx.n_dot_v, 0.0, 1.0);
            let f = pow(max(1.0 - ndv, 0.0), max(power, 0.0));
            stack[sp - 1u] = mix(stack[sp - 1u], edge, clamp(f, 0.0, 1.0));
        }} else if (tag == ALICE_OP_SATURATE) {{
            let factor = bitcast<f32>(alice_npr_load(pc));
            pc = pc + 1u;
            stack[sp - 1u] = alice_npr_saturate_toward_luma(stack[sp - 1u], factor);
        }} else if (tag == ALICE_OP_BLOOM) {{
            let threshold = bitcast<f32>(alice_npr_load(pc));
            let intensity = bitcast<f32>(alice_npr_load(pc + 1u));
            pc = pc + 2u;
            stack[sp - 1u] = alice_npr_bloom(stack[sp - 1u], threshold, intensity);
        }} else if (tag == ALICE_OP_POSTERIZE_COLOR) {{
            let levels = alice_npr_load(pc);
            pc = pc + 1u;
            stack[sp - 1u] = alice_npr_posterize(stack[sp - 1u], levels);
        }} else if (tag == ALICE_OP_VIGNETTE) {{
            let radius = bitcast<f32>(alice_npr_load(pc));
            let softness = bitcast<f32>(alice_npr_load(pc + 1u));
            pc = pc + 2u;
            let mask = alice_npr_vignette_mask(ctx.uv.x, ctx.uv.y, radius, softness);
            stack[sp - 1u] = stack[sp - 1u] * mask;
        }} else if (tag == ALICE_OP_PALETTE3) {{
            let source = alice_npr_load(pc);
            let c0 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 1u)),
                bitcast<f32>(alice_npr_load(pc + 2u)),
                bitcast<f32>(alice_npr_load(pc + 3u)),
            );
            let c1 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 4u)),
                bitcast<f32>(alice_npr_load(pc + 5u)),
                bitcast<f32>(alice_npr_load(pc + 6u)),
            );
            let c2 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 7u)),
                bitcast<f32>(alice_npr_load(pc + 8u)),
                bitcast<f32>(alice_npr_load(pc + 9u)),
            );
            pc = pc + 10u;
            let t = alice_npr_palette_source_scalar(source, ctx);
            stack[sp] = alice_npr_palette3(t, c0, c1, c2);
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_PALETTE5) {{
            let source = alice_npr_load(pc);
            let c0 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 1u)),
                bitcast<f32>(alice_npr_load(pc + 2u)),
                bitcast<f32>(alice_npr_load(pc + 3u)),
            );
            let c1 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 4u)),
                bitcast<f32>(alice_npr_load(pc + 5u)),
                bitcast<f32>(alice_npr_load(pc + 6u)),
            );
            let c2 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 7u)),
                bitcast<f32>(alice_npr_load(pc + 8u)),
                bitcast<f32>(alice_npr_load(pc + 9u)),
            );
            let c3 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 10u)),
                bitcast<f32>(alice_npr_load(pc + 11u)),
                bitcast<f32>(alice_npr_load(pc + 12u)),
            );
            let c4 = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 13u)),
                bitcast<f32>(alice_npr_load(pc + 14u)),
                bitcast<f32>(alice_npr_load(pc + 15u)),
            );
            pc = pc + 16u;
            let t = alice_npr_palette_source_scalar(source, ctx);
            stack[sp] = alice_npr_palette5(t, c0, c1, c2, c3, c4);
            sp = sp + 1u;
        }} else if (tag == ALICE_OP_HATCH) {{
            let angle_rad = bitcast<f32>(alice_npr_load(pc));
            let density = bitcast<f32>(alice_npr_load(pc + 1u));
            let thickness = bitcast<f32>(alice_npr_load(pc + 2u));
            let ink = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 3u)),
                bitcast<f32>(alice_npr_load(pc + 4u)),
                bitcast<f32>(alice_npr_load(pc + 5u)),
            );
            pc = pc + 6u;
            let mask = alice_npr_hatch_mask(ctx.uv.x, ctx.uv.y, angle_rad, density, thickness);
            stack[sp - 1u] = mix(stack[sp - 1u], ink, mask);
        }} else if (tag == ALICE_OP_TONEMAP) {{
            let exposure = bitcast<f32>(alice_npr_load(pc));
            pc = pc + 1u;
            let scaled = stack[sp - 1u] * max(exposure, 0.0);
            stack[sp - 1u] = scaled / (vec3<f32>(1.0) + scaled);
        }} else if (tag == ALICE_OP_SPEED_LINE) {{
            let fx = bitcast<f32>(alice_npr_load(pc));
            let fy = bitcast<f32>(alice_npr_load(pc + 1u));
            let count = alice_npr_load(pc + 2u);
            let thickness = bitcast<f32>(alice_npr_load(pc + 3u));
            let ink = vec3<f32>(
                bitcast<f32>(alice_npr_load(pc + 4u)),
                bitcast<f32>(alice_npr_load(pc + 5u)),
                bitcast<f32>(alice_npr_load(pc + 6u)),
            );
            pc = pc + 7u;
            let mask = alice_npr_speed_line_mask(ctx.uv.x, ctx.uv.y, fx, fy, count, thickness);
            stack[sp - 1u] = mix(stack[sp - 1u], ink, mask);
        }} else {{
            // Unknown opcode: abort by breaking the loop. A well-formed
            // program never hits this branch (opcodes are validated on
            // the CPU by `CompiledColorPipeline::serialize`).
            break;
        }}
    }}

    if (sp == 0u) {{
        return vec3<f32>(0.0);
    }}
    return stack[sp - 1u];
}}
",
        push_constant = t::PUSH_CONSTANT,
        toon = t::TOON,
        soft_toon = t::SOFT_TOON,
        two_tone = t::TWO_TONE,
        multiply = t::MULTIPLY,
        add = t::ADD,
        scale = t::SCALE,
        outline_over = t::OUTLINE_OVER,
        fresnel = t::FRESNEL,
        saturate = t::SATURATE,
        bloom = t::BLOOM,
        posterize_color = t::POSTERIZE_COLOR,
        vignette = t::VIGNETTE,
        palette3 = t::PALETTE3,
        palette5 = t::PALETTE5,
        hatch = t::HATCH,
        tonemap = t::TONEMAP,
        speed_line = t::SPEED_LINE,
        ps_ndotl = p::N_DOT_L,
        ps_ndotv = p::N_DOT_V,
        ps_sdf = p::SDF,
        ps_uvy = p::UV_Y,
        ps_time = p::TIME_CYCLE,
    )
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
            mkctx(0.15, Vec2::new(0.55, 0.45), std::f32::consts::PI, 0.05),
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

    // =================================================================
    // Phase 14 — GPU bytecode serialisation
    // =================================================================

    fn assert_roundtrip_matches(node: &NprColorNode, contexts: &[NprColorContext]) {
        let pipeline = node.compile();
        let program = pipeline.serialize().expect("serialise");
        let decoded = program.deserialize().expect("deserialise");
        assert_eq!(pipeline.ops.len(), decoded.ops.len(), "ops count differs");
        for c in contexts {
            let original = pipeline.eval(c);
            let round = decoded.eval(c);
            assert_close(round, original);
        }
    }

    #[test]
    fn serialize_constant_encodes_tag_and_three_floats() {
        let node = NprColorNode::Constant(Vec3::new(0.25, 0.5, 0.75));
        let program = node.compile().serialize().expect("serialise");
        assert_eq!(program.words.len(), 4);
        assert_eq!(program.words[0], gpu_opcode_tag::PUSH_CONSTANT);
        assert_eq!(f32::from_bits(program.words[1]), 0.25);
        assert_eq!(f32::from_bits(program.words[2]), 0.5);
        assert_eq!(f32::from_bits(program.words[3]), 0.75);
    }

    #[test]
    fn serialize_toon_pipeline_word_layout() {
        let node = NprColorNode::Toon {
            shadow: Vec3::ZERO,
            light: Vec3::ONE,
            bands: 3,
        };
        let program = node.compile().serialize().expect("serialise");
        // PushConstant(shadow) 4 + PushConstant(light) 4 + Toon 2 = 10 words
        assert_eq!(program.words.len(), 10);
        assert_eq!(program.words[0], gpu_opcode_tag::PUSH_CONSTANT);
        assert_eq!(program.words[4], gpu_opcode_tag::PUSH_CONSTANT);
        assert_eq!(program.words[8], gpu_opcode_tag::TOON);
        assert_eq!(program.words[9], 3);
    }

    #[test]
    fn serialize_all_native_variants_roundtrip() {
        let contexts = [ctx(), ctx_grazing()];
        // Cover all 17 native opcodes across several trees.
        let nodes: Vec<NprColorNode> = vec![
            NprColorNode::Constant(Vec3::new(0.3, 0.6, 0.9)),
            NprColorNode::Toon {
                shadow: Vec3::new(0.1, 0.1, 0.2),
                light: Vec3::new(0.9, 0.85, 0.7),
                bands: 4,
            },
            NprColorNode::SoftToon {
                shadow: Vec3::new(0.05, 0.05, 0.1),
                light: Vec3::new(0.95, 0.9, 0.8),
                bands: 5,
                smoothness: 0.07,
            },
            NprColorNode::TwoTone {
                shadow: Vec3::new(0.2, 0.15, 0.3),
                light: Vec3::new(0.85, 0.8, 0.65),
                threshold: 0.55,
            },
            NprColorNode::Constant(Vec3::splat(0.5))
                .multiply(NprColorNode::Constant(Vec3::new(0.4, 0.6, 0.8))),
            NprColorNode::Constant(Vec3::splat(0.3))
                .plus(NprColorNode::Constant(Vec3::new(0.1, 0.2, 0.4))),
            NprColorNode::Constant(Vec3::splat(0.6)).scale(0.85),
            NprColorNode::Constant(Vec3::splat(0.4)).with_outline(Vec3::ZERO, 0.7),
            NprColorNode::Constant(Vec3::splat(0.5)).with_fresnel(Vec3::new(1.0, 0.9, 0.6), 2.5),
            NprColorNode::Constant(Vec3::new(0.6, 0.4, 0.7)).saturate(1.3),
            NprColorNode::Constant(Vec3::new(0.7, 0.5, 0.6)).bloom(0.4, 1.15),
            NprColorNode::Constant(Vec3::new(0.55, 0.65, 0.75)).posterize(5),
            NprColorNode::Constant(Vec3::splat(0.8)).vignetted(0.35, 0.25),
            NprColorNode::Palette3 {
                source: PaletteSource::NDotL,
                c0: Vec3::new(0.1, 0.15, 0.3),
                c1: Vec3::new(0.5, 0.55, 0.6),
                c2: Vec3::new(0.9, 0.85, 0.7),
            },
            NprColorNode::Palette5 {
                source: PaletteSource::TimeCycle,
                c0: Vec3::new(0.10, 0.05, 0.20),
                c1: Vec3::new(0.50, 0.10, 0.30),
                c2: Vec3::new(0.90, 0.40, 0.20),
                c3: Vec3::new(0.95, 0.85, 0.50),
                c4: Vec3::new(0.80, 0.95, 0.95),
            },
            NprColorNode::Constant(Vec3::splat(0.4)).with_hatch(
                0.4,
                40.0,
                0.15,
                Vec3::new(0.05, 0.05, 0.1),
            ),
            NprColorNode::Constant(Vec3::splat(0.9)).tonemap_reinhard(1.2),
            NprColorNode::Constant(Vec3::splat(0.4)).with_speed_lines(
                Vec2::new(0.5, 0.5),
                24,
                0.03,
                Vec3::new(0.02, 0.02, 0.05),
            ),
        ];
        for node in &nodes {
            assert_roundtrip_matches(node, &contexts);
        }
    }

    #[test]
    fn serialize_deep_composition_roundtrip() {
        // Nine-level composition tree exercised end-to-end.
        let node = NprColorNode::TwoTone {
            shadow: Vec3::new(0.15, 0.13, 0.30),
            light: Vec3::new(0.92, 0.85, 0.70),
            threshold: 0.5,
        }
        .with_fresnel(Vec3::new(0.9, 0.8, 0.6), 2.0)
        .with_outline(Vec3::new(0.03, 0.03, 0.06), 0.85)
        .saturate(1.1)
        .bloom(0.35, 1.1)
        .posterize(5)
        .vignetted(0.4, 0.25)
        .with_hatch(0.35, 32.0, 0.12, Vec3::new(0.04, 0.04, 0.08))
        .tonemap_reinhard(1.15)
        .with_speed_lines(Vec2::new(0.5, 0.5), 20, 0.03, Vec3::new(0.02, 0.02, 0.04))
        .scale(0.95);
        assert_roundtrip_matches(&node, &[ctx(), ctx_grazing()]);
    }

    #[test]
    fn serialize_rejects_fallback() {
        // Hand-build a pipeline containing a Fallback opcode.
        let pipeline = CompiledColorPipeline {
            ops: vec![
                ColorOp::PushConstant(Vec3::splat(0.5)),
                ColorOp::Fallback(Box::new(NprColorNode::Constant(Vec3::splat(0.3)))),
            ],
        };
        let err = pipeline.serialize().expect_err("fallback must be rejected");
        assert_eq!(
            err,
            SerializeError::UnsupportedFallback {
                instruction_index: 1,
            }
        );
    }

    #[test]
    fn deserialize_detects_unknown_opcode() {
        let program = GpuColorProgram {
            words: vec![9999, 0, 0, 0],
        };
        let err = program.deserialize().expect_err("unknown tag");
        assert_eq!(
            err,
            DeserializeError::UnknownOpcode {
                word_offset: 0,
                tag: 9999,
            }
        );
    }

    #[test]
    fn deserialize_detects_truncated_payload() {
        // PUSH_CONSTANT expects 3 payload words; provide only 2.
        let program = GpuColorProgram {
            words: vec![gpu_opcode_tag::PUSH_CONSTANT, 0, 0],
        };
        let err = program.deserialize().expect_err("truncated");
        assert_eq!(
            err,
            DeserializeError::Truncated {
                word_offset: 0,
                expected_payload: 3,
            }
        );
    }

    #[test]
    fn deserialize_detects_unknown_palette_source() {
        // PALETTE3 with an invalid PaletteSource tag.
        let mut words = vec![gpu_opcode_tag::PALETTE3, 9999];
        words.extend(std::iter::repeat_n(0u32, 9));
        let program = GpuColorProgram { words };
        let err = program.deserialize().expect_err("unknown palette source");
        assert_eq!(
            err,
            DeserializeError::UnknownPaletteSource {
                word_offset: 1,
                tag: 9999,
            }
        );
    }

    #[test]
    fn gpu_program_byte_len_matches_word_count() {
        let node = NprColorNode::Constant(Vec3::splat(0.5));
        let program = node.compile().serialize().expect("serialise");
        assert_eq!(program.byte_len(), program.words.len() * 4);
        assert_eq!(program.as_words().len(), program.words.len());
    }

    #[test]
    fn opcode_word_count_covers_all_tags() {
        use gpu_opcode_tag as t;
        // Exhaustively cover every known tag; add a case here when a new
        // opcode is introduced so the map stays in lockstep with the enum.
        for tag in [
            t::PUSH_CONSTANT,
            t::TOON,
            t::SOFT_TOON,
            t::TWO_TONE,
            t::MULTIPLY,
            t::ADD,
            t::SCALE,
            t::OUTLINE_OVER,
            t::FRESNEL,
            t::SATURATE,
            t::BLOOM,
            t::POSTERIZE_COLOR,
            t::VIGNETTE,
            t::PALETTE3,
            t::PALETTE5,
            t::HATCH,
            t::TONEMAP,
            t::SPEED_LINE,
        ] {
            assert!(opcode_word_count(tag).is_some(), "missing tag {tag}");
        }
        assert!(opcode_word_count(9999).is_none());
    }

    #[test]
    fn palette_source_tags_roundtrip() {
        for src in [
            PaletteSource::NDotL,
            PaletteSource::NDotV,
            PaletteSource::Sdf,
            PaletteSource::UvY,
            PaletteSource::TimeCycle,
        ] {
            let tag = palette_source_to_tag(src);
            let back = palette_source_from_tag(tag, 0).expect("roundtrip");
            assert_eq!(back, src);
        }
    }

    #[test]
    fn emit_wgsl_bytecode_evaluator_includes_all_opcode_constants() {
        let src = emit_wgsl_bytecode_evaluator();
        for name in [
            "ALICE_OP_PUSH_CONSTANT",
            "ALICE_OP_TOON",
            "ALICE_OP_SOFT_TOON",
            "ALICE_OP_TWO_TONE",
            "ALICE_OP_MULTIPLY",
            "ALICE_OP_ADD",
            "ALICE_OP_SCALE",
            "ALICE_OP_OUTLINE_OVER",
            "ALICE_OP_FRESNEL",
            "ALICE_OP_SATURATE",
            "ALICE_OP_BLOOM",
            "ALICE_OP_POSTERIZE_COLOR",
            "ALICE_OP_VIGNETTE",
            "ALICE_OP_PALETTE3",
            "ALICE_OP_PALETTE5",
            "ALICE_OP_HATCH",
            "ALICE_OP_TONEMAP",
            "ALICE_OP_SPEED_LINE",
            "alice_npr_eval_bytecode",
            "AliceNprBytecodeCtx",
        ] {
            assert!(src.contains(name), "WGSL missing {name}");
        }
    }
}

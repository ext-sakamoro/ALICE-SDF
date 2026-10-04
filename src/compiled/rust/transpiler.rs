//! Bytecode → straight-line Rust.
//!
//! Every emitter below mirrors one arm of `eval_core::eval_bytecode` for
//! `R = f32` (and the law it calls in `primitives` / `operations` /
//! `modifiers` / `real`) operation for operation. Changing a law there without
//! changing the emitter here is caught by `tests/test_rust_transpiler_oracle.rs`
//! (bit comparison against `eval_compiled`).
//!
//! Author: Moroya Sakamoto

use crate::compiled::{CompileError, CompiledSdf, Instruction, OpCode};
use crate::types::SdfNode;
use std::fmt::Write;

/// Finite-difference step baked into the emitted `normal` when the caller does
/// not choose one (same value the raycaster and the FFI normal use).
pub const DEFAULT_NORMAL_EPSILON: f32 = 0.001;

/// Options for [`RustSource::transpile_with`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RustTranspileOptions {
    /// Tetrahedral finite-difference step of the emitted `normal`
    /// (the `epsilon` of [`eval_compiled_normal`](crate::compiled::eval_compiled_normal)).
    /// Must be finite and positive.
    pub normal_epsilon: f32,
    /// Path through which the emitted code reaches `alice-det-math`
    /// (default `::alice_det_math`). Change it when the consuming crate
    /// renames the dependency or re-exports it.
    pub det_math_path: String,
}

impl Default for RustTranspileOptions {
    fn default() -> Self {
        Self {
            normal_epsilon: DEFAULT_NORMAL_EPSILON,
            det_math_path: "::alice_det_math".to_string(),
        }
    }
}

impl RustTranspileOptions {
    /// Default options with a different normal epsilon.
    pub const fn with_normal_epsilon(mut self, epsilon: f32) -> Self {
        self.normal_epsilon = epsilon;
        self
    }

    /// Default options with a different `alice-det-math` path.
    pub fn with_det_math_path(mut self, path: impl Into<String>) -> Self {
        self.det_math_path = path.into();
        self
    }
}

/// Why a scene could not be emitted as Rust.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RustTranspileError {
    /// The tree does not compile to bytecode (the emitter works on bytecode).
    #[error("bytecode compilation failed: {0}")]
    Compile(#[from] CompileError),
    /// The bytecode contains an opcode this emitter has no law for.
    #[error("opcode {opcode:?} (instruction {index}) has no Rust emitter")]
    UnsupportedOpcode {
        /// The opcode without an emitter.
        opcode: OpCode,
        /// Its index in the instruction stream.
        index: usize,
    },
    /// `normal_epsilon` was zero, negative or not finite.
    #[error("normal epsilon must be finite and positive, got {0}")]
    InvalidNormalEpsilon(f32),
    /// The bytecode was not a well-formed stack program (pop without push).
    #[error("malformed bytecode at instruction {0}")]
    MalformedBytecode(usize),
}

/// Whether `op` has a Rust emitter.
///
/// The match is exhaustive on purpose: a new opcode is a compile error here
/// until someone decides whether it gets an emitter.
pub const fn is_supported(op: OpCode) -> bool {
    match op {
        OpCode::Sphere
        | OpCode::Box3d
        | OpCode::RoundedBox
        | OpCode::Cylinder
        | OpCode::Torus
        | OpCode::Plane
        | OpCode::Capsule
        | OpCode::Cone
        | OpCode::HexPrism
        | OpCode::Link
        | OpCode::InfiniteCylinder
        | OpCode::Gyroid
        | OpCode::Union
        | OpCode::Intersection
        | OpCode::Subtraction
        | OpCode::SmoothUnion
        | OpCode::SmoothIntersection
        | OpCode::SmoothSubtraction
        | OpCode::ChamferUnion
        | OpCode::ChamferIntersection
        | OpCode::ChamferSubtraction
        | OpCode::ExpSmoothUnion
        | OpCode::ExpSmoothIntersection
        | OpCode::ExpSmoothSubtraction
        | OpCode::XOR
        | OpCode::Morph
        | OpCode::MetricBlend
        | OpCode::Translate
        | OpCode::Rotate
        | OpCode::Scale
        | OpCode::ScaleNonUniform
        | OpCode::Twist
        | OpCode::Bend
        | OpCode::RepeatInfinite
        | OpCode::RepeatFinite
        | OpCode::Round
        | OpCode::Onion
        | OpCode::Elongate
        | OpCode::Mirror
        | OpCode::OctantMirror
        | OpCode::Revolution
        | OpCode::Extrude
        | OpCode::Taper
        | OpCode::Displacement
        | OpCode::PolarRepeat
        | OpCode::Shear
        | OpCode::Animated
        | OpCode::PopTransform
        | OpCode::End => true,
        OpCode::Ellipsoid
        | OpCode::RoundedCone
        | OpCode::Pyramid
        | OpCode::Octahedron
        | OpCode::CappedCone
        | OpCode::CappedTorus
        | OpCode::RoundedCylinder
        | OpCode::TriangularPrism
        | OpCode::CutSphere
        | OpCode::CutHollowSphere
        | OpCode::DeathStar
        | OpCode::SolidAngle
        | OpCode::Rhombus
        | OpCode::Horseshoe
        | OpCode::Vesica
        | OpCode::InfiniteCone
        | OpCode::Heart
        | OpCode::Tube
        | OpCode::Barrel
        | OpCode::Diamond
        | OpCode::ChamferedCube
        | OpCode::SchwarzP
        | OpCode::Superellipsoid
        | OpCode::RoundedX
        | OpCode::Pie
        | OpCode::Trapezoid
        | OpCode::Parallelogram
        | OpCode::Tunnel
        | OpCode::UnevenCapsule
        | OpCode::Egg
        | OpCode::ArcShape
        | OpCode::Moon
        | OpCode::CrossShape
        | OpCode::BlobbyCross
        | OpCode::ParabolaSegment
        | OpCode::RegularPolygon
        | OpCode::StarPolygon
        | OpCode::Stairs
        | OpCode::Helix
        | OpCode::Tetrahedron
        | OpCode::Dodecahedron
        | OpCode::Icosahedron
        | OpCode::TruncatedOctahedron
        | OpCode::TruncatedIcosahedron
        | OpCode::BoxFrame
        | OpCode::DiamondSurface
        | OpCode::Neovius
        | OpCode::Lidinoid
        | OpCode::IWP
        | OpCode::FRD
        | OpCode::FischerKochS
        | OpCode::PMY
        | OpCode::Circle2D
        | OpCode::Rect2D
        | OpCode::Segment2D
        | OpCode::Polygon2D
        | OpCode::RoundedRect2D
        | OpCode::Annular2D
        | OpCode::MetricBall
        | OpCode::StairsUnion
        | OpCode::StairsIntersection
        | OpCode::StairsSubtraction
        | OpCode::ColumnsUnion
        | OpCode::ColumnsIntersection
        | OpCode::ColumnsSubtraction
        | OpCode::Pipe
        | OpCode::Engrave
        | OpCode::Groove
        | OpCode::Tongue
        | OpCode::ProjectiveTransform
        | OpCode::LatticeDeform
        | OpCode::SdfSkinning
        | OpCode::Noise
        | OpCode::SweepBezier
        | OpCode::IcosahedralSymmetry
        | OpCode::IFS
        | OpCode::HeightmapDisplacement
        | OpCode::SurfaceRoughness => false,
    }
}

/// Emitted Rust source for one scene.
#[derive(Debug, Clone)]
pub struct RustSource {
    source: String,
    instruction_count: usize,
}

impl RustSource {
    /// Emit `node` with [`RustTranspileOptions::default`].
    pub fn transpile(node: &SdfNode) -> Result<Self, RustTranspileError> {
        Self::transpile_with(node, &RustTranspileOptions::default())
    }

    /// Emit `node` with explicit options.
    pub fn transpile_with(
        node: &SdfNode,
        options: &RustTranspileOptions,
    ) -> Result<Self, RustTranspileError> {
        let compiled = CompiledSdf::try_compile(node)?;
        Self::transpile_compiled(&compiled, options)
    }

    /// Emit an already compiled scene (the bytecode `eval_compiled` runs).
    pub fn transpile_compiled(
        compiled: &CompiledSdf,
        options: &RustTranspileOptions,
    ) -> Result<Self, RustTranspileError> {
        let eps = options.normal_epsilon;
        if !(eps.is_finite() && eps > 0.0) {
            return Err(RustTranspileError::InvalidNormalEpsilon(eps));
        }
        let body = Emitter::new(&options.det_math_path).run(&compiled.instructions)?;
        let mut source = String::new();
        let _ = writeln!(
            source,
            "// Generated by alice-sdf {} (compiled::rust). Do not edit.\n\
             // Bit-identical to alice_sdf::compiled::eval_compiled for this scene.",
            env!("CARGO_PKG_VERSION")
        );
        source.push_str(
            "\n/// Signed distance of the scene at `(x, y, z)`.\n\
             #[allow(clippy::all, clippy::pedantic, clippy::nursery, unused_parens, unused_variables, non_snake_case)]\n\
             #[inline]\n\
             pub fn sdf(x: f32, y: f32, z: f32) -> f32 {\n",
        );
        source.push_str(&body);
        source.push_str("}\n");
        let e = lit(eps);
        let _ = write!(
            source,
            "\n/// Unit normal at `(x, y, z)`: tetrahedral finite differences with step {e}\n\
             /// (same samples and normalisation as `eval_compiled_normal`).\n\
             #[allow(clippy::all, clippy::pedantic, clippy::nursery, unused_parens)]\n\
             #[inline]\n\
             pub fn normal(x: f32, y: f32, z: f32) -> (f32, f32, f32) {{\n\
             \x20   let e: f32 = {e};\n\
             \x20   let v0 = sdf(x + e, y + (-e), z + (-e));\n\
             \x20   let v1 = sdf(x + (-e), y + (-e), z + e);\n\
             \x20   let v2 = sdf(x + (-e), y + e, z + (-e));\n\
             \x20   let v3 = sdf(x + e, y + e, z + e);\n\
             \x20   let nx = v0 - v1 - v2 + v3;\n\
             \x20   let ny = -v0 - v1 + v2 + v3;\n\
             \x20   let nz = -v0 + v1 - v2 + v3;\n\
             \x20   let inv = 1.0_f32 / ((nx * nx) + (ny * ny) + (nz * nz)).sqrt();\n\
             \x20   (nx * inv, ny * inv, nz * inv)\n\
             }}\n"
        );
        Ok(Self {
            source,
            instruction_count: compiled.instructions.len(),
        })
    }

    /// The emitted source: `pub fn sdf` and `pub fn normal`, no `use` lines,
    /// suitable for `include!` inside a module.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Number of bytecode instructions the source unrolls.
    pub const fn instruction_count(&self) -> usize {
        self.instruction_count
    }
}

/// A Rust `f32` expression with exactly the bits of `v`.
fn lit(v: f32) -> String {
    if v.is_nan() {
        format!("f32::from_bits({:#010x})", v.to_bits())
    } else if v == f32::INFINITY {
        "f32::INFINITY".to_string()
    } else if v == f32::NEG_INFINITY {
        "f32::NEG_INFINITY".to_string()
    } else if v.is_sign_negative() {
        // `{:?}` is the shortest round-trip form; parenthesised so that
        // `a - (-0.5_f32)` and method calls on it parse as intended.
        format!("({v:?}_f32)")
    } else {
        format!("{v:?}_f32")
    }
}

/// A point held in three local variables.
#[derive(Clone)]
struct P {
    x: String,
    y: String,
    z: String,
}

/// One entry of the emit-time coordinate stack (mirrors `Frame`).
struct Frame {
    saved: P,
    inst: usize,
    lane: Option<String>,
}

struct Emitter<'a> {
    out: String,
    n: usize,
    dm: &'a str,
}

impl<'a> Emitter<'a> {
    const fn new(dm: &'a str) -> Self {
        Self {
            out: String::new(),
            n: 0,
            dm,
        }
    }

    /// `let tN: f32 = expr;` → `tN`
    fn let_(&mut self, expr: &str) -> String {
        let name = format!("t{}", self.n);
        self.n += 1;
        let _ = writeln!(self.out, "    let {name}: f32 = {expr};");
        name
    }

    /// `let (sN, cN) = sin_cos(arg);`
    fn emit_sin_cos(&mut self, arg: &str) -> (String, String) {
        let s = format!("s{}", self.n);
        let c = format!("c{}", self.n);
        self.n += 1;
        let _ = writeln!(
            self.out,
            "    let ({s}, {c}): (f32, f32) = {}::sin_cos({arg});",
            self.dm
        );
        (s, c)
    }

    fn point(&mut self, x: &str, y: &str, z: &str) -> P {
        P {
            x: self.let_(x),
            y: self.let_(y),
            z: self.let_(z),
        }
    }

    fn run(mut self, insts: &[Instruction]) -> Result<String, RustTranspileError> {
        let mut p = P {
            x: "x".into(),
            y: "y".into(),
            z: "z".into(),
        };
        let mut vals: Vec<String> = Vec::new();
        let mut frames: Vec<Frame> = Vec::new();

        for (pc, inst) in insts.iter().enumerate() {
            let op = inst.opcode;
            if !is_supported(op) {
                return Err(RustTranspileError::UnsupportedOpcode {
                    opcode: op,
                    index: pc,
                });
            }
            let k = &inst.params;
            match op {
                OpCode::End => break,
                OpCode::PopTransform => {
                    let frame = frames
                        .pop()
                        .ok_or(RustTranspileError::MalformedBytecode(pc))?;
                    self.pop_post(&insts[frame.inst], &frame, &mut vals, pc)?;
                    p = frame.saved;
                }
                _ if is_prefix(op) => {
                    let saved = p.clone();
                    let mut lane = None;
                    p = self.prefix(op, k, &p, &mut lane);
                    frames.push(Frame {
                        saved,
                        inst: pc,
                        lane,
                    });
                }
                _ if is_binary(op) => {
                    let b = vals
                        .pop()
                        .ok_or(RustTranspileError::MalformedBytecode(pc))?;
                    let a = vals
                        .pop()
                        .ok_or(RustTranspileError::MalformedBytecode(pc))?;
                    let r = self.bin_op(op, k, &a, &b, &p);
                    vals.push(r);
                }
                _ => {
                    let d = self.leaf(op, k, &p);
                    vals.push(d);
                }
            }
        }

        let ret = match vals.first() {
            Some(v) => v.clone(),
            None => lit(f32::MAX),
        };
        let _ = writeln!(self.out, "    {ret}");
        Ok(self.out)
    }

    /// Point transform of a frame-pushing opcode (`eval_core` prefix arms).
    fn prefix(&mut self, op: OpCode, k: &[f32; 7], p: &P, lane: &mut Option<String>) -> P {
        let (x, y, z) = (&p.x, &p.y, &p.z);
        match op {
            OpCode::Translate => self.point(
                &format!("{x} - {}", lit(k[0])),
                &format!("{y} - {}", lit(k[1])),
                &format!("{z} - {}", lit(k[2])),
            ),
            OpCode::Rotate => {
                // real::rotate_inverse: qv = -q.xyz, t = (qv × p) * 2,
                // p' = p + t * w + qv × t
                let (qx, qy, qz) = (lit(-k[0]), lit(-k[1]), lit(-k[2]));
                let w = lit(k[3]);
                let tx = self.let_(&format!("({qy} * {z} - {qz} * {y}) * 2.0_f32"));
                let ty = self.let_(&format!("({qz} * {x} - {qx} * {z}) * 2.0_f32"));
                let tz = self.let_(&format!("({qx} * {y} - {qy} * {x}) * 2.0_f32"));
                self.point(
                    &format!("{x} + {tx} * {w} + ({qy} * {tz} - {qz} * {ty})"),
                    &format!("{y} + {ty} * {w} + ({qz} * {tx} - {qx} * {tz})"),
                    &format!("{z} + {tz} * {w} + ({qx} * {ty} - {qy} * {tx})"),
                )
            }
            OpCode::Scale => {
                let s = lit(k[0]);
                self.point(
                    &format!("{x} * {s}"),
                    &format!("{y} * {s}"),
                    &format!("{z} * {s}"),
                )
            }
            OpCode::ScaleNonUniform => self.point(
                &format!("{x} * {}", lit(k[0])),
                &format!("{y} * {}", lit(k[1])),
                &format!("{z} * {}", lit(k[2])),
            ),
            OpCode::Twist => {
                // modifiers::modifier_twist (the f32 hook of Real::twist)
                let (s, c) = self.emit_sin_cos(&format!("{y} * {}", lit(k[0])));
                self.point(
                    &format!("{x} * {c} + -({z} * {s})"),
                    y,
                    &format!("{x} * {s} + ({z} * {c})"),
                )
            }
            OpCode::Bend => {
                let (s, c) = self.emit_sin_cos(&format!("{x} * {}", lit(k[0])));
                self.point(
                    &format!("{x} * {c} - {y} * {s}"),
                    &format!("{x} * {s} + {y} * {c}"),
                    z,
                )
            }
            OpCode::RepeatInfinite => {
                let axis = |v: &str, sp: f32, rc: f32| {
                    format!("{v} - ({v} * {} + 0.5_f32).floor() * {}", lit(rc), lit(sp))
                };
                let ex = axis(x, k[0], k[3]);
                let ey = axis(y, k[1], k[4]);
                let ez = axis(z, k[2], k[5]);
                self.point(&ex, &ey, &ez)
            }
            OpCode::RepeatFinite => {
                // real::repeat_finite: limit = count * 0.5, inv = 1 / spacing
                let axis = |v: &str, count: f32, sp: f32| {
                    let limit = count * 0.5;
                    let inv = 1.0 / sp;
                    format!(
                        "{v} - ({v} * {} + 0.5_f32).floor().max({}).min({}) * {}",
                        lit(inv),
                        lit(-limit),
                        lit(limit),
                        lit(sp)
                    )
                };
                let ex = axis(x, k[0], k[3]);
                let ey = axis(y, k[1], k[4]);
                let ez = axis(z, k[2], k[5]);
                self.point(&ex, &ey, &ez)
            }
            OpCode::Elongate => {
                let axis = |v: &str, a: f32| format!("{v} - {v}.max({}).min({})", lit(-a), lit(a));
                let ex = axis(x, k[0]);
                let ey = axis(y, k[1]);
                let ez = axis(z, k[2]);
                self.point(&ex, &ey, &ez)
            }
            OpCode::Mirror => {
                let axis = |v: &str, m: f32| {
                    if m != 0.0 {
                        format!("{v}.abs()")
                    } else {
                        v.to_string()
                    }
                };
                let ex = axis(x, k[0]);
                let ey = axis(y, k[1]);
                let ez = axis(z, k[2]);
                self.point(&ex, &ey, &ez)
            }
            OpCode::OctantMirror => {
                let ax = self.let_(&format!("{x}.abs()"));
                let ay = self.let_(&format!("{y}.abs()"));
                let az = self.let_(&format!("{z}.abs()"));
                let x1 = self.let_(&format!("{ax}.max({ay})"));
                let y1 = self.let_(&format!("{ax}.min({ay})"));
                let y2 = self.let_(&format!("{y1}.max({az})"));
                let z2 = self.let_(&format!("{y1}.min({az})"));
                let x3 = self.let_(&format!("{x1}.max({y2})"));
                let y3 = self.let_(&format!("{x1}.min({y2})"));
                P {
                    x: x3,
                    y: y3,
                    z: z2,
                }
            }
            OpCode::Revolution => self.point(
                &format!("({x} * {x} + {z} * {z}).sqrt() - {}", lit(k[0])),
                y,
                "0.0_f32",
            ),
            OpCode::Extrude => {
                *lane = Some(z.clone());
                P {
                    x: x.clone(),
                    y: y.clone(),
                    z: self.let_("0.0_f32"),
                }
            }
            OpCode::Taper => {
                // real::taper
                let den = self.let_(&format!("1.0_f32 - {y} * {}", lit(k[0])));
                let mag = self.let_(&format!("{den}.abs().max(1e-6_f32)"));
                let s = self.let_(&format!(
                    "1.0_f32 / (if {den} < 0.0_f32 {{ -{mag} }} else {{ {mag} }})"
                ));
                self.point(&format!("{x} * {s}"), y, &format!("{z} * {s}"))
            }
            OpCode::PolarRepeat => {
                let a = self.let_(&format!("{}::atan2({z}, {x})", self.dm));
                let r = self.let_(&format!("({x} * {x} + {z} * {z}).sqrt()"));
                let ang = self.let_(&format!(
                    "{a} - ({a} * {} + 0.5_f32).floor() * {}",
                    lit(k[2]),
                    lit(k[1])
                ));
                let (s, c) = self.emit_sin_cos(&ang);
                self.point(&format!("{r} * {c}"), y, &format!("{r} * {s}"))
            }
            OpCode::Shear => self.point(
                x,
                &format!("{y} - {} * {x}", lit(k[0])),
                &format!("{z} - {} * {x} - {} * {y}", lit(k[1]), lit(k[2])),
            ),
            // Distance-only modifiers and Animated push a frame and keep the point
            _ => p.clone(),
        }
    }

    /// `PopTransform` post-processing keyed on the pushing opcode.
    fn pop_post(
        &mut self,
        finst: &Instruction,
        frame: &Frame,
        vals: &mut [String],
        pc: usize,
    ) -> Result<(), RustTranspileError> {
        let k = &finst.params;
        let q = &frame.saved;
        let top = match finst.opcode {
            OpCode::Scale
            | OpCode::ScaleNonUniform
            | OpCode::Round
            | OpCode::Onion
            | OpCode::Extrude
            | OpCode::Displacement
            | OpCode::Taper => vals
                .last_mut()
                .ok_or(RustTranspileError::MalformedBytecode(pc))?,
            _ => return Ok(()),
        };
        let d = top.clone();
        let new = match finst.opcode {
            OpCode::Scale => self.let_(&format!("{d} * {}", lit(k[1]))),
            OpCode::ScaleNonUniform => self.let_(&format!("{d} * {}", lit(k[3]))),
            OpCode::Round => self.let_(&format!("{d} - {}", lit(k[0]))),
            OpCode::Onion => self.let_(&format!("{d}.abs() - {}", lit(k[0]))),
            OpCode::Extrude => {
                // real::extrude_distance
                let oz = frame
                    .lane
                    .clone()
                    .ok_or(RustTranspileError::MalformedBytecode(pc))?;
                let dz = self.let_(&format!("{oz}.abs() - {}", lit(k[0])));
                let wx = self.let_(&format!("{d}.max(0.0_f32)"));
                let wy = self.let_(&format!("{dz}.max(0.0_f32)"));
                self.let_(&format!(
                    "{d}.max({dz}).min(0.0_f32) + ({wx} * {wx} + {wy} * {wy}).sqrt()"
                ))
            }
            OpCode::Displacement => {
                // modifiers::modifier_sine_displacement at the frame point
                let dm = self.dm;
                self.let_(&format!(
                    "{dm}::sin({} * {}) * {dm}::sin({} * {}) * {dm}::sin({} * {}) * {} + {d}",
                    q.x,
                    lit(k[1]),
                    q.y,
                    lit(k[2]),
                    q.z,
                    lit(k[3]),
                    lit(k[0])
                ))
            }
            OpCode::Taper => self.taper_bound(&d, q, k[0], [k[1], k[2]]),
            _ => return Ok(()),
        };
        *vals
            .last_mut()
            .ok_or(RustTranspileError::MalformedBytecode(pc))? = new;
        Ok(())
    }

    /// real::taper_bound
    fn taper_bound(&mut self, d: &str, p: &P, factor: f32, reach: [f32; 2]) -> String {
        let f_abs = factor.abs();
        if f_abs == 0.0 {
            return d.to_string();
        }
        let fa = lit(f_abs);
        let (x, y, z) = (&p.x, &p.y, &p.z);
        let den = self.let_(&format!("1.0_f32 - {y} * {}", lit(factor)));
        let den_abs = self.let_(&format!("{den}.abs().max(1e-6_f32)"));
        let s = self.let_(&format!("1.0_f32 / {den_abs}"));
        let rho = self.let_(&format!("({x} * {x} + {z} * {z}).sqrt()"));
        let d_abs = self.let_(&format!("{d}.abs()"));
        let j0 = self.let_(&format!("{s}.max(1.0_f32) + {fa} * {s} * {s} * {rho}"));
        let r = self.let_(&format!("({d_abs} / {j0}).min({den_abs} / ({fa} + {fa}))"));
        let den1 = self.let_(&format!("({den_abs} - {fa} * {r}).max(1e-6_f32)"));
        let s1 = self.let_(&format!("1.0_f32 / {den1}"));
        let j1 = self.let_(&format!(
            "{s1}.max(1.0_f32) + {fa} * {s1} * {s1} * ({rho} + {r})"
        ));
        let mag = self.let_(&format!("({d_abs} / {j1}).min({r})"));
        let d_j = self.let_(&format!(
            "if {d} < 0.0_f32 {{ 0.0_f32 - {mag} }} else {{ {mag} }}"
        ));
        if reach[0] >= 1e30 || reach[1] >= 1e30 {
            return d_j;
        }
        let kk = reach[0] * f_abs;
        let inv_n = 1.0 / (kk * kk + 1.0).sqrt();
        let big_y = self.let_(&format!("{y} - {}", lit(1.0 / factor)));
        let d_cone = self.let_(&format!(
            "({rho} - {} * {big_y}.abs()) * {}",
            lit(kk),
            lit(inv_n)
        ));
        let d_slab = self.let_(&format!("{y}.abs() - {}", lit(reach[1])));
        self.let_(&format!(
            "if {d} < 0.0_f32 {{ {d_j} }} else {{ {d_j}.max({d_cone}.max({d_slab})) }}"
        ))
    }

    /// CSG binary laws (`operations::*_r`) and the point-reading MetricBlend.
    fn bin_op(&mut self, op: OpCode, k: &[f32; 7], a: &str, b: &str, p: &P) -> String {
        match op {
            OpCode::Union => self.let_(&format!("{a}.min({b})")),
            OpCode::Intersection => self.let_(&format!("{a}.max({b})")),
            OpCode::Subtraction => self.let_(&format!("{a}.max(-{b})")),
            OpCode::SmoothUnion => self.smooth(a, b, k[0], k[1], false),
            OpCode::SmoothIntersection => self.smooth(a, b, k[0], k[1], true),
            OpCode::SmoothSubtraction => {
                let nb = self.let_(&format!("-{b}"));
                self.smooth(a, &nb, k[0], k[1], true)
            }
            OpCode::ChamferUnion => self.chamfer_min(a, b, k[0]),
            OpCode::ChamferIntersection => {
                let na = self.let_(&format!("-{a}"));
                let nb = self.let_(&format!("-{b}"));
                let m = self.chamfer_min(&na, &nb, k[0]);
                self.let_(&format!("-{m}"))
            }
            OpCode::ChamferSubtraction => {
                // chamfer_max(a, -b) = -chamfer_min(-a, b)
                let na = self.let_(&format!("-{a}"));
                let nb = self.let_(&format!("-{b}"));
                let nnb = self.let_(&format!("-{nb}"));
                let m = self.chamfer_min(&na, &nnb, k[0]);
                self.let_(&format!("-{m}"))
            }
            OpCode::ExpSmoothUnion => self.exp_smooth(a, b, k[0], false),
            OpCode::ExpSmoothIntersection => self.exp_smooth(a, b, k[0], true),
            OpCode::ExpSmoothSubtraction => {
                let nb = self.let_(&format!("-{b}"));
                self.exp_smooth(a, &nb, k[0], true)
            }
            OpCode::XOR => self.let_(&format!("{a}.min({b}).max(-{a}.max({b}))")),
            OpCode::Morph => self.let_(&format!("{a} * {} + {b} * {}", lit(1.0 - k[0]), lit(k[0]))),
            OpCode::MetricBlend => {
                // weight from the distance to the centre (glam Vec3::length order)
                let dx = self.let_(&format!("{} - {}", p.x, lit(k[0])));
                let dy = self.let_(&format!("{} - {}", p.y, lit(k[1])));
                let dz = self.let_(&format!("{} - {}", p.z, lit(k[2])));
                let t = self.let_(&format!(
                    "{}::metric::smoothstep({}, {}, ((({dx} * {dx}) + ({dy} * {dy})) + ({dz} * {dz})).sqrt())",
                    self.dm,
                    lit(k[3]),
                    lit(k[3] + k[4])
                ));
                self.let_(&format!("(1.0_f32 - {t}) * {a} + {t} * {b}"))
            }
            _ => unreachable!("is_binary covers exactly the arms above"),
        }
    }

    /// smooth_min_rk_r / smooth_max_rk_r
    fn smooth(&mut self, a: &str, b: &str, k: f32, rk: f32, max: bool) -> String {
        let h = self.let_(&format!(
            "(1.0_f32 - ({a} - {b}).abs() * {}).max(0.0_f32)",
            lit(rk)
        ));
        let q = lit(k * 0.25);
        if max {
            self.let_(&format!("{a}.max({b}) + {h} * {h} * {q}"))
        } else {
            self.let_(&format!("{a}.min({b}) - {h} * {h} * {q}"))
        }
    }

    /// chamfer_min_r
    fn chamfer_min(&mut self, a: &str, b: &str, r: f32) -> String {
        self.let_(&format!(
            "{a}.min({b}).min(({a} + {b}) * {} - {})",
            lit(std::f32::consts::FRAC_1_SQRT_2),
            lit(r)
        ))
    }

    /// sdf_exp_smooth_union_r / sdf_exp_smooth_intersection_r
    fn exp_smooth(&mut self, a: &str, b: &str, k: f32, max: bool) -> String {
        let kk = lit(k.max(1e-6));
        let dm = self.dm;
        let (m, sign) = if max {
            (format!("{a}.max({b})"), "+")
        } else {
            (format!("{a}.min({b})"), "-")
        };
        let delta = self.let_(&format!("({a} - {b}).abs()"));
        self.let_(&format!(
            "{m} {sign} {dm}::ln(1.0_f32 + {dm}::exp(-{delta} / {kk})) * {kk}"
        ))
    }

    /// Leaf primitive laws (`primitives::sdf_*`).
    fn leaf(&mut self, op: OpCode, k: &[f32; 7], p: &P) -> String {
        let (x, y, z) = (&p.x, &p.y, &p.z);
        match op {
            OpCode::Sphere => self.let_(&format!(
                "({x} * {x} + {y} * {y} + {z} * {z}).sqrt() - {}",
                lit(k[0])
            )),
            OpCode::Box3d | OpCode::RoundedBox => {
                let qx = self.let_(&format!("{x}.abs() - {}", lit(k[0])));
                let qy = self.let_(&format!("{y}.abs() - {}", lit(k[1])));
                let qz = self.let_(&format!("{z}.abs() - {}", lit(k[2])));
                let mx = self.let_(&format!("{qx}.max(0.0_f32)"));
                let my = self.let_(&format!("{qy}.max(0.0_f32)"));
                let mz = self.let_(&format!("{qz}.max(0.0_f32)"));
                let d = format!(
                    "({mx} * {mx} + {my} * {my} + {mz} * {mz}).sqrt() + {qx}.max({qy}.max({qz})).min(0.0_f32)"
                );
                if op == OpCode::RoundedBox {
                    self.let_(&format!("{d} - {}", lit(k[3])))
                } else {
                    self.let_(&d)
                }
            }
            OpCode::Cylinder => {
                let dx = self.let_(&format!("({x} * {x} + {z} * {z}).sqrt() - {}", lit(k[0])));
                let dy = self.let_(&format!("{y}.abs() - {}", lit(k[1])));
                let ox = self.let_(&format!("{dx}.max(0.0_f32)"));
                let oy = self.let_(&format!("{dy}.max(0.0_f32)"));
                self.let_(&format!(
                    "{dx}.max({dy}).min(0.0_f32) + ({ox} * {ox} + {oy} * {oy}).sqrt()"
                ))
            }
            OpCode::Torus => {
                let qx = self.let_(&format!("({x} * {x} + {z} * {z}).sqrt() - {}", lit(k[0])));
                self.let_(&format!("({qx} * {qx} + {y} * {y}).sqrt() - {}", lit(k[1])))
            }
            OpCode::Plane => self.let_(&format!(
                "{x} * {} + {y} * {} + {z} * {} - {}",
                lit(k[0]),
                lit(k[1]),
                lit(k[2]),
                lit(k[3])
            )),
            OpCode::Capsule => {
                // sdf_capsule_r: ba = b - a (constant), h = clamp(pa·ba / ba·ba, 0, 1)
                let (ax, ay, az) = (k[0], k[1], k[2]);
                let (ba_x, ba_y, ba_z) = (k[3] - ax, k[4] - ay, k[5] - az);
                let baba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z;
                let pax = self.let_(&format!("{x} - {}", lit(ax)));
                let pay = self.let_(&format!("{y} - {}", lit(ay)));
                let paz = self.let_(&format!("{z} - {}", lit(az)));
                let h = self.let_(&format!(
                    "(({pax} * {} + {pay} * {} + {paz} * {}) / {}).max(0.0_f32).min(1.0_f32)",
                    lit(ba_x),
                    lit(ba_y),
                    lit(ba_z),
                    lit(baba)
                ));
                let ux = self.let_(&format!("{pax} - {} * {h}", lit(ba_x)));
                let uy = self.let_(&format!("{pay} - {} * {h}", lit(ba_y)));
                let uz = self.let_(&format!("{paz} - {} * {h}", lit(ba_z)));
                self.let_(&format!(
                    "({ux} * {ux} + {uy} * {uy} + {uz} * {uz}).sqrt() - {}",
                    lit(k[6])
                ))
            }
            OpCode::Cone => {
                let radius = k[0];
                let h = k[1];
                let k2x = -radius;
                let k2y = h + h;
                let k2k2 = k2x * k2x + k2y * k2y;
                let qx = self.let_(&format!("({x} * {x} + {z} * {z}).sqrt()"));
                let qy = y;
                let ca_r = self.let_(&format!(
                    "if {qy} < 0.0_f32 {{ {} }} else {{ 0.0_f32 }}",
                    lit(radius)
                ));
                let ca_x = self.let_(&format!("{qx} - {qx}.min({ca_r})"));
                let ca_y = self.let_(&format!("{qy}.abs() - {}", lit(h)));
                let t = self.let_(&format!(
                    "((-{qx} * {} + ({} - {qy}) * {}) / {}).max(0.0_f32).min(1.0_f32)",
                    lit(k2x),
                    lit(h),
                    lit(k2y),
                    lit(k2k2)
                ));
                let cb_x = self.let_(&format!("{qx} + {} * {t}", lit(k2x)));
                let cb_y = self.let_(&format!("{qy} - {} + {} * {t}", lit(h), lit(k2y)));
                let s = self.let_(&format!(
                    "if {cb_x} < 0.0_f32 && {ca_y} < 0.0_f32 {{ -1.0_f32 }} else {{ 1.0_f32 }}"
                ));
                self.let_(&format!(
                    "{s} * ({ca_x} * {ca_x} + {ca_y} * {ca_y}).min({cb_x} * {cb_x} + {cb_y} * {cb_y}).sqrt()"
                ))
            }
            OpCode::HexPrism => {
                let r = k[0];
                let kz = 0.57735027_f32 * r;
                let kx = lit(-0.8660254);
                let ky = lit(0.5);
                let ax = self.let_(&format!("{x}.abs()"));
                let ay = self.let_(&format!("{y}.abs()"));
                let az = self.let_(&format!("{z}.abs()"));
                let refl = self.let_(&format!(
                    "2.0_f32 * ({kx} * {ax} + {ky} * {ay}).min(0.0_f32)"
                ));
                let px = self.let_(&format!("{ax} - {refl} * {kx}"));
                let py = self.let_(&format!("{ay} - {refl} * {ky}"));
                let dx = self.let_(&format!("{px} - {px}.max({}).min({})", lit(-kz), lit(kz)));
                let dy = self.let_(&format!("{py} - {}", lit(r)));
                let dxy = self.let_(&format!(
                    "({dx} * {dx} + {dy} * {dy}).sqrt() * (if {dy} < 0.0_f32 {{ -1.0_f32 }} else {{ 1.0_f32 }})"
                ));
                let dz = self.let_(&format!("{az} - {}", lit(k[1])));
                let ox = self.let_(&format!("{dxy}.max(0.0_f32)"));
                let oz = self.let_(&format!("{dz}.max(0.0_f32)"));
                self.let_(&format!(
                    "{dxy}.max({dz}).min(0.0_f32) + ({ox} * {ox} + {oz} * {oz}).sqrt()"
                ))
            }
            OpCode::Link => {
                let qy = self.let_(&format!("({y}.abs() - {}).max(0.0_f32)", lit(k[0])));
                let xy = self.let_(&format!("({x} * {x} + {qy} * {qy}).sqrt() - {}", lit(k[1])));
                self.let_(&format!("({xy} * {xy} + {z} * {z}).sqrt() - {}", lit(k[2])))
            }
            OpCode::InfiniteCylinder => {
                self.let_(&format!("{}::hypot({x}, {z}) - {}", self.dm, lit(k[0])))
            }
            OpCode::Gyroid => {
                let sc = lit(k[0]);
                let sx = self.let_(&format!("{x} * {sc}"));
                let sy = self.let_(&format!("{y} * {sc}"));
                let sz = self.let_(&format!("{z} * {sc}"));
                let dm = self.dm;
                self.let_(&format!(
                    "({dm}::sin({sz}) * {dm}::cos({sx}) + ({dm}::sin({sx}) * {dm}::cos({sy}) + ({dm}::sin({sy}) * {dm}::cos({sz})))).abs() / {sc} - {}",
                    lit(k[1])
                ))
            }
            _ => unreachable!("leaf covers every supported non-prefix non-binary opcode"),
        }
    }
}

/// Opcodes that push a coordinate frame (closed by `PopTransform`).
const fn is_prefix(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Translate
            | OpCode::Rotate
            | OpCode::Scale
            | OpCode::ScaleNonUniform
            | OpCode::Twist
            | OpCode::Bend
            | OpCode::RepeatInfinite
            | OpCode::RepeatFinite
            | OpCode::Round
            | OpCode::Onion
            | OpCode::Elongate
            | OpCode::Mirror
            | OpCode::OctantMirror
            | OpCode::Revolution
            | OpCode::Extrude
            | OpCode::Taper
            | OpCode::Displacement
            | OpCode::PolarRepeat
            | OpCode::Shear
            | OpCode::Animated
    )
}

/// Supported binary opcodes (pop two values, push one).
const fn is_binary(op: OpCode) -> bool {
    matches!(
        op,
        OpCode::Union
            | OpCode::Intersection
            | OpCode::Subtraction
            | OpCode::SmoothUnion
            | OpCode::SmoothIntersection
            | OpCode::SmoothSubtraction
            | OpCode::ChamferUnion
            | OpCode::ChamferIntersection
            | OpCode::ChamferSubtraction
            | OpCode::ExpSmoothUnion
            | OpCode::ExpSmoothIntersection
            | OpCode::ExpSmoothSubtraction
            | OpCode::XOR
            | OpCode::Morph
            | OpCode::MetricBlend
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_round_trips_bits() {
        for v in [
            0.0f32,
            -0.0,
            1.0,
            -1.5,
            1e-6,
            f32::MAX,
            f32::MIN_POSITIVE,
            0.1,
        ] {
            let l = lit(v);
            let s = l
                .trim_start_matches('(')
                .trim_end_matches(')')
                .trim_end_matches("_f32");
            assert_eq!(s.parse::<f32>().unwrap().to_bits(), v.to_bits(), "{l}");
        }
        assert_eq!(lit(f32::INFINITY), "f32::INFINITY");
        assert!(lit(f32::NAN).starts_with("f32::from_bits("));
    }

    #[test]
    fn pop_without_frame_is_malformed() {
        let mut c = CompiledSdf::compile(&SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0));
        assert_eq!(c.instructions[0].opcode, OpCode::Translate);
        c.instructions.remove(0);
        let e = RustSource::transpile_compiled(&c, &RustTranspileOptions::default()).unwrap_err();
        assert!(
            matches!(e, RustTranspileError::MalformedBytecode(1)),
            "{e:?}"
        );
    }

    #[test]
    fn unsupported_opcode_is_an_error() {
        let e = RustSource::transpile(&SdfNode::ellipsoid(1.0, 2.0, 1.0)).unwrap_err();
        assert!(matches!(
            e,
            RustTranspileError::UnsupportedOpcode {
                opcode: OpCode::Ellipsoid,
                index: 0
            }
        ));
    }
}

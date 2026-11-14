//! Compiler: SdfNode tree → CompiledSdf bytecode
//!
//! Converts the recursive Arc-based tree structure into a flat
//! instruction array for cache-efficient evaluation.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use crate::types::SdfNode;

/// Error type for SDF compilation failures.
#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    /// A primitive in the SDF tree is not supported by the bytecode compiler.
    #[error(
        "Unsupported primitive for bytecode compilation: {0}. Use eval() or transpiler instead."
    )]
    UnsupportedPrimitive(String),

    /// The SDF tree is too deep and would overflow the evaluation stack.
    #[error("SDF tree requires stack depth {required} but maximum is {limit} ({kind} stack)")]
    StackOverflow {
        /// Kind of stack that overflows ("value" or "coordinate")
        kind: &'static str,
        /// Required depth
        required: usize,
        /// Maximum allowed depth
        limit: usize,
    },
}

/// Compiled SDF representation
///
/// A flat array of instructions that can be evaluated without
/// recursion or pointer chasing. Construct it with [`CompiledSdf::compile`]
/// / [`CompiledSdf::try_compile`]; the bytecode is read through
/// [`instructions`](Self::instructions) / [`aux_data`](Self::aux_data) and
/// the compile-time facts through [`node_count`](Self::node_count) /
/// [`lipschitz`](Self::lipschitz) (fields are private since 2.0, so new
/// compile-time data can be added without a major bump).
#[derive(Clone, Debug)]
pub struct CompiledSdf {
    /// The instruction bytecode
    pub(crate) instructions: Vec<Instruction>,
    /// Auxiliary data buffer for operations whose data exceeds `Instruction::params`.
    /// Indexed by `Instruction::aux_offset` / `Instruction::aux_len`.
    pub(crate) aux_data: Vec<f32>,
    /// Original node count (for statistics)
    pub(crate) node_count: usize,
    /// Lipschitz bound of the field on its exterior, see [`Self::lipschitz`].
    pub(crate) lipschitz: f32,
}

impl CompiledSdf {
    /// The instruction bytecode (postfix, with `PopTransform` frames).
    #[inline]
    #[must_use]
    pub fn instructions(&self) -> &[Instruction] {
        &self.instructions
    }

    /// Auxiliary data for operations whose data exceeds `Instruction::params`
    /// (heightmaps, lattices, bones, polygon vertices), indexed by
    /// `Instruction::aux_offset` / `aux_len`.
    #[inline]
    #[must_use]
    pub fn aux_data(&self) -> &[f32] {
        &self.aux_data
    }

    /// Number of `SdfNode`s the bytecode was compiled from.
    #[inline]
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.node_count
    }

    /// Lipschitz bound of the field on its exterior
    /// ([`crate::interval::eval_lipschitz`] at compile time). The marchers
    /// step by `d / lipschitz`, so a compiled TPMS (√3 … 7) traces correctly
    /// with the default config; `f32::INFINITY` when the tree has no finite
    /// bound (domain repetition, taper, …), in which case they step by `d`.
    #[inline]
    #[must_use]
    pub const fn lipschitz(&self) -> f32 {
        self.lipschitz
    }

    /// Compile an SdfNode tree into bytecode.
    ///
    /// # Panics
    ///
    /// Panics if the tree contains unsupported primitives (Triangle, Bezier).
    /// Use [`try_compile`](Self::try_compile) for a non-panicking alternative.
    pub fn compile(node: &SdfNode) -> Self {
        Self::try_compile(node)
            .expect("CompiledSdf::compile() failed: unsupported primitive in SDF tree")
    }

    /// Compile an SdfNode tree into bytecode, returning an error on failure.
    ///
    /// Use this instead of [`compile`](Self::compile) when the SDF tree may contain
    /// primitives that cannot be represented in bytecode (e.g. Triangle, Bezier which
    /// exceed the params\[6\] limit) or when the tree may be too deep for the
    /// fixed-size evaluation stacks (value: 64, coordinate: 32).
    pub fn try_compile(node: &SdfNode) -> Result<Self, CompileError> {
        validate_for_compile(node)?;

        let (value_depth, coord_depth) = compute_stack_depths(node);
        if value_depth > MAX_VALUE_STACK {
            return Err(CompileError::StackOverflow {
                kind: "value",
                required: value_depth,
                limit: MAX_VALUE_STACK,
            });
        }
        if coord_depth > MAX_COORD_STACK {
            return Err(CompileError::StackOverflow {
                kind: "coordinate",
                required: coord_depth,
                limit: MAX_COORD_STACK,
            });
        }

        let mut compiler = Compiler::new();
        compiler.compile_node(node);
        if let Some(e) = compiler.refused.take() {
            return Err(e);
        }
        compiler.instructions.push(Instruction::end());

        Ok(Self {
            instructions: compiler.instructions,
            aux_data: compiler.aux_data,
            node_count: compiler.node_count,
            lipschitz: crate::interval::eval_lipschitz(node),
        })
    }

    /// Get the number of instructions
    #[inline]
    pub const fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Get memory usage in bytes
    #[inline]
    pub const fn memory_size(&self) -> usize {
        self.instructions.len() * std::mem::size_of::<Instruction>()
            + self.aux_data.len() * std::mem::size_of::<f32>()
    }
}

/// Internal compiler state
struct Compiler {
    instructions: Vec<Instruction>,
    aux_data: Vec<f32>,
    node_count: usize,
    /// The first node the bytecode has no law for (validate_for_compile refuses the same nodes
    /// before compiling; this keeps the compiler itself total: it records the refusal and emits
    /// nothing for that node, and try_compile returns the error)
    refused: Option<CompileError>,
    /// The operations of the binary nodes whose operands are being compiled, innermost last
    /// (kept here, not in the recursion's frame: an instruction is a 64-byte aligned value)
    pending_ops: Vec<Instruction>,
}

/// How the compiler continues below a node (see `Compiler::open`)
enum Step<'a> {
    /// No children
    Leaf,
    /// One child that adds no instruction of its own (material, animation, pose)
    Pass(&'a SdfNode),
    /// Two operands, then the operation `open` put on `Compiler::pending_ops`
    Binary { a: &'a SdfNode, b: &'a SdfNode },
    /// One child inside the coordinate frame opened by the instruction at `inst_idx`
    Wrap { child: &'a SdfNode, inst_idx: usize },
}

impl Compiler {
    fn new() -> Self {
        Self {
            instructions: Vec::with_capacity(256),
            aux_data: Vec::new(),
            node_count: 0,
            refused: None,
            pending_ops: Vec::new(),
        }
    }

    /// Record that the bytecode cannot represent `name` (the first refusal is kept)
    fn refuse(&mut self, name: &str) {
        if self.refused.is_none() {
            self.refused = Some(CompileError::UnsupportedPrimitive(name.to_owned()));
        }
    }

    /// Append floats to the auxiliary data buffer, returning (offset, len).
    fn push_aux(&mut self, data: &[f32]) -> (u32, u32) {
        let offset = self.aux_data.len() as u32;
        self.aux_data.extend_from_slice(data);
        (offset, data.len() as u32)
    }

    /// Compile a node and its subtree
    ///
    /// The only recursive function of the compiler: it keeps one `Step` per tree level and
    /// nothing else, so its frame stays within the budget `tests/test_compile_stack_budget.rs`
    /// holds in unoptimised and optimised builds. The per-node work (the instruction, its aux
    /// data) is in `open`, which an unoptimised build does not inline into this frame (an
    /// optimised one shares the slots of `open`'s arms and may inline it).
    fn compile_node(&mut self, node: &SdfNode) {
        match self.open(node) {
            Step::Leaf => {}
            Step::Pass(child) => self.compile_node(child),
            Step::Binary { a, b } => {
                // post-order: left, right, op
                self.compile_node(a);
                self.compile_node(b);
                self.close_binary();
            }
            Step::Wrap { child, inst_idx } => {
                self.compile_node(child);
                self.close_wrap(inst_idx);
            }
        }
    }

    /// Close a transform or modifier: pop its coordinate frame and point its skip offset past
    /// the pop
    #[cfg_attr(debug_assertions, inline(never))]
    fn close_wrap(&mut self, inst_idx: usize) {
        self.instructions.push(Instruction::pop_transform());
        self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
    }

    /// Close a binary operation: emit the operation `open` set aside for it (its operands are
    /// compiled between, and each of them takes back exactly what it set aside, so the last one
    /// set aside is this node's)
    #[cfg_attr(debug_assertions, inline(never))]
    fn close_binary(&mut self) {
        let op = self
            .pending_ops
            .pop()
            .expect("open set aside the operation of every binary node");
        self.instructions.push(op);
    }

    /// Emit what a node puts before its children and say how its children are compiled
    ///
    /// A leaf emits its instruction (or records a refusal); a transform or modifier emits its
    /// opening instruction; a binary operation emits nothing yet and hands back the instruction
    /// that follows its two operands.
    #[cfg_attr(debug_assertions, inline(never))]
    fn open<'a>(&mut self, node: &'a SdfNode) -> Step<'a> {
        self.node_count += 1;
        let mut step = Step::Leaf;

        match node {
            // === Primitives ===
            SdfNode::Sphere { radius } => {
                self.instructions.push(Instruction::sphere(*radius));
            }

            SdfNode::Box3d { half_extents } => {
                self.instructions.push(Instruction::box3d(
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                ));
            }

            SdfNode::Cylinder {
                radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::cylinder(*radius, *half_height));
            }

            SdfNode::Torus {
                major_radius,
                minor_radius,
            } => {
                self.instructions
                    .push(Instruction::torus(*major_radius, *minor_radius));
            }

            SdfNode::Plane { normal, distance } => {
                self.instructions
                    .push(Instruction::plane(normal.x, normal.y, normal.z, *distance));
            }

            SdfNode::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                self.instructions.push(Instruction::capsule(
                    point_a.x, point_a.y, point_a.z, point_b.x, point_b.y, point_b.z, *radius,
                ));
            }

            SdfNode::Cone {
                radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::cone(*radius, *half_height));
            }

            SdfNode::Ellipsoid { radii } => {
                self.instructions
                    .push(Instruction::ellipsoid(radii.x, radii.y, radii.z));
            }

            SdfNode::RoundedCone {
                r1,
                r2,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::rounded_cone(*r1, *r2, *half_height));
            }

            SdfNode::Pyramid { half_height } => {
                self.instructions.push(Instruction::pyramid(*half_height));
            }

            SdfNode::Octahedron { size } => {
                self.instructions.push(Instruction::octahedron(*size));
            }

            SdfNode::HexPrism {
                hex_radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::hex_prism(*hex_radius, *half_height));
            }

            SdfNode::Link {
                half_length,
                r1,
                r2,
            } => {
                self.instructions
                    .push(Instruction::link(*half_length, *r1, *r2));
            }

            SdfNode::Triangle { .. } => self.refuse("Triangle"),

            SdfNode::SectionStack(_) => self.refuse("SectionStack"),
            SdfNode::SmoothSectionStack(_) => self.refuse("SmoothSectionStack"),

            SdfNode::Bezier { .. } => self.refuse("Bezier"),

            // === Extended Primitives (38 new) ===
            SdfNode::RoundedBox {
                half_extents,
                round_radius,
            } => {
                self.instructions.push(Instruction::rounded_box(
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                    *round_radius,
                ));
            }

            SdfNode::CappedCone {
                half_height,
                r1,
                r2,
            } => {
                self.instructions
                    .push(Instruction::capped_cone(*half_height, *r1, *r2));
            }

            SdfNode::CappedTorus {
                major_radius,
                minor_radius,
                cap_angle,
            } => {
                self.instructions.push(Instruction::capped_torus(
                    *major_radius,
                    *minor_radius,
                    *cap_angle,
                ));
            }

            SdfNode::RoundedCylinder {
                radius,
                round_radius,
                half_height,
            } => {
                self.instructions.push(Instruction::rounded_cylinder(
                    *radius,
                    *round_radius,
                    *half_height,
                ));
            }

            SdfNode::TriangularPrism { width, half_depth } => {
                self.instructions
                    .push(Instruction::triangular_prism(*width, *half_depth));
            }

            SdfNode::CutSphere { radius, cut_height } => {
                self.instructions
                    .push(Instruction::cut_sphere(*radius, *cut_height));
            }

            SdfNode::CutHollowSphere {
                radius,
                cut_height,
                thickness,
            } => {
                self.instructions.push(Instruction::cut_hollow_sphere(
                    *radius,
                    *cut_height,
                    *thickness,
                ));
            }

            SdfNode::DeathStar { ra, rb, d } => {
                self.instructions
                    .push(Instruction::death_star(*ra, *rb, *d));
            }

            SdfNode::SolidAngle { angle, radius } => {
                self.instructions
                    .push(Instruction::solid_angle(*angle, *radius));
            }

            SdfNode::Rhombus {
                la,
                lb,
                half_height,
                round_radius,
            } => {
                self.instructions
                    .push(Instruction::rhombus(*la, *lb, *half_height, *round_radius));
            }

            SdfNode::Horseshoe {
                angle,
                radius,
                half_length,
                width,
                thickness,
            } => {
                self.instructions.push(Instruction::horseshoe(
                    *angle,
                    *radius,
                    *half_length,
                    *width,
                    *thickness,
                ));
            }

            SdfNode::Vesica { radius, half_dist } => {
                self.instructions
                    .push(Instruction::vesica(*radius, *half_dist));
            }

            SdfNode::InfiniteCylinder { radius } => {
                self.instructions
                    .push(Instruction::infinite_cylinder(*radius));
            }

            SdfNode::InfiniteCone { angle } => {
                self.instructions.push(Instruction::infinite_cone(*angle));
            }

            SdfNode::Gyroid { scale, thickness } => {
                self.instructions
                    .push(Instruction::gyroid(*scale, *thickness));
            }

            SdfNode::MetricBall {
                radius,
                w_l1,
                w_l2,
                w_linf,
            } => {
                self.instructions
                    .push(Instruction::metric_ball(*radius, *w_l1, *w_l2, *w_linf));
            }

            SdfNode::MetricBlend {
                inner,
                outer,
                center,
                radius,
                skin,
            } => {
                // stack order matches the binary ops: inner first, outer second
                self.pending_ops.push(Instruction::metric_blend(
                    center.x, center.y, center.z, *radius, *skin,
                ));
                step = Step::Binary { a: inner, b: outer };
            }

            SdfNode::Heart { size } => {
                self.instructions.push(Instruction::heart(*size));
            }

            SdfNode::Tube {
                outer_radius,
                thickness,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::tube(*outer_radius, *thickness, *half_height));
            }

            SdfNode::Barrel {
                radius,
                half_height,
                bulge,
            } => {
                self.instructions
                    .push(Instruction::barrel(*radius, *half_height, *bulge));
            }

            SdfNode::Diamond {
                radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::diamond(*radius, *half_height));
            }

            SdfNode::ChamferedCube {
                half_extents,
                chamfer,
            } => {
                self.instructions.push(Instruction::chamfered_cube(
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                    *chamfer,
                ));
            }

            SdfNode::SchwarzP { scale, thickness } => {
                self.instructions
                    .push(Instruction::schwarz_p(*scale, *thickness));
            }

            SdfNode::Superellipsoid {
                half_extents,
                e1,
                e2,
            } => {
                self.instructions.push(Instruction::superellipsoid(
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                    *e1,
                    *e2,
                ));
            }

            SdfNode::RoundedX {
                width,
                round_radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::rounded_x(*width, *round_radius, *half_height));
            }

            SdfNode::Pie {
                angle,
                radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::pie(*angle, *radius, *half_height));
            }

            SdfNode::Trapezoid {
                r1,
                r2,
                trap_height,
                half_depth,
            } => {
                self.instructions
                    .push(Instruction::trapezoid(*r1, *r2, *trap_height, *half_depth));
            }

            SdfNode::Parallelogram {
                width,
                para_height,
                skew,
                half_depth,
            } => {
                self.instructions.push(Instruction::parallelogram(
                    *width,
                    *para_height,
                    *skew,
                    *half_depth,
                ));
            }

            SdfNode::Tunnel {
                width,
                height_2d,
                half_depth,
            } => {
                self.instructions
                    .push(Instruction::tunnel(*width, *height_2d, *half_depth));
            }

            SdfNode::UnevenCapsule {
                r1,
                r2,
                cap_height,
                half_depth,
            } => {
                self.instructions.push(Instruction::uneven_capsule(
                    *r1,
                    *r2,
                    *cap_height,
                    *half_depth,
                ));
            }

            SdfNode::Egg { ra, rb } => {
                self.instructions.push(Instruction::egg(*ra, *rb));
            }

            SdfNode::ArcShape {
                aperture,
                radius,
                thickness,
                half_height,
            } => {
                self.instructions.push(Instruction::arc_shape(
                    *aperture,
                    *radius,
                    *thickness,
                    *half_height,
                ));
            }

            SdfNode::Moon {
                d,
                ra,
                rb,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::moon(*d, *ra, *rb, *half_height));
            }

            SdfNode::CrossShape {
                length,
                thickness,
                round_radius,
                half_height,
            } => {
                self.instructions.push(Instruction::cross_shape(
                    *length,
                    *thickness,
                    *round_radius,
                    *half_height,
                ));
            }

            SdfNode::BlobbyCross { size, half_height } => {
                self.instructions
                    .push(Instruction::blobby_cross(*size, *half_height));
            }

            SdfNode::ParabolaSegment {
                width,
                para_height,
                half_depth,
            } => {
                self.instructions.push(Instruction::parabola_segment(
                    *width,
                    *para_height,
                    *half_depth,
                ));
            }

            SdfNode::RegularPolygon {
                radius,
                n_sides,
                half_height,
            } => {
                self.instructions.push(Instruction::regular_polygon(
                    *radius,
                    *n_sides,
                    *half_height,
                ));
            }

            SdfNode::StarPolygon {
                radius,
                n_points,
                m,
                half_height,
            } => {
                self.instructions.push(Instruction::star_polygon(
                    *radius,
                    *n_points,
                    *m,
                    *half_height,
                ));
            }

            SdfNode::Stairs {
                step_width,
                step_height,
                n_steps,
                half_depth,
            } => {
                self.instructions.push(Instruction::stairs(
                    *step_width,
                    *step_height,
                    *n_steps,
                    *half_depth,
                ));
            }

            SdfNode::Helix {
                major_r,
                minor_r,
                pitch,
                half_height,
            } => {
                self.instructions.push(Instruction::helix(
                    *major_r,
                    *minor_r,
                    *pitch,
                    *half_height,
                ));
            }

            SdfNode::Tetrahedron { size } => {
                self.instructions.push(Instruction::tetrahedron(*size));
            }

            SdfNode::Dodecahedron { radius } => {
                self.instructions.push(Instruction::dodecahedron(*radius));
            }

            SdfNode::Icosahedron { radius } => {
                self.instructions.push(Instruction::icosahedron(*radius));
            }

            SdfNode::TruncatedOctahedron { radius } => {
                self.instructions
                    .push(Instruction::truncated_octahedron(*radius));
            }

            SdfNode::TruncatedIcosahedron { radius } => {
                self.instructions
                    .push(Instruction::truncated_icosahedron(*radius));
            }

            SdfNode::BoxFrame { half_extents, edge } => {
                self.instructions
                    .push(Instruction::box_frame(*half_extents, *edge));
            }

            SdfNode::DiamondSurface { scale, thickness } => {
                self.instructions
                    .push(Instruction::diamond_surface(*scale, *thickness));
            }

            SdfNode::Neovius { scale, thickness } => {
                self.instructions
                    .push(Instruction::neovius(*scale, *thickness));
            }

            SdfNode::Lidinoid { scale, thickness } => {
                self.instructions
                    .push(Instruction::lidinoid(*scale, *thickness));
            }

            SdfNode::IWP { scale, thickness } => {
                self.instructions.push(Instruction::iwp(*scale, *thickness));
            }

            SdfNode::FRD { scale, thickness } => {
                self.instructions.push(Instruction::frd(*scale, *thickness));
            }

            SdfNode::FischerKochS { scale, thickness } => {
                self.instructions
                    .push(Instruction::fischer_koch_s(*scale, *thickness));
            }

            SdfNode::PMY { scale, thickness } => {
                self.instructions.push(Instruction::pmy(*scale, *thickness));
            }

            // === Binary Operations ===
            // For binary operations, we use post-order: left, right, op
            SdfNode::Union { a, b } => {
                self.pending_ops.push(Instruction::union());
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Intersection { a, b } => {
                self.pending_ops.push(Instruction::intersection());
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Subtraction { a, b } => {
                self.pending_ops.push(Instruction::subtraction());
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::SmoothUnion { a, b, k } => {
                self.pending_ops.push(Instruction::smooth_union(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::SmoothIntersection { a, b, k } => {
                self.pending_ops.push(Instruction::smooth_intersection(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::SmoothSubtraction { a, b, k } => {
                self.pending_ops.push(Instruction::smooth_subtraction(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ChamferUnion { a, b, r } => {
                self.pending_ops.push(Instruction::chamfer_union(*r));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ChamferIntersection { a, b, r } => {
                self.pending_ops.push(Instruction::chamfer_intersection(*r));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ChamferSubtraction { a, b, r } => {
                self.pending_ops.push(Instruction::chamfer_subtraction(*r));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::StairsUnion { a, b, r, n } => {
                self.pending_ops.push(Instruction::stairs_union(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::StairsIntersection { a, b, r, n } => {
                self.pending_ops
                    .push(Instruction::stairs_intersection(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                self.pending_ops
                    .push(Instruction::stairs_subtraction(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::XOR { a, b } => {
                self.pending_ops.push(Instruction::xor());
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Morph { a, b, t } => {
                self.pending_ops.push(Instruction::morph(*t));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ColumnsUnion { a, b, r, n } => {
                self.pending_ops.push(Instruction::columns_union(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ColumnsIntersection { a, b, r, n } => {
                self.pending_ops
                    .push(Instruction::columns_intersection(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                self.pending_ops
                    .push(Instruction::columns_subtraction(*r, *n));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Pipe { a, b, r } => {
                self.pending_ops.push(Instruction::pipe(*r));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Engrave { a, b, r } => {
                self.pending_ops.push(Instruction::engrave(*r));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Groove { a, b, ra, rb } => {
                self.pending_ops.push(Instruction::groove(*ra, *rb));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::Tongue { a, b, ra, rb } => {
                self.pending_ops.push(Instruction::tongue(*ra, *rb));
                step = Step::Binary { a: a, b: b };
            }

            // === Transforms ===
            // For transforms, we: emit transform, compile child, emit pop
            SdfNode::Translate { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::translate(offset.x, offset.y, offset.z));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Rotate { child, rotation } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::rotate(
                    rotation.x, rotation.y, rotation.z, rotation.w,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Scale { child, factor } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale(*factor));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::ScaleNonUniform { child, factors } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale_non_uniform(
                    factors.x, factors.y, factors.z,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            // === Modifiers ===
            SdfNode::Twist { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::twist(*strength));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Bend { child, curvature } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::bend(*curvature));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::repeat_infinite(
                    spacing.x, spacing.y, spacing.z,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::RepeatFinite {
                child,
                count,
                spacing,
            } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::repeat_finite(
                    count[0] as f32,
                    count[1] as f32,
                    count[2] as f32,
                    spacing.x,
                    spacing.y,
                    spacing.z,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Noise {
                child,
                amplitude,
                frequency,
                seed,
            } => {
                // Noise is a post-processing modifier that adds noise to the distance
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::noise(*amplitude, *frequency, *seed));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Round { child, radius } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::round(*radius));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Onion { child, thickness } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::onion(*thickness));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Elongate { child, amount } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::elongate(amount.x, amount.y, amount.z));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Mirror { child, axes } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::mirror(axes.x, axes.y, axes.z));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::OctantMirror { child } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::octant_mirror());
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Revolution { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::revolution(*offset));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Extrude { child, half_height } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::extrude(*half_height));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sweep_bezier(
                    p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Taper {
                child,
                factor,
                reach,
            } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::taper(*factor, *reach));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::Displacement { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::displacement(*strength));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            // Phase 27: SineDisplacement (= bytecode 経路は未実装、 raymarch は eval/mod.rs 直接経路使用)
            // 注: 現状 Displacement opcode (freq=5 固定) で fallback、 frequency 引数は無視される
            // raymarch_batch_parallel は src/eval/mod.rs::eval を使うため bytecode 経路は不要
            SdfNode::SineDisplacement {
                child,
                amplitude,
                frequency,
            } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sine_displacement(
                    *amplitude,
                    frequency.x,
                    frequency.y,
                    frequency.z,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::PolarRepeat { child, count } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::polar_repeat(*count as f32));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            // WithMaterial is transparent for distance evaluation
            SdfNode::WithMaterial { child, .. } | SdfNode::Pose { child } => {
                step = Step::Pass(child);
            }

            // === 2D Primitives ===
            SdfNode::Circle2D {
                radius,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::circle_2d(*radius, *half_height));
            }

            SdfNode::Rect2D {
                half_extents,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::rect_2d(*half_extents, *half_height));
            }

            SdfNode::Segment2D {
                a,
                b,
                thickness,
                half_height,
            } => {
                self.instructions
                    .push(Instruction::segment_2d(*a, *b, *thickness, *half_height));
            }

            SdfNode::Loft {
                axis,
                up,
                harmonics,
                rows,
                caps,
                bounds,
            } => {
                let (sec, caps) = (
                    crate::primitives::loft::LoftSection::Polar(*harmonics),
                    *caps,
                );
                let flat =
                    crate::primitives::loft::flatten(axis, *up, sec, rows, &[], caps, bounds);
                let (aux_off, aux_len) = self.push_aux(&flat);
                let mut inst = Instruction::loft();
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
            }

            SdfNode::LoftCurves {
                axis,
                up,
                section_points,
                points,
                knots,
                caps,
                bounds,
            } => {
                let sec = crate::primitives::loft::LoftSection::Curve(*section_points);
                let flat = crate::primitives::loft::flatten(
                    axis,
                    *up,
                    sec,
                    points,
                    knots,
                    [caps[0][0], caps[1][0]],
                    bounds,
                );
                let (aux_off, aux_len) = self.push_aux(&flat);
                let mut inst = Instruction::loft();
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
            }

            SdfNode::Polygon2D {
                vertices,
                half_height,
            } => {
                // Serialize vertices as flat [x0, y0, x1, y1, ...] into aux_data
                let flat: Vec<f32> = vertices.iter().flat_map(|v| [v.x, v.y]).collect();
                let (aux_off, aux_len) = self.push_aux(&flat);
                let mut inst = Instruction::polygon_2d(*half_height);
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
            }

            SdfNode::RoundedRect2D {
                half_extents,
                round_radius,
                half_height,
            } => {
                self.instructions.push(Instruction::rounded_rect_2d(
                    *half_extents,
                    *round_radius,
                    *half_height,
                ));
            }

            SdfNode::Annular2D {
                outer_radius,
                thickness,
                half_height,
            } => {
                self.instructions.push(Instruction::annular_2d(
                    *outer_radius,
                    *thickness,
                    *half_height,
                ));
            }

            // Terrain has no bytecode law yet (FBM heightfield lives only in the tree
            // evaluator); rejected by `validate_for_compile` so this arm is unreachable.
            SdfNode::Terrain { .. } => self.refuse("Terrain"),

            // === ExpSmooth operations ===
            SdfNode::ExpSmoothUnion { a, b, k } => {
                self.pending_ops.push(Instruction::exp_smooth_union(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ExpSmoothIntersection { a, b, k } => {
                self.pending_ops
                    .push(Instruction::exp_smooth_intersection(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::ExpSmoothSubtraction { a, b, k } => {
                self.pending_ops
                    .push(Instruction::exp_smooth_subtraction(*k));
                step = Step::Binary { a: a, b: b };
            }

            SdfNode::SmoothBlend {
                a,
                b,
                k,
                kernel,
                mode,
            } => {
                self.pending_ops
                    .push(Instruction::smooth_blend(*k, *kernel, *mode));
                step = Step::Binary { a: a, b: b };
            }

            // === Shear modifier (point transform) ===
            SdfNode::Shear { child, shear } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::shear(*shear));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            // === Animated modifier (passthrough) ===
            SdfNode::Animated { child, .. } => {
                step = Step::Pass(child);
            }

            // === New Transforms (7 variants) ===
            SdfNode::ProjectiveTransform {
                child,
                inv_matrix,
                lipschitz_bound,
            } => {
                let (aux_off, aux_len) = self.push_aux(inv_matrix);
                let inst_idx = self.instructions.len();
                let mut inst = Instruction::projective_transform(*lipschitz_bound);
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::LatticeDeform {
                child,
                control_points,
                nx,
                ny,
                nz,
                bbox_min,
                bbox_max,
            } => {
                // Serialize: [nx, ny, nz, bbox_min.x/y/z, bbox_max.x/y/z, cp0.x, cp0.y, cp0.z, ...]
                let mut aux = Vec::with_capacity(9 + control_points.len() * 3);
                aux.push(*nx as f32);
                aux.push(*ny as f32);
                aux.push(*nz as f32);
                aux.push(bbox_min.x);
                aux.push(bbox_min.y);
                aux.push(bbox_min.z);
                aux.push(bbox_max.x);
                aux.push(bbox_max.y);
                aux.push(bbox_max.z);
                for cp in control_points {
                    aux.push(cp.x);
                    aux.push(cp.y);
                    aux.push(cp.z);
                }
                let (aux_off, aux_len) = self.push_aux(&aux);
                let inst_idx = self.instructions.len();
                let mut inst = Instruction::lattice_deform();
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::GridWarp { .. } => self.refuse("GridWarp"),
            SdfNode::BellPull { .. } => self.refuse("BellPull"),

            SdfNode::SdfSkinning { child, bones } => {
                // Serialize: [bone_count, (inv_bind_pose[16], current_pose[16], weight) × N]
                let mut aux = Vec::with_capacity(1 + bones.len() * 33);
                aux.push(bones.len() as f32);
                for bone in bones {
                    aux.extend_from_slice(&bone.inv_bind_pose);
                    aux.extend_from_slice(&bone.current_pose);
                    aux.push(bone.weight);
                }
                let (aux_off, aux_len) = self.push_aux(&aux);
                let inst_idx = self.instructions.len();
                let mut inst = Instruction::sdf_skinning();
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            // === New Modifiers (4 variants) ===
            SdfNode::IcosahedralSymmetry { child } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::icosahedral_symmetry());
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::IFS {
                child,
                transforms,
                iterations,
            } => {
                // Serialize: [transform_count, mat4[0..16], mat4[16..32], ...]
                let mut aux = Vec::with_capacity(1 + transforms.len() * 16);
                aux.push(transforms.len() as f32);
                for t in transforms {
                    aux.extend_from_slice(t);
                }
                let (aux_off, aux_len) = self.push_aux(&aux);
                let inst_idx = self.instructions.len();
                let mut inst = Instruction::ifs(*iterations);
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::HeightmapDisplacement {
                child,
                heightmap,
                width,
                height,
                amplitude,
                scale,
            } => {
                // Serialize: [width, height, heightmap_data...]
                let mut aux = Vec::with_capacity(2 + heightmap.len());
                aux.push(*width as f32);
                aux.push(*height as f32);
                aux.extend_from_slice(heightmap);
                let (aux_off, aux_len) = self.push_aux(&aux);
                let inst_idx = self.instructions.len();
                let mut inst = Instruction::heightmap_displacement(*amplitude, *scale);
                inst.aux_offset = aux_off;
                inst.aux_len = aux_len;
                self.instructions.push(inst);
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }

            SdfNode::SurfaceRoughness {
                child,
                frequency,
                amplitude,
                octaves,
            } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::surface_roughness(
                    *frequency, *amplitude, *octaves,
                ));
                step = Step::Wrap {
                    child: child,
                    inst_idx,
                };
            }
        }
        step
    }
}

/// Maximum stack depth for value stack (must match eval.rs / eval_simd.rs).
const MAX_VALUE_STACK: usize = 64;
/// Maximum stack depth for coordinate transform stack.
const MAX_COORD_STACK: usize = 32;

/// Compute the maximum value stack depth and coordinate stack depth required
/// to evaluate the given SDF tree.
///
/// Returns `(value_depth, coord_depth)`.
pub fn compute_stack_depths(node: &SdfNode) -> (usize, usize) {
    match node {
        // Primitives push 1 value onto the value stack, no coord push
        SdfNode::Sphere { .. }
        | SdfNode::Box3d { .. }
        | SdfNode::Cylinder { .. }
        | SdfNode::Torus { .. }
        | SdfNode::Plane { .. }
        | SdfNode::Capsule { .. }
        | SdfNode::Cone { .. }
        | SdfNode::Ellipsoid { .. }
        | SdfNode::RoundedCone { .. }
        | SdfNode::Pyramid { .. }
        | SdfNode::Octahedron { .. }
        | SdfNode::HexPrism { .. }
        | SdfNode::Link { .. }
        | SdfNode::Triangle { .. }
        | SdfNode::Bezier { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        | SdfNode::Circle2D { .. }
        | SdfNode::Rect2D { .. }
        | SdfNode::Segment2D { .. }
        | SdfNode::Polygon2D { .. }
        | SdfNode::Loft { .. }
        | SdfNode::LoftCurves { .. }
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. }
        | SdfNode::Terrain { .. } => (1, 0),

        // Binary operations: compile left, then right (both on stack), then consume 2 → produce 1.
        // Peak value depth = max(left_depth, left_result + right_depth)
        // where left_result = 1 (left subtree leaves one value on stack).
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. }
        | SdfNode::SmoothBlend { a, b, .. } => {
            let (va, ca) = compute_stack_depths(a);
            let (vb, cb) = compute_stack_depths(b);
            // Left is evaluated first, leaving 1 value. Then right is evaluated on top.
            let value_depth = va.max(1 + vb);
            let coord_depth = ca.max(cb);
            (value_depth, coord_depth)
        }

        // Transforms: push 1 coord frame, evaluate child, pop coord frame.
        // Value depth = child's value depth.
        // Coord depth = 1 + child's coord depth.
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Scale { child, .. }
        | SdfNode::ScaleNonUniform { child, .. }
        | SdfNode::Twist { child, .. }
        | SdfNode::Bend { child, .. }
        | SdfNode::RepeatInfinite { child, .. }
        | SdfNode::RepeatFinite { child, .. }
        | SdfNode::Noise { child, .. }
        | SdfNode::Elongate { child, .. }
        | SdfNode::Mirror { child, .. }
        | SdfNode::OctantMirror { child }
        | SdfNode::Revolution { child, .. }
        | SdfNode::Extrude { child, .. }
        | SdfNode::SweepBezier { child, .. }
        | SdfNode::Taper { child, .. }
        | SdfNode::Displacement { child, .. }
        | SdfNode::SineDisplacement { child, .. }
        | SdfNode::PolarRepeat { child, .. }
        | SdfNode::Animated { child, .. }
        | SdfNode::Shear { child, .. }
        | SdfNode::ProjectiveTransform { child, .. }
        | SdfNode::LatticeDeform { child, .. }
        | SdfNode::SdfSkinning { child, .. }
        | SdfNode::IcosahedralSymmetry { child }
        | SdfNode::IFS { child, .. }
        | SdfNode::HeightmapDisplacement { child, .. }
        | SdfNode::SurfaceRoughness { child, .. } => {
            let (vc, cc) = compute_stack_depths(child);
            (vc, 1 + cc)
        }

        // Modifiers that don't push a coord frame (value-only modifiers)
        SdfNode::Round { child, .. }
        | SdfNode::Onion { child, .. }
        | SdfNode::WithMaterial { child, .. }
        | SdfNode::Pose { child } => compute_stack_depths(child),

        // Catch-all for future variants
        #[allow(unreachable_patterns)]
        _ => (1, 0),
    }
}

/// Recursively validate that all nodes in the tree are supported by the bytecode compiler.
fn validate_for_compile(node: &SdfNode) -> Result<(), CompileError> {
    let mut refused = None;
    validate_subtree(node, &mut refused);
    refused.map_or(Ok(()), Err)
}

/// Check `node` and its subtree, depth first (left before right); stop at the first refusal and
/// leave it in `refused`. The only recursive part: one `Children` per tree level (the checks,
/// and the error values they build, are in `validate_node`, outside this frame)
fn validate_subtree(node: &SdfNode, refused: &mut Option<CompileError>) -> bool {
    match validate_node(node) {
        Children::Refused(e) => {
            *refused = Some(e);
            false
        }
        Children::None => true,
        Children::One(child) => validate_subtree(child, refused),
        Children::Two(a, b) => validate_subtree(a, refused) && validate_subtree(b, refused),
    }
}

/// What `validate_node` found at a node: the refusal, or the children checked next
enum Children<'a> {
    Refused(CompileError),
    None,
    One(&'a SdfNode),
    Two(&'a SdfNode, &'a SdfNode),
}

/// Check one node; say which children are checked next
#[cfg_attr(debug_assertions, inline(never))]
fn validate_node(node: &SdfNode) -> Children<'_> {
    match node {
        // Metric nodes: four and five params respectively, both within the
        // instruction's seven slots, and both have a VM law.
        SdfNode::MetricBall { .. } => {}
        SdfNode::MetricBlend { inner, outer, .. } => {
            return Children::Two(inner, outer);
        }
        // Unsupported primitives (exceed params[6] limit)
        SdfNode::Triangle { .. } => {
            return Children::Refused(CompileError::UnsupportedPrimitive("Triangle".into()));
        }
        // no bytecode law yet (its levels and loops live in aux data, the next step): refused, not
        // compiled to a stand-in
        SdfNode::SectionStack(_) => {
            return Children::Refused(CompileError::UnsupportedPrimitive("SectionStack".into()));
        }
        // no bytecode law yet (the levels and loops go to aux data next): refused, not compiled to a stand-in
        SdfNode::SmoothSectionStack(_) => {
            return Children::Refused(CompileError::UnsupportedPrimitive(
                "SmoothSectionStack".into(),
            ));
        }
        SdfNode::Bezier { .. } => {
            return Children::Refused(CompileError::UnsupportedPrimitive("Bezier".into()));
        }
        // Unsupported: the grid's maps are not carried by the bytecode (evaluated by `eval` and
        // by the WGSL transpiler, which reads them from the node data buffer)
        SdfNode::GridWarp { .. } => {
            return Children::Refused(CompileError::UnsupportedPrimitive("GridWarp".into()));
        }
        // Unsupported: evaluated by `eval` and written by the shader transpilers
        SdfNode::BellPull { .. } => {
            return Children::Refused(CompileError::UnsupportedPrimitive("BellPull".into()));
        }
        // Unsupported: no bytecode law (previously compiled to a silent sphere)
        SdfNode::Terrain { .. } => {
            return Children::Refused(CompileError::UnsupportedPrimitive("Terrain".into()));
        }
        // Binary operations — validate both children
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b }
        | SdfNode::SmoothUnion { a, b, .. }
        | SdfNode::SmoothIntersection { a, b, .. }
        | SdfNode::SmoothSubtraction { a, b, .. }
        | SdfNode::ChamferUnion { a, b, .. }
        | SdfNode::ChamferIntersection { a, b, .. }
        | SdfNode::ChamferSubtraction { a, b, .. }
        | SdfNode::StairsUnion { a, b, .. }
        | SdfNode::StairsIntersection { a, b, .. }
        | SdfNode::StairsSubtraction { a, b, .. }
        | SdfNode::XOR { a, b }
        | SdfNode::Morph { a, b, .. }
        | SdfNode::ColumnsUnion { a, b, .. }
        | SdfNode::ColumnsIntersection { a, b, .. }
        | SdfNode::ColumnsSubtraction { a, b, .. }
        | SdfNode::Pipe { a, b, .. }
        | SdfNode::Engrave { a, b, .. }
        | SdfNode::Groove { a, b, .. }
        | SdfNode::Tongue { a, b, .. }
        | SdfNode::ExpSmoothUnion { a, b, .. }
        | SdfNode::ExpSmoothIntersection { a, b, .. }
        | SdfNode::ExpSmoothSubtraction { a, b, .. }
        | SdfNode::SmoothBlend { a, b, .. } => {
            return Children::Two(a, b);
        }
        // Transforms and modifiers — validate child
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Scale { child, .. }
        | SdfNode::ScaleNonUniform { child, .. }
        | SdfNode::Twist { child, .. }
        | SdfNode::Bend { child, .. }
        | SdfNode::RepeatInfinite { child, .. }
        | SdfNode::RepeatFinite { child, .. }
        | SdfNode::Noise { child, .. }
        | SdfNode::Round { child, .. }
        | SdfNode::Onion { child, .. }
        | SdfNode::Elongate { child, .. }
        | SdfNode::Mirror { child, .. }
        | SdfNode::OctantMirror { child }
        | SdfNode::Revolution { child, .. }
        | SdfNode::Extrude { child, .. }
        | SdfNode::SweepBezier { child, .. }
        | SdfNode::Taper { child, .. }
        | SdfNode::Displacement { child, .. }
        | SdfNode::SineDisplacement { child, .. }
        | SdfNode::PolarRepeat { child, .. }
        | SdfNode::WithMaterial { child, .. }
        | SdfNode::Pose { child }
        | SdfNode::Animated { child, .. }
        | SdfNode::Shear { child, .. }
        | SdfNode::ProjectiveTransform { child, .. }
        | SdfNode::LatticeDeform { child, .. }
        | SdfNode::SdfSkinning { child, .. }
        | SdfNode::IcosahedralSymmetry { child }
        | SdfNode::IFS { child, .. }
        | SdfNode::HeightmapDisplacement { child, .. }
        | SdfNode::SurfaceRoughness { child, .. } => {
            return Children::One(child);
        }
        // All supported primitives (leaf nodes, no children to validate)
        SdfNode::Sphere { .. }
        | SdfNode::Box3d { .. }
        | SdfNode::Cylinder { .. }
        | SdfNode::Torus { .. }
        | SdfNode::Plane { .. }
        | SdfNode::Capsule { .. }
        | SdfNode::Cone { .. }
        | SdfNode::Ellipsoid { .. }
        | SdfNode::RoundedCone { .. }
        | SdfNode::Pyramid { .. }
        | SdfNode::Octahedron { .. }
        | SdfNode::HexPrism { .. }
        | SdfNode::Link { .. }
        | SdfNode::RoundedBox { .. }
        | SdfNode::CappedCone { .. }
        | SdfNode::CappedTorus { .. }
        | SdfNode::RoundedCylinder { .. }
        | SdfNode::TriangularPrism { .. }
        | SdfNode::CutSphere { .. }
        | SdfNode::CutHollowSphere { .. }
        | SdfNode::DeathStar { .. }
        | SdfNode::SolidAngle { .. }
        | SdfNode::Rhombus { .. }
        | SdfNode::Horseshoe { .. }
        | SdfNode::Vesica { .. }
        | SdfNode::InfiniteCylinder { .. }
        | SdfNode::InfiniteCone { .. }
        | SdfNode::Gyroid { .. }
        | SdfNode::Heart { .. }
        | SdfNode::Tube { .. }
        | SdfNode::Barrel { .. }
        | SdfNode::Diamond { .. }
        | SdfNode::ChamferedCube { .. }
        | SdfNode::SchwarzP { .. }
        | SdfNode::Superellipsoid { .. }
        | SdfNode::RoundedX { .. }
        | SdfNode::Pie { .. }
        | SdfNode::Trapezoid { .. }
        | SdfNode::Parallelogram { .. }
        | SdfNode::Tunnel { .. }
        | SdfNode::UnevenCapsule { .. }
        | SdfNode::Egg { .. }
        | SdfNode::ArcShape { .. }
        | SdfNode::Moon { .. }
        | SdfNode::CrossShape { .. }
        | SdfNode::BlobbyCross { .. }
        | SdfNode::ParabolaSegment { .. }
        | SdfNode::RegularPolygon { .. }
        | SdfNode::StarPolygon { .. }
        | SdfNode::Stairs { .. }
        | SdfNode::Helix { .. }
        | SdfNode::Tetrahedron { .. }
        | SdfNode::Dodecahedron { .. }
        | SdfNode::Icosahedron { .. }
        | SdfNode::TruncatedOctahedron { .. }
        | SdfNode::TruncatedIcosahedron { .. }
        | SdfNode::BoxFrame { .. }
        | SdfNode::DiamondSurface { .. }
        | SdfNode::Neovius { .. }
        | SdfNode::Lidinoid { .. }
        | SdfNode::IWP { .. }
        | SdfNode::FRD { .. }
        | SdfNode::FischerKochS { .. }
        | SdfNode::PMY { .. }
        | SdfNode::Circle2D { .. }
        | SdfNode::Rect2D { .. }
        | SdfNode::Segment2D { .. }
        | SdfNode::Polygon2D { .. }
        | SdfNode::Loft { .. }
        | SdfNode::LoftCurves { .. }
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. } => {}
    }
    Children::None
}

#[cfg(test)]
mod tests {

    /// A `SmoothBlend` instruction whose kernel / mode code the compiler never writes
    /// (a corrupted program) evaluates to NaN in the scalar and the SIMD tables: no
    /// guessed kernel, no panic. Valid codes still evaluate.
    #[test]
    fn undecodable_smooth_blend_instruction_evaluates_to_nan() {
        use crate::compiled::{eval_compiled, eval_compiled_simd, Vec3x8};
        use crate::operations::{BlendMode, SmoothKernel};
        let node = SdfNode::smooth_blend(
            SdfNode::sphere(0.6),
            SdfNode::box3d(0.5, 0.4, 0.3).translate(0.4, 0.0, 0.0),
            0.3,
            SmoothKernel::Quartic,
            BlendMode::Union,
        );
        let mut c = CompiledSdf::compile(&node);
        let p = glam::Vec3::new(0.2, 0.1, 0.0);
        assert!(eval_compiled(&c, p).is_finite());
        let i = c
            .instructions
            .iter()
            .position(|x| x.opcode == OpCode::SmoothBlend)
            .expect("the program has a SmoothBlend instruction");
        for (k, m) in [
            (3.0, 0.0),
            (0.0, 3.0),
            (0.5, 0.0),
            (0.0, -1.0),
            (f32::NAN, 0.0),
        ] {
            let mut bad = c.clone();
            bad.instructions[i].params[1] = k;
            bad.instructions[i].params[2] = m;
            assert!(eval_compiled(&bad, p).is_nan(), "scalar, codes {k} / {m}");
            let simd = eval_compiled_simd(&bad, Vec3x8::splat(p)).to_array();
            assert!(simd.iter().all(|d| d.is_nan()), "simd, codes {k} / {m}");
        }
        c.instructions[i].params[1] = 2.0;
        assert!(eval_compiled(&c, p).is_finite());
    }
    use super::super::opcode::OpCode;
    use super::*;

    /// One tree of each variant validate_for_compile refuses (the bytecode has no law for them),
    /// with the name its refusal carries
    fn refused_trees() -> Vec<(&'static str, SdfNode)> {
        use glam::{Vec2, Vec3};
        let ring = |n: usize, rx: f32, ry: f32| -> Vec<Vec2> {
            (0..n)
                .map(|i| {
                    let (s, c) =
                        alice_det_math::sin_cos(i as f32 / n as f32 * std::f32::consts::TAU);
                    Vec2::new(rx * c, ry * s)
                })
                .collect()
        };
        let identity_maps: Vec<f32> = (0..27)
            .flat_map(|_| [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            .collect();
        vec![
            (
                "Triangle",
                SdfNode::Triangle {
                    point_a: Vec3::X,
                    point_b: Vec3::Y,
                    point_c: Vec3::Z,
                },
            ),
            (
                "Bezier",
                SdfNode::Bezier {
                    point_a: Vec3::ZERO,
                    point_b: Vec3::Y,
                    point_c: Vec3::X,
                    radius: 0.1,
                },
            ),
            (
                "Terrain",
                SdfNode::Terrain {
                    scale: 1.3,
                    amplitude: 0.4,
                },
            ),
            (
                "SectionStack",
                SdfNode::section_stack(
                    vec![-0.4, 0.0, 0.4],
                    vec![
                        vec![ring(16, 0.3, 0.25)],
                        vec![ring(16, 0.35, 0.3)],
                        vec![ring(16, 0.2, 0.15)],
                    ],
                )
                .expect("a valid stack"),
            ),
            (
                "SmoothSectionStack",
                SdfNode::smooth_section_stack(
                    vec![-0.45, -0.15, 0.15, 0.45],
                    vec![
                        vec![ring(10, 0.3, 0.25)],
                        vec![ring(10, 0.35, 0.3)],
                        vec![ring(10, 0.3, 0.28)],
                        vec![ring(10, 0.2, 0.15)],
                    ],
                    0.01,
                    0.1,
                )
                .expect("a valid smooth stack"),
            ),
            (
                "GridWarp",
                SdfNode::box3d(0.5, 0.5, 0.5).grid_warp(
                    crate::transforms::GridWarp::new(crate::transforms::GridWarpData {
                        origin: [-2.0; 3],
                        cell: 2.0,
                        dims: [3, 3, 3],
                        maps: identity_maps,
                        reach: 1.0,
                        floor: Vec::new(),
                    })
                    .expect("a valid grid"),
                ),
            ),
            (
                "BellPull",
                SdfNode::box3d(0.5, 0.5, 0.5).bell_pull(
                    crate::transforms::BellPull::new(
                        [[0.3, 0.0, 0.25, 0.0], [-0.2, 0.4, -0.12, 0.18]],
                        0.5,
                    )
                    .expect("one to one"),
                ),
            ),
        ]
    }

    #[test]
    fn every_refused_variant_is_an_unsupported_primitive_error() {
        let trees = refused_trees();
        assert_eq!(trees.len(), 7);
        for (name, node) in trees {
            // alone and under a union: the whole tree is refused
            for tree in [node.clone(), SdfNode::sphere(0.3).union(node)] {
                match CompiledSdf::try_compile(&tree) {
                    Err(CompileError::UnsupportedPrimitive(n)) => assert_eq!(n, name),
                    other => panic!("{name}: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn the_compiler_refuses_without_validation_and_does_not_panic() {
        // validate_for_compile bypassed: the compiler itself must refuse, not panic
        for (name, node) in refused_trees() {
            for tree in [node.clone(), SdfNode::sphere(0.3).union(node)] {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let mut c = Compiler::new();
                    c.compile_node(&tree);
                    c.refused
                }));
                let recorded = r.unwrap_or_else(|_| panic!("{name}: the compiler panicked"));
                match recorded {
                    Some(CompileError::UnsupportedPrimitive(n)) => assert_eq!(n, name),
                    other => panic!("{name}: the compiler recorded {other:?}"),
                }
            }
        }
    }

    #[test]
    fn test_compile_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        assert_eq!(compiled.node_count, 1);
        assert_eq!(compiled.instruction_count(), 2); // sphere + end
        assert_eq!(compiled.instructions[0].opcode, OpCode::Sphere);
        assert_eq!(compiled.instructions[0].params[0], 1.0);
        assert_eq!(compiled.instructions[1].opcode, OpCode::End);
    }

    #[test]
    fn test_compile_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::sphere(2.0);
        let node = a.union(b);
        let compiled = CompiledSdf::compile(&node);

        assert_eq!(compiled.node_count, 3);
        // sphere(1.0), sphere(2.0), union, end
        assert_eq!(compiled.instruction_count(), 4);
        assert_eq!(compiled.instructions[0].opcode, OpCode::Sphere);
        assert_eq!(compiled.instructions[1].opcode, OpCode::Sphere);
        assert_eq!(compiled.instructions[2].opcode, OpCode::Union);
        assert_eq!(compiled.instructions[3].opcode, OpCode::End);
    }

    #[test]
    fn test_compile_translate() {
        let node = SdfNode::sphere(1.0).translate(1.0, 2.0, 3.0);
        let compiled = CompiledSdf::compile(&node);

        // translate, sphere, pop_transform, end
        assert_eq!(compiled.instruction_count(), 4);
        assert_eq!(compiled.instructions[0].opcode, OpCode::Translate);
        assert_eq!(compiled.instructions[1].opcode, OpCode::Sphere);
        assert_eq!(compiled.instructions[2].opcode, OpCode::PopTransform);
        assert_eq!(compiled.instructions[3].opcode, OpCode::End);
    }

    #[test]
    fn test_compile_complex() {
        // sphere.subtract(box).translate
        let node = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(0.5, 0.5, 0.5))
            .translate(1.0, 0.0, 0.0);
        let compiled = CompiledSdf::compile(&node);

        assert_eq!(compiled.node_count, 4);
        // translate, sphere, box, subtract, pop_transform, end
        assert_eq!(compiled.instruction_count(), 6);
    }

    #[test]
    fn test_memory_size() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdf::compile(&node);

        // 2 instructions * 64 bytes = 128 bytes
        assert_eq!(compiled.memory_size(), 128);
    }

    /// Helper: compare compiled eval vs tree-walker eval at several points
    fn assert_roundtrip(node: &crate::types::SdfNode, tolerance: f32) {
        use crate::compiled::eval::eval_compiled;
        use crate::eval::eval;
        use glam::Vec3;

        let compiled = CompiledSdf::compile(node);
        let test_points = [
            Vec3::new(0.5, 0.3, 0.1),
            Vec3::new(-0.2, 0.8, -0.4),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        for pt in &test_points {
            let d_tree = eval(node, *pt);
            let d_compiled = eval_compiled(&compiled, *pt);
            assert!(
                (d_tree - d_compiled).abs() < tolerance,
                "Mismatch at {pt}: tree={d_tree}, compiled={d_compiled}"
            );
        }
    }

    #[test]
    fn test_roundtrip_icosahedral_symmetry() {
        let node = crate::types::SdfNode::sphere(1.0).icosahedral_symmetry();
        assert_roundtrip(&node, 1e-5);
    }

    #[test]
    fn test_roundtrip_surface_roughness() {
        let node = crate::types::SdfNode::sphere(1.0).surface_roughness(2.0, 0.05, 3);
        assert_roundtrip(&node, 1e-5);
    }

    #[test]
    fn test_roundtrip_heightmap_displacement() {
        // Simple 4x4 flat heightmap (all zeros = no displacement)
        let heightmap = vec![0.0f32; 16];
        let node =
            crate::types::SdfNode::sphere(1.0).heightmap_displacement(heightmap, 4, 4, 0.1, 1.0);
        assert_roundtrip(&node, 1e-5);
    }

    #[test]
    fn test_roundtrip_projective_transform() {
        // Identity inverse matrix
        #[rustfmt::skip]
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let node = crate::types::SdfNode::sphere(1.0).projective_transform(identity, 1.0);
        assert_roundtrip(&node, 1e-5);
    }

    #[test]
    fn test_roundtrip_ifs() {
        // Single identity transform, 1 iteration (should be close to original)
        #[rustfmt::skip]
        let identity = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let node = crate::types::SdfNode::sphere(1.0).ifs(vec![identity], 1);
        assert_roundtrip(&node, 1e-5);
    }

    #[test]
    fn test_compile_noise() {
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        let compiled = CompiledSdf::compile(&node);

        // noise, sphere, pop_transform, end
        assert_eq!(compiled.instruction_count(), 4);
        assert_eq!(compiled.instructions[0].opcode, OpCode::Noise);
        assert_eq!(compiled.instructions[0].params[0], 0.1); // amplitude
        assert_eq!(compiled.instructions[0].params[1], 2.0); // frequency
        assert_eq!(compiled.instructions[0].params[2], 42.0); // seed
        assert_eq!(compiled.instructions[1].opcode, OpCode::Sphere);
        assert_eq!(compiled.instructions[2].opcode, OpCode::PopTransform);
        assert_eq!(compiled.instructions[3].opcode, OpCode::End);
    }
}

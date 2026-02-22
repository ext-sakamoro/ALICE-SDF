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
/// recursion or pointer chasing.
#[derive(Clone, Debug)]
pub struct CompiledSdf {
    /// The instruction bytecode
    pub instructions: Vec<Instruction>,
    /// Original node count (for statistics)
    pub node_count: usize,
}

impl CompiledSdf {
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
        compiler.instructions.push(Instruction::end());

        Ok(CompiledSdf {
            instructions: compiler.instructions,
            node_count: compiler.node_count,
        })
    }

    /// Get the number of instructions
    #[inline]
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Get memory usage in bytes
    #[inline]
    pub fn memory_size(&self) -> usize {
        self.instructions.len() * std::mem::size_of::<Instruction>()
    }
}

/// Internal compiler state
struct Compiler {
    instructions: Vec<Instruction>,
    node_count: usize,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            instructions: Vec::with_capacity(256),
            node_count: 0,
        }
    }

    /// Compile a single node (recursive)
    fn compile_node(&mut self, node: &SdfNode) {
        self.node_count += 1;

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

            SdfNode::Triangle { .. } => {
                unreachable!("Triangle: validated by validate_for_compile()");
            }

            SdfNode::Bezier { .. } => {
                unreachable!("Bezier: validated by validate_for_compile()");
            }

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
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::union());
            }

            SdfNode::Intersection { a, b } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::intersection());
            }

            SdfNode::Subtraction { a, b } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::subtraction());
            }

            SdfNode::SmoothUnion { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::smooth_union(*k));
            }

            SdfNode::SmoothIntersection { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::smooth_intersection(*k));
            }

            SdfNode::SmoothSubtraction { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::smooth_subtraction(*k));
            }

            SdfNode::ChamferUnion { a, b, r } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::chamfer_union(*r));
            }

            SdfNode::ChamferIntersection { a, b, r } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::chamfer_intersection(*r));
            }

            SdfNode::ChamferSubtraction { a, b, r } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::chamfer_subtraction(*r));
            }

            SdfNode::StairsUnion { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::stairs_union(*r, *n));
            }

            SdfNode::StairsIntersection { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::stairs_intersection(*r, *n));
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::stairs_subtraction(*r, *n));
            }

            SdfNode::XOR { a, b } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::xor());
            }

            SdfNode::Morph { a, b, t } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::morph(*t));
            }

            SdfNode::ColumnsUnion { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::columns_union(*r, *n));
            }

            SdfNode::ColumnsIntersection { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::columns_intersection(*r, *n));
            }

            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::columns_subtraction(*r, *n));
            }

            SdfNode::Pipe { a, b, r } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::pipe(*r));
            }

            SdfNode::Engrave { a, b, r } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::engrave(*r));
            }

            SdfNode::Groove { a, b, ra, rb } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::groove(*ra, *rb));
            }

            SdfNode::Tongue { a, b, ra, rb } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::tongue(*ra, *rb));
            }

            // === Transforms ===
            // For transforms, we: emit transform, compile child, emit pop
            SdfNode::Translate { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::translate(offset.x, offset.y, offset.z));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                // Update skip_offset to point past the pop
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Rotate { child, rotation } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::rotate(
                    rotation.x, rotation.y, rotation.z, rotation.w,
                ));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Scale { child, factor } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale(*factor));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::ScaleNonUniform { child, factors } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale_non_uniform(
                    factors.x, factors.y, factors.z,
                ));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            // === Modifiers ===
            SdfNode::Twist { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::twist(*strength));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Bend { child, curvature } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::bend(*curvature));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::RepeatInfinite { child, spacing } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::repeat_infinite(
                    spacing.x, spacing.y, spacing.z,
                ));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
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
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
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
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Round { child, radius } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::round(*radius));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Onion { child, thickness } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::onion(*thickness));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Elongate { child, amount } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::elongate(amount.x, amount.y, amount.z));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Mirror { child, axes } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::mirror(axes.x, axes.y, axes.z));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::OctantMirror { child } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::octant_mirror());
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Revolution { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::revolution(*offset));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Extrude { child, half_height } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::extrude(*half_height));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sweep_bezier(
                    p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                ));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Taper { child, factor } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::taper(*factor));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Displacement { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::displacement(*strength));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::PolarRepeat { child, count } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::polar_repeat(*count as f32));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            // WithMaterial is transparent for distance evaluation
            SdfNode::WithMaterial { child, .. } => {
                self.compile_node(child);
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

            SdfNode::Polygon2D { half_height, .. } => {
                self.instructions
                    .push(Instruction::polygon_2d(*half_height));
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

            // === ExpSmooth operations ===
            SdfNode::ExpSmoothUnion { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::exp_smooth_union(*k));
            }

            SdfNode::ExpSmoothIntersection { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::exp_smooth_intersection(*k));
            }

            SdfNode::ExpSmoothSubtraction { a, b, k } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions
                    .push(Instruction::exp_smooth_subtraction(*k));
            }

            // === Shear modifier (point transform) ===
            SdfNode::Shear { child, shear } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::shear(*shear));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            // === Animated modifier (passthrough) ===
            SdfNode::Animated { child, .. } => {
                self.compile_node(child);
            }

            // === New Transforms (7 variants) ===
            SdfNode::ProjectiveTransform {
                child,
                lipschitz_bound,
                ..
            } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::projective_transform(*lipschitz_bound));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::LatticeDeform { child, .. } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::lattice_deform());
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::SdfSkinning { child, .. } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sdf_skinning());
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            // === New Modifiers (4 variants) ===
            SdfNode::IcosahedralSymmetry { child } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::icosahedral_symmetry());
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::IFS {
                child, iterations, ..
            } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::ifs(*iterations));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::HeightmapDisplacement {
                child,
                amplitude,
                scale,
                ..
            } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::heightmap_displacement(*amplitude, *scale));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
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
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }
        }
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
pub(crate) fn compute_stack_depths(node: &SdfNode) -> (usize, usize) {
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
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. } => (1, 0),

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
        | SdfNode::ExpSmoothSubtraction { a, b, .. } => {
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
        | SdfNode::WithMaterial { child, .. } => compute_stack_depths(child),

        // Catch-all for future variants
        #[allow(unreachable_patterns)]
        _ => (1, 0),
    }
}

/// Recursively validate that all nodes in the tree are supported by the bytecode compiler.
fn validate_for_compile(node: &SdfNode) -> Result<(), CompileError> {
    match node {
        // Unsupported primitives (exceed params[6] limit)
        SdfNode::Triangle { .. } => {
            return Err(CompileError::UnsupportedPrimitive("Triangle".into()));
        }
        SdfNode::Bezier { .. } => {
            return Err(CompileError::UnsupportedPrimitive("Bezier".into()));
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
        | SdfNode::ExpSmoothSubtraction { a, b, .. } => {
            validate_for_compile(a)?;
            validate_for_compile(b)?;
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
        | SdfNode::PolarRepeat { child, .. }
        | SdfNode::WithMaterial { child, .. }
        | SdfNode::Animated { child, .. }
        | SdfNode::Shear { child, .. }
        | SdfNode::ProjectiveTransform { child, .. }
        | SdfNode::LatticeDeform { child, .. }
        | SdfNode::SdfSkinning { child, .. }
        | SdfNode::IcosahedralSymmetry { child }
        | SdfNode::IFS { child, .. }
        | SdfNode::HeightmapDisplacement { child, .. }
        | SdfNode::SurfaceRoughness { child, .. } => {
            validate_for_compile(child)?;
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
        | SdfNode::RoundedRect2D { .. }
        | SdfNode::Annular2D { .. } => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::opcode::OpCode;
    use super::*;

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

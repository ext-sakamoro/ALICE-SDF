//! BVH-accelerated SDF evaluation with spatial pruning
//!
//! This module provides SDF evaluation that uses bounding volume hierarchies
//! to skip computation of distant objects, significantly accelerating
//! evaluation for sparse scenes.
//!
//! Author: Moroya Sakamoto

use super::aabb::{primitives as aabb_prims, AabbPacked};
use super::compiler::CompileError;
use super::instruction::Instruction;
use crate::types::SdfNode;
use glam::Vec3;

/// Compiled SDF with BVH acceleration data
///
/// Each instruction has an associated AABB for spatial pruning.
/// When the evaluation point is far from an AABB, the entire
/// subtree rooted at that instruction can be skipped.
#[derive(Clone, Debug)]
pub struct CompiledSdfBvh {
    /// The instruction bytecode
    pub instructions: Vec<Instruction>,
    /// AABB for each instruction (same length as instructions)
    pub aabbs: Vec<AabbPacked>,
    /// Original node count
    pub node_count: usize,
    /// AABB encompassing the entire scene
    pub scene_aabb: AabbPacked,
    /// Parent instruction index per instruction (same length as `instructions`).
    /// `Some(idx)` when the instruction is consumed by another instruction
    /// (child of a CSG binary op, or wrapped by a transform / modifier).
    /// `None` for the root instruction and structural markers.
    ///
    /// Populated by `try_compile` and used by
    /// `crate::compiled::refit::refit_partial` to walk the ancestor chain
    /// of dirty instructions. Empty when the BVH was constructed by a
    /// path that predates parent tracking (in which case `refit_partial`
    /// will treat the whole BVH as affected).
    pub parent_indices: Vec<Option<u32>>,
    /// End index of the subtree rooted at each instruction, inclusive.
    ///
    /// - Leaves (primitives): `subtree_end[i] == i`.
    /// - Postfix binary CSG at `i`: `subtree_end[i] == i` (children come
    ///   before in bytecode order and are covered via `parent_indices`).
    /// - Prefix transforms / modifiers at `i` (with matching `PopTransform`
    ///   at `k`): `subtree_end[i] == k`, so the whole `[i, k]` range can be
    ///   skipped when no dirty index falls in it.
    /// - `PopTransform` / `End` markers: `subtree_end[i] == i`.
    ///
    /// Populated by `try_compile`. Empty when population failed (e.g.
    /// unsupported opcode) — the refit walker then falls back to the
    /// no-skip path.
    pub subtree_end: Vec<u32>,
}

impl CompiledSdfBvh {
    /// Compile an SdfNode tree with BVH data.
    ///
    /// # Panics
    ///
    /// Panics if the tree contains unsupported primitives.
    /// Use [`try_compile`](Self::try_compile) for a non-panicking alternative.
    pub fn compile(node: &SdfNode) -> Self {
        Self::try_compile(node)
            .expect("CompiledSdfBvh::compile() failed: unsupported primitive in SDF tree")
    }

    /// Compile an SdfNode tree with BVH data, returning an error for unsupported primitives.
    ///
    /// The BVH compiler supports a subset of primitives (basic shapes only).
    /// Extended primitives and 2D shapes require the interpreter or shader transpiler.
    pub fn try_compile(node: &SdfNode) -> Result<Self, CompileError> {
        validate_for_bvh_compile(node)?;

        let (value_depth, coord_depth) = super::compiler::compute_stack_depths(node);
        if value_depth > 64 {
            return Err(CompileError::StackOverflow {
                kind: "value",
                required: value_depth,
                limit: 64,
            });
        }
        if coord_depth > 32 {
            return Err(CompileError::StackOverflow {
                kind: "coordinate",
                required: coord_depth,
                limit: 32,
            });
        }

        let mut compiler = BvhCompiler::new();
        let scene_aabb = compiler.compile_node(node);
        compiler.instructions.push(Instruction::end());
        compiler.aabbs.push(AabbPacked::infinite());

        let mut bvh = Self {
            instructions: compiler.instructions,
            aabbs: compiler.aabbs,
            node_count: compiler.node_count,
            scene_aabb,
            parent_indices: Vec::new(),
            subtree_end: Vec::new(),
        };
        // Populate parent_indices + subtree_end. Silent fallback to empty
        // vectors for scenes containing opcodes not covered by the refit
        // walker — `refit_partial` handles the empty case defensively.
        if let Ok(parents) = super::refit::build_parent_indices(&bvh) {
            bvh.parent_indices = parents;
        }
        if let Ok(ends) = super::refit::build_subtree_ends(&bvh) {
            bvh.subtree_end = ends;
        }
        Ok(bvh)
    }

    /// Recompute every AABB from `Instruction.params[]` in place. See
    /// [`crate::compiled::refit::refit_all`].
    ///
    /// # Errors
    ///
    /// Propagates [`crate::compiled::refit::RefitError`] for unsupported
    /// opcodes or malformed bytecode.
    pub fn refit_all_from_bytecode(&mut self) -> Result<usize, super::refit::RefitError> {
        super::refit::refit_all(self)
    }

    /// Recompute AABBs only for `dirty` instructions and their ancestor
    /// chain. See [`crate::compiled::refit::refit_partial`].
    ///
    /// # Errors
    ///
    /// Propagates [`crate::compiled::refit::RefitError`].
    pub fn refit_partial_from_bytecode(
        &mut self,
        dirty: &[usize],
    ) -> Result<usize, super::refit::RefitError> {
        super::refit::refit_partial(self, dirty)
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
            + self.aabbs.len() * std::mem::size_of::<AabbPacked>()
    }
}

/// Internal compiler state for BVH compilation
struct BvhCompiler {
    instructions: Vec<Instruction>,
    aabbs: Vec<AabbPacked>,
    node_count: usize,
}

impl BvhCompiler {
    fn new() -> Self {
        Self {
            instructions: Vec::with_capacity(256),
            aabbs: Vec::with_capacity(256),
            node_count: 0,
        }
    }

    /// Compile a node and return its AABB
    fn compile_node(&mut self, node: &SdfNode) -> AabbPacked {
        self.node_count += 1;

        match node {
            // === Primitives ===
            SdfNode::Sphere { radius } => {
                let aabb = aabb_prims::sphere_aabb(*radius);
                self.instructions.push(Instruction::sphere(*radius));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Box3d { half_extents } => {
                let aabb = aabb_prims::box_aabb(*half_extents);
                self.instructions.push(Instruction::box3d(
                    half_extents.x,
                    half_extents.y,
                    half_extents.z,
                ));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Cylinder {
                radius,
                half_height,
            } => {
                let aabb = aabb_prims::cylinder_aabb(*radius, *half_height);
                self.instructions
                    .push(Instruction::cylinder(*radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Torus {
                major_radius,
                minor_radius,
            } => {
                let aabb = aabb_prims::torus_aabb(*major_radius, *minor_radius);
                self.instructions
                    .push(Instruction::torus(*major_radius, *minor_radius));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Plane { normal, distance } => {
                let aabb = aabb_prims::plane_aabb();
                self.instructions
                    .push(Instruction::plane(normal.x, normal.y, normal.z, *distance));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Capsule {
                point_a,
                point_b,
                radius,
            } => {
                let aabb = aabb_prims::capsule_aabb(*point_a, *point_b, *radius);
                let mut inst = Instruction::capsule(
                    point_a.x, point_a.y, point_a.z, point_b.x, point_b.y, point_b.z, *radius,
                );
                inst.skip_offset = radius.to_bits();
                self.instructions.push(inst);
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Cone {
                radius,
                half_height,
            } => {
                let aabb = AabbPacked::new(
                    Vec3::new(-*radius, -*half_height, -*radius),
                    Vec3::new(*radius, *half_height, *radius),
                );
                self.instructions
                    .push(Instruction::cone(*radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Ellipsoid { radii } => {
                let aabb = AabbPacked::new(-*radii, *radii);
                self.instructions
                    .push(Instruction::ellipsoid(radii.x, radii.y, radii.z));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::RoundedCone {
                r1,
                r2,
                half_height,
            } => {
                let max_r = r1.max(*r2);
                let aabb = AabbPacked::new(
                    Vec3::new(-max_r, -*half_height, -max_r),
                    Vec3::new(max_r, *half_height, max_r),
                );
                self.instructions
                    .push(Instruction::rounded_cone(*r1, *r2, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Pyramid { half_height } => {
                // Base is unit square (side=1), so half_extent = 0.5
                let aabb = AabbPacked::new(
                    Vec3::new(-0.5, -*half_height, -0.5),
                    Vec3::new(0.5, *half_height, 0.5),
                );
                self.instructions.push(Instruction::pyramid(*half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Octahedron { size } => {
                let aabb = AabbPacked::new(
                    Vec3::new(-*size, -*size, -*size),
                    Vec3::new(*size, *size, *size),
                );
                self.instructions.push(Instruction::octahedron(*size));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::HexPrism {
                hex_radius,
                half_height,
            } => {
                let aabb = AabbPacked::new(
                    Vec3::new(-*hex_radius, -*hex_radius, -*half_height),
                    Vec3::new(*hex_radius, *hex_radius, *half_height),
                );
                self.instructions
                    .push(Instruction::hex_prism(*hex_radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Link {
                half_length,
                r1,
                r2,
            } => {
                let extent = r1 + r2;
                let aabb = AabbPacked::new(
                    Vec3::new(-extent, -(*half_length + extent), -*r2),
                    Vec3::new(extent, *half_length + extent, *r2),
                );
                self.instructions
                    .push(Instruction::link(*half_length, *r1, *r2));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Triangle { .. } => {
                unreachable!("Triangle: validated by validate_for_bvh_compile()");
            }

            SdfNode::Bezier { .. } => {
                unreachable!("Bezier: validated by validate_for_bvh_compile()");
            }

            // New primitives — interpreter-only (use eval() or transpiler)
            SdfNode::RoundedBox { .. }
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
            | SdfNode::Annular2D { .. } => {
                unreachable!("Extended primitive: validated by validate_for_bvh_compile()");
            }

            // === New Binary Operations ===
            SdfNode::XOR { a, b } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b);
                self.instructions.push(Instruction::xor());
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Morph { a, b, t } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b);
                self.instructions.push(Instruction::morph(*t));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ColumnsUnion { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b).expand(*r);
                self.instructions.push(Instruction::columns_union(*r, *n));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ColumnsIntersection { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions
                    .push(Instruction::columns_intersection(*r, *n));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ColumnsSubtraction { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions
                    .push(Instruction::columns_subtraction(*r, *n));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::Pipe { a, b, r } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b);
                self.instructions.push(Instruction::pipe(*r));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Engrave { a, b, r } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions.push(Instruction::engrave(*r));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::Groove { a, b, ra, rb } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions.push(Instruction::groove(*ra, *rb));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::Tongue { a, b, ra, rb } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions.push(Instruction::tongue(*ra, *rb));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            // === Binary Operations ===
            SdfNode::Union { a, b } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b);
                self.instructions.push(Instruction::union());
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Intersection { a, b } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions.push(Instruction::intersection());
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Subtraction { a, b } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                // Subtraction can only shrink or maintain the AABB of 'a'
                self.instructions.push(Instruction::subtraction());
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::SmoothUnion { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                // Smooth union expands the combined AABB by k
                let aabb = aabb_a.union(&aabb_b).expand(*k);
                self.instructions.push(Instruction::smooth_union(*k));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::SmoothIntersection { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions.push(Instruction::smooth_intersection(*k));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::SmoothSubtraction { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions.push(Instruction::smooth_subtraction(*k));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::ChamferUnion { a, b, r } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b).expand(*r);
                self.instructions.push(Instruction::chamfer_union(*r));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ChamferIntersection { a, b, r } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions
                    .push(Instruction::chamfer_intersection(*r));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ChamferSubtraction { a, b, r } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions.push(Instruction::chamfer_subtraction(*r));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::StairsUnion { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b).expand(*r);
                self.instructions.push(Instruction::stairs_union(*r, *n));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::StairsIntersection { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions
                    .push(Instruction::stairs_intersection(*r, *n));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions
                    .push(Instruction::stairs_subtraction(*r, *n));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            // === Transforms ===
            SdfNode::Translate { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::translate(offset.x, offset.y, offset.z));
                self.aabbs.push(AabbPacked::empty()); // Placeholder

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Transform the child's AABB
                let aabb = child_aabb.translate(*offset);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Rotate { child, rotation } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::rotate(
                    rotation.x, rotation.y, rotation.z, rotation.w,
                ));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                let aabb = child_aabb.rotate(*rotation);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Scale { child, factor } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale(*factor));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                let aabb = child_aabb.scale(*factor);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::ScaleNonUniform { child, factors } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::scale_non_uniform(
                    factors.x, factors.y, factors.z,
                ));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                let aabb = child_aabb.scale_nonuniform(*factors);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            // === Modifiers ===
            SdfNode::Twist { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::twist(*strength));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Twist can expand the AABB - conservative estimate
                let max_extent = child_aabb.half_size().max_element();
                let aabb = child_aabb.expand(max_extent * strength.abs() * 0.5);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Bend { child, curvature } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::bend(*curvature));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Bend can expand the AABB - conservative estimate
                let max_extent = child_aabb.half_size().max_element();
                let aabb = child_aabb.expand(max_extent * curvature.abs());
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::RepeatInfinite { child, spacing: _ } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::repeat_infinite(0.0, 0.0, 0.0));
                self.aabbs.push(AabbPacked::infinite()); // Infinite repeat = infinite AABB

                self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                AabbPacked::infinite()
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
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Finite repeat expands the AABB by count * spacing
                let expand = Vec3::new(
                    count[0] as f32 * spacing.x,
                    count[1] as f32 * spacing.y,
                    count[2] as f32 * spacing.z,
                );
                let aabb = AabbPacked::new(child_aabb.min() - expand, child_aabb.max() + expand);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Noise {
                child,
                amplitude,
                frequency,
                seed,
            } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::noise(*amplitude, *frequency, *seed));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Expand AABB by noise amplitude
                let aabb = child_aabb.expand(*amplitude);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Round { child, radius } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::round(*radius));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Round expands the AABB
                let aabb = child_aabb.expand(*radius);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Onion { child, thickness } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::onion(*thickness));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Onion expands the AABB
                let aabb = child_aabb.expand(*thickness);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Elongate { child, amount } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::elongate(amount.x, amount.y, amount.z));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Elongate expands the AABB
                let aabb = AabbPacked::new(child_aabb.min() - *amount, child_aabb.max() + *amount);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Mirror { child, axes } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::mirror(axes.x, axes.y, axes.z));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Mirror makes the AABB symmetric along mirrored axes
                let cmin = child_aabb.min();
                let cmax = child_aabb.max();
                let extent_x = cmax.x.abs().max(cmin.x.abs());
                let extent_y = cmax.y.abs().max(cmin.y.abs());
                let extent_z = cmax.z.abs().max(cmin.z.abs());
                let aabb = AabbPacked::new(
                    Vec3::new(
                        if axes.x != 0.0 { -extent_x } else { cmin.x },
                        if axes.y != 0.0 { -extent_y } else { cmin.y },
                        if axes.z != 0.0 { -extent_z } else { cmin.z },
                    ),
                    Vec3::new(
                        if axes.x != 0.0 { extent_x } else { cmax.x },
                        if axes.y != 0.0 { extent_y } else { cmax.y },
                        if axes.z != 0.0 { extent_z } else { cmax.z },
                    ),
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::OctantMirror { child } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::octant_mirror());
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // abs + sort folds every octant/permutation onto the child, so the
                // result is symmetric in all axes and permutations: cube of max extent
                let cmin = child_aabb.min();
                let cmax = child_aabb.max();
                let m = cmax
                    .x
                    .abs()
                    .max(cmin.x.abs())
                    .max(cmax.y.abs())
                    .max(cmin.y.abs())
                    .max(cmax.z.abs())
                    .max(cmin.z.abs());
                let aabb = AabbPacked::new(Vec3::splat(-m), Vec3::splat(m));
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Revolution { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::revolution(*offset));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Revolution creates a radially symmetric shape around Y
                let max_r = child_aabb.max().x.abs().max(child_aabb.min().x.abs()) + offset.abs();
                let aabb = AabbPacked::new(
                    Vec3::new(-max_r, child_aabb.min().y, -max_r),
                    Vec3::new(max_r, child_aabb.max().y, max_r),
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Extrude { child, half_height } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::extrude(*half_height));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Extrude extends the XY shape along Z
                let aabb = AabbPacked::new(
                    Vec3::new(child_aabb.min().x, child_aabb.min().y, -*half_height),
                    Vec3::new(child_aabb.max().x, child_aabb.max().y, *half_height),
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Taper { child, factor } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::taper(*factor));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Taper can expand the AABB - conservative estimate
                let max_extent = child_aabb.half_size().max_element();
                let aabb = child_aabb.expand(max_extent * factor.abs());
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Displacement { child, strength } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::displacement(*strength));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Displacement expands the AABB by strength
                let aabb = child_aabb.expand(strength.abs());
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::PolarRepeat { child, count } => {
                let inst_idx = self.instructions.len();
                self.instructions
                    .push(Instruction::polar_repeat(*count as f32));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Polar repeat creates a radially symmetric shape around Y
                let max_r = child_aabb.half_size().x.max(child_aabb.half_size().z)
                    + child_aabb.center().x.abs().max(child_aabb.center().z.abs());
                let aabb = AabbPacked::new(
                    Vec3::new(-max_r, child_aabb.min().y, -max_r),
                    Vec3::new(max_r, child_aabb.max().y, max_r),
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sweep_bezier(
                    p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                ));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // SweepBezier: curve in XZ plane, child at (perp_dist, y, 0)
                let bmin_x = p0.x.min(p1.x).min(p2.x);
                let bmax_x = p0.x.max(p1.x).max(p2.x);
                let bmin_z = p0.y.min(p1.y).min(p2.y);
                let bmax_z = p0.y.max(p1.y).max(p2.y);
                let max_perp = child_aabb.max().x.abs().max(child_aabb.min().x.abs());
                let aabb = AabbPacked::new(
                    Vec3::new(bmin_x - max_perp, child_aabb.min().y, bmin_z - max_perp),
                    Vec3::new(bmax_x + max_perp, child_aabb.max().y, bmax_z + max_perp),
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::ExpSmoothUnion { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.union(&aabb_b).expand(*k);
                self.instructions.push(Instruction::exp_smooth_union(*k));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ExpSmoothIntersection { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let aabb_b = self.compile_node(b);
                let aabb = aabb_a.intersection(&aabb_b);
                self.instructions
                    .push(Instruction::exp_smooth_intersection(*k));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::ExpSmoothSubtraction { a, b, k } => {
                let aabb_a = self.compile_node(a);
                let _aabb_b = self.compile_node(b);
                self.instructions
                    .push(Instruction::exp_smooth_subtraction(*k));
                self.aabbs.push(aabb_a);
                aabb_a
            }

            SdfNode::Shear { child, shear } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::shear(*shear));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Conservative: expand AABB by shear amount
                let max_shear = shear.x.abs().max(shear.y.abs()).max(shear.z.abs());
                let half = child_aabb.half_size();
                let expand = half.length() * max_shear;
                let aabb = child_aabb.expand(expand);
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Animated { child, .. } => {
                // Static compilation: just compile child
                self.compile_node(child)
            }

            // WithMaterial is transparent for distance evaluation
            SdfNode::WithMaterial { child, .. } => self.compile_node(child),

            // Rejected up front by `validate_for_bvh_compile` (aux-data dependent or
            // no AABB law yet). Listed explicitly so the match stays exhaustive: a new
            // SdfNode variant is a compile error here, not a silent fallback.
            SdfNode::ProjectiveTransform { .. }
            | SdfNode::LatticeDeform { .. }
            | SdfNode::SdfSkinning { .. }
            | SdfNode::IFS { .. }
            | SdfNode::HeightmapDisplacement { .. }
            | SdfNode::IcosahedralSymmetry { .. }
            | SdfNode::SurfaceRoughness { .. }
            | SdfNode::SineDisplacement { .. }
            | SdfNode::Terrain { .. } => {
                unreachable!("rejected by validate_for_bvh_compile()");
            }
        }
    }
}

/// Evaluate compiled SDF with BVH annotations.
///
/// The BVH bytecode uses the same instruction set as [`super::compiler::CompiledSdf`]
/// and is executed by the shared exhaustive stack machine in
/// `eval_scalar_core`. The per-instruction AABBs (`sdf.aabbs`) are
/// retained for raymarching / refit consumers; this point evaluator does not
/// prune with them (pruning by AABB is unsafe for a single-point SDF query
/// because the distance to a culled subtree still contributes to the result).
///
/// `CompiledSdfBvh` carries no `aux_data`; every opcode that needs aux data
/// is rejected up front by `validate_for_bvh_compile`, so the empty slice is
/// never read.
#[inline]
pub fn eval_compiled_bvh(sdf: &CompiledSdfBvh, point: Vec3) -> f32 {
    super::eval_scalar_core::eval_bytecode(&sdf.instructions, &[], point)
}

/// Get the AABB for the entire compiled SDF
///
/// Returns the scene AABB computed during compilation, which
/// correctly handles all node types including transforms and modifiers as root.
pub const fn get_scene_aabb(sdf: &CompiledSdfBvh) -> AabbPacked {
    sdf.scene_aabb
}

/// Recursively validate that all nodes in the tree are supported by the BVH bytecode compiler.
///
/// The BVH compiler supports only basic primitives. Extended primitives and
/// 2D shapes require the interpreter or shader transpiler.
fn validate_for_bvh_compile(node: &SdfNode) -> Result<(), CompileError> {
    match node {
        // Unsupported: params[6] limit
        SdfNode::Triangle { .. } => {
            return Err(CompileError::UnsupportedPrimitive("Triangle".into()));
        }
        SdfNode::Bezier { .. } => {
            return Err(CompileError::UnsupportedPrimitive("Bezier".into()));
        }
        // Unsupported: BVH bytecode not yet implemented for extended primitives
        SdfNode::RoundedBox { .. }
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
        | SdfNode::Annular2D { .. } => {
            return Err(CompileError::UnsupportedPrimitive(format!(
                "{:?}",
                std::mem::discriminant(node)
            )));
        }
        // Unsupported: needs `aux_data` (CompiledSdfBvh has none) or no AABB law yet.
        // Previously these compiled to a silent `sphere(0.001)` fallback.
        SdfNode::ProjectiveTransform { .. } => {
            return Err(CompileError::UnsupportedPrimitive(
                "ProjectiveTransform".into(),
            ));
        }
        SdfNode::LatticeDeform { .. } => {
            return Err(CompileError::UnsupportedPrimitive("LatticeDeform".into()));
        }
        SdfNode::SdfSkinning { .. } => {
            return Err(CompileError::UnsupportedPrimitive("SdfSkinning".into()));
        }
        SdfNode::IFS { .. } => {
            return Err(CompileError::UnsupportedPrimitive("IFS".into()));
        }
        SdfNode::HeightmapDisplacement { .. } => {
            return Err(CompileError::UnsupportedPrimitive(
                "HeightmapDisplacement".into(),
            ));
        }
        SdfNode::IcosahedralSymmetry { .. } => {
            return Err(CompileError::UnsupportedPrimitive(
                "IcosahedralSymmetry".into(),
            ));
        }
        SdfNode::SurfaceRoughness { .. } => {
            return Err(CompileError::UnsupportedPrimitive(
                "SurfaceRoughness".into(),
            ));
        }
        SdfNode::SineDisplacement { .. } => {
            return Err(CompileError::UnsupportedPrimitive(
                "SineDisplacement".into(),
            ));
        }
        SdfNode::Terrain { .. } => {
            return Err(CompileError::UnsupportedPrimitive("Terrain".into()));
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
            validate_for_bvh_compile(a)?;
            validate_for_bvh_compile(b)?;
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
        | SdfNode::Shear { child, .. } => {
            validate_for_bvh_compile(child)?;
        }
        // All basic primitives are supported (BvhCompiler::compile_node is exhaustive,
        // so a new variant that slips through here fails to compile there).
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;

    #[test]
    fn test_compile_sphere_bvh() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdfBvh::compile(&node);

        assert_eq!(compiled.node_count, 1);
        assert_eq!(compiled.instructions.len(), 2); // sphere + end
        assert_eq!(compiled.aabbs.len(), 2);

        // Check AABB
        let aabb = &compiled.aabbs[0];
        assert_eq!(aabb.min(), Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(aabb.max(), Vec3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_compile_union_bvh() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(3.0, 0.0, 0.0));
        let compiled = CompiledSdfBvh::compile(&node);

        // Check that the union AABB encompasses both spheres
        let scene_aabb = get_scene_aabb(&compiled);
        assert!(scene_aabb.min_x <= -1.0);
        assert!(scene_aabb.max_x >= 4.0); // 3 + 1
    }

    #[test]
    fn test_eval_bvh_sphere() {
        let node = SdfNode::sphere(1.0);
        let compiled = CompiledSdfBvh::compile(&node);

        let d = eval_compiled_bvh(&compiled, Vec3::ZERO);
        assert!((d + 1.0).abs() < 0.001);

        let d = eval_compiled_bvh(&compiled, Vec3::new(1.0, 0.0, 0.0));
        assert!(d.abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_vs_interpreted() {
        let node = SdfNode::sphere(1.0)
            .smooth_union(SdfNode::box3d(0.8, 0.8, 0.8), 0.1)
            .translate(0.5, 0.0, 0.0);
        let compiled = CompiledSdfBvh::compile(&node);

        let test_points = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            Vec3::new(-1.0, -1.0, -1.0),
        ];

        for p in test_points {
            let d_interp = eval(&node, p);
            let d_bvh = eval_compiled_bvh(&compiled, p);
            assert!(
                (d_interp - d_bvh).abs() < 0.001,
                "Mismatch at {:?}: interp={}, bvh={}",
                p,
                d_interp,
                d_bvh
            );
        }
    }

    #[test]
    fn test_scene_aabb() {
        let node = SdfNode::sphere(1.0).union(SdfNode::sphere(1.0).translate(5.0, 0.0, 0.0));
        let compiled = CompiledSdfBvh::compile(&node);

        let aabb = get_scene_aabb(&compiled);
        assert!(aabb.min_x <= -1.0);
        assert!(aabb.max_x >= 6.0);
        assert!(aabb.min_y <= -1.0);
        assert!(aabb.max_y >= 1.0);
    }

    /// Helper: compare BVH evaluation against interpreted evaluation
    fn assert_bvh_matches_interp(node: &SdfNode, test_points: &[Vec3], tolerance: f32) {
        let compiled = CompiledSdfBvh::compile(node);
        for &p in test_points {
            let d_interp = eval(node, p);
            let d_bvh = eval_compiled_bvh(&compiled, p);
            assert!(
                (d_interp - d_bvh).abs() < tolerance,
                "Mismatch at {:?}: interp={}, bvh={} (diff={})",
                p,
                d_interp,
                d_bvh,
                (d_interp - d_bvh).abs()
            );
        }
    }

    const STANDARD_TEST_POINTS: [Vec3; 8] = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.5, 0.5, 0.5),
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.3, -0.7, 1.2),
    ];

    #[test]
    fn test_eval_bvh_cone() {
        let node = SdfNode::cone(1.0, 1.5);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_cone_translated() {
        let node = SdfNode::cone(0.8, 1.0).translate(1.0, 0.5, 0.0);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_cone_aabb() {
        // cone(radius, height) → half_height = height * 0.5
        let node = SdfNode::cone(1.0, 4.0); // half_height = 2.0
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0];
        assert!((aabb.min_x - (-1.0)).abs() < 0.001);
        assert!((aabb.max_x - 1.0).abs() < 0.001);
        assert!((aabb.min_y - (-2.0)).abs() < 0.001);
        assert!((aabb.max_y - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_ellipsoid() {
        let node = SdfNode::ellipsoid(1.0, 0.5, 0.75);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_ellipsoid_union() {
        let node =
            SdfNode::ellipsoid(1.0, 0.5, 0.75).union(SdfNode::sphere(0.5).translate(2.0, 0.0, 0.0));
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_ellipsoid_aabb() {
        let node = SdfNode::ellipsoid(2.0, 1.0, 1.5);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0];
        assert!((aabb.min_x - (-2.0)).abs() < 0.001);
        assert!((aabb.max_x - 2.0).abs() < 0.001);
        assert!((aabb.min_y - (-1.0)).abs() < 0.001);
        assert!((aabb.max_y - 1.0).abs() < 0.001);
        assert!((aabb.min_z - (-1.5)).abs() < 0.001);
        assert!((aabb.max_z - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_eval_bvh_mirror() {
        let node = SdfNode::box3d(1.0, 0.5, 0.5)
            .translate(1.0, 0.0, 0.0)
            .mirror(true, false, false);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_eval_bvh_mirror_xyz() {
        let node = SdfNode::sphere(0.5)
            .translate(1.0, 1.0, 1.0)
            .mirror(true, true, true);
        let extra_points = [
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, -1.0, -1.0),
        ];
        assert_bvh_matches_interp(&node, &extra_points, 0.001);
    }

    #[test]
    fn test_eval_bvh_revolution() {
        let node = SdfNode::sphere(0.3).revolution(1.0);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_revolution_aabb() {
        // sphere(0.5).revolution(2.0)
        // Instructions: revolution, sphere, pop_transform, end
        // Revolution AABB is at index 0
        let node = SdfNode::sphere(0.5).revolution(2.0);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = &compiled.aabbs[0]; // Revolution instruction's AABB
                                       // Child sphere AABB x: [-0.5, 0.5], max abs = 0.5
                                       // Revolution radial extent = 0.5 + |offset| = 2.5
        assert!(aabb.min_x <= -2.5, "min_x={}", aabb.min_x);
        assert!(aabb.max_x >= 2.5, "max_x={}", aabb.max_x);
        assert!(aabb.min_z <= -2.5, "min_z={}", aabb.min_z);
        assert!(aabb.max_z >= 2.5, "max_z={}", aabb.max_z);
    }

    #[test]
    fn test_eval_bvh_extrude() {
        let node = SdfNode::sphere(1.0).extrude(0.5);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.01);
    }

    #[test]
    fn test_eval_bvh_extrude_box() {
        let node = SdfNode::box3d(1.0, 0.5, 0.0).extrude(2.0);
        let test_points = [
            Vec3::ZERO,
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::new(1.5, 0.0, 0.5),
        ];
        assert_bvh_matches_interp(&node, &test_points, 0.01);
    }

    #[test]
    fn test_eval_bvh_noise() {
        // Interpreted eval uses Perlin noise, BVH uses hash noise (different algorithms)
        // So we compare BVH compiled eval vs BVH eval (same noise function)
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        let compiled_bvh = CompiledSdfBvh::compile(&node);
        let compiled_flat = crate::compiled::CompiledSdf::compile(&node);
        for &p in &STANDARD_TEST_POINTS {
            let d_bvh = eval_compiled_bvh(&compiled_bvh, p);
            let d_flat = crate::compiled::eval::eval_compiled(&compiled_flat, p);
            assert!(
                (d_bvh - d_flat).abs() < 0.001,
                "Noise mismatch at {:?}: bvh={}, flat={} (diff={})",
                p,
                d_bvh,
                d_flat,
                (d_bvh - d_flat).abs()
            );
        }
    }

    #[test]
    fn test_eval_bvh_noise_on_union() {
        // Compare BVH vs flat compiled (both use hash noise)
        let node = SdfNode::sphere(1.0)
            .union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.5, 0.0, 0.0))
            .noise(0.05, 3.0, 7);
        let compiled_bvh = CompiledSdfBvh::compile(&node);
        let compiled_flat = crate::compiled::CompiledSdf::compile(&node);
        for &p in &STANDARD_TEST_POINTS {
            let d_bvh = eval_compiled_bvh(&compiled_bvh, p);
            let d_flat = crate::compiled::eval::eval_compiled(&compiled_flat, p);
            assert!(
                (d_bvh - d_flat).abs() < 0.001,
                "Noise mismatch at {:?}: bvh={}, flat={} (diff={})",
                p,
                d_bvh,
                d_flat,
                (d_bvh - d_flat).abs()
            );
        }
    }

    #[test]
    fn test_eval_bvh_combined_new_features() {
        // Cone + Mirror (no noise — compare against interpreted)
        let node = SdfNode::cone(0.5, 1.0).mirror(true, false, true);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }

    #[test]
    fn test_scene_aabb_transform_root() {
        // Bug fix: get_scene_aabb should return correct AABB when root is a transform
        let node = SdfNode::sphere(1.0).translate(5.0, 3.0, 0.0);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&compiled);
        assert!((aabb.min_x - 4.0).abs() < 0.001, "min_x={}", aabb.min_x);
        assert!((aabb.max_x - 6.0).abs() < 0.001, "max_x={}", aabb.max_x);
        assert!((aabb.min_y - 2.0).abs() < 0.001, "min_y={}", aabb.min_y);
        assert!((aabb.max_y - 4.0).abs() < 0.001, "max_y={}", aabb.max_y);
    }

    #[test]
    fn test_scene_aabb_modifier_root() {
        // get_scene_aabb should work when root is a modifier (e.g. noise)
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        let compiled = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&compiled);
        // Sphere AABB is [-1,1]^3, expanded by amplitude=0.1
        assert!(aabb.min_x <= -1.0);
        assert!(aabb.max_x >= 1.0);
    }

    #[test]
    fn test_noise_interpreted_vs_compiled() {
        // Bug fix: all paths now use perlin_noise_3d
        let node = SdfNode::sphere(1.0).noise(0.1, 2.0, 42);
        assert_bvh_matches_interp(&node, &STANDARD_TEST_POINTS, 0.001);
    }
}

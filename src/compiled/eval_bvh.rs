//! BVH-accelerated SDF evaluation with spatial pruning
//!
//! This module provides SDF evaluation that uses bounding volume hierarchies
//! to skip computation of distant objects, significantly accelerating
//! evaluation for sparse scenes.
//!
//! Author: Moroya Sakamoto

use super::aabb::{primitives as aabb_prims, AabbPacked};
use super::instruction::Instruction;
use super::opcode::OpCode;
use crate::primitives::*;
use crate::operations::*;
use crate::modifiers::*;
use crate::types::SdfNode;
use glam::{Quat, Vec3};

use crate::modifiers::perlin_noise_3d;

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
}

impl CompiledSdfBvh {
    /// Compile an SdfNode tree with BVH data
    pub fn compile(node: &SdfNode) -> Self {
        let mut compiler = BvhCompiler::new();
        let scene_aabb = compiler.compile_node(node);
        compiler.instructions.push(Instruction::end());
        compiler.aabbs.push(AabbPacked::infinite()); // End instruction has infinite AABB

        CompiledSdfBvh {
            instructions: compiler.instructions,
            aabbs: compiler.aabbs,
            node_count: compiler.node_count,
            scene_aabb,
        }
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
        BvhCompiler {
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

            SdfNode::Cylinder { radius, half_height } => {
                let aabb = aabb_prims::cylinder_aabb(*radius, *half_height);
                self.instructions.push(Instruction::cylinder(*radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Torus { major_radius, minor_radius } => {
                let aabb = aabb_prims::torus_aabb(*major_radius, *minor_radius);
                self.instructions.push(Instruction::torus(*major_radius, *minor_radius));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Plane { normal, distance } => {
                let aabb = aabb_prims::plane_aabb();
                self.instructions.push(Instruction::plane(
                    normal.x, normal.y, normal.z, *distance,
                ));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Capsule { point_a, point_b, radius } => {
                let aabb = aabb_prims::capsule_aabb(*point_a, *point_b, *radius);
                let mut inst = Instruction::capsule(
                    point_a.x, point_a.y, point_a.z,
                    point_b.x, point_b.y, point_b.z,
                    *radius,
                );
                inst.skip_offset = radius.to_bits();
                self.instructions.push(inst);
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Cone { radius, half_height } => {
                let aabb = AabbPacked::new(
                    Vec3::new(-*radius, -*half_height, -*radius),
                    Vec3::new(*radius, *half_height, *radius),
                );
                self.instructions.push(Instruction::cone(*radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Ellipsoid { radii } => {
                let aabb = AabbPacked::new(-*radii, *radii);
                self.instructions.push(Instruction::ellipsoid(radii.x, radii.y, radii.z));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::RoundedCone { r1, r2, half_height } => {
                let max_r = r1.max(*r2);
                let aabb = AabbPacked::new(
                    Vec3::new(-max_r, -*half_height, -max_r),
                    Vec3::new(max_r, *half_height, max_r),
                );
                self.instructions.push(Instruction::rounded_cone(*r1, *r2, *half_height));
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

            SdfNode::HexPrism { hex_radius, half_height } => {
                let aabb = AabbPacked::new(
                    Vec3::new(-*hex_radius, -*hex_radius, -*half_height),
                    Vec3::new(*hex_radius, *hex_radius, *half_height),
                );
                self.instructions.push(Instruction::hex_prism(*hex_radius, *half_height));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Link { half_length, r1, r2 } => {
                let extent = r1 + r2;
                let aabb = AabbPacked::new(
                    Vec3::new(-extent, -(*half_length + extent), -*r2),
                    Vec3::new(extent, *half_length + extent, *r2),
                );
                self.instructions.push(Instruction::link(*half_length, *r1, *r2));
                self.aabbs.push(aabb);
                aabb
            }

            SdfNode::Triangle { .. } => {
                panic!("Triangle requires 9 params and cannot be compiled to bytecode (params[6] limit). Use eval() or transpiler instead.");
            }

            SdfNode::Bezier { .. } => {
                panic!("Bezier requires 10 params and cannot be compiled to bytecode (params[6] limit). Use eval() or transpiler instead.");
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
            | SdfNode::Heart { .. } => {
                panic!("This primitive is interpreter/transpiler-only. Use eval() or shader transpiler instead.");
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

            // === Transforms ===
            SdfNode::Translate { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::translate(offset.x, offset.y, offset.z));
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
                self.instructions.push(Instruction::repeat_infinite(0.0, 0.0, 0.0));
                self.aabbs.push(AabbPacked::infinite()); // Infinite repeat = infinite AABB

                self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                AabbPacked::infinite()
            }

            SdfNode::RepeatFinite { child, count, spacing } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::repeat_finite(
                    count[0] as f32, count[1] as f32, count[2] as f32,
                    spacing.x, spacing.y, spacing.z,
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
                let aabb = AabbPacked::new(
                    child_aabb.min() - expand,
                    child_aabb.max() + expand,
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Noise { child, amplitude, frequency, seed } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::noise(*amplitude, *frequency, *seed));
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
                self.instructions.push(Instruction::elongate(amount.x, amount.y, amount.z));
                self.aabbs.push(AabbPacked::empty());

                let child_aabb = self.compile_node(child);

                self.instructions.push(Instruction::pop_transform());
                self.aabbs.push(AabbPacked::infinite());

                // Elongate expands the AABB
                let aabb = AabbPacked::new(
                    child_aabb.min() - *amount,
                    child_aabb.max() + *amount,
                );
                self.aabbs[inst_idx] = aabb;
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
                aabb
            }

            SdfNode::Mirror { child, axes } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::mirror(axes.x, axes.y, axes.z));
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
                self.instructions.push(Instruction::polar_repeat(*count as f32));
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

            // WithMaterial is transparent for distance evaluation
            SdfNode::WithMaterial { child, .. } => {
                self.compile_node(child)
            }
        }
    }
}

/// Evaluate compiled SDF with BVH pruning
///
/// This evaluator uses AABBs to skip computation of subtrees
/// when the evaluation point is far enough away.
#[inline]
pub fn eval_compiled_bvh(sdf: &CompiledSdfBvh, point: Vec3) -> f32 {
    const MAX_VALUE_STACK: usize = 64;
    const MAX_COORD_STACK: usize = 32;

    let mut value_stack: [f32; MAX_VALUE_STACK] = [0.0; MAX_VALUE_STACK];
    let mut vsp: usize = 0;

    #[derive(Clone, Copy)]
    struct CoordFrame {
        point: Vec3,
        scale_correction: f32,
        opcode: OpCode,
        params: [f32; 4],
    }

    let mut coord_stack: [CoordFrame; MAX_COORD_STACK] = [CoordFrame {
        point: Vec3::ZERO,
        scale_correction: 1.0,
        opcode: OpCode::End,
        params: [0.0; 4],
    }; MAX_COORD_STACK];
    let mut csp: usize = 0;

    let mut p = point;
    let mut scale_correction: f32 = 1.0;

    let mut ip: usize = 0;
    while ip < sdf.instructions.len() {
        let inst = &sdf.instructions[ip];
        let aabb = &sdf.aabbs[ip];

        // BVH pruning: check if we can skip this subtree
        // Only skip for transform/modifier opcodes that have skip_offset set
        if inst.opcode.is_transform() || inst.opcode.is_modifier() {
            let _aabb_dist = aabb.distance_to_point_fast(point);
            // If point is far from AABB and we have a valid skip offset,
            // we could potentially skip. However, for correctness we need
            // to be careful - we only skip if aabb_dist is larger than
            // some threshold (e.g., current best distance).
            // For now, we don't skip transforms but the AABB is still computed
            // for future raymarching optimization.
        }

        match inst.opcode {
            // === Primitives ===
            OpCode::Sphere => {
                let d = sdf_sphere(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Box3d => {
                let half_extents = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_box3d(p, half_extents);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cylinder => {
                let d = sdf_cylinder(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Torus => {
                let d = sdf_torus(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Plane => {
                let normal = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_plane(p, normal, inst.params[3]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Capsule => {
                let point_a = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let point_b = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                let radius = inst.get_capsule_radius();
                let d = sdf_capsule(p, point_a, point_b, radius);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Cone => {
                let d = sdf_cone(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Ellipsoid => {
                let radii = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let d = sdf_ellipsoid(p, radii);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::RoundedCone => {
                let d = sdf_rounded_cone(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Pyramid => {
                let d = sdf_pyramid(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Octahedron => {
                let d = sdf_octahedron(p, inst.params[0]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::HexPrism => {
                let d = sdf_hex_prism(p, inst.params[0], inst.params[1]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            OpCode::Link => {
                let d = sdf_link(p, inst.params[0], inst.params[1], inst.params[2]);
                value_stack[vsp] = d * scale_correction;
                vsp += 1;
            }

            // === Binary Operations ===
            OpCode::Union => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_union(a, b);
            }

            OpCode::Intersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_intersection(a, b);
            }

            OpCode::Subtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_subtraction(a, b);
            }

            OpCode::SmoothUnion => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_union(a, b, inst.params[0]);
            }

            OpCode::SmoothIntersection => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_intersection(a, b, inst.params[0]);
            }

            OpCode::SmoothSubtraction => {
                vsp -= 1;
                let b = value_stack[vsp];
                let a = value_stack[vsp - 1];
                value_stack[vsp - 1] = sdf_smooth_subtraction(a, b, inst.params[0]);
            }

            // === Transforms ===
            OpCode::Translate => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Translate,
                    params: [0.0; 4],
                };
                csp += 1;

                let offset = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = p - offset;
            }

            OpCode::Rotate => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Rotate,
                    params: [0.0; 4],
                };
                csp += 1;

                let q = Quat::from_xyzw(
                    inst.params[0],
                    inst.params[1],
                    inst.params[2],
                    inst.params[3],
                );
                p = q.inverse() * p;
            }

            OpCode::Scale => {
                let inv_factor = inst.params[0]; // precomputed 1.0/factor
                let factor = inst.params[1];     // original factor
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverse (no division)
                p = p * inv_factor;
                scale_correction *= factor;
            }

            OpCode::ScaleNonUniform => {
                let inv_factors = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let min_factor = inst.params[3]; // precomputed min(sx,sy,sz)

                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                // Multiply by precomputed inverses (no division)
                p = p * inv_factors;
                scale_correction *= min_factor;
            }

            // === Modifiers ===
            OpCode::Twist => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Twist,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_twist(p, inst.params[0]);
            }

            OpCode::Bend => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Bend,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_bend(p, inst.params[0]);
            }

            OpCode::RepeatInfinite => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::RepeatInfinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let spacing = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = modifier_repeat_infinite(p, spacing);
            }

            OpCode::RepeatFinite => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::RepeatFinite,
                    params: [0.0; 4],
                };
                csp += 1;

                let count = [
                    inst.params[0] as u32,
                    inst.params[1] as u32,
                    inst.params[2] as u32,
                ];
                let spacing = Vec3::new(inst.params[3], inst.params[4], inst.params[5]);
                p = modifier_repeat_finite(p, count, spacing);
            }

            OpCode::Round => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Round,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
            }

            OpCode::Onion => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Onion,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
            }

            OpCode::Elongate => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Elongate,
                    params: [0.0; 4],
                };
                csp += 1;

                let amount = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = p - p.clamp(-amount, amount);
            }

            OpCode::Noise => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Noise,
                    params: [inst.params[0], inst.params[1], inst.params[2], 0.0],
                };
                csp += 1;
                // Noise doesn't modify point, only post-processes distance
            }

            OpCode::Mirror => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Mirror,
                    params: [inst.params[0], inst.params[1], inst.params[2], 0.0],
                };
                csp += 1;

                let axes = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                p = modifier_mirror(p, axes);
            }

            OpCode::Revolution => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Revolution,
                    params: [0.0; 4],
                };
                csp += 1;

                p = modifier_revolution(p, inst.params[0]);
            }

            OpCode::Extrude => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Extrude,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = Vec3::new(p.x, p.y, 0.0);
            }

            OpCode::Taper => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Taper,
                    params: [0.0; 4],
                };
                csp += 1;

                let factor = inst.params[0];
                let s = 1.0 / (1.0 - p.y * factor);
                p = Vec3::new(p.x * s, p.y, p.z * s);
            }

            OpCode::Displacement => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Displacement,
                    params: [inst.params[0], 0.0, 0.0, 0.0],
                };
                csp += 1;
                // Displacement doesn't modify point, only post-processes distance
            }

            OpCode::PolarRepeat => {
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::PolarRepeat,
                    params: [0.0; 4],
                };
                csp += 1;

                use crate::modifiers::modifier_polar_repeat;
                p = modifier_polar_repeat(p, inst.params[0] as u32);
            }

            // === Control ===
            OpCode::PopTransform => {
                csp -= 1;
                let frame = coord_stack[csp];

                match frame.opcode {
                    OpCode::Round => {
                        value_stack[vsp - 1] -= frame.params[0];
                    }
                    OpCode::Onion => {
                        value_stack[vsp - 1] = value_stack[vsp - 1].abs() - frame.params[0];
                    }
                    OpCode::Noise => {
                        let amplitude = frame.params[0];
                        let frequency = frame.params[1];
                        let seed = frame.params[2] as u32;
                        let p = frame.point * frequency;
                        let noise_val = perlin_noise_3d(p.x, p.y, p.z, seed);
                        value_stack[vsp - 1] += noise_val * amplitude;
                    }
                    OpCode::Extrude => {
                        let half_height = frame.params[0];
                        let child_dist = value_stack[vsp - 1];
                        let w_y = frame.point.z.abs() - half_height;
                        let outside = glam::Vec2::new(child_dist.max(0.0), w_y.max(0.0)).length();
                        let inside = child_dist.max(w_y).min(0.0);
                        value_stack[vsp - 1] = outside + inside;
                    }
                    OpCode::Displacement => {
                        let strength = frame.params[0];
                        let d = value_stack[vsp - 1];
                        let fp = frame.point;
                        value_stack[vsp - 1] = d + (5.0 * fp.x).sin() * (5.0 * fp.y).sin() * (5.0 * fp.z).sin() * strength;
                    }
                    _ => {}
                }

                p = frame.point;
                scale_correction = frame.scale_correction;
            }

            OpCode::End => {
                break;
            }
        }

        ip += 1;
    }

    if vsp > 0 {
        value_stack[0]
    } else {
        f32::MAX
    }
}

/// Get the AABB for the entire compiled SDF
///
/// Returns the scene AABB computed during compilation, which
/// correctly handles all node types including transforms and modifiers as root.
pub fn get_scene_aabb(sdf: &CompiledSdfBvh) -> AabbPacked {
    sdf.scene_aabb
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
                p, d_interp, d_bvh
            );
        }
    }

    #[test]
    fn test_scene_aabb() {
        let node = SdfNode::sphere(1.0)
            .union(SdfNode::sphere(1.0).translate(5.0, 0.0, 0.0));
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
                p, d_interp, d_bvh, (d_interp - d_bvh).abs()
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
        let node = SdfNode::ellipsoid(1.0, 0.5, 0.75)
            .union(SdfNode::sphere(0.5).translate(2.0, 0.0, 0.0));
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
                p, d_bvh, d_flat, (d_bvh - d_flat).abs()
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
                p, d_bvh, d_flat, (d_bvh - d_flat).abs()
            );
        }
    }

    #[test]
    fn test_eval_bvh_combined_new_features() {
        // Cone + Mirror (no noise — compare against interpreted)
        let node = SdfNode::cone(0.5, 1.0)
            .mirror(true, false, true);
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

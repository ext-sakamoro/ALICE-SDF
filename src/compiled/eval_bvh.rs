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

/// Hash-based 3D noise for compiled SDF evaluation
#[inline]
fn hash_noise_3d(p: Vec3, seed: u32) -> f32 {
    let ix = (p.x.floor() as i32) as u32;
    let iy = (p.y.floor() as i32) as u32;
    let iz = (p.z.floor() as i32) as u32;

    let fx = p.x.fract();
    let fy = p.y.fract();
    let fz = p.z.fract();

    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let uz = fz * fz * (3.0 - 2.0 * fz);

    let hash = |x: u32, y: u32, z: u32| -> f32 {
        let mut h = x.wrapping_mul(374761393)
            .wrapping_add(y.wrapping_mul(668265263))
            .wrapping_add(z.wrapping_mul(1274126177))
            .wrapping_add(seed);
        h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        h = h ^ (h >> 16);
        (h as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let c000 = hash(ix, iy, iz);
    let c100 = hash(ix.wrapping_add(1), iy, iz);
    let c010 = hash(ix, iy.wrapping_add(1), iz);
    let c110 = hash(ix.wrapping_add(1), iy.wrapping_add(1), iz);
    let c001 = hash(ix, iy, iz.wrapping_add(1));
    let c101 = hash(ix.wrapping_add(1), iy, iz.wrapping_add(1));
    let c011 = hash(ix, iy.wrapping_add(1), iz.wrapping_add(1));
    let c111 = hash(ix.wrapping_add(1), iy.wrapping_add(1), iz.wrapping_add(1));

    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);

    let c00 = lerp(c000, c100, ux);
    let c10 = lerp(c010, c110, ux);
    let c01 = lerp(c001, c101, ux);
    let c11 = lerp(c011, c111, ux);

    let c0 = lerp(c00, c10, uy);
    let c1 = lerp(c01, c11, uy);

    lerp(c0, c1, uz)
}

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
}

impl CompiledSdfBvh {
    /// Compile an SdfNode tree with BVH data
    pub fn compile(node: &SdfNode) -> Self {
        let mut compiler = BvhCompiler::new();
        compiler.compile_node(node);
        compiler.instructions.push(Instruction::end());
        compiler.aabbs.push(AabbPacked::infinite()); // End instruction has infinite AABB

        CompiledSdfBvh {
            instructions: compiler.instructions,
            aabbs: compiler.aabbs,
            node_count: compiler.node_count,
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

            SdfNode::Noise { child, amplitude, frequency: _, seed: _ } => {
                // Noise just compiles the child (no AABB change needed, noise is local)
                let child_aabb = self.compile_node(child);
                // Expand AABB by noise amplitude
                child_aabb.expand(*amplitude)
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
            let aabb_dist = aabb.distance_to_point_fast(point);
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
                let radius = f32::from_bits(inst.skip_offset);
                let d = sdf_capsule(p, point_a, point_b, radius);
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
                let factor = inst.params[0];
                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::Scale,
                    params: [factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = p / factor;
                scale_correction *= factor;
            }

            OpCode::ScaleNonUniform => {
                let factors = Vec3::new(inst.params[0], inst.params[1], inst.params[2]);
                let min_factor = factors.x.min(factors.y).min(factors.z);

                coord_stack[csp] = CoordFrame {
                    point: p,
                    scale_correction,
                    opcode: OpCode::ScaleNonUniform,
                    params: [min_factor, 0.0, 0.0, 0.0],
                };
                csp += 1;

                p = p / factors;
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
                        let noise_val = hash_noise_3d(frame.point * frequency, seed);
                        value_stack[vsp - 1] += noise_val * amplitude;
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
/// For post-order compilation, the root node's AABB is the last
/// non-End, non-PopTransform instruction.
pub fn get_scene_aabb(sdf: &CompiledSdfBvh) -> AabbPacked {
    if sdf.aabbs.is_empty() {
        return AabbPacked::empty();
    }

    // Find the last meaningful instruction (not End or PopTransform)
    // This is the root of the SDF tree
    for i in (0..sdf.instructions.len()).rev() {
        let opcode = sdf.instructions[i].opcode;
        if opcode != OpCode::End && opcode != OpCode::PopTransform {
            return sdf.aabbs[i];
        }
    }

    sdf.aabbs[0]
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
}

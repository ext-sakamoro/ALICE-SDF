//! Compiler: SdfNode tree → CompiledSdf bytecode
//!
//! Converts the recursive Arc-based tree structure into a flat
//! instruction array for cache-efficient evaluation.
//!
//! Author: Moroya Sakamoto

use super::instruction::Instruction;
use crate::types::SdfNode;

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
    /// Compile an SdfNode tree into bytecode
    ///
    /// The compilation uses post-order traversal:
    /// - For primitives: emit the primitive instruction
    /// - For binary ops: compile left child, compile right child, emit operation
    /// - For transforms/modifiers: emit transform, compile child, emit pop
    pub fn compile(node: &SdfNode) -> Self {
        let mut compiler = Compiler::new();
        compiler.compile_node(node);
        compiler.instructions.push(Instruction::end());

        CompiledSdf {
            instructions: compiler.instructions,
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

            SdfNode::Cylinder { radius, half_height } => {
                self.instructions.push(Instruction::cylinder(*radius, *half_height));
            }

            SdfNode::Torus { major_radius, minor_radius } => {
                self.instructions.push(Instruction::torus(*major_radius, *minor_radius));
            }

            SdfNode::Plane { normal, distance } => {
                self.instructions.push(Instruction::plane(
                    normal.x, normal.y, normal.z, *distance,
                ));
            }

            SdfNode::Capsule { point_a, point_b, radius } => {
                self.instructions.push(Instruction::capsule(
                    point_a.x, point_a.y, point_a.z,
                    point_b.x, point_b.y, point_b.z,
                    *radius,
                ));
            }

            SdfNode::Cone { radius, half_height } => {
                self.instructions.push(Instruction::cone(*radius, *half_height));
            }

            SdfNode::Ellipsoid { radii } => {
                self.instructions.push(Instruction::ellipsoid(radii.x, radii.y, radii.z));
            }

            SdfNode::RoundedCone { r1, r2, half_height } => {
                self.instructions.push(Instruction::rounded_cone(*r1, *r2, *half_height));
            }

            SdfNode::Pyramid { half_height } => {
                self.instructions.push(Instruction::pyramid(*half_height));
            }

            SdfNode::Octahedron { size } => {
                self.instructions.push(Instruction::octahedron(*size));
            }

            SdfNode::HexPrism { hex_radius, half_height } => {
                self.instructions.push(Instruction::hex_prism(*hex_radius, *half_height));
            }

            SdfNode::Link { half_length, r1, r2 } => {
                self.instructions.push(Instruction::link(*half_length, *r1, *r2));
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
            | SdfNode::Helix { .. } => {
                panic!("This primitive is interpreter/transpiler-only. Use eval() or shader transpiler instead.");
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

            // === Transforms ===
            // For transforms, we: emit transform, compile child, emit pop
            SdfNode::Translate { child, offset } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::translate(offset.x, offset.y, offset.z));
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

            SdfNode::RepeatFinite { child, count, spacing } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::repeat_finite(
                    count[0] as f32, count[1] as f32, count[2] as f32,
                    spacing.x, spacing.y, spacing.z,
                ));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Noise { child, amplitude, frequency, seed } => {
                // Noise is a post-processing modifier that adds noise to the distance
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::noise(*amplitude, *frequency, *seed));
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
                self.instructions.push(Instruction::elongate(amount.x, amount.y, amount.z));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            SdfNode::Mirror { child, axes } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::mirror(axes.x, axes.y, axes.z));
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
                self.instructions.push(Instruction::polar_repeat(*count as f32));
                self.compile_node(child);
                self.instructions.push(Instruction::pop_transform());
                self.instructions[inst_idx].skip_offset = self.instructions.len() as u32;
            }

            // WithMaterial is transparent for distance evaluation
            SdfNode::WithMaterial { child, .. } => {
                self.compile_node(child);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::opcode::OpCode;

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

        // 2 instructions * 32 bytes = 64 bytes
        assert_eq!(compiled.memory_size(), 64);
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

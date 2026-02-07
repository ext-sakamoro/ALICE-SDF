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

            // === Extended Primitives (38 new) ===
            SdfNode::RoundedBox { half_extents, round_radius } => {
                self.instructions.push(Instruction::rounded_box(
                    half_extents.x, half_extents.y, half_extents.z, *round_radius,
                ));
            }

            SdfNode::CappedCone { half_height, r1, r2 } => {
                self.instructions.push(Instruction::capped_cone(*half_height, *r1, *r2));
            }

            SdfNode::CappedTorus { major_radius, minor_radius, cap_angle } => {
                self.instructions.push(Instruction::capped_torus(*major_radius, *minor_radius, *cap_angle));
            }

            SdfNode::RoundedCylinder { radius, round_radius, half_height } => {
                self.instructions.push(Instruction::rounded_cylinder(*radius, *round_radius, *half_height));
            }

            SdfNode::TriangularPrism { width, half_depth } => {
                self.instructions.push(Instruction::triangular_prism(*width, *half_depth));
            }

            SdfNode::CutSphere { radius, cut_height } => {
                self.instructions.push(Instruction::cut_sphere(*radius, *cut_height));
            }

            SdfNode::CutHollowSphere { radius, cut_height, thickness } => {
                self.instructions.push(Instruction::cut_hollow_sphere(*radius, *cut_height, *thickness));
            }

            SdfNode::DeathStar { ra, rb, d } => {
                self.instructions.push(Instruction::death_star(*ra, *rb, *d));
            }

            SdfNode::SolidAngle { angle, radius } => {
                self.instructions.push(Instruction::solid_angle(*angle, *radius));
            }

            SdfNode::Rhombus { la, lb, half_height, round_radius } => {
                self.instructions.push(Instruction::rhombus(*la, *lb, *half_height, *round_radius));
            }

            SdfNode::Horseshoe { angle, radius, half_length, width, thickness } => {
                self.instructions.push(Instruction::horseshoe(*angle, *radius, *half_length, *width, *thickness));
            }

            SdfNode::Vesica { radius, half_dist } => {
                self.instructions.push(Instruction::vesica(*radius, *half_dist));
            }

            SdfNode::InfiniteCylinder { radius } => {
                self.instructions.push(Instruction::infinite_cylinder(*radius));
            }

            SdfNode::InfiniteCone { angle } => {
                self.instructions.push(Instruction::infinite_cone(*angle));
            }

            SdfNode::Gyroid { scale, thickness } => {
                self.instructions.push(Instruction::gyroid(*scale, *thickness));
            }

            SdfNode::Heart { size } => {
                self.instructions.push(Instruction::heart(*size));
            }

            SdfNode::Tube { outer_radius, thickness, half_height } => {
                self.instructions.push(Instruction::tube(*outer_radius, *thickness, *half_height));
            }

            SdfNode::Barrel { radius, half_height, bulge } => {
                self.instructions.push(Instruction::barrel(*radius, *half_height, *bulge));
            }

            SdfNode::Diamond { radius, half_height } => {
                self.instructions.push(Instruction::diamond(*radius, *half_height));
            }

            SdfNode::ChamferedCube { half_extents, chamfer } => {
                self.instructions.push(Instruction::chamfered_cube(
                    half_extents.x, half_extents.y, half_extents.z, *chamfer,
                ));
            }

            SdfNode::SchwarzP { scale, thickness } => {
                self.instructions.push(Instruction::schwarz_p(*scale, *thickness));
            }

            SdfNode::Superellipsoid { half_extents, e1, e2 } => {
                self.instructions.push(Instruction::superellipsoid(
                    half_extents.x, half_extents.y, half_extents.z, *e1, *e2,
                ));
            }

            SdfNode::RoundedX { width, round_radius, half_height } => {
                self.instructions.push(Instruction::rounded_x(*width, *round_radius, *half_height));
            }

            SdfNode::Pie { angle, radius, half_height } => {
                self.instructions.push(Instruction::pie(*angle, *radius, *half_height));
            }

            SdfNode::Trapezoid { r1, r2, trap_height, half_depth } => {
                self.instructions.push(Instruction::trapezoid(*r1, *r2, *trap_height, *half_depth));
            }

            SdfNode::Parallelogram { width, para_height, skew, half_depth } => {
                self.instructions.push(Instruction::parallelogram(*width, *para_height, *skew, *half_depth));
            }

            SdfNode::Tunnel { width, height_2d, half_depth } => {
                self.instructions.push(Instruction::tunnel(*width, *height_2d, *half_depth));
            }

            SdfNode::UnevenCapsule { r1, r2, cap_height, half_depth } => {
                self.instructions.push(Instruction::uneven_capsule(*r1, *r2, *cap_height, *half_depth));
            }

            SdfNode::Egg { ra, rb } => {
                self.instructions.push(Instruction::egg(*ra, *rb));
            }

            SdfNode::ArcShape { aperture, radius, thickness, half_height } => {
                self.instructions.push(Instruction::arc_shape(*aperture, *radius, *thickness, *half_height));
            }

            SdfNode::Moon { d, ra, rb, half_height } => {
                self.instructions.push(Instruction::moon(*d, *ra, *rb, *half_height));
            }

            SdfNode::CrossShape { length, thickness, round_radius, half_height } => {
                self.instructions.push(Instruction::cross_shape(*length, *thickness, *round_radius, *half_height));
            }

            SdfNode::BlobbyCross { size, half_height } => {
                self.instructions.push(Instruction::blobby_cross(*size, *half_height));
            }

            SdfNode::ParabolaSegment { width, para_height, half_depth } => {
                self.instructions.push(Instruction::parabola_segment(*width, *para_height, *half_depth));
            }

            SdfNode::RegularPolygon { radius, n_sides, half_height } => {
                self.instructions.push(Instruction::regular_polygon(*radius, *n_sides, *half_height));
            }

            SdfNode::StarPolygon { radius, n_points, m, half_height } => {
                self.instructions.push(Instruction::star_polygon(*radius, *n_points, *m, *half_height));
            }

            SdfNode::Stairs { step_width, step_height, n_steps, half_depth } => {
                self.instructions.push(Instruction::stairs(*step_width, *step_height, *n_steps, *half_depth));
            }

            SdfNode::Helix { major_r, minor_r, pitch, half_height } => {
                self.instructions.push(Instruction::helix(*major_r, *minor_r, *pitch, *half_height));
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
                self.instructions.push(Instruction::chamfer_intersection(*r));
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
                self.instructions.push(Instruction::stairs_intersection(*r, *n));
            }

            SdfNode::StairsSubtraction { a, b, r, n } => {
                self.compile_node(a);
                self.compile_node(b);
                self.instructions.push(Instruction::stairs_subtraction(*r, *n));
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

            SdfNode::SweepBezier { child, p0, p1, p2 } => {
                let inst_idx = self.instructions.len();
                self.instructions.push(Instruction::sweep_bezier(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y));
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

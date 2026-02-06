//! Instruction structure for compiled SDF
//!
//! Author: Moroya Sakamoto

use super::opcode::OpCode;

/// A single instruction in the compiled SDF bytecode
///
/// This structure is designed for cache efficiency:
/// - 32-byte aligned (half a cache line on most CPUs)
/// - Contiguous memory layout
/// - No pointers or indirection
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Instruction {
    /// Parameters for the operation
    /// - Primitives: dimensions (radius, half_extents, etc.)
    /// - Operations: smoothing factor k
    /// - Transforms: offset, quaternion, scale
    /// - Modifiers: strength, spacing, etc.
    pub params: [f32; 6],  // 24 bytes

    /// The operation code
    pub opcode: OpCode,    // 1 byte

    /// Flags for special behavior
    /// - bit 0: has_scale_correction (for Scale opcode)
    /// - bit 1-7: reserved
    pub flags: u8,         // 1 byte

    /// Number of child nodes (for debugging/validation)
    pub child_count: u16,  // 2 bytes

    /// Offset to skip this subtree (for BVH pruning in Phase 3)
    /// Points to the instruction index after this subtree
    pub skip_offset: u32,  // 4 bytes
}                          // Total: 32 bytes

impl Instruction {
    /// Create a new instruction
    #[inline]
    pub fn new(opcode: OpCode) -> Self {
        Instruction {
            params: [0.0; 6],
            opcode,
            flags: 0,
            child_count: 0,
            skip_offset: 0,
        }
    }

    /// Create a sphere instruction
    #[inline]
    pub fn sphere(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Sphere);
        inst.params[0] = radius;
        inst
    }

    /// Create a box instruction
    #[inline]
    pub fn box3d(hx: f32, hy: f32, hz: f32) -> Self {
        let mut inst = Self::new(OpCode::Box3d);
        inst.params[0] = hx;
        inst.params[1] = hy;
        inst.params[2] = hz;
        inst
    }

    /// Create a cylinder instruction
    #[inline]
    pub fn cylinder(radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Cylinder);
        inst.params[0] = radius;
        inst.params[1] = half_height;
        inst
    }

    /// Create a torus instruction
    #[inline]
    pub fn torus(major_radius: f32, minor_radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Torus);
        inst.params[0] = major_radius;
        inst.params[1] = minor_radius;
        inst
    }

    /// Create a plane instruction
    #[inline]
    pub fn plane(nx: f32, ny: f32, nz: f32, distance: f32) -> Self {
        let mut inst = Self::new(OpCode::Plane);
        inst.params[0] = nx;
        inst.params[1] = ny;
        inst.params[2] = nz;
        inst.params[3] = distance;
        inst
    }

    /// Create a capsule instruction
    ///
    /// Capsule needs 7 parameters but only 6 fit in `params[]`.
    /// The radius is stored in `skip_offset` as `f32::to_bits()`.
    /// Use `get_capsule_radius()` to read it back safely.
    #[inline]
    pub fn capsule(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Capsule);
        inst.params[0] = ax;
        inst.params[1] = ay;
        inst.params[2] = az;
        inst.params[3] = bx;
        inst.params[4] = by;
        inst.params[5] = bz;
        inst.skip_offset = radius.to_bits();
        inst
    }

    /// Get the capsule radius stored in `skip_offset`
    ///
    /// Capsule stores its 7th parameter (radius) as `f32::to_bits()` in `skip_offset`
    /// because `params[6]` is fully used for the two 3D endpoints.
    #[inline]
    pub fn get_capsule_radius(&self) -> f32 {
        debug_assert_eq!(self.opcode, OpCode::Capsule, "get_capsule_radius() called on non-Capsule instruction");
        f32::from_bits(self.skip_offset)
    }

    /// Returns true if this instruction is a leaf node (primitive or binary op)
    ///
    /// Leaf nodes don't have subtrees and always advance by 1 instruction.
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.opcode.is_primitive() || self.opcode.is_binary_op()
    }

    /// Get the index of the next instruction to execute after this subtree
    ///
    /// - For leaf nodes (primitives, binary ops): `current_index + 1`
    /// - For transforms/modifiers: `skip_offset` (points past the PopTransform)
    /// - For PopTransform/End: `current_index + 1`
    #[inline]
    pub fn next_instruction_index(&self, current_index: usize) -> usize {
        if self.is_leaf() || self.opcode == OpCode::PopTransform || self.opcode == OpCode::End {
            current_index + 1
        } else {
            // For transforms/modifiers, skip_offset stores the subtree end index
            // Exception: Capsule is a primitive (is_leaf() = true), so it won't reach here
            self.skip_offset as usize
        }
    }

    /// Create a cone instruction
    #[inline]
    pub fn cone(radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Cone);
        inst.params[0] = radius;
        inst.params[1] = half_height;
        inst
    }

    /// Create an ellipsoid instruction
    #[inline]
    pub fn ellipsoid(rx: f32, ry: f32, rz: f32) -> Self {
        let mut inst = Self::new(OpCode::Ellipsoid);
        inst.params[0] = rx;
        inst.params[1] = ry;
        inst.params[2] = rz;
        inst
    }

    /// Create a rounded cone instruction
    #[inline]
    pub fn rounded_cone(r1: f32, r2: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::RoundedCone);
        inst.params[0] = r1;
        inst.params[1] = r2;
        inst.params[2] = half_height;
        inst
    }

    /// Create a pyramid instruction
    #[inline]
    pub fn pyramid(half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Pyramid);
        inst.params[0] = half_height;
        inst
    }

    /// Create an octahedron instruction
    #[inline]
    pub fn octahedron(size: f32) -> Self {
        let mut inst = Self::new(OpCode::Octahedron);
        inst.params[0] = size;
        inst
    }

    /// Create a hex prism instruction
    #[inline]
    pub fn hex_prism(hex_radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::HexPrism);
        inst.params[0] = hex_radius;
        inst.params[1] = half_height;
        inst
    }

    /// Create a link instruction
    #[inline]
    pub fn link(half_length: f32, r1: f32, r2: f32) -> Self {
        let mut inst = Self::new(OpCode::Link);
        inst.params[0] = half_length;
        inst.params[1] = r1;
        inst.params[2] = r2;
        inst
    }

    /// Create a union instruction
    #[inline]
    pub fn union() -> Self {
        Self::new(OpCode::Union)
    }

    /// Create an intersection instruction
    #[inline]
    pub fn intersection() -> Self {
        Self::new(OpCode::Intersection)
    }

    /// Create a subtraction instruction
    #[inline]
    pub fn subtraction() -> Self {
        Self::new(OpCode::Subtraction)
    }

    /// Create a smooth union instruction
    #[inline]
    pub fn smooth_union(k: f32) -> Self {
        let mut inst = Self::new(OpCode::SmoothUnion);
        inst.params[0] = k;
        inst
    }

    /// Create a smooth intersection instruction
    #[inline]
    pub fn smooth_intersection(k: f32) -> Self {
        let mut inst = Self::new(OpCode::SmoothIntersection);
        inst.params[0] = k;
        inst
    }

    /// Create a smooth subtraction instruction
    #[inline]
    pub fn smooth_subtraction(k: f32) -> Self {
        let mut inst = Self::new(OpCode::SmoothSubtraction);
        inst.params[0] = k;
        inst
    }

    /// Create a translate instruction
    #[inline]
    pub fn translate(x: f32, y: f32, z: f32) -> Self {
        let mut inst = Self::new(OpCode::Translate);
        inst.params[0] = x;
        inst.params[1] = y;
        inst.params[2] = z;
        inst.child_count = 1;
        inst
    }

    /// Create a rotate instruction (quaternion)
    #[inline]
    pub fn rotate(qx: f32, qy: f32, qz: f32, qw: f32) -> Self {
        let mut inst = Self::new(OpCode::Rotate);
        inst.params[0] = qx;
        inst.params[1] = qy;
        inst.params[2] = qz;
        inst.params[3] = qw;
        inst.child_count = 1;
        inst
    }

    /// Create a scale instruction
    ///
    /// Precomputes `1.0/factor` to eliminate per-pixel division:
    /// - `params[0]` = `1.0 / factor` (inverse, for multiplication)
    /// - `params[1]` = `factor` (for scale correction)
    #[inline]
    pub fn scale(factor: f32) -> Self {
        let mut inst = Self::new(OpCode::Scale);
        inst.params[0] = 1.0 / factor; // precomputed inverse
        inst.params[1] = factor;        // original factor
        inst.flags = 1; // has_scale_correction
        inst.child_count = 1;
        inst
    }

    /// Create a non-uniform scale instruction
    ///
    /// Precomputes inverse factors to eliminate per-pixel division:
    /// - `params[0..3]` = `1.0/sx, 1.0/sy, 1.0/sz` (inverse, for multiplication)
    /// - `params[3]` = `min(sx, sy, sz)` (precomputed scale correction)
    #[inline]
    pub fn scale_non_uniform(sx: f32, sy: f32, sz: f32) -> Self {
        let mut inst = Self::new(OpCode::ScaleNonUniform);
        inst.params[0] = 1.0 / sx; // precomputed inverse
        inst.params[1] = 1.0 / sy;
        inst.params[2] = 1.0 / sz;
        inst.params[3] = sx.min(sy).min(sz); // precomputed min factor
        inst.flags = 1; // has_scale_correction
        inst.child_count = 1;
        inst
    }

    /// Create a twist instruction
    #[inline]
    pub fn twist(strength: f32) -> Self {
        let mut inst = Self::new(OpCode::Twist);
        inst.params[0] = strength;
        inst.child_count = 1;
        inst
    }

    /// Create a bend instruction
    #[inline]
    pub fn bend(curvature: f32) -> Self {
        let mut inst = Self::new(OpCode::Bend);
        inst.params[0] = curvature;
        inst.child_count = 1;
        inst
    }

    /// Create an infinite repeat instruction
    #[inline]
    pub fn repeat_infinite(sx: f32, sy: f32, sz: f32) -> Self {
        let mut inst = Self::new(OpCode::RepeatInfinite);
        inst.params[0] = sx;
        inst.params[1] = sy;
        inst.params[2] = sz;
        inst.child_count = 1;
        inst
    }

    /// Create a finite repeat instruction
    #[inline]
    pub fn repeat_finite(cx: f32, cy: f32, cz: f32, sx: f32, sy: f32, sz: f32) -> Self {
        let mut inst = Self::new(OpCode::RepeatFinite);
        inst.params[0] = cx;
        inst.params[1] = cy;
        inst.params[2] = cz;
        inst.params[3] = sx;
        inst.params[4] = sy;
        inst.params[5] = sz;
        inst.child_count = 1;
        inst
    }

    /// Create a round instruction
    #[inline]
    pub fn round(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Round);
        inst.params[0] = radius;
        inst.child_count = 1;
        inst
    }

    /// Create an onion instruction
    #[inline]
    pub fn onion(thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::Onion);
        inst.params[0] = thickness;
        inst.child_count = 1;
        inst
    }

    /// Create an elongate instruction
    #[inline]
    pub fn elongate(ax: f32, ay: f32, az: f32) -> Self {
        let mut inst = Self::new(OpCode::Elongate);
        inst.params[0] = ax;
        inst.params[1] = ay;
        inst.params[2] = az;
        inst.child_count = 1;
        inst
    }

    /// Create a noise instruction
    #[inline]
    pub fn noise(amplitude: f32, frequency: f32, seed: u32) -> Self {
        let mut inst = Self::new(OpCode::Noise);
        inst.params[0] = amplitude;
        inst.params[1] = frequency;
        inst.params[2] = seed as f32;
        inst.child_count = 1;
        inst
    }

    /// Create a mirror instruction
    #[inline]
    pub fn mirror(ax: f32, ay: f32, az: f32) -> Self {
        let mut inst = Self::new(OpCode::Mirror);
        inst.params[0] = ax;
        inst.params[1] = ay;
        inst.params[2] = az;
        inst.child_count = 1;
        inst
    }

    /// Create a revolution instruction
    #[inline]
    pub fn revolution(offset: f32) -> Self {
        let mut inst = Self::new(OpCode::Revolution);
        inst.params[0] = offset;
        inst.child_count = 1;
        inst
    }

    /// Create an extrude instruction
    #[inline]
    pub fn extrude(half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Extrude);
        inst.params[0] = half_height;
        inst.child_count = 1;
        inst
    }

    /// Create a pop transform instruction
    #[inline]
    pub fn pop_transform() -> Self {
        Self::new(OpCode::PopTransform)
    }

    /// Create an end instruction
    #[inline]
    pub fn end() -> Self {
        Self::new(OpCode::End)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn test_instruction_size() {
        // Ensure instruction is exactly 32 bytes
        assert_eq!(mem::size_of::<Instruction>(), 32);
    }

    #[test]
    fn test_instruction_alignment() {
        // Ensure instruction is 32-byte aligned
        assert_eq!(mem::align_of::<Instruction>(), 32);
    }

    #[test]
    fn test_sphere_instruction() {
        let inst = Instruction::sphere(1.5);
        assert_eq!(inst.opcode, OpCode::Sphere);
        assert_eq!(inst.params[0], 1.5);
    }

    #[test]
    fn test_box_instruction() {
        let inst = Instruction::box3d(1.0, 2.0, 3.0);
        assert_eq!(inst.opcode, OpCode::Box3d);
        assert_eq!(inst.params[0], 1.0);
        assert_eq!(inst.params[1], 2.0);
        assert_eq!(inst.params[2], 3.0);
    }

    #[test]
    fn test_translate_instruction() {
        let inst = Instruction::translate(1.0, 2.0, 3.0);
        assert_eq!(inst.opcode, OpCode::Translate);
        assert_eq!(inst.child_count, 1);
    }

    #[test]
    fn test_capsule_stores_radius() {
        let inst = Instruction::capsule(0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.5);
        assert_eq!(inst.opcode, OpCode::Capsule);
        assert!((inst.get_capsule_radius() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_is_leaf() {
        assert!(Instruction::sphere(1.0).is_leaf());
        assert!(Instruction::box3d(1.0, 1.0, 1.0).is_leaf());
        assert!(Instruction::capsule(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3).is_leaf());
        assert!(Instruction::union().is_leaf());
        assert!(Instruction::smooth_union(0.1).is_leaf());
        assert!(!Instruction::translate(1.0, 0.0, 0.0).is_leaf());
        assert!(!Instruction::scale(2.0).is_leaf());
        assert!(!Instruction::twist(0.5).is_leaf());
    }

    #[test]
    fn test_next_instruction_index() {
        // Leaf: always current + 1
        let sphere = Instruction::sphere(1.0);
        assert_eq!(sphere.next_instruction_index(0), 1);
        assert_eq!(sphere.next_instruction_index(5), 6);

        // Transform: uses skip_offset
        let mut translate = Instruction::translate(1.0, 0.0, 0.0);
        translate.skip_offset = 10;
        assert_eq!(translate.next_instruction_index(0), 10);

        // PopTransform/End: always current + 1
        assert_eq!(Instruction::pop_transform().next_instruction_index(7), 8);
        assert_eq!(Instruction::end().next_instruction_index(3), 4);
    }
}

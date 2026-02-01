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
    #[inline]
    pub fn capsule(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Capsule);
        inst.params[0] = ax;
        inst.params[1] = ay;
        inst.params[2] = az;
        inst.params[3] = bx;
        inst.params[4] = by;
        inst.params[5] = bz;
        // Store radius in flags as f32 bits (we'll use a separate method)
        // Actually, we need 7 params for capsule. Let's store radius differently.
        // For now, we'll handle this in the compiler by using a special encoding.
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
    #[inline]
    pub fn scale(factor: f32) -> Self {
        let mut inst = Self::new(OpCode::Scale);
        inst.params[0] = factor;
        inst.flags = 1; // has_scale_correction
        inst.child_count = 1;
        inst
    }

    /// Create a non-uniform scale instruction
    #[inline]
    pub fn scale_non_uniform(sx: f32, sy: f32, sz: f32) -> Self {
        let mut inst = Self::new(OpCode::ScaleNonUniform);
        inst.params[0] = sx;
        inst.params[1] = sy;
        inst.params[2] = sz;
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
}

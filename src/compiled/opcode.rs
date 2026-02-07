//! OpCode definitions for compiled SDF
//!
//! Author: Moroya Sakamoto

/// Operation codes for the SDF virtual machine
///
/// Each opcode represents either a primitive (pushes to stack),
/// an operation (pops operands, pushes result), or a transform
/// (modifies the evaluation point).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OpCode {
    // === Primitives (push distance to stack) ===
    /// Sphere: params[0] = radius
    Sphere = 0,
    /// Box3d: params[0..3] = half_extents (x, y, z)
    Box3d = 1,
    /// Cylinder: params[0] = radius, params[1] = half_height
    Cylinder = 2,
    /// Torus: params[0] = major_radius, params[1] = minor_radius
    Torus = 3,
    /// Plane: params[0..3] = normal (x, y, z), params[3] = distance
    Plane = 4,
    /// Capsule: params[0..3] = point_a, params[3..6] = point_b (radius in params via separate field)
    Capsule = 5,
    /// Cone: params[0] = radius, params[1] = half_height
    Cone = 6,
    /// Ellipsoid: params[0..3] = radii (x, y, z)
    Ellipsoid = 7,
    /// RoundedCone: params[0] = r1, params[1] = r2, params[2] = half_height
    RoundedCone = 8,
    /// Pyramid: params[0] = half_height
    Pyramid = 9,
    /// Octahedron: params[0] = size
    Octahedron = 10,
    /// HexPrism: params[0] = hex_radius, params[1] = half_height
    HexPrism = 11,
    /// Link: params[0] = half_length, params[1] = r1, params[2] = r2
    Link = 12,

    // === Binary Operations (pop 2, push 1) ===
    /// Union: min(a, b)
    Union = 16,
    /// Intersection: max(a, b)
    Intersection = 17,
    /// Subtraction: max(a, -b)
    Subtraction = 18,
    /// SmoothUnion: params[0] = k
    SmoothUnion = 19,
    /// SmoothIntersection: params[0] = k
    SmoothIntersection = 20,
    /// SmoothSubtraction: params[0] = k
    SmoothSubtraction = 21,

    // === Transforms (modify point, then evaluate child) ===
    /// Translate: params[0..3] = offset (x, y, z)
    Translate = 32,
    /// Rotate: params[0..4] = quaternion (x, y, z, w)
    Rotate = 33,
    /// Scale: params[0] = uniform scale factor
    Scale = 34,
    /// ScaleNonUniform: params[0..3] = scale factors (x, y, z)
    ScaleNonUniform = 35,

    // === Modifiers ===
    /// Twist: params[0] = strength
    Twist = 48,
    /// Bend: params[0] = curvature
    Bend = 49,
    /// RepeatInfinite: params[0..3] = spacing (x, y, z)
    RepeatInfinite = 50,
    /// RepeatFinite: params[0..3] = count, params[3..6] = spacing
    RepeatFinite = 51,
    /// Round: params[0] = radius (post-process: d - radius)
    Round = 52,
    /// Onion: params[0] = thickness (post-process: |d| - thickness)
    Onion = 53,
    /// Elongate: params[0..3] = amount (x, y, z)
    Elongate = 54,
    /// Noise: params[0] = amplitude, params[1] = frequency, params[2] = seed
    /// Post-processes distance: d += noise(p * frequency) * amplitude
    Noise = 55,
    /// Mirror: params[0..3] = axes (non-zero = mirror that axis)
    Mirror = 56,
    /// Revolution: params[0] = offset from Y-axis
    Revolution = 57,
    /// Extrude: params[0] = half_height
    Extrude = 58,
    /// Taper: params[0] = factor
    Taper = 59,
    /// Displacement: params[0] = strength (post-process)
    Displacement = 60,
    /// PolarRepeat: params[0] = count (as f32)
    PolarRepeat = 61,

    // === Control ===
    /// Pop transform from coordinate stack
    PopTransform = 240,
    /// End of program
    End = 255,
}

impl OpCode {
    /// Returns true if this opcode is a primitive (pushes to value stack)
    #[inline]
    pub fn is_primitive(self) -> bool {
        (self as u8) < 16
    }

    /// Returns true if this opcode is a binary operation
    #[inline]
    pub fn is_binary_op(self) -> bool {
        let v = self as u8;
        v >= 16 && v < 32
    }

    /// Returns true if this opcode is a transform
    #[inline]
    pub fn is_transform(self) -> bool {
        let v = self as u8;
        v >= 32 && v < 48
    }

    /// Returns true if this opcode is a modifier
    #[inline]
    pub fn is_modifier(self) -> bool {
        let v = self as u8;
        v >= 48 && v < 64
    }

    /// Returns true if this opcode modifies the evaluation point
    #[inline]
    pub fn modifies_point(self) -> bool {
        self.is_transform() || matches!(
            self,
            OpCode::Twist | OpCode::Bend | OpCode::RepeatInfinite |
            OpCode::RepeatFinite | OpCode::Elongate | OpCode::Mirror |
            OpCode::Revolution | OpCode::Extrude | OpCode::Taper |
            OpCode::PolarRepeat
        )
    }

    /// Returns true if this opcode post-processes the distance value
    #[inline]
    pub fn is_post_process(self) -> bool {
        matches!(self, OpCode::Round | OpCode::Onion | OpCode::Scale | OpCode::Noise | OpCode::Extrude | OpCode::Displacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_categories() {
        assert!(OpCode::Sphere.is_primitive());
        assert!(OpCode::Box3d.is_primitive());
        assert!(!OpCode::Union.is_primitive());

        assert!(OpCode::Union.is_binary_op());
        assert!(OpCode::SmoothUnion.is_binary_op());
        assert!(!OpCode::Sphere.is_binary_op());

        assert!(OpCode::Translate.is_transform());
        assert!(OpCode::Rotate.is_transform());
        assert!(!OpCode::Sphere.is_transform());

        assert!(OpCode::Twist.is_modifier());
        assert!(OpCode::Round.is_modifier());
        assert!(!OpCode::Union.is_modifier());
    }

    #[test]
    fn test_opcode_values() {
        // Ensure opcodes have expected values for serialization
        assert_eq!(OpCode::Sphere as u8, 0);
        assert_eq!(OpCode::Union as u8, 16);
        assert_eq!(OpCode::Translate as u8, 32);
        assert_eq!(OpCode::Twist as u8, 48);
        assert_eq!(OpCode::End as u8, 255);
    }
}

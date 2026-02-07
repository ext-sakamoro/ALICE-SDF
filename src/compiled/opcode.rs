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

    // === Extended Primitives (64-101) ===
    /// RoundedBox: params[0..3] = half_extents, params[3] = round_radius
    RoundedBox = 64,
    /// CappedCone: params[0] = half_height, params[1] = r1, params[2] = r2
    CappedCone = 65,
    /// CappedTorus: params[0] = major_radius, params[1] = minor_radius, params[2] = cap_angle
    CappedTorus = 66,
    /// RoundedCylinder: params[0] = radius, params[1] = round_radius, params[2] = half_height
    RoundedCylinder = 67,
    /// TriangularPrism: params[0] = width, params[1] = half_depth
    TriangularPrism = 68,
    /// CutSphere: params[0] = radius, params[1] = cut_height
    CutSphere = 69,
    /// CutHollowSphere: params[0] = radius, params[1] = cut_height, params[2] = thickness
    CutHollowSphere = 70,
    /// DeathStar: params[0] = ra, params[1] = rb, params[2] = d
    DeathStar = 71,
    /// SolidAngle: params[0] = angle, params[1] = radius
    SolidAngle = 72,
    /// Rhombus: params[0] = la, params[1] = lb, params[2] = half_height, params[3] = round_radius
    Rhombus = 73,
    /// Horseshoe: params[0] = angle, params[1] = radius, params[2] = half_length, params[3] = width, params[4] = thickness
    Horseshoe = 74,
    /// Vesica: params[0] = radius, params[1] = half_dist
    Vesica = 75,
    /// InfiniteCylinder: params[0] = radius
    InfiniteCylinder = 76,
    /// InfiniteCone: params[0] = angle
    InfiniteCone = 77,
    /// Gyroid: params[0] = scale, params[1] = thickness
    Gyroid = 78,
    /// Heart: params[0] = size
    Heart = 79,
    /// Tube: params[0] = outer_radius, params[1] = thickness, params[2] = half_height
    Tube = 80,
    /// Barrel: params[0] = radius, params[1] = half_height, params[2] = bulge
    Barrel = 81,
    /// Diamond: params[0] = radius, params[1] = half_height
    Diamond = 82,
    /// ChamferedCube: params[0..3] = half_extents, params[3] = chamfer
    ChamferedCube = 83,
    /// SchwarzP: params[0] = scale, params[1] = thickness
    SchwarzP = 84,
    /// Superellipsoid: params[0..3] = half_extents, params[3] = e1, params[4] = e2
    Superellipsoid = 85,
    /// RoundedX: params[0] = width, params[1] = round_radius, params[2] = half_height
    RoundedX = 86,
    /// Pie: params[0] = angle, params[1] = radius, params[2] = half_height
    Pie = 87,
    /// Trapezoid: params[0] = r1, params[1] = r2, params[2] = trap_height, params[3] = half_depth
    Trapezoid = 88,
    /// Parallelogram: params[0] = width, params[1] = para_height, params[2] = skew, params[3] = half_depth
    Parallelogram = 89,
    /// Tunnel: params[0] = width, params[1] = height_2d, params[2] = half_depth
    Tunnel = 90,
    /// UnevenCapsule: params[0] = r1, params[1] = r2, params[2] = cap_height, params[3] = half_depth
    UnevenCapsule = 91,
    /// Egg: params[0] = ra, params[1] = rb
    Egg = 92,
    /// ArcShape: params[0] = aperture, params[1] = radius, params[2] = thickness, params[3] = half_height
    ArcShape = 93,
    /// Moon: params[0] = d, params[1] = ra, params[2] = rb, params[3] = half_height
    Moon = 94,
    /// CrossShape: params[0] = length, params[1] = thickness, params[2] = round_radius, params[3] = half_height
    CrossShape = 95,
    /// BlobbyCross: params[0] = size, params[1] = half_height
    BlobbyCross = 96,
    /// ParabolaSegment: params[0] = width, params[1] = para_height, params[2] = half_depth
    ParabolaSegment = 97,
    /// RegularPolygon: params[0] = radius, params[1] = n_sides, params[2] = half_height
    RegularPolygon = 98,
    /// StarPolygon: params[0] = radius, params[1] = n_points, params[2] = m, params[3] = half_height
    StarPolygon = 99,
    /// Stairs: params[0] = step_width, params[1] = step_height, params[2] = n_steps, params[3] = half_depth
    Stairs = 100,
    /// Helix: params[0] = major_r, params[1] = minor_r, params[2] = pitch, params[3] = half_height
    Helix = 101,

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
    /// ChamferUnion: params[0] = r (chamfer radius)
    ChamferUnion = 22,
    /// ChamferIntersection: params[0] = r
    ChamferIntersection = 23,
    /// ChamferSubtraction: params[0] = r
    ChamferSubtraction = 24,
    /// StairsUnion: params[0] = r, params[1] = n
    StairsUnion = 25,
    /// StairsIntersection: params[0] = r, params[1] = n
    StairsIntersection = 26,
    /// StairsSubtraction: params[0] = r, params[1] = n
    StairsSubtraction = 27,

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
    /// SweepBezier: params[0..6] = p0.x, p0.z, p1.x, p1.z, p2.x, p2.z
    SweepBezier = 62,

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
        let v = self as u8;
        v < 16 || (v >= 64 && v <= 101)
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
            OpCode::Revolution | OpCode::Extrude | OpCode::SweepBezier | OpCode::Taper |
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
        assert_eq!(OpCode::RoundedBox as u8, 64);
        assert_eq!(OpCode::Helix as u8, 101);
        assert_eq!(OpCode::End as u8, 255);
    }

    #[test]
    fn test_extended_primitives_are_primitives() {
        assert!(OpCode::RoundedBox.is_primitive());
        assert!(OpCode::Stairs.is_primitive());
        assert!(OpCode::Helix.is_primitive());
        assert!(OpCode::Heart.is_primitive());
        assert!(OpCode::Gyroid.is_primitive());
        assert!(!OpCode::RoundedBox.is_binary_op());
        assert!(!OpCode::RoundedBox.is_transform());
        assert!(!OpCode::RoundedBox.is_modifier());
    }
}

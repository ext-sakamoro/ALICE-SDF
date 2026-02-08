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
    pub params: [f32; 6], // 24 bytes

    /// The operation code
    pub opcode: OpCode, // 1 byte

    /// Flags for special behavior
    /// - bit 0: has_scale_correction (for Scale opcode)
    /// - bit 1-7: reserved
    pub flags: u8, // 1 byte

    /// Number of child nodes (for debugging/validation)
    pub child_count: u16, // 2 bytes

    /// Offset to skip this subtree (for BVH pruning in Phase 3)
    /// Points to the instruction index after this subtree
    pub skip_offset: u32, // 4 bytes
} // Total: 32 bytes

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
        debug_assert_eq!(
            self.opcode,
            OpCode::Capsule,
            "get_capsule_radius() called on non-Capsule instruction"
        );
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

    // === Extended Primitives (38 new) ===

    #[inline]
    pub fn rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> Self {
        let mut inst = Self::new(OpCode::RoundedBox);
        inst.params[0] = hx;
        inst.params[1] = hy;
        inst.params[2] = hz;
        inst.params[3] = round_radius;
        inst
    }

    #[inline]
    pub fn capped_cone(half_height: f32, r1: f32, r2: f32) -> Self {
        let mut inst = Self::new(OpCode::CappedCone);
        inst.params[0] = half_height;
        inst.params[1] = r1;
        inst.params[2] = r2;
        inst
    }

    #[inline]
    pub fn capped_torus(major_radius: f32, minor_radius: f32, cap_angle: f32) -> Self {
        let mut inst = Self::new(OpCode::CappedTorus);
        inst.params[0] = major_radius;
        inst.params[1] = minor_radius;
        inst.params[2] = cap_angle;
        inst
    }

    #[inline]
    pub fn rounded_cylinder(radius: f32, round_radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::RoundedCylinder);
        inst.params[0] = radius;
        inst.params[1] = round_radius;
        inst.params[2] = half_height;
        inst
    }

    #[inline]
    pub fn triangular_prism(width: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::TriangularPrism);
        inst.params[0] = width;
        inst.params[1] = half_depth;
        inst
    }

    #[inline]
    pub fn cut_sphere(radius: f32, cut_height: f32) -> Self {
        let mut inst = Self::new(OpCode::CutSphere);
        inst.params[0] = radius;
        inst.params[1] = cut_height;
        inst
    }

    #[inline]
    pub fn cut_hollow_sphere(radius: f32, cut_height: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::CutHollowSphere);
        inst.params[0] = radius;
        inst.params[1] = cut_height;
        inst.params[2] = thickness;
        inst
    }

    #[inline]
    pub fn death_star(ra: f32, rb: f32, d: f32) -> Self {
        let mut inst = Self::new(OpCode::DeathStar);
        inst.params[0] = ra;
        inst.params[1] = rb;
        inst.params[2] = d;
        inst
    }

    #[inline]
    pub fn solid_angle(angle: f32, radius: f32) -> Self {
        let mut inst = Self::new(OpCode::SolidAngle);
        inst.params[0] = angle;
        inst.params[1] = radius;
        inst
    }

    #[inline]
    pub fn rhombus(la: f32, lb: f32, half_height: f32, round_radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Rhombus);
        inst.params[0] = la;
        inst.params[1] = lb;
        inst.params[2] = half_height;
        inst.params[3] = round_radius;
        inst
    }

    #[inline]
    pub fn horseshoe(
        angle: f32,
        radius: f32,
        half_length: f32,
        width: f32,
        thickness: f32,
    ) -> Self {
        let mut inst = Self::new(OpCode::Horseshoe);
        inst.params[0] = angle;
        inst.params[1] = radius;
        inst.params[2] = half_length;
        inst.params[3] = width;
        inst.params[4] = thickness;
        inst
    }

    #[inline]
    pub fn vesica(radius: f32, half_dist: f32) -> Self {
        let mut inst = Self::new(OpCode::Vesica);
        inst.params[0] = radius;
        inst.params[1] = half_dist;
        inst
    }

    #[inline]
    pub fn infinite_cylinder(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::InfiniteCylinder);
        inst.params[0] = radius;
        inst
    }

    #[inline]
    pub fn infinite_cone(angle: f32) -> Self {
        let mut inst = Self::new(OpCode::InfiniteCone);
        inst.params[0] = angle;
        inst
    }

    #[inline]
    pub fn gyroid(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::Gyroid);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn heart(size: f32) -> Self {
        let mut inst = Self::new(OpCode::Heart);
        inst.params[0] = size;
        inst
    }

    #[inline]
    pub fn tube(outer_radius: f32, thickness: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Tube);
        inst.params[0] = outer_radius;
        inst.params[1] = thickness;
        inst.params[2] = half_height;
        inst
    }

    #[inline]
    pub fn barrel(radius: f32, half_height: f32, bulge: f32) -> Self {
        let mut inst = Self::new(OpCode::Barrel);
        inst.params[0] = radius;
        inst.params[1] = half_height;
        inst.params[2] = bulge;
        inst
    }

    #[inline]
    pub fn diamond(radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Diamond);
        inst.params[0] = radius;
        inst.params[1] = half_height;
        inst
    }

    #[inline]
    pub fn chamfered_cube(hx: f32, hy: f32, hz: f32, chamfer: f32) -> Self {
        let mut inst = Self::new(OpCode::ChamferedCube);
        inst.params[0] = hx;
        inst.params[1] = hy;
        inst.params[2] = hz;
        inst.params[3] = chamfer;
        inst
    }

    #[inline]
    pub fn schwarz_p(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::SchwarzP);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn superellipsoid(hx: f32, hy: f32, hz: f32, e1: f32, e2: f32) -> Self {
        let mut inst = Self::new(OpCode::Superellipsoid);
        inst.params[0] = hx;
        inst.params[1] = hy;
        inst.params[2] = hz;
        inst.params[3] = e1;
        inst.params[4] = e2;
        inst
    }

    #[inline]
    pub fn rounded_x(width: f32, round_radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::RoundedX);
        inst.params[0] = width;
        inst.params[1] = round_radius;
        inst.params[2] = half_height;
        inst
    }

    #[inline]
    pub fn pie(angle: f32, radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Pie);
        inst.params[0] = angle;
        inst.params[1] = radius;
        inst.params[2] = half_height;
        inst
    }

    #[inline]
    pub fn trapezoid(r1: f32, r2: f32, trap_height: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::Trapezoid);
        inst.params[0] = r1;
        inst.params[1] = r2;
        inst.params[2] = trap_height;
        inst.params[3] = half_depth;
        inst
    }

    #[inline]
    pub fn parallelogram(width: f32, para_height: f32, skew: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::Parallelogram);
        inst.params[0] = width;
        inst.params[1] = para_height;
        inst.params[2] = skew;
        inst.params[3] = half_depth;
        inst
    }

    #[inline]
    pub fn tunnel(width: f32, height_2d: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::Tunnel);
        inst.params[0] = width;
        inst.params[1] = height_2d;
        inst.params[2] = half_depth;
        inst
    }

    #[inline]
    pub fn uneven_capsule(r1: f32, r2: f32, cap_height: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::UnevenCapsule);
        inst.params[0] = r1;
        inst.params[1] = r2;
        inst.params[2] = cap_height;
        inst.params[3] = half_depth;
        inst
    }

    #[inline]
    pub fn egg(ra: f32, rb: f32) -> Self {
        let mut inst = Self::new(OpCode::Egg);
        inst.params[0] = ra;
        inst.params[1] = rb;
        inst
    }

    #[inline]
    pub fn arc_shape(aperture: f32, radius: f32, thickness: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::ArcShape);
        inst.params[0] = aperture;
        inst.params[1] = radius;
        inst.params[2] = thickness;
        inst.params[3] = half_height;
        inst
    }

    #[inline]
    pub fn moon(d: f32, ra: f32, rb: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Moon);
        inst.params[0] = d;
        inst.params[1] = ra;
        inst.params[2] = rb;
        inst.params[3] = half_height;
        inst
    }

    #[inline]
    pub fn cross_shape(length: f32, thickness: f32, round_radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::CrossShape);
        inst.params[0] = length;
        inst.params[1] = thickness;
        inst.params[2] = round_radius;
        inst.params[3] = half_height;
        inst
    }

    #[inline]
    pub fn blobby_cross(size: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::BlobbyCross);
        inst.params[0] = size;
        inst.params[1] = half_height;
        inst
    }

    #[inline]
    pub fn parabola_segment(width: f32, para_height: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::ParabolaSegment);
        inst.params[0] = width;
        inst.params[1] = para_height;
        inst.params[2] = half_depth;
        inst
    }

    #[inline]
    pub fn regular_polygon(radius: f32, n_sides: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::RegularPolygon);
        inst.params[0] = radius;
        inst.params[1] = n_sides;
        inst.params[2] = half_height;
        inst
    }

    #[inline]
    pub fn star_polygon(radius: f32, n_points: f32, m: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::StarPolygon);
        inst.params[0] = radius;
        inst.params[1] = n_points;
        inst.params[2] = m;
        inst.params[3] = half_height;
        inst
    }

    #[inline]
    pub fn stairs(step_width: f32, step_height: f32, n_steps: f32, half_depth: f32) -> Self {
        let mut inst = Self::new(OpCode::Stairs);
        inst.params[0] = step_width;
        inst.params[1] = step_height;
        inst.params[2] = n_steps;
        inst.params[3] = half_depth;
        inst
    }

    #[inline]
    pub fn helix(major_r: f32, minor_r: f32, pitch: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Helix);
        inst.params[0] = major_r;
        inst.params[1] = minor_r;
        inst.params[2] = pitch;
        inst.params[3] = half_height;
        inst
    }

    #[inline]
    pub fn tetrahedron(size: f32) -> Self {
        let mut inst = Self::new(OpCode::Tetrahedron);
        inst.params[0] = size;
        inst
    }

    #[inline]
    pub fn dodecahedron(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Dodecahedron);
        inst.params[0] = radius;
        inst
    }

    #[inline]
    pub fn icosahedron(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::Icosahedron);
        inst.params[0] = radius;
        inst
    }

    #[inline]
    pub fn truncated_octahedron(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::TruncatedOctahedron);
        inst.params[0] = radius;
        inst
    }

    #[inline]
    pub fn truncated_icosahedron(radius: f32) -> Self {
        let mut inst = Self::new(OpCode::TruncatedIcosahedron);
        inst.params[0] = radius;
        inst
    }

    #[inline]
    pub fn box_frame(half_extents: glam::Vec3, edge: f32) -> Self {
        let mut inst = Self::new(OpCode::BoxFrame);
        inst.params[0] = half_extents.x;
        inst.params[1] = half_extents.y;
        inst.params[2] = half_extents.z;
        inst.params[3] = edge;
        inst
    }

    #[inline]
    pub fn diamond_surface(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::DiamondSurface);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn neovius(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::Neovius);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn lidinoid(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::Lidinoid);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn iwp(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::IWP);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn frd(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::FRD);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn fischer_koch_s(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::FischerKochS);
        inst.params[0] = scale;
        inst.params[1] = thickness;
        inst
    }

    #[inline]
    pub fn pmy(scale: f32, thickness: f32) -> Self {
        let mut inst = Self::new(OpCode::PMY);
        inst.params[0] = scale;
        inst.params[1] = thickness;
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
        let k_safe = k.max(1e-10);
        inst.params[0] = k_safe;
        inst.params[1] = 1.0 / k_safe; // Division Exorcism: precomputed reciprocal
        inst
    }

    /// Create a smooth intersection instruction
    #[inline]
    pub fn smooth_intersection(k: f32) -> Self {
        let mut inst = Self::new(OpCode::SmoothIntersection);
        let k_safe = k.max(1e-10);
        inst.params[0] = k_safe;
        inst.params[1] = 1.0 / k_safe; // Division Exorcism: precomputed reciprocal
        inst
    }

    /// Create a smooth subtraction instruction
    #[inline]
    pub fn smooth_subtraction(k: f32) -> Self {
        let mut inst = Self::new(OpCode::SmoothSubtraction);
        let k_safe = k.max(1e-10);
        inst.params[0] = k_safe;
        inst.params[1] = 1.0 / k_safe; // Division Exorcism: precomputed reciprocal
        inst
    }

    /// Create a chamfer union instruction
    #[inline]
    pub fn chamfer_union(r: f32) -> Self {
        let mut inst = Self::new(OpCode::ChamferUnion);
        inst.params[0] = r.max(0.0);
        inst
    }

    /// Create a chamfer intersection instruction
    #[inline]
    pub fn chamfer_intersection(r: f32) -> Self {
        let mut inst = Self::new(OpCode::ChamferIntersection);
        inst.params[0] = r.max(0.0);
        inst
    }

    /// Create a chamfer subtraction instruction
    #[inline]
    pub fn chamfer_subtraction(r: f32) -> Self {
        let mut inst = Self::new(OpCode::ChamferSubtraction);
        inst.params[0] = r.max(0.0);
        inst
    }

    /// Create a stairs union instruction
    #[inline]
    pub fn stairs_union(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::StairsUnion);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    /// Create a stairs intersection instruction
    #[inline]
    pub fn stairs_intersection(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::StairsIntersection);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    /// Create a stairs subtraction instruction
    #[inline]
    pub fn stairs_subtraction(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::StairsSubtraction);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    #[inline]
    pub fn xor() -> Self {
        Self::new(OpCode::XOR)
    }

    #[inline]
    pub fn morph(t: f32) -> Self {
        let mut inst = Self::new(OpCode::Morph);
        inst.params[0] = t;
        inst
    }

    #[inline]
    pub fn columns_union(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::ColumnsUnion);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    #[inline]
    pub fn columns_intersection(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::ColumnsIntersection);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    #[inline]
    pub fn columns_subtraction(r: f32, n: f32) -> Self {
        let mut inst = Self::new(OpCode::ColumnsSubtraction);
        inst.params[0] = r.max(1e-10);
        inst.params[1] = n.max(1.0);
        inst
    }

    #[inline]
    pub fn pipe(r: f32) -> Self {
        let mut inst = Self::new(OpCode::Pipe);
        inst.params[0] = r;
        inst
    }

    #[inline]
    pub fn engrave(r: f32) -> Self {
        let mut inst = Self::new(OpCode::Engrave);
        inst.params[0] = r;
        inst
    }

    #[inline]
    pub fn groove(ra: f32, rb: f32) -> Self {
        let mut inst = Self::new(OpCode::Groove);
        inst.params[0] = ra;
        inst.params[1] = rb;
        inst
    }

    #[inline]
    pub fn tongue(ra: f32, rb: f32) -> Self {
        let mut inst = Self::new(OpCode::Tongue);
        inst.params[0] = ra;
        inst.params[1] = rb;
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
        inst.params[1] = factor; // original factor
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
        // Division Exorcism: precomputed reciprocal spacing
        inst.params[3] = 1.0 / sx;
        inst.params[4] = 1.0 / sy;
        inst.params[5] = 1.0 / sz;
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

    /// Create a sweep bezier instruction
    #[inline]
    pub fn sweep_bezier(p0x: f32, p0z: f32, p1x: f32, p1z: f32, p2x: f32, p2z: f32) -> Self {
        let mut inst = Self::new(OpCode::SweepBezier);
        inst.params[0] = p0x;
        inst.params[1] = p0z;
        inst.params[2] = p1x;
        inst.params[3] = p1z;
        inst.params[4] = p2x;
        inst.params[5] = p2z;
        inst.child_count = 1;
        inst
    }

    /// Create a taper instruction
    #[inline]
    pub fn taper(factor: f32) -> Self {
        let mut inst = Self::new(OpCode::Taper);
        inst.params[0] = factor;
        inst.child_count = 1;
        inst
    }

    /// Create a displacement instruction
    #[inline]
    pub fn displacement(strength: f32) -> Self {
        let mut inst = Self::new(OpCode::Displacement);
        inst.params[0] = strength;
        inst.child_count = 1;
        inst
    }

    /// Create a polar repeat instruction
    #[inline]
    pub fn polar_repeat(count: f32) -> Self {
        let mut inst = Self::new(OpCode::PolarRepeat);
        inst.params[0] = count;
        // Division Exorcism: precomputed sector angle and its reciprocal
        let sector = std::f32::consts::TAU / count;
        inst.params[1] = sector; // TAU / count
        inst.params[2] = count / std::f32::consts::TAU; // 1.0 / sector = count / TAU
        inst.child_count = 1;
        inst
    }

    /// Create an octant mirror instruction
    #[inline]
    pub fn octant_mirror() -> Self {
        let mut inst = Self::new(OpCode::OctantMirror);
        inst.child_count = 1;
        inst
    }

    /// Create a circle 2D instruction
    #[inline]
    pub fn circle_2d(radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Circle2D);
        inst.params[0] = radius;
        inst.params[1] = half_height;
        inst
    }

    /// Create a rect 2D instruction
    #[inline]
    pub fn rect_2d(half_extents: glam::Vec2, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Rect2D);
        inst.params[0] = half_extents.x;
        inst.params[1] = half_extents.y;
        inst.params[2] = half_height;
        inst
    }

    /// Create a segment 2D instruction
    #[inline]
    pub fn segment_2d(a: glam::Vec2, b: glam::Vec2, thickness: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Segment2D);
        inst.params[0] = a.x;
        inst.params[1] = a.y;
        inst.params[2] = b.x;
        inst.params[3] = b.y;
        inst.params[4] = thickness;
        inst.params[5] = half_height;
        inst
    }

    /// Create a polygon 2D instruction
    #[inline]
    pub fn polygon_2d(half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Polygon2D);
        inst.params[0] = half_height;
        inst
    }

    /// Create a rounded rect 2D instruction
    #[inline]
    pub fn rounded_rect_2d(half_extents: glam::Vec2, round_radius: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::RoundedRect2D);
        inst.params[0] = half_extents.x;
        inst.params[1] = half_extents.y;
        inst.params[2] = round_radius;
        inst.params[3] = half_height;
        inst
    }

    /// Create an annular 2D instruction
    #[inline]
    pub fn annular_2d(outer_radius: f32, thickness: f32, half_height: f32) -> Self {
        let mut inst = Self::new(OpCode::Annular2D);
        inst.params[0] = outer_radius;
        inst.params[1] = thickness;
        inst.params[2] = half_height;
        inst
    }

    /// Create an exp smooth union instruction
    #[inline]
    pub fn exp_smooth_union(k: f32) -> Self {
        let mut inst = Self::new(OpCode::ExpSmoothUnion);
        inst.params[0] = k;
        inst.child_count = 2;
        inst
    }

    /// Create an exp smooth intersection instruction
    #[inline]
    pub fn exp_smooth_intersection(k: f32) -> Self {
        let mut inst = Self::new(OpCode::ExpSmoothIntersection);
        inst.params[0] = k;
        inst.child_count = 2;
        inst
    }

    /// Create an exp smooth subtraction instruction
    #[inline]
    pub fn exp_smooth_subtraction(k: f32) -> Self {
        let mut inst = Self::new(OpCode::ExpSmoothSubtraction);
        inst.params[0] = k;
        inst.child_count = 2;
        inst
    }

    /// Create a shear instruction
    #[inline]
    pub fn shear(shear: glam::Vec3) -> Self {
        let mut inst = Self::new(OpCode::Shear);
        inst.params[0] = shear.x;
        inst.params[1] = shear.y;
        inst.params[2] = shear.z;
        inst.child_count = 1;
        inst
    }

    /// Create an animated instruction
    #[inline]
    pub fn animated(speed: f32, amplitude: f32) -> Self {
        let mut inst = Self::new(OpCode::Animated);
        inst.params[0] = speed;
        inst.params[1] = amplitude;
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

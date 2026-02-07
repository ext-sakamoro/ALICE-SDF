//! Core types for ALICE-SDF
//!
//! Defines the SdfNode tree structure and related types.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Signed Distance Function Node
///
/// Represents a node in the SDF tree. Each node can be:
/// - A primitive shape (sphere, box, etc.)
/// - An operation combining two shapes (union, intersection, etc.)
/// - A transform applied to a child node
/// - A modifier deforming a child node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SdfNode {
    // === Primitives ===
    /// Sphere with radius
    Sphere {
        /// Sphere radius
        radius: f32,
    },

    /// Axis-aligned box with half-extents
    Box3d {
        /// Half-extents along each axis
        half_extents: Vec3,
    },

    /// Cylinder along Y-axis with radius and half-height
    Cylinder {
        /// Cylinder radius
        radius: f32,
        /// Half the cylinder height
        half_height: f32,
    },

    /// Torus in XZ plane with major and minor radius
    Torus {
        /// Distance from center to tube center
        major_radius: f32,
        /// Tube radius
        minor_radius: f32,
    },

    /// Infinite plane with normal and distance from origin
    Plane {
        /// Plane normal direction
        normal: Vec3,
        /// Signed distance from origin along normal
        distance: f32,
    },

    /// Capsule between two points with radius
    Capsule {
        /// First endpoint
        point_a: Vec3,
        /// Second endpoint
        point_b: Vec3,
        /// Capsule radius
        radius: f32,
    },

    /// Cone along Y-axis with base radius and half-height
    Cone {
        /// Base radius
        radius: f32,
        /// Half the cone height
        half_height: f32,
    },

    /// Ellipsoid with semi-axis radii
    Ellipsoid {
        /// Semi-axis radii (x, y, z)
        radii: Vec3,
    },

    /// Rounded cone along Y-axis with bottom radius r1, top radius r2
    RoundedCone {
        /// Bottom sphere radius
        r1: f32,
        /// Top sphere radius
        r2: f32,
        /// Half the cone height
        half_height: f32,
    },

    /// 4-sided pyramid with square base (side=1) centered at origin
    Pyramid {
        /// Half the pyramid height
        half_height: f32,
    },

    /// Regular octahedron centered at origin
    Octahedron {
        /// Distance from center to vertex
        size: f32,
    },

    /// Hexagonal prism centered at origin
    HexPrism {
        /// Hexagon circumradius
        hex_radius: f32,
        /// Half the prism height
        half_height: f32,
    },

    /// Chain link shape centered at origin
    Link {
        /// Half the straight section length
        half_length: f32,
        /// Major radius (center to tube center)
        r1: f32,
        /// Minor radius (tube thickness)
        r2: f32,
    },

    /// Triangle defined by three vertices (unsigned distance)
    Triangle {
        /// First vertex
        point_a: Vec3,
        /// Second vertex
        point_b: Vec3,
        /// Third vertex
        point_c: Vec3,
    },

    /// Quadratic Bezier curve with radius (tube around curve)
    Bezier {
        /// Start point
        point_a: Vec3,
        /// Control point
        point_b: Vec3,
        /// End point
        point_c: Vec3,
        /// Tube radius
        radius: f32,
    },

    /// Box with rounded edges
    RoundedBox {
        /// Half-extents (before rounding)
        half_extents: Vec3,
        /// Edge rounding radius
        round_radius: f32,
    },

    /// Capped cone (frustum) along Y-axis
    CappedCone {
        /// Half the cone height
        half_height: f32,
        /// Bottom radius
        r1: f32,
        /// Top radius
        r2: f32,
    },

    /// Partial torus (arc) in XZ plane
    CappedTorus {
        /// Distance from center to tube center
        major_radius: f32,
        /// Tube radius
        minor_radius: f32,
        /// Half opening angle (radians)
        cap_angle: f32,
    },

    /// Cylinder with rounded edges along Y-axis
    RoundedCylinder {
        /// Cylinder radius
        radius: f32,
        /// Edge rounding radius
        round_radius: f32,
        /// Half the cylinder height
        half_height: f32,
    },

    /// Equilateral triangular prism along Z-axis
    TriangularPrism {
        /// Half-width of the triangle cross-section
        width: f32,
        /// Half the prism depth along Z
        half_depth: f32,
    },

    /// Sphere with a planar cut
    CutSphere {
        /// Sphere radius
        radius: f32,
        /// Y height of the cut plane
        cut_height: f32,
    },

    /// Hollow sphere shell with a planar cut
    CutHollowSphere {
        /// Sphere radius
        radius: f32,
        /// Y height of the cut plane
        cut_height: f32,
        /// Shell thickness
        thickness: f32,
    },

    /// Death Star: sphere with spherical indentation
    DeathStar {
        /// Main sphere radius
        ra: f32,
        /// Carving sphere radius
        rb: f32,
        /// Distance between sphere centers
        d: f32,
    },

    /// Solid angle (cone sector)
    SolidAngle {
        /// Half-angle in radians
        angle: f32,
        /// Bounding radius
        radius: f32,
    },

    /// 3D rhombus (diamond shape)
    Rhombus {
        /// Half-diagonal along X
        la: f32,
        /// Half-diagonal along Z
        lb: f32,
        /// Half-height along Y
        half_height: f32,
        /// Edge rounding
        round_radius: f32,
    },

    /// Horseshoe (U-shape)
    Horseshoe {
        /// Opening half-angle (radians)
        angle: f32,
        /// Ring radius
        radius: f32,
        /// Straight extension half-length
        half_length: f32,
        /// Cross-section half-width
        width: f32,
        /// Cross-section half-thickness
        thickness: f32,
    },

    /// 3D vesica (lens shape, revolved)
    Vesica {
        /// Arc radius
        radius: f32,
        /// Half distance between arc centers
        half_dist: f32,
    },

    /// Infinite cylinder along Y-axis
    InfiniteCylinder {
        /// Cylinder radius
        radius: f32,
    },

    /// Infinite cone along Y-axis
    InfiniteCone {
        /// Half-angle in radians
        angle: f32,
    },

    /// Gyroid triply-periodic minimal surface
    Gyroid {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// 3D heart shape (revolved contour)
    Heart {
        /// Overall size
        size: f32,
    },

    // === Operations ===
    /// Union of two shapes (min distance)
    Union {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
    },

    /// Intersection of two shapes (max distance)
    Intersection {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
    },

    /// Subtraction: a minus b (max of a and -b)
    Subtraction {
        /// Shape to subtract from
        a: Arc<SdfNode>,
        /// Shape to subtract
        b: Arc<SdfNode>,
    },

    /// Smooth union with blending factor k
    SmoothUnion {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Blending radius
        k: f32,
    },

    /// Smooth intersection with blending factor k
    SmoothIntersection {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Blending radius
        k: f32,
    },

    /// Smooth subtraction with blending factor k
    SmoothSubtraction {
        /// Shape to subtract from
        a: Arc<SdfNode>,
        /// Shape to subtract
        b: Arc<SdfNode>,
        /// Blending radius
        k: f32,
    },

    // === Transforms ===
    /// Translation
    Translate {
        /// Child node
        child: Arc<SdfNode>,
        /// Translation offset
        offset: Vec3,
    },

    /// Rotation (quaternion)
    Rotate {
        /// Child node
        child: Arc<SdfNode>,
        /// Rotation quaternion
        rotation: Quat,
    },

    /// Uniform scale
    Scale {
        /// Child node
        child: Arc<SdfNode>,
        /// Scale factor
        factor: f32,
    },

    /// Non-uniform scale (stretches the shape)
    ScaleNonUniform {
        /// Child node
        child: Arc<SdfNode>,
        /// Scale factors per axis
        factors: Vec3,
    },

    // === Modifiers ===
    /// Twist around Y-axis (radians per unit height)
    Twist {
        /// Child node
        child: Arc<SdfNode>,
        /// Twist strength (radians per unit)
        strength: f32,
    },

    /// Bend around Y-axis
    Bend {
        /// Child node
        child: Arc<SdfNode>,
        /// Bend curvature
        curvature: f32,
    },

    /// Infinite repetition with spacing
    RepeatInfinite {
        /// Child node
        child: Arc<SdfNode>,
        /// Spacing between repetitions
        spacing: Vec3,
    },

    /// Finite repetition with count and spacing
    RepeatFinite {
        /// Child node
        child: Arc<SdfNode>,
        /// Repeat count per axis
        count: [u32; 3],
        /// Spacing between repetitions
        spacing: Vec3,
    },

    /// Perlin noise displacement
    Noise {
        /// Child node
        child: Arc<SdfNode>,
        /// Noise amplitude
        amplitude: f32,
        /// Noise frequency
        frequency: f32,
        /// Random seed
        seed: u32,
    },

    /// Round edges by subtracting radius
    Round {
        /// Child node
        child: Arc<SdfNode>,
        /// Rounding radius
        radius: f32,
    },

    /// Onion: creates a shell with thickness
    Onion {
        /// Child node
        child: Arc<SdfNode>,
        /// Shell thickness
        thickness: f32,
    },

    /// Elongate along an axis
    Elongate {
        /// Child node
        child: Arc<SdfNode>,
        /// Elongation amount per axis
        amount: Vec3,
    },

    /// Mirror along specified axes (takes absolute value of coordinates)
    Mirror {
        /// Child node
        child: Arc<SdfNode>,
        /// Mirror axes (nonzero = mirrored)
        axes: Vec3,
    },

    /// Revolution around Y-axis (creates rotational symmetry)
    Revolution {
        /// Child node
        child: Arc<SdfNode>,
        /// Radial offset
        offset: f32,
    },

    /// Extrude along Z-axis (creates 3D from XY cross-section)
    Extrude {
        /// Child node
        child: Arc<SdfNode>,
        /// Half the extrusion height
        half_height: f32,
    },

    /// Taper: scale XZ by inverse of (1 - y*factor)
    Taper {
        /// Child node
        child: Arc<SdfNode>,
        /// Taper factor
        factor: f32,
    },

    /// Sin-based displacement (post-processing modifier)
    Displacement {
        /// Child node
        child: Arc<SdfNode>,
        /// Displacement strength
        strength: f32,
    },

    /// Polar repetition around Y-axis
    PolarRepeat {
        /// Child node
        child: Arc<SdfNode>,
        /// Number of copies around Y-axis
        count: u32,
    },

    /// Assign a material ID to a subtree (transparent for distance evaluation)
    WithMaterial {
        /// Child node
        child: Arc<SdfNode>,
        /// Material ID (indexes into MaterialLibrary)
        material_id: u32,
    },
}

impl SdfNode {
    // === Primitive constructors ===

    /// Create a sphere with the given radius
    #[inline]
    pub fn sphere(radius: f32) -> Self {
        SdfNode::Sphere { radius }
    }

    /// Create an axis-aligned box with the given dimensions
    #[inline]
    pub fn box3d(width: f32, height: f32, depth: f32) -> Self {
        SdfNode::Box3d {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
        }
    }

    /// Create a cylinder along Y-axis
    #[inline]
    pub fn cylinder(radius: f32, height: f32) -> Self {
        SdfNode::Cylinder {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a torus in the XZ plane
    #[inline]
    pub fn torus(major_radius: f32, minor_radius: f32) -> Self {
        SdfNode::Torus {
            major_radius,
            minor_radius,
        }
    }

    /// Create an infinite plane
    #[inline]
    pub fn plane(normal: Vec3, distance: f32) -> Self {
        SdfNode::Plane {
            normal: normal.normalize(),
            distance,
        }
    }

    /// Create a capsule between two points
    #[inline]
    pub fn capsule(a: Vec3, b: Vec3, radius: f32) -> Self {
        SdfNode::Capsule {
            point_a: a,
            point_b: b,
            radius,
        }
    }

    /// Create a cone along Y-axis
    #[inline]
    pub fn cone(radius: f32, height: f32) -> Self {
        SdfNode::Cone {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create an ellipsoid with given semi-axis radii
    #[inline]
    pub fn ellipsoid(rx: f32, ry: f32, rz: f32) -> Self {
        SdfNode::Ellipsoid {
            radii: Vec3::new(rx, ry, rz),
        }
    }

    /// Create a rounded cone along Y-axis
    #[inline]
    pub fn rounded_cone(r1: f32, r2: f32, height: f32) -> Self {
        SdfNode::RoundedCone {
            r1,
            r2,
            half_height: height * 0.5,
        }
    }

    /// Create a 4-sided pyramid
    #[inline]
    pub fn pyramid(height: f32) -> Self {
        SdfNode::Pyramid {
            half_height: height * 0.5,
        }
    }

    /// Create a regular octahedron
    #[inline]
    pub fn octahedron(size: f32) -> Self {
        SdfNode::Octahedron { size }
    }

    /// Create a hexagonal prism
    #[inline]
    pub fn hex_prism(hex_radius: f32, height: f32) -> Self {
        SdfNode::HexPrism {
            hex_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a chain link shape
    #[inline]
    pub fn link(length: f32, r1: f32, r2: f32) -> Self {
        SdfNode::Link {
            half_length: length * 0.5,
            r1,
            r2,
        }
    }

    /// Create a triangle from three vertices
    #[inline]
    pub fn triangle(a: Vec3, b: Vec3, c: Vec3) -> Self {
        SdfNode::Triangle {
            point_a: a,
            point_b: b,
            point_c: c,
        }
    }

    /// Create a quadratic Bezier curve tube
    #[inline]
    pub fn bezier(a: Vec3, b: Vec3, c: Vec3, radius: f32) -> Self {
        SdfNode::Bezier {
            point_a: a,
            point_b: b,
            point_c: c,
            radius,
        }
    }

    /// Create a rounded box
    #[inline]
    pub fn rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> Self {
        SdfNode::RoundedBox {
            half_extents: Vec3::new(hx, hy, hz),
            round_radius,
        }
    }

    /// Create a capped cone (frustum)
    #[inline]
    pub fn capped_cone(height: f32, r1: f32, r2: f32) -> Self {
        SdfNode::CappedCone {
            half_height: height * 0.5,
            r1,
            r2,
        }
    }

    /// Create a capped torus (arc)
    #[inline]
    pub fn capped_torus(major_radius: f32, minor_radius: f32, cap_angle: f32) -> Self {
        SdfNode::CappedTorus {
            major_radius,
            minor_radius,
            cap_angle,
        }
    }

    /// Create a rounded cylinder
    #[inline]
    pub fn rounded_cylinder(radius: f32, round_radius: f32, height: f32) -> Self {
        SdfNode::RoundedCylinder {
            radius,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a triangular prism
    #[inline]
    pub fn triangular_prism(width: f32, depth: f32) -> Self {
        SdfNode::TriangularPrism {
            width,
            half_depth: depth * 0.5,
        }
    }

    /// Create a cut sphere
    #[inline]
    pub fn cut_sphere(radius: f32, cut_height: f32) -> Self {
        SdfNode::CutSphere { radius, cut_height }
    }

    /// Create a cut hollow sphere
    #[inline]
    pub fn cut_hollow_sphere(radius: f32, cut_height: f32, thickness: f32) -> Self {
        SdfNode::CutHollowSphere {
            radius,
            cut_height,
            thickness,
        }
    }

    /// Create a Death Star shape
    #[inline]
    pub fn death_star(ra: f32, rb: f32, d: f32) -> Self {
        SdfNode::DeathStar { ra, rb, d }
    }

    /// Create a solid angle
    #[inline]
    pub fn solid_angle(angle: f32, radius: f32) -> Self {
        SdfNode::SolidAngle { angle, radius }
    }

    /// Create a rhombus
    #[inline]
    pub fn rhombus(la: f32, lb: f32, height: f32, round_radius: f32) -> Self {
        SdfNode::Rhombus {
            la,
            lb,
            half_height: height * 0.5,
            round_radius,
        }
    }

    /// Create a horseshoe shape
    #[inline]
    pub fn horseshoe(angle: f32, radius: f32, length: f32, width: f32, thickness: f32) -> Self {
        SdfNode::Horseshoe {
            angle,
            radius,
            half_length: length * 0.5,
            width,
            thickness,
        }
    }

    /// Create a 3D vesica
    #[inline]
    pub fn vesica(radius: f32, dist: f32) -> Self {
        SdfNode::Vesica {
            radius,
            half_dist: dist * 0.5,
        }
    }

    /// Create an infinite cylinder
    #[inline]
    pub fn infinite_cylinder(radius: f32) -> Self {
        SdfNode::InfiniteCylinder { radius }
    }

    /// Create an infinite cone
    #[inline]
    pub fn infinite_cone(angle: f32) -> Self {
        SdfNode::InfiniteCone { angle }
    }

    /// Create a gyroid surface
    #[inline]
    pub fn gyroid(scale: f32, thickness: f32) -> Self {
        SdfNode::Gyroid { scale, thickness }
    }

    /// Create a 3D heart
    #[inline]
    pub fn heart(size: f32) -> Self {
        SdfNode::Heart { size }
    }

    // === Operation methods ===

    /// Union with another shape
    #[inline]
    pub fn union(self, other: SdfNode) -> Self {
        SdfNode::Union {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Intersection with another shape
    #[inline]
    pub fn intersection(self, other: SdfNode) -> Self {
        SdfNode::Intersection {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Subtract another shape from this one
    #[inline]
    pub fn subtract(self, other: SdfNode) -> Self {
        SdfNode::Subtraction {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Smooth union with another shape
    #[inline]
    pub fn smooth_union(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth intersection with another shape
    #[inline]
    pub fn smooth_intersection(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Smooth subtraction of another shape
    #[inline]
    pub fn smooth_subtract(self, other: SdfNode, k: f32) -> Self {
        SdfNode::SmoothSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    // === Transform methods ===

    /// Translate by offset
    #[inline]
    pub fn translate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset: Vec3::new(x, y, z),
        }
    }

    /// Translate by vector
    #[inline]
    pub fn translate_vec(self, offset: Vec3) -> Self {
        SdfNode::Translate {
            child: Arc::new(self),
            offset,
        }
    }

    /// Rotate by quaternion
    #[inline]
    pub fn rotate(self, rotation: Quat) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation,
        }
    }

    /// Rotate by Euler angles (radians)
    #[inline]
    pub fn rotate_euler(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Rotate {
            child: Arc::new(self),
            rotation: Quat::from_euler(glam::EulerRot::XYZ, x, y, z),
        }
    }

    /// Uniform scale
    #[inline]
    pub fn scale(self, factor: f32) -> Self {
        SdfNode::Scale {
            child: Arc::new(self),
            factor,
        }
    }

    /// Non-uniform scale
    #[inline]
    pub fn scale_xyz(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::ScaleNonUniform {
            child: Arc::new(self),
            factors: Vec3::new(x, y, z),
        }
    }

    // === Modifier methods ===

    /// Twist around Y-axis
    #[inline]
    pub fn twist(self, strength: f32) -> Self {
        SdfNode::Twist {
            child: Arc::new(self),
            strength,
        }
    }

    /// Bend around Y-axis
    #[inline]
    pub fn bend(self, curvature: f32) -> Self {
        SdfNode::Bend {
            child: Arc::new(self),
            curvature,
        }
    }

    /// Infinite repetition
    #[inline]
    pub fn repeat_infinite(self, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        SdfNode::RepeatInfinite {
            child: Arc::new(self),
            spacing: Vec3::new(spacing_x, spacing_y, spacing_z),
        }
    }

    /// Finite repetition
    #[inline]
    pub fn repeat_finite(self, count: [u32; 3], spacing: Vec3) -> Self {
        SdfNode::RepeatFinite {
            child: Arc::new(self),
            count,
            spacing,
        }
    }

    /// Perlin noise displacement
    #[inline]
    pub fn noise(self, amplitude: f32, frequency: f32, seed: u32) -> Self {
        SdfNode::Noise {
            child: Arc::new(self),
            amplitude,
            frequency,
            seed,
        }
    }

    /// Round edges
    #[inline]
    pub fn round(self, radius: f32) -> Self {
        SdfNode::Round {
            child: Arc::new(self),
            radius,
        }
    }

    /// Create a shell (onion)
    #[inline]
    pub fn onion(self, thickness: f32) -> Self {
        SdfNode::Onion {
            child: Arc::new(self),
            thickness,
        }
    }

    /// Elongate along an axis
    #[inline]
    pub fn elongate(self, x: f32, y: f32, z: f32) -> Self {
        SdfNode::Elongate {
            child: Arc::new(self),
            amount: Vec3::new(x, y, z),
        }
    }

    /// Mirror along specified axes
    #[inline]
    pub fn mirror(self, x: bool, y: bool, z: bool) -> Self {
        SdfNode::Mirror {
            child: Arc::new(self),
            axes: Vec3::new(
                if x { 1.0 } else { 0.0 },
                if y { 1.0 } else { 0.0 },
                if z { 1.0 } else { 0.0 },
            ),
        }
    }

    /// Revolution around Y-axis
    #[inline]
    pub fn revolution(self, offset: f32) -> Self {
        SdfNode::Revolution {
            child: Arc::new(self),
            offset,
        }
    }

    /// Extrude along Z-axis
    #[inline]
    pub fn extrude(self, height: f32) -> Self {
        SdfNode::Extrude {
            child: Arc::new(self),
            half_height: height * 0.5,
        }
    }

    /// Taper along Y-axis
    #[inline]
    pub fn taper(self, factor: f32) -> Self {
        SdfNode::Taper {
            child: Arc::new(self),
            factor,
        }
    }

    /// Sin-based displacement
    #[inline]
    pub fn displacement(self, strength: f32) -> Self {
        SdfNode::Displacement {
            child: Arc::new(self),
            strength,
        }
    }

    /// Polar repetition around Y-axis
    #[inline]
    pub fn polar_repeat(self, count: u32) -> Self {
        SdfNode::PolarRepeat {
            child: Arc::new(self),
            count,
        }
    }

    /// Assign a material ID to this subtree
    #[inline]
    pub fn with_material(self, material_id: u32) -> Self {
        SdfNode::WithMaterial {
            child: Arc::new(self),
            material_id,
        }
    }

    /// Count total nodes in the tree
    pub fn node_count(&self) -> u32 {
        match self {
            // Primitives: 1 node
            SdfNode::Sphere { .. }
            | SdfNode::Box3d { .. }
            | SdfNode::Cylinder { .. }
            | SdfNode::Torus { .. }
            | SdfNode::Plane { .. }
            | SdfNode::Capsule { .. }
            | SdfNode::Cone { .. }
            | SdfNode::Ellipsoid { .. }
            | SdfNode::RoundedCone { .. }
            | SdfNode::Pyramid { .. }
            | SdfNode::Octahedron { .. }
            | SdfNode::HexPrism { .. }
            | SdfNode::Link { .. }
            | SdfNode::Triangle { .. }
            | SdfNode::Bezier { .. }
            | SdfNode::RoundedBox { .. }
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
            | SdfNode::Heart { .. } => 1,

            // Operations: 1 + children
            SdfNode::Union { a, b }
            | SdfNode::Intersection { a, b }
            | SdfNode::Subtraction { a, b }
            | SdfNode::SmoothUnion { a, b, .. }
            | SdfNode::SmoothIntersection { a, b, .. }
            | SdfNode::SmoothSubtraction { a, b, .. } => 1 + a.node_count() + b.node_count(),

            // Transforms and modifiers: 1 + child
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
            | SdfNode::Revolution { child, .. }
            | SdfNode::Extrude { child, .. }
            | SdfNode::Taper { child, .. }
            | SdfNode::Displacement { child, .. }
            | SdfNode::PolarRepeat { child, .. }
            | SdfNode::WithMaterial { child, .. } => 1 + child.node_count(),
        }
    }
}

/// SDF Tree - top-level container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfTree {
    /// Version string
    pub version: String,
    /// Root node
    pub root: SdfNode,
    /// Optional metadata
    pub metadata: Option<SdfMetadata>,
}

impl SdfTree {
    /// Create a new SDF tree
    pub fn new(root: SdfNode) -> Self {
        SdfTree {
            version: env!("CARGO_PKG_VERSION").to_string(),
            root,
            metadata: None,
        }
    }

    /// Create with metadata
    pub fn with_metadata(root: SdfNode, metadata: SdfMetadata) -> Self {
        SdfTree {
            version: env!("CARGO_PKG_VERSION").to_string(),
            root,
            metadata: Some(metadata),
        }
    }

    /// Get total node count
    pub fn node_count(&self) -> u32 {
        self.root.node_count()
    }
}

/// Optional metadata for SDF trees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdfMetadata {
    /// Name of the model
    pub name: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Bounding box hint (min, max)
    pub bounds: Option<(Vec3, Vec3)>,
    /// Custom key-value pairs
    pub custom: Option<std::collections::HashMap<String, String>>,
}

impl Default for SdfMetadata {
    fn default() -> Self {
        SdfMetadata {
            name: None,
            description: None,
            author: None,
            bounds: None,
            custom: None,
        }
    }
}

/// Axis-aligned bounding box
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Aabb {
    /// Minimum corner
    pub min: Vec3,
    /// Maximum corner
    pub max: Vec3,
}

impl Aabb {
    /// Create a new AABB
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Aabb { min, max }
    }

    /// Create from center and half-extents
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Aabb {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get center point
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get size
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Get half-extents
    pub fn half_extents(&self) -> Vec3 {
        self.size() * 0.5
    }

    /// Check if point is inside
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// Expand to include another AABB
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// Ray for raycasting
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Ray origin point
    pub origin: Vec3,
    /// Ray direction (normalized)
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Ray {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Get point along ray at distance t
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }
}

/// Hit result from raycasting
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Distance along ray
    pub distance: f32,
    /// Hit point
    pub point: Vec3,
    /// Surface normal (approximate)
    pub normal: Vec3,
    /// Number of marching steps
    pub steps: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sphere_creation() {
        let sphere = SdfNode::sphere(1.0);
        assert_eq!(sphere.node_count(), 1);
    }

    #[test]
    fn test_union() {
        let a = SdfNode::sphere(1.0);
        let b = SdfNode::box3d(2.0, 2.0, 2.0);
        let union = a.union(b);
        assert_eq!(union.node_count(), 3);
    }

    #[test]
    fn test_transform_chain() {
        let shape = SdfNode::sphere(1.0)
            .translate(1.0, 0.0, 0.0)
            .rotate_euler(0.0, std::f32::consts::PI, 0.0)
            .scale(2.0);
        assert_eq!(shape.node_count(), 4);
    }

    #[test]
    fn test_sdf_tree() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        assert_eq!(tree.node_count(), 1);
    }

    #[test]
    fn test_aabb() {
        let aabb = Aabb::new(Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, 1.0));
        assert!(aabb.contains(Vec3::ZERO));
        assert!(!aabb.contains(Vec3::new(2.0, 0.0, 0.0)));
    }
}

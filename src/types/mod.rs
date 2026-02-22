//! Core types for ALICE-SDF
//!
//! Defines the SdfNode tree structure and related types.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

mod constructors;
mod containers;
mod modifiers;
mod operations;
mod transforms;

pub use containers::{Aabb, Hit, Ray, SdfMetadata, SdfTree};

/// Category of an SDF node variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdfCategory {
    /// Leaf geometry nodes (spheres, boxes, cylinders, etc.)
    Primitive,
    /// Boolean and blending operations (union, intersection, smooth union, etc.)
    Operation,
    /// Spatial transform nodes (translate, rotate, scale, etc.)
    Transform,
    /// Surface and domain modifiers (twist, bend, shell, etc.)
    Modifier,
}

impl SdfCategory {
    /// Number of SdfNode variants in this category
    pub fn count(self) -> u32 {
        match self {
            SdfCategory::Primitive => 72,
            SdfCategory::Operation => 24,
            SdfCategory::Transform => 7,
            SdfCategory::Modifier => 23,
        }
    }

    /// Total number of all SdfNode variants
    pub fn total() -> u32 {
        SdfCategory::Primitive.count()
            + SdfCategory::Operation.count()
            + SdfCategory::Transform.count()
            + SdfCategory::Modifier.count()
    }
}

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

    /// Hollow cylinder (tube/pipe) along Y-axis
    Tube {
        /// Radius of the tube center-line
        outer_radius: f32,
        /// Half-wall thickness
        thickness: f32,
        /// Half the tube height
        half_height: f32,
    },

    /// Cylinder with parabolic radial bulge along Y-axis
    Barrel {
        /// Base radius at caps
        radius: f32,
        /// Half the barrel height
        half_height: f32,
        /// Additional radius at the middle
        bulge: f32,
    },

    /// Bipyramid (double-cone) with revolution symmetry
    Diamond {
        /// Equator radius
        radius: f32,
        /// Half the total height
        half_height: f32,
    },

    /// Box with chamfered (beveled) edges
    ChamferedCube {
        /// Half-extents along each axis
        half_extents: Vec3,
        /// Chamfer amount
        chamfer: f32,
    },

    /// Schwarz P triply-periodic minimal surface
    SchwarzP {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// Generalized ellipsoid (sphere-box morph)
    Superellipsoid {
        /// Semi-axis radii
        half_extents: Vec3,
        /// North-south roundness
        e1: f32,
        /// East-west roundness
        e2: f32,
    },

    /// Rounded X shape extruded along Y-axis
    RoundedX {
        /// Arm length
        width: f32,
        /// Rounding radius
        round_radius: f32,
        /// Half the extrusion height
        half_height: f32,
    },

    /// Pie (sector) shape extruded along Y-axis
    Pie {
        /// Half opening angle in radians
        angle: f32,
        /// Pie radius
        radius: f32,
        /// Half the extrusion height
        half_height: f32,
    },

    /// Trapezoid prism in XY plane, extruded along Z
    Trapezoid {
        /// Half-width at bottom
        r1: f32,
        /// Half-width at top
        r2: f32,
        /// Half-height of the 2D trapezoid
        trap_height: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Parallelogram prism in XY plane, extruded along Z
    Parallelogram {
        /// Half-width
        width: f32,
        /// Half-height
        para_height: f32,
        /// Horizontal skew
        skew: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Tunnel (arch opening) in XY plane, extruded along Z
    Tunnel {
        /// Half-width of the tunnel
        width: f32,
        /// Height of the 2D tunnel opening
        height_2d: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Uneven capsule prism in XY plane, extruded along Z
    UnevenCapsule {
        /// Bottom circle radius
        r1: f32,
        /// Top circle radius
        r2: f32,
        /// Half-height between circle centers
        cap_height: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Egg shape (revolution body) around Y-axis
    Egg {
        /// Overall size / base radius
        ra: f32,
        /// Top deformation (controls pointiness)
        rb: f32,
    },

    /// Arc shape (thick ring sector) in XZ plane, extruded along Y
    ArcShape {
        /// Half opening angle in radians
        aperture: f32,
        /// Arc center radius
        radius: f32,
        /// Ring half-thickness
        thickness: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Moon (crescent) shape in XZ plane, extruded along Y
    Moon {
        /// Distance between circle centers
        d: f32,
        /// Outer circle radius
        ra: f32,
        /// Inner (subtracted) circle radius
        rb: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Cross (plus) shape in XZ plane, extruded along Y
    CrossShape {
        /// Half-length of cross arms
        length: f32,
        /// Half-thickness of cross arms
        thickness: f32,
        /// Rounding radius
        round_radius: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Blobby cross (organic) in XZ plane, extruded along Y
    BlobbyCross {
        /// Overall size
        size: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Parabola segment in XY plane, extruded along Z
    ParabolaSegment {
        /// Half-width of the parabola base
        width: f32,
        /// Height of the parabola
        para_height: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Regular N-sided polygon prism in XZ plane, extruded along Y
    RegularPolygon {
        /// Circumscribed circle radius (center to vertex)
        radius: f32,
        /// Number of sides (as f32)
        n_sides: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Star polygon prism in XZ plane, extruded along Y
    StarPolygon {
        /// Outer vertex radius
        radius: f32,
        /// Number of star points (as f32)
        n_points: f32,
        /// Inner vertex radius (spike depth)
        m: f32,
        /// Half the extrusion height along Y
        half_height: f32,
    },

    /// Staircase shape in XY plane, extruded along Z
    Stairs {
        /// Width of each step
        step_width: f32,
        /// Height of each step
        step_height: f32,
        /// Number of steps (as f32)
        n_steps: f32,
        /// Half the extrusion depth along Z
        half_depth: f32,
    },

    /// Helix (spiral tube) along Y-axis
    Helix {
        /// Major radius (distance from Y-axis to helix center)
        major_r: f32,
        /// Minor radius (tube thickness)
        minor_r: f32,
        /// Vertical distance per full revolution
        pitch: f32,
        /// Half the height along Y
        half_height: f32,
    },

    /// Regular tetrahedron centered at origin
    Tetrahedron {
        /// Distance from center to face
        size: f32,
    },

    /// Regular dodecahedron centered at origin (GDF)
    Dodecahedron {
        /// Distance from center to face
        radius: f32,
    },

    /// Regular icosahedron centered at origin (GDF)
    Icosahedron {
        /// Distance from center to face
        radius: f32,
    },

    /// Truncated octahedron centered at origin (GDF)
    TruncatedOctahedron {
        /// Distance from center to face
        radius: f32,
    },

    /// Truncated icosahedron (soccer ball) centered at origin (GDF)
    TruncatedIcosahedron {
        /// Distance from center to face
        radius: f32,
    },

    /// Box frame (wireframe box, edges only)
    BoxFrame {
        /// Half-extents along each axis
        half_extents: Vec3,
        /// Edge thickness
        edge: f32,
    },

    /// Diamond surface (Schwarz D TPMS)
    DiamondSurface {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// Neovius surface (TPMS)
    Neovius {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// Lidinoid surface (TPMS)
    Lidinoid {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// IWP surface (TPMS)
    IWP {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// FRD surface (TPMS)
    FRD {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// Fischer-Koch S surface (TPMS)
    FischerKochS {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    /// PMY surface (TPMS)
    PMY {
        /// Spatial frequency
        scale: f32,
        /// Shell half-thickness
        thickness: f32,
    },

    // === 2D Primitives (extruded to 3D) ===
    /// 2D Circle extruded along Z
    Circle2D {
        /// Circle radius
        radius: f32,
        /// Half-height of extrusion along Z
        half_height: f32,
    },
    /// 2D Rectangle extruded along Z
    Rect2D {
        /// Half-width and half-height of the rectangle
        half_extents: Vec2,
        /// Half-depth of extrusion along Z
        half_height: f32,
    },
    /// 2D Line Segment extruded along Z
    Segment2D {
        /// Start point
        a: Vec2,
        /// End point
        b: Vec2,
        /// Segment thickness (radius)
        thickness: f32,
        /// Half-depth of extrusion along Z
        half_height: f32,
    },
    /// 2D Polygon (convex, up to 8 vertices) extruded along Z
    Polygon2D {
        /// Vertices (up to 8)
        vertices: Vec<Vec2>,
        /// Half-depth of extrusion along Z
        half_height: f32,
    },
    /// 2D Rounded Rectangle extruded along Z
    RoundedRect2D {
        /// Half-extents of the rectangle
        half_extents: Vec2,
        /// Corner rounding radius
        round_radius: f32,
        /// Half-depth of extrusion along Z
        half_height: f32,
    },
    /// 2D Annular (ring) shape extruded along Z
    Annular2D {
        /// Outer radius
        outer_radius: f32,
        /// Ring thickness
        thickness: f32,
        /// Half-depth of extrusion along Z
        half_height: f32,
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

    /// Chamfer union: 45-degree beveled blend
    ChamferUnion {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Chamfer radius
        r: f32,
    },

    /// Chamfer intersection: 45-degree beveled blend
    ChamferIntersection {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Chamfer radius
        r: f32,
    },

    /// Chamfer subtraction: 45-degree beveled blend
    ChamferSubtraction {
        /// Shape to subtract from
        a: Arc<SdfNode>,
        /// Shape to subtract
        b: Arc<SdfNode>,
        /// Chamfer radius
        r: f32,
    },

    /// Stairs union: stepped/terraced blend (Mercury hg_sdf)
    StairsUnion {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Blend radius
        r: f32,
        /// Number of steps (n-1 visible steps)
        n: f32,
    },

    /// Stairs intersection: stepped/terraced blend
    StairsIntersection {
        /// First operand
        a: Arc<SdfNode>,
        /// Second operand
        b: Arc<SdfNode>,
        /// Blend radius
        r: f32,
        /// Number of steps
        n: f32,
    },

    /// Stairs subtraction: stepped/terraced blend
    StairsSubtraction {
        /// Shape to subtract from
        a: Arc<SdfNode>,
        /// Shape to subtract
        b: Arc<SdfNode>,
        /// Blend radius
        r: f32,
        /// Number of steps
        n: f32,
    },

    /// XOR (symmetric difference)
    XOR {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
    },

    /// Morph (linear interpolation between two shapes)
    Morph {
        /// Source shape
        a: Arc<SdfNode>,
        /// Target shape
        b: Arc<SdfNode>,
        /// Blend factor (0=a, 1=b)
        t: f32,
    },

    /// Columns union: column-shaped blend
    ColumnsUnion {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Columns intersection: column-shaped blend
    ColumnsIntersection {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Columns subtraction: column-shaped blend
    ColumnsSubtraction {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Pipe: cylindrical surface at intersection
    Pipe {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Pipe radius
        r: f32,
    },

    /// Engrave: engrave shape b into shape a
    Engrave {
        /// Base shape (receives the engraving)
        a: Arc<SdfNode>,
        /// Engraving shape
        b: Arc<SdfNode>,
        /// Engrave depth
        r: f32,
    },

    /// Groove: cut a groove of shape b into shape a
    Groove {
        /// Base shape (receives the groove)
        a: Arc<SdfNode>,
        /// Groove profile shape
        b: Arc<SdfNode>,
        /// Groove width
        ra: f32,
        /// Groove depth
        rb: f32,
    },

    /// Tongue: add a tongue protrusion
    Tongue {
        /// Base shape
        a: Arc<SdfNode>,
        /// Tongue profile shape
        b: Arc<SdfNode>,
        /// Tongue width
        ra: f32,
        /// Tongue height
        rb: f32,
    },

    /// Exponential smooth union (IQ): exp-weighted smooth min
    ExpSmoothUnion {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Smoothness parameter
        k: f32,
    },
    /// Exponential smooth intersection
    ExpSmoothIntersection {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Smoothness parameter
        k: f32,
    },
    /// Exponential smooth subtraction
    ExpSmoothSubtraction {
        /// Left operand
        a: Arc<SdfNode>,
        /// Right operand
        b: Arc<SdfNode>,
        /// Smoothness parameter
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

    /// Projective (perspective) transform with Lipschitz correction
    ProjectiveTransform {
        /// Child node
        child: Arc<SdfNode>,
        /// Inverse projection matrix (column-major)
        inv_matrix: [f32; 16],
        /// Lipschitz bound for distance correction
        lipschitz_bound: f32,
    },

    /// Free-Form Deformation via control point lattice
    LatticeDeform {
        /// Child node
        child: Arc<SdfNode>,
        /// Control points array
        control_points: Vec<Vec3>,
        /// Number of control points along X
        nx: u32,
        /// Number of control points along Y
        ny: u32,
        /// Number of control points along Z
        nz: u32,
        /// Lattice bounding box minimum
        bbox_min: Vec3,
        /// Lattice bounding box maximum
        bbox_max: Vec3,
    },

    /// SDF Skinning — bone-weight based spatial blending
    SdfSkinning {
        /// Child node
        child: Arc<SdfNode>,
        /// Bone transforms with weights
        bones: Vec<crate::transforms::skinning::BoneTransform>,
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

    /// Sweep along a quadratic Bezier curve in the XZ plane.
    /// The child SDF is evaluated at (perpendicular_distance, y, 0).
    /// Creates tubes, channels, or any cross-section shape along a curved path.
    SweepBezier {
        /// Child node (2D cross-section, evaluated in XY)
        child: Arc<SdfNode>,
        /// Bezier start point (XZ plane)
        p0: Vec2,
        /// Bezier control point (XZ plane)
        p1: Vec2,
        /// Bezier end point (XZ plane)
        p2: Vec2,
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

    /// Octant mirror: maps point to first octant with x >= y >= z (48-fold symmetry)
    OctantMirror {
        /// Child node
        child: Arc<SdfNode>,
    },

    /// Shear deformation (modifies evaluation point)
    Shear {
        /// Child node
        child: Arc<SdfNode>,
        /// Shear factors: (xy, xz, yz)
        shear: Vec3,
    },

    /// Animated modifier: applies time-based transformation
    Animated {
        /// Child node
        child: Arc<SdfNode>,
        /// Animation speed multiplier
        speed: f32,
        /// Animation amplitude
        amplitude: f32,
    },

    /// Assign a material ID to a subtree (transparent for distance evaluation)
    WithMaterial {
        /// Child node
        child: Arc<SdfNode>,
        /// Material ID (indexes into MaterialLibrary)
        material_id: u32,
    },

    /// Icosahedral symmetry (120-fold) — maps to fundamental domain
    IcosahedralSymmetry {
        /// Child node
        child: Arc<SdfNode>,
    },

    /// Iterated Function System — fractal self-similar folding
    IFS {
        /// Child node
        child: Arc<SdfNode>,
        /// Affine transforms (column-major Mat4)
        transforms: Vec<[f32; 16]>,
        /// Number of iterations
        iterations: u32,
    },

    /// Heightmap displacement — image-based surface perturbation
    HeightmapDisplacement {
        /// Child node
        child: Arc<SdfNode>,
        /// Heightmap data (row-major, grayscale)
        heightmap: Vec<f32>,
        /// Heightmap width
        width: u32,
        /// Heightmap height
        height: u32,
        /// Displacement amplitude
        amplitude: f32,
        /// UV scale
        scale: f32,
    },

    /// Surface roughness — FBM micro-detail noise
    SurfaceRoughness {
        /// Child node
        child: Arc<SdfNode>,
        /// Noise frequency
        frequency: f32,
        /// Noise amplitude
        amplitude: f32,
        /// Number of octaves
        octaves: u32,
    },
}

impl SdfNode {
    /// Returns the category of this node variant
    pub fn category(&self) -> SdfCategory {
        match self {
            // === Primitives ===
            Self::Sphere { .. }
            | Self::Box3d { .. }
            | Self::Cylinder { .. }
            | Self::Torus { .. }
            | Self::Plane { .. }
            | Self::Capsule { .. }
            | Self::Cone { .. }
            | Self::Ellipsoid { .. }
            | Self::RoundedCone { .. }
            | Self::Pyramid { .. }
            | Self::Octahedron { .. }
            | Self::HexPrism { .. }
            | Self::Link { .. }
            | Self::Triangle { .. }
            | Self::Bezier { .. }
            | Self::RoundedBox { .. }
            | Self::CappedCone { .. }
            | Self::CappedTorus { .. }
            | Self::RoundedCylinder { .. }
            | Self::TriangularPrism { .. }
            | Self::CutSphere { .. }
            | Self::CutHollowSphere { .. }
            | Self::DeathStar { .. }
            | Self::SolidAngle { .. }
            | Self::Rhombus { .. }
            | Self::Horseshoe { .. }
            | Self::Vesica { .. }
            | Self::InfiniteCylinder { .. }
            | Self::InfiniteCone { .. }
            | Self::Gyroid { .. }
            | Self::Heart { .. }
            | Self::Tube { .. }
            | Self::Barrel { .. }
            | Self::Diamond { .. }
            | Self::ChamferedCube { .. }
            | Self::SchwarzP { .. }
            | Self::Superellipsoid { .. }
            | Self::RoundedX { .. }
            | Self::Pie { .. }
            | Self::Trapezoid { .. }
            | Self::Parallelogram { .. }
            | Self::Tunnel { .. }
            | Self::UnevenCapsule { .. }
            | Self::Egg { .. }
            | Self::ArcShape { .. }
            | Self::Moon { .. }
            | Self::CrossShape { .. }
            | Self::BlobbyCross { .. }
            | Self::ParabolaSegment { .. }
            | Self::RegularPolygon { .. }
            | Self::StarPolygon { .. }
            | Self::Stairs { .. }
            | Self::Helix { .. }
            | Self::Tetrahedron { .. }
            | Self::Dodecahedron { .. }
            | Self::Icosahedron { .. }
            | Self::TruncatedOctahedron { .. }
            | Self::TruncatedIcosahedron { .. }
            | Self::BoxFrame { .. }
            | Self::DiamondSurface { .. }
            | Self::Neovius { .. }
            | Self::Lidinoid { .. }
            | Self::IWP { .. }
            | Self::FRD { .. }
            | Self::FischerKochS { .. }
            | Self::PMY { .. }
            | Self::Circle2D { .. }
            | Self::Rect2D { .. }
            | Self::Segment2D { .. }
            | Self::Polygon2D { .. }
            | Self::RoundedRect2D { .. }
            | Self::Annular2D { .. } => SdfCategory::Primitive,

            // === Operations ===
            Self::Union { .. }
            | Self::Intersection { .. }
            | Self::Subtraction { .. }
            | Self::SmoothUnion { .. }
            | Self::SmoothIntersection { .. }
            | Self::SmoothSubtraction { .. }
            | Self::ChamferUnion { .. }
            | Self::ChamferIntersection { .. }
            | Self::ChamferSubtraction { .. }
            | Self::StairsUnion { .. }
            | Self::StairsIntersection { .. }
            | Self::StairsSubtraction { .. }
            | Self::XOR { .. }
            | Self::Morph { .. }
            | Self::ColumnsUnion { .. }
            | Self::ColumnsIntersection { .. }
            | Self::ColumnsSubtraction { .. }
            | Self::Pipe { .. }
            | Self::Engrave { .. }
            | Self::Groove { .. }
            | Self::Tongue { .. }
            | Self::ExpSmoothUnion { .. }
            | Self::ExpSmoothIntersection { .. }
            | Self::ExpSmoothSubtraction { .. } => SdfCategory::Operation,

            // === Transforms ===
            Self::Translate { .. }
            | Self::Rotate { .. }
            | Self::Scale { .. }
            | Self::ScaleNonUniform { .. }
            | Self::ProjectiveTransform { .. }
            | Self::LatticeDeform { .. }
            | Self::SdfSkinning { .. } => SdfCategory::Transform,

            // === Modifiers ===
            Self::Twist { .. }
            | Self::Bend { .. }
            | Self::RepeatInfinite { .. }
            | Self::RepeatFinite { .. }
            | Self::Noise { .. }
            | Self::Round { .. }
            | Self::Onion { .. }
            | Self::Elongate { .. }
            | Self::Mirror { .. }
            | Self::Revolution { .. }
            | Self::Extrude { .. }
            | Self::SweepBezier { .. }
            | Self::Taper { .. }
            | Self::Displacement { .. }
            | Self::PolarRepeat { .. }
            | Self::OctantMirror { .. }
            | Self::Shear { .. }
            | Self::Animated { .. }
            | Self::WithMaterial { .. }
            | Self::IcosahedralSymmetry { .. }
            | Self::IFS { .. }
            | Self::HeightmapDisplacement { .. }
            | Self::SurfaceRoughness { .. } => SdfCategory::Modifier,
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
            | SdfNode::Helix { .. }
            | SdfNode::Tetrahedron { .. }
            | SdfNode::Dodecahedron { .. }
            | SdfNode::Icosahedron { .. }
            | SdfNode::TruncatedOctahedron { .. }
            | SdfNode::TruncatedIcosahedron { .. }
            | SdfNode::BoxFrame { .. }
            | SdfNode::DiamondSurface { .. }
            | SdfNode::Neovius { .. }
            | SdfNode::Lidinoid { .. }
            | SdfNode::IWP { .. }
            | SdfNode::FRD { .. }
            | SdfNode::FischerKochS { .. }
            | SdfNode::PMY { .. } => 1,

            // Operations: 1 + children
            SdfNode::Union { a, b }
            | SdfNode::Intersection { a, b }
            | SdfNode::Subtraction { a, b }
            | SdfNode::SmoothUnion { a, b, .. }
            | SdfNode::SmoothIntersection { a, b, .. }
            | SdfNode::SmoothSubtraction { a, b, .. }
            | SdfNode::ChamferUnion { a, b, .. }
            | SdfNode::ChamferIntersection { a, b, .. }
            | SdfNode::ChamferSubtraction { a, b, .. }
            | SdfNode::StairsUnion { a, b, .. }
            | SdfNode::StairsIntersection { a, b, .. }
            | SdfNode::StairsSubtraction { a, b, .. }
            | SdfNode::XOR { a, b }
            | SdfNode::Morph { a, b, .. }
            | SdfNode::ColumnsUnion { a, b, .. }
            | SdfNode::ColumnsIntersection { a, b, .. }
            | SdfNode::ColumnsSubtraction { a, b, .. }
            | SdfNode::Pipe { a, b, .. }
            | SdfNode::Engrave { a, b, .. }
            | SdfNode::Groove { a, b, .. }
            | SdfNode::Tongue { a, b, .. } => 1 + a.node_count() + b.node_count(),

            // Transforms and modifiers: 1 + child
            SdfNode::Translate { child, .. }
            | SdfNode::Rotate { child, .. }
            | SdfNode::Scale { child, .. }
            | SdfNode::ScaleNonUniform { child, .. }
            | SdfNode::ProjectiveTransform { child, .. }
            | SdfNode::LatticeDeform { child, .. }
            | SdfNode::SdfSkinning { child, .. }
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
            | SdfNode::SweepBezier { child, .. }
            | SdfNode::Taper { child, .. }
            | SdfNode::Displacement { child, .. }
            | SdfNode::PolarRepeat { child, .. }
            | SdfNode::OctantMirror { child, .. }
            | SdfNode::Shear { child, .. }
            | SdfNode::Animated { child, .. }
            | SdfNode::WithMaterial { child, .. }
            | SdfNode::IcosahedralSymmetry { child, .. }
            | SdfNode::IFS { child, .. }
            | SdfNode::HeightmapDisplacement { child, .. }
            | SdfNode::SurfaceRoughness { child, .. } => 1 + child.node_count(),

            #[allow(unreachable_patterns)]
            _ => 1,
        }
    }
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
    fn test_category_primitive() {
        assert_eq!(SdfNode::sphere(1.0).category(), SdfCategory::Primitive);
        assert_eq!(
            SdfNode::box3d(1.0, 1.0, 1.0).category(),
            SdfCategory::Primitive
        );
        assert_eq!(
            SdfNode::cylinder(0.5, 1.0).category(),
            SdfCategory::Primitive
        );
        assert_eq!(SdfNode::torus(1.0, 0.3).category(), SdfCategory::Primitive);
    }

    #[test]
    fn test_category_operation() {
        let u = SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        assert_eq!(u.category(), SdfCategory::Operation);
        let s = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2);
        assert_eq!(s.category(), SdfCategory::Operation);
    }

    #[test]
    fn test_category_transform() {
        let t = SdfNode::sphere(1.0).translate(1.0, 0.0, 0.0);
        assert_eq!(t.category(), SdfCategory::Transform);
        let r = SdfNode::sphere(1.0).rotate_euler(0.0, 1.0, 0.0);
        assert_eq!(r.category(), SdfCategory::Transform);
        let s = SdfNode::sphere(1.0).scale(2.0);
        assert_eq!(s.category(), SdfCategory::Transform);
    }

    #[test]
    fn test_category_modifier() {
        let t = SdfNode::sphere(1.0).twist(0.5);
        assert_eq!(t.category(), SdfCategory::Modifier);
        let b = SdfNode::sphere(1.0).bend(0.3);
        assert_eq!(b.category(), SdfCategory::Modifier);
    }

    #[test]
    fn test_category_total() {
        assert_eq!(SdfCategory::total(), 126);
    }

    #[test]
    fn test_node_count_leaf() {
        assert_eq!(SdfNode::sphere(1.0).node_count(), 1);
        assert_eq!(SdfNode::box3d(1.0, 1.0, 1.0).node_count(), 1);
        assert_eq!(SdfNode::torus(1.0, 0.3).node_count(), 1);
    }

    #[test]
    fn test_node_count_deep_tree() {
        let tree = SdfNode::sphere(1.0)
            .union(SdfNode::box3d(1.0, 1.0, 1.0))
            .translate(1.0, 0.0, 0.0)
            .scale(2.0)
            .twist(0.5);
        assert_eq!(tree.node_count(), 6); // sphere + box + union + translate + scale + twist
    }

    #[test]
    fn test_clone_equivalence() {
        let a = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(1.0, 1.0, 1.0), 0.2);
        let b = a.clone();
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
    }

    #[test]
    fn test_sdf_tree_creation() {
        let node = SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0));
        let tree = SdfTree::new(node);
        assert!(tree.node_count() >= 3);
    }

    #[test]
    fn test_node_count_operations() {
        let xor = SdfNode::sphere(1.0).xor(SdfNode::box3d(1.0, 1.0, 1.0));
        assert_eq!(xor.node_count(), 3);
        let morph = SdfNode::sphere(1.0).morph(SdfNode::box3d(1.0, 1.0, 1.0), 0.5);
        assert_eq!(morph.node_count(), 3);
    }
}

//! Core types for ALICE-SDF
//!
//! Defines the SdfNode tree structure and related types.
//!
//! Author: Moroya Sakamoto

use glam::{Quat, Vec2, Vec3};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Category of an SDF node variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdfCategory {
    Primitive,
    Operation,
    Transform,
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
    XOR { a: Arc<SdfNode>, b: Arc<SdfNode> },

    /// Morph (linear interpolation between two shapes)
    Morph {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Blend factor (0=a, 1=b)
        t: f32,
    },

    /// Columns union: column-shaped blend
    ColumnsUnion {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Columns intersection: column-shaped blend
    ColumnsIntersection {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Columns subtraction: column-shaped blend
    ColumnsSubtraction {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Column radius
        r: f32,
        /// Number of columns
        n: f32,
    },

    /// Pipe: cylindrical surface at intersection
    Pipe {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Pipe radius
        r: f32,
    },

    /// Engrave: engrave shape b into shape a
    Engrave {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Engrave depth
        r: f32,
    },

    /// Groove: cut a groove of shape b into shape a
    Groove {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Groove width
        ra: f32,
        /// Groove depth
        rb: f32,
    },

    /// Tongue: add a tongue protrusion
    Tongue {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Tongue width
        ra: f32,
        /// Tongue height
        rb: f32,
    },

    /// Exponential smooth union (IQ): exp-weighted smooth min
    ExpSmoothUnion {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Smoothness parameter
        k: f32,
    },
    /// Exponential smooth intersection
    ExpSmoothIntersection {
        a: Arc<SdfNode>,
        b: Arc<SdfNode>,
        /// Smoothness parameter
        k: f32,
    },
    /// Exponential smooth subtraction
    ExpSmoothSubtraction {
        a: Arc<SdfNode>,
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

    /// Create a tube (hollow cylinder) along Y-axis
    #[inline]
    pub fn tube(outer_radius: f32, thickness: f32, height: f32) -> Self {
        SdfNode::Tube {
            outer_radius,
            thickness,
            half_height: height * 0.5,
        }
    }

    /// Create a barrel along Y-axis
    #[inline]
    pub fn barrel(radius: f32, height: f32, bulge: f32) -> Self {
        SdfNode::Barrel {
            radius,
            half_height: height * 0.5,
            bulge,
        }
    }

    /// Create a diamond (bipyramid)
    #[inline]
    pub fn diamond(radius: f32, height: f32) -> Self {
        SdfNode::Diamond {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a chamfered cube
    #[inline]
    pub fn chamfered_cube(width: f32, height: f32, depth: f32, chamfer: f32) -> Self {
        SdfNode::ChamferedCube {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
            chamfer,
        }
    }

    /// Create a Schwarz P surface
    #[inline]
    pub fn schwarz_p(scale: f32, thickness: f32) -> Self {
        SdfNode::SchwarzP { scale, thickness }
    }

    /// Create a superellipsoid
    #[inline]
    pub fn superellipsoid(width: f32, height: f32, depth: f32, e1: f32, e2: f32) -> Self {
        SdfNode::Superellipsoid {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
            e1,
            e2,
        }
    }

    /// Create a rounded X shape
    #[inline]
    pub fn rounded_x(width: f32, round_radius: f32, height: f32) -> Self {
        SdfNode::RoundedX {
            width,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a pie (sector) shape
    #[inline]
    pub fn pie(angle: f32, radius: f32, height: f32) -> Self {
        SdfNode::Pie {
            angle,
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a trapezoid prism
    #[inline]
    pub fn trapezoid(r1: f32, r2: f32, trap_height: f32, depth: f32) -> Self {
        SdfNode::Trapezoid {
            r1,
            r2,
            trap_height: trap_height * 0.5,
            half_depth: depth * 0.5,
        }
    }

    /// Create a parallelogram prism
    #[inline]
    pub fn parallelogram(width: f32, para_height: f32, skew: f32, depth: f32) -> Self {
        SdfNode::Parallelogram {
            width,
            para_height: para_height * 0.5,
            skew,
            half_depth: depth * 0.5,
        }
    }

    /// Create a tunnel shape
    #[inline]
    pub fn tunnel(width: f32, height_2d: f32, depth: f32) -> Self {
        SdfNode::Tunnel {
            width,
            height_2d,
            half_depth: depth * 0.5,
        }
    }

    /// Create an uneven capsule prism
    #[inline]
    pub fn uneven_capsule(r1: f32, r2: f32, cap_height: f32, depth: f32) -> Self {
        SdfNode::UnevenCapsule {
            r1,
            r2,
            cap_height: cap_height * 0.5,
            half_depth: depth * 0.5,
        }
    }

    /// Create an egg shape
    #[inline]
    pub fn egg(ra: f32, rb: f32) -> Self {
        SdfNode::Egg { ra, rb }
    }

    /// Create an arc shape (thick ring sector)
    pub fn arc_shape(aperture: f32, radius: f32, thickness: f32, height: f32) -> Self {
        SdfNode::ArcShape {
            aperture,
            radius,
            thickness,
            half_height: height * 0.5,
        }
    }

    /// Create a moon (crescent) shape
    pub fn moon(d: f32, ra: f32, rb: f32, height: f32) -> Self {
        SdfNode::Moon {
            d,
            ra,
            rb,
            half_height: height * 0.5,
        }
    }

    /// Create a cross (plus) shape
    pub fn cross_shape(length: f32, thickness: f32, round_radius: f32, height: f32) -> Self {
        SdfNode::CrossShape {
            length,
            thickness,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a blobby cross (organic) shape
    pub fn blobby_cross(size: f32, height: f32) -> Self {
        SdfNode::BlobbyCross {
            size,
            half_height: height * 0.5,
        }
    }

    /// Create a parabola segment
    pub fn parabola_segment(width: f32, para_height: f32, depth: f32) -> Self {
        SdfNode::ParabolaSegment {
            width,
            para_height,
            half_depth: depth * 0.5,
        }
    }

    /// Create a regular N-sided polygon prism
    pub fn regular_polygon(radius: f32, n_sides: u32, height: f32) -> Self {
        SdfNode::RegularPolygon {
            radius,
            n_sides: n_sides as f32,
            half_height: height * 0.5,
        }
    }

    /// Create a star polygon prism
    pub fn star_polygon(radius: f32, n_points: u32, m: f32, height: f32) -> Self {
        SdfNode::StarPolygon {
            radius,
            n_points: n_points as f32,
            m,
            half_height: height * 0.5,
        }
    }

    /// Create a staircase shape
    pub fn stairs(step_width: f32, step_height: f32, n_steps: u32, depth: f32) -> Self {
        SdfNode::Stairs {
            step_width,
            step_height,
            n_steps: n_steps as f32,
            half_depth: depth * 0.5,
        }
    }

    /// Create a helix (spiral tube)
    pub fn helix(major_r: f32, minor_r: f32, pitch: f32, height: f32) -> Self {
        SdfNode::Helix {
            major_r,
            minor_r,
            pitch,
            half_height: height * 0.5,
        }
    }

    /// Create a regular tetrahedron
    #[inline]
    pub fn tetrahedron(size: f32) -> Self {
        SdfNode::Tetrahedron { size }
    }

    /// Create a regular dodecahedron
    #[inline]
    pub fn dodecahedron(radius: f32) -> Self {
        SdfNode::Dodecahedron { radius }
    }

    /// Create a regular icosahedron
    #[inline]
    pub fn icosahedron(radius: f32) -> Self {
        SdfNode::Icosahedron { radius }
    }

    /// Create a truncated octahedron
    #[inline]
    pub fn truncated_octahedron(radius: f32) -> Self {
        SdfNode::TruncatedOctahedron { radius }
    }

    /// Create a truncated icosahedron (soccer ball)
    #[inline]
    pub fn truncated_icosahedron(radius: f32) -> Self {
        SdfNode::TruncatedIcosahedron { radius }
    }

    /// Create a box frame (wireframe box)
    #[inline]
    pub fn box_frame(half_extents: Vec3, edge: f32) -> Self {
        SdfNode::BoxFrame { half_extents, edge }
    }

    /// Create a diamond surface (TPMS)
    #[inline]
    pub fn diamond_surface(scale: f32, thickness: f32) -> Self {
        SdfNode::DiamondSurface { scale, thickness }
    }

    /// Create a neovius surface (TPMS)
    #[inline]
    pub fn neovius(scale: f32, thickness: f32) -> Self {
        SdfNode::Neovius { scale, thickness }
    }

    /// Create a lidinoid surface (TPMS)
    #[inline]
    pub fn lidinoid(scale: f32, thickness: f32) -> Self {
        SdfNode::Lidinoid { scale, thickness }
    }

    /// Create an IWP surface (TPMS)
    #[inline]
    pub fn iwp(scale: f32, thickness: f32) -> Self {
        SdfNode::IWP { scale, thickness }
    }

    /// Create an FRD surface (TPMS)
    #[inline]
    pub fn frd(scale: f32, thickness: f32) -> Self {
        SdfNode::FRD { scale, thickness }
    }

    /// Create a Fischer-Koch S surface (TPMS)
    #[inline]
    pub fn fischer_koch_s(scale: f32, thickness: f32) -> Self {
        SdfNode::FischerKochS { scale, thickness }
    }

    /// Create a PMY surface (TPMS)
    #[inline]
    pub fn pmy(scale: f32, thickness: f32) -> Self {
        SdfNode::PMY { scale, thickness }
    }

    // === 2D Primitive constructors ===

    /// Create a 2D circle extruded along Z
    #[inline]
    pub fn circle_2d(radius: f32, half_height: f32) -> Self {
        SdfNode::Circle2D {
            radius,
            half_height,
        }
    }

    /// Create a 2D rectangle extruded along Z
    #[inline]
    pub fn rect_2d(half_w: f32, half_h: f32, half_height: f32) -> Self {
        SdfNode::Rect2D {
            half_extents: Vec2::new(half_w, half_h),
            half_height,
        }
    }

    /// Create a 2D line segment extruded along Z
    #[inline]
    pub fn segment_2d(
        ax: f32,
        ay: f32,
        bx: f32,
        by: f32,
        thickness: f32,
        half_height: f32,
    ) -> Self {
        SdfNode::Segment2D {
            a: Vec2::new(ax, ay),
            b: Vec2::new(bx, by),
            thickness,
            half_height,
        }
    }

    /// Create a 2D polygon extruded along Z
    #[inline]
    pub fn polygon_2d(vertices: Vec<Vec2>, half_height: f32) -> Self {
        SdfNode::Polygon2D {
            vertices,
            half_height,
        }
    }

    /// Create a 2D rounded rectangle extruded along Z
    #[inline]
    pub fn rounded_rect_2d(half_w: f32, half_h: f32, round_radius: f32, half_height: f32) -> Self {
        SdfNode::RoundedRect2D {
            half_extents: Vec2::new(half_w, half_h),
            round_radius,
            half_height,
        }
    }

    /// Create a 2D annular (ring) shape extruded along Z
    #[inline]
    pub fn annular_2d(outer_radius: f32, thickness: f32, half_height: f32) -> Self {
        SdfNode::Annular2D {
            outer_radius,
            thickness,
            half_height,
        }
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

    /// Chamfer union with another shape
    #[inline]
    pub fn chamfer_union(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer intersection with another shape
    #[inline]
    pub fn chamfer_intersection(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Chamfer subtraction of another shape
    #[inline]
    pub fn chamfer_subtract(self, other: SdfNode, r: f32) -> Self {
        SdfNode::ChamferSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Stairs union with another shape
    #[inline]
    pub fn stairs_union(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs intersection with another shape
    #[inline]
    pub fn stairs_intersection(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Stairs subtraction of another shape
    #[inline]
    pub fn stairs_subtract(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::StairsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// XOR (symmetric difference) with another shape
    #[inline]
    pub fn xor(self, other: SdfNode) -> Self {
        SdfNode::XOR {
            a: Arc::new(self),
            b: Arc::new(other),
        }
    }

    /// Morph with another shape
    #[inline]
    pub fn morph(self, other: SdfNode, t: f32) -> Self {
        SdfNode::Morph {
            a: Arc::new(self),
            b: Arc::new(other),
            t,
        }
    }

    /// Columns union with another shape
    #[inline]
    pub fn columns_union(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns intersection with another shape
    #[inline]
    pub fn columns_intersection(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Columns subtraction of another shape
    #[inline]
    pub fn columns_subtract(self, other: SdfNode, r: f32, n: f32) -> Self {
        SdfNode::ColumnsSubtraction {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
            n,
        }
    }

    /// Pipe operation with another shape
    #[inline]
    pub fn pipe(self, other: SdfNode, r: f32) -> Self {
        SdfNode::Pipe {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Engrave another shape into this one
    #[inline]
    pub fn engrave(self, other: SdfNode, r: f32) -> Self {
        SdfNode::Engrave {
            a: Arc::new(self),
            b: Arc::new(other),
            r,
        }
    }

    /// Cut a groove of another shape into this one
    #[inline]
    pub fn groove(self, other: SdfNode, ra: f32, rb: f32) -> Self {
        SdfNode::Groove {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Add a tongue protrusion of another shape
    #[inline]
    pub fn tongue(self, other: SdfNode, ra: f32, rb: f32) -> Self {
        SdfNode::Tongue {
            a: Arc::new(self),
            b: Arc::new(other),
            ra,
            rb,
        }
    }

    /// Exponential smooth union with another shape
    #[inline]
    pub fn exp_smooth_union(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothUnion {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth intersection with another shape
    #[inline]
    pub fn exp_smooth_intersection(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothIntersection {
            a: Arc::new(self),
            b: Arc::new(other),
            k,
        }
    }

    /// Exponential smooth subtraction of another shape
    #[inline]
    pub fn exp_smooth_subtract(self, other: SdfNode, k: f32) -> Self {
        SdfNode::ExpSmoothSubtraction {
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

    /// Projective (perspective) transform
    #[inline]
    pub fn projective_transform(self, inv_matrix: [f32; 16], lipschitz_bound: f32) -> Self {
        SdfNode::ProjectiveTransform {
            child: Arc::new(self),
            inv_matrix,
            lipschitz_bound,
        }
    }

    /// Lattice deformation (Free-Form Deformation)
    #[inline]
    pub fn lattice_deform(
        self,
        control_points: Vec<Vec3>,
        nx: u32,
        ny: u32,
        nz: u32,
        bbox_min: Vec3,
        bbox_max: Vec3,
    ) -> Self {
        SdfNode::LatticeDeform {
            child: Arc::new(self),
            control_points,
            nx,
            ny,
            nz,
            bbox_min,
            bbox_max,
        }
    }

    /// SDF skinning (bone-weight based deformation)
    #[inline]
    pub fn sdf_skinning(self, bones: Vec<crate::transforms::skinning::BoneTransform>) -> Self {
        SdfNode::SdfSkinning {
            child: Arc::new(self),
            bones,
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

    /// Sweep along a quadratic Bezier curve in the XZ plane.
    /// Control points are (x, z) coordinates.
    #[inline]
    pub fn sweep_bezier(self, p0: Vec2, p1: Vec2, p2: Vec2) -> Self {
        SdfNode::SweepBezier {
            child: Arc::new(self),
            p0,
            p1,
            p2,
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

    /// Octant mirror (48-fold symmetry)
    #[inline]
    pub fn octant_mirror(self) -> Self {
        SdfNode::OctantMirror {
            child: Arc::new(self),
        }
    }

    /// Apply shear deformation
    #[inline]
    pub fn shear(self, xy: f32, xz: f32, yz: f32) -> Self {
        SdfNode::Shear {
            child: Arc::new(self),
            shear: Vec3::new(xy, xz, yz),
        }
    }

    /// Apply time-based animation
    #[inline]
    pub fn animated(self, speed: f32, amplitude: f32) -> Self {
        SdfNode::Animated {
            child: Arc::new(self),
            speed,
            amplitude,
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

    /// Icosahedral symmetry (120-fold)
    #[inline]
    pub fn icosahedral_symmetry(self) -> Self {
        SdfNode::IcosahedralSymmetry {
            child: Arc::new(self),
        }
    }

    /// Iterated Function System
    #[inline]
    pub fn ifs(self, transforms: Vec<[f32; 16]>, iterations: u32) -> Self {
        SdfNode::IFS {
            child: Arc::new(self),
            transforms,
            iterations,
        }
    }

    /// Heightmap displacement
    #[inline]
    pub fn heightmap_displacement(
        self,
        heightmap: Vec<f32>,
        width: u32,
        height: u32,
        amplitude: f32,
        scale: f32,
    ) -> Self {
        SdfNode::HeightmapDisplacement {
            child: Arc::new(self),
            heightmap,
            width,
            height,
            amplitude,
            scale,
        }
    }

    /// Surface roughness (FBM noise)
    #[inline]
    pub fn surface_roughness(self, frequency: f32, amplitude: f32, octaves: u32) -> Self {
        SdfNode::SurfaceRoughness {
            child: Arc::new(self),
            frequency,
            amplitude,
            octaves,
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

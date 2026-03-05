//! Primitive and 2D constructors for SdfNode
//!
//! Author: Moroya Sakamoto

use glam::{Vec2, Vec3};

use super::SdfNode;

impl SdfNode {
    // === Primitive constructors ===

    /// Create a sphere with the given radius
    #[must_use]
    #[inline]
    pub const fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create an axis-aligned box with the given dimensions
    #[must_use]
    #[inline]
    pub fn box3d(width: f32, height: f32, depth: f32) -> Self {
        Self::Box3d {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
        }
    }

    /// Create a cylinder along Y-axis
    #[must_use]
    #[inline]
    pub fn cylinder(radius: f32, height: f32) -> Self {
        Self::Cylinder {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a torus in the XZ plane
    #[must_use]
    #[inline]
    pub const fn torus(major_radius: f32, minor_radius: f32) -> Self {
        Self::Torus {
            major_radius,
            minor_radius,
        }
    }

    /// Create an infinite plane
    #[must_use]
    #[inline]
    pub fn plane(normal: Vec3, distance: f32) -> Self {
        Self::Plane {
            normal: normal.normalize(),
            distance,
        }
    }

    /// Create a capsule between two points
    #[must_use]
    #[inline]
    pub const fn capsule(a: Vec3, b: Vec3, radius: f32) -> Self {
        Self::Capsule {
            point_a: a,
            point_b: b,
            radius,
        }
    }

    /// Create a cone along Y-axis
    #[must_use]
    #[inline]
    pub fn cone(radius: f32, height: f32) -> Self {
        Self::Cone {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create an ellipsoid with given semi-axis radii
    #[must_use]
    #[inline]
    pub const fn ellipsoid(rx: f32, ry: f32, rz: f32) -> Self {
        Self::Ellipsoid {
            radii: Vec3::new(rx, ry, rz),
        }
    }

    /// Create a rounded cone along Y-axis
    #[must_use]
    #[inline]
    pub fn rounded_cone(r1: f32, r2: f32, height: f32) -> Self {
        Self::RoundedCone {
            r1,
            r2,
            half_height: height * 0.5,
        }
    }

    /// Create a 4-sided pyramid
    #[must_use]
    #[inline]
    pub fn pyramid(height: f32) -> Self {
        Self::Pyramid {
            half_height: height * 0.5,
        }
    }

    /// Create a regular octahedron
    #[must_use]
    #[inline]
    pub const fn octahedron(size: f32) -> Self {
        Self::Octahedron { size }
    }

    /// Create a hexagonal prism
    #[must_use]
    #[inline]
    pub fn hex_prism(hex_radius: f32, height: f32) -> Self {
        Self::HexPrism {
            hex_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a chain link shape
    #[must_use]
    #[inline]
    pub fn link(length: f32, r1: f32, r2: f32) -> Self {
        Self::Link {
            half_length: length * 0.5,
            r1,
            r2,
        }
    }

    /// Create a triangle from three vertices
    #[must_use]
    #[inline]
    pub const fn triangle(a: Vec3, b: Vec3, c: Vec3) -> Self {
        Self::Triangle {
            point_a: a,
            point_b: b,
            point_c: c,
        }
    }

    /// Create a quadratic Bezier curve tube
    #[must_use]
    #[inline]
    pub const fn bezier(a: Vec3, b: Vec3, c: Vec3, radius: f32) -> Self {
        Self::Bezier {
            point_a: a,
            point_b: b,
            point_c: c,
            radius,
        }
    }

    /// Create a rounded box
    #[must_use]
    #[inline]
    pub const fn rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> Self {
        Self::RoundedBox {
            half_extents: Vec3::new(hx, hy, hz),
            round_radius,
        }
    }

    /// Create a capped cone (frustum)
    #[must_use]
    #[inline]
    pub fn capped_cone(height: f32, r1: f32, r2: f32) -> Self {
        Self::CappedCone {
            half_height: height * 0.5,
            r1,
            r2,
        }
    }

    /// Create a capped torus (arc)
    #[must_use]
    #[inline]
    pub const fn capped_torus(major_radius: f32, minor_radius: f32, cap_angle: f32) -> Self {
        Self::CappedTorus {
            major_radius,
            minor_radius,
            cap_angle,
        }
    }

    /// Create a rounded cylinder
    #[must_use]
    #[inline]
    pub fn rounded_cylinder(radius: f32, round_radius: f32, height: f32) -> Self {
        Self::RoundedCylinder {
            radius,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a triangular prism
    #[must_use]
    #[inline]
    pub fn triangular_prism(width: f32, depth: f32) -> Self {
        Self::TriangularPrism {
            width,
            half_depth: depth * 0.5,
        }
    }

    /// Create a cut sphere
    #[must_use]
    #[inline]
    pub const fn cut_sphere(radius: f32, cut_height: f32) -> Self {
        Self::CutSphere { radius, cut_height }
    }

    /// Create a cut hollow sphere
    #[must_use]
    #[inline]
    pub const fn cut_hollow_sphere(radius: f32, cut_height: f32, thickness: f32) -> Self {
        Self::CutHollowSphere {
            radius,
            cut_height,
            thickness,
        }
    }

    /// Create a Death Star shape
    #[must_use]
    #[inline]
    pub const fn death_star(ra: f32, rb: f32, d: f32) -> Self {
        Self::DeathStar { ra, rb, d }
    }

    /// Create a solid angle
    #[must_use]
    #[inline]
    pub const fn solid_angle(angle: f32, radius: f32) -> Self {
        Self::SolidAngle { angle, radius }
    }

    /// Create a rhombus
    #[must_use]
    #[inline]
    pub fn rhombus(la: f32, lb: f32, height: f32, round_radius: f32) -> Self {
        Self::Rhombus {
            la,
            lb,
            half_height: height * 0.5,
            round_radius,
        }
    }

    /// Create a horseshoe shape
    #[must_use]
    #[inline]
    pub fn horseshoe(angle: f32, radius: f32, length: f32, width: f32, thickness: f32) -> Self {
        Self::Horseshoe {
            angle,
            radius,
            half_length: length * 0.5,
            width,
            thickness,
        }
    }

    /// Create a 3D vesica
    #[must_use]
    #[inline]
    pub fn vesica(radius: f32, dist: f32) -> Self {
        Self::Vesica {
            radius,
            half_dist: dist * 0.5,
        }
    }

    /// Create an infinite cylinder
    #[must_use]
    #[inline]
    pub const fn infinite_cylinder(radius: f32) -> Self {
        Self::InfiniteCylinder { radius }
    }

    /// Create an infinite cone
    #[must_use]
    #[inline]
    pub const fn infinite_cone(angle: f32) -> Self {
        Self::InfiniteCone { angle }
    }

    /// Create a gyroid surface
    #[must_use]
    #[inline]
    pub const fn gyroid(scale: f32, thickness: f32) -> Self {
        Self::Gyroid { scale, thickness }
    }

    /// Create a 3D heart
    #[must_use]
    #[inline]
    pub const fn heart(size: f32) -> Self {
        Self::Heart { size }
    }

    /// Create a tube (hollow cylinder) along Y-axis
    #[must_use]
    #[inline]
    pub fn tube(outer_radius: f32, thickness: f32, height: f32) -> Self {
        Self::Tube {
            outer_radius,
            thickness,
            half_height: height * 0.5,
        }
    }

    /// Create a barrel along Y-axis
    #[must_use]
    #[inline]
    pub fn barrel(radius: f32, height: f32, bulge: f32) -> Self {
        Self::Barrel {
            radius,
            half_height: height * 0.5,
            bulge,
        }
    }

    /// Create a diamond (bipyramid)
    #[must_use]
    #[inline]
    pub fn diamond(radius: f32, height: f32) -> Self {
        Self::Diamond {
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a chamfered cube
    #[must_use]
    #[inline]
    pub fn chamfered_cube(width: f32, height: f32, depth: f32, chamfer: f32) -> Self {
        Self::ChamferedCube {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
            chamfer,
        }
    }

    /// Create a Schwarz P surface
    #[must_use]
    #[inline]
    pub const fn schwarz_p(scale: f32, thickness: f32) -> Self {
        Self::SchwarzP { scale, thickness }
    }

    /// Create a superellipsoid
    #[must_use]
    #[inline]
    pub fn superellipsoid(width: f32, height: f32, depth: f32, e1: f32, e2: f32) -> Self {
        Self::Superellipsoid {
            half_extents: Vec3::new(width * 0.5, height * 0.5, depth * 0.5),
            e1,
            e2,
        }
    }

    /// Create a rounded X shape
    #[must_use]
    #[inline]
    pub fn rounded_x(width: f32, round_radius: f32, height: f32) -> Self {
        Self::RoundedX {
            width,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a pie (sector) shape
    #[must_use]
    #[inline]
    pub fn pie(angle: f32, radius: f32, height: f32) -> Self {
        Self::Pie {
            angle,
            radius,
            half_height: height * 0.5,
        }
    }

    /// Create a trapezoid prism
    #[must_use]
    #[inline]
    pub fn trapezoid(r1: f32, r2: f32, trap_height: f32, depth: f32) -> Self {
        Self::Trapezoid {
            r1,
            r2,
            trap_height: trap_height * 0.5,
            half_depth: depth * 0.5,
        }
    }

    /// Create a parallelogram prism
    #[must_use]
    #[inline]
    pub fn parallelogram(width: f32, para_height: f32, skew: f32, depth: f32) -> Self {
        Self::Parallelogram {
            width,
            para_height: para_height * 0.5,
            skew,
            half_depth: depth * 0.5,
        }
    }

    /// Create a tunnel shape
    #[must_use]
    #[inline]
    pub fn tunnel(width: f32, height_2d: f32, depth: f32) -> Self {
        Self::Tunnel {
            width,
            height_2d,
            half_depth: depth * 0.5,
        }
    }

    /// Create an uneven capsule prism
    #[must_use]
    #[inline]
    pub fn uneven_capsule(r1: f32, r2: f32, cap_height: f32, depth: f32) -> Self {
        Self::UnevenCapsule {
            r1,
            r2,
            cap_height: cap_height * 0.5,
            half_depth: depth * 0.5,
        }
    }

    /// Create an egg shape
    #[must_use]
    #[inline]
    pub const fn egg(ra: f32, rb: f32) -> Self {
        Self::Egg { ra, rb }
    }

    /// Create an arc shape (thick ring sector)
    #[must_use]
    pub fn arc_shape(aperture: f32, radius: f32, thickness: f32, height: f32) -> Self {
        Self::ArcShape {
            aperture,
            radius,
            thickness,
            half_height: height * 0.5,
        }
    }

    /// Create a moon (crescent) shape
    #[must_use]
    pub fn moon(d: f32, ra: f32, rb: f32, height: f32) -> Self {
        Self::Moon {
            d,
            ra,
            rb,
            half_height: height * 0.5,
        }
    }

    /// Create a cross (plus) shape
    #[must_use]
    pub fn cross_shape(length: f32, thickness: f32, round_radius: f32, height: f32) -> Self {
        Self::CrossShape {
            length,
            thickness,
            round_radius,
            half_height: height * 0.5,
        }
    }

    /// Create a blobby cross (organic) shape
    #[must_use]
    pub fn blobby_cross(size: f32, height: f32) -> Self {
        Self::BlobbyCross {
            size,
            half_height: height * 0.5,
        }
    }

    /// Create a parabola segment
    #[must_use]
    pub fn parabola_segment(width: f32, para_height: f32, depth: f32) -> Self {
        Self::ParabolaSegment {
            width,
            para_height,
            half_depth: depth * 0.5,
        }
    }

    /// Create a regular N-sided polygon prism
    #[must_use]
    pub fn regular_polygon(radius: f32, n_sides: u32, height: f32) -> Self {
        Self::RegularPolygon {
            radius,
            n_sides: n_sides as f32,
            half_height: height * 0.5,
        }
    }

    /// Create a star polygon prism
    #[must_use]
    pub fn star_polygon(radius: f32, n_points: u32, m: f32, height: f32) -> Self {
        Self::StarPolygon {
            radius,
            n_points: n_points as f32,
            m,
            half_height: height * 0.5,
        }
    }

    /// Create a staircase shape
    #[must_use]
    pub fn stairs(step_width: f32, step_height: f32, n_steps: u32, depth: f32) -> Self {
        Self::Stairs {
            step_width,
            step_height,
            n_steps: n_steps as f32,
            half_depth: depth * 0.5,
        }
    }

    /// Create a helix (spiral tube)
    #[must_use]
    pub fn helix(major_r: f32, minor_r: f32, pitch: f32, height: f32) -> Self {
        Self::Helix {
            major_r,
            minor_r,
            pitch,
            half_height: height * 0.5,
        }
    }

    /// Create a regular tetrahedron
    #[must_use]
    #[inline]
    pub const fn tetrahedron(size: f32) -> Self {
        Self::Tetrahedron { size }
    }

    /// Create a regular dodecahedron
    #[must_use]
    #[inline]
    pub const fn dodecahedron(radius: f32) -> Self {
        Self::Dodecahedron { radius }
    }

    /// Create a regular icosahedron
    #[must_use]
    #[inline]
    pub const fn icosahedron(radius: f32) -> Self {
        Self::Icosahedron { radius }
    }

    /// Create a truncated octahedron
    #[must_use]
    #[inline]
    pub const fn truncated_octahedron(radius: f32) -> Self {
        Self::TruncatedOctahedron { radius }
    }

    /// Create a truncated icosahedron (soccer ball)
    #[must_use]
    #[inline]
    pub const fn truncated_icosahedron(radius: f32) -> Self {
        Self::TruncatedIcosahedron { radius }
    }

    /// Create a box frame (wireframe box)
    #[must_use]
    #[inline]
    pub const fn box_frame(half_extents: Vec3, edge: f32) -> Self {
        Self::BoxFrame { half_extents, edge }
    }

    /// Create a diamond surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn diamond_surface(scale: f32, thickness: f32) -> Self {
        Self::DiamondSurface { scale, thickness }
    }

    /// Create a neovius surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn neovius(scale: f32, thickness: f32) -> Self {
        Self::Neovius { scale, thickness }
    }

    /// Create a lidinoid surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn lidinoid(scale: f32, thickness: f32) -> Self {
        Self::Lidinoid { scale, thickness }
    }

    /// Create an IWP surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn iwp(scale: f32, thickness: f32) -> Self {
        Self::IWP { scale, thickness }
    }

    /// Create an FRD surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn frd(scale: f32, thickness: f32) -> Self {
        Self::FRD { scale, thickness }
    }

    /// Create a Fischer-Koch S surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn fischer_koch_s(scale: f32, thickness: f32) -> Self {
        Self::FischerKochS { scale, thickness }
    }

    /// Create a PMY surface (TPMS)
    #[must_use]
    #[inline]
    pub const fn pmy(scale: f32, thickness: f32) -> Self {
        Self::PMY { scale, thickness }
    }

    // === 2D Primitive constructors ===

    /// Create a 2D circle extruded along Z
    #[must_use]
    #[inline]
    pub const fn circle_2d(radius: f32, half_height: f32) -> Self {
        Self::Circle2D {
            radius,
            half_height,
        }
    }

    /// Create a 2D rectangle extruded along Z
    #[must_use]
    #[inline]
    pub const fn rect_2d(half_w: f32, half_h: f32, half_height: f32) -> Self {
        Self::Rect2D {
            half_extents: Vec2::new(half_w, half_h),
            half_height,
        }
    }

    /// Create a 2D line segment extruded along Z
    #[must_use]
    #[inline]
    pub const fn segment_2d(
        ax: f32,
        ay: f32,
        bx: f32,
        by: f32,
        thickness: f32,
        half_height: f32,
    ) -> Self {
        Self::Segment2D {
            a: Vec2::new(ax, ay),
            b: Vec2::new(bx, by),
            thickness,
            half_height,
        }
    }

    /// Create a 2D polygon extruded along Z
    #[must_use]
    #[inline]
    pub const fn polygon_2d(vertices: Vec<Vec2>, half_height: f32) -> Self {
        Self::Polygon2D {
            vertices,
            half_height,
        }
    }

    /// Create a 2D rounded rectangle extruded along Z
    #[must_use]
    #[inline]
    pub const fn rounded_rect_2d(
        half_w: f32,
        half_h: f32,
        round_radius: f32,
        half_height: f32,
    ) -> Self {
        Self::RoundedRect2D {
            half_extents: Vec2::new(half_w, half_h),
            round_radius,
            half_height,
        }
    }

    /// Create a 2D annular (ring) shape extruded along Z
    #[must_use]
    #[inline]
    pub const fn annular_2d(outer_radius: f32, thickness: f32, half_height: f32) -> Self {
        Self::Annular2D {
            outer_radius,
            thickness,
            half_height,
        }
    }
}

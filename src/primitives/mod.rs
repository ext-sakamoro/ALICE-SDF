//! Primitive SDF shapes (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Enum Dispatch**: Replaced string matching with fast Enum matching (integer comparison).
//! - **Unchecked Variants**: Added `_unchecked` variants for hot paths where
//!   parameter length is guaranteed by the caller (e.g. from compiled bytecode).
//!
//! Author: Moroya Sakamoto

mod sphere;
mod box3d;
mod cylinder;
mod torus;
mod plane;
mod capsule;
mod cone;
mod ellipsoid;
mod rounded_cone;
mod pyramid;
mod octahedron;
mod hex_prism;
mod link;
mod triangle;
mod bezier;
mod rounded_box;
mod capped_cone;
mod capped_torus;
mod rounded_cylinder;
mod triangular_prism;
mod cut_sphere;
mod cut_hollow_sphere;
mod death_star;
mod solid_angle;
mod rhombus;
mod horseshoe;
mod vesica;
mod infinite_cylinder;
mod infinite_cone;
mod gyroid;
mod heart;
mod tube;
mod barrel;
mod diamond;
mod chamfered_cube;
mod schwarz_p;
mod superellipsoid;
mod rounded_x;
mod pie;
mod trapezoid;
mod parallelogram;
mod tunnel;
mod uneven_capsule;
mod egg;
mod arc_shape;
mod moon;
mod cross_shape;
mod blobby_cross;
mod parabola_segment;
mod regular_polygon;
mod star_polygon;
mod stairs;
mod helix;

pub use sphere::{sdf_sphere, sdf_sphere_at};
pub use box3d::{sdf_box3d, sdf_box3d_at, sdf_rounded_box3d};
pub use cylinder::{sdf_cylinder, sdf_cylinder_capped, sdf_cylinder_infinite};
pub use torus::{sdf_torus, sdf_torus_capped};
pub use plane::{sdf_plane, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz, sdf_plane_from_points};
pub use capsule::{sdf_capsule, sdf_capsule_vertical, sdf_capsule_horizontal};
pub use cone::sdf_cone;
pub use ellipsoid::sdf_ellipsoid;
pub use rounded_cone::sdf_rounded_cone;
pub use pyramid::sdf_pyramid;
pub use octahedron::sdf_octahedron;
pub use hex_prism::sdf_hex_prism;
pub use link::sdf_link;
pub use triangle::sdf_triangle;
pub use bezier::sdf_bezier;
pub use rounded_box::sdf_rounded_box;
pub use capped_cone::sdf_capped_cone;
pub use capped_torus::sdf_capped_torus;
pub use rounded_cylinder::sdf_rounded_cylinder;
pub use triangular_prism::sdf_triangular_prism;
pub use cut_sphere::sdf_cut_sphere;
pub use cut_hollow_sphere::sdf_cut_hollow_sphere;
pub use death_star::sdf_death_star;
pub use solid_angle::sdf_solid_angle;
pub use rhombus::sdf_rhombus;
pub use horseshoe::sdf_horseshoe;
pub use vesica::sdf_vesica;
pub use infinite_cylinder::sdf_infinite_cylinder;
pub use infinite_cone::sdf_infinite_cone;
pub use gyroid::sdf_gyroid;
pub use heart::sdf_heart;
pub use tube::sdf_tube;
pub use barrel::sdf_barrel;
pub use diamond::sdf_diamond;
pub use chamfered_cube::sdf_chamfered_cube;
pub use schwarz_p::sdf_schwarz_p;
pub use superellipsoid::sdf_superellipsoid;
pub use rounded_x::sdf_rounded_x;
pub use pie::sdf_pie;
pub use trapezoid::sdf_trapezoid;
pub use parallelogram::sdf_parallelogram;
pub use tunnel::sdf_tunnel;
pub use uneven_capsule::sdf_uneven_capsule;
pub use egg::sdf_egg;
pub use arc_shape::sdf_arc_shape;
pub use moon::sdf_moon;
pub use cross_shape::sdf_cross_shape;
pub use blobby_cross::sdf_blobby_cross;
pub use parabola_segment::sdf_parabola_segment;
pub use regular_polygon::sdf_regular_polygon;
pub use star_polygon::sdf_star_polygon;
pub use stairs::sdf_stairs;
pub use helix::sdf_helix;

use glam::Vec3;

/// Primitive type identifier for fast dispatch
///
/// #[repr(u8)] ensures it compiles to a single byte, friendly for serialization/bytecode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrimitiveType {
    /// Sphere
    Sphere,
    /// Axis-aligned box
    Box3d,
    /// Cylinder along Y-axis
    Cylinder,
    /// Torus in XZ plane
    Torus,
    /// Infinite plane
    Plane,
    /// Capsule between two points
    Capsule,
    /// Cone along Y-axis
    Cone,
    /// Ellipsoid
    Ellipsoid,
    /// Rounded cone along Y-axis
    RoundedCone,
    /// Square-base pyramid
    Pyramid,
    /// Regular octahedron
    Octahedron,
    /// Hexagonal prism
    HexPrism,
    /// Chain link
    Link,
    /// Triangle (3 vertices)
    Triangle,
    /// Quadratic Bezier curve with radius
    Bezier,
    /// Rounded box
    RoundedBox,
    /// Capped cone (frustum)
    CappedCone,
    /// Capped torus (arc)
    CappedTorus,
    /// Rounded cylinder
    RoundedCylinder,
    /// Triangular prism
    TriangularPrism,
    /// Cut sphere
    CutSphere,
    /// Cut hollow sphere
    CutHollowSphere,
    /// Death Star
    DeathStar,
    /// Solid angle
    SolidAngle,
    /// Rhombus
    Rhombus,
    /// Horseshoe
    Horseshoe,
    /// Vesica
    Vesica,
    /// Infinite cylinder
    InfiniteCylinder,
    /// Infinite cone
    InfiniteCone,
    /// Gyroid
    Gyroid,
    /// Heart
    Heart,
    /// Tube (hollow cylinder)
    Tube,
    /// Barrel (bulged cylinder)
    Barrel,
    /// Diamond (bipyramid)
    Diamond,
    /// Chamfered cube
    ChamferedCube,
    /// Schwarz P surface
    SchwarzP,
    /// Superellipsoid
    Superellipsoid,
    /// Rounded X
    RoundedX,
    /// Pie (sector prism)
    Pie,
    /// Trapezoid prism
    Trapezoid,
    /// Parallelogram prism
    Parallelogram,
    /// Tunnel (arch opening)
    Tunnel,
    /// Uneven capsule prism
    UnevenCapsule,
    /// Egg (revolution body)
    Egg,
    /// Arc shape (thick ring sector)
    ArcShape,
    /// Moon (crescent)
    Moon,
    /// Cross shape (plus)
    CrossShape,
    /// Blobby cross (organic)
    BlobbyCross,
    /// Parabola segment
    ParabolaSegment,
    /// Regular N-sided polygon prism
    RegularPolygon,
    /// Star polygon prism
    StarPolygon,
    /// Staircase shape
    Stairs,
    /// Helix (spiral tube)
    Helix,
}

/// Evaluate a primitive SDF using fast Enum dispatch (Safe version)
///
/// # Arguments
/// * `prim` - Primitive type enum (fast integer comparison)
/// * `point` - Point to evaluate
/// * `params` - Parameters array
#[inline(always)]
pub fn eval_primitive(prim: PrimitiveType, point: Vec3, params: &[f32]) -> Option<f32> {
    match prim {
        PrimitiveType::Sphere => {
            if params.len() >= 1 {
                Some(sdf_sphere(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Box3d => {
            if params.len() >= 3 {
                Some(sdf_box3d(point, Vec3::new(params[0], params[1], params[2])))
            } else {
                None
            }
        }
        PrimitiveType::Cylinder => {
            if params.len() >= 2 {
                Some(sdf_cylinder(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Torus => {
            if params.len() >= 2 {
                Some(sdf_torus(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Plane => {
            if params.len() >= 4 {
                Some(sdf_plane(point, Vec3::new(params[0], params[1], params[2]), params[3]))
            } else {
                None
            }
        }
        PrimitiveType::Capsule => {
            if params.len() >= 7 {
                Some(sdf_capsule(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    params[6],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Cone => {
            if params.len() >= 2 {
                Some(sdf_cone(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Ellipsoid => {
            if params.len() >= 3 {
                Some(sdf_ellipsoid(point, Vec3::new(params[0], params[1], params[2])))
            } else {
                None
            }
        }
        PrimitiveType::RoundedCone => {
            if params.len() >= 3 {
                Some(sdf_rounded_cone(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Pyramid => {
            if params.len() >= 1 {
                Some(sdf_pyramid(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Octahedron => {
            if params.len() >= 1 {
                Some(sdf_octahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::HexPrism => {
            if params.len() >= 2 {
                Some(sdf_hex_prism(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Link => {
            if params.len() >= 3 {
                Some(sdf_link(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Triangle => {
            if params.len() >= 9 {
                Some(sdf_triangle(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    Vec3::new(params[6], params[7], params[8]),
                ))
            } else {
                None
            }
        }
        PrimitiveType::Bezier => {
            if params.len() >= 10 {
                Some(sdf_bezier(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    Vec3::new(params[3], params[4], params[5]),
                    Vec3::new(params[6], params[7], params[8]),
                    params[9],
                ))
            } else {
                None
            }
        }
        PrimitiveType::RoundedBox => {
            if params.len() >= 4 {
                Some(sdf_rounded_box(point, Vec3::new(params[0], params[1], params[2]), params[3]))
            } else { None }
        }
        PrimitiveType::CappedCone => {
            if params.len() >= 3 {
                Some(sdf_capped_cone(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::CappedTorus => {
            if params.len() >= 3 {
                Some(sdf_capped_torus(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::RoundedCylinder => {
            if params.len() >= 3 {
                Some(sdf_rounded_cylinder(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::TriangularPrism => {
            if params.len() >= 2 {
                Some(sdf_triangular_prism(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::CutSphere => {
            if params.len() >= 2 {
                Some(sdf_cut_sphere(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::CutHollowSphere => {
            if params.len() >= 3 {
                Some(sdf_cut_hollow_sphere(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::DeathStar => {
            if params.len() >= 3 {
                Some(sdf_death_star(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::SolidAngle => {
            if params.len() >= 2 {
                Some(sdf_solid_angle(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::Rhombus => {
            if params.len() >= 4 {
                Some(sdf_rhombus(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Horseshoe => {
            if params.len() >= 5 {
                Some(sdf_horseshoe(point, params[0], params[1], params[2], params[3], params[4]))
            } else { None }
        }
        PrimitiveType::Vesica => {
            if params.len() >= 2 {
                Some(sdf_vesica(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::InfiniteCylinder => {
            if params.len() >= 1 {
                Some(sdf_infinite_cylinder(point, params[0]))
            } else { None }
        }
        PrimitiveType::InfiniteCone => {
            if params.len() >= 1 {
                Some(sdf_infinite_cone(point, params[0]))
            } else { None }
        }
        PrimitiveType::Gyroid => {
            if params.len() >= 2 {
                Some(sdf_gyroid(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::Heart => {
            if params.len() >= 1 {
                Some(sdf_heart(point, params[0]))
            } else { None }
        }
        PrimitiveType::Tube => {
            if params.len() >= 3 {
                Some(sdf_tube(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::Barrel => {
            if params.len() >= 3 {
                Some(sdf_barrel(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::Diamond => {
            if params.len() >= 2 {
                Some(sdf_diamond(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::ChamferedCube => {
            if params.len() >= 4 {
                Some(sdf_chamfered_cube(point, Vec3::new(params[0], params[1], params[2]), params[3]))
            } else { None }
        }
        PrimitiveType::SchwarzP => {
            if params.len() >= 2 {
                Some(sdf_schwarz_p(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::Superellipsoid => {
            if params.len() >= 5 {
                Some(sdf_superellipsoid(point, Vec3::new(params[0], params[1], params[2]), params[3], params[4]))
            } else { None }
        }
        PrimitiveType::RoundedX => {
            if params.len() >= 3 {
                Some(sdf_rounded_x(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::Pie => {
            if params.len() >= 3 {
                Some(sdf_pie(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::Trapezoid => {
            if params.len() >= 4 {
                Some(sdf_trapezoid(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Parallelogram => {
            if params.len() >= 4 {
                Some(sdf_parallelogram(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Tunnel => {
            if params.len() >= 3 {
                Some(sdf_tunnel(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::UnevenCapsule => {
            if params.len() >= 4 {
                Some(sdf_uneven_capsule(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Egg => {
            if params.len() >= 2 {
                Some(sdf_egg(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::ArcShape => {
            if params.len() >= 4 {
                Some(sdf_arc_shape(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Moon => {
            if params.len() >= 4 {
                Some(sdf_moon(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::CrossShape => {
            if params.len() >= 4 {
                Some(sdf_cross_shape(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::BlobbyCross => {
            if params.len() >= 2 {
                Some(sdf_blobby_cross(point, params[0], params[1]))
            } else { None }
        }
        PrimitiveType::ParabolaSegment => {
            if params.len() >= 3 {
                Some(sdf_parabola_segment(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::RegularPolygon => {
            if params.len() >= 3 {
                Some(sdf_regular_polygon(point, params[0], params[1], params[2]))
            } else { None }
        }
        PrimitiveType::StarPolygon => {
            if params.len() >= 4 {
                Some(sdf_star_polygon(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Stairs => {
            if params.len() >= 4 {
                Some(sdf_stairs(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
        PrimitiveType::Helix => {
            if params.len() >= 4 {
                Some(sdf_helix(point, params[0], params[1], params[2], params[3]))
            } else { None }
        }
    }
}

/// Evaluate a primitive SDF using fast Enum dispatch (Unsafe/Deep Fried version)
///
/// # Deep Fried Optimization
///
/// - **No Bounds Checks**: Uses `get_unchecked` for parameter access.
/// - **No Option Wrapping**: Returns `f32` directly.
///
/// # Safety
/// Caller must ensure `params` has enough elements for the given primitive type.
/// This is intended for use in the inner loop where validation happened upstream.
#[inline(always)]
pub unsafe fn eval_primitive_unchecked(prim: PrimitiveType, point: Vec3, params: &[f32]) -> f32 {
    match prim {
        PrimitiveType::Sphere => sdf_sphere(point, *params.get_unchecked(0)),
        PrimitiveType::Box3d => sdf_box3d(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2))
        ),
        PrimitiveType::Cylinder => sdf_cylinder(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Torus => sdf_torus(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Plane => sdf_plane(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            *params.get_unchecked(3)
        ),
        PrimitiveType::Capsule => sdf_capsule(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            *params.get_unchecked(6)
        ),
        PrimitiveType::Cone => sdf_cone(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Ellipsoid => sdf_ellipsoid(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2))
        ),
        PrimitiveType::RoundedCone => sdf_rounded_cone(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1),
            *params.get_unchecked(2)
        ),
        PrimitiveType::Pyramid => sdf_pyramid(point, *params.get_unchecked(0)),
        PrimitiveType::Octahedron => sdf_octahedron(point, *params.get_unchecked(0)),
        PrimitiveType::HexPrism => sdf_hex_prism(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1)
        ),
        PrimitiveType::Link => sdf_link(
            point,
            *params.get_unchecked(0),
            *params.get_unchecked(1),
            *params.get_unchecked(2)
        ),
        PrimitiveType::Triangle => sdf_triangle(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            Vec3::new(*params.get_unchecked(6), *params.get_unchecked(7), *params.get_unchecked(8)),
        ),
        PrimitiveType::Bezier => sdf_bezier(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            Vec3::new(*params.get_unchecked(3), *params.get_unchecked(4), *params.get_unchecked(5)),
            Vec3::new(*params.get_unchecked(6), *params.get_unchecked(7), *params.get_unchecked(8)),
            *params.get_unchecked(9),
        ),
        PrimitiveType::RoundedBox => sdf_rounded_box(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            *params.get_unchecked(3),
        ),
        PrimitiveType::CappedCone => sdf_capped_cone(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::CappedTorus => sdf_capped_torus(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::RoundedCylinder => sdf_rounded_cylinder(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::TriangularPrism => sdf_triangular_prism(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::CutSphere => sdf_cut_sphere(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::CutHollowSphere => sdf_cut_hollow_sphere(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::DeathStar => sdf_death_star(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::SolidAngle => sdf_solid_angle(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::Rhombus => sdf_rhombus(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Horseshoe => sdf_horseshoe(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3), *params.get_unchecked(4),
        ),
        PrimitiveType::Vesica => sdf_vesica(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::InfiniteCylinder => sdf_infinite_cylinder(point, *params.get_unchecked(0)),
        PrimitiveType::InfiniteCone => sdf_infinite_cone(point, *params.get_unchecked(0)),
        PrimitiveType::Gyroid => sdf_gyroid(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::Heart => sdf_heart(point, *params.get_unchecked(0)),
        PrimitiveType::Tube => sdf_tube(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::Barrel => sdf_barrel(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::Diamond => sdf_diamond(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::ChamferedCube => sdf_chamfered_cube(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            *params.get_unchecked(3),
        ),
        PrimitiveType::SchwarzP => sdf_schwarz_p(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::Superellipsoid => sdf_superellipsoid(
            point,
            Vec3::new(*params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2)),
            *params.get_unchecked(3), *params.get_unchecked(4),
        ),
        PrimitiveType::RoundedX => sdf_rounded_x(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::Pie => sdf_pie(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::Trapezoid => sdf_trapezoid(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Parallelogram => sdf_parallelogram(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Tunnel => sdf_tunnel(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::UnevenCapsule => sdf_uneven_capsule(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Egg => sdf_egg(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::ArcShape => sdf_arc_shape(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Moon => sdf_moon(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::CrossShape => sdf_cross_shape(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::BlobbyCross => sdf_blobby_cross(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
        ),
        PrimitiveType::ParabolaSegment => sdf_parabola_segment(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::RegularPolygon => sdf_regular_polygon(
            point, *params.get_unchecked(0), *params.get_unchecked(1), *params.get_unchecked(2),
        ),
        PrimitiveType::StarPolygon => sdf_star_polygon(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Stairs => sdf_stairs(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
        PrimitiveType::Helix => sdf_helix(
            point, *params.get_unchecked(0), *params.get_unchecked(1),
            *params.get_unchecked(2), *params.get_unchecked(3),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_primitive_enum() {
        // Sphere at origin with radius 1
        let d = eval_primitive(PrimitiveType::Sphere, Vec3::ZERO, &[1.0]).unwrap();
        assert!((d + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_eval_primitive_unchecked() {
        let params = [1.0];
        // Safety: params length is 1, Sphere needs 1.
        let d = unsafe { eval_primitive_unchecked(PrimitiveType::Sphere, Vec3::ZERO, &params) };
        assert!((d + 1.0).abs() < 0.001);
    }
}

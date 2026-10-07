//! Primitive SDF shapes (Deep Fried Edition)
//!
//! # Deep Fried Optimizations
//! - **Enum Dispatch**: Replaced string matching with fast Enum matching (integer comparison).
//! - **Unchecked Variants**: Added `_unchecked` variants for hot paths where
//!   parameter length is guaranteed by the caller (e.g. from compiled bytecode).
//!
//! Author: Moroya Sakamoto

/// `clamp` that never panics.
///
/// `f32::clamp(min, max)` panics when `min > max` or either bound is NaN, so a primitive whose
/// parameter is a clamp bound (`radius`, `w`, ...) took the process down for a negative or NaN
/// value. This version returns `lo` for `x < lo`, `hi` for `x > hi` and `x` otherwise: identical to
/// `clamp` for every valid range, a NaN `x` stays NaN, and an inverted range is merely wrong
/// geometry instead of an abort.
#[inline(always)]
pub(crate) fn clamp_total(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

mod arc_shape;
mod barrel;
mod bezier;
mod blobby_cross;
mod box3d;
mod box_frame;
mod capped_cone;
mod capped_torus;
mod capsule;
mod chamfered_cube;
mod cone;
mod cross_shape;
mod cut_hollow_sphere;
mod cut_sphere;
mod cylinder;
mod death_star;
mod diamond;
mod diamond_surface;
mod dodecahedron;
mod egg;
mod ellipsoid;
mod fischer_koch_s;
mod frd;
mod gdf_vectors;
mod gyroid;
mod heart;
mod helix;
mod hex_prism;
mod horseshoe;
mod icosahedron;
mod infinite_cone;
mod infinite_cylinder;
mod iwp;
mod lidinoid;
mod link;
mod metric_ball;
mod moon;
mod neovius;
mod octahedron;
mod parabola_segment;
mod parallelogram;
mod pie;
mod plane;
mod pmy;
mod pyramid;
mod regular_polygon;
mod rhombus;
mod rounded_box;
mod rounded_cone;
mod rounded_cylinder;
mod rounded_x;
mod schwarz_p;
mod shapes_2d;
mod solid_angle;
mod sphere;
mod stairs;
mod star_polygon;
mod superellipsoid;
mod tetrahedron;
mod torus;
mod trapezoid;
mod triangle;
mod triangular_prism;
mod truncated_icosahedron;
mod truncated_octahedron;
mod tube;
mod tunnel;
mod uneven_capsule;
mod vesica;

pub use arc_shape::sdf_arc_shape;
pub use barrel::sdf_barrel;
pub use bezier::sdf_bezier;
pub use blobby_cross::sdf_blobby_cross;
pub use box3d::{sdf_box3d, sdf_box3d_at, sdf_box3d_r, sdf_rounded_box3d};
pub use box_frame::sdf_box_frame;
pub use capped_cone::sdf_capped_cone;
pub use capped_torus::sdf_capped_torus;
pub use capsule::{sdf_capsule, sdf_capsule_horizontal, sdf_capsule_r, sdf_capsule_vertical};
pub use chamfered_cube::sdf_chamfered_cube;
pub use cone::{sdf_cone, sdf_cone_r};
pub use cross_shape::sdf_cross_shape;
pub use cut_hollow_sphere::sdf_cut_hollow_sphere;
pub use cut_sphere::sdf_cut_sphere;
#[allow(deprecated)] // re-export kept until the next major
pub use cylinder::sdf_cylinder_infinite;
pub use cylinder::{sdf_cylinder, sdf_cylinder_capped, sdf_cylinder_r};
pub use death_star::sdf_death_star;
pub use diamond::sdf_diamond;
pub use diamond_surface::sdf_diamond_surface;
pub use dodecahedron::sdf_dodecahedron;
pub use egg::sdf_egg;
pub use ellipsoid::{sdf_ellipsoid, sdf_ellipsoid_exact, sdf_ellipsoid_r};
pub use fischer_koch_s::sdf_fischer_koch_s;
pub use frd::sdf_frd;
pub use gyroid::sdf_gyroid;
pub use heart::sdf_heart;
pub use helix::sdf_helix;
pub use hex_prism::{sdf_hex_prism, sdf_hex_prism_r};
pub use horseshoe::sdf_horseshoe;
pub use icosahedron::sdf_icosahedron;
pub use infinite_cone::sdf_infinite_cone;
pub use infinite_cylinder::sdf_infinite_cylinder;
pub use iwp::sdf_iwp;
pub use lidinoid::sdf_lidinoid;
pub use link::{sdf_link, sdf_link_r};
pub use metric_ball::sdf_metric_ball;
pub use moon::sdf_moon;
pub use neovius::sdf_neovius;
pub use octahedron::{sdf_octahedron, sdf_octahedron_r};
pub use parabola_segment::sdf_parabola_segment;
pub use parallelogram::sdf_parallelogram;
pub use pie::sdf_pie;
pub use plane::{
    sdf_plane, sdf_plane_from_points, sdf_plane_r, sdf_plane_xy, sdf_plane_xz, sdf_plane_yz,
};
pub use pmy::sdf_pmy;
pub use pyramid::{sdf_pyramid, sdf_pyramid_r};
pub use regular_polygon::sdf_regular_polygon;
pub use rhombus::sdf_rhombus;
pub use rounded_box::sdf_rounded_box;
pub use rounded_cone::{sdf_rounded_cone, sdf_rounded_cone_r};
pub use rounded_cylinder::sdf_rounded_cylinder;
pub use rounded_x::sdf_rounded_x;
pub use schwarz_p::sdf_schwarz_p;
pub use shapes_2d::{
    extrude_2d, sdf_annular_2d, sdf_circle_2d, sdf_polygon_2d, sdf_polygon_2d_flat,
    sdf_polygon_2d_xy, sdf_rect_2d, sdf_rounded_rect_2d, sdf_segment_2d,
};
pub use solid_angle::sdf_solid_angle;
pub use sphere::{sdf_sphere, sdf_sphere_at, sdf_sphere_r};
pub use stairs::sdf_stairs;
pub use star_polygon::sdf_star_polygon;
pub use superellipsoid::sdf_superellipsoid;
pub use tetrahedron::sdf_tetrahedron;
pub use torus::{sdf_torus, sdf_torus_capped, sdf_torus_r};
pub use trapezoid::sdf_trapezoid;
pub use triangle::sdf_triangle;
pub use triangular_prism::sdf_triangular_prism;
pub use truncated_icosahedron::sdf_truncated_icosahedron;
pub use truncated_octahedron::sdf_truncated_octahedron;
pub use tube::sdf_tube;
pub use tunnel::sdf_tunnel;
pub use uneven_capsule::sdf_uneven_capsule;
pub use vesica::sdf_vesica;

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
    /// Tetrahedron
    Tetrahedron,
    /// Dodecahedron
    Dodecahedron,
    /// Icosahedron
    Icosahedron,
    /// Truncated octahedron
    TruncatedOctahedron,
    /// Truncated icosahedron
    TruncatedIcosahedron,
    /// Box frame (wireframe box)
    BoxFrame,
    /// Diamond surface (TPMS)
    DiamondSurface,
    /// Neovius surface (TPMS)
    Neovius,
    /// Lidinoid surface (TPMS)
    Lidinoid,
    /// IWP surface (TPMS)
    IWP,
    /// FRD surface (TPMS)
    FRD,
    /// Fischer-Koch S surface (TPMS)
    FischerKochS,
    /// PMY surface (TPMS)
    PMY,
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
            if !params.is_empty() {
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
                Some(sdf_plane(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    params[3],
                ))
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
                Some(sdf_ellipsoid(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                ))
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
            if !params.is_empty() {
                Some(sdf_pyramid(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Octahedron => {
            if !params.is_empty() {
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
                Some(sdf_rounded_box(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::CappedCone => {
            if params.len() >= 3 {
                Some(sdf_capped_cone(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::CappedTorus => {
            if params.len() >= 3 {
                Some(sdf_capped_torus(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::RoundedCylinder => {
            if params.len() >= 3 {
                Some(sdf_rounded_cylinder(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::TriangularPrism => {
            if params.len() >= 2 {
                Some(sdf_triangular_prism(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::CutSphere => {
            if params.len() >= 2 {
                Some(sdf_cut_sphere(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::CutHollowSphere => {
            if params.len() >= 3 {
                Some(sdf_cut_hollow_sphere(
                    point, params[0], params[1], params[2],
                ))
            } else {
                None
            }
        }
        PrimitiveType::DeathStar => {
            if params.len() >= 3 {
                Some(sdf_death_star(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::SolidAngle => {
            if params.len() >= 2 {
                Some(sdf_solid_angle(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Rhombus => {
            if params.len() >= 4 {
                Some(sdf_rhombus(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Horseshoe => {
            if params.len() >= 5 {
                Some(sdf_horseshoe(
                    point, params[0], params[1], params[2], params[3], params[4],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Vesica => {
            if params.len() >= 2 {
                Some(sdf_vesica(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::InfiniteCylinder => {
            if !params.is_empty() {
                Some(sdf_infinite_cylinder(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::InfiniteCone => {
            if !params.is_empty() {
                Some(sdf_infinite_cone(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Gyroid => {
            if params.len() >= 2 {
                Some(sdf_gyroid(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Heart => {
            if !params.is_empty() {
                Some(sdf_heart(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Tube => {
            if params.len() >= 3 {
                Some(sdf_tube(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Barrel => {
            if params.len() >= 3 {
                Some(sdf_barrel(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Diamond => {
            if params.len() >= 2 {
                Some(sdf_diamond(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::ChamferedCube => {
            if params.len() >= 4 {
                Some(sdf_chamfered_cube(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::SchwarzP => {
            if params.len() >= 2 {
                Some(sdf_schwarz_p(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Superellipsoid => {
            if params.len() >= 5 {
                Some(sdf_superellipsoid(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    params[3],
                    params[4],
                ))
            } else {
                None
            }
        }
        PrimitiveType::RoundedX => {
            if params.len() >= 3 {
                Some(sdf_rounded_x(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Pie => {
            if params.len() >= 3 {
                Some(sdf_pie(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::Trapezoid => {
            if params.len() >= 4 {
                Some(sdf_trapezoid(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Parallelogram => {
            if params.len() >= 4 {
                Some(sdf_parallelogram(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Tunnel => {
            if params.len() >= 3 {
                Some(sdf_tunnel(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::UnevenCapsule => {
            if params.len() >= 4 {
                Some(sdf_uneven_capsule(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Egg => {
            if params.len() >= 2 {
                Some(sdf_egg(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::ArcShape => {
            if params.len() >= 4 {
                Some(sdf_arc_shape(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Moon => {
            if params.len() >= 4 {
                Some(sdf_moon(point, params[0], params[1], params[2], params[3]))
            } else {
                None
            }
        }
        PrimitiveType::CrossShape => {
            if params.len() >= 4 {
                Some(sdf_cross_shape(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::BlobbyCross => {
            if params.len() >= 2 {
                Some(sdf_blobby_cross(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::ParabolaSegment => {
            if params.len() >= 3 {
                Some(sdf_parabola_segment(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::RegularPolygon => {
            if params.len() >= 3 {
                Some(sdf_regular_polygon(point, params[0], params[1], params[2]))
            } else {
                None
            }
        }
        PrimitiveType::StarPolygon => {
            if params.len() >= 4 {
                Some(sdf_star_polygon(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Stairs => {
            if params.len() >= 4 {
                Some(sdf_stairs(
                    point, params[0], params[1], params[2], params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::Helix => {
            if params.len() >= 4 {
                Some(sdf_helix(point, params[0], params[1], params[2], params[3]))
            } else {
                None
            }
        }
        PrimitiveType::Tetrahedron => {
            if !params.is_empty() {
                Some(sdf_tetrahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Dodecahedron => {
            if !params.is_empty() {
                Some(sdf_dodecahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::Icosahedron => {
            if !params.is_empty() {
                Some(sdf_icosahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::TruncatedOctahedron => {
            if !params.is_empty() {
                Some(sdf_truncated_octahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::TruncatedIcosahedron => {
            if !params.is_empty() {
                Some(sdf_truncated_icosahedron(point, params[0]))
            } else {
                None
            }
        }
        PrimitiveType::BoxFrame => {
            if params.len() >= 4 {
                Some(sdf_box_frame(
                    point,
                    Vec3::new(params[0], params[1], params[2]),
                    params[3],
                ))
            } else {
                None
            }
        }
        PrimitiveType::DiamondSurface => {
            if params.len() >= 2 {
                Some(sdf_diamond_surface(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Neovius => {
            if params.len() >= 2 {
                Some(sdf_neovius(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::Lidinoid => {
            if params.len() >= 2 {
                Some(sdf_lidinoid(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::IWP => {
            if params.len() >= 2 {
                Some(sdf_iwp(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::FRD => {
            if params.len() >= 2 {
                Some(sdf_frd(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::FischerKochS => {
            if params.len() >= 2 {
                Some(sdf_fischer_koch_s(point, params[0], params[1]))
            } else {
                None
            }
        }
        PrimitiveType::PMY => {
            if params.len() >= 2 {
                Some(sdf_pmy(point, params[0], params[1]))
            } else {
                None
            }
        }
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
}

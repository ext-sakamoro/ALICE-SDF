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

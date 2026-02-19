//! FFI functions for creating primitive SDF shapes.
//!
//! Author: Moroya Sakamoto

use super::registry::register_node;
use super::types::*;
use crate::prelude::*;

// ============================================================================
// Primitives
// ============================================================================

/// Create a sphere SDF
#[no_mangle]
pub extern "C" fn alice_sdf_sphere(radius: f32) -> SdfHandle {
    let node = SdfNode::Sphere { radius };
    register_node(node)
}

/// Create a box SDF
#[no_mangle]
pub extern "C" fn alice_sdf_box(hx: f32, hy: f32, hz: f32) -> SdfHandle {
    let node = SdfNode::Box3d {
        half_extents: glam::Vec3::new(hx, hy, hz),
    };
    register_node(node)
}

/// Create a cylinder SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cylinder(radius: f32, half_height: f32) -> SdfHandle {
    let node = SdfNode::Cylinder {
        radius,
        half_height,
    };
    register_node(node)
}

/// Create a torus SDF
#[no_mangle]
pub extern "C" fn alice_sdf_torus(major_radius: f32, minor_radius: f32) -> SdfHandle {
    let node = SdfNode::Torus {
        major_radius,
        minor_radius,
    };
    register_node(node)
}

/// Create a capsule SDF
#[no_mangle]
pub extern "C" fn alice_sdf_capsule(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    radius: f32,
) -> SdfHandle {
    let node = SdfNode::Capsule {
        point_a: glam::Vec3::new(ax, ay, az),
        point_b: glam::Vec3::new(bx, by, bz),
        radius,
    };
    register_node(node)
}

/// Create a plane SDF
#[no_mangle]
pub extern "C" fn alice_sdf_plane(nx: f32, ny: f32, nz: f32, distance: f32) -> SdfHandle {
    let node = SdfNode::Plane {
        normal: glam::Vec3::new(nx, ny, nz).normalize(),
        distance,
    };
    register_node(node)
}

/// Create a cone SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cone(radius: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::Cone {
        radius,
        half_height,
    })
}

/// Create an ellipsoid SDF
#[no_mangle]
pub extern "C" fn alice_sdf_ellipsoid(rx: f32, ry: f32, rz: f32) -> SdfHandle {
    register_node(SdfNode::Ellipsoid {
        radii: glam::Vec3::new(rx, ry, rz),
    })
}

/// Create a rounded cone SDF
#[no_mangle]
pub extern "C" fn alice_sdf_rounded_cone(r1: f32, r2: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::RoundedCone {
        r1,
        r2,
        half_height,
    })
}

/// Create a pyramid SDF
#[no_mangle]
pub extern "C" fn alice_sdf_pyramid(half_height: f32) -> SdfHandle {
    register_node(SdfNode::Pyramid { half_height })
}

/// Create an octahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_octahedron(size: f32) -> SdfHandle {
    register_node(SdfNode::Octahedron { size })
}

/// Create a hexagonal prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_hex_prism(hex_radius: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::HexPrism {
        hex_radius,
        half_height,
    })
}

/// Create a chain link SDF
#[no_mangle]
pub extern "C" fn alice_sdf_link(half_length: f32, r1: f32, r2: f32) -> SdfHandle {
    register_node(SdfNode::Link {
        half_length,
        r1,
        r2,
    })
}

/// Create a triangle SDF
#[no_mangle]
pub extern "C" fn alice_sdf_triangle(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
) -> SdfHandle {
    register_node(SdfNode::Triangle {
        point_a: glam::Vec3::new(ax, ay, az),
        point_b: glam::Vec3::new(bx, by, bz),
        point_c: glam::Vec3::new(cx, cy, cz),
    })
}

/// Create a quadratic Bezier curve tube SDF
#[no_mangle]
pub extern "C" fn alice_sdf_bezier(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    radius: f32,
) -> SdfHandle {
    register_node(SdfNode::Bezier {
        point_a: glam::Vec3::new(ax, ay, az),
        point_b: glam::Vec3::new(bx, by, bz),
        point_c: glam::Vec3::new(cx, cy, cz),
        radius,
    })
}

/// Create a rounded box SDF
#[no_mangle]
pub extern "C" fn alice_sdf_rounded_box(hx: f32, hy: f32, hz: f32, round_radius: f32) -> SdfHandle {
    register_node(SdfNode::RoundedBox {
        half_extents: glam::Vec3::new(hx, hy, hz),
        round_radius,
    })
}

/// Create a capped cone (frustum) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_capped_cone(half_height: f32, r1: f32, r2: f32) -> SdfHandle {
    register_node(SdfNode::CappedCone {
        half_height,
        r1,
        r2,
    })
}

/// Create a capped torus (arc) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_capped_torus(
    major_radius: f32,
    minor_radius: f32,
    cap_angle: f32,
) -> SdfHandle {
    register_node(SdfNode::CappedTorus {
        major_radius,
        minor_radius,
        cap_angle,
    })
}

/// Create a rounded cylinder SDF
#[no_mangle]
pub extern "C" fn alice_sdf_rounded_cylinder(
    radius: f32,
    round_radius: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::RoundedCylinder {
        radius,
        round_radius,
        half_height,
    })
}

/// Create a triangular prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_triangular_prism(width: f32, half_depth: f32) -> SdfHandle {
    register_node(SdfNode::TriangularPrism { width, half_depth })
}

/// Create a cut sphere SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cut_sphere(radius: f32, cut_height: f32) -> SdfHandle {
    register_node(SdfNode::CutSphere { radius, cut_height })
}

/// Create a cut hollow sphere SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cut_hollow_sphere(
    radius: f32,
    cut_height: f32,
    thickness: f32,
) -> SdfHandle {
    register_node(SdfNode::CutHollowSphere {
        radius,
        cut_height,
        thickness,
    })
}

/// Create a Death Star SDF
#[no_mangle]
pub extern "C" fn alice_sdf_death_star(ra: f32, rb: f32, d: f32) -> SdfHandle {
    register_node(SdfNode::DeathStar { ra, rb, d })
}

/// Create a solid angle SDF
#[no_mangle]
pub extern "C" fn alice_sdf_solid_angle(angle: f32, radius: f32) -> SdfHandle {
    register_node(SdfNode::SolidAngle { angle, radius })
}

/// Create a rhombus SDF
#[no_mangle]
pub extern "C" fn alice_sdf_rhombus(
    la: f32,
    lb: f32,
    half_height: f32,
    round_radius: f32,
) -> SdfHandle {
    register_node(SdfNode::Rhombus {
        la,
        lb,
        half_height,
        round_radius,
    })
}

/// Create a horseshoe SDF
#[no_mangle]
pub extern "C" fn alice_sdf_horseshoe(
    angle: f32,
    radius: f32,
    half_length: f32,
    width: f32,
    thickness: f32,
) -> SdfHandle {
    register_node(SdfNode::Horseshoe {
        angle,
        radius,
        half_length,
        width,
        thickness,
    })
}

/// Create a vesica SDF
#[no_mangle]
pub extern "C" fn alice_sdf_vesica(radius: f32, half_dist: f32) -> SdfHandle {
    register_node(SdfNode::Vesica { radius, half_dist })
}

/// Create an infinite cylinder SDF
#[no_mangle]
pub extern "C" fn alice_sdf_infinite_cylinder(radius: f32) -> SdfHandle {
    register_node(SdfNode::InfiniteCylinder { radius })
}

/// Create an infinite cone SDF
#[no_mangle]
pub extern "C" fn alice_sdf_infinite_cone(angle: f32) -> SdfHandle {
    register_node(SdfNode::InfiniteCone { angle })
}

/// Create a gyroid surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_gyroid(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::Gyroid { scale, thickness })
}

/// Create a heart SDF
#[no_mangle]
pub extern "C" fn alice_sdf_heart(size: f32) -> SdfHandle {
    register_node(SdfNode::Heart { size })
}

/// Create a tube (hollow cylinder) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_tube(outer_radius: f32, thickness: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::Tube {
        outer_radius,
        thickness,
        half_height,
    })
}

/// Create a barrel SDF
#[no_mangle]
pub extern "C" fn alice_sdf_barrel(radius: f32, half_height: f32, bulge: f32) -> SdfHandle {
    register_node(SdfNode::Barrel {
        radius,
        half_height,
        bulge,
    })
}

/// Create a diamond (bipyramid) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_diamond(radius: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::Diamond {
        radius,
        half_height,
    })
}

/// Create a chamfered cube SDF
#[no_mangle]
pub extern "C" fn alice_sdf_chamfered_cube(hx: f32, hy: f32, hz: f32, chamfer: f32) -> SdfHandle {
    register_node(SdfNode::ChamferedCube {
        half_extents: glam::Vec3::new(hx, hy, hz),
        chamfer,
    })
}

/// Create a Schwarz P surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_schwarz_p(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::SchwarzP { scale, thickness })
}

/// Create a superellipsoid SDF
#[no_mangle]
pub extern "C" fn alice_sdf_superellipsoid(
    hx: f32,
    hy: f32,
    hz: f32,
    e1: f32,
    e2: f32,
) -> SdfHandle {
    register_node(SdfNode::Superellipsoid {
        half_extents: glam::Vec3::new(hx, hy, hz),
        e1,
        e2,
    })
}

/// Create a rounded X shape SDF
#[no_mangle]
pub extern "C" fn alice_sdf_rounded_x(
    width: f32,
    round_radius: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::RoundedX {
        width,
        round_radius,
        half_height,
    })
}

/// Create a pie (sector) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_pie(angle: f32, radius: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::Pie {
        angle,
        radius,
        half_height,
    })
}

/// Create a trapezoid prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_trapezoid(
    r1: f32,
    r2: f32,
    trap_height: f32,
    half_depth: f32,
) -> SdfHandle {
    register_node(SdfNode::Trapezoid {
        r1,
        r2,
        trap_height,
        half_depth,
    })
}

/// Create a parallelogram prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_parallelogram(
    width: f32,
    para_height: f32,
    skew: f32,
    half_depth: f32,
) -> SdfHandle {
    register_node(SdfNode::Parallelogram {
        width,
        para_height,
        skew,
        half_depth,
    })
}

/// Create a tunnel SDF
#[no_mangle]
pub extern "C" fn alice_sdf_tunnel(width: f32, height_2d: f32, half_depth: f32) -> SdfHandle {
    register_node(SdfNode::Tunnel {
        width,
        height_2d,
        half_depth,
    })
}

/// Create an uneven capsule SDF
#[no_mangle]
pub extern "C" fn alice_sdf_uneven_capsule(
    r1: f32,
    r2: f32,
    cap_height: f32,
    half_depth: f32,
) -> SdfHandle {
    register_node(SdfNode::UnevenCapsule {
        r1,
        r2,
        cap_height,
        half_depth,
    })
}

/// Create an egg SDF
#[no_mangle]
pub extern "C" fn alice_sdf_egg(ra: f32, rb: f32) -> SdfHandle {
    register_node(SdfNode::Egg { ra, rb })
}

/// Create an arc shape SDF
#[no_mangle]
pub extern "C" fn alice_sdf_arc_shape(
    aperture: f32,
    radius: f32,
    thickness: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::ArcShape {
        aperture,
        radius,
        thickness,
        half_height,
    })
}

/// Create a moon (crescent) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_moon(d: f32, ra: f32, rb: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::Moon {
        d,
        ra,
        rb,
        half_height,
    })
}

/// Create a cross shape SDF
#[no_mangle]
pub extern "C" fn alice_sdf_cross_shape(
    length: f32,
    thickness: f32,
    round_radius: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::CrossShape {
        length,
        thickness,
        round_radius,
        half_height,
    })
}

/// Create a blobby cross SDF
#[no_mangle]
pub extern "C" fn alice_sdf_blobby_cross(size: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::BlobbyCross { size, half_height })
}

/// Create a parabola segment SDF
#[no_mangle]
pub extern "C" fn alice_sdf_parabola_segment(
    width: f32,
    para_height: f32,
    half_depth: f32,
) -> SdfHandle {
    register_node(SdfNode::ParabolaSegment {
        width,
        para_height,
        half_depth,
    })
}

/// Create a regular polygon prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_regular_polygon(
    radius: f32,
    n_sides: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::RegularPolygon {
        radius,
        n_sides,
        half_height,
    })
}

/// Create a star polygon prism SDF
#[no_mangle]
pub extern "C" fn alice_sdf_star_polygon(
    radius: f32,
    n_points: f32,
    m: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::StarPolygon {
        radius,
        n_points,
        m,
        half_height,
    })
}

/// Create a stairs SDF
#[no_mangle]
pub extern "C" fn alice_sdf_stairs(
    step_width: f32,
    step_height: f32,
    n_steps: f32,
    half_depth: f32,
) -> SdfHandle {
    register_node(SdfNode::Stairs {
        step_width,
        step_height,
        n_steps,
        half_depth,
    })
}

/// Create a helix SDF
#[no_mangle]
pub extern "C" fn alice_sdf_helix(
    major_r: f32,
    minor_r: f32,
    pitch: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::Helix {
        major_r,
        minor_r,
        pitch,
        half_height,
    })
}

/// Create a tetrahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_tetrahedron(size: f32) -> SdfHandle {
    register_node(SdfNode::Tetrahedron { size })
}

/// Create a dodecahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_dodecahedron(radius: f32) -> SdfHandle {
    register_node(SdfNode::Dodecahedron { radius })
}

/// Create an icosahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_icosahedron(radius: f32) -> SdfHandle {
    register_node(SdfNode::Icosahedron { radius })
}

/// Create a truncated octahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_truncated_octahedron(radius: f32) -> SdfHandle {
    register_node(SdfNode::TruncatedOctahedron { radius })
}

/// Create a truncated icosahedron SDF
#[no_mangle]
pub extern "C" fn alice_sdf_truncated_icosahedron(radius: f32) -> SdfHandle {
    register_node(SdfNode::TruncatedIcosahedron { radius })
}

/// Create a box frame SDF
#[no_mangle]
pub extern "C" fn alice_sdf_box_frame(hx: f32, hy: f32, hz: f32, edge: f32) -> SdfHandle {
    register_node(SdfNode::BoxFrame {
        half_extents: glam::Vec3::new(hx, hy, hz),
        edge,
    })
}

/// Create a diamond surface (Schwarz D TPMS) SDF
#[no_mangle]
pub extern "C" fn alice_sdf_diamond_surface(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::DiamondSurface { scale, thickness })
}

/// Create a Neovius surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_neovius(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::Neovius { scale, thickness })
}

/// Create a Lidinoid surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_lidinoid(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::Lidinoid { scale, thickness })
}

/// Create an IWP surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_iwp(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::IWP { scale, thickness })
}

/// Create an FRD surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_frd(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::FRD { scale, thickness })
}

/// Create a Fischer-Koch S surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_fischer_koch_s(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::FischerKochS { scale, thickness })
}

/// Create a PMY surface SDF
#[no_mangle]
pub extern "C" fn alice_sdf_pmy(scale: f32, thickness: f32) -> SdfHandle {
    register_node(SdfNode::PMY { scale, thickness })
}

// ============================================================================
// 2D Primitives (extruded along Z)
// ============================================================================

/// Create a 2D circle extruded along Z
#[no_mangle]
pub extern "C" fn alice_sdf_circle_2d(radius: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::circle_2d(radius, half_height))
}

/// Create a 2D rectangle extruded along Z
#[no_mangle]
pub extern "C" fn alice_sdf_rect_2d(half_w: f32, half_h: f32, half_height: f32) -> SdfHandle {
    register_node(SdfNode::rect_2d(half_w, half_h, half_height))
}

/// Create a 2D line segment extruded along Z
#[no_mangle]
pub extern "C" fn alice_sdf_segment_2d(
    ax: f32,
    ay: f32,
    bx: f32,
    by: f32,
    thickness: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::segment_2d(ax, ay, bx, by, thickness, half_height))
}

/// Create a 2D rounded rectangle extruded along Z
#[no_mangle]
pub extern "C" fn alice_sdf_rounded_rect_2d(
    half_w: f32,
    half_h: f32,
    round_radius: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::rounded_rect_2d(
        half_w,
        half_h,
        round_radius,
        half_height,
    ))
}

/// Create a 2D annular (ring) shape extruded along Z
#[no_mangle]
pub extern "C" fn alice_sdf_annular_2d(
    outer_radius: f32,
    thickness: f32,
    half_height: f32,
) -> SdfHandle {
    register_node(SdfNode::annular_2d(outer_radius, thickness, half_height))
}

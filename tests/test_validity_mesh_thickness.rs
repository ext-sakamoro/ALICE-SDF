//! Per-triangle local thickness on a real tessellation, against closed-form walls
//!
//! `validate_for_printing` measures thickness from each mesh triangle. The existing oracles call
//! `local_thickness` with an exact surface point and an exact normal; these feed the marching-cubes
//! mesh itself, so the sample point and direction come from the tessellation, as in production.
//!
//! Each shape has a known thickness everywhere on its surface: no triangle may report less than
//! that thickness minus the meshing tolerance, and the report must count no thin triangle when the
//! requirement is below the true wall. Every case asserts it compared at least one triangle.

use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex};
use alice_sdf::validity::{validate_for_printing, PrintRequirements};
use alice_sdf::SdfNode;
use glam::Vec3;

fn mesh(node: &SdfNode, half: Vec3, resolution: usize) -> Mesh {
    let config = MarchingCubesConfig {
        resolution,
        ..MarchingCubesConfig::default()
    };
    sdf_to_mesh(node, -half, half, &config)
}

/// Meshes `node`, runs the validator with `min_wall = wall - tol` and checks every triangle
/// against the closed-form `wall`
fn check(name: &str, node: &SdfNode, half: Vec3, wall: f32, tol: f32) {
    let m = mesh(node, half, 96);
    let triangles = m.indices.len() / 3;
    assert!(
        triangles > 0,
        "{name}: the mesh has no triangle (compared nothing)"
    );
    let req = PrintRequirements {
        min_wall: wall - tol,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let r = validate_for_printing(node, &m, -half * 1.05, half * 1.05, req);
    let min = r.min_local_thickness.expect("no triangle was measured");
    assert!(
        r.thin_triangles == 0 && min >= wall - tol,
        "{name}: true wall {wall} everywhere, but {} of {triangles} triangles measured below {} (min {min})",
        r.thin_triangles,
        wall - tol
    );
}

/// A 1.7 mm disc (the 100-yen-coin template's shape): every surface point, rim included, is on a
/// 1.7 mm wall (the radius is much larger)
#[test]
fn a_disc_measures_its_thickness_on_every_triangle_including_the_rim() {
    let disc = SdfNode::cylinder(11.4, 1.7);
    check("disc 1.7", &disc, Vec3::new(12.0, 1.2, 12.0), 1.7, 0.1);
}

/// A 2 mm plate (box): the faces and the edges are on a 2 mm wall
#[test]
fn a_plate_measures_its_thickness_on_every_triangle_including_the_edges() {
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    check("plate 2.0", &plate, Vec3::new(11.0, 1.5, 11.0), 2.0, 0.1);
}

/// The spherical shell sphere(10) - sphere(8): smooth, 2 mm everywhere (the case the ray already
/// measures right; kept so a new measurement cannot regress it)
#[test]
fn a_spherical_shell_measures_its_wall_on_every_triangle() {
    let shell = SdfNode::sphere(10.0).subtract(SdfNode::sphere(8.0));
    check("shell 2.0", &shell, Vec3::splat(10.6), 2.0, 0.1);
}

/// A thin wall must still be reported: a 0.5 mm plate under a 0.8 mm requirement has thin triangles
/// on its faces (the negative case, so a measurement that never reports thin cannot pass)
#[test]
fn a_plate_thinner_than_the_requirement_reports_thin_triangles() {
    let plate = SdfNode::box3d(20.0, 0.5, 20.0);
    let half = Vec3::new(11.0, 1.0, 11.0);
    let m = mesh(&plate, half, 96);
    assert!(!m.indices.is_empty(), "no triangle");
    let r = validate_for_printing(&plate, &m, -half, half, PrintRequirements::fdm_0_4_nozzle());
    let min = r.min_local_thickness.expect("no triangle was measured");
    assert!(
        r.thin_triangles > 0,
        "a 0.5 mm plate reported no thin triangle"
    );
    assert!(min < 0.8, "a 0.5 mm plate measured {min}");
}

/// A mesh of one triangle centred on `centre`, lying in the plane normal to `normal`, wound so that
/// `cross(b - a, c - a)` points along `normal` (the face normal the validator reads)
fn triangle_with_normal(centre: Vec3, normal: Vec3, size: f32) -> Mesh {
    let n = normal.normalize();
    let u = n.any_orthonormal_vector();
    let v = n.cross(u);
    // an equilateral triangle without trigonometry: corners at angles 0, 120 and 240 degrees
    const HALF_SQRT3: f32 = 0.866_025_4;
    let corners = [
        centre + u * size,
        centre + (u * -0.5 + v * HALF_SQRT3) * size,
        centre + (u * -0.5 - v * HALF_SQRT3) * size,
    ];
    let vertex = |p: Vec3| Vertex {
        position: p,
        ..Default::default()
    };
    let m = Mesh {
        vertices: corners.iter().map(|&p| vertex(p)).collect(),
        indices: vec![0, 1, 2],
    };
    let face = (m.vertices[1].position - m.vertices[0].position)
        .cross(m.vertices[2].position - m.vertices[0].position)
        .normalize();
    assert!(
        face.dot(n) > 0.999,
        "the test triangle is not wound along the given normal"
    );
    m
}

/// The sliver measured on the 100-yen-coin template's preview mesh (core's remesh + repair):
/// a 1.6e-4 mm² triangle on the rim of the 1.7 mm disc at (2.353, -0.848, 11.155) with its face
/// normal tilted the wrong way, (-0.19, 0.39, 0.90) at the bottom rim. Its inward ray left through
/// the bottom face after 0.0047 mm; the disc is 1.7 mm thick there.
#[test]
fn a_rim_sliver_with_a_tilted_face_normal_measures_the_disc_thickness() {
    let disc = SdfNode::cylinder(11.4, 1.7);
    let m = triangle_with_normal(
        Vec3::new(2.353, -0.848, 11.155),
        Vec3::new(-0.19, 0.39, 0.90),
        0.01,
    );
    let half = Vec3::new(12.0, 1.2, 12.0);
    let req = PrintRequirements {
        min_wall: 1.6,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let r = validate_for_printing(&disc, &m, -half, half, req);
    let t = r
        .min_local_thickness
        .expect("the triangle was not measured (compared nothing)");
    assert!(
        r.thin_triangles == 0 && t >= 1.6,
        "a rim sliver on a 1.7 mm disc measured {t} ({} thin)",
        r.thin_triangles
    );
}

/// A face whose normal opposes the SDF gradient by 60 degrees (cos -0.5, as measured on the mug and
/// caliper templates' preview meshes), lying 0.0024 mm inside the top face of a 2 mm plate: the
/// wall there is 2 mm, whatever direction the mesh face claims
#[test]
fn a_face_whose_normal_opposes_the_gradient_measures_the_plate_thickness() {
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    let m = triangle_with_normal(
        Vec3::new(0.0, 1.0 - 0.0024, 0.0),
        Vec3::new(0.87, -0.5, 0.0),
        0.3,
    );
    let half = Vec3::new(11.0, 1.5, 11.0);
    let req = PrintRequirements {
        min_wall: 1.9,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let r = validate_for_printing(&plate, &m, -half, half, req);
    let t = r
        .min_local_thickness
        .expect("the triangle was not measured (compared nothing)");
    assert!(
        r.thin_triangles == 0 && t >= 1.9,
        "a face on a 2 mm plate with a mis-oriented normal measured {t}"
    );
}

/// A real thin wall stays thin whatever the face normal says: the same mis-oriented face on a
/// 0.5 mm plate measures below the 0.8 mm requirement (so the new measurement cannot hide walls)
#[test]
fn a_mis_oriented_face_on_a_thin_plate_still_measures_thin() {
    let plate = SdfNode::box3d(20.0, 0.5, 20.0);
    let m = triangle_with_normal(
        Vec3::new(0.0, 0.25 - 0.0024, 0.0),
        Vec3::new(0.87, -0.5, 0.0),
        0.3,
    );
    let half = Vec3::new(11.0, 1.0, 11.0);
    let r = validate_for_printing(&plate, &m, -half, half, PrintRequirements::fdm_0_4_nozzle());
    let t = r
        .min_local_thickness
        .expect("the triangle was not measured (compared nothing)");
    assert!(
        r.thin_triangles == 1 && (t - 0.5).abs() < 0.05,
        "a 0.5 mm plate measured {t}"
    );
}

/// A sample just outside a convex edge (a tessellation corner that rounds the edge off): there the
/// SDF gradient points diagonally away from the edge, and a ray along it leaves through the other
/// face at once. The direction must be read below the surface, where the gradient is one face's
/// normal, so the wall measured is the plate's 2 mm (or its 20 mm width), never the corner
#[test]
fn a_sample_outside_a_convex_edge_measures_a_face_wall_not_the_corner() {
    let plate = SdfNode::box3d(20.0, 2.0, 20.0);
    let m = triangle_with_normal(
        Vec3::new(10.002, 1.0021, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        0.001,
    );
    let half = Vec3::new(11.0, 1.5, 11.0);
    let req = PrintRequirements {
        min_wall: 1.9,
        ..PrintRequirements::fdm_0_4_nozzle()
    };
    let r = validate_for_printing(&plate, &m, -half, half, req);
    let t = r
        .min_local_thickness
        .expect("the triangle was not measured (compared nothing)");
    assert!(
        r.thin_triangles == 0 && t >= 1.9,
        "a sample at the edge of a 2 mm plate measured {t}"
    );
}

/// The same edge sample on a 0.7 mm slab under a 0.8 mm requirement: the diagonal chord through
/// the corner is about 0.99 mm and would hide the thin wall; read below the surface, the direction
/// is the top face's normal and the wall measures 0.7 mm
#[test]
fn a_sample_outside_the_edge_of_a_thin_slab_still_measures_the_slab() {
    let slab = SdfNode::box3d(20.0, 0.7, 20.0);
    let m = triangle_with_normal(
        Vec3::new(10.0021, 0.352, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        0.001,
    );
    let half = Vec3::new(11.0, 1.0, 11.0);
    let r = validate_for_printing(&slab, &m, -half, half, PrintRequirements::fdm_0_4_nozzle());
    let t = r
        .min_local_thickness
        .expect("the triangle was not measured (compared nothing)");
    assert!(
        r.thin_triangles == 1 && (t - 0.7).abs() < 0.02,
        "the edge of a 0.7 mm slab measured {t}"
    );
}

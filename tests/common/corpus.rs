//! Every compilable `SdfNode` variant with non-trivial parameters — the
//! shared corpus for the evaluator parity, Lipschitz, transpiler and
//! shader-validation tests. A new variant that is not listed here fails
//! `corpus_covers_every_emitted_opcode`.
//!
//! Author: Moroya Sakamoto

#![allow(dead_code)]

use alice_sdf::prelude::*;
use alice_sdf::transforms::skinning::BoneTransform;
use glam::Vec2;

pub fn sphere() -> SdfNode {
    SdfNode::sphere(0.6)
}

pub fn unit_box() -> SdfNode {
    SdfNode::box3d(0.5, 0.4, 0.3)
}

pub fn identity_mat() -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

pub fn square_verts() -> Vec<Vec2> {
    vec![
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
    ]
}

/// Every compilable SdfNode variant with non-trivial parameters.
pub fn corpus() -> Vec<(&'static str, SdfNode)> {
    let a = sphere();
    let b = SdfNode::sphere(0.5).translate(0.8, 0.0, 0.0);
    vec![
        // --- primitives ---
        ("sphere", sphere()),
        ("box3d", unit_box()),
        ("cylinder", SdfNode::cylinder(0.4, 0.6)),
        ("torus", SdfNode::torus(0.7, 0.2)),
        ("plane", SdfNode::plane(Vec3::Y, 0.2)),
        (
            "capsule",
            SdfNode::capsule(Vec3::new(-0.5, 0.0, 0.0), Vec3::new(0.5, 0.2, 0.0), 0.25),
        ),
        ("cone", SdfNode::cone(0.5, 0.8)),
        ("ellipsoid", SdfNode::ellipsoid(0.6, 0.4, 0.3)),
        ("rounded_cone", SdfNode::rounded_cone(0.4, 0.2, 0.6)),
        ("pyramid", SdfNode::pyramid(0.7)),
        ("octahedron", SdfNode::octahedron(0.6)),
        ("hex_prism", SdfNode::hex_prism(0.5, 0.3)),
        ("link", SdfNode::link(0.4, 0.3, 0.1)),
        // --- extended primitives ---
        ("rounded_box", SdfNode::rounded_box(0.5, 0.4, 0.3, 0.1)),
        ("capped_cone", SdfNode::capped_cone(0.6, 0.4, 0.2)),
        ("capped_torus", SdfNode::capped_torus(0.6, 0.2, 1.0)),
        ("rounded_cylinder", SdfNode::rounded_cylinder(0.4, 0.1, 0.5)),
        ("triangular_prism", SdfNode::triangular_prism(0.5, 0.3)),
        ("cut_sphere", SdfNode::cut_sphere(0.6, 0.2)),
        (
            "cut_hollow_sphere",
            SdfNode::cut_hollow_sphere(0.6, 0.2, 0.05),
        ),
        ("death_star", SdfNode::death_star(0.6, 0.4, 0.5)),
        ("solid_angle", SdfNode::solid_angle(0.8, 0.6)),
        ("rhombus", SdfNode::rhombus(0.5, 0.3, 0.2, 0.05)),
        ("horseshoe", SdfNode::horseshoe(0.8, 0.5, 0.3, 0.1, 0.2)),
        ("vesica", SdfNode::vesica(0.6, 0.3)),
        ("infinite_cylinder", SdfNode::infinite_cylinder(0.4)),
        ("infinite_cone", SdfNode::infinite_cone(0.6)),
        ("gyroid", SdfNode::gyroid(2.0, 0.1)),
        ("heart", SdfNode::heart(0.6)),
        ("tube", SdfNode::tube(0.5, 0.1, 0.4)),
        ("barrel", SdfNode::barrel(0.5, 0.6, 0.2)),
        ("diamond", SdfNode::diamond(0.5, 0.7)),
        (
            "chamfered_cube",
            SdfNode::chamfered_cube(0.5, 0.4, 0.3, 0.1),
        ),
        ("schwarz_p", SdfNode::schwarz_p(2.0, 0.1)),
        (
            "superellipsoid",
            SdfNode::superellipsoid(0.5, 0.4, 0.3, 0.8, 1.2),
        ),
        ("rounded_x", SdfNode::rounded_x(0.6, 0.1, 0.2)),
        ("pie", SdfNode::pie(0.8, 0.6, 0.2)),
        ("trapezoid", SdfNode::trapezoid(0.5, 0.3, 0.4, 0.2)),
        ("parallelogram", SdfNode::parallelogram(0.5, 0.3, 0.2, 0.2)),
        ("tunnel", SdfNode::tunnel(0.5, 0.4, 0.3)),
        (
            "uneven_capsule",
            SdfNode::uneven_capsule(0.3, 0.2, 0.5, 0.2),
        ),
        ("egg", SdfNode::egg(0.5, 0.3)),
        ("arc_shape", SdfNode::arc_shape(0.8, 0.6, 0.1, 0.2)),
        ("moon", SdfNode::moon(0.3, 0.6, 0.5, 0.2)),
        ("cross_shape", SdfNode::cross_shape(0.6, 0.2, 0.05, 0.2)),
        ("blobby_cross", SdfNode::blobby_cross(0.6, 0.2)),
        ("parabola_segment", SdfNode::parabola_segment(0.5, 0.4, 0.2)),
        ("regular_polygon", SdfNode::regular_polygon(0.6, 6, 0.2)),
        ("star_polygon", SdfNode::star_polygon(0.6, 5, 0.3, 0.2)),
        ("stairs", SdfNode::stairs(0.3, 0.2, 4, 0.3)),
        ("helix", SdfNode::helix(0.6, 0.1, 0.5, 0.8)),
        ("tetrahedron", SdfNode::tetrahedron(0.6)),
        ("dodecahedron", SdfNode::dodecahedron(0.6)),
        ("icosahedron", SdfNode::icosahedron(0.6)),
        ("truncated_octahedron", SdfNode::truncated_octahedron(0.6)),
        ("truncated_icosahedron", SdfNode::truncated_icosahedron(0.6)),
        (
            "box_frame",
            SdfNode::box_frame(Vec3::new(0.5, 0.4, 0.3), 0.05),
        ),
        ("diamond_surface", SdfNode::diamond_surface(2.0, 0.1)),
        ("neovius", SdfNode::neovius(2.0, 0.1)),
        ("lidinoid", SdfNode::lidinoid(2.0, 0.1)),
        ("iwp", SdfNode::iwp(2.0, 0.1)),
        ("frd", SdfNode::frd(2.0, 0.1)),
        ("fischer_koch_s", SdfNode::fischer_koch_s(2.0, 0.1)),
        ("pmy", SdfNode::pmy(2.0, 0.1)),
        // --- 2D primitives (extruded) ---
        ("circle_2d", SdfNode::circle_2d(0.25, 0.5)),
        ("rect_2d", SdfNode::rect_2d(0.4, 0.3, 0.5)),
        (
            "rounded_rect_2d",
            SdfNode::RoundedRect2D {
                half_extents: Vec2::new(0.4, 0.3),
                round_radius: 0.1,
                half_height: 0.5,
            },
        ),
        (
            "segment_2d",
            SdfNode::Segment2D {
                a: Vec2::new(-0.5, -0.2),
                b: Vec2::new(0.5, 0.3),
                thickness: 0.1,
                half_height: 0.5,
            },
        ),
        ("polygon_2d", SdfNode::polygon_2d(square_verts(), 0.5)),
        ("annular_2d", SdfNode::annular_2d(0.6, 0.1, 0.5)),
        // --- binary operations ---
        ("union", a.clone().union(b.clone())),
        ("intersection", a.clone().intersection(b.clone())),
        ("subtract", a.clone().subtract(b.clone())),
        ("smooth_union", a.clone().smooth_union(b.clone(), 0.3)),
        (
            "smooth_intersection",
            a.clone().smooth_intersection(b.clone(), 0.3),
        ),
        ("smooth_subtract", a.clone().smooth_subtract(b.clone(), 0.3)),
        ("chamfer_union", a.clone().chamfer_union(b.clone(), 0.2)),
        (
            "chamfer_intersection",
            a.clone().chamfer_intersection(b.clone(), 0.2),
        ),
        (
            "chamfer_subtract",
            a.clone().chamfer_subtract(b.clone(), 0.2),
        ),
        ("stairs_union", a.clone().stairs_union(b.clone(), 0.3, 3.0)),
        (
            "stairs_intersection",
            a.clone().stairs_intersection(b.clone(), 0.3, 3.0),
        ),
        (
            "stairs_subtract",
            a.clone().stairs_subtract(b.clone(), 0.3, 3.0),
        ),
        ("xor", a.clone().xor(b.clone())),
        ("morph", a.clone().morph(b.clone(), 0.4)),
        (
            "columns_union",
            a.clone().columns_union(b.clone(), 0.3, 3.0),
        ),
        (
            "columns_intersection",
            a.clone().columns_intersection(b.clone(), 0.3, 3.0),
        ),
        (
            "columns_subtract",
            a.clone().columns_subtract(b.clone(), 0.3, 3.0),
        ),
        ("pipe", a.clone().pipe(b.clone(), 0.2)),
        ("engrave", a.clone().engrave(b.clone(), 0.1)),
        ("groove", a.clone().groove(b.clone(), 0.2, 0.1)),
        ("tongue", a.clone().tongue(b.clone(), 0.2, 0.1)),
        (
            "exp_smooth_union",
            a.clone().exp_smooth_union(b.clone(), 0.3),
        ),
        (
            "exp_smooth_intersection",
            a.clone().exp_smooth_intersection(b.clone(), 0.3),
        ),
        (
            "exp_smooth_subtract",
            a.clone().exp_smooth_subtract(b.clone(), 0.3),
        ),
        // --- transforms ---
        ("translate", unit_box().translate(0.3, -0.2, 0.1)),
        ("rotate", unit_box().rotate(Quat::from_rotation_y(0.7))),
        ("scale", sphere().scale(1.7)),
        ("scale_xyz", sphere().scale_xyz(1.5, 0.8, 1.2)),
        (
            "projective_transform",
            unit_box().projective_transform(identity_mat(), 1.0),
        ),
        (
            "lattice_deform",
            unit_box().lattice_deform(
                (0..8)
                    .map(|i| {
                        Vec3::new(
                            if i & 1 == 0 { -1.0 } else { 1.0 },
                            if i & 2 == 0 { -1.0 } else { 1.0 },
                            if i & 4 == 0 { -1.0 } else { 1.0 },
                        ) * 1.1
                    })
                    .collect(),
                2,
                2,
                2,
                Vec3::splat(-1.0),
                Vec3::splat(1.0),
            ),
        ),
        (
            "sdf_skinning",
            unit_box().sdf_skinning(vec![BoneTransform {
                inv_bind_pose: identity_mat(),
                current_pose: identity_mat(),
                weight: 1.0,
            }]),
        ),
        // --- modifiers ---
        ("twist", unit_box().twist(1.5)),
        ("bend", unit_box().bend(0.8)),
        ("repeat_infinite", sphere().repeat_infinite(2.0, 2.0, 2.0)),
        (
            "repeat_finite",
            sphere().repeat_finite([2, 1, 2], Vec3::splat(1.5)),
        ),
        // Non-linear ops *inside* a Scale: `s * f(p / s)` must be applied after
        // the blend, not to each leaf (1.11.0, found by fuzz_eval_parity:
        // Scale(ExpSmoothUnion) was 21% off on the compiled / SIMD / JIT-SIMD
        // paths). Every op with an absolute width is covered.
        (
            "scale_exp_smooth_union",
            sphere().exp_smooth_union(sphere(), 0.8625).scale(0.25),
        ),
        (
            "scale_smooth_union",
            sphere()
                .translate(0.6, 0.0, 0.0)
                .smooth_union(unit_box(), 0.4)
                .scale(0.5),
        ),
        (
            "scale_chamfer_union",
            sphere()
                .translate(0.6, 0.0, 0.0)
                .chamfer_union(unit_box(), 0.3)
                .scale(2.0),
        ),
        (
            "scale_stairs_union",
            sphere()
                .translate(0.6, 0.0, 0.0)
                .stairs_union(unit_box(), 0.3, 3.0)
                .scale(0.5),
        ),
        ("scale_round", unit_box().round(0.2).scale(0.5)),
        ("scale_onion", sphere().onion(0.1).scale(3.0)),
        (
            "scale_xyz_smooth_union",
            sphere()
                .smooth_union(unit_box(), 0.3)
                .scale_xyz(0.5, 2.0, 1.0),
        ),
        (
            "scale_nested",
            sphere()
                .smooth_union(unit_box(), 0.3)
                .scale(0.5)
                .round(0.1)
                .scale(2.0),
        ),
        // fuzz_eval_parity findings 3-7 (1.11.0), each a whole-cell / sector /
        // sign disagreement between paths on the pre-fix code:
        // 3: exp smooth with d ≫ k underflowed to ln(0) (inf vs NaN)
        (
            "exp_smooth_union_far",
            SdfNode::cone(0.05, 0.7625)
                .exp_smooth_union(SdfNode::rounded_box(1.5, 1.5, 0.05, 0.01), 0.125)
                .scale(0.25),
        ),
        // 4: odd sector count, atan2 = π lands on k + 0.5 (SIMD atan2 was polynomial)
        (
            "polar_repeat_7_offset",
            SdfNode::capsule(Vec3::new(0.0, -3.0, 0.0), Vec3::new(0.0, 3.0, 0.0), 3.0)
                .translate(-3.0, 0.0, 3.0)
                .polar_repeat(7),
        ),
        // 5: repeat_finite tie, `p / s` vs `p * (1 / s)` differ by an ulp
        (
            "repeat_finite_tie_translate",
            SdfNode::pyramid(1.5)
                .repeat_finite([3, 3, 3], Vec3::splat(3.5))
                .translate(0.0, 3.0, 0.0),
        ),
        // 6: rotation rounding (glam vs generic) at the pyramid sign discontinuity
        (
            "rotate_pyramid",
            SdfNode::pyramid(1.5).rotate(glam::Quat::from_xyzw(0.0, -0.9995736, 0.0, -0.029199546)),
        ),
        // 7: four nested polar repeats amplify an ulp across a sector boundary
        (
            "polar_repeat_nested",
            SdfNode::octahedron(0.5)
                .polar_repeat(13)
                .polar_repeat(13)
                .translate(3.0, 3.0, 3.0)
                .polar_repeat(13)
                .translate(3.0, 3.0, 3.0)
                .polar_repeat(13),
        ),
        // Pyramid base centre: `sign(max(qz, -py))` sees -0.0 there; f32::signum
        // gives -1, the SIMD / JIT / shader convention gives +1 (1.11.0, fuzz).
        ("pyramid_base", SdfNode::pyramid(1.5)),
        // Asymmetric children: a symmetric sphere hides a wrong cell choice at
        // the tie points above (same distance from every cell), an offset one
        // does not.
        (
            "repeat_infinite_offset",
            sphere()
                .translate(0.6, 0.0, 0.0)
                .repeat_infinite(2.0, 2.0, 2.0),
        ),
        (
            "repeat_finite_offset",
            sphere()
                .translate(0.6, 0.2, 0.0)
                .repeat_finite([3, 2, 3], Vec3::splat(2.0)),
        ),
        ("noise", sphere().noise(0.1, 2.0, 42)),
        ("round", unit_box().round(0.1)),
        ("onion", sphere().onion(0.1)),
        ("elongate", sphere().elongate(0.3, 0.1, 0.0)),
        (
            "mirror",
            unit_box()
                .translate(0.4, 0.0, 0.0)
                .mirror(true, false, false),
        ),
        ("revolution", SdfNode::circle_2d(0.2, 0.2).revolution(0.6)),
        ("extrude", SdfNode::circle_2d(0.4, 1.0).extrude(0.3)),
        (
            "sweep_bezier",
            sphere().sweep_bezier(
                Vec2::new(-1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(1.0, 0.0),
            ),
        ),
        ("taper", unit_box().taper(0.5)),
        ("displacement", sphere().displacement(0.1)),
        ("sine_displacement", sphere().sine_displacement(0.1, 3.0)),
        (
            "polar_repeat",
            sphere().translate(0.8, 0.0, 0.0).polar_repeat(6),
        ),
        (
            "octant_mirror",
            unit_box().translate(0.3, 0.2, 0.1).octant_mirror(),
        ),
        ("shear", unit_box().shear(1.0, 0.3, 0.0)),
        ("animated", sphere().animated(1.0, 0.2)),
        ("with_material", sphere().with_material(3)),
        (
            "icosahedral_symmetry",
            unit_box().translate(0.3, 0.0, 0.0).icosahedral_symmetry(),
        ),
        ("ifs", sphere().ifs(vec![identity_mat()], 2)),
        (
            "surface_roughness",
            sphere().surface_roughness(3.0, 0.05, 2),
        ),
        (
            "heightmap_displacement",
            sphere().heightmap_displacement(vec![0.0, 0.5, 1.0, 0.5], 2, 2, 0.1, 1.0),
        ),
        // --- nesting: transform inside op inside modifier ---
        (
            "nested",
            a.clone()
                .smooth_union(b.clone().rotate(Quat::from_rotation_z(0.3)), 0.2)
                .twist(0.5)
                .translate(0.1, 0.1, 0.1)
                .round(0.05),
        ),
    ]
}

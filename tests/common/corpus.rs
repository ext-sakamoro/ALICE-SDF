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

pub const fn sphere() -> SdfNode {
    SdfNode::sphere(0.6)
}

pub fn unit_box() -> SdfNode {
    SdfNode::box3d(0.5, 0.4, 0.3)
}

pub const fn identity_mat() -> [f32; 16] {
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
        (
            "metric_ball_cube",
            SdfNode::metric_ball(0.6, alice_det_math::metric::MetricWeights::LINF),
        ),
        (
            "metric_ball_octahedron",
            SdfNode::metric_ball(0.6, alice_det_math::metric::MetricWeights::L1),
        ),
        (
            "metric_ball_mix",
            SdfNode::metric_ball(
                0.6,
                alice_det_math::metric::MetricWeights::new(0.3, 0.5, 0.2)
                    .expect("non-negative weights are a metric"),
            ),
        ),
        (
            "metric_blend",
            SdfNode::metric_blend(sphere(), unit_box(), glam::Vec3::ZERO, 0.7, 0.4),
        ),
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
        // Large enough that the sampled boxes land *inside* it: the interval
        // arm scaled `lo` and `hi` by different factors, which inverts the
        // bounds once the child interval is wholly negative (2026-09-27).
        (
            "scale_xyz_interior",
            SdfNode::sphere(1.6).scale_xyz(0.8, 1.7, 2.4),
        ),
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
        (
            "sdf_skinning_two_bones",
            unit_box().sdf_skinning(vec![
                BoneTransform {
                    inv_bind_pose: glam::Mat4::from_translation(glam::Vec3::new(0.2, -0.1, 0.0))
                        .to_cols_array(),
                    current_pose: glam::Mat4::from_rotation_z(0.6).to_cols_array(),
                    weight: 0.3,
                },
                BoneTransform {
                    inv_bind_pose: glam::Mat4::from_scale(glam::Vec3::new(1.2, 0.9, 1.0))
                        .to_cols_array(),
                    current_pose: glam::Mat4::from_rotation_x(-0.4).to_cols_array(),
                    weight: 0.7,
                },
            ]),
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
        // Wedges of a half turn and a full turn: the interval arm folded the
        // sector with `sin(ha)`, which collapses the z extent at `ha ≥ π/2`
        // and made the enclosure too tight for `count ≤ 2` (2026-09-27).
        (
            "polar_repeat_2",
            sphere().translate(0.8, 0.0, 0.0).polar_repeat(2),
        ),
        (
            "polar_repeat_1",
            sphere().translate(0.8, 0.0, 0.0).polar_repeat(1),
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
        // non-trivial transforms (column-major): the identity entries above
        // would pass a shader that ignores the matrices
        (
            "ifs_scale_rotate",
            sphere().ifs(
                vec![
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::splat(0.5),
                        glam::Quat::IDENTITY,
                        glam::Vec3::new(0.8, 0.0, 0.0),
                    )
                    .to_cols_array(),
                    glam::Mat4::from_scale_rotation_translation(
                        glam::Vec3::new(0.7, 0.5, 0.6),
                        glam::Quat::from_rotation_y(0.9),
                        glam::Vec3::new(-0.3, 0.4, 0.2),
                    )
                    .to_cols_array(),
                ],
                3,
            ),
        ),
        (
            "surface_roughness",
            sphere().surface_roughness(3.0, 0.05, 2),
        ),
        (
            "terrain",
            SdfNode::Terrain {
                scale: 1.3,
                amplitude: 0.4,
            },
        ),
        (
            "heightmap_displacement",
            sphere().heightmap_displacement(vec![0.0, 0.5, 1.0, 0.5], 2, 2, 0.1, 1.0),
        ),
        // --- nesting: transform inside op inside modifier ---
        (
            "nested",
            a.smooth_union(b.rotate(Quat::from_rotation_z(0.3)), 0.2)
                .twist(0.5)
                .translate(0.1, 0.1, 0.1)
                .round(0.05),
        ),
    ]
}

// ============================================================================
// Projective transform away from the identity (shader oracle)
// ============================================================================

/// Inverse matrix (column-major, as `SdfNode::ProjectiveTransform::inv_matrix`)
/// of a projective transform far from the identity. The corpus entry
/// `projective_transform` is the identity, which a transpiler that forwards the
/// child unchanged also renders correctly; this case is the one that tells them
/// apart.
///
/// Every non-zero entry is a decimal with no exact binary representation. The
/// w row (entries 3, 7, 11, 15) gives `w = 0.07 x - 0.04 y + 0.06 z + 1.3`,
/// which runs from 0.79 to 1.81 over the ±3 box the parity tests sample, so the
/// divide changes every point and `|1 / w|` (0.55 to 1.27) falls on both sides
/// of [`PROJECTIVE_BOUND`].
pub const PROJECTIVE_INV: [f32; 16] = [
    0.9, 0.1, -0.2, 0.07, // column 0
    0.15, 1.1, 0.05, -0.04, // column 1
    -0.1, 0.2, 0.8, 0.06, // column 2
    0.3, -0.1, 0.2, 1.3, // column 3
];

/// Lipschitz bound of [`projective_nonidentity`]: `min(|1 / w|, bound)` picks
/// `|1 / w|` where `w > 1 / 0.77` (about 1.3, half of the sampled box) and
/// the bound elsewhere.
pub const PROJECTIVE_BOUND: f32 = 0.77;

/// `unit_box()` under [`PROJECTIVE_INV`] with [`PROJECTIVE_BOUND`].
pub fn projective_nonidentity() -> SdfNode {
    unit_box().projective_transform(PROJECTIVE_INV, PROJECTIVE_BOUND)
}

/// One way of getting the projective law wrong, for [`projective_law`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectiveMutation {
    /// The law as `src/` defines it.
    None,
    /// Use the transposed matrix.
    Transpose,
    /// Skip the divide by `w`.
    NoDivide,
    /// Skip the `min(|1 / w|, bound)` distance scale.
    NoScale,
    /// Scale the distance by the bound alone.
    ConstantBound,
    /// Ignore the transform (evaluate the child at `p`).
    PassThrough,
}

/// Distance of [`projective_nonidentity`] at `p`, the law written out
/// independently of `src/` in the same operation order (every `a * b + c`
/// rounded twice, the row times `1 / w`, the child distance times
/// `min(|1 / w|, bound)`), with `mutation` applied.
pub fn projective_law(p: Vec3, mutation: ProjectiveMutation) -> f32 {
    let child = unit_box();
    if mutation == ProjectiveMutation::PassThrough {
        return eval(&child, p);
    }
    let src = PROJECTIVE_INV;
    let m: [f32; 16] = if mutation == ProjectiveMutation::Transpose {
        std::array::from_fn(|i| src[(i % 4) * 4 + i / 4])
    } else {
        src
    };
    let row = |r: usize| m[r] * p.x + m[4 + r] * p.y + m[8 + r] * p.z + m[12 + r];
    let w = row(3);
    let inv_w = 1.0 / w;
    let q = if mutation == ProjectiveMutation::NoDivide {
        Vec3::new(row(0), row(1), row(2))
    } else {
        Vec3::new(row(0) * inv_w, row(1) * inv_w, row(2) * inv_w)
    };
    let d = eval(&child, q);
    match mutation {
        ProjectiveMutation::NoScale => d,
        ProjectiveMutation::ConstantBound => d * PROJECTIVE_BOUND,
        _ => d * inv_w.abs().min(PROJECTIVE_BOUND),
    }
}

/// Checks, on the CPU, that [`projective_nonidentity`] over `pts` can tell
/// every [`ProjectiveMutation`] from the right one at relative
/// tolerance `rel_tol` (the bar the calling shader oracle applies), and that
/// the unmutated law reproduces the tree evaluator bit for bit. A shader oracle
/// that calls this first cannot pass vacuously because its case happens to be
/// insensitive to the error it is meant to catch.
pub fn assert_projective_case_discriminates(pts: &[Vec3], rel_tol: f32) {
    let node = projective_nonidentity();
    for p in pts {
        assert_eq!(
            projective_law(*p, ProjectiveMutation::None).to_bits(),
            eval(&node, *p).to_bits(),
            "the reference law and the tree evaluator disagree at {p:?}"
        );
    }
    // both branches of min(|1 / w|, bound)
    let m = PROJECTIVE_INV;
    let above = pts
        .iter()
        .filter(|p| {
            let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
            (1.0 / w).abs() > PROJECTIVE_BOUND
        })
        .count();
    let quarter = pts.len() / 4;
    assert!(
        above >= quarter && pts.len() - above >= quarter,
        "|1 / w| exceeds the bound at {above} of {} points: both branches need a quarter",
        pts.len()
    );
    for mutation in [
        ProjectiveMutation::Transpose,
        ProjectiveMutation::NoDivide,
        ProjectiveMutation::NoScale,
        ProjectiveMutation::ConstantBound,
        ProjectiveMutation::PassThrough,
    ] {
        let seen = pts
            .iter()
            .filter(|p| {
                let c = eval(&node, **p);
                (projective_law(**p, mutation) - c).abs() / c.abs().max(1.0) > rel_tol
            })
            .count();
        assert!(
            seen >= quarter,
            "{mutation:?}: only {seen} of {} points differ by more than {rel_tol:e}",
            pts.len()
        );
    }
}

/// The child of a node with exactly one child (a `child` field and no other
/// `SdfNode` field), read through serde: the enum is `#[non_exhaustive]` and
/// the crate's own child walker is private, and a `match` written here would
/// silently miss a new variant.
pub fn single_child(node: &SdfNode) -> Option<SdfNode> {
    let value = serde_json::to_value(node).expect("SdfNode serializes");
    let fields = value.as_object()?.values().next()?.as_object()?;
    let nodes = fields
        .values()
        .filter(|v| serde_json::from_value::<SdfNode>((*v).clone()).is_ok())
        .count();
    if nodes != 1 {
        return None;
    }
    serde_json::from_value(fields.get("child")?.clone()).ok()
}

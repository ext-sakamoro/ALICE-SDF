//! Integration tests: evaluator parity across every compiled opcode
//!
//! The tree evaluator (`eval`) is the oracle. For every compilable `SdfNode`
//! variant we compare:
//!
//! - `eval_compiled`      — scalar bytecode (shared exhaustive stack machine)
//! - `eval_compiled_simd` — 8-lane SIMD, lane 0 of a splatted point
//! - `eval_compiled_bvh`  — BVH bytecode (same stack machine; since 1.9.2 the
//!   BVH accepts every tree the main compiler does)
//! - `JitCompiledSdf` / `JitSimdSdf` (feature `jit`) — Cranelift scalar JIT and
//!   the SIMD JIT, **or** a loud `Err` from `compile` — never a silent MAX
//!   distance
//!
//! A final test asserts that the corpus below reaches every `OpCode` the
//! compiler can emit, so a new opcode that is not covered here fails CI.
//!
//! Background (1.9.1): before this test the scalar and BVH evaluators were
//! hand-copied with a `_ =>` fallback that turned unknown primitives into a
//! unit sphere and unknown modifiers into a no-op.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{CompileError, OpCode};
use alice_sdf::prelude::*;
use alice_sdf::transforms::skinning::BoneTransform;
use glam::Vec2;
use std::collections::BTreeSet;

const TOL: f32 = 1e-4;

fn simd_lane0(sdf: &CompiledSdf, p: Vec3) -> f32 {
    eval_compiled_simd(sdf, Vec3x8::splat(p)).to_array()[0]
}

fn sample_points() -> Vec<Vec3> {
    vec![
        Vec3::ZERO,
        Vec3::new(0.25, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(0.3, -0.7, 1.1),
        Vec3::new(1.5, 1.5, 1.5),
        Vec3::new(-2.0, 0.4, -0.9),
    ]
}

/// Compare tree vs scalar vs SIMD; BVH is compared when it compiles and must
/// otherwise reject loudly. Returns (mismatch lines, opcodes the bytecode emitted).
fn check_parity(name: &str, node: &SdfNode) -> (Vec<String>, BTreeSet<String>) {
    let compiled = CompiledSdf::try_compile(node)
        .unwrap_or_else(|e| panic!("{name}: CompiledSdf::try_compile failed: {e}"));
    let bvh = CompiledSdfBvh::try_compile(node)
        .unwrap_or_else(|e| panic!("{name}: CompiledSdfBvh::try_compile failed: {e}"));
    #[cfg(feature = "jit")]
    let jit_scalar = alice_sdf::compiled::jit::JitCompiledSdf::compile(node).ok();
    #[cfg(feature = "jit")]
    let jit_simd = alice_sdf::compiled::jit::JitSimdSdf::compile(&compiled).ok();
    let mut failures = Vec::new();
    for p in sample_points() {
        let tree = eval(node, p);
        // Comparing non-finite or astronomically large values is meaningless;
        // the degenerate-polygon sentinel (1e10) is covered separately.
        if !tree.is_finite() || tree.abs() > 1e6 {
            continue;
        }
        let scalar = eval_compiled(&compiled, p);
        let simd = simd_lane0(&compiled, p);
        #[allow(unused_mut)]
        let mut paths = vec![
            ("scalar", scalar),
            ("simd", simd),
            ("bvh", eval_compiled_bvh(&bvh, p)),
        ];
        #[cfg(feature = "jit")]
        {
            if let Some(j) = &jit_scalar {
                paths.push(("jit", j.eval(p)));
            }
            if let Some(j) = &jit_simd {
                let out = j.eval_batch(&[p.x; 8], &[p.y; 8], &[p.z; 8]);
                paths.push(("jit_simd", out[0]));
            }
        }
        for (path, v) in paths {
            if (v - tree).abs() > TOL * tree.abs().max(1.0) {
                failures.push(format!("{name} @ {p:?}: tree={tree:.6} {path}={v:.6}"));
            }
        }
    }
    let ops = compiled
        .instructions
        .iter()
        .map(|i| format!("{:?}", i.opcode))
        .collect();
    (failures, ops)
}

fn sphere() -> SdfNode {
    SdfNode::sphere(0.6)
}

fn unit_box() -> SdfNode {
    SdfNode::box3d(0.5, 0.4, 0.3)
}

fn identity_mat() -> [f32; 16] {
    let mut m = [0.0; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m
}

fn square_verts() -> Vec<Vec2> {
    vec![
        Vec2::new(-0.5, -0.5),
        Vec2::new(0.5, -0.5),
        Vec2::new(0.5, 0.5),
        Vec2::new(-0.5, 0.5),
    ]
}

/// Every compilable SdfNode variant with non-trivial parameters.
fn corpus() -> Vec<(&'static str, SdfNode)> {
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

#[test]
fn every_compilable_node_matches_tree_eval() {
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        failures.extend(check_parity(name, &node).0);
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The corpus must reach every opcode the compiler can emit, so that adding
/// an opcode without adding a parity case here fails CI.
#[test]
fn corpus_covers_every_emitted_opcode() {
    let mut seen = BTreeSet::new();
    for (name, node) in corpus() {
        seen.extend(check_parity(name, &node).1);
    }
    // 125 OpCode variants minus `Animated` (compiler inlines the child) = 124.
    // `End` and `PopTransform` are emitted and counted.
    let expected = 124;
    assert!(
        !seen.contains("Animated"),
        "compiler started emitting Animated; update the expected count"
    );
    assert_eq!(
        seen.len(),
        expected,
        "corpus reaches {} opcodes, expected {}: {:?}",
        seen.len(),
        expected,
        seen
    );
    // Spot-check the historically missing ones are actually reached.
    for op in [
        "Circle2D",
        "Rect2D",
        "RoundedRect2D",
        "Segment2D",
        "Polygon2D",
        "Annular2D",
        "ExpSmoothUnion",
        "ExpSmoothIntersection",
        "ExpSmoothSubtraction",
        "Shear",
    ] {
        assert!(seen.contains(op), "{op} not reached by corpus");
    }
    let _ = OpCode::Sphere; // keep the import meaningful for readers
}

#[test]
fn unsupported_nodes_are_rejected_loudly_not_silently() {
    // Main compiler: Terrain used to compile to a silent sphere.
    let terrain = SdfNode::Terrain {
        scale: 1.0,
        amplitude: 0.5,
    };
    assert!(matches!(
        CompiledSdf::try_compile(&terrain),
        Err(CompileError::UnsupportedPrimitive(_))
    ));

    // Both compilers reject the same set (no bytecode law): Terrain, Triangle, Bezier.
    let rejected: Vec<(&str, SdfNode)> = vec![
        ("terrain", terrain),
        ("triangle", SdfNode::triangle(Vec3::ZERO, Vec3::X, Vec3::Y)),
        ("bezier", SdfNode::bezier(Vec3::ZERO, Vec3::X, Vec3::Y, 0.1)),
    ];
    for (name, node) in rejected {
        assert!(
            matches!(
                CompiledSdf::try_compile(&node),
                Err(CompileError::UnsupportedPrimitive(_))
            ),
            "{name}: CompiledSdf must reject with UnsupportedPrimitive"
        );
        assert!(
            matches!(
                CompiledSdfBvh::try_compile(&node),
                Err(CompileError::UnsupportedPrimitive(_))
            ),
            "{name}: CompiledSdfBvh must reject with UnsupportedPrimitive"
        );
    }
    // Since 1.9.2 the BVH accepts what used to be silently replaced by sphere(0.001).
    for (name, node) in [
        ("ifs", sphere().ifs(vec![identity_mat()], 2)),
        (
            "projective",
            unit_box().projective_transform(identity_mat(), 1.0),
        ),
        ("icosahedral", unit_box().icosahedral_symmetry()),
        ("circle_2d", SdfNode::circle_2d(0.25, 0.5)),
    ] {
        assert!(
            CompiledSdfBvh::try_compile(&node).is_ok(),
            "{name}: BVH must compile"
        );
    }
}

/// Every primitive's AABB must contain every point where the SDF is ≤ 0.
/// Sampled on a grid so a too-tight conservative bound in `refit::primitive_aabb`
/// fails here instead of silently culling geometry downstream.
#[test]
fn primitive_and_scene_aabbs_are_conservative() {
    let mut failures = Vec::new();
    for (name, node) in corpus() {
        let bvh = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&bvh);
        if !aabb.is_valid() {
            failures.push(format!("{name}: scene AABB is empty/invalid"));
            continue;
        }
        if !aabb.min().is_finite() || !aabb.max().is_finite() {
            continue; // unbounded shape: infinite AABB is trivially conservative
        }
        let steps = 25;
        let extent = 3.0f32;
        for ix in 0..steps {
            for iy in 0..steps {
                for iz in 0..steps {
                    let f = |i: usize| -extent + 2.0 * extent * (i as f32) / ((steps - 1) as f32);
                    let p = Vec3::new(f(ix), f(iy), f(iz));
                    let d = eval(&node, p);
                    if d <= 0.0 && aabb.distance_to_point(p) > 1e-4 {
                        failures.push(format!(
                            "{name}: inside point {p:?} (d={d:.4}) outside AABB [{:?}, {:?}]",
                            aabb.min(),
                            aabb.max()
                        ));
                        break;
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} AABB violations:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn polygon_2d_vertices_survive_compilation() {
    // A triangle and a square must evaluate differently at the same point
    // (previously the compiler dropped the vertices and both were a unit sphere).
    let tri = SdfNode::polygon_2d(
        vec![
            Vec2::new(-0.5, -0.5),
            Vec2::new(0.5, -0.5),
            Vec2::new(0.0, 0.5),
        ],
        0.5,
    );
    let sq = SdfNode::polygon_2d(square_verts(), 0.5);
    let p = Vec3::new(0.4, 0.4, 0.0);
    let d_tri = eval_compiled(&CompiledSdf::compile(&tri), p);
    let d_sq = eval_compiled(&CompiledSdf::compile(&sq), p);
    assert!((d_tri - eval(&tri, p)).abs() < TOL);
    assert!((d_sq - eval(&sq, p)).abs() < TOL);
    assert!((d_tri - d_sq).abs() > 0.05, "tri={d_tri} sq={d_sq}");
}

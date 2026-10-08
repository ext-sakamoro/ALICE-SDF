//! Oracles for the compiled bytecode and its CPU evaluators: the opcode
//! classification, the instruction skip structure, the compile-time facts of
//! `CompiledSdf` / `CompiledSdfBvh`, the per-primitive AABBs, the batch /
//! distance-and-normal / SIMD gradient entry points, the `Vec3R` helpers and
//! `InstancedSdf`.
//!
//! Expected values come from closed forms, from the tree evaluator, or from
//! an independent walk written here; none is produced by the function under
//! test. Every check counts its comparisons and fails on zero.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

mod common;

use alice_sdf::animation::AnimationParams;
use alice_sdf::compiled::real::Vec3R;
use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch, eval_compiled_distance_and_normal, eval_compiled_simd,
    eval_gradient_simd, get_scene_aabb, AabbPacked, CompiledSdf, CompiledSdfBvh, InstancedSdf,
    Instruction, OpCode, OpKind, Vec3x8,
};
use alice_sdf::prelude::*;
use glam::{EulerRot, Quat};
use wide::f32x8;

/// The corpus entries the bytecode compiler accepts (the tree-only ones,
/// e.g. `Terrain`, return `Err` and are covered by the tree evaluator tests).
fn compilable() -> Vec<(&'static str, SdfNode)> {
    let all = common::corpus::corpus();
    let n = all.len();
    let ok: Vec<_> = all
        .into_iter()
        .filter(|(_, node)| CompiledSdf::try_compile(node).is_ok())
        .collect();
    assert!(
        ok.len() * 10 >= n * 9,
        "only {} of {n} corpus trees compile",
        ok.len()
    );
    ok
}

fn corpus_bytecode() -> Vec<(&'static str, CompiledSdf)> {
    compilable()
        .into_iter()
        .map(|(name, node)| (name, CompiledSdf::compile(&node)))
        .collect()
}

// ---------------------------------------------------------------------------
// OpCode classification
// ---------------------------------------------------------------------------

/// Frame-pushing opcodes for which the VM rewrites the evaluation point
/// before the child runs (read off `eval_core`: every arm that assigns `p`).
const POINT_MODIFIERS: &[OpCode] = &[
    OpCode::Translate,
    OpCode::Rotate,
    OpCode::Scale,
    OpCode::ScaleNonUniform,
    OpCode::ProjectiveTransform,
    OpCode::LatticeDeform,
    OpCode::SdfSkinning,
    OpCode::Twist,
    OpCode::Bend,
    OpCode::RepeatInfinite,
    OpCode::RepeatFinite,
    OpCode::Elongate,
    OpCode::Mirror,
    OpCode::OctantMirror,
    OpCode::Revolution,
    OpCode::Extrude,
    OpCode::Taper,
    OpCode::PolarRepeat,
    OpCode::SweepBezier,
    OpCode::Shear,
    OpCode::IcosahedralSymmetry,
    OpCode::IFS,
];

/// Opcodes whose frame the VM post-processes at `PopTransform` (read off the
/// `PopTransform` arm of `eval_core`).
const POST_PROCESS: &[OpCode] = &[
    OpCode::Scale,
    OpCode::ScaleNonUniform,
    OpCode::ProjectiveTransform,
    OpCode::LatticeDeform,
    OpCode::IFS,
    OpCode::Round,
    OpCode::Onion,
    OpCode::Noise,
    OpCode::Extrude,
    OpCode::Displacement,
    OpCode::Taper,
    OpCode::HeightmapDisplacement,
    OpCode::SurfaceRoughness,
];

/// Every frame-pushing opcode (transforms and modifiers), including the ones
/// the compiler never emits (`Animated`).
const FRAME_OPCODES: &[OpCode] = &[
    OpCode::Translate,
    OpCode::Rotate,
    OpCode::Scale,
    OpCode::ScaleNonUniform,
    OpCode::ProjectiveTransform,
    OpCode::LatticeDeform,
    OpCode::SdfSkinning,
    OpCode::Twist,
    OpCode::Bend,
    OpCode::RepeatInfinite,
    OpCode::RepeatFinite,
    OpCode::Round,
    OpCode::Onion,
    OpCode::Elongate,
    OpCode::Noise,
    OpCode::Mirror,
    OpCode::Revolution,
    OpCode::Extrude,
    OpCode::Taper,
    OpCode::Displacement,
    OpCode::PolarRepeat,
    OpCode::SweepBezier,
    OpCode::OctantMirror,
    OpCode::Shear,
    OpCode::Animated,
    OpCode::IcosahedralSymmetry,
    OpCode::IFS,
    OpCode::HeightmapDisplacement,
    OpCode::SurfaceRoughness,
];

#[test]
fn opcode_predicates_match_the_vm_and_kind() {
    let mut seen: Vec<OpCode> = FRAME_OPCODES.to_vec();
    for (_, c) in corpus_bytecode() {
        seen.extend(c.instructions().iter().map(|i| i.opcode));
    }
    seen.extend([OpCode::PopTransform, OpCode::End]);

    let mut compared = 0usize;
    for op in &seen {
        let kind = op.kind();
        assert_eq!(op.is_primitive(), kind == OpKind::Primitive, "{op:?}");
        assert_eq!(op.is_binary_op(), kind == OpKind::Binary, "{op:?}");
        assert_eq!(op.is_transform(), kind == OpKind::Transform, "{op:?}");
        assert_eq!(op.is_modifier(), kind == OpKind::Modifier, "{op:?}");
        assert_eq!(
            op.modifies_point(),
            POINT_MODIFIERS.contains(op),
            "modifies_point({op:?})"
        );
        assert_eq!(
            op.is_post_process(),
            POST_PROCESS.contains(op),
            "is_post_process({op:?})"
        );
        compared += 1;
    }
    // Every frame opcode is a transform or a modifier and nothing else is.
    for op in FRAME_OPCODES {
        assert!(op.is_transform() || op.is_modifier(), "{op:?}");
    }
    for op in POINT_MODIFIERS.iter().chain(POST_PROCESS) {
        assert!(FRAME_OPCODES.contains(op), "{op:?} pushes no frame");
    }
    assert!(compared > 100, "compared {compared}");
}

#[test]
fn every_emitted_frame_opcode_is_classified_here() {
    let mut n = 0;
    for (name, c) in corpus_bytecode() {
        for inst in c.instructions() {
            let k = inst.opcode.kind();
            if k == OpKind::Transform || k == OpKind::Modifier {
                assert!(
                    FRAME_OPCODES.contains(&inst.opcode),
                    "{name}: {:?} missing from FRAME_OPCODES",
                    inst.opcode
                );
                n += 1;
            }
        }
    }
    assert!(n > 20, "frame instructions seen: {n}");
}

// ---------------------------------------------------------------------------
// Instruction structure
// ---------------------------------------------------------------------------

#[test]
fn next_instruction_index_skips_exactly_the_subtree() {
    let mut compared = 0usize;
    for (name, c) in corpus_bytecode() {
        let insts = c.instructions();
        // Independent matching: a frame opcode at i is closed by the
        // PopTransform at which the frame depth returns to its level.
        let mut open: Vec<usize> = Vec::new();
        let mut close = vec![usize::MAX; insts.len()];
        for (i, inst) in insts.iter().enumerate() {
            match inst.opcode.kind() {
                OpKind::Transform | OpKind::Modifier => open.push(i),
                OpKind::PopTransform => close[open.pop().expect("balanced frames")] = i,
                _ => {}
            }
        }
        assert!(open.is_empty(), "{name}: unbalanced frames");
        for (i, inst) in insts.iter().enumerate() {
            let leaf = inst.opcode.is_primitive() || inst.opcode.is_binary_op();
            assert_eq!(inst.is_leaf(), leaf, "{name}[{i}] {:?}", inst.opcode);
            let expected = if close[i] == usize::MAX {
                i + 1
            } else {
                close[i] + 1
            };
            assert_eq!(
                inst.next_instruction_index(i),
                expected,
                "{name}[{i}] {:?}",
                inst.opcode
            );
            compared += 1;
        }
        assert_eq!(insts.last().map(|i| i.opcode), Some(OpCode::End), "{name}");
    }
    assert!(compared > 300, "compared {compared}");
}

#[test]
fn animated_instruction_is_a_point_preserving_frame() {
    let inst = Instruction::animated(2.5, 0.75);
    assert_eq!(inst.opcode, OpCode::Animated);
    assert_eq!(inst.params[0], 2.5);
    assert_eq!(inst.params[1], 0.75);
    assert_eq!(inst.child_count, 1);
    assert!(inst.opcode.is_modifier());
    assert!(!inst.is_leaf());
    assert!(!inst.opcode.modifies_point());
    assert!(!inst.opcode.is_post_process());
}

// ---------------------------------------------------------------------------
// CompiledSdf / CompiledSdfBvh compile-time facts
// ---------------------------------------------------------------------------

#[test]
fn compiled_facts_match_the_tree_and_the_layout() {
    let inst_size = std::mem::size_of::<Instruction>();
    let mut compared = 0usize;
    for (name, node) in compilable() {
        let c = CompiledSdf::compile(&node);
        assert_eq!(c.node_count(), node.node_count() as usize, "{name}");
        assert_eq!(
            c.memory_size(),
            c.instruction_count() * inst_size + c.aux_data().len() * 4,
            "{name}"
        );
        assert_eq!(
            c.lipschitz().to_bits(),
            alice_sdf::interval::eval_lipschitz(&node).to_bits(),
            "{name}"
        );
        let bvh = CompiledSdfBvh::compile(&node);
        assert_eq!(bvh.instruction_count(), c.instruction_count(), "{name}");
        assert_eq!(bvh.aux_data, c.aux_data(), "{name}");
        assert_eq!(
            bvh.memory_size(),
            bvh.instruction_count() * inst_size
                + bvh.aabbs.len() * std::mem::size_of::<AabbPacked>()
                + bvh.aux_data.len() * 4,
            "{name}"
        );
        compared += 1;
    }
    assert!(compared > 50, "compared {compared}");

    // Closed forms: a sphere is one node, a polygon carries its vertices.
    let sphere = CompiledSdf::compile(&SdfNode::sphere(1.0));
    assert_eq!(sphere.node_count(), 1);
    assert!(sphere.aux_data().is_empty());
    assert_eq!(sphere.lipschitz(), 1.0);
    let verts = common::corpus::square_verts();
    let poly = CompiledSdf::compile(&SdfNode::polygon_2d(verts.clone(), 0.2));
    let flat: Vec<f32> = verts.iter().flat_map(|v| [v.x, v.y]).collect();
    assert_eq!(poly.aux_data(), flat.as_slice());
    // A gyroid is not 1-Lipschitz; the compiled bound is the tree's.
    let g = CompiledSdf::compile(&SdfNode::gyroid(3.0, 0.1));
    assert!(g.lipschitz() > 1.0, "gyroid L = {}", g.lipschitz());
}

#[test]
fn refit_all_from_bytecode_restores_the_fresh_aabbs() {
    let mut compared = 0usize;
    for (name, node) in compilable() {
        let fresh = CompiledSdfBvh::compile(&node);
        let mut bvh = fresh.clone();
        for a in &mut bvh.aabbs {
            *a = AabbPacked::empty();
        }
        bvh.scene_aabb = AabbPacked::empty();
        let n = bvh.refit_all_from_bytecode().expect("refit");
        // One box per instruction that produces a value or a frame
        // (`PopTransform` writes its transform's box, `End` none).
        let producing = bvh
            .instructions
            .iter()
            .filter(|i| !matches!(i.opcode.kind(), OpKind::PopTransform | OpKind::End))
            .count();
        assert_eq!(n, producing, "{name}");
        for (i, (a, b)) in bvh.aabbs.iter().zip(&fresh.aabbs).enumerate() {
            assert_eq!(a.min().to_array(), b.min().to_array(), "{name}[{i}]");
            assert_eq!(a.max().to_array(), b.max().to_array(), "{name}[{i}]");
            compared += 1;
        }
    }
    assert!(compared > 300, "compared {compared}");

    // Editing a radius and refitting moves the box (closed form).
    let mut bvh = CompiledSdfBvh::compile(&SdfNode::sphere(1.0));
    bvh.instructions[0].params[0] = 2.5;
    bvh.refit_all_from_bytecode().expect("refit");
    assert_eq!(get_scene_aabb(&bvh).max(), Vec3::splat(2.5));
    assert_eq!(get_scene_aabb(&bvh).min(), Vec3::splat(-2.5));
}

/// Closed-form bounds of the primitives whose box comes from
/// `aabb::primitives` (rounded cone, pyramid, octahedron, hex prism, link).
#[test]
fn primitive_boxes_are_the_closed_form_extent() {
    // Constructors take full heights / lengths and store halves.
    let cases: Vec<(&str, SdfNode, Vec3)> = vec![
        // spheres r1 at y = -h/2, r2 at y = +h/2
        (
            "rounded_cone",
            SdfNode::rounded_cone(0.4, 0.2, 0.6),
            Vec3::new(0.4, 0.3 + 0.4, 0.4),
        ),
        ("pyramid", SdfNode::pyramid(0.7), Vec3::new(0.5, 0.35, 0.5)),
        ("octahedron", SdfNode::octahedron(0.6), Vec3::splat(0.6)),
        // apothem 0.5 along Y (flat edges), vertices at x = ±0.5 · 2/√3
        (
            "hex_prism",
            SdfNode::hex_prism(0.5, 0.3),
            Vec3::new(0.5 * 2.0 / 3f32.sqrt(), 0.5, 0.15),
        ),
        // torus (r1, r2) in XY stretched by l/2 along Y: |y| <= l/2 + r1 + r2
        (
            "link",
            SdfNode::link(0.3, 0.25, 0.08),
            Vec3::new(0.25 + 0.08, 0.15 + 0.25 + 0.08, 0.08),
        ),
    ];
    let mut compared = 0usize;
    for (name, node, half) in cases {
        let bvh = CompiledSdfBvh::compile(&node);
        let aabb = get_scene_aabb(&bvh);
        let lo = aabb.min();
        let hi = aabb.max();
        // Box is symmetric except the rounded cone (r1 below, r2 above).
        if name == "rounded_cone" {
            assert!(
                (lo - Vec3::new(-0.4, -0.7, -0.4)).abs().max_element() < 1e-6,
                "{name}: {lo}"
            );
            assert!(
                (hi - Vec3::new(0.4, 0.5, 0.4)).abs().max_element() < 1e-6,
                "{name}: {hi}"
            );
        } else {
            assert!(
                (hi - half).abs().max_element() < 1e-6,
                "{name}: {hi} vs {half}"
            );
            assert!(
                (lo + half).abs().max_element() < 1e-6,
                "{name}: {lo} vs {half}"
            );
        }
        // Conservative: every interior / surface sample lies in the box.
        // Tight: on every axis the solid samples reach within two grid
        // cells of both faces.
        let n = 64;
        let span = hi.max(-lo) * 1.25;
        let cell = 2.0 * span / (n - 1) as f32;
        let mut smin = Vec3::splat(f32::INFINITY);
        let mut smax = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let t = Vec3::new(i as f32, j as f32, k as f32) / (n - 1) as f32;
                    let p = -span + 2.0 * span * t;
                    if eval(&node, p) <= 0.0 {
                        let inside = p.cmpge(lo - 1e-5).all() && p.cmple(hi + 1e-5).all();
                        assert!(inside, "{name}: solid sample {p} outside [{lo}, {hi}]");
                        smin = smin.min(p);
                        smax = smax.max(p);
                        compared += 1;
                    }
                }
            }
        }
        assert!(
            (hi - smax).cmple(2.0 * cell).all(),
            "{name}: box {hi} not tight, samples reach {smax}"
        );
        assert!(
            (smin - lo).cmple(2.0 * cell).all(),
            "{name}: box {lo} not tight, samples reach {smin}"
        );
    }
    assert!(compared > 1000, "compared {compared}");
}

#[test]
fn aabb_from_half_size_and_fast_distance_closed_form() {
    let half = Vec3::new(1.0, 2.0, 0.5);
    let b = AabbPacked::from_half_size(half);
    assert_eq!(b.min(), -half);
    assert_eq!(b.max(), half);
    let mut compared = 0;
    for i in -8..=8 {
        for j in -8..=8 {
            for k in -8..=8 {
                let p = Vec3::new(i as f32, j as f32, k as f32) * 0.4;
                let out = (p.abs() - half).max(Vec3::ZERO);
                // L∞ distance outside, 0 inside
                let linf = out.max_element();
                assert_eq!(b.distance_to_point_fast(p), linf, "{p}");
                // a lower bound of the Euclidean distance to the box
                assert!(b.distance_to_point_fast(p) <= b.distance_to_point(p).max(0.0) + 1e-6);
                compared += 1;
            }
        }
    }
    assert!(compared > 1000);
}

// ---------------------------------------------------------------------------
// Evaluator entry points
// ---------------------------------------------------------------------------

#[test]
fn eval_compiled_batch_is_pointwise_eval_compiled() {
    let pts = common::test_grid_points(9);
    let mut compared = 0usize;
    for (name, c) in corpus_bytecode() {
        let batch = eval_compiled_batch(&c, &pts);
        assert_eq!(batch.len(), pts.len(), "{name}");
        for (d, p) in batch.iter().zip(&pts) {
            assert_eq!(d.to_bits(), eval_compiled(&c, *p).to_bits(), "{name} @ {p}");
            compared += 1;
        }
    }
    assert!(compared > 10_000, "compared {compared}");
}

#[test]
fn distance_and_normal_closed_forms() {
    let e = 1e-3;
    let mut compared = 0;
    // Plane n·p - d: linear, so the tetrahedral average is the centre
    // value and the normal is n.
    let n = Vec3::new(1.0, 2.0, -2.0).normalize();
    let plane = CompiledSdf::compile(&SdfNode::plane(n, 0.3));
    // Sphere |p| - r: the four offsets δ satisfy Σδ = 0 and Σδδᵀ = 4e²I, so
    // the average is d + e²/|p| + O(e⁴) and the normal is p/|p| + O(e²).
    let r = 0.8;
    let sphere = CompiledSdf::compile(&SdfNode::sphere(r));
    for i in 0..64 {
        let a = i as f32 * 0.37;
        let p = Vec3::new(
            a.cos() * 1.3,
            (a * 0.7).sin() * 0.9,
            (a * 1.3).cos() * 0.6 + 0.2,
        );
        let (d, nn) = eval_compiled_distance_and_normal(&plane, p, e);
        assert!((d - (n.dot(p) - 0.3)).abs() < 2e-5, "plane d at {p}");
        assert!((nn - n).length() < 1e-2, "plane n at {p}: {nn}");
        let (d, nn) = eval_compiled_distance_and_normal(&sphere, p, e);
        let l = p.length();
        let expected = l - r + e * e / l;
        assert!(
            (d - expected).abs() < 2e-6,
            "sphere d at {p}: {d} vs {expected}"
        );
        assert!((nn - p / l).length() < 1e-3, "sphere n at {p}");
        compared += 2;
    }
    assert!(compared > 100);
}

#[test]
fn simd_gradient_is_the_scalar_tetrahedral_gradient() {
    let e = 1e-3;
    let pts = common::test_grid_points(8);
    let mut compared = 0usize;
    for (name, c) in corpus_bytecode() {
        for chunk in pts.chunks_exact(8) {
            let p = Vec3x8 {
                x: f32x8::new(std::array::from_fn(|i| chunk[i].x)),
                y: f32x8::new(std::array::from_fn(|i| chunk[i].y)),
                z: f32x8::new(std::array::from_fn(|i| chunk[i].z)),
            };
            let (gx, gy, gz) = eval_gradient_simd(&c, p, e);
            let (gx, gy, gz) = (gx.to_array(), gy.to_array(), gz.to_array());
            for (l, q) in chunk.iter().enumerate() {
                let f = |o: Vec3| eval_compiled(&c, *q + o * e);
                let v0 = f(Vec3::new(1.0, -1.0, -1.0));
                let v1 = f(Vec3::new(-1.0, -1.0, 1.0));
                let v2 = f(Vec3::new(-1.0, 1.0, -1.0));
                let v3 = f(Vec3::new(1.0, 1.0, 1.0));
                let g = [v0 - v1 - v2 + v3, -v0 - v1 + v2 + v3, -v0 + v1 - v2 + v3];
                assert_eq!(gx[l].to_bits(), g[0].to_bits(), "{name} gx @ {q}");
                assert_eq!(gy[l].to_bits(), g[1].to_bits(), "{name} gy @ {q}");
                assert_eq!(gz[l].to_bits(), g[2].to_bits(), "{name} gz @ {q}");
                compared += 1;
            }
        }
    }
    assert!(compared > 5_000, "compared {compared}");

    // Sphere: the gradient points along p.
    let c = CompiledSdf::compile(&SdfNode::sphere(1.0));
    let p = Vec3::new(0.6, -0.8, 1.1);
    let v = Vec3x8 {
        x: f32x8::splat(p.x),
        y: f32x8::splat(p.y),
        z: f32x8::splat(p.z),
    };
    let (gx, gy, gz) = eval_gradient_simd(&c, v, e);
    let g = Vec3::new(gx.to_array()[0], gy.to_array()[0], gz.to_array()[0]).normalize();
    assert!((g - p.normalize()).length() < 1e-3);
}

#[test]
fn vec3r_round_and_max_element_match_std() {
    let vals = [
        -2.5f32,
        -1.5,
        -0.5,
        -0.49,
        0.0,
        0.49,
        0.5,
        1.5,
        2.5,
        3.7,
        -3.7,
        1e7 + 0.5,
    ];
    let mut compared = 0;
    for &a in &vals {
        for &b in &vals {
            let c = a * 0.5 - b;
            let v: Vec3R<f32> = Vec3R::new(a, b, c);
            let r = v.round();
            // std `f32::round`: nearest, ties away from zero
            assert_eq!(r.x.to_bits(), a.round().to_bits(), "{a}");
            assert_eq!(r.y.to_bits(), b.round().to_bits(), "{b}");
            assert_eq!(r.z.to_bits(), c.round().to_bits(), "{c}");
            assert_eq!(v.max_element(), a.max(b).max(c));
            // the 8-lane impl gives the same lanes
            let w: Vec3R<f32x8> = Vec3R::new(f32x8::splat(a), f32x8::splat(b), f32x8::splat(c));
            let rw = w.round();
            assert_eq!(rw.x.to_array()[3].to_bits(), r.x.to_bits());
            assert_eq!(rw.z.to_array()[7].to_bits(), r.z.to_bits());
            assert_eq!(w.max_element().to_array()[5], v.max_element());
            compared += 1;
        }
    }
    assert!(compared > 100);
}

// ---------------------------------------------------------------------------
// InstancedSdf
// ---------------------------------------------------------------------------

fn instances() -> Vec<AnimationParams> {
    (0..19)
        .map(|i| {
            let f = i as f32;
            AnimationParams {
                translate_x: (f * 0.9).cos() * 2.0,
                translate_y: (f * 0.4).sin() * 1.5,
                translate_z: f * 0.1 - 0.9,
                rotate_x: if i % 3 == 0 { 0.0 } else { f * 0.31 },
                rotate_y: f * 0.17,
                rotate_z: if i % 2 == 0 { 0.4 } else { -0.2 },
                scale: if i % 4 == 0 { 1.0 } else { 0.5 + f * 0.05 },
                ..Default::default()
            }
        })
        .collect()
}

/// The tree an instance stands for: scale, then rotate (XYZ Euler), then
/// translate — the inverse of `AnimationParams::transform_point`.
fn instance_tree(base: &SdfNode, a: &AnimationParams) -> SdfNode {
    base.clone()
        .scale(a.scale)
        .rotate(Quat::from_euler(
            EulerRot::XYZ,
            a.rotate_x,
            a.rotate_y,
            a.rotate_z,
        ))
        .translate(a.translate_x, a.translate_y, a.translate_z)
}

#[test]
fn instanced_min_is_the_union_of_the_instance_trees() {
    let base = SdfNode::box3d(0.3, 0.2, 0.4).smooth_union(SdfNode::sphere(0.25), 0.1);
    let params = instances();
    let mut inst = InstancedSdf::with_capacity(CompiledSdf::compile(&base), params.len());
    for a in &params {
        inst.add_instance(*a);
    }
    inst.add_at(-3.0, 0.5, 0.25);
    let mut all = params;
    all.push(AnimationParams {
        translate_x: -3.0,
        translate_y: 0.5,
        translate_z: 0.25,
        scale: 1.0,
        ..Default::default()
    });
    assert_eq!(inst.instance_count(), all.len());
    let trees: Vec<SdfNode> = all.iter().map(|a| instance_tree(&base, a)).collect();

    let pts = common::test_grid_points(13)
        .into_iter()
        .map(|p| p * 3.0)
        .collect::<Vec<_>>();
    let mut compared = 0usize;
    for p in &pts {
        let per = inst.eval_per_instance(*p);
        assert_eq!(per.len(), trees.len());
        for (d, t) in per.iter().zip(&trees) {
            let e = eval(t, *p);
            assert!((d - e).abs() < 1e-4 * (1.0 + e.abs()), "{p}: {d} vs {e}");
            compared += 1;
        }
        let union = trees.iter().map(|t| eval(t, *p)).fold(f32::MAX, f32::min);
        let m = inst.eval_min(*p);
        assert!(
            (m - union).abs() < 1e-4 * (1.0 + union.abs()),
            "{p}: {m} vs {union}"
        );
        let ms = inst.eval_min_simd(*p);
        assert!(
            (ms - m).abs() < 1e-5 * (1.0 + m.abs()),
            "simd {p}: {ms} vs {m}"
        );
        compared += 2;
    }
    // Batches are the pointwise results, on both sides of the parallel cut-off.
    for n in [37usize, pts.len()] {
        let sub = &pts[..n];
        let b = inst.eval_min_batch(sub);
        let bs = inst.eval_min_batch_simd(sub);
        for (k, p) in sub.iter().enumerate() {
            assert_eq!(b[k].to_bits(), inst.eval_min(*p).to_bits());
            assert_eq!(bs[k].to_bits(), inst.eval_min_simd(*p).to_bits());
            compared += 2;
        }
    }
    assert!(compared > 50_000, "compared {compared}");
}

#[test]
fn instanced_translation_only_is_bit_identical_and_empty_is_max() {
    let base = SdfNode::sphere(0.5);
    let c = CompiledSdf::compile(&base);
    let mut inst = InstancedSdf::new(c.clone());
    assert_eq!(inst.instance_count(), 0);
    assert_eq!(inst.eval_min(Vec3::ZERO), f32::MAX);
    assert_eq!(inst.eval_min_simd(Vec3::ZERO), f32::MAX);
    let offsets: Vec<Vec3> = (0..11)
        .map(|i| Vec3::new(i as f32 * 1.25 - 6.0, (i % 3) as f32, -(i as f32) * 0.5))
        .collect();
    for o in &offsets {
        inst.add_at(o.x, o.y, o.z);
    }
    let mut compared = 0;
    for p in common::test_grid_points(9).into_iter().map(|p| p * 4.0) {
        let expected = offsets
            .iter()
            .map(|o| eval_compiled(&c, p - *o))
            .fold(f32::MAX, f32::min);
        assert_eq!(inst.eval_min(p).to_bits(), expected.to_bits(), "{p}");
        // SIMD: one lane of eval_compiled_simd is eval_compiled
        let lane = eval_compiled_simd(
            &c,
            Vec3x8 {
                x: f32x8::splat(p.x - offsets[0].x),
                y: f32x8::splat(p.y - offsets[0].y),
                z: f32x8::splat(p.z - offsets[0].z),
            },
        )
        .to_array()[0];
        assert_eq!(lane.to_bits(), eval_compiled(&c, p - offsets[0]).to_bits());
        assert_eq!(
            inst.eval_min_simd(p).to_bits(),
            expected.to_bits(),
            "simd {p}"
        );
        compared += 1;
    }
    assert!(compared > 500);
}

/// Sampled conservativeness of boxes whose solid leaves the naive bound: the
/// tube ring (`R + t`, not `max(R, hh)`), the pipe (a tube of radius r around
/// the curve a = b = 0, outside both operands where they touch) and the
/// tongue (material added outside the LHS).
#[test]
fn boxes_contain_material_outside_the_naive_bound() {
    let cube_a = SdfNode::box3d(1.0, 1.0, 1.0);
    // full sizes: unit cubes touching on the face x = 0.5
    let cube_b = SdfNode::box3d(1.0, 1.0, 1.0).translate(1.0, 0.0, 0.0);
    let a = SdfNode::sphere(0.6);
    let b = SdfNode::sphere(0.5).translate(0.8, 0.0, 0.0);
    let cases = vec![
        ("tube", SdfNode::tube(0.5, 0.1, 0.4)),
        ("pipe_touching_boxes", cube_a.pipe(cube_b, 0.3)),
        ("tongue", a.tongue(b, 0.2, 0.1)),
    ];
    let mut compared = 0usize;
    for (name, node) in cases {
        let aabb = get_scene_aabb(&CompiledSdfBvh::compile(&node));
        let (lo, hi) = (aabb.min(), aabb.max());
        assert!(
            lo.is_finite() && hi.is_finite() && aabb.is_valid(),
            "{name}: {lo} {hi}"
        );
        let n = 72;
        let span = Vec3::splat(3.5);
        let centre = Vec3::new(1.0, 0.0, 0.0);
        let mut solid = 0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let t = Vec3::new(i as f32, j as f32, k as f32) / (n - 1) as f32;
                    let p = centre - span + 2.0 * span * t;
                    if eval(&node, p) <= 0.0 {
                        assert!(
                            p.cmpge(lo - 1e-5).all() && p.cmple(hi + 1e-5).all(),
                            "{name}: solid sample {p} outside [{lo}, {hi}]"
                        );
                        solid += 1;
                    }
                    compared += 1;
                }
            }
        }
        assert!(solid > 50, "{name}: only {solid} solid samples");
    }
    assert!(compared > 1_000_000);

    // Tube closed form: radial R + t, height ±h/2 (the constructor takes the full height).
    let aabb = get_scene_aabb(&CompiledSdfBvh::compile(&SdfNode::tube(0.5, 0.1, 0.4)));
    assert!(
        (aabb.max() - Vec3::new(0.6, 0.2, 0.6)).abs().max_element() < 1e-6,
        "{}",
        aabb.max()
    );
}

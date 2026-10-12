//! Seeded, non-tolerance oracle for the one min/max rule across evaluators.
//!
//! The tree and SIMD paths now route `min`/`max` through the Law helper, which
//! matches Cranelift's ordered `fmax`/`fmin` on signed zero. This test seeds the
//! case the Law exists for: **coincident boundaries**, where a CSG `max` sees
//! `(+0, -0)` because both operands sit on their surfaces at once. On a target
//! whose `f32::max(+0,-0)` is `-0` (x86_64), this test is red before the Law and
//! green after; on aarch64 it is green either way. Separately it guards that no
//! tree/JIT disagreement is ever a signed-zero split, and that a tree-built SVO
//! and a bytecode-built SVO serialize to identical bytes (same-arch determinism).
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "jit")]
mod common;
use alice_sdf::compiled::jit::JitCompiledSdf;
use alice_sdf::prelude::*;

/// `-0` and `+0` compare equal but carry different bits; this is a sign split.
fn is_signed_zero_split(a: f32, b: f32) -> bool {
    a == 0.0 && b == 0.0 && a.to_bits() != b.to_bits()
}

/// Scenes whose two operands share a surface, so a `max`/`min` on the boundary
/// is handed `(+0, -0)`. Identical boxes make every surface point coincident.
fn coincident_scenes() -> Vec<(&'static str, SdfNode)> {
    let b = || SdfNode::box3d(1.0, 1.0, 1.0); // faces at ±0.5, matching boundary_points
    vec![
        ("subtract_identical", b().subtract(b())),
        ("intersect_identical", b().intersection(b())),
        ("union_identical", b().union(b())),
        (
            "subtract_shared_face",
            SdfNode::box3d(1.0, 1.0, 1.0).subtract(SdfNode::box3d(1.0, 2.0, 2.0)),
        ),
    ]
}

/// Points that land exactly on the shared faces (x,y,z ∈ {±0.5, 0}).
fn boundary_points() -> Vec<Vec3> {
    let c = [-0.5f32, 0.0, 0.5];
    let mut v = vec![];
    for &x in &c {
        for &y in &c {
            for &z in &c {
                v.push(Vec3::new(x, y, z));
            }
        }
    }
    v
}

#[test]
fn tree_and_jit_agree_on_signed_zero_at_coincident_boundaries() {
    let mut checked = 0usize;
    for (name, node) in coincident_scenes() {
        let jit = JitCompiledSdf::compile(&node).expect("jit compile");
        for p in boundary_points() {
            let t = eval(&node, p);
            let j = jit.eval(p);
            assert!(
                !is_signed_zero_split(t, j),
                "{name} @ {p:?}: tree {:08x} vs jit {:08x} is a signed-zero split",
                t.to_bits(),
                j.to_bits()
            );
            // on the coincident scenes the distance on a shared face is a zero;
            // after the Law the two evaluators agree on its sign bit exactly.
            if t == 0.0 && j == 0.0 {
                assert_eq!(t.to_bits(), j.to_bits(), "{name} @ {p:?}: zero sign bit");
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "no coincident zero was exercised (seed is vacuous)"
    );
}

#[test]
fn no_signed_zero_split_over_a_csg_grid() {
    let scenes = coincident_scenes();
    let mut coords = vec![];
    let mut x = -1.5f32;
    while x <= 1.5001 {
        coords.push(x);
        x += 0.25;
    }
    for (name, node) in scenes {
        let jit = JitCompiledSdf::compile(&node).expect("jit");
        for &a in &coords {
            for &b in &coords {
                for &c in &coords {
                    let p = Vec3::new(a, b, c);
                    let (t, j) = (eval(&node, p), jit.eval(p));
                    assert!(
                        !is_signed_zero_split(t, j),
                        "{name} @ {p:?}: signed-zero split tree {:08x} jit {:08x}",
                        t.to_bits(),
                        j.to_bits()
                    );
                }
            }
        }
    }
}

#[cfg(feature = "svo")]
#[test]
fn tree_and_bytecode_svo_serialize_identically() {
    use alice_sdf::compiled::CompiledSdf;
    use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};
    // SVO distances come from the Real path (tree `build`, bytecode
    // `build_compiled`); the Law makes both agree on every zero's sign, so the
    // serialized chunks are byte-identical. A ±0 divergence would change bytes.
    let cfg = SvoBuildConfig::default();
    let mut compared = 0usize;
    for (name, node) in coincident_scenes() {
        let compiled = CompiledSdf::compile(&node);
        let tree_svo = SparseVoxelOctree::build(&node, &cfg);
        let comp_svo = SparseVoxelOctree::build_compiled(&compiled, &cfg);
        for p in boundary_points() {
            let t = tree_svo.query_point(p);
            let c = comp_svo.query_point(p);
            // the stored distances must agree bit-for-bit, signed zero included:
            // the serialized chunk writes `distance.to_le_bytes()`, so a ±0 split
            // would make the persisted bytes diverge across architectures.
            assert_eq!(
                t.to_bits(),
                c.to_bits(),
                "{name} @ {p:?}: tree vs bytecode SVO distance"
            );
            compared += 1;
        }
    }
    assert!(compared > 0, "SVO comparison was vacuous");
}

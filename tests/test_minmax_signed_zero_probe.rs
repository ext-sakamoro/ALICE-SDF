//! Step-1 probe: does the tree evaluator (Rust std max/min) and the JIT
//! (cranelift fmax/fmin) ever disagree bit-for-bit, and is the disagreement a
//! signed-zero split, on ops that can feed -0 into a min/max argument
//! (subtract = max(a,-b); a point on b's surface makes -b = -0).
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "jit")]
mod common;
use alice_sdf::compiled::jit::JitCompiledSdf;
use alice_sdf::prelude::*;

fn scenes() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("subtract_box_box", SdfNode::box3d(1.0, 1.0, 1.0).subtract(SdfNode::box3d(0.5, 2.0, 2.0))),
        ("intersect_box_box", SdfNode::box3d(1.0, 1.0, 1.0).intersection(SdfNode::box3d(0.5, 2.0, 2.0))),
        ("union_box_box", SdfNode::box3d(0.5, 0.5, 0.5).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.0, 0.0, 0.0))),
        ("round_subtract", SdfNode::box3d(1.0, 1.0, 1.0).subtract(SdfNode::box3d(0.5, 2.0, 2.0)).round(0.1)),
        ("smooth_union_box", SdfNode::box3d(0.5, 0.5, 0.5).smooth_union(SdfNode::box3d(0.5, 0.5, 0.5).translate(0.8, 0.0, 0.0), 0.2)),
    ]
}

#[test]
fn tree_vs_jit_signed_zero_probe() {
    // grid hits exact face/boundary coords (0.5, 1.0, 0.0) where a sub-distance is +0 -> -0 into max
    let mut coords = vec![];
    let mut x = -1.5f32;
    while x <= 1.5001 { coords.push(x); x += 0.25; }        // includes 0.5,1.0,0.0,-0.5,...
    let pts: Vec<Vec3> = {
        let mut v = vec![];
        for &a in &coords { for &b in &coords { for &c in &coords { v.push(Vec3::new(a,b,c)); } } }
        v
    };
    let mut any_mismatch = 0usize;
    let mut signed_zero_only = 0usize;
    let mut examples: Vec<String> = vec![];
    for (name, node) in scenes() {
        let jit = JitCompiledSdf::compile(&node).expect("jit compile");
        for p in &pts {
            let t = eval(&node, *p);
            let j = jit.eval(*p);
            if t.to_bits() != j.to_bits() {
                any_mismatch += 1;
                let sz = (t == 0.0 && j == 0.0) && (t.to_bits() != j.to_bits()); // ±0 split
                if sz { signed_zero_only += 1; }
                if examples.len() < 8 {
                    examples.push(format!("{name} @ {p:?}: tree={:08x} jit={:08x}{}", t.to_bits(), j.to_bits(), if sz {" [±0 split]"} else {""}));
                }
            }
        }
    }
    eprintln!("probe: pts/scene={}, scenes={}, any_to_bits_mismatch={any_mismatch}, signed_zero_split={signed_zero_only}",
              pts.len(), scenes().len());
    for e in &examples { eprintln!("  {e}"); }
    // report-only: never fail; this test exists to MEASURE, not gate (step 1)
}

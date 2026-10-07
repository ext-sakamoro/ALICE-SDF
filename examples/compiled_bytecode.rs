//! Inspecting compiled SDF bytecode and its CPU evaluators.
//!
//! Compiles a small scene, prints a disassembly (opcode, role, whether it
//! warps the point / post-processes the distance, where its subtree ends),
//! the compile-time facts of `CompiledSdf` and `CompiledSdfBvh`, refits the
//! BVH boxes after editing a radius in place, and evaluates through the batch,
//! distance-and-normal and 8-wide gradient entry points. Every printed value
//! is checked against a closed form or the scalar evaluator.
//!
//! # Running
//! ```bash
//! cargo run --release --example compiled_bytecode
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::real::Vec3R;
use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch, eval_compiled_distance_and_normal, eval_gradient_simd,
    get_scene_aabb, AabbPacked, CompiledSdfBvh, Instruction, OpCode, Vec3x8,
};
use alice_sdf::prelude::*;
use wide::f32x8;

fn main() {
    println!("ALICE-SDF — compiled bytecode");
    println!("=============================");

    let scene = SdfNode::sphere(0.6)
        .translate(0.3, 0.0, 0.0)
        .smooth_union(SdfNode::box3d(0.8, 0.4, 0.4).round(0.05), 0.1)
        .scale(1.5);
    let compiled = CompiledSdf::compile(&scene);

    // --- Disassembly ---
    println!(
        "\n{:>3}  {:<18} {:<10} warp  post  next",
        "pc", "opcode", "role"
    );
    for (pc, inst) in compiled.instructions().iter().enumerate() {
        let op = inst.opcode;
        let role = if op.is_primitive() {
            "primitive"
        } else if op.is_binary_op() {
            "binary"
        } else if op.is_transform() {
            "transform"
        } else if op.is_modifier() {
            "modifier"
        } else {
            "control"
        };
        println!(
            "{pc:>3}  {:<18} {role:<10} {:<5} {:<5} {}",
            format!("{op:?}"),
            op.modifies_point(),
            op.is_post_process(),
            inst.next_instruction_index(pc)
        );
        assert_eq!(inst.is_leaf(), op.is_primitive() || op.is_binary_op());
    }
    // Scale warps the point and rescales the distance; Round only offsets it.
    assert!(OpCode::Scale.modifies_point() && OpCode::Scale.is_post_process());
    assert!(!OpCode::Round.modifies_point() && OpCode::Round.is_post_process());
    // The outermost frame (Scale at pc 0) skips to the End instruction.
    assert_eq!(
        compiled.instructions()[0].next_instruction_index(0),
        compiled.instruction_count() - 1
    );
    // `Animated` frames are never emitted by the compiler but exist in the ISA.
    let animated = Instruction::animated(1.0, 0.5);
    assert!(animated.opcode.is_modifier() && !animated.opcode.modifies_point());

    // --- Compile-time facts ---
    let inst_size = std::mem::size_of::<Instruction>();
    println!(
        "\nnodes {}  instructions {}  aux {}  bytes {}  lipschitz {}",
        compiled.node_count(),
        compiled.instruction_count(),
        compiled.aux_data().len(),
        compiled.memory_size(),
        compiled.lipschitz()
    );
    assert_eq!(compiled.node_count(), scene.node_count() as usize);
    assert_eq!(
        compiled.memory_size(),
        compiled.instruction_count() * inst_size
    );

    // --- BVH boxes and refit ---
    let mut bvh = CompiledSdfBvh::compile(&SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0));
    let before = get_scene_aabb(&bvh);
    bvh.instructions[1].params[0] = 0.5; // the sphere radius
    let refitted = bvh.refit_all_from_bytecode().expect("refit");
    let after = get_scene_aabb(&bvh);
    println!(
        "\nBVH: {} instructions, {} bytes, refitted {refitted} boxes",
        bvh.instruction_count(),
        bvh.memory_size()
    );
    println!(
        "  box before [{:?} .. {:?}]  after [{:?} .. {:?}]",
        before.min(),
        before.max(),
        after.min(),
        after.max()
    );
    assert_eq!(after.min(), Vec3::new(1.5, -0.5, -0.5));
    assert_eq!(after.max(), Vec3::new(2.5, 0.5, 0.5));
    let probe = Vec3::new(4.0, 0.0, 0.0);
    // Euclidean and L∞ distances to the box (equal along an axis).
    assert_eq!(after.distance_to_point(probe), 1.5);
    assert_eq!(after.distance_to_point_fast(probe), 1.5);
    let corner = AabbPacked::from_half_size(Vec3::ONE);
    let d_fast = corner.distance_to_point_fast(Vec3::new(2.0, 3.0, 1.0));
    println!("  L∞ distance from (2,3,1) to the unit cube: {d_fast}");
    assert_eq!(d_fast, 2.0);

    // --- Evaluators ---
    let sphere = CompiledSdf::compile(&SdfNode::sphere(1.0));
    let pts: Vec<Vec3> = (0..8)
        .map(|i| Vec3::new(1.0 + i as f32 * 0.25, 0.5, -0.25))
        .collect();
    let batch = eval_compiled_batch(&sphere, &pts);
    for (d, p) in batch.iter().zip(&pts) {
        assert_eq!(*d, eval_compiled(&sphere, *p));
    }
    let e = 1e-3;
    let (d, n) = eval_compiled_distance_and_normal(&sphere, pts[0], e);
    let l = pts[0].length();
    println!("\ndistance+normal at {:?}: {d:.6} {n:?}", pts[0]);
    // average of the 4 tetrahedral samples = |p| - r + e²/|p| + O(e⁴)
    assert!((d - (l - 1.0 + e * e / l)).abs() < 1e-5);
    assert!((n - pts[0] / l).length() < 1e-3);

    let lanes = Vec3x8 {
        x: f32x8::new(std::array::from_fn(|i| pts[i].x)),
        y: f32x8::new(std::array::from_fn(|i| pts[i].y)),
        z: f32x8::new(std::array::from_fn(|i| pts[i].z)),
    };
    let (gx, gy, gz) = eval_gradient_simd(&sphere, lanes, e);
    let (gx, gy, gz) = (gx.to_array(), gy.to_array(), gz.to_array());
    for (i, p) in pts.iter().enumerate() {
        let g = Vec3::new(gx[i], gy[i], gz[i]).normalize();
        assert!((g - p.normalize()).length() < 1e-3);
    }
    println!("8-wide gradient: all lanes along p/|p|");

    // --- Vec3R helpers (the lane type the laws are written in) ---
    let v: Vec3R<f32> = Vec3R::new(2.5, -1.5, 0.4);
    let r = v.round();
    println!("Vec3R round {:?} max {}", (r.x, r.y, r.z), v.max_element());
    assert_eq!((r.x, r.y, r.z), (3.0, -2.0, 0.0)); // ties away from zero
    assert_eq!(v.max_element(), 2.5);

    println!("\nall checks passed");
}

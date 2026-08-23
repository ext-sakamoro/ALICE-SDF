//! Incremental parameter update via `ParamDependencyIndex`.
//!
//! Demonstrates the binding layer that maps `ParamId` (from
//! `ConstraintSolver`) to `Instruction.params[slot]` inside a `CompiledSdf`,
//! so that a solved / animated parameter can be pushed into the bytecode
//! in place without recompiling the whole tree.
//!
//! # Running
//! ```bash
//! cargo run --example incremental_param_index
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::cache::{ChunkedCacheConfig, ChunkedMeshCache};
use alice_sdf::compiled::OpCode;
use alice_sdf::prelude::*;
use glam::Vec3;

fn first_of(compiled: &CompiledSdf, opcode: OpCode) -> usize {
    compiled
        .instructions
        .iter()
        .position(|i| i.opcode == opcode)
        .expect("opcode not present in compiled scene")
}

fn main() {
    // Scene: Union(Sphere(r=1.0), Translate(2,0,0) . Box(hx=0.5, hy=0.5, hz=0.5))
    let sphere = SdfNode::sphere(1.0);
    let cube = SdfNode::box3d(0.5, 0.5, 0.5).translate(2.0, 0.0, 0.0);
    let scene = sphere.union(cube);
    let mut compiled = CompiledSdf::compile(&scene);
    let mut bvh = CompiledSdfBvh::compile(&scene);

    let sphere_idx = first_of(&compiled, OpCode::Sphere);
    let box_idx = first_of(&compiled, OpCode::Box3d);

    // Bind ParamId → instruction slots.
    // `radius_param` drives the sphere radius and the box width (shared).
    // `height_param` drives the box height alone.
    let mut index = ParamDependencyIndex::new();
    let radius_param = ParamId::from_raw(0);
    let height_param = ParamId::from_raw(1);
    index
        .bind(radius_param, &compiled, sphere_idx, 0)
        .expect("bind sphere radius");
    index
        .bind(radius_param, &compiled, box_idx, 0)
        .expect("bind box hx (shared with radius)");
    index
        .bind(height_param, &compiled, box_idx, 1)
        .expect("bind box hy");

    println!("ALICE-SDF — Incremental Parameter Index");
    println!("=======================================");
    println!(
        "bindings: {} params, {} slots",
        index.param_count(),
        index.binding_count()
    );

    // Frame 0: initial state.
    let d0 = eval_compiled(&compiled, Vec3::ZERO);
    println!(
        "\nframe 0: sphere.r=1.00, box.hx=0.50, box.hy=0.50 → d(origin)={:.4}",
        d0
    );

    // Frame 1: shrink shared radius via ConstraintSolver, then push to bytecode.
    let solver = ConstraintSolver::new(vec![0.6, 0.5]);
    index
        .apply_all(&mut compiled, &solver)
        .expect("apply_all should succeed");
    let d1 = eval_compiled(&compiled, Vec3::ZERO);
    println!(
        "frame 1: apply_all(solver=[0.6, 0.5]) → d(origin)={:.4}, dirty_params={}",
        d1,
        index.dirty_params().count()
    );

    // Enumerate dirty instructions — the caller can feed this into
    // `ChunkedMeshCache::invalidate_region()` or a BVH refit routine.
    let dirty: Vec<usize> = index.dirty_instructions().collect();
    println!("dirty instruction indices: {:?}", dirty);

    // Frame 2: bump just the height param.
    index.mark_clean();
    index
        .apply(&mut compiled, height_param, 1.25)
        .expect("apply height");
    let d2 = eval_compiled(&compiled, Vec3::new(2.0, 0.0, 0.0));
    println!(
        "\nframe 2: apply(height=1.25) → d(box_center)={:.4}, dirty_params={}",
        d2,
        index.dirty_params().count()
    );
    println!(
        "dirty instruction indices (after mark_clean + 1 apply): {:?}",
        index.dirty_instructions().collect::<Vec<_>>()
    );

    // Frame 3: query per-param AABB and invalidate a chunked mesh cache.
    let radius_aabb = index
        .affected_aabb(radius_param, &bvh)
        .expect("radius param must cover a real AABB");
    println!(
        "\naffected_aabb(radius_param) = min={:?} max={:?}",
        radius_aabb.min(),
        radius_aabb.max()
    );

    let dirty_aabb = index.dirty_aabb(&bvh).expect("height_param is still dirty");
    println!(
        "dirty_aabb() (only height_param dirty) = min={:?} max={:?}",
        dirty_aabb.min(),
        dirty_aabb.max()
    );

    let cache = ChunkedMeshCache::new(ChunkedCacheConfig::default());
    let invalidated = index.invalidate_chunked_cache(&cache, &bvh);
    println!(
        "invalidate_chunked_cache() → aabb={:?}, dirty_now={}",
        invalidated.is_some(),
        index.is_dirty()
    );

    // Frame 4: SdfNode-first update + refit_bvh (case B.4 wrapper).
    // The apply()-based path above only touches bytecode `params[]`; the BVH
    // AABBs stay frozen at compile time. When the caller drives updates by
    // mutating the SdfNode instead, `refit_bvh` recompiles the BVH so
    // downstream pruning sees the fresh geometry.
    let grown_sphere = SdfNode::sphere(2.5);
    let grown_box = SdfNode::box3d(0.5, 1.25, 0.5).translate(2.0, 0.0, 0.0);
    let grown_scene = grown_sphere.union(grown_box);
    let old_sphere_max = bvh.aabbs[sphere_idx].max();
    index
        .refit_bvh(&mut bvh, &grown_scene)
        .expect("refit must succeed on a valid scene");
    let new_sphere_max = bvh.aabbs[sphere_idx].max();
    println!(
        "\nframe 4: refit_bvh(grown_scene) → sphere AABB max: {:?} → {:?}",
        old_sphere_max, new_sphere_max
    );

    // Frame 5: apply → refit_bvh_partial (case B P1+P2 bytecode-driven).
    // Reset the workflow with a fresh scene, mutate the bytecode via apply,
    // then use the bytecode-driven partial refit so `bvh.aabbs` reflect the
    // mutated `Instruction.params[]` without needing an updated SdfNode.
    let fresh_scene = SdfNode::sphere(1.0).union(SdfNode::sphere(0.5).translate(3.0, 0.0, 0.0));
    let mut compiled2 = CompiledSdf::compile(&fresh_scene);
    let mut bvh2 = CompiledSdfBvh::compile(&fresh_scene);
    let mut index2 = ParamDependencyIndex::new();
    let s2_idx = first_of(&compiled2, OpCode::Sphere);
    let r2 = ParamId::from_raw(0);
    index2.bind(r2, &compiled2, s2_idx, 0).unwrap();

    let old_max = bvh2.aabbs[s2_idx].max();
    index2.apply(&mut compiled2, r2, 3.0).unwrap();
    let refit_count = index2
        .refit_bvh_partial(&mut bvh2, &compiled2)
        .expect("partial refit must succeed");
    let new_max = bvh2.aabbs[s2_idx].max();
    println!(
        "\nframe 5: apply(r=3.0) + refit_bvh_partial(bvh, compiled) → sphere AABB max: {:?} → {:?}, {} instructions recomputed",
        old_max, new_max, refit_count
    );
}

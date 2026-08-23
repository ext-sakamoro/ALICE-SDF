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
    let bvh = CompiledSdfBvh::compile(&scene);

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
}

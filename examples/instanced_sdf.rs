//! One compiled shape evaluated at many instance transforms.
//!
//! Builds an `InstancedSdf` of 20 rotated / scaled / translated copies of a
//! rounded box, evaluates the minimum distance with the scalar and the 8-wide
//! SoA paths (single points and batches), the per-instance distances, and
//! checks them against the union of the equivalent `SdfNode` trees. With the
//! `gpu` feature it also prints the size of the generated WGSL compute shader.
//!
//! # Running
//! ```bash
//! cargo run --release --example instanced_sdf
//! cargo run --release --example instanced_sdf --features gpu
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::animation::AnimationParams;
use alice_sdf::compiled::InstancedSdf;
use alice_sdf::prelude::*;
use glam::{EulerRot, Quat};

fn main() {
    println!("ALICE-SDF — instanced SDF");
    println!("=========================");

    let base = SdfNode::box3d(0.4, 0.2, 0.3).round(0.05);
    let params: Vec<AnimationParams> = (0..19)
        .map(|i| {
            let a = i as f32 * 0.33;
            AnimationParams {
                translate_x: a.cos() * 3.0,
                translate_y: (a * 2.0).sin() * 0.5,
                translate_z: a.sin() * 3.0,
                rotate_y: a,
                rotate_z: 0.2 * a,
                scale: 0.8 + 0.02 * i as f32,
                ..Default::default()
            }
        })
        .collect();

    let mut inst = InstancedSdf::with_capacity(CompiledSdf::compile(&base), params.len() + 1);
    for p in &params {
        inst.add_instance(*p);
    }
    inst.add_at(0.0, 2.0, 0.0); // one more, translation only
    println!("instances: {}", inst.instance_count());
    assert_eq!(inst.instance_count(), params.len() + 1);

    // The equivalent trees: scale, then XYZ Euler rotation, then translation.
    let mut trees: Vec<SdfNode> = params
        .iter()
        .map(|a| {
            base.clone()
                .scale(a.scale)
                .rotate(Quat::from_euler(
                    EulerRot::XYZ,
                    a.rotate_x,
                    a.rotate_y,
                    a.rotate_z,
                ))
                .translate(a.translate_x, a.translate_y, a.translate_z)
        })
        .collect();
    trees.push(base.clone().translate(0.0, 2.0, 0.0));

    let queries: Vec<Vec3> = (0..300)
        .map(|i| {
            let t = i as f32 * 0.1;
            Vec3::new(t.cos() * 3.2, (t * 0.7).sin(), t.sin() * 2.9)
        })
        .collect();
    let mut worst = 0.0f32;
    for q in &queries {
        let union = trees.iter().map(|t| eval(t, *q)).fold(f32::MAX, f32::min);
        let scalar = inst.eval_min(*q);
        let simd = inst.eval_min_simd(*q);
        worst = worst.max((scalar - union).abs()).max((simd - union).abs());
        let per = inst.eval_per_instance(*q);
        assert_eq!(per.len(), trees.len());
        assert_eq!(per.iter().copied().fold(f32::MAX, f32::min), scalar);
    }
    println!(
        "max |instanced - tree union| over {} points: {worst:e}",
        queries.len()
    );
    assert!(worst < 1e-4);

    let batch = inst.eval_min_batch(&queries);
    let batch_simd = inst.eval_min_batch_simd(&queries);
    for (k, q) in queries.iter().enumerate() {
        assert_eq!(batch[k], inst.eval_min(*q));
        assert_eq!(batch_simd[k], inst.eval_min_simd(*q));
    }
    println!(
        "batches: {} scalar + {} SIMD distances",
        batch.len(),
        batch_simd.len()
    );

    #[cfg(feature = "gpu")]
    {
        let wgsl = InstancedSdf::to_instanced_wgsl(&base);
        println!(
            "instanced WGSL compute shader: {} lines",
            wgsl.lines().count()
        );
        assert!(wgsl.contains("@compute"));
    }

    println!("\nall checks passed");
}

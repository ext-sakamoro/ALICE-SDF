//! Ray casting with the JIT-compiled evaluators (scalar and 8-wide SIMD).
//!
//! Compiles a sphere to native code with Cranelift, renders a depth image
//! with both JIT renderers, marches single rays, a parallel batch and an
//! 8-ray packet, and checks every result against the closed-form
//! ray–sphere intersection `t = −b − √(b² − c)`.
//!
//! # Running
//! ```bash
//! cargo run --release --example raycast_jit --features jit
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::jit::{JitCompiledSdf, JitSimdSdf};
use alice_sdf::compiled::Vec3x8;
use alice_sdf::prelude::*;
use alice_sdf::raycast::{
    raymarch_jit, raymarch_jit_batch_parallel, raymarch_jit_simd_8, raymarch_jit_with_config,
    render_depth_jit, render_depth_jit_simd,
};

/// Closed-form first hit on the unit sphere, `None` on a miss.
fn sphere_hit(o: Vec3, d: Vec3) -> Option<f32> {
    let d = d.normalize();
    let (b, c) = (o.dot(d), o.length_squared() - 1.0);
    let disc = b * b - c;
    (disc >= 0.0)
        .then(|| -b - disc.sqrt())
        .filter(|t| *t >= 0.0)
}

fn main() {
    println!("ALICE-SDF — JIT ray casting");
    println!("===========================");

    let node = SdfNode::sphere(1.0);
    let compiled = CompiledSdf::compile(&node);
    let jit = JitCompiledSdf::compile(&node).expect("JIT compile");
    let simd = JitSimdSdf::compile(&compiled).expect("JIT SIMD compile");
    let eye = Vec3::new(0.0, 0.0, 4.0);
    let (w, h, fov, far) = (32, 16, 0.9, 50.0);

    let depth = render_depth_jit(&jit, eye, Vec3::NEG_Z, Vec3::Y, w, h, fov, far);
    let depth_s =
        render_depth_jit_simd(&simd, &compiled, eye, Vec3::NEG_Z, Vec3::Y, w, h, fov, far);
    for row in depth.chunks(w) {
        let line: String = row
            .iter()
            .map(|&d| if d == f32::MAX { '.' } else { '#' })
            .collect();
        println!("{line}");
    }
    let centre = (h / 2) * w + w / 2;
    println!(
        "centre depth: scalar {} / simd {} (closed form 3)",
        depth[centre], depth_s[centre]
    );
    assert!((depth[centre] - 3.0).abs() < 1e-3 && (depth_s[centre] - 3.0).abs() < 1e-3);
    let both = depth
        .iter()
        .zip(&depth_s)
        .filter(|(a, b)| **a != f32::MAX && **b != f32::MAX);
    assert!(both.clone().count() > 0);
    assert!(both.map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max) < 1e-3);

    // Slopes kept away from the silhouette (tan(asin(1/4)) ≈ 0.258): four
    // rays hit, four miss.
    let xs = [-0.5_f32, -0.35, -0.15, -0.05, 0.05, 0.15, 0.35, 0.5];
    let rays: Vec<Ray> = xs
        .iter()
        .map(|&x| Ray::new(eye, Vec3::new(x, 0.05, -1.0)))
        .collect();
    let batch = raymarch_jit_batch_parallel(&jit, &rays, far);
    let packet = raymarch_jit_simd_8(
        &simd,
        &compiled,
        Vec3x8::splat(eye),
        Vec3x8::from_vecs(std::array::from_fn(|i| rays[i].direction)),
        far,
        &RaymarchConfig::default(),
    );
    let mut compared = 0;
    for (i, ray) in rays.iter().enumerate() {
        let expected = sphere_hit(ray.origin, ray.direction);
        let got = [
            raymarch_jit(&jit, ray.origin, ray.direction, far).map(|h| h.distance),
            raymarch_jit_with_config(
                &jit,
                ray.origin,
                ray.direction,
                far,
                &RaymarchConfig::fast(),
            )
            .map(|h| h.distance),
            batch[i].map(|h| h.distance),
            packet[i].map(|p| p.0),
        ];
        println!("ray {i}: closed form {expected:?}, jit {got:?}");
        for g in got {
            match (g, expected) {
                (Some(a), Some(b)) => assert!((a - b).abs() < 2e-3),
                (None, None) => {}
                _ => panic!("ray {i}: {g:?} vs {expected:?}"),
            }
            compared += 1;
        }
    }
    assert!(compared == 32);
    println!("all checks passed");
}

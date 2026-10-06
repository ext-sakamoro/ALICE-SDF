//! Ray casting and offline rendering helpers.
//!
//! Renders a depth image and a normal image of a sphere resting on a ground
//! slab with the tree interpreter, the compiled VM and the 8-wide SIMD
//! marcher, then shades one pixel with hard / soft shadows and ambient
//! occlusion. The centre pixel's depth and normal are checked against the
//! closed-form ray–sphere intersection, and all backends against each other.
//!
//! # Running
//! ```bash
//! cargo run --release --example raycast_render
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::Vec3x8;
use alice_sdf::prelude::*;
use alice_sdf::raycast::{
    ambient_occlusion_compiled, hard_shadow, hard_shadow_compiled, raymarch_batch,
    raymarch_compiled, raymarch_compiled_batch_parallel, raymarch_compiled_with_config,
    raymarch_detailed, raymarch_relaxed, raymarch_simd_8, render_depth, render_depth_compiled,
    render_depth_compiled_simd, render_normals, render_normals_compiled, soft_shadow_compiled,
};

fn main() {
    println!("ALICE-SDF — ray casting");
    println!("=======================");

    // Unit sphere at the origin on a ground slab whose top is y = −1.
    let scene =
        SdfNode::sphere(1.0).union(SdfNode::box3d(20.0, 1.0, 20.0).translate(0.0, -1.5, 0.0));
    let compiled = CompiledSdf::compile(&scene);
    let (eye, fwd, up) = (Vec3::new(0.0, 0.0, 4.0), Vec3::NEG_Z, Vec3::Y);
    let (w, h, fov, far) = (48, 24, 0.9, 50.0);

    let depth = render_depth(&scene, eye, fwd, up, w, h, fov, far);
    let depth_c = render_depth_compiled(&compiled, eye, fwd, up, w, h, fov, far);
    let depth_s = render_depth_compiled_simd(&compiled, eye, fwd, up, w, h, fov, far);
    let normals = render_normals(&scene, eye, fwd, up, w, h, fov, far);
    let normals_c = render_normals_compiled(&compiled, eye, fwd, up, w, h, fov, far);

    let shades = b" .:-=+*#%@";
    for row in depth.chunks(w) {
        let line: String = row
            .iter()
            .map(|&d| {
                if d == f32::MAX {
                    ' '
                } else {
                    let k = ((1.0 - ((d - 2.5) / 6.0).clamp(0.0, 1.0)) * 9.0) as usize;
                    shades[k] as char
                }
            })
            .collect();
        println!("|{line}|");
    }

    // Centre pixel (u = v = 0) looks straight down −Z and hits the sphere at
    // z = 1: depth 3, normal +Z → colour (127, 127, 255).
    let centre = (h / 2) * w + w / 2;
    println!(
        "centre: depth {} / {} / {}, normal rgb {:?}",
        depth[centre], depth_c[centre], depth_s[centre], normals[centre]
    );
    for d in [depth[centre], depth_c[centre], depth_s[centre]] {
        assert!((d - 3.0).abs() < 1e-3);
    }
    assert_eq!(normals[centre][2], 255);
    assert!((normals[centre][0] as i32 - 127).abs() <= 1);
    let max_diff = depth
        .iter()
        .zip(&depth_c)
        .zip(&depth_s)
        .filter(|((a, _), _)| **a != f32::MAX)
        .map(|((a, b), c)| (a - b).abs().max((a - c).abs()))
        .fold(0.0_f32, f32::max);
    println!("max depth difference between backends: {max_diff:.2e}");
    assert!(max_diff < 1e-3);
    assert_eq!(normals.len(), normals_c.len());

    // Single rays through the same point with every marcher.
    let dir = Vec3::NEG_Z;
    let configs = [
        RaymarchConfig::fast(),
        RaymarchConfig::high_quality(),
        RaymarchConfig::relaxed(&scene),
    ];
    let mut ts = vec![
        raymarch_relaxed(&scene, eye, dir, far).map(|h| h.distance),
        raymarch_compiled(&compiled, eye, dir, far).map(|h| h.distance),
    ];
    for c in &configs {
        ts.push(raymarch_compiled_with_config(&compiled, eye, dir, far, c).map(|h| h.distance));
    }
    let detail: RaymarchResult =
        raymarch_detailed(&scene, eye, dir, far, &RaymarchConfig::default());
    println!(
        "detailed: hit={} t={:.4} steps={}",
        detail.hit, detail.distance, detail.steps
    );
    ts.push(detail.hit.then_some(detail.distance));
    let rays: Vec<Ray> = (0..8)
        .map(|i| Ray::new(eye, Vec3::new(i as f32 * 0.01, 0.0, -1.0)))
        .collect();
    ts.push(raymarch_batch(&scene, &rays, far)[0].map(|h| h.distance));
    ts.push(raycast_batch(&scene, &rays, far)[0].map(|h| h.distance));
    ts.push(raymarch_compiled_batch_parallel(&compiled, &rays, far)[0].map(|h| h.distance));
    let packet = raymarch_simd_8(
        &compiled,
        Vec3x8::splat(eye),
        Vec3x8::from_vecs(std::array::from_fn(|i| rays[i].direction)),
        far,
        &RaymarchConfig::default(),
    );
    ts.push(packet[0].map(|p| p.0));
    for t in &ts {
        assert!((t.expect("hit") - 3.0).abs() < 1e-3, "{ts:?}");
    }

    // Shading a ground point next to the sphere, lit from straight above.
    let ground = Vec3::new(1.5, -1.0, 0.0);
    let under = Vec3::new(0.0, -1.0, 2.0); // in front, clear sky
    let lit = !hard_shadow(&scene, ground, Vec3::Y, 0.01, 20.0);
    let shadowed = hard_shadow_compiled(&compiled, Vec3::new(0.5, -1.0, 0.0), Vec3::Y, 0.01, 20.0);
    let soft = soft_shadow(
        &scene,
        ground,
        Vec3::new(-0.5, 1.0, 0.0).normalize(),
        0.01,
        20.0,
        8.0,
    );
    let soft_c = soft_shadow_compiled(
        &compiled,
        ground,
        Vec3::new(-0.5, 1.0, 0.0).normalize(),
        0.01,
        20.0,
        8.0,
    );
    let ao_open = ambient_occlusion(&scene, under, Vec3::Y, 6, 0.9);
    let ao_near = ambient_occlusion_compiled(&compiled, ground, Vec3::Y, 6, 0.9);
    println!("hard: lit={lit} shadowed={shadowed}; soft={soft:.3}/{soft_c:.3}; ao open={ao_open:.3} near sphere={ao_near:.3}");
    assert!(lit && shadowed);
    assert!((soft - soft_c).abs() < 1e-5);
    assert!((ao_open - 1.0).abs() < 1e-5, "open ground has no occlusion");
    assert!(ao_near < 1.0, "the sphere occludes the point beside it");

    println!("all checks passed");
}

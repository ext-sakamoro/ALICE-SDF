//! VRChat Mochi golden — the Mochi sample scene evaluated by the canonical
//! CPU evaluator on a fixed grid.
//!
//! `vrchat-package/Samples~/SDF Gallery/SampleMochi/SampleMochi_Collider.cs`
//! hand-ports `SmoothUnion(ground, SmoothUnion(mochi_i, 0.5), 0.15)` into
//! UdonSharp for player collision. This example prints what `alice_sdf::eval`
//! says the same scene is, one `x y z d` line per grid point, and
//! `vrchat-package/HostTests~/MochiParity` compares the C# `EvaluateSdf`
//! against it (`scripts/vrchat-host-parity.sh`).
//!
//! The five spheres are the initial layout of `SampleMochi_Collider.Start`
//! (= `mochi.asdf.json`); keep the three in sync.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_mochi_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

/// (centre, radius) of each initial mochi, as spawned by the collider.
const MOCHIS: [([f32; 3], f32); 5] = [
    ([-0.6, 0.35, 0.5], 0.35),
    ([0.5, 0.30, 0.3], 0.30),
    ([0.0, 0.28, -0.4], 0.28),
    ([-0.9, 0.40, -0.2], 0.40),
    ([0.4, 0.25, -0.8], 0.25),
];

/// Blend between mochis (`blendK` / `_BlendK`).
const BLEND_K: f32 = 0.5;
/// Blend with the ground (`groundK` / `_GroundK`).
const GROUND_K: f32 = 0.15;

fn main() {
    // Same left fold as the C# loop: fold the spheres, then union with the ground.
    let mut blob: Option<SdfNode> = None;
    for ([x, y, z], r) in MOCHIS {
        let sphere = SdfNode::sphere(r).translate(x, y, z);
        blob = Some(match blob {
            None => sphere,
            Some(acc) => acc.smooth_union(sphere, BLEND_K),
        });
    }
    let scene = SdfNode::plane(Vec3::Y, 0.0).smooth_union(blob.expect("five mochis"), GROUND_K);

    // 13 x 9 x 13 grid over the play area: x, z in [-1.5, 1.5] step 0.25, y in [0, 1] step 0.125
    for iy in 0..=8 {
        for ix in 0..=12 {
            for iz in 0..=12 {
                let p = Vec3::new(
                    (ix as f32).mul_add(0.25, -1.5),
                    iy as f32 * 0.125,
                    (iz as f32).mul_add(0.25, -1.5),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

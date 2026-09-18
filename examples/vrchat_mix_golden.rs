//! VRChat Mix golden — the Mix sample at `animTime = 0` (a fractal planet:
//! sphere minus a repeated cross; a torus ring; a three-layer onion shell at
//! (16, 0, 0)) evaluated by the canonical CPU evaluator on a fixed grid;
//! `SampleMix_Collider.Evaluate` (also the shader's `map()`) is compared
//! against it by `vrchat-package/HostTests~/StaticParity`.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_mix_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

const PLANET_RADIUS: f32 = 6.0;
const HOLE_HALF: f32 = 0.8;
const REPEAT: f32 = 5.0;
const RING_MAJOR: f32 = 10.0;
const RING_MINOR: f32 = 0.3;
const ONION_RADIUS: f32 = 3.0;
const ONION_THICKNESS: f32 = 0.15;
const ONION_ORBIT: f32 = 16.0;
const SMOOTHNESS: f32 = 0.8;
const BAR_HALF: f32 = 1000.0;

fn main() {
    let cross = SdfNode::box3d(BAR_HALF * 2.0, HOLE_HALF * 2.0, HOLE_HALF * 2.0)
        .union(SdfNode::box3d(
            HOLE_HALF * 2.0,
            BAR_HALF * 2.0,
            HOLE_HALF * 2.0,
        ))
        .union(SdfNode::box3d(
            HOLE_HALF * 2.0,
            HOLE_HALF * 2.0,
            BAR_HALF * 2.0,
        ))
        .repeat_infinite(REPEAT, REPEAT, REPEAT);
    let planet = SdfNode::sphere(PLANET_RADIUS).subtract(cross);
    let ring = SdfNode::torus(RING_MAJOR, RING_MINOR);
    // Three onion layers: |.| - t applied three times, as the collider's loop
    let onion = SdfNode::sphere(ONION_RADIUS)
        .onion(ONION_THICKNESS)
        .onion(ONION_THICKNESS)
        .onion(ONION_THICKNESS)
        .translate(ONION_ORBIT, 0.0, 0.0);
    let scene = planet
        .smooth_union(ring, SMOOTHNESS)
        .smooth_union(onion, SMOOTHNESS * 0.5);

    // 17 x 9 x 13: x in [-12, 20] step 2, y in [-4, 4] step 1, z in [-12, 12] step 2
    // (even coordinates never sit on a repeat cell boundary at 2.5 + 5 n)
    for iy in 0..=8 {
        for ix in 0..=16 {
            for iz in 0..=12 {
                let p = Vec3::new(
                    (ix as f32).mul_add(2.0, -12.0),
                    (iy as f32).mul_add(1.0, -4.0),
                    (iz as f32).mul_add(2.0, -12.0),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

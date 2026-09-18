//! VRChat Cosmic golden — the Cosmic sample at `animTime = 0` (sun, planet on
//! its orbit, the ring tilted 15 degrees and twisted, the moon, six
//! asteroids) evaluated by the canonical CPU evaluator on a fixed grid;
//! `SampleCosmic_Collider.Evaluate` (which is also the shader's `map()`) is
//! compared against it by `vrchat-package/HostTests~/StaticParity`.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_cosmic_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::{Quat, Vec3};

const SUN_RADIUS: f32 = 8.0;
const PLANET_RADIUS: f32 = 2.5;
const PLANET_DISTANCE: f32 = 18.0;
const SMOOTHNESS: f32 = 1.5;
const RING_TWIST: f32 = 0.5;

fn main() {
    // t = 0: the planet at (D, 0, 0), the moon at planet + (4, 0, 0)
    let planet_pos = Vec3::new(PLANET_DISTANCE, 0.0, 0.0);
    let sun = SdfNode::sphere(SUN_RADIUS);
    let planet = SdfNode::sphere(PLANET_RADIUS).translate(planet_pos.x, planet_pos.y, planet_pos.z);
    // The shader rotates the sample point by +15 degrees about X, then twists
    // it about Y by `RING_TWIST * y`, then evaluates the torus: a Rotate node
    // evaluates its child at R^-1 p, so R = rotX(-15 degrees).
    let ring = SdfNode::torus(PLANET_RADIUS * 1.8, 0.12)
        .twist(RING_TWIST)
        .rotate(Quat::from_rotation_x(-15.0_f32.to_radians()))
        .translate(planet_pos.x, planet_pos.y, planet_pos.z);
    let moon = SdfNode::sphere(0.6).translate(planet_pos.x + 4.0, planet_pos.y, planet_pos.z);

    let belt_r = PLANET_DISTANCE * 0.75;
    let mut asteroids: Option<SdfNode> = None;
    for i in 0..6 {
        let angle = i as f32 * 1.0472;
        let y = if i % 2 == 0 { 0.5 } else { -0.5 };
        let a = SdfNode::sphere((i as f32).mul_add(0.1, 0.3)).translate(
            angle.cos() * belt_r,
            y,
            angle.sin() * belt_r,
        );
        asteroids = Some(match asteroids {
            None => a,
            Some(acc) => acc.union(a),
        });
    }

    let scene = sun
        .smooth_union(planet, SMOOTHNESS)
        .smooth_union(ring, SMOOTHNESS * 0.5)
        .smooth_union(moon, SMOOTHNESS)
        .smooth_union(asteroids.expect("six asteroids"), SMOOTHNESS * 0.3);

    // 19 x 9 x 17: x in [-10, 26] step 2, y in [-6, 6] step 1.5, z in [-16, 16] step 2
    for iy in 0..=8 {
        for ix in 0..=18 {
            for iz in 0..=16 {
                let p = Vec3::new(
                    (ix as f32).mul_add(2.0, -10.0),
                    (iy as f32).mul_add(1.5, -6.0),
                    (iz as f32).mul_add(2.0, -16.0),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

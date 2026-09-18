//! VRChat Fractal golden — the Fractal sample (a Menger-sponge box: a 100 m
//! box minus an infinitely repeated cross, the whole twisted about Y)
//! evaluated by the canonical CPU evaluator on a fixed grid;
//! `SampleFractal_Collider.Evaluate` is compared against it by
//! `vrchat-package/HostTests~/StaticParity`.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_fractal_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

const BOX_HALF: f32 = 50.0;
const HOLE_HALF: f32 = 2.0;
const REPEAT: f32 = 15.0;
const TWIST: f32 = 0.02;
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
    let scene = SdfNode::box3d(BOX_HALF * 2.0, BOX_HALF * 2.0, BOX_HALF * 2.0)
        .subtract(cross)
        .twist(TWIST);

    // 15^3: each axis in [-19.7, 18.1] step 2.7 (off the repeat cell boundaries at 7.5 + 15 n)
    for iy in 0..=14 {
        for ix in 0..=14 {
            for iz in 0..=14 {
                let p = Vec3::new(
                    (ix as f32).mul_add(2.7, -19.7),
                    (iy as f32).mul_add(2.7, -19.7),
                    (iz as f32).mul_add(2.7, -19.7),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

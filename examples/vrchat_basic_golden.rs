//! VRChat Basic golden — the Basic sample (ground plane + sphere) evaluated by
//! the canonical CPU evaluator on a fixed grid; `SampleBasic_Collider.Evaluate`
//! is compared against it by `vrchat-package/HostTests~/StaticParity`
//! (`scripts/vrchat-host-parity.sh`).
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_basic_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

fn main() {
    let scene = SdfNode::plane(Vec3::Y, 0.0).union(SdfNode::sphere(1.5).translate(0.0, 1.5, 0.0));

    // 13 x 15 x 13: x, z in [-3, 3] step 0.5, y in [0, 3.5] step 0.25
    for iy in 0..=14 {
        for ix in 0..=12 {
            for iz in 0..=12 {
                let p = Vec3::new(
                    (ix as f32).mul_add(0.5, -3.0),
                    iy as f32 * 0.25,
                    (iz as f32).mul_add(0.5, -3.0),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

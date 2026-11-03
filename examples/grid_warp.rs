//! Grid warp: pose a rest-pose shape with a grid of affine maps
//!
//! A capsule stands along y. The grid holds, at each node in the posed space, the map back to the
//! rest pose: the identity above the "joint" at y = 0 and a rotation by 60° about x below it,
//! blended over 10 cm around it. The warped field is the capsule bent at the joint; the example prints
//! the warp's computed Lipschitz bound and the field along a line through the bent part.
//!
//! Run: `cargo run --example grid_warp`

use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};

fn main() {
    let rest = SdfNode::capsule(Vec3::new(0.0, -0.3, 0.0), Vec3::new(0.0, 0.3, 0.0), 0.05);
    let (cell, dims) = (0.025f32, [9u32, 29, 21]);
    let origin = [-0.1f32, -0.4, -0.35];
    let mut maps = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for _i in 0..dims[0] {
                let y = origin[1] + j as f32 * cell;
                let _z = origin[2] + k as f32 * cell;
                // the posed lower half is the rest lower half rotated by +60° about x: the map back
                // to the rest pose rotates by the blended angle in the other direction
                let t = ((0.05 - y) / 0.1).clamp(0.0, 1.0);
                let (s, c) = alice_det_math::sin_cos(-t * std::f32::consts::FRAC_PI_3);
                maps.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, c, -s, 0.0, 0.0, s, c, 0.0]);
            }
        }
    }
    let warp = GridWarp::new(GridWarpData {
        origin,
        cell,
        dims,
        maps,
        reach: 4.0 * cell,
        floor: Vec::new(),
    })
    .expect("valid grid");
    println!("Lipschitz bound of the warp: {:.4}", warp.lipschitz());
    let posed = rest.grid_warp(warp);
    let (s, c) = alice_det_math::sin_cos(std::f32::consts::FRAC_PI_3);
    for i in 0..=6 {
        // points along the posed lower axis: the rest axis (0, -t, 0) rotated by +60° about x
        let t = 0.05 * i as f32;
        let p = Vec3::new(0.0, -t * c, -t * s);
        println!("t {t:.2}: field {:+.4}", eval(&posed, p));
    }
}

//! VRChat TerrainSculpt golden — the TerrainSculpt sample terrain after a
//! fixed set of sculpt operations, evaluated by the canonical CPU evaluator
//! on a fixed grid.
//!
//! `vrchat-package/Samples~/SDF Gallery/SampleTerrainSculpt/SampleTerrainSculpt_Collider.cs`
//! hand-ports the terrain law into UdonSharp for player collision: a ground
//! plane folded, in order, with `SmoothUnion(., sphere, 0.25)` for every
//! "add" and `SmoothSubtraction(., sphere, 0.15)` for every "dig". This
//! example prints what `alice_sdf::eval` says the same terrain is, one
//! `x y z d` line per grid point, and
//! `vrchat-package/HostTests~/TerrainSculptParity` records the same
//! operations into the C# collider and compares its `EvaluateSdf`
//! (`scripts/vrchat-host-parity.sh`).
//!
//! The operation list is the one `TerrainSculptParity/Program.cs` replays;
//! keep the two in sync.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_terrain_sculpt_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

/// (centre, radius) of each sculpt in the order recorded; a positive radius
/// adds a hill, a negative one digs a hole (the collider's `_SculptData.w`).
const SCULPTS: [([f32; 3], f32); 6] = [
    ([0.0, 0.0, 0.0], 0.3),
    ([0.3, 0.2, 0.0], 0.3),
    ([-1.0, 0.0, 0.5], -0.3),
    ([-1.2, -0.1, 0.6], -0.3),
    ([1.5, 0.1, -1.0], 0.3),
    ([0.0, 0.0, 1.5], -0.3),
];

/// SmoothUnion factor for adding (`addSmooth` / `_AddSmooth`).
const ADD_K: f32 = 0.25;
/// SmoothSubtraction factor for digging (`subSmooth` / `_SubSmooth`).
const SUB_K: f32 = 0.15;

fn main() {
    // Same left fold as the C# loop: the plane, then every sculpt in order.
    let mut terrain = SdfNode::plane(Vec3::Y, 0.0);
    for ([x, y, z], r) in SCULPTS {
        let sphere = SdfNode::sphere(r.abs()).translate(x, y, z);
        terrain = if r > 0.0 {
            terrain.smooth_union(sphere, ADD_K)
        } else {
            terrain.smooth_subtract(sphere, SUB_K)
        };
    }

    // 17 x 15 x 17 grid over the sculpted area: x, z in [-2, 2] step 0.25,
    // y in [-0.75, 1] step 0.125 (holes reach -0.4, hills 0.5)
    for iy in 0..=14 {
        for ix in 0..=16 {
            for iz in 0..=16 {
                let p = Vec3::new(
                    (ix as f32).mul_add(0.25, -2.0),
                    (iy as f32).mul_add(0.125, -0.75),
                    (iz as f32).mul_add(0.25, -2.0),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&terrain, p));
            }
        }
    }
}

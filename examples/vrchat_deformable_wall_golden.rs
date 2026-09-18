//! VRChat DeformableWall golden — the DeformableWall sample wall with a fixed
//! set of dents, evaluated by the canonical CPU evaluator on a fixed grid.
//!
//! `vrchat-package/Samples~/SDF Gallery/SampleDeformableWall/SampleDeformableWall_Collider.cs`
//! hand-ports the wall law into UdonSharp for player collision and hand
//! contact: a box (half extents 5 × 2.5 × 0.2) standing on the plane, with
//! every live dent `SmoothSubtraction(., sphere(0.35 * strength), 0.08)`
//! folded in slot order, then `min(plane, wall)`. This example prints what
//! `alice_sdf::eval` says the same wall is, one `x y z d` line per grid
//! point, and `vrchat-package/HostTests~/DeformableWallParity` loads the
//! same dents into the C# collider and compares its `EvaluateSdf`
//! (`scripts/vrchat-host-parity.sh`).
//!
//! The dent list is the one `DeformableWallParity/Program.cs` loads; keep
//! the two in sync.
//!
//! # Running
//! ```bash
//! cargo run --example vrchat_deformable_wall_golden > golden.txt
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

/// (centre, strength) of each dent in slot order; the dent radius is
/// `DENT_RADIUS * strength` (the collider's `_ImpactPoints.w`).
const DENTS: [([f32; 3], f32); 4] = [
    ([1.0, 1.2, 0.2], 1.0),
    ([-2.0, 0.8, -0.2], 0.6),
    ([3.0, 2.0, 0.2], 0.3),
    ([0.0, 3.5, 0.2], 0.05),
];

/// Full-strength dent radius (`dentRadius` / `_DentRadius`).
const DENT_RADIUS: f32 = 0.35;
/// SmoothSubtraction factor of a dent (`dentSmooth` / `_DentSmooth`).
const DENT_K: f32 = 0.08;
/// Wall half extents (`wallWidth` / `wallHeight` / `wallThickness`).
const WALL_HALF: [f32; 3] = [5.0, 2.5, 0.2];

fn main() {
    // Same left fold as the C# loop: the box, then every dent in slot order.
    let mut wall = SdfNode::box3d(WALL_HALF[0] * 2.0, WALL_HALF[1] * 2.0, WALL_HALF[2] * 2.0)
        .translate(0.0, WALL_HALF[1], 0.0);
    for ([x, y, z], strength) in DENTS {
        let dent = SdfNode::sphere(DENT_RADIUS * strength).translate(x, y, z);
        wall = wall.smooth_subtract(dent, DENT_K);
    }
    let scene = SdfNode::plane(Vec3::Y, 0.0).union(wall);

    // 25 x 12 x 17 grid around the wall: x in [-6, 6] step 0.5, y in [0, 5.5]
    // step 0.5, z in [-1, 1] step 0.125 (the dents are within 0.35 of the faces)
    for iy in 0..=11 {
        for ix in 0..=24 {
            for iz in 0..=16 {
                let p = Vec3::new(
                    (ix as f32).mul_add(0.5, -6.0),
                    iy as f32 * 0.5,
                    (iz as f32).mul_add(0.125, -1.0),
                );
                println!("{} {} {} {:.7}", p.x, p.y, p.z, eval(&scene, p));
            }
        }
    }
}

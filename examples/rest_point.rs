//! Rest point: where a colour law defined on the unposed model reads its value
//!
//! An arm part is built along its own y axis and placed into the model by a translate and a rotate
//! (modelling); a grid warp rotates the posed space back to the rest pose (a pose); a rigid copy of
//! the part, moved into a posed position, is marked as a pose (`rigid_pose`). The rest point undoes
//! the two poses and maps back through the placement, so it lands on the model's rest surface, never
//! in the part's own frame. The example prints the rest points on the CPU and the size of the WGSL
//! module that evaluates them on the GPU (`sdf_eval_rest`).
//!
//! Run: `cargo run --example rest_point --features gpu`

use alice_sdf::compiled::{TranspileMode, WgslShader};
use alice_sdf::eval::{eval, eval_rest_point};
use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};
use glam::Quat;

fn main() {
    // the part in its own frame (along y), placed into the model along x
    let part = SdfNode::capsule(Vec3::new(0.0, -0.1, 0.0), Vec3::new(0.0, 0.1, 0.0), 0.03)
        .rotate(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2))
        .translate(0.1, 0.0, 0.0);
    // the pose: the posed space is the rest space rotated by 30° about y; each node holds the map back
    let (s, c) = alice_det_math::sin_cos(-std::f32::consts::FRAC_PI_6);
    let m = [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0];
    let warp = GridWarp::new(GridWarpData {
        origin: [-0.1, -0.15, -0.25],
        cell: 0.05,
        dims: [7, 7, 11],
        maps: m.repeat(7 * 7 * 11),
        reach: 0.1,
        floor: Vec::new(),
    })
    .expect("valid grid");
    let warped = part.clone().grid_warp(warp);
    // a rigid copy of the placed part moved up by 0.2 and turned about x: a pose, not a placement
    let copy = part.rigid_pose(Quat::from_rotation_x(0.4), Vec3::new(0.0, 0.2, 0.0));
    let model = warped.union(copy);
    for p in [
        Vec3::new(0.08, 0.0, -0.05),
        Vec3::new(0.15, 0.03, -0.08),
        Vec3::new(0.1, 0.23, 0.01),
    ] {
        let r = eval_rest_point(&model, p).expect("every node is followed");
        println!(
            "p {p:.3}: field {:+.4}, rest point {r:.3} (rest field {:+.4})",
            eval(&model, p),
            eval(&warped_rest(), r)
        );
    }
    let shader = WgslShader::transpile_with_rest_point(&model, TranspileMode::Hardcoded)
        .expect("every node is followed");
    println!("WGSL with sdf_eval_rest: {} bytes", shader.source.len());
}

/// The unposed model: the placed part, where the rest points above should land near the surface
fn warped_rest() -> SdfNode {
    SdfNode::capsule(Vec3::new(0.0, -0.1, 0.0), Vec3::new(0.0, 0.1, 0.0), 0.03)
        .rotate(Quat::from_rotation_z(-std::f32::consts::FRAC_PI_2))
        .translate(0.1, 0.0, 0.0)
}

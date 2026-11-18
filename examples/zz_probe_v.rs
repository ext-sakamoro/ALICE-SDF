use alice_sdf::validity::{prove_gaps, GAP_PROOF_WORK};
use alice_sdf::SdfNode;
use glam::Vec3;
fn v_groove(alpha: f32, depth: f32) -> SdfNode {
    let slab = SdfNode::box3d(30.0, 10.0, 20.0).translate(0.0, -5.0, 0.0);
    let half = alpha.to_radians() * 0.5;
    let hs = |t: f32| SdfNode::box3d(60.0, 60.0, 40.0).translate(30.0, 0.0, 0.0).rotate_euler(0.0, 0.0, t);
    slab.subtract(hs(half).intersection(hs(std::f32::consts::PI - half)).translate(0.0, -depth, 0.0))
}
fn main() {
    let d: u32 = std::env::args().nth(1).map_or(8, |s| s.parse().unwrap());
    let p = prove_gaps(&v_groove(40.0, 3.0), 1.9, Vec3::new(-5.0, -4.0, -3.0), Vec3::new(5.0, 1.0, 3.0), d, GAP_PROOF_WORK).unwrap();
    println!("w {} u {} b {} wit {} ex {}", p.work, p.undecided, p.boxes, p.witnesses.len(), p.exhausted);
}

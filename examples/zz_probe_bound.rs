use alice_sdf::validity::{prove_gaps, GAP_PROOF_WORK};
use alice_sdf::SdfNode;
use glam::Vec3;
fn main() {
    let s = SdfNode::box3d(8.0, 2.0, 8.0).subtract(SdfNode::box3d(1.4, 2.2, 10.0));
    let p = prove_gaps(&s, 0.5, Vec3::new(-3.0, -1.6, -2.0), Vec3::new(3.0, 1.6, 2.0), 9, GAP_PROOF_WORK).unwrap();
    println!("undecided {} boxes {} work {} regions {}", p.undecided, p.boxes, p.work, p.undecided_at.len());
    for u in p.undecided_at.iter().take(12) { println!("{:?} {:?}", u.at, u.walls); }
}

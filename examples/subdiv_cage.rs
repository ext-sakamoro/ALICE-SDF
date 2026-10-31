//! Subdivision cage — Catmull–Clark subdivision of a quad cage and its limit surface
//!
//! Builds a closed cube cage with one infinite crease (an edge loop around the middle) and one
//! semi-sharp edge (sharpness 2), checks its topology (Euler characteristic, valence defect),
//! subdivides it, samples the limit surface three levels down and reads the exact limit normals
//! at the smooth vertices. Each printed value is checked against its closed form: a closed quad
//! surface of genus 0 has χ = 2 and a valence defect of 4χ = 8, and every step multiplies the
//! face count by 4. The full oracle is the `mesh::subdiv` module's tests.
//!
//! # Running
//! ```bash
//! cargo run --example subdiv_cage
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::subdiv::{check, limit_mesh, limit_normals, limit_positions, subdivide, Cage, INFINITE};
use glam::DVec3;
use std::collections::BTreeMap;

fn main() {
    // a cube split at its middle height: 12 vertices, 10 quads (top, bottom, 8 sides)
    let p = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
    let positions = vec![
        p(-1.0, -1.0, -1.0), p(1.0, -1.0, -1.0), p(1.0, 1.0, -1.0), p(-1.0, 1.0, -1.0),
        p(-1.0, -1.0, 0.0), p(1.0, -1.0, 0.0), p(1.0, 1.0, 0.0), p(-1.0, 1.0, 0.0),
        p(-1.0, -1.0, 1.0), p(1.0, -1.0, 1.0), p(1.0, 1.0, 1.0), p(-1.0, 1.0, 1.0),
    ];
    let mut faces = vec![[0, 3, 2, 1], [8, 9, 10, 11]];
    for k in 0..4u32 {
        let (a, b) = (k, (k + 1) % 4);
        faces.push([a, b, b + 4, a + 4]);
        faces.push([a + 4, b + 4, b + 8, a + 8]);
    }
    let mut sharpness = BTreeMap::new();
    // the middle loop: an infinite crease
    for k in 0..4u32 {
        let (a, b) = (4 + k, 4 + (k + 1) % 4);
        sharpness.insert((a.min(b), a.max(b)), INFINITE);
    }
    // one top edge: semi-sharp for two steps
    sharpness.insert((8, 9), 2);
    let cage = Cage { positions, faces, sharpness };

    let t = check(&cage).expect("the cage is valid");
    println!("cage: {} vertices, {} edges, {} faces, χ = {}, valence defect = {}", t.vertices, t.edges, t.faces, t.euler(), t.valence_defect());
    assert_eq!((t.euler(), t.valence_defect()), (2, 8), "a closed genus-0 quad surface: χ = 2, defect 4χ");

    let once = subdivide(&cage).expect("the cage subdivides");
    println!("one step: {} faces", once.faces.len());
    assert_eq!(once.faces.len(), 4 * cage.faces.len());

    let lp = limit_positions(&cage).expect("the cage is valid");
    let lm = limit_mesh(&cage, 3).expect("the cage is valid");
    println!("limit mesh three levels down: {} vertices, {} faces", lm.positions.len(), lm.faces.len());
    assert_eq!(lm.faces.len(), 64 * cage.faces.len());
    // the original vertices keep their indices: the limit mesh's first vertices are their limits
    let dev = lp.iter().zip(&lm.positions).map(|(a, b)| (*a - *b).length()).fold(0.0, f64::max);
    println!("the cage vertices' limits agree with the limit mesh to {dev:.1e}");
    assert!(dev < 1e-12);

    let nrm = limit_normals(&lm).expect("the limit mesh is a valid cage");
    let smooth = nrm.iter().filter(|n| n.is_some()).count();
    println!("limit normals: {smooth} of {} vertices are smooth (the crease loop has none)", nrm.len());
    assert!(smooth > 0 && smooth < nrm.len());
}

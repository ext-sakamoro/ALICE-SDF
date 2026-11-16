//! Finding zero-thickness sheets in a part: a seam where two solids only touch, and a cutter that
//! stops exactly on the material's face (a membrane over the opening)
//!
//! Run: `cargo run --example coplanar_faces`

use alice_sdf::coplanar::{
    coplanar_faces, SHEET_DELTA_FLOOR_ULPS, SHEET_DELTA_REL, SHEET_MAX_SAMPLES,
};
use alice_sdf::SdfNode;

fn main() {
    // a rib standing on a base (a union seam) and a pocket that stops on the base's top (a flush cut)
    let base = SdfNode::box3d(40.0, 4.0, 20.0);
    let rib = SdfNode::box3d(10.0, 6.0, 20.0).translate(0.0, 5.0, 0.0);
    let pocket = SdfNode::box3d(8.0, 2.0, 8.0).translate(12.0, 1.0, 0.0);
    let part = base.subtract(pocket).union(rib);

    let report = coplanar_faces(&part);
    println!(
        "classified with δ = {SHEET_DELTA_REL} x extent (floor {SHEET_DELTA_FLOOR_ULPS} ulp), at most {SHEET_MAX_SAMPLES} samples per pair"
    );
    println!("fully analysed: {}", report.is_fully_analysed());
    for p in &report.pairs {
        println!(
            "{:?} at {:?}, {} sheet region(s)",
            p.kind,
            p.point,
            p.sheets.len()
        );
        for s in &p.sheets {
            println!(
                "  {:?} sheet from {:?} to {:?} ({} samples)",
                s.side, s.min, s.max, s.samples
            );
        }
    }
    println!(
        "{} candidate(s) are real surfaces, not sheets",
        report.surfaces.len()
    );
}

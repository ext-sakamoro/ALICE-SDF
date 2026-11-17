//! Bounding the direction of a distance field over a box: next to one pin the field points one way;
//! on the plane between two pins it holds both pins' directions (the ridge between facing walls).
//! The point gradient at the box's corner is checked against the bound.
//!
//! Run: `cargo run --example interval_gradient`

use alice_sdf::eval::eval_gradient;
use alice_sdf::interval::Vec3Interval;
use alice_sdf::interval_gradient::eval_interval_gradient;
use alice_sdf::SdfNode;
use glam::Vec3;

fn main() {
    let pins = SdfNode::cylinder(0.3, 6.0)
        .translate(-0.8, 3.0, 0.0)
        .union(SdfNode::cylinder(0.3, 6.0).translate(0.8, 3.0, 0.0));
    for (what, lo, hi) in [
        (
            "beside the left pin",
            Vec3::new(-1.6, 2.9, -0.05),
            Vec3::new(-1.5, 3.0, 0.05),
        ),
        (
            "between the pins",
            Vec3::new(-0.02, 2.9, -0.02),
            Vec3::new(0.02, 3.0, 0.02),
        ),
        (
            "under a torus (no enclosure)",
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.1, 0.1, 0.1),
        ),
    ] {
        let node = if what.contains("torus") {
            SdfNode::torus(1.0, 0.3)
        } else {
            pins.clone()
        };
        let g = eval_interval_gradient(&node, Vec3Interval::from_bounds(lo, hi));
        // an unknown bound encloses nothing, so there is no point gradient to check against it
        let corner = if g.known {
            let p = eval_gradient(&node, lo);
            assert!(
                g.contains(p),
                "{what}: the bound misses the gradient {p} at {lo}"
            );
            "holds the corner's gradient"
        } else {
            "no enclosure to check"
        };
        println!(
            "{what}: known {} dx [{:.3}, {:.3}] dy [{:.3}, {:.3}] dz [{:.3}, {:.3}], {corner}",
            g.known, g.x.lo, g.x.hi, g.y.lo, g.y.hi, g.z.lo, g.z.hi
        );
    }
}

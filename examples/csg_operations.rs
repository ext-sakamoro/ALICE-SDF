//! CSG Operations — Constructive Solid Geometry examples
//!
//! Demonstrates union, intersection, subtraction, and smooth blending.
//!
//! # Running
//! ```bash
//! cargo run --example csg_operations
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

fn main() {
    let sphere = SdfNode::sphere(1.0);
    let box3d = SdfNode::box3d(0.8, 0.8, 0.8);

    // Boolean operations
    let union = sphere.clone().union(box3d.clone());
    let intersection = sphere.clone().intersection(box3d.clone());
    let subtraction = sphere.clone().subtract(box3d.clone());

    // Smooth blend (k = smoothness factor)
    let smooth = sphere.smooth_union(box3d, 0.3);

    let p = Vec3::new(1.0, 0.0, 0.0);
    println!("ALICE-SDF — CSG Operations");
    println!("==========================");
    println!("  Test point: ({}, {}, {})", p.x, p.y, p.z);
    println!();
    println!("  Union:        {:.4}", eval(&union, p));
    println!("  Intersection: {:.4}", eval(&intersection, p));
    println!("  Subtraction:  {:.4}", eval(&subtraction, p));
    println!("  Smooth Union: {:.4}", eval(&smooth, p));
}

//! Hello Sphere — Minimal ALICE-SDF example
//!
//! Creates a sphere, evaluates its distance at several points,
//! and prints the results.
//!
//! # Running
//! ```bash
//! cargo run --example hello_sphere
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use glam::Vec3;

fn main() {
    // Create a unit sphere (radius 1.0)
    let sphere = SdfNode::sphere(1.0);

    // Evaluate distances at several points
    let points = [
        Vec3::new(0.0, 0.0, 0.0), // center → -1.0
        Vec3::new(1.0, 0.0, 0.0), // surface → 0.0
        Vec3::new(2.0, 0.0, 0.0), // outside → 1.0
        Vec3::new(0.0, 0.5, 0.0), // inside → -0.5
    ];

    println!("ALICE-SDF — Hello Sphere");
    println!("========================");
    for p in &points {
        let d = eval(&sphere, *p);
        println!(
            "  point ({:5.1}, {:5.1}, {:5.1}) → distance = {:.4}",
            p.x, p.y, p.z, d
        );
    }
}

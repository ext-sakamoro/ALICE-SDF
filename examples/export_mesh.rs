//! Export Mesh — SDF to mesh generation
//!
//! Generates a mesh from an SDF using Marching Cubes and prints stats.
//!
//! # Running
//! ```bash
//! cargo run --example export_mesh
//! ```
//!
//! Author: Moroya Sakamoto

use std::fmt::Write;

use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use alice_sdf::prelude::*;
use glam::Vec3;

fn main() {
    // Create a shape: sphere with a box cut out
    let shape = SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.6, 0.6, 0.6));

    // Configure marching cubes
    let config = MarchingCubesConfig {
        resolution: 64,
        ..Default::default()
    };

    // Generate mesh
    let bounds_min = Vec3::splat(-1.5);
    let bounds_max = Vec3::splat(1.5);
    let mesh = sdf_to_mesh(&shape, bounds_min, bounds_max, &config);

    println!("ALICE-SDF — Export Mesh");
    println!("=======================");
    println!("  Vertices:  {}", mesh.vertices.len());
    println!("  Triangles: {}", mesh.indices.len() / 3);

    // Write simple OBJ manually
    let path = "output.obj";
    let mut obj = String::new();
    for v in &mesh.vertices {
        let _ = writeln!(obj, "v {} {} {}", v.position.x, v.position.y, v.position.z);
    }
    for tri in mesh.indices.chunks(3) {
        let _ = writeln!(obj, "f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1);
    }
    std::fs::write(path, obj).expect("Failed to write OBJ");
    println!("  Exported to: {path}");
}

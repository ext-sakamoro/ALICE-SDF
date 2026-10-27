//! Dual contouring with one vertex per surface sheet
//!
//! A thin offset layer over a bent limb (two capsules joined by a smooth
//! union) cut on a 4 mm grid. Where the layer crosses a grid face in a band
//! (the face's four corners alternate in sign), one vertex per cell joins
//! the two cells' vertices through all four of that face's edges, so four
//! triangles share one edge. `DcTopology::Manifold` gives each sheet in a
//! cell its own vertex and the mesh stays closed and manifold.
//!
//! Run: `cargo run --release --example dual_contouring_topology`
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::dual_contouring::{
    dual_contouring_field, dual_contouring_with_topology, DcTopology,
};
use alice_sdf::mesh::{validate_mesh, DualContouringConfig};
use alice_sdf::prelude::*;

fn main() {
    let joint = Vec3::new(0.0, 0.3, 0.0);
    let limb = SdfNode::capsule(Vec3::ZERO, joint, 0.05).smooth_union(
        SdfNode::capsule(
            joint,
            joint + Vec3::new(0.0, 1.0, 1.0).normalize() * 0.3, // bent 45°
            0.05,
        ),
        0.04,
    );
    let (lo, hi) = (Vec3::splat(-0.4), Vec3::splat(0.7));
    let cfg = DualContouringConfig {
        resolution: 275,
        ..Default::default()
    };

    // the limb itself, from the SdfNode
    let body = dual_contouring_with_topology(&limb, lo, hi, &cfg, DcTopology::Manifold);
    let v = validate_mesh(&body);
    println!(
        "limb: {} triangles, non-manifold edges {}, open edges {}",
        v.triangle_count, v.non_manifold_edges, v.boundary_edges
    );

    // a layer 2 mm above it, from a field closure
    let h = 0.002;
    let layer = |p: Vec3| eval(&limb, p) - h;
    for topology in [DcTopology::OneVertexPerCell, DcTopology::Manifold] {
        let m = dual_contouring_field(&layer, lo, hi, &cfg, topology);
        let v = validate_mesh(&m);
        println!(
            "layer {h} m, {topology:?}: {} vertices, {} triangles, non-manifold edges {}, open edges {}",
            v.vertex_count, v.triangle_count, v.non_manifold_edges, v.boundary_edges
        );
    }
}

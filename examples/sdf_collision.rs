//! Collision queries between two SDF shapes.
//!
//! Two unit spheres are moved closer step by step; at each step the example
//! asks whether they overlap, how far apart they are, and (once they touch)
//! for the contact manifold. A fast ball is then swept against a sphere with
//! continuous collision detection, and a point is projected onto a surface.
//! Every answer is checked against the closed form for spheres.
//!
//! # Running
//! ```bash
//! cargo run --example sdf_collision
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

fn main() {
    println!("ALICE-SDF — SDF collision");
    println!("=========================");

    let bounds = Aabb {
        min: Vec3::splat(-3.0),
        max: Vec3::splat(3.0),
    };
    let res = 32;
    let a = SdfNode::sphere(1.0);

    for &d in &[3.0_f32, 2.5, 1.5] {
        let b = SdfNode::sphere(1.0).translate(d, 0.0, 0.0);
        let overlap = sdf_overlap(&a, &b, &bounds, res);
        let sep = sdf_distance(&a, &b, &bounds, res);
        println!(
            "centre distance {d}: overlap={overlap}, separation≈{sep:.3} (gap {})",
            (d - 2.0).max(0.0)
        );
        assert_eq!(overlap, d < 2.0);
        assert!(sep >= (d - 2.0).max(0.0) - 1e-5);

        if overlap {
            let contacts: Vec<SdfContact> = sdf_collide(&a, &b, &bounds, res);
            let m: ContactManifold = compute_manifold(&contacts).expect("contacts");
            println!(
                "  {} contacts, centre {:.3}, normal {:.3}, max depth {:.3} (lens depth {})",
                m.count,
                m.center,
                m.normal,
                m.max_depth,
                1.0 - d / 2.0
            );
            assert!(m.max_depth <= 1.0 - d / 2.0 + 1e-6);
            assert!(m.normal.x < -0.99, "normal points from B to A");
        }
    }

    // A ball of radius 0.25 at x = −5 moving at 10 units/s hits the unit
    // sphere when its centre reaches x = −1.25: t = 3.75 / 10.
    let (toi, at) = sdf_ccd(
        &a,
        Vec3::new(-5.0, 0.0, 0.0),
        Vec3::new(10.0, 0.0, 0.0),
        1.0,
        0.25,
    )
    .expect("impact within the step");
    println!("ccd: time of impact {toi:.4} at {at:.3}");
    assert!((toi - 0.375).abs() < 1e-4);

    // Closest point on a radius-2 sphere to (3, 4, 0) is 2·(3, 4, 0)/5.
    let (p, residual) = sdf_closest_point(&SdfNode::sphere(2.0), Vec3::new(3.0, 4.0, 0.0), 32);
    println!("closest point: {p:.4} (residual {residual:.2e})");
    assert!((p - Vec3::new(1.2, 1.6, 0.0)).length() < 1e-3);

    println!("all checks passed");
}

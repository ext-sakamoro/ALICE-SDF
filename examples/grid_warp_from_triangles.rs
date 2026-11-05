//! Grid warp from a surface given at rest and posed: a bar laid on a sheet follows the sheet
//!
//! A 9×9 sheet lies flat at rest over [0, 1]² in z = 0; posed, it is bent into an arc that
//! rises to 0.3 at its middle (a pulled cloth). A thin bar lies on the rest sheet. The grid of
//! posed→rest maps built by `grid_warp_from_triangles` (each node takes the map of the nearest
//! posed triangle, `affine_posed_to_rest`) warps the bar onto the bent sheet. The example
//! checks that the bar's top surface, found by marching down a ray at a few places, sits on the
//! arc (the bar's thickness measured along the arc's normal), and prints the heights.
//!
//! Run: `cargo run --example grid_warp_from_triangles`

use alice_sdf::eval::eval;
use alice_sdf::transforms::{affine_posed_to_rest, grid_warp_from_triangles, GridWarp, WarpGrid};
use alice_sdf::SdfNode;
use glam::Vec3;

const N: usize = 9;

/// Height of the posed sheet over x (an arc: 0 at both ends, 0.3 in the middle).
fn arc(x: f32) -> f32 {
    1.2 * x * (1.0 - x)
}

fn main() {
    let mut rest = Vec::new();
    for j in 0..N {
        for i in 0..N {
            rest.push(Vec3::new(
                i as f32 / (N - 1) as f32,
                j as f32 / (N - 1) as f32,
                0.0,
            ));
        }
    }
    let posed: Vec<Vec3> = rest.iter().map(|p| Vec3::new(p.x, p.y, arc(p.x))).collect();
    let mut tris = Vec::new();
    for j in 0..N - 1 {
        for i in 0..N - 1 {
            let a = (j * N + i) as u32;
            let (b, c, d) = (a + 1, a + N as u32, a + N as u32 + 1);
            tris.push([a, b, d]);
            tris.push([a, d, c]);
        }
    }
    // every triangle has a map (the sheet has no fold)
    for t in &tris {
        let r = [
            rest[t[0] as usize],
            rest[t[1] as usize],
            rest[t[2] as usize],
        ];
        let p = [
            posed[t[0] as usize],
            posed[t[1] as usize],
            posed[t[2] as usize],
        ];
        assert!(affine_posed_to_rest(r, p).is_some());
    }
    let data = grid_warp_from_triangles(
        &rest,
        &posed,
        &tris,
        &WarpGrid {
            box_min: Vec3::new(-0.3, -0.3, -0.3),
            box_max: Vec3::new(1.3, 1.3, 0.7),
            cell: 0.025,
            reach: 0.3,
            shell: None,
        },
    )
    .expect("the sheet builds a grid");
    let warp = GridWarp::new(data).expect("a valid grid");
    println!(
        "Lipschitz bound of the warped field: {:.3}",
        warp.lipschitz()
    );
    // a bar 0.04 thick lying on the rest sheet along x, at y = 0.5
    let bar = SdfNode::box3d(0.8, 0.1, 0.04).translate(0.5, 0.5, 0.02);
    let field = bar.grid_warp(warp);
    for x in [0.2f32, 0.35, 0.5, 0.65, 0.8] {
        // march down from above to the bar's top surface
        let mut z = 0.6f32;
        let mut steps = 0;
        loop {
            let d = eval(&field, Vec3::new(x, 0.5, z));
            if d < 1e-4 || steps > 400 {
                break;
            }
            z -= d.max(1e-4);
            steps += 1;
        }
        // the bar's 0.04 is along the sheet's normal: straight up it is 0.04·sqrt(1 + slope²)
        let slope = 1.2 * (1.0 - 2.0 * x);
        let want = arc(x) + 0.04 * (1.0 + slope * slope).sqrt();
        println!("x = {x:.2}: top of the bar at z = {z:.4}, on the arc {want:.4}");
        // the posed sheet is 8 flat strips, so the arc is met to the chord error (< 0.005)
        assert!(
            (z - want).abs() < 0.005,
            "the bar does not follow the sheet at x = {x}"
        );
    }
}

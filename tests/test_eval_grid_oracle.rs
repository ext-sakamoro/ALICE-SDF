//! Batch / grid evaluators and the finite-difference gradient.
//!
//! oracle:
//! - `eval_batch` / `eval_grid` only reorganise the loop, so each value is
//!   bit-identical to `eval` at the point the documented layout names
//!   (`x + y·res + z·res²`, coordinate `min + i·(max − min)/(res − 1)`)
//! - `grid_index` / `grid_coords` are inverse bijections over `0..res³`
//! - `eval::gradient` of a sphere is p/|p|; a central difference of the exact
//!   distance |p| − r has truncation error ε²/(6|p|²)·(3 bound) plus the f32
//!   cancellation of the difference, so 1e-3 relative is a safe bound for
//!   ε = 1e-3 and |p| ≥ 0.5
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::eval::parallel::{grid_coords, grid_index};
use alice_sdf::eval::{eval, eval_batch, eval_grid, eval_grid_with_normals, gradient, normal};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn scene() -> SdfNode {
    SdfNode::sphere(0.9)
        .union(SdfNode::box3d(1.2, 0.5, 0.7).translate(0.5, -0.25, 0.0))
        .subtract(SdfNode::sphere(0.3).translate(-0.5, 0.25, 0.5))
}

#[test]
fn eval_batch_is_bit_identical_to_eval() {
    let node = scene();
    let pts: Vec<Vec3> = (0..97)
        .map(|i| {
            let t = i as f32 * 0.37;
            Vec3::new(t.sin() * 1.5, (t * 1.7).cos(), (t * 0.4).sin() * 2.0)
        })
        .collect();
    let got = eval_batch(&node, &pts);
    assert_eq!(got.len(), pts.len());
    let mut compared = 0;
    for (p, d) in pts.iter().zip(&got) {
        assert_eq!(d.to_bits(), eval(&node, *p).to_bits());
        compared += 1;
    }
    assert_eq!(compared, 97);
    assert!(eval_batch(&node, &[]).is_empty());
}

#[test]
fn eval_grid_follows_its_documented_layout() {
    let node = scene();
    let (min, max) = (Vec3::new(-1.5, -1.0, -1.25), Vec3::new(1.5, 1.0, 1.25));
    let mut compared = 0;
    for res in [2usize, 3, 9, 17] {
        let grid = eval_grid(&node, min, max, res);
        let (dists, normals) = eval_grid_with_normals(&node, min, max, res, 1e-3);
        assert_eq!(grid.len(), res * res * res);
        assert_eq!(dists, grid);
        let step = (max - min) / (res as f32 - 1.0);
        for z in 0..res {
            for y in 0..res {
                for x in 0..res {
                    let p = Vec3::new(
                        (x as f32) * step.x + min.x,
                        (y as f32) * step.y + min.y,
                        (z as f32) * step.z + min.z,
                    );
                    let i = grid_index(x, y, z, res);
                    assert_eq!(i, x + y * res + z * res * res);
                    assert_eq!(
                        grid[i].to_bits(),
                        eval(&node, p).to_bits(),
                        "res={res} {x},{y},{z}"
                    );
                    assert_eq!(normals[i], normal(&node, p, 1e-3));
                    compared += 1;
                }
            }
        }
        // the grid's corners are exactly min and max
        assert_eq!(grid[0].to_bits(), eval(&node, min).to_bits());
        assert_eq!(
            grid[res * res * res - 1].to_bits(),
            eval(&node, max).to_bits()
        );
    }
    assert_eq!(compared, 8 + 27 + 729 + 4913);
}

#[test]
fn grid_index_and_coords_are_inverse() {
    let mut compared = 0;
    for res in [1usize, 2, 5, 16] {
        // every flat index maps to a distinct in-range triple and back
        let mut seen = std::collections::HashSet::new();
        for i in 0..res * res * res {
            let (x, y, z) = grid_coords(i, res);
            assert!(x < res && y < res && z < res);
            assert_eq!(grid_index(x, y, z, res), i);
            assert!(seen.insert((x, y, z)));
            compared += 1;
        }
        assert_eq!(seen.len(), res * res * res);
    }
    assert_eq!(compared, 1 + 8 + 125 + 4096);
}

#[test]
fn finite_difference_gradient_of_a_sphere_is_radial() {
    let r = 0.75_f32;
    let sphere = SdfNode::sphere(r);
    let mut compared = 0;
    for p in [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.3, -0.8, 0.6),
        Vec3::new(-1.2, 0.4, 2.0),
        Vec3::new(0.2, 0.5, -0.1),
    ] {
        let g = gradient(&sphere, p, 1e-3);
        let want = p / p.length();
        assert!((g - want).length() < 1e-3, "p={p:?} g={g:?}");
        // not normalised: |∇| of a distance field is 1 up to the same bound
        assert!((g.length() - 1.0).abs() < 1e-3);
        compared += 1;
    }
    // Scale multiplies the child distance back by the factor, so the field
    // stays a distance and the gradient stays unit (radial)
    let doubled = SdfNode::sphere(r).scale(0.5);
    let g = gradient(&doubled, Vec3::new(0.0, 0.6, 0.0), 1e-3);
    assert!((g - Vec3::Y).length() < 1e-3, "g={g:?}");
    compared += 1;
    assert_eq!(compared, 5);
}

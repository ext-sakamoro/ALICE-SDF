//! `ExteriorField::is_exterior` near slivers.
//!
//! A sign guard, not a test of the sliver fix in `MeshBvh`: it passes before that fix too (a wrong
//! closest point on a sliver lying on the local surface keeps the walk's normal component, so the
//! sign holds). It keeps the sign from regressing near slivers.
//!
//! Inside the blocked band the classification walks away from
//! `MeshBvh::closest_point`, so a closest point taken from the wrong Voronoi
//! region of a sliver walks in the wrong direction.
//!
//! The mesh: an icosphere (closed, outward wound) in which every third
//! triangle `abc` is replaced by `(a, m, c)`, `(m, b, c)` and the sliver
//! `(a, b, m)` with `m = a + (b − a)·0.967` rounded in f32. The three keep the
//! mesh closed and consistently wound (`ab` keeps its direction, `am` / `mb`
//! each appear once in each direction), and the sliver sits on the crease
//! between `abc` and its neighbour across `ab`.
//!
//! oracle: a point at radius `ρ(1 ± δ)` along a sliver's centroid direction,
//! with `ρ` the centroid's radius and `δ` larger than the polyhedron's local
//! departure from that radius, is outside for `+δ` and inside for `−δ`. `δ`
//! is kept inside the blocked band (half a cell diagonal), so every query
//! takes the walk.

use alice_sdf::mesh::{ExteriorField, MeshBvh};
use glam::{DVec3, Vec3};
use std::collections::HashMap;

fn icosphere(levels: u32) -> (Vec<Vec3>, Vec<[u32; 3]>) {
    let mut dirs: Vec<DVec3> = vec![
        DVec3::X,
        DVec3::NEG_X,
        DVec3::Y,
        DVec3::NEG_Y,
        DVec3::Z,
        DVec3::NEG_Z,
    ];
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];
    for _ in 0..levels {
        let mut mid: HashMap<(usize, usize), usize> = HashMap::new();
        let mut midpoint = |a: usize, b: usize, dirs: &mut Vec<DVec3>| -> usize {
            let key = (a.min(b), a.max(b));
            *mid.entry(key).or_insert_with(|| {
                dirs.extend([(dirs[a] + dirs[b]).normalize()]);
                dirs.len() - 1
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(a, b, &mut dirs);
            let bc = midpoint(b, c, &mut dirs);
            let ca = midpoint(c, a, &mut dirs);
            next.extend_from_slice(&[[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]);
        }
        faces = next;
    }
    let verts = dirs.iter().map(|d| d.as_vec3()).collect();
    let faces = faces
        .iter()
        .map(|f| [f[0] as u32, f[1] as u32, f[2] as u32])
        .collect();
    (verts, faces)
}

#[test]
fn walks_near_slivers_keep_the_sign() {
    let (mut verts, faces) = icosphere(3);
    let mut idx: Vec<u32> = Vec::new();
    let mut slivers = Vec::new();
    for (i, &[a, b, c]) in faces.iter().enumerate() {
        if i % 3 != 0 {
            idx.extend_from_slice(&[a, b, c]);
            continue;
        }
        let (va, vb) = (verts[a as usize], verts[b as usize]);
        let m = verts.len() as u32;
        verts.extend([va + (vb - va) * 0.967]);
        idx.extend_from_slice(&[a, m, c, m, b, c, a, b, m]);
        slivers.extend([(a, b, m)]);
    }
    let bvh = MeshBvh::build(&verts, &idx, 4);
    let field = ExteriorField::build(&bvh, 24).expect("closed mesh builds");
    // cell ≈ 2.2 / 24 ≈ 0.09, half diagonal ≈ 0.08; the level-3 sphere departs
    // from a centroid's radius by < 0.006 within the offsets below
    let mut compared = 0;
    let mut wrong = Vec::new();
    for &(a, b, m) in &slivers {
        let (va, vb, vm) = (verts[a as usize], verts[b as usize], verts[m as usize]);
        let centroid = (va + vb + vm) / 3.0;
        let rho = centroid.length();
        let n = centroid / rho;
        let along = (vb - va).normalize();
        for delta in [0.015_f32, 0.03, 0.05] {
            for s in [-0.004_f32, 0.0, 0.004] {
                for (sign, outside) in [(1.0_f32, true), (-1.0, false)] {
                    let p = n * (rho * (1.0 + sign * delta)) + along * s;
                    compared += 1;
                    if field.is_exterior(&bvh, p) != outside {
                        wrong.extend([(p, outside)]);
                    }
                }
            }
        }
    }
    assert!(compared > 1000, "compared {compared}");
    assert!(
        wrong.is_empty(),
        "{} of {compared} near-sliver points take the wrong sign, first {:?}",
        wrong.len(),
        wrong.first()
    );
}

//! Nearest triangle of a mesh: which input triangle, where on it, and with
//! which barycentric weights (what skinning needs to carry a point along with
//! the surface).
//!
//! The mesh is the octahedron `|x| + |y| + |z| = 1`, so every answer has a
//! closed form: a point straight out from a face centre lands on that face's
//! centre with weights (1/3, 1/3, 1/3), and a point straight out from a vertex
//! lands on the vertex.
//!
//! Run: `cargo run --example mesh_closest_triangle`
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::MeshBvh;
use glam::Vec3;

fn main() {
    let verts = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    #[rustfmt::skip]
    let idx: [u32; 24] = [
        0, 2, 4,  2, 1, 4,  1, 3, 4,  3, 0, 4,
        2, 0, 5,  1, 2, 5,  3, 1, 5,  0, 3, 5,
    ];
    let bvh = MeshBvh::build(&verts, &idx, 2);

    // straight out from the centre of face 0 (+x, +y, +z)
    let (i, w, p) = bvh
        .closest_triangle(Vec3::splat(1.0))
        .expect("non-empty mesh");
    println!("face centre query: triangle {i}, weights {w:?}, point {p:?}");
    assert_eq!(i, 0);
    assert!((p - Vec3::splat(1.0 / 3.0)).length() < 1e-6);
    assert!((w - Vec3::splat(1.0 / 3.0)).length() < 1e-6);

    // straight out from vertex +y: four triangles reach it at the same
    // distance, and the lowest input index (0) answers
    let (i, w, p) = bvh.closest_triangle(Vec3::new(0.0, 3.0, 0.0)).unwrap();
    println!("vertex query: triangle {i}, weights {w:?}, point {p:?}");
    assert_eq!((i, p), (0, Vec3::Y));
    assert_eq!(w, Vec3::Y);

    // the weights rebuild the point on the input triangle
    let q = Vec3::new(-0.2, -0.7, 0.9);
    let (i, w, p) = bvh.closest_triangle(q).unwrap();
    let t = &idx[3 * i..3 * i + 3];
    let rebuilt =
        verts[t[0] as usize] * w.x + verts[t[1] as usize] * w.y + verts[t[2] as usize] * w.z;
    println!("query {q:?}: triangle {i}, weights {w:?}, point {p:?}");
    assert!((rebuilt - p).length() < 1e-6);
    assert_eq!(Some(p), bvh.closest_point(q));
}

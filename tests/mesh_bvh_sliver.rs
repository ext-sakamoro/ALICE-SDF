//! `MeshBvh::closest_triangle` on slivers: a triangle whose computed area is
//! not zero but whose third vertex lies almost on the line of the other two.
//!
//! oracle: every point of an edge is a point of the triangle, so the closest
//! point of the triangle is never farther than the closest point of its
//! nearest edge. The edge points are the clamped projections onto `ab`, `bc`
//! and `ca`, computed here on their own; they need no region test.
//!
//! The inputs below are the three worst of 166,380 slivers drawn as
//! `c = a + (b - a) * 0.967` (the generator is in
//! `slivers_never_answer_farther_than_an_edge`). Before the fix the answer
//! was up to 2.8 times the edge distance² farther: the region tests took the
//! sign of a 2×2 determinant smaller than its own rounding error and answered
//! from the wrong Voronoi region.

use alice_sdf::mesh::MeshBvh;
use glam::Vec3;

fn v(bits: [u32; 3]) -> Vec3 {
    Vec3::new(
        f32::from_bits(bits[0]),
        f32::from_bits(bits[1]),
        f32::from_bits(bits[2]),
    )
}

fn segment_d2(p: Vec3, a: Vec3, b: Vec3) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    let q = if len_sq == 0.0 {
        a
    } else {
        a + ab * (ab.dot(p - a) / len_sq).clamp(0.0, 1.0)
    };
    (p - q).length_squared()
}

/// distance² from `p` to the nearest of the three edges
fn nearest_edge_d2(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    segment_d2(p, a, b)
        .min(segment_d2(p, b, c))
        .min(segment_d2(p, c, a))
}

fn answered_d2(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let bvh = MeshBvh::build(&[a, b, c], &[0, 1, 2], 4);
    let (index, _, q) = bvh.closest_triangle(p).expect("non-empty mesh answers");
    assert_eq!(index, 0);
    (p - q).length_squared()
}

#[test]
fn the_worst_slivers_answer_no_farther_than_the_nearest_edge() {
    let cases = [
        (
            [0x3f4f_38e6, 0xbec1_7aee, 0x3f10_5d64],
            [0xbf3d_61f6, 0x3f4e_6538, 0x3eac_486c],
            [0xbf30_4b72, 0x3f44_6455, 0x3eb0_202a],
            [0xbe07_70c0, 0xbf41_d9d4, 0x3e41_9b90],
        ),
        (
            [0xbedc_1074, 0x3c51_8500, 0x3d88_a930],
            [0x3f48_fae0, 0xbf69_f4c3, 0x3ebe_e200],
            [0x3f3e_b772, 0xbf62_20a5, 0x3eb9_b60d],
            [0x3dcf_b760, 0x3d8b_d240, 0xbf3f_2842],
        ),
        (
            [0x3e24_05f8, 0x3eb9_8f40, 0xbf54_ee5b],
            [0xbf0e_2be0, 0xbf3f_a084, 0x3f4d_56e2],
            [0xbf08_2065, 0xbf36_3dd8, 0x3f3f_8955],
            [0x3fad_882a, 0xbf45_14a0, 0xbe4e_5150],
        ),
    ];
    for (a, b, c, p) in cases {
        let (a, b, c, p) = (v(a), v(b), v(c), v(p));
        assert_ne!((b - a).cross(c - a), Vec3::ZERO, "a sliver, not zero area");
        let got = answered_d2(p, a, b, c);
        let edge = nearest_edge_d2(p, a, b, c);
        assert!(got <= edge, "p {p:?}: answered d² {got} > nearest edge d² {edge}");
    }
}

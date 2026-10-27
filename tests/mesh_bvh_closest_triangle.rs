//! `MeshBvh::closest_triangle`: which input triangle is nearest, where on it,
//! and in which barycentric coordinates.
//!
//! oracle (closed form): the unit right triangle a=(0,0,0) b=(1,0,0) c=(0,1,0)
//! split into its seven Voronoi regions (Ericson, *Real-Time Collision
//! Detection* §5.1.5). Every expected point and weight below is written by
//! hand from that partition and is exact in binary, so the comparisons are
//! `==`.
//!
//! oracle (exhaustive parity): for a mesh the answer is the minimum of
//! `(distance², input index)` over every triangle, so the BVH walk is compared
//! with a plain loop over all triangles. The loop only uses the per-triangle
//! closest point, which the closed-form cases above pin.

use alice_sdf::mesh::{BvhTriangle, MeshBvh};
use glam::Vec3;

const A: Vec3 = Vec3::new(0.0, 0.0, 0.0);
const B: Vec3 = Vec3::new(1.0, 0.0, 0.0);
const C: Vec3 = Vec3::new(0.0, 1.0, 0.0);

fn unit_triangle() -> MeshBvh {
    MeshBvh::build(&[A, B, C], &[0, 1, 2], 4)
}

fn check(bvh: &MeshBvh, query: Vec3, index: usize, bary: Vec3, point: Vec3) {
    let got = bvh.closest_triangle(query).expect("non-empty mesh answers");
    assert_eq!(got, (index, bary, point), "query {query:?}");
}

#[test]
fn vertex_regions_answer_the_vertex_with_unit_weight() {
    let bvh = unit_triangle();
    check(&bvh, Vec3::new(-1.0, -1.0, 0.5), 0, Vec3::X, A);
    check(&bvh, Vec3::new(2.0, -1.0, 0.0), 0, Vec3::Y, B);
    check(&bvh, Vec3::new(-1.0, 2.0, -3.0), 0, Vec3::Z, C);
}

#[test]
fn edge_regions_answer_the_foot_on_the_edge() {
    let bvh = unit_triangle();
    // AB: foot at x = 0.25
    check(
        &bvh,
        Vec3::new(0.25, -1.0, 3.0),
        0,
        Vec3::new(0.75, 0.25, 0.0),
        Vec3::new(0.25, 0.0, 0.0),
    );
    // AC: foot at y = 0.5
    check(
        &bvh,
        Vec3::new(-1.0, 0.5, 0.0),
        0,
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(0.0, 0.5, 0.0),
    );
    // BC: foot at the midpoint
    check(
        &bvh,
        Vec3::new(1.0, 1.0, 0.0),
        0,
        Vec3::new(0.0, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.0),
    );
    // BC off the midpoint: (1, 0.5) projects onto (0.75, 0.25)
    // = 0.75 b + 0.25 c
    check(
        &bvh,
        Vec3::new(1.0, 0.5, 1.0),
        0,
        Vec3::new(0.0, 0.75, 0.25),
        Vec3::new(0.75, 0.25, 0.0),
    );
}

#[test]
fn face_region_answers_the_projection() {
    let bvh = unit_triangle();
    check(
        &bvh,
        Vec3::new(0.25, 0.25, 2.0),
        0,
        Vec3::new(0.5, 0.25, 0.25),
        Vec3::new(0.25, 0.25, 0.0),
    );
    check(
        &bvh,
        Vec3::new(0.5, 0.25, -1.0),
        0,
        Vec3::new(0.25, 0.5, 0.25),
        Vec3::new(0.5, 0.25, 0.0),
    );
}

/// A zero-area triangle has no face region; the answer is the nearest point
/// of its three edges (ab, then bc, then ca on a tie) and the weights put the
/// edge parameter on that edge's two vertices.
#[test]
fn degenerate_triangle_answers_the_nearest_edge() {
    // collinear: a=(0,0,0) b=(2,0,0) c=(1,0,0)
    let v = [
        Vec3::ZERO,
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    ];
    let bvh = MeshBvh::build(&v, &[0, 1, 2], 4);
    // (1.5, 1, 0): ab and bc both reach (1.5, 0, 0) at d² = 1; ab is first
    check(
        &bvh,
        Vec3::new(1.5, 1.0, 0.0),
        0,
        Vec3::new(0.25, 0.75, 0.0),
        Vec3::new(1.5, 0.0, 0.0),
    );
    // (-1, 0, 0): only the a end is nearest; ab reaches it first
    check(&bvh, Vec3::new(-1.0, 0.0, 0.0), 0, Vec3::X, Vec3::ZERO);

    // c beyond b: the region tests alone would answer through edge ac with
    // weights (0.25, 0, 0.75); the rule ties bc and ca at d² = 1, and bc wins
    let v = [
        Vec3::ZERO,
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
    ];
    let bvh = MeshBvh::build(&v, &[0, 1, 2], 4);
    check(
        &bvh,
        Vec3::new(1.5, 1.0, 0.0),
        0,
        Vec3::new(0.0, 0.5, 0.5),
        Vec3::new(1.5, 0.0, 0.0),
    );
    // all three vertices on one point
    let p = Vec3::new(3.0, -2.0, 1.0);
    let bvh = MeshBvh::build(&[p, p, p], &[0, 1, 2], 4);
    check(&bvh, Vec3::new(0.0, 0.0, 0.0), 0, Vec3::X, p);

    // two distinct vertices, c on a: segment a..b, query beyond b
    let v = [Vec3::ZERO, Vec3::new(4.0, 0.0, 0.0), Vec3::ZERO];
    let bvh = MeshBvh::build(&v, &[0, 1, 2], 4);
    check(
        &bvh,
        Vec3::new(1.0, 5.0, 0.0),
        0,
        Vec3::new(0.75, 0.25, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
    );
}

/// The index is the input triangle's position in `indices / 3`, also when a
/// degenerate triangle sits before the nearest one and a trailing partial
/// chunk is dropped.
#[test]
fn index_is_the_input_position() {
    let v = [
        Vec3::new(10.0, 10.0, 10.0),
        Vec3::new(11.0, 10.0, 10.0),
        Vec3::new(12.0, 10.0, 10.0), // 0: degenerate, far
        A,
        B,
        C, // 1: the unit triangle
        Vec3::new(-10.0, 0.0, 0.0),
        Vec3::new(-9.0, 0.0, 0.0),
        Vec3::new(-10.0, 1.0, 0.0), // 2: far
    ];
    // trailing [0, 1] is a partial chunk and is not a triangle
    let idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1];
    let bvh = MeshBvh::build(&v, &idx, 1);
    assert_eq!(bvh.triangle_count(), 3);
    check(
        &bvh,
        Vec3::new(0.25, 0.25, 2.0),
        1,
        Vec3::new(0.5, 0.25, 0.25),
        Vec3::new(0.25, 0.25, 0.0),
    );
    check(&bvh, Vec3::new(13.0, 10.0, 10.0), 0, Vec3::Z, v[2]);
    check(&bvh, Vec3::new(-10.0, 0.0, -1.0), 2, Vec3::X, v[6]);
}

/// Equal distance: the lower input index wins, wherever the BVH puts the
/// copies.
#[test]
fn ties_go_to_the_lower_input_index() {
    // the same triangle at inputs 0..8 (one per leaf): every query ties
    let mut v = Vec::new();
    let mut idx = Vec::new();
    for i in 0..8u32 {
        v.extend_from_slice(&[A, B, C]);
        idx.extend_from_slice(&[3 * i, 3 * i + 1, 3 * i + 2]);
    }
    let bvh = MeshBvh::build(&v, &idx, 1);
    for q in [
        Vec3::new(0.25, 0.25, 1.0),
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ] {
        assert_eq!(bvh.closest_triangle(q).unwrap().0, 0, "query {q:?}");
    }

    // two triangles sharing edge BC, mirrored; a query on the plane through
    // BC normal to the mesh is equidistant from both. Input order 1 then 0 in
    // space, so the walk cannot win by visiting the left child first.
    let d = Vec3::new(1.0, 1.0, 0.0);
    let v = [B, d, C, A, B, C];
    let bvh = MeshBvh::build(&v, &[0, 1, 2, 3, 4, 5], 1);
    let got = bvh.closest_triangle(Vec3::new(0.5, 0.5, 1.0)).unwrap();
    assert_eq!(got.0, 0);
    assert_eq!(got.2, Vec3::new(0.5, 0.5, 0.0));
}

/// A tie between two subtrees: the lower index lies in the subtree visited
/// second, at a box distance equal to the best distance.
#[test]
fn a_tie_in_the_second_subtree_still_wins() {
    let v = [
        // input 0, right of the query: edge x = 1 nearest
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        // input 1, the mirror image, left of the query (the left child)
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(-2.0, 0.0, 0.0),
        Vec3::new(-1.0, 1.0, 0.0),
    ];
    let bvh = MeshBvh::build(&v, &[0, 1, 2, 3, 4, 5], 1);
    // both at d² = 2; the boxes' distances are √2 as well
    check(
        &bvh,
        Vec3::new(0.0, 0.5, 1.0),
        0,
        Vec3::new(0.5, 0.0, 0.5),
        Vec3::new(1.0, 0.5, 0.0),
    );
}

/// A tie whose box distance rounds above the tied distance: triangle 1 is the
/// mirror image of triangle 0 across x = 0.86330026, and both are nearest at
/// their vertex v2. The left box's computed distance comes out a few ulp above
/// the tied distance, so a walk that prunes on the bare box distance answers
/// triangle 1 (found by a random search over mirrored pairs).
#[test]
fn a_rounded_box_distance_does_not_lose_a_tie() {
    let v = [
        Vec3::new(2.753103, 1.4901774, 1.360981),
        Vec3::new(2.6968179, 1.7402129, 1.6204355),
        Vec3::new(2.603341, 1.0192674, 1.0668589),
        Vec3::new(-1.0265025, 1.4901774, 1.360981),
        Vec3::new(-0.97021735, 1.7402129, 1.6204355),
        Vec3::new(-0.8767406, 1.0192674, 1.0668589),
    ];
    let bvh = MeshBvh::build(&v, &[0, 1, 2, 3, 4, 5], 1);
    let q = Vec3::new(0.86330026, 0.95373714, 0.4978857);
    // both triangles answer their v2 at the same distance (the mirror is exact)
    assert_eq!(
        (q - bvh.triangles[0].closest_point(q)).length_squared(),
        (q - bvh.triangles[1].closest_point(q)).length_squared()
    );
    check(&bvh, q, 0, Vec3::Z, v[2]);
}

#[test]
fn empty_mesh_and_non_finite_query_answer_none() {
    let empty = MeshBvh::build(&[], &[], 4);
    assert_eq!(empty.closest_triangle(Vec3::ZERO), None);
    // only a partial chunk: no triangle
    let partial = MeshBvh::build(&[A, B], &[0, 1], 4);
    assert_eq!(partial.closest_triangle(Vec3::ZERO), None);

    let bvh = unit_triangle();
    assert_eq!(bvh.closest_triangle(Vec3::new(f32::NAN, 0.0, 0.0)), None);
    assert_eq!(bvh.closest_triangle(Vec3::splat(f32::INFINITY)), None);
}

/// Small deterministic generator so the test needs no extra dependency.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // 24 high bits → [0, 1), then to [-2, 2): exact in f32
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
    fn v(&mut self) -> Vec3 {
        Vec3::new(self.next(), self.next(), self.next())
    }
}

fn brute_force(tris: &[BvhTriangle], q: Vec3) -> (usize, Vec3, f32) {
    let mut best = (usize::MAX, Vec3::ZERO, f32::INFINITY);
    for (i, t) in tris.iter().enumerate() {
        let p = t.closest_point(q);
        let d = (q - p).length_squared();
        if d < best.2 {
            best = (i, p, d);
        }
    }
    best
}

/// BVH walk vs. the exhaustive loop on soups that contain duplicates,
/// degenerate triangles and coplanar neighbours; every leaf size from 1 to 8.
#[test]
fn bvh_walk_matches_the_exhaustive_minimum() {
    let mut rng = Lcg(0x5eed_c105_e57f_ace5);
    let mut compared = 0usize;
    for leaf in 1..=8 {
        let mut v = Vec::new();
        for _ in 0..60 {
            v.extend_from_slice(&[rng.v(), rng.v(), rng.v()]);
        }
        // duplicates of earlier triangles, later in the input
        for k in 0..10 {
            let t = 3 * (k * 5);
            v.extend_from_slice(&[v[t], v[t + 1], v[t + 2]]);
        }
        // degenerate: collinear and a single point
        let (p, q) = (rng.v(), rng.v());
        v.extend_from_slice(&[p, q, (p + q) * 0.5, p, p, p]);
        let idx: Vec<u32> = (0..v.len() as u32).collect();
        let bvh = MeshBvh::build(&v, &idx, leaf);

        for _ in 0..400 {
            let query = rng.v() * 1.5;
            let (bi, bp, bd) = brute_force(&bvh.triangles, query);
            let (gi, bary, gp) = bvh.closest_triangle(query).unwrap();
            assert_eq!((gi, gp), (bi, bp), "leaf {leaf} query {query:?}");
            assert_eq!((query - gp).length_squared(), bd);
            // weights are a convex combination that rebuilds the point
            let t = &bvh.triangles[gi];
            // face weights are (1 - v - w, v, w): the first may round to -ulp
            assert!(bary.min_element() >= -1e-6, "{bary:?}");
            assert!((bary.x + bary.y + bary.z - 1.0).abs() <= 1e-6, "{bary:?}");
            let rebuilt = t.v0 * bary.x + t.v1 * bary.y + t.v2 * bary.z;
            assert!((rebuilt - gp).length() <= 1e-5, "{rebuilt:?} vs {gp:?}");
            compared += 1;
        }
    }
    assert_eq!(compared, 8 * 400);
}

/// The point returned with the weights is bit-identical to the existing
/// `closest_point` queries, so adding the index does not move any distance.
#[test]
fn point_agrees_with_closest_point() {
    let mut rng = Lcg(42);
    let mut v = Vec::new();
    for _ in 0..40 {
        v.extend_from_slice(&[rng.v(), rng.v(), rng.v()]);
    }
    let idx: Vec<u32> = (0..v.len() as u32).collect();
    let bvh = MeshBvh::build(&v, &idx, 4);
    for _ in 0..500 {
        let q = rng.v() * 2.0;
        let (_, _, p) = bvh.closest_triangle(q).unwrap();
        assert_eq!(Some(p), bvh.closest_point(q));
        assert_eq!((q - p).length(), bvh.unsigned_distance(q));
    }

    // an exact tie between two subtrees: closest_point answers the same tied
    // triangle (the lower index) as closest_triangle, not whichever the walk
    // meets first
    let v = [
        Vec3::new(2.753103, 1.4901774, 1.360981),
        Vec3::new(2.6968179, 1.7402129, 1.6204355),
        Vec3::new(2.603341, 1.0192674, 1.0668589),
        Vec3::new(-1.0265025, 1.4901774, 1.360981),
        Vec3::new(-0.97021735, 1.7402129, 1.6204355),
        Vec3::new(-0.8767406, 1.0192674, 1.0668589),
    ];
    let bvh = MeshBvh::build(&v, &[0, 1, 2, 3, 4, 5], 1);
    for q in [
        Vec3::new(0.86330026, 0.95373714, 0.4978857),
        Vec3::new(0.86330026, 0.4272502, 0.6176987),
    ] {
        let (i, _, p) = bvh.closest_triangle(q).unwrap();
        assert_eq!((i, bvh.closest_point(q)), (0, Some(p)), "query {q:?}");
        assert_eq!((q - p).length(), bvh.unsigned_distance(q));
    }
}

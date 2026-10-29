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

const fn v(bits: [u32; 3]) -> Vec3 {
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
        assert!(
            got <= edge,
            "p {p:?}: answered d² {got} > nearest edge d² {edge}"
        );
    }
}

/// The bound of the answer outside the ill-conditioned path: an edge or
/// vertex expression moves its point by at most `2^-19·(|ap|₁ + |bp|₁ + |cp|₁)`
/// (the factor and its derivation are `REGION_DET_ERR` in `src/mesh/bvh.rs`;
/// a face answer that could move farther is sent to the ill-conditioned
/// path). On that path the answer is never farther than the nearest edge.
/// Either way the answered distance is at most the nearest edge's distance
/// plus that bound.
fn bound(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let l1 = |x: Vec3| x.x.abs() + x.y.abs() + x.z.abs();
    (l1(p - a) + l1(p - b) + l1(p - c)) / 524_288.0
}

type D3 = [f64; 3];

fn d3(v: Vec3) -> D3 {
    [f64::from(v.x), f64::from(v.y), f64::from(v.z)]
}

fn sub(a: D3, b: D3) -> D3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: D3, b: D3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: D3, b: D3) -> D3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn seg_dist(p: D3, a: D3, b: D3) -> f64 {
    let ab = sub(b, a);
    let len_sq = dot(ab, ab);
    let t = if len_sq == 0.0 {
        0.0
    } else {
        (dot(sub(p, a), ab) / len_sq).clamp(0.0, 1.0)
    };
    let q = [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t];
    dot(sub(p, q), sub(p, q)).sqrt()
}

/// The distance from `p` to the triangle, computed independently in f64 on the exact f32 inputs
/// (29 more bits than the code under test): the foot of the perpendicular when it lies inside
/// (the three edge-side cross products agree with the normal), else the nearest of the three
/// edges
fn true_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (p, a, b, c) = (d3(p), d3(a), d3(b), d3(c));
    let n = cross(sub(b, a), sub(c, a));
    let nn = dot(n, n);
    let edges = seg_dist(p, a, b)
        .min(seg_dist(p, b, c))
        .min(seg_dist(p, c, a));
    if nn == 0.0 {
        return edges;
    }
    let h = dot(sub(p, a), n) / nn;
    let q = [p[0] - n[0] * h, p[1] - n[1] * h, p[2] - n[2] * h];
    let inside = [(a, b), (b, c), (c, a)]
        .iter()
        .all(|&(u, w)| dot(cross(sub(w, u), sub(q, u)), n) >= 0.0);
    if inside {
        (h.abs() * nn.sqrt()).min(edges)
    } else {
        edges
    }
}

/// The error bound of a face answer, from the error model of `REGION_DET_ERR` in
/// `src/mesh/bvh.rs` evaluated in f64: each region determinant is within
/// `2^-19·(S·S' + S''·S''')` of its exact value, so a face weight is within `2·Σerr/denom` and
/// the face point within `(|ab|₁ + |ac|₁)·2·Σerr/denom`
fn face_bound(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (p, a, b, c) = (d3(p), d3(a), d3(b), d3(c));
    let abs = |v: D3| [v[0].abs(), v[1].abs(), v[2].abs()];
    let l1 = |v: D3| v[0].abs() + v[1].abs() + v[2].abs();
    let (ab, ac) = (sub(b, a), sub(c, a));
    let (ap, bp, cp) = (sub(p, a), sub(p, b), sub(p, c));
    let s = |u: D3, w: D3| dot(abs(u), abs(w));
    let (s1, s2, s3, s4, s5, s6) = (
        s(ab, ap),
        s(ac, ap),
        s(ab, bp),
        s(ac, bp),
        s(ab, cp),
        s(ac, cp),
    );
    let e = 1.0 / 524_288.0;
    let err = e * ((s1 * s4 + s3 * s2) + (s5 * s2 + s1 * s6) + (s3 * s6 + s5 * s4));
    let n = cross(ab, ac);
    let denom = dot(n, n);
    if denom == 0.0 {
        return f64::INFINITY;
    }
    (l1(ab) + l1(ac)) * 2.0 * err / denom
}

/// `(compared, beyond)`: how many non-degenerate triangles were compared and how many answered
/// a distance outside what the code guarantees, against the independent [`true_distance`]:
///
/// - never nearer than the truth by more than [`bound`]: every candidate is a point of the
///   triangle;
/// - never farther than the truth plus [`bound`] (an edge or vertex expression, or a face answer
///   that passed the face gate), or else than the ill-conditioned answer allows: no farther than
///   the nearest edge and no farther than the face point, which is within [`face_bound`] of the
///   true nearest point.
///
/// `face_bound` alone would be no oracle on slivers: it divides by the small determinant sum and
/// grows past any wrong-region answer. Each triangle is also tried with its vertices rotated, so
/// the near-collinear vertex takes each role and each region determinant is in turn the one that
/// trips the guard.
fn beyond_bound(tris: &[(Vec3, Vec3, Vec3, Vec3)]) -> (usize, usize) {
    let (mut compared, mut beyond) = (0, 0);
    for &(a, b, c, p) in tris {
        for (a, b, c) in [(a, b, c), (b, c, a), (c, a, b)] {
            if (b - a).cross(c - a) == Vec3::ZERO {
                continue;
            }
            compared += 1;
            let got = f64::from(answered_d2(p, a, b, c).sqrt());
            let truth = true_distance(p, a, b, c);
            // plus the f32 rounding of the answered point: three coordinates of magnitude at
            // most 2 in these sweeps, each within ε·|x|
            let eps = 6.0 * f64::from(f32::EPSILON);
            let bnd = f64::from(bound(p, a, b, c));
            let (pd, ad, bd, cd) = (d3(p), d3(a), d3(b), d3(c));
            let edges = seg_dist(pd, ad, bd)
                .min(seg_dist(pd, bd, cd))
                .min(seg_dist(pd, cd, ad));
            let ill = edges.min(truth + face_bound(p, a, b, c));
            if got < truth - bnd - eps || got > (truth + bnd).max(ill) + eps {
                beyond += 1;
            }
        }
    }
    (compared, beyond)
}

/// The 200,000 draws the three worst cases above come from:
/// `c = a + (b − a)·0.967` rounded in f32, so `c` lies within rounding of the
/// line `ab`, and `p = 2·v` for a fresh draw `v`.
#[test]
fn slivers_never_answer_farther_than_an_edge() {
    let mut x: u64 = 11;
    let mut f = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((x >> 33) % 2_000_001) as f32 / 1e6 - 1.0
    };
    let mut v = || Vec3::new(f(), f(), f());
    let tris: Vec<_> = (0..200_000)
        .map(|_| {
            let a = v();
            let b = v();
            let c = a + (b - a) * 0.967;
            let p = v() * 2.0;
            (a, b, c, p)
        })
        .collect();
    let (compared, beyond) = beyond_bound(&tris);
    assert!(compared > 150_000, "compared {compared}");
    assert_eq!(beyond, 0, "{beyond} of {compared} beyond the bound");
}

/// Thin triangles from well conditioned to a few ulp off a line:
/// a = (0, 0, 0), b = (1, 0, 0), c = (t, h, 0) with h = 2^-1 … 2^-40 and the
/// query on a grid around the triangle, so the gate is crossed on both sides.
#[test]
fn every_aspect_ratio_stays_within_the_bound() {
    let mut tris = Vec::new();
    for k in 1_u32..=40 {
        // 2^-k, exact
        let h = f32::from_bits((127 - k) << 23);
        for t in [-0.3_f32, 0.0, 0.25, 0.5, 0.967, 1.0, 1.4] {
            let (a, b, c) = (Vec3::ZERO, Vec3::X, Vec3::new(t, h, 0.0));
            for i in -4..=4 {
                // below, on, inside (0.25 … 0.75 of the height) and above the
                // thin triangle
                for fy in [-8.0_f32, -2.0, 0.0, 0.25, 0.5, 0.75, 2.0, 8.0] {
                    for z in [0.0_f32, 0.3] {
                        let p = Vec3::new(0.5 + 0.3 * i as f32, h * fy, z);
                        tris.extend([(a, b, c, p)]);
                    }
                }
            }
        }
    }
    let (compared, beyond) = beyond_bound(&tris);
    assert!(compared > 10_000, "compared {compared}");
    assert_eq!(beyond, 0, "{beyond} of {compared} beyond the bound");
}

/// Queries that project inside thin triangles (height 2^-6 … 2^-22 of the
/// length, all weights 0.05 … 0.9, a small offset along the normal): the
/// region signs are certain there, but the face weights divide by a small
/// determinant sum, so the face answer is the one that can be off. Without
/// the face-weight test 809 of these 100,000 answered beyond the bound.
#[test]
fn interior_queries_of_thin_triangles_stay_within_the_bound() {
    let mut x: u64 = 23;
    let mut r = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((x >> 33) % 1_000_001) as f32 / 1e6
    };
    let mut tris = Vec::new();
    for n in 0..100_000_u32 {
        let k = 6 + n % 17;
        let h = f32::from_bits((127 - k) << 23);
        let a = Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5);
        let b = Vec3::new(r() + 0.5, r() - 0.5, r() - 0.5);
        let ab = b - a;
        let side = ab
            .cross(Vec3::new(r() - 0.5, r() - 0.5, r() - 0.5))
            .normalize();
        let c = a + ab * (0.1 + 0.8 * r()) + side * (h * ab.length());
        let (mut v, mut w) = (0.05 + 0.85 * r(), 0.05 + 0.85 * r());
        if v + w > 0.95 {
            v *= 0.5;
            w *= 0.5;
        }
        let normal = (b - a).cross(c - a).normalize_or_zero();
        let foot = a + (b - a) * v + (c - a) * w;
        tris.extend([(a, b, c, foot + normal * ((r() - 0.5) * 0.2))]);
    }
    let (compared, beyond) = beyond_bound(&tris);
    assert!(compared > 90_000, "compared {compared}");
    assert_eq!(beyond, 0, "{beyond} of {compared} beyond the bound");
}

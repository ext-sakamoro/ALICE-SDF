//! The Law max / min at the library's vector sites, through the public API
//! (ALICE-Law G2 and G13)
//!
//! `glam`'s `min` / `max` / `clamp` and the lane reductions call `f32::min` /
//! `f32::max` per lane, whose result for an equal pair of zeros is the target's
//! (`min(+0, -0)` is `-0` on aarch64 and the second argument on x86_64), so a
//! bound folded from such values took a sign that depended on the platform and,
//! in a parallel fold, on how rayon split the work. The library uses the Law
//! helpers instead (`law_min` gives `-0` for any pair holding a `-0`, `law_max`
//! gives `+0` for any pair holding a `+0`, NaN gives the other operand), so the
//! expected values below follow from that rule, written out in this file, not
//! from the library.
//!
//! Site classes: (a) bounds built from points (`compute_aabb`, the lightmap
//! projection bounds), checked here bit for bit, including the independence
//! from the thread count; (b) the primitive and evaluator distances, where the
//! clamped vector goes into a length or a square, so the sign of a zero is not
//! observable in the distance: the distance bits at points on the zero
//! components are pinned instead; (c) the wasm render helpers, where one operand
//! is a non-zero constant (`min(1.0)`, `max(0.001)`), so no tie exists.

use alice_sdf::mesh::{
    compute_aabb, generate_lightmap_uvs, sdf_to_mesh, MarchingCubesConfig, Mesh, Vertex,
};
use alice_sdf::{eval, SdfNode};
use glam::Vec3;

fn vertex(p: Vec3) -> Vertex {
    Vertex {
        position: p,
        ..Vertex::default()
    }
}

fn mesh(points: &[Vec3]) -> Mesh {
    Mesh {
        vertices: points.iter().copied().map(vertex).collect(),
        indices: Vec::new(),
    }
}

/// The Law minimum, written here as the rule states it
fn law_min(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    if a == b {
        // an equal pair: -0 if either is -0
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b {
        a
    } else {
        b
    }
}

/// The Law maximum, written here as the rule states it
fn law_max(a: f32, b: f32) -> f32 {
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    if a == b {
        // an equal pair: +0 if either is +0
        return if a.is_sign_negative() { b } else { a };
    }
    if a > b {
        a
    } else {
        b
    }
}

const fn bits(v: Vec3) -> [u32; 3] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

#[test]
fn the_bounds_of_points_take_the_law_sign_of_zero_and_skip_nan() {
    let (p, n) = (0.0f32, -0.0f32);
    for points in [
        vec![Vec3::new(p, n, 1.0), Vec3::new(n, p, f32::NAN)],
        vec![Vec3::new(n, p, f32::NAN), Vec3::new(p, n, 1.0)],
    ] {
        let b = compute_aabb(&mesh(&points));
        assert_eq!(
            bits(b.min),
            [n.to_bits(), n.to_bits(), 1.0f32.to_bits()],
            "{points:?}"
        );
        assert_eq!(
            bits(b.max),
            [p.to_bits(), p.to_bits(), 1.0f32.to_bits()],
            "{points:?}"
        );
    }
}

/// Points with many equal pairs of zeros of both signs and some NaN lanes
fn awkward_points(n: usize) -> Vec<Vec3> {
    let mut s = 0x9e37_79b9_u32;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    (0..n)
        .map(|_| {
            let mut lane = || match next() % 6 {
                0 => 0.0,
                1 => -0.0,
                2 => f32::NAN,
                _ => 0.0,
            };
            Vec3::new(lane(), lane(), lane())
        })
        .collect()
}

#[test]
fn the_parallel_bounds_do_not_depend_on_the_thread_count() {
    let points = awkward_points(20_000);
    // the expected bounds: a sequential fold under the rule above
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for q in &points {
        for (i, x) in [q.x, q.y, q.z].into_iter().enumerate() {
            lo[i] = law_min(lo[i], x);
            hi[i] = law_max(hi[i], x);
        }
    }
    let want = (lo.map(f32::to_bits), hi.map(f32::to_bits));
    // every lane holds both zeros, so the rule fixes the signs
    assert_eq!(want.0, [(-0.0f32).to_bits(); 3]);
    assert_eq!(want.1, [0.0f32.to_bits(); 3]);
    let m = mesh(&points);
    for threads in [1, 2, 3, 8] {
        let b = pool(threads).install(|| compute_aabb(&m));
        assert_eq!((bits(b.min), bits(b.max)), want, "{threads} threads");
    }
}

#[test]
fn the_lightmap_uvs_do_not_depend_on_the_thread_count() {
    let node = SdfNode::sphere(1.0);
    let config = MarchingCubesConfig {
        resolution: 24,
        ..MarchingCubesConfig::default()
    };
    let base = sdf_to_mesh(&node, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    assert!(base.vertices.len() > 100);
    let uvs = |threads: usize| {
        let mut m = base.clone();
        pool(threads).install(|| generate_lightmap_uvs(&mut m, 256));
        m.vertices
            .iter()
            .map(|v| [v.uv2.x.to_bits(), v.uv2.y.to_bits()])
            .collect::<Vec<_>>()
    };
    let one = uvs(1);
    for threads in [2, 8] {
        assert_eq!(uvs(threads), one, "{threads} threads");
    }
}

#[test]
fn distances_at_points_on_the_zero_components_are_the_recorded_bits() {
    // class (b): the clamped vector feeds a length, so only the distance bits
    // are observable; they were recorded once (aarch64) and must hold on every target CI runs
    let cases: [(&str, SdfNode, Vec3, u32); 4] = [
        (
            "box frame on a face",
            SdfNode::box_frame(Vec3::splat(1.0), 0.1),
            Vec3::new(1.0, 0.0, -0.0),
            0x3f4c_cccc,
        ),
        (
            "rounded box at a corner",
            SdfNode::rounded_box(1.0, 1.0, 1.0, 0.25),
            Vec3::new(1.0, 1.0, 1.0),
            0xbe80_0000,
        ),
        (
            "rounded box at the centre",
            SdfNode::rounded_box(1.0, 1.0, 1.0, 0.25),
            Vec3::new(0.0, -0.0, 0.0),
            0xbfa0_0000,
        ),
        (
            "box frame at the origin",
            SdfNode::box_frame(Vec3::splat(1.0), 0.1),
            Vec3::new(-0.0, 0.0, -0.0),
            0x3f90_d0c2,
        ),
    ];
    for (name, node, p, want) in cases {
        assert_eq!(eval(&node, p).to_bits(), want, "{name}");
    }
}

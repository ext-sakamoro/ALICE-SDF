//! SoA storage and the SoA batch evaluators against the scalar evaluator.
//!
//! oracle: the SoA path only changes the memory layout, so every SoA result
//! must be bit-identical to `eval_compiled` at the same point — for each way a
//! `SoAPoints` can be built (`from_vec3_slice`, `FromIterator`, `push`, `push`
//! after padding) and for each evaluator (`eval_compiled_batch_soa`,
//! `_parallel`, `_into`). The storage accessors are checked against the
//! sequence of points that went in.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_soa, eval_compiled_batch_soa_into,
    eval_compiled_batch_soa_parallel, CompiledSdf,
};
use alice_sdf::soa::{AlignedVec, SoADistances, SoAPoints, SIMD_ALIGNMENT, SIMD_WIDTH};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn scene() -> CompiledSdf {
    let node = SdfNode::sphere(1.0)
        .smooth_union(SdfNode::box3d(0.6, 0.4, 0.8).translate(0.7, 0.2, -0.1), 0.2)
        .subtract(SdfNode::cylinder(0.25, 2.0));
    CompiledSdf::compile(&node)
}

fn points(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let t = i as f32 * 0.618_034;
            Vec3::new(
                (t * 1.3).sin() * 1.7,
                (t * 0.7).cos() * 1.4,
                (t * 2.1).sin() * (t * 0.3).cos() * 1.9,
            )
        })
        .collect()
}

fn assert_bits(sdf: &CompiledSdf, pts: &[Vec3], got: &[f32], what: &str) -> usize {
    assert_eq!(got.len(), pts.len(), "{what}: length");
    for (i, (&p, &d)) in pts.iter().zip(got).enumerate() {
        let want = eval_compiled(sdf, p);
        assert_eq!(d.to_bits(), want.to_bits(), "{what}: point {i} {p:?}");
    }
    pts.len()
}

#[test]
fn soa_batch_evaluators_are_bit_identical_to_scalar() {
    let sdf = scene();
    let mut compared = 0;
    // 0, below one lane group, exact multiples, ragged tails, above the
    // parallel threshold (256)
    for n in [0usize, 1, 7, 8, 9, 23, 64, 255, 257, 1000] {
        let pts = points(n);

        let soa = SoAPoints::from_vec3_slice(&pts);
        compared += assert_bits(&sdf, &pts, &eval_compiled_batch_soa(&sdf, &soa), "slice");
        compared += assert_bits(
            &sdf,
            &pts,
            &eval_compiled_batch_soa_parallel(&sdf, &soa),
            "parallel",
        );

        let mut out = SoADistances::with_capacity(n);
        eval_compiled_batch_soa_into(&sdf, &soa, &mut out);
        assert_eq!(out.len(), n);
        assert_eq!(out.is_empty(), n == 0);
        compared += assert_bits(&sdf, &pts, out.as_slice(), "into");
        assert_eq!(out.to_vec(), out.as_slice());

        let collected: SoAPoints = pts.iter().copied().collect();
        compared += assert_bits(
            &sdf,
            &pts,
            &eval_compiled_batch_soa(&sdf, &collected),
            "iter",
        );

        // push one by one, without calling ensure_padding
        let mut pushed = SoAPoints::new();
        for &p in &pts {
            pushed.push_vec3(p);
        }
        compared += assert_bits(&sdf, &pts, &eval_compiled_batch_soa(&sdf, &pushed), "push");
        compared += assert_bits(
            &sdf,
            &pts,
            &eval_compiled_batch_soa_parallel(&sdf, &pushed),
            "push parallel",
        );
        let mut out = SoADistances::with_capacity(n);
        eval_compiled_batch_soa_into(&sdf, &pushed, &mut out);
        compared += assert_bits(&sdf, &pts, out.as_slice(), "push into");
    }
    assert!(compared > 10_000, "compared {compared}");
}

/// A point pushed after `from_vec3_slice` (which pads to the SIMD width) is the
/// next point: `get(len - 1)` returns it and the evaluators see it.
#[test]
fn push_after_padding_appends_the_point() {
    let sdf = scene();
    let mut pts = points(3);
    let mut soa = SoAPoints::from_vec3_slice(&pts);
    assert_eq!(soa.padded_len(), SIMD_WIDTH);
    for p in points(13).into_iter().skip(3) {
        soa.push_vec3(p);
        pts.push(p);
        assert_eq!(soa.len(), pts.len());
        assert_eq!(soa.get(soa.len() - 1), Some(p));
    }
    assert_eq!(soa.iter().collect::<Vec<_>>(), pts);
    assert_bits(
        &sdf,
        &pts,
        &eval_compiled_batch_soa(&sdf, &soa),
        "padded push",
    );

    // ensure_padding then push again: still appends at len
    soa.ensure_padding();
    let extra = Vec3::new(0.1, -0.2, 0.3);
    soa.push(extra.x, extra.y, extra.z);
    pts.push(extra);
    assert_eq!(soa.get(pts.len() - 1), Some(extra));
    assert_eq!(soa.iter().collect::<Vec<_>>(), pts);
}

#[test]
fn soa_accessors_return_what_went_in() {
    let pts = points(11);
    let mut soa = SoAPoints::with_capacity(pts.len());
    assert!(soa.is_empty());
    for &p in &pts {
        soa.push(p.x, p.y, p.z);
    }
    assert_eq!(soa.len(), 11);
    assert_eq!(soa.padded_len(), 16);
    for (i, &p) in pts.iter().enumerate() {
        assert_eq!(soa.get(i), Some(p));
    }
    assert_eq!(soa.get(11), None);

    let (xs, ys, zs) = soa.as_slices();
    assert!(xs.len() >= soa.padded_len());
    for (i, p) in pts.iter().enumerate() {
        assert_eq!((xs[i], ys[i], zs[i]), (p.x, p.y, p.z));
    }

    // checked and unchecked SIMD loads agree with the scalar layout
    for start in [0usize, 8] {
        let (x, y, z) = soa.load_simd(start).expect("padded");
        // SAFETY: start is a multiple of 8 and start + 8 <= padded_len (16)
        let (ux, uy, uz) = unsafe { soa.load_simd_unchecked(start) };
        let (x, y, z): ([f32; 8], [f32; 8], [f32; 8]) = (x.into(), y.into(), z.into());
        let (ux, uy, uz): ([f32; 8], [f32; 8], [f32; 8]) = (ux.into(), uy.into(), uz.into());
        assert_eq!((x, y, z), (ux, uy, uz));
        for lane in 0..8 {
            let i = start + lane;
            let want = pts.get(i).copied().unwrap_or(Vec3::ZERO);
            assert_eq!(Vec3::new(x[lane], y[lane], z[lane]), want, "index {i}");
        }
    }
    assert!(soa.load_simd(16).is_none());

    let (px, py, pz) = soa.as_ptrs();
    // SAFETY: index 4 < len
    unsafe {
        assert_eq!(
            (*px.add(4), *py.add(4), *pz.add(4)),
            (pts[4].x, pts[4].y, pts[4].z)
        );
    }

    soa.clear();
    assert!(soa.is_empty());
    assert_eq!(soa.get(0), None);
    assert_eq!(soa.iter().count(), 0);
}

#[test]
fn aligned_vec_is_aligned_and_behaves_like_a_vec() {
    assert_eq!(SIMD_ALIGNMENT, SIMD_WIDTH * std::mem::size_of::<f32>());
    let mut v = AlignedVec::with_capacity(3);
    assert!(v.is_empty());
    for i in 0..37 {
        v.push(i as f32 * 0.5);
        assert_eq!(v.as_ptr() as usize % SIMD_ALIGNMENT, 0, "len {}", v.len());
    }
    assert_eq!(v.len(), 37);
    assert_eq!(v.as_slice()[36], 18.0);

    // SAFETY: index 5 < len
    unsafe { v.as_mut_ptr().add(5).write(-1.0) };
    assert_eq!(v[5], -1.0);
    v.as_mut_slice()[6] = -2.0;
    assert_eq!(v.as_slice()[6], -2.0);

    let c = v.clone();
    assert_eq!(c.as_slice(), v.as_slice());
    assert_eq!(c.as_ptr() as usize % SIMD_ALIGNMENT, 0);

    v.clear();
    assert!(v.is_empty());
    v.push(4.0);
    assert_eq!(v.as_slice(), &[4.0]);
}

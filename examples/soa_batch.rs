//! Structure-of-arrays point storage for the 8-wide batch evaluators.
//!
//! Builds `SoAPoints` three ways (slice, iterator, `push`), evaluates them with
//! the SoA batch evaluators, reuses a `SoADistances` output buffer, and reads
//! the storage back. Every result is checked bit for bit against the scalar
//! compiled evaluator.
//!
//! Run: `cargo run --example soa_batch`
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{
    eval_compiled, eval_compiled_batch_soa, eval_compiled_batch_soa_into,
    eval_compiled_batch_soa_parallel, eval_compiled_simd, CompiledSdf, Vec3x8,
};
use alice_sdf::soa::{AlignedVec, SoADistances, SoAPoints, SIMD_ALIGNMENT, SIMD_WIDTH};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn main() {
    let node = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(1.4, 0.4, 0.4), 0.2);
    let sdf = CompiledSdf::compile(&node);

    let pts: Vec<Vec3> = (0..300)
        .map(|i| {
            let t = i as f32 * 0.1;
            Vec3::new(t.sin() * 1.5, (t * 0.7).cos() * 1.2, (t * 1.3).sin())
        })
        .collect();
    let check = |what: &str, got: &[f32]| {
        assert_eq!(got.len(), pts.len());
        let same = pts
            .iter()
            .zip(got)
            .filter(|(p, d)| d.to_bits() == eval_compiled(&sdf, **p).to_bits())
            .count();
        println!(
            "{what:<10} {} points, {same} bit-identical to the scalar path",
            got.len()
        );
        assert_eq!(same, pts.len());
    };

    // slice → SoA (padded to the SIMD width)
    let soa = SoAPoints::from_vec3_slice(&pts);
    println!(
        "SoA: {} points, padded to {} (lane width {SIMD_WIDTH})",
        soa.len(),
        soa.padded_len()
    );
    check("serial", &eval_compiled_batch_soa(&sdf, &soa));
    check("parallel", &eval_compiled_batch_soa_parallel(&sdf, &soa));

    // reusable output buffer
    let mut out = SoADistances::with_capacity(soa.len());
    eval_compiled_batch_soa_into(&sdf, &soa, &mut out);
    assert!(!out.is_empty() && out.len() == pts.len());
    check("into", out.as_slice());
    assert_eq!(out.to_vec(), out.as_slice());

    // iterator and push, then appending past the padding
    let collected: SoAPoints = pts[..200].iter().collect();
    let mut grown = collected;
    for p in &pts[200..] {
        grown.push_vec3(*p);
    }
    grown.ensure_padding();
    check("pushed", &eval_compiled_batch_soa(&sdf, &grown));
    assert_eq!(grown.iter().collect::<Vec<_>>(), pts);
    assert_eq!(grown.get(299), Some(pts[299]));

    // one lane group by hand: load 8 points, evaluate, store
    let (x, y, z) = soa.load_simd(8).expect("index 8..16 is inside the padding");
    // SAFETY: 8 is a multiple of SIMD_WIDTH and 8 + 8 <= padded_len
    let (ux, uy, uz) = unsafe { soa.load_simd_unchecked(8) };
    let lanes = eval_compiled_simd(&sdf, Vec3x8 { x, y, z });
    let lanes_u = eval_compiled_simd(
        &sdf,
        Vec3x8 {
            x: ux,
            y: uy,
            z: uz,
        },
    );
    let mut buf = SoADistances::with_capacity(16);
    // SAFETY: 8 + 8 <= 16, the buffer's padded length
    unsafe { buf.store_simd_unchecked(8, lanes) };
    let a: [f32; 8] = lanes.into();
    let b: [f32; 8] = lanes_u.into();
    assert_eq!(a, b);
    assert_eq!(&buf.as_slice()[8..16], &a);
    let (px, _, _) = soa.as_ptrs();
    let (xs, _, _) = soa.as_slices();
    // SAFETY: index 9 < len
    assert_eq!(unsafe { *px.add(9) }, xs[9]);
    println!("lane group 8..16: {a:?}");

    // the aligned buffer behind each coordinate array
    let mut v = AlignedVec::with_capacity(4);
    for i in 0..20 {
        v.push(i as f32);
    }
    v.as_mut_slice()[0] = -1.0;
    // SAFETY: index 1 < len
    unsafe { v.as_mut_ptr().add(1).write(-2.0) };
    println!(
        "AlignedVec: {} values, {}-byte aligned: {}",
        v.len(),
        SIMD_ALIGNMENT,
        v.as_ptr() as usize % SIMD_ALIGNMENT == 0
    );
    assert_eq!(v.as_ptr() as usize % SIMD_ALIGNMENT, 0);
    assert_eq!(&v.as_slice()[..3], &[-1.0, -2.0, 2.0]);
    v.clear();
    assert!(v.is_empty());

    let mut s = soa;
    s.clear();
    assert!(s.is_empty());
}

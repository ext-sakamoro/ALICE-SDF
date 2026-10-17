//! Cache distances by the exact point
//!
//! `ExactEvalCache` returns a cached distance only for the very point it was computed
//! at, so evaluating through it gives the same bits as evaluating directly, whatever is
//! cached and however threads interleave.
//!
//! Run: `cargo run --example exact_eval_cache`

use alice_sdf::{eval, ExactEvalCache, SdfNode};
use glam::Vec3;

fn main() {
    let node = SdfNode::sphere(1.0);
    let cache = ExactEvalCache::new(1024);
    let points: Vec<Vec3> = (0..16)
        .flat_map(|i| (0..16).map(move |j| Vec3::new(i as f32 * 0.125, j as f32 * 0.125, 0.0)))
        .collect();
    for _ in 0..3 {
        for &p in &points {
            let d = cache.get_or_eval(p, |q| eval(&node, q));
            assert_eq!(d.to_bits(), eval(&node, p).to_bits());
        }
    }
    let (hits, misses) = cache.hits_and_misses();
    println!(
        "{} points cached, {hits} hits, {misses} misses, first entry {:?}",
        cache.len(),
        cache.entries().first()
    );
    cache.clear();
    assert!(cache.is_empty());
}

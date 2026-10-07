//! Caching SDF evaluations on a quantised grid with `alice-cache`
//! (feature `sdf-cache`).
//!
//! Samples a field repeatedly, as an editor preview does, and serves repeated
//! samples from the cache.
//!
//! ```sh
//! cargo run --example sdf_eval_cache --features sdf-cache
//! ```

use alice_sdf::cache_bridge::{GridPoint, SdfEvalCache};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;

fn main() {
    let node = SdfNode::sphere(1.0).union(SdfNode::box3d(0.4, 0.4, 1.4));
    let cell = 0.05;
    let cache = SdfEvalCache::new(1 << 16, cell);
    println!("empty: {}", cache.is_empty());

    let mut evaluated = 0;
    for _frame in 0..3 {
        for i in 0..40 {
            for j in 0..40 {
                let (x, y, z) = (-1.0 + 0.05 * i as f32, -1.0 + 0.05 * j as f32, 0.25);
                if cache.get(x, y, z).is_none() {
                    cache.put(x, y, z, eval(&node, Vec3::new(x, y, z)));
                    evaluated += 1;
                }
            }
        }
    }
    let g = GridPoint::from_f32(0.26, -0.24, 0.25, 1.0 / cell);
    println!(
        "{} cached, {evaluated} evaluated, hit rate {:.3}; (0.26, -0.24, 0.25) is cell ({}, {}, {})",
        cache.len(),
        cache.hit_rate(),
        g.x,
        g.y,
        g.z
    );
    assert_eq!(cache.len(), evaluated);
    assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
}

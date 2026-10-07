//! Oracle tests for `cache_bridge::SdfEvalCache` (feature `sdf-cache`).
//!
//! The expected behaviour is modelled here independently:
//!
//! * key: `GridPoint::from_f32` is `round(c / cell_size)` per axis (Rust
//!   `f32::round`, half away from zero); two points share an entry exactly when
//!   their rounded coordinates agree;
//! * value: a `HashMap` of the last value put per key — a hit must return it;
//! * capacity: never more entries than the capacity (alice-cache evicts by
//!   sampling, so *which* entry goes is not modelled, only the bound and that
//!   no hit returns a stale or foreign value);
//! * hit rate: hits / (hits + misses) counted by the test, 0 before any lookup.

#![cfg(feature = "sdf-cache")]

use std::collections::HashMap;

use alice_sdf::cache_bridge::{GridPoint, SdfEvalCache};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;

fn key(p: [f32; 3], cell: f32) -> [i32; 3] {
    p.map(|c| (c / cell).round() as i32)
}

#[test]
fn grid_point_is_round_half_away_from_zero() {
    let cases = [
        // (coordinate, inv_cell, expected)
        (0.0f32, 4.0f32, 0),
        (0.125, 4.0, 1),   // 0.5 -> 1
        (-0.125, 4.0, -1), // -0.5 -> -1
        (0.124, 4.0, 0),
        (0.375, 4.0, 2),   // 1.5 -> 2
        (-0.625, 4.0, -3), // -2.5 -> -3
        (10.0, 0.5, 5),
    ];
    let mut compared = 0;
    for (c, inv, want) in cases {
        let g = GridPoint::from_f32(c, -c, 2.0 * c, inv);
        assert_eq!(
            (g.x, g.y, g.z),
            (want, -want, (2.0 * c * inv).round() as i32),
            "{c}"
        );
        compared += 1;
    }
    assert_eq!(compared, 7);
}

#[test]
fn cached_values_match_the_model_and_the_evaluator() {
    let node = SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 1.5));
    let cell = 0.25f32;
    // Large capacity: 64 entries per shard, far more than any shard receives.
    let cache = SdfEvalCache::new(256 * 64, cell);
    assert!(cache.is_empty());
    assert_eq!(cache.len(), 0);
    assert_eq!(cache.hit_rate(), 0.0);

    let mut model: HashMap<[i32; 3], f32> = HashMap::new();
    let (mut hits, mut misses) = (0u64, 0u64);
    let mut compared = 0;
    // Sample on a grid finer than the cell: several points share a key.
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                let p = [
                    -1.5 + 0.25 * i as f32 + 0.06,
                    -1.5 + 0.25 * j as f32 - 0.07,
                    -0.5 + 0.1 * k as f32,
                ];
                let got = cache.get(p[0], p[1], p[2]);
                let want = model.get(&key(p, cell)).copied();
                assert_eq!(got.map(f32::to_bits), want.map(f32::to_bits), "{p:?}");
                if let Some(d) = got {
                    hits += 1;
                    // A hit returns the distance of the point that filled the
                    // cell, which is within one cell diagonal of the exact one.
                    let exact = eval(&node, Vec3::from(p));
                    assert!((d - exact).abs() <= cell * 3f32.sqrt());
                } else {
                    misses += 1;
                    let d = eval(&node, Vec3::from(p));
                    cache.put(p[0], p[1], p[2], d);
                    model.insert(key(p, cell), d);
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 13 * 13 * 5);
    assert!(hits > 0 && misses > 0);
    assert_eq!(cache.len(), model.len());
    assert!(!cache.is_empty());
    assert_eq!(cache.hit_rate(), hits as f64 / (hits + misses) as f64);
}

#[test]
fn put_overwrites_the_cell_value() {
    let cache = SdfEvalCache::new(4096, 0.5);
    cache.put(1.0, 1.0, 1.0, 0.25);
    cache.put(1.1, 0.9, 1.2, -0.75); // same cell: round(2.2)=2, round(1.8)=2, round(2.4)=2
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.get(1.0, 1.0, 1.0), Some(-0.75));
    assert_eq!(cache.get(1.3, 1.0, 1.0), None); // round(2.6) = 3: another cell
    assert_eq!(cache.hit_rate(), 0.5);
}

#[test]
fn eviction_respects_capacity_and_never_returns_a_foreign_value() {
    let capacity = 512; // 2 entries per shard
    let cache = SdfEvalCache::new(capacity, 1.0);
    let mut model: HashMap<[i32; 3], f32> = HashMap::new();
    for i in 0..4000i32 {
        let p = [i as f32, (i % 7) as f32, -(i / 7) as f32];
        let v = i as f32 * 0.5 - 3.0;
        cache.put(p[0], p[1], p[2], v);
        model.insert(key(p, 1.0), v);
        assert!(cache.len() <= capacity, "len {} after {i}", cache.len());
    }
    assert!(!cache.is_empty());
    let mut hits = 0;
    for (k, &v) in &model {
        if let Some(got) = cache.get(k[0] as f32, k[1] as f32, k[2] as f32) {
            assert_eq!(got, v, "{k:?}");
            hits += 1;
        }
    }
    assert_eq!(hits, cache.len());
}

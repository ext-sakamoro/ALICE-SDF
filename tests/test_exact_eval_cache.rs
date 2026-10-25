//! `ExactEvalCache`: exact keys, first-in-first-out eviction, and values independent of
//! the thread count (ALICE-Law G10)
//!
//! The expected eviction comes from a model written here (a queue of keys in insertion
//! order); the expected distances come from evaluating the tree directly.

use alice_sdf::{eval, ExactEvalCache, SdfNode};
use glam::Vec3;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, VecDeque};

fn lcg(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        s >> 33
    }
}

/// A point from a small grid, so that the sequence revisits points
fn point(r: u64) -> Vec3 {
    let c = |v: u64| (v % 9) as f32 * 0.25 - 1.0;
    Vec3::new(c(r), c(r / 9), c(r / 81))
}

const fn bits(p: Vec3) -> [u32; 3] {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

#[test]
fn a_hit_is_the_exact_point_only() {
    let c = ExactEvalCache::new(8);
    let p = Vec3::new(0.5, 0.25, 0.0);
    c.put(p, 1.5);
    assert_eq!(c.get(p), Some(1.5));
    // one ulp away, and the other sign of zero, are other points
    assert_eq!(
        c.get(Vec3::new(f32::from_bits(0.5f32.to_bits() + 1), 0.25, 0.0)),
        None
    );
    assert_eq!(c.get(Vec3::new(0.5, 0.25, -0.0)), None);
    assert_eq!(c.hits_and_misses(), (1, 2));
    // putting a present key replaces its value and keeps one entry
    c.put(p, 2.5);
    assert_eq!((c.get(p), c.len()), (Some(2.5), 1));
}

#[test]
fn a_zero_capacity_stores_nothing() {
    let c = ExactEvalCache::new(0);
    c.put(Vec3::ZERO, 1.0);
    assert!(c.is_empty());
    assert_eq!(c.get(Vec3::ZERO), None);
}

/// Runs one fixed sequence of puts and gets; returns the hits, misses and a digest of
/// the surviving entries, and checks the survivors against a FIFO model
fn run_sequence(capacity: usize) -> (u64, u64, String) {
    let c = ExactEvalCache::new(capacity);
    let mut model: VecDeque<[u32; 3]> = VecDeque::new();
    let mut values: BTreeMap<[u32; 3], f32> = BTreeMap::new();
    let mut next = lcg(7);
    for i in 0..5_000u32 {
        let p = point(next());
        let k = bits(p);
        if next().is_multiple_of(3) {
            let got = c.get(p);
            assert_eq!(got, values.get(&k).copied(), "step {i}");
        } else {
            let d = i as f32 * 0.5;
            c.put(p, d);
            if !values.contains_key(&k) {
                if model.len() == capacity {
                    let old = model.pop_front().expect("full");
                    values.remove(&old);
                }
                model.push_back(k);
            }
            values.insert(k, d);
        }
    }
    let entries = c.entries();
    assert_eq!(entries.len(), values.len());
    let mut h = Sha256::new();
    for (p, d) in entries {
        assert_eq!(values.get(&bits(p)), Some(&d));
        h.update(p.x.to_bits().to_le_bytes());
        h.update(p.y.to_bits().to_le_bytes());
        h.update(p.z.to_bits().to_le_bytes());
        h.update(d.to_bits().to_le_bytes());
    }
    let (hits, misses) = c.hits_and_misses();
    (hits, misses, format!("{:x}", h.finalize()))
}

#[test]
fn the_same_sequence_gives_the_same_hits_and_entries_and_evicts_first_in_first_out() {
    let a = run_sequence(64);
    let b = run_sequence(64);
    assert_eq!(a, b);
    assert!(a.0 > 0 && a.1 > 0, "the sequence must hit and miss: {a:?}");
}

#[test]
fn distances_through_the_cache_do_not_depend_on_the_thread_count() {
    let node = SdfNode::sphere(0.75).smooth_union(SdfNode::box3d(0.5, 0.25, 0.5), 0.1);
    let mut next = lcg(11);
    let points: Vec<Vec3> = (0..4_000).map(|_| point(next())).collect();
    let direct: Vec<u32> = points.iter().map(|&p| eval(&node, p).to_bits()).collect();
    for threads in [1, 2, 8] {
        // a small capacity, so entries are evicted while threads race
        let c = ExactEvalCache::new(32);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let got: Vec<u32> = pool.install(|| {
            use rayon::prelude::*;
            points
                .par_iter()
                .map(|&p| c.get_or_eval(p, |q| eval(&node, q)).to_bits())
                .collect()
        });
        assert_eq!(got, direct, "{threads} threads");
        for (p, d) in c.entries() {
            assert_eq!(d.to_bits(), eval(&node, p).to_bits());
        }
    }
}

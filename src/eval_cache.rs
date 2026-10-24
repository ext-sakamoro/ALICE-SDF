//! An exact-key cache of evaluated distances
//!
//! [`ExactEvalCache`] keys an entry by the bit pattern of the point (`x`, `y`, `z` as
//! `f32::to_bits`), so a hit returns the distance computed at exactly that point and
//! never another point's. The evaluators are deterministic, so a cached distance is the
//! one evaluating would give: the value a caller sees does not depend on what is cached,
//! on the order of the calls or on how threads interleave. Only the hit / miss counts
//! do.
//!
//! The cache holds at most `capacity` entries and evicts in insertion order (first in,
//! first out); putting a key that is already present replaces its value and keeps its
//! place. The entries live in ordered maps, with no hasher and no seed, so the same
//! sequence of calls on one thread gives the same hits, misses and entries on every run
//! and target.
//!
//! This is the cache of the Law path (ALICE-Law G10). `cache_bridge::SdfEvalCache`
//! (feature `sdf-cache`, the `alice-cache` crate, AGPL-3.0-or-later at the pinned version) quantises points to grid cells and
//! evicts by its dependency's hasher; it is outside the Law guarantee.

use glam::Vec3;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// The bit pattern of a point, the key of an entry
type Key = [u32; 3];

const fn key(p: Vec3) -> Key {
    [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
}

#[derive(Debug, Default)]
struct Inner {
    /// key -> (distance, insertion sequence number)
    entries: BTreeMap<Key, (f32, u64)>,
    /// insertion sequence number -> key, oldest first
    order: BTreeMap<u64, Key>,
    next: u64,
    hits: u64,
    misses: u64,
}

/// A bounded cache of distances keyed by the exact point (see the module documentation)
#[derive(Debug)]
pub struct ExactEvalCache {
    inner: Mutex<Inner>,
    capacity: usize,
}

impl ExactEvalCache {
    /// A cache that holds at most `capacity` distances (`0` stores nothing).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(Inner::default()),
            capacity,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // a poisoned lock still holds consistent maps: every update below completes
        // before it returns
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The cached distance at exactly `p`, if any (counts a hit or a miss).
    pub fn get(&self, p: Vec3) -> Option<f32> {
        let mut g = self.lock();
        let found = g.entries.get(&key(p)).map(|&(d, _)| d);
        if found.is_some() {
            g.hits += 1;
        } else {
            g.misses += 1;
        }
        found
    }

    /// Stores the distance at exactly `p`, evicting the oldest entry when full.
    pub fn put(&self, p: Vec3, distance: f32) {
        if self.capacity == 0 {
            return;
        }
        let k = key(p);
        let mut g = self.lock();
        if let Some(e) = g.entries.get_mut(&k) {
            e.0 = distance;
            return;
        }
        while g.entries.len() >= self.capacity {
            let Some((_, oldest)) = g.order.pop_first() else {
                break;
            };
            g.entries.remove(&oldest);
        }
        let seq = g.next;
        g.next += 1;
        g.entries.insert(k, (distance, seq));
        g.order.insert(seq, k);
    }

    /// The distance at `p`: the cached one, or `eval(p)`, which is then stored.
    pub fn get_or_eval(&self, p: Vec3, eval: impl FnOnce(Vec3) -> f32) -> f32 {
        if let Some(d) = self.get(p) {
            return d;
        }
        let d = eval(p);
        self.put(p, d);
        d
    }

    /// Number of cached distances.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Whether nothing is cached.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `(hits, misses)` counted by [`Self::get`] (and [`Self::get_or_eval`]).
    #[must_use]
    pub fn hits_and_misses(&self) -> (u64, u64) {
        let g = self.lock();
        (g.hits, g.misses)
    }

    /// The cached points and distances, in key order (the bit patterns of x, y, z).
    #[must_use]
    pub fn entries(&self) -> Vec<(Vec3, f32)> {
        self.lock()
            .entries
            .iter()
            .map(|(k, &(d, _))| {
                (
                    Vec3::new(
                        f32::from_bits(k[0]),
                        f32::from_bits(k[1]),
                        f32::from_bits(k[2]),
                    ),
                    d,
                )
            })
            .collect()
    }

    /// Removes every entry and resets the counts.
    pub fn clear(&self) {
        *self.lock() = Inner::default();
    }
}

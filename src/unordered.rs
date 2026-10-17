//! Hash collections that cannot be iterated (ALICE-Law G13)
//!
//! A `HashMap` / `HashSet` iterates in an order that is unspecified and differs between
//! runs, so a result built by iterating one can differ between runs. The library uses
//! hash collections only through these wrappers: lookup, insertion, removal and size,
//! and a few aggregates whose result does not depend on the order (a count, the
//! smallest key under a total order, the entries sorted by key). There is no
//! `iter`, no `IntoIterator` and no `Deref` to the inner collection, so a `for` loop
//! or an iterator chain over one does not compile; this module is the one place in
//! the library that names `HashMap` / `HashSet` (.github/clippy-lib/clippy.toml bans
//! them elsewhere).
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the one module that holds the hash collections; nothing here exposes their iteration order (ALICE-Law G13)"
)]

use std::borrow::Borrow;
use std::collections::{hash_map, HashMap, HashSet};
use std::hash::Hash;

/// A `HashMap` without iteration: see the module documentation
#[derive(Debug, Clone)]
pub struct UnorderedMap<K, V>(HashMap<K, V>);

impl<K, V> Default for UnorderedMap<K, V> {
    fn default() -> Self {
        Self(HashMap::new())
    }
}

impl<K: Eq + Hash, V> UnorderedMap<K, V> {
    /// An empty map
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    /// An empty map with room for `n` entries
    pub fn with_capacity(n: usize) -> Self {
        Self(HashMap::with_capacity(n))
    }

    /// Inserts, returning the previous value of the key
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.0.insert(k, v)
    }

    /// The value of a key
    pub fn get<Q: Eq + Hash + ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.0.get(k)
    }

    /// The value of a key, mutable
    pub fn get_mut<Q: Eq + Hash + ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.0.get_mut(k)
    }

    /// Whether the key is present
    pub fn contains_key<Q: Eq + Hash + ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.contains_key(k)
    }

    /// Removes a key, returning its value
    pub fn remove<Q: Eq + Hash + ?Sized>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.0.remove(k)
    }

    /// The entry of a key, for in-place insertion or update
    pub fn entry(&mut self, k: K) -> hash_map::Entry<'_, K, V> {
        self.0.entry(k)
    }

    /// Number of entries
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are no entries
    #[cfg(feature = "svo")] // the svo chunk cache is the one user
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Removes every entry
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// How many values satisfy `pred` (an integer count, the same in any order)
    pub fn count_values_where(&self, pred: impl Fn(&V) -> bool) -> usize {
        self.0.values().filter(|v| pred(v)).count()
    }

    /// The sum of `f` over the values: an integer, so the same in any order
    pub fn sum_values_by(&self, f: impl Fn(&V) -> usize) -> usize {
        self.0.values().map(f).sum()
    }

    /// Applies `f` to every value. `f` is `Fn`, so it cannot carry state from one value
    /// to the next: each value's result does not depend on the order
    pub fn update_each_value(&mut self, f: impl Fn(&mut V)) {
        for v in self.0.values_mut() {
            f(v);
        }
    }
}

impl<K: Eq + Hash + Ord, V> UnorderedMap<K, V> {
    /// The entries in key order
    pub fn sorted_entries(&self) -> Vec<(&K, &V)> {
        let mut out: Vec<(&K, &V)> = self.0.iter().collect();
        out.sort_unstable_by(|a, b| a.0.cmp(b.0));
        out
    }

    /// The entries in key order, values mutable
    pub fn sorted_entries_mut(&mut self) -> Vec<(&K, &mut V)> {
        let mut out: Vec<(&K, &mut V)> = self.0.iter_mut().collect();
        out.sort_unstable_by(|a, b| a.0.cmp(b.0));
        out
    }

    /// The keys in key order
    pub fn sorted_keys(&self) -> Vec<&K> {
        let mut out: Vec<&K> = self.0.keys().collect();
        out.sort_unstable();
        out
    }

    /// The key with the smallest `(by(key, value), key)`: a total order with the key
    /// breaking ties, so the answer does not depend on the iteration order
    #[cfg(feature = "svo")] // the svo chunk cache's eviction is the one user
    pub fn min_key_by<O: Ord>(&self, by: impl Fn(&K, &V) -> O) -> Option<&K> {
        self.0
            .iter()
            .map(|(k, v)| (by(k, v), k))
            .min()
            .map(|(_, k)| k)
    }
}

impl<K: Eq + Hash + Borrow<Q>, Q: Eq + Hash + ?Sized, V> std::ops::Index<&Q>
    for UnorderedMap<K, V>
{
    type Output = V;

    /// The value of a key; panics when the key is absent, as `HashMap` does
    fn index(&self, k: &Q) -> &V {
        &self.0[k]
    }
}

impl<K: Eq + Hash, V: Ord + Copy> UnorderedMap<K, V> {
    /// The largest value (integers and other totally ordered values; the same in any order)
    pub fn max_value(&self) -> Option<V> {
        self.0.values().copied().max()
    }
}

/// A `HashSet` without iteration: see the module documentation
#[derive(Debug, Clone)]
pub struct UnorderedSet<T>(HashSet<T>);

impl<T> Default for UnorderedSet<T> {
    fn default() -> Self {
        Self(HashSet::new())
    }
}

impl<T: Eq + Hash> UnorderedSet<T> {
    /// An empty set
    pub fn new() -> Self {
        Self(HashSet::new())
    }

    /// An empty set with room for `n` values
    pub fn with_capacity(n: usize) -> Self {
        Self(HashSet::with_capacity(n))
    }

    /// Inserts, returning whether the value was new
    pub fn insert(&mut self, v: T) -> bool {
        self.0.insert(v)
    }

    /// Whether the value is present
    pub fn contains<Q: Eq + Hash + ?Sized>(&self, v: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.0.contains(v)
    }

    /// Number of values
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Removes every value
    pub fn clear(&mut self) {
        self.0.clear();
    }

    /// Inserts every value of `it`
    pub fn extend(&mut self, it: impl IntoIterator<Item = T>) {
        self.0.extend(it);
    }
}

impl<T: Eq + Hash> FromIterator<T> for UnorderedSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(it: I) -> Self {
        Self(it.into_iter().collect())
    }
}

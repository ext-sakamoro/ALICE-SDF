//! Hash collections that cannot be iterated (ALICE-Law G13)
//!
//! A `HashMap` / `HashSet` iterates in an order that is unspecified and differs between
//! runs, so a result built by iterating one can differ between runs. The library uses
//! hash collections only through these wrappers: lookup, insertion, removal and size,
//! and a few aggregates that walk the entries in key order (so a callback with side
//! effects sees the same sequence on every run). There is no
//! `iter`, no `IntoIterator` and no `Deref` to the inner collection, so a `for` loop
//! or an iterator chain over one does not compile.
//!
//! The types are public so another crate can apply the same rule: it bans `HashMap` /
//! `HashSet` in its own library code (for example with clippy's `disallowed-types`) and
//! uses these wrappers instead. In this crate this module is the one place in the
//! library that names the std hash collections.
#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the one module that holds the hash collections; nothing here exposes their iteration order (ALICE-Law G13)"
)]

use std::borrow::Borrow;
use std::collections::{hash_map, HashMap, HashSet};
use std::hash::Hash;

/// A `HashMap` without iteration: see the module documentation
#[derive(Clone)]
pub struct UnorderedMap<K, V>(HashMap<K, V>);

/// The entries in key order: a derived `Debug` would print them in hash order, which
/// differs between runs
impl<K: Eq + Hash + Ord + std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug
    for UnorderedMap<K, V>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("UnorderedMap")
            .field(&DebugMap(self.sorted_entries()))
            .finish()
    }
}

/// `{k: v, ...}` for entries already in key order
struct DebugMap<'a, K, V>(Vec<(&'a K, &'a V)>);

impl<K: std::fmt::Debug, V: std::fmt::Debug> std::fmt::Debug for DebugMap<'_, K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.0.iter().copied()).finish()
    }
}

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
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Removes every entry
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<K: Eq + Hash + Ord, V> UnorderedMap<K, V> {
    // Every traversal of the values below walks them in key order: a callback is free
    // to have side effects (a `Cell`, a `RefCell`, a static), and what it sees is the
    // same sequence on every run

    /// How many values satisfy `pred`
    pub fn count_values_where(&self, pred: impl Fn(&V) -> bool) -> usize {
        self.sorted_entries()
            .into_iter()
            .filter(|(_, v)| pred(v))
            .count()
    }

    /// The sum of `f` over the values (an integer)
    pub fn sum_values_by(&self, f: impl Fn(&V) -> usize) -> usize {
        self.sorted_entries().into_iter().map(|(_, v)| f(v)).sum()
    }

    /// Applies `f` to every value, in key order
    pub fn update_each_value(&mut self, f: impl Fn(&mut V)) {
        for (_, v) in self.sorted_entries_mut() {
            f(v);
        }
    }

    /// The key with the smallest `(by(key, value), key)`, the key breaking ties;
    /// `by` is called in key order
    pub fn min_key_by<O: Ord>(&self, by: impl Fn(&K, &V) -> O) -> Option<&K> {
        self.sorted_entries()
            .into_iter()
            .map(|(k, v)| (by(k, v), k))
            .min()
            .map(|(_, k)| k)
    }

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

impl<K: Eq + Hash + Ord, V: Ord + Copy> UnorderedMap<K, V> {
    /// The largest value (integers and other totally ordered values; the same in any order)
    pub fn max_value(&self) -> Option<V> {
        self.0.values().copied().max()
    }
}

/// A `HashSet` without iteration: see the module documentation
#[derive(Clone)]
pub struct UnorderedSet<T>(HashSet<T>);

/// The values in order: a derived `Debug` would print them in hash order, which differs
/// between runs
impl<T: Eq + Hash + Ord + std::fmt::Debug> std::fmt::Debug for UnorderedSet<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut values: Vec<&T> = self.0.iter().collect();
        values.sort_unstable();
        f.debug_tuple("UnorderedSet")
            .field(&DebugSet(values))
            .finish()
    }
}

/// `{v, ...}` for values already in order
struct DebugSet<'a, T>(Vec<&'a T>);

impl<T: std::fmt::Debug> std::fmt::Debug for DebugSet<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.iter().copied()).finish()
    }
}

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

    /// Whether there are no values
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
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

#[cfg(test)]
mod tests {
    #[test]
    fn debug_prints_in_key_order_whatever_the_insertion_order() {
        let keys: Vec<u32> = (0..64).map(|i| (i * 37) % 64).collect();
        let mut a = UnorderedMap::new();
        let mut b = UnorderedMap::new();
        let mut sa = UnorderedSet::new();
        let mut sb = UnorderedSet::new();
        for &k in &keys {
            a.insert(k, k * 10);
            sa.insert(k);
        }
        for &k in keys.iter().rev() {
            b.insert(k, k * 10);
            sb.insert(k);
        }
        let want_map = format!(
            "UnorderedMap({{{}}})",
            (0..64u32)
                .map(|k| format!("{k}: {}", k * 10))
                .collect::<Vec<_>>()
                .join(", ")
        );
        let want_set = format!(
            "UnorderedSet({{{}}})",
            (0..64u32)
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        assert_eq!(format!("{a:?}"), want_map);
        assert_eq!(format!("{b:?}"), want_map);
        assert_eq!(format!("{sa:?}"), want_set);
        assert_eq!(format!("{sb:?}"), want_set);
    }

    use super::*;
    use std::cell::RefCell;

    /// Keys inserted out of order, many enough that the hash order is not key order
    fn map() -> UnorderedMap<u32, u32> {
        let mut m = UnorderedMap::new();
        for i in 0..64u32 {
            let k = (i * 37) % 64;
            m.insert(k, k * 10);
        }
        m
    }

    fn key_order() -> Vec<u32> {
        (0..64u32).map(|k| k * 10).collect()
    }

    /// What a callback with a side effect observes is the values in key order
    #[test]
    fn a_callback_sees_the_values_in_key_order() {
        let mut m = map();
        let seen = RefCell::new(Vec::new());
        let _ = m.count_values_where(|v| {
            seen.borrow_mut().extend([*v]);
            true
        });
        assert_eq!(seen.take(), key_order(), "count_values_where");
        let _ = m.sum_values_by(|v| {
            seen.borrow_mut().extend([*v]);
            0
        });
        assert_eq!(seen.take(), key_order(), "sum_values_by");
        m.update_each_value(|v| seen.borrow_mut().extend([*v]));
        assert_eq!(seen.take(), key_order(), "update_each_value");
        let got: Vec<u32> = m.sorted_entries().into_iter().map(|(_, v)| *v).collect();
        assert_eq!(got, key_order(), "sorted_entries");
    }

    #[test]
    fn min_key_by_calls_in_key_order_and_breaks_ties_by_key() {
        let m = map();
        let seen = RefCell::new(Vec::new());
        let k = m.min_key_by(|k, _| {
            seen.borrow_mut().extend([*k]);
            0u8
        });
        assert_eq!(k, Some(&0));
        assert_eq!(seen.take(), (0..64u32).collect::<Vec<_>>());
    }
}

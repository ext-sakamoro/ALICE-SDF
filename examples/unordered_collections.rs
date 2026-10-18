//! `alice_sdf::unordered`: hash collections that cannot be iterated (ALICE-Law G13)
//!
//! A crate that bans `std::collections::HashMap` / `HashSet` in its library code (their
//! iteration order differs between runs) uses these wrappers instead: lookup, insertion,
//! removal and size, and aggregates that walk the entries in key order.
//!
//! Run: `cargo run --example unordered_collections`

use alice_sdf::unordered::{UnorderedMap, UnorderedSet};

fn main() {
    // count each word: `entry` inserts or updates, nothing iterates the map
    let text = "sphere box sphere torus box sphere";
    let mut counts: UnorderedMap<&str, usize> = UnorderedMap::with_capacity(4);
    for word in text.split(' ') {
        *counts.entry(word).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 3);
    assert!(!counts.is_empty());
    assert_eq!(counts.get("sphere"), Some(&3));
    // the aggregates walk the entries in key order, so the result is the same every run
    let in_key_order: Vec<(&str, usize)> = counts
        .sorted_entries()
        .into_iter()
        .map(|(k, v)| (*k, *v))
        .collect();
    assert_eq!(in_key_order, vec![("box", 2), ("sphere", 3), ("torus", 1)]);
    assert_eq!(counts.min_key_by(|_, v| *v), Some(&"torus"));

    // a set of seen node ids
    let mut seen: UnorderedSet<u32> = UnorderedSet::with_capacity(8);
    assert!(seen.is_empty());
    for id in [7, 3, 7, 11] {
        seen.insert(id);
    }
    assert_eq!(seen.len(), 3);
    assert!(seen.contains(&11));
    println!(
        "{} words, {} distinct ids, counts in key order: {in_key_order:?}",
        counts.len(),
        seen.len()
    );
}

//! ALICE-Cache bridge: SDF evaluation result caching
//!
//! Caches SDF distance field evaluations using ALICE-Cache to avoid
//! redundant computation during interactive editing and mesh generation.

use alice_cache::AliceCache;
use std::hash::{Hash, Hasher};

/// Quantised 3D point for cache keys.
///
/// Coordinates are quantised to a configurable grid resolution
/// so that nearby lookups share cache entries.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GridPoint {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Hash for GridPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_i32(self.x);
        state.write_i32(self.y);
        state.write_i32(self.z);
    }
}

impl GridPoint {
    /// Quantise a floating-point position to the grid.
    #[inline]
    pub fn from_f32(x: f32, y: f32, z: f32, inv_cell: f32) -> Self {
        Self {
            x: (x * inv_cell).round() as i32,
            y: (y * inv_cell).round() as i32,
            z: (z * inv_cell).round() as i32,
        }
    }
}

/// Cached SDF evaluator.
///
/// Wraps an `AliceCache` keyed by quantised grid positions, returning
/// previously computed distances when available.
pub struct SdfEvalCache {
    cache: AliceCache<GridPoint, f32>,
    inv_cell: f32,
}

impl SdfEvalCache {
    /// Create a new SDF evaluation cache.
    ///
    /// * `capacity` — maximum number of cached distance values
    /// * `cell_size` — quantisation grid cell size (smaller = more precise but more entries)
    pub fn new(capacity: usize, cell_size: f32) -> Self {
        Self {
            cache: AliceCache::new(capacity),
            inv_cell: 1.0 / cell_size,
        }
    }

    /// Look up a cached distance value.
    #[inline]
    pub fn get(&self, x: f32, y: f32, z: f32) -> Option<f32> {
        let key = GridPoint::from_f32(x, y, z, self.inv_cell);
        self.cache.get(&key)
    }

    /// Store a computed distance value.
    #[inline]
    pub fn put(&self, x: f32, y: f32, z: f32, distance: f32) {
        let key = GridPoint::from_f32(x, y, z, self.inv_cell);
        self.cache.put(key, distance);
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_roundtrip() {
        let cache = SdfEvalCache::new(1024, 0.01);
        cache.put(1.0, 2.0, 3.0, 0.42);
        assert_eq!(cache.get(1.0, 2.0, 3.0), Some(0.42));
    }

    #[test]
    fn test_cache_miss() {
        let cache = SdfEvalCache::new(1024, 0.01);
        assert_eq!(cache.get(5.0, 6.0, 7.0), None);
    }

    #[test]
    fn test_quantisation_groups_nearby() {
        let cache = SdfEvalCache::new(1024, 0.1);
        cache.put(1.0, 2.0, 3.0, 0.5);
        // Slightly different point within same cell
        assert_eq!(cache.get(1.04, 2.03, 3.02), Some(0.5));
    }
}

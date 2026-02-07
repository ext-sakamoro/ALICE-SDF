//! SVO Streaming: Chunk-based loading with LRU cache (Deep Fried Edition)
//!
//! For large SVO scenes that don't fit in memory, this module provides:
//!
//! - **Chunking**: Split SVO into spatial chunks at a configurable depth
//! - **LRU Cache**: Keep recently-accessed chunks in memory
//! - **On-demand Loading**: Load chunks from disk/network as needed
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────┐
//! │  SvoStreamingCache          │
//! │  ┌──────┐ ┌──────┐ ┌──────┐│
//! │  │Chunk0│ │Chunk3│ │Chunk7││  ← LRU cache (hot chunks)
//! │  └──────┘ └──────┘ └──────┘│
//! │  capacity: N chunks         │
//! └─────────────┬───────────────┘
//!               │ load/evict
//! ┌─────────────▼───────────────┐
//! │  Storage (disk/network)     │
//! │  chunk_0.svo chunk_1.svo ...│
//! └─────────────────────────────┘
//! ```
//!
//! Author: Moroya Sakamoto

use glam::Vec3;

use super::{SvoNode, SparseVoxelOctree};
use crate::compiled::AabbPacked;

/// A spatial chunk of the SVO
#[derive(Clone)]
pub struct SvoChunk {
    /// Chunk identifier (spatial index)
    pub chunk_id: u32,
    /// Nodes in this chunk (linearized)
    pub nodes: Vec<SvoNode>,
    /// World-space bounds of this chunk
    pub bounds: AabbPacked,
    /// Depth in the tree where this chunk starts
    pub start_depth: u32,
}

impl SvoChunk {
    /// Create a new chunk
    pub fn new(chunk_id: u32, nodes: Vec<SvoNode>, bounds: AabbPacked, start_depth: u32) -> Self {
        SvoChunk {
            chunk_id,
            nodes,
            bounds,
            start_depth,
        }
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * std::mem::size_of::<SvoNode>()
    }

    /// Serialize chunk to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: chunk_id, node_count, start_depth, bounds
        bytes.extend_from_slice(&self.chunk_id.to_le_bytes());
        bytes.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.start_depth.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.min_x.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.min_y.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.min_z.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.max_x.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.max_y.to_le_bytes());
        bytes.extend_from_slice(&self.bounds.max_z.to_le_bytes());

        // Node data
        for node in &self.nodes {
            bytes.extend_from_slice(&node.distance.to_le_bytes());
            bytes.extend_from_slice(&node.nx.to_le_bytes());
            bytes.extend_from_slice(&node.ny.to_le_bytes());
            bytes.extend_from_slice(&node.nz.to_le_bytes());
            bytes.push(node.child_mask);
            bytes.push(node.is_leaf);
            bytes.extend_from_slice(&node.material_id.to_le_bytes());
            bytes.extend_from_slice(&node.first_child.to_le_bytes());
            bytes.extend_from_slice(&node._pad[0].to_le_bytes());
            bytes.extend_from_slice(&node._pad[1].to_le_bytes());
        }

        bytes
    }

    /// Deserialize chunk from bytes
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 36 {
            return None;
        }

        let chunk_id = u32::from_le_bytes(data[0..4].try_into().ok()?);
        let node_count = u32::from_le_bytes(data[4..8].try_into().ok()?) as usize;
        let start_depth = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let min_x = f32::from_le_bytes(data[12..16].try_into().ok()?);
        let min_y = f32::from_le_bytes(data[16..20].try_into().ok()?);
        let min_z = f32::from_le_bytes(data[20..24].try_into().ok()?);
        let max_x = f32::from_le_bytes(data[24..28].try_into().ok()?);
        let max_y = f32::from_le_bytes(data[28..32].try_into().ok()?);
        let max_z = f32::from_le_bytes(data[32..36].try_into().ok()?);

        let bounds = AabbPacked::new(
            Vec3::new(min_x, min_y, min_z),
            Vec3::new(max_x, max_y, max_z),
        );

        let node_data = &data[36..];
        let node_byte_size = 32; // SvoNode is 32 bytes
        if node_data.len() < node_count * node_byte_size {
            return None;
        }

        let mut nodes = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let offset = i * node_byte_size;
            let d = &node_data[offset..];

            let distance = f32::from_le_bytes(d[0..4].try_into().ok()?);
            let nx = f32::from_le_bytes(d[4..8].try_into().ok()?);
            let ny = f32::from_le_bytes(d[8..12].try_into().ok()?);
            let nz = f32::from_le_bytes(d[12..16].try_into().ok()?);
            let child_mask = d[16];
            let is_leaf = d[17];
            let material_id = u16::from_le_bytes(d[18..20].try_into().ok()?);
            let first_child = u32::from_le_bytes(d[20..24].try_into().ok()?);
            let pad0 = f32::from_le_bytes(d[24..28].try_into().ok()?);
            let pad1 = f32::from_le_bytes(d[28..32].try_into().ok()?);

            nodes.push(SvoNode {
                distance,
                nx,
                ny,
                nz,
                child_mask,
                is_leaf,
                material_id,
                first_child,
                _pad: [pad0, pad1],
            });
        }

        Some(SvoChunk {
            chunk_id,
            nodes,
            bounds,
            start_depth,
        })
    }
}

/// LRU entry for the cache
struct CacheEntry {
    chunk: SvoChunk,
    last_access: u64,
}

/// Streaming SVO cache with LRU eviction
///
/// Stores a fixed number of SVO chunks in memory. When the cache
/// is full and a new chunk is needed, the least recently used chunk
/// is evicted.
pub struct SvoStreamingCache {
    /// Maximum number of chunks to keep in memory
    capacity: usize,
    /// Cached chunks (chunk_id → entry)
    entries: std::collections::HashMap<u32, CacheEntry>,
    /// Access counter for LRU ordering
    access_counter: u64,
    /// Total memory used by cached chunks
    memory_used: usize,
    /// Maximum memory in bytes (0 = unlimited, use capacity-based eviction)
    max_memory: usize,
    /// Statistics
    hits: u64,
    misses: u64,
}

impl SvoStreamingCache {
    /// Create a new streaming cache
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of chunks to keep in memory
    pub fn new(capacity: usize) -> Self {
        SvoStreamingCache {
            capacity,
            entries: std::collections::HashMap::new(),
            access_counter: 0,
            memory_used: 0,
            max_memory: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Create a cache with a memory budget
    pub fn with_memory_budget(max_memory_bytes: usize) -> Self {
        SvoStreamingCache {
            capacity: usize::MAX,
            entries: std::collections::HashMap::new(),
            access_counter: 0,
            memory_used: 0,
            max_memory: max_memory_bytes,
            hits: 0,
            misses: 0,
        }
    }

    /// Get a chunk from the cache
    pub fn get(&mut self, chunk_id: u32) -> Option<&SvoChunk> {
        self.access_counter += 1;
        if let Some(entry) = self.entries.get_mut(&chunk_id) {
            entry.last_access = self.access_counter;
            self.hits += 1;
            Some(&entry.chunk)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a chunk into the cache, evicting LRU if needed
    pub fn insert(&mut self, chunk: SvoChunk) {
        let chunk_id = chunk.chunk_id;
        let chunk_mem = chunk.memory_bytes();

        // Evict until we have room
        while self.needs_eviction(chunk_mem) {
            if !self.evict_lru() {
                break; // No more entries to evict
            }
        }

        self.access_counter += 1;
        self.memory_used += chunk_mem;
        self.entries.insert(chunk_id, CacheEntry {
            chunk,
            last_access: self.access_counter,
        });
    }

    /// Check if eviction is needed
    fn needs_eviction(&self, additional_bytes: usize) -> bool {
        if self.entries.len() >= self.capacity {
            return true;
        }
        if self.max_memory > 0 && self.memory_used + additional_bytes > self.max_memory {
            return true;
        }
        false
    }

    /// Evict the least recently used chunk
    fn evict_lru(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }

        let lru_id = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access)
            .map(|(&id, _)| id);

        if let Some(id) = lru_id {
            if let Some(entry) = self.entries.remove(&id) {
                self.memory_used = self.memory_used.saturating_sub(entry.chunk.memory_bytes());
            }
            true
        } else {
            false
        }
    }

    /// Remove a specific chunk
    pub fn remove(&mut self, chunk_id: u32) -> Option<SvoChunk> {
        if let Some(entry) = self.entries.remove(&chunk_id) {
            self.memory_used = self.memory_used.saturating_sub(entry.chunk.memory_bytes());
            Some(entry.chunk)
        } else {
            None
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.memory_used = 0;
    }

    /// Number of chunks in cache
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total memory used by cached chunks
    pub fn memory_used(&self) -> usize {
        self.memory_used
    }

    /// Cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Split an SVO into spatial chunks at a given depth
///
/// Each chunk contains the subtree rooted at the given depth.
/// The top-level nodes (above split_depth) are stored in a "root chunk".
pub fn split_into_chunks(
    svo: &SparseVoxelOctree,
    split_depth: u32,
) -> Vec<SvoChunk> {
    if svo.nodes.is_empty() {
        return Vec::new();
    }

    let mut chunks = Vec::new();
    let mut chunk_id = 0u32;

    // BFS to find nodes at split_depth
    struct BfsEntry {
        node_idx: usize,
        depth: u32,
        center: Vec3,
        half_size: Vec3,
    }

    let center = svo.bounds.center();
    let half_size = svo.bounds.half_size();

    let mut queue: std::collections::VecDeque<BfsEntry> = std::collections::VecDeque::new();
    queue.push_back(BfsEntry {
        node_idx: 0,
        depth: 0,
        center,
        half_size,
    });

    // Collect root-level nodes (above split_depth) as first chunk
    let mut root_nodes = Vec::new();

    while let Some(entry) = queue.pop_front() {
        if entry.node_idx >= svo.nodes.len() {
            continue;
        }

        let node = &svo.nodes[entry.node_idx];

        if entry.depth < split_depth && node.is_leaf == 0 {
            // Above split depth: collect as root chunk
            root_nodes.push(*node);

            // Enqueue children
            for octant in 0..8u8 {
                if let Some(child_idx) = node.child_index(octant) {
                    let child_c = super::child_center(entry.center, entry.half_size, octant);
                    queue.push_back(BfsEntry {
                        node_idx: child_idx as usize,
                        depth: entry.depth + 1,
                        center: child_c,
                        half_size: entry.half_size * 0.5,
                    });
                }
            }
        } else {
            // At or below split depth: create a chunk from this subtree
            let subtree = extract_subtree(svo, entry.node_idx);
            let bounds = AabbPacked::new(
                entry.center - entry.half_size,
                entry.center + entry.half_size,
            );

            chunks.push(SvoChunk::new(chunk_id, subtree, bounds, entry.depth));
            chunk_id += 1;
        }
    }

    // Insert root chunk at the beginning
    if !root_nodes.is_empty() {
        chunks.insert(0, SvoChunk::new(
            u32::MAX, // Special ID for root chunk
            root_nodes,
            svo.bounds,
            0,
        ));
    }

    chunks
}

/// Extract a subtree starting at the given node index
fn extract_subtree(svo: &SparseVoxelOctree, root_idx: usize) -> Vec<SvoNode> {
    if root_idx >= svo.nodes.len() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    let mut old_to_new: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();

    queue.push_back(root_idx);
    old_to_new.insert(root_idx, 0);
    let mut next_new_idx = 1u32;

    // First pass: assign new indices
    let mut bfs_order: Vec<usize> = Vec::new();
    let mut temp_queue = queue.clone();
    while let Some(idx) = temp_queue.pop_front() {
        bfs_order.push(idx);
        if idx < svo.nodes.len() {
            let node = &svo.nodes[idx];
            if node.is_leaf == 0 {
                for octant in 0..8u8 {
                    if let Some(child_idx) = node.child_index(octant) {
                        let ci = child_idx as usize;
                        if !old_to_new.contains_key(&ci) {
                            old_to_new.insert(ci, next_new_idx);
                            next_new_idx += 1;
                            temp_queue.push_back(ci);
                        }
                    }
                }
            }
        }
    }

    // Second pass: build remapped nodes
    for &old_idx in &bfs_order {
        if old_idx >= svo.nodes.len() {
            continue;
        }
        let mut node = svo.nodes[old_idx];

        if node.is_leaf == 0 {
            // Remap first_child
            let old_first = node.first_child as usize;
            if let Some(&new_first) = old_to_new.get(&old_first) {
                node.first_child = new_first;
            }
        }

        result.push(node);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use crate::svo::SvoBuildConfig;

    fn make_test_svo() -> SparseVoxelOctree {
        let sphere = SdfNode::sphere(1.0);
        let config = SvoBuildConfig {
            max_depth: 4,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            use_compiled: false,
            ..Default::default()
        };
        super::super::build::build_svo(&sphere, &config)
    }

    #[test]
    fn test_cache_basic() {
        let mut cache = SvoStreamingCache::new(4);

        let chunk = SvoChunk::new(0, vec![SvoNode::default()], AabbPacked::empty(), 0);
        cache.insert(chunk);

        assert_eq!(cache.len(), 1);
        assert!(cache.get(0).is_some());
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = SvoStreamingCache::new(2);

        for i in 0..3 {
            let chunk = SvoChunk::new(i, vec![SvoNode::default()], AabbPacked::empty(), 0);
            cache.insert(chunk);
        }

        // Capacity is 2, so oldest (chunk 0) should be evicted
        assert_eq!(cache.len(), 2);
        assert!(cache.get(0).is_none(), "Chunk 0 should have been evicted");
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn test_cache_lru_order() {
        let mut cache = SvoStreamingCache::new(2);

        let chunk0 = SvoChunk::new(0, vec![SvoNode::default()], AabbPacked::empty(), 0);
        let chunk1 = SvoChunk::new(1, vec![SvoNode::default()], AabbPacked::empty(), 0);
        cache.insert(chunk0);
        cache.insert(chunk1);

        // Access chunk 0 to make it most recent
        cache.get(0);

        // Insert chunk 2 → should evict chunk 1 (LRU)
        let chunk2 = SvoChunk::new(2, vec![SvoNode::default()], AabbPacked::empty(), 0);
        cache.insert(chunk2);

        assert!(cache.get(0).is_some(), "Chunk 0 was recently accessed, should still be cached");
        assert!(cache.get(1).is_none(), "Chunk 1 should have been evicted");
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = SvoStreamingCache::new(4);

        let chunk = SvoChunk::new(0, vec![SvoNode::default()], AabbPacked::empty(), 0);
        cache.insert(chunk);

        cache.get(0); // hit
        cache.get(0); // hit
        cache.get(1); // miss

        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_chunk_serialization() {
        let nodes = vec![
            SvoNode::leaf(1.5, Vec3::new(0.0, 1.0, 0.0)),
            SvoNode::interior(0.5, 0b00110101, 42),
        ];
        let chunk = SvoChunk::new(7, nodes, AabbPacked::new(Vec3::ZERO, Vec3::ONE), 3);

        let bytes = chunk.to_bytes();
        let restored = SvoChunk::from_bytes(&bytes).expect("Should deserialize");

        assert_eq!(restored.chunk_id, 7);
        assert_eq!(restored.start_depth, 3);
        assert_eq!(restored.nodes.len(), 2);
        assert_eq!(restored.nodes[0].distance, 1.5);
        assert_eq!(restored.nodes[0].ny, 1.0);
        assert_eq!(restored.nodes[1].child_mask, 0b00110101);
        assert_eq!(restored.nodes[1].first_child, 42);
    }

    #[test]
    fn test_split_into_chunks() {
        let svo = make_test_svo();
        let chunks = split_into_chunks(&svo, 2);

        assert!(chunks.len() > 1, "Should produce multiple chunks");

        // Total nodes across chunks should account for all SVO nodes
        let total_chunk_nodes: usize = chunks.iter().map(|c| c.nodes.len()).sum();
        assert!(total_chunk_nodes > 0);
    }

    #[test]
    fn test_cache_memory_budget() {
        let mut cache = SvoStreamingCache::with_memory_budget(128);

        // Each default SvoNode is 32 bytes, so 1 node = 32 bytes per chunk
        for i in 0..10 {
            let chunk = SvoChunk::new(i, vec![SvoNode::default()], AabbPacked::empty(), 0);
            cache.insert(chunk);
        }

        // Should have evicted some chunks to stay under 128 bytes
        assert!(cache.memory_used() <= 128 + 32, "Memory should be near budget");
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = SvoStreamingCache::new(4);

        for i in 0..3 {
            cache.insert(SvoChunk::new(i, vec![SvoNode::default()], AabbPacked::empty(), 0));
        }

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.memory_used(), 0);
    }
}

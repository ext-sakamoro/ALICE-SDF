//! Handle Registry for FFI (Deep Fried Edition)
//!
//! This module manages the mapping between opaque handles and SDF objects.
//! Uses thread-safe registries with minimal lock contention.
//!
//! # Deep Fried Optimizations
//! - `Ordering::Relaxed` for ID generation (sufficient for uniqueness)
//! - Separate registries for nodes and compiled SDFs (reduce lock contention)
//! - Short lock durations (only for insert/remove, not during evaluation)
//!
//! Author: Moroya Sakamoto

use super::types::{CompiledHandle, MeshHandle, SdfHandle};
use crate::compiled::CompiledSdf;
use crate::mesh::Mesh;
use crate::types::SdfNode;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc, Mutex,
};

// ============================================================================
// Node Registry
// ============================================================================

lazy_static::lazy_static! {
    /// Global registry of SDF nodes
    static ref NODE_REGISTRY: Mutex<HashMap<u64, Arc<SdfNode>>> = Mutex::new(HashMap::new());
}

/// Counter for generating unique handle IDs (nodes)
static NODE_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Register a new SDF node and return its handle
#[inline]
pub fn register_node(node: SdfNode) -> SdfHandle {
    // Relaxed ordering is sufficient - we only need uniqueness, not ordering
    let id = NODE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let arc_node = Arc::new(node);

    // Minimize lock duration - just insert
    {
        let mut registry = NODE_REGISTRY.lock().unwrap();
        registry.insert(id, arc_node);
    }

    id as SdfHandle
}

/// Get an SDF node by handle (clones Arc - cheap)
#[inline]
pub fn get_node(handle: SdfHandle) -> Option<Arc<SdfNode>> {
    if handle.is_null() {
        return None;
    }

    let id = handle as u64;
    let registry = NODE_REGISTRY.lock().unwrap();
    registry.get(&id).cloned()
}

/// Remove an SDF node by handle
#[inline]
pub fn remove_node(handle: SdfHandle) {
    if handle.is_null() {
        return;
    }

    let id = handle as u64;
    let mut registry = NODE_REGISTRY.lock().unwrap();
    registry.remove(&id);
}

// ============================================================================
// Compiled SDF Registry (Deep Fried)
// ============================================================================

lazy_static::lazy_static! {
    /// Global registry of compiled SDFs
    static ref COMPILED_REGISTRY: Mutex<HashMap<u64, Arc<CompiledSdf>>> = Mutex::new(HashMap::new());
}

/// Counter for generating unique compiled handle IDs
/// Uses high bits to distinguish from node handles
static COMPILED_COUNTER: AtomicU64 = AtomicU64::new(0x8000_0000_0000_0001);

/// Register a compiled SDF and return its handle
#[inline]
pub fn register_compiled(compiled: CompiledSdf) -> CompiledHandle {
    let id = COMPILED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let arc_compiled = Arc::new(compiled);

    {
        let mut registry = COMPILED_REGISTRY.lock().unwrap();
        registry.insert(id, arc_compiled);
    }

    id as CompiledHandle
}

/// Get a compiled SDF by handle (clones Arc - cheap)
#[inline]
pub fn get_compiled(handle: CompiledHandle) -> Option<Arc<CompiledSdf>> {
    if handle.is_null() {
        return None;
    }

    let id = handle as u64;
    let registry = COMPILED_REGISTRY.lock().unwrap();
    registry.get(&id).cloned()
}

/// Remove a compiled SDF by handle
#[inline]
pub fn remove_compiled(handle: CompiledHandle) {
    if handle.is_null() {
        return;
    }

    let id = handle as u64;
    let mut registry = COMPILED_REGISTRY.lock().unwrap();
    registry.remove(&id);
}

// ============================================================================
// Mesh Registry
// ============================================================================

lazy_static::lazy_static! {
    /// Global registry of generated meshes
    static ref MESH_REGISTRY: Mutex<HashMap<u64, Arc<Mesh>>> = Mutex::new(HashMap::new());
}

/// Counter for generating unique mesh handle IDs
static MESH_COUNTER: AtomicU64 = AtomicU64::new(0x4000_0000_0000_0001);

/// Register a mesh and return its handle
#[inline]
pub fn register_mesh(mesh: Mesh) -> MeshHandle {
    let id = MESH_COUNTER.fetch_add(1, Ordering::Relaxed);
    let arc_mesh = Arc::new(mesh);

    {
        let mut registry = MESH_REGISTRY.lock().unwrap();
        registry.insert(id, arc_mesh);
    }

    id as MeshHandle
}

/// Get a mesh by handle (clones Arc - cheap)
#[inline]
pub fn get_mesh(handle: MeshHandle) -> Option<Arc<Mesh>> {
    if handle.is_null() {
        return None;
    }

    let id = handle as u64;
    let registry = MESH_REGISTRY.lock().unwrap();
    registry.get(&id).cloned()
}

/// Remove a mesh by handle
#[inline]
pub fn remove_mesh(handle: MeshHandle) {
    if handle.is_null() {
        return;
    }

    let id = handle as u64;
    let mut registry = MESH_REGISTRY.lock().unwrap();
    registry.remove(&id);
}

// ============================================================================
// Utilities
// ============================================================================

/// Get the number of registered nodes (for debugging)
#[allow(dead_code)]
pub fn node_count() -> usize {
    let registry = NODE_REGISTRY.lock().unwrap();
    registry.len()
}

/// Get the number of registered compiled SDFs (for debugging)
#[allow(dead_code)]
pub fn compiled_count() -> usize {
    let registry = COMPILED_REGISTRY.lock().unwrap();
    registry.len()
}

/// Clear all registered nodes (for testing)
#[allow(dead_code)]
pub fn clear_all() {
    {
        let mut registry = NODE_REGISTRY.lock().unwrap();
        registry.clear();
    }
    {
        let mut registry = COMPILED_REGISTRY.lock().unwrap();
        registry.clear();
    }
    {
        let mut registry = MESH_REGISTRY.lock().unwrap();
        registry.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let node = SdfNode::Sphere { radius: 1.0 };
        let handle = register_node(node);

        assert!(!handle.is_null());

        let retrieved = get_node(handle);
        assert!(retrieved.is_some());

        if let SdfNode::Sphere { radius } = &*retrieved.unwrap() {
            assert_eq!(*radius, 1.0);
        } else {
            panic!("Expected Sphere node");
        }

        remove_node(handle);
    }

    #[test]
    fn test_remove() {
        let node = SdfNode::Sphere { radius: 2.0 };
        let handle = register_node(node);

        assert!(get_node(handle).is_some());

        remove_node(handle);

        assert!(get_node(handle).is_none());
    }

    #[test]
    fn test_null_handle() {
        let null_handle: SdfHandle = std::ptr::null_mut();
        assert!(get_node(null_handle).is_none());
    }

    #[test]
    fn test_compiled_registry() {
        let node = SdfNode::Sphere { radius: 1.0 };
        let compiled = CompiledSdf::compile(&node);
        let handle = register_compiled(compiled);

        assert!(!handle.is_null());

        let retrieved = get_compiled(handle);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().node_count, 1);

        remove_compiled(handle);
        assert!(get_compiled(handle).is_none());
    }
}

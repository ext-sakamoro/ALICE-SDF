//! ASDF JSON format (Deep Fried Edition)
//!
//! Human-readable JSON representation of SDF trees.
//!
//! # Deep Fried Optimizations
//! - **Streaming I/O**: Uses `serde_json::to_writer`/`from_reader` with `BufWriter`/`BufReader`.
//! - **Reduced Allocations**: Eliminates intermediate Strings for file I/O.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::types::SdfTree;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

/// Save an SDF tree to JSON format (Streaming)
///
/// Writes directly to file buffer, avoiding intermediate String allocation.
pub fn save_asdf_json(tree: &SdfTree, path: impl AsRef<Path>) -> Result<(), IoError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);

    // Write directly to file buffer (pretty for readability)
    serde_json::to_writer_pretty(writer, tree)
        .map_err(|e| IoError::Serialization(e.to_string()))?;

    Ok(())
}

/// Load an SDF tree from JSON format (Streaming)
///
/// Reads directly from file buffer, avoiding intermediate String allocation.
pub fn load_asdf_json(path: impl AsRef<Path>) -> Result<SdfTree, IoError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Read directly from file buffer
    let tree: SdfTree =
        serde_json::from_reader(reader).map_err(|e| IoError::Serialization(e.to_string()))?;

    Ok(tree)
}

/// Serialize an SDF tree to JSON string (Memory-bound, prefer file I/O)
#[allow(dead_code)]
pub fn to_json_string(tree: &SdfTree) -> Result<String, IoError> {
    serde_json::to_string_pretty(tree).map_err(|e| IoError::Serialization(e.to_string()))
}

/// Parse an SDF tree from JSON string
#[allow(dead_code)]
pub fn from_json_string(json: &str) -> Result<SdfTree, IoError> {
    serde_json::from_str(json).map_err(|e| IoError::Serialization(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SdfMetadata, SdfNode};
    use std::fs;

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_json_{}", name));
        path
    }

    #[test]
    fn test_save_load_simple() {
        let tree = SdfTree::new(SdfNode::sphere(1.5));
        let path = temp_path("simple.asdf.json");

        save_asdf_json(&tree, &path).unwrap();
        let loaded = load_asdf_json(&path).unwrap();

        assert_eq!(loaded.node_count(), 1);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_with_metadata() {
        let mut metadata = SdfMetadata::default();
        metadata.name = Some("Test Model".to_string());
        metadata.author = Some("Test Author".to_string());

        let tree = SdfTree::with_metadata(SdfNode::sphere(1.0), metadata);
        let path = temp_path("metadata.asdf.json");

        save_asdf_json(&tree, &path).unwrap();
        let loaded = load_asdf_json(&path).unwrap();

        assert!(loaded.metadata.is_some());
        let meta = loaded.metadata.unwrap();
        assert_eq!(meta.name, Some("Test Model".to_string()));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_json_string_roundtrip() {
        let tree = SdfTree::new(SdfNode::sphere(1.0).union(SdfNode::box3d(2.0, 2.0, 2.0)));

        let json = to_json_string(&tree).unwrap();
        let parsed = from_json_string(&json).unwrap();

        assert_eq!(parsed.node_count(), tree.node_count());
    }

    #[test]
    fn test_json_readability() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let json = to_json_string(&tree).unwrap();

        // Should be valid JSON and contain expected fields
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"root\""));
        assert!(json.contains("\"Sphere\""));
    }

    #[test]
    fn test_complex_tree() {
        let shape = SdfNode::sphere(1.0)
            .subtract(SdfNode::box3d(1.5, 1.5, 1.5))
            .smooth_union(SdfNode::cylinder(0.5, 2.0), 0.2)
            .translate(1.0, 2.0, 3.0)
            .rotate_euler(0.1, 0.2, 0.3);

        let tree = SdfTree::new(shape);
        let json = to_json_string(&tree).unwrap();
        let parsed = from_json_string(&json).unwrap();

        assert_eq!(parsed.node_count(), tree.node_count());
    }

    #[test]
    fn test_streaming_large_json() {
        // Test streaming with a larger tree
        let mut shape = SdfNode::sphere(1.0);
        for i in 0..10 {
            shape = shape.union(SdfNode::box3d(1.0, 1.0, 1.0).translate(i as f32, 0.0, 0.0));
        }
        let tree = SdfTree::new(shape);
        let path = temp_path("streaming_large.asdf.json");

        save_asdf_json(&tree, &path).unwrap();
        let loaded = load_asdf_json(&path).unwrap();

        assert_eq!(loaded.node_count(), tree.node_count());
        fs::remove_file(&path).ok();
    }
}

//! ASDF JSON format (Deep Fried Edition)
//!
//! Human-readable JSON representation of SDF trees.
//!
//! # Semantics identifier (ALICE-Law G8)
//!
//! A `.asdf.json` document is the tree's JSON object with one more top-level field,
//! `"alice:semantics_id"` ([`crate::provenance::KEY`]): the 64 hex digits of the identifier
//! of the arithmetic the tree was written under (this crate's `SDF_SEMANTICS_ID` unless a
//! `*_with_provenance` writer is given another). The readers check it: [`load_asdf_json`] and
//! [`from_json_string`] refuse a document without one ([`IoError::Unpinned`], a document
//! written before 7.0.0) and one with another identifier ([`IoError::SemanticsMismatch`]);
//! [`load_asdf_json_unverified`] and [`from_json_string_unverified`] also read a document
//! without one and return the tree with what is known about it ([`Checked`]).
//!
//! # Deep Fried Optimizations
//! - **Streaming I/O**: Uses `serde_json::to_writer`/`from_reader` with `BufWriter`/`BufReader`.
//! - **Reduced Allocations**: Eliminates intermediate Strings for file I/O.
//!
//! Author: Moroya Sakamoto

use crate::io::IoError;
use crate::provenance::{sdf_load_verdict, Checked, Provenance, KEY};
use crate::types::SdfTree;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

pub const FORMAT: &str = "ASDF JSON";

/// The tree's JSON object with the identifier as a top-level field
fn stamped(tree: &SdfTree, provenance: &Provenance) -> Result<serde_json::Value, IoError> {
    let mut value =
        serde_json::to_value(tree).map_err(|e| IoError::Serialization(e.to_string()))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| IoError::Serialization("an SDF tree is not a JSON object".to_string()))?;
    object.insert(KEY.to_string(), serde_json::Value::String(provenance.hex()));
    Ok(value)
}

/// The tree and the identifier a document carries (`None`: no identifier field)
pub fn split(mut value: serde_json::Value) -> Result<(SdfTree, Option<[u8; 32]>), IoError> {
    let stored = match value.as_object_mut().and_then(|o| o.remove(KEY)) {
        None => None,
        Some(serde_json::Value::String(hex)) => Some(
            Provenance::from_hex(&hex)
                .ok_or_else(|| {
                    IoError::InvalidFormat(format!("{FORMAT}: {KEY} is not 64 hex digits"))
                })?
                .semantics_id,
        ),
        Some(_) => {
            return Err(IoError::InvalidFormat(format!(
                "{FORMAT}: {KEY} is not a string"
            )));
        }
    };
    #[allow(
        clippy::disallowed_methods,
        reason = "stored form checked: split returns the stored identifier with the tree; checked() and validate_sdf_json apply sdf_load_verdict"
    )]
    let tree: SdfTree =
        serde_json::from_value(value).map_err(|e| IoError::Serialization(e.to_string()))?;
    Ok((tree, stored))
}

fn checked(value: serde_json::Value, allow_unpinned: bool) -> Result<Checked<SdfTree>, IoError> {
    let (tree, stored) = split(value)?;
    let verdict = sdf_load_verdict(stored, FORMAT, allow_unpinned)?;
    Ok(Checked::new(tree, verdict))
}

/// Save an SDF tree to JSON format (Streaming), stamped with this crate's identifier
pub fn save_asdf_json(tree: &SdfTree, path: impl AsRef<Path>) -> Result<(), IoError> {
    save_asdf_json_with_provenance(tree, path, &Provenance::sdf())
}

/// [`save_asdf_json`] with the identifier the caller gives
pub fn save_asdf_json_with_provenance(
    tree: &SdfTree,
    path: impl AsRef<Path>,
    provenance: &Provenance,
) -> Result<(), IoError> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &stamped(tree, provenance)?)
        .map_err(|e| IoError::Serialization(e.to_string()))?;
    Ok(())
}

#[allow(
    clippy::disallowed_methods,
    reason = "stored form checked: a serde_json::Value only; the tree is built by split() and checked before it is returned"
)]
fn read_value(path: impl AsRef<Path>) -> Result<serde_json::Value, IoError> {
    let file = File::open(path)?;
    serde_json::from_reader(BufReader::new(file)).map_err(|e| IoError::Serialization(e.to_string()))
}

/// Load an SDF tree from JSON format, checking its identifier
///
/// # Errors
///
/// [`IoError::Unpinned`] for a document without an identifier ([`load_asdf_json_unverified`]
/// reads it), [`IoError::SemanticsMismatch`] for another identifier, and the I/O and JSON errors.
pub fn load_asdf_json(path: impl AsRef<Path>) -> Result<SdfTree, IoError> {
    checked(read_value(path)?, false).map(|c| c.into_parts().0)
}

/// [`load_asdf_json`] that also reads a document without an identifier, marking it
/// [`crate::provenance::StoredSemantics::Unpinned`]; another identifier is still refused
pub fn load_asdf_json_unverified(path: impl AsRef<Path>) -> Result<Checked<SdfTree>, IoError> {
    checked(read_value(path)?, true)
}

/// Serialize an SDF tree to a JSON string (Memory-bound, prefer file I/O), stamped with this
/// crate's identifier
pub fn to_json_string(tree: &SdfTree) -> Result<String, IoError> {
    to_json_string_with_provenance(tree, &Provenance::sdf())
}

/// [`to_json_string`] with the identifier the caller gives
pub fn to_json_string_with_provenance(
    tree: &SdfTree,
    provenance: &Provenance,
) -> Result<String, IoError> {
    serde_json::to_string_pretty(&stamped(tree, provenance)?)
        .map_err(|e| IoError::Serialization(e.to_string()))
}

#[allow(
    clippy::disallowed_methods,
    reason = "stored form checked: a serde_json::Value only; the tree is built by split() and checked before it is returned"
)]
fn parse_value(json: &str) -> Result<serde_json::Value, IoError> {
    serde_json::from_str(json).map_err(|e| IoError::Serialization(e.to_string()))
}

/// Parse an SDF tree from a JSON string, checking its identifier
///
/// # Errors
///
/// As [`load_asdf_json`].
pub fn from_json_string(json: &str) -> Result<SdfTree, IoError> {
    checked(parse_value(json)?, false).map(|c| c.into_parts().0)
}

/// [`from_json_string`] that also reads a document without an identifier (see
/// [`load_asdf_json_unverified`])
pub fn from_json_string_unverified(json: &str) -> Result<Checked<SdfTree>, IoError> {
    checked(parse_value(json)?, true)
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
        let metadata = SdfMetadata {
            name: Some("Test Model".to_string()),
            author: Some("Test Author".to_string()),
            ..SdfMetadata::default()
        };

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

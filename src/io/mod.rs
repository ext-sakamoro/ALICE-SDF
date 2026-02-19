//! File I/O for SDF trees (Deep Fried Edition)
//!
//! Supports two formats:
//! - .asdf: Binary format with CRC32 integrity check
//! - .asdf.json: Human-readable JSON format
//!
//! # Deep Fried Optimizations
//! - **Buffered I/O**: Uses `BufReader`/`BufWriter` for efficient buffering.
//! - **Streaming Write**: On-the-fly CRC with seek patching for ASDF.
//! - **CRC-First Read**: Validates integrity before deserialization (fail-fast).
//! - **Direct JSON I/O**: Uses `serde_json::to_writer`/`from_reader`.
//!
//! Author: Moroya Sakamoto

pub mod abm;
pub mod alembic;
mod asdf;
pub mod fbx;
pub mod gltf;
mod json;
pub mod nanite;
pub mod obj;
pub mod ply;
pub mod stl;
pub mod threemf;
pub mod ue5_asset;
pub mod unity_mesh;
pub mod usd;

pub use abm::{
    load_abm, load_abm_with_lods, read_abm_header, save_abm, save_abm_with_lods, AbmHeader,
    ABM_FLAG_HAS_COLORS, ABM_FLAG_HAS_LODS, ABM_FLAG_HAS_MATERIAL_IDS, ABM_FLAG_HAS_NORMALS,
    ABM_FLAG_HAS_TANGENTS, ABM_FLAG_HAS_UV2, ABM_FLAG_HAS_UVS, ABM_MAGIC, ABM_VERSION,
};
pub use alembic::{export_alembic, AlembicConfig};
pub use asdf::{load_asdf, save_asdf, AsdfHeader, ASDF_MAGIC};
pub use fbx::{
    export_fbx, fbx_animation_to_timeline, import_fbx, import_fbx_full, FbxAnimClip, FbxAnimCurve,
    FbxBone, FbxConfig, FbxFormat, FbxUpAxis, ImportedFbx,
};
pub use gltf::{
    export_glb, export_glb_bytes, export_gltf_json, import_glb, import_glb_bytes, GltfConfig,
};
pub use json::{from_json_string, load_asdf_json, save_asdf_json, to_json_string};
pub use nanite::{
    export_nanite, export_nanite_json, export_nanite_with_config, NaniteExportConfig,
};
pub use obj::{export_obj, import_obj, ObjConfig};
pub use ply::{export_ply, import_ply, PlyConfig};
pub use stl::{export_stl, export_stl_ascii, import_stl};
pub use threemf::export_3mf;
pub use ue5_asset::{
    export_ue5_mesh, export_ue5_mesh_binary, export_ue5_mesh_with_lods, Ue5MeshConfig,
};
pub use unity_mesh::{export_unity_mesh, export_unity_mesh_binary, UnityMeshConfig};
pub use usd::{export_usda, import_usda, ImportedUsdMaterial, ImportedUsda, UsdConfig, UsdUpAxis};

use crate::types::SdfTree;
use std::path::Path;
use thiserror::Error;

/// File I/O errors
#[derive(Error, Debug)]
pub enum IoError {
    /// I/O error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid file format
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    /// CRC checksum mismatch
    #[error("CRC mismatch: expected {expected}, got {actual}")]
    CrcMismatch {
        /// Expected CRC value
        expected: u32,
        /// Actual CRC value
        actual: u32,
    },

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Unsupported file version
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u16),
}

/// Save an SDF tree to file (auto-detect format from extension)
///
/// # Arguments
/// * `tree` - The SDF tree to save
/// * `path` - File path (.asdf or .asdf.json)
///
/// # Returns
/// Result indicating success or error
pub fn save(tree: &SdfTree, path: impl AsRef<Path>) -> Result<(), IoError> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str.ends_with(".asdf.json") {
        save_asdf_json(tree, path)
    } else if path_str.ends_with(".asdf") {
        save_asdf(tree, path)
    } else {
        Err(IoError::InvalidFormat(
            "Unknown file extension. Use .asdf or .asdf.json".to_string(),
        ))
    }
}

/// Load an SDF tree from file (auto-detect format from extension)
///
/// # Arguments
/// * `path` - File path (.asdf or .asdf.json)
///
/// # Returns
/// Loaded SDF tree or error
pub fn load(path: impl AsRef<Path>) -> Result<SdfTree, IoError> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str.ends_with(".asdf.json") {
        load_asdf_json(path)
    } else if path_str.ends_with(".asdf") {
        load_asdf(path)
    } else {
        Err(IoError::InvalidFormat(
            "Unknown file extension. Use .asdf or .asdf.json".to_string(),
        ))
    }
}

/// Get file info without loading the full tree
///
/// # Arguments
/// * `path` - File path
///
/// # Returns
/// File information string
pub fn get_info(path: impl AsRef<Path>) -> Result<String, IoError> {
    let path = path.as_ref();
    let path_str = path.to_string_lossy();

    if path_str.ends_with(".asdf.json") {
        let tree = load_asdf_json(path)?;
        Ok(format!(
            "Format: ASDF JSON\nVersion: {}\nNode count: {}",
            tree.version,
            tree.node_count()
        ))
    } else if path_str.ends_with(".asdf") {
        let header = asdf::read_header(path)?;
        Ok(format!(
            "Format: ASDF Binary\nVersion: {}\nNode count: {}\nCRC32: 0x{:08X}",
            header.version, header.node_count, header.crc32
        ))
    } else {
        Err(IoError::InvalidFormat("Unknown file extension".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;
    use std::fs;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("alice_sdf_test_{}", name));
        path
    }

    #[test]
    fn test_save_load_binary() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let path = temp_path("test.asdf");

        save(&tree, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.node_count(), tree.node_count());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_load_json() {
        let tree = SdfTree::new(SdfNode::sphere(1.0).union(SdfNode::box3d(2.0, 2.0, 2.0)));
        let path = temp_path("test.asdf.json");

        save(&tree, &path).unwrap();
        let loaded = load(&path).unwrap();

        assert_eq!(loaded.node_count(), tree.node_count());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_get_info() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let path = temp_path("info_test.asdf");

        save(&tree, &path).unwrap();
        let info = get_info(&path).unwrap();

        assert!(info.contains("ASDF Binary"));
        assert!(info.contains("Node count: 1"));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_unknown_extension() {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let result = save(&tree, "test.unknown");
        assert!(result.is_err());
    }
}

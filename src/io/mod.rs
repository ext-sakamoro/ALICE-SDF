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
pub mod iges;
pub(crate) mod json;
pub mod meshopt_gltf;
pub mod nanite;
pub mod obj;
pub mod ply;
pub mod splat;
pub mod step;
pub mod stl;
pub mod threemf;
pub mod ue5_asset;
pub mod unity_mesh;
pub mod usd;
#[cfg(feature = "openvdb")]
pub mod vdb;
pub mod vox;

pub use abm::{
    load_abm, load_abm_unverified, load_abm_with_lods, load_abm_with_lods_unverified,
    read_abm_header, save_abm, save_abm_with_lods, save_abm_with_lods_with_provenance,
    save_abm_with_provenance, AbmHeader, ABM_FLAG_HAS_COLORS, ABM_FLAG_HAS_LODS,
    ABM_FLAG_HAS_MATERIAL_IDS, ABM_FLAG_HAS_NORMALS, ABM_FLAG_HAS_TANGENTS, ABM_FLAG_HAS_UV2,
    ABM_FLAG_HAS_UVS, ABM_HEADER_LEN, ABM_MAGIC, ABM_VERSION,
};
pub use alembic::{export_alembic, export_alembic_with_provenance, AlembicConfig};
pub use asdf::{
    load_asdf, load_asdf_unverified, save_asdf, save_asdf_with_provenance, AsdfHeader,
    ASDF_HEADER_LEN, ASDF_MAGIC, ASDF_VERSION,
};
pub use fbx::{
    export_fbx, export_fbx_with_provenance, export_fbx_with_stamp, fbx_animation_to_timeline,
    import_fbx, import_fbx_full, import_fbx_full_pinned, import_fbx_pinned, FbxAnimClip,
    FbxAnimCurve, FbxBone, FbxConfig, FbxFormat, FbxUpAxis, ImportedFbx,
};
pub use gltf::{
    export_glb, export_glb_bytes, export_glb_bytes_with_provenance, export_glb_with_provenance,
    export_gltf_json, export_gltf_json_with_provenance, import_glb, import_glb_bytes,
    import_glb_bytes_pinned, import_glb_pinned, GltfConfig,
};
pub use json::{
    from_json_string, from_json_string_unverified, load_asdf_json, load_asdf_json_unverified,
    save_asdf_json, save_asdf_json_with_provenance, to_json_string, to_json_string_with_provenance,
};
pub use nanite::{
    export_nanite, export_nanite_json, export_nanite_json_with_provenance,
    export_nanite_with_config, export_nanite_with_provenance, NaniteExportConfig,
};
pub use obj::{
    export_obj, export_obj_with_provenance, export_obj_with_stamp, import_obj, import_obj_pinned,
    ObjConfig,
};
pub use ply::{export_ply, export_ply_with_provenance, import_ply, import_ply_pinned, PlyConfig};
pub use stl::{
    export_stl, export_stl_ascii, export_stl_ascii_with_provenance, export_stl_ascii_with_stamp,
    export_stl_with_provenance, export_stl_with_stamp, import_stl, import_stl_pinned,
};
pub use threemf::{export_3mf, export_3mf_with_provenance, export_3mf_with_stamp};
pub use ue5_asset::{
    export_ue5_mesh, export_ue5_mesh_binary, export_ue5_mesh_binary_with_provenance,
    export_ue5_mesh_with_lods, export_ue5_mesh_with_lods_with_provenance,
    export_ue5_mesh_with_provenance, Ue5MeshConfig,
};
pub use unity_mesh::{
    export_unity_mesh, export_unity_mesh_binary, export_unity_mesh_binary_with_provenance,
    export_unity_mesh_with_provenance, UnityMeshConfig,
};
pub use usd::{
    export_usda, export_usda_with_provenance, import_usda, import_usda_pinned, ImportedUsdMaterial,
    ImportedUsda, UsdConfig, UsdUpAxis,
};

use crate::types::SdfTree;
use std::path::Path;
use thiserror::Error;

/// File I/O errors
#[derive(Error, Debug)]
#[non_exhaustive]
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

    /// A stored meaning identifier that is neither this crate's nor one of an earlier identifier
    #[error(
        "{format}: {}: stored {stored}, expected {expected} ({form:?} version {version})",
        crate::provenance::MEANING_KEY
    )]
    MeaningMismatch {
        /// The format read
        format: &'static str,
        /// The stored form the expected identifier was computed for
        form: crate::provenance::StoredForm,
        /// Its layout version
        version: u32,
        /// The stored identifier (hex)
        stored: String,
        /// The meaning identifier of this crate's identifier (hex)
        expected: String,
    },
    /// The file carries a semantics identifier that is neither this crate's nor one it still
    /// reproduces (see [`crate::provenance`])
    #[error("{format}: semantics identifier mismatch: stored {stored}, expected {expected}")]
    SemanticsMismatch {
        /// The format read
        format: &'static str,
        /// The identifier in the file
        stored: crate::provenance::Provenance,
        /// This crate's identifier
        expected: crate::provenance::Provenance,
    },

    /// The file carries no semantics identifier (written before identifiers were stored, or by
    /// another program); read it with the `*_unverified` loader, or a mesh with the default
    /// importer (not `*_pinned`), to get it marked as such
    #[error("{format}: no semantics identifier (read it with the *_unverified loader, or a mesh with the importer that is not *_pinned)")]
    Unpinned {
        /// The format read
        format: &'static str,
    },
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

//! MagicaVoxel (`.vox`) I/O
//!
//! voxel art / indie game 用途。MagicaVoxel の RIFF ベース `.vox` 形式 (v150) で
//! SDF を voxel 化した結果を読み書きする。
//!
//! 形式: <https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt>
//!
//! # 使用例
//!
//! ```ignore
//! use alice_sdf::io::vox::{sdf_to_vox, save_vox, VoxConfig};
//! use alice_sdf::prelude::*;
//!
//! let node = SdfNode::sphere(1.0);
//! let cfg = VoxConfig { size: 64, bounds: (-1.5, 1.5), color_index: 79 };
//! let vox = sdf_to_vox(&node, &cfg);
//! save_vox("sphere.vox", &vox).unwrap();
//! ```

use crate::eval::eval;
use crate::io::IoError;
use crate::provenance::Checked;
use crate::types::SdfNode;
use glam::Vec3;
use std::path::Path;

/// 単一 voxel (x, y, z, color_index)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Voxel {
    /// X 座標 (0..=255)
    pub x: u8,
    /// Y 座標 (0..=255)
    pub y: u8,
    /// Z 座標 (0..=255)
    pub z: u8,
    /// MagicaVoxel palette index (1-255)、0 は空 voxel
    pub color: u8,
}

/// MagicaVoxel データ (1 モデル = 1 SIZE + 1 XYZI)
#[derive(Clone, Debug)]
pub struct VoxModel {
    /// グリッドサイズ (X, Y, Z)、各 1..=256
    pub size: (u32, u32, u32),
    /// voxel 配列
    pub voxels: Vec<Voxel>,
}

/// SDF → Vox 変換設定
#[derive(Clone, Copy, Debug)]
pub struct VoxConfig {
    /// 1 辺の voxel 数 (1..=256)
    pub size: u32,
    /// グリッド範囲 (min, max)
    pub bounds: (f32, f32),
    /// 内部 voxel に割り当てる palette index (1..=255)
    pub color_index: u8,
}

impl Default for VoxConfig {
    fn default() -> Self {
        Self {
            size: 64,
            bounds: (-1.5, 1.5),
            color_index: 79,
        }
    }
}

/// SDF tree を voxelize して VoxModel に変換
///
/// `eval(p) <= 0` の voxel を内部とみなして color_index を設定する。
pub fn sdf_to_vox(node: &SdfNode, cfg: &VoxConfig) -> VoxModel {
    let n = cfg.size.clamp(1, 256);
    let (lo, hi) = cfg.bounds;
    let step = (hi - lo) / n.max(1) as f32;
    let mut voxels = Vec::new();
    for k in 0..n {
        let z = (k as f32 + 0.5) * step + lo;
        for j in 0..n {
            let y = (j as f32 + 0.5) * step + lo;
            for i in 0..n {
                let x = (i as f32 + 0.5) * step + lo;
                let d = eval(node, Vec3::new(x, y, z));
                if d <= 0.0 {
                    voxels.push(Voxel {
                        x: i as u8,
                        y: j as u8,
                        z: k as u8,
                        color: cfg.color_index,
                    });
                }
            }
        }
    }
    VoxModel {
        size: (n, n, n),
        voxels,
    }
}

/// MagicaVoxel `.vox` v150 バイナリへ書き出し
pub fn save_vox(path: impl AsRef<Path>, model: &VoxModel) -> std::io::Result<()> {
    save_vox_with_provenance(path, model, &crate::provenance::Provenance::sdf())
}

/// [`save_vox`] storing the given semantics identifier instead of this crate's
///
/// The identifier is a private `ALSD` chunk among `MAIN`'s children (layout version 1, then the
/// 32 bytes); MagicaVoxel and other readers skip chunks they do not know.
///
/// # Errors
///
/// An I/O error writing the file.
pub fn save_vox_with_provenance(
    path: impl AsRef<Path>,
    model: &VoxModel,
    provenance: &crate::provenance::Provenance,
) -> std::io::Result<()> {
    let bytes = encode_vox(model, provenance);
    std::fs::write(path, bytes)
}

fn encode_vox(model: &VoxModel, provenance: &crate::provenance::Provenance) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64 + model.voxels.len() * 4);
    // Header
    buf.extend_from_slice(b"VOX ");
    buf.extend_from_slice(&150u32.to_le_bytes());

    // MAIN chunk
    buf.extend_from_slice(b"MAIN");
    buf.extend_from_slice(&0u32.to_le_bytes()); // content size
    let children_size_pos = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes()); // children size (patched later)
    let children_start = buf.len();

    // SIZE chunk
    buf.extend_from_slice(b"SIZE");
    buf.extend_from_slice(&12u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // no children
    buf.extend_from_slice(&model.size.0.to_le_bytes());
    buf.extend_from_slice(&model.size.1.to_le_bytes());
    buf.extend_from_slice(&model.size.2.to_le_bytes());

    // XYZI chunk
    buf.extend_from_slice(b"XYZI");
    let num_voxels = model.voxels.len() as u32;
    let xyzi_content = 4 + num_voxels * 4;
    buf.extend_from_slice(&xyzi_content.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // no children
    buf.extend_from_slice(&num_voxels.to_le_bytes());
    for v in &model.voxels {
        buf.push(v.x);
        buf.push(v.y);
        buf.push(v.z);
        buf.push(v.color);
    }

    // ALSD chunk: the semantics identifier (layout version 1 + 32 bytes), a child of MAIN
    buf.extend_from_slice(crate::provenance::VOX_CHUNK);
    buf.extend_from_slice(&36u32.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // no children
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&provenance.semantics_id);

    // Patch MAIN children size
    let children_end = buf.len();
    let children_size = (children_end - children_start) as u32;
    buf[children_size_pos..children_size_pos + 4].copy_from_slice(&children_size.to_le_bytes());
    buf
}

/// `.vox` バイナリ → VoxModel 読込 (SIZE + XYZI のみ対応、RGBA / TRANSFORM は無視)
///
/// MagicaVoxel and other programs write no identifier, so a file without the `ALSD` chunk is read
/// and marked [`crate::provenance::StoredSemantics::Unpinned`]; [`load_vox_pinned`] refuses it.
///
/// # Errors
///
/// An I/O error reading the file, [`IoError::SemanticsMismatch`] for an identifier that is
/// neither this crate's nor one it still reproduces, and [`IoError::InvalidFormat`] for bytes that
/// are not a VOX file or an `ALSD` chunk that is cut short, has another length or layout version,
/// or occurs twice.
pub fn load_vox(path: impl AsRef<Path>) -> Result<Checked<VoxModel>, IoError> {
    load_vox_with(path.as_ref(), true)
}

/// [`load_vox`] that refuses a file without this crate's identifier
///
/// # Errors
///
/// [`IoError::Unpinned`] for a file without an identifier, otherwise as [`load_vox`].
pub fn load_vox_pinned(path: impl AsRef<Path>) -> Result<VoxModel, IoError> {
    load_vox_with(path.as_ref(), false).map(|c| c.into_parts().0)
}

fn load_vox_with(path: &Path, allow_unpinned: bool) -> Result<Checked<VoxModel>, IoError> {
    let bytes = std::fs::read(path)?;
    // a present but unreadable ALSD chunk is an error, never a file without an identifier
    let stored = crate::provenance::vox_stored_id(&bytes)
        .map_err(|e| IoError::InvalidFormat(format!("VOX: {e}")))?;
    let semantics = crate::provenance::sdf_load_verdict(stored, "VOX", allow_unpinned)?;
    let model = parse_vox(&bytes).map_err(|e| IoError::InvalidFormat(format!("VOX: {e}")))?;
    Ok(Checked::new(model, semantics))
}

fn parse_vox(bytes: &[u8]) -> Result<VoxModel, String> {
    if bytes.len() < 8 || &bytes[0..4] != b"VOX " {
        return Err("not a VOX file".into());
    }
    let mut size = (0u32, 0u32, 0u32);
    let mut voxels = Vec::new();
    let mut off = 8;
    // skip MAIN header (8 bytes id + 4 content size + 4 children size)
    if bytes.len() < off + 12 || &bytes[off..off + 4] != b"MAIN" {
        return Err("missing MAIN".into());
    }
    off += 12;
    while off + 12 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let content_size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let _children_size =
            u32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap()) as usize;
        let body_start = off + 12;
        let body_end = body_start + content_size;
        if body_end > bytes.len() {
            break;
        }
        if id == b"SIZE" && content_size >= 12 {
            size = (
                u32::from_le_bytes(bytes[body_start..body_start + 4].try_into().unwrap()),
                u32::from_le_bytes(bytes[body_start + 4..body_start + 8].try_into().unwrap()),
                u32::from_le_bytes(bytes[body_start + 8..body_start + 12].try_into().unwrap()),
            );
        } else if id == b"XYZI" && content_size >= 4 {
            let num =
                u32::from_le_bytes(bytes[body_start..body_start + 4].try_into().unwrap()) as usize;
            for i in 0..num {
                let p = body_start + 4 + i * 4;
                if p + 4 > body_end {
                    break;
                }
                voxels.push(Voxel {
                    x: bytes[p],
                    y: bytes[p + 1],
                    z: bytes[p + 2],
                    color: bytes[p + 3],
                });
            }
        }
        off = body_end;
    }
    Ok(VoxModel { size, voxels })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn sphere_voxelize_non_empty() {
        let n = SdfNode::sphere(1.0);
        let cfg = VoxConfig {
            size: 32,
            bounds: (-1.5, 1.5),
            color_index: 79,
        };
        let m = sdf_to_vox(&n, &cfg);
        assert_eq!(m.size, (32, 32, 32));
        assert!(!m.voxels.is_empty());
    }

    #[test]
    fn encode_starts_with_vox_magic() {
        let m = VoxModel {
            size: (1, 1, 1),
            voxels: vec![Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
            }],
        };
        let b = encode_vox(&m, &crate::provenance::Provenance::sdf());
        assert_eq!(&b[0..4], b"VOX ");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 150);
    }

    #[test]
    fn roundtrip_save_load() {
        let path = std::env::temp_dir().join("alice_sdf_test.vox");
        let n = SdfNode::sphere(1.0);
        let cfg = VoxConfig {
            size: 16,
            bounds: (-1.5, 1.5),
            color_index: 42,
        };
        let m = sdf_to_vox(&n, &cfg);
        save_vox(&path, &m).unwrap();
        let loaded = load_vox_pinned(&path).unwrap();
        assert_eq!(loaded.size, m.size);
        assert_eq!(loaded.voxels.len(), m.voxels.len());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_voxels_still_valid_file() {
        let m = VoxModel {
            size: (1, 1, 1),
            voxels: vec![],
        };
        let b = encode_vox(&m, &crate::provenance::Provenance::sdf());
        let parsed = parse_vox(&b).unwrap();
        assert_eq!(parsed.size, (1, 1, 1));
        assert_eq!(parsed.voxels.len(), 0);
    }
}

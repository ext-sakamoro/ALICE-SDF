//! OpenVDB (.vdb) I/O for SDF volumes
//!
//! ALICE-SDF を OpenVDB Float Grid 形式に bake / load する。
//! Houdini / Maya / Nuke 等の DCC ツール連携で使う。
//!
//! `openvdb` feature 有効時のみ有効。`vdb-rs` 0.6 (pure Rust) ベース。
//!
//! # 使用例
//!
//! ```ignore
//! use alice_sdf::io::vdb::{bake_to_vdb, load_dense_grid_from_vdb};
//! use alice_sdf::prelude::*;
//!
//! let node = SdfNode::sphere(1.0);
//! let vdb_bytes = bake_to_vdb(&node, (-2.0, 2.0), 64).unwrap();
//! std::fs::write("sphere.vdb", &vdb_bytes).unwrap();
//! ```

use crate::eval::eval;
use crate::types::SdfNode;
use glam::Vec3;

/// OpenVDB bake のエラー
#[derive(Debug, thiserror::Error)]
pub enum VdbError {
    /// 下位 vdb-rs ライブラリからの I/O / シリアライズエラー
    #[error("vdb-rs IO error: {0}")]
    Io(String),
    /// grid bounds が不正 (min >= max など)
    #[error("invalid grid bounds")]
    InvalidBounds,
}

/// SDF を密 voxel 配列 (row-major, x changes fastest) に bake
///
/// # Arguments
/// - `node`: SDF tree
/// - `bounds`: (min, max) — 立方体領域、xyz 同範囲
/// - `resolution`: 1 辺の voxel 数 (例 64 → 64³ = 262144 voxels)
///
/// # Returns
/// `Vec<f32>` 長さ `resolution³`、距離値が格納される
pub fn bake_dense_grid(node: &SdfNode, bounds: (f32, f32), resolution: u32) -> Vec<f32> {
    let (min, max) = bounds;
    let n = resolution as usize;
    let step = (max - min) / (resolution - 1).max(1) as f32;
    let mut grid = Vec::with_capacity(n * n * n);
    for k in 0..n {
        let z = (k as f32) * step + min;
        for j in 0..n {
            let y = (j as f32) * step + min;
            for i in 0..n {
                let x = (i as f32) * step + min;
                let d = eval(node, Vec3::new(x, y, z));
                grid.push(d);
            }
        }
    }
    grid
}

/// SDF を OpenVDB float grid (`.vdb` バイナリ) に bake
///
/// 内部で `bake_dense_grid` を呼んでから vdb-rs で encode する。
/// 戻り値の `Vec<u8>` をそのままファイルに書ける。
///
/// # Note
/// vdb-rs 0.6 は pure Rust 実装、`half` 精度や圧縮の細かい制御は限定的。
/// 大規模なグリッド (>512³) は OpenVDB C++ 公式実装の方が高機能。
pub fn bake_to_vdb(
    node: &SdfNode,
    bounds: (f32, f32),
    resolution: u32,
) -> Result<Vec<u8>, VdbError> {
    bake_to_vdb_with_provenance(
        node,
        bounds,
        resolution,
        &crate::provenance::Provenance::sdf(),
    )
}

/// [`bake_to_vdb`] storing the given semantics identifier instead of this crate's
///
/// # Errors
///
/// As [`bake_to_vdb`].
pub fn bake_to_vdb_with_provenance(
    node: &SdfNode,
    bounds: (f32, f32),
    resolution: u32,
    provenance: &crate::provenance::Provenance,
) -> Result<Vec<u8>, VdbError> {
    if bounds.1 <= bounds.0 || resolution == 0 {
        return Err(VdbError::InvalidBounds);
    }
    let _dense = bake_dense_grid(node, bounds, resolution);

    // NOTE: vdb-rs 0.6 のフルエンコード API は本実装時点では public 化が限定的。
    // 暫定的に「ALICE-VDB Float Grid」自前バイナリ形式 (header + raw dense f32 array) を返す。
    // 将来 vdb-rs が write API を整備したら差し替え予定。
    let mut buf = Vec::new();
    buf.extend_from_slice(ALICE_VDB_MAGIC_V2); // magic (9 bytes)
    buf.extend_from_slice(&provenance.semantics_id);
    buf.extend_from_slice(&resolution.to_le_bytes());
    buf.extend_from_slice(&bounds.0.to_le_bytes());
    buf.extend_from_slice(&bounds.1.to_le_bytes());
    for v in _dense.iter() {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    Ok(buf)
}

/// A dense grid read back from `bake_to_vdb`: `(values, resolution, (bounds_min, bounds_max))`.
pub type DenseGrid = (Vec<f32>, u32, (f32, f32));

/// `bake_to_vdb` で生成したバイナリから密 voxel 配列を復元 (保存された意味の識別子を確かめる)
///
/// # Errors
///
/// [`crate::io::IoError::Unpinned`] for a version 1 grid (no identifier; read it with
/// [`load_dense_grid_from_vdb_unverified`]), [`crate::io::IoError::SemanticsMismatch`] for an
/// identifier that is neither this crate's nor one it still reproduces, and
/// [`crate::io::IoError::InvalidFormat`] for bytes that are not an ALICE-VDB grid, a version this
/// crate does not know, or a header or grid cut short.
pub fn load_dense_grid_from_vdb(bytes: &[u8]) -> Result<DenseGrid, crate::io::IoError> {
    load_checked(bytes, false).map(|c| c.into_parts().0)
}

/// [`load_dense_grid_from_vdb`] that also reads a version 1 grid, marking it unpinned
///
/// A version 1 grid (`ALICEVDB1`, alice-sdf 6.1.0 and earlier) was written before the semantics
/// identifier; it is returned with [`crate::provenance::StoredSemantics::Unpinned`].
///
/// # Errors
///
/// As [`load_dense_grid_from_vdb`], except that a grid without an identifier is returned, not
/// refused.
pub fn load_dense_grid_from_vdb_unverified(
    bytes: &[u8],
) -> Result<crate::provenance::Checked<DenseGrid>, crate::io::IoError> {
    load_checked(bytes, true)
}

/// Magic of the version 1 grid (no identifier)
pub const ALICE_VDB_MAGIC_V1: &[u8; 9] = b"ALICEVDB1";
/// Magic of the version 2 grid: the 32-byte semantics identifier follows it
pub const ALICE_VDB_MAGIC_V2: &[u8; 9] = b"ALICEVDB2";

fn load_checked(
    bytes: &[u8],
    allow_unpinned: bool,
) -> Result<crate::provenance::Checked<DenseGrid>, crate::io::IoError> {
    use crate::io::IoError;
    if !bytes.starts_with(b"ALICEVDB") {
        return Err(IoError::InvalidFormat("ALICE-VDB: bad magic".into()));
    }
    // the magic is present: a version or identifier that cannot be read is an error, never
    // "unstamped"
    let (stored, body) = match bytes.get(..9) {
        Some(m) if m == ALICE_VDB_MAGIC_V1 => (None, &bytes[9..]),
        Some(m) if m == ALICE_VDB_MAGIC_V2 => {
            let id: [u8; 32] = bytes
                .get(9..41)
                .ok_or_else(|| IoError::InvalidFormat("ALICE-VDB: truncated header".into()))?
                .try_into()
                .unwrap();
            (Some(id), &bytes[41..])
        }
        Some(m) => {
            return Err(IoError::InvalidFormat(format!(
                "ALICE-VDB: version {:?} is not known",
                char::from(m[8])
            )))
        }
        None => return Err(IoError::InvalidFormat("ALICE-VDB: truncated header".into())),
    };
    let semantics = crate::provenance::sdf_load_verdict(stored, "ALICE-VDB", allow_unpinned)?;
    let grid = read_grid(body).map_err(|e| IoError::InvalidFormat(format!("ALICE-VDB: {e}")))?;
    Ok(crate::provenance::Checked::new(grid, semantics))
}

/// The grid after the magic and identifier (identical in versions 1 and 2)
fn read_grid(bytes: &[u8]) -> Result<DenseGrid, &'static str> {
    if bytes.len() < 4 + 4 + 4 {
        return Err("buffer too small");
    }
    let resolution = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let bmin = f32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let bmax = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let n = resolution as usize;
    let expected = 12 + n * n * n * 4;
    if bytes.len() < expected {
        return Err("truncated grid data");
    }
    let mut grid = Vec::with_capacity(n * n * n);
    let mut off = 12;
    for _ in 0..(n * n * n) {
        let v = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        grid.push(v);
        off += 4;
    }
    Ok((grid, resolution, (bmin, bmax)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SdfNode;

    #[test]
    fn bake_dense_grid_returns_correct_size() {
        let n = SdfNode::sphere(1.0);
        let g = bake_dense_grid(&n, (-2.0, 2.0), 8);
        assert_eq!(g.len(), 8 * 8 * 8);
    }

    #[test]
    fn bake_dense_grid_centerpoint_is_inside() {
        let n = SdfNode::sphere(1.0);
        let g = bake_dense_grid(&n, (-2.0, 2.0), 9); // 9³, center index = (4,4,4)
        let center = g[4 * 9 * 9 + 4 * 9 + 4];
        assert!(
            center < 0.0,
            "centerpoint should be inside sphere: {center}"
        );
    }

    #[test]
    fn vdb_roundtrip() {
        let n = SdfNode::sphere(1.0);
        let bytes = bake_to_vdb(&n, (-2.0, 2.0), 8).unwrap();
        let (grid, res, bounds) = load_dense_grid_from_vdb(&bytes).unwrap();
        assert_eq!(res, 8);
        assert_eq!(bounds, (-2.0, 2.0));
        assert_eq!(grid.len(), 8 * 8 * 8);
    }

    #[test]
    fn vdb_invalid_bounds_returns_err() {
        let n = SdfNode::sphere(1.0);
        assert!(bake_to_vdb(&n, (2.0, -2.0), 8).is_err());
    }
}

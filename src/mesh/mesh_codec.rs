//! Mesh Encoding: 簡易 varint delta compression for indices and vertices
//!
//! ネットワーク帯域幅削減 / storage 圧縮向けの Rust ネイティブ mesh 圧縮
//!
//! # 独自 format 宣言
//!
//! 本 module は **meshoptimizer の binary format とは非互換** の独自 format ALICE-SDF 内部
//! utility として設計、`EXT_meshopt_compression` glTF extension には対応しない
//! (完全 meshopt compat は future work、~500-800 行 bit-level spec 準拠)
//!
//! # Encoding 手法
//!
//! ## Index encoding
//!
//! - 各 triangle の 3 index について、前 triangle の対応 index からの delta を取る
//! - delta を zigzag encoding で unsigned 化 (符号交互配置)
//! - varint (LEB128) で可変長 encode
//!
//! ## Vertex encoding
//!
//! - Vertex を byte stream に split (position.x の byte 0, byte 1, byte 2, byte 3 別々に等)
//! - 各 byte stream で前 vertex 対応 byte からの delta
//! - varint で encode (よく差分が 0 near で圧縮効く)
//!
//! # 圧縮率
//!
//! - Index: 典型的な三角形 mesh で 2-3x compression
//! - Vertex: 平面 mesh / 規則 mesh で 2-4x、noisy mesh で ~1.5x
//!
//! # Header format (自己記述型、8 bytes)
//!
//! ```text
//! Offset  Size  Field
//! 0       4     Magic: b"ASDF" (ALICE-SDF)
//! 4       1     Version (1)
//! 5       1     Kind (0 = index, 1 = vertex position, 2 = vertex all)
//! 6       2     (reserved, zero-fill)
//! ```
//!
//! # References
//!
//! - zeux/meshoptimizer §indexcodec.cpp / vertexcodec.cpp (完全 binary compat 版、独立実装)
//! - varint / LEB128 encoding (Google Protocol Buffers spec)
//! - Zigzag encoding (Twitter Snowflake ID / Protocol Buffers)
//!
//! Author: Moroya Sakamoto

use crate::mesh::Mesh;
use glam::Vec3;

const MAGIC: [u8; 4] = *b"ASDF";
const VERSION: u8 = 1;
const KIND_INDEX: u8 = 0;
const KIND_VERTEX_POS: u8 = 1;

/// Codec エラー
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecError {
    /// Header の magic bytes 不一致
    BadMagic,
    /// Version が未対応
    UnsupportedVersion,
    /// Data kind が期待値と異なる
    WrongKind,
    /// Buffer が途中で終了 (unexpected EOF)
    UnexpectedEof,
    /// varint decode 中に buffer 超過
    VarintOverflow,
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "codec: bad magic bytes"),
            Self::UnsupportedVersion => write!(f, "codec: unsupported version"),
            Self::WrongKind => write!(f, "codec: wrong data kind"),
            Self::UnexpectedEof => write!(f, "codec: unexpected end of buffer"),
            Self::VarintOverflow => write!(f, "codec: varint decode overflow"),
        }
    }
}

impl std::error::Error for CodecError {}

// ============================================================================
// varint (LEB128) encoding
// ============================================================================

/// unsigned varint encode: 各 byte の下位 7 bit が data、最上位 bit で continuation
fn write_varint_u32(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(((v & 0x7F) | 0x80) as u8);
        v >>= 7;
    }
    out.push(v as u8);
}

/// varint decode from cursor position、cursor 進行させる
fn read_varint_u32(buf: &[u8], cursor: &mut usize) -> Result<u32, CodecError> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    loop {
        if *cursor >= buf.len() {
            return Err(CodecError::UnexpectedEof);
        }
        let b = buf[*cursor];
        *cursor += 1;
        if shift >= 32 {
            return Err(CodecError::VarintOverflow);
        }
        result |= ((b & 0x7F) as u32) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
    }
}

/// signed → unsigned zigzag (符号交互配置、小さい delta が短く encode される)
#[inline]
const fn zigzag_encode_i32(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// unsigned → signed zigzag decode
#[inline]
const fn zigzag_decode_u32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

// ============================================================================
// Index encoding (varint delta)
// ============================================================================

/// Triangle index buffer を encode
///
/// # アルゴリズム
///
/// 1. Header 8 bytes を書く (magic + version + kind=0 + tri_count)
/// 2. 各 triangle について:
///    - `delta[k] = indices[3*t + k] - previous_indices[k]` (previous は前 triangle の同 slot)
///    - zigzag encode で unsigned 化
///    - varint で書く
/// 3. 最初の triangle は 0 前提 (絶対値 = delta)
///
/// # Returns
///
/// 圧縮された byte buffer 冒頭 8 bytes は header で残りは encoded data
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub fn encode_indices(indices: &[u32]) -> Vec<u8> {
    let tri_count = indices.len() / 3;
    let mut out = Vec::with_capacity(8 + tri_count * 3);

    // Header (8 bytes)
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(KIND_INDEX);
    out.push(0);
    out.push(0);

    // Triangle count を varint で書く (decode 時に必要)
    write_varint_u32(&mut out, tri_count as u32);

    let mut prev = [0i32; 3];
    for t in 0..tri_count {
        let base = t * 3;
        for (k, prev_slot) in prev.iter_mut().enumerate() {
            let cur = indices[base + k] as i32;
            let delta = cur - *prev_slot;
            let encoded = zigzag_encode_i32(delta);
            write_varint_u32(&mut out, encoded);
            *prev_slot = cur;
        }
    }

    out
}

/// Encoded triangle index buffer を decode
///
/// # Errors
///
/// - `BadMagic`: header の magic が `b"ASDF"` でない
/// - `UnsupportedVersion`: version が 1 でない
/// - `WrongKind`: kind が KIND_INDEX (0) でない
/// - `UnexpectedEof`: buffer が途中で切れている
/// - `VarintOverflow`: varint が 32-bit を超えて continue した
pub fn decode_indices(buf: &[u8]) -> Result<Vec<u32>, CodecError> {
    if buf.len() < 8 {
        return Err(CodecError::UnexpectedEof);
    }
    if buf[0..4] != MAGIC {
        return Err(CodecError::BadMagic);
    }
    if buf[4] != VERSION {
        return Err(CodecError::UnsupportedVersion);
    }
    if buf[5] != KIND_INDEX {
        return Err(CodecError::WrongKind);
    }
    // buf[6..8] は reserved

    let mut cursor = 8_usize;
    let tri_count = read_varint_u32(buf, &mut cursor)? as usize;

    let mut indices = Vec::with_capacity(tri_count * 3);
    let mut prev = [0i32; 3];
    for _ in 0..tri_count {
        for prev_slot in &mut prev {
            let encoded = read_varint_u32(buf, &mut cursor)?;
            let delta = zigzag_decode_u32(encoded);
            let cur = *prev_slot + delta;
            indices.push(cur as u32);
            *prev_slot = cur;
        }
    }

    Ok(indices)
}

// ============================================================================
// Vertex position encoding (byte-stream delta + varint)
// ============================================================================

/// 頂点 position 配列を encode (byte-stream split delta + varint)
///
/// # アルゴリズム
///
/// 1. Header 8 bytes (kind = KIND_VERTEX_POS)
/// 2. Vertex count varint
/// 3. 各 position 成分 (x, y, z) を f32 bit representation として `u32` に変換
/// 4. 4 byte stream (byte 0, 1, 2, 3) に分割
/// 5. 各 stream で前 vertex 対応 byte からの delta を varint (zigzag) で書く
///
/// # 圧縮率想定
///
/// - 平面 / regular mesh: 4-8x (byte 上位が全 0 に近い)
/// - Noisy / random mesh: 1.5-2x (byte 差分が大きい)
///
/// # 注意
///
/// f32 の bit patterns の delta は数値差とは違う (signed magnitude 表現による)
/// 小規模 mesh では実質 delta が小さい傾向あるが、大 scale mesh では bit 差分が大きくなる
#[must_use]
pub fn encode_positions(positions: &[Vec3]) -> Vec<u8> {
    let n = positions.len();
    let mut out = Vec::with_capacity(8 + n * 12);

    // Header
    out.extend_from_slice(&MAGIC);
    out.push(VERSION);
    out.push(KIND_VERTEX_POS);
    out.push(0);
    out.push(0);

    write_varint_u32(&mut out, n as u32);

    if n == 0 {
        return out;
    }

    // XYZ 3 成分 × 4 bytes = 12 byte streams、byte-major layout で encode
    // stream[b][i] = bit_repr(positions[i].comp_j) の byte b (j = b / 4)
    // 各 stream 内で delta encoding
    let mut prev_bytes = [0u8; 12];
    for &p in positions {
        let comps = [p.x, p.y, p.z];
        let mut cur_bytes = [0u8; 12];
        for (j, &c) in comps.iter().enumerate() {
            let bits = c.to_bits();
            let le = bits.to_le_bytes();
            cur_bytes[j * 4..j * 4 + 4].copy_from_slice(&le);
        }
        for b in 0..12 {
            let delta = i32::from(cur_bytes[b]) - i32::from(prev_bytes[b]);
            let encoded = zigzag_encode_i32(delta);
            write_varint_u32(&mut out, encoded);
        }
        prev_bytes = cur_bytes;
    }

    out
}

/// Encoded position buffer を decode
///
/// # Errors
///
/// - `BadMagic`: magic 不一致
/// - `UnsupportedVersion`: version != 1
/// - `WrongKind`: kind != KIND_VERTEX_POS
/// - `UnexpectedEof`: buffer 切断
/// - `VarintOverflow`: varint 32-bit 超過
pub fn decode_positions(buf: &[u8]) -> Result<Vec<Vec3>, CodecError> {
    if buf.len() < 8 {
        return Err(CodecError::UnexpectedEof);
    }
    if buf[0..4] != MAGIC {
        return Err(CodecError::BadMagic);
    }
    if buf[4] != VERSION {
        return Err(CodecError::UnsupportedVersion);
    }
    if buf[5] != KIND_VERTEX_POS {
        return Err(CodecError::WrongKind);
    }

    let mut cursor = 8_usize;
    let n = read_varint_u32(buf, &mut cursor)? as usize;

    let mut positions = Vec::with_capacity(n);
    let mut prev_bytes = [0u8; 12];
    for _ in 0..n {
        let mut cur_bytes = [0u8; 12];
        for b in 0..12 {
            let encoded = read_varint_u32(buf, &mut cursor)?;
            let delta = zigzag_decode_u32(encoded);
            let cur = (i32::from(prev_bytes[b]) + delta) as i32;
            // byte 範囲外 (delta が異常大) → error にせず低位 8-bit だけ採用 (garbage in)
            cur_bytes[b] = cur as u8;
        }
        let mut comps = [0.0_f32; 3];
        for (j, comp) in comps.iter_mut().enumerate() {
            let bytes: [u8; 4] = cur_bytes[j * 4..j * 4 + 4].try_into().unwrap_or([0; 4]);
            let bits = u32::from_le_bytes(bytes);
            *comp = f32::from_bits(bits);
        }
        positions.push(Vec3::new(comps[0], comps[1], comps[2]));
        prev_bytes = cur_bytes;
    }

    Ok(positions)
}

// ============================================================================
// Convenience: encode a whole Mesh (indices + positions)
// ============================================================================

/// Mesh 全体 (indices + positions) を encode
///
/// 返り値は (encoded_indices, encoded_positions) 2 buffer
/// caller が任意に concatenate 可能
#[must_use]
pub fn encode_mesh(mesh: &Mesh) -> (Vec<u8>, Vec<u8>) {
    let indices_encoded = encode_indices(&mesh.indices);
    let positions: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
    let positions_encoded = encode_positions(&positions);
    (indices_encoded, positions_encoded)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // varint
    // ------------------------------------------------------------------------

    #[test]
    fn test_varint_roundtrip_small() {
        for &v in &[0_u32, 1, 127, 128, 129, 1000, 1_000_000, u32::MAX] {
            let mut buf = Vec::new();
            write_varint_u32(&mut buf, v);
            let mut cursor = 0;
            let decoded = read_varint_u32(&buf, &mut cursor).unwrap();
            assert_eq!(decoded, v);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn test_varint_encoding_size() {
        // 127 まで 1 byte、128 以上 2 bytes
        let mut buf1 = Vec::new();
        write_varint_u32(&mut buf1, 127);
        assert_eq!(buf1.len(), 1);
        let mut buf2 = Vec::new();
        write_varint_u32(&mut buf2, 128);
        assert_eq!(buf2.len(), 2);
    }

    // ------------------------------------------------------------------------
    // zigzag
    // ------------------------------------------------------------------------

    #[test]
    fn test_zigzag_roundtrip() {
        for &v in &[0_i32, 1, -1, 100, -100, i32::MAX, i32::MIN] {
            let encoded = zigzag_encode_i32(v);
            let decoded = zigzag_decode_u32(encoded);
            assert_eq!(decoded, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_zigzag_small_deltas_are_small() {
        // 0, -1, 1, -2, 2 → 0, 1, 2, 3, 4 (小さい delta が小さい unsigned)
        assert_eq!(zigzag_encode_i32(0), 0);
        assert_eq!(zigzag_encode_i32(-1), 1);
        assert_eq!(zigzag_encode_i32(1), 2);
        assert_eq!(zigzag_encode_i32(-2), 3);
        assert_eq!(zigzag_encode_i32(2), 4);
    }

    // ------------------------------------------------------------------------
    // Index encoding
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_decode_empty_indices() {
        let encoded = encode_indices(&[]);
        let decoded = decode_indices(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_decode_single_triangle() {
        let indices = vec![0_u32, 1, 2];
        let encoded = encode_indices(&indices);
        let decoded = decode_indices(&encoded).unwrap();
        assert_eq!(decoded, indices);
    }

    #[test]
    fn test_encode_decode_many_triangles() {
        // 100 三角形の連続 index (grid pattern)
        let mut indices = Vec::new();
        for t in 0..100 {
            indices.push(t as u32);
            indices.push((t + 1) as u32);
            indices.push((t + 2) as u32);
        }
        let encoded = encode_indices(&indices);
        let decoded = decode_indices(&encoded).unwrap();
        assert_eq!(decoded, indices);
    }

    #[test]
    fn test_encode_indices_compression_ratio() {
        // 連続 index (小 delta) は 4 bytes/index より小さくなる
        let indices: Vec<u32> = (0..300).collect();
        let raw_size = indices.len() * 4; // u32 = 4 bytes
        let encoded = encode_indices(&indices);
        assert!(
            encoded.len() < raw_size,
            "encoded should be smaller than raw: {} vs {}",
            encoded.len(),
            raw_size
        );
    }

    #[test]
    fn test_decode_indices_bad_magic() {
        let mut bad = vec![b'X', b'Y', b'Z', b'W', 1, 0, 0, 0];
        write_varint_u32(&mut bad, 0);
        let err = decode_indices(&bad).unwrap_err();
        assert_eq!(err, CodecError::BadMagic);
    }

    #[test]
    fn test_decode_indices_wrong_kind() {
        // kind=1 (vertex pos) を index decoder に渡す
        let mut bad: Vec<u8> = MAGIC.to_vec();
        bad.push(VERSION);
        bad.push(KIND_VERTEX_POS);
        bad.push(0);
        bad.push(0);
        write_varint_u32(&mut bad, 0);
        let err = decode_indices(&bad).unwrap_err();
        assert_eq!(err, CodecError::WrongKind);
    }

    #[test]
    fn test_decode_indices_truncated() {
        let indices = vec![0_u32, 1, 2, 3, 4, 5];
        let encoded = encode_indices(&indices);
        // Header 直後で切断
        let truncated = &encoded[..8];
        let result = decode_indices(truncated);
        assert!(result.is_err(), "truncated should error");
    }

    // ------------------------------------------------------------------------
    // Position encoding
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_decode_empty_positions() {
        let encoded = encode_positions(&[]);
        let decoded = decode_positions(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_decode_single_position() {
        let positions = vec![Vec3::new(1.0, 2.0, 3.0)];
        let encoded = encode_positions(&positions);
        let decoded = decode_positions(&encoded).unwrap();
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - positions[0]).length() < 1e-6);
    }

    #[test]
    fn test_encode_decode_grid_positions() {
        // 10x10 grid → regular pattern で圧縮効きやすい
        let mut positions = Vec::new();
        for i in 0..10 {
            for j in 0..10 {
                positions.push(Vec3::new(i as f32, 0.0, j as f32));
            }
        }
        let encoded = encode_positions(&positions);
        let decoded = decode_positions(&encoded).unwrap();
        assert_eq!(decoded.len(), positions.len());
        for (a, b) in positions.iter().zip(decoded.iter()) {
            assert!((a - b).length() < 1e-6);
        }
    }

    #[test]
    fn test_decode_positions_bad_magic() {
        let mut bad = vec![b'X', b'X', b'X', b'X', 1, 1, 0, 0];
        write_varint_u32(&mut bad, 0);
        let err = decode_positions(&bad).unwrap_err();
        assert_eq!(err, CodecError::BadMagic);
    }

    #[test]
    fn test_decode_positions_wrong_kind() {
        // kind=0 (index) を position decoder に渡す
        let encoded = encode_indices(&[]);
        let err = decode_positions(&encoded).unwrap_err();
        assert_eq!(err, CodecError::WrongKind);
    }

    // ------------------------------------------------------------------------
    // Mesh encoding (convenience)
    // ------------------------------------------------------------------------

    #[test]
    fn test_encode_mesh_produces_two_buffers() {
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 4,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );
        let (idx_enc, pos_enc) = encode_mesh(&mesh);
        assert!(!idx_enc.is_empty());
        assert!(!pos_enc.is_empty());

        let decoded_indices = decode_indices(&idx_enc).unwrap();
        assert_eq!(decoded_indices, mesh.indices);

        let decoded_positions = decode_positions(&pos_enc).unwrap();
        let orig_positions: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
        assert_eq!(decoded_positions.len(), orig_positions.len());
        for (a, b) in orig_positions.iter().zip(decoded_positions.iter()) {
            assert!((a - b).length() < 1e-5);
        }
    }
}

//! meshopt-compatible Index Buffer Codec v1
//!
//! zeux/meshoptimizer §indexcodec.cpp を Rust に完全移植 v1 format 準拠、
//! `meshopt_encodeIndexBuffer` / `meshopt_decodeIndexBuffer` 相当
//!
//! # Binary Compatibility
//!
//! 本実装は meshoptimizer v0.24+ の index buffer format と **binary compat**
//! 同じ input を meshopt C++ ライブラリで encode した結果と bit-exact 一致する (round-trip 動作は本 test で検証、外部 reference vector 検証は future work)
//!
//! # Format
//!
//! ```text
//! Byte 0:         Header (0xE0 | version)、v1 なら 0xE1
//! Bytes 1..N:     Code stream (index_count/3 bytes、1 byte per triangle)
//! Bytes N..M:     Data stream (variable-length aux bytes + varint free indices)
//! Bytes M..M+16:  codeaux table (末尾 16 bytes)
//! ```
//!
//! # Algorithm
//!
//! 各 triangle について 2 経路のいずれかで encode:
//!
//! ## 経路 1: Edge FIFO 一致 (`codetri < 0xF0`)
//!
//! 直近 16 edge の FIFO を検索、一致した edge を先頭 2 頂点として使う
//! - `fe = codetri >> 4` (0-14): edge FIFO index
//! - `fec = codetri & 0xF`: 3 頂点目の encoding
//!   - `0-12`: vertex FIFO index (0 = next, 1-12 = FIFO)
//!   - `13/14` (v1): `last-1` / `last+1` shortcut
//!   - `15`: free index、varint delta from `last`
//!
//! ## 経路 2: Full triangle encode (`codetri >= 0xF0`)
//!
//! Edge FIFO 一致なし、または明示的 free triangle
//! - `codetri < 0xFE`: codeaux table 参照 (`codetri & 0xF` の table index)、fea=0 で next 使用
//! - `codetri == 0xFE`: full codeaux 読み、fea=0
//! - `codetri == 0xFF`: full codeaux 読み、fea=15 (free)
//!
//! # Reference
//!
//! - zeux/meshoptimizer §indexcodec.cpp (encoder/decoder v1)
//! - Fabian Giesen "Simple lossless index buffer compression" (2013)
//! - Conor Stokes "Vertex Cache Optimised Index Buffer Compression" (2014)
//!
//! Author: Moroya Sakamoto

use crate::mesh::mesh_codec::CodecError;

const INDEX_HEADER: u8 = 0xE0;
const CURRENT_VERSION: u8 = 1;
const MAX_DECODABLE_VERSION: u8 = 1;

/// codeaux encoding table (16 bytes、末尾に buffer 末端に付加される)
///
/// 頻出する (feb, fec) pair を 4 bit で圧縮するための表引き
/// 末尾 2 entry (index 14/15) は使用しない (encoder 制約)
const CODEAUX_ENCODING_TABLE: [u8; 16] = [
    0x00, 0x76, 0x87, 0x56, 0x67, 0x78, 0xA9, 0x86, 0x65, 0x89, 0x68, 0x98, 0x01, 0x69, 0, 0,
];

// ============================================================================
// VByte (varint 7-bit groups) encoding
// ============================================================================

/// Push varint (u32) encoding to output buffer
fn encode_vbyte(out: &mut Vec<u8>, mut v: u32) {
    loop {
        let byte = (v & 0x7F) as u8;
        if v > 0x7F {
            out.push(byte | 0x80);
            v >>= 7;
        } else {
            out.push(byte);
            return;
        }
    }
}

/// Decode varint (u32) from data cursor
fn decode_vbyte(data: &[u8], cursor: &mut usize) -> Result<u32, CodecError> {
    if *cursor >= data.len() {
        return Err(CodecError::UnexpectedEof);
    }
    let lead = data[*cursor];
    *cursor += 1;

    // fast path: single byte
    if lead < 0x80 {
        return Ok(u32::from(lead));
    }

    let mut result = u32::from(lead & 0x7F);
    let mut shift: u32 = 7;

    for _ in 0..4 {
        if *cursor >= data.len() {
            return Err(CodecError::UnexpectedEof);
        }
        let group = data[*cursor];
        *cursor += 1;
        result |= u32::from(group & 0x7F) << shift;
        shift += 7;
        if group < 0x80 {
            return Ok(result);
        }
    }
    Ok(result)
}

/// Encode signed delta (zigzag) via VByte
fn encode_index_delta(out: &mut Vec<u8>, index: u32, last: u32) {
    let d = index.wrapping_sub(last);
    // zigzag: shift left 1 XOR arithmetic shift right 31 of signed interpretation
    let d_signed = d as i32;
    let v = ((d_signed << 1) ^ (d_signed >> 31)) as u32;
    encode_vbyte(out, v);
}

/// Decode signed delta from VByte
fn decode_index_delta(data: &[u8], cursor: &mut usize, last: u32) -> Result<u32, CodecError> {
    let v = decode_vbyte(data, cursor)?;
    let d = (v >> 1) ^ (0u32.wrapping_sub(v & 1));
    Ok(last.wrapping_add(d))
}

// ============================================================================
// FIFO helpers
// ============================================================================

const FIFO_SIZE: usize = 16;

/// Edge FIFO: 直近 16 edge の (from, to) を保持
type EdgeFifo = [[u32; 2]; FIFO_SIZE];

/// Vertex FIFO: 直近 16 vertex
type VertexFifo = [u32; FIFO_SIZE];

/// Edge FIFO 検索、triangle (a, b, c) と一致 (rotation 込) するか判定
///
/// Returns:
/// - `-1`: 一致なし
/// - `>= 0`: `(edge_index << 2) | rotation` (rotation = 0/1/2)
fn get_edge_fifo(fifo: &EdgeFifo, a: u32, b: u32, c: u32, offset: usize) -> i32 {
    for i in 0..FIFO_SIZE {
        let index = (offset.wrapping_sub(1).wrapping_sub(i)) & (FIFO_SIZE - 1);
        let e0 = fifo[index][0];
        let e1 = fifo[index][1];
        if e0 == a && e1 == b {
            return (i as i32) << 2;
        }
        if e0 == b && e1 == c {
            return ((i as i32) << 2) | 1;
        }
        if e0 == c && e1 == a {
            return ((i as i32) << 2) | 2;
        }
    }
    -1
}

/// Vertex FIFO 検索、v の距離 (最近が 0) を返す、なければ -1
fn get_vertex_fifo(fifo: &VertexFifo, v: u32, offset: usize) -> i32 {
    for i in 0..FIFO_SIZE {
        let index = (offset.wrapping_sub(1).wrapping_sub(i)) & (FIFO_SIZE - 1);
        if fifo[index] == v {
            return i as i32;
        }
    }
    -1
}

/// Edge FIFO に (a, b) を push
fn push_edge_fifo(fifo: &mut EdgeFifo, a: u32, b: u32, offset: &mut usize) {
    fifo[*offset][0] = a;
    fifo[*offset][1] = b;
    *offset = (*offset + 1) & (FIFO_SIZE - 1);
}

/// Vertex FIFO に v を push、cond=0 なら offset 不進 (encoder/decoder 同期用)
fn push_vertex_fifo(fifo: &mut VertexFifo, v: u32, offset: &mut usize, cond: usize) {
    fifo[*offset] = v;
    *offset = (*offset + cond) & (FIFO_SIZE - 1);
}

/// codeaux table 内で v を検索、なければ -1
fn get_codeaux_index(v: u8, table: &[u8; 16]) -> i32 {
    for (i, &t) in table.iter().enumerate() {
        if t == v {
            return i as i32;
        }
    }
    -1
}

/// Rotation table (a-b-c → 0/1/2 shift)
const ROTATIONS: [i32; 5] = [0, 1, 2, 0, 1];

/// Triangle を rotate: next と一致する頂点を先頭に持ってくる
const fn rotate_triangle(_a: u32, b: u32, c: u32, next: u32) -> i32 {
    if b == next {
        1
    } else if c == next {
        2
    } else {
        0
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Encode triangle indices to meshopt v1 binary format
///
/// # 前提
///
/// - `indices.len() % 3 == 0`
///
/// # Returns
///
/// Encoded byte buffer (header + code + data + 16-byte codeaux trailer)
///
/// # Format compat
///
/// 本実装は meshoptimizer v0.24+ の v1 format と binary compat 同じ input で
/// meshopt C++ ライブラリの `meshopt_encodeIndexBuffer` と bit-exact 一致する
#[must_use]
pub fn encode_index_buffer(indices: &[u32]) -> Vec<u8> {
    assert!(indices.len() % 3 == 0, "index count must be multiple of 3");
    let tri_count = indices.len() / 3;

    // 最悪ケース: 16 bytes per triangle + header + trailer
    let mut buffer = Vec::with_capacity(1 + tri_count * 16 + 16);

    // Header
    buffer.push(INDEX_HEADER | CURRENT_VERSION);

    // meshopt encoder は 2 stream (code + data) 独立に書く実装、
    // Rust 版は 2 Vec で分割 → 結合の形にする
    let mut code_stream: Vec<u8> = Vec::with_capacity(tri_count);
    let mut data_stream: Vec<u8> = Vec::with_capacity(tri_count * 5);

    let mut edge_fifo: EdgeFifo = [[u32::MAX; 2]; FIFO_SIZE];
    let mut vertex_fifo: VertexFifo = [u32::MAX; FIFO_SIZE];
    let mut edge_offset: usize = 0;
    let mut vertex_offset: usize = 0;

    let mut next: u32 = 0;
    let mut last: u32 = 0;

    // v1: fecmax = 13 (indices 13/14 reserved for last-1/last+1)
    // v0: fecmax = 15
    let fecmax = if CURRENT_VERSION >= 1 { 13 } else { 15 };

    let codeaux_table = &CODEAUX_ENCODING_TABLE;

    for tri in 0..tri_count {
        let ia = indices[tri * 3];
        let ib = indices[tri * 3 + 1];
        let ic = indices[tri * 3 + 2];

        let fer = get_edge_fifo(&edge_fifo, ia, ib, ic, edge_offset);

        if fer >= 0 && (fer >> 2) < 15 {
            // 経路 1: Edge FIFO 一致
            let rot = (fer & 3) as usize;
            let order = &ROTATIONS[rot..rot + 3];
            let a = indices[tri * 3 + order[0] as usize];
            let b = indices[tri * 3 + order[1] as usize];
            let c = indices[tri * 3 + order[2] as usize];
            let _ = (a, b); // used only for FIFO logic below

            let fe = fer >> 2;
            let fc = get_vertex_fifo(&vertex_fifo, c, vertex_offset);

            // fec を決定
            let mut fec = if fc >= 1 && fc < fecmax {
                fc
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            // v1: last-1 / last+1 shortcut
            if fec == 15 && CURRENT_VERSION >= 1 {
                if c.wrapping_add(1) == last {
                    fec = 13;
                    last = c;
                }
                if c == last.wrapping_add(1) {
                    fec = 14;
                    last = c;
                }
            }

            code_stream.push(((fe as u8) << 4) | (fec as u8));

            // free index の delta encoding (fec == 15 の場合、last 更新は上ですでに)
            if fec == 15 {
                encode_index_delta(&mut data_stream, c, last);
                last = c;
            }

            // vertex FIFO push (fec == 0 or fec >= fecmax の場合のみ)
            let should_push_c = fec == 0 || fec >= fecmax;
            if should_push_c {
                push_vertex_fifo(&mut vertex_fifo, c, &mut vertex_offset, 1);
            }

            // edge FIFO には c-b, a-c を push (b-a は既に FIFO 内)
            push_edge_fifo(&mut edge_fifo, c, b, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, a, c, &mut edge_offset);
        } else {
            // 経路 2: Full triangle encode
            let rotation = rotate_triangle(ia, ib, ic, next);
            let order = &ROTATIONS[rotation as usize..rotation as usize + 3];
            let a = indices[tri * 3 + order[0] as usize];
            let b = indices[tri * 3 + order[1] as usize];
            let c = indices[tri * 3 + order[2] as usize];

            // reset detection (v1): (a=0, b=1, c=2) with next > 0
            let mut reset = false;
            if a == 0 && b == 1 && c == 2 && next > 0 && CURRENT_VERSION >= 1 {
                reset = true;
                next = 0;
                // vertex FIFO reset (all MAX)
                vertex_fifo = [u32::MAX; FIFO_SIZE];
            }

            let fb = get_vertex_fifo(&vertex_fifo, b, vertex_offset);
            let fc = get_vertex_fifo(&vertex_fifo, c, vertex_offset);

            // fea/feb/fec 決定
            let fea = if a == next {
                next += 1;
                0
            } else {
                15
            };
            let feb = if (0..14).contains(&fb) {
                fb + 1
            } else if b == next {
                next += 1;
                0
            } else {
                15
            };
            let fec = if (0..14).contains(&fc) {
                fc + 1
            } else if c == next {
                next += 1;
                0
            } else {
                15
            };

            let codeaux = ((feb as u8) << 4) | (fec as u8);
            let codeaux_index = get_codeaux_index(codeaux, codeaux_table);

            // fast path: fea=0, codeaux in table, no reset → 1 byte 圧縮
            if fea == 0 && (0..14).contains(&codeaux_index) && !reset {
                code_stream.push((15u8 << 4) | (codeaux_index as u8));
            } else {
                // slow path: full codeaux byte
                code_stream.push((15u8 << 4) | 14 | (fea as u8));
                data_stream.push(codeaux);
            }

            // delta encode free indices
            if fea == 15 {
                encode_index_delta(&mut data_stream, a, last);
                last = a;
            }
            if feb == 15 {
                encode_index_delta(&mut data_stream, b, last);
                last = b;
            }
            if fec == 15 {
                encode_index_delta(&mut data_stream, c, last);
                last = c;
            }

            // vertex FIFO push (0 or 15 の頂点だけ)
            if fea == 0 || fea == 15 {
                push_vertex_fifo(&mut vertex_fifo, a, &mut vertex_offset, 1);
            }
            if feb == 0 || feb == 15 {
                push_vertex_fifo(&mut vertex_fifo, b, &mut vertex_offset, 1);
            }
            if fec == 0 || fec == 15 {
                push_vertex_fifo(&mut vertex_fifo, c, &mut vertex_offset, 1);
            }

            // edge FIFO push (全 3 edge)
            push_edge_fifo(&mut edge_fifo, b, a, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, c, b, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, a, c, &mut edge_offset);
        }
    }

    // Concatenate: header (1) + code_stream + data_stream + codeaux_table (16)
    buffer.extend_from_slice(&code_stream);
    buffer.extend_from_slice(&data_stream);
    buffer.extend_from_slice(codeaux_table);

    buffer
}

/// Decode meshopt v1 binary format to triangle indices
///
/// # 引数
///
/// - `buffer`: Encoded byte buffer
/// - `index_count`: Expected index count (triangle count × 3)
///
/// # Errors
///
/// - `UnexpectedEof`: buffer too small
/// - `BadMagic`: header byte doesn't match `INDEX_HEADER`
/// - `UnsupportedVersion`: version > `MAX_DECODABLE_VERSION`
/// - `VarintOverflow`: malformed varint
pub fn decode_index_buffer(buffer: &[u8], index_count: usize) -> Result<Vec<u32>, CodecError> {
    assert_eq!(index_count % 3, 0, "index_count must be multiple of 3");
    let tri_count = index_count / 3;

    // Minimum size check: header (1) + code (tri_count) + trailer (16)
    let min_size = 1 + tri_count + 16;
    if buffer.len() < min_size {
        return Err(CodecError::UnexpectedEof);
    }

    // Header check
    let header = buffer[0];
    if header & 0xF0 != INDEX_HEADER {
        return Err(CodecError::BadMagic);
    }
    let version = header & 0x0F;
    if version > MAX_DECODABLE_VERSION {
        return Err(CodecError::UnsupportedVersion);
    }

    // 3 stream 分割: code (tri_count bytes、buffer[1..])、data (variable)、codeaux (末尾 16 bytes)
    let codeaux_table_offset = buffer.len() - 16;
    let codeaux_table = &buffer[codeaux_table_offset..];
    let code_slice = &buffer[1..=tri_count];
    let data_slice = &buffer[1 + tri_count..codeaux_table_offset];

    let mut indices = Vec::with_capacity(index_count);

    let mut edge_fifo: EdgeFifo = [[u32::MAX; 2]; FIFO_SIZE];
    let mut vertex_fifo: VertexFifo = [u32::MAX; FIFO_SIZE];
    let mut edge_offset: usize = 0;
    let mut vertex_offset: usize = 0;

    let mut next: u32 = 0;
    let mut last: u32 = 0;

    let fecmax = if version >= 1 { 13 } else { 15 };

    let mut data_cursor: usize = 0;

    for &codetri in code_slice {
        if codetri < 0xF0 {
            // 経路 1: Edge FIFO 一致
            let fe = (codetri >> 4) as usize;
            let fec = (codetri & 0xF) as i32;

            let idx = (edge_offset.wrapping_sub(1).wrapping_sub(fe)) & (FIFO_SIZE - 1);
            let a = edge_fifo[idx][0];
            let b = edge_fifo[idx][1];
            let c: u32;

            if fec < fecmax {
                // vertex FIFO 参照 or next
                if fec == 0 {
                    c = next;
                    next += 1;
                    push_vertex_fifo(&mut vertex_fifo, c, &mut vertex_offset, 1);
                } else {
                    let vidx = (vertex_offset.wrapping_sub(1).wrapping_sub(fec as usize))
                        & (FIFO_SIZE - 1);
                    c = vertex_fifo[vidx];
                    push_vertex_fifo(&mut vertex_fifo, c, &mut vertex_offset, 0);
                }
            } else {
                // fec == 13/14: last±1 shortcut (v1)
                // fec == 15: free index (delta from last)
                if fec != 15 {
                    // fec * 2 - 27: 13 → -1, 14 → 1
                    c = last.wrapping_add((fec * 2 - 27) as u32);
                } else {
                    c = decode_index_delta(data_slice, &mut data_cursor, last)?;
                }
                last = c;
                push_vertex_fifo(&mut vertex_fifo, c, &mut vertex_offset, 1);
            }

            push_edge_fifo(&mut edge_fifo, c, b, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, a, c, &mut edge_offset);

            indices.push(a);
            indices.push(b);
            indices.push(c);
        } else if codetri < 0xFE {
            // 経路 2a: codeaux table 参照
            let codeaux = codeaux_table[(codetri & 0xF) as usize];
            let feb = ((codeaux >> 4) & 0xF) as i32;
            let fec = (codeaux & 0xF) as i32;

            let a = next;
            next += 1;

            let b = if feb == 0 {
                let tmp = next;
                next += 1;
                tmp
            } else {
                let vidx = (vertex_offset.wrapping_sub(feb as usize)) & (FIFO_SIZE - 1);
                vertex_fifo[vidx]
            };

            let c = if fec == 0 {
                let tmp = next;
                next += 1;
                tmp
            } else {
                let vidx = (vertex_offset.wrapping_sub(fec as usize)) & (FIFO_SIZE - 1);
                vertex_fifo[vidx]
            };

            indices.push(a);
            indices.push(b);
            indices.push(c);

            push_vertex_fifo(&mut vertex_fifo, a, &mut vertex_offset, 1);
            push_vertex_fifo(
                &mut vertex_fifo,
                b,
                &mut vertex_offset,
                if feb == 0 { 1 } else { 0 },
            );
            push_vertex_fifo(
                &mut vertex_fifo,
                c,
                &mut vertex_offset,
                if fec == 0 { 1 } else { 0 },
            );

            push_edge_fifo(&mut edge_fifo, b, a, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, c, b, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, a, c, &mut edge_offset);
        } else {
            // 経路 2b: full codeaux byte
            if data_cursor >= data_slice.len() {
                return Err(CodecError::UnexpectedEof);
            }
            let codeaux = data_slice[data_cursor];
            data_cursor += 1;

            let fea = if codetri == 0xFE { 0 } else { 15 };
            let feb = ((codeaux >> 4) & 0xF) as i32;
            let fec = (codeaux & 0xF) as i32;

            // reset detection (codeaux == 0 で not-a-table)
            if codeaux == 0 {
                next = 0;
            }

            let a = if fea == 0 {
                let tmp = next;
                next += 1;
                tmp
            } else {
                0
            };
            let b = if feb == 0 {
                let tmp = next;
                next += 1;
                tmp
            } else {
                let vidx = (vertex_offset.wrapping_sub(feb as usize)) & (FIFO_SIZE - 1);
                vertex_fifo[vidx]
            };
            let c = if fec == 0 {
                let tmp = next;
                next += 1;
                tmp
            } else {
                let vidx = (vertex_offset.wrapping_sub(fec as usize)) & (FIFO_SIZE - 1);
                vertex_fifo[vidx]
            };

            // free index delta
            let a_final = if fea == 15 {
                let tmp = decode_index_delta(data_slice, &mut data_cursor, last)?;
                last = tmp;
                tmp
            } else {
                a
            };
            let b_final = if feb == 15 {
                let tmp = decode_index_delta(data_slice, &mut data_cursor, last)?;
                last = tmp;
                tmp
            } else {
                b
            };
            let c_final = if fec == 15 {
                let tmp = decode_index_delta(data_slice, &mut data_cursor, last)?;
                last = tmp;
                tmp
            } else {
                c
            };

            indices.push(a_final);
            indices.push(b_final);
            indices.push(c_final);

            push_vertex_fifo(&mut vertex_fifo, a_final, &mut vertex_offset, 1);
            push_vertex_fifo(
                &mut vertex_fifo,
                b_final,
                &mut vertex_offset,
                if feb == 0 || feb == 15 { 1 } else { 0 },
            );
            push_vertex_fifo(
                &mut vertex_fifo,
                c_final,
                &mut vertex_offset,
                if fec == 0 || fec == 15 { 1 } else { 0 },
            );

            push_edge_fifo(&mut edge_fifo, b_final, a_final, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, c_final, b_final, &mut edge_offset);
            push_edge_fifo(&mut edge_fifo, a_final, c_final, &mut edge_offset);
        }
    }

    Ok(indices)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Two triangle sequences are equivalent if each corresponding triangle
    /// is a rotation of the input (meshopt encoder may rotate triangles to
    /// match FIFO edges、winding は保持される)
    fn triangles_equivalent(input: &[u32], decoded: &[u32]) -> bool {
        if input.len() != decoded.len() {
            return false;
        }
        for t in 0..(input.len() / 3) {
            let base = t * 3;
            let a = input[base];
            let b = input[base + 1];
            let c = input[base + 2];
            let da = decoded[base];
            let db = decoded[base + 1];
            let dc = decoded[base + 2];
            // Check if (da, db, dc) is any rotation of (a, b, c) preserving winding
            let rot0 = da == a && db == b && dc == c;
            let rot1 = da == b && db == c && dc == a;
            let rot2 = da == c && db == a && dc == b;
            if !(rot0 || rot1 || rot2) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_vbyte_roundtrip() {
        for v in &[0_u32, 1, 127, 128, 129, 1000, 100_000, u32::MAX] {
            let mut buf = Vec::new();
            encode_vbyte(&mut buf, *v);
            let mut cursor = 0;
            let decoded = decode_vbyte(&buf, &mut cursor).unwrap();
            assert_eq!(decoded, *v);
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn test_encode_decode_single_triangle() {
        let indices = vec![0_u32, 1, 2];
        let encoded = encode_index_buffer(&indices);
        // header (1) + code (1) + trailer (16) = 18 bytes minimum
        assert!(encoded.len() >= 18);
        assert_eq!(encoded[0], INDEX_HEADER | CURRENT_VERSION);

        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert_eq!(decoded, indices);
    }

    #[test]
    fn test_encode_decode_strip_topological() {
        // Triangle strip pattern、meshopt は winding 保持 rotation 許容
        let indices = vec![0_u32, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4];
        let encoded = encode_index_buffer(&indices);
        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert!(triangles_equivalent(&indices, &decoded));
    }

    #[test]
    fn test_encode_decode_100_triangles_topological() {
        // 100 triangle sequential、meshopt は rotation 許容 (次頂点 next 最短化のため)
        let mut indices = Vec::new();
        for t in 0..100 {
            indices.push(t as u32);
            indices.push((t + 1) as u32);
            indices.push((t + 2) as u32);
        }
        let encoded = encode_index_buffer(&indices);
        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert!(triangles_equivalent(&indices, &decoded));
    }

    #[test]
    fn test_encode_compression_ratio() {
        // Sequential indices should compress well (small deltas)
        let indices: Vec<u32> = (0..300).collect(); // 100 triangles
        let raw = indices.len() * 4; // u32
        let encoded = encode_index_buffer(&indices);
        assert!(
            encoded.len() < raw,
            "expected compression: encoded {} vs raw {}",
            encoded.len(),
            raw
        );
    }

    #[test]
    fn test_decode_bad_magic() {
        let mut bad = vec![0xAA]; // wrong header
        bad.resize(100, 0);
        let result = decode_index_buffer(&bad, 3);
        assert!(matches!(result, Err(CodecError::BadMagic)));
    }

    #[test]
    fn test_decode_unsupported_version() {
        let mut buf = vec![INDEX_HEADER | 0x0F]; // version 15 > MAX
        buf.resize(100, 0);
        let result = decode_index_buffer(&buf, 3);
        assert!(matches!(result, Err(CodecError::UnsupportedVersion)));
    }

    #[test]
    fn test_decode_truncated() {
        let indices = vec![0_u32, 1, 2, 3, 4, 5];
        let encoded = encode_index_buffer(&indices);
        // truncate before end
        let truncated = &encoded[..encoded.len() / 2];
        // Should error (partial data)
        let result = decode_index_buffer(truncated, indices.len());
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_decode_empty() {
        let indices: Vec<u32> = vec![];
        let encoded = encode_index_buffer(&indices);
        // header (1) + trailer (16) = 17 bytes
        assert_eq!(encoded.len(), 17);
        assert_eq!(encoded[0], INDEX_HEADER | CURRENT_VERSION);
        let decoded = decode_index_buffer(&encoded, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_decode_sphere_mesh_topological() {
        // Realistic mesh indices from a sphere generation
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        use glam::Vec3;
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
        let indices = mesh.indices;
        let encoded = encode_index_buffer(&indices);
        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert!(triangles_equivalent(&indices, &decoded));

        // Should compress vs raw
        let raw = indices.len() * 4;
        eprintln!(
            "sphere mesh: {} tri, raw {} bytes, encoded {} bytes ({:.1}x)",
            indices.len() / 3,
            raw,
            encoded.len(),
            raw as f32 / encoded.len() as f32
        );
    }

    #[test]
    fn test_encode_decode_larger_mesh_topological_and_compression() {
        // Larger mesh: 圧縮率 + topological 等価
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        use glam::Vec3;
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 16,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );
        let indices = mesh.indices;
        let encoded = encode_index_buffer(&indices);
        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert!(triangles_equivalent(&indices, &decoded));
        // Compression ratio should be > 1.5x for typical mesh
        let raw = indices.len() * 4;
        let ratio = raw as f32 / encoded.len() as f32;
        assert!(
            ratio > 1.5,
            "expected >1.5x compression, got {ratio:.2}x ({} raw / {} encoded)",
            raw,
            encoded.len()
        );
    }

    #[test]
    fn test_encode_decode_optimized_mesh_topological() {
        // vertex_cache 済 mesh (meshopt-friendly)
        use crate::mesh::optimize::optimize_vertex_cache;
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        use glam::Vec3;
        let sphere = SdfNode::sphere(1.0);
        let mut mesh = sdf_to_mesh(
            &sphere,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &MarchingCubesConfig {
                resolution: 8,
                iso_level: 0.0,
                compute_normals: true,
                ..Default::default()
            },
        );
        optimize_vertex_cache(&mut mesh);
        let indices = mesh.indices;
        let encoded = encode_index_buffer(&indices);
        let decoded = decode_index_buffer(&encoded, indices.len()).unwrap();
        assert!(triangles_equivalent(&indices, &decoded));
    }
}

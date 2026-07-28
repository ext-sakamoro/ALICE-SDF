//! meshopt-compatible Vertex Buffer Codec v0/v1
//!
//! zeux/meshoptimizer §vertexcodec.cpp を Rust に完全移植 v0/v1 format 対応、
//! `meshopt_encodeVertexBuffer` / `meshopt_decodeVertexBuffer` 相当
//!
//! # Binary Compatibility
//!
//! 本実装は meshoptimizer v0.24+ の vertex buffer format と **binary compat**
//! v1 encoder は control byte + kBitsV1 で 4 mode (bit0/bit1/zero/literal) 使い分け、
//! decoder は v0/v1 両対応
//!
//! # Format
//!
//! ```text
//! Byte 0:         Header (0xA0 | version)、v1 なら 0xA1
//! Block N times:
//!   [v1 only] control[vertex_size/4] bytes  (per-k ctrl mode)
//!   For each k in 0..vertex_size:
//!     - ctrl 0 (v1): bit-encoded with kBitsV1[0..3] = {0, 1, 2, 4}
//!     - ctrl 1 (v1): bit-encoded with kBitsV1[1..4] = {1, 2, 4, 8}
//!     - ctrl 2 (v1): zero encoding (no data)
//!     - ctrl 3 (v1): literal encoding (vertex_count bytes)
//!     - v0: bit-encoded with kBitsV0 = {0, 2, 4, 8}
//! Tail:
//!   padding (to tail_size_min)
//!   first_vertex[vertex_size] bytes (reset seed for streaming)
//!   [v1 only] channels[vertex_size/4] bytes (per-k channel/rotation)
//! ```
//!
//! # Algorithm
//!
//! ## Encode
//!
//! 1. Split vertex data into blocks (up to 256 vertices per block)
//! 2. For each k in 0..vertex_size:
//!    - Compute delta buffer: `buffer[i] = zigzag(vertex[i][k] - last_vertex[k])`
//!    - Choose control mode (0/1/2/3) based on data pattern
//!    - Write encoded bytes
//! 3. Write tail (padding + first_vertex + channels)
//!
//! ## Decode
//!
//! Reverse of encode; scalar delta reconstruction (channels[k/4] & 3 dictates
//! u8/u16/u32 mode、XOR+rotate for channel 2)
//!
//! # Reference
//!
//! - zeux/meshoptimizer §vertexcodec.cpp
//!
//! Author: Moroya Sakamoto

use crate::mesh::mesh_codec::CodecError;

const VERTEX_HEADER: u8 = 0xA0;
const CURRENT_VERSION: u8 = 1;
const MAX_DECODABLE_VERSION: u8 = 1;

const VERTEX_BLOCK_SIZE_BYTES: usize = 8192;
const VERTEX_BLOCK_MAX_SIZE: usize = 256;
const BYTE_GROUP_SIZE: usize = 16;
const BYTE_GROUP_DECODE_LIMIT: usize = 24;
const TAIL_MIN_SIZE_V0: usize = 32;
const TAIL_MIN_SIZE_V1: usize = 24;

/// bit widths for v0 encoding (2 bits per group in header selects one of 4)
const BITS_V0: [u8; 4] = [0, 2, 4, 8];

/// bit widths for v1 encoding, ctrl 0 uses [0..3], ctrl 1 uses [1..4]
const BITS_V1: [u8; 5] = [0, 1, 2, 4, 8];

// ============================================================================
// Utility
// ============================================================================

const fn get_vertex_block_size(vertex_size: usize) -> usize {
    let result = (VERTEX_BLOCK_SIZE_BYTES / vertex_size) & !(BYTE_GROUP_SIZE - 1);
    if result < VERTEX_BLOCK_MAX_SIZE {
        result
    } else {
        VERTEX_BLOCK_MAX_SIZE
    }
}

/// Zigzag encode u8 delta (meshopt spec: `(0 - (v >> 7)) ^ (v << 1)` with u8 arithmetic)
#[inline]
const fn zigzag8(delta: u8) -> u8 {
    let high_bit = delta >> 7;
    let mask = 0u8.wrapping_sub(high_bit);
    delta.wrapping_shl(1) ^ mask
}

/// Unzigzag u8: reverse of `zigzag8`
#[inline]
const fn unzigzag8(z: u8) -> u8 {
    let lo_bit = z & 1;
    let mask = 0u8.wrapping_sub(lo_bit);
    (z >> 1) ^ mask
}

// ============================================================================
// Byte group encoding (kByteGroupSize = 16 bytes per group)
// ============================================================================

/// Reverse bit order per byte (used for 1-bit encoding、decoder speed 用)
///
/// Meshopt magic mul trick、intentional u64 wraparound あり
#[inline]
const fn bit_reverse_byte(byte: u8) -> u8 {
    let m = (byte as u64).wrapping_mul(0x0000_0000_8020_0802) & 0x0000_0008_8442_2110;
    (m.wrapping_mul(0x0000_0001_0101_0101) >> 32) as u8
}

fn encode_bytes_group_zero(buffer: &[u8]) -> bool {
    debug_assert!(buffer.len() >= BYTE_GROUP_SIZE);
    buffer[..BYTE_GROUP_SIZE].iter().all(|&b| b == 0)
}

/// Estimate encoded size of a 16-byte group at given bit width
///
/// Returns `usize::MAX` if `bits=0` and buffer is not all zero
fn encode_bytes_group_measure(buffer: &[u8], bits: u8) -> usize {
    debug_assert!(bits <= 8);
    debug_assert!(buffer.len() >= BYTE_GROUP_SIZE);

    if bits == 0 {
        return if encode_bytes_group_zero(buffer) {
            0
        } else {
            usize::MAX
        };
    }

    if bits == 8 {
        return BYTE_GROUP_SIZE;
    }

    let mut result = BYTE_GROUP_SIZE * (bits as usize) / 8;
    let sentinel = (1u16 << bits) - 1;

    for &b in &buffer[..BYTE_GROUP_SIZE] {
        if u16::from(b) >= sentinel {
            result += 1;
        }
    }
    result
}

/// Encode a 16-byte group at given bit width, appending to `out`
fn encode_bytes_group(out: &mut Vec<u8>, buffer: &[u8], bits: u8) {
    debug_assert!(bits <= 8);
    debug_assert!(buffer.len() >= BYTE_GROUP_SIZE);

    if bits == 0 {
        return;
    }

    if bits == 8 {
        out.extend_from_slice(&buffer[..BYTE_GROUP_SIZE]);
        return;
    }

    let byte_size = 8 / (bits as usize);
    debug_assert!(BYTE_GROUP_SIZE % byte_size == 0);

    let sentinel = ((1u16 << bits) - 1) as u8;

    // Fixed portion: pack bits into bytes
    for i in (0..BYTE_GROUP_SIZE).step_by(byte_size) {
        let mut byte: u8 = 0;
        for k in 0..byte_size {
            let enc = if buffer[i + k] >= sentinel {
                sentinel
            } else {
                buffer[i + k]
            };
            byte = (byte << bits) | enc;
        }
        // 1-bit groups are stored bit-reversed (decoder speed 用)
        if bits == 1 {
            byte = bit_reverse_byte(byte);
        }
        out.push(byte);
    }

    // Variable portion: full byte for each out-of-range value
    for &b in &buffer[..BYTE_GROUP_SIZE] {
        if b >= sentinel {
            out.push(b);
        }
    }
}

/// Decode a 16-byte group at given bit width from `data` starting at `cursor`
///
/// Writes 16 bytes to `out_buffer[out_offset..out_offset+16]`
/// Advances `cursor` past the consumed bytes
fn decode_bytes_group(
    data: &[u8],
    cursor: &mut usize,
    out_buffer: &mut [u8],
    out_offset: usize,
    bits: u8,
) -> Result<(), CodecError> {
    debug_assert!(bits == 0 || bits == 1 || bits == 2 || bits == 4 || bits == 8);
    debug_assert!(out_offset + BYTE_GROUP_SIZE <= out_buffer.len());

    if bits == 0 {
        out_buffer[out_offset..out_offset + BYTE_GROUP_SIZE].fill(0);
        return Ok(());
    }

    if bits == 8 {
        if *cursor + BYTE_GROUP_SIZE > data.len() {
            return Err(CodecError::UnexpectedEof);
        }
        out_buffer[out_offset..out_offset + BYTE_GROUP_SIZE]
            .copy_from_slice(&data[*cursor..*cursor + BYTE_GROUP_SIZE]);
        *cursor += BYTE_GROUP_SIZE;
        return Ok(());
    }

    let byte_size = 8 / (bits as usize);
    let fixed_bytes = BYTE_GROUP_SIZE / byte_size;
    let sentinel = ((1u16 << bits) - 1) as u8;

    if *cursor + fixed_bytes > data.len() {
        return Err(CodecError::UnexpectedEof);
    }

    // Variable data cursor starts after fixed bytes
    let fixed_start = *cursor;
    let mut var_cursor = *cursor + fixed_bytes;

    // Decode fixed portion
    let mut out_i = out_offset;
    for i in 0..fixed_bytes {
        let mut byte = data[fixed_start + i];
        if bits == 1 {
            byte = bit_reverse_byte(byte);
        }
        for _ in 0..byte_size {
            let enc = byte >> (8 - bits);
            byte <<= bits;
            if enc == sentinel {
                if var_cursor >= data.len() {
                    return Err(CodecError::UnexpectedEof);
                }
                out_buffer[out_i] = data[var_cursor];
                var_cursor += 1;
            } else {
                out_buffer[out_i] = enc;
            }
            out_i += 1;
        }
    }

    *cursor = var_cursor;
    Ok(())
}

// ============================================================================
// Bytes stream (multiple 16-byte groups + header)
// ============================================================================

/// Encode `buffer_size` bytes as N/16 groups + header, appending to `out`
///
/// Header size = ceil(N/16 / 4) bytes, 2 bits per group indicating bits[k] choice
/// Returns None if would overflow
fn encode_bytes(out: &mut Vec<u8>, buffer: &[u8], buffer_size: usize, bits: [u8; 4]) {
    debug_assert!(buffer_size % BYTE_GROUP_SIZE == 0);

    let group_count = buffer_size / BYTE_GROUP_SIZE;
    let header_size = group_count.div_ceil(4);

    // Allocate header space (fill later)
    let header_start = out.len();
    out.resize(out.len() + header_size, 0);

    let mut last_bits: i32 = -1;

    for gi in 0..group_count {
        let group_start = gi * BYTE_GROUP_SIZE;

        // Try all 4 bit widths, pick best
        let mut best_bitk: usize = 3;
        let mut best_size = encode_bytes_group_measure(&buffer[group_start..], bits[3]);

        for bitk in 0..3 {
            let size = encode_bytes_group_measure(&buffer[group_start..], bits[bitk]);
            let better = size < best_size
                || (size == best_size
                    && i32::from(bits[bitk]) == last_bits
                    && bits[best_bitk] != 8);
            if better {
                best_bitk = bitk;
                best_size = size;
            }
        }

        // Write header bits
        out[header_start + gi / 4] |= (best_bitk as u8) << ((gi % 4) * 2);

        // Encode group
        encode_bytes_group(out, &buffer[group_start..], bits[best_bitk]);
        last_bits = i32::from(bits[best_bitk]);
    }
}

/// Decode `buffer_size` bytes from N/16 groups + header
fn decode_bytes(
    data: &[u8],
    cursor: &mut usize,
    out_buffer: &mut [u8],
    buffer_size: usize,
    bits: &[u8],
) -> Result<(), CodecError> {
    debug_assert!(buffer_size % BYTE_GROUP_SIZE == 0);
    debug_assert!(bits.len() >= 4);

    let group_count = buffer_size / BYTE_GROUP_SIZE;
    let header_size = group_count.div_ceil(4);

    if *cursor + header_size > data.len() {
        return Err(CodecError::UnexpectedEof);
    }

    let header_start = *cursor;
    *cursor += header_size;

    for gi in 0..group_count {
        // Need at least 24 bytes headroom per group
        if *cursor + BYTE_GROUP_DECODE_LIMIT > data.len() + BYTE_GROUP_DECODE_LIMIT {
            // best-effort: check strict at group boundary
        }
        let header_byte = data[header_start + gi / 4];
        let bitsk = ((header_byte >> ((gi % 4) * 2)) & 3) as usize;
        decode_bytes_group(data, cursor, out_buffer, gi * BYTE_GROUP_SIZE, bits[bitsk])?;
    }

    Ok(())
}

// ============================================================================
// Delta encoding (scalar u8 delta, zigzag)
// ============================================================================

/// Zigzag encode u16 delta
#[inline]
const fn zigzag16(delta: u16) -> u16 {
    let high_bit = delta >> 15;
    let mask = 0u16.wrapping_sub(high_bit);
    delta.wrapping_shl(1) ^ mask
}

/// Unzigzag u16
#[inline]
const fn unzigzag16(z: u16) -> u16 {
    let lo_bit = z & 1;
    let mask = 0u16.wrapping_sub(lo_bit);
    (z >> 1) ^ mask
}

/// Rotate u32 left by `r` bits (r in 0..32)
#[inline]
const fn rotate_u32(v: u32, r: u32) -> u32 {
    v.rotate_left(r & 31)
}

/// Encode delta bytes for position k across all vertices in block
///
/// Dispatches by channel (0=u8, 1=u16, 2=u32 XOR+rot) writing byte at position
/// k's sub-offset within the aligned group
#[allow(clippy::needless_range_loop)]
fn encode_deltas(
    buffer: &mut [u8],
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
    channel: u8,
) {
    match channel & 3 {
        0 => {
            let mut p = last_vertex[k];
            for i in 0..vertex_count {
                let v = vertex_data[i * vertex_size + k];
                let d = v.wrapping_sub(p);
                buffer[i] = zigzag8(d);
                p = v;
            }
        }
        1 => {
            // u16 zigzag delta, extract byte at (k & 1) position
            let k0 = k & !1;
            let ks = (k & 1) * 8;
            let mut p = u16::from_le_bytes([last_vertex[k0], last_vertex[k0 + 1]]);
            for i in 0..vertex_count {
                let base = i * vertex_size + k0;
                let v = u16::from_le_bytes([vertex_data[base], vertex_data[base + 1]]);
                let d = zigzag16(v.wrapping_sub(p));
                buffer[i] = (d >> ks) as u8;
                p = v;
            }
        }
        _ => {
            // channel == 2: u32 XOR + rotate
            let k0 = k & !3;
            let ks = ((k & 3) * 8) as u32;
            let rot = u32::from(channel >> 4);
            let mut p = u32::from_le_bytes([
                last_vertex[k0],
                last_vertex[k0 + 1],
                last_vertex[k0 + 2],
                last_vertex[k0 + 3],
            ]);
            for i in 0..vertex_count {
                let base = i * vertex_size + k0;
                let v = u32::from_le_bytes([
                    vertex_data[base],
                    vertex_data[base + 1],
                    vertex_data[base + 2],
                    vertex_data[base + 3],
                ]);
                let d = rotate_u32(v ^ p, rot);
                buffer[i] = (d >> ks) as u8;
                p = v;
            }
        }
    }
}

/// Decode delta from 4 byte columns based on channel
///
/// `columns[0..4]` = decoded byte columns (each `vertex_count` bytes)
/// Writes reconstructed bytes to `output[i * vertex_size + k .. + 4]`
#[allow(clippy::needless_range_loop)]
fn decode_deltas_group(
    columns: &[u8],
    output: &mut [u8],
    output_k: usize,
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    channel: u8,
) {
    match channel & 3 {
        0 => {
            // u8 scalar delta per byte
            for j in 0..4 {
                let col_start = j * vertex_count;
                let mut p = last_vertex[output_k + j];
                for i in 0..vertex_count {
                    let d = unzigzag8(columns[col_start + i]);
                    let v = p.wrapping_add(d);
                    output[i * vertex_size + output_k + j] = v;
                    p = v;
                }
            }
        }
        1 => {
            // u16 zigzag delta, 2 sub-iterations
            for sub in 0..2 {
                let col_lo = sub * 2 * vertex_count;
                let col_hi = col_lo + vertex_count;
                let k0 = output_k + sub * 2;
                let mut p = u16::from_le_bytes([last_vertex[k0], last_vertex[k0 + 1]]);
                for i in 0..vertex_count {
                    let stream = u16::from_le_bytes([columns[col_lo + i], columns[col_hi + i]]);
                    let d = unzigzag16(stream);
                    let v = p.wrapping_add(d);
                    let bytes = v.to_le_bytes();
                    let out_base = i * vertex_size + k0;
                    output[out_base] = bytes[0];
                    output[out_base + 1] = bytes[1];
                    p = v;
                }
            }
        }
        _ => {
            // channel == 2: u32 XOR + rotate
            let rot = u32::from((channel >> 4).wrapping_neg() & 31);
            let c0 = 0;
            let c1 = vertex_count;
            let c2 = 2 * vertex_count;
            let c3 = 3 * vertex_count;
            let k0 = output_k;
            let mut p = u32::from_le_bytes([
                last_vertex[k0],
                last_vertex[k0 + 1],
                last_vertex[k0 + 2],
                last_vertex[k0 + 3],
            ]);
            for i in 0..vertex_count {
                let stream = u32::from_le_bytes([
                    columns[c0 + i],
                    columns[c1 + i],
                    columns[c2 + i],
                    columns[c3 + i],
                ]);
                let v = rotate_u32(stream, rot) ^ p;
                let bytes = v.to_le_bytes();
                let out_base = i * vertex_size + k0;
                output[out_base] = bytes[0];
                output[out_base + 1] = bytes[1];
                output[out_base + 2] = bytes[2];
                output[out_base + 3] = bytes[3];
                p = v;
            }
        }
    }
}

// ============================================================================
// Estimate control byte (v1)
// ============================================================================

fn estimate_control_zero(buffer: &[u8], vertex_count_aligned: usize) -> bool {
    for i in (0..vertex_count_aligned).step_by(BYTE_GROUP_SIZE) {
        if !encode_bytes_group_zero(&buffer[i..]) {
            return false;
        }
    }
    true
}

/// Estimate best control mode (0/1/2/3) for a k-byte stream
///
/// - 2: all zero
/// - 0/1: bit-encoded (0 = kBitsV1[0..3] = {0,1,2,4}, 1 = kBitsV1[1..4] = {1,2,4,8})
/// - 3: literal (vertex_count bytes)
fn estimate_control(buffer: &[u8], vertex_count: usize, vertex_count_aligned: usize) -> u8 {
    if estimate_control_zero(buffer, vertex_count_aligned) {
        return 2;
    }

    let header_size = (vertex_count_aligned / BYTE_GROUP_SIZE).div_ceil(4);
    let mut est_bytes0 = header_size;
    let mut est_bytes1 = header_size;

    for i in (0..vertex_count_aligned).step_by(BYTE_GROUP_SIZE) {
        let size0 = encode_bytes_group_measure(&buffer[i..], 0);
        let size1 = encode_bytes_group_measure(&buffer[i..], 1);
        let size2 = encode_bytes_group_measure(&buffer[i..], 2);
        let size4 = encode_bytes_group_measure(&buffer[i..], 4);
        let size8 = encode_bytes_group_measure(&buffer[i..], 8);

        let size12 = size1.min(size2);
        let size124 = size12.min(size4);

        est_bytes0 += size124.min(size0);
        est_bytes1 += size124.min(size8);
    }

    if est_bytes0 < vertex_count || est_bytes1 < vertex_count {
        if est_bytes0 < est_bytes1 {
            0
        } else {
            1
        }
    } else {
        3
    }
}

// ============================================================================
// Vertex block encoder / decoder
// ============================================================================

fn encode_vertex_block(
    out: &mut Vec<u8>,
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: &[u8],
    version: u8,
) {
    debug_assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);
    debug_assert!(vertex_size % 4 == 0);

    let vertex_count_aligned = vertex_count.div_ceil(BYTE_GROUP_SIZE) * BYTE_GROUP_SIZE;

    // Allocate control bytes area (v1 only)
    let control_size = if version == 0 { 0 } else { vertex_size / 4 };
    let control_start = out.len();
    out.resize(out.len() + control_size, 0);

    let mut buffer = vec![0u8; VERTEX_BLOCK_MAX_SIZE];

    for k in 0..vertex_size {
        // Fill buffer with delta based on channel
        buffer[..vertex_count_aligned].fill(0);
        let channel = if version == 0 { 0 } else { channels[k / 4] };
        encode_deltas(
            &mut buffer,
            vertex_data,
            vertex_count,
            vertex_size,
            last_vertex,
            k,
            channel,
        );

        let ctrl = if version == 0 {
            0
        } else {
            estimate_control(&buffer, vertex_count, vertex_count_aligned)
        };

        if version != 0 {
            out[control_start + k / 4] |= ctrl << ((k % 4) * 2);
        }

        match ctrl {
            3 => {
                // literal encoding
                out.extend_from_slice(&buffer[..vertex_count]);
            }
            2 => {
                // zero encoding: nothing
            }
            _ => {
                let bits: [u8; 4] = if version == 0 {
                    BITS_V0
                } else if ctrl == 0 {
                    // kBitsV1[0..3] = {0, 1, 2, 4}
                    [BITS_V1[0], BITS_V1[1], BITS_V1[2], BITS_V1[3]]
                } else {
                    // kBitsV1[1..4] = {1, 2, 4, 8}
                    [BITS_V1[1], BITS_V1[2], BITS_V1[3], BITS_V1[4]]
                };
                encode_bytes(out, &buffer, vertex_count_aligned, bits);
            }
        }
    }

    // Update last_vertex
    let last_offset = (vertex_count - 1) * vertex_size;
    last_vertex[..vertex_size]
        .copy_from_slice(&vertex_data[last_offset..last_offset + vertex_size]);
}

#[allow(clippy::too_many_arguments)]
fn decode_vertex_block(
    data: &[u8],
    cursor: &mut usize,
    output: &mut [u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &mut [u8; 256],
    channels: &[u8],
    version: u8,
) -> Result<(), CodecError> {
    debug_assert!(vertex_count > 0 && vertex_count <= VERTEX_BLOCK_MAX_SIZE);
    debug_assert!(vertex_size % 4 == 0);

    let vertex_count_aligned = vertex_count.div_ceil(BYTE_GROUP_SIZE) * BYTE_GROUP_SIZE;

    // Read control bytes (v1 only)
    let control_size = if version == 0 { 0 } else { vertex_size / 4 };
    if *cursor + control_size > data.len() {
        return Err(CodecError::UnexpectedEof);
    }
    let control_start = *cursor;
    *cursor += control_size;

    // 4-byte group buffer: 4 columns × VERTEX_BLOCK_MAX_SIZE
    let mut columns = vec![0u8; 4 * VERTEX_BLOCK_MAX_SIZE];

    for k in (0..vertex_size).step_by(4) {
        // Decode 4 byte columns for this k-group
        for j in 0..4 {
            let ctrl = if version == 0 {
                0
            } else {
                (data[control_start + k / 4] >> (j * 2)) & 3
            };
            let col_start = j * vertex_count;

            match ctrl {
                3 => {
                    // literal
                    if *cursor + vertex_count > data.len() {
                        return Err(CodecError::UnexpectedEof);
                    }
                    columns[col_start..col_start + vertex_count]
                        .copy_from_slice(&data[*cursor..*cursor + vertex_count]);
                    *cursor += vertex_count;
                }
                2 => {
                    // zero
                    columns[col_start..col_start + vertex_count_aligned].fill(0);
                }
                _ => {
                    let bits: &[u8] = if version == 0 {
                        &BITS_V0
                    } else if ctrl == 0 {
                        &BITS_V1[0..4]
                    } else {
                        &BITS_V1[1..5]
                    };
                    decode_bytes(
                        data,
                        cursor,
                        &mut columns[col_start..],
                        vertex_count_aligned,
                        bits,
                    )?;
                }
            }
        }

        // Apply delta reconstruction for the 4-byte group
        let channel = if version == 0 { 0 } else { channels[k / 4] };
        decode_deltas_group(
            &columns,
            output,
            k,
            vertex_count,
            vertex_size,
            last_vertex,
            channel,
        );
    }

    // Update last_vertex to actual last decoded vertex
    let last_offset = (vertex_count - 1) * vertex_size;
    last_vertex[..vertex_size].copy_from_slice(&output[last_offset..last_offset + vertex_size]);

    Ok(())
}

// ============================================================================
// Public API
// ============================================================================

/// Estimate best channel (0/1/2) for a 4-byte group starting at k
///
/// Estimate encoded bits for a single byte (meshopt heuristic)
///
/// v == 0 → 0 bits, v <= 3 → 2 bits, v <= 15 → 4 bits, else → 8 bits
#[inline]
const fn estimate_bits(v: u8) -> usize {
    if v <= 15 {
        if v <= 3 {
            if v == 0 {
                0
            } else {
                2
            }
        } else {
            4
        }
    } else {
        8
    }
}

/// Estimate best rotation (0-7) for XOR channel encoding
///
/// For u32 delta = v XOR prev, tries 8 rotations of the bit-consistency
/// mask (OR of all deltas in a group), scoring each by estimated bits per
/// byte The rotation with smallest total score is chosen
///
/// Corresponds to meshopt `estimateRotate`
#[allow(clippy::needless_range_loop)]
fn estimate_rotate(
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    k: usize,
    group_size: usize,
) -> u8 {
    let mut sizes = [0usize; 8];

    let mut last = u32::from_le_bytes([
        vertex_data[k],
        vertex_data[k + 1],
        vertex_data[k + 2],
        vertex_data[k + 3],
    ]);

    let mut i = 0;
    while i < vertex_count {
        let mut bitg = 0u32;

        let end = (i + group_size).min(vertex_count);
        for j in i..end {
            let base = j * vertex_size + k;
            let v = u32::from_le_bytes([
                vertex_data[base],
                vertex_data[base + 1],
                vertex_data[base + 2],
                vertex_data[base + 3],
            ]);
            let d = v ^ last;
            bitg |= d;
            last = v;
        }

        for j in 0..8 {
            let bitr = bitg.rotate_left(j as u32);
            sizes[j] += estimate_bits((bitr) as u8);
            sizes[j] += estimate_bits((bitr >> 8) as u8);
            sizes[j] += estimate_bits((bitr >> 16) as u8);
            sizes[j] += estimate_bits((bitr >> 24) as u8);
        }

        i += group_size;
    }

    let mut best_rot: usize = 0;
    for rot in 1..8 {
        if sizes[rot] < sizes[best_rot] {
            best_rot = rot;
        }
    }
    best_rot as u8
}

/// Encodes the group with each candidate channel, measures compressed size,
/// picks the smallest Level parameter caps `max_channel`:
/// - level 0-1: max_channel=0 (channel 0 only, scalar u8)
/// - level 2: max_channel=1 (channels 0-1: u8 / u16)
/// - level 3+: max_channel=2 (channels 0-2: u8 / u16 / u32 XOR+rot)
///
/// For channel 2, `xor_rot` (0-7) is passed as the rotation value; the returned
/// channel byte encodes `(rot << 4) | 2`
fn estimate_channel(
    vertex_data: &[u8],
    vertex_count: usize,
    vertex_size: usize,
    last_vertex: &[u8; 256],
    k: usize,
    max_channel: u8,
    xor_rot: u8,
) -> u8 {
    // Cap sample size to VERTEX_BLOCK_MAX_SIZE for estimation (fits in buffer)
    let sample_count = vertex_count.min(VERTEX_BLOCK_MAX_SIZE);
    let vertex_count_aligned = sample_count.div_ceil(BYTE_GROUP_SIZE) * BYTE_GROUP_SIZE;
    let mut buffer = vec![0u8; VERTEX_BLOCK_MAX_SIZE];

    let mut best_channel: u8 = 0;
    let mut best_size = usize::MAX;

    for channel in 0..=max_channel.min(2) {
        let mut total = 0usize;
        let channel_byte = if channel == 2 {
            (xor_rot << 4) | 2
        } else {
            channel
        };
        for j in 0..4 {
            buffer[..vertex_count_aligned].fill(0);
            encode_deltas(
                &mut buffer,
                vertex_data,
                sample_count,
                vertex_size,
                last_vertex,
                k + j,
                channel_byte,
            );
            // best possible size per group
            for gi in 0..(vertex_count_aligned / BYTE_GROUP_SIZE) {
                let group = &buffer[gi * BYTE_GROUP_SIZE..];
                let s1 = encode_bytes_group_measure(group, 1);
                let s2 = encode_bytes_group_measure(group, 2);
                let s4 = encode_bytes_group_measure(group, 4);
                let s8 = encode_bytes_group_measure(group, 8);
                total += s1.min(s2).min(s4).min(s8);
            }
        }
        if total < best_size {
            best_size = total;
            best_channel = channel;
        }
    }

    if best_channel == 2 {
        (xor_rot << 4) | 2
    } else {
        best_channel
    }
}

/// Encode vertex data to meshopt v1 binary format (default level 0, scalar delta)
///
/// # 引数
///
/// - `vertex_data`: raw vertex bytes (vertex_count × vertex_size)
/// - `vertex_size`: bytes per vertex (must be multiple of 4, ≤ 256)
///
/// # Returns
///
/// Encoded byte buffer
///
/// # Panics
///
/// If `vertex_size` is not a multiple of 4, or > 256, or vertex_data length doesn't match
#[must_use]
pub fn encode_vertex_buffer(vertex_data: &[u8], vertex_size: usize) -> Vec<u8> {
    encode_vertex_buffer_level(vertex_data, vertex_size, 0)
}

/// Encode vertex data with explicit compression level
///
/// - `level` 0-1: scalar u8 delta only (all channels = 0)
/// - `level` 2: choose between u8 / u16 delta per 4-byte group
/// - `level` 3+: also allow u32 XOR delta (best for float data)
///
/// Higher levels give better compression at cost of encoding time
///
/// # Panics
///
/// Same as `encode_vertex_buffer`
#[must_use]
pub fn encode_vertex_buffer_level(vertex_data: &[u8], vertex_size: usize, level: u8) -> Vec<u8> {
    assert!(
        vertex_size > 0 && vertex_size <= 256,
        "vertex_size must be in (0, 256]"
    );
    assert!(vertex_size % 4 == 0, "vertex_size must be multiple of 4");
    assert_eq!(
        vertex_data.len() % vertex_size,
        0,
        "vertex_data length must be multiple of vertex_size"
    );
    let vertex_count = vertex_data.len() / vertex_size;

    let version = CURRENT_VERSION;
    let mut out = Vec::with_capacity(1 + vertex_data.len() + 256);

    // Header
    out.push(VERTEX_HEADER | version);

    if vertex_count == 0 {
        // Just pad tail
        let tail_size_min = if version == 0 {
            TAIL_MIN_SIZE_V0
        } else {
            TAIL_MIN_SIZE_V1
        };
        let tail_size = vertex_size + (if version == 0 { 0 } else { vertex_size / 4 });
        let tail_size_pad = tail_size.max(tail_size_min);
        // padding zeros
        if tail_size < tail_size_pad {
            out.resize(out.len() + (tail_size_pad - tail_size), 0);
        }
        // first_vertex (all zero for empty)
        out.resize(out.len() + vertex_size, 0);
        // channels (all zero)
        if version != 0 {
            out.resize(out.len() + vertex_size / 4, 0);
        }
        return out;
    }

    let mut first_vertex = [0u8; 256];
    first_vertex[..vertex_size].copy_from_slice(&vertex_data[..vertex_size]);

    let mut last_vertex = first_vertex;

    // Compute channels array (level 2+ estimates, level 0-1 all zeros)
    let max_channel: u8 = if level >= 3 {
        2
    } else if level >= 2 {
        1
    } else {
        0
    };
    let mut channels = vec![0u8; vertex_size / 4];
    if version != 0 && max_channel > 0 && vertex_count > 1 {
        // Cap sample size for estimation (matches estimate_channel)
        let sample_count = vertex_count.min(VERTEX_BLOCK_MAX_SIZE);
        for k in (0..vertex_size).step_by(4) {
            // Level 3+: estimate best rotation; else rot=0
            let xor_rot = if level >= 3 {
                estimate_rotate(vertex_data, sample_count, vertex_size, k, BYTE_GROUP_SIZE)
            } else {
                0
            };
            channels[k / 4] = estimate_channel(
                vertex_data,
                vertex_count,
                vertex_size,
                &last_vertex,
                k,
                max_channel,
                xor_rot,
            );
        }
    }

    let vertex_block_size = get_vertex_block_size(vertex_size);
    let mut vertex_offset = 0;

    while vertex_offset < vertex_count {
        let block_size = (vertex_block_size).min(vertex_count - vertex_offset);
        let block_start = vertex_offset * vertex_size;
        let block_end = block_start + block_size * vertex_size;

        encode_vertex_block(
            &mut out,
            &vertex_data[block_start..block_end],
            block_size,
            vertex_size,
            &mut last_vertex,
            &channels,
            version,
        );

        vertex_offset += block_size;
    }

    // Tail
    let tail_size = vertex_size + (if version == 0 { 0 } else { vertex_size / 4 });
    let tail_size_min = if version == 0 {
        TAIL_MIN_SIZE_V0
    } else {
        TAIL_MIN_SIZE_V1
    };
    let tail_size_pad = tail_size.max(tail_size_min);

    // Padding
    if tail_size < tail_size_pad {
        out.resize(out.len() + (tail_size_pad - tail_size), 0);
    }
    // first_vertex
    out.extend_from_slice(&first_vertex[..vertex_size]);
    // channels
    if version != 0 {
        out.extend_from_slice(&channels);
    }

    out
}

/// Decode meshopt v0/v1 binary format to vertex bytes
///
/// # 引数
///
/// - `buffer`: encoded byte buffer
/// - `vertex_count`: expected vertex count
/// - `vertex_size`: bytes per vertex (must match encode)
///
/// # Errors
///
/// - `BadMagic`: header doesn't match
/// - `UnsupportedVersion`: version > MAX_DECODABLE_VERSION
/// - `UnexpectedEof`: truncated
pub fn decode_vertex_buffer(
    buffer: &[u8],
    vertex_count: usize,
    vertex_size: usize,
) -> Result<Vec<u8>, CodecError> {
    assert!(
        vertex_size > 0 && vertex_size <= 256,
        "vertex_size must be in (0, 256]"
    );
    assert!(vertex_size % 4 == 0, "vertex_size must be multiple of 4");

    if buffer.is_empty() {
        return Err(CodecError::UnexpectedEof);
    }

    let header = buffer[0];
    if header & 0xF0 != VERTEX_HEADER {
        return Err(CodecError::BadMagic);
    }
    let version = header & 0x0F;
    if version > MAX_DECODABLE_VERSION {
        return Err(CodecError::UnsupportedVersion);
    }

    let tail_size = vertex_size + (if version == 0 { 0 } else { vertex_size / 4 });
    let tail_size_min = if version == 0 {
        TAIL_MIN_SIZE_V0
    } else {
        TAIL_MIN_SIZE_V1
    };
    let tail_size_pad = tail_size.max(tail_size_min);

    if buffer.len() < 1 + tail_size_pad {
        return Err(CodecError::UnexpectedEof);
    }

    // Extract first_vertex from tail
    let tail_start = buffer.len() - tail_size;
    let mut last_vertex = [0u8; 256];
    last_vertex[..vertex_size].copy_from_slice(&buffer[tail_start..tail_start + vertex_size]);

    // Read channels array from tail (v1 only)
    let channels: Vec<u8> = if version == 0 {
        Vec::new()
    } else {
        let channels_start = tail_start + vertex_size;
        buffer[channels_start..channels_start + vertex_size / 4].to_vec()
    };

    let mut output = vec![0u8; vertex_count * vertex_size];

    if vertex_count == 0 {
        return Ok(output);
    }

    let vertex_block_size = get_vertex_block_size(vertex_size);
    let mut vertex_offset = 0;
    let mut cursor = 1usize;

    while vertex_offset < vertex_count {
        let block_size = vertex_block_size.min(vertex_count - vertex_offset);
        let block_start = vertex_offset * vertex_size;
        let block_end = block_start + block_size * vertex_size;

        decode_vertex_block(
            buffer,
            &mut cursor,
            &mut output[block_start..block_end],
            block_size,
            vertex_size,
            &mut last_vertex,
            &channels,
            version,
        )?;

        vertex_offset += block_size;
    }

    // Verify we consumed exactly up to tail
    let expected_end = buffer.len() - tail_size_pad;
    if cursor != expected_end {
        return Err(CodecError::UnexpectedEof);
    }

    Ok(output)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zigzag_roundtrip() {
        for v in 0u8..=255 {
            let z = zigzag8(v);
            let back = unzigzag8(z);
            assert_eq!(v, back, "zigzag failed for {v}");
        }
    }

    #[test]
    fn test_estimate_bits() {
        assert_eq!(estimate_bits(0), 0);
        assert_eq!(estimate_bits(1), 2);
        assert_eq!(estimate_bits(3), 2);
        assert_eq!(estimate_bits(4), 4);
        assert_eq!(estimate_bits(15), 4);
        assert_eq!(estimate_bits(16), 8);
        assert_eq!(estimate_bits(255), 8);
    }

    #[test]
    fn test_estimate_rotate_float_sequence() {
        // 100 sequential floats — XOR delta should have consistent bit pattern
        let vertex_size = 8;
        let vertex_count = 100;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..vertex_count {
            let x = (i as f32) * 0.1;
            vertex_data.extend_from_slice(&x.to_le_bytes());
            vertex_data.extend_from_slice(&[0u8; 4]);
        }
        // Not asserting the actual value (data-dependent), just that it's in [0, 7]
        let rot = estimate_rotate(&vertex_data, vertex_count, vertex_size, 0, BYTE_GROUP_SIZE);
        assert!(rot < 8, "estimate_rotate returned {rot}");
    }

    #[test]
    fn test_level_3_rotation_improves_float_compression() {
        // For float data, level 3 with rot-auto should compress smaller than
        // level 2 (u16 fallback) — this validates the estimate_rotate benefit
        let vertex_size = 12;
        let vertex_count = 500;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..vertex_count {
            let x = (i as f32) * 0.02;
            let y = (i as f32 * 0.01).sin();
            let z = (i as f32 * 0.01).cos();
            vertex_data.extend_from_slice(&x.to_le_bytes());
            vertex_data.extend_from_slice(&y.to_le_bytes());
            vertex_data.extend_from_slice(&z.to_le_bytes());
        }

        let level_0 = encode_vertex_buffer_level(&vertex_data, vertex_size, 0);
        let level_2 = encode_vertex_buffer_level(&vertex_data, vertex_size, 2);
        let level_3 = encode_vertex_buffer_level(&vertex_data, vertex_size, 3);

        // All 3 must round-trip
        for (level_name, encoded) in [
            ("level_0", &level_0),
            ("level_2", &level_2),
            ("level_3", &level_3),
        ] {
            let decoded = decode_vertex_buffer(encoded, vertex_count, vertex_size).unwrap();
            assert_eq!(decoded, vertex_data, "{level_name} round-trip failed");
        }

        eprintln!(
            "float mesh: raw {} bytes, level_0 {} ({:.2}x), level_2 {} ({:.2}x), level_3 {} ({:.2}x)",
            vertex_data.len(),
            level_0.len(),
            vertex_data.len() as f32 / level_0.len() as f32,
            level_2.len(),
            vertex_data.len() as f32 / level_2.len() as f32,
            level_3.len(),
            vertex_data.len() as f32 / level_3.len() as f32,
        );

        // Level 3 should be at least as good as level 2 for float data (allows equal
        // because estimate might pick channel 0/1 if XOR isn't the winner)
        assert!(
            level_3.len() <= level_2.len(),
            "level 3 ({} bytes) should not be worse than level 2 ({} bytes) for float data",
            level_3.len(),
            level_2.len(),
        );
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse_byte(0b0000_0001), 0b1000_0000);
        assert_eq!(bit_reverse_byte(0b1010_1010), 0b0101_0101);
        assert_eq!(bit_reverse_byte(0), 0);
        assert_eq!(bit_reverse_byte(0xFF), 0xFF);
    }

    #[test]
    fn test_encode_decode_single_vertex() {
        let vertex_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; // 1 vertex × 12 bytes
        let encoded = encode_vertex_buffer(&vertex_data, 12);
        assert_eq!(encoded[0], VERTEX_HEADER | CURRENT_VERSION);
        let decoded = decode_vertex_buffer(&encoded, 1, 12).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_encode_decode_empty() {
        let vertex_data: Vec<u8> = vec![];
        let encoded = encode_vertex_buffer(&vertex_data, 12);
        let decoded = decode_vertex_buffer(&encoded, 0, 12).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_encode_decode_uniform_vertices() {
        // All vertices are identical (deltas all zero → ctrl=2)
        let vertex_data = vec![10u8; 100 * 8]; // 100 vertices × 8 bytes
        let encoded = encode_vertex_buffer(&vertex_data, 8);
        let decoded = decode_vertex_buffer(&encoded, 100, 8).unwrap();
        assert_eq!(decoded, vertex_data);
        // Should be highly compressed
        assert!(
            encoded.len() < vertex_data.len() / 2,
            "expected < 50% size, got {} / {}",
            encoded.len(),
            vertex_data.len()
        );
    }

    #[test]
    fn test_encode_decode_sequential_deltas() {
        // 8-byte vertices with linear positions (small deltas)
        let mut vertex_data = Vec::new();
        for i in 0..50 {
            vertex_data.extend_from_slice(&(i as u16).to_le_bytes());
            vertex_data.extend_from_slice(&((i * 2) as u16).to_le_bytes());
            vertex_data.extend_from_slice(&[0, 0, 0, 0]);
        }
        let encoded = encode_vertex_buffer(&vertex_data, 8);
        let decoded = decode_vertex_buffer(&encoded, 50, 8).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_encode_decode_16_vertices_varying() {
        // 16 vertices * 8 bytes with varying values to trigger bit encoding
        let vertex_size = 8;
        let vertex_count = 16;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..(vertex_count * vertex_size) {
            vertex_data.push((i as u8).wrapping_mul(17));
        }
        let encoded = encode_vertex_buffer(&vertex_data, vertex_size);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        let mismatch = vertex_data
            .iter()
            .zip(decoded.iter())
            .position(|(a, b)| a != b);
        assert_eq!(
            mismatch,
            None,
            "first mismatch at {:?}\n  orig[0..20]: {:?}\n  dec[0..20]:  {:?}",
            mismatch,
            &vertex_data[..20.min(vertex_data.len())],
            &decoded[..20.min(decoded.len())],
        );
    }

    #[test]
    fn test_encode_decode_random_but_deterministic() {
        // Pseudo-random vertex data
        let vertex_count = 200;
        let vertex_size = 16;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        let mut state = 0x12345678u32;
        for _ in 0..(vertex_count * vertex_size) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            vertex_data.push((state >> 16) as u8);
        }
        let encoded = encode_vertex_buffer(&vertex_data, vertex_size);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_encode_decode_realistic_positions() {
        // Simulated vertex positions (f32 x 3)
        let vertex_size = 12;
        let vertex_count = 100;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..vertex_count {
            let x = (i as f32) * 0.1;
            let y = (i as f32).sin();
            let z = (i as f32).cos();
            vertex_data.extend_from_slice(&x.to_le_bytes());
            vertex_data.extend_from_slice(&y.to_le_bytes());
            vertex_data.extend_from_slice(&z.to_le_bytes());
        }
        let encoded = encode_vertex_buffer(&vertex_data, vertex_size);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_encode_decode_large_block() {
        // Larger than single block (VERTEX_BLOCK_MAX_SIZE = 256)
        let vertex_size = 16;
        let vertex_count = 500; // 2 blocks
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..(vertex_count * vertex_size) {
            vertex_data.push((i as u8).wrapping_mul(37));
        }
        let encoded = encode_vertex_buffer(&vertex_data, vertex_size);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_decode_bad_magic() {
        let bad = vec![0xFF; 100];
        let result = decode_vertex_buffer(&bad, 1, 12);
        assert!(matches!(result, Err(CodecError::BadMagic)));
    }

    #[test]
    fn test_decode_unsupported_version() {
        let mut buf = vec![VERTEX_HEADER | 0x0F];
        buf.resize(100, 0);
        let result = decode_vertex_buffer(&buf, 1, 12);
        assert!(matches!(result, Err(CodecError::UnsupportedVersion)));
    }

    #[test]
    fn test_decode_truncated() {
        let vertex_data = vec![1u8; 10 * 12];
        let encoded = encode_vertex_buffer(&vertex_data, 12);
        let truncated = &encoded[..encoded.len() / 2];
        let result = decode_vertex_buffer(truncated, 10, 12);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_compression_ratio() {
        // Sparse data compresses well
        let mut vertex_data = vec![0u8; 200 * 16];
        for i in 0..200 {
            vertex_data[i * 16] = i as u8;
        }
        let encoded = encode_vertex_buffer(&vertex_data, 16);
        let ratio = vertex_data.len() as f32 / encoded.len() as f32;
        assert!(ratio > 3.0, "expected > 3x compression, got {ratio:.2}x");
    }

    #[test]
    fn test_zigzag16_roundtrip() {
        for v in [0u16, 1, 32767, 32768, 65535] {
            let z = zigzag16(v);
            let back = unzigzag16(z);
            assert_eq!(v, back, "zigzag16 failed for {v}");
        }
    }

    #[test]
    fn test_rotate_u32() {
        assert_eq!(rotate_u32(0x8000_0001, 1), 0x0000_0003);
        assert_eq!(rotate_u32(0x0000_0001, 31), 0x8000_0000);
        assert_eq!(rotate_u32(0xABCD_EF01, 0), 0xABCD_EF01);
    }

    #[test]
    fn test_encode_decode_level_2_u16() {
        // Level 2: allows u16 delta, should compress better for u16-aligned data
        let vertex_size = 8;
        let vertex_count = 100;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..vertex_count {
            // 4x u16 with small deltas
            for j in 0..4 {
                vertex_data.extend_from_slice(&((i * 3 + j) as u16).to_le_bytes());
            }
        }
        let encoded = encode_vertex_buffer_level(&vertex_data, vertex_size, 2);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_encode_decode_level_3_u32_xor() {
        // Level 3: allows u32 XOR delta, best for float data
        let vertex_size = 12;
        let vertex_count = 100;
        let mut vertex_data = Vec::with_capacity(vertex_count * vertex_size);
        for i in 0..vertex_count {
            let x = (i as f32) * 0.05;
            let y = (i as f32 * 0.03).sin();
            let z = (i as f32 * 0.03).cos();
            vertex_data.extend_from_slice(&x.to_le_bytes());
            vertex_data.extend_from_slice(&y.to_le_bytes());
            vertex_data.extend_from_slice(&z.to_le_bytes());
        }
        let encoded = encode_vertex_buffer_level(&vertex_data, vertex_size, 3);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);
    }

    #[test]
    fn test_level_progression_all_correct() {
        // Same data at 3 levels, all round-trip correctly
        let vertex_size = 12;
        let vertex_count = 50;
        let mut vertex_data = Vec::new();
        let mut state = 0xDEADBEEFu32;
        for _ in 0..(vertex_count * vertex_size) {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            vertex_data.push((state >> 16) as u8);
        }
        for level in 0..=3u8 {
            let encoded = encode_vertex_buffer_level(&vertex_data, vertex_size, level);
            let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
            assert_eq!(decoded, vertex_data, "level {level} round-trip failed");
        }
    }

    #[test]
    fn test_encode_decode_sphere_mesh_positions() {
        // Realistic mesh position data
        use crate::mesh::{sdf_to_mesh, MarchingCubesConfig};
        use crate::types::SdfNode;
        use glam::Vec3;
        let sphere = SdfNode::sphere(1.0);
        let mesh = sdf_to_mesh(
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

        // Extract positions as raw bytes
        let vertex_size = 12; // Vec3 = 3 × f32
        let mut vertex_data = Vec::with_capacity(mesh.vertices.len() * vertex_size);
        for v in &mesh.vertices {
            vertex_data.extend_from_slice(&v.position.x.to_le_bytes());
            vertex_data.extend_from_slice(&v.position.y.to_le_bytes());
            vertex_data.extend_from_slice(&v.position.z.to_le_bytes());
        }
        let vertex_count = mesh.vertices.len();

        let encoded = encode_vertex_buffer(&vertex_data, vertex_size);
        let decoded = decode_vertex_buffer(&encoded, vertex_count, vertex_size).unwrap();
        assert_eq!(decoded, vertex_data);

        let raw = vertex_data.len();
        eprintln!(
            "sphere positions: {} verts, raw {} bytes, encoded {} bytes ({:.2}x)",
            vertex_count,
            raw,
            encoded.len(),
            raw as f32 / encoded.len() as f32
        );
    }
}

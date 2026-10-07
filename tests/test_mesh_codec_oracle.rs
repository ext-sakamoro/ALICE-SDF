//! Lossless mesh codecs vs their format definition (`mesh::mesh_codec`,
//! `mesh::meshopt_index_codec`, `mesh::meshopt_vertex_codec`).
//!
//! - `mesh_codec` (the crate's own varint delta format): the encoder must
//!   produce exactly the bytes of the format written in the module doc
//!   (8-byte header `ASDF`, version 1, kind, 2 reserved; count as LEB128;
//!   per slot the zigzag LEB128 of the delta to the previous triangle's slot,
//!   or to the previous vertex's byte). The reference encoder below is written
//!   from that description, not from the implementation. Decoding must give
//!   the input back bit for bit, over the whole `u32` index range and over
//!   arbitrary `f32` bit patterns (NaN payloads, infinities, subnormals,
//!   `-0.0`). Each header field that is wrong is reported with its own error.
//! - meshopt codecs (meshoptimizer v1 format): `decode(encode(x))` is `x`
//!   bit for bit for vertex data, and the same triangles in the same order
//!   with the winding kept for index data (the format may rotate a triangle's
//!   first vertex). Decoding the C++ reference vectors is pinned separately in
//!   `tests/meshopt_reference_vectors.rs`.
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::mesh_codec::{
    decode_indices, decode_positions, encode_indices, encode_mesh, encode_positions, CodecError,
};
use alice_sdf::mesh::meshopt_index_codec::{decode_index_buffer, encode_index_buffer};
use alice_sdf::mesh::meshopt_vertex_codec::{decode_vertex_buffer, encode_vertex_buffer};
use alice_sdf::mesh::{Mesh, Vertex};
use glam::Vec3;

struct Rng(u64);
impl Rng {
    const fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
}

fn leb128(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// zigzag of a delta taken modulo 2^32 and read as a signed 32-bit value
const fn zigzag(delta: i64) -> u64 {
    let d = delta as i32 as i64;
    (if d >= 0 { 2 * d } else { -2 * d - 1 }) as u64
}

fn reference_encode_indices(indices: &[u32]) -> Vec<u8> {
    let mut out = b"ASDF".to_vec();
    out.extend([1, 0, 0, 0]);
    let tris = indices.len() / 3;
    leb128(&mut out, tris as u64);
    let mut prev = [0i64; 3];
    for t in 0..tris {
        for k in 0..3 {
            let cur = i64::from(indices[3 * t + k]);
            leb128(&mut out, zigzag(cur - prev[k]));
            prev[k] = cur;
        }
    }
    out
}

fn reference_encode_positions(ps: &[Vec3]) -> Vec<u8> {
    let mut out = b"ASDF".to_vec();
    out.extend([1, 1, 0, 0]);
    leb128(&mut out, ps.len() as u64);
    let mut prev = [0i64; 12];
    for p in ps {
        let mut bytes = Vec::with_capacity(12);
        for c in [p.x, p.y, p.z] {
            bytes.extend(c.to_bits().to_le_bytes());
        }
        for b in 0..12 {
            let cur = i64::from(bytes[b]);
            leb128(&mut out, zigzag(cur - prev[b]));
            prev[b] = cur;
        }
    }
    out
}

fn index_cases() -> Vec<Vec<u32>> {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    let mut cases = vec![
        vec![],
        vec![0, 1, 2],
        vec![0, 1, 2, 2, 1, 3, 4, 6, 5, 7, 8, 9],
        // full-range jumps: deltas that do not fit an i32 without wrapping
        vec![
            0,
            u32::MAX,
            1 << 31,
            (1 << 31) - 1,
            0,
            u32::MAX,
            5,
            1 << 31,
            7,
        ],
    ];
    for len in [3usize, 30, 300, 3000] {
        cases.push((0..len).map(|_| rng.next() % 1000).collect());
        cases.push((0..len).map(|_| rng.next()).collect());
    }
    cases
}

#[test]
fn varint_index_codec_matches_the_documented_format_and_round_trips() {
    let mut n = 0;
    for idx in index_cases() {
        let enc = encode_indices(&idx);
        assert_eq!(
            enc,
            reference_encode_indices(&idx),
            "bytes for {} indices",
            idx.len()
        );
        assert_eq!(decode_indices(&enc).unwrap(), idx);
        n += 1;
    }
    assert_eq!(n, 12);
}

#[test]
fn varint_position_codec_matches_the_documented_format_and_is_bit_exact() {
    let mut rng = Rng(0xDEAD_BEEF_0BAD_F00D);
    let specials = [
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7FC0_1234), // NaN with a payload
        f32::from_bits(0xFF80_0001), // signalling NaN pattern
        f32::from_bits(1),           // smallest subnormal
        f32::MIN_POSITIVE,
        f32::MAX,
        -1.5,
    ];
    let mut cases: Vec<Vec<Vec3>> = vec![
        vec![],
        specials
            .chunks(1)
            .map(|c| Vec3::new(c[0], -c[0], 0.25))
            .collect(),
    ];
    for len in [1usize, 17, 500] {
        cases.push(
            (0..len)
                .map(|_| {
                    Vec3::new(
                        f32::from_bits(rng.next()),
                        f32::from_bits(rng.next()),
                        f32::from_bits(rng.next()),
                    )
                })
                .collect(),
        );
    }
    let mut n = 0;
    for ps in cases {
        let enc = encode_positions(&ps);
        assert_eq!(enc, reference_encode_positions(&ps));
        let dec = decode_positions(&enc).unwrap();
        assert_eq!(dec.len(), ps.len());
        for (a, b) in ps.iter().zip(&dec) {
            for (x, y) in a.to_array().iter().zip(b.to_array()) {
                assert_eq!(x.to_bits(), y.to_bits());
                n += 1;
            }
        }
    }
    assert!(n > 1500, "compared {n}");
}

#[test]
fn encode_mesh_is_the_pair_of_the_two_buffers() {
    let mut mesh = Mesh::new();
    for i in 0..6 {
        let p = Vec3::new(i as f32, (i * i) as f32 * 0.5, -(i as f32));
        mesh.vertices.push(Vertex::new(p, Vec3::Z));
    }
    mesh.indices = vec![0, 1, 2, 2, 1, 3, 3, 4, 5];
    let (ib, pb) = encode_mesh(&mesh);
    assert_eq!(ib, reference_encode_indices(&mesh.indices));
    let ps: Vec<Vec3> = mesh.vertices.iter().map(|v| v.position).collect();
    assert_eq!(pb, reference_encode_positions(&ps));
}

#[test]
fn varint_codec_reports_each_broken_header_field() {
    let good = encode_indices(&[0, 1, 2]);
    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    let mut bad_version = good.clone();
    bad_version[4] = 2;
    let truncated = &good[..good.len() - 1];
    assert_eq!(decode_indices(&bad_magic), Err(CodecError::BadMagic));
    assert_eq!(
        decode_indices(&bad_version),
        Err(CodecError::UnsupportedVersion)
    );
    assert_eq!(decode_indices(truncated), Err(CodecError::UnexpectedEof));
    assert_eq!(decode_indices(&good[..5]), Err(CodecError::UnexpectedEof));
    // a position buffer is not an index buffer, and the other way round
    let pos = encode_positions(&[Vec3::ONE]);
    assert_eq!(decode_indices(&pos), Err(CodecError::WrongKind));
    assert_eq!(decode_positions(&good), Err(CodecError::WrongKind));
    // six continuation bytes run past 32 bits
    let mut overflow = good[..8].to_vec();
    overflow.extend([0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    assert_eq!(decode_indices(&overflow), Err(CodecError::VarintOverflow));
}

fn same_triangles_keep_winding(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len()
        && a.chunks_exact(3).zip(b.chunks_exact(3)).all(|(x, y)| {
            (0..3).any(|r| x[0] == y[r] && x[1] == y[(r + 1) % 3] && x[2] == y[(r + 2) % 3])
        })
}

#[test]
fn meshopt_index_codec_round_trips_triangles_in_order() {
    let mut rng = Rng(0x0F0F_1234_ABCD_0001);
    let mut cases: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![0, 1, 2, 2, 1, 3, 2, 3, 4]];
    // a vertex-cache friendly grid strip and random soups
    let mut grid = Vec::new();
    for y in 0..20u32 {
        for x in 0..20u32 {
            let i = y * 21 + x;
            grid.extend([i, i + 1, i + 21, i + 1, i + 22, i + 21]);
        }
    }
    cases.push(grid);
    for (tris, range) in [(50usize, 30u32), (500, 2000), (1000, 1 << 20)] {
        let mut v = Vec::new();
        for _ in 0..tris {
            // distinct vertices per triangle (the format requires non-degenerate input)
            let a = rng.next() % range;
            let b = (a + 1 + rng.next() % (range - 2)) % range;
            let mut c = rng.next() % range;
            while c == a || c == b {
                c = (c + 1) % range;
            }
            v.extend([a, b, c]);
        }
        cases.push(v);
    }
    let mut n = 0;
    for idx in cases {
        let enc = encode_index_buffer(&idx);
        let dec = decode_index_buffer(&enc, idx.len()).unwrap();
        assert!(
            same_triangles_keep_winding(&idx, &dec),
            "{} indices",
            idx.len()
        );
        n += idx.len() / 3;
    }
    assert!(n > 2000, "compared {n} triangles");
}

#[test]
fn meshopt_vertex_codec_round_trips_bit_exact() {
    let mut rng = Rng(0x5555_AAAA_1357_2468);
    let mut n = 0;
    for &(count, size) in &[
        (1usize, 4usize),
        (3, 12),
        (16, 16),
        (100, 32),
        (257, 8),
        (1000, 64),
    ] {
        for smooth in [false, true] {
            let mut data = vec![0u8; count * size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = if smooth {
                    // slowly varying attribute stream: small deltas
                    ((i / size) as u32 / 3 + (i % size) as u32 * 7) as u8
                } else {
                    rng.next() as u8
                };
            }
            let enc = encode_vertex_buffer(&data, size);
            let dec = decode_vertex_buffer(&enc, count, size).unwrap();
            assert_eq!(dec, data, "{count} x {size} smooth={smooth}");
            n += data.len();
        }
    }
    assert!(n > 100_000, "compared {n} bytes");
}

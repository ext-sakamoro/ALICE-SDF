//! Mesh codecs — lossless index / vertex compression, vertex filters and
//! attribute quantization
//!
//! Meshes the unit sphere, then
//! - compresses it with the crate's varint delta format (`encode_mesh`) and
//!   with the meshoptimizer v1 index / vertex codecs, and decodes both back;
//! - packs normals with the octahedral filter, rotations with the quaternion
//!   filter and floats with the exponential filter, and unpacks them;
//! - quantizes attributes to snorm / unorm / binary16 and decodes them.
//!
//! Every round trip is printed with its size and error and checked with
//! `assert!`. The oracles (format bytes, closed-form error bounds, IEEE 754
//! binary16) are `tests/test_mesh_codec_oracle.rs`,
//! `tests/test_meshopt_filter_oracle.rs` and
//! `tests/test_mesh_quantization_oracle.rs`.
//!
//! # Running
//! ```bash
//! cargo run --example mesh_codecs
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::mesh_codec::{decode_indices, decode_positions, encode_mesh, CodecError};
use alice_sdf::mesh::meshopt_filter::{
    decode_filter_exp_u32_in_place, decode_filter_oct_i16_in_place,
    decode_filter_quat_i16_in_place, encode_filter_exp_u32, encode_filter_oct_i16,
    encode_filter_quat_i16,
};
use alice_sdf::mesh::meshopt_index_codec::{decode_index_buffer, encode_index_buffer};
use alice_sdf::mesh::meshopt_vertex_codec::{decode_vertex_buffer, encode_vertex_buffer};
use alice_sdf::mesh::quantization::{
    half_decode, half_encode, snorm_i16_decode, snorm_i16_encode, snorm_i8_decode, snorm_i8_encode,
    unorm_u16_decode, unorm_u16_encode, unorm_u8_decode, unorm_u8_encode,
};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig};
use alice_sdf::SdfNode;
use glam::Vec3;

fn main() {
    println!("ALICE-SDF — mesh codecs");
    println!("=======================");

    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 24,
            ..Default::default()
        },
    );
    let raw_index_bytes = mesh.indices.len() * 4;
    let raw_pos_bytes = mesh.vertices.len() * 12;
    println!(
        "sphere: {} vertices, {} triangles",
        mesh.vertex_count(),
        mesh.triangle_count()
    );

    // --- crate varint delta format ---
    let (ib, pb) = encode_mesh(&mesh);
    let indices = decode_indices(&ib).expect("decode indices");
    let positions = decode_positions(&pb).expect("decode positions");
    assert_eq!(indices, mesh.indices);
    for (p, v) in positions.iter().zip(&mesh.vertices) {
        assert_eq!(
            p.to_array().map(f32::to_bits),
            v.position.to_array().map(f32::to_bits)
        );
    }
    println!(
        "varint delta: indices {raw_index_bytes} -> {} bytes, positions {raw_pos_bytes} -> {} bytes (bit exact)",
        ib.len(),
        pb.len()
    );
    assert_eq!(decode_positions(&ib), Err(CodecError::WrongKind));
    println!(
        "  an index buffer read as positions: {}",
        CodecError::WrongKind
    );

    // --- meshoptimizer v1 codecs ---
    let enc_i = encode_index_buffer(&mesh.indices);
    let dec_i = decode_index_buffer(&enc_i, mesh.indices.len()).expect("meshopt index decode");
    for (a, b) in mesh.indices.chunks_exact(3).zip(dec_i.chunks_exact(3)) {
        let same = (0..3).any(|r| a[0] == b[r] && a[1] == b[(r + 1) % 3] && a[2] == b[(r + 2) % 3]);
        assert!(same, "triangle {a:?} came back as {b:?}");
    }
    let vbytes: Vec<u8> = mesh
        .vertices
        .iter()
        .flat_map(|v| {
            v.position
                .to_array()
                .into_iter()
                .chain(v.normal.to_array())
                .flat_map(f32::to_le_bytes)
        })
        .collect();
    let enc_v = encode_vertex_buffer(&vbytes, 24);
    let dec_v =
        decode_vertex_buffer(&enc_v, mesh.vertices.len(), 24).expect("meshopt vertex decode");
    assert_eq!(dec_v, vbytes);
    println!(
        "meshopt v1: indices {raw_index_bytes} -> {} bytes, position+normal {} -> {} bytes",
        enc_i.len(),
        vbytes.len(),
        enc_v.len()
    );

    // --- vertex filters ---
    let normals: Vec<[f32; 4]> = mesh
        .vertices
        .iter()
        .map(|v| [v.normal.x, v.normal.y, v.normal.z, 1.0])
        .collect();
    let mut oct = encode_filter_oct_i16(&normals, 12);
    decode_filter_oct_i16_in_place(&mut oct);
    let worst_deg = normals
        .iter()
        .zip(oct.chunks_exact(4))
        .map(|(n, d)| {
            let a = Vec3::new(n[0], n[1], n[2]).normalize();
            let b = Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32).normalize();
            a.dot(b).clamp(-1.0, 1.0).acos().to_degrees()
        })
        .fold(0.0f32, f32::max);
    println!("octahedral normals (12 bit): worst angle {worst_deg:.4} deg");
    assert!(worst_deg < 0.15);

    let quats: Vec<[f32; 4]> = (0..64)
        .map(|i| {
            let a = i as f32 * 0.37;
            let axis = Vec3::new(a.sin(), a.cos(), 0.5).normalize();
            let s = (a * 0.5).sin();
            [axis.x * s, axis.y * s, axis.z * s, (a * 0.5).cos()]
        })
        .collect();
    let mut qd = encode_filter_quat_i16(&quats, 16);
    decode_filter_quat_i16_in_place(&mut qd);
    let mut worst_q = 0.0f32;
    for (q, d) in quats.iter().zip(qd.chunks_exact(4)) {
        let dot: f32 = (0..4).map(|k| q[k] * d[k] as f32 / 32767.0).sum();
        worst_q = worst_q.max(1.0 - dot.abs());
    }
    println!("quaternions (16 bit): worst 1 - |dot| {worst_q:.2e}");
    // per component the error is below 4 · 0.5/(√2 · 32767) + 2/32767 ≈ 1e-4 and
    // the decoded vector is not renormalised, so |dot| is off by at most 2e-4
    assert!(worst_q < 2e-4);

    let floats: Vec<f32> = mesh.vertices.iter().map(|v| v.position.x * 100.0).collect();
    let mut ex = encode_filter_exp_u32(&floats, 16);
    decode_filter_exp_u32_in_place(&mut ex);
    let worst_rel = floats
        .iter()
        .zip(&ex)
        .filter(|(f, _)| f.abs() > 1e-3)
        .map(|(f, e)| ((f32::from_bits(*e) - f) / f).abs())
        .fold(0.0f32, f32::max);
    println!("exponential (16 bit mantissa): worst relative error {worst_rel:.2e}");
    assert!(worst_rel <= 2f32.powi(-14));

    // --- quantization ---
    println!("quantization of 0.3:");
    let v = 0.3f32;
    let rows = [
        ("snorm i8", snorm_i8_decode(snorm_i8_encode(v)), 0.5 / 127.0),
        (
            "snorm i16",
            snorm_i16_decode(snorm_i16_encode(v)),
            0.5 / 32767.0,
        ),
        ("unorm u8", unorm_u8_decode(unorm_u8_encode(v)), 0.5 / 255.0),
        (
            "unorm u16",
            unorm_u16_decode(unorm_u16_encode(v)),
            0.5 / 65535.0,
        ),
        ("binary16", half_decode(half_encode(v)), 2f32.powi(-13)),
    ];
    for (name, got, bound) in rows {
        println!(
            "  {name:<10} {got:.7}  (|error| {:.2e} <= {bound:.2e})",
            (got - v).abs()
        );
        assert!((got - v).abs() <= bound + f32::EPSILON);
    }
    let tiny = 1.0e-6f32; // a binary16 subnormal
    let h = half_encode(tiny);
    println!(
        "  binary16 subnormal {tiny:e} -> {h:#06x} -> {:e}",
        half_decode(h)
    );
    assert!((half_decode(h) - tiny).abs() <= 2f32.powi(-25));
}

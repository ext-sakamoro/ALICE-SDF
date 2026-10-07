//! Oracle tests for `asp_bridge` (feature `asp`).
//!
//! * I-packet: the tree that comes back evaluates bit-identically to the tree
//!   that went in, both from the packet value and after the packet has been
//!   serialised with serde and parsed again (the network path);
//! * D-packet: the delta bytes, the reference sequence and the documented
//!   `asdf_len` parameter are carried unchanged;
//! * `estimate_packet_size`: the documented closed form
//!   `(n, 12 n + 84)` where `n` is the ASDF (bincode) length, with `n` read from
//!   the length the I-packet itself records.

#![cfg(feature = "asp")]

use alice_sdf::asp_bridge::{
    create_sdf_d_packet, create_sdf_i_packet, decode_sdf_i_packet, estimate_packet_size,
};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;
use libasp::AspPacket;

fn scene() -> SdfNode {
    SdfNode::sphere(0.7)
        .smooth_union(SdfNode::box3d(0.8, 0.4, 0.6).translate(0.5, 0.2, 0.0), 0.15)
        .subtract(SdfNode::cylinder(0.2, 2.0))
        .twist(0.3)
}

fn probe_points() -> Vec<Vec3> {
    let mut pts = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            for k in 0..5 {
                pts.push(Vec3::new(
                    -1.2 + 0.6 * i as f32,
                    -1.2 + 0.6 * j as f32,
                    -1.2 + 0.6 * k as f32,
                ));
            }
        }
    }
    pts
}

fn assert_same_field(a: &SdfNode, b: &SdfNode) {
    let mut compared = 0;
    for p in probe_points() {
        assert_eq!(eval(a, p).to_bits(), eval(b, p).to_bits(), "at {p:?}");
        compared += 1;
    }
    assert_eq!(compared, 125);
}

#[test]
fn i_packet_round_trip_preserves_the_field_bit_for_bit() {
    let node = scene();
    let tree = SdfTree::new(node.clone());
    let packet = create_sdf_i_packet(&tree, 42).expect("packet");
    assert_eq!(packet.header.sequence, 42);
    let back = decode_sdf_i_packet(&packet).expect("decode");
    assert_same_field(&back.root, &node);
}

#[test]
fn i_packet_survives_a_serde_wire_format() {
    let node = scene();
    let tree = SdfTree::new(node.clone());
    let packet = create_sdf_i_packet(&tree, 9).expect("packet");
    // `AspPacket` is serde-serialisable; JSON stands in for any serde format.
    let bytes = serde_json::to_vec(&packet).expect("serialise");
    let parsed: AspPacket = serde_json::from_slice(&bytes).expect("parse");
    assert_eq!(parsed.header.sequence, 9);
    let back = decode_sdf_i_packet(&parsed).expect("decode after the wire");
    assert_same_field(&back.root, &node);
}

/// libasp's default `to_bytes` (FlatBuffers) does not carry I-packet region
/// descriptors, where the bridge stores the scene. The documented transport
/// is therefore a serde format; this pins the limitation so a libasp
/// release that starts carrying regions shows up here.
#[test]
fn flatbuffers_wire_format_drops_the_scene() {
    let tree = SdfTree::new(scene());
    let packet = create_sdf_i_packet(&tree, 9).expect("packet");
    let parsed = AspPacket::from_bytes(&packet.to_bytes().expect("serialise")).expect("parse");
    assert_eq!(
        decode_sdf_i_packet(&parsed).unwrap_err(),
        "No regions in I-packet"
    );
}

#[test]
fn d_packet_carries_the_delta_bytes_unchanged() {
    // Every byte value, so a byte -> f32 -> byte path that loses any value fails.
    let delta: Vec<u8> = (0..=255u8).chain([0, 255, 128, 1]).collect();
    let packet = create_sdf_d_packet(&delta, 3, 4).expect("packet");
    assert_eq!(packet.header.sequence, 4);
    let d = packet.as_d_packet().expect("a D-packet");
    assert_eq!(d.ref_sequence, 3);
    assert_eq!(d.region_deltas.len(), 1);
    let rd = &d.region_deltas[0];
    assert_eq!(rd.region_index, 0);
    let coeffs = rd.dct_delta.as_ref().expect("delta bytes");
    assert_eq!(coeffs.len(), delta.len());
    let mut compared = 0;
    for (i, &(idx, zero, v)) in coeffs.iter().enumerate() {
        assert_eq!(idx as usize, i);
        assert_eq!(zero, 0);
        assert_eq!(v as u8, delta[i]);
        assert_eq!(v, f32::from(delta[i]));
        compared += 1;
    }
    assert_eq!(compared, 260);
    let params = rd.param_delta.as_ref().expect("params");
    assert_eq!(params, &vec![("asdf_len".to_string(), 260.0)]);
    // A D-packet is not an I-packet.
    assert!(decode_sdf_i_packet(&packet).is_err());
}

#[test]
fn estimate_packet_size_is_the_documented_closed_form() {
    let mut compared = 0;
    for node in [
        SdfNode::sphere(1.0),
        scene(),
        SdfNode::box3d(1.0, 2.0, 3.0).union(SdfNode::torus(1.0, 0.25)),
    ] {
        let tree = SdfTree::new(node);
        let (asdf, total) = estimate_packet_size(&tree);
        // The I-packet records the ASDF length twice: as the `asdf_len`
        // parameter and as one coefficient per byte.
        let packet = create_sdf_i_packet(&tree, 1).expect("packet");
        let region = &packet.as_i_packet().expect("I-packet").regions[0];
        let recorded = region.params.as_ref().unwrap()[0].1 as usize;
        assert_eq!(region.dct_coefficients.as_ref().unwrap().len(), recorded);
        assert_eq!(asdf, recorded);
        assert!(asdf > 0);
        assert_eq!(total, 12 * asdf + 16 + 4 + 64);
        compared += 1;
    }
    assert_eq!(compared, 3);
}

//! SDF scene streaming over ALICE Streaming Protocol packets (feature `asp`).
//!
//! Packs a scene into an I-packet, sends it as serde JSON (libasp's default
//! FlatBuffers `to_bytes` does not carry the region that holds the scene),
//! decodes it on the receiving side and checks that the field is unchanged.
//!
//! ```sh
//! cargo run --example asp_bridge --features asp
//! ```

use alice_sdf::asp_bridge::{
    create_sdf_d_packet, create_sdf_i_packet, decode_sdf_i_packet, estimate_packet_size,
};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;

fn main() {
    let node = SdfNode::sphere(0.7)
        .smooth_union(SdfNode::box3d(0.8, 0.4, 0.6).translate(0.5, 0.2, 0.0), 0.15);
    let tree = SdfTree::new(node.clone());

    let (asdf, total) = estimate_packet_size(&tree);
    println!("scene: {asdf} ASDF bytes, about {total} bytes as a packet");

    let packet = create_sdf_i_packet(&tree, 1).expect("I-packet");
    let wire = serde_json::to_vec(&packet).expect("serialise");
    println!(
        "I-packet #{} -> {} wire bytes",
        packet.header.sequence,
        wire.len()
    );

    let received: libasp::AspPacket = serde_json::from_slice(&wire).expect("parse");
    let scene = decode_sdf_i_packet(&received).expect("decode");
    let p = Vec3::new(0.3, -0.2, 0.4);
    let (a, b) = (eval(&node, p), eval(&scene.root, p));
    println!("distance at {p:?}: sent {a}, received {b}");
    assert_eq!(a.to_bits(), b.to_bits());

    // A follow-up delta referencing the I-packet.
    let delta = serde_json::to_vec(&SdfTree::new(node.translate(0.0, 0.1, 0.0))).expect("delta");
    let d = create_sdf_d_packet(&delta, 1, 2).expect("D-packet");
    let payload = d.as_d_packet().expect("D-packet payload");
    println!(
        "D-packet #{} refers to #{} with {} delta bytes",
        d.header.sequence,
        payload.ref_sequence,
        payload.region_deltas[0]
            .dct_delta
            .as_ref()
            .map_or(0, Vec::len)
    );
    assert_eq!(payload.ref_sequence, 1);
}

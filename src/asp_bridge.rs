//! ALICE-Streaming-Protocol bridge: SDF scene delivery via ASP packets
//!
//! Packages SDF trees into ASP I-packets (full scene) and D-packets
//! (delta updates) for real-time procedural scene streaming.
//!
//! # Transport
//!
//! The scene travels in the I-packet's region descriptor. libasp's default
//! `AspPacket::to_bytes` (FlatBuffers) does not carry region descriptors, so a
//! packet sent that way arrives without the scene. Serialise the packet with
//! serde instead (`AspPacket` implements `Serialize` / `Deserialize`; libasp's
//! `bincode-compat` feature adds `to_bytes_bincode`).
//!
//! # Example
//!
//! ```ignore
//! use alice_sdf::asp_bridge::*;
//! use alice_sdf::prelude::*;
//!
//! let tree = SdfTree::new(SdfNode::sphere(1.0));
//! let packet = create_sdf_i_packet(&tree, 1).unwrap();
//! let bytes = serde_json::to_vec(&packet).unwrap();
//! // Send `bytes` over network
//!
//! // On receiver:
//! let packet: libasp::AspPacket = serde_json::from_slice(&bytes).unwrap();
//! let recovered = decode_sdf_i_packet(&packet).unwrap();
//! ```
//!
//! `examples/asp_bridge.rs` runs the same round trip.
//!
//! # Semantics identifier (ALICE-Law G8)
//!
//! An I-packet carries the identifier of the arithmetic the tree was built under: the parameter
//! `alice_semantics_v` = 2 marks it, the first 32 coefficient entries hold the identifier's
//! bytes and the tree's bytes follow (`asdf_offset` = 32). [`decode_sdf_i_packet`] refuses a
//! packet without one (a sender of 7.0.0 or earlier) and one with another identifier;
//! [`decode_sdf_i_packet_unverified`] also reads a packet without one and returns the tree with
//! what is known about it.

use libasp::{
    AspPacket, AspResult, Color, ColorPalette, DPacketPayload, IPacketPayload, QualityLevel, Rect,
    RegionDescriptor,
};

use crate::provenance::{sdf_load_verdict, Checked, Provenance};
use crate::types::SdfTree;

/// Create an ASP I-packet containing a full SDF scene.
///
/// The SDF tree is serialized to ASDF binary format and embedded
/// in the I-packet's region descriptors as a `Complex` pattern.
/// bincode allocation upper bound (256 MB) fuzz 検出の capacity overflow panic 予防
/// src/io/asdf.rs::ASDF_BINCODE_LIMIT と同期
const ASP_BINCODE_LIMIT: usize = 256 * 1024 * 1024;

/// The parameter marking a packet that carries the semantics identifier, and its value
const SEMANTICS_PARAM: &str = "alice_semantics_v";
const SEMANTICS_LAYOUT: f32 = 2.0;
/// The coefficient index where the tree's bytes start in such a packet
const TREE_OFFSET: usize = 32;

/// Serialise `tree` (ASDF bincode wire format) into an ASP I-packet with the
/// given `sequence` number.
///
/// # Errors
///
/// Returns the `libasp` error when the tree does not fit the packet or the
/// serialisation exceeds the 256 MiB bincode limit.
pub fn create_sdf_i_packet(tree: &SdfTree, sequence: u32) -> AspResult<AspPacket> {
    create_sdf_i_packet_with_provenance(tree, sequence, &Provenance::sdf())
}

/// [`create_sdf_i_packet`] stamped with the identifier the caller gives
///
/// # Errors
///
/// As [`create_sdf_i_packet`].
pub fn create_sdf_i_packet_with_provenance(
    tree: &SdfTree,
    sequence: u32,
    provenance: &Provenance,
) -> AspResult<AspPacket> {
    // Serialize SDF tree to binary via bincode (same format as ASDF files)
    // bincode 2 API: serde compat + legacy config で bincode 1 wire format 互換
    // with_limit で DoS 予防 (fuzz_asdf_decode で発覚した panic の同 root cause)
    let asdf_bytes = bincode::serde::encode_to_vec(
        tree,
        bincode::config::legacy().with_limit::<ASP_BINCODE_LIMIT>(),
    )
    .map_err(|e| libasp::AspError::SerializationError(e.to_string()))?;

    // Build I-packet with ASDF payload embedded as DCT coefficients
    // (reusing the sparse coefficient field for binary data)
    let mut payload = IPacketPayload::new(0, 0, 0.0);
    payload.quality = QualityLevel::High;
    payload.timestamp_ms = 0;

    // Store ASDF as a region descriptor
    let region = RegionDescriptor {
        bounds: Rect::new(0, 0, 0, 0),
        pattern_type: libasp::PatternType::Complex,
        palette: ColorPalette {
            colors: vec![Color::new(
                (asdf_bytes.len() >> 16) as u8,
                (asdf_bytes.len() >> 8) as u8,
                asdf_bytes.len() as u8,
            )],
            weights: None,
        },
        dct_coefficients: Some(
            provenance
                .semantics_id
                .iter()
                .chain(asdf_bytes.iter())
                .enumerate()
                .map(|(i, &b)| (i as u32, 0, b as f32))
                .collect(),
        ),
        texture_id: None,
        params: Some(vec![
            ("asdf_len".to_string(), asdf_bytes.len() as f32),
            ("asdf_offset".to_string(), TREE_OFFSET as f32),
            (SEMANTICS_PARAM.to_string(), SEMANTICS_LAYOUT),
        ]),
    };
    payload.regions.push(region);

    AspPacket::create_i_packet(sequence, payload)
}

/// Create an ASP D-packet with SDF parameter deltas.
///
/// `delta_asdf` should be a serialized ASDF representing the updated tree.
pub fn create_sdf_d_packet(
    delta_asdf: &[u8],
    ref_sequence: u32,
    sequence: u32,
) -> AspResult<AspPacket> {
    let mut payload = DPacketPayload::new(ref_sequence);
    payload.timestamp_ms = 0;

    // Embed delta as a correction-style region delta
    let delta = libasp::RegionDelta {
        region_index: 0,
        palette_delta: None,
        dct_delta: Some(
            delta_asdf
                .iter()
                .enumerate()
                .map(|(i, &b)| (i as u32, 0, b as f32))
                .collect(),
        ),
        param_delta: Some(vec![("asdf_len".to_string(), delta_asdf.len() as f32)]),
    };
    payload.region_deltas.push(delta);

    AspPacket::create_d_packet(sequence, payload)
}

/// Decode an SDF tree from a received ASP I-packet.
///
/// Extracts the ASDF binary from the packet's region descriptor
/// and deserializes it back into an `SdfTree`.
///
/// # Errors
///
/// A packet without a semantics identifier (sent by 7.0.0 or earlier; read it with
/// [`decode_sdf_i_packet_unverified`]), one with another identifier, and the format errors.
pub fn decode_sdf_i_packet(packet: &AspPacket) -> Result<SdfTree, String> {
    decode_checked(packet, false).map(|c| c.into_parts().0)
}

/// [`decode_sdf_i_packet`] that also reads a packet without a semantics identifier, marking it
/// [`crate::provenance::StoredSemantics::Unpinned`]; another identifier is still refused
///
/// # Errors
///
/// A packet with another identifier, and the format errors.
pub fn decode_sdf_i_packet_unverified(packet: &AspPacket) -> Result<Checked<SdfTree>, String> {
    decode_checked(packet, true)
}

fn decode_checked(packet: &AspPacket, allow_unpinned: bool) -> Result<Checked<SdfTree>, String> {
    let i_payload = packet
        .as_i_packet()
        .ok_or_else(|| "Not an I-packet".to_string())?;

    let region = i_payload
        .regions
        .first()
        .ok_or_else(|| "No regions in I-packet".to_string())?;

    // Extract ASDF length from params
    let asdf_len = region
        .params
        .as_ref()
        .and_then(|p| p.iter().find(|(k, _)| k == "asdf_len"))
        .map(|(_, v)| *v as usize)
        .ok_or_else(|| "Missing asdf_len param".to_string())?;

    // Reconstruct ASDF bytes from DCT coefficient field
    let coeffs = region
        .dct_coefficients
        .as_ref()
        .ok_or_else(|| "No DCT data in region".to_string())?;

    let param = |name: &str| {
        region
            .params
            .as_ref()
            .and_then(|p| p.iter().find(|(k, _)| k == name))
            .map(|(_, v)| *v)
    };
    let stamped = match param(SEMANTICS_PARAM) {
        None => false,
        Some(v) if v == SEMANTICS_LAYOUT => true,
        Some(v) => return Err(format!("{SEMANTICS_PARAM} {v} is not a known layout")),
    };
    let offset = if stamped { TREE_OFFSET } else { 0 };

    let mut bytes = vec![0u8; offset + asdf_len];
    for &(idx, _, val) in coeffs {
        if (idx as usize) < bytes.len() {
            bytes[idx as usize] = val as u8;
        }
    }
    let stored = stamped.then(|| {
        let mut id = [0u8; 32];
        id.copy_from_slice(&bytes[..TREE_OFFSET]);
        id
    });
    let semantics =
        sdf_load_verdict(stored, "ASP I-packet", allow_unpinned).map_err(|e| e.to_string())?;

    #[allow(
        clippy::disallowed_methods,
        reason = "stored form checked: the identifier in the first 32 coefficient entries went through sdf_load_verdict above"
    )]
    let decoded = bincode::serde::decode_from_slice(
        &bytes[offset..],
        bincode::config::legacy().with_limit::<ASP_BINCODE_LIMIT>(),
    );
    decoded
        .map(|(tree, _)| Checked::new(tree, semantics))
        .map_err(|e| format!("ASDF decode failed: {}", e))
}

/// Compute the ASP packet overhead for an SDF scene.
///
/// Returns `(asdf_bytes, total_packet_bytes)`.
pub fn estimate_packet_size(tree: &SdfTree) -> (usize, usize) {
    let asdf_bytes = bincode::serde::encode_to_vec(
        tree,
        bincode::config::legacy().with_limit::<ASP_BINCODE_LIMIT>(),
    )
    .unwrap_or_default();
    let asdf_len = asdf_bytes.len();
    // ASP overhead: 16-byte header + FlatBuffers framing + CRC32
    // Each byte stored as (u32, u32, f32) = 12 bytes in DCT field
    // Total overhead is significant but acceptable for small SDF scenes
    let packet_overhead = 16 + 4 + 64; // header + CRC + FlatBuffers framing
                                       // the semantics identifier rides in the first TREE_OFFSET coefficient entries
    (asdf_len, (TREE_OFFSET + asdf_len) * 12 + packet_overhead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::eval;
    use crate::types::SdfNode;
    use glam::Vec3;

    /// I-packet round trip: the tree that comes back evaluates identically.
    #[test]
    fn i_packet_round_trip_preserves_the_tree() {
        let node = SdfNode::sphere(0.7)
            .smooth_union(SdfNode::box3d(0.8, 0.4, 0.6).translate(0.5, 0.2, 0.0), 0.15)
            .twist(0.3);
        let tree = SdfTree::new(node.clone());
        let packet = create_sdf_i_packet(&tree, 7).expect("packet");
        let back = decode_sdf_i_packet(&packet).expect("decode");
        for p in [
            Vec3::ZERO,
            Vec3::new(0.4, -0.3, 0.9),
            Vec3::new(-1.2, 0.8, 0.1),
        ] {
            assert_eq!(eval(&back.root, p).to_bits(), eval(&node, p).to_bits());
        }
        let (asdf, total) = estimate_packet_size(&tree);
        assert!(asdf > 0 && total >= asdf);
    }
}

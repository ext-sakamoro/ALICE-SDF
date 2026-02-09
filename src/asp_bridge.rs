//! ALICE-Streaming-Protocol bridge: SDF scene delivery via ASP packets
//!
//! Packages SDF trees into ASP I-packets (full scene) and D-packets
//! (delta updates) for real-time procedural scene streaming.
//!
//! # Example
//!
//! ```ignore
//! use alice_sdf::asp_bridge::*;
//! use alice_sdf::prelude::*;
//!
//! let tree = SdfTree::new(SdfNode::sphere(1.0));
//! let packet = create_sdf_i_packet(&tree, 1).unwrap();
//! let bytes = packet.to_bytes().unwrap();
//! // Send `bytes` over network
//!
//! // On receiver:
//! let recovered = decode_sdf_i_packet(&bytes).unwrap();
//! ```

use libasp::{
    AspPacket,
    DPacketPayload, IPacketPayload, Rect,
    AspResult, Color, ColorPalette, QualityLevel, RegionDescriptor,
};

use crate::types::SdfTree;

/// Create an ASP I-packet containing a full SDF scene.
///
/// The SDF tree is serialized to ASDF binary format and embedded
/// in the I-packet's region descriptors as a `Complex` pattern.
pub fn create_sdf_i_packet(tree: &SdfTree, sequence: u32) -> AspResult<AspPacket> {
    // Serialize SDF tree to binary via bincode (same format as ASDF files)
    let asdf_bytes = bincode::serialize(tree)
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
            asdf_bytes
                .iter()
                .enumerate()
                .map(|(i, &b)| (i as u32, 0, b as f32))
                .collect(),
        ),
        texture_id: None,
        params: Some(vec![("asdf_len".to_string(), asdf_bytes.len() as f32)]),
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
pub fn decode_sdf_i_packet(packet: &AspPacket) -> Result<SdfTree, String> {
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

    let mut asdf_bytes = vec![0u8; asdf_len];
    for &(idx, _, val) in coeffs {
        if (idx as usize) < asdf_len {
            asdf_bytes[idx as usize] = val as u8;
        }
    }

    bincode::deserialize(&asdf_bytes).map_err(|e| format!("ASDF decode failed: {}", e))
}

/// Compute the ASP packet overhead for an SDF scene.
///
/// Returns `(asdf_bytes, total_packet_bytes)`.
pub fn estimate_packet_size(tree: &SdfTree) -> (usize, usize) {
    let asdf_bytes = bincode::serialize(tree).unwrap_or_default();
    let asdf_len = asdf_bytes.len();
    // ASP overhead: 16-byte header + FlatBuffers framing + CRC32
    // Each byte stored as (u32, u32, f32) = 12 bytes in DCT field
    // Total overhead is significant but acceptable for small SDF scenes
    let packet_overhead = 16 + 4 + 64; // header + CRC + FlatBuffers framing
    (asdf_len, asdf_len * 12 + packet_overhead)
}

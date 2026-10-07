//! SDF volume compression with `alice-codec` (feature `codec`).
//!
//! Voxelizes a shape, compresses it with the lossless and the lossy presets,
//! decodes it again and prints the size and the reconstruction error.
//!
//! ```sh
//! cargo run --example codec_bridge --features codec
//! ```

use alice_sdf::codec_bridge::{
    compress_sdf, compression_ratio, decode_sdf_volume, decompress_sdf, encode_sdf_volume,
    try_decode_sdf_volume, volume_stats, voxelize_sdf, voxelize_sdf_uniform, EncodeConfig,
};
use alice_sdf::prelude::*;

fn main() {
    let node = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.3, 1.4, 0.3), 0.2);
    let (lo, hi) = (Vec3::splat(-2.0), Vec3::splat(2.0));

    // One call: voxelize + stats + encode.
    let one = compress_sdf(&node, lo, hi, 24, &EncodeConfig::lossless());
    println!(
        "compress_sdf {:?}: {} bytes, ratio {:.2}, {} zero crossings, rms {:.3}",
        one.resolution,
        one.encoded.len(),
        one.ratio,
        one.stats.zero_crossings,
        one.stats.rms
    );
    let back = decompress_sdf(&one.encoded);
    let vol = voxelize_sdf_uniform(&node, lo, hi, 24);
    let scale = EncodeConfig::lossless().fixed_point_scale;
    let exact = vol.data.iter().zip(&back.data).all(|(&d, &r)| {
        r.to_bits() == (((d * scale).round() as i32) as f32 * (1.0 / scale)).to_bits()
    });
    println!("lossless round trip reproduces the fixed-point samples: {exact}");
    assert!(exact);

    // Step by step, on an anisotropic grid, with each preset.
    let vol = voxelize_sdf(&node, lo, hi, [32, 24, 16]);
    let stats = volume_stats(&vol);
    println!(
        "volume {}x{}x{} ({} voxels, empty: {}), distance range [{:.3}, {:.3}], centre {:.3}, corner at {:?}",
        vol.width,
        vol.height,
        vol.depth,
        vol.len(),
        vol.is_empty(),
        stats.min_distance,
        stats.max_distance,
        vol.get(16, 12, 8),
        vol.world_pos(0, 0, 0)
    );
    for (name, config) in [
        ("lossless", EncodeConfig::lossless()),
        ("high_quality", EncodeConfig::high_quality()),
        ("default", EncodeConfig::default()),
        ("fast", EncodeConfig::fast()),
    ] {
        let bytes = encode_sdf_volume(&vol, &config);
        let dec = decode_sdf_volume(&bytes);
        let max_err = vol
            .data
            .iter()
            .zip(&dec.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!(
            "{name:>12}: {:6} bytes, ratio {:5.2}, max error {max_err:.5}",
            bytes.len(),
            compression_ratio(&vol, &bytes)
        );
        assert!(max_err < 0.2);
    }

    // Malformed input is reported, not misread.
    let bytes = encode_sdf_volume(&vol, &EncodeConfig::fast());
    let err = try_decode_sdf_volume(&bytes[..20]).unwrap_err();
    println!("truncated stream: {err}");
}

//! Oracle tests for `codec_bridge` (feature `codec`).
//!
//! Every expectation here is built without calling the code under test:
//!
//! * voxelization: each voxel against a direct `eval` at the grid point given
//!   by the documented `origin + i * (extent - origin) / (n - 1)` spacing, bit
//!   for bit, and against the closed-form sphere distance;
//! * lossless round trip (CDF 5/3, quantizer step 1): the decoded value is the
//!   fixed-point value `round(d * scale) / scale`, bit for bit, on both the raw
//!   and the rANS payload paths;
//! * lossy round trip (CDF 9/7): an error envelope in units of the quantizer
//!   step written in the stream;
//! * statistics: min / max / RMS / zero crossings counted by hand on a volume
//!   whose values are chosen so the answer is known;
//! * bitstream header: the documented flags byte.
//!
//! Every loop counts its comparisons and fails on zero.

#![cfg(feature = "codec")]

use alice_sdf::codec_bridge::{
    compress_sdf, compression_ratio, decode_sdf_volume, decompress_sdf, encode_sdf_volume,
    try_decode_sdf_volume, volume_stats, voxelize_sdf, voxelize_sdf_uniform, DecodeError,
    EncodeConfig, SdfVolume,
};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;

/// Byte offset of the flags byte in the encoded header (documented layout:
/// width, height, depth, origin[3], voxel_size, fixed_point_scale, quality, flags).
const FLAGS_OFFSET: usize = 4 * 3 + 4 * 3 + 4 + 4 + 1;
const FLAG_LOSSLESS_WAVELET: u8 = 0x01;
const FLAG_RANS: u8 = 0x02;
const FLAG_WIDE: u8 = 0x04;

fn grid_point(origin: Vec3, extent: Vec3, dims: [usize; 3], ix: [usize; 3]) -> Vec3 {
    let step = |a: usize| {
        let size = extent[a] - origin[a];
        if dims[a] > 1 {
            size / (dims[a] - 1) as f32
        } else {
            size
        }
    };
    Vec3::new(
        (ix[0] as f32).mul_add(step(0), origin.x),
        (ix[1] as f32).mul_add(step(1), origin.y),
        (ix[2] as f32).mul_add(step(2), origin.z),
    )
}

fn scene() -> SdfNode {
    SdfNode::sphere(0.8).smooth_union(SdfNode::box3d(0.3, 1.1, 0.4).translate(0.4, 0.0, 0.1), 0.2)
}

#[test]
fn voxels_equal_direct_evaluation_bit_for_bit() {
    let node = scene();
    let (origin, extent) = (Vec3::new(-1.5, -1.25, -1.0), Vec3::new(1.5, 1.25, 1.0));
    // Anisotropic dims: the row-major layout must be [z][y][x].
    let dims = [7, 5, 4];
    let vol = voxelize_sdf(&node, origin, extent, dims);
    assert_eq!((vol.width, vol.height, vol.depth), (7, 5, 4));
    assert_eq!(vol.len(), 7 * 5 * 4);
    assert_eq!(vol.data.len(), vol.len());
    assert!(!vol.is_empty());
    assert_eq!(vol.origin, origin);

    let mut compared = 0;
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                let p = grid_point(origin, extent, dims, [x, y, z]);
                let want = eval(&node, p);
                assert_eq!(
                    vol.get(x, y, z).to_bits(),
                    want.to_bits(),
                    "voxel ({x},{y},{z}) at {p:?}"
                );
                assert_eq!(vol.data[z * 5 * 7 + y * 7 + x].to_bits(), want.to_bits());
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 140);
    // The metadata spacing is the smallest axis step: 3.0 / 6 on x
    // (y: 2.5 / 4, z: 2.0 / 3).
    assert_eq!(vol.voxel_size, 0.5);
}

#[test]
fn uniform_voxelization_matches_the_closed_form_sphere() {
    let r = 1.0f32;
    let node = SdfNode::sphere(r);
    let n = 9; // step 0.5 on [-2, 2]: every grid point is exactly representable
    let vol = voxelize_sdf_uniform(&node, Vec3::splat(-2.0), Vec3::splat(2.0), n);
    assert_eq!((vol.width, vol.height, vol.depth), (n, n, n));
    assert_eq!(vol.voxel_size, 0.5);
    let mut compared = 0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = Vec3::new(
                    -2.0 + 0.5 * x as f32,
                    -2.0 + 0.5 * y as f32,
                    -2.0 + 0.5 * z as f32,
                );
                // world_pos is exact for a cubic grid.
                assert_eq!(vol.world_pos(x, y, z), p);
                let exact = (p.length() as f64 - r as f64) as f32;
                assert!(
                    (vol.get(x, y, z) - exact).abs() <= 1e-6,
                    "({x},{y},{z}): {} vs {exact}",
                    vol.get(x, y, z)
                );
                compared += 1;
            }
        }
    }
    assert_eq!(compared, n * n * n);
}

/// The value a lossless codec must reproduce: the fixed-point sample.
fn fixed_point(d: f32, scale: f32) -> f32 {
    ((d * scale).round() as i32) as f32 * (1.0 / scale)
}

fn assert_lossless(vol: &SdfVolume, config: &EncodeConfig, expect_rans: bool) {
    let bytes = encode_sdf_volume(vol, config);
    let flags = bytes[FLAGS_OFFSET];
    assert_eq!(flags & FLAG_LOSSLESS_WAVELET, FLAG_LOSSLESS_WAVELET);
    assert_eq!(flags & FLAG_RANS != 0, expect_rans, "rANS flag");
    assert_eq!(bytes[FLAGS_OFFSET - 1], config.quality, "quality byte");

    let back = decode_sdf_volume(&bytes);
    assert_eq!(
        (back.width, back.height, back.depth),
        (vol.width, vol.height, vol.depth)
    );
    assert_eq!(back.origin, vol.origin);
    assert_eq!(back.voxel_size.to_bits(), vol.voxel_size.to_bits());
    let mut compared = 0;
    for (i, (&d, &r)) in vol.data.iter().zip(&back.data).enumerate() {
        let want = fixed_point(d, config.fixed_point_scale);
        assert_eq!(
            r.to_bits(),
            want.to_bits(),
            "voxel {i}: d={d} got {r} want {want}"
        );
        compared += 1;
    }
    assert_eq!(compared, vol.len());
    assert!(compared > 0);
}

#[test]
fn lossless_presets_reproduce_the_fixed_point_samples_exactly() {
    let node = scene();
    // 4^3 voxels = 128 payload bytes: below the rANS threshold, raw path.
    let small = voxelize_sdf_uniform(&node, Vec3::splat(-1.5), Vec3::splat(1.5), 4);
    // 16^3 voxels = 8 KiB payload: the rANS path.
    let large = voxelize_sdf_uniform(&node, Vec3::splat(-1.5), Vec3::splat(1.5), 16);
    for config in [EncodeConfig::lossless(), EncodeConfig::high_quality()] {
        assert!(config.lossless_wavelet);
        assert!(config.quality >= 95);
        assert_lossless(&small, &config, false);
        assert_lossless(&large, &config, true);
    }
}

#[test]
fn lossy_round_trip_stays_within_the_quantizer_bound_and_keeps_the_sign() {
    let node = scene();
    let res = 16;
    let vol = voxelize_sdf_uniform(&node, Vec3::splat(-1.5), Vec3::splat(1.5), res);
    for config in [EncodeConfig::default(), EncodeConfig::fast()] {
        assert!(!config.lossless_wavelet);
        let bytes = encode_sdf_volume(&vol, &config);
        assert_eq!(bytes[FLAGS_OFFSET] & FLAG_LOSSLESS_WAVELET, 0);
        // Quantizer step as written after the 38-byte header.
        let step = i32::from_le_bytes(bytes[38..42].try_into().unwrap());
        assert!(step >= 1);
        let back = decompress_sdf(&bytes);
        assert_eq!(back.data.len(), vol.data.len());
        // Envelope: 4 quantizer steps in fixed-point units plus the half-unit
        // rounding of the fixed-point conversion. A quantizer of step s with
        // dead zone s/2 moves each coefficient by at most s; the 4x factor is
        // an allowance for the CDF 9/7 synthesis gain, not a proven bound
        // (measured on this grid: 1.4 steps at quality 75, 1.1 at quality 50).
        let bound = (4.0 * step as f32 + 0.5) / config.fixed_point_scale;
        let mut compared = 0;
        let mut max_err = 0.0f32;
        for (&d, &r) in vol.data.iter().zip(&back.data) {
            let err = (d - r).abs();
            max_err = max_err.max(err);
            if d.abs() > bound {
                assert_eq!(d.signum(), r.signum(), "sign flip at d={d} r={r}");
            }
            compared += 1;
        }
        assert_eq!(compared, res * res * res);
        assert!(
            max_err <= bound,
            "quality {}: max error {max_err} > bound {bound} (step {step})",
            config.quality
        );
    }
}

#[test]
fn compress_sdf_is_voxelize_then_encode() {
    let node = scene();
    let (lo, hi) = (Vec3::splat(-1.5), Vec3::splat(1.5));
    let config = EncodeConfig::lossless();
    let res = compress_sdf(&node, lo, hi, 8, &config);
    let vol = voxelize_sdf_uniform(&node, lo, hi, 8);
    assert_eq!(res.encoded, encode_sdf_volume(&vol, &config));
    assert_eq!(res.resolution, [8, 8, 8]);
    assert_eq!(res.stats.total_voxels, 512);
    // ratio = (voxels * 4 bytes) / encoded bytes
    assert_eq!(res.ratio, 2048.0 / res.encoded.len() as f64);
    assert_eq!(compression_ratio(&vol, &res.encoded), res.ratio);
    let back = decompress_sdf(&res.encoded);
    let mut compared = 0;
    for (&d, &r) in vol.data.iter().zip(&back.data) {
        assert_eq!(
            r.to_bits(),
            fixed_point(d, config.fixed_point_scale).to_bits()
        );
        compared += 1;
    }
    assert_eq!(compared, 512);
}

#[test]
fn volume_stats_match_hand_counted_values() {
    // 3 x 2 x 2 volume; x rows (each row is 3 values along x):
    //   z0 y0: [-1,  2,  3]  -> 1 crossing
    //   z0 y1: [ 4, -5,  6]  -> 2 crossings
    //   z1 y0: [ 0,  1, -1]  -> 1 crossing (0 * 1 is not < 0)
    //   z1 y1: [ 2,  2,  2]  -> 0
    let data = vec![
        -1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 0.0, 1.0, -1.0, 2.0, 2.0, 2.0,
    ];
    let sum_sq: f64 = data.iter().map(|&v: &f32| (v as f64) * (v as f64)).sum();
    let vol = SdfVolume {
        data,
        width: 3,
        height: 2,
        depth: 2,
        origin: Vec3::ZERO,
        voxel_size: 1.0,
    };
    let s = volume_stats(&vol);
    assert_eq!(s.min_distance, -5.0);
    assert_eq!(s.max_distance, 6.0);
    assert_eq!(s.zero_crossings, 4);
    assert_eq!(s.total_voxels, 12);
    assert_eq!(s.surface_ratio, 4.0 / 12.0);
    assert!((s.rms - (sum_sq / 12.0).sqrt()).abs() < 1e-12);
    // get() follows data[z * H * W + y * W + x].
    assert_eq!(vol.get(1, 1, 0), -5.0);
    assert_eq!(vol.get(2, 0, 1), -1.0);
    assert_eq!(vol.world_pos(2, 1, 1), Vec3::new(2.0, 1.0, 1.0));
}

#[test]
fn encode_config_presets_are_the_documented_values() {
    let d = EncodeConfig::default();
    assert_eq!(
        (d.quality, d.fixed_point_scale, d.lossless_wavelet),
        (75, 1024.0, false)
    );
    let f = EncodeConfig::fast();
    assert_eq!(
        (f.quality, f.fixed_point_scale, f.lossless_wavelet),
        (50, 512.0, false)
    );
    let h = EncodeConfig::high_quality();
    assert_eq!(
        (h.quality, h.fixed_point_scale, h.lossless_wavelet),
        (95, 4096.0, true)
    );
    let l = EncodeConfig::lossless();
    assert_eq!(
        (l.quality, l.fixed_point_scale, l.lossless_wavelet),
        (100, 4096.0, true)
    );
}

/// FNV-1a 64 of a byte string (pins encoder output across versions).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// A sphere of radius 1 sampled on an n^3 grid over [-1.5, 1.5]^3, computed
/// here (no evaluator) so the bytes do not depend on the evaluator.
fn closed_form_volume(n: usize) -> SdfVolume {
    let step = 3.0f32 / (n - 1) as f32;
    let mut data = Vec::with_capacity(n * n * n);
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let p = [x, y, z].map(|i| (i as f32).mul_add(step, -1.5));
                data.push((p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt() - 1.0);
            }
        }
    }
    SdfVolume {
        data,
        width: n,
        height: n,
        depth: n,
        origin: Vec3::splat(-1.5),
        voxel_size: step,
    }
}

/// Encoder output for in-range volumes, recorded with the encoder as it was
/// before i32 coefficients existed (i16 only). The format extension must not
/// change a single byte of these streams: (n, preset, length, FNV-1a 64, flags).
const PRE_EXTENSION_STREAMS: [(usize, &str, usize, u64, u8); 8] = [
    (4, "lossless", 178, 0xf0ae_c212_e217_ca1f, 1),
    (4, "high_quality", 178, 0xacf3_5f35_de01_7f70, 1),
    (4, "default", 178, 0xd357_441f_6349_a1d9, 0),
    (4, "fast", 178, 0x9357_1639_ae35_fd19, 0),
    (16, "lossless", 5580, 0xeba0_8404_0ec5_124b, 3),
    (16, "high_quality", 5580, 0x637d_ea0b_d97f_b33c, 3),
    (16, "default", 8242, 0xa31e_b7d0_5978_0133, 0),
    (16, "fast", 8242, 0x0942_a962_097a_affb, 0),
];

fn preset(name: &str) -> EncodeConfig {
    match name {
        "lossless" => EncodeConfig::lossless(),
        "high_quality" => EncodeConfig::high_quality(),
        "default" => EncodeConfig::default(),
        "fast" => EncodeConfig::fast(),
        _ => unreachable!("{name}"),
    }
}

#[test]
fn in_range_streams_are_byte_identical_to_the_i16_format() {
    let mut compared = 0;
    for (n, name, len, hash, flags) in PRE_EXTENSION_STREAMS {
        let bytes = encode_sdf_volume(&closed_form_volume(n), &preset(name));
        assert_eq!(bytes[FLAGS_OFFSET], flags, "{n} {name}: flags");
        assert_eq!(bytes.len(), len, "{n} {name}: length");
        assert_eq!(fnv1a(&bytes), hash, "{n} {name}: bytes changed");
        compared += 1;
    }
    assert_eq!(compared, 8);
}

/// Deterministic values spread over +-`amp` (64-bit LCG, top 24 bits).
fn noise_volume(n: usize, amp: f32, seed: u64) -> SdfVolume {
    let mut state = seed;
    let data = (0..n * n * n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32).mul_add(2.0 * amp, -amp)
        })
        .collect();
    SdfVolume {
        data,
        width: n,
        height: n,
        depth: n,
        origin: Vec3::ZERO,
        voxel_size: 1.0,
    }
}

fn assert_wide_lossless(vol: &SdfVolume, expect_rans: bool) {
    let config = EncodeConfig::lossless();
    // The values themselves exceed the i16 fixed-point range (32767 / 4096 ~ 8).
    assert!(vol
        .data
        .iter()
        .any(|d| (d * config.fixed_point_scale).abs() > 32767.0));
    let bytes = encode_sdf_volume(vol, &config);
    assert_eq!(bytes[FLAGS_OFFSET] & FLAG_WIDE, FLAG_WIDE, "wide flag");
    assert_eq!(
        bytes[FLAGS_OFFSET] & FLAG_RANS != 0,
        expect_rans,
        "rANS flag"
    );
    let back = decode_sdf_volume(&bytes);
    let mut compared = 0;
    for (i, (&d, &r)) in vol.data.iter().zip(&back.data).enumerate() {
        let want = fixed_point(d, config.fixed_point_scale);
        assert_eq!(
            r.to_bits(),
            want.to_bits(),
            "voxel {i}: d={d} got {r} want {want}"
        );
        compared += 1;
    }
    assert_eq!(compared, vol.len());
}

#[test]
fn lossless_round_trip_is_exact_beyond_the_i16_range_raw_path() {
    // Sphere on [-10, 10]^3: corner distances reach 16.3 (66 900 fixed-point
    // units). 4^3 = 64 voxels -> 256 payload bytes: below the rANS threshold.
    let node = SdfNode::sphere(1.0);
    let vol = voxelize_sdf_uniform(&node, Vec3::splat(-10.0), Vec3::splat(10.0), 4);
    assert_wide_lossless(&vol, false);
}

#[test]
fn lossless_round_trip_is_exact_beyond_the_i16_range_rans_path() {
    // 8^3 voxels -> 2 KiB of i32 coefficients: the rANS path.
    let node = SdfNode::sphere(1.0);
    let vol = voxelize_sdf_uniform(&node, Vec3::splat(-10.0), Vec3::splat(10.0), 8);
    assert_wide_lossless(&vol, true);
    // +-100 noise: coefficients span ~ +-400 000.
    assert_wide_lossless(&noise_volume(12, 100.0, 7), true);
}

#[test]
fn lossy_wavelet_is_not_truncated_beyond_the_i16_range() {
    // Quality >= 95 uses quantizer step 1; with the CDF 9/7 wavelet the
    // coefficients of a [-10, 10]^3 sphere exceed i16.
    let config = EncodeConfig {
        quality: 95,
        fixed_point_scale: 4096.0,
        lossless_wavelet: false,
    };
    let node = SdfNode::sphere(1.0);
    for (vol, expect_rans) in [
        (
            voxelize_sdf_uniform(&node, Vec3::splat(-10.0), Vec3::splat(10.0), 4),
            false,
        ),
        (
            voxelize_sdf_uniform(&node, Vec3::splat(-10.0), Vec3::splat(10.0), 8),
            true,
        ),
        (noise_volume(12, 100.0, 11), true),
    ] {
        let bytes = encode_sdf_volume(&vol, &config);
        assert_eq!(bytes[FLAGS_OFFSET] & FLAG_WIDE, FLAG_WIDE);
        assert_eq!(bytes[FLAGS_OFFSET] & FLAG_RANS != 0, expect_rans);
        let back = decode_sdf_volume(&bytes);
        // Envelope as in the in-range lossy test, step 1: 4.5 fixed-point units.
        let bound = 4.5 / config.fixed_point_scale;
        let mut compared = 0;
        for (&d, &r) in vol.data.iter().zip(&back.data) {
            assert!((d - r).abs() <= bound, "d={d} r={r}");
            compared += 1;
        }
        assert_eq!(compared, vol.len());
    }
}

#[test]
fn decoder_rejects_reserved_flag_bits_and_truncated_streams() {
    let good = encode_sdf_volume(&closed_form_volume(4), &EncodeConfig::lossless());
    assert!(try_decode_sdf_volume(&good).is_ok());
    let mut rejected = 0;
    for bit in 3..8 {
        let mut bad = good.clone();
        bad[FLAGS_OFFSET] |= 1 << bit;
        assert_eq!(
            try_decode_sdf_volume(&bad).unwrap_err(),
            DecodeError::UnknownFlags(1 << bit)
        );
        assert!(std::panic::catch_unwind(|| decode_sdf_volume(&bad)).is_err());
        rejected += 1;
    }
    assert_eq!(rejected, 5);
    // Cut anywhere: header, quantizer params, payload length, payload.
    for cut in [0, 10, 38, 45, 49, good.len() - 1] {
        assert!(
            matches!(
                try_decode_sdf_volume(&good[..cut]),
                Err(DecodeError::Truncated { len, .. }) if len == cut
            ),
            "cut at {cut}"
        );
    }
}

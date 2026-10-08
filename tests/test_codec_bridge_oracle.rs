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
//! * bitstream header: the documented flags byte;
//! * per-axis voxel size (5.0.0): `world_pos` of a grid with three different
//!   dyadic steps against `origin + i * step` per axis, the bit-3 layout, and
//!   streams without bit 3 decoding with one step on every axis.
//!
//! Every loop counts its comparisons and fails on zero.
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "codec")]

use alice_sdf::codec_bridge::{
    compress_sdf, compression_ratio, decode_sdf_volume, decompress_sdf, encode_sdf_volume,
    try_decode_sdf_volume, volume_stats, voxelize_sdf, voxelize_sdf_uniform, DecodeError,
    EncodeConfig, SdfVolume,
};
use alice_sdf::eval::eval;
use alice_sdf::prelude::*;

/// Byte offset of the flags byte in the encoded header (documented layout:
/// width, height, depth, origin[3], voxel size along x, fixed_point_scale,
/// quality, flags).
const FLAGS_OFFSET: usize = 4 * 3 + 4 * 3 + 4 + 4 + 1;
const FLAG_LOSSLESS_WAVELET: u8 = 0x01;
const FLAG_RANS: u8 = 0x02;
const FLAG_WIDE: u8 = 0x04;
const FLAG_ANISOTROPIC_VOXEL: u8 = 0x08;
/// Header length (documented layout, up to and including the symbol count).
const HEADER_LEN: usize = FLAGS_OFFSET + 1 + 4;

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
    // The per-axis spacing: (extent - origin) / (n - 1) on each axis.
    assert_eq!(vol.voxel_size, Vec3::new(3.0 / 6.0, 2.5 / 4.0, 2.0 / 3.0));
}

#[test]
fn uniform_voxelization_matches_the_closed_form_sphere() {
    let r = 1.0f32;
    let node = SdfNode::sphere(r);
    let n = 9; // step 0.5 on [-2, 2]: every grid point is exactly representable
    let vol = voxelize_sdf_uniform(&node, Vec3::splat(-2.0), Vec3::splat(2.0), n);
    assert_eq!((vol.width, vol.height, vol.depth), (n, n, n));
    assert_eq!(vol.voxel_size, Vec3::splat(0.5));
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
    assert_eq!(
        back.voxel_size.to_array().map(f32::to_bits),
        vol.voxel_size.to_array().map(f32::to_bits)
    );
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
        voxel_size: Vec3::splat(1.0),
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
        voxel_size: Vec3::splat(step),
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
        voxel_size: Vec3::splat(1.0),
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
    // bit 3 is the per-axis voxel size (5.0.0); bits 4-7 stay reserved
    for bit in 4..8 {
        let mut bad = good.clone();
        bad[FLAGS_OFFSET] |= 1 << bit;
        assert_eq!(
            try_decode_sdf_volume(&bad).unwrap_err(),
            DecodeError::UnknownFlags(1 << bit)
        );
        assert!(std::panic::catch_unwind(|| decode_sdf_volume(&bad)).is_err());
        rejected += 1;
    }
    assert_eq!(rejected, 4);
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

// ───────────────────── per-axis voxel size (5.0.0) ─────────────────────

/// A grid whose three steps differ and are all dyadic, so every grid point
/// and every `origin + i * step` is exact in f32: x 2/8, y 4/4, z 1/8.
const ANISO_ORIGIN: Vec3 = Vec3::new(-1.0, -2.0, -0.5);
const ANISO_EXTENT: Vec3 = Vec3::new(1.0, 2.0, 0.5);
const ANISO_DIMS: [usize; 3] = [9, 5, 9];
const ANISO_STEP: [f64; 3] = [0.25, 1.0, 0.125];

fn aniso_volume() -> SdfVolume {
    voxelize_sdf(&scene(), ANISO_ORIGIN, ANISO_EXTENT, ANISO_DIMS)
}

/// `origin + i * step` per axis, in f64 (exact here, then exact in f32).
fn aniso_closed_form(ix: [usize; 3]) -> Vec3 {
    let o = ANISO_ORIGIN.to_array();
    let c = |a: usize| (f64::from(o[a]) + ix[a] as f64 * ANISO_STEP[a]) as f32;
    Vec3::new(c(0), c(1), c(2))
}

#[test]
fn non_cubic_grid_world_pos_matches_the_closed_form() {
    let vol = aniso_volume();
    assert_eq!(
        vol.voxel_size,
        Vec3::new(
            ANISO_STEP[0] as f32,
            ANISO_STEP[1] as f32,
            ANISO_STEP[2] as f32
        )
    );
    let node = scene();
    let mut compared = 0;
    for z in 0..ANISO_DIMS[2] {
        for y in 0..ANISO_DIMS[1] {
            for x in 0..ANISO_DIMS[0] {
                let want = aniso_closed_form([x, y, z]);
                let got = vol.world_pos(x, y, z);
                assert_eq!(
                    got.to_array().map(f32::to_bits),
                    want.to_array().map(f32::to_bits),
                    "world_pos({x},{y},{z})"
                );
                // and it is the point the voxel was sampled at
                assert_eq!(vol.get(x, y, z).to_bits(), eval(&node, want).to_bits());
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 9 * 5 * 9);
}

#[test]
fn per_axis_voxel_size_survives_the_round_trip() {
    let vol = aniso_volume();
    let mut compared = 0;
    for config in [EncodeConfig::lossless(), EncodeConfig::default()] {
        let bytes = encode_sdf_volume(&vol, &config);
        assert_eq!(
            bytes[FLAGS_OFFSET] & FLAG_ANISOTROPIC_VOXEL,
            FLAG_ANISOTROPIC_VOXEL
        );
        // documented layout: x in the header, then y and z after it
        let f = |o: usize| f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
        assert_eq!(f(24).to_bits(), 0.25f32.to_bits());
        assert_eq!(f(HEADER_LEN).to_bits(), 1.0f32.to_bits());
        assert_eq!(f(HEADER_LEN + 4).to_bits(), 0.125f32.to_bits());

        let back = try_decode_sdf_volume(&bytes).unwrap();
        assert_eq!(
            back.voxel_size.to_array().map(f32::to_bits),
            vol.voxel_size.to_array().map(f32::to_bits)
        );
        assert_eq!(back.origin, vol.origin);
        let (x, y, z) = (8, 4, 8);
        assert_eq!(back.world_pos(x, y, z), aniso_closed_form([x, y, z]));
        compared += 1;

        // cut inside the per-axis sizes
        for cut in [HEADER_LEN, HEADER_LEN + 4, HEADER_LEN + 12] {
            assert!(
                matches!(
                    try_decode_sdf_volume(&bytes[..cut]),
                    Err(DecodeError::Truncated { len, .. }) if len == cut
                ),
                "cut at {cut}"
            );
        }
    }
    assert_eq!(compared, 2);
}

#[test]
fn streams_without_the_voxel_flag_decode_with_one_step_on_every_axis() {
    // A cubic-voxel stream is the pre-5.0.0 format byte for byte (pinned by
    // `in_range_streams_are_byte_identical_to_the_i16_format`).
    let mut compared = 0;
    for n in [4usize, 16] {
        let vol = closed_form_volume(n);
        let mut bytes = encode_sdf_volume(&vol, &EncodeConfig::lossless());
        assert_eq!(bytes[FLAGS_OFFSET] & FLAG_ANISOTROPIC_VOXEL, 0);
        let step = 3.0f32 / (n - 1) as f32;
        assert_eq!(decode_sdf_volume(&bytes).voxel_size, Vec3::splat(step));
        // an older stream carrying another step: read as that step on x, y, z
        bytes[24..28].copy_from_slice(&0.75f32.to_le_bytes());
        let back = decode_sdf_volume(&bytes);
        assert_eq!(back.voxel_size, Vec3::splat(0.75));
        assert_eq!(
            back.world_pos(1, 2, 3),
            back.origin + Vec3::new(0.75, 1.5, 2.25)
        );
        compared += 1;
    }
    assert_eq!(compared, 2);
}

// ───────────────────── corrupted streams (5.0.0) ─────────────────────
//
// `try_decode_sdf_volume` must return `Err` instead of panicking on any input.
// Until 5.0.0 it trusted the quantizer fields, the dimensions and the rANS
// histogram: a large step overflowed `i32` in alice-codec's dequantizer (a
// panic with overflow checks, a wrapped value without them), and corrupted
// dimensions could ask for an allocation of any size.

const QUANT_OFFSET: usize = HEADER_LEN;

/// A stream written by hand from the documented layout: raw payload of i32
/// coefficients (flag bit 2), no rANS, cubic voxels.
fn hand_stream(
    dims: [u32; 3],
    symbol_count: u32,
    flags: u8,
    step: i32,
    dead_zone: i32,
    coeffs: &[i32],
) -> Vec<u8> {
    let mut b = Vec::new();
    for v in dims {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0.0f32, 0.0, 0.0, 1.0, 1.0] {
        // origin, voxel size, fixed_point_scale = 1
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.push(100); // quality
    b.push(flags | FLAG_WIDE);
    b.extend_from_slice(&symbol_count.to_le_bytes());
    assert_eq!(b.len(), QUANT_OFFSET);
    b.extend_from_slice(&step.to_le_bytes());
    b.extend_from_slice(&dead_zone.to_le_bytes());
    b.extend_from_slice(&((coeffs.len() * 4) as u32).to_le_bytes());
    for c in coeffs {
        b.extend_from_slice(&c.to_le_bytes());
    }
    b
}

/// Dequantized magnitude of the documented law, in i64:
/// `max(dz, 1) + (|q| - 1)·step + step/2`, 0 for q = 0
fn dequant_i64(q: i64, step: i64, dz: i64) -> i64 {
    if q == 0 {
        return 0;
    }
    let mag = dz.max(1) + (q.abs() - 1) * step + step / 2;
    mag * q.signum()
}

/// A 1x1x1 volume runs no wavelet step, so the voxel is the dequantized
/// coefficient itself (scale 1): the decoder accepts exactly the coefficients
/// whose magnitude fits in i32 and returns them unchanged
#[test]
fn dequantization_is_checked_against_the_i32_range() {
    let mut compared = 0;
    for (step, dz) in [(1i32, 0i32), (2, 1), (7, 3), (1_000_003, 500_001)] {
        let (s, d) = (i64::from(step), i64::from(dz));
        // largest |q| whose magnitude still fits: dz' + (q - 1)·s + s/2 <= MAX
        let q_max = (i64::from(i32::MAX) - d.max(1) - s / 2) / s + 1;
        for q in [1, 2, q_max - 1, q_max, q_max + 1, i64::from(i32::MAX)] {
            for sign in [1i64, -1] {
                let qq = q * sign;
                let Ok(q32) = i32::try_from(qq) else { continue };
                let bytes = hand_stream([1, 1, 1], 1, 0, step, dz, &[q32]);
                let got = try_decode_sdf_volume(&bytes);
                let want = dequant_i64(qq, s, d);
                // the magnitude is formed before the sign, so it must fit too
                if want.abs() <= i64::from(i32::MAX) {
                    let v = got.unwrap_or_else(|e| panic!("q={qq} step={step}: {e}"));
                    assert_eq!(v.data[0].to_bits(), (want as i32 as f32).to_bits());
                } else {
                    assert_eq!(got.unwrap_err(), DecodeError::CoefficientOverflow);
                }
                compared += 1;
            }
        }
    }
    // q = i32::MIN: |q| is 2^31, outside i32 for any step
    let bytes = hand_stream([1, 1, 1], 1, 0, 1, 0, &[i32::MIN]);
    assert_eq!(
        try_decode_sdf_volume(&bytes).unwrap_err(),
        DecodeError::CoefficientOverflow
    );
    assert!(compared >= 40, "{compared}");
}

#[test]
fn quantizer_fields_outside_the_encoder_range_are_rejected() {
    let mut rejected = 0;
    for (step, dz) in [(0, 0), (-1, 0), (i32::MIN, 1), (1, -1), (5, i32::MIN)] {
        let bytes = hand_stream([1, 1, 1], 1, 0, step, dz, &[0]);
        assert_eq!(
            try_decode_sdf_volume(&bytes).unwrap_err(),
            DecodeError::InvalidQuantizer {
                step,
                dead_zone: dz
            }
        );
        rejected += 1;
    }
    // the same fields on an encoder stream: the sign bit of the step / dead zone
    let good = encode_sdf_volume(&closed_form_volume(4), &EncodeConfig::default());
    for off in [QUANT_OFFSET + 3, QUANT_OFFSET + 7] {
        let mut bad = good.clone();
        bad[off] ^= 0x80;
        assert!(matches!(
            try_decode_sdf_volume(&bad),
            Err(DecodeError::InvalidQuantizer { .. })
        ));
        rejected += 1;
    }
    assert_eq!(rejected, 7);
}

/// The CDF 5/3 inverse of one pair `[e, o]` (low band, high band), computed
/// from the lifting steps in i64: update `e -= (2o·1024 + 2048) >> 12`, then
/// predict `o -= (2e·(-2048) + 2048) >> 12` (the mirror boundary doubles the
/// one neighbour). `None` when a sum or a result leaves i32.
fn cdf53_pair_inverse(e: i64, o: i64) -> Option<(i64, i64)> {
    let fits = |v: i64| i32::try_from(v).is_ok();
    let s1 = 2 * o;
    let e1 = e - ((s1 * 1024 + 2048) >> 12);
    let s2 = 2 * e1;
    let o1 = o - ((s2 * -2048 + 2048) >> 12);
    [s1, e1, s2, o1]
        .iter()
        .all(|&v| fits(v))
        .then_some((e1, o1))
}

/// A 2x1x1 volume runs exactly one wavelet pair along x: the decoder returns
/// the closed-form inverse when it stays in i32 and `CoefficientOverflow`
/// when it does not
#[test]
fn inverse_wavelet_is_checked_against_the_i32_range() {
    let big = i64::from(i32::MAX);
    let mut ok = 0;
    let mut overflow = 0;
    for (e, o) in [
        (0i64, 0i64),
        (100, -7),
        (-5, 3),
        (big / 2, 0),
        (big / 2 + 1, 0),
        (big, 0),
        (0, big / 2),
        (0, big / 2 + 1),
        (-big, -big),
        (big / 3, -big / 3),
        // the sums fit but the updated low band does not: e - (2o + 2)/4 > MAX
        (big, -big / 2),
        (-big, big / 2),
    ] {
        let coeffs = [e as i32, o as i32];
        // step 1, dead zone 0: q maps to q (dz' = 1, mag = 1 + (|q| - 1))
        let bytes = hand_stream([2, 1, 1], 2, FLAG_LOSSLESS_WAVELET, 1, 0, &coeffs);
        match (cdf53_pair_inverse(e, o), try_decode_sdf_volume(&bytes)) {
            (Some((e1, o1)), Ok(v)) => {
                assert_eq!(
                    v.data[0].to_bits(),
                    (e1 as i32 as f32).to_bits(),
                    "e={e} o={o}"
                );
                assert_eq!(
                    v.data[1].to_bits(),
                    (o1 as i32 as f32).to_bits(),
                    "e={e} o={o}"
                );
                ok += 1;
            }
            (None, Err(err)) => {
                assert_eq!(err, DecodeError::CoefficientOverflow);
                overflow += 1;
            }
            (want, got) => panic!("e={e} o={o}: want {want:?}, got {got:?}"),
        }
    }
    assert!(ok >= 3 && overflow >= 3, "ok={ok} overflow={overflow}");
}

/// The CDF 5/3 inverse of `[e0, e1, o0]` (3 samples: 2 low, 1 high), from the
/// lifting steps in i64: update `e0 -= (2·o0·1024 + 2048) >> 12` (o0 mirrored),
/// then predict `o0 -= ((e0 + e1)·(-2048) + 2048) >> 12`; output order
/// `[e0, o0, e1]`. `None` when a sum or a result leaves i32.
fn cdf53_triple_inverse(e0: i64, e1: i64, o0: i64) -> Option<[i64; 3]> {
    let fits = |v: i64| i32::try_from(v).is_ok();
    let s1 = 2 * o0;
    let e0n = e0 - ((s1 * 1024 + 2048) >> 12);
    let s2 = e0n + e1;
    let o0n = o0 - ((s2 * -2048 + 2048) >> 12);
    [s1, e0n, s2, o0n]
        .iter()
        .all(|&v| fits(v))
        .then_some([e0n, o0n, e1])
}

/// Three samples along x: an overflow of the updated low band that does not
/// feed a later sum (the high band's right neighbour is the untouched `e1`)
/// must still be reported
#[test]
fn inverse_wavelet_reports_an_overflow_that_no_later_sum_sees() {
    let big = i64::from(i32::MAX);
    let (mut ok, mut overflow) = (0, 0);
    for (e0, e1, o0) in [
        (1_744_830_451i64, 0i64, -1_073_741_824i64),
        (big, 0, -big / 2),
        (10, 20, -3),
        (big / 2, big / 4, 1000),
    ] {
        let bytes = hand_stream(
            [3, 1, 1],
            3,
            FLAG_LOSSLESS_WAVELET,
            1,
            0,
            &[e0 as i32, e1 as i32, o0 as i32],
        );
        match (
            cdf53_triple_inverse(e0, e1, o0),
            try_decode_sdf_volume(&bytes),
        ) {
            (Some(want), Ok(v)) => {
                for (k, w) in want.iter().enumerate() {
                    assert_eq!(
                        v.data[k].to_bits(),
                        (*w as i32 as f32).to_bits(),
                        "{e0} {e1} {o0}"
                    );
                }
                ok += 1;
            }
            (None, Err(err)) => {
                assert_eq!(err, DecodeError::CoefficientOverflow);
                overflow += 1;
            }
            (want, got) => panic!("{e0} {e1} {o0}: want {want:?}, got {got:?}"),
        }
    }
    assert!(ok >= 2 && overflow >= 2, "ok={ok} overflow={overflow}");
}

#[test]
fn dimensions_must_match_the_symbol_count() {
    let mut rejected = 0;
    for (dims, count) in [
        ([2u32, 1, 1], 1u32),
        ([1, 1, 1], 2),
        ([1 << 22, 1 << 22, 1 << 22], 0),
        ([u32::MAX, u32::MAX, 2], 2),
        ([0, 5, 5], 1),
    ] {
        let bytes = hand_stream(dims, count, 0, 1, 0, &[0]);
        assert!(
            matches!(
                try_decode_sdf_volume(&bytes),
                Err(DecodeError::InvalidDimensions { .. })
            ),
            "{dims:?} {count}"
        );
        rejected += 1;
    }
    // an empty volume is consistent and decodes to nothing
    let empty = hand_stream([0, 5, 5], 0, 0, 1, 0, &[]);
    assert!(try_decode_sdf_volume(&empty).unwrap().data.is_empty());
    assert_eq!(rejected, 5);
}

#[test]
fn rans_histogram_must_match_the_payload() {
    let good = encode_sdf_volume(&closed_form_volume(8), &EncodeConfig::lossless());
    assert_eq!(good[FLAGS_OFFSET] & FLAG_RANS, FLAG_RANS);
    assert!(try_decode_sdf_volume(&good).is_ok());
    let hist = QUANT_OFFSET + 8;
    let count = |b: &[u8], i: usize| {
        u32::from_le_bytes(b[hist + 4 * i..hist + 4 * i + 4].try_into().unwrap())
    };
    let set = |b: &mut [u8], i: usize, v: u32| {
        b[hist + 4 * i..hist + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
    };
    let mut rejected = 0;
    // sum off by one
    let mut bad = good.clone();
    set(&mut bad, 0, count(&good, 0) + 1);
    assert_eq!(
        try_decode_sdf_volume(&bad).unwrap_err(),
        DecodeError::InvalidHistogram
    );
    rejected += 1;
    // one value holds half of the bytes (the sum stays right)
    let total: u32 = (0..256).map(|i| count(&good, i)).sum();
    let mut bad = good.clone();
    for i in 0..256 {
        set(&mut bad, i, 0);
    }
    set(&mut bad, 0, total / 2);
    set(&mut bad, 1, total - total / 2);
    assert_eq!(
        try_decode_sdf_volume(&bad).unwrap_err(),
        DecodeError::InvalidHistogram
    );
    rejected += 1;
    // at most 16 coefficient bytes per payload byte: claim a k x 1 x 1 volume
    // (2k bytes of i16) with a consistent header and an even histogram over
    // the same payload; 2k = 16 · payload passes the histogram check, 2k =
    // 16 · payload + 2 does not
    let plen_at = hist + 256 * 4;
    let payload = u32::from_le_bytes(good[plen_at..plen_at + 4].try_into().unwrap());
    let claim = |k: u32| {
        let mut bad = good.clone();
        for (a, v) in [k, 1, 1].into_iter().enumerate() {
            bad[4 * a..4 * a + 4].copy_from_slice(&v.to_le_bytes());
        }
        bad[HEADER_LEN - 4..HEADER_LEN].copy_from_slice(&k.to_le_bytes());
        let bytes = 2 * k;
        for i in 0..256u32 {
            set(
                &mut bad,
                i as usize,
                bytes / 256 + u32::from(i < bytes % 256),
            );
        }
        try_decode_sdf_volume(&bad)
    };
    assert_ne!(
        claim(8 * payload).err(),
        Some(DecodeError::InvalidHistogram)
    );
    assert_eq!(
        claim(8 * payload + 1).unwrap_err(),
        DecodeError::InvalidHistogram
    );
    rejected += 1;
    assert_eq!(rejected, 3);
}

/// Every single-bit flip and every truncation of streams from each encoder
/// path (raw i16, raw i32, rANS i16, rANS i32, lossy, per-axis voxel size)
/// decodes to `Ok` or `Err` without a panic; a truncated stream is always
/// `Err`
#[test]
fn bit_flips_and_truncations_never_panic() {
    let streams = [
        (
            encode_sdf_volume(&closed_form_volume(4), &EncodeConfig::lossless()),
            0,
        ),
        (
            encode_sdf_volume(&closed_form_volume(4), &EncodeConfig::default()),
            0,
        ),
        (
            encode_sdf_volume(&closed_form_volume(8), &EncodeConfig::lossless()),
            FLAG_RANS,
        ),
        (
            encode_sdf_volume(&noise_volume(4, 100.0, 3), &EncodeConfig::lossless()),
            FLAG_WIDE,
        ),
        (
            encode_sdf_volume(&noise_volume(8, 100.0, 5), &EncodeConfig::lossless()),
            FLAG_WIDE | FLAG_RANS,
        ),
        (
            encode_sdf_volume(&aniso_volume(), &EncodeConfig::default()),
            FLAG_ANISOTROPIC_VOXEL,
        ),
    ];
    let (mut tried, mut ok, mut err) = (0usize, 0usize, 0usize);
    for (k, (good, flags)) in streams.iter().enumerate() {
        assert_eq!(good[FLAGS_OFFSET] & flags, *flags, "stream {k} path");
        assert!(try_decode_sdf_volume(good).is_ok(), "stream {k}");
        for cut in 0..good.len() {
            let r = std::panic::catch_unwind(|| try_decode_sdf_volume(&good[..cut]));
            match r {
                Ok(Err(_)) => err += 1,
                Ok(Ok(_)) => panic!("stream {k} cut at {cut}: decoded"),
                Err(payload) => panic!("stream {k} cut at {cut}: panicked: {payload:?}"),
            }
            tried += 1;
        }
        for byte in 0..good.len() {
            for bit in 0..8 {
                let mut bad = good.clone();
                bad[byte] ^= 1 << bit;
                match std::panic::catch_unwind(|| try_decode_sdf_volume(&bad)) {
                    Ok(Ok(_)) => ok += 1,
                    Ok(Err(_)) => err += 1,
                    Err(payload) => {
                        panic!("stream {k} byte {byte} bit {bit}: panicked: {payload:?}")
                    }
                }
                tried += 1;
            }
        }
    }
    assert!(ok > 0 && err > 0, "ok={ok} err={err}");
    assert!(tried > 10_000, "{tried}");
}

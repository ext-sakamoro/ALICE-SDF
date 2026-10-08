//! Oracles for the volume API that `examples/volume_bake.rs` wires: the
//! compiled and distance + gradient bakes, the distance + gradient mip chain,
//! `Volume3D` addressing, and the raw / DDS writers.
//!
//! | target | oracle |
//! |---|---|
//! | `bake_volume_compiled` | bit-identical to `bake_volume` (the CPU evaluators are bit-exact) and equal to `|p| − r` at the grid nodes `world_min + i·size/(res−1)` |
//! | `bake_volume_with_normals` | the distance channel is `bake_volume`'s, the normal is the central difference of the analytic sphere `|p ± ε·eᵢ| − r` (computed here in f64) normalised, and `Vec3::Y` where that difference vanishes |
//! | `generate_mip_chain_distgrad` | each mip voxel is a voxel of its footprint (spans `[2i, 2i+2)`, the last absorbing an odd leftover) holding the footprint's minimum distance |
//! | `Volume3D` addressing | `index = x + y·rx + z·rx·ry`, `voxel_size = size/(res−1)`, `mip_count = 1 + mips` |
//! | `export_raw*` | file bytes are the voxels little-endian, the JSON sidecar's offsets are the prefix sums of the level sizes |
//! | `export_dds_3d*` | Microsoft's DDS_HEADER / DDS_HEADER_DXT10 layout: `DDSD_PITCH` (not `DDSD_LINEARSIZE`) for an uncompressed pitch, `DDSD_MIPMAPCOUNT` and `DDSCAPS_COMPLEX | DDSCAPS_MIPMAP` exactly when a mip chain is present, `DDSCAPS2_VOLUME`, `D3D10_RESOURCE_DIMENSION_TEXTURE3D`; R16 texels are IEEE 754 binary16 rounded to nearest, ties to even (checked against the neighbouring codes) |
//! | `gpu_bake_volume_with_normals` | the analytic sphere at the nodes and the tetrahedral difference of the analytic sphere at the **requested** ε |
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "volume")]

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::types::SdfNode;
use alice_sdf::volume::bake::{bake_volume, bake_volume_compiled, bake_volume_with_normals};
use alice_sdf::volume::export::{
    export_dds_3d, export_dds_3d_distgrad, export_raw, export_raw_with_mips, DdsFormat,
};
use alice_sdf::volume::mipchain::{generate_mip_chain, generate_mip_chain_distgrad};
use alice_sdf::volume::{BakeConfig, Volume3D, VoxelDistGrad};
use glam::Vec3;
use std::path::PathBuf;

fn config(res: u32, mips: bool) -> BakeConfig {
    BakeConfig {
        resolution: [res, res, res],
        bounds_min: Vec3::splat(-2.0),
        bounds_max: Vec3::splat(2.0),
        generate_mips: mips,
        ..Default::default()
    }
}

fn node_pos(cfg: &BakeConfig, x: u32, y: u32, z: u32) -> Vec3 {
    let size = cfg.bounds_max - cfg.bounds_min;
    let r = cfg.resolution;
    cfg.bounds_min
        + Vec3::new(
            x as f32 * size.x / (r[0] - 1) as f32,
            y as f32 * size.y / (r[1] - 1) as f32,
            z as f32 * size.z / (r[2] - 1) as f32,
        )
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "alice_sdf_volume_api_{}_{name}",
        std::process::id()
    ))
}

#[test]
fn compiled_bake_is_bit_identical_and_matches_the_analytic_sphere() {
    let node = SdfNode::sphere(1.1);
    let cfg = config(13, false);
    let compiled = bake_volume_compiled(&CompiledSdf::compile(&node), &cfg);
    let interp = bake_volume(&node, &cfg);
    assert_eq!(compiled.resolution, interp.resolution);
    assert_eq!(compiled.data.len(), 13 * 13 * 13);
    for (a, b) in compiled.data.iter().zip(&interp.data) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let mut worst = 0.0f32;
    for z in 0..13 {
        for y in 0..13 {
            for x in 0..13 {
                let p = node_pos(&cfg, x, y, z);
                worst = worst.max((compiled.get(x, y, z) - (p.length() - 1.1)).abs());
                assert_eq!(
                    compiled.get(x, y, z),
                    compiled.data[compiled.index(x, y, z)]
                );
                assert_eq!(compiled.index(x, y, z), (x + 13 * y + 169 * z) as usize);
            }
        }
    }
    assert!(worst < 2e-6, "worst {worst}");

    // `generate_mips` is honoured on the compiled path too
    let with_mips = config(8, true);
    let c = bake_volume_compiled(&CompiledSdf::compile(&node), &with_mips);
    let i = bake_volume(&node, &with_mips);
    assert_eq!(c.mip_count(), 4, "8 → 4 → 2 → 1");
    assert_eq!(c.mips, i.mips);
}

/// Central-difference normal of the analytic sphere, f64.
fn sphere_central_normal(p: Vec3, r: f64, eps: f64) -> Option<Vec3> {
    let p = [p.x as f64, p.y as f64, p.z as f64];
    let f = |q: [f64; 3]| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - r;
    let mut g = [0.0f64; 3];
    for i in 0..3 {
        let (mut a, mut b) = (p, p);
        a[i] += eps;
        b[i] -= eps;
        g[i] = f(a) - f(b);
    }
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    (len > 1e-9).then(|| {
        Vec3::new(
            (g[0] / len) as f32,
            (g[1] / len) as f32,
            (g[2] / len) as f32,
        )
    })
}

#[test]
fn normal_bake_is_the_central_difference_of_the_analytic_field() {
    let node = SdfNode::sphere(1.0);
    let cfg = config(9, false); // node 4 is the origin, where the gradient vanishes
    for eps in [0.01f32, 0.2] {
        let vol = bake_volume_with_normals(&node, &cfg, eps);
        let dist = bake_volume(&node, &cfg);
        let mut compared = 0;
        for z in 0..9 {
            for y in 0..9 {
                for x in 0..9 {
                    let v = vol.get(x, y, z);
                    assert_eq!(v.distance.to_bits(), dist.get(x, y, z).to_bits());
                    let p = vol.voxel_to_world(x, y, z);
                    assert!((p - node_pos(&cfg, x, y, z)).length() < 1e-6);
                    let n = Vec3::new(v.nx, v.ny, v.nz);
                    match sphere_central_normal(p, 1.0, eps as f64) {
                        Some(want) => {
                            assert!(
                                (n - want).length() < 1e-4,
                                "eps {eps} at {p}: {n} vs {want}"
                            );
                            compared += 1;
                        }
                        None => assert_eq!(n, Vec3::Y, "degenerate gradient falls back to +Y"),
                    }
                }
            }
        }
        assert_eq!(compared, 9 * 9 * 9 - 1);
    }
}

/// Mip footprint on one axis, from the mip chain's documentation.
const fn span(i: usize, next: usize, prev: usize) -> std::ops::Range<usize> {
    let start = 2 * i;
    let end = if i + 1 == next { prev } else { 2 * i + 2 };
    start..end
}

#[test]
fn distgrad_mips_hold_a_footprint_voxel_with_the_minimum_distance() {
    for res in [8u32, 7] {
        let base = bake_volume_with_normals(&SdfNode::sphere(1.2), &config(res, true), 0.01);
        let mips = generate_mip_chain_distgrad(&base);
        assert_eq!(base.mips.len(), mips.len());
        assert_eq!(base.mip_count(), mips.len() + 1);
        let dist_only: Volume3D<f32> = Volume3D {
            data: base.data.iter().map(|v| v.distance).collect(),
            resolution: base.resolution,
            world_min: base.world_min,
            world_max: base.world_max,
            mips: Vec::new(),
        };
        let fmips = generate_mip_chain(&dist_only);
        let mut prev: Vec<VoxelDistGrad> = base.data.clone();
        let mut pres = res as usize;
        for (level, mip) in mips.iter().enumerate() {
            let nres = (pres / 2).max(1);
            assert_eq!(mip.len(), nres * nres * nres, "level {}", level + 1);
            for z in 0..nres {
                for y in 0..nres {
                    for x in 0..nres {
                        let got = mip[x + y * nres + z * nres * nres];
                        let mut min = f32::MAX;
                        let mut members = Vec::new();
                        for cz in span(z, nres, pres) {
                            for cy in span(y, nres, pres) {
                                for cx in span(x, nres, pres) {
                                    let c = prev[cx + cy * pres + cz * pres * pres];
                                    min = min.min(c.distance);
                                    members.push(c);
                                }
                            }
                        }
                        assert_eq!(got.distance, min);
                        assert_eq!(fmips[level][x + y * nres + z * nres * nres], min);
                        assert!(
                            members.iter().any(|c| c.distance == min
                                && (c.nx, c.ny, c.nz) == (got.nx, got.ny, got.nz)),
                            "res {res} level {}: gradient not from a minimal child",
                            level + 1
                        );
                    }
                }
            }
            prev = mip.clone();
            pres = nres;
        }
        assert_eq!(pres, 1, "chain ends at 1³");
    }
}

#[test]
fn volume_addressing_follows_its_closed_form() {
    let mut v = Volume3D::<f32>::new(
        [4, 3, 5],
        Vec3::new(-1.0, 0.0, 2.0),
        Vec3::new(2.0, 4.0, 6.0),
    );
    assert_eq!(v.voxel_count(), 60);
    assert_eq!(v.data.len(), 60);
    assert_eq!(v.world_size(), Vec3::new(3.0, 4.0, 4.0));
    assert_eq!(v.voxel_size(), Vec3::new(1.0, 2.0, 1.0));
    assert_eq!(v.mip_count(), 1);
    v.set(3, 2, 4, 9.5);
    assert_eq!(v.data[3 + 2 * 4 + 4 * 12], 9.5);
    assert_eq!(v.get(3, 2, 4), 9.5);
    assert_eq!(v.voxel_to_world(3, 2, 4), Vec3::new(2.0, 4.0, 6.0));
    // a 1-voxel axis has no step: voxel_size divides by max(res − 1, 1)
    let thin = Volume3D::<f32>::new([1, 1, 1], Vec3::ZERO, Vec3::ONE);
    assert_eq!(thin.voxel_size(), Vec3::ONE);
    assert_eq!(thin.voxel_to_world(0, 0, 0), Vec3::ZERO);
}

#[test]
fn raw_export_writes_the_voxels_and_prefix_summed_mip_offsets() {
    let vol = bake_volume(&SdfNode::sphere(1.0), &config(8, true));
    let path = tmp("a.raw");
    let p = path.to_str().unwrap();
    export_raw(&vol, p).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let want: Vec<u8> = vol.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(bytes, want);
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path.with_extension("raw.meta.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(meta["resolution"], serde_json::json!([8, 8, 8]));
    assert_eq!(meta["voxel_count"], 512);
    assert_eq!(meta["byte_size"], 2048);
    assert_eq!(meta["mip_levels"], 4);

    let path2 = tmp("b.raw");
    export_raw_with_mips(&vol, path2.to_str().unwrap()).unwrap();
    let bytes2 = std::fs::read(&path2).unwrap();
    let mut want2 = want;
    let mut offsets = vec![0u64];
    let mut off = 512 * 4u64;
    for m in &vol.mips {
        offsets.push(off);
        off += m.len() as u64 * 4;
        want2.extend(m.iter().flat_map(|v| v.to_le_bytes()));
    }
    assert_eq!(bytes2, want2);
    let meta2: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(path2.with_extension("raw.meta.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(meta2["mip_byte_offsets"], serde_json::json!(offsets));
    assert_eq!(meta2["byte_size"], off);
    assert_eq!(offsets, vec![0, 2048, 2048 + 256, 2048 + 256 + 32]);
    for f in [&path, &path2] {
        std::fs::remove_file(f).ok();
        std::fs::remove_file(f.with_extension("raw.meta.json")).ok();
    }
}

fn le32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

/// Check the 4 + 124 + 20 byte DDS prologue against the Microsoft layout and
/// return the offset of the texel data.
fn check_dds_header(b: &[u8], res: [u32; 3], bpp: u32, mips: u32, dxgi: u32) -> usize {
    assert_eq!(&b[0..4], b"DDS ");
    let h = 4;
    assert_eq!(le32(b, h), 124, "dwSize");
    let flags = le32(b, h + 4);
    let (caps, height, width, pitch, pixfmt, mipcount, linsize, depth) =
        (0x1, 0x2, 0x4, 0x8, 0x1000, 0x2_0000, 0x8_0000, 0x80_0000);
    for (bit, name) in [
        (caps, "CAPS"),
        (height, "HEIGHT"),
        (width, "WIDTH"),
        (pitch, "PITCH"),
        (pixfmt, "PIXELFORMAT"),
        (depth, "DEPTH"),
    ] {
        assert_ne!(flags & bit, 0, "DDSD_{name} must be set");
    }
    assert_eq!(
        flags & linsize,
        0,
        "DDSD_LINEARSIZE is for compressed textures only"
    );
    assert_eq!(
        flags & mipcount != 0,
        mips > 1,
        "DDSD_MIPMAPCOUNT iff mipmapped"
    );
    assert_eq!(le32(b, h + 8), res[1], "dwHeight");
    assert_eq!(le32(b, h + 12), res[0], "dwWidth");
    assert_eq!(
        le32(b, h + 16),
        res[0] * bpp,
        "dwPitchOrLinearSize = row bytes"
    );
    assert_eq!(le32(b, h + 20), res[2], "dwDepth");
    assert_eq!(le32(b, h + 24), mips, "dwMipMapCount");
    let pf = h + 72;
    assert_eq!(le32(b, pf), 32, "ddspf.dwSize");
    assert_eq!(le32(b, pf + 4), 0x4, "DDPF_FOURCC");
    assert_eq!(&b[pf + 8..pf + 12], b"DX10");
    let caps_off = h + 104;
    let dw_caps = le32(b, caps_off);
    assert_ne!(dw_caps & 0x1000, 0, "DDSCAPS_TEXTURE");
    assert_eq!(
        dw_caps & 0x8 != 0,
        mips > 1,
        "DDSCAPS_COMPLEX iff more than one surface"
    );
    assert_eq!(
        dw_caps & 0x40_0000 != 0,
        mips > 1,
        "DDSCAPS_MIPMAP iff mipmapped"
    );
    assert_eq!(le32(b, caps_off + 4), 0x20_0000, "DDSCAPS2_VOLUME");
    let d = 4 + 124;
    assert_eq!(le32(b, d), dxgi, "dxgiFormat");
    assert_eq!(le32(b, d + 4), 4, "D3D10_RESOURCE_DIMENSION_TEXTURE3D");
    assert_eq!(le32(b, d + 12), 1, "arraySize");
    d + 20
}

fn half_to_f64(h: u16) -> f64 {
    let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => s * f64::INFINITY,
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

#[test]
fn dds_export_follows_the_microsoft_header_layout() {
    for mips in [false, true] {
        let vol = bake_volume(&SdfNode::sphere(1.0), &config(8, mips));
        let levels = vol.mip_count() as u32;
        let texels: usize = 512 + vol.mips.iter().map(Vec::len).sum::<usize>();
        for (fmt, bpp, dxgi) in [
            (DdsFormat::R32Float, 4u32, 41u32),
            (DdsFormat::R16Float, 2, 54),
            (DdsFormat::R32G32B32A32Float, 16, 2),
        ] {
            let path = tmp(&format!("{mips}_{dxgi}.dds"));
            export_dds_3d(&vol, path.to_str().unwrap(), fmt).unwrap();
            let b = std::fs::read(&path).unwrap();
            let data = check_dds_header(&b, [8, 8, 8], bpp, levels, dxgi);
            assert_eq!(b.len() - data, texels * bpp as usize, "{fmt:?} mips {mips}");
            let all: Vec<f32> = vol
                .data
                .iter()
                .chain(vol.mips.iter().flatten())
                .copied()
                .collect();
            for (i, &v) in all.iter().enumerate() {
                let o = data + i * bpp as usize;
                match fmt {
                    DdsFormat::R32Float => assert_eq!(le32(&b, o), v.to_bits()),
                    DdsFormat::R32G32B32A32Float => {
                        assert_eq!(le32(&b, o), v.to_bits());
                        assert_eq!(&b[o + 4..o + 16], &[0u8; 12]);
                    }
                    DdsFormat::R16Float => {
                        let h = u16::from_le_bytes([b[o], b[o + 1]]);
                        assert_eq!(half_to_f64(h), half_to_f64(nearest_half(v as f64)), "{v}");
                    }
                }
            }
            std::fs::remove_file(&path).ok();
        }
        let grad = bake_volume_with_normals(&SdfNode::sphere(1.0), &config(8, mips), 0.01);
        let path = tmp(&format!("{mips}_distgrad.dds"));
        export_dds_3d_distgrad(&grad, path.to_str().unwrap()).unwrap();
        let b = std::fs::read(&path).unwrap();
        let data = check_dds_header(&b, [8, 8, 8], 16, grad.mip_count() as u32, 2);
        let all: Vec<VoxelDistGrad> = grad
            .data
            .iter()
            .chain(grad.mips.iter().flatten())
            .copied()
            .collect();
        assert_eq!(b.len() - data, all.len() * 16);
        for (i, v) in all.iter().enumerate() {
            let o = data + 16 * i;
            assert_eq!(
                [
                    le32(&b, o),
                    le32(&b, o + 4),
                    le32(&b, o + 8),
                    le32(&b, o + 12)
                ],
                [
                    v.distance.to_bits(),
                    v.nx.to_bits(),
                    v.ny.to_bits(),
                    v.nz.to_bits()
                ]
            );
        }
        std::fs::remove_file(&path).ok();
    }
}

/// IEEE 754 binary16 nearest to `v`, ties to the even code, by scanning the
/// finite codes of the right sign (independent of any bit manipulation).
fn nearest_half(v: f64) -> u16 {
    let sign = if v.is_sign_negative() { 0x8000u16 } else { 0 };
    let a = v.abs();
    if a >= 65520.0 {
        return sign | 0x7C00; // rounds past the largest finite value
    }
    let mut best = 0u16;
    let mut best_err = f64::INFINITY;
    for code in 0u16..0x7C00 {
        let err = (half_to_f64(code) - a).abs();
        if err < best_err || (err == best_err && code & 1 == 0) {
            best = code;
            best_err = err;
        }
    }
    sign | best
}

#[test]
fn r16_texels_round_to_nearest_even() {
    // values chosen on and around rounding boundaries of binary16
    let one_ulp = 2f32.powi(-10);
    let values = [
        1.0 + one_ulp * 0.5,  // tie: rounds down to the even 1.0
        1.0 + one_ulp * 1.5,  // tie: rounds up to the even 1 + 2 ulp
        1.0 + one_ulp * 0.75, // above half: rounds up
        -2.0 - 2.0 * one_ulp * 0.6,
        3.0e-6,  // subnormal binary16
        65519.0, // largest value that still rounds to 65504
        0.1,
        -0.333,
    ];
    let mut vol = Volume3D::<f32>::new([values.len() as u32, 1, 1], Vec3::ZERO, Vec3::ONE);
    vol.data.copy_from_slice(&values);
    let path = tmp("ties.dds");
    export_dds_3d(&vol, path.to_str().unwrap(), DdsFormat::R16Float).unwrap();
    let b = std::fs::read(&path).unwrap();
    let data = check_dds_header(&b, [values.len() as u32, 1, 1], 2, 1, 54);
    for (i, &v) in values.iter().enumerate() {
        let h = u16::from_le_bytes([b[data + 2 * i], b[data + 2 * i + 1]]);
        assert_eq!(h, nearest_half(v as f64), "value {v}");
    }
    std::fs::remove_file(&path).ok();
}

// ---------------------------------------------------------------------------
// GPU device required (the gpu-parity CI job sets ALICE_SDF_REQUIRE_GPU=1, then
// a missing adapter is a failure, not a skip)
// ---------------------------------------------------------------------------

/// Tetrahedral difference of the analytic sphere, f64, at the given ε.
fn sphere_tetra_normal(p: Vec3, r: f64, e: f64) -> Vec3 {
    let k = [
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0f64],
    ];
    let mut g = [0.0f64; 3];
    for ki in k {
        let q = [
            p.x as f64 + ki[0] * e,
            p.y as f64 + ki[1] * e,
            p.z as f64 + ki[2] * e,
        ];
        let d = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt() - r;
        for a in 0..3 {
            g[a] += ki[a] * d;
        }
    }
    let len = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
    Vec3::new(
        (g[0] / len) as f32,
        (g[1] / len) as f32,
        (g[2] / len) as f32,
    )
}

#[test]
fn gpu_normal_bake_uses_the_requested_epsilon() {
    use alice_sdf::volume::gpu_bake::gpu_bake_volume_with_normals;
    let node = SdfNode::sphere(1.0);
    let cfg = config(8, false); // nodes at ±0.286 … ±2: |p| ≥ 0.49 > ε
    for eps in [0.001f32, 0.2] {
        let vol = match gpu_bake_volume_with_normals(&node, &cfg, eps) {
            Ok(v) => v,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but the GPU normal bake failed: {e}"
                );
                eprintln!("skipping GPU normal bake oracle: {e}");
                return;
            }
        };
        let mut worst_d = 0.0f32;
        let mut worst_n = 0.0f32;
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let p = node_pos(&cfg, x, y, z);
                    let v = vol.get(x, y, z);
                    worst_d = worst_d.max((v.distance - (p.length() - 1.0)).abs());
                    let want = sphere_tetra_normal(p, 1.0, eps as f64);
                    worst_n = worst_n.max((Vec3::new(v.nx, v.ny, v.nz) - want).length());
                }
            }
        }
        // f32 rounding of four `length(p) − 1` values against differences of
        // size ~4ε: 2e-7 / (4·0.001) ≈ 5e-5 at the smaller ε
        assert!(worst_d < 1e-5, "eps {eps}: distance error {worst_d}");
        assert!(
            worst_n < 2e-4,
            "eps {eps}: normal differs from the ε-tetrahedral one by {worst_n}"
        );
    }
}

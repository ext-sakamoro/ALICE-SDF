//! Bake a distance field into 3D volume textures and export them.
//!
//! Run: `cargo run --example volume_bake --features volume`
//!
//! Bakes a sphere with the interpreted and the compiled evaluator, bakes
//! distance + normal, samples the volume trilinearly, builds the mip chains and
//! writes raw and DDS files to the temp directory. Values are checked against
//! the analytic sphere (pinned in `tests/test_volume_api_oracle.rs`). The GPU
//! bake is tried last and reported as skipped when no adapter is available.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::types::SdfNode;
use alice_sdf::volume::bake::{bake_volume, bake_volume_compiled, bake_volume_with_normals};
use alice_sdf::volume::export::{
    export_dds_3d, export_dds_3d_distgrad, export_raw, export_raw_with_mips, DdsFormat,
};
use alice_sdf::volume::gpu_bake::gpu_bake_volume_with_normals;
use alice_sdf::volume::mipchain::generate_mip_chain_distgrad;
use alice_sdf::volume::{BakeConfig, Volume3D, VoxelDistGrad};
use glam::Vec3;

fn main() {
    let r = 1.0;
    let sphere = SdfNode::sphere(r);
    let config = BakeConfig {
        resolution: [32, 32, 32],
        generate_mips: true,
        ..Default::default()
    };

    let vol = bake_volume(&sphere, &config);
    let compiled = bake_volume_compiled(&CompiledSdf::compile(&sphere), &config);
    assert_eq!(vol.data, compiled.data, "compiled bake is bit-identical");
    assert_eq!(vol.mips, compiled.mips);
    println!(
        "volume {:?}: {} voxels, voxel size {:.4}, world size {}, {} mip levels",
        vol.resolution,
        vol.voxel_count(),
        vol.voxel_size().x,
        vol.world_size(),
        compiled.mip_count()
    );
    let (x, y, z) = (20, 9, 14);
    let p = vol.voxel_to_world(x, y, z);
    println!(
        "  voxel ({x},{y},{z}) at {p}: {:+.5} (analytic {:+.5})",
        vol.get(x, y, z),
        p.length() - r
    );
    assert!((vol.get(x, y, z) - (p.length() - r)).abs() < 1e-5);
    assert_eq!(vol.data[vol.index(x, y, z)], vol.get(x, y, z));
    let q = Vec3::new(0.31, -0.42, 0.77);
    let s = vol.sample_trilinear(q);
    println!(
        "  trilinear at {q}: {s:+.4} (analytic {:+.4})",
        q.length() - r
    );
    assert!((s - (q.length() - r)).abs() < 0.01);

    // distance + normal
    let grad = bake_volume_with_normals(&sphere, &config, 0.01);
    let mut worst = 0.0f32;
    for i in 0..grad.voxel_count() {
        let v: VoxelDistGrad = grad.data[i];
        let (x, y, z) = (i as u32 % 32, (i as u32 / 32) % 32, i as u32 / 1024);
        let p = grad.voxel_to_world(x, y, z);
        worst = worst.max((Vec3::new(v.nx, v.ny, v.nz) - p.normalize()).length());
    }
    let sampled = grad.sample_trilinear(Vec3::new(0.0, 1.0, 0.0));
    println!("  normals: worst deviation from radial {worst:.2e}; sampled at the pole n = ({:.3}, {:.3}, {:.3})", sampled.nx, sampled.ny, sampled.nz);
    assert!(worst < 1e-3);
    let mips = generate_mip_chain_distgrad(&grad);
    assert_eq!(mips.len(), grad.mips.len());

    // a hand-filled volume
    let mut ramp = Volume3D::<f32>::new([4, 4, 4], Vec3::ZERO, Vec3::splat(3.0));
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                ramp.set(x, y, z, x as f32 + 2.0 * y as f32 - z as f32);
            }
        }
    }
    let v = ramp.sample_trilinear(Vec3::new(1.25, 2.5, 0.75));
    println!(
        "  linear field sampled exactly: {v} (closed form {})",
        1.25 + 5.0 - 0.75
    );
    assert!((v - 5.5).abs() < 1e-5);

    // export
    let dir = std::env::temp_dir();
    let path = |name: &str| {
        dir.join(format!("alice_sdf_volume_bake_{name}"))
            .to_string_lossy()
            .into_owned()
    };
    export_raw(&vol, &path("dist.raw")).unwrap();
    export_raw_with_mips(&vol, &path("dist_mips.raw")).unwrap();
    for (fmt, name) in [
        (DdsFormat::R32Float, "r32.dds"),
        (DdsFormat::R16Float, "r16.dds"),
        (DdsFormat::R32G32B32A32Float, "rgba.dds"),
    ] {
        export_dds_3d(&vol, &path(name), fmt).unwrap();
    }
    export_dds_3d_distgrad(&grad, &path("distgrad.dds")).unwrap();
    let raw_len = std::fs::metadata(path("dist.raw")).unwrap().len();
    let dds_len = std::fs::metadata(path("r16.dds")).unwrap().len();
    let texels = vol.voxel_count() + vol.mips.iter().map(Vec::len).sum::<usize>();
    println!(
        "  wrote {raw_len} raw bytes and a {dds_len}-byte R16 DDS ({texels} texels over {} levels)",
        vol.mip_count()
    );
    assert_eq!(raw_len, 4 * 32 * 32 * 32);
    assert_eq!(dds_len as usize, 4 + 124 + 20 + 2 * texels);
    for name in [
        "dist.raw",
        "dist_mips.raw",
        "r32.dds",
        "r16.dds",
        "rgba.dds",
        "distgrad.dds",
    ] {
        std::fs::remove_file(path(name)).ok();
    }
    for name in ["dist.raw", "dist_mips.raw"] {
        std::fs::remove_file(std::path::Path::new(&path(name)).with_extension("raw.meta.json"))
            .ok();
    }

    // GPU bake (optional at run time)
    match gpu_bake_volume_with_normals(
        &sphere,
        &BakeConfig {
            resolution: [16, 16, 16],
            ..Default::default()
        },
        0.01,
    ) {
        Ok(gpu) => {
            let v = gpu.get(12, 8, 8);
            let p = gpu.voxel_to_world(12, 8, 8);
            println!(
                "  GPU bake: voxel (12,8,8) {:+.5} (analytic {:+.5})",
                v.distance,
                p.length() - r
            );
            assert!((v.distance - (p.length() - r)).abs() < 1e-5);
        }
        Err(e) => println!("  GPU bake skipped: {e}"),
    }
}

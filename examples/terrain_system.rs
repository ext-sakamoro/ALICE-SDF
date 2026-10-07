//! Terrain: a heightmap from an image, a ramp from raw data, caves and a
//! chamber cut into it, clipmap LOD meshes and a splatmap.
//!
//! Run: `cargo run --example terrain_system --features terrain,image`
//!
//! Every printed value is checked against a closed form (pinned in
//! `tests/test_terrain_api_oracle.rs` and `tests/test_terrain_destruction_oracle.rs`).
//!
//! Author: Moroya Sakamoto

use alice_sdf::eval::eval;
use alice_sdf::terrain::caves::generate_chamber;
use alice_sdf::terrain::{
    generate_cave_sdf, terrain_sdf, CaveConfig, ClipmapLevel, ClipmapMesh, ClipmapTerrain,
    Heightmap, HeightmapImageConfig, SplatLayer, Splatmap, TerrainConfig,
};
use glam::Vec3;

fn main() {
    let tc = TerrainConfig {
        width: 64.0,
        depth: 64.0,
        height_scale: 10.0,
        resolution_x: 32,
        resolution_z: 32,
        clipmap_levels: 3,
        clipmap_resolution: 17,
    };

    // --- a heightmap from an in-memory 8-bit grey PNG ---------------------
    let (w, d) = (tc.resolution_x, tc.resolution_z);
    let pixels: Vec<u8> = (0..w * d).map(|i| ((i % w) * 8) as u8).collect(); // ramp in x
    let img = image::GrayImage::from_raw(w, d, pixels).unwrap();
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let cfg = HeightmapImageConfig::new(tc.height_scale, tc.width, tc.depth);
    let from_png = Heightmap::from_image_bytes(&png, &cfg).unwrap();
    let path = std::env::temp_dir().join("alice_sdf_terrain_system.png");
    std::fs::write(&path, &png).unwrap();
    let from_file = Heightmap::from_image(&path, &cfg).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(from_png.get_height(31, 0), from_file.get_height(31, 0));
    println!(
        "png heightmap {}x{}: height at column 31 = {:.4} (closed form 248·{}/255 = {:.4})",
        from_png.width,
        from_png.depth,
        from_png.get_height(31, 0),
        tc.height_scale,
        248.0 * tc.height_scale / 255.0
    );
    assert!((from_png.get_height(31, 0) - 248.0 * tc.height_scale / 255.0).abs() < 1e-5);

    // --- an analytic ramp from raw data: h = a·x + b·z --------------------
    let (a, b) = (0.2f32, -0.1f32);
    let data: Vec<f32> = (0..w * d)
        .map(|i| {
            let (x, z) = ((i % w) as f32 * 2.0, (i / w) as f32 * 2.0);
            a * x + b * z
        })
        .collect();
    let hm = Heightmap::from_data(data, w, d, tc.width, tc.depth);
    let (sx, sz) = (21.0, 13.0);
    let n = hm.normal_at(sx, sz);
    let n_true = Vec3::new(-a, 1.0, -b).normalize();
    println!(
        "ramp: bilinear {:.4}, bicubic {:.4}, analytic {:.4}; normal {n:.4} (analytic {n_true:.4})",
        hm.sample(sx, sz),
        hm.sample_bicubic(sx, sz),
        a * sx + b * sz
    );
    assert!((hm.sample_bicubic(sx, sz) - (a * sx + b * sz)).abs() < 1e-4);
    assert!((n - n_true).length() < 1e-5);

    // --- caves and a chamber subtracted from the terrain ------------------
    let caves = CaveConfig {
        density: 0.2,
        octaves: 0,
        ..Default::default()
    };
    let cave = generate_cave_sdf(&caves);
    let chamber = generate_chamber(Vec3::new(30.0, -12.0, 30.0), 6.0, 4.0);
    let underground = cave.clone().union(chamber);
    let p_room = Vec3::new(30.0, -12.0, 30.0);
    let ground = terrain_sdf(&hm, p_room, None);
    let carved = terrain_sdf(&hm, p_room, Some(&underground));
    println!(
        "terrain sdf at the chamber centre: solid {ground:+.3}, with the chamber {carved:+.3}"
    );
    assert!(ground < 0.0 && carved > 0.0, "the chamber is hollow");
    assert_eq!(carved, ground.max(-eval(&underground, p_room)));
    let ceiling = -caves.min_depth + 1.5 * caves.tunnel_radius + caves.tunnel_radius / 8.0;
    assert!(eval(&cave, Vec3::new(0.0, ceiling + 0.1, 0.0)) > 0.0);

    // --- clipmap LOD -------------------------------------------------------
    let mut cm = ClipmapTerrain::new(tc.clipmap_levels, tc.clipmap_resolution, 1.0);
    cm.update(Vec3::new(30.0, 5.0, 30.0));
    let meshes: Vec<ClipmapMesh> = cm.generate_meshes(&hm);
    let fine = cm.generate_level_mesh(0, &hm).unwrap();
    let levels: &[ClipmapLevel] = &cm.levels;
    println!(
        "clipmap: {} levels, {} vertices total, spacings {:?}",
        cm.level_count(),
        cm.total_vertices(),
        levels.iter().map(|l| l.spacing).collect::<Vec<_>>()
    );
    assert_eq!(meshes.len(), 3);
    assert_eq!(fine.mesh.vertices.len(), 17 * 17);
    for v in &fine.mesh.vertices {
        assert!((v.position.y - (a * v.position.x + b * v.position.z)).abs() < 1e-3);
    }

    // --- splatmap ------------------------------------------------------------
    let mut sp = Splatmap::new(16, 16);
    let grass = sp.add_layer("grass", 1, 1.0);
    let rock = sp.add_layer("rock", 2, 0.0);
    sp.set_weight(rock, 3, 4, 2.0);
    sp.normalize();
    let layer: &SplatLayer = &sp.layers[rock];
    println!(
        "splatmap: {} layers; texel (3,4) rock {:.3} grass {:.3}, dominant id {} ({})",
        sp.layer_count(),
        sp.get_weight(rock, 3, 4),
        sp.get_weight(grass, 3, 4),
        sp.dominant_material(3, 4),
        layer.name
    );
    assert!((sp.get_weight(rock, 3, 4) - 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(sp.dominant_material(3, 4), 2);
    assert_eq!(sp.dominant_material(0, 0), 1);
    sp.auto_splat_from_heightmap(&hm, 0.3, 5.0);
    assert!((sp.get_weight(0, 5, 5) + sp.get_weight(1, 5, 5) - 1.0).abs() < 1e-6);
}

//! Oracles for the terrain API that `examples/terrain_system.rs` wires: caves
//! and chambers, the clipmap meshes, heightmaps from raw data and images, and
//! the splatmap accessors.
//!
//! | target | oracle |
//! |---|---|
//! | `generate_cave_sdf` (no noise) | containment bound from the documented construction: every tunnel sphere has its centre at `y ∈ [−max_depth, −min_depth]` and radius `≤ 1.5·tunnel_radius`, and the polynomial smooth union lowers a minimum by at most `k/4` (`k = tunnel_radius/2`), so the field is positive above `−min_depth + 1.5·r + r/8`; the deepest sphere centre is inside |
//! | `generate_chamber` | the rounded box closed form `‖max(q, 0)‖ + min(max_i qᵢ, 0) − 0.3·radius`, `q = |p − c| − b`, with `b = (radius, height/2, radius)/2` because `box3d` takes full sizes |
//! | clipmap meshes | grid vertices `origin + g·spacing` on the analytic ramp `h = a·x + b·z`, vertex normals `normalize(−a, 1, −b)`, every face wound counter-clockwise seen from above, `res²` vertices and `2·(res−1)²` faces per level |
//! | `Heightmap::from_data` / `from_image*` | interpolation at the nodes returns the data; an 8-bit grey pixel `p` maps to `p · height_scale / 255`, world size defaults to the pixel size |
//! | `Splatmap` accessors | `dominant_material` is the arg-max of `get_weight` over the layers (first layer on ties, 0 where every weight is 0, including outside the map) |
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "terrain")]

use alice_sdf::eval::eval;
use alice_sdf::terrain::caves::generate_chamber;
use alice_sdf::terrain::{generate_cave_sdf, CaveConfig, ClipmapTerrain, Heightmap, Splatmap};
use glam::Vec3;

#[test]
fn caves_stay_below_their_documented_ceiling() {
    // density 0.8 (64 segments) is left out: the segments are chained as a
    // left-deep smooth union, and evaluating that 64-deep tree overflows the
    // 2 MiB test-thread stack in a debug build
    for (density, seed) in [(0.05f32, 1u64), (0.3, 42), (0.5, 7)] {
        let cfg = CaveConfig {
            density,
            seed,
            octaves: 0,
            ..Default::default()
        };
        let cave = generate_cave_sdf(&cfg);
        let r = cfg.tunnel_radius;
        let ceiling = -cfg.min_depth + 1.5 * r + r / 8.0;
        let mut checked = 0;
        for ix in -12..=12 {
            for iz in -12..=12 {
                for dy in [0.01f32, 1.0, 10.0] {
                    let p = Vec3::new(ix as f32 * 5.0, ceiling + dy, iz as f32 * 5.0);
                    assert!(
                        eval(&cave, p) > 0.0,
                        "density {density} seed {seed}: cave above {p}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 25 * 25 * 3);
        // some point between the depth limits is inside a tunnel
        let mut inside = false;
        'scan: for ix in -30..=30 {
            for iz in -30..=30 {
                for iy in 0..=20 {
                    let y = -cfg.min_depth - (cfg.max_depth - cfg.min_depth) * iy as f32 / 20.0;
                    if eval(&cave, Vec3::new(ix as f32 * 2.0, y, iz as f32 * 2.0)) < 0.0 {
                        inside = true;
                        break 'scan;
                    }
                }
            }
        }
        assert!(inside, "density {density}: no tunnel found below ground");
        // determinism
        let again = generate_cave_sdf(&cfg);
        let p = Vec3::new(1.5, -20.0, -3.0);
        assert_eq!(eval(&cave, p).to_bits(), eval(&again, p).to_bits());
    }
}

#[test]
fn chamber_is_the_rounded_box_closed_form() {
    let c = Vec3::new(2.0, -10.0, 1.0);
    let (radius, height) = (5.0f32, 4.0f32);
    let chamber = generate_chamber(c, radius, height);
    let b = Vec3::new(radius, height * 0.5, radius) * 0.5;
    let mut n = 0;
    for i in 0..343 {
        let f = Vec3::new((i % 7) as f32, ((i / 7) % 7) as f32, (i / 49) as f32) / 6.0;
        let p = c + (f - 0.5) * Vec3::new(4.0 * radius, 2.0 * height + 4.0, 4.0 * radius);
        let q = (p - c).abs() - b;
        let want = q.max(Vec3::ZERO).length() + q.max_element().min(0.0) - 0.3 * radius;
        assert!((eval(&chamber, p) - want).abs() < 1e-4, "at {p}");
        n += 1;
    }
    assert_eq!(n, 343);
}

fn ramp(a: f32, b: f32) -> Heightmap {
    let (w, d, world) = (64u32, 64u32, 128.0f32);
    let mut data = Vec::with_capacity((w * d) as usize);
    for z in 0..d {
        for x in 0..w {
            let wx = x as f32 * world / w as f32;
            let wz = z as f32 * world / d as f32;
            data.push(a * wx + b * wz);
        }
    }
    Heightmap::from_data(data, w, d, world, world)
}

#[test]
fn clipmap_meshes_lie_on_the_ramp_with_upward_faces() {
    let (a, b) = (0.25f32, -0.4f32);
    let hm = ramp(a, b);
    let mut cm = ClipmapTerrain::new(3, 9, 1.5);
    cm.update(Vec3::new(40.0, 0.0, 50.0));
    assert_eq!(cm.level_count(), 3);
    assert_eq!(cm.total_vertices(), 3 * 81);
    let all = cm.generate_meshes(&hm);
    assert_eq!(all.len(), 3);
    let n_want = Vec3::new(-a, 1.0, -b).normalize();
    for (i, level) in cm.levels.iter().enumerate() {
        let one = cm
            .generate_level_mesh(i as u32, &hm)
            .expect("level in range");
        assert_eq!(one.level, level.level);
        assert_eq!(one.mesh.indices, all[i].mesh.indices);
        assert_eq!(one.mesh.vertices.len(), 81);
        assert_eq!(one.mesh.indices.len(), 6 * 64);
        for (k, v) in one.mesh.vertices.iter().enumerate() {
            let (gx, gz) = ((k % 9) as f32, (k / 9) as f32);
            let wx = level.origin_x + gx * level.spacing;
            let wz = level.origin_z + gz * level.spacing;
            assert!((v.position.x - wx).abs() < 1e-4 && (v.position.z - wz).abs() < 1e-4);
            assert!(
                (v.position.y - (a * wx + b * wz)).abs() < 1e-3,
                "height at {wx},{wz}"
            );
            assert!((v.normal - n_want).length() < 1e-4);
            assert_eq!(all[i].mesh.vertices[k].position, v.position);
        }
        for t in one.mesh.indices.chunks_exact(3) {
            let p = |j: u32| one.mesh.vertices[j as usize].position;
            let face = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
            assert!(face.normalize().dot(n_want) > 0.999, "face winding");
        }
    }
    assert!(cm.generate_level_mesh(3, &hm).is_none());
}

#[test]
fn heightmap_from_data_interpolates_its_nodes() {
    let data: Vec<f32> = (0..12).map(|i| (i * i) as f32 * 0.5 - 3.0).collect();
    let hm = Heightmap::from_data(data.clone(), 4, 3, 8.0, 6.0);
    for z in 0..3u32 {
        for x in 0..4u32 {
            let want = data[(x + 4 * z) as usize];
            assert_eq!(hm.get_height(x, z), want);
            // node (x, z) sits at world (x·8/4, z·6/3)
            assert_eq!(hm.sample(x as f32 * 2.0, z as f32 * 2.0), want);
        }
    }
}

#[test]
#[should_panic(expected = "assertion")]
fn heightmap_from_data_rejects_a_wrong_length() {
    let _ = Heightmap::from_data(vec![0.0; 5], 2, 3, 1.0, 1.0);
}

#[cfg(feature = "image")]
mod image_oracle {
    use alice_sdf::terrain::{Heightmap, HeightmapImageConfig};

    fn png(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
        let pixels: Vec<u8> = (0..width * height)
            .map(|i| ((i * 37) % 256) as u8)
            .collect();
        let img = image::GrayImage::from_raw(width, height, pixels.clone()).unwrap();
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .unwrap();
        (bytes, pixels)
    }

    #[test]
    fn grey_pixels_map_linearly_to_height() {
        let (bytes, pixels) = png(7, 5);
        let cfg = HeightmapImageConfig::new(51.0, 70.0, 25.0);
        let from_bytes = Heightmap::from_image_bytes(&bytes, &cfg).unwrap();
        let path = std::env::temp_dir().join(format!("alice_sdf_hm_{}.png", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let from_file = Heightmap::from_image(&path, &cfg).unwrap();
        std::fs::remove_file(&path).ok();
        for hm in [&from_bytes, &from_file] {
            assert_eq!((hm.width, hm.depth), (7, 5));
            assert_eq!((hm.world_width, hm.world_depth), (70.0, 25.0));
            for z in 0..5u32 {
                for x in 0..7u32 {
                    let p = pixels[(x + 7 * z) as usize] as f32;
                    assert_eq!(hm.get_height(x, z), p * (51.0 / 255.0));
                }
            }
        }
        // defaults: world size in pixels, 255 maps to 100
        let d = Heightmap::from_image_bytes(&bytes, &HeightmapImageConfig::default()).unwrap();
        assert_eq!((d.world_width, d.world_depth), (7.0, 5.0));
        assert_eq!(d.get_height(0, 0), 0.0);
        assert!(Heightmap::from_image_bytes(b"not an image", &cfg).is_err());
        assert!(Heightmap::from_image("/nonexistent/alice.png", &cfg).is_err());
    }
}

#[test]
fn dominant_material_is_the_argmax_of_the_layer_weights() {
    let mut sp = Splatmap::new(5, 4);
    assert_eq!(sp.layer_count(), 0);
    assert_eq!(sp.dominant_material(1, 1), 0, "no layers");
    let grass = sp.add_layer("grass", 10, 0.2);
    let rock = sp.add_layer("rock", 20, 0.2);
    let snow = sp.add_layer("snow", 30, 0.0);
    assert_eq!((grass, rock, snow, sp.layer_count()), (0, 1, 2, 3));
    sp.set_weight(rock, 3, 2, 0.9);
    sp.set_weight(snow, 0, 3, 0.5);
    sp.set_weight(grass, 4, 0, 0.0);
    sp.set_weight(rock, 4, 0, 0.0);
    sp.set_weight(9, 0, 0, 1.0); // missing layer: ignored
    sp.set_weight(rock, 5, 0, 1.0); // outside: ignored
    for z in 0..4u32 {
        for x in 0..5u32 {
            let w: Vec<f32> = (0..3).map(|l| sp.get_weight(l, x, z)).collect();
            let mut best = (0.0f32, 0u16);
            for (l, &wl) in w.iter().enumerate() {
                if wl > best.0 {
                    best = (wl, sp.layers[l].material_id);
                }
            }
            assert_eq!(
                sp.dominant_material(x, z),
                best.1,
                "texel ({x},{z}) weights {w:?}"
            );
        }
    }
    assert_eq!(sp.dominant_material(3, 2), 20);
    assert_eq!(sp.dominant_material(0, 3), 30);
    assert_eq!(
        sp.dominant_material(1, 1),
        10,
        "tie goes to the first layer"
    );
    assert_eq!(sp.dominant_material(4, 0), 0, "all weights zero");
    // outside the map every weight reads 0, so there is no dominant material:
    // (5, 0) must not alias texel (0, 1)
    assert_eq!(sp.get_weight(rock, 5, 0), 0.0);
    assert_eq!(sp.dominant_material(5, 0), 0);
    assert_eq!(sp.dominant_material(0, 4), 0);
}

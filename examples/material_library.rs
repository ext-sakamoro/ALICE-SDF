//! Build a PBR material library and export it with a mesh
//!
//! Walks the whole `material` module: the standard presets, every builder
//! (scalar factors, texture slots with UV channel and tiling, the extended
//! PBR layers), the particle projection, `material_lerp`, and the library
//! lookups. The library is then exported to glTF (GLB, in memory) so the
//! materials end up in the file a renderer reads.
//!
//! # Running
//! ```bash
//! cargo run --example material_library
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::io::{export_glb_bytes, GltfConfig};
use alice_sdf::material::{
    material_lerp, Material, MaterialLibrary, ParticleMaterial, StandardMaterials, TextureSlot,
};
use alice_sdf::prelude::{sdf_to_mesh, MarchingCubesConfig, SdfNode};
use glam::Vec3;

fn main() {
    // --- Presets -----------------------------------------------------------
    let presets = [
        StandardMaterials::gold(),
        StandardMaterials::aluminum(),
        StandardMaterials::copper(),
        StandardMaterials::chrome(),
        StandardMaterials::plastic_white(),
        StandardMaterials::plastic_red(),
        StandardMaterials::glass(),
        StandardMaterials::diamond(),
        StandardMaterials::water(),
        StandardMaterials::skin(),
        StandardMaterials::marble(),
        StandardMaterials::wet_asphalt(),
        StandardMaterials::velvet(),
        StandardMaterials::rubber(),
        StandardMaterials::concrete(),
    ];
    println!(
        "{:<14} {:>8} {:>9} {:>5}",
        "preset", "metallic", "roughness", "ior"
    );
    for m in &presets {
        println!(
            "{:<14} {:>8.2} {:>9.2} {:>5.2}",
            m.name, m.metallic, m.roughness, m.ior
        );
        assert!(m.metallic == 0.0 || m.metallic == 1.0, "{}", m.name);
    }

    // --- Builders ----------------------------------------------------------
    let car_paint = Material::new("CarPaint")
        .with_color(0.6, 0.02, 0.05, 1.0)
        .with_metallic(0.4)
        .with_roughness(0.35)
        .with_clearcoat(1.0, 0.05)
        .with_albedo_map("car_albedo.png")
        .with_normal_map("car_normal.png")
        .with_metallic_roughness_map("car_mr.png");
    let mut lightmapped = Material::dielectric("Floor", 0.5, 0.5, 0.5, 0.8)
        .with_ao_map("floor_ao.png")
        .with_emissive_map("floor_emit.png")
        .with_emission(1.0, 0.8, 0.6, 0.5);
    // Baked occlusion lives on the lightmap UV set (TEXCOORD_1), tiled 4x.
    lightmapped.ao_map = Some(
        TextureSlot::new("floor_ao.png")
            .with_uv_channel(1)
            .with_tiling(4.0, 4.0),
    );
    let tinted_glass = Material::glass("TintedGlass", 1.52)
        .with_transmission(0.95)
        .with_volume(0.01, 0.25, 0.4, 0.8, 0.6);
    let brushed = Material::metal("Brushed", 0.9, 0.9, 0.92, 0.3).with_anisotropy(0.7, 0.25);
    let fabric = Material::dielectric("Fabric", 0.2, 0.2, 0.5, 0.9).with_sheen(0.5, 0.5, 0.9, 0.6);
    let wax = Material::dielectric("Wax", 0.9, 0.85, 0.7, 0.4).with_subsurface(0.6, 1.0, 0.7, 0.4);
    let lamp = Material::emissive("Lamp", 1.0, 0.9, 0.7, 8.0);

    assert_eq!(car_paint.clearcoat, 1.0);
    assert_eq!(lightmapped.ao_map.as_ref().map(|s| s.uv_channel), Some(1));
    assert_eq!(tinted_glass.attenuation_distance, 0.25);
    assert_eq!(lamp.emission_strength, 8.0);

    // --- Blend -------------------------------------------------------------
    let half = material_lerp(
        &StandardMaterials::gold(),
        &StandardMaterials::copper(),
        0.5,
    );
    println!(
        "\n{}: base colour {:?}, roughness {:.3}",
        half.name, half.base_color, half.roughness
    );
    let ends = material_lerp(&brushed, &fabric, 0.0);
    assert_eq!(
        ends.roughness, brushed.roughness,
        "t = 0 is the first input"
    );

    // --- Particles ---------------------------------------------------------
    let spark = lamp.to_particle();
    let ember = ParticleMaterial::from_material(&StandardMaterials::copper());
    let solid = ParticleMaterial::solid(0.2, 0.4, 0.9);
    let glow = ParticleMaterial::glow(1.0, 0.5, 0.1, 3.0);
    println!(
        "particles: spark emission {}, ember colour {:?}, solid {:?}, glow {}",
        spark.emission_strength, ember.color, solid.color, glow.emission_strength
    );
    assert_eq!(spark.emission_strength, 8.0);
    assert_eq!(std::mem::size_of::<ParticleMaterial>(), 24);

    // --- Library -----------------------------------------------------------
    let mut lib = MaterialLibrary::new();
    for m in [
        car_paint,
        lightmapped,
        tinted_glass,
        brushed,
        fabric,
        wax,
        lamp,
    ] {
        lib.add(m);
    }
    for m in presets {
        lib.add(m);
    }
    println!(
        "\nlibrary: {} materials (default: {}), empty: {}",
        lib.len(),
        lib.default_material().name,
        lib.is_empty()
    );
    let (floor_id, floor) = lib.find_by_name("Floor").expect("Floor is in the library");
    assert_eq!(lib.get(floor_id).map(|m| &m.name), Some(&floor.name));
    assert_eq!(lib.iter().count(), lib.len());

    // --- Export ------------------------------------------------------------
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 12,
            ..Default::default()
        },
    );
    let glb = export_glb_bytes(
        &mesh,
        &GltfConfig {
            export_extensions: true,
            ..Default::default()
        },
        Some(&lib),
    )
    .expect("GLB export");
    let json_len = u32::from_le_bytes(glb[12..16].try_into().expect("GLB header")) as usize;
    let json = std::str::from_utf8(&glb[20..20 + json_len]).expect("JSON chunk");
    println!(
        "GLB: {} bytes, lightmap texCoord written: {}",
        glb.len(),
        json.contains(r#""texCoord":1"#)
    );
    assert!(json.contains(r#""texCoord":1"#));
    assert!(json.contains("KHR_materials_clearcoat"));
}

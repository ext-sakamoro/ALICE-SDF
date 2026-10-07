//! PBR material data model vs its own contract (`material`).
//!
//! The material module holds data, not shading: every builder is a field
//! assignment with a documented clamp, the presets are named points of the
//! metallic-roughness workflow, and `material_lerp` is a per-field linear
//! blend. The oracles below are written from those definitions:
//!
//! - each `with_*` builder changes exactly the fields it names (checked by
//!   diffing the serde JSON of the result against `Material::default()`, so a
//!   builder that also touches a neighbouring field is caught) and applies the
//!   clamp `min(max(x, lo), hi)` written in its doc
//! - `metal` / `dielectric` / `glass` / `emissive` set the metallic factor,
//!   transmission and emission to the values that define those classes
//! - presets: metals have metallic 1, dielectrics 0, and the indices of
//!   refraction are the physical constants (glass 1.5, diamond 2.42, water
//!   1.33 — the same constants the `ior` field doc names)
//! - `material_lerp` returns its end points at `t = 0` / `t = 1`, the
//!   closed form `a + t (b - a)` in between, clamps `t` to `[0, 1]`, and takes
//!   texture slots from `a` below one half and from `b` from one half on
//! - `MaterialLibrary` ids are the insertion order starting after the default
//!   material at id 0
//!
//! Author: Moroya Sakamoto

use alice_sdf::material::{
    material_lerp, Material, MaterialLibrary, ParticleMaterial, StandardMaterials, TextureSlot,
};
use serde_json::Value;
use std::collections::BTreeSet;

/// Top-level JSON keys whose values differ between two materials.
fn changed_fields(a: &Material, b: &Material) -> BTreeSet<String> {
    let va = serde_json::to_value(a).expect("serialize a");
    let vb = serde_json::to_value(b).expect("serialize b");
    let (Value::Object(ma), Value::Object(mb)) = (va, vb) else {
        panic!("Material must serialize to a JSON object");
    };
    assert_eq!(
        ma.keys().collect::<Vec<_>>(),
        mb.keys().collect::<Vec<_>>(),
        "both materials serialize the same field set"
    );
    ma.iter()
        .filter(|(k, v)| mb.get(*k) != Some(*v))
        .map(|(k, _)| k.clone())
        .collect()
}

fn set(keys: &[&str]) -> BTreeSet<String> {
    keys.iter().map(|k| (*k).to_string()).collect()
}

/// The clamp each builder documents, written out instead of calling `clamp`.
fn clamp_ref(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

const SWEEP: [f32; 9] = [-2.0, -1.0, -0.25, 0.0, 0.3, 0.5, 1.0, 1.5, 7.0];

#[test]
fn texture_slot_defaults_and_builders() {
    let s = TextureSlot::new("albedo.png");
    assert_eq!(s.path, "albedo.png");
    assert_eq!(s.uv_channel, 0, "primary UV set by default");
    assert_eq!(s.tiling, [1.0, 1.0], "no repeat by default");
    assert_eq!(s.offset, [0.0, 0.0]);

    let t = TextureSlot::new("lightmap.exr")
        .with_uv_channel(1)
        .with_tiling(4.0, 0.5);
    assert_eq!(t.uv_channel, 1);
    assert_eq!(t.tiling, [4.0, 0.5]);
    assert_eq!(t.offset, [0.0, 0.0], "tiling does not move the offset");
    assert_eq!(t.path, "lightmap.exr");
}

#[test]
fn scalar_builders_change_only_their_fields_and_clamp() {
    let base = Material::default();
    let mut compared = 0usize;
    for &x in &SWEEP {
        let m = base.clone().with_metallic(x);
        assert_eq!(m.metallic.to_bits(), clamp_ref(x, 0.0, 1.0).to_bits());
        assert!(changed_fields(&base, &m).is_subset(&set(&["metallic"])));

        let m = base.clone().with_roughness(x);
        assert_eq!(m.roughness.to_bits(), clamp_ref(x, 0.0, 1.0).to_bits());
        assert!(changed_fields(&base, &m).is_subset(&set(&["roughness"])));

        let m = base.clone().with_transmission(x);
        assert_eq!(m.transmission.to_bits(), clamp_ref(x, 0.0, 1.0).to_bits());
        assert!(changed_fields(&base, &m).is_subset(&set(&["transmission"])));

        let m = base.clone().with_clearcoat(x, 1.0 - x);
        assert_eq!(m.clearcoat.to_bits(), clamp_ref(x, 0.0, 1.0).to_bits());
        assert_eq!(
            m.clearcoat_roughness.to_bits(),
            clamp_ref(1.0 - x, 0.0, 1.0).to_bits()
        );
        assert!(changed_fields(&base, &m).is_subset(&set(&["clearcoat", "clearcoat_roughness"])));

        let m = base.clone().with_anisotropy(x, 0.25);
        assert_eq!(m.anisotropy.to_bits(), clamp_ref(x, -1.0, 1.0).to_bits());
        assert_eq!(m.anisotropy_rotation, 0.25, "rotation is not clamped");
        assert!(changed_fields(&base, &m).is_subset(&set(&["anisotropy", "anisotropy_rotation"])));

        let m = base.clone().with_sheen(0.1, 0.2, 0.3, x);
        assert_eq!(m.sheen_color, [0.1, 0.2, 0.3]);
        assert_eq!(
            m.sheen_roughness.to_bits(),
            clamp_ref(x, 0.0, 1.0).to_bits()
        );
        assert!(changed_fields(&base, &m).is_subset(&set(&["sheen_color", "sheen_roughness"])));

        let m = base.clone().with_subsurface(x, 0.9, 0.4, 0.3);
        assert_eq!(m.subsurface.to_bits(), clamp_ref(x, 0.0, 1.0).to_bits());
        assert_eq!(m.subsurface_color, [0.9, 0.4, 0.3]);
        assert!(changed_fields(&base, &m).is_subset(&set(&["subsurface", "subsurface_color"])));
        compared += 7;
    }
    assert_eq!(compared, 7 * SWEEP.len());
}

#[test]
fn unclamped_builders_store_their_arguments() {
    let base = Material::default();

    let m = base.clone().with_color(0.1, 0.2, 0.3, 0.4);
    assert_eq!(m.base_color, [0.1, 0.2, 0.3, 0.4]);
    assert_eq!(changed_fields(&base, &m), set(&["base_color"]));

    let m = base.clone().with_emission(1.0, 0.5, 0.25, 3.0);
    assert_eq!(m.emission, [1.0, 0.5, 0.25]);
    assert_eq!(m.emission_strength, 3.0);
    assert_eq!(
        changed_fields(&base, &m),
        set(&["emission", "emission_strength"])
    );

    let m = base.clone().with_volume(0.02, 0.5, 0.9, 0.8, 0.7);
    assert_eq!(m.thickness, 0.02);
    assert_eq!(m.attenuation_distance, 0.5);
    assert_eq!(m.attenuation_color, [0.9, 0.8, 0.7]);
    assert_eq!(
        changed_fields(&base, &m),
        set(&["thickness", "attenuation_distance", "attenuation_color"])
    );
}

#[test]
fn texture_builders_fill_exactly_one_slot() {
    let base = Material::default();
    type Builder = fn(Material, &str) -> Material;
    let cases: [(&str, Builder); 5] = [
        ("albedo_map", |m, p| m.with_albedo_map(p)),
        ("normal_map", |m, p| m.with_normal_map(p)),
        ("metallic_roughness_map", |m, p| {
            m.with_metallic_roughness_map(p)
        }),
        ("ao_map", |m, p| m.with_ao_map(p)),
        ("emissive_map", |m, p| m.with_emissive_map(p)),
    ];
    for (field, build) in cases {
        let path = format!("{field}.png");
        let m = build(base.clone(), &path);
        assert_eq!(changed_fields(&base, &m), set(&[field]), "{field}");
        let v = serde_json::to_value(&m).unwrap();
        assert_eq!(v[field]["path"], Value::from(path.as_str()), "{field}");
        assert_eq!(v[field]["uv_channel"], Value::from(0), "{field}");
        assert_eq!(v[field]["tiling"], serde_json::json!([1.0, 1.0]), "{field}");
    }
}

#[test]
fn class_constructors_set_the_defining_fields() {
    let m = Material::new("n");
    assert_eq!(changed_fields(&Material::default(), &m), set(&["name"]));

    let m = Material::metal("Steel", 0.6, 0.6, 0.62, 3.0);
    assert_eq!(m.metallic, 1.0);
    assert_eq!(
        m.roughness, 1.0,
        "roughness argument goes through the clamp"
    );
    assert_eq!(m.base_color, [0.6, 0.6, 0.62, 1.0], "metals are opaque");

    let m = Material::dielectric("Plastic", 0.8, 0.1, 0.1, -1.0);
    assert_eq!(m.metallic, 0.0);
    assert_eq!(m.roughness, 0.0);
    assert_eq!(m.base_color[3], 1.0);

    let m = Material::glass("Crown", 1.52);
    assert_eq!(m.ior, 1.52);
    assert_eq!(m.transmission, 1.0, "glass transmits");
    assert_eq!(m.metallic, 0.0);
    assert_eq!(m.roughness, 0.0, "clear glass is smooth");

    let m = Material::emissive("Lamp", 1.0, 0.9, 0.8, 5.0);
    assert_eq!(m.emission, [1.0, 0.9, 0.8]);
    assert_eq!(m.emission_strength, 5.0);
    assert_eq!(
        changed_fields(&Material::default(), &m),
        set(&["name", "emission", "emission_strength"])
    );
}

#[test]
fn particle_material_is_the_lossy_projection() {
    assert_eq!(std::mem::size_of::<ParticleMaterial>(), 24, "4 + 1 + 1 f32");

    let p = ParticleMaterial::solid(0.1, 0.2, 0.3);
    assert_eq!(p.color, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!((p.emission_strength, p.opacity), (0.0, 1.0));

    let p = ParticleMaterial::glow(1.0, 0.5, 0.0, 4.0);
    assert_eq!(p.color, [1.0, 0.5, 0.0, 1.0]);
    assert_eq!((p.emission_strength, p.opacity), (4.0, 1.0));

    let mut m = Material::emissive("Spark", 1.0, 0.4, 0.1, 2.5).with_color(0.9, 0.8, 0.7, 0.6);
    m.opacity = 0.35;
    for p in [m.to_particle(), ParticleMaterial::from_material(&m)] {
        assert_eq!(p.color, m.base_color);
        assert_eq!(p.emission_strength, m.emission_strength);
        assert_eq!(p.opacity, m.opacity);
    }
}

#[test]
fn library_ids_follow_insertion_order() {
    let mut lib = MaterialLibrary::new();
    assert_eq!(lib.len(), 1, "a new library holds the default material");
    assert!(!lib.is_empty());
    assert_eq!(lib.default_material().name, "Default");
    assert!(
        MaterialLibrary::default().is_empty(),
        "derived Default is empty"
    );

    let names = ["A", "B", "A", "C"];
    for (k, name) in names.iter().enumerate() {
        let id = lib.add(Material::new(*name));
        assert_eq!(id as usize, k + 1);
    }
    assert_eq!(lib.len(), names.len() + 1);
    assert!(
        lib.get(names.len() as u32 + 1).is_none(),
        "one past the end"
    );
    assert!(lib.get(u32::MAX).is_none());

    let ids: Vec<u32> = lib.iter().map(|(id, _)| id).collect();
    assert_eq!(ids, (0..lib.len() as u32).collect::<Vec<_>>());
    for (id, m) in lib.iter() {
        assert_eq!(lib.get(id).map(|g| &g.name), Some(&m.name));
    }
    assert_eq!(
        lib.find_by_name("A").map(|(id, _)| id),
        Some(1),
        "first match"
    );
    assert_eq!(lib.find_by_name("C").map(|(id, _)| id), Some(4));
    assert!(lib.find_by_name("missing").is_none());
}

fn presets() -> Vec<(Material, bool)> {
    vec![
        (StandardMaterials::gold(), true),
        (StandardMaterials::aluminum(), true),
        (StandardMaterials::copper(), true),
        (StandardMaterials::chrome(), true),
        (StandardMaterials::plastic_white(), false),
        (StandardMaterials::plastic_red(), false),
        (StandardMaterials::glass(), false),
        (StandardMaterials::diamond(), false),
        (StandardMaterials::water(), false),
        (StandardMaterials::skin(), false),
        (StandardMaterials::marble(), false),
        (StandardMaterials::wet_asphalt(), false),
        (StandardMaterials::velvet(), false),
        (StandardMaterials::rubber(), false),
        (StandardMaterials::concrete(), false),
    ]
}

#[test]
fn presets_are_points_of_the_metallic_roughness_workflow() {
    let all = presets();
    let names: BTreeSet<_> = all.iter().map(|(m, _)| m.name.clone()).collect();
    assert_eq!(names.len(), all.len(), "preset names are unique");
    for (m, is_metal) in &all {
        let want = if *is_metal { 1.0 } else { 0.0 };
        assert_eq!(m.metallic, want, "{}: metallic", m.name);
        for (field, v) in [
            ("roughness", m.roughness),
            ("clearcoat", m.clearcoat),
            ("transmission", m.transmission),
            ("subsurface", m.subsurface),
        ] {
            assert!((0.0..=1.0).contains(&v), "{}: {field} = {v}", m.name);
        }
        assert!(
            m.base_color.iter().all(|c| (0.0..=1.0).contains(c)),
            "{}: base colour",
            m.name
        );
    }
    // Indices of refraction (sodium D line): crown glass 1.5, diamond 2.42,
    // water 1.33 — the values the `Material::ior` doc lists.
    let ior = |m: Material| (m.name.clone(), m.ior);
    assert_eq!(ior(StandardMaterials::glass()), ("Glass".into(), 1.5));
    assert_eq!(ior(StandardMaterials::diamond()), ("Diamond".into(), 2.42));
    assert_eq!(ior(StandardMaterials::water()), ("Water".into(), 1.33));
    for m in [
        StandardMaterials::glass(),
        StandardMaterials::diamond(),
        StandardMaterials::water(),
    ] {
        assert!(m.transmission > 0.5, "{} transmits", m.name);
    }
    // Each layered preset carries the layer that defines it.
    assert!(StandardMaterials::wet_asphalt().clearcoat > 0.0);
    assert!(StandardMaterials::velvet().sheen_roughness > 0.0);
    assert!(StandardMaterials::skin().subsurface > 0.0);
    assert!(StandardMaterials::marble().subsurface > 0.0);
    assert!(StandardMaterials::aluminum().anisotropy > 0.0, "brushed");
}

fn sample_pair() -> (Material, Material) {
    let a = Material::metal("A", 0.9, 0.2, 0.1, 0.1)
        .with_emission(1.0, 0.0, 0.0, 2.0)
        .with_clearcoat(0.5, 0.25)
        .with_sheen(0.1, 0.2, 0.3, 0.4)
        .with_anisotropy(-0.5, 0.125)
        .with_subsurface(0.25, 0.5, 0.75, 1.0)
        .with_albedo_map("a_albedo.png")
        .with_normal_map("a_normal.png");
    let b = Material::glass("B", 1.33)
        .with_color(0.1, 0.4, 0.8, 0.5)
        .with_transmission(0.75)
        .with_ao_map("b_ao.png")
        .with_emissive_map("b_emit.png");
    (a, b)
}

/// Scalar (and vector component) fields that `material_lerp` blends
/// linearly, as JSON paths.
fn blended(m: &Material) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let scal = [
        ("metallic", m.metallic),
        ("roughness", m.roughness),
        ("emission_strength", m.emission_strength),
        ("opacity", m.opacity),
        ("ior", m.ior),
        ("normal_scale", m.normal_scale),
        ("clearcoat", m.clearcoat),
        ("clearcoat_roughness", m.clearcoat_roughness),
        ("transmission", m.transmission),
        ("thickness", m.thickness),
        ("anisotropy", m.anisotropy),
        ("anisotropy_rotation", m.anisotropy_rotation),
        ("subsurface", m.subsurface),
        ("sheen_roughness", m.sheen_roughness),
    ];
    out.extend(scal.iter().map(|(k, v)| ((*k).to_string(), *v)));
    for (k, v) in m.base_color.iter().enumerate() {
        out.push((format!("base_color[{k}]"), *v));
    }
    for (name, arr) in [
        ("emission", m.emission),
        ("attenuation_color", m.attenuation_color),
        ("subsurface_color", m.subsurface_color),
        ("sheen_color", m.sheen_color),
    ] {
        for (k, v) in arr.iter().enumerate() {
            out.push((format!("{name}[{k}]"), *v));
        }
    }
    out
}

#[test]
fn lerp_end_points_are_the_inputs() {
    let (a, b) = sample_pair();
    let mut compared = 0usize;
    for (t, want) in [(0.0, &a), (1.0, &b), (-3.0, &a), (9.0, &b)] {
        let m = material_lerp(&a, &b, t);
        for ((k, got), (_, w)) in blended(&m).into_iter().zip(blended(want)) {
            assert_eq!(got.to_bits(), w.to_bits(), "t={t}: {k}");
            compared += 1;
        }
        // Texture slots come from the nearer end.
        for field in [
            "albedo_map",
            "normal_map",
            "ao_map",
            "emissive_map",
            "metallic_roughness_map",
        ] {
            let got = serde_json::to_value(&m).unwrap()[field].clone();
            let w = serde_json::to_value(want).unwrap()[field].clone();
            assert_eq!(got, w, "t={t}: {field}");
            compared += 1;
        }
    }
    assert!(compared > 100);
}

#[test]
fn lerp_interior_is_the_linear_closed_form() {
    let (a, b) = sample_pair();
    let mut compared = 0usize;
    for k in 1..16 {
        let t = k as f32 / 16.0;
        let m = material_lerp(&a, &b, t);
        for (((key, got), (_, va)), (_, vb)) in
            blended(&m).into_iter().zip(blended(&a)).zip(blended(&b))
        {
            let want = f64::from(va) + f64::from(t) * (f64::from(vb) - f64::from(va));
            let scale = f64::from(va.abs().max(vb.abs())).max(1.0);
            assert!(
                (f64::from(got) - want).abs() <= 2.0 * f64::from(f32::EPSILON) * scale,
                "t={t}: {key} = {got}, closed form {want}"
            );
            compared += 1;
        }
        let tex_from_b = t >= 0.5;
        assert_eq!(
            m.albedo_map.is_some(),
            !tex_from_b,
            "t={t}: albedo map only on a"
        );
        assert_eq!(m.ao_map.is_some(), tex_from_b, "t={t}: ao map only on b");
        assert_eq!(m.name, "A_B_blend");
    }
    assert!(compared > 300);
}

/// The JSON chunk of a GLB container (glTF 2.0 §4.4: 12-byte header, then
/// chunk length, chunk type `JSON`, payload). Parsed here from the format
/// definition, not through the crate's importer.
fn glb_json(glb: &[u8]) -> Value {
    assert_eq!(&glb[0..4], b"glTF", "GLB magic");
    assert_eq!(
        u32::from_le_bytes(glb[4..8].try_into().unwrap()),
        2,
        "version 2"
    );
    let len = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
    assert_eq!(&glb[16..20], b"JSON", "first chunk is JSON");
    serde_json::from_slice(&glb[20..20 + len]).expect("JSON chunk parses")
}

/// glTF 2.0 §5.36 `textureInfo.texCoord`: the TEXCOORD_n set the texture is
/// sampled with, default 0. A slot built with `with_uv_channel(1)` (the
/// lightmap UV the exporter writes as TEXCOORD_1) must say so, otherwise a
/// viewer samples it with TEXCOORD_0.
#[test]
fn gltf_texture_info_carries_the_uv_channel() {
    use alice_sdf::io::{export_glb_bytes, GltfConfig};
    use alice_sdf::prelude::{sdf_to_mesh, MarchingCubesConfig, SdfNode};
    use glam::Vec3;

    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 8,
            ..Default::default()
        },
    );
    let mut m = Material::metal("Lightmapped", 0.8, 0.8, 0.8, 0.3);
    m.albedo_map = Some(TextureSlot::new("albedo.png"));
    m.ao_map = Some(TextureSlot::new("baked_ao.png").with_uv_channel(1));
    m.emissive_map = Some(TextureSlot::new("emit.png").with_uv_channel(1));
    m.normal_map = Some(TextureSlot::new("normal.png").with_uv_channel(0));
    let mut lib = MaterialLibrary::new();
    let id = lib.add(m);

    let glb = export_glb_bytes(&mesh, &GltfConfig::default(), Some(&lib)).expect("export");
    let doc = glb_json(&glb);
    let mat = &doc["materials"][id as usize];
    assert_eq!(mat["name"], "Lightmapped");
    let tex_coord = |info: &Value| info.get("texCoord").map_or(0, |v| v.as_u64().unwrap());
    let mut compared = 0;
    for (info, want) in [
        (&mat["pbrMetallicRoughness"]["baseColorTexture"], 0),
        (&mat["normalTexture"], 0),
        (&mat["occlusionTexture"], 1),
        (&mat["emissiveTexture"], 1),
    ] {
        assert!(info.get("index").is_some(), "texture info present: {info}");
        assert_eq!(tex_coord(info), want, "{info}");
        compared += 1;
    }
    assert_eq!(compared, 4);
}

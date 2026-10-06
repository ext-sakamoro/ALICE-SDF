//! Mesh and volume file formats: write a part in each format, read it back
//! where the format has a reader, and check what came back.
//!
//! Run: `cargo run --example mesh_formats`
//! (`--features openvdb` adds the dense volume grid, `--features hlsl` the
//! Nanite material function)
//!
//! Author: Moroya Sakamoto

use alice_sdf::io::abm::{read_abm_header, save_abm};
use alice_sdf::io::fbx::{
    export_fbx, fbx_animation_to_timeline, import_fbx, import_fbx_full, FbxAnimClip, FbxAnimCurve,
    FbxConfig,
};
use alice_sdf::io::gltf::{export_gltf_json, GltfConfig};
use alice_sdf::io::iges::{export_iges, IgesConfig};
use alice_sdf::io::nanite::{
    export_nanite, export_nanite_json, export_nanite_with_config, NaniteExportConfig, NANITE_MAGIC,
    NANITE_VERSION,
};
use alice_sdf::io::obj::{export_obj, import_obj, ObjConfig};
use alice_sdf::io::ply::{export_ply, import_ply, PlyConfig};
use alice_sdf::io::splat::{load_splat, save_splat, Splat, SPLAT_BYTES};
use alice_sdf::io::stl::{export_stl_ascii, import_stl};
use alice_sdf::io::usd::{export_usda, import_usda, UsdConfig};
use alice_sdf::io::vox::{load_vox, save_vox, VoxModel, Voxel};
use alice_sdf::material::{Material, MaterialLibrary};
use alice_sdf::mesh::nanite::{generate_nanite_mesh, NaniteConfig};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh};
use alice_sdf::types::SdfNode;
use glam::Vec3;
use std::collections::HashMap;

fn positions(mesh: &Mesh) -> Vec<Vec3> {
    mesh.indices
        .iter()
        .map(|&i| mesh.vertices[i as usize].position)
        .collect()
}

fn same_positions(a: &Mesh, b: &Mesh, tol: f32) -> bool {
    let (pa, pb) = (positions(a), positions(b));
    pa.len() == pb.len()
        && pa
            .iter()
            .zip(&pb)
            .all(|(x, y)| (*x - *y).abs().max_element() <= tol)
}

fn main() {
    let dir = std::env::temp_dir().join(format!("alice_sdf_mesh_formats_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let part = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(1.2, 0.4, 1.2), 0.2);
    let mesh = sdf_to_mesh(
        &part,
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 16,
            ..Default::default()
        },
    );
    let tris = mesh.indices.len() / 3;
    println!("part: {} vertices, {tris} triangles", mesh.vertices.len());

    // ── text formats with a reader: exact round trips ──────
    let stl = dir.join("part.stl");
    export_stl_ascii(&mesh, &stl).unwrap();
    assert!(same_positions(&import_stl(&stl).unwrap(), &mesh, 0.0));
    let obj = dir.join("part.obj");
    export_obj(&mesh, &obj, &ObjConfig::default(), None).unwrap();
    assert!(same_positions(&import_obj(&obj).unwrap(), &mesh, 0.0));
    let ply = dir.join("part.ply");
    export_ply(&mesh, &ply, &PlyConfig::default()).unwrap();
    assert!(same_positions(&import_ply(&ply).unwrap(), &mesh, 0.0));
    let fbx = dir.join("part.fbx");
    export_fbx(&mesh, &fbx, &FbxConfig::default(), None).unwrap();
    assert!(same_positions(&import_fbx(&fbx).unwrap(), &mesh, 1e-6));
    let full = import_fbx_full(&fbx).unwrap();
    assert!(full.skeleton.is_none() && full.animations.is_empty());
    println!("STL ASCII / OBJ / PLY / FBX ASCII: {tris} triangles read back at the same positions");

    // binary FBX is written but not read: the reader says so by name
    let fbx_bin = dir.join("part_binary.fbx");
    export_fbx(&mesh, &fbx_bin, &FbxConfig::binary(), None).unwrap();
    let refused = import_fbx(&fbx_bin).unwrap_err();
    println!(
        "FBX binary: {} bytes written; import: {refused}",
        std::fs::metadata(&fbx_bin).unwrap().len()
    );

    // ── USD with a material ────────────────────────────────
    let lib = MaterialLibrary {
        materials: vec![Material {
            base_color: [0.75, 0.25, 0.5, 1.0],
            metallic: 0.5,
            roughness: 0.125,
            ..Default::default()
        }],
    };
    let usda = dir.join("part.usda");
    export_usda(&mesh, &usda, &UsdConfig::default(), Some(&lib)).unwrap();
    let imported = import_usda(&usda).unwrap();
    let mat = imported.material.expect("material");
    println!(
        "USDA: diffuse {:?}, metallic {}, roughness {}",
        mat.diffuse_color, mat.metallic, mat.roughness
    );
    assert!(same_positions(&imported.mesh, &mesh, 0.0));
    assert_eq!(mat.diffuse_color, [0.75, 0.25, 0.5]);

    // ── an FBX animation clip as a timeline ────────────────
    let mut curves = HashMap::new();
    curves.insert(
        "root".to_string(),
        vec![FbxAnimCurve {
            property: "Lcl Rotation".to_string(),
            channel: 1,
            times: vec![0.0, 1.0],
            values: vec![0.0, 90.0],
        }],
    );
    let clip = FbxAnimClip {
        name: "turn".to_string(),
        bone_curves: curves,
        duration: 1.0,
        fps: 24.0,
    };
    let timeline = fbx_animation_to_timeline(&clip, "root");
    let half = timeline.get_value("rotate.y", 0.5).unwrap();
    println!(
        "timeline: rotate.y at 0.5 s = {half:.6} rad (π/4 = {:.6})",
        std::f32::consts::FRAC_PI_4
    );
    assert!((half - std::f32::consts::FRAC_PI_4).abs() < 1e-6);

    // ── write-only formats ─────────────────────────────────
    let gltf = dir.join("part.gltf");
    export_gltf_json(&mesh, &gltf, &GltfConfig::aaa(), Some(&lib)).unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&gltf).unwrap()).unwrap();
    let uri = doc["buffers"][0]["uri"].as_str().unwrap_or("");
    println!(
        "glTF JSON: buffer of {} bytes as a data URI",
        doc["buffers"][0]["byteLength"]
    );
    assert!(uri.starts_with("data:application/octet-stream;base64,"));

    let iges = dir.join("part.igs");
    export_iges(&iges, &part, &IgesConfig::default()).unwrap();
    let iges_text = std::fs::read_to_string(&iges).unwrap();
    println!("IGES: {} records of 80 columns", iges_text.lines().count());
    assert!(iges_text.lines().all(|l| l.len() == 80));

    let nanite = generate_nanite_mesh(
        &part,
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &NaniteConfig {
            lod_levels: 2,
            base_resolution: 16,
            ..NaniteConfig::default()
        },
    );
    let nan = dir.join("part.nanite");
    export_nanite(&nanite, &nan).unwrap();
    let head = std::fs::read(&nan).unwrap();
    assert_eq!(&head[0..4], NANITE_MAGIC);
    assert_eq!(
        u32::from_le_bytes(head[4..8].try_into().unwrap()),
        NANITE_VERSION
    );
    let lean = dir.join("part_lean.nanite");
    export_nanite_with_config(
        &nanite,
        &lean,
        &NaniteExportConfig {
            export_uvs: false,
            ..NaniteExportConfig::default()
        },
    )
    .unwrap();
    export_nanite_json(&nanite, dir.join("part.nanite.json")).unwrap();
    println!(
        "Nanite: {} clusters, {} bytes (without UVs {} bytes)",
        nanite.clusters.len(),
        head.len(),
        std::fs::metadata(&lean).unwrap().len()
    );
    #[cfg(feature = "hlsl")]
    {
        let ush = dir.join("part_material.ush");
        alice_sdf::io::nanite::export_nanite_hlsl_material(&part, &ush).unwrap();
        println!(
            "Nanite material: {} lines of HLSL",
            std::fs::read_to_string(&ush).unwrap().lines().count()
        );
    }

    let abm = dir.join("part.abm");
    save_abm(&mesh, &abm).unwrap();
    let header = read_abm_header(&abm).unwrap();
    println!(
        "ABM header: {} vertices, {} indices",
        header.vertex_count, header.index_count
    );
    assert_eq!(header.vertex_count as usize, mesh.vertices.len());
    assert_eq!(header.index_count as usize, mesh.indices.len());

    // ── point and voxel formats ────────────────────────────
    let splats: Vec<Splat> = mesh
        .vertices
        .iter()
        .take(64)
        .map(|v| Splat {
            position: v.position.to_array(),
            scale: [0.05; 3],
            color: [200, 200, 220, 255],
            rotation: [255, 128, 128, 128],
        })
        .collect();
    let splat_path = dir.join("part.splat");
    save_splat(&splat_path, &splats).unwrap();
    let back = load_splat(&splat_path).unwrap();
    assert_eq!(
        std::fs::metadata(&splat_path).unwrap().len() as usize,
        splats.len() * SPLAT_BYTES
    );
    assert!(back
        .iter()
        .zip(&splats)
        .all(|(a, b)| a.position == b.position));
    println!("splat: {} records of {SPLAT_BYTES} bytes", back.len());

    let model = VoxModel {
        size: (3, 3, 3),
        voxels: (0..3)
            .map(|i| Voxel {
                x: i,
                y: i,
                z: i,
                color: 1 + i,
            })
            .collect(),
    };
    let vox = dir.join("diag.vox");
    save_vox(&vox, &model).unwrap();
    let vback = load_vox(&vox).unwrap();
    assert_eq!(vback.size, model.size);
    assert_eq!(vback.voxels.len(), 3);
    println!("vox: {:?} with {} voxels", vback.size, vback.voxels.len());

    #[cfg(feature = "openvdb")]
    {
        use alice_sdf::io::vdb::{bake_dense_grid, bake_to_vdb, load_dense_grid_from_vdb};
        let bytes = bake_to_vdb(&part, (-1.5, 1.5), 16).unwrap();
        let (grid, res, bounds) = load_dense_grid_from_vdb(&bytes).unwrap();
        assert_eq!(grid, bake_dense_grid(&part, (-1.5, 1.5), 16));
        println!("dense grid: {res}³ over {bounds:?}, {} bytes", bytes.len());
    }

    std::fs::remove_dir_all(&dir).ok();
    println!("mesh_formats: all checks passed");
}

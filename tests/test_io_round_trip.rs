//! Integration tests: I/O round-trip consistency
//!
//! Verifies ASDF binary/JSON save-load, ABM mesh persistence, OBJ/STL/GLB/PLY/FBX/USDA
//! export-import round-trips.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::io;
use alice_sdf::prelude::*;
use common::*;
use std::path::PathBuf;

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("alice_sdf_test_io");
    std::fs::create_dir_all(&dir).ok();
    dir
}

// ============================================================================
// ASDF binary round-trip
// ============================================================================

#[test]
fn asdf_binary_round_trip_sphere() {
    let tree = SdfTree::new(test_sphere());
    let path = temp_dir().join("sphere.asdf");

    save_asdf(&tree, &path).expect("save_asdf failed");
    let loaded = load_asdf(&path).expect("load_asdf failed");

    // Verify loaded tree evaluates identically
    for p in test_points() {
        let d_original = eval(&tree.root, p);
        let d_loaded = eval(&loaded.root, p);
        assert_close(
            d_original,
            d_loaded,
            1e-6,
            &format!("ASDF binary mismatch at {:?}", p),
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn asdf_binary_round_trip_complex() {
    let tree = SdfTree::new(test_complex_shape());
    let path = temp_dir().join("complex.asdf");

    save_asdf(&tree, &path).expect("save_asdf failed");
    let loaded = load_asdf(&path).expect("load_asdf failed");

    for p in test_points() {
        let d_original = eval(&tree.root, p);
        let d_loaded = eval(&loaded.root, p);
        assert_close(
            d_original,
            d_loaded,
            1e-5,
            &format!("ASDF complex mismatch at {:?}", p),
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn asdf_binary_preserves_metadata() {
    let mut metadata = SdfMetadata::default();
    metadata.name = Some("test_model".to_string());
    metadata.author = Some("Moroya Sakamoto".to_string());
    metadata.description = Some("Integration test shape".to_string());

    let tree = SdfTree::with_metadata(test_sphere(), metadata);
    let path = temp_dir().join("metadata.asdf");

    save_asdf(&tree, &path).expect("save_asdf failed");
    let loaded = load_asdf(&path).expect("load_asdf failed");

    let meta = loaded.metadata.expect("Metadata should be preserved");
    assert_eq!(meta.name.as_deref(), Some("test_model"));
    assert_eq!(meta.author.as_deref(), Some("Moroya Sakamoto"));

    std::fs::remove_file(&path).ok();
}

// ============================================================================
// ASDF JSON round-trip
// ============================================================================

#[test]
fn asdf_json_round_trip() {
    let tree = SdfTree::new(test_csg());
    let path = temp_dir().join("csg.asdf.json");

    save_asdf_json(&tree, &path).expect("save_asdf_json failed");
    let loaded = load_asdf_json(&path).expect("load_asdf_json failed");

    for p in test_points() {
        let d_original = eval(&tree.root, p);
        let d_loaded = eval(&loaded.root, p);
        assert_close(
            d_original,
            d_loaded,
            1e-6,
            &format!("JSON mismatch at {:?}", p),
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn json_string_round_trip() {
    let tree = SdfTree::new(test_sphere());
    let json = alice_sdf::io::to_json_string(&tree).expect("to_json_string failed");
    let loaded = alice_sdf::io::from_json_string(&json).expect("from_json_string failed");

    let d_original = eval(&tree.root, Vec3::new(2.0, 0.0, 0.0));
    let d_loaded = eval(&loaded.root, Vec3::new(2.0, 0.0, 0.0));
    assert_close(d_original, d_loaded, 1e-6, "JSON string round-trip");
}

// ============================================================================
// ABM mesh round-trip
// ============================================================================

#[test]
fn abm_mesh_round_trip() {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 16,
        compute_normals: true,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config);
    let path = temp_dir().join("sphere.abm");

    alice_sdf::io::save_abm(&mesh, &path).expect("save_abm failed");
    let loaded = alice_sdf::io::load_abm(&path).expect("load_abm failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.vertices.len(),
        "Vertex count mismatch"
    );
    assert_eq!(
        mesh.indices.len(),
        loaded.indices.len(),
        "Index count mismatch"
    );

    // Check first few vertices match
    for i in 0..mesh.vertices.len().min(10) {
        let orig = &mesh.vertices[i];
        let load = &loaded.vertices[i];
        assert_close(
            orig.position.x,
            load.position.x,
            1e-5,
            &format!("Vertex {} position X", i),
        );
        assert_close(
            orig.position.y,
            load.position.y,
            1e-5,
            &format!("Vertex {} position Y", i),
        );
        assert_close(
            orig.position.z,
            load.position.z,
            1e-5,
            &format!("Vertex {} position Z", i),
        );
    }

    std::fs::remove_file(&path).ok();
}

#[test]
fn abm_with_lods_round_trip() {
    let shape = test_sphere();
    let min = Vec3::splat(-1.5);
    let max = Vec3::splat(1.5);

    let lod_config = DecimationLodConfig {
        num_levels: 3,
        base_resolution: 32,
        decimation_ratio: 0.5,
        distance_multiplier: 2.0,
        base_distance: 5.0,
        compute_normals: true,
        preserve_materials: false,
    };
    let lod_chain = generate_lod_chain_decimated(&shape, min, max, &lod_config);

    // Extract meshes and transition distances (N-1 distances for N meshes)
    let meshes: Vec<Mesh> = lod_chain.levels.iter().map(|l| l.mesh.clone()).collect();
    let distances: Vec<f32> = lod_chain
        .levels
        .windows(2)
        .map(|w| w[0].max_distance)
        .collect();

    let path = temp_dir().join("sphere_lods.abm");
    alice_sdf::io::save_abm_with_lods(&meshes, &distances, &path)
        .expect("save_abm_with_lods failed");
    let (loaded_meshes, loaded_distances) =
        alice_sdf::io::load_abm_with_lods(&path).expect("load_abm_with_lods failed");

    assert_eq!(meshes.len(), loaded_meshes.len(), "Mesh count mismatch");
    assert_eq!(
        distances.len(),
        loaded_distances.len(),
        "Distance count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

// ============================================================================
// OBJ export validation
// ============================================================================

#[test]
fn obj_export_creates_valid_file() {
    let shape = test_box();
    let config = MarchingCubesConfig {
        resolution: 16,
        ..Default::default()
    };
    let mesh = sdf_to_mesh(&shape, Vec3::splat(-1.0), Vec3::splat(1.0), &config);
    let path = temp_dir().join("box.obj");

    let obj_config = ObjConfig::default();
    export_obj(&mesh, &path, &obj_config, None).expect("export_obj failed");

    let content = std::fs::read_to_string(&path).expect("read OBJ failed");
    assert!(content.contains("v "), "OBJ should contain vertex data");
    assert!(content.contains("f "), "OBJ should contain face data");

    std::fs::remove_file(&path).ok();
}

// ============================================================================
// Mesh format round-trip tests
// ============================================================================

/// Generate a sphere mesh for round-trip testing
fn test_mesh_sphere() -> Mesh {
    let shape = test_sphere();
    let config = MarchingCubesConfig {
        resolution: 16,
        compute_normals: true,
        ..Default::default()
    };
    sdf_to_mesh(&shape, Vec3::splat(-1.5), Vec3::splat(1.5), &config)
}

#[test]
fn stl_binary_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.stl");

    io::export_stl(&mesh, &path).expect("export_stl failed");
    let loaded = io::import_stl(&path).expect("import_stl failed");

    assert_eq!(
        mesh.indices.len(),
        loaded.indices.len(),
        "STL binary: index count mismatch"
    );
    // STL stores per-face vertices, so vertex count may differ (de-duplicated on import)
    assert!(
        loaded.vertices.len() > 0,
        "STL binary: loaded mesh should have vertices"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn stl_ascii_export_valid() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere_ascii.stl");

    io::export_stl_ascii(&mesh, &path).expect("export_stl_ascii failed");

    let content = std::fs::read_to_string(&path).expect("read STL ASCII failed");
    assert!(
        content.starts_with("solid"),
        "STL ASCII should start with 'solid'"
    );
    assert!(
        content.contains("endsolid"),
        "STL ASCII should contain 'endsolid'"
    );
    assert!(
        content.contains("facet normal"),
        "STL ASCII should contain facet normals"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn glb_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.glb");

    let config = GltfConfig::default();
    export_glb(&mesh, &path, &config, None).expect("export_glb failed");
    let loaded = import_glb(&path).expect("import_glb failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.vertices.len(),
        "GLB: vertex count mismatch"
    );
    assert_eq!(
        mesh.indices.len(),
        loaded.indices.len(),
        "GLB: index count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn ply_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.ply");

    let config = io::PlyConfig {
        binary: true,
        export_normals: true,
    };
    io::export_ply(&mesh, &path, &config).expect("export_ply failed");
    let loaded = io::import_ply(&path).expect("import_ply failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.vertices.len(),
        "PLY: vertex count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn fbx_ascii_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.fbx");

    let config = FbxConfig {
        format: FbxFormat::Ascii,
        ..Default::default()
    };
    export_fbx(&mesh, &path, &config, None).expect("export_fbx failed");
    let loaded = import_fbx(&path).expect("import_fbx failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.vertices.len(),
        "FBX ASCII: vertex count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn usda_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.usda");

    let config = UsdConfig::default();
    export_usda(&mesh, &path, &config, None).expect("export_usda failed");
    let loaded = io::import_usda(&path).expect("import_usda failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.mesh.vertices.len(),
        "USDA: vertex count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

#[test]
fn obj_round_trip() {
    let mesh = test_mesh_sphere();
    let path = temp_dir().join("rt_sphere.obj");

    let config = ObjConfig::default();
    export_obj(&mesh, &path, &config, None).expect("export_obj failed");
    let loaded = import_obj(&path).expect("import_obj failed");

    assert_eq!(
        mesh.vertices.len(),
        loaded.vertices.len(),
        "OBJ: vertex count mismatch"
    );
    assert_eq!(
        mesh.indices.len(),
        loaded.indices.len(),
        "OBJ: index count mismatch"
    );

    std::fs::remove_file(&path).ok();
}

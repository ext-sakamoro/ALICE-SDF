//! Provenance (ALICE-Law G8): every durable writer stores the semantics identifier, and it reads back
//!
//! For each writer in [`WRITERS`]: the plain writer stores [`Provenance::sdf`], the
//! `*_with_provenance` writer stores the caller's identifier, `provenance::read` returns it,
//! `provenance::check` refuses another identifier and names both, and the file still opens in
//! this crate's own importer (and, for binary STL, in an independent STL parser).
//! `scripts/provenance_writers.py` checks that every public writer appears in [`WRITERS`] or in
//! its list of writers still to do.

use alice_sdf::io;
use alice_sdf::io::meshopt_gltf as meshopt;
use alice_sdf::mesh::lod_persist::{self, LodChainConfig, LodChainPersist};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{self, Provenance, ProvenanceError, StoredSemantics};
use std::path::{Path, PathBuf};

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("alice_sdf_provenance_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn mesh() -> Mesh {
    let config = MarchingCubesConfig {
        resolution: 12,
        compute_normals: true,
        ..Default::default()
    };
    sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &config,
    )
}

fn caller_id() -> Provenance {
    let mut id = [0u8; 32];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::try_from(i).unwrap().wrapping_mul(29).wrapping_add(7);
    }
    Provenance::new(id)
}

type Write = fn(&Mesh, &Path, Option<&Provenance>);

/// (writer name as the scanner reads it, file name, write with the default / with the caller's id)
const WRITERS: &[(&str, &str, Write)] = &[
    ("src/io/stl.rs::export_stl", "m.stl", |m, p, id| match id {
        None => io::export_stl(m, p).unwrap(),
        Some(id) => io::export_stl_with_provenance(m, p, id).unwrap(),
    }),
    (
        "src/io/stl.rs::export_stl_ascii",
        "m_ascii.stl",
        |m, p, id| match id {
            None => io::export_stl_ascii(m, p).unwrap(),
            Some(id) => io::export_stl_ascii_with_provenance(m, p, id).unwrap(),
        },
    ),
    (
        "src/io/threemf.rs::export_3mf",
        "m.3mf",
        |m, p, id| match id {
            None => io::export_3mf(m, p).unwrap(),
            Some(id) => io::export_3mf_with_provenance(m, p, id).unwrap(),
        },
    ),
    ("src/io/fbx.rs::export_fbx", "m_bin.fbx", |m, p, id| {
        let c = io::FbxConfig {
            format: io::FbxFormat::Binary,
            ..Default::default()
        };
        match id {
            None => io::export_fbx(m, p, &c, None).unwrap(),
            Some(id) => io::export_fbx_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    ("src/io/fbx.rs::export_fbx", "m_ascii.fbx", |m, p, id| {
        let c = io::FbxConfig {
            format: io::FbxFormat::Ascii,
            ..Default::default()
        };
        match id {
            None => io::export_fbx(m, p, &c, None).unwrap(),
            Some(id) => io::export_fbx_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    ("src/io/obj.rs::export_obj", "m.obj", |m, p, id| {
        let c = io::ObjConfig::default();
        match id {
            None => io::export_obj(m, p, &c, None).unwrap(),
            Some(id) => io::export_obj_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    ("src/io/ply.rs::export_ply", "m_ascii.ply", |m, p, id| {
        let c = io::PlyConfig {
            binary: false,
            ..Default::default()
        };
        match id {
            None => io::export_ply(m, p, &c).unwrap(),
            Some(id) => io::export_ply_with_provenance(m, p, &c, id).unwrap(),
        }
    }),
    ("src/io/ply.rs::export_ply", "m_bin.ply", |m, p, id| {
        let c = io::PlyConfig {
            binary: true,
            ..Default::default()
        };
        match id {
            None => io::export_ply(m, p, &c).unwrap(),
            Some(id) => io::export_ply_with_provenance(m, p, &c, id).unwrap(),
        }
    }),
    ("src/io/gltf.rs::export_glb", "m.glb", |m, p, id| {
        let c = io::GltfConfig::default();
        match id {
            None => io::export_glb(m, p, &c, None).unwrap(),
            Some(id) => io::export_glb_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    (
        "src/io/gltf.rs::export_glb",
        "m_meshopt_via_glb.glb",
        |m, p, id| {
            let c = io::GltfConfig {
                meshopt_compress: true,
                ..Default::default()
            };
            match id {
                None => io::export_glb(m, p, &c, None).unwrap(),
                Some(id) => io::export_glb_with_provenance(m, p, &c, None, id).unwrap(),
            }
        },
    ),
    (
        "src/io/gltf.rs::export_glb_bytes",
        "m_bytes.glb",
        |m, p, id| {
            let c = io::GltfConfig::default();
            let bytes = match id {
                None => io::export_glb_bytes(m, &c, None).unwrap(),
                Some(id) => io::export_glb_bytes_with_provenance(m, &c, None, id).unwrap(),
            };
            std::fs::write(p, bytes).unwrap();
        },
    ),
    ("src/io/gltf.rs::export_gltf_json", "m.gltf", |m, p, id| {
        let c = io::GltfConfig::default();
        match id {
            None => io::export_gltf_json(m, p, &c, None).unwrap(),
            Some(id) => io::export_gltf_json_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    (
        "src/io/meshopt_gltf.rs::export_glb_meshopt",
        "mo.glb",
        |m, p, id| {
            let c = meshopt::MeshoptGltfConfig::default();
            match id {
                None => meshopt::export_glb_meshopt(m, p, &c).unwrap(),
                Some(id) => std::fs::write(
                    p,
                    meshopt::export_glb_meshopt_bytes_skinned_with_provenance(m, None, &c, id)
                        .unwrap(),
                )
                .unwrap(),
            }
        },
    ),
    (
        "src/io/meshopt_gltf.rs::export_glb_meshopt_skinned",
        "mo_sk.glb",
        |m, p, id| {
            let c = meshopt::MeshoptGltfConfig::default();
            match id {
                None => meshopt::export_glb_meshopt_skinned(m, None, p, &c).unwrap(),
                Some(id) => std::fs::write(
                    p,
                    meshopt::export_glb_meshopt_bytes_skinned_with_provenance(m, None, &c, id)
                        .unwrap(),
                )
                .unwrap(),
            }
        },
    ),
    (
        "src/io/meshopt_gltf.rs::export_glb_meshopt_bytes",
        "mo_b.glb",
        |m, p, id| {
            let c = meshopt::MeshoptGltfConfig::default();
            let bytes = match id {
                None => meshopt::export_glb_meshopt_bytes(m, &c).unwrap(),
                Some(id) => {
                    meshopt::export_glb_meshopt_bytes_skinned_with_provenance(m, None, &c, id)
                        .unwrap()
                }
            };
            std::fs::write(p, bytes).unwrap();
        },
    ),
    (
        "src/io/meshopt_gltf.rs::export_glb_meshopt_bytes_skinned",
        "mo_bs.glb",
        |m, p, id| {
            let c = meshopt::MeshoptGltfConfig::default();
            let bytes = match id {
                None => meshopt::export_glb_meshopt_bytes_skinned(m, None, &c).unwrap(),
                Some(id) => {
                    meshopt::export_glb_meshopt_bytes_skinned_with_provenance(m, None, &c, id)
                        .unwrap()
                }
            };
            std::fs::write(p, bytes).unwrap();
        },
    ),
    ("src/io/usd.rs::export_usda", "m.usda", |m, p, id| {
        let c = io::UsdConfig::default();
        match id {
            None => io::export_usda(m, p, &c, None).unwrap(),
            Some(id) => io::export_usda_with_provenance(m, p, &c, None, id).unwrap(),
        }
    }),
    ("src/io/step.rs::export_step", "m.step", |_, p, id| {
        let c = io::step::StepConfig::default();
        let node = SdfNode::sphere(1.0);
        match id {
            None => io::step::export_step(p, &node, &c).unwrap(),
            Some(id) => io::step::export_step_with_provenance(p, &node, &c, id).unwrap(),
        }
    }),
    (
        "src/validity.rs::export_step_validated",
        "v.step",
        |_, p, id| {
            let c = io::step::StepConfig {
                bounds: (-6.0, 6.0),
                resolution: 24,
                name: "ThickBlock".into(),
            };
            let node = SdfNode::box3d(8.0, 4.0, 8.0);
            let req = alice_sdf::validity::PrintRequirements {
                max_overhang: std::f32::consts::FRAC_PI_2,
                ..alice_sdf::validity::PrintRequirements::fdm_0_4_nozzle()
            };
            match id {
                None => {
                    alice_sdf::validity::export_step_validated(p, &node, &c, req).unwrap();
                }
                Some(id) => {
                    alice_sdf::validity::export_step_validated_with_provenance(
                        p, &node, &c, req, id,
                    )
                    .unwrap();
                }
            }
        },
    ),
    ("src/io/iges.rs::export_iges", "m.igs", |_, p, id| {
        let c = io::iges::IgesConfig::default();
        let node = SdfNode::sphere(1.0);
        match id {
            None => io::iges::export_iges(p, &node, &c).unwrap(),
            Some(id) => io::iges::export_iges_with_provenance(p, &node, &c, id).unwrap(),
        }
    }),
    ("src/io/asdf.rs::save_asdf", "t.asdf", |_, p, id| {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        match id {
            None => io::save_asdf(&tree, p).unwrap(),
            Some(id) => io::save_asdf_with_provenance(&tree, p, id).unwrap(),
        }
    }),
    (
        "src/io/json.rs::save_asdf_json",
        "t.asdf.json",
        |_, p, id| {
            let tree = SdfTree::new(SdfNode::sphere(1.0));
            match id {
                None => io::save_asdf_json(&tree, p).unwrap(),
                Some(id) => io::save_asdf_json_with_provenance(&tree, p, id).unwrap(),
            }
        },
    ),
    ("src/io/asdf.rs::to_bytes", "t_header.asdf", |_, p, id| {
        let tree = SdfTree::new(SdfNode::sphere(1.0));
        let header = match id {
            None => io::AsdfHeader::new(&tree, 0),
            Some(id) => io::AsdfHeader::with_provenance(&tree, 0, id),
        };
        std::fs::write(p, header.to_bytes()).unwrap();
    }),
    ("src/io/abm.rs::save_abm", "m.abm", |m, p, id| match id {
        None => io::save_abm(m, p).unwrap(),
        Some(id) => io::save_abm_with_provenance(m, p, id).unwrap(),
    }),
    (
        "src/io/abm.rs::save_abm_with_lods",
        "m_lods.abm",
        |m, p, id| {
            let lods = [m.clone(), m.clone()];
            match id {
                None => io::save_abm_with_lods(&lods, &[10.0], p).unwrap(),
                Some(id) => io::save_abm_with_lods_with_provenance(&lods, &[10.0], p, id).unwrap(),
            }
        },
    ),
    (
        "src/mesh/lod_persist.rs::save_lod_chain",
        "m_chain.abm",
        |m, p, id| {
            let chain = lod_chain(m);
            match id {
                None => lod_persist::save_lod_chain(&chain, p).unwrap(),
                Some(id) => lod_persist::save_lod_chain_with_provenance(&chain, p, id).unwrap(),
            }
        },
    ),
    // the sidecar alone: written next to `<base>`, read from `<base>.meta.json` (the row's file)
    (
        "src/mesh/lod_persist.rs::save_lod_chain_metadata",
        "m_meta.abm.meta.json",
        |m, p, id| {
            let base = p.with_file_name("m_meta.abm");
            let chain = lod_chain(m);
            match id {
                None => lod_persist::save_lod_chain_metadata(&chain, &base).unwrap(),
                Some(id) => {
                    lod_persist::save_lod_chain_metadata_with_provenance(&chain, &base, id)
                        .unwrap();
                }
            }
        },
    ),
    ("src/io/abm.rs::to_bytes", "m_header.abm", |m, p, id| {
        let header = io::AbmHeader {
            magic: io::ABM_MAGIC,
            version: io::ABM_VERSION,
            flags: 0,
            vertex_count: u32::try_from(m.vertices.len()).unwrap(),
            index_count: u32::try_from(m.indices.len()).unwrap(),
            lod_count: 0,
            _reserved: [0; 7],
            crc32: 0,
            semantics_id: Some(id.map_or(alice_sdf::SDF_SEMANTICS_ID, |id| id.semantics_id)),
        };
        std::fs::write(p, header.to_bytes()).unwrap();
    }),
];

#[test]
fn every_writer_stores_the_identifier_and_it_reads_back() {
    let m = mesh();
    assert!(!m.indices.is_empty());
    let mut compared = 0;
    for &(name, file, write) in WRITERS {
        let path = temp(file);
        write(&m, &path, None);
        assert_eq!(
            provenance::read(&path).unwrap_or_else(|e| panic!("{name}: {e}")),
            Provenance::sdf(),
            "{name}: the plain writer stores this crate's identifier"
        );
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            provenance::StoredSemantics::Current
        );

        write(&m, &path, Some(&caller_id()));
        assert_eq!(
            provenance::read(&path).unwrap(),
            caller_id(),
            "{name}: the caller's identifier"
        );
        assert_eq!(
            provenance::check(&path, &Provenance::sdf(), &[caller_id().semantics_id]).unwrap(),
            provenance::StoredSemantics::VerifiedPrevious(caller_id()),
            "{name}: an earlier identifier the caller lists is semantics and named"
        );
        match provenance::check_sdf(&path) {
            Err(ProvenanceError::Mismatch { stored, expected }) => {
                assert_eq!(stored, caller_id());
                assert_eq!(expected, Provenance::sdf());
                let text = ProvenanceError::Mismatch { stored, expected }.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex()),
                    "{name}: the mismatch names both identifiers: {text}"
                );
            }
            other => panic!("{name}: expected a mismatch, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, WRITERS.len());
    assert!(compared > 0, "compared no writer");
}

#[test]
fn files_with_the_identifier_still_open_in_this_crates_importers() {
    let m = mesh();
    let tris = m.indices.len() / 3;
    for (file, ascii) in [("i.stl", false), ("i_ascii.stl", true)] {
        let path = temp(file);
        if ascii {
            io::export_stl_ascii_with_provenance(&m, &path, &Provenance::sdf()).unwrap();
        } else {
            io::export_stl_with_provenance(&m, &path, &Provenance::sdf()).unwrap();
        }
        let back = io::import_stl_pinned(&path).unwrap();
        assert_eq!(back.indices.len() / 3, tris, "{file}");
        std::fs::remove_file(&path).ok();
    }
    let path = temp("i.obj");
    io::export_obj_with_provenance(
        &m,
        &path,
        &io::ObjConfig::default(),
        None,
        &Provenance::sdf(),
    )
    .unwrap();
    assert_eq!(
        io::import_obj_pinned(&path).unwrap().indices.len() / 3,
        tris,
        "obj"
    );
    std::fs::remove_file(&path).ok();
    for binary in [false, true] {
        let path = temp(&format!("i_{binary}.ply"));
        let c = io::PlyConfig {
            binary,
            ..Default::default()
        };
        io::export_ply_with_provenance(&m, &path, &c, &Provenance::sdf()).unwrap();
        assert_eq!(
            io::import_ply_pinned(&path).unwrap().indices.len() / 3,
            tris,
            "ply binary={binary}"
        );
        std::fs::remove_file(&path).ok();
    }
    // this crate imports ASCII FBX only
    let path = temp("i_ascii.fbx");
    let c = io::FbxConfig {
        format: io::FbxFormat::Ascii,
        ..Default::default()
    };
    io::export_fbx_with_provenance(&m, &path, &c, None, &Provenance::sdf()).unwrap();
    let back = io::import_fbx_pinned(&path).unwrap();
    assert_eq!(back.indices.len() / 3, tris);
    std::fs::remove_file(&path).ok();
}

/// The OBJ writer also writes the material file (`.mtl`) next to it; both carry the identifier
#[test]
fn the_obj_material_file_carries_the_identifier_too() {
    let m = mesh();
    let path = temp("with_materials.obj");
    let mut lib = alice_sdf::material::MaterialLibrary::new();
    lib.add(alice_sdf::material::Material::new("red"));
    io::export_obj_with_provenance(
        &m,
        &path,
        &io::ObjConfig::default(),
        Some(&lib),
        &caller_id(),
    )
    .unwrap();
    let mtl = path.with_extension("mtl");
    assert_eq!(provenance::read(&path).unwrap(), caller_id(), "obj");
    assert_eq!(provenance::read(&mtl).unwrap(), caller_id(), "mtl");
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&mtl).ok();
}

/// An independent glTF parser (`gltf`) reads the `.glb` and `.gltf` carrying the identifier in
/// `asset.extras`, and this crate's importer reads the `.glb` back; the meshopt `.glb` has a JSON
/// chunk that parses as JSON with the identifier under `asset.extras`
#[test]
fn gltf_files_with_the_identifier_stay_valid() {
    let m = mesh();
    let tris = m.indices.len() / 3;
    let c = io::GltfConfig::default();
    let glb = io::export_glb_bytes_with_provenance(&m, &c, None, &Provenance::sdf()).unwrap();
    gltf::Gltf::from_slice(&glb).expect("gltf reads the .glb");
    assert_eq!(
        io::import_glb_bytes_pinned(&glb).unwrap().indices.len() / 3,
        tris
    );
    let path = temp("strict.gltf");
    io::export_gltf_json_with_provenance(&m, &path, &c, None, &caller_id()).unwrap();
    gltf::Gltf::from_slice(&std::fs::read(&path).unwrap()).expect("gltf reads the .gltf");
    std::fs::remove_file(&path).ok();

    let mo = meshopt::export_glb_meshopt_bytes_skinned_with_provenance(
        &m,
        None,
        &meshopt::MeshoptGltfConfig::default(),
        &caller_id(),
    )
    .unwrap();
    let len = u32::from_le_bytes([mo[12], mo[13], mo[14], mo[15]]) as usize;
    let json: serde_json::Value = serde_json::from_slice(&mo[20..20 + len]).unwrap();
    assert_eq!(
        json["asset"]["extras"][provenance::KEY].as_str(),
        Some(caller_id().hex().as_str())
    );
}

/// USD reads back in this crate's importer; STEP keeps its ISO header and trailer; every IGES line
/// stays 80 columns with the Start-section count in the Terminate record matching the Start lines
#[test]
fn usd_step_iges_with_the_identifier_stay_well_formed() {
    let m = mesh();
    let path = temp("wf.usda");
    io::export_usda_with_provenance(
        &m,
        &path,
        &io::UsdConfig::default(),
        None,
        &Provenance::sdf(),
    )
    .unwrap();
    assert!(!io::import_usda_pinned(&path)
        .unwrap()
        .mesh
        .indices
        .is_empty());
    std::fs::remove_file(&path).ok();

    let node = SdfNode::sphere(1.0);
    let path = temp("wf.step");
    io::step::export_step_with_provenance(
        &path,
        &node,
        &io::step::StepConfig::default(),
        &caller_id(),
    )
    .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("ISO-10303-21;\nHEADER;\n"));
    assert!(text.trim_end().ends_with("END-ISO-10303-21;"));
    std::fs::remove_file(&path).ok();

    let path = temp("wf.igs");
    io::iges::export_iges_with_provenance(
        &path,
        &node,
        &io::iges::IgesConfig::default(),
        &caller_id(),
    )
    .unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines.iter().all(|l| l.len() == 80),
        "every IGES line is 80 columns"
    );
    let starts = lines.iter().filter(|l| &l[72..73] == "S").count();
    assert_eq!(starts, 3, "the banner and the two identifier lines");
    let term = lines.last().unwrap();
    assert!(
        term.starts_with(&format!("S{starts:>7}")),
        "Terminate counts the Start lines: {term}"
    );
    std::fs::remove_file(&path).ok();
}

/// A file stamped with an identifier this crate published before (6.1's, `alice-det-math` 0.5.1's)
/// reads as a verified earlier identifier, the same verdict as `check_sdf_semantics`
#[test]
fn an_earlier_published_identifier_reads_as_verified_previous() {
    let m = mesh();
    let earlier = alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0];
    assert_ne!(earlier, alice_sdf::SDF_SEMANTICS_ID);
    let path = temp("earlier.stl");
    io::export_stl_with_provenance(&m, &path, &Provenance::new(earlier)).unwrap();
    assert_eq!(
        provenance::check_sdf(&path).unwrap(),
        provenance::StoredSemantics::VerifiedPrevious(Provenance::new(earlier))
    );
    assert!(matches!(
        alice_sdf::check_sdf_semantics(&earlier),
        alice_det_math::SemanticsCheck::VerifiedPrevious(_)
    ));
    assert_eq!(Provenance::sdf().semantics_id, alice_sdf::SDF_SEMANTICS_ID);
    std::fs::remove_file(&path).ok();
}

/// Volume exports (feature `volume`): DDS carries the identifier in the header's reserved dwords;
/// raw data has no header, so it is in the sidecar `<path>.semantics`, and the raw bytes stay the
/// voxel data only
#[cfg(feature = "volume")]
#[test]
fn volume_writers_store_the_identifier() {
    use alice_sdf::volume::export as vx;
    use alice_sdf::volume::{Volume3D, VoxelDistGrad};
    type VolWrite = fn(&str, Option<&Provenance>);
    const VOLUME_WRITERS: &[(&str, &str, VolWrite)] = &[
        ("src/volume/export.rs::export_raw", "v.raw", |p, id| {
            let v = Volume3D::<f32>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));
            match id {
                None => vx::export_raw(&v, p).unwrap(),
                Some(id) => vx::export_raw_with_provenance(&v, p, id).unwrap(),
            }
        }),
        (
            "src/volume/export.rs::export_raw_with_mips",
            "vm.raw",
            |p, id| {
                let v = Volume3D::<f32>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));
                match id {
                    None => vx::export_raw_with_mips(&v, p).unwrap(),
                    Some(id) => vx::export_raw_with_mips_with_provenance(&v, p, id).unwrap(),
                }
            },
        ),
        ("src/volume/export.rs::export_dds_3d", "v.dds", |p, id| {
            let v = Volume3D::<f32>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));
            match id {
                None => vx::export_dds_3d(&v, p, vx::DdsFormat::R32Float).unwrap(),
                Some(id) => {
                    vx::export_dds_3d_with_provenance(&v, p, vx::DdsFormat::R32Float, id).unwrap();
                }
            }
        }),
        (
            "src/volume/export.rs::export_dds_3d_distgrad",
            "vg.dds",
            |p, id| {
                let v =
                    Volume3D::<VoxelDistGrad>::new([4, 4, 4], Vec3::splat(-1.0), Vec3::splat(1.0));
                match id {
                    None => vx::export_dds_3d_distgrad(&v, p).unwrap(),
                    Some(id) => vx::export_dds_3d_distgrad_with_provenance(&v, p, id).unwrap(),
                }
            },
        ),
    ];
    let mut compared = 0;
    for &(name, file, write) in VOLUME_WRITERS {
        let path = temp(file);
        let p = path.to_str().unwrap();
        write(p, None);
        assert_eq!(
            provenance::read(&path).unwrap(),
            Provenance::sdf(),
            "{name}"
        );
        write(p, Some(&caller_id()));
        assert_eq!(provenance::read(&path).unwrap(), caller_id(), "{name}");
        assert!(matches!(
            provenance::check_sdf(&path),
            Err(ProvenanceError::Mismatch { .. })
        ));
        let bytes = std::fs::read(&path).unwrap();
        if file.ends_with(".dds") {
            assert_eq!(&bytes[..4], b"DDS ", "{name}: magic");
            assert_eq!(
                u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
                124
            );
            assert_eq!(
                &bytes[84..88],
                b"DX10",
                "{name}: pixel format FourCC still DX10"
            );
        } else {
            assert_eq!(bytes.len() % 4, 0, "{name}: raw stays f32 voxels");
            assert!(
                !bytes.windows(8).any(|w| w == b"semantic"),
                "{name}: no tag in the raw bytes"
            );
            std::fs::remove_file(provenance::sidecar_path(&path)).ok();
            std::fs::remove_file(path.with_extension("raw.meta.json")).ok();
        }
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, VOLUME_WRITERS.len());
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/provenance")
        .join(name)
}

/// A file written by alice-sdf 6.1.0 (`tests/fixtures/provenance/tree_v1.asdf`, ASDF version 1,
/// no identifier): the default loader refuses it, the explicit unverified loader returns the tree
/// marked `Unpinned`
#[test]
fn an_asdf_file_from_6_1_0_is_refused_by_default_and_read_unverified() {
    let v1 = fixture("tree_v1.asdf");
    assert_eq!(
        &std::fs::read(&v1).unwrap()[4..6],
        &1u16.to_le_bytes(),
        "fixture is version 1"
    );
    match io::load_asdf(&v1) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "ASDF"),
        other => panic!("expected Unpinned, got {other:?}"),
    }
    let checked = io::load_asdf_unverified(&v1).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let (tree, semantics) = checked.into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    let expected = SdfTree::new(SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5)));
    assert_eq!(tree.node_count(), expected.node_count());
    for p in [Vec3::ZERO, Vec3::new(0.7, 0.2, -0.4), Vec3::splat(1.3)] {
        assert_eq!(
            eval(&tree.root, p).to_bits(),
            eval(&expected.root, p).to_bits(),
            "the 6.1.0 tree evaluates the same at {p}"
        );
    }
    assert!(matches!(
        provenance::read(&v1),
        Err(ProvenanceError::Missing { format: "ASDF" })
    ));
}

/// ASDF version 2: this crate's identifier loads as `Current`, an earlier verified one as
/// `VerifiedPrevious`, another one is refused by both loaders naming both identifiers
#[test]
fn asdf_loaders_check_the_identifier() {
    let tree = SdfTree::new(SdfNode::sphere(1.0));
    let path = temp("check.asdf");
    io::save_asdf(&tree, &path).unwrap();
    assert_eq!(
        io::load_asdf(&path).unwrap().node_count(),
        tree.node_count()
    );
    assert_eq!(
        io::load_asdf_unverified(&path).unwrap().semantics(),
        StoredSemantics::Current
    );

    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);
    io::save_asdf_with_provenance(&tree, &path, &earlier).unwrap();
    assert!(io::load_asdf(&path).is_ok());
    assert_eq!(
        io::load_asdf_unverified(&path).unwrap().semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    io::save_asdf_with_provenance(&tree, &path, &caller_id()).unwrap();
    for result in [
        io::load_asdf(&path).map(|_| ()),
        io::load_asdf_unverified(&path).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex())
                );
            }
            other => panic!("expected SemanticsMismatch, got {other:?}"),
        }
    }
    std::fs::remove_file(&path).ok();
}

/// An independent FBX parser (`fbxcel`) reads the whole binary file (nodes and footer), and its
/// `FBXHeaderExtension/Creator` string carries the identifier
#[test]
fn an_independent_fbx_parser_reads_the_binary_fbx() {
    let m = mesh();
    let path = temp("strict.fbx");
    let c = io::FbxConfig {
        format: io::FbxFormat::Binary,
        ..Default::default()
    };
    io::export_fbx_with_provenance(&m, &path, &c, None, &caller_id()).unwrap();
    let f = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let fbxcel::tree::any::AnyTree::V7400(_, tree, footer) =
        fbxcel::tree::any::AnyTree::from_seekable_reader(f).expect("fbxcel reads the file")
    else {
        panic!("not an FBX 7.4 tree")
    };
    footer.expect("fbxcel reads the footer");
    let creator = tree
        .root()
        .first_child_by_name("FBXHeaderExtension")
        .and_then(|h| h.first_child_by_name("Creator"))
        .expect("FBXHeaderExtension/Creator");
    match creator.attributes() {
        [fbxcel::low::v7400::AttributeValue::String(s)] => {
            assert_eq!(*s, format!("ALICE-SDF {}", caller_id().tagged()));
        }
        other => panic!("Creator attributes: {other:?}"),
    }
    std::fs::remove_file(&path).ok();
}

/// An independent binary STL parser (`stl_io`) reads the file with the identifier in the header
#[test]
fn an_independent_stl_parser_reads_the_binary_stl() {
    let m = mesh();
    let path = temp("strict.stl");
    io::export_stl_with_provenance(&m, &path, &caller_id()).unwrap();
    let mut f = std::fs::File::open(&path).unwrap();
    let parsed = stl_io::read_stl(&mut f).expect("stl_io reads the file");
    assert_eq!(parsed.faces.len(), m.indices.len() / 3);
    std::fs::remove_file(&path).ok();
}

/// The 3MF model is well-formed for a 3MF reader: the namespace of the metadata name is declared on
/// `<model>`, and the metadata comes before `<resources>` (3MF core specification, 3.4)
#[test]
fn the_3mf_metadata_is_declared_and_placed_before_resources() {
    let m = mesh();
    let path = temp("order.3mf");
    io::export_3mf_with_provenance(&m, &path, &caller_id()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let decl = format!("xmlns:alice=\"{}\"", provenance::NAMESPACE_3MF);
    let model = text.find("<model ").expect("model element");
    let model_end = model + text[model..].find('>').unwrap();
    assert!(
        text[model..model_end].contains(&decl),
        "namespace declared on <model>"
    );
    let meta = text.find("<metadata name=\"alice:semantics_id\">").unwrap();
    let resources = text.find("<resources>").unwrap();
    assert!(model_end < meta && meta < resources);
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_file_without_the_identifier_is_missing_not_semantics() {
    let m = mesh();
    let path = temp("plain_header.stl");
    io::export_stl(&m, &path).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..80].fill(0);
    std::fs::write(&path, &bytes).unwrap();
    assert!(matches!(
        provenance::read(&path),
        Err(ProvenanceError::Missing { .. })
    ));
    std::fs::remove_file(&path).ok();
}

// ── producer identifier (`*_with_stamp`) ────────────────────────────────────────────────

use alice_sdf::provenance::Stamp;

fn producer_id() -> Provenance {
    let mut id = [0u8; 32];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::try_from(i).unwrap().wrapping_mul(53).wrapping_add(11);
    }
    Provenance::new(id)
}

type StampWrite = fn(&Mesh, &Path, &Stamp);

/// (writer, file name, write with a stamp, write with a provenance only)
const STAMPED: &[(&str, &str, StampWrite, Write)] = &[
    (
        "export_stl_with_stamp",
        "s.stl",
        |m, p, s| io::export_stl_with_stamp(m, p, s).unwrap(),
        |m, p, id| io::export_stl_with_provenance(m, p, id.unwrap()).unwrap(),
    ),
    (
        "export_stl_ascii_with_stamp",
        "s_ascii.stl",
        |m, p, s| io::export_stl_ascii_with_stamp(m, p, s).unwrap(),
        |m, p, id| io::export_stl_ascii_with_provenance(m, p, id.unwrap()).unwrap(),
    ),
    (
        "export_3mf_with_stamp",
        "s.3mf",
        |m, p, s| io::export_3mf_with_stamp(m, p, s).unwrap(),
        |m, p, id| io::export_3mf_with_provenance(m, p, id.unwrap()).unwrap(),
    ),
    (
        "export_fbx_with_stamp (binary)",
        "s_bin.fbx",
        |m, p, s| {
            let c = io::FbxConfig {
                format: io::FbxFormat::Binary,
                ..Default::default()
            };
            io::export_fbx_with_stamp(m, p, &c, None, s).unwrap();
        },
        |m, p, id| {
            let c = io::FbxConfig {
                format: io::FbxFormat::Binary,
                ..Default::default()
            };
            io::export_fbx_with_provenance(m, p, &c, None, id.unwrap()).unwrap();
        },
    ),
    (
        "export_obj_with_stamp",
        "s.obj",
        |m, p, s| io::export_obj_with_stamp(m, p, &io::ObjConfig::default(), None, s).unwrap(),
        |m, p, id| {
            io::export_obj_with_provenance(m, p, &io::ObjConfig::default(), None, id.unwrap())
                .unwrap();
        },
    ),
    (
        "export_fbx_with_stamp (ascii)",
        "s_ascii.fbx",
        |m, p, s| {
            let c = io::FbxConfig {
                format: io::FbxFormat::Ascii,
                ..Default::default()
            };
            io::export_fbx_with_stamp(m, p, &c, None, s).unwrap();
        },
        |m, p, id| {
            let c = io::FbxConfig {
                format: io::FbxFormat::Ascii,
                ..Default::default()
            };
            io::export_fbx_with_provenance(m, p, &c, None, id.unwrap()).unwrap();
        },
    ),
];

/// The OBJ writer's companion MTL file carries both identifiers too
#[test]
fn a_stamped_obj_writes_both_identifiers_into_its_mtl() {
    use alice_sdf::material::{Material, MaterialLibrary};
    let m = mesh();
    let lib = MaterialLibrary {
        materials: vec![Material {
            name: "paint".into(),
            ..Default::default()
        }],
    };
    let path = temp("s_mtl.obj");
    io::export_obj_with_stamp(
        &m,
        &path,
        &io::ObjConfig::default(),
        Some(&lib),
        &Stamp::sdf_with_producer(producer_id()),
    )
    .unwrap();
    for p in [path.clone(), path.with_extension("mtl")] {
        assert_eq!(
            provenance::read_with_producer(&p).unwrap(),
            (Provenance::sdf(), Some(producer_id())),
            "{}",
            p.display()
        );
    }
}

/// Each stamped writer keeps the two identifiers in their own slots (in either assignment), a
/// stamp without a producer writes exactly the bytes of `*_with_provenance`, and `check_producer`
/// answers present / absent / another producer distinctly
#[test]
fn stamped_writers_keep_both_identifiers_in_their_slots() {
    let m = mesh();
    let mut compared = 0;
    for &(name, file, stamped, plain) in STAMPED {
        let path = temp(file);
        // this crate's identifier as primary, the caller's as producer
        stamped(&m, &path, &Stamp::sdf_with_producer(producer_id()));
        assert_eq!(
            provenance::read_with_producer(&path).unwrap(),
            (Provenance::sdf(), Some(producer_id())),
            "{name}"
        );
        assert_eq!(
            provenance::read(&path).unwrap(),
            Provenance::sdf(),
            "{name}"
        );
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            StoredSemantics::Current,
            "{name}"
        );
        provenance::check_producer(&path, &producer_id()).unwrap();
        match provenance::check_producer(&path, &caller_id()) {
            Err(ProvenanceError::ProducerMismatch { stored, expected }) => {
                assert_eq!((stored, expected), (producer_id(), caller_id()), "{name}");
            }
            other => panic!("{name}: expected a producer mismatch, got {other:?}"),
        }
        // the other assignment: the two must not swap places
        stamped(
            &m,
            &path,
            &Stamp::new(producer_id(), Some(Provenance::sdf())),
        );
        assert_eq!(
            provenance::read_with_producer(&path).unwrap(),
            (producer_id(), Some(Provenance::sdf())),
            "{name}: swapped assignment"
        );
        // no producer: the same bytes as the provenance writer, and no producer read back
        stamped(&m, &path, &Stamp::from(caller_id()));
        let with_stamp = std::fs::read(&path).unwrap();
        plain(&m, &path, Some(&caller_id()));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            with_stamp,
            "{name}: no producer"
        );
        assert_eq!(
            provenance::read_with_producer(&path).unwrap(),
            (caller_id(), None),
            "{name}"
        );
        assert!(
            matches!(
                provenance::check_producer(&path, &producer_id()),
                Err(ProvenanceError::ProducerMissing)
            ),
            "{name}"
        );
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, STAMPED.len());
    assert!(compared > 0, "compared no stamped writer");
}

/// The binary STL header with a producer: version 2, bytes 41..48 zero, the producer in 48..80,
/// never starting with `solid`; an all-zero producer is still present (the version says so)
#[test]
fn the_binary_stl_header_with_a_producer_has_the_documented_layout() {
    let m = mesh();
    let path = temp("layout.stl");
    io::export_stl_with_stamp(&m, &path, &Stamp::sdf_with_producer(producer_id())).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], provenance::STL_MAGIC);
    assert_eq!(bytes[8], provenance::STL_HEADER_VERSION_PRODUCER);
    assert_eq!(&bytes[9..41], &Provenance::sdf().semantics_id);
    assert_eq!(&bytes[41..48], &[0u8; 7]);
    assert_eq!(&bytes[48..80], &producer_id().semantics_id);
    assert!(!bytes.starts_with(b"solid"));
    // without a producer: version 1 and zeros after the identifier
    io::export_stl_with_stamp(&m, &path, &Stamp::from(Provenance::sdf())).unwrap();
    let plain = std::fs::read(&path).unwrap();
    assert_eq!(plain[8], provenance::STL_HEADER_VERSION);
    assert_eq!(&plain[41..80], &[0u8; 39]);
    // an all-zero producer is distinguishable from none
    io::export_stl_with_stamp(
        &m,
        &path,
        &Stamp::sdf_with_producer(Provenance::new([0; 32])),
    )
    .unwrap();
    assert_eq!(
        provenance::read_with_producer(&path).unwrap(),
        (Provenance::sdf(), Some(Provenance::new([0; 32])))
    );
    // a version 2 header with a non-zero reserved byte carries this crate's magic but is not a
    // layout it writes: refused, never read as a file without an identifier
    let mut bad = std::fs::read(&path).unwrap();
    bad[44] = 1;
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(
        provenance::read(&path),
        Err(ProvenanceError::Unreadable(_))
    ));
    std::fs::remove_file(&path).ok();
}

/// ASCII STL: with both tokens the primary is read from its own key; a solid name with only the
/// producer token has no primary identifier (the producer is not misread as one)
#[test]
fn ascii_stl_reads_each_identifier_from_its_own_key() {
    let m = mesh();
    let path = temp("keys.stl");
    io::export_stl_ascii_with_stamp(&m, &path, &Stamp::sdf_with_producer(producer_id())).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let first = text.lines().next().unwrap();
    assert_eq!(
        first,
        format!(
            "solid {} {}={}",
            Provenance::sdf().tagged(),
            provenance::PRODUCER_KEY,
            producer_id().hex()
        )
    );
    let only_producer = text.replacen(&format!("{} ", Provenance::sdf().tagged()), "", 1);
    std::fs::write(&path, only_producer).unwrap();
    assert!(matches!(
        provenance::read(&path),
        Err(ProvenanceError::Missing { .. })
    ));
    std::fs::remove_file(&path).ok();
}

/// The independent parsers still read the stamped files: `stl_io` the version 2 binary STL,
/// `fbxcel` the binary FBX (with `ProducerSemanticsId` beside `Creator`), and the 3MF model keeps
/// both metadata entries before `<resources>`
#[test]
fn independent_parsers_read_the_stamped_files() {
    let m = mesh();
    let stamp = Stamp::sdf_with_producer(producer_id());

    let stl = temp("strict_stamp.stl");
    io::export_stl_with_stamp(&m, &stl, &stamp).unwrap();
    let mut f = std::fs::File::open(&stl).unwrap();
    let parsed = stl_io::read_stl(&mut f).expect("stl_io reads the file");
    assert_eq!(parsed.faces.len(), m.indices.len() / 3);
    std::fs::remove_file(&stl).ok();

    let fbx = temp("strict_stamp.fbx");
    let c = io::FbxConfig {
        format: io::FbxFormat::Binary,
        ..Default::default()
    };
    io::export_fbx_with_stamp(&m, &fbx, &c, None, &stamp).unwrap();
    let f = std::io::BufReader::new(std::fs::File::open(&fbx).unwrap());
    let fbxcel::tree::any::AnyTree::V7400(_, tree, footer) =
        fbxcel::tree::any::AnyTree::from_seekable_reader(f).expect("fbxcel reads the file")
    else {
        panic!("not an FBX 7.4 tree")
    };
    footer.expect("fbxcel reads the footer");
    let header = tree
        .root()
        .first_child_by_name("FBXHeaderExtension")
        .expect("FBXHeaderExtension");
    let string_of = |name: &str| match header
        .first_child_by_name(name)
        .unwrap_or_else(|| panic!("{name}"))
        .attributes()
    {
        [fbxcel::low::v7400::AttributeValue::String(s)] => s.clone(),
        other => panic!("{name} attributes: {other:?}"),
    };
    assert_eq!(
        string_of("Creator"),
        format!("ALICE-SDF {}", Provenance::sdf().tagged())
    );
    assert_eq!(
        string_of("ProducerSemanticsId"),
        format!("{}={}", provenance::PRODUCER_KEY, producer_id().hex())
    );
    std::fs::remove_file(&fbx).ok();

    let tmf = temp("order_stamp.3mf");
    io::export_3mf_with_stamp(&m, &tmf, &stamp).unwrap();
    let bytes = std::fs::read(&tmf).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let primary = text.find("<metadata name=\"alice:semantics_id\">").unwrap();
    let producer = text
        .find("<metadata name=\"alice:producer_semantics_id\">")
        .unwrap();
    let resources = text.find("<resources>").unwrap();
    assert!(primary < producer && producer < resources);
    std::fs::remove_file(&tmf).ok();
}

/// The mesh the ABM fixtures hold (written by alice-sdf 6.1.0 from crates.io with the same
/// construction): a triangle and a quad, normals along +Z, scaled by `scale`
fn abm_fixture_mesh(scale: f32) -> Mesh {
    let p = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(2.0, 0.0, 0.5),
        Vec3::new(3.0, 0.0, 0.5),
        Vec3::new(2.0, 1.0, 0.5),
    ];
    Mesh {
        vertices: p
            .iter()
            .map(|&v| Vertex {
                position: v * scale,
                normal: Vec3::Z,
                ..Vertex::default()
            })
            .collect(),
        indices: vec![0, 1, 2, 3, 4, 5, 0, 2, 3],
    }
}

fn same_mesh(a: &Mesh, b: &Mesh) -> bool {
    a.indices == b.indices
        && a.vertices.len() == b.vertices.len()
        && a.vertices.iter().zip(&b.vertices).all(|(x, y)| {
            x.position.to_array().map(f32::to_bits) == y.position.to_array().map(f32::to_bits)
                && x.normal.to_array().map(f32::to_bits) == y.normal.to_array().map(f32::to_bits)
        })
}

/// Files written by alice-sdf 6.1.0 (`tests/fixtures/provenance/mesh_v1.abm` and `lods_v1.abm`,
/// ABM version 1, no identifier): both default loaders refuse them, the unverified loaders
/// return the meshes bit for bit, marked `Unpinned`
#[test]
fn an_abm_file_from_6_1_0_is_refused_by_default_and_read_unverified() {
    let single = fixture("mesh_v1.abm");
    let lods = fixture("lods_v1.abm");
    for path in [&single, &lods] {
        assert_eq!(
            &std::fs::read(path).unwrap()[8..10],
            &1u16.to_le_bytes(),
            "fixture is version 1"
        );
        for refused in [
            io::load_abm(path).map(|_| ()),
            io::load_abm_with_lods(path).map(|_| ()),
        ] {
            match refused {
                Err(io::IoError::Unpinned { format }) => assert_eq!(format, "ABM"),
                other => panic!("expected Unpinned, got {other:?}"),
            }
        }
        assert!(matches!(
            provenance::read(path),
            Err(ProvenanceError::Missing { format: "ABM" })
        ));
        assert_eq!(io::read_abm_header(path).unwrap().semantics_id, None);
    }

    let checked = io::load_abm_unverified(&single).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let (m, _) = checked.into_parts();
    assert!(
        same_mesh(&m, &abm_fixture_mesh(1.0)),
        "the 6.1.0 mesh reads back bit for bit"
    );

    let checked = io::load_abm_with_lods_unverified(&lods).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let ((meshes, distances), _) = checked.into_parts();
    assert_eq!(meshes.len(), 2);
    assert!(same_mesh(&meshes[0], &abm_fixture_mesh(1.0)));
    assert!(same_mesh(&meshes[1], &abm_fixture_mesh(0.5)));
    assert_eq!(distances, [10.0]);
}

/// ABM version 2: this crate's identifier loads as `Current`, an earlier verified one as
/// `VerifiedPrevious`, another one is refused by all four loaders naming both identifiers
#[test]
fn abm_loaders_check_the_identifier() {
    let m = abm_fixture_mesh(1.0);
    let single = temp("check.abm");
    let lods = temp("check_lods.abm");
    let write = |id: &Provenance| {
        io::save_abm_with_provenance(&m, &single, id).unwrap();
        io::save_abm_with_lods_with_provenance(&[m.clone(), m.clone()], &[5.0], &lods, id).unwrap();
    };

    write(&Provenance::sdf());
    assert!(same_mesh(&io::load_abm(&single).unwrap(), &m));
    assert_eq!(io::load_abm_with_lods(&lods).unwrap().0.len(), 2);
    assert_eq!(
        io::load_abm_unverified(&single).unwrap().semantics(),
        StoredSemantics::Current
    );
    assert_eq!(
        io::load_abm_with_lods_unverified(&lods)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);
    write(&earlier);
    assert!(io::load_abm(&single).is_ok());
    assert!(io::load_abm_with_lods(&lods).is_ok());
    assert_eq!(
        io::load_abm_unverified(&single).unwrap().semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );
    assert_eq!(
        io::load_abm_with_lods_unverified(&lods)
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    write(&caller_id());
    for result in [
        io::load_abm(&single).map(|_| ()),
        io::load_abm_unverified(&single).map(|_| ()),
        io::load_abm_with_lods(&lods).map(|_| ()),
        io::load_abm_with_lods_unverified(&lods).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex()),
                    "{text}"
                );
            }
            other => panic!("expected SemanticsMismatch, got {other:?}"),
        }
    }
    std::fs::remove_file(&single).ok();
    std::fs::remove_file(&lods).ok();
}

/// The ABM header is read by its version: a version the crate does not know is refused, not
/// read at the version 2 offset, and a truncated header is refused by the reader and the loaders
#[test]
fn abm_headers_are_read_by_their_exact_version() {
    let path = temp("version.abm");
    io::save_abm(&abm_fixture_mesh(1.0), &path).unwrap();
    let good = std::fs::read(&path).unwrap();

    // 0 (the old `version > ABM_VERSION` check let it through), 3 (the next one) and the largest
    for v in [0u16, 3, u16::MAX] {
        let mut other = good.clone();
        other[8..10].copy_from_slice(&v.to_le_bytes());
        assert!(
            matches!(
                provenance::from_bytes(&other),
                Err(ProvenanceError::Unreadable(ref why)) if why.contains(&format!("version {v}"))
            ),
            "provenance reads ABM version {v}"
        );
        assert!(
            matches!(io::AbmHeader::from_bytes(&other), Err(io::IoError::UnsupportedVersion(x)) if x == v),
            "the header parser accepts version {v}"
        );
        std::fs::write(&path, &other).unwrap();
        assert!(
            matches!(io::load_abm(&path), Err(io::IoError::UnsupportedVersion(x)) if x == v),
            "load_abm accepts version {v}"
        );
        assert!(
            io::load_abm_unverified(&path).is_err(),
            "the unverified loader reads version {v}"
        );
    }

    let truncated = &good[..40];
    assert!(matches!(
        provenance::from_bytes(truncated),
        Err(ProvenanceError::Unreadable(_))
    ));
    assert!(matches!(
        io::AbmHeader::from_bytes(truncated),
        Err(io::IoError::InvalidFormat(_))
    ));
    std::fs::write(&path, truncated).unwrap();
    assert!(io::load_abm_unverified(&path).is_err());
    std::fs::remove_file(&path).ok();
}

fn lod_chain(m: &Mesh) -> LodChainPersist {
    LodChainPersist::new(
        vec![m.clone(), m.clone()],
        vec![0.0, 10.0],
        0x5eed,
        LodChainConfig::default(),
    )
}

/// A LOD chain written by alice-sdf 6.1.0 (`tests/fixtures/provenance/chain_v1.abm` and its
/// sidecar `chain_v1.abm.meta.json`, format 1, neither part has an identifier): the default
/// loader refuses it, the unverified loader returns it marked `Unpinned` with the stored
/// values bit for bit, and the metadata loader returns the authored config marked `Unpinned`
#[test]
fn a_lod_chain_from_6_1_0_is_refused_by_default_and_read_unverified() {
    let dir = temp("chain_v1");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("chain_v1.abm");
    std::fs::copy(fixture("chain_v1.abm"), &path).unwrap();
    std::fs::copy(
        fixture("chain_v1.abm.meta.json"),
        dir.join("chain_v1.abm.meta.json"),
    )
    .unwrap();

    match lod_persist::load_lod_chain(&path) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "ABM"),
        other => panic!("expected Unpinned, got {other:?}"),
    }
    let (chain, semantics) = lod_persist::load_lod_chain_unverified(&path)
        .unwrap()
        .into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(chain.sdf_hash, 0x5eed);
    assert_eq!(chain.meshes.len(), 2);
    assert!(same_mesh(&chain.meshes[0].mesh, &abm_fixture_mesh(1.0)));
    assert!(same_mesh(&chain.meshes[1].mesh, &abm_fixture_mesh(0.5)));
    assert_eq!(
        chain.meshes[0].aabb_max,
        [3.0, 1.0, 0.5],
        "the stored AABB, read back"
    );
    assert_eq!(chain.meshes[1].transition_distance, 10.0);

    let (config, semantics) =
        lod_persist::load_lod_chain_metadata(dir.join("chain_v1.abm.meta.json"))
            .unwrap()
            .into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(config.lod_count, LodChainConfig::default().lod_count);
    std::fs::remove_dir_all(&dir).ok();
}

/// Both parts of a chain carry the identifier: this crate's reads as `Current`, an earlier
/// verified one as `VerifiedPrevious`, another one in either part is refused naming both, and
/// the unverified loader reports the weaker of the two parts
#[test]
fn lod_chain_loaders_check_both_parts() {
    let m = abm_fixture_mesh(1.0);
    let chain = lod_chain(&m);
    let path = temp("check_chain.abm");
    let meta = path.with_file_name("check_chain.abm.meta.json");
    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);

    lod_persist::save_lod_chain(&chain, &path).unwrap();
    assert_eq!(lod_persist::load_lod_chain(&path).unwrap().meshes.len(), 2);
    assert_eq!(
        lod_persist::load_lod_chain_unverified(&path)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );
    assert_eq!(
        lod_persist::load_lod_chain_metadata(&meta)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    // the sidecar written under an earlier identifier: the chain reads as the weaker part
    lod_persist::save_lod_chain_metadata_with_provenance(&chain, &path, &earlier).unwrap();
    assert!(lod_persist::load_lod_chain(&path).is_ok());
    assert_eq!(
        lod_persist::load_lod_chain_unverified(&path)
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    // the mesh file under the earlier identifier, the sidecar current: the weaker part is the
    // mesh file, so the chain is VerifiedPrevious, not Current
    lod_persist::save_lod_chain_with_provenance(&chain, &path, &earlier).unwrap();
    lod_persist::save_lod_chain_metadata(&chain, &path).unwrap();
    assert!(lod_persist::load_lod_chain(&path).is_ok());
    assert_eq!(
        lod_persist::load_lod_chain_unverified(&path)
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    // a mesh file without an identifier (written by 6.1.0) and a current sidecar: the default
    // loader refuses, the unverified one reads the chain as Unpinned, not Current
    std::fs::copy(fixture("chain_v1.abm"), &path).unwrap();
    lod_persist::save_lod_chain_metadata(&chain, &path).unwrap();
    assert!(matches!(
        lod_persist::load_lod_chain(&path),
        Err(io::IoError::Unpinned { .. })
    ));
    assert_eq!(
        lod_persist::load_lod_chain_unverified(&path)
            .unwrap()
            .semantics(),
        StoredSemantics::Unpinned
    );

    // another identifier in the sidecar only, then in the mesh file only: refused either way
    lod_persist::save_lod_chain(&chain, &path).unwrap();
    lod_persist::save_lod_chain_metadata_with_provenance(&chain, &path, &caller_id()).unwrap();
    for result in [
        lod_persist::load_lod_chain(&path).map(|_| ()),
        lod_persist::load_lod_chain_unverified(&path).map(|_| ()),
        lod_persist::load_lod_chain_metadata(&meta).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex())
                );
            }
            other => panic!("sidecar: expected SemanticsMismatch, got {other:?}"),
        }
    }
    lod_persist::save_lod_chain_with_provenance(&chain, &path, &caller_id()).unwrap();
    lod_persist::save_lod_chain_metadata(&chain, &path).unwrap();
    assert!(matches!(
        lod_persist::load_lod_chain(&path),
        Err(io::IoError::SemanticsMismatch { .. })
    ));

    // the mesh file current, the sidecar without an identifier (format 1): default refuses,
    // unverified reads it as Unpinned
    lod_persist::save_lod_chain(&chain, &path).unwrap();
    std::fs::copy(fixture("chain_v1.abm.meta.json"), &meta).unwrap();
    assert!(matches!(
        lod_persist::load_lod_chain(&path),
        Err(io::IoError::Unpinned { .. })
    ));
    assert_eq!(
        lod_persist::load_lod_chain_unverified(&path)
            .unwrap()
            .semantics(),
        StoredSemantics::Unpinned
    );
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&meta).ok();
}

/// The sidecar is read by its format: format 2 without the identifier, an identifier that is not
/// 64 hex digits and a format this crate does not know are refused, not read
#[test]
fn lod_metadata_is_read_by_its_exact_format() {
    let chain = lod_chain(&abm_fixture_mesh(1.0));
    let path = temp("format_chain.abm");
    let meta = path.with_file_name("format_chain.abm.meta.json");
    lod_persist::save_lod_chain(&chain, &path).unwrap();
    let good: serde_json::Value = serde_json::from_slice(&std::fs::read(&meta).unwrap()).unwrap();
    assert_eq!(good["format"], 2);
    assert_eq!(good[provenance::KEY], Provenance::sdf().hex());

    let mut cases = Vec::new();
    let mut v = good.clone();
    v.as_object_mut().unwrap().remove(provenance::KEY);
    cases.push(("format 2 without the identifier", v));
    let mut v = good.clone();
    v[provenance::KEY] = serde_json::Value::from("abc");
    cases.push(("a short identifier", v));
    for f in [0, 3, 1000] {
        let mut v = good.clone();
        v["format"] = serde_json::Value::from(f);
        cases.push(("an unknown format", v.clone()));
        v.as_object_mut().unwrap().remove(provenance::KEY);
        cases.push(("an unknown format without the identifier", v));
    }
    for (what, v) in cases {
        std::fs::write(&meta, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(
            lod_persist::load_lod_chain_unverified(&path).is_err(),
            "{what}: read"
        );
        assert!(
            lod_persist::load_lod_chain_metadata(&meta).is_err(),
            "{what}: metadata read"
        );
    }
    std::fs::remove_file(&path).ok();
    std::fs::remove_file(&meta).ok();
}

// ---------------------------------------------------------------------------------------------
// G8 group F: this crate's own stored forms that close only as reader + writer
//
// The version 1 fixtures were written by alice-sdf 6.1.0 from crates.io (a throwaway crate on
// `alice-sdf = "=6.1.0"`, features `openvdb` and `svo`); none of them carries an identifier:
//   neural_v1.nsdf     fc28931eacf667c5504e035dfbcbf87e4f3c3f08e23adebdc5a764509eeda564
//                      NeuralSdf::new(&NeuralSdfConfig { hidden_layers: 1, hidden_width: 4,
//                      pos_encoding_freqs: 1, ..Default::default() }).save(..)
//   grid_v1.alicevdb   66c2dae8e96828c2c8cd4c2cddd24ad54ef7bbdc345ec1d9c7273ae551e5566a
//                      io::vdb::bake_to_vdb(&SdfNode::sphere(1.0), (-1.5, 1.5), 4)
//   chunk_v1.svoc      5cf74507b60fac586b8a9b260a68496e22e07f0df4deb8d5bea14855038273ae
//                      svo::SvoChunk::new(7, vec![node, node], AabbPacked(-1..1), 2).to_bytes()
// ---------------------------------------------------------------------------------------------

use alice_sdf::neural::{NeuralSdf, NeuralSdfConfig};

/// (writer name as the scanner reads it, what it writes) for the own stored forms of group F; each
/// one is written, read back through `provenance::from_bytes` and judged by the tests below
const STORED_WRITERS: &[(&str, &str)] = &[
    ("src/neural.rs::save", "NSDF weights"),
    ("src/svo/streaming.rs::to_bytes", "SVO chunk"),
];

fn small_net() -> NeuralSdf {
    NeuralSdf::new(&NeuralSdfConfig {
        hidden_layers: 1,
        hidden_width: 4,
        pos_encoding_freqs: 1,
        ..Default::default()
    })
}

fn net_bits(net: &NeuralSdf) -> Vec<u32> {
    (0..27)
        .map(|i| {
            let c = |k: i32| (k % 3 - 1) as f32 * 0.7;
            net.eval(Vec3::new(c(i), c(i / 3), c(i / 9))).to_bits()
        })
        .collect()
}

fn nsdf_bytes(id: Option<&Provenance>) -> Vec<u8> {
    let mut b = Vec::new();
    match id {
        None => small_net().save(&mut b).unwrap(),
        Some(id) => small_net().save_with_provenance(&mut b, id).unwrap(),
    }
    b
}

/// The NSDF file written by alice-sdf 6.1.0 (version 1, no identifier): `NeuralSdf::load`
/// refuses it as `Unpinned`, `load_unverified` returns the same network marked `Unpinned`
#[test]
fn an_nsdf_file_from_6_1_0_is_refused_by_default_and_read_unverified() {
    let bytes = std::fs::read(fixture("neural_v1.nsdf")).unwrap();
    assert_eq!(&bytes[..4], b"NSDF");
    assert_eq!(&bytes[4..8], &1u32.to_le_bytes(), "fixture is version 1");
    assert!(matches!(
        provenance::from_bytes(&bytes),
        Err(ProvenanceError::Missing { format: "NSDF" })
    ));
    match NeuralSdf::load(&mut bytes.as_slice()) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "NSDF"),
        other => panic!("expected Unpinned, got {:?}", other.map(|_| ())),
    }
    let checked = NeuralSdf::load_unverified(&mut bytes.as_slice()).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let (net, _) = checked.into_parts();
    assert_eq!(net.param_count(), small_net().param_count());
    assert_eq!(
        net_bits(&net),
        net_bits(&small_net()),
        "the 6.1.0 weights read back bit for bit"
    );
}

/// NSDF version 2: `save` stores this crate's identifier, `save_with_provenance` the caller's;
/// this crate's loads as `Current`, an earlier verified one as `VerifiedPrevious`, another one
/// is refused by both loaders naming both identifiers
#[test]
fn nsdf_writers_store_and_loaders_check_the_identifier() {
    assert!(STORED_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/neural.rs::save"));
    let own = nsdf_bytes(None);
    assert_eq!(&own[4..8], &2u32.to_le_bytes());
    assert_eq!(provenance::from_bytes(&own).unwrap(), Provenance::sdf());
    let back = NeuralSdf::load(&mut own.as_slice()).unwrap();
    assert_eq!(net_bits(&back), net_bits(&small_net()));
    assert_eq!(
        NeuralSdf::load_unverified(&mut own.as_slice())
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);
    let prev = nsdf_bytes(Some(&earlier));
    assert_eq!(provenance::from_bytes(&prev).unwrap(), earlier);
    assert!(NeuralSdf::load(&mut prev.as_slice()).is_ok());
    assert_eq!(
        NeuralSdf::load_unverified(&mut prev.as_slice())
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = nsdf_bytes(Some(&caller_id()));
    assert_eq!(provenance::from_bytes(&other).unwrap(), caller_id());
    for result in [
        NeuralSdf::load(&mut other.as_slice()).map(|_| ()),
        NeuralSdf::load_unverified(&mut other.as_slice()).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex()),
                    "{text}"
                );
            }
            other => panic!("expected SemanticsMismatch, got {other:?}"),
        }
    }
}

/// The NSDF header is read by its exact version, and a header cut short after the magic is an
/// error of the format, never a file without an identifier
#[test]
fn nsdf_headers_are_read_by_their_exact_version_and_never_as_unstamped() {
    let own = nsdf_bytes(None);
    for v in [0u32, 3, u32::MAX] {
        let mut b = own.clone();
        b[4..8].copy_from_slice(&v.to_le_bytes());
        assert!(
            matches!(provenance::from_bytes(&b), Err(ProvenanceError::Unreadable(ref why)) if why.contains(&format!("version {v}"))),
            "provenance reads NSDF version {v}"
        );
        for result in [
            NeuralSdf::load(&mut b.as_slice()).map(|_| ()),
            NeuralSdf::load_unverified(&mut b.as_slice()).map(|_| ()),
        ] {
            // a version that fits UnsupportedVersion's u16 is reported there, a larger one with
            // its real value
            let named = match u16::try_from(v) {
                Ok(v16) => matches!(result, Err(io::IoError::UnsupportedVersion(x)) if x == v16),
                Err(_) => {
                    matches!(result, Err(io::IoError::InvalidFormat(ref why)) if *why == format!("NSDF: version {v} is not known"))
                }
            };
            assert!(named, "version {v}: {result:?}");
        }
    }
    // the magic, then 0..8+31 bytes: the version or the identifier is cut short
    for len in [4usize, 6, 8, 20, 39] {
        let b = &own[..len];
        assert!(
            matches!(
                provenance::from_bytes(b),
                Err(ProvenanceError::Unreadable(_))
            ),
            "provenance reads {len} bytes"
        );
        for result in [
            NeuralSdf::load(&mut &b[..]).map(|_| ()),
            NeuralSdf::load_unverified(&mut &b[..]).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why.contains("truncated")),
                "{len} bytes: {result:?}"
            );
        }
    }
}

// ALICE-VDB dense grid (`io::vdb`, feature `openvdb`): grid_v1.alicevdb
//   66c2dae8e96828c2c8cd4c2cddd24ad54ef7bbdc345ec1d9c7273ae551e5566a
//   bake_to_vdb(&SdfNode::sphere(1.0), (-1.5, 1.5), 4) by alice-sdf 6.1.0 (`ALICEVDB1`)

#[cfg(feature = "openvdb")]
fn vdb_bytes(id: Option<&Provenance>) -> Vec<u8> {
    use alice_sdf::io::vdb;
    let node = SdfNode::sphere(1.0);
    match id {
        None => vdb::bake_to_vdb(&node, (-1.5, 1.5), 4).unwrap(),
        Some(id) => vdb::bake_to_vdb_with_provenance(&node, (-1.5, 1.5), 4, id).unwrap(),
    }
}

/// The ALICE-VDB grid written by alice-sdf 6.1.0 (`ALICEVDB1`, no identifier): the default loader
/// refuses it as `Unpinned`, the unverified loader returns the same grid marked `Unpinned`
#[cfg(feature = "openvdb")]
#[test]
fn alice_vdb_grid_from_6_1_0_is_refused_by_default_and_read_unverified() {
    use alice_sdf::io::vdb;
    let bytes = std::fs::read(fixture("grid_v1.alicevdb")).unwrap();
    assert_eq!(
        &bytes[..9],
        b"ALICEVDB1",
        "fixture is version 1, no identifier"
    );
    assert!(matches!(
        provenance::from_bytes(&bytes),
        Err(ProvenanceError::Missing {
            format: "ALICE-VDB"
        })
    ));
    match vdb::load_dense_grid_from_vdb(&bytes) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "ALICE-VDB"),
        other => panic!("expected Unpinned, got {:?}", other.map(|_| ())),
    }
    let checked = vdb::load_dense_grid_from_vdb_unverified(&bytes).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let ((grid, res, bounds), _) = checked.into_parts();
    let (now, now_res, now_bounds) = vdb::load_dense_grid_from_vdb(&vdb_bytes(None)).unwrap();
    assert_eq!((res, bounds), (now_res, now_bounds));
    assert_eq!(
        grid.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        now.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "the 6.1.0 grid reads back bit for bit"
    );
}

/// `ALICEVDB2`: `bake_to_vdb` stores this crate's identifier, `bake_to_vdb_with_provenance` the
/// caller's; this crate's loads as `Current`, an earlier verified one as `VerifiedPrevious`,
/// another one is refused by both loaders naming both identifiers
#[cfg(feature = "openvdb")]
#[test]
fn alice_vdb_writers_store_and_loaders_check_the_identifier() {
    use alice_sdf::io::vdb;
    let own = vdb_bytes(None);
    assert_eq!(&own[..9], b"ALICEVDB2");
    assert_eq!(provenance::from_bytes(&own).unwrap(), Provenance::sdf());
    assert_eq!(
        vdb::load_dense_grid_from_vdb_unverified(&own)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);
    let prev = vdb_bytes(Some(&earlier));
    assert_eq!(provenance::from_bytes(&prev).unwrap(), earlier);
    assert!(vdb::load_dense_grid_from_vdb(&prev).is_ok());
    assert_eq!(
        vdb::load_dense_grid_from_vdb_unverified(&prev)
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = vdb_bytes(Some(&caller_id()));
    assert_eq!(provenance::from_bytes(&other).unwrap(), caller_id());
    for result in [
        vdb::load_dense_grid_from_vdb(&other).map(|_| ()),
        vdb::load_dense_grid_from_vdb_unverified(&other).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex()),
                    "{text}"
                );
            }
            other => panic!("expected SemanticsMismatch, got {other:?}"),
        }
    }
}

/// An ALICE-VDB magic with a version the crate does not know, or a header cut short, is an error
/// of the format, never a grid without an identifier
#[cfg(feature = "openvdb")]
#[test]
fn alice_vdb_headers_are_read_by_their_exact_version_and_never_as_unstamped() {
    use alice_sdf::io::vdb;
    let own = vdb_bytes(None);
    for v in *b"039X" {
        let mut b = own.clone();
        b[8] = v;
        assert!(
            matches!(provenance::from_bytes(&b), Err(ProvenanceError::Unreadable(ref why)) if why.contains("is not known")),
            "provenance reads version byte {v}"
        );
        for result in [
            vdb::load_dense_grid_from_vdb(&b).map(|_| ()),
            vdb::load_dense_grid_from_vdb_unverified(&b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why.contains("is not known")),
                "version byte {v}: {result:?}"
            );
        }
    }
    for len in [8usize, 9, 20, 40] {
        let b = &own[..len];
        assert!(
            matches!(
                provenance::from_bytes(b),
                Err(ProvenanceError::Unreadable(_))
            ),
            "provenance reads {len} bytes"
        );
        for result in [
            vdb::load_dense_grid_from_vdb(b).map(|_| ()),
            vdb::load_dense_grid_from_vdb_unverified(b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why.contains("truncated")),
                "{len} bytes: {result:?}"
            );
        }
    }
    // a resolution whose cell count or byte length wraps usize: an error, never a wrapped length
    // that passes the size check and indexes past the buffer
    let body = 9 + 32;
    for r in [u32::MAX, 2_700_000, 1_700_000] {
        let mut b = own.clone();
        b[body..body + 4].copy_from_slice(&r.to_le_bytes());
        for result in [
            vdb::load_dense_grid_from_vdb(&b).map(|_| ()),
            vdb::load_dense_grid_from_vdb_unverified(&b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why.contains("overflows")),
                "resolution {r}: {result:?}"
            );
        }
    }
}

// SVO chunk wire form (`svo::SvoChunk`, feature `svo`): chunk_v1.svoc
//   5cf74507b60fac586b8a9b260a68496e22e07f0df4deb8d5bea14855038273ae
//   SvoChunk::new(7, vec![node, node], AabbPacked::new(Vec3::splat(-1.0), Vec3::splat(1.0)), 2)
//   .to_bytes() by alice-sdf 6.1.0 (node: distance 0.25, normal (0, 1, 0), leaf, material 3)

#[cfg(feature = "svo")]
fn svo_chunk() -> alice_sdf::svo::SvoChunk {
    let node = alice_sdf::svo::SvoNode {
        distance: 0.25,
        nx: 0.0,
        ny: 1.0,
        nz: 0.0,
        child_mask: 0,
        is_leaf: 1,
        material_id: 3,
        first_child: 0,
        _pad: [0.0, 0.0],
    };
    let bounds = alice_sdf::compiled::AabbPacked::new(Vec3::splat(-1.0), Vec3::splat(1.0));
    alice_sdf::svo::SvoChunk::new(7, vec![node, node], bounds, 2)
}

/// The SVO chunk written by alice-sdf 6.1.0 (version 1: no magic, no identifier): `from_bytes`
/// refuses it as `Unpinned`, `from_bytes_unverified` returns the same chunk marked `Unpinned`
#[cfg(feature = "svo")]
#[test]
fn an_svo_chunk_from_6_1_0_is_refused_by_default_and_read_unverified() {
    use alice_sdf::svo::SvoChunk;
    let bytes = std::fs::read(fixture("chunk_v1.svoc")).unwrap();
    assert_ne!(&bytes[..4], b"ASVC", "fixture is version 1, no magic");
    assert_eq!(
        &bytes[..4],
        &7u32.to_le_bytes(),
        "a version 1 chunk starts with its chunk_id"
    );
    match SvoChunk::from_bytes(&bytes) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "SVO chunk"),
        other => panic!("expected Unpinned, got {:?}", other.map(|_| ())),
    }
    let checked = SvoChunk::from_bytes_unverified(&bytes).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let (chunk, _) = checked.into_parts();
    // the version 2 bytes of the same chunk are the 40-byte header followed by the 6.1.0 bytes
    assert_eq!(
        &chunk.to_bytes()[40..],
        &bytes[..],
        "the 6.1.0 chunk reads back bit for bit"
    );
    assert_eq!(&svo_chunk().to_bytes()[40..], &bytes[..]);
}

/// `ASVC` version 2: `to_bytes` stores this crate's identifier, `to_bytes_with_provenance` the
/// caller's; this crate's reads as `Current`, an earlier verified one as `VerifiedPrevious`,
/// another one is refused by both readers naming both identifiers
#[cfg(feature = "svo")]
#[test]
fn svo_chunk_writers_store_and_readers_check_the_identifier() {
    use alice_sdf::svo::SvoChunk;
    assert!(STORED_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/svo/streaming.rs::to_bytes"));
    let own = svo_chunk().to_bytes();
    assert_eq!(&own[..4], b"ASVC");
    assert_eq!(&own[8..40], &alice_sdf::SDF_SEMANTICS_ID);
    assert_eq!(provenance::from_bytes(&own).unwrap(), Provenance::sdf());
    assert_eq!(
        provenance::from_bytes(&svo_chunk().to_bytes_with_provenance(&caller_id())).unwrap(),
        caller_id()
    );
    assert_eq!(SvoChunk::from_bytes(&own).unwrap().to_bytes(), own);
    assert_eq!(
        SvoChunk::from_bytes_unverified(&own).unwrap().semantics(),
        StoredSemantics::Current
    );

    let earlier = Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0]);
    let prev = svo_chunk().to_bytes_with_provenance(&earlier);
    assert!(SvoChunk::from_bytes(&prev).is_ok());
    assert_eq!(
        SvoChunk::from_bytes_unverified(&prev).unwrap().semantics(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = svo_chunk().to_bytes_with_provenance(&caller_id());
    for result in [
        SvoChunk::from_bytes(&other).map(|_| ()),
        SvoChunk::from_bytes_unverified(&other).map(|_| ()),
    ] {
        match result {
            Err(e @ io::IoError::SemanticsMismatch { .. }) => {
                let text = e.to_string();
                assert!(
                    text.contains(&caller_id().hex()) && text.contains(&Provenance::sdf().hex()),
                    "{text}"
                );
            }
            other => panic!("expected SemanticsMismatch, got {other:?}"),
        }
    }
}

/// With the magic present, an unknown version or a header cut short is an error of the format,
/// never a version 1 chunk; bytes without the magic are never accepted by the checked reader; a
/// version 1 chunk whose chunk_id spells the magic is refused, never read as version 1
#[cfg(feature = "svo")]
#[test]
fn svo_chunk_headers_are_read_by_their_exact_version_and_never_as_version_1() {
    use alice_sdf::svo::SvoChunk;
    let own = svo_chunk().to_bytes();
    for v in [0u32, 1, 3, u32::MAX] {
        let mut b = own.clone();
        b[4..8].copy_from_slice(&v.to_le_bytes());
        assert!(
            matches!(provenance::from_bytes(&b), Err(ProvenanceError::Unreadable(ref why)) if why.contains("is not known")),
            "provenance reads SVO chunk version {v}"
        );
        for result in [
            SvoChunk::from_bytes(&b).map(|_| ()),
            SvoChunk::from_bytes_unverified(&b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if *why == format!("SVO chunk: version {v} is not known")),
                "version {v}: {result:?}"
            );
        }
    }
    for len in [4usize, 7, 8, 39] {
        let b = &own[..len];
        assert!(
            matches!(
                provenance::from_bytes(b),
                Err(ProvenanceError::Unreadable(_))
            ),
            "provenance reads {len} bytes"
        );
        for result in [
            SvoChunk::from_bytes(b).map(|_| ()),
            SvoChunk::from_bytes_unverified(b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why == "SVO chunk: truncated header"),
                "{len} bytes: {result:?}"
            );
        }
    }
    // bytes that are not a chunk and carry no magic: the checked reader refuses them as
    // unstamped, it never parses them
    let mut state = 0x9e37_79b9_u32;
    let random: Vec<u8> = (0..256)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[0]
        })
        .collect();
    assert_ne!(&random[..4], b"ASVC");
    assert!(matches!(
        SvoChunk::from_bytes(&random),
        Err(io::IoError::Unpinned {
            format: "SVO chunk"
        })
    ));
    // a version 1 chunk whose chunk_id is the magic's bytes: read as a version 2 header, refused
    let colliding = alice_sdf::svo::SvoChunk::new(
        u32::from_le_bytes(*b"ASVC"),
        svo_chunk().nodes,
        svo_chunk().bounds,
        2,
    )
    .to_bytes()[40..]
        .to_vec();
    assert_eq!(&colliding[..4], b"ASVC");
    for result in [
        SvoChunk::from_bytes(&colliding).map(|_| ()),
        SvoChunk::from_bytes_unverified(&colliding).map(|_| ()),
    ] {
        // its node_count (2) lands on the version field, so the next 32 bytes are read as an
        // identifier: refused as a mismatch (or, for another node_count, an unknown version)
        assert!(
            matches!(
                result,
                Err(io::IoError::SemanticsMismatch { .. } | io::IoError::InvalidFormat(_))
            ),
            "{result:?}"
        );
    }
}

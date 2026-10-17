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
            io::export_stl_ascii_with_provenance(&m, &path, &caller_id()).unwrap();
        } else {
            io::export_stl_with_provenance(&m, &path, &caller_id()).unwrap();
        }
        let back = io::import_stl(&path).unwrap();
        assert_eq!(back.indices.len() / 3, tris, "{file}");
        std::fs::remove_file(&path).ok();
    }
    let path = temp("i.obj");
    io::export_obj_with_provenance(&m, &path, &io::ObjConfig::default(), None, &caller_id())
        .unwrap();
    assert_eq!(
        io::import_obj(&path).unwrap().indices.len() / 3,
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
        io::export_ply_with_provenance(&m, &path, &c, &caller_id()).unwrap();
        assert_eq!(
            io::import_ply(&path).unwrap().indices.len() / 3,
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
    io::export_fbx_with_provenance(&m, &path, &c, None, &caller_id()).unwrap();
    let back = io::import_fbx(&path).unwrap();
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
    let glb = io::export_glb_bytes_with_provenance(&m, &c, None, &caller_id()).unwrap();
    gltf::Gltf::from_slice(&glb).expect("gltf reads the .glb");
    assert_eq!(io::import_glb_bytes(&glb).unwrap().indices.len() / 3, tris);
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
    io::export_usda_with_provenance(&m, &path, &io::UsdConfig::default(), None, &caller_id())
        .unwrap();
    assert!(!io::import_usda(&path).unwrap().mesh.indices.is_empty());
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
    // a version 2 header with a non-zero reserved byte is not this crate's
    let mut bad = std::fs::read(&path).unwrap();
    bad[44] = 1;
    std::fs::write(&path, &bad).unwrap();
    assert!(matches!(
        provenance::read(&path),
        Err(ProvenanceError::Missing { .. })
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

    let mut v3 = good.clone();
    v3[8..10].copy_from_slice(&3u16.to_le_bytes());
    assert!(matches!(
        provenance::from_bytes(&v3),
        Err(ProvenanceError::Unreadable(why)) if why.contains("version 3")
    ));
    std::fs::write(&path, &v3).unwrap();
    assert!(matches!(
        io::load_abm(&path),
        Err(io::IoError::UnsupportedVersion(3))
    ));

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

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

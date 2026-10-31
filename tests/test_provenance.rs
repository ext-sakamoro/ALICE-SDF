//! Provenance (ALICE-Law G8): every durable writer stores the semantics identifier, and it reads back
//!
//! For each writer in [`WRITERS`]: the plain writer stores [`Provenance::sdf`], the
//! `*_with_provenance` writer stores the caller's identifier, `provenance::read` returns it,
//! `provenance::check` refuses another identifier and names both, and the file still opens in
//! this crate's own importer (and, for binary STL, in an independent STL parser).
//! `scripts/provenance_writers.py` checks that every public writer appears in [`WRITERS`] or in
//! its list of writers still to do.

#[path = "common/temp.rs"]
mod temp;

use alice_sdf::io;
use alice_sdf::io::meshopt_gltf as meshopt;
use alice_sdf::mesh::lod_persist::{self, LodChainConfig, LodChainPersist};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{self, Provenance, ProvenanceError, StoredForm, StoredSemantics};
use std::path::{Path, PathBuf};

/// Lower-case hex of 32 bytes
fn hex_of(id: &[u8; 32]) -> String {
    id.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        write!(s, "{b:02x}").expect("writing to a String");
        s
    })
}

fn temp(name: &str) -> temp::TempPath {
    temp::TempPath::new("alice_sdf_provenance", name)
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

/// The identifier alice-sdf stamped with alice-det-math 0.5.2 at 39507d1b (`6e98f0be…`); this
/// build no longer reproduces it (`PREVIOUS_SDF_SEMANTICS_IDS` is empty), so the loaders refuse
/// it, while the shared judge given it as an earlier identifier reads it as `VerifiedPrevious`
const EARLIER: [u8; 32] = {
    let hex = *b"6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37";
    let mut id = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        id[i] = hex_digit(hex[2 * i]) << 4 | hex_digit(hex[2 * i + 1]);
        i += 1;
    }
    id
};

const fn hex_digit(c: u8) -> u8 {
    if c <= b'9' {
        c - b'0'
    } else {
        c - b'a' + 10
    }
}

const fn earlier_id() -> Provenance {
    Provenance::new(EARLIER)
}

/// A loader refused [`EARLIER`], naming it and this crate's identifier (as meaning identifiers
/// for a form that stores them)
/// The loader refuses a file written under [`caller_id`]: the meaning identifier of the form
/// (ALICE-Law G8) under the caller's identifier, named with the one this build expects
/// The meaning identifier of `form` under `p`, as an identifier (the codecs' errors name them so)
fn meaning_of(form: StoredForm, p: &Provenance) -> Provenance {
    Provenance::new(provenance::meaning_id(form, p))
}

fn refused_as_caller<T: std::fmt::Debug>(result: Result<T, io::IoError>) {
    match result {
        Err(e @ io::IoError::MeaningMismatch { .. }) => {
            let io::IoError::MeaningMismatch {
                form,
                ref stored,
                ref expected,
                ..
            } = e
            else {
                unreachable!()
            };
            let hex = |id: [u8; 32]| Provenance::new(id).hex();
            assert_eq!(*stored, hex(provenance::meaning_id(form, &caller_id())));
            assert_eq!(
                *expected,
                hex(provenance::meaning_id(form, &Provenance::sdf()))
            );
            let text = e.to_string();
            assert!(
                text.contains(stored.as_str()) && text.contains(expected.as_str()),
                "{text}"
            );
        }
        other => panic!("expected MeaningMismatch, got {other:?}"),
    }
}

fn refused_as_earlier<T>(result: Result<T, io::IoError>) {
    match result.err() {
        Some(io::IoError::SemanticsMismatch {
            stored, expected, ..
        }) => assert_eq!((stored, expected), (earlier_id(), Provenance::sdf())),
        Some(io::IoError::MeaningMismatch {
            form,
            stored,
            expected,
            ..
        }) => {
            let hex = |id: [u8; 32]| Provenance::new(id).hex();
            assert_eq!(
                (stored, expected),
                (
                    hex(provenance::meaning_id(form, &earlier_id())),
                    hex(provenance::meaning_id(form, &Provenance::sdf()))
                )
            );
        }
        other => panic!("the earlier identifier was not refused: {other:?}"),
    }
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
    ("src/io/alembic.rs::export_alembic", "m.abc", |m, p, id| {
        let c = io::AlembicConfig::default();
        match id {
            None => io::export_alembic(m, p, &c).unwrap(),
            Some(id) => io::export_alembic_with_provenance(m, p, &c, id).unwrap(),
        }
    }),
    ("src/io/nanite.rs::export_nanite", "m.nanite", |_, p, id| {
        let n = nanite_mesh();
        match id {
            None => io::export_nanite(&n, p).unwrap(),
            Some(id) => io::nanite::export_nanite_with_provenance(
                &n,
                p,
                &io::NaniteExportConfig::default(),
                id,
            )
            .unwrap(),
        }
    }),
    (
        "src/io/nanite.rs::export_nanite_with_config",
        "m_lean.nanite",
        |_, p, id| {
            let n = nanite_mesh();
            let c = io::NaniteExportConfig {
                export_uvs: false,
                export_normals: false,
                quantize_positions: false,
            };
            match id {
                None => io::export_nanite_with_config(&n, p, &c).unwrap(),
                Some(id) => io::nanite::export_nanite_with_provenance(&n, p, &c, id).unwrap(),
            }
        },
    ),
    (
        "src/io/nanite.rs::export_nanite_json",
        "m.nanite.json",
        |_, p, id| {
            let n = nanite_mesh();
            match id {
                None => io::export_nanite_json(&n, p).unwrap(),
                Some(id) => io::nanite::export_nanite_json_with_provenance(&n, p, id).unwrap(),
            }
        },
    ),
    (
        "src/io/ue5_asset.rs::export_ue5_mesh",
        "m.ue5_mesh",
        |m, p, id| {
            let c = io::Ue5MeshConfig::default();
            match id {
                None => io::export_ue5_mesh(m, p, &c).unwrap(),
                Some(id) => io::ue5_asset::export_ue5_mesh_with_provenance(m, p, &c, id).unwrap(),
            }
        },
    ),
    (
        "src/io/ue5_asset.rs::export_ue5_mesh_binary",
        "m.ue5_mesh_bin",
        |m, p, id| {
            let c = io::Ue5MeshConfig::default();
            match id {
                None => io::export_ue5_mesh_binary(m, p, &c).unwrap(),
                Some(id) => {
                    io::ue5_asset::export_ue5_mesh_binary_with_provenance(m, p, &c, id).unwrap();
                }
            }
        },
    ),
    (
        "src/io/ue5_asset.rs::export_ue5_mesh_with_lods",
        "m_lods.ue5_mesh",
        |m, p, id| {
            let c = io::Ue5MeshConfig::default();
            let lods = [m.clone(), m.clone()];
            match id {
                None => io::export_ue5_mesh_with_lods(&lods, &[1.0, 0.5], p, &c).unwrap(),
                Some(id) => io::ue5_asset::export_ue5_mesh_with_lods_with_provenance(
                    &lods,
                    &[1.0, 0.5],
                    p,
                    &c,
                    id,
                )
                .unwrap(),
            }
        },
    ),
    (
        "src/io/unity_mesh.rs::export_unity_mesh",
        "m.unity_mesh",
        |m, p, id| {
            let c = io::UnityMeshConfig::default();
            match id {
                None => io::export_unity_mesh(m, p, &c).unwrap(),
                Some(id) => {
                    io::unity_mesh::export_unity_mesh_with_provenance(m, p, &c, id).unwrap();
                }
            }
        },
    ),
    (
        "src/io/unity_mesh.rs::export_unity_mesh_binary",
        "m.unity_mesh_bin",
        |m, p, id| {
            let c = io::UnityMeshConfig::default();
            match id {
                None => io::export_unity_mesh_binary(m, p, &c).unwrap(),
                Some(id) => {
                    io::unity_mesh::export_unity_mesh_binary_with_provenance(m, p, &c, id).unwrap();
                }
            }
        },
    ),
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
            // the header's slot holds what a writer of the ABM form stores: the meaning identifier
            semantics_id: Some(id.unwrap_or(&Provenance::sdf()).slot_for(StoredForm::Abm)),
        };
        std::fs::write(p, header.to_bytes()).unwrap();
    }),
];

/// Every writer stores the meaning identifier of its form under this crate's identifier, never the
/// raw one (ALICE-Law G8): through the meaning key or in a binary slot
fn stores_the_meaning_identifier(name: &str, stored: &provenance::Stored) {
    assert_eq!(
        stored.id,
        provenance::meaning_id(stored.form, &Provenance::sdf()),
        "{name}: the meaning identifier of {:?}",
        stored.form
    );
    assert_ne!(
        stored.id,
        alice_sdf::SDF_SEMANTICS_ID,
        "{name}: not the raw identifier"
    );
    assert_ne!(
        stored.via,
        provenance::StoredVia::SemanticsKey,
        "{name}: not under the earlier key"
    );
}

#[test]
fn every_writer_stores_the_identifier_and_it_reads_back() {
    let m = mesh();
    assert!(!m.indices.is_empty());
    let mut compared = 0;
    for &(name, file, write) in WRITERS {
        let path = temp(file);
        write(&m, &path, None);
        stores_the_meaning_identifier(name, &provenance::read(&path).unwrap());
        assert!(
            provenance::read(&path)
                .unwrap_or_else(|e| panic!("{name}: {e}"))
                .is_written_under(&Provenance::sdf()),
            "{name}: the plain writer stores this crate's identifier"
        );
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            provenance::StoredSemantics::Current
        );

        write(&m, &path, Some(&caller_id()));
        assert!(
            provenance::read(&path)
                .unwrap()
                .is_written_under(&caller_id()),
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
            // a form that stores meaning identifiers: both meaning identifiers, the form and its version
            Err(e @ ProvenanceError::MeaningMismatch { .. }) => {
                let ProvenanceError::MeaningMismatch {
                    form,
                    version,
                    ref stored,
                    ref expected,
                } = e
                else {
                    unreachable!()
                };
                assert!(form.writes_meaning(), "{name}");
                assert_eq!(version, form.version(), "{name}");
                assert_eq!(
                    *stored,
                    hex_of(&provenance::meaning_id(form, &caller_id())),
                    "{name}"
                );
                assert_eq!(
                    *expected,
                    hex_of(&provenance::meaning_id(form, &Provenance::sdf())),
                    "{name}"
                );
                let text = e.to_string();
                assert!(
                    text.contains(stored.as_str())
                        && text.contains(expected.as_str())
                        && text.contains(&format!("{form:?}")),
                    "{name}: the mismatch names both identifiers and the form: {text}"
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
    assert!(
        provenance::read(&path)
            .unwrap()
            .is_written_under(&caller_id()),
        "obj"
    );
    assert!(
        provenance::read(&mtl)
            .unwrap()
            .is_written_under(&caller_id()),
        "mtl"
    );
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
    // GLB stores the meaning identifier of the caller's identifier, and not the earlier key
    let (key, id) = caller_id().named_for(StoredForm::Glb);
    assert_eq!(key, provenance::MEANING_KEY);
    assert_eq!(json["asset"]["extras"][key].as_str(), Some(id.as_str()));
    assert!(json["asset"]["extras"].get(provenance::KEY).is_none());
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

/// A file stamped with an identifier this crate published before and no longer reproduces
/// (`6e98f0be…`) is refused by `check_sdf` and the importer, the same verdict as
/// `check_sdf_semantics`; the shared judge given it as an earlier identifier reads it as a
/// verified earlier one
#[test]
fn an_earlier_published_identifier_is_refused_and_judged_previous_when_listed() {
    let m = mesh();
    let earlier = EARLIER;
    assert_ne!(earlier, alice_sdf::SDF_SEMANTICS_ID);
    assert!(!alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS.contains(&earlier));
    let path = temp("earlier.stl");
    io::export_stl_with_provenance(&m, &path, &Provenance::new(earlier)).unwrap();
    assert!(matches!(
        provenance::check_sdf(&path),
        Err(ProvenanceError::MeaningMismatch { .. } | ProvenanceError::Mismatch { .. })
    ));
    refused_as_earlier(io::import_stl(&path));
    assert_eq!(
        provenance::check(&path, &Provenance::sdf(), &[earlier]).unwrap(),
        provenance::StoredSemantics::VerifiedPrevious(Provenance::new(earlier))
    );
    assert_eq!(
        alice_sdf::check_sdf_semantics(&earlier),
        alice_det_math::SemanticsCheck::Mismatch
    );
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
        stores_the_meaning_identifier(name, &provenance::read(&path).unwrap());
        assert!(
            provenance::read(&path)
                .unwrap()
                .is_written_under(&Provenance::sdf()),
            "{name}"
        );
        write(p, Some(&caller_id()));
        assert!(
            provenance::read(&path)
                .unwrap()
                .is_written_under(&caller_id()),
            "{name}"
        );
        // a slot or the meaning key holds the caller's meaning identifier: refused naming the form
        assert!(matches!(
            provenance::check_sdf(&path),
            Err(ProvenanceError::MeaningMismatch { .. })
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
                !bytes
                    .windows(8)
                    .any(|w| w == b"semantic" || w == b"meaning_"),
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

    let earlier = earlier_id();
    io::save_asdf_with_provenance(&tree, &path, &earlier).unwrap();
    refused_as_earlier(io::load_asdf(&path));
    refused_as_earlier(io::load_asdf_unverified(&path));
    assert_eq!(
        provenance::check(&path, &Provenance::sdf(), &[EARLIER]).unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    io::save_asdf_with_provenance(&tree, &path, &caller_id()).unwrap();
    for result in [
        io::load_asdf(&path).map(|_| ()),
        io::load_asdf_unverified(&path).map(|_| ()),
    ] {
        refused_as_caller(result);
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
            assert_eq!(
                *s,
                format!(
                    "ALICE-SDF {}",
                    caller_id().tagged_for(StoredForm::FbxBinary)
                )
            );
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
    let meta = text
        .find(&format!("<metadata name=\"{}\">", provenance::MEANING_KEY))
        .unwrap();
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

use alice_sdf::provenance::{Stamp, StoredProducer};

/// The producer read back is the producer meaning identifier of `want` in the file's form, and
/// judges as current against `want`
fn assert_current_producer(
    stored: &provenance::Stored,
    producer: Option<StoredProducer>,
    want: &Provenance,
    what: &str,
) {
    let p = producer.unwrap_or_else(|| panic!("{what}: no producer read back"));
    assert!(p.meaning, "{what}: read as an earlier raw producer");
    assert_eq!(p.form, stored.form, "{what}");
    assert_eq!(
        p.id,
        provenance::producer_meaning_id(stored.form, want),
        "{what}"
    );
    assert_ne!(p.id, stored.id, "{what}: the producer equals the primary");
    assert_eq!(p.judge(want).unwrap(), StoredSemantics::Current, "{what}");
}

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
        "export_glb_bytes_with_stamp",
        "s.glb",
        |m, p, s| {
            let b = io::export_glb_bytes_with_stamp(m, &io::GltfConfig::default(), None, s);
            std::fs::write(p, b.unwrap()).unwrap();
        },
        |m, p, id| {
            let c = io::GltfConfig::default();
            let b = io::export_glb_bytes_with_provenance(m, &c, None, id.unwrap());
            std::fs::write(p, b.unwrap()).unwrap();
        },
    ),
    (
        "export_glb_bytes_with_stamp (meshopt)",
        "s_meshopt.glb",
        |m, p, s| {
            let c = io::GltfConfig {
                meshopt_compress: true,
                ..Default::default()
            };
            std::fs::write(p, io::export_glb_bytes_with_stamp(m, &c, None, s).unwrap()).unwrap();
        },
        |m, p, id| {
            let c = io::GltfConfig {
                meshopt_compress: true,
                ..Default::default()
            };
            let b = io::export_glb_bytes_with_provenance(m, &c, None, id.unwrap());
            std::fs::write(p, b.unwrap()).unwrap();
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
    for p in [path.to_path_buf(), path.with_extension("mtl")] {
        {
            let (stored, producer) = provenance::read_with_producer(&p).unwrap();
            assert!(
                stored.is_written_under(&Provenance::sdf()),
                "{}",
                p.display()
            );
            assert_current_producer(&stored, producer, &producer_id(), &p.display().to_string());
        }
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
        {
            let (stored, producer) = provenance::read_with_producer(&path).unwrap();
            assert!(stored.is_written_under(&Provenance::sdf()), "{name}");
            assert_current_producer(&stored, producer, &producer_id(), name);
        }
        assert!(
            provenance::read(&path)
                .unwrap()
                .is_written_under(&Provenance::sdf()),
            "{name}"
        );
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            StoredSemantics::Current,
            "{name}"
        );
        assert_eq!(
            provenance::check_producer(&path, &producer_id()).unwrap(),
            StoredSemantics::Current,
            "{name}"
        );
        match provenance::check_producer(&path, &caller_id()) {
            Err(ProvenanceError::ProducerMismatch {
                form,
                stored,
                expected,
            }) => {
                // both named as producer meaning identifiers of the file's form
                assert_eq!(
                    (stored, expected),
                    (
                        Provenance::new(provenance::producer_meaning_id(form, &producer_id())),
                        Provenance::new(provenance::producer_meaning_id(form, &caller_id())),
                    ),
                    "{name}"
                );
            }
            other => panic!("{name}: expected a producer mismatch, got {other:?}"),
        }
        // the other assignment: the two must not swap places
        stamped(
            &m,
            &path,
            &Stamp::new(producer_id(), Some(Provenance::sdf())),
        );
        {
            let (stored, producer) = provenance::read_with_producer(&path).unwrap();
            assert!(
                stored.is_written_under(&producer_id()),
                "{name}: swapped assignment"
            );
            assert_current_producer(&stored, producer, &Provenance::sdf(), name);
        }
        // no producer: the same bytes as the provenance writer, and no producer read back
        stamped(&m, &path, &Stamp::from(caller_id()));
        let with_stamp = std::fs::read(&path).unwrap();
        plain(&m, &path, Some(&caller_id()));
        assert_eq!(
            std::fs::read(&path).unwrap(),
            with_stamp,
            "{name}: no producer"
        );
        {
            let (stored, producer) = provenance::read_with_producer(&path).unwrap();
            assert!(stored.is_written_under(&caller_id()), "{name}");
            assert_eq!(producer, None, "{name}");
        }
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

/// The binary STL header with a producer: version 3, bytes 41..48 zero, the producer meaning
/// identifier in 48..80, never starting with `solid`; an all-zero producer is still present (the
/// version says so); an earlier version 2 header (the raw producer in 48..80) reads as the previous
/// generation
#[test]
fn the_binary_stl_header_with_a_producer_has_the_documented_layout() {
    let m = mesh();
    let path = temp("layout.stl");
    io::export_stl_with_stamp(&m, &path, &Stamp::sdf_with_producer(producer_id())).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], provenance::STL_MAGIC);
    assert_eq!(bytes[8], provenance::STL_HEADER_VERSION_PRODUCER_MEANING);
    assert_eq!(
        &bytes[9..41],
        &provenance::meaning_id(StoredForm::StlBinary, &Provenance::sdf())
    );
    assert_eq!(&bytes[41..48], &[0u8; 7]);
    assert_eq!(
        &bytes[48..80],
        &provenance::producer_meaning_id(StoredForm::StlBinary, &producer_id())
    );
    // the earlier layout: version 2 with the raw producer, read as the previous generation
    let mut earlier = bytes.clone();
    earlier[8] = provenance::STL_HEADER_VERSION_PRODUCER;
    earlier[48..80].copy_from_slice(&producer_id().semantics_id);
    {
        let (stored, producer) = provenance::from_bytes_with_producer(&earlier).unwrap();
        assert!(stored.is_written_under(&Provenance::sdf()));
        let producer = producer.expect("an earlier producer");
        assert!(!producer.meaning);
        assert_eq!(producer.id, producer_id().semantics_id);
        assert_eq!(
            producer.judge(&producer_id()).unwrap(),
            StoredSemantics::VerifiedPrevious
        );
        assert!(matches!(
            producer.judge(&caller_id()),
            Err(ProvenanceError::ProducerMismatch { stored, expected, .. })
                if stored == producer_id() && expected == caller_id()
        ));
    }
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
    {
        let (stored, producer) = provenance::read_with_producer(&path).unwrap();
        assert!(stored.is_written_under(&Provenance::sdf()));
        assert_current_producer(
            &stored,
            producer,
            &Provenance::new([0; 32]),
            "zero producer",
        );
    }
    // a version 3 header with a non-zero reserved byte carries this crate's magic but is not a
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
            Provenance::sdf().tagged_for(StoredForm::StlAscii),
            provenance::PRODUCER_MEANING_KEY,
            Provenance::new(provenance::producer_meaning_id(
                StoredForm::StlAscii,
                &producer_id()
            ))
            .hex()
        )
    );
    let only_producer = text.replacen(
        &format!("{} ", Provenance::sdf().tagged_for(StoredForm::StlAscii)),
        "",
        1,
    );
    std::fs::write(&path, only_producer).unwrap();
    assert!(matches!(
        provenance::read(&path),
        Err(ProvenanceError::Missing { .. })
    ));
    std::fs::remove_file(&path).ok();
}

/// The independent parsers still read the stamped files: `stl_io` the version 3 binary STL,
/// `fbxcel` the binary FBX (with `ProducerMeaningId` beside `Creator`), and the 3MF model keeps
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
        format!(
            "ALICE-SDF {}",
            Provenance::sdf().tagged_for(StoredForm::FbxBinary)
        )
    );
    assert_eq!(
        string_of("ProducerMeaningId"),
        format!(
            "{}={}",
            provenance::PRODUCER_MEANING_KEY,
            Provenance::new(provenance::producer_meaning_id(
                StoredForm::FbxBinary,
                &producer_id()
            ))
            .hex()
        )
    );
    std::fs::remove_file(&fbx).ok();

    let tmf = temp("order_stamp.3mf");
    io::export_3mf_with_stamp(&m, &tmf, &stamp).unwrap();
    let bytes = std::fs::read(&tmf).unwrap();
    let text = String::from_utf8_lossy(&bytes);
    let primary = text
        .find(&format!("<metadata name=\"{}\">", provenance::MEANING_KEY))
        .unwrap();
    let producer = text
        .find(&format!(
            "<metadata name=\"{}\">",
            provenance::PRODUCER_MEANING_KEY
        ))
        .unwrap();
    let resources = text.find("<resources>").unwrap();
    assert!(primary < producer && producer < resources);
    std::fs::remove_file(&tmf).ok();
}

/// The stamped files main wrote before the producer stored a meaning identifier (the STAMPED
/// writers at 7b8ddf9 with `Stamp::sdf_with_producer(producer_id())`): each holds the raw producer
/// (binary STL header version 2, key `alice:producer_semantics_id`, FBX `ProducerSemanticsId`), read
/// as the previous generation against that producer, a mismatch against another, both named raw
const PRODUCER_RAW_MAIN: [&str; 6] = [
    "producer_raw_main.stl",
    "producer_raw_main_ascii.stl",
    "producer_raw_main.3mf",
    "producer_raw_main_bin.fbx",
    "producer_raw_main.obj",
    "producer_raw_main_ascii.fbx",
];

#[test]
fn an_earlier_raw_producer_reads_as_the_previous_generation() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/provenance");
    let mut read = 0;
    for file in PRODUCER_RAW_MAIN {
        let path = dir.join(file);
        let (_, producer) = provenance::read_with_producer(&path).unwrap();
        let producer = producer.unwrap_or_else(|| panic!("{file}: no producer"));
        assert!(!producer.meaning, "{file}: read as a meaning identifier");
        assert_eq!(producer.id, producer_id().semantics_id, "{file}");
        assert_eq!(
            provenance::check_producer(&path, &producer_id()).unwrap(),
            StoredSemantics::VerifiedPrevious,
            "{file}"
        );
        match provenance::check_producer(&path, &caller_id()) {
            Err(ProvenanceError::ProducerMismatch {
                stored, expected, ..
            }) => assert_eq!((stored, expected), (producer_id(), caller_id()), "{file}"),
            other => panic!("{file}: expected a producer mismatch, got {other:?}"),
        }
        read += 1;
    }
    assert_eq!(read, PRODUCER_RAW_MAIN.len());
}

/// A single stored zip entry (method 0, sizes in the local header), as the 3MF reader reads them
fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut z = Vec::new();
    z.extend_from_slice(b"PK\x03\x04");
    z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let len = u32::try_from(data.len()).unwrap().to_le_bytes();
    z.extend_from_slice(&len);
    z.extend_from_slice(&len);
    z.extend_from_slice(&u16::try_from(name.len()).unwrap().to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z.extend_from_slice(name.as_bytes());
    z.extend_from_slice(data);
    z
}

/// A file holding both the current and the earlier producer key is refused, in every text form, the
/// binary FBX and the 3MF model; with either key alone it reads
#[test]
fn a_file_with_both_producer_keys_is_refused() {
    let m = mesh();
    let stamp = Stamp::sdf_with_producer(producer_id());
    let raw = format!("{}={}", provenance::PRODUCER_KEY, producer_id().hex());
    let mut refused = 0;
    for &(name, file, stamped, _) in STAMPED {
        if file.ends_with(".stl") && !file.contains("ascii")
            || file.ends_with(".3mf")
            || file.ends_with(".glb")
        {
            continue; // binary STL: the header version decides; 3MF and glTF: below
        }
        let path = temp(file);
        stamped(&m, &path, &stamp);
        let mut both = std::fs::read(&path).unwrap();
        provenance::from_bytes_with_producer(&both).unwrap();
        both.extend_from_slice(format!("\n# {raw}\n").as_bytes());
        assert!(
            matches!(
                provenance::from_bytes_with_producer(&both),
                Err(ProvenanceError::Unreadable(ref e)) if e.contains("both")
            ),
            "{name}"
        );
        std::fs::remove_file(&path).ok();
        refused += 1;
    }
    assert_eq!(refused, 4, "ASCII STL, binary and ASCII FBX, OBJ");
    let primary = format!(
        "<metadata name=\"{}\">{}</metadata>",
        provenance::MEANING_KEY,
        Provenance::new(provenance::meaning_id(
            StoredForm::ThreeMf,
            &Provenance::sdf()
        ))
        .hex()
    );
    let current = format!(
        "<metadata name=\"{}\">{}</metadata>",
        provenance::PRODUCER_MEANING_KEY,
        Provenance::new(provenance::producer_meaning_id(
            StoredForm::ThreeMf,
            &producer_id()
        ))
        .hex()
    );
    let earlier = format!(
        "<metadata name=\"{}\">{}</metadata>",
        provenance::PRODUCER_KEY,
        producer_id().hex()
    );
    let model = |entries: &str| {
        stored_zip(
            "3D/3dmodel.model",
            format!("<model>{entries}</model>").as_bytes(),
        )
    };
    let (_, p) =
        provenance::from_bytes_with_producer(&model(&format!("{primary}{current}"))).unwrap();
    assert_eq!(
        p.map(|p| (p.meaning, p.judge(&producer_id()).unwrap())),
        Some((true, StoredSemantics::Current))
    );
    let (_, p) =
        provenance::from_bytes_with_producer(&model(&format!("{primary}{earlier}"))).unwrap();
    assert_eq!(
        p.map(|p| (p.meaning, p.judge(&producer_id()).unwrap())),
        Some((false, StoredSemantics::VerifiedPrevious))
    );
    assert!(matches!(
        provenance::from_bytes_with_producer(&model(&format!("{primary}{current}{earlier}"))),
        Err(ProvenanceError::Unreadable(ref e)) if e.contains("both")
    ));
    // glTF: the keys of asset.extras
    let gltf = |extras: &str| {
        format!(
            r#"{{"asset":{{"version":"2.0","extras":{{"{}":"{}"{extras}}}}}}}"#,
            provenance::MEANING_KEY,
            Provenance::new(provenance::meaning_id(StoredForm::Gltf, &Provenance::sdf())).hex()
        )
        .into_bytes()
    };
    let current = format!(
        r#","{}":"{}""#,
        provenance::PRODUCER_MEANING_KEY,
        Provenance::new(provenance::producer_meaning_id(
            StoredForm::Gltf,
            &producer_id()
        ))
        .hex()
    );
    let earlier = format!(
        r#","{}":"{}""#,
        provenance::PRODUCER_KEY,
        producer_id().hex()
    );
    let (_, p) = provenance::from_bytes_with_producer(&gltf(&current)).unwrap();
    assert_eq!(
        p.map(|p| (p.meaning, p.judge(&producer_id()).unwrap())),
        Some((true, StoredSemantics::Current))
    );
    let (_, p) = provenance::from_bytes_with_producer(&gltf(&earlier)).unwrap();
    assert_eq!(
        p.map(|p| (p.meaning, p.judge(&producer_id()).unwrap())),
        Some((false, StoredSemantics::VerifiedPrevious))
    );
    assert!(matches!(
        provenance::from_bytes_with_producer(&gltf(&format!("{current}{earlier}"))),
        Err(ProvenanceError::Unreadable(ref e)) if e.contains("both")
    ));
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

    let earlier = earlier_id();
    write(&earlier);
    refused_as_earlier(io::load_abm(&single));
    refused_as_earlier(io::load_abm_with_lods(&lods));
    refused_as_earlier(io::load_abm_unverified(&single));
    refused_as_earlier(io::load_abm_with_lods_unverified(&lods));
    for path in [&single, &lods] {
        assert_eq!(
            provenance::check(path, &Provenance::sdf(), &[EARLIER]).unwrap(),
            StoredSemantics::VerifiedPrevious(earlier)
        );
    }

    write(&caller_id());
    for result in [
        io::load_abm(&single).map(|_| ()),
        io::load_abm_unverified(&single).map(|_| ()),
        io::load_abm_with_lods(&lods).map(|_| ()),
        io::load_abm_with_lods_unverified(&lods).map(|_| ()),
    ] {
        refused_as_caller(result);
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

/// Both parts of a chain carry the identifier: this crate's reads as `Current`, an earlier one
/// this build no longer reproduces in either part is refused (the shared judge given it reads
/// that part as `VerifiedPrevious`), another one in either part is refused naming both, and the
/// unverified loader reports the weaker of the two parts
#[test]
fn lod_chain_loaders_check_both_parts() {
    let m = abm_fixture_mesh(1.0);
    let chain = lod_chain(&m);
    let path = temp("check_chain.abm");
    let meta = path.with_file_name("check_chain.abm.meta.json");
    let earlier = earlier_id();

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

    // the sidecar written under the earlier identifier: the chain is refused, the sidecar alone
    // judged against it is VerifiedPrevious
    lod_persist::save_lod_chain_metadata_with_provenance(&chain, &path, &earlier).unwrap();
    refused_as_earlier(lod_persist::load_lod_chain(&path));
    refused_as_earlier(lod_persist::load_lod_chain_unverified(&path));
    assert_eq!(
        provenance::check(&meta, &Provenance::sdf(), &[EARLIER]).unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    // the mesh file under the earlier identifier, the sidecar current: refused through the mesh
    // file, never read as Current
    lod_persist::save_lod_chain_with_provenance(&chain, &path, &earlier).unwrap();
    lod_persist::save_lod_chain_metadata(&chain, &path).unwrap();
    refused_as_earlier(lod_persist::load_lod_chain(&path));
    refused_as_earlier(lod_persist::load_lod_chain_unverified(&path));
    assert_eq!(
        provenance::check(&path, &Provenance::sdf(), &[EARLIER]).unwrap(),
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
        refused_as_caller(result);
    }
    lod_persist::save_lod_chain_with_provenance(&chain, &path, &caller_id()).unwrap();
    lod_persist::save_lod_chain_metadata(&chain, &path).unwrap();
    assert!(matches!(
        lod_persist::load_lod_chain(&path),
        Err(io::IoError::MeaningMismatch { .. })
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
    // this build writes the meaning identifier of a JSON document, never the raw key
    assert_eq!(
        good[provenance::MEANING_KEY],
        Provenance::sdf().named_for(StoredForm::JsonDocument).1
    );
    assert!(good.get(provenance::KEY).is_none());

    let mut cases = Vec::new();
    let mut v = good.clone();
    v.as_object_mut().unwrap().remove(provenance::MEANING_KEY);
    cases.push(("format 2 without the identifier", v));
    let mut v = good.clone();
    v[provenance::MEANING_KEY] = serde_json::Value::from("abc");
    cases.push(("a short identifier", v));
    // both keys: read as neither
    let mut v = good.clone();
    v[provenance::KEY] = serde_json::Value::from(Provenance::sdf().hex());
    cases.push(("both keys", v));
    for f in [0, 3, 1000] {
        let mut v = good.clone();
        v["format"] = serde_json::Value::from(f);
        cases.push(("an unknown format", v.clone()));
        v.as_object_mut().unwrap().remove(provenance::MEANING_KEY);
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
    ("src/io/vox.rs::save_vox", "MagicaVoxel .vox"),
    (
        "src/io/splat.rs::save_splat",
        ".splat with its .semantics sidecar",
    ),
    ("src/io/vdb.rs::bake_to_vdb", "ALICE-VDB dense grid"),
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
    assert!(provenance::from_bytes(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let back = NeuralSdf::load(&mut own.as_slice()).unwrap();
    assert_eq!(net_bits(&back), net_bits(&small_net()));
    assert_eq!(
        NeuralSdf::load_unverified(&mut own.as_slice())
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    let earlier = earlier_id();
    let prev = nsdf_bytes(Some(&earlier));
    assert!(provenance::from_bytes(&prev)
        .unwrap()
        .is_written_under(&earlier));
    refused_as_earlier(NeuralSdf::load(&mut prev.as_slice()));
    refused_as_earlier(NeuralSdf::load_unverified(&mut prev.as_slice()));
    assert_eq!(
        provenance::from_bytes(&prev)
            .unwrap()
            .judge(&Provenance::sdf(), &[EARLIER])
            .unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = nsdf_bytes(Some(&caller_id()));
    assert!(provenance::from_bytes(&other)
        .unwrap()
        .is_written_under(&caller_id()));
    for result in [
        NeuralSdf::load(&mut other.as_slice()).map(|_| ()),
        NeuralSdf::load_unverified(&mut other.as_slice()).map(|_| ()),
    ] {
        refused_as_caller(result);
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
    assert!(STORED_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/io/vdb.rs::bake_to_vdb"));
    let own = vdb_bytes(None);
    assert_eq!(&own[..9], b"ALICEVDB2");
    assert!(provenance::from_bytes(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    assert_eq!(
        vdb::load_dense_grid_from_vdb_unverified(&own)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );

    let earlier = earlier_id();
    let prev = vdb_bytes(Some(&earlier));
    assert!(provenance::from_bytes(&prev)
        .unwrap()
        .is_written_under(&earlier));
    refused_as_earlier(vdb::load_dense_grid_from_vdb(&prev));
    refused_as_earlier(vdb::load_dense_grid_from_vdb_unverified(&prev));
    assert_eq!(
        provenance::from_bytes(&prev)
            .unwrap()
            .judge(&Provenance::sdf(), &[EARLIER])
            .unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = vdb_bytes(Some(&caller_id()));
    assert!(provenance::from_bytes(&other)
        .unwrap()
        .is_written_under(&caller_id()));
    for result in [
        vdb::load_dense_grid_from_vdb(&other).map(|_| ()),
        vdb::load_dense_grid_from_vdb_unverified(&other).map(|_| ()),
    ] {
        refused_as_caller(result);
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
    assert_eq!(&own[8..40], &Provenance::sdf().slot_for(StoredForm::Svo));
    assert!(provenance::from_bytes(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    assert!(
        provenance::from_bytes(&svo_chunk().to_bytes_with_provenance(&caller_id()))
            .unwrap()
            .is_written_under(&caller_id())
    );
    assert_eq!(SvoChunk::from_bytes(&own).unwrap().to_bytes(), own);
    assert_eq!(
        SvoChunk::from_bytes_unverified(&own).unwrap().semantics(),
        StoredSemantics::Current
    );

    let earlier = earlier_id();
    let prev = svo_chunk().to_bytes_with_provenance(&earlier);
    refused_as_earlier(SvoChunk::from_bytes(&prev));
    refused_as_earlier(SvoChunk::from_bytes_unverified(&prev));
    assert_eq!(
        provenance::from_bytes(&prev)
            .unwrap()
            .judge(&Provenance::sdf(), &[EARLIER])
            .unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = svo_chunk().to_bytes_with_provenance(&caller_id());
    for result in [
        SvoChunk::from_bytes(&other).map(|_| ()),
        SvoChunk::from_bytes_unverified(&other).map(|_| ()),
    ] {
        refused_as_caller(result);
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
    // a node_count far beyond the data (its byte length wraps usize on a 32-bit target): the
    // body is cut short, never indexed past
    for count in [3u32, 1 << 27, u32::MAX] {
        let mut b = own.clone();
        b[44..48].copy_from_slice(&count.to_le_bytes());
        for result in [
            SvoChunk::from_bytes(&b).map(|_| ()),
            SvoChunk::from_bytes_unverified(&b).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref why)) if why == "SVO chunk: truncated body"),
                "node_count {count}: {result:?}"
            );
        }
    }
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
        // identifier: refused as a meaning mismatch of the SVO form (or, for another
        // node_count, an unknown version)
        assert!(
            matches!(
                result,
                Err(io::IoError::MeaningMismatch { .. } | io::IoError::InvalidFormat(_))
            ),
            "{result:?}"
        );
    }
}

// MagicaVoxel `.vox` (`io::vox`): model_v1.vox
//   612f6227cb642393edea6751a297133072233ce5cad4d2454de006bff3cd8fdc
//   save_vox of size (4, 4, 4), voxels (1, 2, 3, colour 79) and (0, 0, 0, colour 1) by alice-sdf
//   6.1.0 (no ALSD chunk)

fn vox_model() -> alice_sdf::io::vox::VoxModel {
    use alice_sdf::io::vox::{VoxModel, Voxel};
    VoxModel {
        size: (4, 4, 4),
        voxels: vec![
            Voxel {
                x: 1,
                y: 2,
                z: 3,
                color: 79,
            },
            Voxel {
                x: 0,
                y: 0,
                z: 0,
                color: 1,
            },
        ],
    }
}

/// (x, y, z, colour) of each voxel
type VoxCells = Vec<(u8, u8, u8, u8)>;

fn vox_voxels(m: &alice_sdf::io::vox::VoxModel) -> VoxCells {
    m.voxels.iter().map(|v| (v.x, v.y, v.z, v.color)).collect()
}

/// The voxels as a reader that knows only SIZE and XYZI sees them: it walks MAIN's children and
/// skips every chunk it does not know, as MagicaVoxel does
fn vox_read_skipping_unknown_chunks(b: &[u8]) -> ((u32, u32, u32), VoxCells) {
    let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
    assert_eq!(&b[0..4], b"VOX ");
    assert_eq!(&b[8..12], b"MAIN");
    let end = 20 + u32_at(16) as usize;
    assert_eq!(
        end,
        b.len(),
        "MAIN's children size covers the rest of the file"
    );
    let mut size = (0, 0, 0);
    let mut voxels = Vec::new();
    let mut off = 20;
    while off < end {
        let id = &b[off..off + 4];
        let content = u32_at(off + 4) as usize;
        let children = u32_at(off + 8) as usize;
        let body = off + 12;
        if id == b"SIZE" {
            size = (u32_at(body), u32_at(body + 4), u32_at(body + 8));
        } else if id == b"XYZI" {
            let n = u32_at(body) as usize;
            voxels = (0..n)
                .map(|i| {
                    let q = body + 4 + 4 * i;
                    (b[q], b[q + 1], b[q + 2], b[q + 3])
                })
                .collect();
        }
        off = body + content + children;
    }
    assert_eq!(off, end);
    (size, voxels)
}

/// The `.vox` written by alice-sdf 6.1.0 (no `ALSD` chunk): MagicaVoxel writes none either, so
/// `load_vox` reads it marked `Unpinned` and `load_vox_pinned` refuses it; the bytes this build
/// writes for the same model are the 6.1.0 bytes with the `ALSD` chunk appended to MAIN's children
#[test]
fn a_vox_from_6_1_0_reads_as_unpinned_and_is_refused_when_pinned() {
    use alice_sdf::io::vox::{load_vox, load_vox_pinned, save_vox};
    let path = fixture("model_v1.vox");
    let old = std::fs::read(&path).unwrap();
    assert!(
        !old.windows(4).any(|w| w == b"ALSD"),
        "fixture carries no identifier"
    );
    assert!(matches!(
        provenance::from_bytes(&old),
        Err(ProvenanceError::Missing { format: "VOX" })
    ));
    let checked = load_vox(&path).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    let (model, _) = checked.into_parts();
    assert_eq!(model.size, vox_model().size);
    assert_eq!(vox_voxels(&model), vox_voxels(&vox_model()));
    match load_vox_pinned(&path) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "VOX"),
        other => panic!("expected Unpinned, got {:?}", other.map(|_| ())),
    }

    let own_path = temp("model_v2.vox");
    save_vox(&own_path, &vox_model()).unwrap();
    let own = std::fs::read(&own_path).unwrap();
    assert_eq!(own.len(), old.len() + 48);
    assert_eq!(&own[..16], &old[..16]);
    let main_children = |b: &[u8]| u32::from_le_bytes(b[16..20].try_into().unwrap());
    assert_eq!(main_children(&own), main_children(&old) + 48);
    assert_eq!(&own[20..old.len()], &old[20..]);
}

/// `save_vox` stores this crate's identifier in the `ALSD` chunk, `save_vox_with_provenance` the
/// caller's; this crate's reads as `Current`, an earlier verified one as `VerifiedPrevious`,
/// another one is refused by both loaders naming both identifiers; a reader that skips unknown
/// chunks still reads the voxels; removing the chunk gives a file without an identifier
#[test]
fn vox_writers_store_and_loaders_check_the_identifier() {
    use alice_sdf::io::vox::{load_vox, load_vox_pinned, save_vox, save_vox_with_provenance};
    assert!(STORED_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/io/vox.rs::save_vox"));
    let own_path = temp("own.vox");
    save_vox(&own_path, &vox_model()).unwrap();
    let own = std::fs::read(&own_path).unwrap();
    assert!(provenance::from_bytes(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    assert_eq!(
        load_vox(&own_path).unwrap().semantics(),
        StoredSemantics::Current
    );
    assert_eq!(
        vox_voxels(&load_vox_pinned(&own_path).unwrap()),
        vox_voxels(&vox_model())
    );
    assert_eq!(
        vox_read_skipping_unknown_chunks(&own),
        ((4, 4, 4), vox_voxels(&vox_model()))
    );

    let earlier = earlier_id();
    let prev_path = temp("prev.vox");
    save_vox_with_provenance(&prev_path, &vox_model(), &earlier).unwrap();
    refused_as_earlier(load_vox_pinned(&prev_path));
    refused_as_earlier(load_vox(&prev_path));
    assert_eq!(
        provenance::check(&prev_path, &Provenance::sdf(), &[EARLIER]).unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other_path = temp("other.vox");
    save_vox_with_provenance(&other_path, &vox_model(), &caller_id()).unwrap();
    assert!(provenance::from_bytes(&std::fs::read(&other_path).unwrap())
        .unwrap()
        .is_written_under(&caller_id()));
    for result in [
        load_vox(&other_path).map(|_| ()),
        load_vox_pinned(&other_path).map(|_| ()),
    ] {
        refused_as_caller(result);
    }

    // the same file with the ALSD chunk cut out (and MAIN's children size fixed up)
    let at = own.windows(4).position(|w| w == b"ALSD").unwrap();
    let mut stripped = [&own[..at], &own[at + 48..]].concat();
    let children = u32::from_le_bytes(stripped[16..20].try_into().unwrap()) - 48;
    stripped[16..20].copy_from_slice(&children.to_le_bytes());
    let stripped_path = temp("stripped.vox");
    std::fs::write(&stripped_path, &stripped).unwrap();
    assert_eq!(
        load_vox(&stripped_path).unwrap().semantics(),
        StoredSemantics::Unpinned
    );
    assert!(matches!(
        load_vox_pinned(&stripped_path),
        Err(io::IoError::Unpinned { format: "VOX" })
    ));
}

/// An `ALSD` chunk that is present but cannot be read (cut short, another length, children,
/// another layout version, or twice) is an error of the format, never a file without an identifier
#[test]
fn vox_identifier_chunks_that_cannot_be_read_are_errors_never_unpinned() {
    use alice_sdf::io::vox::{load_vox, load_vox_pinned, save_vox};
    let path = temp("bad_base.vox");
    save_vox(&path, &vox_model()).unwrap();
    let own = std::fs::read(&path).unwrap();
    let at = own.windows(4).position(|w| w == b"ALSD").unwrap();
    assert_eq!(at + 48, own.len(), "ALSD is the last chunk");
    let set = |b: &mut [u8], o: usize, v: u32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());

    // MAIN's children size set to what the bytes hold, so the check under test is the ALSD one
    let fit = |mut b: Vec<u8>| {
        let n = u32::try_from(b.len() - 20).unwrap();
        b[16..20].copy_from_slice(&n.to_le_bytes());
        b
    };

    let mut cases: Vec<(String, Vec<u8>, &str)> = [2usize, 4, 11, 12, 20, 47]
        .iter()
        .map(|&cut| {
            let why = if cut < 12 {
                "truncated header"
            } else {
                "truncated body"
            };
            (
                format!("cut to {cut} bytes"),
                fit(own[..at + cut].to_vec()),
                why,
            )
        })
        .collect();
    let mut b = own.clone();
    set(&mut b, at + 4, 35);
    cases.extend([("content 35".to_string(), b, "length 35")]);
    let mut b = own.clone();
    set(&mut b, at + 8, 4);
    cases.extend([("children 4".to_string(), b, "children 4")]);
    let mut b = own.clone();
    set(&mut b, at + 12, 2);
    cases.extend([("layout 2".to_string(), b, "layout version 2 is not known")]);
    let twice = fit([&own[..], &own[at..]].concat());
    cases.extend([("twice".to_string(), twice, "more than one ALSD chunk")]);
    // a stamped file cut short (inside XYZI, or inside ALSD) keeps MAIN's size of the whole file:
    // an error, never a partial model read as unstamped
    cases.extend([
        (
            "stamped, cut in XYZI".to_string(),
            own[..at - 2].to_vec(),
            "MAIN children size",
        ),
        (
            "stamped, cut in ALSD".to_string(),
            own[..at + 20].to_vec(),
            "MAIN children size",
        ),
        (
            "stamped, cut in XYZI, MAIN refitted".to_string(),
            fit(own[..at - 2].to_vec()),
            "chunk XYZI at byte",
        ),
    ]);
    // the same for a file without the chunk (6.1.0, as MagicaVoxel writes it)
    let old = std::fs::read(fixture("model_v1.vox")).unwrap();
    cases.extend([(
        "6.1.0 file cut in XYZI".to_string(),
        old[..old.len() - 3].to_vec(),
        "MAIN children size",
    )]);
    assert_eq!(cases.len(), 14);

    for (name, bytes, why) in cases {
        let from = provenance::from_bytes(&bytes);
        assert!(
            matches!(from, Err(ProvenanceError::Unreadable(ref e)) if e.contains(why)),
            "provenance, {name}: {from:?}"
        );
        let p = temp("bad.vox");
        std::fs::write(&p, &bytes).unwrap();
        for result in [load_vox(&p).map(|_| ()), load_vox_pinned(&p).map(|_| ())] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref e)) if e.starts_with("VOX: ") && e.contains(why)),
                "{name}: {result:?}"
            );
        }
    }
}

// Gaussian splats `.splat` (`io::splat`): cloud_v1.splat
//   d987565d4725012cb6ae108b8029caef7add6fbf82f8218af83b177ab97cfe01
//   save_splat of the two splats of `splats()` by alice-sdf 6.1.0 (no sidecar)

fn splats() -> Vec<alice_sdf::io::splat::Splat> {
    use alice_sdf::io::splat::Splat;
    vec![
        Splat {
            position: [0.5, -0.25, 1.0],
            scale: [0.1, 0.1, 0.1],
            color: [200, 200, 220, 255],
            rotation: [128, 128, 255, 128],
        },
        Splat {
            position: [-1.0, 0.0, 0.0],
            scale: [0.2, 0.05, 0.1],
            color: [10, 20, 30, 40],
            rotation: [0, 255, 128, 64],
        },
    ]
}

/// (record encoder as the scanner reads it, why it stores no identifier, who else calls it): a
/// fixed-size record of a third-party format; the identifier belongs to the file, which the
/// writer named here stamps through its sidecar
const RECORD_CODECS: &[(&str, &str, &str)] = &[(
    "src/io/splat.rs::to_bytes",
    "one 32-byte .splat record (antimatter15 layout, no room for a header); save_splat writes the records and the sidecar",
    "server/src/main.rs (the HTTP .splat body, sent without a sidecar) and ALICE-LOL's sdf_writers.lock",
)];

fn splat_bytes(v: &[alice_sdf::io::splat::Splat]) -> Vec<u8> {
    v.iter()
        .flat_map(alice_sdf::io::splat::Splat::to_bytes)
        .collect()
}

/// The `.splat` written by alice-sdf 6.1.0 has no sidecar (no viewer writes one either):
/// `load_splat` reads it marked `Unpinned`, `load_splat_pinned` refuses it; this build writes the
/// same `.splat` bytes and a sidecar beside them
#[test]
fn a_splat_from_6_1_0_reads_as_unpinned_and_is_refused_when_pinned() {
    use alice_sdf::io::splat::{load_splat, load_splat_pinned, save_splat};
    use sha2::{Digest, Sha256};
    let path = fixture("cloud_v1.splat");
    let old = std::fs::read(&path).unwrap();
    assert_eq!(
        Provenance::new(Sha256::digest(&old).into()).hex(),
        "d987565d4725012cb6ae108b8029caef7add6fbf82f8218af83b177ab97cfe01"
    );
    assert!(!provenance::sidecar_path(&path).exists());
    let checked = load_splat(&path).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Unpinned);
    assert_eq!(splat_bytes(&checked.into_parts().0), old);
    match load_splat_pinned(&path) {
        Err(io::IoError::Unpinned { format }) => assert_eq!(format, "splat"),
        other => panic!("expected Unpinned, got {:?}", other.map(|_| ())),
    }

    let own = temp("cloud_v2.splat");
    save_splat(&own, &splats()).unwrap();
    assert_eq!(
        std::fs::read(&own).unwrap(),
        old,
        "the .splat bytes are unchanged"
    );
    let sidecar = std::fs::read_to_string(provenance::sidecar_path(&own)).unwrap();
    assert_eq!(
        sidecar,
        format!(
            "{}\nalice:content_sha256={}\n",
            Provenance::sdf().tagged_for(StoredForm::Sidecar),
            Provenance::new(Sha256::digest(&old).into()).hex()
        )
    );
}

/// `save_splat` stores this crate's identifier and the file's SHA-256 in the sidecar,
/// `save_splat_with_provenance` the caller's identifier; this crate's reads as `Current`, an
/// earlier verified one as `VerifiedPrevious`, another one is refused by both loaders naming both
/// identifiers; the record encoder is the writer's records, its other callers named
#[test]
fn splat_writers_store_and_loaders_check_the_identifier() {
    use alice_sdf::io::splat::{
        load_splat, load_splat_pinned, save_splat, save_splat_with_provenance,
    };
    assert!(STORED_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/io/splat.rs::save_splat"));
    let own = temp("own.splat");
    save_splat(&own, &splats()).unwrap();
    assert_eq!(std::fs::read(&own).unwrap(), splat_bytes(&splats()));
    assert_eq!(RECORD_CODECS.len(), 1);
    assert!(RECORD_CODECS[0].2.contains("server/src/main.rs"));
    assert!(provenance::read(&own)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    assert_eq!(
        load_splat(&own).unwrap().semantics(),
        StoredSemantics::Current
    );
    assert_eq!(
        splat_bytes(&load_splat_pinned(&own).unwrap()),
        splat_bytes(&splats())
    );

    let earlier = earlier_id();
    let prev = temp("prev.splat");
    save_splat_with_provenance(&prev, &splats(), &earlier).unwrap();
    refused_as_earlier(load_splat_pinned(&prev));
    refused_as_earlier(load_splat(&prev));
    assert_eq!(
        provenance::check(&prev, &Provenance::sdf(), &[EARLIER]).unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );

    let other = temp("other.splat");
    save_splat_with_provenance(&other, &splats(), &caller_id()).unwrap();
    assert!(provenance::read(&other)
        .unwrap()
        .is_written_under(&caller_id()));
    for result in [
        load_splat(&other).map(|_| ()),
        load_splat_pinned(&other).map(|_| ()),
    ] {
        refused_as_caller(result);
    }

    // the .splat rewritten by another program after its sidecar: provenance::read refuses the
    // sidecar's identifier as it does load_splat, never reports it as the file's
    let rewritten = temp("rewritten.splat");
    save_splat(&rewritten, &splats()).unwrap();
    std::fs::write(&rewritten, splat_bytes(&splats()[..1])).unwrap();
    assert!(matches!(
        provenance::read(&rewritten),
        Err(ProvenanceError::Unreadable(ref e)) if e.contains("rewritten after its sidecar")
    ));

    // the SHA-256 line removed from a .splat sidecar: load_splat refuses it, and so does read()
    let unbound = temp("unbound.splat");
    save_splat(&unbound, &splats()).unwrap();
    let sidecar = std::fs::read_to_string(provenance::sidecar_path(&unbound)).unwrap();
    let id_only: String = sidecar
        .lines()
        .filter(|l| !l.starts_with("alice:content_sha256="))
        .flat_map(|l| [l, "\n"])
        .collect();
    std::fs::write(provenance::sidecar_path(&unbound), id_only).unwrap();
    for e in [
        provenance::read(&unbound)
            .map(|_| ())
            .unwrap_err()
            .to_string(),
        load_splat(&unbound).map(|_| ()).unwrap_err().to_string(),
    ] {
        assert!(e.contains("has no alice:content_sha256"), "{e}");
    }
    // the same under other names (a .splat has no magic, the name is all there is): refused too
    for name in ["unbound.bin", "unbound", "unbound.gsplat", "other.SPLAT"] {
        let p = temp(name);
        std::fs::copy(&unbound, &p).unwrap();
        std::fs::copy(
            provenance::sidecar_path(&unbound),
            provenance::sidecar_path(&p),
        )
        .unwrap();
        let e = provenance::read(&p).map(|_| ()).unwrap_err().to_string();
        assert!(e.contains("has no alice:content_sha256"), "{name}: {e}");
    }
    assert_eq!(provenance::PLAIN_SIDECAR_EXTENSIONS, ["raw", "dds"]);

    // the sidecar removed: a file without an identifier
    std::fs::remove_file(provenance::sidecar_path(&own)).unwrap();
    assert_eq!(
        load_splat(&own).unwrap().semantics(),
        StoredSemantics::Unpinned
    );
    assert!(matches!(
        load_splat_pinned(&own),
        Err(io::IoError::Unpinned { format: "splat" })
    ));
}

/// A sidecar that is present but cannot be read, or whose SHA-256 is not the file's (the `.splat`
/// was rewritten by another program after its sidecar), is an error of the format, never a file
/// without an identifier
#[test]
fn splat_sidecars_that_cannot_be_read_or_describe_other_bytes_are_errors_never_unpinned() {
    use alice_sdf::io::splat::{load_splat, load_splat_pinned, save_splat};
    let base = temp("bad_base.splat");
    save_splat(&base, &splats()).unwrap();
    let good = std::fs::read_to_string(provenance::sidecar_path(&base)).unwrap();
    let (id_line, sha_line) = good.split_once('\n').unwrap();
    let sha_line = sha_line.trim_end();
    let id_hex = Provenance::sdf().named_for(StoredForm::Sidecar).1;

    let mut cases: Vec<(&str, Vec<u8>, String, &str)> = vec![
        (
            "rewritten file",
            splat_bytes(&splats()[..1]),
            good.clone(),
            "rewritten after its sidecar",
        ),
        (
            "no id",
            splat_bytes(&splats()),
            format!("{sha_line}\n"),
            "has no alice:meaning_id",
        ),
        (
            "no sha",
            splat_bytes(&splats()),
            format!("{id_line}\n"),
            "has no alice:content_sha256",
        ),
        (
            "id twice",
            splat_bytes(&splats()),
            format!("{id_line}\n{good}"),
            "alice:meaning_id twice",
        ),
        (
            "unknown key",
            splat_bytes(&splats()),
            format!("{good}alice:other=1\n"),
            "is not known",
        ),
        (
            "not key=value",
            splat_bytes(&splats()),
            format!("{good}junk\n"),
            "is not key=value",
        ),
        (
            "short id",
            splat_bytes(&splats()),
            good.replace(&id_hex, &id_hex[..63]),
            "alice:meaning_id is not 64 hex",
        ),
        (
            "both keys",
            splat_bytes(&splats()),
            format!(
                "{id_line}\nalice:semantics_id={}\n{sha_line}\n",
                Provenance::sdf().hex()
            ),
            "has both alice:meaning_id and alice:semantics_id",
        ),
    ];
    cases.extend([(
        "not utf-8",
        splat_bytes(&splats()),
        String::new(),
        "is not UTF-8",
    )]);
    assert_eq!(cases.len(), 9);

    for (name, bytes, sidecar, why) in cases {
        let p = temp("bad.splat");
        std::fs::write(&p, &bytes).unwrap();
        if name == "not utf-8" {
            std::fs::write(provenance::sidecar_path(&p), [0xff, 0xfe, b'\n']).unwrap();
        } else {
            std::fs::write(provenance::sidecar_path(&p), sidecar).unwrap();
        }
        for result in [
            load_splat(&p).map(|_| ()),
            load_splat_pinned(&p).map(|_| ()),
        ] {
            assert!(
                matches!(result, Err(io::IoError::InvalidFormat(ref e)) if e.starts_with("splat: ") && e.contains(why)),
                "{name}: {result:?}"
            );
        }
    }
}

/// (writer name as the scanner reads it, what it writes) for the encoded buffers; each one is
/// written, its identifier read back and another one refused by the codec tests below
const CODEC_WRITERS: &[(&str, &str)] = &[
    (
        "src/mesh/mesh_codec.rs::encode_indices",
        "delta index buffer",
    ),
    (
        "src/mesh/mesh_codec.rs::encode_positions",
        "delta position buffer",
    ),
    (
        "src/mesh/mesh_codec.rs::encode_mesh",
        "delta index and position buffers",
    ),
    (
        "src/mesh/meshopt_index_codec.rs::encode_index_buffer",
        "stamped meshoptimizer index buffer",
    ),
    (
        "src/mesh/meshopt_index_codec.rs::try_encode_index_buffer",
        "stamped meshoptimizer index buffer",
    ),
    (
        "src/mesh/meshopt_vertex_codec.rs::encode_vertex_buffer",
        "stamped meshoptimizer vertex buffer",
    ),
    (
        "src/mesh/meshopt_vertex_codec.rs::encode_vertex_buffer_level",
        "stamped meshoptimizer vertex buffer",
    ),
    (
        "src/codec_bridge.rs::encode_sdf_volume",
        "compressed SDF volume stream",
    ),
];

fn codec_writer(name: &str) -> bool {
    CODEC_WRITERS.iter().any(|(w, _)| *w == name)
}

/// The crate's varint delta mesh codec (`mesh::mesh_codec`): version 2 buffers carry the
/// identifier; the decoders read this crate's and a reproduced earlier one, refuse another one
/// naming both, and refuse a version 1 buffer (written by the published 6.1.0) unless
/// `*_unverified`, which returns it as Unpinned
#[test]
fn mesh_codec_decoders_check_the_identifier() {
    use alice_sdf::mesh::mesh_codec::{
        decode_indices, decode_indices_unverified, decode_positions, decode_positions_unverified,
        encode_indices, encode_indices_with_provenance, encode_mesh, encode_mesh_with_provenance,
        encode_positions, encode_positions_with_provenance, CodecError,
    };
    for w in ["encode_indices", "encode_positions", "encode_mesh"] {
        assert!(codec_writer(&format!("src/mesh/mesh_codec.rs::{w}")), "{w}");
    }
    let idx = [0u32, 1, 2, 2, 1, 3];
    let pos = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.5, -2.0),
        Vec3::new(3.25, -0.125, 7.5),
        Vec3::new(-1.0, 2.0, 0.25),
    ];
    let bits = |v: &[Vec3]| {
        v.iter()
            .map(|p| p.to_array().map(f32::to_bits))
            .collect::<Vec<_>>()
    };

    // this crate's identifier: both decoders, Current
    let mesh = Mesh {
        vertices: pos.iter().map(|&p| Vertex::new(p, Vec3::Z)).collect(),
        indices: idx.to_vec(),
    };
    let (ib, pb) = encode_mesh(&mesh);
    assert_eq!(
        &ib[8..40],
        &Provenance::sdf().slot_for(StoredForm::MeshCodec)
    );
    assert_eq!(
        (encode_indices(&idx), encode_positions(&pos)),
        (ib.clone(), pb.clone())
    );
    assert_eq!(decode_indices(&ib).unwrap(), idx);
    assert_eq!(bits(&decode_positions(&pb).unwrap()), bits(&pos));
    assert_eq!(
        decode_indices_unverified(&ib).unwrap().semantics(),
        StoredSemantics::Current
    );

    // an earlier identifier this build no longer reproduces: refused by both decoders (the
    // mapping of a listed earlier identifier is sdf_load_verdict's, tested in the crate)
    let earlier = earlier_id();
    let ib = encode_indices_with_provenance(&idx, &earlier);
    let pb = encode_positions_with_provenance(&pos, &earlier);
    let refused = CodecError::MeaningMismatch {
        form: StoredForm::MeshCodec,
        stored: meaning_of(StoredForm::MeshCodec, &earlier),
        expected: meaning_of(StoredForm::MeshCodec, &Provenance::sdf()),
    };
    assert_eq!(decode_indices(&ib), Err(refused));
    assert_eq!(decode_indices_unverified(&ib).map(|_| ()), Err(refused));
    assert_eq!(decode_positions(&pb).map(|_| ()), Err(refused));
    assert_eq!(decode_positions_unverified(&pb).map(|_| ()), Err(refused));

    // another identifier: refused by both decoders, the error names both
    let ib = encode_indices_with_provenance(&idx, &caller_id());
    let pb = encode_positions_with_provenance(&pos, &caller_id());
    assert_eq!(
        encode_mesh_with_provenance(&mesh, &caller_id()),
        (ib.clone(), pb.clone())
    );
    let mismatch = CodecError::MeaningMismatch {
        form: StoredForm::MeshCodec,
        stored: meaning_of(StoredForm::MeshCodec, &caller_id()),
        expected: meaning_of(StoredForm::MeshCodec, &Provenance::sdf()),
    };
    assert_eq!(decode_indices(&ib), Err(mismatch));
    assert_eq!(decode_indices_unverified(&ib).map(|_| ()), Err(mismatch));
    assert_eq!(decode_positions(&pb), Err(mismatch));
    assert_eq!(decode_positions_unverified(&pb).map(|_| ()), Err(mismatch));
    let text = mismatch.to_string();
    assert!(
        text.contains(&meaning_of(StoredForm::MeshCodec, &caller_id()).hex())
            && text.contains(&meaning_of(StoredForm::MeshCodec, &Provenance::sdf()).hex())
    );

    // version 1 buffers written by alice-sdf 6.1.0 (crates.io): refused by default, read
    // unverified as Unpinned with the same values
    let ib1 = std::fs::read(fixture("mesh_codec_indices_v1.bin")).unwrap();
    let pb1 = std::fs::read(fixture("mesh_codec_positions_v1.bin")).unwrap();
    assert_eq!(ib1[4], 1);
    assert_eq!(decode_indices(&ib1), Err(CodecError::Unpinned));
    assert_eq!(decode_positions(&pb1), Err(CodecError::Unpinned));
    let (got, status) = decode_indices_unverified(&ib1).unwrap().into_parts();
    assert_eq!((got, status), (idx.to_vec(), StoredSemantics::Unpinned));
    let (got, status) = decode_positions_unverified(&pb1).unwrap().into_parts();
    assert_eq!(
        (bits(&got), status),
        (bits(&pos), StoredSemantics::Unpinned)
    );
}

/// The meshoptimizer codecs: the default encoders write the crate's stamped form (the identifier
/// around the meshoptimizer bytes) and the default decoders read only that; bare meshoptimizer
/// bytes (interchange, and everything 6.1.0 wrote) are refused by the default decoders as
/// Unpinned and read by name with `*_raw`, which marks them Unpinned; `*_raw` does not read a
/// stamped buffer
#[test]
fn meshopt_codecs_read_the_stamped_form_and_raw_bytes_by_name() {
    use alice_sdf::mesh::mesh_codec::CodecError;
    use alice_sdf::mesh::meshopt_index_codec::{
        decode_index_buffer, decode_index_buffer_raw, encode_index_buffer, encode_index_buffer_raw,
        try_encode_index_buffer, try_encode_index_buffer_with_provenance,
    };
    use alice_sdf::mesh::meshopt_vertex_codec::{
        decode_vertex_buffer, decode_vertex_buffer_raw, encode_vertex_buffer,
        encode_vertex_buffer_level, encode_vertex_buffer_raw, encode_vertex_buffer_with_provenance,
    };
    for w in [
        "meshopt_index_codec.rs::encode_index_buffer",
        "meshopt_index_codec.rs::try_encode_index_buffer",
        "meshopt_vertex_codec.rs::encode_vertex_buffer",
        "meshopt_vertex_codec.rs::encode_vertex_buffer_level",
    ] {
        assert!(codec_writer(&format!("src/mesh/{w}")), "{w}");
    }
    let idx = [0u32, 1, 2, 2, 1, 3];
    let pos = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.5, -2.0),
        Vec3::new(3.25, -0.125, 7.5),
        Vec3::new(-1.0, 2.0, 0.25),
    ];
    let mut verts = Vec::new();
    for p in pos {
        for c in [p.x, p.y, p.z] {
            verts.extend_from_slice(&c.to_le_bytes());
        }
    }
    let same_tris = |a: &[u32], b: &[u32]| {
        a.chunks_exact(3).zip(b.chunks_exact(3)).all(|(x, y)| {
            (0..3).any(|r| x[0] == y[r] && x[1] == y[(r + 1) % 3] && x[2] == y[(r + 2) % 3])
        })
    };

    // the stamped form: this crate's identifier around the raw bytes
    let vb = encode_vertex_buffer(&verts, 12);
    let ib = encode_index_buffer(&idx);
    assert_eq!(&vb[..4], b"ASMO");
    assert_eq!(
        &vb[8..40],
        &Provenance::sdf().slot_for(StoredForm::MeshoptStamp)
    );
    assert_eq!(&vb[40..], encode_vertex_buffer_raw(&verts, 12).as_slice());
    assert_eq!(decode_vertex_buffer(&vb, 4, 12).unwrap(), verts);
    assert!(same_tris(&idx, &decode_index_buffer(&ib, 6).unwrap()));
    assert_eq!(try_encode_index_buffer(&idx).unwrap(), ib);
    let vb_level = encode_vertex_buffer_level(&verts, 12, 0);
    assert_eq!(
        &vb_level[8..40],
        &Provenance::sdf().slot_for(StoredForm::MeshoptStamp)
    );
    assert_eq!(decode_vertex_buffer(&vb_level, 4, 12).unwrap(), verts);
    // an earlier identifier this build no longer reproduces is stored and refused (the mapping
    // of a listed earlier identifier is sdf_load_verdict's, tested in the crate)
    let earlier = earlier_id();
    let prev = encode_vertex_buffer_with_provenance(&verts, 12, &earlier);
    assert_eq!(
        decode_vertex_buffer(&prev, 4, 12),
        Err(CodecError::MeaningMismatch {
            form: StoredForm::MeshoptStamp,
            stored: meaning_of(StoredForm::MeshoptStamp, &earlier),
            expected: meaning_of(StoredForm::MeshoptStamp, &Provenance::sdf()),
        })
    );
    assert_eq!(
        &prev[8..40],
        &meaning_of(StoredForm::MeshoptStamp, &earlier).semantics_id
    );
    // another identifier is refused, naming both
    let mismatch = CodecError::MeaningMismatch {
        form: StoredForm::MeshoptStamp,
        stored: meaning_of(StoredForm::MeshoptStamp, &caller_id()),
        expected: meaning_of(StoredForm::MeshoptStamp, &Provenance::sdf()),
    };
    assert_eq!(
        decode_vertex_buffer(
            &encode_vertex_buffer_with_provenance(&verts, 12, &caller_id()),
            4,
            12
        ),
        Err(mismatch)
    );
    assert_eq!(
        decode_index_buffer(
            &try_encode_index_buffer_with_provenance(&idx, &caller_id()).unwrap(),
            6
        ),
        Err(mismatch)
    );
    // a vertex stamp is not an index buffer
    assert_eq!(decode_index_buffer(&vb, 6), Err(CodecError::WrongKind));

    // bare meshoptimizer bytes, as alice-sdf 6.1.0 (crates.io) wrote them: the default decoders
    // refuse them as Unpinned; *_raw reads them and says so
    let raw_v = std::fs::read(fixture("meshopt_vertex_v1.bin")).unwrap();
    let raw_i = std::fs::read(fixture("meshopt_index_v1.bin")).unwrap();
    assert_eq!(raw_v, encode_vertex_buffer_raw(&verts, 12));
    assert_eq!(raw_i, encode_index_buffer_raw(&idx));
    assert_eq!(
        decode_vertex_buffer(&raw_v, 4, 12),
        Err(CodecError::Unpinned)
    );
    assert_eq!(decode_index_buffer(&raw_i, 6), Err(CodecError::Unpinned));
    let (v, status) = decode_vertex_buffer_raw(&raw_v, 4, 12)
        .unwrap()
        .into_parts();
    assert_eq!((v, status), (verts.clone(), StoredSemantics::Unpinned));
    let (i, status) = decode_index_buffer_raw(&raw_i, 6).unwrap().into_parts();
    assert!(same_tris(&idx, &i));
    assert_eq!(status, StoredSemantics::Unpinned);
    // *_raw does not read the stamped form (it is not meshoptimizer bytes)
    assert_eq!(
        decode_vertex_buffer_raw(&vb, 4, 12).map(|_| ()),
        Err(CodecError::BadMagic)
    );
    assert_eq!(
        decode_index_buffer_raw(&ib, 6).map(|_| ()),
        Err(CodecError::BadMagic)
    );
}

/// The compressed SDF volume stream (`codec_bridge`): the identifier follows the header (flag
/// 0x10); `try_decode_sdf_volume` reads this crate's and a reproduced earlier one and refuses
/// another (naming both) and a stream written by 7.0.0 or earlier (6.1.0's, without the flag);
/// `decode_sdf_volume` / `decompress_sdf` panic on those with a message naming the identifiers;
/// `try_decode_sdf_volume_unverified` returns an old stream marked Unpinned
#[cfg(feature = "codec")]
#[test]
fn codec_volume_stream_checks_the_identifier() {
    use alice_sdf::codec_bridge::{
        decode_sdf_volume, decompress_sdf, encode_sdf_volume, encode_sdf_volume_with_provenance,
        try_decode_sdf_volume, try_decode_sdf_volume_unverified, voxelize_sdf_uniform, DecodeError,
        EncodeConfig,
    };
    assert!(codec_writer("src/codec_bridge.rs::encode_sdf_volume"));
    let volume = voxelize_sdf_uniform(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        8,
    );
    let config = EncodeConfig::default();
    let current = encode_sdf_volume(&volume, &config);
    let ok = try_decode_sdf_volume(&current).unwrap();
    assert_eq!(ok.data.len(), volume.data.len());
    assert_eq!(
        try_decode_sdf_volume_unverified(&current)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );
    let earlier = earlier_id();
    let prev = encode_sdf_volume_with_provenance(&volume, &config, &earlier);
    let refused = DecodeError::MeaningMismatch {
        stored: meaning_of(StoredForm::EncodedVolume, &earlier),
        expected: meaning_of(StoredForm::EncodedVolume, &Provenance::sdf()),
    };
    assert_eq!(
        try_decode_sdf_volume_unverified(&prev).map(|_| ()),
        Err(refused.clone())
    );
    assert_eq!(try_decode_sdf_volume(&prev).map(|_| ()), Err(refused));
    // the stream is not a form provenance::from_bytes reads; the mapping of a listed earlier
    // identifier is sdf_load_verdict's, tested in the crate

    let other = encode_sdf_volume_with_provenance(&volume, &config, &caller_id());
    let mismatch = DecodeError::MeaningMismatch {
        stored: meaning_of(StoredForm::EncodedVolume, &caller_id()),
        expected: meaning_of(StoredForm::EncodedVolume, &Provenance::sdf()),
    };
    assert_eq!(
        try_decode_sdf_volume(&other).map(|_| ()),
        Err(mismatch.clone())
    );
    assert_eq!(
        try_decode_sdf_volume_unverified(&other).map(|_| ()),
        Err(mismatch)
    );
    for decode in [decode_sdf_volume, decompress_sdf] {
        let panic = std::panic::catch_unwind(|| decode(&other)).unwrap_err();
        let text = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            text.contains(&meaning_of(StoredForm::EncodedVolume, &caller_id()).hex())
                && text.contains(&meaning_of(StoredForm::EncodedVolume, &Provenance::sdf()).hex()),
            "{text}"
        );
    }

    // a stream written by alice-sdf 6.1.0 (crates.io), without the identifier flag
    let old = std::fs::read(fixture("codec_volume_v1.bin")).unwrap();
    assert_eq!(old[33] & 0x10, 0);
    assert_eq!(
        try_decode_sdf_volume(&old).map(|_| ()),
        Err(DecodeError::Unpinned)
    );
    let (vol, status) = try_decode_sdf_volume_unverified(&old).unwrap().into_parts();
    assert_eq!(status, StoredSemantics::Unpinned);
    assert_eq!(vol.data.len(), 8 * 8 * 8);
    for decode in [decode_sdf_volume, decompress_sdf] {
        let panic = std::panic::catch_unwind(|| decode(&old)).unwrap_err();
        let text = panic.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(text.contains("no semantics identifier"), "{text}");
    }
}

/// `io::save` picks the writer by extension; both forms it writes carry this crate's identifier
/// (scripts/provenance-not-writers.txt names both writers; the scanner checks the body calls each)
#[test]
fn io_save_stores_the_identifier_for_both_extensions() {
    let tree = SdfTree::new(SdfNode::sphere(1.0));
    let mut compared = 0;
    for file in ["t_io_save.asdf", "t_io_save.asdf.json"] {
        let path = temp(file);
        io::save(&tree, &path).unwrap();
        assert!(
            provenance::read(&path)
                .unwrap()
                .is_written_under(&Provenance::sdf()),
            "{file}"
        );
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// The smallest Ogawa archive `provenance::read` reads the archive metadata from: the finished
/// header, a root group of four children (three empty data blocks, then `meta`)
fn ogawa_with_metadata(meta: &str) -> Vec<u8> {
    let empty = 1u64 << 63;
    let mut b = b"Ogawa\xFF\x00\x01".to_vec();
    let data_at = 16u64;
    let root_at = data_at + 8 + meta.len() as u64;
    b.extend_from_slice(&root_at.to_le_bytes());
    b.extend_from_slice(&(meta.len() as u64).to_le_bytes());
    b.extend_from_slice(meta.as_bytes());
    for child in [4, empty, empty, empty, data_at | empty] {
        b.extend_from_slice(&child.to_le_bytes());
    }
    b
}

/// The identifier in an Alembic archive's metadata: an archive another program wrote without it
/// (the Alembic library, tests/fixtures/provenance/alembic_foreign.abc) is Missing, never read as
/// some value; one that holds the key but no readable value, two values, an unfinished or cut
/// archive and the layout this crate wrote up to 7.0.0 (which no Alembic reader opens) are
/// unreadable
#[test]
fn alembic_without_identifier_is_missing_and_a_malformed_archive_unreadable() {
    let path = temp("m_unstamped.abc");
    io::export_alembic(&mesh(), &path, &io::AlembicConfig::default()).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(provenance::from_bytes(&bytes)
        .unwrap()
        .is_written_under(&Provenance::sdf()));

    let foreign = std::fs::read(fixture("alembic_foreign.abc")).unwrap();
    assert!(matches!(
        provenance::from_bytes(&foreign),
        Err(ProvenanceError::Missing { format: "Alembic" })
    ));

    // edits in place keep every length and position
    let key = format!("{}=", provenance::MEANING_KEY);
    let at = bytes
        .windows(key.len())
        .position(|w| w == key.as_bytes())
        .unwrap();
    let edit = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = bytes.clone();
        f(&mut b);
        provenance::from_bytes(&b)
    };
    // another key name: no identifier
    assert!(matches!(
        edit(&|b| b[at + 5] = b'X'),
        Err(ProvenanceError::Missing { format: "Alembic" })
    ));
    let unreadable = |r: Result<provenance::Stored, ProvenanceError>, what: &str| {
        assert!(
            matches!(r, Err(ProvenanceError::Unreadable(_))),
            "{what}: {r:?}"
        );
    };
    unreadable(
        edit(&|b| b[at + key.len() + 10] = b'g'),
        "a non-hex identifier",
    );
    unreadable(
        edit(&|b| b[5] = 0x00),
        "an unfinished archive (frozen byte 0)",
    );
    unreadable(edit(&|b| b[7] = 0x02), "another Ogawa version");
    unreadable(provenance::from_bytes(&bytes[..20]), "a cut archive");
    // the same key twice, in an archive built from the Ogawa layout (see `ogawa_with_metadata`)
    let (_, hex) = Provenance::sdf().named_for(StoredForm::Alembic);
    assert!(
        provenance::from_bytes(&ogawa_with_metadata(&format!("{key}{hex}")))
            .unwrap()
            .is_written_under(&Provenance::sdf())
    );
    unreadable(
        provenance::from_bytes(&ogawa_with_metadata(&format!("{key}{hex};{key}{hex}"))),
        "two identifiers",
    );
    // an archive of 7.0.0 holds the raw identifier under the earlier key: the earlier generation
    let raw = format!("{}={}", provenance::KEY, Provenance::sdf().hex());
    let earlier = provenance::from_bytes(&ogawa_with_metadata(&raw)).unwrap();
    assert!(!earlier.is_written_under(&Provenance::sdf()));
    assert_eq!(
        earlier
            .judge(&Provenance::sdf(), alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS)
            .unwrap(),
        StoredSemantics::VerifiedPrevious(Provenance::sdf())
    );
    // both keys in one archive: unreadable, never read as either
    unreadable(
        provenance::from_bytes(&ogawa_with_metadata(&format!("{key}{hex};{raw}"))),
        "the meaning key and the earlier key",
    );
    // the layout this crate wrote up to 7.0.0: magic, version 1.0 in three bytes, a group tag
    let mut old = b"Ogawa\x00\x01\x00\xFF".to_vec();
    old.extend_from_slice(&6u32.to_le_bytes());
    match provenance::from_bytes(&old) {
        Err(ProvenanceError::Unreadable(why)) => assert!(why.contains("7.0.0"), "{why}"),
        other => panic!("the 7.0.0 layout gave {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// G8: the image readers and the heightmap PNG writer
//
// A 16 x 12 grey image, level (13 x + 7 z) mod 256, encoded by the `image` crate (0.25) as a
// paint tool would write it (no identifier), and the heightmap writer's output for it:
//   heightmap_plain.png    34151b0d0efbd9f7fa65ccc76299d474e00ad48ddfe2b40ecfed7916f0db8098
//   heightmap_plain.jpg    3b8303187e7a141798d8c3a8c72b26da9795da077b87fcc00856a717afc7dbcd
//   heightmap_stamped.png  e0514a7706ecf2b7831f63f78b8cfcf4c39c988c17ba2426ce5eaa7dfc1cfedd
//                          Heightmap::from_image_bytes(plain.png, &HeightmapImageConfig::new(
//                          40.0, 16.0, 12.0)).to_png(..) (16-bit grey, the meaning identifier in
//                          tEXt); built outside the writer: the earlier file's tEXt chunk
//                          replaced by `alice:meaning_id` and the PNG meaning identifier of
//                          python/tests/meaning_id.py, its CRC recomputed
//   heightmap_raw_main.png e13af87f802f9dde3dbb1af755b841ee15f3eb2946cf02c05bdd362a179b2308
//                          the same writer before meaning identifiers (`alice:semantics_id`)
// ---------------------------------------------------------------------------------------------

/// The CRC-32 of a PNG chunk (its type and data), for building malformed images in the tests
#[cfg(feature = "image")]
fn png_chunk_crc(kind: &[u8], data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in kind.iter().chain(data) {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// `png` with a chunk of `kind` and `data` inserted after IHDR (the signature and IHDR are 33 bytes)
#[cfg(feature = "image")]
fn with_png_chunk(png: &[u8], kind: [u8; 4], data: &[u8]) -> Vec<u8> {
    let mut chunk = u32::try_from(data.len()).unwrap().to_be_bytes().to_vec();
    chunk.extend_from_slice(&kind);
    chunk.extend_from_slice(data);
    chunk.extend_from_slice(&png_chunk_crc(&kind, data).to_be_bytes());
    [&png[..33], &chunk, &png[33..]].concat()
}

/// `jpeg` with a comment segment (`COM`) of `payload` inserted after the start of image
#[cfg(feature = "image")]
fn with_jpeg_comment(jpeg: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut seg = vec![0xFF, 0xFE];
    seg.extend_from_slice(&u16::try_from(payload.len() + 2).unwrap().to_be_bytes());
    seg.extend_from_slice(payload);
    [&jpeg[..2], &seg, &jpeg[2..]].concat()
}

/// (writer name as the scanner reads it, what it writes) for the image writer; the test below
/// writes it, reads the identifier back and checks a mismatch
#[cfg(all(feature = "image", feature = "terrain"))]
const IMAGE_WRITERS: &[(&str, &str)] = &[(
    "src/terrain/heightmap.rs::to_png",
    "16-bit grey PNG heightmap",
)];

/// The heightmap images: `Heightmap::to_png` stores the identifier in a `tEXt` chunk before the
/// image data (`tests/fixtures/provenance/heightmap_stamped.png` pins the bytes); the readers
/// (`from_image_bytes` / `from_image`) read this crate's identifier as Current, a reproduced
/// earlier one as VerifiedPrevious, refuse another (naming both), and read an image without one
/// (a paint tool's PNG or JPEG) as Unpinned, which the `*_pinned` readers refuse; a malformed or
/// repeated identifier, the key in a compressed text chunk, a cut chunk and a JPEG comment that is
/// not exactly the tagged form are invalid, never read as unstamped
#[cfg(all(feature = "image", feature = "terrain"))]
#[test]
fn heightmap_images_carry_and_check_the_identifier() {
    use alice_sdf::terrain::{Heightmap, HeightmapImageConfig};
    assert_eq!(IMAGE_WRITERS.len(), 1, "{IMAGE_WRITERS:?}");
    let cfg = HeightmapImageConfig::new(40.0, 16.0, 12.0);
    let invalid = |bytes: &[u8]| {
        assert!(
            matches!(
                Heightmap::from_image_bytes(bytes, &cfg).map(|_| ()),
                Err(io::IoError::InvalidFormat(_))
            ),
            "read as valid"
        );
        assert!(matches!(
            provenance::from_bytes(bytes),
            Err(ProvenanceError::Unreadable(_))
        ));
    };

    // a paint tool's 8-bit grey PNG and JPEG: read, marked Unpinned; the pinned readers refuse
    let plain = std::fs::read(fixture("heightmap_plain.png")).unwrap();
    let jpeg = std::fs::read(fixture("heightmap_plain.jpg")).unwrap();
    for (bytes, format) in [(&plain, "PNG"), (&jpeg, "JPEG")] {
        let (hm, status) = Heightmap::from_image_bytes(bytes, &cfg)
            .unwrap()
            .into_parts();
        assert_eq!(status, StoredSemantics::Unpinned);
        assert_eq!((hm.width, hm.depth), (16, 12));
        assert!(matches!(
            Heightmap::from_image_bytes_pinned(bytes, &cfg),
            Err(io::IoError::Unpinned { format: f }) if f == format
        ));
        assert!(matches!(
            provenance::from_bytes(bytes),
            Err(ProvenanceError::Missing { format: f }) if f == format
        ));
    }
    let (hm, _) = Heightmap::from_image_bytes(&plain, &cfg)
        .unwrap()
        .into_parts();

    // this crate's writer: 16-bit grey, the identifier before the image data, pinned bytes
    let stamped = hm.to_png(&cfg).unwrap();
    assert_eq!(
        stamped,
        std::fs::read(fixture("heightmap_stamped.png")).unwrap()
    );
    let key_at = stamped
        .windows(provenance::MEANING_KEY.len())
        .position(|w| w == provenance::MEANING_KEY.as_bytes())
        .unwrap();
    let idat_at = stamped.windows(4).position(|w| w == b"IDAT").unwrap();
    assert!(
        key_at < idat_at,
        "the identifier is written before the image data"
    );
    assert!(provenance::from_bytes(&stamped)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let (back, status) = Heightmap::from_image_bytes(&stamped, &cfg)
        .unwrap()
        .into_parts();
    assert_eq!(status, StoredSemantics::Current);
    // an 8-bit level p is stored as 257·p, read back as the same height to the 16-bit step
    for (a, b) in back.heights.iter().zip(&hm.heights) {
        assert!((a - b).abs() <= 40.0 / 65535.0, "{a} {b}");
    }
    // heights read from such a PNG are written back to the same bytes
    assert_eq!(back.to_png(&cfg).unwrap(), stamped);
    let path = temp("g8_image.png");
    std::fs::write(&path, &stamped).unwrap();
    let pinned = Heightmap::from_image_pinned(&path, &cfg).unwrap();
    assert_eq!(
        Heightmap::from_image(&path, &cfg).unwrap().semantics(),
        StoredSemantics::Current
    );
    std::fs::remove_file(&path).ok();
    assert_eq!(pinned.heights, back.heights);

    // an earlier identifier this build no longer reproduces is refused (the shared judge given
    // it reads it as VerifiedPrevious); another is refused, naming both
    let earlier = earlier_id();
    let prev = hm.to_png_with_provenance(&cfg, &earlier).unwrap();
    refused_as_earlier(Heightmap::from_image_bytes(&prev, &cfg));
    refused_as_earlier(Heightmap::from_image_bytes_pinned(&prev, &cfg));
    assert_eq!(
        provenance::from_bytes(&prev)
            .unwrap()
            .judge(&Provenance::sdf(), &[EARLIER])
            .unwrap(),
        StoredSemantics::VerifiedPrevious(earlier)
    );
    let foreign = hm.to_png_with_provenance(&cfg, &caller_id()).unwrap();
    for result in [
        Heightmap::from_image_bytes(&foreign, &cfg).map(|_| ()),
        Heightmap::from_image_bytes_pinned(&foreign, &cfg).map(|_| ()),
    ] {
        refused_as_caller(result);
    }

    // malformed PNGs: a changed digit (the CRC no longer matches), a value that is not 64 hex
    // digits, a second key chunk, the key in zTXt / iTXt (also on an unstamped image), a cut file
    let mut flipped = stamped.clone();
    let digit = key_at + provenance::KEY.len() + 1;
    flipped[digit] = if flipped[digit] == b'0' { b'1' } else { b'0' };
    invalid(&flipped);
    let key = provenance::KEY.as_bytes();
    let tagged = |value: &[u8]| [key, b"\0", value].concat();
    invalid(&with_png_chunk(&plain, *b"tEXt", &tagged(b"not hex")));
    // the keyword alone, without the NUL separator and a value
    invalid(&with_png_chunk(&plain, *b"tEXt", key));
    invalid(&with_png_chunk(&plain, *b"tEXt", &tagged(&[b'a'; 63])));
    invalid(&with_png_chunk(
        &stamped,
        *b"tEXt",
        &tagged(Provenance::sdf().hex().as_bytes()),
    ));
    invalid(&with_png_chunk(
        &plain,
        *b"zTXt",
        &[key, b"\0\0", b"x"].concat(),
    ));
    invalid(&with_png_chunk(
        &plain,
        *b"iTXt",
        &[key, b"\0\0\0\0\0", b"x"].concat(),
    ));
    // the key in a text chunk other than tEXt is refused even when its value would read as an
    // identifier (only the uncompressed tEXt form is the stored one)
    for kind in [*b"zTXt", *b"iTXt"] {
        invalid(&with_png_chunk(
            &plain,
            kind,
            &tagged(Provenance::sdf().hex().as_bytes()),
        ));
    }
    invalid(&stamped[..stamped.len() - 12]); // no IEND
    invalid(&stamped[..idat_at + 10]); // cut inside the image data
    let mut overrun = stamped;
    overrun[33..37].copy_from_slice(&u32::MAX.to_be_bytes());
    invalid(&overrun);
    // the key in another chunk's text is not the key chunk
    let unrelated = with_png_chunk(&plain, *b"tEXt", &[b"Comment\0".as_slice(), key].concat());
    assert_eq!(
        Heightmap::from_image_bytes(&unrelated, &cfg)
            .unwrap()
            .semantics(),
        StoredSemantics::Unpinned
    );

    // JPEG: the comment segment before the start of scan; this build's meaning identifier is
    // current, a raw identifier of this crate (a file stamped before meaning identifiers) the
    // previous generation
    let tag = Provenance::sdf().tagged_for(StoredForm::Jpeg);
    let jpeg_stamped = with_jpeg_comment(&jpeg, tag.as_bytes());
    assert_eq!(
        Heightmap::from_image_bytes(&jpeg_stamped, &cfg)
            .unwrap()
            .semantics(),
        StoredSemantics::Current
    );
    let jpeg_raw = with_jpeg_comment(
        &jpeg,
        format!("{}={}", provenance::KEY, Provenance::sdf().hex()).as_bytes(),
    );
    assert_eq!(
        Heightmap::from_image_bytes(&jpeg_raw, &cfg)
            .unwrap()
            .semantics(),
        StoredSemantics::VerifiedPrevious(Provenance::sdf())
    );
    // both keys: read as neither
    invalid(&with_jpeg_comment(&jpeg_raw, tag.as_bytes()));
    let jpeg_foreign =
        with_jpeg_comment(&jpeg, caller_id().tagged_for(StoredForm::Jpeg).as_bytes());
    assert!(matches!(
        Heightmap::from_image_bytes(&jpeg_foreign, &cfg).map(|_| ()),
        Err(io::IoError::MeaningMismatch { format: "JPEG", .. })
    ));
    invalid(&with_jpeg_comment(&jpeg_stamped, tag.as_bytes()));
    invalid(&with_jpeg_comment(
        &jpeg,
        format!("{}=zz", provenance::MEANING_KEY).as_bytes(),
    ));
    invalid(&with_jpeg_comment(&jpeg, format!("{} ", tag).as_bytes()));
    // fill bytes (0xFF) before the comment's marker, and markers without a length (RST0, TEM)
    // before it: skipped, the identifier is still read
    let filled = [&jpeg_stamped[..2], &[0xFF, 0xFF][..], &jpeg_stamped[2..]].concat();
    assert!(provenance::from_bytes(&filled)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let markers = [
        &jpeg_stamped[..2],
        &[0xFF, 0xD0, 0xFF, 0x01][..],
        &jpeg_stamped[2..],
    ]
    .concat();
    assert!(provenance::from_bytes(&markers)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    // the end of image before the start of scan, even when the bytes after it would walk on to
    // the identifier (read as a 2-byte segment, they would)
    invalid(
        &[
            &jpeg_stamped[..2],
            &[0xFF, 0xD9, 0x00, 0x02][..],
            &jpeg_stamped[2..],
        ]
        .concat(),
    );
    let mut seg_overrun = with_jpeg_comment(&jpeg, b"hello");
    seg_overrun[4..6].copy_from_slice(&u16::MAX.to_be_bytes());
    invalid(&seg_overrun);
    invalid(&jpeg[..20]);
}

/// The texture fit reads the identifier of its image as the heightmap readers do: an image
/// without one is fitted and marked Unpinned (refused by `fit_texture_pinned`), this crate's PNG
/// is Current, another identifier is refused naming both
#[cfg(all(feature = "texture-fit", feature = "terrain"))]
#[test]
fn texture_fit_checks_the_image_identifier() {
    use alice_sdf::terrain::{Heightmap, HeightmapImageConfig};
    use alice_sdf::texture::{fit_texture, fit_texture_pinned, TextureFitConfig, TextureFitError};
    let config = TextureFitConfig {
        max_octaves: 1,
        iterations_per_octave: 20,
        ..TextureFitConfig::default()
    };
    let plain = fixture("heightmap_plain.png");
    let (fit, status) = fit_texture(&plain, &config).unwrap().into_parts();
    assert_eq!(status, StoredSemantics::Unpinned);
    assert_eq!((fit.width, fit.height), (16, 12));
    assert!(matches!(
        fit_texture_pinned(&plain, &config),
        Err(TextureFitError::Io(io::IoError::Unpinned { format: "PNG" }))
    ));

    let cfg = HeightmapImageConfig::default();
    let hm = Heightmap::from_image_bytes(&std::fs::read(&plain).unwrap(), &cfg)
        .unwrap()
        .into_parts()
        .0;
    let current = temp("g8_image_fit.png");
    std::fs::write(&current, hm.to_png(&cfg).unwrap()).unwrap();
    assert_eq!(
        fit_texture(&current, &config).unwrap().semantics(),
        StoredSemantics::Current
    );
    assert!(fit_texture_pinned(&current, &config).is_ok());
    std::fs::write(
        &current,
        hm.to_png_with_provenance(&cfg, &caller_id()).unwrap(),
    )
    .unwrap();
    let refused = fit_texture(&current, &config).map(|_| ());
    std::fs::remove_file(&current).ok();
    match refused {
        Err(TextureFitError::Io(io::IoError::MeaningMismatch {
            form,
            stored,
            expected,
            ..
        })) => {
            let hex = |id: [u8; 32]| Provenance::new(id).hex();
            assert_eq!(form, StoredForm::Png);
            assert_eq!(
                (stored, expected),
                (
                    hex(provenance::meaning_id(form, &caller_id())),
                    hex(provenance::meaning_id(form, &Provenance::sdf()))
                )
            );
        }
        other => panic!("another identifier was read: {other:?}"),
    }
}

/// A small Nanite cluster hierarchy of the unit sphere (two LOD levels)
fn nanite_mesh() -> alice_sdf::mesh::NaniteMesh {
    alice_sdf::mesh::generate_nanite_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &alice_sdf::mesh::NaniteConfig {
            lod_levels: 2,
            base_resolution: 8,
            ..alice_sdf::mesh::NaniteConfig::default()
        },
    )
}

/// `.nanite` version 4 stores the identifier in the 32 bytes after the 32-byte header; a file of
/// versions 1-3 (no identifier) is reported missing, another version or a cut header unreadable,
/// never read as some identifier
#[test]
fn nanite_binary_versions_without_the_identifier_are_missing() {
    let path = temp("versions.nanite");
    io::export_nanite(&nanite_mesh(), &path).unwrap();
    let b = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(&b[..4], io::nanite::NANITE_MAGIC);
    assert_eq!(b[4..8], 4u32.to_le_bytes());
    assert_eq!(b[32..64], Provenance::sdf().slot_for(StoredForm::Nanite));
    assert!(provenance::from_bytes(&b)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let mut compared = 0;
    for version in [1u32, 2, 3] {
        // the layout of versions 1-3: the same header and body without the identifier
        let mut old = [&b[..32], &b[64..]].concat();
        old[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(
            matches!(
                provenance::from_bytes(&old),
                Err(ProvenanceError::Missing { format: "Nanite" })
            ),
            "version {version}"
        );
        compared += 1;
    }
    assert_eq!(compared, 3);
    for version in [0u32, 5, u32::MAX] {
        let mut other = b.clone();
        other[4..8].copy_from_slice(&version.to_le_bytes());
        assert!(
            matches!(
                provenance::from_bytes(&other),
                Err(ProvenanceError::Unreadable(_))
            ),
            "version {version}"
        );
    }
    for cut in [6, 40, 63] {
        assert!(
            matches!(
                provenance::from_bytes(&b[..cut]),
                Err(ProvenanceError::Unreadable(_))
            ),
            "cut at {cut}"
        );
    }
}

/// (writer name as the scanner reads it, what it writes) for the shader text writer (feature
/// hlsl); the test below writes it, reads the identifier back and refuses malformed forms
#[cfg(feature = "hlsl")]
const SHADER_TEXT_WRITERS: &[(&str, &str)] = &[(
    "src/io/nanite.rs::export_nanite_hlsl_material",
    "Nanite HLSL material (.usf)",
)];

/// The Nanite HLSL material stores the identifier as a comment in its leading comment block
/// (`// alice:semantics_id=<64 hex>`, the form of the shader text this crate writes); a `//`-led
/// text without it is missing, a malformed value or two different values are unreadable
#[cfg(feature = "hlsl")]
#[test]
fn nanite_hlsl_material_carries_the_identifier() {
    use io::nanite::{export_nanite_hlsl_material, export_nanite_hlsl_material_with_provenance};
    assert_eq!(SHADER_TEXT_WRITERS.len(), 1, "{SHADER_TEXT_WRITERS:?}");
    let node = SdfNode::sphere(1.0);
    let path = temp("material.usf");
    export_nanite_hlsl_material(&node, &path).unwrap();
    assert!(provenance::read(&path)
        .unwrap()
        .is_written_under(&Provenance::sdf()));
    let text = std::fs::read_to_string(&path).unwrap();
    let tag = format!(
        "// {}",
        Provenance::sdf().tagged_for(StoredForm::ShaderText)
    );
    let at = text
        .lines()
        .position(|l| l == tag)
        .expect("the identifier line");
    assert!(
        text.lines().take(at).all(|l| l.starts_with("//")),
        "the identifier is in the leading comment block"
    );
    export_nanite_hlsl_material_with_provenance(&node, &path, &caller_id()).unwrap();
    assert!(provenance::read(&path)
        .unwrap()
        .is_written_under(&caller_id()));
    std::fs::remove_file(&path).ok();

    let without: String = text
        .lines()
        .filter(|l| *l != tag)
        .flat_map(|l| [l, "\n"])
        .collect();
    assert!(matches!(
        provenance::from_bytes(without.as_bytes()),
        Err(ProvenanceError::Missing {
            format: "shader text"
        })
    ));
    // the key below the leading comment block (in the shader body) is not the identifier, and a
    // present key is never read as absent: the text is unreadable
    let below = format!("{without}{tag}\n");
    assert!(matches!(
        provenance::from_bytes(below.as_bytes()),
        Err(ProvenanceError::Unreadable(_))
    ));
    let not_hex = text.replace(
        &Provenance::sdf().named_for(StoredForm::ShaderText).1,
        &"g".repeat(64),
    );
    assert!(matches!(
        provenance::from_bytes(not_hex.as_bytes()),
        Err(ProvenanceError::Unreadable(_))
    ));
    let two = text.replacen(
        &tag,
        &format!(
            "{tag}\n// {}",
            caller_id().tagged_for(StoredForm::ShaderText)
        ),
        1,
    );
    assert!(matches!(
        provenance::from_bytes(two.as_bytes()),
        Err(ProvenanceError::Unreadable(_))
    ));
}

/// The UE5 (`ALUE`) and Unity (`ALMB`) binary meshes: version 2 stores the identifier in the 32
/// bytes after the 16-byte header; a version 1 file (no identifier) is reported missing, another
/// version or a cut header unreadable, never read as some identifier
#[test]
fn engine_binary_meshes_of_version_1_are_missing() {
    type Bin = fn(&Mesh, &Path);
    let writers: [(&str, &[u8; 4], Bin); 2] = [
        ("UE5", b"ALUE", |m, p| {
            io::export_ue5_mesh_binary(m, p, &io::Ue5MeshConfig::default()).unwrap();
        }),
        ("Unity", b"ALMB", |m, p| {
            io::export_unity_mesh_binary(m, p, &io::UnityMeshConfig::default()).unwrap();
        }),
    ];
    let m = mesh();
    let mut compared = 0;
    for (format, magic, write) in writers {
        let path = temp(&format!("versions_{format}.bin"));
        write(&m, &path);
        let b = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(&b[..4], magic);
        assert_eq!(b[4..6], 2u16.to_le_bytes(), "{format}");
        let form = if format == "UE5" {
            StoredForm::Ue5Mesh
        } else {
            StoredForm::UnityMesh
        };
        assert_eq!(b[16..48], Provenance::sdf().slot_for(form), "{format}");
        assert!(provenance::from_bytes(&b)
            .unwrap()
            .is_written_under(&Provenance::sdf()));
        // version 1: the same header and body without the identifier
        let mut old = [&b[..16], &b[48..]].concat();
        old[4..6].copy_from_slice(&1u16.to_le_bytes());
        match provenance::from_bytes(&old) {
            Err(ProvenanceError::Missing { format: f }) => assert_eq!(f, format),
            other => panic!("{format} version 1: {other:?}"),
        }
        for version in [0u16, 3, u16::MAX] {
            let mut other = b.clone();
            other[4..6].copy_from_slice(&version.to_le_bytes());
            assert!(
                matches!(
                    provenance::from_bytes(&other),
                    Err(ProvenanceError::Unreadable(_))
                ),
                "{format} version {version}"
            );
        }
        for cut in [5, 20, 47] {
            assert!(
                matches!(
                    provenance::from_bytes(&b[..cut]),
                    Err(ProvenanceError::Unreadable(_))
                ),
                "{format} cut at {cut}"
            );
        }
        compared += 1;
    }
    assert_eq!(compared, 2);
}

/// The stored form is exactly 64 lower-case hex digits that end the value: a 65th digit, letters
/// after the digits, or upper-case digits make the identifier unreadable (never `Current`, never
/// read as absent). The text forms read through the key (OBJ, shader text, ASCII STL, ASCII FBX,
/// sidecar-style text) and binary FBX (whose value must end its length-prefixed string) are
/// covered.
///
/// Oracle: the documented tag format `alice:semantics_id=<64 hex>` and the writers' lower-case
/// [`Provenance::hex`]; `Unreadable` per the doc of [`provenance::read`] (the identifier is present
/// but cannot be read). Not produced by the reader under test.
#[test]
fn only_64_lower_case_hex_digits_ending_the_value_are_read() {
    let hex = Provenance::sdf().hex();
    assert_eq!(hex, hex.to_ascii_lowercase());
    let key = provenance::KEY;
    type Form = (&'static str, fn(&str) -> String);
    let forms: [Form; 4] = [
        ("OBJ", |v| format!("# {}={v}\nv 0 0 0\n", provenance::KEY)),
        ("shader text", |v| {
            format!("// {}={v}\nfn f() {{}}\n", provenance::KEY)
        }),
        ("ASCII STL", |v| {
            format!("solid {}={v}\nendsolid\n", provenance::KEY)
        }),
        ("ASCII FBX", |v| {
            format!(
                "; FBX 7.4.0 project file\nFBXHeaderExtension:  {{\n\tCreator: \"ALICE-SDF {}={v}\"\n}}\n",
                provenance::KEY
            )
        }),
    ];
    let mut checked = 0;
    for (format, make) in forms {
        assert_eq!(
            provenance::from_bytes(make(&hex).as_bytes())
                .ok()
                .map(|s| s.id),
            Some(Provenance::sdf().semantics_id),
            "{format}: the valid tag"
        );
        for (what, value) in [
            ("a 65th hex digit", format!("{hex}0")),
            ("letters after the digits", format!("{hex}zz")),
            ("upper-case hex", hex.to_ascii_uppercase()),
            ("63 digits at the end", hex[..63].to_string()),
        ] {
            let r = provenance::from_bytes(make(&value).as_bytes());
            assert!(
                matches!(r, Err(ProvenanceError::Unreadable(_))),
                "{format} {what} ({key}={value}): {r:?}"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 16);

    // binary FBX: the value ends the length-prefixed `Creator` string
    let m = mesh();
    let path = temp("strict_tail.fbx");
    let c = io::FbxConfig {
        format: io::FbxFormat::Binary,
        ..Default::default()
    };
    io::export_fbx(&m, &path, &c, None).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert!(
        provenance::from_bytes(&bytes).is_ok_and(|s| s.is_written_under(&Provenance::sdf())),
        "binary FBX: the valid tag"
    );
    // the binary FBX writer stores the meaning identifier under its key
    let key = provenance::MEANING_KEY;
    let hex = hex_of(&provenance::meaning_id(
        StoredForm::FbxBinary,
        &Provenance::sdf(),
    ));
    let at = bytes
        .windows(key.len())
        .position(|w| w == key.as_bytes())
        .unwrap();
    let start = at + key.len() + 1;
    let mut upper = bytes.clone();
    upper[start..start + 64].make_ascii_uppercase();
    assert!(matches!(
        provenance::from_bytes(&upper),
        Err(ProvenanceError::Unreadable(_))
    ));
    // the string's declared length one byte longer: the digits no longer end the value
    let creator = format!("ALICE-SDF {key}={hex}");
    let s = at - "ALICE-SDF ".len() - 5;
    assert_eq!(bytes[s], b'S');
    assert_eq!(
        u32::from_le_bytes(bytes[s + 1..s + 5].try_into().unwrap()) as usize,
        creator.len()
    );
    let mut longer = bytes;
    longer[s + 1..s + 5].copy_from_slice(&(creator.len() as u32 + 1).to_le_bytes());
    assert!(matches!(
        provenance::from_bytes(&longer),
        Err(ProvenanceError::Unreadable(_))
    ));
}

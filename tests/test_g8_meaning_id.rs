//! ALICE-Law G8: the meaning identifier the stored forms carry
//!
//! A stored form carries `meaning_id(form, semantics)` (law-id v2 over the form, its layout version
//! and the semantics identifier), not the in-process identifier itself, so the stored identifier
//! names what the stored bytes mean and differs from G6's. The reader recomputes it for the
//! expected identifier and for G6's earlier ones; a raw identifier (a file written before the form
//! stored meaning identifiers) reads as the earlier generation.
//!
//! Author: Moroya Sakamoto

use alice_sdf::io;
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{
    self, meaning_id, Provenance, ProvenanceError, StoredForm, StoredSemantics,
};
use alice_sdf::{PREVIOUS_SDF_SEMANTICS_IDS, SDF_SEMANTICS_BY_DET_MATH, SDF_SEMANTICS_ID};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The forms whose writers store meaning identifiers in this build
const MEANING_FORMS: [StoredForm; 13] = [
    StoredForm::StlBinary,
    StoredForm::StlAscii,
    StoredForm::ThreeMf,
    StoredForm::FbxBinary,
    StoredForm::FbxAscii,
    StoredForm::ObjText,
    StoredForm::Ply,
    StoredForm::Glb,
    StoredForm::Gltf,
    StoredForm::Usd,
    StoredForm::Step,
    StoredForm::Iges,
    StoredForm::Alembic,
];

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("alice_sdf_g8_meaning_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

fn mesh() -> Mesh {
    sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 4,
            ..Default::default()
        },
    )
}

fn hex(id: &[u8; 32]) -> String {
    id.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        write!(s, "{b:02x}").expect("writing to a String");
        s
    })
}

/// Writes `m` to `path` as `form` under `p`
fn write(form: StoredForm, m: &Mesh, path: &Path, p: &Provenance) {
    let fbx = |format| io::FbxConfig {
        format,
        ..Default::default()
    };
    match form {
        StoredForm::StlBinary => io::export_stl_with_provenance(m, path, p).unwrap(),
        StoredForm::StlAscii => io::export_stl_ascii_with_provenance(m, path, p).unwrap(),
        StoredForm::ThreeMf => io::export_3mf_with_provenance(m, path, p).unwrap(),
        StoredForm::FbxBinary => {
            io::export_fbx_with_provenance(m, path, &fbx(io::FbxFormat::Binary), None, p).unwrap();
        }
        StoredForm::FbxAscii => {
            io::export_fbx_with_provenance(m, path, &fbx(io::FbxFormat::Ascii), None, p).unwrap();
        }
        StoredForm::ObjText => {
            io::export_obj_with_provenance(m, path, &io::ObjConfig::default(), None, p).unwrap();
        }
        StoredForm::Ply => {
            io::export_ply_with_provenance(m, path, &io::PlyConfig::default(), p).unwrap();
        }
        StoredForm::Glb => {
            io::export_glb_with_provenance(m, path, &io::GltfConfig::default(), None, p).unwrap();
        }
        StoredForm::Gltf => {
            io::export_gltf_json_with_provenance(m, path, &io::GltfConfig::default(), None, p)
                .unwrap();
        }
        StoredForm::Usd => {
            io::export_usda_with_provenance(m, path, &io::UsdConfig::default(), None, p).unwrap();
        }
        StoredForm::Alembic => {
            io::export_alembic_with_provenance(m, path, &io::AlembicConfig::default(), p).unwrap();
        }
        // STEP and IGES tessellate the SDF themselves: the sphere `mesh` is built from
        StoredForm::Step => {
            let cfg = io::step::StepConfig {
                resolution: 4,
                ..Default::default()
            };
            io::step::export_step_with_provenance(path, &SdfNode::sphere(1.0), &cfg, p).unwrap();
        }
        StoredForm::Iges => {
            let cfg = io::iges::IgesConfig {
                resolution: 4,
                ..Default::default()
            };
            io::iges::export_iges_with_provenance(path, &SdfNode::sphere(1.0), &cfg, p).unwrap();
        }
        other => panic!("no writer for {other:?} in this test"),
    }
}

const fn ext(form: StoredForm) -> &'static str {
    match form {
        StoredForm::StlBinary | StoredForm::StlAscii => "stl",
        StoredForm::ThreeMf => "3mf",
        StoredForm::FbxBinary | StoredForm::FbxAscii => "fbx",
        StoredForm::ObjText => "obj",
        StoredForm::Ply => "ply",
        StoredForm::Glb => "glb",
        StoredForm::Gltf => "gltf",
        StoredForm::Usd => "usda",
        StoredForm::Step => "step",
        StoredForm::Iges => "igs",
        StoredForm::Alembic => "abc",
        _ => "bin",
    }
}

/// An identifier of another arithmetic: alice-sdf's identifier for `alice-det-math` 0.5.2 at
/// 39507d1b (`6e98f0be…`), a generation this build does not compute with (its row left
/// SDF_SEMANTICS_BY_DET_MATH with 2d01c63f; `tests/sdf_semantics_id.rs` freezes its pins)
fn other_det_math() -> Provenance {
    let hex = "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37";
    let id: [u8; 32] =
        core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap());
    assert_ne!(id, SDF_SEMANTICS_ID);
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.contains(&id));
    assert!(SDF_SEMANTICS_BY_DET_MATH.iter().all(|(_, sdf)| *sdf != id));
    Provenance::new(id)
}

#[test]
fn every_meaning_form_writes_the_meaning_identifier_and_it_is_current() {
    let m = mesh();
    let caller = Provenance::new([0x5a; 32]);
    let mut compared = 0;
    for form in MEANING_FORMS {
        assert!(form.writes_meaning(), "{form:?}");
        let path = temp(&format!("own_{}.{}", form.name(), ext(form)));
        write(form, &m, &path, &Provenance::sdf());
        let bytes = std::fs::read(&path).unwrap();
        let meaning = meaning_id(form, &Provenance::sdf());
        let raw = Provenance::sdf().semantics_id;
        // the file holds the meaning identifier, and nowhere the raw one
        let holds = |id: &[u8; 32]| {
            bytes.windows(32).any(|w| w == id) || bytes.windows(64).any(|w| w == hex(id).as_bytes())
        };
        assert!(
            holds(&meaning),
            "{form:?}: the meaning identifier is stored"
        );
        assert!(!holds(&raw), "{form:?}: the raw identifier is not stored");
        let stored = provenance::read(&path).unwrap();
        assert_eq!(stored.form, form);
        assert_eq!(stored.id, meaning);
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            StoredSemantics::Current,
            "{form:?}"
        );
        // a caller's identifier: its own meaning identifier, current for that caller
        write(form, &m, &path, &caller);
        assert_eq!(
            provenance::read(&path).unwrap().id,
            meaning_id(form, &caller)
        );
        assert_eq!(
            provenance::check(&path, &caller, &[]).unwrap(),
            StoredSemantics::Current,
            "{form:?}"
        );
        // written under an identifier an earlier release stamped: its meaning identifier, refused
        // by check_sdf (G6's list is empty) and VerifiedPrevious when the judge is given it
        let previous = other_det_math();
        write(form, &m, &path, &previous);
        assert_eq!(
            provenance::read(&path).unwrap().id,
            meaning_id(form, &previous)
        );
        assert!(
            matches!(
                provenance::check_sdf(&path),
                Err(ProvenanceError::MeaningMismatch { .. })
            ),
            "{form:?}"
        );
        assert_eq!(
            provenance::check(&path, &Provenance::sdf(), &[previous.semantics_id]).unwrap(),
            StoredSemantics::VerifiedPrevious(previous),
            "{form:?}"
        );
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, MEANING_FORMS.len());
}

/// The files alice-sdf 7.0.0 wrote for the meaning forms: the raw identifier, under the earlier key
/// or in the slot (the mesh forms of the first set from main 115240f, glTF / GLB / USD / STEP / IGES
/// / Alembic from main 7f22542, each from the sphere of `mesh()` at resolution 4)
const RAW_V2: [(StoredForm, &str); 13] = [
    (StoredForm::StlBinary, "mesh_raw_v2.stl"),
    (StoredForm::StlAscii, "mesh_raw_v2_ascii.stl"),
    (StoredForm::ThreeMf, "mesh_raw_v2.3mf"),
    (StoredForm::FbxBinary, "mesh_raw_v2_binary.fbx"),
    (StoredForm::FbxAscii, "mesh_raw_v2_ascii.fbx"),
    (StoredForm::ObjText, "mesh_raw_v2.obj"),
    (StoredForm::Ply, "mesh_raw_v2.ply"),
    (StoredForm::Glb, "mesh_raw_v2.glb"),
    (StoredForm::Gltf, "mesh_raw_v2.gltf"),
    (StoredForm::Usd, "mesh_raw_v2.usda"),
    (StoredForm::Step, "sphere_raw_v2.step"),
    (StoredForm::Iges, "sphere_raw_v2.igs"),
    (StoredForm::Alembic, "mesh_raw_v2.abc"),
];

/// The raw identifier 7.0.0 wrote: `SDF_SEMANTICS_ID` with alice-det-math 0.5.1, which this build
/// no longer reproduces (not in `PREVIOUS_SDF_SEMANTICS_IDS`)
const SDF_SEMANTICS_ID_7_0_0: [u8; 32] = [
    0x15, 0x10, 0x79, 0x59, 0x5c, 0x1e, 0x8d, 0x8f, 0xc6, 0x61, 0x90, 0x9b, 0xac, 0x7e, 0xa0, 0xdd,
    0xe0, 0xba, 0x5a, 0x8b, 0x3e, 0xf5, 0x00, 0x01, 0x9d, 0x01, 0xc6, 0xaf, 0x64, 0xde, 0xfb, 0x69,
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/provenance")
        .join(name)
}

#[test]
fn a_raw_identifier_of_a_meaning_form_is_the_earlier_generation_refused_unless_listed() {
    // a file written before the form stored meaning identifiers holds the raw identifier: read as
    // the raw identifier, refused by check_sdf and the importers (7.0.0's identifier is not in
    // G6's list), and VerifiedPrevious, never Current, when the judge is given it
    let raw_7_0_0 = Provenance::new(SDF_SEMANTICS_ID_7_0_0);
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.contains(&SDF_SEMANTICS_ID_7_0_0));
    for (form, name) in RAW_V2 {
        let path = fixture(name);
        let stored = provenance::read(&path).unwrap();
        assert_eq!(stored.form, form, "{name}");
        assert_eq!(
            stored.id, SDF_SEMANTICS_ID_7_0_0,
            "{name}: the raw identifier"
        );
        assert!(!stored.is_written_under(&Provenance::sdf()), "{name}");
        assert!(provenance::check_sdf(&path).is_err(), "{name}");
        assert_eq!(
            provenance::check(&path, &Provenance::sdf(), &[SDF_SEMANTICS_ID_7_0_0]).unwrap(),
            StoredSemantics::VerifiedPrevious(raw_7_0_0),
            "{name}: a raw identifier is the earlier generation"
        );
    }
    let named = |e: io::IoError| {
        let text = e.to_string();
        assert!(text.contains(&raw_7_0_0.hex()), "{text}");
    };
    named(io::import_stl(fixture("mesh_raw_v2.stl")).unwrap_err());
    named(io::import_stl(fixture("mesh_raw_v2_ascii.stl")).unwrap_err());
    named(io::import_obj(fixture("mesh_raw_v2.obj")).unwrap_err());
    named(io::import_ply(fixture("mesh_raw_v2.ply")).unwrap_err());
    named(io::import_fbx(fixture("mesh_raw_v2_ascii.fbx")).unwrap_err());
    named(io::import_glb(fixture("mesh_raw_v2.glb")).unwrap_err());
    named(io::import_usda(fixture("mesh_raw_v2.usda")).unwrap_err());
}

#[test]
fn a_file_written_under_another_det_math_is_refused_naming_both_and_the_form() {
    let m = mesh();
    let other = other_det_math();
    for form in MEANING_FORMS {
        let path = temp(&format!("other_{}.{}", form.name(), ext(form)));
        write(form, &m, &path, &other);
        match provenance::check_sdf(&path) {
            Err(ProvenanceError::MeaningMismatch {
                form: f,
                version,
                stored,
                expected,
            }) => {
                assert_eq!(f, form);
                assert_eq!(version, form.version());
                assert_eq!(stored, hex(&meaning_id(form, &other)));
                assert_eq!(expected, hex(&meaning_id(form, &Provenance::sdf())));
            }
            r => panic!("{form:?}: expected MeaningMismatch, got {r:?}"),
        }
        std::fs::remove_file(&path).ok();
    }
}

#[test]
fn the_meaning_identifiers_are_distinct_from_each_other_and_from_the_in_process_ones() {
    let mut ids = vec![
        ("SDF_SEMANTICS_ID".to_string(), SDF_SEMANTICS_ID),
        (
            "alice_det_math::SEMANTICS_ID".to_string(),
            alice_det_math::SEMANTICS_ID,
        ),
    ];
    for form in StoredForm::ALL {
        ids.push((format!("{form:?}"), meaning_id(form, &Provenance::sdf())));
    }
    for (i, (a, x)) in ids.iter().enumerate() {
        for (b, y) in &ids[i + 1..] {
            assert_ne!(x, y, "{a} and {b}");
        }
    }
    assert_eq!(ids.len(), StoredForm::ALL.len() + 2);
    assert_eq!(StoredForm::ALL.len(), 34, "every stored form compared");
}

#[test]
fn the_meaning_identifier_is_law_id_v2_as_its_byte_layout_reads() {
    // oracle: the law-id v2 bytes written out by hand from alice_zip::law_id's documented layout
    // (every length a big-endian u64; a str value tag 0x05 with its length; a u32 tag 0x02)
    for form in StoredForm::ALL {
        let mut h = Sha256::new();
        let domain = b"alice-law-id/v2";
        h.update((domain.len() as u64).to_be_bytes());
        h.update(domain);
        h.update((provenance::MEANING_KIND.len() as u64).to_be_bytes());
        h.update(provenance::MEANING_KIND);
        h.update(SDF_SEMANTICS_ID);
        h.update([0x05]);
        h.update((form.name().len() as u64).to_be_bytes());
        h.update(form.name().as_bytes());
        h.update([0x02]);
        h.update(form.version().to_be_bytes());
        let want: [u8; 32] = h.finalize().into();
        assert_eq!(meaning_id(form, &Provenance::sdf()), want, "{form:?}");
    }
}

#[test]
fn the_meshopt_glb_writer_stores_the_glb_meaning_identifier() {
    // the meshopt path of the GLB writer builds its own JSON: it stores the same identifier as the
    // plain path, read back as Current and imported as Current
    let path = temp("own_meshopt.glb");
    let cfg = io::GltfConfig {
        meshopt_compress: true,
        ..Default::default()
    };
    io::export_glb_with_provenance(&mesh(), &path, &cfg, None, &Provenance::sdf()).unwrap();
    let stored = provenance::read(&path).unwrap();
    assert_eq!(stored.form, StoredForm::Glb);
    assert_eq!(stored.id, meaning_id(StoredForm::Glb, &Provenance::sdf()));
    assert_eq!(
        provenance::check_sdf(&path).unwrap(),
        StoredSemantics::Current
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn a_document_holding_both_keys_is_refused_never_read_as_either() {
    // glTF JSON and USDA read the identifier through their own walk (not the text-key scan of the
    // other text forms): a document carrying both the meaning key and the earlier key is
    // unreadable, as it is for the text forms (Alembic, whose metadata block is length framed, is
    // covered in test_provenance.rs through an archive built with both keys)
    let m = mesh();
    let mut compared = 0;
    for form in [StoredForm::Gltf, StoredForm::Usd] {
        let path = temp(&format!("both_{}.{}", form.name(), ext(form)));
        write(form, &m, &path, &Provenance::sdf());
        let text = String::from_utf8(std::fs::read(&path).unwrap()).unwrap();
        let raw = hex(&SDF_SEMANTICS_ID);
        let both = if form == StoredForm::Gltf {
            let extra = format!(r#""extras":{{"{}":"{raw}","#, provenance::KEY);
            text.replacen(r#""extras":{"#, &extra, 1)
        } else {
            let extra = format!(
                "customLayerData = {{ string \"{}\" = \"{raw}\", ",
                provenance::KEY
            );
            text.replacen("customLayerData = { ", &extra, 1)
        };
        assert_ne!(both, text, "{form:?}: the earlier key was added");
        std::fs::write(&path, both.as_bytes()).unwrap();
        match provenance::read(&path) {
            Err(ProvenanceError::Unreadable(why)) => {
                assert!(why.contains(provenance::MEANING_KEY), "{form:?}: {why}");
            }
            r => panic!("{form:?}: expected Unreadable, got {r:?}"),
        }
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, 2);
}

#[test]
fn every_stored_form_writes_meaning_identifiers_under_its_own_name() {
    // ALICE-Law G8: no form of this crate still stores the raw identifier; each form has its own
    // name, so no two forms share a meaning identifier
    let mut names = std::collections::BTreeSet::new();
    for form in StoredForm::ALL {
        assert!(form.writes_meaning(), "{form:?}");
        assert!(
            names.insert(form.name()),
            "{form:?}: the name {} twice",
            form.name()
        );
        assert_eq!(
            form.version(),
            1,
            "{form:?}: the layout version this build writes"
        );
    }
    assert_eq!(names.len(), 34);
}

#[test]
fn the_meaning_identifier_agrees_with_the_independent_python_one() {
    // oracle: python/tests/meaning_id.py (standard library only, from ALICE-Zip's documented
    // law-id v2 layout) computed these for this build's semantics identifier; the Python binding
    // tests use the same file against what the writers store
    let sdf = Provenance::sdf();
    let pinned = [
        (
            StoredForm::JsonDocument,
            "a7541b24b44eb49edac41d3f6deacb0770028128d24e39f376f4021d9a6a8ae4",
        ),
        (
            StoredForm::Vdb,
            "0233911e4f3aabac7db28568878192b0445842fcac400ff1300610e7423e354a",
        ),
    ];
    for (form, want) in pinned {
        assert_eq!(hex(&meaning_id(form, &sdf)), want, "{form:?}");
    }
    // the producer meaning identifier, the same oracle's `producer_meaning_id`, for a producer
    // other than this crate (ALICE-LOL's geometry identifier, the producer it stamps)
    let lol =
        Provenance::from_hex("d1cbba69a1b33ce7363ac178f16ad0cb062699ff8c9c76af7058c6d006bc8a24")
            .unwrap();
    let pinned = [
        (
            StoredForm::StlBinary,
            "a43ff2e7a95c54ea7da6dac461534073279badbc1ccfc25370007329b368db9e",
        ),
        (
            StoredForm::ThreeMf,
            "8fc5289f1a0e2c8b502db7b1ed18a76fd0376d0f223a1cab607b799dc6d81f79",
        ),
    ];
    for (form, want) in pinned {
        assert_eq!(
            hex(&alice_sdf::provenance::producer_meaning_id(form, &lol)),
            want,
            "producer {form:?}"
        );
    }
    // under its own kind: never the primary meaning identifier of the same identifier
    for form in StoredForm::ALL {
        assert_ne!(
            alice_sdf::provenance::producer_meaning_id(form, &sdf),
            meaning_id(form, &sdf),
            "{form:?}"
        );
    }
}

/// The files the writers of the forms converted last stored before they wrote meaning
/// identifiers: written by main at eb84934 / 7b8ddf9 (the last commit whose writers stored the raw
/// identifier) through each form's plain writer, from the inputs of `tests/test_provenance.rs`;
/// the shader text is main's generated Unreal corpus shader. Each holds this crate's raw
/// identifier (`SDF_SEMANTICS_ID`, not 7.0.0's). The mesh codec's buffers (magic `ASDF`, as the ASDF
/// tree's) are read by their decoder below, not by `from_bytes`
const RAW_MAIN: [(StoredForm, &str); 14] = [
    (StoredForm::JsonDocument, "tree_raw_main.asdf.json"),
    (StoredForm::Asdf, "tree_raw_main.asdf"),
    (StoredForm::Abm, "mesh_raw_main.abm"),
    (StoredForm::Nanite, "mesh_raw_main.nanite"),
    (StoredForm::Ue5Mesh, "mesh_raw_main.ue5_mesh_bin"),
    (StoredForm::UnityMesh, "mesh_raw_main.unity_mesh_bin"),
    (StoredForm::Neural, "net_raw_main.nsdf"),
    (StoredForm::Vdb, "grid_raw_main.alicevdb"),
    (StoredForm::Svo, "chunk_raw_main.svoc"),
    (StoredForm::Vox, "model_raw_main.vox"),
    (StoredForm::Dds, "volume_raw_main.dds"),
    (StoredForm::Png, "heightmap_raw_main.png"),
    (StoredForm::ShaderText, "shader_raw_main.usf"),
    (StoredForm::Sidecar, "volume_raw_main.raw"),
];

#[test]
fn a_raw_file_this_crate_wrote_before_meaning_identifiers_is_the_previous_generation() {
    // read back: this crate's raw identifier, through the earlier key or a slot; judged the
    // previous generation (VerifiedPrevious), never Current, and the importers read it
    let raw = Provenance::sdf();
    let mut compared = 0;
    for (form, name) in RAW_MAIN {
        let path = fixture(name);
        let bytes = std::fs::read(&path).unwrap();
        let stored = if form == StoredForm::Sidecar {
            provenance::read(&path).unwrap()
        } else {
            provenance::from_bytes(&bytes).unwrap()
        };
        assert_eq!(stored.form, form, "{name}");
        assert_eq!(stored.id, SDF_SEMANTICS_ID, "{name}: the raw identifier");
        assert_ne!(stored.via, provenance::StoredVia::MeaningKey, "{name}");
        assert!(
            !stored.is_written_under(&raw),
            "{name}: not what this build writes"
        );
        assert_eq!(
            stored.judge(&raw, PREVIOUS_SDF_SEMANTICS_IDS).unwrap(),
            StoredSemantics::VerifiedPrevious(raw),
            "{name}: the previous generation"
        );
        compared += 1;
    }
    assert_eq!(compared, RAW_MAIN.len());
    // the loaders read them as the previous generation (one per kind of reader)
    let tree = io::load_asdf_json_unverified(fixture("tree_raw_main.asdf.json")).unwrap();
    assert_eq!(tree.semantics(), StoredSemantics::VerifiedPrevious(raw));
    let tree = io::load_asdf_unverified(fixture("tree_raw_main.asdf")).unwrap();
    assert_eq!(tree.semantics(), StoredSemantics::VerifiedPrevious(raw));
    let idx = alice_sdf::mesh::mesh_codec::decode_indices_unverified(
        &std::fs::read(fixture("mesh_codec_indices_raw_main.bin")).unwrap(),
    )
    .unwrap();
    assert_eq!(idx.semantics(), StoredSemantics::VerifiedPrevious(raw));
    assert_eq!(idx.into_parts().0, [0, 1, 2, 2, 1, 3]);
    // the meshopt stamp, the encoded volume and the ASP I-packet are read through their decoders
    let verts = alice_sdf::mesh::meshopt_vertex_codec::decode_vertex_buffer(
        &std::fs::read(fixture("meshopt_vertex_raw_main.bin")).unwrap(),
        4,
        12,
    )
    .unwrap();
    assert_eq!(verts.len(), 48);
    #[cfg(feature = "codec")]
    {
        let volume = alice_sdf::codec_bridge::try_decode_sdf_volume_unverified(
            &std::fs::read(fixture("codec_volume_raw_main.bin")).unwrap(),
        )
        .unwrap();
        assert_eq!(volume.semantics(), StoredSemantics::VerifiedPrevious(raw));
    }
    #[cfg(feature = "asp")]
    {
        let packet: libasp::AspPacket = serde_json::from_str(
            &std::fs::read_to_string(fixture("i_packet_raw_main.json")).unwrap(),
        )
        .unwrap();
        let back = alice_sdf::asp_bridge::decode_sdf_i_packet_unverified(&packet).unwrap();
        assert_eq!(back.semantics(), StoredSemantics::VerifiedPrevious(raw));
    }
}

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
const MEANING_FORMS: [StoredForm; 7] = [
    StoredForm::StlBinary,
    StoredForm::StlAscii,
    StoredForm::ThreeMf,
    StoredForm::FbxBinary,
    StoredForm::FbxAscii,
    StoredForm::ObjText,
    StoredForm::Ply,
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
        _ => "bin",
    }
}

/// An identifier of another arithmetic: alice-sdf's identifier for `alice-det-math` 0.5.0, a
/// generation this build does not compute with (its row in SDF_SEMANTICS_BY_DET_MATH)
fn other_det_math() -> Provenance {
    let det_050 = SDF_SEMANTICS_BY_DET_MATH
        .iter()
        .find(|(dm, _)| *dm != alice_det_math::SEMANTICS_ID)
        .expect("a row of another alice-det-math generation");
    assert_ne!(det_050.1, SDF_SEMANTICS_ID);
    assert!(!PREVIOUS_SDF_SEMANTICS_IDS.contains(&det_050.1));
    Provenance::new(det_050.1)
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
        // written under one of G6's earlier identifiers: its meaning identifier, VerifiedPrevious
        let previous = Provenance::new(PREVIOUS_SDF_SEMANTICS_IDS[0]);
        write(form, &m, &path, &previous);
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            StoredSemantics::VerifiedPrevious(previous),
            "{form:?}"
        );
        std::fs::remove_file(&path).ok();
        compared += 1;
    }
    assert_eq!(compared, MEANING_FORMS.len());
}

/// The files alice-sdf 7.0.0 (main 115240f) wrote for the meaning forms: the raw identifier, under
/// the earlier key or in the slot
const RAW_V2: [(StoredForm, &str); 7] = [
    (StoredForm::StlBinary, "mesh_raw_v2.stl"),
    (StoredForm::StlAscii, "mesh_raw_v2_ascii.stl"),
    (StoredForm::ThreeMf, "mesh_raw_v2.3mf"),
    (StoredForm::FbxBinary, "mesh_raw_v2_binary.fbx"),
    (StoredForm::FbxAscii, "mesh_raw_v2_ascii.fbx"),
    (StoredForm::ObjText, "mesh_raw_v2.obj"),
    (StoredForm::Ply, "mesh_raw_v2.ply"),
];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/provenance")
        .join(name)
}

#[test]
fn a_raw_identifier_of_a_meaning_form_reads_as_the_earlier_generation() {
    // a file written before the form stored meaning identifiers holds the raw identifier: it is
    // VerifiedPrevious, never Current, and the importers read it the same way
    for (form, name) in RAW_V2 {
        let path = fixture(name);
        let stored = provenance::read(&path).unwrap();
        assert_eq!(stored.form, form, "{name}");
        assert_eq!(stored.id, SDF_SEMANTICS_ID, "{name}: the raw identifier");
        assert!(!stored.is_written_under(&Provenance::sdf()), "{name}");
        assert_eq!(
            provenance::check_sdf(&path).unwrap(),
            StoredSemantics::VerifiedPrevious(Provenance::sdf()),
            "{name}: a raw identifier is the earlier generation"
        );
    }
    let earlier = StoredSemantics::VerifiedPrevious(Provenance::sdf());
    assert_eq!(
        io::import_stl(fixture("mesh_raw_v2.stl"))
            .unwrap()
            .semantics(),
        earlier
    );
    assert_eq!(
        io::import_stl(fixture("mesh_raw_v2_ascii.stl"))
            .unwrap()
            .semantics(),
        earlier
    );
    assert_eq!(
        io::import_obj(fixture("mesh_raw_v2.obj"))
            .unwrap()
            .semantics(),
        earlier
    );
    assert_eq!(
        io::import_ply(fixture("mesh_raw_v2.ply"))
            .unwrap()
            .semantics(),
        earlier
    );
    assert_eq!(
        io::import_fbx(fixture("mesh_raw_v2_ascii.fbx"))
            .unwrap()
            .semantics(),
        earlier
    );

    // one of G6's earlier identifiers in a binary slot is VerifiedPrevious as well
    let mut b = std::fs::read(fixture("mesh_raw_v2.stl")).unwrap();
    b[9..41].copy_from_slice(&PREVIOUS_SDF_SEMANTICS_IDS[0]);
    let path = temp("raw_previous.stl");
    std::fs::write(&path, &b).unwrap();
    assert_eq!(
        provenance::check_sdf(&path).unwrap(),
        StoredSemantics::VerifiedPrevious(Provenance::new(PREVIOUS_SDF_SEMANTICS_IDS[0]))
    );
    std::fs::remove_file(&path).ok();
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
    for form in MEANING_FORMS {
        ids.push((format!("{form:?}"), meaning_id(form, &Provenance::sdf())));
    }
    for (i, (a, x)) in ids.iter().enumerate() {
        for (b, y) in &ids[i + 1..] {
            assert_ne!(x, y, "{a} and {b}");
        }
    }
    assert_eq!(ids.len(), MEANING_FORMS.len() + 2);
}

#[test]
fn the_meaning_identifier_is_law_id_v2_as_its_byte_layout_reads() {
    // oracle: the law-id v2 bytes written out by hand from alice_zip::law_id's documented layout
    // (every length a big-endian u64; a str value tag 0x05 with its length; a u32 tag 0x02)
    for form in MEANING_FORMS {
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

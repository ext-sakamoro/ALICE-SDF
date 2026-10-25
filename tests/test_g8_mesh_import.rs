//! ALICE-Law G8: the mesh importers read the semantics identifier this crate's writers store
//!
//! A mesh file is external input: a file from another program has no identifier. Each
//! importer reads it and marks it `StoredSemantics::Unpinned` in the `Checked` it returns,
//! reads this crate's files as `Current` (or `VerifiedPrevious` for an earlier identifier it
//! still reproduces), and refuses a file stamped by another SDF arithmetic. The `*_pinned`
//! importers also refuse a file without an identifier. A file that holds the key with a value
//! that is not an identifier is an error, never read as unstamped.
//!
//! The files without an identifier are the bytes the published alice-sdf 6.1.0 wrote
//! (`tests/fixtures/provenance/mesh_v1.*`, a sphere meshed at resolution 4: 56 triangles).

#[path = "common/temp.rs"]
mod temp;

use alice_sdf::io::{self, IoError};
use alice_sdf::mesh::{sdf_to_mesh, MarchingCubesConfig, Mesh};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{meaning_id, Provenance, StoredForm, StoredSemantics};
use std::path::{Path, PathBuf};

const V1_TRIANGLES: usize = 56;

/// The forms whose meaning identifier a writer may store
const FORMS: [StoredForm; 13] = [
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

fn hex(id: &[u8; 32]) -> String {
    id.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write as _;
        write!(s, "{b:02x}").expect("writing to a String");
        s
    })
}

/// The 64 hex digits a writer stored for `p` in `bytes`: its meaning identifier for the file's form
/// (a form that stores meaning identifiers), or the raw identifier
fn stored_hex_in(bytes: &[u8], p: &Provenance) -> String {
    let found = |h: &String| bytes.windows(64).any(|w| w == h.as_bytes());
    FORMS
        .iter()
        .map(|f| hex(&meaning_id(*f, p)))
        .chain(std::iter::once(p.hex()))
        .find(found)
        .expect("the written file holds this identifier, raw or as a meaning identifier")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/provenance")
        .join(name)
}

fn temp(name: &str) -> temp::TempPath {
    temp::TempPath::new("alice_sdf_g8_mesh_import", name)
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

const fn other() -> Provenance {
    Provenance::new([0x5a; 32])
}

const fn earlier() -> Provenance {
    Provenance::new(alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS[0])
}

type Write = fn(&Mesh, &Path, &Provenance);
type Read = fn(&Path) -> Result<(usize, StoredSemantics), IoError>;
type ReadPinned = fn(&Path) -> Result<usize, IoError>;

/// One importer: its file name, a writer with an identifier, the default reader and the
/// pinned reader, each giving the triangle count
struct Format {
    name: &'static str,
    file: &'static str,
    write: Write,
    read: Read,
    read_pinned: ReadPinned,
}

const fn tris(m: &Mesh) -> usize {
    m.indices.len() / 3
}

fn formats() -> Vec<Format> {
    vec![
        Format {
            name: "STL",
            file: "mesh_v1.stl",
            write: |m, p, v| io::export_stl_with_provenance(m, p, v).unwrap(),
            read: |p| {
                io::import_stl(p).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_stl_pinned(p).map(|m| tris(&m)),
        },
        Format {
            name: "OBJ",
            file: "mesh_v1.obj",
            write: |m, p, v| {
                io::export_obj_with_provenance(m, p, &io::ObjConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_obj(p).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_obj_pinned(p).map(|m| tris(&m)),
        },
        Format {
            name: "PLY",
            file: "mesh_v1.ply",
            write: |m, p, v| {
                io::export_ply_with_provenance(m, p, &io::PlyConfig::default(), v).unwrap();
            },
            read: |p| {
                io::import_ply(p).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_ply_pinned(p).map(|m| tris(&m)),
        },
        Format {
            name: "FBX",
            file: "mesh_v1.fbx",
            write: |m, p, v| {
                io::export_fbx_with_provenance(m, p, &io::FbxConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_fbx(p).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_fbx_pinned(p).map(|m| tris(&m)),
        },
        Format {
            name: "FBX (full)",
            file: "mesh_v1.fbx",
            write: |m, p, v| {
                io::export_fbx_with_provenance(m, p, &io::FbxConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_fbx_full(p).map(|c| {
                    let (v, s) = c.into_parts();
                    (tris(&v.mesh), s)
                })
            },
            read_pinned: |p| io::import_fbx_full_pinned(p).map(|f| tris(&f.mesh)),
        },
        Format {
            name: "GLB",
            file: "mesh_v1.glb",
            write: |m, p, v| {
                io::export_glb_with_provenance(m, p, &io::GltfConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_glb(p).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_glb_pinned(p).map(|m| tris(&m)),
        },
        Format {
            name: "GLB (bytes)",
            file: "mesh_v1.glb",
            write: |m, p, v| {
                io::export_glb_with_provenance(m, p, &io::GltfConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_glb_bytes(&std::fs::read(p)?).map(|c| {
                    let (m, s) = c.into_parts();
                    (tris(&m), s)
                })
            },
            read_pinned: |p| io::import_glb_bytes_pinned(&std::fs::read(p)?).map(|m| tris(&m)),
        },
        Format {
            name: "USDA",
            file: "mesh_v1.usda",
            write: |m, p, v| {
                io::export_usda_with_provenance(m, p, &io::UsdConfig::default(), None, v).unwrap();
            },
            read: |p| {
                io::import_usda(p).map(|c| {
                    let (v, s) = c.into_parts();
                    (tris(&v.mesh), s)
                })
            },
            read_pinned: |p| io::import_usda_pinned(p).map(|u| tris(&u.mesh)),
        },
    ]
}

fn ext(f: &Format) -> &'static str {
    f.file.rsplit('.').next().unwrap()
}

#[test]
fn a_file_from_6_1_0_is_read_unpinned_and_refused_by_the_pinned_importer() {
    for f in formats() {
        let path = fixture(f.file);
        let bytes = std::fs::read(&path).unwrap();
        assert!(
            !bytes
                .windows(provenance_key().len())
                .any(|w| w == provenance_key()),
            "{}: the fixture holds no identifier",
            f.name
        );
        let (n, semantics) = (f.read)(&path).unwrap_or_else(|e| panic!("{}: {e}", f.name));
        assert_eq!(
            (n, semantics),
            (V1_TRIANGLES, StoredSemantics::Unpinned),
            "{}",
            f.name
        );
        match (f.read_pinned)(&path) {
            Err(IoError::Unpinned { .. }) => {}
            other => panic!("{}: expected Unpinned, got {other:?}", f.name),
        }
    }
}

const fn provenance_key() -> &'static [u8] {
    alice_sdf::provenance::KEY.as_bytes()
}

#[test]
fn this_crates_file_reads_as_current_and_through_the_pinned_importer() {
    let m = mesh();
    for f in formats() {
        let path = temp(&format!(
            "current_{}.{}",
            f.name.replace([' ', '(', ')'], ""),
            ext(&f)
        ));
        (f.write)(&m, &path, &Provenance::sdf());
        assert_eq!(
            (f.read)(&path).unwrap_or_else(|e| panic!("{}: {e}", f.name)),
            (tris(&m), StoredSemantics::Current),
            "{}",
            f.name
        );
        assert_eq!((f.read_pinned)(&path).unwrap(), tris(&m), "{}", f.name);
    }
}

#[test]
fn an_earlier_published_identifier_reads_as_verified_previous() {
    let m = mesh();
    for f in formats() {
        let path = temp(&format!(
            "earlier_{}.{}",
            f.name.replace([' ', '(', ')'], ""),
            ext(&f)
        ));
        (f.write)(&m, &path, &earlier());
        assert_eq!(
            (f.read)(&path).unwrap().1,
            StoredSemantics::VerifiedPrevious(earlier()),
            "{}",
            f.name
        );
        assert!((f.read_pinned)(&path).is_ok(), "{}", f.name);
    }
}

#[test]
fn another_identifier_is_refused_by_both_importers() {
    let m = mesh();
    for f in formats() {
        let path = temp(&format!(
            "other_{}.{}",
            f.name.replace([' ', '(', ')'], ""),
            ext(&f)
        ));
        (f.write)(&m, &path, &other());
        for result in [(f.read)(&path).map(|r| r.0), (f.read_pinned)(&path)] {
            match result {
                // a form that stores meaning identifiers names the stored one, this crate's and the form
                Err(IoError::MeaningMismatch {
                    form,
                    version,
                    stored,
                    expected,
                    ..
                }) => {
                    assert_eq!(version, form.version(), "{}", f.name);
                    assert_eq!(
                        (stored, expected),
                        (
                            hex(&meaning_id(form, &other())),
                            hex(&meaning_id(form, &Provenance::sdf()))
                        ),
                        "{}",
                        f.name
                    );
                }
                Err(IoError::SemanticsMismatch {
                    stored, expected, ..
                }) => {
                    assert_eq!(
                        (stored, expected),
                        (other(), Provenance::sdf()),
                        "{}",
                        f.name
                    );
                }
                r => panic!("{}: expected a mismatch, got {r:?}", f.name),
            }
        }
    }
}

#[test]
fn a_binary_stl_header_of_an_unknown_version_is_an_error_not_unstamped() {
    // this crate's binary STL header (magic, version, identifier): a version it does not know,
    // or a version 2 header whose padding is not zero, is refused by both importers
    let m = mesh();
    let path = temp("header.stl");
    io::export_stl_with_provenance(&m, &path, &Provenance::sdf()).unwrap();
    let good = std::fs::read(&path).unwrap();
    assert_eq!(&good[..8], alice_sdf::provenance::STL_MAGIC);
    let mut cases: Vec<(&str, Vec<u8>)> = Vec::new();
    for v in [0u8, 3, 255] {
        let mut b = good.clone();
        b[8] = v;
        cases.push(("version", b));
    }
    let mut b = good;
    b[8] = alice_sdf::provenance::STL_HEADER_VERSION_PRODUCER;
    b[41..48].fill(1);
    cases.push(("version 2 with padding", b));
    for (what, bytes) in cases {
        std::fs::write(&path, &bytes).unwrap();
        for result in [
            io::import_stl(&path).map(|_| ()),
            io::import_stl_pinned(&path).map(|_| ()),
        ] {
            match result {
                Err(IoError::InvalidFormat(why)) => {
                    assert!(why.contains("binary STL"), "{what}: {why}");
                }
                r => panic!(
                    "{what} (byte 8 = {}): expected InvalidFormat, got {r:?}",
                    bytes[8]
                ),
            }
        }
    }
}

#[test]
fn a_malformed_identifier_in_an_ascii_stl_is_an_error() {
    let m = mesh();
    let path = temp("bad_ascii.stl");
    io::export_stl_ascii_with_provenance(&m, &path, &Provenance::sdf()).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let hex = stored_hex_in(text.as_bytes(), &Provenance::sdf());
    assert_eq!(
        hex,
        self::hex(&meaning_id(StoredForm::StlAscii, &Provenance::sdf()))
    );
    std::fs::write(&path, text.replace(&hex, &"z".repeat(64))).unwrap();
    match io::import_stl(&path) {
        Err(IoError::InvalidFormat(why)) => assert!(why.contains("STL"), "{why}"),
        r => panic!("expected InvalidFormat, got {r:?}"),
    }
}

#[test]
fn a_malformed_identifier_is_an_error_not_unstamped() {
    // the formats that store the identifier as text: its 64 hex digits replaced by non-hex
    // characters
    let m = mesh();
    for f in formats()
        .into_iter()
        .filter(|f| matches!(f.name, "OBJ" | "PLY" | "FBX" | "GLB" | "USDA"))
    {
        let path = temp(&format!("bad_{}.{}", f.name, ext(&f)));
        (f.write)(&m, &path, &Provenance::sdf());
        let bytes = std::fs::read(&path).unwrap();
        let hex = stored_hex_in(&bytes, &Provenance::sdf());
        let at = bytes
            .windows(64)
            .position(|w| w == hex.as_bytes())
            .unwrap_or_else(|| panic!("{}: the written file holds the identifier", f.name));
        let mut bad = bytes;
        bad[at..at + 64].fill(b'z');
        std::fs::write(&path, bad).unwrap();
        match (f.read)(&path) {
            Err(IoError::InvalidFormat(why)) => {
                assert!(why.contains(f.name.split(' ').next().unwrap()), "{why}");
            }
            r => panic!("{}: expected InvalidFormat, got {r:?}", f.name),
        }
    }
}

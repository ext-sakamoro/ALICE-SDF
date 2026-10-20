//! ALICE-Law G8: the two stored forms `examples/bake_assets` writes itself (`law.bytecode` and
//! `manifest.json`) store the semantics identifier, and their readers check it
//!
//! oracle: the identifier is the 32 bytes `alice_sdf::SDF_SEMANTICS_ID`; a reader accepts it and
//! the earlier `PREVIOUS_SDF_SEMANTICS_IDS` and refuses another one, a form without one and a cut
//! header (the rule of `provenance::check_sdf`). The bytecode identifier sits at bytes 8..40 after
//! the magic `ASDFBC02`, read here by slicing, not by the module's reader.
//!
//! Author: Moroya Sakamoto

#[path = "../examples/bake_assets/stored.rs"]
mod stored;

use alice_sdf::prelude::*;
use alice_sdf::provenance::{Provenance, StoredSemantics};
use alice_sdf::{PREVIOUS_SDF_SEMANTICS_IDS, SDF_SEMANTICS_ID};
use std::path::{Path, PathBuf};

type Write = fn(&Path, &CompiledSdf, &serde_json::Value, &Provenance) -> Result<(), String>;

/// The writers of the bake's own stored forms, each writing with a given identifier
const WRITERS: &[(&str, &str, Write)] = &[
    (
        "examples/bake_assets/stored.rs::write_bytecode",
        "law.bytecode",
        |p, c, _, id| stored::write_bytecode_with_provenance(p, c, id),
    ),
    (
        "examples/bake_assets/stored.rs::write_manifest",
        "manifest.json",
        |p, _, m, id| stored::write_manifest_with_provenance(p, m, id),
    ),
];

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("alice_sdf_bake_stored_{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn compiled() -> CompiledSdf {
    CompiledSdf::compile(&SdfNode::sphere(1.0).subtract(SdfNode::box3d(0.6, 0.6, 0.6)))
}

fn manifest() -> serde_json::Value {
    serde_json::json!({ "format": "alice-sdf-bake/2", "scenes": [] })
}

fn status(name: &str, path: &Path) -> Result<StoredSemantics, String> {
    if name == "law.bytecode" {
        stored::bytecode_semantics(&std::fs::read(path).unwrap())
    } else {
        stored::manifest_semantics(path)
    }
}

/// The identifier as the file holds it, read without the module's readers
fn stored_id(name: &str, path: &Path) -> [u8; 32] {
    let bytes = std::fs::read(path).unwrap();
    if name == "law.bytecode" {
        assert_eq!(&bytes[..8], b"ASDFBC02");
        bytes[8..40].try_into().unwrap()
    } else {
        let doc: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let hex = doc["alice:semantics_id"].as_str().unwrap();
        let mut id = [0u8; 32];
        for (i, b) in id.iter_mut().enumerate() {
            *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap();
        }
        id
    }
}

#[test]
fn bake_writers_store_the_identifier_and_readers_check_it() {
    let (c, m) = (compiled(), manifest());
    let other = Provenance::new([0xAB; 32]);
    for (key, name, write) in WRITERS {
        let path = dir().join(format!("current_{name}"));
        // the plain writer stores this crate's identifier
        if *name == "law.bytecode" {
            stored::write_bytecode(&path, &c).unwrap();
        } else {
            stored::write_manifest(&path, &m).unwrap();
        }
        assert_eq!(stored_id(name, &path), SDF_SEMANTICS_ID, "{key}");
        assert_eq!(status(name, &path), Ok(StoredSemantics::Current), "{key}");
        // an earlier identifier this build still reproduces
        let prev = Provenance::new(PREVIOUS_SDF_SEMANTICS_IDS[0]);
        write(&path, &c, &m, &prev).unwrap();
        assert_eq!(stored_id(name, &path), prev.semantics_id, "{key}");
        assert_eq!(
            status(name, &path),
            Ok(StoredSemantics::VerifiedPrevious(prev)),
            "{key}"
        );
        // another identifier is refused, naming both
        write(&path, &c, &m, &other).unwrap();
        let e = status(name, &path).unwrap_err();
        assert!(
            e.contains(&other.hex()) && e.contains(&Provenance::sdf().hex()),
            "{key}: {e}"
        );
    }
}

#[test]
fn bake_forms_without_the_identifier_or_cut_are_refused() {
    let c = compiled();
    let good = stored::bytecode_bytes(&c, &Provenance::sdf());
    // the earlier layout: the same bytes without the identifier
    let mut v1 = b"ASDFBC01".to_vec();
    v1.extend_from_slice(&good[40..]);
    assert!(stored::bytecode_semantics(&v1)
        .unwrap_err()
        .contains("ASDFBC01"));
    // a header cut inside the identifier, and an unknown magic
    assert!(stored::bytecode_semantics(&good[..39])
        .unwrap_err()
        .contains("cut"));
    assert!(stored::bytecode_semantics(b"ASDFBC03").is_err());
    // the bytes after the identifier are the compiled program, the same for any identifier
    let other = stored::bytecode_bytes(&c, &Provenance::new([0xAB; 32]));
    assert_eq!(good[40..], other[40..]);
    assert_eq!(
        good.len(),
        40 + 12 + 44 * c.instructions().len() + 4 * c.aux_data().len()
    );

    // a manifest without the identifier
    let path = dir().join("unpinned_manifest.json");
    std::fs::write(&path, serde_json::to_string(&manifest()).unwrap()).unwrap();
    assert!(stored::manifest_semantics(&path).is_err());
}

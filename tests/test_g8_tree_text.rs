//! ALICE-Law G8: the SDF tree's own text and packet forms carry the semantics identifier and
//! their readers check it.
//!
//! * `.asdf.json` (`io::save_asdf_json` / `to_json_string`): a top-level `"alice:semantics_id"`.
//!   `load_asdf_json`, `from_json_string` and `io::load` refuse a document without one and one
//!   with another identifier; the `*_unverified` readers return the tree with its status.
//! * the ASP I-packet: the identifier in the first 32 coefficient entries, marked by the
//!   `alice_semantics_v` parameter; `decode_sdf_i_packet` refuses as above,
//!   `decode_sdf_i_packet_unverified` returns the status.
//! * `llm_schema::validate_sdf_json` reads a model's answer, which carries no identifier: it is
//!   read and marked Unpinned, and a document stamped by another SDF arithmetic is refused.
//!
//! The documents without an identifier are the bytes the published alice-sdf 6.1.0 wrote
//! (`tests/fixtures/provenance/tree_v1.asdf.json`, `i_packet_v1.json`, a sphere unioned with a
//! translated box: 4 nodes).

#[path = "common/temp.rs"]
mod temp;

use alice_sdf::io::{self, IoError};
use alice_sdf::llm_schema::validate_sdf_json;
use alice_sdf::prelude::*;
use alice_sdf::provenance::{self, Provenance, StoredSemantics};

const V1_JSON: &str = include_str!("fixtures/provenance/tree_v1.asdf.json");

fn tree() -> SdfTree {
    SdfTree::new(SdfNode::sphere(1.0).union(SdfNode::box3d(0.5, 0.5, 0.5).translate(1.0, 0.0, 0.0)))
}

fn temp(name: &str) -> temp::TempPath {
    temp::TempPath::new("alice_sdf_g8_tree_text", name)
}

const fn other() -> Provenance {
    Provenance::new([0x5a; 32])
}

/// The identifier alice-sdf stamped with alice-det-math 0.5.2 at 39507d1b (`6e98f0be…`), which
/// this build no longer reproduces
fn earlier() -> Provenance {
    let hex = "6e98f0bed3f8691ca470ae2fe2c404c893856a3e3f6a9cda9d67528ae8074b37";
    let id: [u8; 32] =
        core::array::from_fn(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap());
    assert!(!alice_sdf::PREVIOUS_SDF_SEMANTICS_IDS.contains(&id));
    Provenance::new(id)
}

/// (writer as scripts/provenance_writers.py names it, what it writes): the text and packet forms
/// of this file, each stamped and read back by the tests below
const TEXT_WRITERS: &[(&str, &str)] = &[
    ("src/io/json.rs::to_json_string", "ASDF JSON text"),
    (
        "src/asp_bridge.rs::create_sdf_i_packet",
        "ASP I-packet (feature asp)",
    ),
];

#[test]
fn asdf_json_carries_and_checks_the_identifier() {
    assert!(TEXT_WRITERS
        .iter()
        .any(|(w, _)| *w == "src/io/json.rs::to_json_string"));
    let t = tree();
    let path = temp("stamped.asdf.json");
    io::save_asdf_json(&t, &path).unwrap();
    // the stored field, read by the format-independent reader and by the loaders
    assert_eq!(
        provenance::read(&path).unwrap().id,
        Provenance::sdf().semantics_id
    );
    assert_eq!(
        io::load_asdf_json(&path).unwrap().node_count(),
        t.node_count()
    );
    assert_eq!(io::load(&path).unwrap().node_count(), t.node_count());
    let checked = io::load_asdf_json_unverified(&path).unwrap();
    assert_eq!(checked.semantics(), StoredSemantics::Current);
    // the string form is the same document
    let json = io::to_json_string(&t).unwrap();
    assert!(json.contains(&Provenance::sdf().tagged().replace('=', "\": \"")));
    assert_eq!(
        io::from_json_string(&json).unwrap().node_count(),
        t.node_count()
    );
}

#[test]
fn an_asdf_json_from_6_1_0_is_refused_by_default_and_read_unverified() {
    assert!(!V1_JSON.contains(provenance::KEY));
    match io::from_json_string(V1_JSON) {
        Err(IoError::Unpinned { format }) => assert_eq!(format, "ASDF JSON"),
        other => panic!("expected Unpinned, got {other:?}"),
    }
    let path = temp("v1.asdf.json");
    std::fs::write(&path, V1_JSON).unwrap();
    assert!(matches!(
        io::load_asdf_json(&path),
        Err(IoError::Unpinned { .. })
    ));
    assert!(matches!(io::load(&path), Err(IoError::Unpinned { .. })));
    assert!(matches!(
        provenance::read(&path),
        Err(provenance::ProvenanceError::Missing { .. })
    ));
    let (back, semantics) = io::from_json_string_unverified(V1_JSON)
        .unwrap()
        .into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(back.node_count(), 4);
    let (back, semantics) = io::load_asdf_json_unverified(&path).unwrap().into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(back.node_count(), 4);
}

#[test]
fn asdf_json_with_another_identifier_is_refused_even_unverified() {
    let json = io::to_json_string_with_provenance(&tree(), &other()).unwrap();
    for result in [
        io::from_json_string(&json).map(|_| ()),
        io::from_json_string_unverified(&json).map(|_| ()),
    ] {
        match result {
            Err(IoError::SemanticsMismatch {
                stored, expected, ..
            }) => {
                assert_eq!(stored, other());
                assert_eq!(expected, Provenance::sdf());
            }
            r => panic!("expected SemanticsMismatch, got {r:?}"),
        }
    }
}

#[test]
fn asdf_json_with_an_earlier_published_identifier_is_refused_and_judged_previous_when_listed() {
    let json = io::to_json_string_with_provenance(&tree(), &earlier()).unwrap();
    for result in [
        io::from_json_string(&json).map(|_| ()),
        io::from_json_string_unverified(&json).map(|_| ()),
    ] {
        match result {
            Err(io::IoError::SemanticsMismatch {
                stored, expected, ..
            }) => assert_eq!((stored, expected), (earlier(), Provenance::sdf())),
            other => panic!("the earlier identifier was not refused: {other:?}"),
        }
    }
    assert_eq!(
        alice_sdf::provenance::from_bytes(json.as_bytes())
            .unwrap()
            .judge(&Provenance::sdf(), &[earlier().semantics_id])
            .unwrap(),
        StoredSemantics::VerifiedPrevious(earlier())
    );
}

#[test]
fn asdf_json_with_a_malformed_identifier_is_an_error() {
    let mut value: serde_json::Value =
        serde_json::from_str(&io::to_json_string(&tree()).unwrap()).unwrap();
    value[provenance::KEY] = serde_json::Value::String("not hex".into());
    assert!(matches!(
        io::from_json_string(&value.to_string()),
        Err(IoError::InvalidFormat(_))
    ));
    value[provenance::KEY] = serde_json::json!(7);
    assert!(matches!(
        io::from_json_string_unverified(&value.to_string()),
        Err(IoError::InvalidFormat(_))
    ));
}

#[test]
fn a_model_answer_is_read_unpinned_and_another_stamp_is_refused() {
    let answer = r#"{"version": "0.1.0", "root": {"Sphere": {"radius": 1.0}}}"#;
    let (back, semantics) = validate_sdf_json(answer).unwrap().into_parts();
    assert_eq!(semantics, StoredSemantics::Unpinned);
    assert_eq!(back.node_count(), 1);
    let stamped = io::to_json_string(&tree()).unwrap();
    assert_eq!(
        validate_sdf_json(&stamped).unwrap().semantics(),
        StoredSemantics::Current
    );
    let foreign = io::to_json_string_with_provenance(&tree(), &other()).unwrap();
    let err = validate_sdf_json(&foreign).unwrap_err();
    assert!(err.contains("semantics identifier"), "{err}");
}

#[cfg(feature = "asp")]
mod packet {
    use super::*;
    use alice_sdf::asp_bridge::{
        create_sdf_i_packet, create_sdf_i_packet_with_provenance, decode_sdf_i_packet,
        decode_sdf_i_packet_unverified,
    };

    const V1_PACKET: &str = include_str!("fixtures/provenance/i_packet_v1.json");

    #[test]
    fn an_i_packet_carries_and_checks_the_identifier() {
        assert!(TEXT_WRITERS
            .iter()
            .any(|(w, _)| *w == "src/asp_bridge.rs::create_sdf_i_packet"));
        let t = tree();
        let packet = create_sdf_i_packet(&t, 1).unwrap();
        assert_eq!(
            decode_sdf_i_packet(&packet).unwrap().node_count(),
            t.node_count()
        );
        assert_eq!(
            decode_sdf_i_packet_unverified(&packet).unwrap().semantics(),
            StoredSemantics::Current
        );
        // an identifier an earlier release stamped, no longer reproduced: refused, naming both
        let previous = create_sdf_i_packet_with_provenance(&t, 1, &earlier()).unwrap();
        for err in [
            decode_sdf_i_packet(&previous).map(|_| ()).unwrap_err(),
            decode_sdf_i_packet_unverified(&previous)
                .map(|_| ())
                .unwrap_err(),
        ] {
            assert!(
                err.contains(&earlier().hex()) && err.contains(&Provenance::sdf().hex()),
                "{err}"
            );
        }
        let foreign = create_sdf_i_packet_with_provenance(&t, 1, &other()).unwrap();
        assert!(decode_sdf_i_packet(&foreign).is_err());
        assert!(decode_sdf_i_packet_unverified(&foreign).is_err());
    }

    #[test]
    fn an_i_packet_from_6_1_0_is_refused_by_default_and_read_unverified() {
        let packet: libasp::AspPacket = serde_json::from_str(V1_PACKET).unwrap();
        let err = decode_sdf_i_packet(&packet).unwrap_err();
        assert!(
            err.contains("Unpinned") || err.contains("identifier"),
            "{err}"
        );
        let (back, semantics) = decode_sdf_i_packet_unverified(&packet)
            .unwrap()
            .into_parts();
        assert_eq!(semantics, StoredSemantics::Unpinned);
        assert_eq!(back.node_count(), 4);
    }
}

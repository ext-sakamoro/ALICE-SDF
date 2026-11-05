//! A stored tree written under an identifier before the distance laws moved: it reads when it has none of
//! the moved nodes, it is refused with the nodes named when it has one, and a form without a tree stays
//! refused. Goes through the public load functions with files stamped with the real earlier identifiers.
//!
//! Author: Moroya Sakamoto

use alice_sdf::io::{self, IoError};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{Provenance, StoredSemantics};
use glam::Vec2;

fn earlier(hex: &str) -> Provenance {
    Provenance::new(core::array::from_fn(|i| {
        u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex digits")
    }))
}
const EARLIER: [&str; 2] = [
    "0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4",
    "ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938",
];

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("alice_sdf_moved_laws_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// The shapes stored body trees are made of: capsules, spheres and boxes under smooth operations, a
/// transform, a modifier
fn body_style_tree() -> SdfTree {
    let limb = SdfNode::capsule(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0), 0.1)
        .smooth_union(SdfNode::sphere(0.2).translate(0.0, 1.0, 0.0), 0.05);
    let torso = SdfNode::box3d(0.3, 0.5, 0.2).round(0.02).twist(0.1);
    SdfTree::new(limb.union(torso).subtract(SdfNode::sphere(0.05)))
}

fn moved_cases() -> Vec<(&'static str, SdfTree)> {
    let v2 = Vec2::new;
    vec![
        (
            "ParabolaSegment",
            SdfTree::new(SdfNode::parabola_segment(0.5, 0.4, 0.2)),
        ),
        (
            "ParabolaSegment",
            SdfTree::new(
                SdfNode::sphere(1.0).union(
                    SdfNode::parabola_segment(0.5, 0.4, 0.2)
                        .translate(1.0, 0.0, 0.0)
                        .twist(0.5),
                ),
            ),
        ),
        ("BlobbyCross", SdfTree::new(SdfNode::blobby_cross(0.6, 0.2))),
        ("Helix", SdfTree::new(SdfNode::helix(0.6, 0.1, 0.5, 0.8))),
        (
            "Bezier",
            SdfTree::new(SdfNode::bezier(Vec3::ZERO, Vec3::Y, Vec3::X, 0.1)),
        ),
        (
            "SweepBezier",
            SdfTree::new(SdfNode::sphere(0.6).sweep_bezier(
                v2(-1.0, 0.0),
                v2(0.0, 1.0),
                v2(1.0, 0.0),
            )),
        ),
    ]
}

#[test]
#[ignore = "takes effect when the identifier moves: both earlier identifiers are still decided before this rule"]
fn a_tree_without_a_moved_node_reads_under_each_earlier_identifier() {
    let tree = body_style_tree();
    let p = Vec3::new(0.1, 0.2, 0.3);
    let want = eval(&tree.root, p).to_bits();
    let mut n = 0;
    for hex in EARLIER {
        let id = earlier(hex);
        let json = temp(&format!("plain_{}.asdf.json", &hex[..8]));
        io::save_asdf_json_with_provenance(&tree, &json, &id).unwrap();
        let bin = temp(&format!("plain_{}.asdf", &hex[..8]));
        io::save_asdf_with_provenance(&tree, &bin, &id).unwrap();
        for read in [
            io::load_asdf_json(&json).unwrap(),
            io::load_asdf(&bin).unwrap(),
        ] {
            assert_eq!(eval(&read.root, p).to_bits(), want);
            n += 1;
        }
        let (_, status) = io::load_asdf_json_unverified(&json).unwrap().into_parts();
        assert_eq!(status, StoredSemantics::VerifiedPrevious(id));
    }
    assert_eq!(n, 4);
}

#[test]
#[ignore = "takes effect when the identifier moves: both earlier identifiers are still decided before this rule"]
fn a_tree_with_a_moved_node_is_refused_and_named() {
    let mut n = 0;
    for hex in EARLIER {
        let id = earlier(hex);
        for (i, (law, tree)) in moved_cases().into_iter().enumerate() {
            let json = temp(&format!("moved_{}_{i}.asdf.json", &hex[..8]));
            io::save_asdf_json_with_provenance(&tree, &json, &id).unwrap();
            let bin = temp(&format!("moved_{}_{i}.asdf", &hex[..8]));
            io::save_asdf_with_provenance(&tree, &bin, &id).unwrap();
            for got in [
                io::load_asdf_json(&json),
                io::load_asdf(&bin),
                io::load_asdf_json_unverified(&json).map(|c| c.into_parts().0),
            ] {
                assert!(
                    matches!(&got, Err(IoError::MovedLaw { stored, laws, .. }) if *stored == id && laws == &[law]),
                    "{law}: {:?}",
                    got.map(|_| "read")
                );
                n += 1;
            }
            // this build's own stamp on the same tree reads: the refusal is about the old identifier
            let now = temp(&format!("now_{i}.asdf.json"));
            io::save_asdf_json(&tree, &now).unwrap();
            assert!(
                io::load_asdf_json(&now).is_ok(),
                "{law} under this build's identifier"
            );
        }
    }
    assert_eq!(n, 36);
}

#[test]
fn an_older_identifier_is_refused_for_any_tree() {
    // 7.0.0's identifier, not in the table
    let id = earlier("151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69");
    assert!(!alice_sdf::SDF_SEMANTICS_MOVED_LAWS
        .iter()
        .any(|(e, _)| *e == id.semantics_id));
    let json = temp("older.asdf.json");
    io::save_asdf_json_with_provenance(&body_style_tree(), &json, &id).unwrap();
    assert!(matches!(
        io::load_asdf_json(&json),
        Err(IoError::SemanticsMismatch { .. })
    ));
}

#[test]
#[ignore = "takes effect when the identifier moves: both earlier identifiers are still decided before this rule"]
fn a_mesh_under_an_earlier_identifier_stays_refused() {
    let mesh = sdf_to_mesh(
        &SdfNode::sphere(1.0),
        Vec3::splat(-1.5),
        Vec3::splat(1.5),
        &MarchingCubesConfig {
            resolution: 4,
            ..Default::default()
        },
    );
    let mut n = 0;
    for hex in EARLIER {
        let id = earlier(hex);
        let ply = temp(&format!("{}.ply", &hex[..8]));
        io::export_ply_with_provenance(&mesh, &ply, &io::PlyConfig::default(), &id).unwrap();
        assert!(
            matches!(
                io::import_ply(&ply),
                Err(IoError::SemanticsMismatch { .. } | IoError::MeaningMismatch { .. })
            ),
            "PLY under {hex}"
        );
        let stl = temp(&format!("{}.stl", &hex[..8]));
        io::export_stl_with_provenance(&mesh, &stl, &id).unwrap();
        assert!(
            matches!(
                io::import_stl(&stl),
                Err(IoError::SemanticsMismatch { .. } | IoError::MeaningMismatch { .. })
            ),
            "STL under {hex}"
        );
        n += 2;
    }
    assert_eq!(n, 4);
}

/// Every node of the frozen list is a real tag: a misspelled entry would let its node through
#[test]
fn the_frozen_list_names_real_nodes() {
    let mut seen: Vec<&str> = moved_cases().iter().map(|(law, _)| *law).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen, alice_sdf::SDF_MOVED_LAWS_OVERSTATE_FIX);
    for (law, tree) in moved_cases() {
        assert_eq!(
            alice_sdf::moved_laws_in(&tree, alice_sdf::SDF_MOVED_LAWS_OVERSTATE_FIX),
            vec![law]
        );
    }
    assert!(
        alice_sdf::moved_laws_in(&body_style_tree(), alice_sdf::SDF_MOVED_LAWS_OVERSTATE_FIX)
            .is_empty()
    );
}

/// The LLM JSON path follows the same rule (it reads an identifier and accepts none)
#[test]
#[ignore = "takes effect when the identifier moves: both earlier identifiers are still decided before this rule"]
fn llm_json_follows_the_rule() {
    let id = earlier(EARLIER[0]);
    let plain = io::to_json_string_with_provenance(&body_style_tree(), &id).unwrap();
    assert!(alice_sdf::llm_schema::validate_sdf_json(&plain).is_ok());
    let moved = io::to_json_string_with_provenance(&moved_cases().remove(0).1, &id).unwrap();
    let e = alice_sdf::llm_schema::validate_sdf_json(&moved)
        .map(|_| ())
        .unwrap_err();
    assert!(e.contains("ParabolaSegment"), "{e}");
}

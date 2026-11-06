//! A stored tree written under an identifier before the distance laws moved: it reads when it has none of
//! the moved nodes, it is refused with the nodes named when it has one, and a form without a tree stays
//! refused. Goes through the public load functions with files stamped with the real earlier identifiers.
//!
//! Author: Moroya Sakamoto

mod common;

use alice_sdf::io::{self, IoError};
use alice_sdf::prelude::*;
use alice_sdf::provenance::{Provenance, StoredSemantics};
use glam::Vec2;

fn earlier(hex: &str) -> Provenance {
    Provenance::new(core::array::from_fn(|i| {
        u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex digits")
    }))
}
const EARLIER: [&str; 3] = [
    "150a80c505111a736fe705e0fa4e58aa521de2136cf0abc2762d84fbbb614f79",
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
    assert_eq!(n, 6);
}

#[test]
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
    assert_eq!(n, 54);
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
    assert_eq!(n, 6);
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

/// The corpus tree of a row, built by its constructor: its bounds are right, so only the identifier can
/// refuse it
fn corpus_tree(row: &str) -> SdfTree {
    let (_, node) = common::corpus::corpus()
        .into_iter()
        .find(|(n, _)| *n == row)
        .unwrap_or_else(|| panic!("no corpus row {row}"));
    SdfTree::new(node)
}

/// (identifier, the node tags and corpus rows that did not exist there). `150a80c5…` holds them all
const NOT_AT_ID: [(&str, &[(&str, &str)]); 2] = [
    (
        "0f6c58d0244a39c92712977c7cdbd2282e5c13d3cbd435f57189f79d2f6b69a4",
        &[
            ("Loft", "loft"),
            ("LoftCurves", "loft_curves"),
            ("SectionStack", "section_stack"),
            ("SmoothSectionStack", "smooth_section_stack"),
        ],
    ),
    (
        "ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938",
        &[
            ("GridWarp", "grid_warp"),
            ("Pose", "pose"),
            ("Loft", "loft"),
            ("LoftCurves", "loft_curves"),
            ("SectionStack", "section_stack"),
            ("SmoothSectionStack", "smooth_section_stack"),
        ],
    ),
];

/// A stamp says what computed the tree: a tree holding a node that did not exist under its identifier was
/// not computed by that build, so the identifier check refuses it and names the node. The same tree under
/// this build's identifier reads, so neither its data nor its bounds are the reason
#[test]
fn a_tree_with_a_node_newer_than_its_identifier_is_refused() {
    let mut n = 0;
    for (hex, kinds) in NOT_AT_ID {
        let id = earlier(hex);
        for (tag, row) in kinds {
            let tree = corpus_tree(row);
            let json = temp(&format!("new_{}_{row}.asdf.json", &hex[..8]));
            io::save_asdf_json_with_provenance(&tree, &json, &id).unwrap();
            let bin = temp(&format!("new_{}_{row}.asdf", &hex[..8]));
            io::save_asdf_with_provenance(&tree, &bin, &id).unwrap();
            for got in [
                io::load_asdf_json(&json),
                io::load_asdf(&bin),
                io::load_asdf_json_unverified(&json).map(|c| c.into_parts().0),
            ] {
                assert!(
                    matches!(&got, Err(IoError::NodeAfterIdentifier { stored, nodes, .. }) if *stored == id && nodes == &[*tag]),
                    "{tag} under {hex}: {:?}",
                    got.map(|_| "read")
                );
                n += 1;
            }
            let now = temp(&format!("new_now_{row}.asdf.json"));
            io::save_asdf_json(&tree, &now).unwrap();
            assert!(io::load_asdf_json(&now).is_ok(), "{tag} under this build");
        }
    }
    assert_eq!(n, 30);
}

/// A tree with a node newer than its identifier and a moved law: the newer node is what it is refused for
#[test]
fn a_newer_node_is_named_before_a_moved_law() {
    let id = earlier(EARLIER[2]);
    let tree = SdfTree::new(
        corpus_tree("loft_curves")
            .root
            .union(SdfNode::parabola_segment(0.5, 0.4, 0.2)),
    );
    let json = temp("both.asdf.json");
    io::save_asdf_json_with_provenance(&tree, &json, &id).unwrap();
    let got = io::load_asdf_json(&json);
    assert!(
        matches!(&got, Err(IoError::NodeAfterIdentifier { nodes, .. }) if nodes == &["LoftCurves"]),
        "{:?}",
        got.map(|_| "read")
    );
}

/// The node kinds of `SdfNode` when `0f6c58d0…` was the identifier
const KINDS_AT_0F6C58D0: &[&str] = &[
    "Animated",
    "Annular2D",
    "ArcShape",
    "Barrel",
    "Bend",
    "Bezier",
    "BlobbyCross",
    "Box3d",
    "BoxFrame",
    "CappedCone",
    "CappedTorus",
    "Capsule",
    "ChamferIntersection",
    "ChamferSubtraction",
    "ChamferUnion",
    "ChamferedCube",
    "Circle2D",
    "ColumnsIntersection",
    "ColumnsSubtraction",
    "ColumnsUnion",
    "Cone",
    "CrossShape",
    "CutHollowSphere",
    "CutSphere",
    "Cylinder",
    "DeathStar",
    "Diamond",
    "DiamondSurface",
    "Displacement",
    "Dodecahedron",
    "Egg",
    "Ellipsoid",
    "Elongate",
    "Engrave",
    "ExpSmoothIntersection",
    "ExpSmoothSubtraction",
    "ExpSmoothUnion",
    "Extrude",
    "FRD",
    "FischerKochS",
    "GridWarp",
    "Groove",
    "Gyroid",
    "Heart",
    "HeightmapDisplacement",
    "Helix",
    "HexPrism",
    "Horseshoe",
    "IFS",
    "IWP",
    "IcosahedralSymmetry",
    "Icosahedron",
    "InfiniteCone",
    "InfiniteCylinder",
    "Intersection",
    "LatticeDeform",
    "Lidinoid",
    "Link",
    "MetricBall",
    "MetricBlend",
    "Mirror",
    "Moon",
    "Morph",
    "Neovius",
    "Noise",
    "Octahedron",
    "OctantMirror",
    "Onion",
    "PMY",
    "ParabolaSegment",
    "Parallelogram",
    "Pie",
    "Pipe",
    "Plane",
    "PolarRepeat",
    "Polygon2D",
    "Pose",
    "ProjectiveTransform",
    "Pyramid",
    "Rect2D",
    "RegularPolygon",
    "RepeatFinite",
    "RepeatInfinite",
    "Revolution",
    "Rhombus",
    "Rotate",
    "Round",
    "RoundedBox",
    "RoundedCone",
    "RoundedCylinder",
    "RoundedRect2D",
    "RoundedX",
    "Scale",
    "ScaleNonUniform",
    "SchwarzP",
    "SdfSkinning",
    "Segment2D",
    "Shear",
    "SineDisplacement",
    "SmoothIntersection",
    "SmoothSubtraction",
    "SmoothUnion",
    "SolidAngle",
    "Sphere",
    "Stairs",
    "StairsIntersection",
    "StairsSubtraction",
    "StairsUnion",
    "StarPolygon",
    "Subtraction",
    "Superellipsoid",
    "SurfaceRoughness",
    "SweepBezier",
    "Taper",
    "Terrain",
    "Tetrahedron",
    "Tongue",
    "Torus",
    "Translate",
    "Trapezoid",
    "Triangle",
    "TriangularPrism",
    "TruncatedIcosahedron",
    "TruncatedOctahedron",
    "Tube",
    "Tunnel",
    "Twist",
    "UnevenCapsule",
    "Union",
    "Vesica",
    "WithMaterial",
    "XOR",
];

fn enum_kinds() -> Vec<String> {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/types/mod.rs"))
        .expect("the node enum's source");
    let body = &src[src.find("pub enum SdfNode").expect("the enum")..];
    let body = &body[..body.find("\n}\n").expect("its end")];
    body.lines()
        .filter_map(|l| {
            let name = l.strip_prefix("    ")?;
            if !name.starts_with(|c: char| c.is_ascii_uppercase()) {
                return None;
            }
            let end = name.find(|c: char| !c.is_ascii_alphanumeric())?;
            matches!(
                name[end..].trim_start().chars().next(),
                Some('{' | '(' | ',')
            )
            .then(|| name[..end].to_owned())
        })
        .collect()
}

/// Every node kind since `0f6c58d0…` is listed as absent there: a new kind fails here until it is in each
/// row of `SDF_SEMANTICS_ABSENT_NODES`, so an old stamp cannot pass for a node that came later
#[test]
fn the_absent_lists_cover_every_kind_added_since() {
    let kinds = enum_kinds();
    assert!(
        kinds.len() >= KINDS_AT_0F6C58D0.len(),
        "read {} kinds",
        kinds.len()
    );
    let mut added: Vec<&str> = kinds
        .iter()
        .map(String::as_str)
        .filter(|k| !KINDS_AT_0F6C58D0.contains(k))
        .collect();
    added.sort_unstable();
    assert_eq!(added, alice_sdf::SDF_NODES_ADDED_WITH_LOFTS);
    let mut before: Vec<&str> = added.clone();
    before.extend(["GridWarp", "Pose"]);
    before.sort_unstable();
    assert_eq!(before, alice_sdf::SDF_NODES_ADDED_WITH_WARP_AND_LOFTS);
    for k in KINDS_AT_0F6C58D0 {
        assert!(kinds.iter().any(|x| x == k), "{k} is no longer a kind");
    }
    assert_eq!(alice_sdf::SDF_SEMANTICS_ABSENT_NODES.len(), 2);
}

/// The lofts and section stacks read under the identifier that added them
#[test]
fn a_tree_of_lofts_reads_under_the_identifier_that_had_them() {
    let id = earlier(EARLIER[0]);
    for row in [
        "loft",
        "loft_curves",
        "section_stack",
        "smooth_section_stack",
    ] {
        let json = temp(&format!("had_{row}.asdf.json"));
        io::save_asdf_json_with_provenance(&corpus_tree(row), &json, &id).unwrap();
        let (_, status) = io::load_asdf_json_unverified(&json).unwrap().into_parts();
        assert_eq!(status, StoredSemantics::VerifiedPrevious(id), "{row}");
    }
}

/// The LLM JSON path follows the same rule (it reads an identifier and accepts none)
#[test]
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

//! LLM schema helpers oracle.
//!
//! - `validate_sdf_json` round trip: a tree serialised with serde comes back
//!   structurally identical (compared through `Debug`) and evaluates to the
//!   scene's closed-form distance
//! - invalid dimensions and malformed JSON are rejected with a message that
//!   names the problem (the message is fed back to the generator)
//! - `schema_summary` reports the variant counts `SdfCategory` holds (those
//!   are checked against one node per variant in
//!   `test_node_backend_matrix.rs`)
//!
//! Author: Moroya Sakamoto

use alice_sdf::llm_schema::{schema_summary, validate_sdf_json};
use alice_sdf::prelude::*;

#[test]
fn serialised_trees_round_trip_through_validation() {
    let scenes = [
        SdfNode::sphere(1.0),
        SdfNode::sphere(1.0).translate(2.0, 0.0, 0.0),
        SdfNode::sphere(1.0).subtract(SdfNode::box3d(1.0, 1.0, 1.0)),
        SdfNode::cylinder(0.5, 1.0).union(SdfNode::sphere(0.25).translate(0.0, 2.0, 0.0)),
    ];
    let mut compared = 0;
    for node in &scenes {
        let json = serde_json::to_string(&SdfTree::new(node.clone())).unwrap();
        let tree = validate_sdf_json(&json).expect("valid tree");
        assert_eq!(format!("{:?}", tree.root), format!("{node:?}"));
        compared += 1;
    }
    // Evaluate the translated sphere: |p − (2,0,0)| − 1
    let json = serde_json::to_string(&SdfTree::new(scenes[1].clone())).unwrap();
    let tree = validate_sdf_json(&json).unwrap();
    for &p in &[
        Vec3::ZERO,
        Vec3::new(2.0, 3.0, 0.0),
        Vec3::new(4.0, 0.0, 0.0),
    ] {
        let expected = (p - Vec3::new(2.0, 0.0, 0.0)).length() - 1.0;
        assert!((eval(&tree.root, p) - expected).abs() < 1e-6);
        compared += 1;
    }
    assert!(compared > 0);
}

#[test]
fn invalid_input_is_rejected_with_a_reason() {
    let bad_radius = serde_json::to_string(&SdfTree::new(
        SdfNode::sphere(1.0).union(SdfNode::sphere(-2.0)),
    ))
    .unwrap();
    let err = validate_sdf_json(&bad_radius).unwrap_err();
    assert!(err.contains("Sphere radius must be positive"), "{err}");
    assert!(err.contains("-2"), "{err}");

    let bad_box = serde_json::to_string(&SdfTree::new(
        SdfNode::box3d(1.0, 0.0, 1.0).translate(1.0, 0.0, 0.0),
    ))
    .unwrap();
    assert!(validate_sdf_json(&bad_box).unwrap_err().contains("Box3d"));

    let err = validate_sdf_json("{\n  \"version\": \"0.1.0\",\n  \"root\": [oops]\n}").unwrap_err();
    assert!(err.starts_with("JSON parse error at line 3"), "{err}");
}

fn number_after(s: &str, label: &str) -> u32 {
    let at = s
        .find(label)
        .unwrap_or_else(|| panic!("{label} missing in {s}"))
        + label.len();
    let digits: String = s[at..]
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().unwrap()
}

#[test]
fn schema_summary_counts_match_the_enum() {
    let s = schema_summary();
    let pairs = [
        ("Primitives", SdfCategory::Primitive.count()),
        ("Operations", SdfCategory::Operation.count()),
        ("Transforms", SdfCategory::Transform.count()),
        ("Modifiers", SdfCategory::Modifier.count()),
        ("Node Types", SdfCategory::total()),
    ];
    let mut compared = 0;
    for (label, expected) in pairs {
        assert_eq!(number_after(&s, label), expected, "{label} in:\n{s}");
        compared += 1;
    }
    assert_eq!(compared, 5);
}

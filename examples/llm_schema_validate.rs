//! Validating generated SDF JSON before using it.
//!
//! A text generator is given `schema_summary()` as the list of node types and
//! answers with an SDF tree as JSON. `validate_sdf_json` parses it and checks
//! dimensions; on failure it returns a message meant to be sent back to the
//! generator for a corrected answer. Here the "answers" are fixed strings:
//! one broken, one with a negative radius, one valid.
//!
//! # Running
//! ```bash
//! cargo run --example llm_schema_validate
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::llm_schema::{schema_summary, validate_sdf_json};
use alice_sdf::prelude::*;

fn main() {
    println!("ALICE-SDF — validating generated SDF JSON");
    println!("=========================================");

    let summary = schema_summary();
    println!("{summary}");
    assert!(summary.contains(&format!("({} total)", SdfCategory::total())));

    let answers = [
        r#"{"version": "0.1.0", "root": {"Sphere": {"radius": 1.0}"#,
        r#"{"version": "0.1.0", "root": {"Sphere": {"radius": -1.0}}}"#,
        r#"{"version": "0.1.0", "root": {"Translate": {"child": {"Sphere": {"radius": 1.0}}, "offset": [0.0, 2.0, 0.0]}}}"#,
    ];

    let mut accepted = None;
    for (i, answer) in answers.iter().enumerate() {
        match validate_sdf_json(answer) {
            Ok(tree) => {
                println!("answer {i}: accepted ({} nodes)", tree.node_count());
                accepted = Some(tree);
            }
            Err(feedback) => println!("answer {i}: rejected, feedback = {feedback}"),
        }
    }
    assert!(validate_sdf_json(answers[0])
        .unwrap_err()
        .starts_with("JSON parse error"));
    assert!(validate_sdf_json(answers[1])
        .unwrap_err()
        .contains("radius must be positive"));

    // The accepted tree is a unit sphere at (0, 2, 0): d(p) = |p − (0,2,0)| − 1.
    let tree = accepted.expect("the last answer is valid");
    let p = Vec3::new(0.0, 5.0, 0.0);
    let d = eval(&tree.root, p);
    println!("d({p}) = {d}");
    assert!((d - 2.0).abs() < 1e-6);
    println!("all checks passed");
}

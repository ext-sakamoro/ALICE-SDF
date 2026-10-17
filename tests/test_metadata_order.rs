//! The custom key-value pairs of `SdfMetadata` are written in key order (ALICE-Law
//! G13): the same pairs, inserted in different orders, give the same JSON bytes, and
//! the keys appear sorted
use alice_sdf::io::to_json_string;
use alice_sdf::types::{SdfMetadata, SdfNode, SdfTree};

fn tree_with(pairs: &[(String, String)]) -> SdfTree {
    let metadata = SdfMetadata {
        name: Some("order".to_string()),
        custom: Some(pairs.iter().cloned().collect()),
        ..SdfMetadata::default()
    };
    SdfTree::with_metadata(SdfNode::sphere(1.0), metadata)
}

#[test]
fn custom_pairs_are_written_in_key_order() {
    let pairs: Vec<(String, String)> = (0..32)
        .map(|i| (format!("key{:02}", (i * 13) % 32), format!("v{i}")))
        .collect();
    let mut reversed = pairs.clone();
    reversed.reverse();
    let a = to_json_string(&tree_with(&pairs)).unwrap();
    let b = to_json_string(&tree_with(&reversed)).unwrap();
    assert_eq!(
        a, b,
        "the same pairs inserted in another order wrote other bytes"
    );
    let at: Vec<usize> = (0..32)
        .map(|k| a.find(&format!("\"key{k:02}\"")).expect("key written"))
        .collect();
    assert!(
        at.windows(2).all(|w| w[0] < w[1]),
        "keys are not in sorted order: {at:?}"
    );
}

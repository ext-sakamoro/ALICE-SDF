//! split <tree.asdf.json> <out prefix>: save each Union leaf of the root as its own pinned tree (save_asdf_json)
use alice_sdf::prelude::*;
fn leaves(n: &SdfNode, out: &mut Vec<SdfNode>) {
    match n {
        SdfNode::Union { a, b } => { leaves(a, out); leaves(b, out); }
        other => out.push(other.clone()),
    }
}
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t = alice_sdf::io::load_asdf_json(&a[1]).expect("load");
    let mut v = Vec::new(); leaves(&t.root, &mut v);
    for (i, n) in v.into_iter().enumerate() {
        let p = format!("{}{:02}.asdf.json", a[2], i);
        alice_sdf::io::save_asdf_json(&SdfTree::new(n), &p).expect("save");
        println!("{p}");
    }
}

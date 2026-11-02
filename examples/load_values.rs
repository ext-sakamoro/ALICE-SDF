//! Local only: load a saved tree through the json loader and write its values at given points.
//! usage: load_values <tree.json> <points.json [[x, y, z], ...]> <values.json>
use alice_sdf::io::load_asdf_json;
use glam::Vec3;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = load_asdf_json(&a[0]).expect("the tree loads");
    let pts: Vec<[f32; 3]> = serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("points")).expect("points json");
    let vals: Vec<f32> = pts.iter().map(|p| alice_sdf::eval(&t.root, Vec3::from_array(*p))).collect();
    std::fs::write(&a[2], serde_json::to_string(&vals).expect("values")).expect("write");
    println!("loaded {}; values at {} points -> {}", a[0], vals.len(), a[2]);
}

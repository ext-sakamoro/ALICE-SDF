//! Prints the field of a saved tree at given points: `probe_tree <tree.asdf.json> x,y,z [x,y,z ...]` (m)
use alice_sdf::io::load_asdf_json;
use glam::Vec3;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let t = load_asdf_json(&a[0]).expect("tree");
    for s in &a[1..] {
        let v: Vec<f32> = s.split(',').map(|x| x.parse().unwrap()).collect();
        let p = Vec3::new(v[0], v[1], v[2]);
        println!("{s}: {:e} m", alice_sdf::eval(&t.root, p));
    }
}

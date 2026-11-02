//! Local only (not committed): the foot as parts, each its own hull ∩ its own SectionStack.
//! body = hull.body ∩ (the foot-wide stack along the long axis); toe k = hull.toe_k ∩ (toe k's stack along its own
//! axis); toes joined by hard union (sub-mm gaps between pressed toes survive), the body smooth-unioned with them (k).
//! Stacks are built through the constructor (it computes the Lipschitz bound) and placed by a rotation whose columns
//! are the local axes in world coordinates, then a translation; the placement is checked against the local stack.
//! usage: foot_stack2 <foot_sections.json> <toe_sections.json> <hull stem (…/tree_hull)> <k mm> <out.json>
use alice_sdf::io::{from_json_string_unverified, to_json_string};
use alice_sdf::types::{SdfNode, SdfTree};
use glam::{Mat3, Quat, Vec2, Vec3};
use serde_json::Value;

fn f(v: &Value) -> f32 {
    v.as_f64().expect("number") as f32
}
fn v3(v: &Value) -> Vec3 {
    Vec3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}
fn stack_of(levels: &Value, loops: &Value) -> SdfNode {
    let lv: Vec<f32> = levels.as_array().expect("levels").iter().map(f).collect();
    let lp: Vec<Vec<Vec<Vec2>>> = loops
        .as_array()
        .expect("loops")
        .iter()
        .map(|l| l.as_array().expect("level").iter().map(|r| r.as_array().expect("loop").iter().map(|p| Vec2::new(f(&p[0]), f(&p[1]))).collect()).collect())
        .collect();
    SdfNode::section_stack(lv, lp).expect("section stack")
}
fn lip(n: &SdfNode) -> f32 {
    if let SdfNode::SectionStack(s) = n { s.lipschitz() } else { f32::NAN }
}
fn hull(stem: &str, name: &str) -> SdfNode {
    from_json_string_unverified(&std::fs::read_to_string(format!("{stem}.{name}.json")).expect("hull part")).expect("hull json").into_parts().0.root
}
/// local stack placed so local X, Y, Z lie along bx, d, bz and the local origin at o; checked on a few points
fn place(local: SdfNode, bx: Vec3, d: Vec3, bz: Vec3, o: Vec3, probe: &[Vec3]) -> SdfNode {
    let q = Quat::from_mat3(&Mat3::from_cols(bx, d, bz));
    let world = local.clone().rotate(q).translate(o.x, o.y, o.z);
    for &l in probe {
        let p = o + bx * l.x + d * l.y + bz * l.z;
        let (a, b) = (alice_sdf::eval::eval(&local, l), alice_sdf::eval::eval(&world, p));
        assert!((a - b).abs() < 2e-6, "placement: local {a} vs world {b} at {l:?}");
    }
    world
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let fs: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("foot sections")).expect("json");
    let ts: Value = serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("toe sections")).expect("json");
    let (stem, k) = (a[3].as_str(), a[4].parse::<f32>().expect("k") * 1e-3);
    let fr = &fs["frame"];
    let fstack = stack_of(&fs["levels"], &fs["loops"]);
    println!("foot stack: L {:.3}", lip(&fstack));
    let probe = [Vec3::new(0.01, 0.02, -0.03), Vec3::new(-0.02, 0.06, -0.05), Vec3::new(0.0, 0.11, -0.055)];
    let body = hull(stem, "body").intersection(place(fstack, v3(&fr["x"]), v3(&fr["axis"]), v3(&fr["z"]), v3(&fr["origin"]), &probe));
    let mut toes: Option<SdfNode> = None;
    for (i, t) in ts["toes"].as_array().expect("toes").iter().enumerate() {
        let st = stack_of(&t["levels"], &t["loops"]);
        println!("toe {i} stack: L {:.3}", lip(&st));
        let tp = [Vec3::new(0.002, 0.003, 0.001), Vec3::new(-0.004, 0.01, 0.003)];
        let node = hull(stem, &format!("toe{i}")).intersection(place(st, v3(&t["bx"]), v3(&t["d"]), v3(&t["bz"]), v3(&t["o"]), &tp));
        toes = Some(match toes { None => node, Some(u) => u.union(node) });
    }
    let tree = body.smooth_union(toes.expect("toes"), k);
    std::fs::write(&a[5], to_json_string(&SdfTree::new(tree)).expect("tree json")).expect("write");
    println!("wrote {} (toes hard union, body smooth union k {:.1} mm)", a[5], k * 1e3);
}

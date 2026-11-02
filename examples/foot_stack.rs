//! Local only (not committed): the foot as the outline hull intersected with a stack of R0's cross sections along
//! the foot's long axis. Reads the sections (levels and loops in the stack's local frame) and the hull tree, builds the
//! stack through its constructor (which computes the Lipschitz bound), turns local Y onto the long axis with the same
//! rotate chain the extraction checked, and writes the tree (hull ∩ stack) and the stack alone.
//! usage: foot_stack <sections.json> <hull_tree.json> <out_tree.json> <out_stack.json>
use alice_sdf::io::{from_json_string_unverified, to_json_string};
use alice_sdf::types::{SdfNode, SdfTree};
use glam::Vec2;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let s: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("sections")).expect("sections json");
    let levels: Vec<f32> = s["levels"].as_array().expect("levels").iter().map(|v| v.as_f64().expect("level") as f32).collect();
    let loops: Vec<Vec<Vec<Vec2>>> = s["loops"]
        .as_array()
        .expect("loops")
        .iter()
        .map(|lv| {
            lv.as_array()
                .expect("level loops")
                .iter()
                .map(|lp| lp.as_array().expect("loop").iter().map(|p| Vec2::new(p[0].as_f64().expect("x") as f32, p[1].as_f64().expect("z") as f32)).collect())
                .collect()
        })
        .collect();
    let phi = s["frame"]["phi"].as_f64().expect("phi") as f32;
    let stack = SdfNode::section_stack(levels.clone(), loops).expect("section stack");
    if let SdfNode::SectionStack(st) = &stack {
        println!("stack: {} levels, Lipschitz bound {:.4}", levels.len(), st.lipschitz());
    }
    let stack = stack.rotate_euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0).rotate_euler(0.0, phi, 0.0);
    let hull = from_json_string_unverified(&std::fs::read_to_string(&a[2]).expect("hull")).expect("hull json").into_parts().0.root;
    let tree = hull.intersection(stack.clone());
    std::fs::write(&a[3], to_json_string(&SdfTree::new(tree)).expect("tree json")).expect("write tree");
    std::fs::write(&a[4], to_json_string(&SdfTree::new(stack)).expect("stack json")).expect("write stack");
    println!("wrote {} and {}", a[3], a[4]);
}

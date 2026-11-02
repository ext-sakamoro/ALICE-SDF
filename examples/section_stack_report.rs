//! Builds a section stack from measured horizontal sections and reports its Lipschitz bound: the
//! intervals with the largest boundary change per height (G = H / Δy), the bound L = sqrt(1 + G²),
//! and the sphere-tracing step counts on given rays for a tree alone, the stack alone and their
//! intersection.
//!
//! `cargo run --release --example section_stack_report -- <sections.json> <tree.asdf.json> <rays.json> [y_lo y_hi ...]`
//!
//! sections.json: {"levels": [y…], "loops": [[[[x, z]…]…]…]} (m); rays.json: {"rays": [{"o", "d"}…]}; each
//! y_lo y_hi pair selects the rays whose origin height lies in it.
use alice_sdf::interval::eval_lipschitz;
use alice_sdf::io::load_asdf_json;
use alice_sdf::primitives::{boundary_hausdorff_bound, SectionStack};
use alice_sdf::raycast::{raymarch_with_config, RaymarchConfig};
use alice_sdf::SdfNode;
use glam::{Vec2, Vec3};
use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 3,
        "section_stack_report <sections> <tree> <rays> [y_lo y_hi ...]"
    );
    let s: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[0]).expect("sections")).expect("JSON");
    let levels: Vec<f32> = s["levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let loops: Vec<Vec<Vec<Vec2>>> = s["loops"]
        .as_array()
        .unwrap()
        .iter()
        .map(|lv| {
            lv.as_array()
                .unwrap()
                .iter()
                .map(|l| {
                    l.as_array()
                        .unwrap()
                        .iter()
                        .map(|q| {
                            Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    let mut g: Vec<(f32, f32, f32)> = (0..levels.len() - 1)
        .map(|i| {
            let h = boundary_hausdorff_bound(&loops[i], &loops[i + 1]);
            (h / (levels[i + 1] - levels[i]), levels[i], h)
        })
        .collect();
    g.sort_by(|x, y| y.0.total_cmp(&x.0));
    println!("[stack] {} levels; largest G = H/Δy:", levels.len());
    for (gi, y, h) in g.iter().take(12) {
        println!("[stack]   y {:.4} m: H {:.1} mm, G {:.1}", y, h * 1e3, gi);
    }
    let below: usize = g.iter().filter(|x| x.0 <= 2.0).count();
    println!("[stack] intervals with G ≤ 2: {below} of {}", g.len());
    let stack = SdfNode::SectionStack(SectionStack::new(levels, loops).expect("a valid stack"));
    if let SdfNode::SectionStack(st) = &stack {
        println!("[stack] L = {:.2}", st.lipschitz());
    }
    let tree = load_asdf_json(&a[1]).expect("tree").root;
    let both = tree.clone().intersection(stack.clone());
    let r: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[2]).expect("rays")).expect("JSON");
    let rays: Vec<(Vec3, Vec3)> = r["rays"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| {
            let v = |k: &str| {
                let q = x[k].as_array().unwrap();
                Vec3::new(
                    q[0].as_f64().unwrap() as f32,
                    q[1].as_f64().unwrap() as f32,
                    q[2].as_f64().unwrap() as f32,
                )
            };
            (v("o"), v("d"))
        })
        .collect();
    let bands: Vec<(f32, f32)> = a[3..]
        .chunks(2)
        .map(|c| (c[0].parse().unwrap(), c[1].parse().unwrap()))
        .collect();
    for (lo, hi) in bands {
        let sel: Vec<&(Vec3, Vec3)> = rays
            .iter()
            .filter(|(o, _)| o.y >= lo && o.y <= hi)
            .collect();
        assert!(!sel.is_empty(), "no rays in {lo}..{hi}");
        for (name, node) in [("tree", &tree), ("stack", &stack), ("tree ∩ stack", &both)] {
            let mut cfg = RaymarchConfig::default().with_bound(eval_lipschitz(node));
            cfg.max_steps = 4096;
            let mut steps: Vec<u32> = Vec::new();
            let mut miss = 0;
            for (o, d) in &sel {
                match raymarch_with_config(node, *o, *d, 2.0, &cfg) {
                    Some(h) => steps.push(h.steps),
                    None => miss += 1,
                }
            }
            steps.sort_unstable();
            let p = |q: f32| {
                steps
                    .get(((steps.len().max(1) - 1) as f32 * q) as usize)
                    .copied()
                    .unwrap_or(0)
            };
            println!("[steps] y {lo}..{hi}, {} rays, {name}: hits {} misses {miss}, steps p50 {} p90 {} max {}", sel.len(), steps.len(), p(0.5), p(0.9), p(1.0));
        }
    }
}

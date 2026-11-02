//! Reports the Lipschitz bound of section stacks built from per-part sections: for each part the
//! largest boundary change per spacing (G = H / Δ) with where it is along the axis, and L = sqrt(1 + G²).
//!
//! `cargo run --release --example part_stack_g -- <part_sections.json>`
//!
//! part_sections.json: {"parts": {name: {"levels": [s…], "loops": [[[[x, z]…]…]…]}}} (m, local axis frame)
use alice_sdf::primitives::{boundary_hausdorff_bound, SectionStack};
use glam::Vec2;
use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(a.len(), 1, "part_stack_g <part_sections.json>");
    let d: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[0]).expect("sections")).expect("JSON");
    let mut n = 0;
    for (name, p) in d["parts"].as_object().expect("parts") {
        let levels: Vec<f32> = p["levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        let loops: Vec<Vec<Vec<Vec2>>> = p["loops"]
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
                                Vec2::new(
                                    q[0].as_f64().unwrap() as f32,
                                    q[1].as_f64().unwrap() as f32,
                                )
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        if levels.len() < 2 {
            println!("[g] {name}: {} levels, no stack", levels.len());
            continue;
        }
        let mut g: Vec<(f32, f32, f32, f32)> = (0..levels.len() - 1)
            .map(|i| {
                let h = boundary_hausdorff_bound(&loops[i], &loops[i + 1]);
                let dy = levels[i + 1] - levels[i];
                (h / dy, levels[i], h, dy)
            })
            .collect();
        g.sort_by(|x, y| y.0.total_cmp(&x.0));
        let st = SectionStack::new(levels.clone(), loops).expect("a valid stack");
        let top: Vec<String> = g
            .iter()
            .take(4)
            .map(|(gi, s, h, dy)| {
                format!(
                    "G {gi:.1} at {:.0} mm (H {:.1} / Δ {:.1} mm)",
                    s * 1e3,
                    h * 1e3,
                    dy * 1e3
                )
            })
            .collect();
        let le2 = g.iter().filter(|x| x.0 <= 2.0).count();
        println!(
            "[g] {name}: {} levels, L {:.2}; G ≤ 2 in {le2} of {}; largest: {}",
            levels.len(),
            st.lipschitz(),
            g.len(),
            top.join("; ")
        );
        n += 1;
    }
    assert!(n > 0, "no part had a stack");
}

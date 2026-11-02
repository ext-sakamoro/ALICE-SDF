//! Local only: the figure's top-level parts, each through WGSL on the GPU against the CPU tree
#![allow(clippy::disallowed_methods)]
use alice_sdf::compiled::GpuEvaluator;
use alice_sdf::prelude::*;

fn leaves(n: &SdfNode, out: &mut Vec<SdfNode>) {
    match n {
        SdfNode::Union { a, b } => {
            leaves(a, out);
            leaves(b, out);
        }
        other => out.push(other.clone()),
    }
}

/// one result line, appended to PARTS_OUT and flushed at once, so a cut-off run still shows the
/// parts it finished
fn line(text: &str) {
    use std::io::Write;
    println!("{text}");
    std::io::stdout().flush().ok();
    if let Ok(path) = std::env::var("PARTS_OUT") {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path).expect("results file");
        writeln!(f, "{text}").expect("write");
        f.flush().ok();
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("tree");
    let tree = alice_sdf::io::load_asdf_json(&path).expect("load");
    let mut parts = Vec::new();
    leaves(&tree.root, &mut parts);
    // PARTS_SPLIT=i: measure the children of top-level part i instead (Union leaves under its
    // Subtraction's minuend, the subtrahend, and the minuend as a whole)
    if let Some(k) = std::env::var("PARTS_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()) {
        let part = parts[k].clone();
        let mut kids = Vec::new();
        if let SdfNode::Subtraction { a, b } = &part {
            leaves(a, &mut kids);
            kids.push((**b).clone());
            kids.push((**a).clone());
        } else {
            leaves(&part, &mut kids);
        }
        parts = kids;
    }
    let mut s = 99u64;
    let mut r = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    for (i, part) in parts.iter().enumerate() {
        let mut pts = Vec::new();
        let mut tries = 0;
        let want: usize = std::env::var("PARTS_N").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
        while pts.len() < want && tries < 4_000_000 {
            tries += 1;
            let p = Vec3::new(r() * 1.8 - 0.9, r() * 1.9 - 0.1, r() * 1.0 - 0.5);
            if pts.len() % 2 == 0 || eval(part, p).abs() < 0.02 {
                pts.push(p);
            }
        }
        let kind = format!("{:?}", part).chars().take(40).collect::<String>();
        match GpuEvaluator::new(part) {
            Ok(g) => {
                let got = g.eval_batch(&pts).expect("eval");
                let (mut w, mut at) = (0.0f32, Vec3::ZERO);
                for (p, gv) in pts.iter().zip(&got) {
                    let d = (eval(part, *p) - gv).abs();
                    if d > w { w = d; at = *p; }
                }
                line(&format!("part {i} [{kind}]: {} points, max |gpu - cpu| {w:.3e} at {at:?}", pts.len()));
            }
            Err(e) => line(&format!("part {i} [{kind}]: GPU evaluator failed: {e}")),
        }
    }
}

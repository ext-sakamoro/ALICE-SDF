//! Head loft tool: build an ALICE-SDF tree from a part spec, save it, evaluate points, sphere trace views
//!
//! build <spec.json> <out.asdf.json>
//! eval <tree.asdf.json> <points.f32> <out.f32>            (points: xyz f32 little endian)
//! trace <tree.asdf.json> <rays.f32> <out.f32>             (rays: ox oy oz dx dy dz tmax; out: t (-1 miss) nx ny nz)
use alice_sdf::prelude::*;
use rayon::prelude::*;
use std::fs;

fn f32s(path: &str) -> Vec<f32> {
    let b = fs::read(path).expect("read");
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn write_f32s(path: &str, v: &[f32]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    fs::write(path, b).expect("write");
}
fn v3(a: &serde_json::Value) -> Vec3 {
    Vec3::new(a[0].as_f64().unwrap() as f32, a[1].as_f64().unwrap() as f32, a[2].as_f64().unwrap() as f32)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a[1].as_str() {
        "build" => {
            let spec: serde_json::Value = serde_json::from_str(&fs::read_to_string(&a[2]).unwrap()).unwrap();
            let mut node: Option<SdfNode> = None;
            for p in spec["parts"].as_array().unwrap() {
                let axis: Vec<Vec3> = p["axis"].as_array().unwrap().iter().map(v3).collect();
                let up = v3(&p["up"]);
                let part = match p["kind"].as_str().unwrap() {
                    "curves" => {
                        let rows: Vec<Vec<glam::Vec2>> = p["rows"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|r| r.as_array().unwrap().iter().map(|q| glam::Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32)).collect())
                            .collect();
                        SdfNode::loft_curves(axis, up, rows)
                    }
                    "capped" => {
                        let rows: Vec<Vec<f32>> = p["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
                        let caps = [p["caps"][0].as_f64().unwrap() as f32, p["caps"][1].as_f64().unwrap() as f32];
                        SdfNode::loft_capped(axis, up, p["harmonics"].as_u64().unwrap() as u32, rows, caps)
                    }
                    k => panic!("kind {k}"),
                };
                let part = match part {
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("REFUSED part {}: {}", p["name"], e.0);
                        std::process::exit(2);
                    }
                };
                if let SdfNode::LoftCurves { bounds, .. } | SdfNode::Loft { bounds, .. } = &part {
                    println!("part {} bounds {:?}", p["name"], bounds);
                }
                let k = p["k"].as_f64().unwrap_or(0.0) as f32;
                node = Some(match node {
                    None => part,
                    Some(n) if k > 0.0 => n.smooth_union(part, k),
                    Some(n) => n.union(part),
                });
            }
            let tree = SdfTree::new(node.unwrap());
            alice_sdf::io::save(&tree, &a[3]).expect("save");
            println!("saved {}", a[3]);
        }
        "eval" => {
            let tree = alice_sdf::io::load(&a[2]).expect("load");
            let p = f32s(&a[3]);
            let comp = alice_sdf::compiled::CompiledSdf::compile(&tree.root);
            let compiled = std::env::var("EVAL_COMPILED").is_ok();
            let d: Vec<f32> = p.par_chunks_exact(3).map(|c| { let q = Vec3::new(c[0], c[1], c[2]); if compiled { alice_sdf::compiled::eval_compiled(&comp, q) } else { eval(&tree.root, q) } }).collect();
            write_f32s(&a[4], &d);
        }
        "trace" => {
            let tree = alice_sdf::io::load(&a[2]).expect("load");
            let r = f32s(&a[3]);
            let comp = alice_sdf::compiled::CompiledSdf::compile(&tree.root);
            let ev = |q: Vec3| if std::env::var("TRACE_TREE").is_ok() { eval(&tree.root, q) } else { alice_sdf::compiled::eval_compiled(&comp, q) };
            let out: Vec<f32> = r
                .par_chunks_exact(7)
                .flat_map_iter(|c| {
                    let o = Vec3::new(c[0], c[1], c[2]);
                    let dir = Vec3::new(c[3], c[4], c[5]);
                    let tmax = c[6];
                    let mut t = 0.0f32;
                    let mut hit = -1.0f32;
                    for _ in 0..4000 {
                        let d = ev(o + dir * t);
                        if d < 0.002 {
                            hit = t;
                            break;
                        }
                        t += d;
                        if t > tmax {
                            break;
                        }
                    }
                    let mut nrm = Vec3::ZERO;
                    if hit >= 0.0 {
                        let p = o + dir * hit;
                        let h = 0.01;
                        nrm = Vec3::new(
                            ev(p + Vec3::X * h) - ev(p - Vec3::X * h),
                            ev(p + Vec3::Y * h) - ev(p - Vec3::Y * h),
                            ev(p + Vec3::Z * h) - ev(p - Vec3::Z * h),
                        )
                        .normalize_or_zero();
                    }
                    [hit, nrm.x, nrm.y, nrm.z]
                })
                .collect();
            write_f32s(&a[4], &out);
        }
        m => panic!("mode {m}"),
    }
}

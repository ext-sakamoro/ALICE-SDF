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
            // every length in the spec is multiplied by unit_scale (0.001: spec in mm, tree in metres)
            let us = spec["unit_scale"].as_f64().unwrap_or(1.0) as f32;
            for p in spec["parts"].as_array().unwrap() {
                let axis: Vec<Vec3> = p["axis"].as_array().unwrap().iter().map(|q| v3(q) * us).collect();
                let up = v3(&p["up"]);
                let part = match p["kind"].as_str().unwrap() {
                    "curves" => {
                        let rows: Vec<Vec<glam::Vec2>> = p["rows"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|r| r.as_array().unwrap().iter().map(|q| glam::Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32) * us).collect())
                            .collect();
                        SdfNode::loft_curves(axis, up, rows)
                    }
                    "capped" => {
                        let rows: Vec<Vec<f32>> = p["rows"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32 * us).collect()).collect();
                        let caps = [p["caps"][0].as_f64().unwrap() as f32 * us, p["caps"][1].as_f64().unwrap() as f32 * us];
                        SdfNode::loft_capped(axis, up, p["harmonics"].as_u64().unwrap() as u32, rows, caps)
                    }
                    "capsules" => {
                        // union of capsules (a, b, r); a == b gives a sphere
                        let mut items: Vec<SdfNode> = Vec::new();
                        for seg in p["segments"].as_array().unwrap() {
                            let a = v3(&seg[0]) * us;
                            let b = v3(&seg[1]) * us;
                            let r = seg[2].as_f64().unwrap() as f32 * us;
                            items.push(if (a - b).length() < 1e-9 { SdfNode::sphere(r).translate(a.x, a.y, a.z) } else { SdfNode::capsule(a, b, r) });
                        }
                        // balanced union (pairwise), so the tree depth is log2(n) (a chain overflows the JSON reader's recursion limit)
                        while items.len() > 1 {
                            let mut next = Vec::with_capacity(items.len() / 2 + 1);
                            let mut it = items.into_iter();
                            while let Some(x) = it.next() {
                                next.push(match it.next() {
                                    Some(y) => x.union(y),
                                    None => x,
                                });
                            }
                            items = next;
                        }
                        Ok(items.pop().unwrap())
                    }
                    "stack" => {
                        let levels: Vec<f32> = p["levels"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32 * us).collect();
                        let loops: Vec<Vec<Vec<glam::Vec2>>> = p["loops"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|lv| {
                                lv.as_array()
                                    .unwrap()
                                    .iter()
                                    .map(|lp| lp.as_array().unwrap().iter().map(|q| glam::Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32) * us).collect())
                                    .collect()
                            })
                            .collect();
                        let q = &p["quat"];
                        let quat = glam::Quat::from_xyzw(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32, q[2].as_f64().unwrap() as f32, q[3].as_f64().unwrap() as f32);
                        match SdfNode::smooth_section_stack(levels, loops, p["r0"].as_f64().unwrap() as f32 * us, p["eps"].as_f64().unwrap() as f32) {
                            Ok(n) => {
                                if let SdfNode::SmoothSectionStack(st) = &n {
                                    println!("part {} stack lipschitz {} knots {:?}", p["name"], st.lipschitz(), st.knots());
                                }
                                Ok(n.rotate(quat))
                            }
                            Err(e) => {
                                eprintln!("REFUSED part {}: {}", p["name"], e);
                                std::process::exit(2);
                            }
                        }
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
                let sub = p["op"].as_str() == Some("subtract");
                node = Some(match node {
                    None => part,
                    Some(n) if sub => n.subtract(part),
                    Some(n) if k > 0.0 => n.smooth_union(part, k),
                    Some(n) => n.union(part),
                });
            }
            let tree = SdfTree::new(node.unwrap());
            alice_sdf::io::save(&tree, &a[3]).expect("save");
            println!("saved {}", a[3]);
            if a.len() > 4 {
                // a[4]: points (xyz f32); compare the in-process tree with the saved-and-reloaded file, bit for bit
                let p = f32s(&a[4]);
                let back = alice_sdf::io::load(&a[3]).expect("reload");
                let mut diff = 0usize;
                let n = p.len() / 3;
                for c in p.chunks_exact(3) {
                    let q = Vec3::new(c[0], c[1], c[2]);
                    if eval(&tree.root, q).to_bits() != eval(&back.root, q).to_bits() {
                        diff += 1;
                    }
                }
                assert!(n > 0, "zero compared points");
                println!("reload check: {n} points, {diff} differ in bits");
            }
        }
        "eval" => {
            let tree = alice_sdf::io::load(&a[2]).expect("load");
            let p = f32s(&a[3]);
            let compiled = std::env::var("EVAL_COMPILED").is_ok();
            let comp = if compiled { Some(alice_sdf::compiled::CompiledSdf::compile(&tree.root)) } else { None };
            let d: Vec<f32> = p.par_chunks_exact(3).map(|c| { let q = Vec3::new(c[0], c[1], c[2]); if let Some(c) = &comp { alice_sdf::compiled::eval_compiled(c, q) } else { eval(&tree.root, q) } }).collect();
            write_f32s(&a[4], &d);
        }
        "trace" => {
            let tree = alice_sdf::io::load(&a[2]).expect("load");
            let r = f32s(&a[3]);
            let comp = if std::env::var("TRACE_COMPILED").is_ok() { Some(alice_sdf::compiled::CompiledSdf::compile(&tree.root)) } else { None };
            let ev = |q: Vec3| if let Some(c) = &comp { alice_sdf::compiled::eval_compiled(c, q) } else { eval(&tree.root, q) };
            let teps: f32 = std::env::var("TRACE_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.002);
            let th: f32 = std::env::var("TRACE_H").ok().and_then(|v| v.parse().ok()).unwrap_or(0.01);
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
                        if d < teps {
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
                        let h = th;
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
        "--trace" => {
            // 66's G5 interface: --trace <tree> <rays.f32 (ox oy oz dx dy dz)> <out prefix> <eps> <tmax> <h>
            // writes <out>.t.f32 (hit t; NaN = miss within tmax; -1 = the ray starts inside), <out>.hg.f32 (central-difference
            // gradient at the hit, h), <out>.steps.f32 (march steps); plain tree eval, no step scaling
            let tree = alice_sdf::io::load(&a[2]).expect("load");
            let r = f32s(&a[3]);
            let (eps, tmax, h): (f32, f32, f32) = (a[5].parse().unwrap(), a[6].parse().unwrap(), a[7].parse().unwrap());
            let root = &tree.root;
            let res: Vec<(f32, Vec3, f32)> = r
                .par_chunks_exact(6)
                .map(|c| {
                    let o = Vec3::new(c[0], c[1], c[2]);
                    let dir = Vec3::new(c[3], c[4], c[5]);
                    if eval(root, o) < 0.0 {
                        return (-1.0, Vec3::ZERO, 0.0);
                    }
                    let mut t = 0.0f32;
                    let mut steps = 0u32;
                    while steps < 100_000 {
                        let d = eval(root, o + dir * t);
                        steps += 1;
                        if d < eps {
                            let p = o + dir * t;
                            let g = Vec3::new(
                                eval(root, p + Vec3::X * h) - eval(root, p - Vec3::X * h),
                                eval(root, p + Vec3::Y * h) - eval(root, p - Vec3::Y * h),
                                eval(root, p + Vec3::Z * h) - eval(root, p - Vec3::Z * h),
                            ) / (2.0 * h);
                            return (t, g, steps as f32);
                        }
                        t += d;
                        if t > tmax {
                            break;
                        }
                    }
                    (f32::NAN, Vec3::ZERO, steps as f32)
                })
                .collect();
            write_f32s(&format!("{}.t.f32", a[4]), &res.iter().map(|x| x.0).collect::<Vec<_>>());
            write_f32s(&format!("{}.hg.f32", a[4]), &res.iter().flat_map(|x| [x.1.x, x.1.y, x.1.z]).collect::<Vec<_>>());
            write_f32s(&format!("{}.steps.f32", a[4]), &res.iter().map(|x| x.2).collect::<Vec<_>>());
            println!("traced {} rays", res.len());
        }
        m => panic!("mode {m}"),
    }
}

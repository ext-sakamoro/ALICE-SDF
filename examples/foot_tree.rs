//! Local only (not committed): the foot as one ALICE-SDF tree, heel/midfoot loft + forefoot band smooth
//! section stack + five toe lofts, unioned (no blend yet), evaluated on the CPU. Writes the tree (JSON) and,
//! per view (side, top, bottom), a raw render: per pixel the sphere-traced depth and the field's normal, at
//! long side / LONG pixels. The PNGs (and R0's, same view and scale) are made from the raw files.
//! usage: foot_tree <foot_tree_input.json> <out dir> [LONG px]
use alice_sdf::eval::eval;
use alice_sdf::io::to_json_string;
use alice_sdf::types::{SdfNode, SdfTree};
use glam::{Mat3, Quat, Vec2, Vec3};
use rayon::prelude::*;
use serde_json::Value;
use std::io::Write;

fn f(v: &Value) -> f32 {
    v.as_f64().expect("number") as f32
}
fn v3(v: &Value) -> Vec3 {
    Vec3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}
fn v2s(v: &Value) -> Vec<Vec2> {
    v.as_array().expect("points").iter().map(|p| Vec2::new(f(&p[0]), f(&p[1]))).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let inp: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("input")).expect("json");
    let out = &a[2];
    let long: f32 = a.get(3).map_or(2000.0, |s| s.parse().expect("LONG"));
    std::fs::create_dir_all(out).expect("out dir");
    // each part with a box that holds its surface (the axis points widened by the largest section
    // control point: the curve lies in its control points' hull); for tracing, a part whose box is
    // farther than the best distance so far is skipped: its surface is at least that far, so the
    // value stays at most the distance to the union's surface (the surface itself is unchanged)
    let loft = |key: &str| -> (SdfNode, Vec3, Vec3) {
        let l = &inp[key];
        let axis: Vec<Vec3> = l["axis"].as_array().expect("axis").iter().map(v3).collect();
        let rows: Vec<Vec<Vec2>> = l["rows"].as_array().expect("rows").iter().map(v2s).collect();
        let r = rows.iter().flatten().map(|q| q.length()).fold(0.0f32, f32::max) * 1.05 + 1e-3;
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for a in &axis {
            lo = lo.min(*a - Vec3::splat(r));
            hi = hi.max(*a + Vec3::splat(r));
        }
        (SdfNode::loft_curves(axis, v3(&l["up"]), rows).unwrap_or_else(|e| panic!("{key}: {e}")), lo, hi)
    };
    let mut parts: Vec<(String, (SdfNode, Vec3, Vec3))> = vec![("body".into(), loft("body"))];
    let toes = std::env::var("TOES").map_or(true, |v| v != "0");
    for k in (0..5).filter(|_| toes) {
        parts.push((format!("toe{k}"), loft(&format!("toe{k}"))));
    }
    let b = &inp["band"];
    let levels: Vec<f32> = b["levels"].as_array().expect("levels").iter().map(f).collect();
    let loops: Vec<Vec<Vec<Vec2>>> = b["loops"].as_array().expect("loops").iter().map(|lv| lv.as_array().expect("level").iter().map(v2s).collect()).collect();
    let stack = SdfNode::smooth_section_stack(levels.clone(), loops.clone(), 0.001, 0.1).expect("band stack");
    if let SdfNode::SmoothSectionStack(st) = &stack {
        println!("band stack: Lipschitz bound {:.3}", st.lipschitz());
    }
    let q = Quat::from_mat3(&Mat3::from_cols(v3(&b["x"]), v3(&b["axis"]), v3(&b["z"])));
    let (mut blo, mut bhi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    let (y0, y1) = (levels[1], levels[levels.len() - 2]);
    for lv in &loops {
        for c in lv.iter().flatten() {
            for y in [y0, y1] {
                let w = q * Vec3::new(c.x, y, c.y);
                blo = blo.min(w);
                bhi = bhi.max(w);
            }
        }
    }
    parts.push(("band".into(), (stack.rotate(q), blo - Vec3::splat(1e-3), bhi + Vec3::splat(1e-3))));
    let culled = |p: Vec3| -> f32 {
        let p = Vec3::new(p.x.abs(), p.y, p.z);
        let mut best = f32::INFINITY;
        for (_, (n, lo, hi)) in &parts {
            let db = (*lo - p).max(p - *hi).max(Vec3::ZERO).length();
            if db < best {
                best = best.min(eval(n, p));
            }
        }
        best
    };
    let mut tree: Option<SdfNode> = None;
    for (_, (p, _, _)) in &parts {
        tree = Some(match tree {
            None => p.clone(),
            Some(t) => t.union(p.clone()),
        });
    }
    // both feet: R0's right foot is the left one mirrored in x = 0 (its vertices within 2e-7 m of the
    // mirrored left surface), and the left foot lies wholly in x > 0, so |x| gives each point the nearer foot
    let tree = tree.expect("parts").mirror(true, false, false);
    std::fs::write(format!("{out}/foot_tree.json"), to_json_string(&SdfTree::new(tree.clone())).expect("json")).expect("write tree");
    // POINTS=<json [[x, y, z], ...]>: the tree's values there, to <out>/values.json (to compare builds)
    if let Ok(pf) = std::env::var("POINTS") {
        let pts: Value = serde_json::from_str(&std::fs::read_to_string(&pf).expect("points")).expect("points json");
        let vals: Vec<f32> = pts.as_array().expect("points").iter().map(|p| eval(&tree, v3(p))).collect();
        std::fs::write(format!("{out}/values.json"), serde_json::to_string(&vals).expect("values")).expect("write values");
        println!("values at {} points -> {out}/values.json", vals.len());
    }
    // PROBE=x,y,z (m): every part's value there
    if let Ok(pr) = std::env::var("PROBE") {
        let v: Vec<f32> = pr.split(',').map(|x| x.parse().expect("probe")).collect();
        let p = Vec3::new(v[0].abs(), v[1], v[2]);
        for (name, (n, _, _)) in &parts {
            println!("probe {name}: {:.6e} m", eval(n, p));
        }
    }
    // G5=<json {p: [[x,y,z]], n: [[x,y,z]]}> (R0 PN points and normals): each point moved onto the tree's
    // surface along R0's normal, the tree's normal there, the part that sets the field and the gap to the
    // next part, to <out>/g5.json
    if let Ok(gf) = std::env::var("G5") {
        let g: Value = serde_json::from_str(&std::fs::read_to_string(&gf).expect("g5 points")).expect("g5 json");
        let ps: Vec<Vec3> = g["p"].as_array().expect("p").iter().map(v3).collect();
        let ns: Vec<Vec3> = g["n"].as_array().expect("n").iter().map(v3).collect();
        let rows: Vec<Value> = ps
            .par_iter()
            .zip(ns.par_iter())
            .map(|(p0, n0)| {
                let f = |q: Vec3| eval(&tree, q);
                let h = 5e-5;
                let grad = |q: Vec3| {
                    Vec3::new(f(q + Vec3::X * h) - f(q - Vec3::X * h), f(q + Vec3::Y * h) - f(q - Vec3::Y * h), f(q + Vec3::Z * h) - f(q - Vec3::Z * h)) / (2.0 * h)
                };
                let mut q = *p0;
                for _ in 0..40 {
                    let d = f(q);
                    let gn = grad(q).dot(*n0);
                    if d.abs() < 1e-7 || gn.abs() < 1e-3 {
                        break;
                    }
                    q -= *n0 * (d / gn).clamp(-0.002, 0.002);
                }
                let nn = grad(q).normalize_or_zero();
                let qa = Vec3::new(q.x.abs(), q.y, q.z);
                let mut vals: Vec<(f32, usize)> = parts.iter().enumerate().map(|(i, (_, (n, _, _)))| (eval(n, qa), i)).collect();
                vals.sort_by(|a, b| a.0.total_cmp(&b.0));
                serde_json::json!({"q": [q.x, q.y, q.z], "f": f(q), "n": [nn.x, nn.y, nn.z], "part": parts[vals[0].1].0, "next": parts[vals[1].1].0, "gap": vals[1].0 - vals[0].0})
            })
            .collect();
        std::fs::write(format!("{out}/g5.json"), serde_json::to_string(&rows).expect("g5 out")).expect("write g5");
        println!("g5: {} points -> {out}/g5.json", rows.len());
    }
    if std::env::var("NORENDER").is_ok_and(|v| v == "1") {
        return;
    }
    // the bounds: the parts' axis points and the band's levels, widened by 15 mm
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for key in ["body", "toe0", "toe1", "toe2", "toe3", "toe4"] {
        for p in inp[key]["axis"].as_array().expect("axis") {
            lo = lo.min(v3(p));
            hi = hi.max(v3(p));
        }
    }
    lo -= Vec3::splat(0.015);
    hi += Vec3::splat(0.015);
    lo.x = -hi.x;
    for (name, dv) in [("side", Vec3::new(-1.0, 0.0, 0.0)), ("top", Vec3::new(0.0, -1.0, 0.0)), ("bottom", Vec3::new(0.0, 1.0, 0.0))] {
        let ax = if dv.x != 0.0 { 0 } else { 1 };
        let uv: [usize; 2] = if ax == 0 { [1, 2] } else { [0, 2] };
        let px = (hi[uv[0]] - lo[uv[0]]).max(hi[uv[1]] - lo[uv[1]]) / long;
        let (nu, nv) = (((hi[uv[0]] - lo[uv[0]]) / px).ceil() as usize, ((hi[uv[1]] - lo[uv[1]]) / px).ceil() as usize);
        let far = hi[ax] - lo[ax];
        let start = if dv[ax] > 0.0 { lo[ax] } else { hi[ax] };
        let pix: Vec<[f32; 4]> = (0..nu * nv)
            .into_par_iter()
            .map(|r| {
                let (i, j) = (r / nv, r % nv);
                let mut o = Vec3::ZERO;
                o[uv[0]] = lo[uv[0]] + (i as f32 + 0.5) * px;
                o[uv[1]] = lo[uv[1]] + (j as f32 + 0.5) * px;
                o[ax] = start;
                let mut t = 0.0f32;
                for _ in 0..3000 {
                    let p = o + dv * t;
                    let d = culled(p);
                    if d < 1e-6 {
                        let h = 5e-5;
                        let n = Vec3::new(
                            culled(p + Vec3::X * h) - culled(p - Vec3::X * h),
                            culled(p + Vec3::Y * h) - culled(p - Vec3::Y * h),
                            culled(p + Vec3::Z * h) - culled(p - Vec3::Z * h),
                        )
                        .normalize_or_zero();
                        return [t, n.x, n.y, n.z];
                    }
                    t += d.max(1e-7);
                    if t > far {
                        break;
                    }
                }
                [f32::INFINITY, 0.0, 0.0, 0.0]
            })
            .collect();
        let mut fh = std::fs::File::create(format!("{out}/raw_{name}.bin")).expect("raw");
        let hdr = [nu as f32, nv as f32, px, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z, ax as f32, uv[0] as f32, uv[1] as f32, dv.x, dv.y, dv.z, start];
        for x in hdr.iter() {
            fh.write_all(&x.to_le_bytes()).expect("w");
        }
        for p in &pix {
            for x in p {
                fh.write_all(&x.to_le_bytes()).expect("w");
            }
        }
        println!("{name}: {nu}x{nv} px at {:.4} mm, hits {}", px * 1e3, pix.iter().filter(|p| p[0].is_finite()).count());
    }
}

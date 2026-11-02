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
    let loft = |key: &str| -> SdfNode {
        let l = &inp[key];
        let axis: Vec<Vec3> = l["axis"].as_array().expect("axis").iter().map(v3).collect();
        let rows: Vec<Vec<Vec2>> = l["rows"].as_array().expect("rows").iter().map(v2s).collect();
        SdfNode::loft_curves(axis, v3(&l["up"]), rows).unwrap_or_else(|e| panic!("{key}: {e}"))
    };
    let mut parts: Vec<(String, SdfNode)> = vec![("body".into(), loft("body"))];
    let toes = std::env::var("TOES").map_or(true, |v| v != "0");
    for k in (0..5).filter(|_| toes) {
        parts.push((format!("toe{k}"), loft(&format!("toe{k}"))));
    }
    let b = &inp["band"];
    let levels: Vec<f32> = b["levels"].as_array().expect("levels").iter().map(f).collect();
    let loops: Vec<Vec<Vec<Vec2>>> = b["loops"].as_array().expect("loops").iter().map(|lv| lv.as_array().expect("level").iter().map(v2s).collect()).collect();
    let stack = SdfNode::smooth_section_stack(levels, loops, 0.001, 0.1).expect("band stack");
    if let SdfNode::SmoothSectionStack(st) = &stack {
        println!("band stack: Lipschitz bound {:.3}", st.lipschitz());
    }
    let q = Quat::from_mat3(&Mat3::from_cols(v3(&b["x"]), v3(&b["axis"]), v3(&b["z"])));
    parts.push(("band".into(), stack.rotate(q)));
    let mut tree: Option<SdfNode> = None;
    for (_, p) in &parts {
        tree = Some(match tree {
            None => p.clone(),
            Some(t) => t.union(p.clone()),
        });
    }
    let tree = tree.expect("parts");
    std::fs::write(format!("{out}/foot_tree.json"), to_json_string(&SdfTree::new(tree.clone())).expect("json")).expect("write tree");
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
                    let d = eval(&tree, p);
                    if d < 1e-6 {
                        let h = 5e-5;
                        let n = Vec3::new(
                            eval(&tree, p + Vec3::X * h) - eval(&tree, p - Vec3::X * h),
                            eval(&tree, p + Vec3::Y * h) - eval(&tree, p - Vec3::Y * h),
                            eval(&tree, p + Vec3::Z * h) - eval(&tree, p - Vec3::Z * h),
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

//! relaw <old.asdf.json> <new.asdf.json> <old samples.json>: rebuild every Loft / LoftCurves of a stored tree
//! under the current loft bound law. The stored JSON is only an input reader: each loft's axis / up / rows
//! (or points + knots) / caps go through the current public constructor (`loft_capped` /
//! `loft_curves_knots_capped`), which computes the bounds; stored bounds are dropped, never copied.
//! The rebuilt tree is saved with save_asdf_json (fresh semantics id), loaded back with the verifying loader,
//! and evaluated at the old export's 4096 sample points (CPU values of the old library) to report |Δd|.
use alice_sdf::prelude::*;
use glam::Vec2;
use serde_json::Value;

fn f(v: &Value) -> f32 { v.as_f64().expect("number") as f32 }
fn v3(v: &Value) -> Vec3 { Vec3::new(f(&v[0]), f(&v[1]), f(&v[2])) }

fn rebuild(kind: &str, x: &Value, n: &mut [usize; 2]) -> Value {
    let axis: Vec<Vec3> = x["axis"].as_array().unwrap().iter().map(v3).collect();
    let up = v3(&x["up"]);
    let node = if kind == "Loft" {
        let k = x["harmonics"].as_u64().unwrap() as u32;
        let flat: Vec<f32> = x["rows"].as_array().unwrap().iter().map(f).collect();
        let w = (2 * k + 1) as usize;
        assert!(flat.len() % w == 0, "Loft rows not a multiple of 2K+1");
        let rows: Vec<Vec<f32>> = flat.chunks(w).map(|c| c.to_vec()).collect();
        let caps = [f(&x["caps"][0]), f(&x["caps"][1])];
        n[0] += 1;
        SdfNode::loft_capped(axis, up, k, rows, caps).unwrap_or_else(|e| panic!("Loft rebuild: {e}"))
    } else {
        let m = x["section_points"].as_u64().unwrap() as usize;
        let flat: Vec<f32> = x["points"].as_array().unwrap().iter().map(f).collect();
        assert!(flat.len() % (2 * m) == 0, "LoftCurves points not a multiple of 2M");
        let rows: Vec<Vec<Vec2>> = flat.chunks(2 * m).map(|r| r.chunks(2).map(|p| Vec2::new(p[0], p[1])).collect()).collect();
        let knots: Vec<f32> = x["knots"].as_array().unwrap().iter().map(f).collect();
        // trees saved before caps existed have flat ends
        let caps = match x.get("caps") {
            Some(c) if !c.is_null() => [[f(&c[0][0]), f(&c[0][1]), f(&c[0][2])], [f(&c[1][0]), f(&c[1][1]), f(&c[1][2])]],
            _ => [[0.0; 3]; 2],
        };
        n[1] += 1;
        SdfNode::loft_curves_knots_capped(axis, up, knots, rows, caps).unwrap_or_else(|e| panic!("LoftCurves rebuild: {e}"))
    };
    serde_json::to_value(&node).expect("serialize")
}

fn walk(v: &mut Value, n: &mut [usize; 2]) {
    match v {
        Value::Object(o) => {
            let key = ["Loft", "LoftCurves"].into_iter().find(|k| o.len() == 1 && o.contains_key(*k));
            if let Some(k) = key {
                *v = rebuild(k, &o[k].clone(), n);
                return;
            }
            for x in o.values_mut() { walk(x, n); }
        }
        Value::Array(a) => for x in a.iter_mut() { walk(x, n); },
        _ => {}
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    assert!(a.len() == 4, "usage: relaw <old.asdf.json> <new.asdf.json> <old samples.json>");
    let mut j: Value = serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("read")).expect("json");
    let mut n = [0usize; 2];
    walk(&mut j["root"], &mut n);
    eprintln!("rebuilt Loft {} LoftCurves {}", n[0], n[1]);
    let root: SdfNode = serde_json::from_value(j["root"].clone()).expect("root deserialize");
    let version = j["version"].as_str().unwrap_or("").to_string();
    let mut tree = SdfTree::new(root);
    if !version.is_empty() { tree.version = version; }
    alice_sdf::io::save_asdf_json(&tree, &a[2]).expect("save");
    let back = alice_sdf::io::load_asdf_json(&a[2]).expect("reload with the verifying loader");
    let s: Value = serde_json::from_str(&std::fs::read_to_string(&a[3]).expect("samples")).expect("samples json");
    let (mut maxd, mut maxb, mut nb, mut diff, mut diffb) = (0f64, 0f64, 0usize, 0usize, 0usize);
    let pts = s.as_array().unwrap();
    let mut dump = String::from("x,y,z,old,new\n");
    for p in pts {
        let q = Vec3::new(f(&p[0]), f(&p[1]), f(&p[2]));
        let old = p[3].as_f64().unwrap();
        let new = eval(&back.root, q) as f64;
        let d = (new - old).abs();
        dump.push_str(&format!("{},{},{},{},{}\n", q.x, q.y, q.z, old, new));
        maxd = maxd.max(d);
        if d > 1e-6 { diff += 1; }
        if old.abs() < 0.005 { nb += 1; maxb = maxb.max(d); if d > 1e-6 { diffb += 1; } }
    }
    println!("compared {} points: max |dd| {:.3e} m, {} differ by > 1e-6 m; band |d|<5mm: {} points, max |dd| {:.3e} m, {} differ", pts.len(), maxd, diff, nb, maxb, diffb);
    std::fs::write(format!("{}.cmp.csv", a[2]), dump).expect("write cmp");
    assert!(!pts.is_empty(), "compared 0 points");
}

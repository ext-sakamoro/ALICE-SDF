//! Measures a body SDF built from measured curves against reference samples, without a mesh:
//! the SDF evaluated at reference surface samples (should be 0), its sign 2 mm either side along
//! the reference normal, and sphere-traced hits against the reference's first hits on the same rays.
//!
//! `cargo run --release --example curve_body_oracle -- <oracle_data.json> <candidate.json>`
//!
//! oracle_data.json: {"samples": [[x,y,z],…], "normals": [[…],…], "rays": [{"o","d","t_ref"|null},…]} (m)
//! candidate.json: a node tree, {"kind": "smooth_union", "k", "a", "b"} | {"kind": "cylinder", "r", "h",
//! "at": [x,y,z]} | {"kind": "superellipsoid", "size": [w,h,d], "e1", "e2", "at"} | {"kind": "loft_world",
//! "axis", "up", "curves"} (the loft needs the loft primitive). Exit 1 when a gate fails or nothing was compared.
use alice_sdf::{eval, SdfNode};
use glam::Vec3;
use serde_json::Value;

fn v3(v: &Value) -> Vec3 {
    let a = v.as_array().expect("a 3-vector");
    Vec3::new(
        a[0].as_f64().unwrap() as f32,
        a[1].as_f64().unwrap() as f32,
        a[2].as_f64().unwrap() as f32,
    )
}

fn build(c: &Value) -> Result<SdfNode, String> {
    let f = |k: &str| {
        c[k].as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{}: no {k}", c["kind"]))
    };
    match c["kind"].as_str().unwrap_or("") {
        "smooth_union" => Ok(build(&c["a"])?.smooth_union(build(&c["b"])?, f("k")?)),
        "union" => Ok(build(&c["a"])?.union(build(&c["b"])?)),
        "cylinder" => {
            let at = v3(&c["at"]);
            Ok(SdfNode::cylinder(f("r")?, f("h")?).translate(at.x, at.y, at.z))
        }
        "superellipsoid" => {
            let s = v3(&c["size"]);
            let at = v3(&c["at"]);
            Ok(SdfNode::superellipsoid(s.x, s.y, s.z, f("e1")?, f("e2")?)
                .translate(at.x, at.y, at.z))
        }
        "loft_world" => Err("loft_world: the loft primitive is not on this branch yet".into()),
        "asdf" => alice_sdf::io::load_asdf_json(c["path"].as_str().ok_or("asdf: no path")?)
            .map(|t| t.root)
            .map_err(|e| format!("asdf: {e}")),
        k => Err(format!("unknown candidate kind {k:?}")),
    }
}

fn pct(mut v: Vec<f32>, q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(f32::total_cmp);
    v[((v.len() - 1) as f32 * q).round() as usize]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        a.len(),
        2,
        "curve_body_oracle <oracle_data.json> <candidate.json>"
    );
    let d: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[0]).expect("oracle data")).expect("JSON");
    let c: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[1]).expect("candidate")).expect("JSON");
    let node = match build(&c) {
        Ok(n) => n,
        Err(e) => {
            println!("[oracle] RED: no candidate SDF: {e}");
            std::process::exit(1);
        }
    };
    let samples: Vec<Vec3> = d["samples"]
        .as_array()
        .expect("samples")
        .iter()
        .map(v3)
        .collect();
    let normals: Vec<Vec3> = d["normals"]
        .as_array()
        .expect("normals")
        .iter()
        .map(v3)
        .collect();
    assert_eq!(samples.len(), normals.len());
    let (mut abs_mm, mut sign_bad) = (Vec::new(), 0usize);
    for (p, n) in samples.iter().zip(&normals) {
        abs_mm.push(eval(&node, *p).abs() * 1e3);
        if eval(&node, *p + *n * 0.002) <= 0.0 || eval(&node, *p - *n * 0.002) >= 0.0 {
            sign_bad += 1;
        }
    }
    // per part (when the data carries labels): |sdf| p99 and the share of wrong signs
    if let Some(parts) = d["parts"].as_array() {
        let mut by: std::collections::BTreeMap<String, (Vec<f32>, usize)> =
            std::collections::BTreeMap::new();
        for ((p, n), lab) in samples.iter().zip(&normals).zip(parts) {
            let e = by
                .entry(lab.as_str().unwrap_or("?").to_string())
                .or_default();
            e.0.push(eval(&node, *p).abs() * 1e3);
            if eval(&node, *p + *n * 0.002) <= 0.0 || eval(&node, *p - *n * 0.002) >= 0.0 {
                e.1 += 1;
            }
        }
        for (k, (v, bad)) in by {
            let m = v.len();
            println!("[oracle]   {k:6}: {m} samples, |sdf| p50 {:.3} p90 {:.3} p99 {:.3} max {:.3} mm, sign wrong {:.2} %", pct(v.clone(), 0.5), pct(v.clone(), 0.9), pct(v.clone(), 0.99), pct(v, 1.0), 100.0 * bad as f32 / m as f32);
        }
    }
    // ORACLE_DUMP=path: per sample, the SDF at the sample and 2 mm outside / inside along the normal (m)
    if let Ok(path) = std::env::var("ORACLE_DUMP") {
        let rows: Vec<[f32; 3]> = samples
            .iter()
            .zip(&normals)
            .map(|(p, n)| {
                [
                    eval(&node, *p),
                    eval(&node, *p + *n * 0.002),
                    eval(&node, *p - *n * 0.002),
                ]
            })
            .collect();
        std::fs::write(&path, serde_json::to_string(&rows).unwrap()).expect("dump");
        println!("[oracle] per-sample values written to {path}");
    }
    let rays = d["rays"].as_array().expect("rays");
    let (mut dt_mm, mut disagree) = (Vec::new(), 0usize);
    let mut steps: Vec<f32> = Vec::new();
    for r in rays {
        let (o, dir) = (v3(&r["o"]), v3(&r["d"]));
        let hit = alice_sdf::raycast::raymarch(&node, o, dir, 1.0);
        match (r["t_ref"].as_f64(), hit) {
            (Some(t), Some(h)) => {
                dt_mm.push((h.distance - t as f32).abs() * 1e3);
                steps.push(h.steps as f32);
            }
            (None, None) => {}
            _ => disagree += 1,
        }
    }
    let n = samples.len();
    assert!(n > 0 && !rays.is_empty(), "nothing compared");
    let (p99, mx) = (pct(abs_mm.clone(), 0.99), pct(abs_mm.clone(), 1.0));
    let sign_frac = sign_bad as f32 / n as f32;
    let dt99 = pct(dt_mm.clone(), 0.99);
    println!("[oracle] samples {n}: |sdf| p50 {:.3} p99 {p99:.3} max {mx:.3} mm; sign wrong 2 mm either side {sign_bad} ({:.3} %)", pct(abs_mm, 0.5), sign_frac * 100.0);
    println!("[oracle] rays {}: hits compared {}, hit/miss disagree {disagree}; |Δt| p50 {:.3} p99 {dt99:.3} mm", rays.len(), dt_mm.len(), pct(dt_mm.clone(), 0.5));
    println!(
        "[oracle] steps on hits: p50 {} p90 {} p99 {} max {}",
        pct(steps.clone(), 0.5),
        pct(steps.clone(), 0.9),
        pct(steps.clone(), 0.99),
        pct(steps, 1.0)
    );
    let green = p99 <= 0.5
        && mx <= 2.0
        && sign_frac <= 0.001
        && disagree == 0
        && dt99 <= 0.5
        && !dt_mm.is_empty();
    println!("[oracle] gates: |sdf| p99 ≤ 0.5 mm, max ≤ 2 mm, sign wrong ≤ 0.1 %, hit/miss agree, |Δt| p99 ≤ 0.5 mm → {}", if green { "GREEN" } else { "RED" });
    std::process::exit(i32::from(!green));
}

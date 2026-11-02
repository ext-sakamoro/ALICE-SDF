//! Builds a body from curve-section lofts (rows files) and joins chosen pairs by ramp blends instead of min.
//! `body_seams <rows dir> <out.asdf.json> <plain|blend>`: plain = every part by min (the twin); blend =
//! torso–legs across y = 0.700 and arm–palm across x = ±0.516 by ramp blends, the rest by min.
use alice_sdf::io::save_asdf_json;
use alice_sdf::prelude::*;
use glam::{Vec2, Vec3};

fn v3(x: &serde_json::Value) -> Vec3 {
    Vec3::new(
        x[0].as_f64().unwrap() as f32,
        x[1].as_f64().unwrap() as f32,
        x[2].as_f64().unwrap() as f32,
    )
}

fn part(dir: &str, name: &str) -> SdfNode {
    let j: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/{name}.json")).expect("rows"))
            .expect("JSON");
    let axis: Vec<Vec3> = j["axis_m"].as_array().unwrap().iter().map(v3).collect();
    let rows: Vec<Vec<Vec2>> = j["rows_m"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|q| Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32))
                .collect()
        })
        .collect();
    SdfNode::loft_curves(axis, v3(&j["up"]), rows).unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, out, mode) = (&a[0], &a[1], a[2].as_str());
    let h: f32 = std::env::var("SEAM_H")
        .ok()
        .map_or(0.001, |s| s.parse().unwrap());
    let p = |n: &str| part(dir, n);
    let legs = p("leg.L").union(p("leg.R")).union(p("perineum"));
    let (torso, neck) = (p("torso"), p("neck"));
    let (arm_l, arm_r, palm_l, palm_r) = (p("arm.L"), p("arm.R"), p("palm.L"), p("palm.R"));
    let t0 = std::time::Instant::now();
    let root = if mode == "blend" {
        let lower = legs
            .ramp_blend(
                torso,
                Vec3::Y,
                0.700,
                0.020,
                0.010,
                Vec3::new(-0.15, 0.684, -0.12),
                Vec3::new(0.15, 0.716, 0.13),
                0.006,
                h,
            )
            .expect("torso-legs");
        let left = palm_l
            .ramp_blend(
                arm_l,
                -Vec3::X,
                -0.516,
                0.006,
                0.006,
                Vec3::new(0.512, 0.99, -0.06),
                Vec3::new(0.520, 1.10, 0.06),
                0.0015,
                h,
            )
            .expect("left wrist");
        let right = palm_r
            .ramp_blend(
                arm_r,
                Vec3::X,
                -0.516,
                0.006,
                0.006,
                Vec3::new(-0.520, 0.99, -0.06),
                Vec3::new(-0.512, 1.10, 0.06),
                0.0015,
                h,
            )
            .expect("right wrist");
        for (n, s) in [
            ("torso-legs", &lower),
            ("left wrist", &left),
            ("right wrist", &right),
        ] {
            if let SdfNode::RampBlend { params, .. } = s {
                println!(
                    "[seams] {n}: D {:.3} mm, Λ {:.3}",
                    params.diff_bound * 1e3,
                    params.lipschitz
                );
            }
        }
        lower.union(neck).union(left).union(right)
    } else {
        legs.union(torso)
            .union(neck)
            .union(arm_l.union(palm_l))
            .union(arm_r.union(palm_r))
    };
    save_asdf_json(&SdfTree::new(root), out).expect("write");
    println!(
        "[seams] wrote {out} ({mode}, grid {h} m, {:.1} s)",
        t0.elapsed().as_secs_f64()
    );
}

//! A composite SDF: per part, the intersection of its view outlines (extruded Polygon2D) and of
//! its section stacks (each in its own axis frame, trimmed to the run of levels whose boundary change
//! per spacing stays within G_max, and unioned with the half-spaces outside its valid region so it only
//! carves inside); parts joined by a smooth union; an extra named subtree from a second outline file
//! above a height. Writes `.asdf.json` and prints probe rows.
//!
//! `cargo run --release --example body_v3 -- <outlines.json> <sections.json> <extra_outlines.json> <extra_part> <extra_y_m> <g_max> <k_m> <out.asdf.json>`
use alice_sdf::eval::eval_gradient;
use alice_sdf::io::save_asdf_json;
use alice_sdf::primitives::{boundary_hausdorff_bound, SectionStack};
use alice_sdf::{eval, SdfNode, SdfTree};
use glam::{Mat3, Quat, Vec2, Vec3};
use serde_json::Value;

fn hull(views: &Value, reach: f32) -> SdfNode {
    let mut part: Option<SdfNode> = None;
    for v in views.as_array().expect("views") {
        let rot = match (v["az"].as_f64(), v["name"].as_str()) {
            (Some(az), _) => Quat::from_rotation_y((az as f32).to_radians()),
            (None, Some("top")) => Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            (None, Some("bottom")) => Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
            _ => panic!("view {}: no azimuth", v["name"]),
        };
        let mut view: Option<SdfNode> = None;
        for poly in v["polys"].as_array().expect("polys") {
            let vertices: Vec<Vec2> = poly
                .as_array()
                .unwrap()
                .iter()
                .map(|q| Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32))
                .collect();
            let n = SdfNode::Polygon2D {
                vertices,
                half_height: reach,
            }
            .rotate(rot);
            view = Some(match view {
                None => n,
                Some(u) => u.union(n),
            });
        }
        if let Some(view) = view {
            part = Some(match part {
                None => view,
                Some(p) => p.intersection(view),
            });
        }
    }
    part.expect("a view")
}

/// Solid where the half-space `sg·(p[ax] − v_mm) ≥ 0` does not hold (outside the region)
fn outside(ax: usize, sg: f32, v_mm: f32) -> SdfNode {
    let mut n = Vec3::ZERO;
    n[ax] = sg;
    SdfNode::plane(n, sg * v_mm / 1000.0)
}

fn carve(sec: &Value, g_max: f32, report: &mut Vec<String>, name: &str) -> SdfNode {
    let col = |i: usize| {
        let c = &sec["R_columns"][i];
        Vec3::new(
            c[0].as_f64().unwrap() as f32,
            c[1].as_f64().unwrap() as f32,
            c[2].as_f64().unwrap() as f32,
        )
    };
    let rm = Mat3::from_cols(col(0), col(1), col(2));
    let levels: Vec<f32> = sec["levels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let loops: Vec<Vec<Vec<Vec2>>> = sec["loops"]
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
    let g: Vec<f32> = (0..levels.len() - 1)
        .map(|i| boundary_hausdorff_bound(&loops[i], &loops[i + 1]) / (levels[i + 1] - levels[i]))
        .collect();
    // the longest run of consecutive intervals with G ≤ g_max
    let (mut best, mut start) = ((0usize, 0usize), 0usize);
    for i in 0..=g.len() {
        if i == g.len() || g[i] > g_max {
            if i - start > best.1 - best.0 {
                best = (start, i);
            }
            start = i + 1;
        }
    }
    let (i0, i1) = (best.0, best.1); // levels i0..=i1
    assert!(i1 > i0, "{name}: no interval with G ≤ {g_max}");
    // the trimmed span's end sections repeated EXT past each end: the stack's own cap then lies outside
    // the span, so at the span ends (where the half-spaces below take over) the stack is still solidly
    // negative inside (a cap there would leave a zero sheet); the repeats add intervals with G = 0
    const EXT: f32 = 0.03;
    let mut lv = vec![levels[i0] - EXT];
    lv.extend_from_slice(&levels[i0..=i1]);
    lv.push(levels[i1] + EXT);
    let mut lp = vec![loops[i0].clone()];
    lp.extend_from_slice(&loops[i0..=i1]);
    lp.push(loops[i1].clone());
    let st = SectionStack::new(lv, lp).expect("a valid stack");
    report.push(format!(
        "{name}: levels {}..{} of {} (s {:.0}..{:.0} mm), L {:.2}",
        i0,
        i1,
        levels.len(),
        levels[i0] * 1e3,
        levels[i1] * 1e3,
        st.lipschitz()
    ));
    let axis = col(1);
    let mut c = SdfNode::SectionStack(st).rotate(Quat::from_mat3(&rm));
    // outside the trimmed span along the axis, and outside the part's region: no carving there
    c = c
        .union(SdfNode::plane(-axis, -levels[i1]))
        .union(SdfNode::plane(axis, levels[i0]));
    for h in sec["half_spaces_mm"].as_array().unwrap() {
        let (ax, sg, v) = (
            h[0].as_u64().unwrap() as usize,
            h[1].as_f64().unwrap() as f32,
            h[2].as_f64().unwrap() as f32,
        );
        c = c.union(outside(ax, sg, v));
    }
    c
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(a.len() >= 8, "body_v3 <outlines> <sections> <extra_outlines> <extra_part> <extra_y> <g_max> <k> <out> [probes.json]");
    let rd = |p: &str| -> Value {
        serde_json::from_str(&std::fs::read_to_string(p).expect(p)).expect("JSON")
    };
    let (ol, sc, ex) = (rd(&a[0]), rd(&a[1]), rd(&a[2]));
    let (extra_y, g_max, k): (f32, f32, f32) = (
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
        a[6].parse().unwrap(),
    );
    let carves: &[(&str, &[&str])] = &[
        ("torso", &["torso"]),
        ("arm.L", &["arm.L"]),
        ("arm.R", &["arm.R"]),
        ("leg.L", &["leg.L", "foot.L"]),
        ("leg.R", &["leg.R", "foot.R"]),
    ];
    let mut report = Vec::new();
    let mut root: Option<SdfNode> = None;
    for (part, stacks) in carves {
        let mut p = hull(&ol["parts"][*part], 2.0);
        for s in *stacks {
            p = p.intersection(carve(&sc["parts"][*s], g_max, &mut report, s));
        }
        root = Some(match root {
            None => p,
            Some(r) => r.smooth_union(p, k),
        });
    }
    // the extra part (an interim subtree a later part replaces): its outline hull above extra_y
    let extra =
        hull(&ex["parts"][a[3].as_str()], 2.0).intersection(SdfNode::plane(-Vec3::Y, -extra_y));
    let root = root.unwrap().smooth_union(extra, k);
    for r in &report {
        println!("[v3] stack {r}");
    }
    // probe rows: points inside each part with their true depth (from the reference); the field is divided
    // by its stacks' bounds, so it must reach at least half the true depth after undoing the largest bound,
    // and be negative (a zero sheet reads ≈ 0 and fails both)
    let l_max: f32 = report
        .iter()
        .filter_map(|r| r.rsplit("L ").next()?.parse::<f32>().ok())
        .fold(1.0, f32::max);
    let mut bad = 0;
    if let Some(path) = a.get(8) {
        let pr = rd(path);
        for (n, e) in pr.as_object().expect("probes") {
            let q = &e["point"];
            let q = Vec3::new(
                q[0].as_f64().unwrap() as f32,
                q[1].as_f64().unwrap() as f32,
                q[2].as_f64().unwrap() as f32,
            );
            let depth = e["depth_mm"].as_f64().unwrap() as f32 / 1000.0;
            let (d, gr) = (eval(&root, q), eval_gradient(&root, q).length());
            // (no gradient condition: a centre probe sits on the medial axis, where any distance field's
            // gradient vanishes; a zero sheet reads ≈ 0 and fails the depth condition)
            let ok = d < 0.0 && -d * l_max >= 0.5 * depth;
            println!("[v3] probe {n:24} sdf {:.1} mm (×L {:.1}) vs true depth {:.1} mm, |∇|·L {:.2} → {}", d * 1e3, -d * l_max * 1e3, depth * 1e3, gr * l_max, if ok { "ok" } else { "FAIL" });
            if !ok {
                bad += 1;
                // the field ±1 mm along each axis: a zero sheet reads ≈ 0 all round, a crease of a min / max reads
                // a kink (one side steeper) with a negative value
                for (ax, e) in [("x", Vec3::X), ("y", Vec3::Y), ("z", Vec3::Z)] {
                    let (m, p) = (eval(&root, q - e * 0.001), eval(&root, q + e * 0.001));
                    println!(
                        "[v3]     {ax}: -1 mm {:.2} / 0 {:.2} / +1 mm {:.2} mm",
                        m * 1e3,
                        d * 1e3,
                        p * 1e3
                    );
                }
            }
        }
    } else {
        println!("[v3] no probe file given");
        bad += 1;
    }
    save_asdf_json(&SdfTree::new(root), &a[7]).expect("write");
    println!("[v3] wrote {} (extra part \"{}\" above y {extra_y}: interim, replaced later); probe failures {bad}", a[7], a[3]);
    std::process::exit(i32::from(bad > 0));
}

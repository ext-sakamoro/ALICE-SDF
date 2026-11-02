//! A body SDF from outlines: per part, each view's closed outline becomes a 2D polygon SDF extruded
//! along that view's direction, the views of a part are intersected, and the parts are joined by a
//! smooth union. Writes the tree as `.asdf.json`.
//!
//! `cargo run --release --example outline_hull -- <outlines.json> <out.asdf.json> [k_m] [y_lo_m] [y_hi_m]`
//!
//! outlines.json: {"parts": {name: [{"name", "az" (degrees, or null for "top" / "bottom"), "polys": [[[u, v], …]]}]}}
//! in m; a view at azimuth a projects (x, y, z) to (u, v) = (x cos a − z sin a, y); "top" to (x, z), "bottom" to
//! (x, −z). The legs below y_lo (feet) and the core above y_hi (head) are kept as their own subtrees.
use alice_sdf::io::save_asdf_json;
use alice_sdf::{eval, SdfNode, SdfTree};
use glam::{Quat, Vec2, Vec3};
use serde_json::Value;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 2,
        "outline_hull <outlines.json> <out.asdf.json> [k] [y_lo] [y_hi]"
    );
    let k: f32 = a.get(2).map_or(0.01, |s| s.parse().expect("k"));
    let y_lo: f32 = a.get(3).map_or(0.125, |s| s.parse().expect("y_lo"));
    let y_hi: f32 = a.get(4).map_or(1.13, |s| s.parse().expect("y_hi"));
    let d: Value =
        serde_json::from_str(&std::fs::read_to_string(&a[0]).expect("outlines")).expect("JSON");
    let reach = 2.0; // half-depth of each extrusion (m): longer than the body
    let mut parts: Vec<(String, SdfNode)> = Vec::new();
    for (name, views) in d["parts"].as_object().expect("parts") {
        let mut part: Option<SdfNode> = None;
        for v in views.as_array().expect("views") {
            let rot = match (v["az"].as_f64(), v["name"].as_str()) {
                (Some(az), _) => Quat::from_rotation_y((az as f32).to_radians()),
                (None, Some("top")) => Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
                (None, Some("bottom")) => Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2),
                _ => panic!("view {}: no azimuth", v["name"]),
            };
            // a view with several outer loops: their union (each loop its own polygon)
            let mut view: Option<SdfNode> = None;
            for poly in v["polys"].as_array().expect("polys") {
                let vertices: Vec<Vec2> = poly
                    .as_array()
                    .expect("a loop")
                    .iter()
                    .map(|q| {
                        Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32)
                    })
                    .collect();
                assert!(
                    vertices.len() >= 3,
                    "{name} {}: a loop of {} vertices",
                    v["name"],
                    vertices.len()
                );
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
            let Some(view) = view else { continue };
            part = Some(match part {
                None => view,
                Some(p) => p.intersection(view),
            });
        }
        let p = part.unwrap_or_else(|| panic!("{name}: no view"));
        // the head (core above y_hi) and the feet (legs below y_lo) as their own parts, from the same hull,
        // so they can be replaced later; a part and its own split rejoin by a hard union (exactly the part).
        // ALICE-SDF's plane is dot(p, n) − distance (positive on the normal side = outside)
        let below = |y: f32| SdfNode::plane(Vec3::Y, y); // keeps y < y
        let above = |y: f32| SdfNode::plane(-Vec3::Y, -y); // keeps y > y
        let p = if name.starts_with("leg") {
            p.clone()
                .intersection(above(y_lo))
                .union(p.intersection(below(y_lo)))
        } else if name == "core" {
            p.clone()
                .intersection(below(y_hi))
                .union(p.intersection(above(y_hi)))
        } else {
            p
        };
        parts.push((name.clone(), p));
    }
    parts.sort_by(|x, y| x.0.cmp(&y.0));
    let names: Vec<String> = parts.iter().map(|p| p.0.clone()).collect();
    let mut it = parts.into_iter().map(|p| p.1);
    let mut root = it.next().expect("a part");
    for p in it {
        root = root.smooth_union(p, k);
    }
    // points inside the body must be inside the tree (an empty part would read as outside everywhere)
    let inside = [
        Vec3::new(0.0, 0.95, 0.0),
        Vec3::new(0.06, 0.40, 0.0),
        Vec3::new(-0.06, 0.40, 0.0),
    ];
    let vals: Vec<f32> = inside.iter().map(|&q| eval(&root, q)).collect();
    println!("[hull] parts {names:?} (head and feet split off as their own subtrees), smooth union k {k} m, splits at y {y_lo} / {y_hi}; inside probes {vals:?} (all < 0 expected)");
    assert!(
        vals.iter().all(|&v| v < 0.0),
        "an inside probe reads outside: a part is empty"
    );
    save_asdf_json(&SdfTree::new(root), &a[1]).expect("write");
    println!("[hull] wrote {}", a[1]);
}

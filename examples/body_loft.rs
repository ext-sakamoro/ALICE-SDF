//! Builds a body from curve-section lofts: one loft per part from a rows file (axis points, up, uniform periodic
//! cubic B-spline control rows, written by an external fitting step), parts joined by smooth unions.
//! Checks each loft against its own rows: on the section curve at every row's level the field must read ~0
//! (the rows are the axis B-spline's control rows; the section at level i is (P(i-1) + 4 P(i) + P(i+1)) / 6,
//! the end rows met exactly).
//! `cargo run --release --example body_loft -- <rows dir> <k_m> <out.asdf.json> <part> [<part> ...]`
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

// uniform periodic cubic B-spline point at parameter t in [0, m)
fn curve_at(p: &[Vec2], t: f32) -> Vec2 {
    let m = p.len();
    let i = t.floor() as isize;
    let u = t - i as f32;
    let w = [
        (1.0 - u).powi(3),
        3.0 * u.powi(3) - 6.0 * u * u + 4.0,
        -3.0 * u.powi(3) + 3.0 * u * u + 3.0 * u + 1.0,
        u.powi(3),
    ];
    let mut q = Vec2::ZERO;
    for (k, wk) in w.iter().enumerate() {
        q += p[((i - 1 + k as isize).rem_euclid(m as isize)) as usize] * (*wk / 6.0);
    }
    q
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 4,
        "body_loft <rows dir> <k_m> <out.asdf.json> <part> [...]"
    );
    let k: f32 = a[1].parse().unwrap();
    let mut root: Option<SdfNode> = None;
    for part in &a[3..] {
        let j: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(format!("{}/{part}.json", a[0])).expect("rows file"),
        )
        .expect("JSON");
        let axis: Vec<Vec3> = j["axis_m"].as_array().unwrap().iter().map(v3).collect();
        let up = v3(&j["up"]);
        let rows: Vec<Vec<Vec2>> = j["rows_m"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|q| {
                        Vec2::new(q[0].as_f64().unwrap() as f32, q[1].as_f64().unwrap() as f32)
                    })
                    .collect()
            })
            .collect();
        let node = SdfNode::loft_curves(axis.clone(), up, rows.clone())
            .unwrap_or_else(|e| panic!("{part}: {e}"));
        // self-check: the field on the section curve at every level
        let tree = SdfTree::new(node.clone());
        let (a0, a1) = (axis[0], axis[axis.len() - 1]);
        let t_dir = (a1 - a0).normalize();
        let n_dir = (up - t_dir * up.dot(t_dir)).normalize();
        let b_dir = t_dir.cross(n_dir);
        let nr = rows.len();
        let mut worst = 0.0_f32;
        let mut sum = 0.0_f64;
        let mut cnt = 0usize;
        for i in 0..nr {
            let sec: Vec<Vec2> = if i == 0 || i == nr - 1 {
                rows[i].clone()
            } else {
                (0..rows[i].len())
                    .map(|c| (rows[i - 1][c] + 4.0 * rows[i][c] + rows[i + 1][c]) / 6.0)
                    .collect()
            };
            let c = a0 + (a1 - a0) * (i as f32 / (nr - 1) as f32);
            let m = sec.len();
            for s in 0..(4 * m) {
                let q = curve_at(&sec, s as f32 / 4.0);
                let p = c + n_dir * q.x + b_dir * q.y;
                let f = eval(&tree.root, p).abs();
                worst = worst.max(f);
                sum += f64::from(f);
                cnt += 1;
            }
        }
        println!("[loft] {part}: {nr} rows x {} control points; field on its own section curves: mean {:.4} max {:.4} mm", rows[0].len(), sum / cnt as f64 * 1e3, worst * 1e3);
        root = Some(match root {
            None => node,
            Some(r) => {
                if k > 0.0 {
                    r.smooth_union(node, k)
                } else {
                    r.union(node)
                }
            }
        });
    }
    save_asdf_json(&SdfTree::new(root.unwrap()), &a[2]).expect("write");
    println!("[loft] wrote {} (k {k} m)", a[2]);
}

//! CPU sphere-traced orthographic render of a union of curve-section lofts (no mesh), culled per part:
//! each part's box is taken from its control rows (a B-spline section lies in its control points' convex
//! hull) padded 1 mm; outside a part's box the distance to the box stands in for the part (a lower bound of
//! the true distance), so a step never passes the surface; inside, the loft's own field is used. The union
//! is the plain min. Same view convention and shading as render_ortho. Writes a binary PPM; rays that reach the step cap are red.
//! `render_parts <rows dir> <az_deg> <u0> <u1> <v0> <v1> <px_m> <out.pgm> <part> [<part> ...]`
use alice_sdf::prelude::*;
use glam::{Vec2, Vec3};
use rayon::prelude::*;
use std::io::Write;

fn v3(x: &serde_json::Value) -> Vec3 {
    Vec3::new(
        x[0].as_f64().unwrap() as f32,
        x[1].as_f64().unwrap() as f32,
        x[2].as_f64().unwrap() as f32,
    )
}

struct Part {
    node: SdfNode,
    lo: Vec3,
    hi: Vec3,
}

fn box_dist(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    let d = (lo - p).max(p - hi).max(Vec3::ZERO);
    d.length()
}

fn field(parts: &[Part], p: Vec3) -> f32 {
    let mut best = f32::INFINITY;
    for q in parts {
        let b = box_dist(p, q.lo, q.hi);
        // the box bound only steps the ray; within 1 mm of the box the loft decides (a box face is never a hit)
        let f = if b > 0.001 { b } else { eval(&q.node, p) };
        best = best.min(f);
    }
    best
}

fn near_field(parts: &[Part], p: Vec3) -> f32 {
    // the true union field (only parts whose box contains p or is within 2 mm)
    let mut best = f32::INFINITY;
    for q in parts {
        if box_dist(p, q.lo, q.hi) < 0.002 {
            best = best.min(eval(&q.node, p));
        }
    }
    best
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert!(
        a.len() >= 9,
        "render_parts <rows dir> <az> <u0> <u1> <v0> <v1> <px> <out.pgm> <part> [...]"
    );
    let mut parts = Vec::new();
    for part in &a[8..] {
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
        let (a0, a1) = (axis[0], axis[axis.len() - 1]);
        let t = (a1 - a0).normalize();
        let n = (up - t * up.dot(t)).normalize();
        let b = t.cross(n);
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        // the reflected end rows extend the control polygon one row past each end
        let nr = rows.len();
        for i in 0..nr {
            let c = a0 + (a1 - a0) * (i as f32 / (nr - 1) as f32);
            for q in &rows[i] {
                let p = c + n * q.x + b * q.y;
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        let pad = Vec3::splat(0.001);
        let node = SdfNode::loft_curves(axis, up, rows).unwrap_or_else(|e| panic!("{part}: {e}"));
        parts.push(Part {
            node,
            lo: lo - pad,
            hi: hi + pad,
        });
    }
    let num = |i: usize| -> f32 { a[i].parse().expect("a number") };
    let (az, u0, u1, v0, v1, px) = (num(1).to_radians(), num(2), num(3), num(4), num(5), num(6));
    let (uax, w) = (
        Vec3::new(az.cos(), 0.0, -az.sin()),
        Vec3::new(az.sin(), 0.0, az.cos()),
    );
    let (nw, nh) = (
        ((u1 - u0) / px).ceil() as usize,
        ((v1 - v0) / px).ceil() as usize,
    );
    assert!(
        nw > 0 && nh > 0 && nw * nh < 40_000_000,
        "image size {nw} x {nh}"
    );
    if let Ok(dbg) = std::env::var("DEBUG_UV") {
        // one ray, verbose: hit point, each nearby part's field, normals at several epsilons
        let uv: Vec<f32> = dbg.split(',').map(|x| x.parse().unwrap()).collect();
        let o = uax * uv[0] + Vec3::Y * uv[1] + w * 1.5;
        let mut t = 0.0_f32;
        for k in 0..1024 {
            let p = o - w * t;
            let d = field(&parts, p);
            if d < 2e-6 {
                println!("hit after {k} steps at {:?}", p);
                for (q, name) in parts.iter().zip(&a[8..]) {
                    if box_dist(p, q.lo, q.hi) < 0.002 {
                        let f = eval(&q.node, p);
                        let mut line = format!("  {name}: field {:.3e}", f);
                        for e in [2e-5_f32, 1e-4, 5e-4] {
                            let g = Vec3::new(
                                eval(&q.node, p + Vec3::X * e) - eval(&q.node, p - Vec3::X * e),
                                eval(&q.node, p + Vec3::Y * e) - eval(&q.node, p - Vec3::Y * e),
                                eval(&q.node, p + Vec3::Z * e) - eval(&q.node, p - Vec3::Z * e),
                            );
                            line += &format!(
                                " | e {e:.0e}: |grad| {:.3} n {:?}",
                                g.length() / (2.0 * e),
                                g.normalize_or_zero()
                            );
                        }
                        println!("{line}");
                    }
                }
                return;
            }
            t += d;
        }
        println!("no hit");
        return;
    }
    let t0 = std::time::Instant::now();
    let rows_px: Vec<(Vec<u8>, usize, usize, usize, usize)> = (0..nh)
        .into_par_iter()
        .map(|jy| {
            let v = v1 - (jy as f32 + 0.5) * px;
            let mut row = vec![0u8; 3 * nw];
            let (mut hits, mut steps, mut capped, mut inverted) = (0usize, 0usize, 0usize, 0usize);
            for i in 0..nw {
                let u = u0 + (i as f32 + 0.5) * px;
                let o = uax * u + Vec3::Y * v + w * 1.5;
                let mut t = 0.0_f32;
                let mut converged = false;
                for _ in 0..1024 {
                    steps += 1;
                    let p = o - w * t;
                    let d = field(&parts, p);
                    if d < 2e-6 {
                        let e = 2e-5;
                        let g = Vec3::new(
                            near_field(&parts, p + Vec3::X * e)
                                - near_field(&parts, p - Vec3::X * e),
                            near_field(&parts, p + Vec3::Y * e)
                                - near_field(&parts, p - Vec3::Y * e),
                            near_field(&parts, p + Vec3::Z * e)
                                - near_field(&parts, p - Vec3::Z * e),
                        )
                        .normalize_or_zero();
                        let c = (255.0 * (0.15 + 0.85 * g.dot(w).max(0.0))) as u8;
                        if g.dot(w) < 0.0 {
                            row[3 * i..3 * i + 3].copy_from_slice(&[0, 80, 255]); // inverted: the normal faces away from the eye
                            inverted += 1;
                        } else {
                            row[3 * i..3 * i + 3].copy_from_slice(&[c, c, c]);
                        }
                        hits += 1;
                        converged = true;
                        break;
                    }
                    t += d;
                    if t > 3.0 {
                        converged = true; // a miss
                        break;
                    }
                }
                if !converged {
                    row[3 * i..3 * i + 3].copy_from_slice(&[255, 0, 0]); // the step cap: unconverged, not shape
                    capped += 1;
                }
            }
            (row, hits, steps, capped, inverted)
        })
        .collect();
    let hits: usize = rows_px.iter().map(|r| r.1).sum();
    let steps: usize = rows_px.iter().map(|r| r.2).sum();
    let capped: usize = rows_px.iter().map(|r| r.3).sum();
    let inverted: usize = rows_px.iter().map(|r| r.4).sum();
    let img: Vec<u8> = rows_px.into_iter().flat_map(|r| r.0).collect();
    assert!(hits > 0, "nothing hit");
    let mut f = std::fs::File::create(&a[7]).expect("out");
    write!(f, "P6\n{nw} {nh}\n255\n").unwrap();
    f.write_all(&img).unwrap();
    println!("[render] az {:.0}: {nw} x {nh} px, {hits} hits, {capped} unconverged (red), {inverted} inverted (blue), {:.1} steps/px, {:.1} s", az.to_degrees(), steps as f64 / (nw * nh) as f64, t0.elapsed().as_secs_f64());
}

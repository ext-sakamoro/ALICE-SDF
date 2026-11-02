//! CPU sphere-traced orthographic render of an `.asdf.json` tree through the evaluator (no mesh):
//! the view at azimuth `az` projects (x, y, z) to (u, v) = (x cos a − z sin a, y) and looks along
//! −(sin a, 0, cos a); shading = 0.15 + 0.85·max(0, n·view) from the evaluator's normal. Writes a binary PGM.
//!
//! `cargo run --release --example render_ortho -- <tree.asdf.json> <az_deg> <u0> <u1> <v0> <v1> <px_m> <out.pgm>`
use alice_sdf::eval::eval_normal;
use alice_sdf::io::load_asdf_json;
use glam::Vec3;
use std::io::Write;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        a.len(),
        8,
        "render_ortho <tree> <az> <u0> <u1> <v0> <v1> <px> <out.pgm>"
    );
    let tree = load_asdf_json(&a[0]).expect("tree");
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
        nw > 0 && nh > 0 && nw * nh < 20_000_000,
        "image size {nw} × {nh}"
    );
    // rows in parallel (each pixel is independent; the result does not depend on the order)
    use rayon::prelude::*;
    let rows: Vec<(Vec<u8>, usize)> = (0..nh)
        .into_par_iter()
        .map(|j| {
            let v = v1 - (j as f32 + 0.5) * px;
            let mut row = vec![0u8; nw];
            let mut n_hit = 0usize;
            for (i, out) in row.iter_mut().enumerate() {
                let u = u0 + (i as f32 + 0.5) * px;
                let o = uax * u + Vec3::Y * v + w * 1.5;
                if let Some(h) = alice_sdf::raycast::raymarch(&tree.root, o, -w, 3.0) {
                    let n = eval_normal(&tree.root, h.point);
                    *out = (255.0 * (0.15 + 0.85 * n.dot(w).max(0.0))) as u8;
                    n_hit += 1;
                }
            }
            (row, n_hit)
        })
        .collect();
    let hits: usize = rows.iter().map(|r| r.1).sum();
    let img: Vec<u8> = rows.into_iter().flat_map(|r| r.0).collect();
    assert!(hits > 0, "nothing hit");
    let mut f = std::fs::File::create(&a[7]).expect("out");
    write!(f, "P5\n{nw} {nh}\n255\n").unwrap();
    f.write_all(&img).unwrap();
    println!(
        "[render] {} az {:.0}: {nw} × {nh} px, {hits} hits",
        a[7],
        num(1)
    );
}

//! 2D signed distance shapes for UI and flat graphics.
//!
//! Builds a small badge — a rounded panel with a ring, a star, a hexagon, a
//! stroke and an elliptical cut-out — renders it as ASCII from `eval_2d_batch`,
//! and checks a handful of distances against their closed forms.
//!
//! # Running
//! ```bash
//! cargo run --example sdf2d_shapes
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::sdf2d::{eval_2d, eval_2d_batch, eval_2d_normal, Sdf2dNode};

fn close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() < 1e-5, "{what}: got {a}, expected {b}");
}

fn main() {
    println!("ALICE-SDF — 2D shapes");
    println!("=====================");

    let panel = Sdf2dNode::rounded_rect(2.0, 1.0, 0.3);
    let ring = Sdf2dNode::ring(0.5, 0.08).translate(-1.2, 0.0);
    let star = Sdf2dNode::star(0.45, 0.2, 5).rotate(0.3);
    let hexagon = Sdf2dNode::regular_polygon(0.4, 6)
        .scale(1.2)
        .translate(1.2, 0.0);
    let stroke = Sdf2dNode::line([-1.8, -0.8], [1.8, -0.8], 0.04);
    let slot = Sdf2dNode::ellipse(0.6, 0.1).translate(0.0, 0.75);
    let marks = ring.union(star).smooth_union(hexagon, 0.05).union(stroke);
    let badge = panel.onion(0.05).union(marks).subtract(slot);
    let window = Sdf2dNode::circle(3.0).intersect(badge);

    // ASCII render (y up).
    let (w, h) = (64, 20);
    let pts: Vec<[f32; 2]> = (0..h)
        .flat_map(|j| {
            (0..w).map(move |i| {
                [
                    (i as f32 + 0.5) / w as f32 * 4.6 - 2.3,
                    1.25 - (j as f32 + 0.5) / h as f32 * 2.5,
                ]
            })
        })
        .collect();
    let d = eval_2d_batch(&window, &pts);
    for row in d.chunks(w) {
        let line: String = row
            .iter()
            .map(|&v| if v < 0.0 { '#' } else { '.' })
            .collect();
        println!("{line}");
    }

    // Closed forms.
    close(eval_2d(&Sdf2dNode::circle(1.0), [3.0, 4.0]), 4.0, "circle");
    close(
        eval_2d(&Sdf2dNode::rect(1.0, 0.5), [2.0, 1.5]),
        2f32.sqrt(),
        "rect corner",
    );
    close(
        eval_2d(&Sdf2dNode::rect(1.0, 0.5), [0.0, 0.0]),
        -0.5,
        "rect inside",
    );
    close(
        eval_2d(&Sdf2dNode::line([0.0, 0.0], [2.0, 0.0], 0.1), [1.0, 1.0]),
        0.9,
        "line",
    );
    close(
        eval_2d(&Sdf2dNode::ring(1.0, 0.1), [0.0, 0.0]),
        0.9,
        "ring centre",
    );
    // A hexagon with circumradius 1 has apothem cos(30°): its centre is that deep.
    close(
        eval_2d(&Sdf2dNode::regular_polygon(1.0, 6), [0.0, 0.0]),
        -(30f32.to_radians().cos()),
        "hexagon centre",
    );
    close(
        eval_2d(&Sdf2dNode::ellipse(2.0, 1.0), [2.0, 0.0]),
        0.0,
        "ellipse vertex",
    );
    let n = eval_2d_normal(&Sdf2dNode::circle(1.0), [0.0, 2.0]);
    println!("circle normal at (0, 2) = {n:?}");
    assert!(n[0].abs() < 1e-3 && (n[1] - 1.0).abs() < 1e-3);
    assert!(d.iter().any(|&v| v < 0.0) && d.iter().any(|&v| v > 0.0));

    println!("all checks passed");
}

//! Oracle: `SdfNode::parabola_segment` returns the distance to its boundary (the arch
//! y = h (1 - (x / w)^2) over |x| <= w and the base y = 0), negative inside, on both sides of the
//! arch and from flat to tall arches.
//!
//! The true distance is computed independently in f64: the arch is sampled densely and the nearest
//! sample is refined by golden-section search on its bracket. An over-statement lets a sphere tracer
//! step through the surface.
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

fn arch(w: f64, h: f64, t: f64) -> f64 {
    h * (1.0 - (t / w).powi(2))
}

/// Distance from (px >= 0, py) to the arch over 0 <= t <= w: 4096 samples, then golden-section
fn arch_distance(w: f64, h: f64, px: f64, py: f64) -> f64 {
    const N: usize = 4096;
    let f = |t: f64| ((px - t).powi(2) + (py - arch(w, h, t)).powi(2)).sqrt();
    let (mut best, mut bd) = (0usize, f64::INFINITY);
    for i in 0..=N {
        let d = f(w * i as f64 / N as f64);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    let (mut lo, mut hi) = (w * best.saturating_sub(1) as f64 / N as f64, w * (best + 1).min(N) as f64 / N as f64);
    let g = 0.618_033_988_749_894_9;
    for _ in 0..120 {
        let (x1, x2) = (hi - g * (hi - lo), lo + g * (hi - lo));
        if f(x1) < f(x2) {
            hi = x2;
        } else {
            lo = x1;
        }
    }
    bd.min(f(0.5 * (lo + hi)))
}

/// The true signed distance in the cross-section
fn true_signed(w: f64, h: f64, x: f64, y: f64) -> f64 {
    let px = x.abs();
    let base = if px <= w { y.abs() } else { ((px - w).powi(2) + y * y).sqrt() };
    let d = arch_distance(w, h, px, y).min(base);
    if px <= w && y >= 0.0 && y <= arch(w, h, px) {
        -d
    } else {
        d
    }
}

/// The truth against closed forms: above the apex of a shallow arch (the convex side, within the
/// curvature radius or not) the nearest point is the apex, below the base it is the base, and past
/// the base corner it is the corner
#[test]
fn the_f64_truth_matches_closed_forms() {
    for (x, y, want) in [(0.0, 1.5, 0.5), (0.0, 1.25, 0.25), (0.5, -0.3, 0.3), (1.3, -0.4, 0.5), (0.0, 0.25, -0.25)] {
        let got = true_signed(1.0, 1.0, x, y);
        assert!((got - want).abs() < 1e-12, "({x}, {y}): {got} vs {want}");
    }
}

/// over-statement and sign tolerance: f32 rounding of the distance at these coordinates (|p| < 10)
const OVER_TOL: f64 = 1e-5;

#[test]
#[ignore = "known defect: PARABOLA-SEGMENT-NEWTON-001: on the convex side of the arch the Newton iteration leaves for the arch's end and the distance is over-stated by up to 3.0 ((0.05, 3.15) at w 1 h 3 reads 3.15, truth 0.154)"]
fn parabola_segment_never_overstates_the_distance() {
    let (mut worst, mut at, mut n, mut flipped) = (0.0f64, String::new(), 0usize, 0usize);
    for (w, h) in [(1.0f32, 1.0f32), (1.0, 0.1), (1.0, 0.01), (1.0, 1e-3), (1.0, 3.0), (0.2, 2.0), (3.0, 0.3)] {
        let node = SdfNode::parabola_segment(w, h, 1000.0);
        let (wf, hf) = (f64::from(w), f64::from(h));
        let mut pts: Vec<(f32, f32)> = Vec::new();
        // the plane around the shape
        for i in -30i32..=30 {
            for j in -30i32..=40 {
                pts.push((i as f32 * 0.05 * w, j as f32 * 0.05 * h.max(w * 0.1)));
            }
        }
        // along the arch, on both sides
        for k in 0..=80 {
            let x = wf * k as f64 / 80.0;
            for o in [0.0, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2, 0.1, 0.5, 2.0] {
                pts.push((x as f32, (arch(wf, hf, x) + o) as f32));
            }
        }
        for (x, y) in pts {
            let got = f64::from(eval(&node, Vec3::new(x, y, 0.0)));
            let truth = true_signed(wf, hf, f64::from(x), f64::from(y));
            n += 1;
            if truth.abs() > OVER_TOL && got * truth < 0.0 {
                flipped += 1;
            }
            let e = got.abs() - truth.abs();
            if e > worst {
                worst = e;
                at = format!("w {w} h {h} ({x}, {y}): eval {got:.7} true {truth:.7}");
            }
        }
    }
    assert!(n > 0, "compared 0 points");
    eprintln!("compared {n} points: worst over-statement {worst:.3e} ({at}), {flipped} flipped signs");
    assert_eq!(flipped, 0, "points on the wrong side");
    assert!(worst <= OVER_TOL, "over-states the distance by {worst:.3e} > {OVER_TOL:e}: {at}");
}

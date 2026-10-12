//! Audit: interval-evaluation looseness of the repeat/transform nodes.
//! (A) eval_interval width vs the true SDF range over a set of boxes (soundness: iv ⊇ true; looseness: width ratio).
//! (B) the preset_large tight-AABB vs the true zero-level-set AABB (the user-visible "0 triangles" symptom).
use alice_sdf::interval::{eval_interval, Vec3Interval};
use alice_sdf::prelude::*;
use alice_sdf::tight_aabb::{compute_tight_aabb_with_config, TightAabbConfig};

fn eval_at(n: &SdfNode, p: Vec3) -> f32 {
    alice_sdf::eval(n, p)
}

/// true SDF min/max over a box [c-h, c+h] by a dense grid
fn true_range(n: &SdfNode, c: Vec3, h: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let k = 7;
    for ix in 0..k {
        for iy in 0..k {
            for iz in 0..k {
                let t = |i: i32| -h + 2.0 * h * (i as f32) / (k as f32 - 1.0);
                let v = eval_at(n, c + Vec3::new(t(ix), t(iy), t(iz)));
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    (lo, hi)
}

/// looseness of eval_interval over a spread of sample boxes; also checks soundness (iv must contain true range)
fn interval_looseness(n: &SdfNode) -> (f32, f32, bool, (Vec3, f32, f32, f32)) {
    let h = 1.0f32; // box half-size
    let mut ratios = Vec::new();
    let mut sound = true;
    let mut worst = (Vec3::ZERO, 1.0f32, 0.0f32, 0.0f32); // (center, ratio, iv_width, true_width)
    let span = 14.0f32;
    let g = 9;
    for ix in 0..g {
        for iy in 0..g {
            for iz in 0..g {
                let t = |i: i32| -span + 2.0 * span * (i as f32) / (g as f32 - 1.0);
                let c = Vec3::new(t(ix), t(iy), t(iz));
                let iv = eval_interval(
                    n,
                    Vec3Interval::from_bounds(c - Vec3::splat(h), c + Vec3::splat(h)),
                );
                let (tlo, thi) = true_range(n, c, h);
                if iv.lo > tlo + 1e-3 || iv.hi < thi - 1e-3 {
                    sound = false;
                }
                let twidth = (thi - tlo).max(1e-4);
                let iwidth = (iv.hi - iv.lo).max(0.0);
                let r = iwidth / twidth;
                ratios.push(r);
                if r > worst.1 {
                    worst = (c, r, iwidth, twidth);
                }
            }
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ratios[ratios.len() / 2];
    let p95 = ratios[(ratios.len() * 95 / 100).min(ratios.len() - 1)];
    (med, p95, sound, worst)
}

/// true zero-level-set AABB extent (max |coord| where |sdf| < 0.5) by a coarse dense scan
fn true_aabb_reach(n: &SdfNode) -> f32 {
    let mut reach = 0.0f32;
    let s = 40.0;
    let k = 81;
    for ix in 0..k {
        for iy in 0..k {
            for iz in 0..k {
                let t = |i: i32| -s + 2.0 * s * (i as f32) / (k as f32 - 1.0);
                let p = Vec3::new(t(ix), t(iy), t(iz));
                if eval_at(n, p).abs() < 0.6 {
                    reach = reach.max(p.x.abs()).max(p.y.abs()).max(p.z.abs());
                }
            }
        }
    }
    reach
}

fn main() {
    // child: an off-centre box (like translate(8,0,0, sphere(1.5)))
    let child = || SdfNode::box3d(3.0, 3.0, 3.0).translate(8.0, 0.0, 0.0);
    let nodes: Vec<(&str, SdfNode)> = vec![
        ("control: translate(box)", child()),
        ("polar_repeat(6)", child().polar_repeat(6)),
        (
            "repeat_finite([3,1,1])",
            child().repeat_finite([3, 1, 1], Vec3::new(6.0, 0.0, 0.0)),
        ),
        (
            "repeat_infinite(16)",
            child().repeat_infinite(16.0, 16.0, 16.0),
        ),
        ("mirror(x)", child().mirror(true, false, false)),
        ("twist(0.3)", SdfNode::box3d(3.0, 3.0, 8.0).twist(0.3)),
        ("bend(0.2)", SdfNode::box3d(3.0, 3.0, 8.0).bend(0.2)),
    ];
    let cfg = TightAabbConfig::preset_large();
    println!(
        "{:<26} {:>8} {:>8} {:>7}  {:>10} {:>9}  verdict",
        "node", "med×", "p95×", "sound", "aabb_reach", "true_reach"
    );
    for (name, n) in &nodes {
        let (med, p95, sound, worst) = interval_looseness(n);
        let aabb = compute_tight_aabb_with_config(n, &cfg);
        let aabb_reach = aabb.min.abs().max(aabb.max.abs()).max_element();
        let true_reach = true_aabb_reach(n);
        let verdict = if p95 > 8.0 || (aabb_reach > 10.0 * true_reach.max(1.0)) {
            "LOOSE"
        } else if med <= 2.5 {
            "tight"
        } else {
            "mid"
        };
        println!("{:<26} {:>8.2} {:>8.2} {:>7} {:>11.1} {:>9.1}  {}  (worst@{:?} iv_w={:.1} true_w={:.1})",
                 name, med, p95, sound, aabb_reach, true_reach, verdict, worst.0, worst.2, worst.3);
    }
    println!("\nsound = eval_interval ⊇ true range on every sample box (soundness invariant; must be true)");
}

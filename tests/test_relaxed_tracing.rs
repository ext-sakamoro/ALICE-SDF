//! Over-relaxed sphere tracing against a brute-force oracle.
//!
//! `RaymarchConfig::fast()` (ω = 1.2) and `RaymarchConfig::relaxed(node)`
//! (ω = 1.6, L from `eval_lipschitz`) must find the same first intersection as
//! plain sphere tracing and as a fixed-step march refined by bisection. Until
//! 1.10.3 the overshoot check advanced instead of retreating, so a unit sphere
//! lost 76 % of its rays at ω = 1.6 (external review, 2026-09-15).
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;
use alice_sdf::raycast::{raymarch_compiled_with_config, raymarch_with_config, RaymarchConfig};

const MAX_DIST: f32 = 12.0;
/// Fixed-step oracle resolution (0.5 mm) — coarse scan, then bisection.
const ORACLE_STEP: f32 = 5e-4;
/// Hit-distance agreement between paths (default ε = 1e-4, oracle refined to 1e-6).
const T_TOL: f32 = 2e-3;

/// First surface crossing along the ray by exhaustive scan + bisection.
fn oracle_hit(node: &SdfNode, origin: Vec3, dir: Vec3) -> Option<f32> {
    let mut t = 0.0f32;
    let mut prev = eval(node, origin);
    if prev <= 0.0 {
        return Some(0.0);
    }
    while t < MAX_DIST {
        let next_t = t + ORACLE_STEP;
        let d = eval(node, origin + dir * next_t);
        if d <= 0.0 {
            let (mut lo, mut hi) = (t, next_t);
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if eval(node, origin + dir * mid) <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(0.5 * (lo + hi));
        }
        prev = d;
        t = next_t;
    }
    let _ = prev;
    None
}

/// 24 × 24 parallel rays from z = −6 towards +Z, covering the ±1.5 window.
fn rays() -> Vec<(Vec3, Vec3)> {
    let n = 24;
    let mut out = Vec::with_capacity(n * n);
    for iy in 0..n {
        for ix in 0..n {
            let f = |i: usize| -1.5 + 3.0 * (i as f32) / ((n - 1) as f32);
            out.push((Vec3::new(f(ix), f(iy), -6.0), Vec3::Z));
        }
    }
    out
}

fn shapes() -> Vec<(&'static str, SdfNode)> {
    vec![
        ("sphere", SdfNode::sphere(1.0)),
        ("torus", SdfNode::torus(0.7, 0.25)),
        ("box", SdfNode::box3d(0.8, 0.5, 0.6)),
        (
            "union_offset",
            SdfNode::sphere(0.6).union(SdfNode::box3d(0.4, 0.9, 0.3).translate(0.7, 0.2, 0.5)),
        ),
        (
            "smooth_subtract",
            SdfNode::box3d(0.9, 0.9, 0.9)
                .smooth_subtract(SdfNode::sphere(1.1).translate(0.5, 0.3, -0.4), 0.2),
        ),
        ("ellipsoid", SdfNode::ellipsoid(1.2, 0.6, 0.9)),
    ]
}

struct Stats {
    rays: usize,
    hit_mismatch: usize,
    t_mismatch: usize,
    worst_dt: f32,
    steps: u64,
}

/// A hit is declared at `|d| < ε`, so along a ray the reported `t` can be off
/// by `ε / |n·dir|` — unbounded at grazing incidence. Tolerance scales with it
/// (×8 absorbs bound-not-exact fields such as the ellipsoid).
fn t_tolerance(node: &SdfNode, p: Vec3, dir: Vec3, epsilon: f32) -> f32 {
    let n = normal(node, p, 1e-4);
    let cos = n.dot(dir).abs().max(0.01);
    T_TOL.max(8.0 * epsilon / cos)
}

/// Oracle hits for every ray, computed once per shape (the scan dominates runtime).
fn oracle_hits(node: &SdfNode, rays: &[(Vec3, Vec3)]) -> Vec<Option<f32>> {
    rays.iter().map(|&(o, d)| oracle_hit(node, o, d)).collect()
}

fn compare<F: Fn(Vec3, Vec3) -> Option<(f32, u32)>>(
    node: &SdfNode,
    rays: &[(Vec3, Vec3)],
    oracle: &[Option<f32>],
    epsilon: f32,
    march: F,
) -> Stats {
    let mut s = Stats {
        rays: 0,
        hit_mismatch: 0,
        t_mismatch: 0,
        worst_dt: 0.0,
        steps: 0,
    };
    for (&(o, d), &want) in rays.iter().zip(oracle) {
        s.rays += 1;
        let got = march(o, d);
        match (want, got) {
            (None, None) => {}
            (Some(tw), Some((tg, steps))) => {
                s.steps += u64::from(steps);
                let dt = (tw - tg).abs();
                s.worst_dt = s.worst_dt.max(dt);
                let tol = t_tolerance(node, o + d * tw, d, epsilon);
                if dt > tol {
                    s.t_mismatch += 1;
                    eprintln!(
                        "  ray o={o:?}: oracle t={tw:.5} got t={tg:.5} (Δ {dt:.2e} > tol {tol:.2e}, {steps} steps, d_at_got={:.2e})",
                        eval(node, o + d * tg)
                    );
                }
            }
            _ => s.hit_mismatch += 1,
        }
    }
    s
}

fn configs(node: &SdfNode) -> Vec<(&'static str, RaymarchConfig)> {
    let mut stress = RaymarchConfig::relaxed(node);
    stress.omega = 1.9;
    vec![
        ("default", RaymarchConfig::default()),
        ("fast", RaymarchConfig::fast()),
        ("relaxed", RaymarchConfig::relaxed(node)),
        ("relaxed_omega_1.9", stress),
    ]
}

#[test]
fn relaxed_tracing_matches_oracle_tree_and_compiled() {
    let mut failures = Vec::new();
    for (shape, node) in shapes() {
        let compiled = CompiledSdf::compile(&node);
        let rays = rays();
        let oracle = oracle_hits(&node, &rays);
        for (cfg_name, cfg) in configs(&node) {
            let tree = compare(&node, &rays, &oracle, cfg.epsilon, |o, d| {
                raymarch_with_config(&node, o, d, MAX_DIST, &cfg).map(|h| (h.distance, h.steps))
            });
            let comp = compare(&node, &rays, &oracle, cfg.epsilon, |o, d| {
                raymarch_compiled_with_config(&compiled, o, d, MAX_DIST, &cfg)
                    .map(|h| (h.distance, h.steps))
            });
            for (path, s) in [("tree", &tree), ("compiled", &comp)] {
                if s.hit_mismatch > 0 || s.t_mismatch > 0 {
                    failures.push(format!(
                        "{shape}/{cfg_name}/{path}: {}/{} hit-miss mismatches, {} t mismatches (worst Δt {:.2e})",
                        s.hit_mismatch, s.rays, s.t_mismatch, s.worst_dt
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The point of over-relaxation: fewer steps than plain tracing where plain
/// tracing crawls — rays at grazing incidence to a plane (Keinert et al. 2014,
/// fig. 3). Steps by `ω·d` stay covered while `sin θ ≤ (2 − ω) / ω`.
#[test]
fn relaxed_tracing_takes_fewer_steps_than_default() {
    let node = SdfNode::plane(Vec3::Y, 0.0);
    let mut grazing = Vec::new();
    for i in 0..10 {
        for j in 0..10 {
            let height = 0.2 + 0.08 * i as f32;
            let deg = 4.0 + 0.8 * j as f32;
            let (s, c) = deg.to_radians().sin_cos();
            grazing.push((Vec3::new(-3.0, height, 0.0), Vec3::new(c, -s, 0.0)));
        }
    }
    let eps = RaymarchConfig::default().epsilon;
    let oracle = oracle_hits(&node, &grazing);
    let default = compare(&node, &grazing, &oracle, eps, |o, d| {
        raymarch_with_config(&node, o, d, MAX_DIST, &RaymarchConfig::default())
            .map(|h| (h.distance, h.steps))
    });
    let relaxed = compare(&node, &grazing, &oracle, eps, |o, d| {
        raymarch_with_config(&node, o, d, MAX_DIST, &RaymarchConfig::relaxed(&node))
            .map(|h| (h.distance, h.steps))
    });
    assert_eq!(default.t_mismatch + relaxed.t_mismatch, 0);
    assert_eq!(default.hit_mismatch + relaxed.hit_mismatch, 0);
    assert!(
        relaxed.steps < default.steps,
        "relaxed {} steps vs default {} steps",
        relaxed.steps,
        default.steps
    );
}

/// The review's reproducer: a unit sphere, rays parallel to +Z at y ∈ {0, 0.3, 0.6, 0.9}
/// — every one of them must hit at the analytic distance.
#[test]
fn review_reproducer_unit_sphere_rows() {
    let node = SdfNode::sphere(1.0);
    let cfg = RaymarchConfig::relaxed(&node);
    for y in [0.0f32, 0.3, 0.6, 0.9] {
        let o = Vec3::new(0.0, y, -6.0);
        let hit = raymarch_with_config(&node, o, Vec3::Z, MAX_DIST, &cfg)
            .unwrap_or_else(|| panic!("y = {y}: relaxed tracing missed the unit sphere"));
        let expect = 6.0 - (1.0 - y * y).sqrt();
        assert!(
            (hit.distance - expect).abs() < T_TOL,
            "y = {y}: t = {} expected {expect}",
            hit.distance
        );
    }
}

#[cfg(feature = "jit")]
#[test]
fn relaxed_tracing_matches_oracle_jit() {
    use alice_sdf::compiled::jit::JitCompiledSdf;
    use alice_sdf::raycast::raymarch_jit_with_config;
    let mut failures = Vec::new();
    for (shape, node) in shapes() {
        let jit = JitCompiledSdf::compile(&node).expect("jit compile");
        let rays = rays();
        let oracle = oracle_hits(&node, &rays);
        for (cfg_name, cfg) in configs(&node) {
            let s = compare(&node, &rays, &oracle, cfg.epsilon, |o, d| {
                raymarch_jit_with_config(&jit, o, d, MAX_DIST, &cfg).map(|h| (h.distance, h.steps))
            });
            if s.hit_mismatch > 0 || s.t_mismatch > 0 {
                failures.push(format!(
                    "{shape}/{cfg_name}/jit: {}/{} hit-miss mismatches, {} t mismatches (worst Δt {:.2e})",
                    s.hit_mismatch, s.rays, s.t_mismatch, s.worst_dt
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

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
    if eval(node, origin) <= 0.0 {
        return Some(0.0);
    }
    let steps = (MAX_DIST / ORACLE_STEP) as usize;
    for i in 0..steps {
        let t = i as f32 * ORACLE_STEP;
        let next_t = t + ORACLE_STEP;
        let d = eval(node, origin + dir * next_t);
        if d <= 0.0 {
            let (mut lo, mut hi) = (t, next_t);
            for _ in 0..24 {
                let mid = f32::midpoint(lo, hi);
                if eval(node, origin + dir * mid) <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            return Some(f32::midpoint(lo, hi));
        }
    }
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
            let height = 0.08f32.mul_add(i as f32, 0.2);
            let deg = 0.8f32.mul_add(j as f32, 4.0);
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
        let expect = 6.0 - y.mul_add(-y, 1.0).sqrt();
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

/// TPMS surfaces are implicit trigonometric functions, not distance fields:
/// `eval_lipschitz` reports √3 … 7 for them (1.0 until 1.10.3, which made
/// Neovius and IWP unrenderable). With the bound applied every ray must hit
/// where the oracle does — plain L = 1 tracing is expected to miss.
#[test]
fn tpms_trace_correctly_with_lipschitz_bound() {
    use alice_sdf::interval::eval_lipschitz;
    let mut failures = Vec::new();
    for (name, node) in [
        ("gyroid", SdfNode::gyroid(2.0, 0.08)),
        ("schwarz_p", SdfNode::schwarz_p(2.0, 0.08)),
        ("neovius", SdfNode::neovius(2.0, 0.08)),
        ("iwp", SdfNode::iwp(2.0, 0.08)),
        ("lidinoid", SdfNode::lidinoid(2.0, 0.08)),
        ("pmy", SdfNode::pmy(2.0, 0.08)),
    ] {
        // TPMS fill space periodically: keep the rays that start outside.
        let rays: Vec<(Vec3, Vec3)> = rays()
            .into_iter()
            .filter(|&(o, _)| eval(&node, o) > 0.02)
            .collect();
        let oracle = oracle_hits(&node, &rays);
        let hits = oracle.iter().filter(|h| h.is_some()).count();
        assert!(
            hits > rays.len() / 2 && rays.len() > 100,
            "{name}: {hits} hits / {} rays",
            rays.len()
        );
        // Steps are d / L with L up to 7: give the budget the bound needs so
        // the comparison exercises the bound, not `max_steps`.
        let mut relaxed = RaymarchConfig::relaxed(&node);
        assert!(
            relaxed.omega > 1.0 && relaxed.lipschitz > 1.5,
            "{name}: {relaxed:?}"
        );
        relaxed.max_steps = 4096;
        let plain_bounded = RaymarchConfig {
            lipschitz: eval_lipschitz(&node),
            max_steps: 4096,
            ..Default::default()
        };
        for (cfg_name, cfg) in [("relaxed", relaxed), ("plain_L", plain_bounded)] {
            let s = compare(&node, &rays, &oracle, cfg.epsilon, |o, d| {
                raymarch_with_config(&node, o, d, MAX_DIST, &cfg).map(|h| (h.distance, h.steps))
            });
            if s.hit_mismatch > 0 || s.t_mismatch > 0 {
                failures.push(format!(
                    "{name}/{cfg_name}: {}/{} hit-miss mismatches, {} t mismatches (worst Δt {:.2e})",
                    s.hit_mismatch, s.rays, s.t_mismatch, s.worst_dt
                ));
            }
        }
        // Documented failure of the unbounded default: at least one ray of
        // Neovius / IWP is lost with L = 1 (this is what the bound fixes).
        if name == "neovius" || name == "iwp" {
            let unbounded = RaymarchConfig {
                max_steps: 4096,
                ..Default::default()
            };
            let s = compare(&node, &rays, &oracle, 1e-4, |o, d| {
                raymarch_with_config(&node, o, d, MAX_DIST, &unbounded)
                    .map(|h| (h.distance, h.steps))
            });
            assert!(
                s.hit_mismatch > 0,
                "{name}: L = 1 tracing unexpectedly hit every ray"
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The config-less entry points (`raymarch`, `raymarch_compiled`,
/// `raymarch_simd_8` with the default config, JIT) apply the field's
/// Lipschitz bound themselves — the tree's `eval_lipschitz` or the value
/// recorded at compile time — so a TPMS traces correctly without the caller
/// knowing it is not a distance field. Every reported hit must be the
/// oracle's first crossing; a `None` is only tolerated when the default
/// budget (256 · L steps since 2.1.0) runs out, never as a skipped
/// surface.
#[test]
fn default_entry_points_apply_the_lipschitz_bound() {
    use alice_sdf::compiled::Vec3x8;
    use alice_sdf::raycast::{raymarch, raymarch_compiled, raymarch_simd_8};
    use wide::f32x8;
    let mut failures = Vec::new();
    for (name, node) in [
        ("gyroid", SdfNode::gyroid(2.0, 0.08)),
        ("neovius", SdfNode::neovius(2.0, 0.08)),
        ("iwp", SdfNode::iwp(2.0, 0.08)),
    ] {
        let rays: Vec<(Vec3, Vec3)> = rays()
            .into_iter()
            .filter(|&(o, _)| eval(&node, o) > 0.02)
            .collect();
        let oracle = oracle_hits(&node, &rays);
        let compiled = CompiledSdf::compile(&node);
        assert!(
            compiled.lipschitz() > 1.5,
            "{name}: compiled bound {}",
            compiled.lipschitz()
        );

        // SIMD: 8 rays per packet, padded with copies of the last ray
        let mut simd_hits: Vec<Option<(f32, u32)>> = Vec::with_capacity(rays.len());
        for chunk in rays.chunks(8) {
            let mut ox = [0.0f32; 8];
            let mut oy = [0.0f32; 8];
            let mut oz = [0.0f32; 8];
            let mut dx = [0.0f32; 8];
            let mut dy = [0.0f32; 8];
            let mut dz = [0.0f32; 8];
            for i in 0..8 {
                let (o, d) = chunk[i.min(chunk.len() - 1)];
                ox[i] = o.x;
                oy[i] = o.y;
                oz[i] = o.z;
                dx[i] = d.x;
                dy[i] = d.y;
                dz[i] = d.z;
            }
            let res = raymarch_simd_8(
                &compiled,
                Vec3x8 {
                    x: f32x8::from(ox),
                    y: f32x8::from(oy),
                    z: f32x8::from(oz),
                },
                Vec3x8 {
                    x: f32x8::from(dx),
                    y: f32x8::from(dy),
                    z: f32x8::from(dz),
                },
                MAX_DIST,
                &RaymarchConfig::default(),
            );
            for r in res.iter().take(chunk.len()) {
                simd_hits.push(r.map(|(t, _, steps)| (t, steps)));
            }
        }

        // The Cranelift JIT has no codegen for TPMS yet (`UnsupportedNode`):
        // cover it where it compiles, skip loudly otherwise.
        #[cfg(feature = "jit")]
        let jit = alice_sdf::compiled::jit::JitCompiledSdf::compile(&node)
            .map_err(|e| eprintln!("{name}: JIT skipped ({e:?})"))
            .ok();
        type March<'a> = Box<dyn Fn(usize, Vec3, Vec3) -> Option<(f32, u32)> + 'a>;
        #[allow(unused_mut)]
        let mut paths: Vec<(&str, March)> = vec![
            (
                "raymarch",
                Box::new(|_, o, d| raymarch(&node, o, d, MAX_DIST).map(|h| (h.distance, h.steps))),
            ),
            (
                "raymarch_compiled",
                Box::new(|_, o, d| {
                    raymarch_compiled(&compiled, o, d, MAX_DIST).map(|h| (h.distance, h.steps))
                }),
            ),
            ("raymarch_simd_8", Box::new(|i, _, _| simd_hits[i])),
        ];
        #[cfg(feature = "jit")]
        if let Some(jit) = &jit {
            paths.push((
                "raymarch_jit",
                Box::new(move |_, o, d| {
                    alice_sdf::raycast::raymarch_jit(jit, o, d, MAX_DIST)
                        .map(|h| (h.distance, h.steps))
                }),
            ));
        }
        for (path, march) in &paths {
            let (mut false_hit, mut budget_miss, mut agree) = (0usize, 0usize, 0usize);
            for (i, (&(o, d), &want)) in rays.iter().zip(&oracle).enumerate() {
                match (want, march(i, o, d)) {
                    (Some(tw), Some((tg, _))) => {
                        if (tw - tg).abs() > t_tolerance(&node, o + d * tw, d, 1e-4) {
                            false_hit += 1;
                        } else {
                            agree += 1;
                        }
                    }
                    (None, Some(_)) => false_hit += 1,
                    (Some(_), None) => budget_miss += 1,
                    (None, None) => agree += 1,
                }
            }
            eprintln!("{name}/{path}: agree {agree}, budget miss {budget_miss}, false hit {false_hit} / {}", rays.len());
            if false_hit > 0 || budget_miss * 10 > rays.len() {
                failures.push(format!(
                    "{name}/{path}: {false_hit} false hits, {budget_miss} misses of {}",
                    rays.len()
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Laws that `eval_lipschitz` declares non-Lipschitz (`INFINITY`): the
/// marchers step by `d` there, which is sound only when the field is still
/// a distance bound. Domain repetition of a child that is symmetric inside
/// its cell *is* a distance field — every ray must agree with the oracle.
/// An off-centre child makes the repeated field jump on the cell borders;
/// that miss rate is pinned as a documented number (regression detector),
/// not claimed as correct. Taper returns a distance bound since 2.0
/// (`real::taper_bound`: Jacobian ball + cone ∩ slab), so it is in the
/// exact set — including rays that cross the singular plane `y = 1/f`,
/// which the Jacobian term alone would report as a surface.
#[test]
fn non_lipschitz_laws_default_tracing() {
    use alice_sdf::raycast::raymarch;
    let mut failures = Vec::new();
    let exact: Vec<(&str, SdfNode)> = vec![
        (
            "repeat_infinite_centred",
            SdfNode::sphere(0.35).repeat_infinite(1.2, 1.2, 1.2),
        ),
        (
            "repeat_finite_centred",
            SdfNode::box3d(0.5, 0.5, 0.5).repeat_finite([3, 3, 3], Vec3::splat(1.1)),
        ),
        (
            "polar_repeat_centred_child",
            SdfNode::sphere(0.3)
                .translate(0.9, 0.0, 0.0)
                .polar_repeat(6),
        ),
        // 2.0: was pinned at 6 % (28 / 576 rays missed on the shrinking side)
        ("taper_box", SdfNode::box3d(1.0, 1.0, 1.0).taper(0.3)),
        // singular plane y = 1/f = 2 inside the ray box (±3): crossing rays
        // must pass, not hit a phantom
        ("taper_sphere_f05", SdfNode::sphere(0.8).taper(0.5)),
        ("taper_torus_neg", SdfNode::torus(0.7, 0.2).taper(-0.6)),
        (
            "taper_offset_box",
            SdfNode::box3d(0.6, 1.2, 0.8)
                .translate(0.3, 0.2, -0.2)
                .taper(0.35),
        ),
    ];
    for (name, node) in &exact {
        let rays: Vec<(Vec3, Vec3)> = rays()
            .into_iter()
            .filter(|&(o, _)| eval(node, o) > 0.02)
            .collect();
        let oracle = oracle_hits(node, &rays);
        let s = compare(node, &rays, &oracle, 1e-4, |o, d| {
            raymarch(node, o, d, MAX_DIST).map(|h| (h.distance, h.steps))
        });
        if s.hit_mismatch > 0 || s.t_mismatch > 0 {
            failures.push(format!(
                "{name}: {} hit-miss, {} t mismatches / {}",
                s.hit_mismatch, s.t_mismatch, s.rays
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Pinned miss rates — documented ceilings, not correctness claims;
    // lowering one is progress, raising one is a regression.
    let pinned: Vec<(&str, SdfNode, f32)> = vec![(
        "repeat_infinite_offset",
        SdfNode::sphere(0.3)
            .translate(0.45, 0.0, 0.0)
            .repeat_infinite(1.2, 1.2, 1.2),
        0.10,
    )];
    for (name, node, ceiling) in &pinned {
        let rays: Vec<(Vec3, Vec3)> = rays()
            .into_iter()
            .filter(|&(o, _)| eval(node, o) > 0.02)
            .collect();
        let oracle = oracle_hits(node, &rays);
        let s = compare(node, &rays, &oracle, 1e-4, |o, d| {
            raymarch(node, o, d, MAX_DIST).map(|h| (h.distance, h.steps))
        });
        let rate = (s.hit_mismatch + s.t_mismatch) as f32 / s.rays as f32;
        eprintln!(
            "{name}: {} hit-miss + {} t mismatches / {} rays ({:.1} %)",
            s.hit_mismatch,
            s.t_mismatch,
            s.rays,
            100.0 * rate
        );
        assert!(
            rate <= *ceiling,
            "{name}: miss rate {rate:.3} exceeds the pinned {ceiling}"
        );
    }
}

/// The taper map is singular on the plane `y = 1/f`; the Jacobian bound
/// alone goes to 0 there and every ray crossing the plane away from the
/// shape would stop on it. The cone ∩ slab term from the child's reach is
/// what prevents that — pinned here by evaluating the same node with the
/// reach removed (`[INFINITY; 2]`, the pre-2.0 / unbounded-child form).
#[test]
fn taper_singular_plane_is_not_a_surface() {
    use alice_sdf::raycast::raymarch;
    use std::sync::Arc;
    let child = SdfNode::sphere(0.8);
    let with_reach = child.clone().taper(0.5);
    let without_reach = SdfNode::Taper {
        child: Arc::new(child),
        factor: 0.5,
        reach: [f32::INFINITY; 2],
    };
    let (mut phantom_without, mut rays_total) = (0, 0);
    for i in 0..24 {
        for j in 0..24 {
            let x = -3.0 + 6.0 * i as f32 / 23.0;
            let z = -3.0 + 6.0 * j as f32 / 23.0;
            if x.hypot(z) < 1.2 {
                continue; // near the axis the cone is close: not a phantom test
            }
            rays_total += 1;
            let o = Vec3::new(x, -3.0, z);
            let hit = raymarch(&with_reach, o, Vec3::Y, MAX_DIST);
            assert!(
                hit.is_none(),
                "ray from {o:?} up through y = 2 hit a phantom at t = {:?}",
                hit.map(|h| h.distance)
            );
            if raymarch(&without_reach, o, Vec3::Y, MAX_DIST).is_some() {
                phantom_without += 1;
            }
        }
    }
    eprintln!("taper singular plane: {phantom_without} / {rays_total} rays stop on it without the reach, 0 with");
    // the test is only meaningful if the plane really is a phantom without the bound
    assert!(
        phantom_without > rays_total / 2,
        "{phantom_without} / {rays_total}"
    );
}

/// Random-direction rays through thin TPMS shells with the config-less
/// `raymarch` (the 9/16 self-review measured 8.3 % misses on a gyroid with
/// the 1.x defaults). Every miss was budget exhaustion: steps are `d / √3`
/// and 128 of them do not carry a ray across several empty cells, and a ray
/// grazing the shell creeps by ≈ ε per step. Since 2.1.0 the budget scales
/// with the bound and the default is 256; what is left (≲ 0.1 %) is the
/// grazing creep, which `max_steps = 4096` resolves completely — pinned
/// here as the documented ceiling.
///
/// A hit is "a point within ε of the surface": a ray that grazes the shell
/// at `f = 7e-6` stops there while the fixed-step oracle reports the first
/// sign change (which can be units further along); such hits are accepted
/// when `|f| ≤ ε` at the reported point.
#[test]
fn tpms_default_budget_random_rays() {
    use alice_sdf::interval::eval_lipschitz;
    use alice_sdf::raycast::raymarch;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    };
    let random_rays: Vec<(Vec3, Vec3)> = (0..3000)
        .map(|_| {
            let o = Vec3::new(
                next().mul_add(6.0, -3.0),
                next().mul_add(6.0, -3.0),
                next().mul_add(6.0, -3.0),
            );
            let d = Vec3::new(
                next().mul_add(2.0, -1.0),
                next().mul_add(2.0, -1.0),
                next().mul_add(2.0, -1.0),
            )
            .normalize();
            (o, d)
        })
        .collect();
    // (hit-miss mismatches, t mismatches) of a marcher against the oracle
    let judge = |node: &SdfNode,
                 rays: &[(Vec3, Vec3)],
                 oracle: &[Option<f32>],
                 march: &dyn Fn(Vec3, Vec3) -> Option<f32>| {
        let (mut hm, mut tm) = (0usize, 0usize);
        for (&(o, d), want) in rays.iter().zip(oracle) {
            match (want, march(o, d)) {
                (Some(tw), Some(tg)) => {
                    let on_surface = eval(node, o + d * tg).abs() <= 1e-4;
                    if (tw - tg).abs() > t_tolerance(node, o + d * tg, d, 1e-4) && !on_surface {
                        tm += 1;
                    }
                }
                (None, None) => {}
                _ => hm += 1,
            }
        }
        (hm, tm)
    };
    let mut failures = Vec::new();
    for (name, node) in [
        ("gyroid s1 t0.1", SdfNode::gyroid(1.0, 0.1)),
        ("gyroid s2 t0.1", SdfNode::gyroid(2.0, 0.1)),
        ("gyroid s2 t0.08", SdfNode::gyroid(2.0, 0.08)),
    ] {
        let rays: Vec<(Vec3, Vec3)> = random_rays
            .iter()
            .copied()
            .filter(|&(o, _)| eval(&node, o) > 0.02)
            .collect();
        let oracle = oracle_hits(&node, &rays);
        let default_cfg = RaymarchConfig::default().with_bound(eval_lipschitz(&node));
        let (hm, tm) = judge(&node, &rays, &oracle, &|o, d| {
            raymarch(&node, o, d, MAX_DIST).map(|h| h.distance)
        });
        let budget = RaymarchConfig {
            max_steps: 4096,
            ..default_cfg
        };
        let (hm_big, tm_big) = judge(&node, &rays, &oracle, &|o, d| {
            raymarch_with_config(&node, o, d, MAX_DIST, &budget).map(|h| h.distance)
        });
        eprintln!(
            "{name}: default (max_steps {}) {hm} hit-miss + {tm} t / {} rays; 4096 steps {hm_big} hit-miss + {tm_big} t",
            default_cfg.max_steps,
            rays.len()
        );
        if hm as f32 > 0.001 * rays.len() as f32 || tm > 0 {
            failures.push(format!(
                "{name}: default budget {hm} hit-miss + {tm} t / {} rays",
                rays.len()
            ));
        }
        if hm_big > 0 || tm_big > 0 {
            failures.push(format!(
                "{name}: 4096 steps still {hm_big} hit-miss + {tm_big} t"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

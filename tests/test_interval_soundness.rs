//! Integration tests: interval arithmetic soundness beyond point sampling
//!
//! `interval_eval_contains_point_values` (test_evaluator_opcode_parity.rs)
//! already samples 8 corners + 8 interior points per box over the whole
//! corpus. That catches an enclosure that is grossly too tight, but an
//! enclosure whose violation sits between the samples survives it — and
//! `alice-lol`'s law verifier turns `eval_interval` into a *proof* ("no
//! surface in this box"), so an enclosure that is too tight anywhere becomes a
//! law that passes without holding.
//!
//! Two oracles that do not depend on where the samples happen to land:
//!
//! 1. **Octree inclusion isotonicity** — for a box `B` split into its 8 octree
//!    children `B'`, sound interval arithmetic must satisfy
//!    `eval_interval(B') ⊆ eval_interval(B)`: the child encloses a subset of
//!    the values. A child interval that pokes outside its parent proves that
//!    at least one of the two is wrong, with no witness point needed. This is
//!    exactly the refinement the verifier performs (`box_children` +
//!    `BALL_PROBE_DEPTH`), so a violation here is a violation on its hot path.
//!    (For the `ia_lipschitz` arms the property holds because an octree child
//!    satisfies `|c' − c| + ρ' = ρ`.)
//!
//! 2. **Gradient-guided adversarial extrema** — instead of sampling blindly,
//!    walk downhill (and uphill) inside the box with `eval_gradient` and
//!    assert the extremum found lies inside the interval. This searches for
//!    the witness that random sampling misses.
//!
//! Author: Moroya Sakamoto

#![allow(clippy::cast_precision_loss)]

mod common;

use alice_sdf::eval::gradient::eval_gradient;
use alice_sdf::interval::{eval_interval, Vec3Interval};
use alice_sdf::prelude::*;
use common::corpus::corpus;

/// Deterministic LCG in [0, 1).
fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed;
    move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Absolute slack for a bound of magnitude `v`.
///
/// **Why this is not ulp-level.** The interval ops round outward by one ulp
/// (3.1.1), and with that the *enclosure* check below measures **exactly 0**
/// drift over the corpus — the enclosure is rigorous with respect to the
/// evaluated expression. The *inclusion isotonicity* check still measures
/// ~1.3·10⁻⁴ relative, and that residue is **not a rounding artefact**:
///
/// - Arms built on a centre sample (`ia_lipschitz`: `f(centre) ± L·ρ`) are
///   sound for each box on its own, but they are isotonic only when the child
///   box's bounding ball sits inside the parent's. Octree children satisfy
///   that (`|c′ − c| + ρ′ = ρ`); a box **reshaped by a transform** does not.
///   The worst case measured is `revolution` (a `Circle2D` under
///   `length_xz` reshaping), where the child's derived box is a general
///   sub-box of the parent's derived box.
/// - So isotonicity is a *heuristic* invariant here, not a theorem. It is
///   still worth checking: it is what caught the stairs arms assuming a
///   Lipschitz constant that `glsl_mod`'s jump does not give them (fixed in
///   3.1.1 by evaluating the stairs law op by op on intervals).
///
/// The envelope stays at 10⁻³ relative: loose enough for the centre-sample
/// residue, tight enough for a structurally wrong enclosure (the ones fixed in
/// 3.1.1 were 5.8·10⁻³, 1.8·10⁻² and 2.1·10⁻¹). Tightening it further needs
/// the centre-sample arms replaced by interval-composed ones, not a smaller
/// number here.
const ENVELOPE: f32 = 1e-3;

fn tol(v: f32) -> f32 {
    ENVELOPE * v.abs().max(1.0)
}

/// Relative overshoot of `value` past `bound` (0 when inside).
fn overshoot(excess: f32, bound: f32) -> f32 {
    if excess <= 0.0 {
        0.0
    } else {
        excess / bound.abs().max(1.0)
    }
}

/// The 8 octree children of `[lo, hi]`.
fn octree_children(lo: Vec3, hi: Vec3) -> [(Vec3, Vec3); 8] {
    let mid = (lo + hi) * 0.5;
    let pick = |k: usize, axis: usize| -> (f32, f32) {
        let (l, m, h) = match axis {
            0 => (lo.x, mid.x, hi.x),
            1 => (lo.y, mid.y, hi.y),
            _ => (lo.z, mid.z, hi.z),
        };
        if k & (1 << axis) == 0 {
            (l, m)
        } else {
            (m, h)
        }
    };
    let mut out = [(Vec3::ZERO, Vec3::ZERO); 8];
    for (k, slot) in out.iter_mut().enumerate() {
        let (x0, x1) = pick(k, 0);
        let (y0, y1) = pick(k, 1);
        let (z0, z1) = pick(k, 2);
        *slot = (Vec3::new(x0, y0, z0), Vec3::new(x1, y1, z1));
    }
    out
}

/// Random boxes: centre in ±2.5, half-size 0.05..0.8 (the verifier's cell
/// sizes at `resolution` 8 over a ±5 AABB, and its refinements).
fn random_boxes(seed: u64, n: usize) -> Vec<(Vec3, Vec3)> {
    let mut rnd = lcg(seed);
    (0..n)
        .map(|_| {
            let c = Vec3::new(
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
                rnd().mul_add(5.0, -2.5),
            );
            let h = Vec3::new(
                rnd().mul_add(0.75, 0.05),
                rnd().mul_add(0.75, 0.05),
                rnd().mul_add(0.75, 0.05),
            );
            (c - h, c + h)
        })
        .collect()
}

/// `eval_interval` on an octree child must stay inside the parent's interval.
#[test]
fn interval_is_inclusion_isotonic_over_corpus() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut worst_drift = 0.0f32;
    let mut worst_at = String::from("(none)");
    for (name, node) in corpus() {
        for (lo, hi) in random_boxes(0x150_7000 ^ name.len() as u64, 40) {
            let parent = eval_interval(&node, Vec3Interval::from_bounds(lo, hi));
            if parent.lo.is_nan() || parent.hi.is_nan() {
                failures.push(format!(
                    "{name}: NaN parent interval {parent:?} on {lo:?}..{hi:?}"
                ));
                continue;
            }
            for (clo, chi) in octree_children(lo, hi) {
                let child = eval_interval(&node, Vec3Interval::from_bounds(clo, chi));
                checked += 1;
                if child.lo.is_nan() || child.hi.is_nan() {
                    failures.push(format!(
                        "{name}: NaN child interval {child:?} on {clo:?}..{chi:?}"
                    ));
                    continue;
                }
                let drift = overshoot(parent.lo - child.lo, parent.lo)
                    .max(overshoot(child.hi - parent.hi, parent.hi));
                if drift > worst_drift {
                    worst_drift = drift;
                    worst_at = format!(
                        "{name}: child [{}, {}] vs parent [{}, {}] (box {clo:?}..{chi:?})",
                        child.lo, child.hi, parent.lo, parent.hi
                    );
                }
                if drift > ENVELOPE {
                    failures.push(format!(
                        "{name}: child [{}, {}] escapes parent [{}, {}] by {drift:.3e} relative (box {clo:?}..{chi:?} ⊂ {lo:?}..{hi:?})",
                        child.lo, child.hi, parent.lo, parent.hi
                    ));
                }
            }
        }
    }
    eprintln!("inclusion isotonicity: {checked} parent/child interval pairs");
    eprintln!("worst inclusion drift: {worst_drift:.3e} relative — {worst_at}");
    assert!(checked > 30_000, "corpus too small: {checked}");
    assert!(
        failures.is_empty(),
        "{} inclusion-isotonicity failures:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Walk inside `[lo, hi]` following ∓∇f and return the extremum found.
/// `sign` = −1 minimises, +1 maximises.
fn guided_extremum(node: &SdfNode, lo: Vec3, hi: Vec3, sign: f32, seed: u64) -> (f32, Vec3) {
    let mut rnd = lcg(seed);
    let half = (hi - lo) * 0.5;
    let mut starts = vec![(lo + hi) * 0.5];
    for _ in 0..4 {
        starts.push(Vec3::new(
            rnd().mul_add(hi.x - lo.x, lo.x),
            rnd().mul_add(hi.y - lo.y, lo.y),
            rnd().mul_add(hi.z - lo.z, lo.z),
        ));
    }
    let mut best = f32::INFINITY * -sign;
    let mut best_at = starts[0];
    for start in starts {
        let mut p = start;
        let mut value = eval(node, p);
        if !value.is_finite() {
            continue;
        }
        let mut step = half.min_element().max(1e-4) * 0.7;
        for _ in 0..48 {
            let g = eval_gradient(node, p);
            if !g.is_finite() || g.length_squared() < 1e-20 {
                break;
            }
            let dir = g.normalize() * sign;
            let q = (p + dir * step).clamp(lo, hi);
            let fq = eval(node, q);
            if fq.is_finite() && (fq - value) * sign > 0.0 {
                p = q;
                value = fq;
            } else {
                step *= 0.6;
                if step < 1e-6 {
                    break;
                }
            }
        }
        if (value - best) * sign > 0.0 {
            best = value;
            best_at = p;
        }
    }
    (best, best_at)
}

/// The interval must contain the extrema an adversarial search can reach.
#[test]
fn interval_encloses_gradient_guided_extrema_over_corpus() {
    let mut failures = Vec::new();
    let mut checked = 0usize;
    let mut worst_drift = 0.0f32;
    let mut worst_at = String::from("(none)");
    for (name, node) in corpus() {
        for (i, (lo, hi)) in random_boxes(0x9e37_0001 ^ name.len() as u64, 40)
            .into_iter()
            .enumerate()
        {
            let iv = eval_interval(&node, Vec3Interval::from_bounds(lo, hi));
            if iv.lo == f32::NEG_INFINITY && iv.hi == f32::INFINITY {
                continue; // no claim to violate
            }
            let seed = 0xabcd_0001 ^ ((i as u64) << 8) ^ name.len() as u64;
            let (lowest, lo_at) = guided_extremum(&node, lo, hi, -1.0, seed);
            let (highest, hi_at) = guided_extremum(&node, lo, hi, 1.0, seed ^ 0x5555);
            checked += 1;
            if lowest.is_finite() {
                let drift = overshoot(iv.lo - lowest, lowest);
                if drift > worst_drift {
                    worst_drift = drift;
                    worst_at = format!("{name}: minimum {lowest} vs interval lo {}", iv.lo);
                }
                if lowest < iv.lo - tol(lowest) {
                    failures.push(format!(
                        "{name}: minimum {lowest} at {lo_at:?} below interval lo {} (box {lo:?}..{hi:?})",
                        iv.lo
                    ));
                }
            }
            if highest.is_finite() {
                let drift = overshoot(highest - iv.hi, highest);
                if drift > worst_drift {
                    worst_drift = drift;
                    worst_at = format!("{name}: maximum {highest} vs interval hi {}", iv.hi);
                }
                if highest > iv.hi + tol(highest) {
                    failures.push(format!(
                        "{name}: maximum {highest} at {hi_at:?} above interval hi {} (box {lo:?}..{hi:?})",
                        iv.hi
                    ));
                }
            }
        }
    }
    eprintln!("gradient-guided extrema: {checked} boxes searched");
    eprintln!("worst enclosure drift: {worst_drift:.3e} relative — {worst_at}");
    assert!(checked > 3_000, "corpus too small: {checked}");
    assert!(
        failures.is_empty(),
        "{} enclosure failures found by guided search:\n{}",
        failures.len(),
        failures
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

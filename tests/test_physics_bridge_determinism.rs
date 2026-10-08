//! Determinism and closed-form oracles for the ALICE-SDF ↔ ALICE-Physics boundary.
//!
//! Both crates advertise bit-exactness — ALICE-Physics is a `Fix128` lockstep
//! engine, ALICE-SDF has been cross-platform bit-exact since 3.1.0
//! (`alice-det-math`) — but the seam between them was not measured by anything.
//! `SdfCollider` holds a `Box<dyn SdfField>` whose methods take and return
//! `f32`, so a query runs
//!
//! ```text
//! Vec3Fix --world_to_local--> f32 --SDF eval--> f32 --* scale_f32--> compare
//! ```
//!
//! and the `f32` stretch in the middle is exactly the shape the lockstep
//! discipline warns about: a non-`Fix128` layer inside a `Fix128` engine is
//! where the chain breaks, regardless of how exact the two ends are.
//!
//! These tests do not assume the seam is broken. They pin the properties that
//! have to hold for the two determinism claims to compose, so that a change
//! that breaks the composition fails here instead of surfacing as a desync.
//!
//! Expectation sources, never the bridge itself:
//!
//! 1. **closed forms** — a unit sphere's signed distance is `|p| - r`, and its
//!    outward normal is `p / |p|`. Points are chosen so the answer is exactly
//!    representable in `f32`, which makes the assertions bit-exact rather than
//!    tolerance-based.
//! 2. **the native evaluator** — `eval_compiled`. The bridge documents itself
//!    as a thin wrapper, so "thin" is testable to the bit.
//! 3. **repetition** — a deterministic query returns the same bits every time.
//!    `Fix128` is compared as its `(hi, lo)` words, not through `to_f32`.
//!
//! ⚠️ `distance_and_normal` is **not** overridden by this crate any more (see
//! the comment where the override used to be in `src/physics_bridge.rs`): the
//! trait default `(distance, normal)` applies, so both calls are the same
//! evaluation and are required to agree to the bit. The override that was there
//! until 2026-09-30 approximated the distance from the normal's four samples and
//! was larger by 4.768e-7 to 7.451e-7 at every point measured.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![cfg(feature = "physics")]

use alice_physics::sdf_collider::{collide_point_sdf, SdfCollider, SdfField};
use alice_physics::{Fix128, QuatFix, Vec3Fix};
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::physics_bridge::sdf_to_physics_field;
use alice_sdf::prelude::*;

// ───────────────────────────── helpers ─────────────────────────────

/// `Fix128` as its two raw words. `to_f32` would hide a difference below the
/// `f32` mantissa, which is the difference a lockstep desync is made of.
const fn words(v: Fix128) -> (i64, u64) {
    (v.hi, v.lo)
}

const fn vec_words(v: Vec3Fix) -> [(i64, u64); 3] {
    [words(v.x), words(v.y), words(v.z)]
}

/// Every field of a contact as raw words, in a fixed order.
fn contact_words(c: &alice_physics::collider::Contact) -> Vec<(i64, u64)> {
    let mut out = vec![words(c.depth)];
    out.extend(vec_words(c.normal));
    out.extend(vec_words(c.point_a));
    out.extend(vec_words(c.point_b));
    out
}

fn unit_sphere_collider() -> SdfCollider {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.0));
    SdfCollider::new_static(Box::new(field), Vec3Fix::ZERO, QuatFix::IDENTITY)
}

/// Points whose exact signed distance to the unit sphere is representable in
/// `f32`: each `|p|` is an exact power of two or a small integer.
const EXACT_POINTS: [([f32; 3], f32); 6] = [
    ([0.0, 0.0, 0.0], -1.0),  // centre
    ([2.0, 0.0, 0.0], 1.0),   // |p| = 2
    ([0.0, -4.0, 0.0], 3.0),  // |p| = 4, negative axis
    ([0.0, 0.0, 0.5], -0.5),  // inside
    ([1.0, 0.0, 0.0], 0.0),   // exactly on the surface
    ([0.0, 0.0, 16.0], 15.0), // far field
];

// ─────────────────────── determinism at the seam ───────────────────────

/// The same point queried repeatedly must produce the same bits. A cache, a
/// lazily initialised table or an uninitialised scratch buffer anywhere on the
/// path would show up as a difference between the first call and a later one.
#[test]
fn repeated_queries_return_bit_identical_results() {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.0).union(SdfNode::box3d(1.4, 1.4, 1.4)));
    let probes = [
        (0.3_f32, -0.9_f32, 0.42_f32),
        (1.7, 0.0, -0.25),
        (-2.5, 3.1, 0.8),
        (0.0, 0.0, 0.0),
    ];
    for (x, y, z) in probes {
        let first_d = field.distance(x, y, z).to_bits();
        let first_n = field.normal(x, y, z);
        let first_n = (
            first_n.0.to_bits(),
            first_n.1.to_bits(),
            first_n.2.to_bits(),
        );
        for i in 0..64 {
            let d = field.distance(x, y, z).to_bits();
            assert_eq!(d, first_d, "distance drifted on call {i} at ({x},{y},{z})");
            let n = field.normal(x, y, z);
            let n = (n.0.to_bits(), n.1.to_bits(), n.2.to_bits());
            assert_eq!(n, first_n, "normal drifted on call {i} at ({x},{y},{z})");
        }
    }
}

/// Interleaving other points must not change what a given point answers. This
/// is the property a hidden last-query cache would break, and it is the one
/// that matters for lockstep: two clients visit points in different orders.
#[test]
fn query_order_does_not_change_any_answer() {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.3).twist(0.6));
    let pts: Vec<(f32, f32, f32)> = (0..32)
        .map(|i| {
            let t = i as f32 * 0.37;
            (t.sin() * 2.0, t.cos() * 1.5, t * 0.1 - 1.0)
        })
        .collect();

    let forward: Vec<u32> = pts
        .iter()
        .map(|&(x, y, z)| field.distance(x, y, z).to_bits())
        .collect();
    let mut backward: Vec<u32> = pts
        .iter()
        .rev()
        .map(|&(x, y, z)| field.distance(x, y, z).to_bits())
        .collect();
    backward.reverse();
    assert_eq!(forward, backward, "answers depend on visit order");
}

/// A collider rebuilt from the same node must behave identically. Construction
/// order, allocation addresses and the `Arc` identity must not reach the value.
#[test]
fn a_rebuilt_collider_gives_bit_identical_contacts() {
    let a = unit_sphere_collider();
    let b = unit_sphere_collider();
    let mut compared = 0;
    for (p, _) in EXACT_POINTS {
        let pf = Vec3Fix::from_f32(p[0], p[1], p[2]);
        match (collide_point_sdf(pf, &a), collide_point_sdf(pf, &b)) {
            (Some(ca), Some(cb)) => {
                assert_eq!(
                    contact_words(&ca),
                    contact_words(&cb),
                    "contact differs at {p:?}"
                );
                compared += 1;
            }
            (None, None) => {}
            (x, y) => panic!(
                "collision disagrees at {p:?}: {:?} vs {:?}",
                x.is_some(),
                y.is_some()
            ),
        }
    }
    assert!(
        compared >= 2,
        "only {compared} contacts compared; the scene stopped colliding"
    );
}

/// Repeated collision queries must be bit-identical in every `Contact` field,
/// including the ones `SimulationChecksum` does not look at.
#[test]
fn repeated_collisions_are_bit_identical_in_every_contact_field() {
    let collider = unit_sphere_collider();
    let inside = Vec3Fix::from_f32(0.25, -0.5, 0.125);
    let first = collide_point_sdf(inside, &collider).expect("point inside must collide");
    let first = contact_words(&first);
    assert!(
        first.len() == 10,
        "Contact grew a field; extend contact_words"
    );
    for i in 0..32 {
        let c = collide_point_sdf(inside, &collider).expect("still colliding");
        assert_eq!(contact_words(&c), first, "contact drifted on call {i}");
    }
}

// ─────────────────────── the bridge is thin (parity) ───────────────────────

/// `CompiledSdfField::distance` documents itself as `eval_compiled`. Bit-for-bit.
#[test]
fn bridge_distance_equals_the_native_evaluator_bit_for_bit() {
    let node = SdfNode::sphere(1.0)
        .union(SdfNode::sphere(0.6).translate(0.9, 0.0, 0.0))
        .round(0.05);
    let compiled = CompiledSdf::compile(&node);
    let field = sdf_to_physics_field(&node);
    let mut checked = 0;
    for i in 0..24 {
        let t = i as f32 * 0.41;
        let p = Vec3::new(t.cos() * 2.2, t.sin() * 1.7, t * 0.13 - 1.1);
        let via_bridge = field.distance(p.x, p.y, p.z).to_bits();
        let native = eval_compiled(&compiled, p).to_bits();
        assert_eq!(via_bridge, native, "bridge is not thin at {p:?}");
        checked += 1;
    }
    assert_eq!(checked, 24);
}

/// `distance_and_normal` must return the same distance as `distance`, to the bit.
///
/// This is the contract the trait itself sets: `SdfField::distance_and_normal`
/// has a default implementation of `(self.distance(..), self.normal(..))` and
/// documents the point of overriding it as computing "both efficiently" — a
/// cost property, not a different answer.
///
/// Until 2026-09-30 this crate overrode it with
/// `eval_compiled_distance_and_normal`, which reuses the normal's four
/// tetrahedral samples and takes their *average* as the centre distance. That
/// answered a different question: measured over a sphere-union-box field at 24
/// points, the override's distance exceeded `distance()` at **every** point, by
/// 4.768e-7 to 7.451e-7 — one-sided, so a bias rather than rounding.
///
/// It was not academic. `alice_physics::sdf_adaptive` (1.4.0) caches the
/// distance that `distance_and_normal` returns, while `collide_point_sdf`
/// derives contact depth from `distance`, so the two disagreed about the same
/// point. The override is gone; the trait default now applies and the two are
/// the same evaluation.
///
/// ⚠️ Re-adding an override that only approximates the distance puts this test
/// red. That is the intent — see the comment where the override used to be in
/// `src/physics_bridge.rs`.
#[test]
fn distance_and_normal_returns_the_exact_distance() {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.0).union(SdfNode::box3d(1.0, 1.0, 1.0)));
    let mut compared = 0;
    for i in 0..24 {
        let t = i as f32 * 0.29;
        let (x, y, z) = (t.cos() * 1.9, t.sin() * 1.4, t * 0.11 - 0.9);
        let standalone = field.distance(x, y, z);
        let (combined, n) = field.distance_and_normal(x, y, z);
        assert_eq!(
            combined.to_bits(),
            standalone.to_bits(),
            "the combined call returned a different distance at ({x},{y},{z}): \
             {combined} ({:#x}) vs {standalone} ({:#x})",
            combined.to_bits(),
            standalone.to_bits()
        );
        // The normal must still be the real one — an implementation that
        // returned the exact distance and a zero normal would pass the line
        // above.
        let len = (n.0 * n.0 + n.1 * n.1 + n.2 * n.2).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-4,
            "the combined call's normal is not unit at ({x},{y},{z}): |n| = {len}"
        );
        let separate = field.normal(x, y, z);
        assert_eq!(
            (n.0.to_bits(), n.1.to_bits(), n.2.to_bits()),
            (
                separate.0.to_bits(),
                separate.1.to_bits(),
                separate.2.to_bits()
            ),
            "the combined call's normal differs from `normal()` at ({x},{y},{z})"
        );
        compared += 1;
    }
    assert_eq!(
        compared, 24,
        "the sample shrank; the pin is weaker than it reads"
    );
}

// ─────────────────────── closed forms at the seam ───────────────────────

/// `|p| - r` for the points chosen to be exactly representable. This replaces a
/// 0.01 tolerance: the sphere's distance is exact, so anything but equality is
/// a defect in the seam, not rounding.
#[test]
fn distance_matches_the_closed_form_exactly_at_representable_points() {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.0));
    for (p, want) in EXACT_POINTS {
        let got = field.distance(p[0], p[1], p[2]);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "|p| - r at {p:?}: got {got} ({:#x}) want {want} ({:#x})",
            got.to_bits(),
            want.to_bits()
        );
    }
}

/// The outward normal of a sphere is `p / |p|`. Both stencils are held to it.
///
/// This also pins the *direction convention*: `SdfField::normal` is documented
/// as pointing away from the surface. A sign flip keeps the vector a unit
/// normal, so length and orthogonality checks cannot see it — the dot product
/// with the analytic outward direction is what catches it. (An existing
/// tangent-basis helper in this workspace was left-handed for exactly that
/// reason: its test checked orthogonality and not handedness.)
#[test]
fn both_normal_stencils_match_the_analytic_outward_normal() {
    let field = sdf_to_physics_field(&SdfNode::sphere(1.0));
    // Away from the surface both stencils are well conditioned; on the surface
    // the central difference straddles it, so stay off it.
    let pts = [
        [2.0_f32, 0.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, 0.0, 4.0],
        [1.5, 1.5, 1.5],
        [-2.0, 1.0, -0.5],
        [0.4, 0.0, 0.0], // inside: the gradient still points outward
    ];
    for p in pts {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let want = [p[0] / len, p[1] / len, p[2] / len];

        for (label, n) in [
            ("normal", {
                let n = field.normal(p[0], p[1], p[2]);
                [n.0, n.1, n.2]
            }),
            ("distance_and_normal", {
                let (_, n) = field.distance_and_normal(p[0], p[1], p[2]);
                [n.0, n.1, n.2]
            }),
        ] {
            let nl = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (nl - 1.0).abs() < 1e-4,
                "{label} at {p:?} is not unit: |n| = {nl}"
            );
            let dot = n[0] * want[0] + n[1] * want[1] + n[2] * want[2];
            assert!(
                dot > 0.999,
                "{label} at {p:?} does not point outward: n = {n:?}, want {want:?}, dot = {dot}"
            );
            for k in 0..3 {
                assert!(
                    (n[k] - want[k]).abs() < 1e-3,
                    "{label} at {p:?} component {k}: got {} want {}",
                    n[k],
                    want[k]
                );
            }
        }
    }
}

/// `collide_point_sdf` treats `world_dist >= 0.0` as not colliding, so a point
/// exactly on the surface is outside. The same boundary contract holds for the
/// interval sign in this workspace (`d == 0` is the outside side); pinning it
/// here keeps the two from drifting apart.
#[test]
fn a_point_exactly_on_the_surface_does_not_collide() {
    let collider = unit_sphere_collider();
    for p in [[1.0_f32, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]] {
        assert_eq!(
            field_distance_on_surface(p),
            0.0_f32.to_bits(),
            "the point {p:?} is not exactly on the surface; the case is vacuous"
        );
        let c = collide_point_sdf(Vec3Fix::from_f32(p[0], p[1], p[2]), &collider);
        assert!(c.is_none(), "a surface point collided at {p:?}");
    }
    // Positive control: just inside must collide, or the test above proves
    // nothing (a collider that never collides would pass it).
    let just_inside = Vec3Fix::from_f32(0.9375, 0.0, 0.0);
    assert!(
        collide_point_sdf(just_inside, &collider).is_some(),
        "the control point inside did not collide; the assertion above is vacuous"
    );
}

fn field_distance_on_surface(p: [f32; 3]) -> u32 {
    sdf_to_physics_field(&SdfNode::sphere(1.0))
        .distance(p[0], p[1], p[2])
        .to_bits()
}

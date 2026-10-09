//! Oracles for the contact → crater → shape update loop between the physics
//! world and `live_sdf` (alice-physics 2.1 `last_step_sdf_contacts` and
//! `SdfField::generation`).
//!
//! The collider is placed off the origin, rotated and scaled, so the
//! world-to-shape conversion is exercised. Expected values come from closed
//! forms evaluated in f64 (the inverse pose, the impact radius rule, the
//! crater depth), never from the code under test. Each test fails when it
//! compared nothing.
#![cfg(feature = "physics")]

use alice_physics::sdf_collider::SdfCollider;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, Vec3Fix};
use alice_sdf::live_sdf::{FracturePolicy, ImpactContact, LiveSdf};
use alice_sdf::SdfNode;
use glam::{DQuat, DVec3, Vec3};

/// A 5 × 2 × 5 slab whose top face is the local plane y = 0.
fn slab() -> SdfNode {
    SdfNode::box3d(5.0, 2.0, 5.0).translate(0.0, -1.0, 0.0)
}

const POS: [f64; 3] = [5.0, 1.0, -2.0];
const YAW_DEG: f64 = 30.0;
const SCALE: f64 = 2.0;
const RHO: f32 = 0.1;

fn pose() -> (Vec3Fix, QuatFix, Fix128) {
    let (s, c) = alice_det_math::sin_cos((YAW_DEG.to_radians() / 2.0) as f32);
    let rotation = QuatFix::new(
        Fix128::ZERO,
        Fix128::from_f32(s),
        Fix128::ZERO,
        Fix128::from_f32(c),
    );
    (
        Vec3Fix::from_f32(POS[0] as f32, POS[1] as f32, POS[2] as f32),
        rotation,
        Fix128::from_f32(SCALE as f32),
    )
}

/// World with the slab placed at `pose()`, registered as collider 0 and as a
/// participant, and a second (plain) slab far away as collider 1.
fn world(live: &LiveSdf) -> PhysicsWorld {
    let mut world = PhysicsWorld::new(PhysicsConfig {
        substeps: 8,
        ..PhysicsConfig::default()
    });
    let (p, r, s) = pose();
    world.add_sdf_collider(SdfCollider::new_static(Box::new(live.clone()), p, r).with_scale(s));
    world
        .add_participant(Box::new(live.clone()))
        .expect("register the shape as a participant");
    let other = LiveSdf::new(slab());
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(other),
        Vec3Fix::from_f32(-40.0, 0.0, 0.0),
        QuatFix::IDENTITY,
    ));
    world.set_sdf_collision_radius(Fix128::from_f32(RHO));
    world
}

/// World point → shape point, the inverse of translate · rotate · scale, in f64.
fn to_local(p: DVec3) -> DVec3 {
    let q = DQuat::from_rotation_y(YAW_DEG.to_radians());
    q.inverse() * (p - DVec3::from(POS)) / SCALE
}

fn v3(v: Vec3Fix) -> DVec3 {
    let (x, y, z) = v.to_f32();
    DVec3::new(f64::from(x), f64::from(y), f64::from(z))
}

const K: f32 = 0.05;
const R_MIN: f32 = 0.4;
const R_MAX: f32 = 0.6;

fn policy() -> FracturePolicy {
    FracturePolicy::new(Fix128::from_f32(1.0), K, R_MIN, R_MAX).for_collider(0)
}

/// Drop one ball from `height` above the slab's world top at world `(x, z)`
/// and step until the first step that records a contact with collider 0.
fn drop_until_contact(world: &mut PhysicsWorld, x: f32, z: f32, height: f32) -> usize {
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(x, POS[1] as f32 + height, z),
        Fix128::ONE,
    ));
    let dt = Fix128::from_f32(1.0 / 60.0);
    for step in 0..600 {
        world.step(dt);
        if world
            .last_step_sdf_contacts()
            .iter()
            .any(|c| c.collider_index == 0)
        {
            return step;
        }
    }
    panic!("no contact with collider 0 in 600 steps");
}

/// oracle: every recorded contact converted by `ImpactContact::from_world`
/// equals the inverse pose applied in f64 (point), the inverse rotation
/// (normal) and the division by the scale (depth, speed).
#[test]
fn world_contacts_are_converted_into_the_shapes_frame() {
    let live = LiveSdf::new(slab());
    let mut w = world(&live);
    drop_until_contact(&mut w, 5.5, -2.3, 3.0);
    let mut compared = 0;
    for c in w
        .last_step_sdf_contacts()
        .iter()
        .filter(|c| c.collider_index == 0)
    {
        let local = ImpactContact::from_world(c, &w.sdf_colliders[0]);
        let want = to_local(v3(c.point));
        assert!(
            (v3(local.point) - want).length() < 1e-5,
            "point {:?} vs {want:?}",
            v3(local.point)
        );
        // The slab's top normal is +y in both frames (the rotation is a yaw).
        assert!((v3(local.normal) - DVec3::Y).length() < 1e-5);
        let ds = c.depth.to_f32() as f64 / SCALE;
        assert!((f64::from(local.depth.to_f32()) - ds).abs() < 1e-6);
        let vs = c.approach_speed.to_f32() as f64 / SCALE;
        assert!((f64::from(local.approach_speed.to_f32()) - vs).abs() < 1e-6);
        assert_eq!(local.body_index, c.body_index);
        assert_eq!(local.substep as usize, c.substep);
        compared += 1;
    }
    assert!(compared > 0, "compared nothing");
}

/// oracle: one fast impact carves one crater, centred on the converted
/// contact point, radius `clamp(v_local · K, R_MIN, R_MAX)` (the impact rule,
/// re-derived here); the next steps wake the resting ball and it sinks to
/// the crater floor, `R · SCALE` below the flat surface in world units.
#[test]
fn a_fast_impact_carves_and_the_ball_sinks_into_its_crater() {
    let live = LiveSdf::new(slab());
    let mut w = world(&live);
    let g0 = live.generation();
    drop_until_contact(&mut w, 5.5, -2.3, 3.0);

    let first = *w
        .last_step_sdf_contacts()
        .iter()
        .filter(|c| c.collider_index == 0)
        .max_by(|a, b| a.approach_speed.cmp(&b.approach_speed))
        .expect("a contact");
    let v_local = first.approach_speed.to_f32() as f64 / SCALE;
    assert!(
        v_local > 1.0,
        "the impact is above the threshold: {v_local}"
    );
    let r = (v_local * f64::from(K)).clamp(f64::from(R_MIN), f64::from(R_MAX));
    let centre = to_local(v3(first.point));

    let records = w
        .last_step_sdf_contacts()
        .iter()
        .filter(|c| c.collider_index == 0 && c.body_index == 0)
        .count();
    assert!(records >= 1);
    let carved = live.apply_world_contacts(&policy(), &w, 0);
    assert_eq!(
        carved, 1,
        "one body, one impact per step, whatever the number of substep records ({records})"
    );
    assert!(live.generation() > g0, "the carve raised the generation");
    // The deepest point of the crater of the fastest contact is r below its
    // centre on the surface.
    let floor = Vec3::new(
        centre.x as f32,
        (centre.y - r) as f32 + 1e-3,
        centre.z as f32,
    );
    let d_floor = live.eval(floor);
    assert!(
        d_floor.abs() < 2e-3,
        "the crater floor of radius {r} is at the surface: {d_floor}"
    );

    // Contacts with collider 1 never carve this shape, slow ones never carve.
    let crater_count = live.crater_count();
    assert_eq!(live.apply_world_contacts(&policy(), &w, 1), 0);
    assert_eq!(live.crater_count(), crater_count);

    let dt = Fix128::from_f32(1.0 / 60.0);
    for _ in 0..360 {
        w.step(dt);
    }
    let y = w.bodies[0].position.y.to_f32() as f64;
    let top = POS[1];
    let sink = top + f64::from(RHO) - y;
    let r_world = r * SCALE;
    // A ball much smaller than the crater rests on its floor: its centre
    // is `R·SCALE − ρ` below the flat surface, so it sank `R·SCALE` below
    // its resting height on the flat surface.
    assert!(
        (sink - r_world).abs() < 0.02 * r_world,
        "the ball sank {sink}, closed form {r_world}"
    );
}

/// oracle: a slow contact (below the threshold) carves nothing and leaves
/// the generation unchanged, so resting bodies are not woken.
#[test]
fn a_slow_contact_carves_nothing() {
    let live = LiveSdf::new(slab());
    let mut w = world(&live);
    drop_until_contact(&mut w, 5.5, -2.3, 0.01);
    let g = live.generation();
    let slow = FracturePolicy::new(Fix128::from_f32(100.0), K, R_MIN, R_MAX).for_collider(0);
    let n = w
        .last_step_sdf_contacts()
        .iter()
        .filter(|c| c.collider_index == 0)
        .count();
    assert!(n > 0, "compared nothing");
    assert_eq!(live.apply_world_contacts(&slow, &w, 0), 0);
    assert_eq!(live.generation(), g);
    assert_eq!(
        live.apply_world_contacts(&slow, &w, 99),
        0,
        "out-of-range collider"
    );
}

/// oracle: a body that stays in contact over a step is pushed out in every
/// substep, so the step records several contacts of the same body; that is
/// one impact and carves one crater (the count of distinct bodies), even
/// with a policy that accepts every contact.
#[test]
fn several_substep_records_of_one_body_carve_one_crater() {
    let live = LiveSdf::new(slab());
    let mut w = world(&live);
    w.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(5.5, POS[1] as f32 + 3.0, -2.3),
        Fix128::ONE,
    ));
    let dt = Fix128::from_f32(1.0 / 60.0);
    let mut records = 0;
    for _ in 0..400 {
        w.step(dt);
        records = w
            .last_step_sdf_contacts()
            .iter()
            .filter(|c| c.collider_index == 0 && c.body_index == 0)
            .count();
        if records > 1 {
            break;
        }
    }
    assert!(
        records > 1,
        "a step with several records of one body: {records}"
    );
    let any = FracturePolicy::new(Fix128::from_f32(-1.0), K, R_MIN, R_MAX).for_collider(0);
    assert_eq!(live.apply_world_contacts(&any, &w, 0), 1);
    assert_eq!(live.crater_count(), 1);
}

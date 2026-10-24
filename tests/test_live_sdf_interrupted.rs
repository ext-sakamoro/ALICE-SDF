//! A `LiveSdf` whose modifier panics in the middle of an update or a restore does not keep going on a
//! half-applied state. After the panic is caught, the shape is either exactly what it was before the
//! interrupted call, or it refuses to be read (the snapshot writer panics instead of handing out a mix).
//! Today the lock's poison is ignored and the shape keeps the modifiers updated before the panic next
//! to the ones after it.
//!
//! oracle: the shape's own snapshot payload (`Participant::write_state`) taken before the call.

#![cfg(feature = "physics")]

use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::world_participant::{
    ObservationSink, Participant, ParticipantFault, ParticipantKind, StateError, SubstepCtx,
};
use alice_physics::Fix128;
use alice_sdf::live_sdf::{LiveModifier, LiveSdf};
use alice_sdf::SdfNode;
use glam::Vec3;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A modifier with a step counter that offsets the distance by the count, and that can be told to
/// panic on a given update (before counting it) or when its state is restored.
struct Counter {
    n: u32,
    panic_on_update: Option<u32>,
    panic_on_restore: bool,
}

impl Counter {
    fn new() -> Self {
        Self {
            n: 0,
            panic_on_update: None,
            panic_on_restore: false,
        }
    }
}

impl PhysicsModifier for Counter {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d + 0.01 * self.n as f32
    }
    fn update(&mut self, _dt: f32) {
        assert!(
            self.panic_on_update != Some(self.n + 1),
            "modifier panics during the update"
        );
        self.n += 1;
    }
    fn name(&self) -> &str {
        "counter"
    }
}

impl Participant for Counter {
    fn kind(&self) -> ParticipantKind {
        ParticipantKind::new(0x434e_5452)
    }
    fn substep(&mut self, _: &mut SubstepCtx<'_>, _: Fix128) -> Result<(), ParticipantFault> {
        Ok(())
    }
    fn observe(&self, _: &mut ObservationSink) {}
    fn write_state(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.n.to_le_bytes());
    }
    fn check_state(&self, bytes: &[u8]) -> Result<(), StateError> {
        if bytes.len() == 4 {
            Ok(())
        } else {
            Err(StateError::InvalidValue)
        }
    }
    fn read_state(&mut self, bytes: &[u8]) {
        assert!(
            !self.panic_on_restore,
            "modifier panics while its state is restored"
        );
        self.n = u32::from_le_bytes(bytes.try_into().unwrap());
    }
}

impl LiveModifier for Counter {}

fn slab() -> SdfNode {
    SdfNode::box3d(5.0, 2.0, 5.0).translate(0.0, -1.0, 0.0)
}

fn payload(live: &LiveSdf) -> Vec<u8> {
    let mut out = Vec::new();
    Participant::write_state(live, &mut out);
    out
}

/// After the interrupted call: the shape is the one from before, or it refuses to hand out its state.
fn unchanged_or_refused(live: &LiveSdf, before: &[u8], what: &str) {
    match catch_unwind(AssertUnwindSafe(|| payload(live))) {
        Err(_) => {} // refused
        Ok(after) => assert_eq!(
            after, before,
            "{what}: the shape keeps a half-applied state (some modifiers changed, some not)"
        ),
    }
}

#[test]
fn a_modifier_panic_during_update_leaves_no_half_updated_shape() {
    let live = LiveSdf::new(slab());
    live.add_modifier(Counter::new());
    live.add_modifier(Counter {
        panic_on_update: Some(1),
        ..Counter::new()
    });
    let before = payload(&live);
    let probe = Vec3::new(0.0, 0.0, 0.0);
    let d_before = live.eval(probe);

    let caught = catch_unwind(AssertUnwindSafe(|| live.update(1.0 / 30.0)));
    assert!(caught.is_err(), "the second modifier panicked");
    unchanged_or_refused(&live, &before, "update");
    if let Ok(d) = catch_unwind(AssertUnwindSafe(|| live.eval(probe))) {
        assert_eq!(
            d.to_bits(),
            d_before.to_bits(),
            "update: the distance moved by half an update"
        );
    }
}

#[test]
fn a_modifier_panic_during_restore_leaves_no_half_restored_shape() {
    // the source shape: a crater and both modifiers stepped
    let source = LiveSdf::new(slab());
    source.add_modifier(Counter::new());
    source.add_modifier(Counter::new());
    source
        .subtract_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5)
        .expect("crater");
    for _ in 0..3 {
        source.update(1.0 / 30.0);
    }
    let blob = payload(&source);

    // the target: same kinds, the second modifier panics in its restore
    let mut target = LiveSdf::new(slab());
    target.add_modifier(Counter::new());
    target.add_modifier(Counter {
        panic_on_restore: true,
        ..Counter::new()
    });
    assert_eq!(
        Participant::check_state(&target, &blob),
        Ok(()),
        "the payload is acceptable"
    );
    let before = payload(&target);
    assert_ne!(before, blob);

    let caught = catch_unwind(AssertUnwindSafe(|| {
        Participant::read_state(&mut target, &blob)
    }));
    assert!(
        caught.is_err(),
        "the second modifier panicked in its restore"
    );
    unchanged_or_refused(&target, &before, "restore");
    if let Ok(n) = catch_unwind(AssertUnwindSafe(|| target.crater_count())) {
        assert_eq!(
            n, 0,
            "restore: the craters of the snapshot were applied without its modifiers"
        );
    }
}

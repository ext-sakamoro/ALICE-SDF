//! Keyframe animation of an SDF: tracks, timeline, `AnimatedSdf`, `morph`.
//!
//! Builds a timeline that slides a unit sphere along +X while it grows,
//! samples it at a few times and checks every value against the closed form
//! of the interpolation it uses (linear lerp, step hold, cubic Hermite, and
//! the three loop modes).
//!
//! # Running
//! ```bash
//! cargo run --example animation_timeline
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

fn close(a: f32, b: f32, tol: f32, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: got {a}, expected {b}");
}

fn main() {
    println!("ALICE-SDF — keyframe animation");
    println!("==============================");

    // translate.x: linear 0 → 2 over one second.
    let mut slide = Track::new("translate.x");
    slide.add_keyframe(Keyframe::new(1.0, 2.0)); // inserted out of order on purpose
    slide.add_keyframe(Keyframe::new(0.0, 0.0));

    // scale: step 1 → 2 at t = 0.5 (holds 1.0 until then).
    let mut grow = Track::new("scale");
    grow.add_keyframe(Keyframe::step(0.0, 1.0));
    grow.add_keyframe(Keyframe::step(0.5, 2.0));
    grow.add_keyframe(Keyframe::step(1.0, 2.0));

    let mut timeline = Timeline::new("slide-and-grow");
    timeline.add_track(slide);
    timeline.add_track(grow);
    println!(
        "timeline '{}' duration = {}",
        timeline.name,
        timeline.duration()
    );
    close(timeline.duration(), 1.0, 0.0, "duration");

    for &(t, x, s) in &[
        (0.0_f32, 0.0_f32, 1.0_f32),
        (0.25, 0.5, 1.0),
        (0.75, 1.5, 2.0),
    ] {
        let values = timeline.evaluate(t);
        println!("  t={t:.2}: {values:?}");
        close(
            timeline.get_value("translate.x", t).unwrap(),
            x,
            1e-6,
            "translate.x",
        );
        close(timeline.get_value("scale", t).unwrap(), s, 0.0, "scale");
    }
    assert!(timeline.get_value("missing", 0.0).is_none());

    // The animated sphere: at t = 0.75 it is centred at x = 1.5 with radius 2.
    let animated = AnimatedSdf::new(SdfNode::sphere(1.0), timeline);
    let params = animated.evaluate_params(0.75);
    let node = animated.evaluate_at(0.75);
    let p = Vec3::new(5.0, 0.0, 0.0);
    let d = eval(&node, p);
    println!(
        "AnimatedSdf @0.75: translate_x={}, scale={}, d({p}) = {d}",
        params.translate_x, params.scale
    );
    close(params.translate_x, 1.5, 1e-6, "params.translate_x");
    close(params.scale, 2.0, 0.0, "params.scale");
    close(d, (5.0 - 1.5) - 2.0, 1e-5, "animated distance");

    // Cubic Hermite at the midpoint: 0.5(v0+v1) + 0.125·span·(m0 − m1).
    let mut ease = Track::new("ease");
    ease.add_keyframe(Keyframe::cubic(0.0, 0.0, 4.0, 1.0));
    ease.add_keyframe(Keyframe::new(2.0, 1.0));
    let mid = ease.evaluate(1.0);
    println!("cubic Hermite midpoint = {mid}");
    close(
        mid,
        0.5 + 0.125 * 2.0 * (4.0 - 1.0),
        1e-6,
        "hermite midpoint",
    );

    // Loop modes on a 0 → 10 ramp of length 2.
    let ramp = |mode| {
        let mut t = Track::new("ramp").with_loop(mode);
        t.add_keyframe(Keyframe::new(0.0, 0.0));
        t.add_keyframe(Keyframe::new(2.0, 10.0));
        t
    };
    let once = ramp(LoopMode::Once);
    let looping = ramp(LoopMode::Loop);
    let ping = ramp(LoopMode::PingPong);
    println!(
        "loop modes at t=3: once={}, loop={}, pingpong={}",
        once.evaluate(3.0),
        looping.evaluate(3.0),
        ping.evaluate(3.0)
    );
    close(once.duration(), 2.0, 0.0, "track duration");
    close(once.evaluate(3.0), 10.0, 0.0, "once clamps");
    close(looping.evaluate(3.0), 5.0, 1e-6, "loop wraps");
    close(ping.evaluate(3.0), 5.0, 1e-6, "pingpong reflects");
    close(ping.evaluate(3.5), 2.5, 1e-6, "pingpong reflects");
    assert_eq!(Interpolation::default(), Interpolation::Linear);

    // morph: a linear blend of the two distance fields. `box3d` takes full
    // widths, so d_to(2,0,0) = 2 − 0.25 = 1.75 and d_from = 2 − 1 = 1.
    let from = SdfNode::sphere(1.0);
    let to = SdfNode::box3d(0.5, 0.5, 0.5);
    let q = Vec3::new(2.0, 0.0, 0.0);
    for &blend in &[0.0_f32, 0.5, 1.0] {
        let dm = eval(&morph(&from, &to, blend), q);
        println!("morph(blend={blend}) d({q}) = {dm}");
        close(dm, (1.0 - blend) * 1.0 + blend * 1.75, 1e-6, "morph lerp");
    }

    println!("all checks passed");
}

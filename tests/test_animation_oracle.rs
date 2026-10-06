//! Animation oracle: every value is checked against the closed form of the
//! interpolation it uses, never against the implementation.
//!
//! - linear: `v0 + (v1 − v0)·α`
//! - step: `v0` until the next key
//! - cubic Hermite: `h00·v0 + h10·span·m0 + h01·v1 + h11·span·m1`
//! - loop modes: wrap (`t mod d`), reflect (`d − |t mod 2d − d|`), clamp
//! - an animated sphere: `|p − c(t)| − r(t)`
//! - `morph`: `(1 − b)·d_from + b·d_to` (the linear blend `SdfNode::Morph`
//!   evaluates, and the endpoints the doc promises: 0 = from, 1 = to)
//!
//! Author: Moroya Sakamoto

use alice_sdf::prelude::*;

fn ramp(mode: LoopMode) -> Track {
    let mut t = Track::new("ramp").with_loop(mode);
    t.add_keyframe(Keyframe::new(0.0, 0.0));
    t.add_keyframe(Keyframe::new(2.0, 10.0));
    t
}

/// Hermite basis evaluated independently of the crate.
fn hermite(v0: f32, v1: f32, m0: f32, m1: f32, span: f32, a: f32) -> f32 {
    let (a2, a3) = (a * a, a * a * a);
    let h00 = 2.0 * a3 - 3.0 * a2 + 1.0;
    let h10 = a3 - 2.0 * a2 + a;
    let h01 = -2.0 * a3 + 3.0 * a2;
    let h11 = a3 - a2;
    h00 * v0 + h10 * span * m0 + h01 * v1 + h11 * span * m1
}

#[test]
fn linear_track_matches_lerp_at_known_times() {
    let track = ramp(LoopMode::Once);
    let mut compared = 0;
    for i in 0..=20 {
        let t = i as f32 * 0.1;
        let expected = 10.0 * (t / 2.0);
        let got = track.evaluate(t);
        assert!((got - expected).abs() < 1e-5, "t={t}: {got} vs {expected}");
        compared += 1;
    }
    assert_eq!(track.evaluate(0.0), 0.0);
    assert_eq!(track.evaluate(1.0), 5.0);
    assert_eq!(track.evaluate(2.0), 10.0);
    assert_eq!(track.duration(), 2.0);
    assert!(compared > 0);
}

#[test]
fn step_track_holds_the_previous_key() {
    let mut track = Track::new("s");
    track.add_keyframe(Keyframe::step(0.0, 1.0));
    track.add_keyframe(Keyframe::step(1.0, 3.0));
    track.add_keyframe(Keyframe::step(2.0, 7.0));
    for &(t, v) in &[
        (0.0, 1.0),
        (0.5, 1.0),
        (0.999, 1.0),
        (1.0, 3.0),
        (1.5, 3.0),
        (2.0, 7.0),
    ] {
        assert_eq!(track.evaluate(t), v, "t={t}");
    }
}

#[test]
fn cubic_track_matches_hermite_basis() {
    let (v0, v1, m0, m1, span) = (2.0_f32, -1.0_f32, 3.0_f32, -0.5_f32, 4.0_f32);
    let mut track = Track::new("c");
    track.add_keyframe(Keyframe::cubic(1.0, v0, m0, m1));
    track.add_keyframe(Keyframe::new(1.0 + span, v1));
    let mut compared = 0;
    for i in 1..16 {
        let a = i as f32 / 16.0;
        let got = track.evaluate(1.0 + a * span);
        let expected = hermite(v0, v1, m0, m1, span, a);
        assert!((got - expected).abs() < 1e-4, "α={a}: {got} vs {expected}");
        compared += 1;
    }
    // α = 0.5: 0.5·(v0 + v1) + 0.125·span·(m0 − m1)
    let mid = track.evaluate(1.0 + 0.5 * span);
    assert!((mid - (f32::midpoint(v0, v1) + 0.125 * span * (m0 - m1))).abs() < 1e-5);
    assert!(compared > 0);
}

#[test]
fn loop_modes_wrap_reflect_and_clamp() {
    let once = ramp(LoopMode::Once);
    let looping = ramp(LoopMode::Loop);
    let ping = ramp(LoopMode::PingPong);
    let d = 2.0_f32;
    let mut compared = 0;
    for i in 0..40 {
        let t = i as f32 * 0.37;
        let wrapped = t % d;
        let phase = t % (2.0 * d);
        let reflected = d - (phase - d).abs();
        let clamped = t.clamp(0.0, d);
        assert!(
            (looping.evaluate(t) - 5.0 * wrapped).abs() < 1e-4,
            "loop t={t}"
        );
        assert!(
            (ping.evaluate(t) - 5.0 * reflected).abs() < 1e-4,
            "pingpong t={t}"
        );
        assert!(
            (once.evaluate(t) - 5.0 * clamped).abs() < 1e-4,
            "once t={t}"
        );
        compared += 1;
    }
    // Negative time wraps into [0, d) for Loop.
    assert!((looping.evaluate(-0.5) - 5.0 * 1.5).abs() < 1e-5);
    assert!(compared > 0);
}

#[test]
fn timeline_speed_and_lookup() {
    let mut timeline = Timeline::new("tl");
    timeline.add_track(ramp(LoopMode::Once));
    let mut other = Track::new("other");
    other.add_keyframe(Keyframe::new(0.0, 1.0));
    other.add_keyframe(Keyframe::new(4.0, 2.0));
    timeline.add_track(other);
    assert_eq!(timeline.duration(), 4.0);

    timeline.speed = 2.0;
    // time 0.5 at speed 2 → track time 1.0
    assert_eq!(timeline.get_value("ramp", 0.5), Some(5.0));
    assert_eq!(timeline.get_value("other", 1.0), Some(1.5));
    assert_eq!(timeline.get_value("missing", 0.5), None);
    let all = timeline.evaluate(0.5);
    assert_eq!(all, vec![("ramp", 5.0), ("other", 1.25)]);
}

#[test]
fn animated_sphere_matches_moving_sphere_closed_form() {
    let mut timeline = Timeline::new("move");
    let mut x = Track::new("translate.x");
    x.add_keyframe(Keyframe::new(0.0, -1.0));
    x.add_keyframe(Keyframe::new(1.0, 3.0));
    timeline.add_track(x);
    let mut s = Track::new("scale");
    s.add_keyframe(Keyframe::new(0.0, 1.0));
    s.add_keyframe(Keyframe::new(1.0, 3.0));
    timeline.add_track(s);
    let animated = AnimatedSdf::new(SdfNode::sphere(0.5), timeline);

    let points = [
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(-2.0, -1.0, 1.0),
        Vec3::new(1.0, 0.25, -0.5),
    ];
    let mut compared = 0;
    for i in 0..=4 {
        let t = i as f32 * 0.25;
        let cx = -1.0 + 4.0 * t;
        let r = 0.5 + t; // 0.5 · (1 + 2t)
        let params = animated.evaluate_params(t);
        assert!((params.translate_x - cx).abs() < 1e-6);
        assert!((params.scale - (1.0 + 2.0 * t)).abs() < 1e-6);
        let node = animated.evaluate_at(t);
        for &p in &points {
            let expected = (p - Vec3::new(cx, 0.0, 0.0)).length() - r;
            let got = eval(&node, p);
            assert!(
                (got - expected).abs() < 1e-4,
                "t={t} p={p}: {got} vs {expected}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0);
}

#[test]
fn morph_is_the_linear_blend_with_doc_endpoints() {
    let from = SdfNode::sphere(1.0);
    let to = SdfNode::box3d(0.5, 0.5, 0.5);
    let points = [
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.7, 0.7, 0.0),
        Vec3::new(-1.0, 2.0, 0.5),
    ];
    let mut compared = 0;
    for &b in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let node = morph(&from, &to, b);
        for &p in &points {
            // oracle: sphere |p| − 1; box3d takes full widths, so the box
            // has half-extent 0.25 (exterior/interior closed form)
            let ds = p.length() - 1.0;
            let q = p.abs() - Vec3::splat(0.25);
            let db = q.max(Vec3::ZERO).length() + q.x.max(q.y).max(q.z).min(0.0);
            let expected = (1.0 - b) * ds + b * db;
            let got = eval(&node, p);
            assert!(
                (got - expected).abs() < 1e-5,
                "b={b} p={p}: {got} vs {expected}"
            );
            compared += 1;
        }
    }
    assert!(compared > 0);
}

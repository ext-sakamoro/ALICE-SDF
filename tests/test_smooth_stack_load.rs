//! Loading a smooth section stack rebuilds it from its levels and loops: the stored bound (knots,
//! lmax) must equal the one computed from them bit for bit, the level count and order are checked
//! (a short list is refused instead of panicking at evaluation), and a loop stored clockwise is
//! loaded counter-clockwise (the same curve, so the even-odd sign holds)

use alice_sdf::prelude::*;
use glam::Vec2;
use serde_json::Value;

fn circle(r: f32, c: Vec2, m: usize) -> Vec<Vec2> {
    (0..m)
        .map(|j| {
            let a = std::f32::consts::TAU * j as f32 / m as f32;
            let (s, co) = alice_det_math::sin_cos(a);
            c + Vec2::new(co, s) * r
        })
        .collect()
}

fn stack() -> SdfNode {
    let levels: Vec<f32> = (0..7).map(|i| -0.3 + 0.1 * i as f32).collect();
    let loops = levels.iter().map(|&y| vec![circle(0.2 + 0.1 * y, Vec2::new(0.05 * y, 0.0), 10)]).collect();
    SdfNode::smooth_section_stack(levels, loops, 0.01, 0.1).expect("a valid stack")
}

fn body(v: &mut Value) -> &mut serde_json::Map<String, Value> {
    v.get_mut("SmoothSectionStack").and_then(Value::as_object_mut).expect("the stack's stored form")
}

fn load(v: &Value) -> Result<SdfNode, serde_json::Error> {
    serde_json::from_value(v.clone())
}

#[test]
fn a_stored_stack_loads_back_equal() {
    let s = stack();
    let v = serde_json::to_value(&s).expect("serialise");
    assert_eq!(serde_json::to_value(load(&v).expect("a stack as built loads")).expect("serialise"), v);
}

#[test]
fn a_smaller_stored_lmax_is_refused() {
    let mut v = serde_json::to_value(stack()).expect("serialise");
    let m = body(&mut v);
    let l = m["lmax"].as_f64().expect("lmax");
    m.insert("lmax".into(), Value::from(l * 0.5));
    assert!(load(&v).is_err(), "a stored lmax below the bound would overstate the distance");
}

#[test]
fn a_changed_stored_knot_is_refused() {
    let mut v = serde_json::to_value(stack()).expect("serialise");
    let m = body(&mut v);
    let knots = m.get_mut("knots").and_then(Value::as_array_mut).expect("knots");
    let k = knots[3].as_f64().expect("a knot");
    knots[3] = Value::from(k * 0.75);
    assert!(load(&v).is_err(), "a stored knot that differs from the recomputed one is refused");
}

#[test]
fn three_levels_are_refused_on_load() {
    let mut v = serde_json::to_value(stack()).expect("serialise");
    let m = body(&mut v);
    for key in ["levels", "loops", "knots"] {
        m.get_mut(key).and_then(Value::as_array_mut).expect("an array").truncate(3);
    }
    assert!(load(&v).is_err(), "fewer than 4 levels is refused (evaluation would index past them)");
}

#[test]
fn levels_out_of_order_are_refused_on_load() {
    let mut v = serde_json::to_value(stack()).expect("serialise");
    let m = body(&mut v);
    m.get_mut("levels").and_then(Value::as_array_mut).expect("levels").swap(1, 2);
    assert!(load(&v).is_err(), "levels not strictly increasing are refused");
}

#[test]
fn a_bad_r0_or_eps_is_refused_on_load() {
    for (key, bad) in [("r0", -1.0), ("eps", 1.5)] {
        let mut v = serde_json::to_value(stack()).expect("serialise");
        body(&mut v).insert(key.into(), Value::from(bad));
        assert!(load(&v).is_err(), "{key} = {bad} is refused");
    }
}

#[test]
fn a_loop_stored_clockwise_loads_counter_clockwise() {
    let s = stack();
    let mut v = serde_json::to_value(&s).expect("serialise");
    let m = body(&mut v);
    let lp = m["loops"][2][0].as_array_mut().expect("a loop");
    lp.reverse();
    let loaded = load(&v).expect("a reversed loop is the same curve, loaded counter-clockwise");
    assert_eq!(serde_json::to_value(loaded).expect("serialise"), serde_json::to_value(&s).expect("serialise"), "the loaded stack equals the one built (the loop re-oriented)");
}

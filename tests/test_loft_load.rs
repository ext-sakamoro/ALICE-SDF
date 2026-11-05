//! Loading a loft rebuilds its bounds from the axis, rows and caps: a stored tree whose bounds are
//! of the current law version but differ from the recomputed ones in any field (the Lipschitz
//! factors, the knots, the saturation widths, the far-bound pieces, the cap curve) is refused, so a
//! stale or edited bound cannot overstate the distance. The law version stays as an early refusal
//! with its own message.

use alice_sdf::prelude::*;
use glam::Vec2;
use serde_json::Value;

fn polar() -> SdfNode {
    let axis = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.05, 0.3, 0.02),
        Vec3::new(0.12, 0.6, 0.0),
        Vec3::new(0.15, 0.9, -0.05),
    ];
    // [a0, a1, b1, a2, b2, a3, b3]
    let rows = vec![
        vec![0.10, 0.0, 0.01, 0.02, 0.0, 0.0, 0.004],
        vec![0.09, 0.005, 0.0, -0.015, 0.0, 0.0, 0.0],
        vec![0.08, 0.0, -0.01, 0.01, 0.005, 0.003, 0.0],
        vec![0.06, 0.0, 0.0, 0.005, 0.0, 0.0, -0.002],
    ];
    SdfNode::loft_capped(axis, Vec3::Z, 3, rows, [0.04, 0.0]).expect("a curved capped loft")
}

fn ellipse(m: usize, a: f32, b: f32) -> Vec<Vec2> {
    (0..m)
        .map(|k| {
            let (s, c) = alice_det_math::sin_cos(std::f32::consts::TAU * k as f32 / m as f32);
            Vec2::new(a * c, b * s)
        })
        .collect()
}

fn curves() -> SdfNode {
    let axis = vec![
        Vec3::ZERO,
        Vec3::new(0.02, 0.3, 0.0),
        Vec3::new(0.05, 0.6, 0.01),
        Vec3::new(0.06, 0.9, 0.0),
    ];
    let rows: Vec<Vec<Vec2>> = (0..4)
        .map(|i| ellipse(8, 0.12 - 0.01 * i as f32, 0.08 + 0.005 * i as f32))
        .collect();
    SdfNode::loft_curves_capped(axis, Vec3::Z, rows, [[0.03, 0.0, 0.0], [0.0, 0.0, 0.0]])
        .expect("a curve loft with a cap")
}

fn variant(v: &mut Value) -> &mut serde_json::Map<String, Value> {
    let o = v.as_object_mut().expect("an externally tagged node");
    let key = o.keys().next().expect("the variant").clone();
    o.get_mut(&key)
        .and_then(Value::as_object_mut)
        .expect("the variant's fields")
}

fn bounds(v: &mut Value) -> &mut serde_json::Map<String, Value> {
    variant(v)
        .get_mut("bounds")
        .and_then(Value::as_object_mut)
        .expect("bounds")
}

fn load(v: &Value) -> Result<SdfNode, String> {
    serde_json::from_value::<SdfNode>(v.clone()).map_err(|e| e.to_string())
}

fn scale_scalar(v: &mut Value, key: &str, f: f64) {
    let b = bounds(v);
    let x = b[key].as_f64().unwrap_or_else(|| panic!("{key}"));
    b.insert(key.into(), Value::from(x * f));
}

fn scale_elem(v: &mut Value, path: &[&str], i: usize, f: f64) {
    let mut cur = &mut *bounds(v);
    for k in &path[..path.len() - 1] {
        cur = cur
            .get_mut(*k)
            .and_then(Value::as_object_mut)
            .unwrap_or_else(|| panic!("{k}"));
    }
    let arr = cur
        .get_mut(path[path.len() - 1])
        .and_then(Value::as_array_mut)
        .expect("an array");
    let x = arr[i].as_f64().expect("a number");
    arr[i] = Value::from(x * f + if x == 0.0 { 0.01 } else { 0.0 });
}

#[test]
fn stored_lofts_load_back_unchanged() {
    for node in [polar(), curves()] {
        let v = serde_json::to_value(&node).expect("serialise");
        let back = load(&v).expect("a loft as built loads");
        assert_eq!(serde_json::to_value(back).expect("serialise"), v);
    }
}

#[test]
fn a_changed_scalar_bound_of_the_current_law_is_refused() {
    for node in [polar(), curves()] {
        for key in ["l", "l2", "core", "m", "reach"] {
            let mut v = serde_json::to_value(&node).expect("serialise");
            scale_scalar(&mut v, key, 0.5);
            let e = load(&v).expect_err(key);
            assert!(e.contains("rebuild"), "{key}: {e}");
        }
    }
}

#[test]
fn a_changed_knot_saturation_width_or_piece_is_refused() {
    for node in [polar(), curves()] {
        for key in ["knots", "bknots", "pieces"] {
            let mut v = serde_json::to_value(&node).expect("serialise");
            if bounds(&mut v)[key].as_array().is_none_or(Vec::is_empty) {
                continue;
            }
            scale_elem(&mut v, &[key], 0, 0.9);
            assert!(load(&v).is_err(), "{key}");
        }
        let mut v = serde_json::to_value(&node).expect("serialise");
        scale_elem(&mut v, &["lq"], 0, 0.5);
        assert!(load(&v).is_err(), "lq");
    }
}

#[test]
fn bounds_from_other_caps_are_refused() {
    // the stored caps changed while the bounds were left as they were
    let mut v = serde_json::to_value(polar()).expect("serialise");
    variant(&mut v).insert("caps".into(), serde_json::json!([0.0, 0.0]));
    assert!(load(&v).is_err(), "polar caps");
    let mut v = serde_json::to_value(curves()).expect("serialise");
    variant(&mut v).insert(
        "caps".into(),
        serde_json::json!([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    );
    assert!(load(&v).is_err(), "curve caps");
}

#[test]
fn malformed_stored_rows_are_refused_not_panicking() {
    let mut v = serde_json::to_value(polar()).expect("serialise");
    variant(&mut v)
        .get_mut("rows")
        .and_then(Value::as_array_mut)
        .expect("rows")
        .pop();
    assert!(load(&v).is_err(), "rows not a multiple of 2K + 1");
    let mut v = serde_json::to_value(curves()).expect("serialise");
    variant(&mut v)
        .get_mut("points")
        .and_then(Value::as_array_mut)
        .expect("points")
        .pop();
    assert!(load(&v).is_err(), "points not a multiple of 2M");
}

#[test]
fn another_law_version_is_refused_by_the_version() {
    let mut v = serde_json::to_value(polar()).expect("serialise");
    bounds(&mut v).insert("law".into(), Value::from(0));
    let e = load(&v).expect_err("law 0");
    assert!(e.contains("bound law"), "{e}");
}

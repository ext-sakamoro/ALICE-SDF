//! Fuzz target: SDF tree の bincode 1 layout (alice-serial の bincode1、.asdf body と ASP の
//! 形) の roundtrip が形状を保ち、bytes が往復で変わらないことを検証
//!
//! 起こり得る危険:
//! - serialize / deserialize の非対称 (再 encode した bytes が違う)
//! - SdfNode variant の一部 skip / mis-align

#![no_main]

use alice_sdf::eval::eval;
use alice_sdf::types::{SdfNode, SdfTree};
use arbitrary::Arbitrary;
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum PrimitiveKind {
    Sphere { radius: f32 },
    Box { width: f32, height: f32, depth: f32 },
    Cylinder { radius: f32, height: f32 },
}

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    primitive: PrimitiveKind,
    px: f32,
    py: f32,
    pz: f32,
}

fuzz_target!(|input: FuzzInput| {
    let node = match input.primitive {
        PrimitiveKind::Sphere { radius } => SdfNode::sphere(radius),
        PrimitiveKind::Box {
            width,
            height,
            depth,
        } => SdfNode::box3d(width, height, depth),
        PrimitiveKind::Cylinder { radius, height } => SdfNode::cylinder(radius, height),
    };
    let tree = SdfTree::new(node.clone());

    // the bincode 1 layout, as the .asdf body writes it
    let Ok(bytes) = alice_serial::bincode1::to_vec(&tree, 1 << 20) else {
        return;
    };
    let decoded: SdfTree = alice_serial::bincode1::from_slice(&bytes, bytes.len())
        .expect("bytes this encoder wrote decode");
    assert_eq!(
        alice_serial::bincode1::to_vec(&decoded, 1 << 20).ok().as_deref(),
        Some(&bytes[..]),
        "re-encoding gives the same bytes"
    );

    let point = Vec3::new(input.px, input.py, input.pz);
    let d1 = eval(&node, point);
    let d2 = eval(&decoded.root, point);

    // NaN 同士は != なので、両者 finite の時のみ厳格比較
    if d1.is_finite() && d2.is_finite() {
        // f32 精度の許容 (roundtrip は bit-exact であるべき)
        let diff = (d1 - d2).abs();
        assert!(
            diff < 1e-5,
            "roundtrip drift: node={:?} point={:?} d1={} d2={} diff={}",
            input.primitive,
            point,
            d1,
            d2,
            diff
        );
    }
});

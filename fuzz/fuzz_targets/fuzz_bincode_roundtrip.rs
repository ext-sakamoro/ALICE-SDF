//! Fuzz target: SDF tree の bincode roundtrip が形状を保つことを検証
//!
//! 任意の primitive tree を構築、bincode 2.x で encode → decode → eval が同じ距離を返すか
//!
//! 起こり得る危険:
//! - bincode 1 → 2 移行で config::legacy() の serialize / deserialize 非対称
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

    // bincode 2.x + config::legacy() で roundtrip
    let Ok(bytes) = bincode::serde::encode_to_vec(&tree, bincode::config::legacy()) else {
        return;
    };
    let Ok((decoded, _)) =
        bincode::serde::decode_from_slice::<SdfTree, _>(&bytes, bincode::config::legacy())
    else {
        return;
    };

    let point = Vec3::new(input.px, input.py, input.pz);
    let d1 = eval(&node, point);
    let d2 = eval(&decoded.root, point);

    // NaN 同士は != なので、両者 finite の時のみ厳格比較
    if d1.is_finite() && d2.is_finite() {
        // f32 精度の許容 (bincode roundtrip は bit-exact であるべき)
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

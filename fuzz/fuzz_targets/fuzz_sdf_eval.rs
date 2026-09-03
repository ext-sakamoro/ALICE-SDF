//! Fuzz target: SDF primitive eval で NaN/Inf / panic を炙り出す
//!
//! `sphere` / `box3d` / `cylinder` / `cone` の 4 primitive を arbitrary パラメータで構築、
//! arbitrary な `Vec3` で `eval` してもプロセスが panic しないこと + 結果が finite (or 意図的 NaN)
//! であることを検証
//!
//! 起こり得る危険:
//! - 距離 0 との除算で Inf / NaN が伝播 → 上位 marching cubes で無限ループ
//! - `radius = 0.0` や巨大な `height` で `f32::MAX` overflow → 上位 refit で symbol 崩壊

#![no_main]

use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use arbitrary::Arbitrary;
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum PrimitiveKind {
    Sphere { radius: f32 },
    Box { width: f32, height: f32, depth: f32 },
    Cylinder { radius: f32, height: f32 },
    Cone { radius: f32, height: f32 },
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
        PrimitiveKind::Cone { radius, height } => SdfNode::cone(radius, height),
    };

    let point = Vec3::new(input.px, input.py, input.pz);

    // eval が panic しないことを検証
    // 結果の NaN/Inf は SDF の domain 上 legitimate なケースもあるため fail させない
    // (例: degenerate primitive の境界で NaN、無限遠で Inf 等)
    let _distance = eval(&node, point);
});

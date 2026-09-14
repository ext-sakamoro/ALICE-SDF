//! Iterative `Drop` for [`SdfNode`]
//!
//! `SdfNode` は子を `Arc<Self>` で持つ再帰 enum なので、default の drop は
//! 子 → 孫 → … と再帰し、深さに比例して native stack を消費する
//! LOL stdlib の product tree は `subtract` を 2,400 段 nest する (正当な
//! 形) ため、2 MB stack の thread では drop だけで overflow していた
//!
//! ここでは子を heap 上の明示 stack に移してから解放する 再帰は一切ない
//!
//! 共有 subtree (`Arc::clone` で複数の親から参照される node) は
//! `Arc::try_unwrap` が `Err` を返すのでそのまま手放す 最後の所有者が
//! drop する時に同じ routine が走るので、深さに依らず stack 消費は O(1)
//!
//! Author: Moroya Sakamoto

use super::SdfNode;
use std::sync::{Arc, OnceLock};

/// 子 field を抜き取った跡に置く leaf `Arc` は refcount の増減だけで
/// allocation を伴わないので、drop の総 cost は node 数に線形
fn leaf_sentinel() -> Arc<SdfNode> {
    static LEAF: OnceLock<Arc<SdfNode>> = OnceLock::new();
    Arc::clone(LEAF.get_or_init(|| Arc::new(SdfNode::Sphere { radius: 0.0 })))
}

impl SdfNode {
    /// 子 `Arc` を全て `out` に移し、自身は leaf 相当の状態にする
    ///
    /// 子を持たない variant では何もしない 子を持つ variant を追加した時は
    /// ここにも arm を足す (`tests` の `take_children_covers_every_family`
    /// が代表 variant で検知する)
    fn take_children_into(&mut self, out: &mut Vec<Arc<Self>>) {
        match self {
            // 二項 operation (24)
            Self::Union { a, b, .. }
            | Self::Intersection { a, b, .. }
            | Self::Subtraction { a, b, .. }
            | Self::SmoothUnion { a, b, .. }
            | Self::SmoothIntersection { a, b, .. }
            | Self::SmoothSubtraction { a, b, .. }
            | Self::ChamferUnion { a, b, .. }
            | Self::ChamferIntersection { a, b, .. }
            | Self::ChamferSubtraction { a, b, .. }
            | Self::StairsUnion { a, b, .. }
            | Self::StairsIntersection { a, b, .. }
            | Self::StairsSubtraction { a, b, .. }
            | Self::ColumnsUnion { a, b, .. }
            | Self::ColumnsIntersection { a, b, .. }
            | Self::ColumnsSubtraction { a, b, .. }
            | Self::ExpSmoothUnion { a, b, .. }
            | Self::ExpSmoothIntersection { a, b, .. }
            | Self::ExpSmoothSubtraction { a, b, .. }
            | Self::XOR { a, b, .. }
            | Self::Pipe { a, b, .. }
            | Self::Engrave { a, b, .. }
            | Self::Groove { a, b, .. }
            | Self::Tongue { a, b, .. }
            | Self::Morph { a, b, .. } => {
                out.push(std::mem::replace(a, leaf_sentinel()));
                out.push(std::mem::replace(b, leaf_sentinel()));
            }

            // 単項 transform (7) + modifier (23) + Animated
            Self::Translate { child, .. }
            | Self::Rotate { child, .. }
            | Self::Scale { child, .. }
            | Self::ScaleNonUniform { child, .. }
            | Self::Shear { child, .. }
            | Self::ProjectiveTransform { child, .. }
            | Self::SdfSkinning { child, .. }
            | Self::Round { child, .. }
            | Self::Onion { child, .. }
            | Self::Elongate { child, .. }
            | Self::Twist { child, .. }
            | Self::Bend { child, .. }
            | Self::Taper { child, .. }
            | Self::Mirror { child, .. }
            | Self::OctantMirror { child, .. }
            | Self::IcosahedralSymmetry { child, .. }
            | Self::RepeatInfinite { child, .. }
            | Self::RepeatFinite { child, .. }
            | Self::PolarRepeat { child, .. }
            | Self::Displacement { child, .. }
            | Self::SineDisplacement { child, .. }
            | Self::HeightmapDisplacement { child, .. }
            | Self::Noise { child, .. }
            | Self::SurfaceRoughness { child, .. }
            | Self::Revolution { child, .. }
            | Self::Extrude { child, .. }
            | Self::SweepBezier { child, .. }
            | Self::LatticeDeform { child, .. }
            | Self::IFS { child, .. }
            | Self::WithMaterial { child, .. }
            | Self::Animated { child, .. } => {
                out.push(std::mem::replace(child, leaf_sentinel()));
            }

            // 葉 (primitive 72 種) は子を持たない
            _ => {}
        }
    }
}

impl Drop for SdfNode {
    fn drop(&mut self) {
        let mut stack: Vec<Arc<Self>> = Vec::new();
        self.take_children_into(&mut stack);
        if stack.is_empty() {
            return;
        }
        while let Some(child) = stack.pop() {
            // 唯一の所有者なら中身を取り出して、その子を先に stack へ退避する
            // `node` はここで scope を抜けて drop されるが、子は既に空なので
            // 再帰は 1 段で止まる 共有中 (`Err`) なら refcount -1 のみ
            if let Ok(mut node) = Arc::try_unwrap(child) {
                node.take_children_into(&mut stack);
            }
        }
    }
}

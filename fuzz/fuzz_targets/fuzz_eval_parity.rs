//! Fuzz target: every evaluation path must agree on the same tree.
//!
//! An arbitrary opcode list is folded into an `SdfNode` (primitives, CSG
//! operators, transforms, modifiers), then evaluated at arbitrary points plus
//! points deliberately placed on cell / sector boundaries (integers, halves,
//! multiples of the repeat spacing) through
//!
//! - the tree evaluator (`eval`)
//! - the compiled scalar stack machine (`eval_compiled`)
//! - the compiled SIMD stack machine (`eval_compiled_batch_simd`, 8 lanes)
//!
//! and any disagreement beyond a relative tolerance is a crash. NaN / Inf must
//! agree too (both paths or neither). This is the fuzz form of
//! `tests/test_evaluator_opcode_parity.rs`: the 1.10.3 `round` tie-break bug
//! (a whole cell of error at spacing/2) was invisible to the hand-written
//! corpus because its repeat child was symmetric; the fuzzer builds asymmetric
//! trees and lands on the ties on purpose.
//!
//! 起こり得る危険:
//! - scalar / SIMD の law drift (`wide` の多項式 vs libm、tie-break、operand 順)
//! - compile 時の param 欠落 / silent fallback (1.9.1 で 20+ 件)
//! - NaN を片方だけが出す (degenerate primitive の境界)
#![no_main]

use alice_sdf::compiled::{eval_compiled, eval_compiled_batch_simd, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::types::SdfNode;
use arbitrary::Arbitrary;
use glam::Vec3;
use libfuzzer_sys::fuzz_target;

/// Relative tolerance: the SIMD transcendental approximations (`wide`) differ
/// from libm by a few ulp, and the smooth operators amplify that slightly.
const TOL: f32 = 1e-3;
const MAX_DEPTH: usize = 6;

/// Keep parameters finite and in a sane range so the fuzzer explores law
/// disagreements rather than IEEE edge cases of the inputs themselves.
fn p(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        (lo + hi) * 0.5
    }
}

/// Positive parameter in `[0.05, 3]`.
fn pos(v: f32) -> f32 {
    p(v.abs(), 0.05, 3.0)
}

#[derive(Arbitrary, Debug)]
enum Prim {
    Sphere(f32),
    Box(f32, f32, f32),
    Cylinder(f32, f32),
    Torus(f32, f32),
    Cone(f32, f32),
    Capsule(f32, f32),
    Ellipsoid(f32, f32, f32),
    HexPrism(f32, f32),
    Octahedron(f32),
    Pyramid(f32),
    Helix(f32, f32, f32, f32),
    RoundedBox(f32, f32, f32, f32),
}

#[derive(Arbitrary, Debug)]
enum Op {
    /// Wrap the current node.
    Translate(f32, f32, f32),
    Scale(f32),
    RotateEuler(f32, f32, f32),
    Twist(f32),
    Bend(f32),
    Round(f32),
    Onion(f32),
    Elongate(f32, f32, f32),
    Mirror(bool, bool, bool),
    RepeatInfinite(f32, f32, f32),
    RepeatFinite(u8, u8, u8, f32),
    PolarRepeat(u8),
    Taper(f32),
    Shear(f32, f32, f32),
    /// Combine with a fresh primitive.
    Union(Prim),
    Intersection(Prim),
    Subtract(Prim),
    SmoothUnion(Prim, f32),
    SmoothIntersection(Prim, f32),
    SmoothSubtract(Prim, f32),
    ChamferUnion(Prim, f32),
    StairsUnion(Prim, f32, u8),
    Xor(Prim),
    Morph(Prim, f32),
    ExpSmoothUnion(Prim, f32),
}

#[derive(Arbitrary, Debug)]
struct Input {
    root: Prim,
    ops: Vec<Op>,
    /// Free sample points.
    points: Vec<(f32, f32, f32)>,
    /// Which axis-aligned tie grid to sample (see `tie_points`).
    tie_scale: u8,
}

fn prim(q: Prim) -> SdfNode {
    match q {
        Prim::Sphere(r) => SdfNode::sphere(pos(r)),
        Prim::Box(a, b, c) => SdfNode::box3d(pos(a), pos(b), pos(c)),
        Prim::Cylinder(r, h) => SdfNode::cylinder(pos(r), pos(h)),
        Prim::Torus(a, b) => SdfNode::torus(pos(a) + 0.2, pos(b) * 0.3),
        Prim::Cone(r, h) => SdfNode::cone(pos(r), pos(h)),
        Prim::Capsule(r, h) => SdfNode::capsule(
            Vec3::new(0.0, -pos(h), 0.0),
            Vec3::new(0.0, pos(h), 0.0),
            pos(r),
        ),
        Prim::Ellipsoid(a, b, c) => SdfNode::ellipsoid(pos(a), pos(b), pos(c)),
        Prim::HexPrism(r, h) => SdfNode::hex_prism(pos(r), pos(h)),
        Prim::Octahedron(s) => SdfNode::octahedron(pos(s)),
        Prim::Pyramid(h) => SdfNode::pyramid(pos(h)),
        // pitch ≥ 0.25: with 20+ turns per unit the nearest-turn choice is
        // ill-conditioned and the SIMD sin/cos polynomial's few-ulp difference
        // from libm becomes a 0.2% distance difference (tolerance domain, not a
        // law bug — see Backlog "SIMD transcendental parity")
        Prim::Helix(a, b, c, d) => {
            SdfNode::helix(pos(a) + 0.2, pos(b) * 0.3, pos(c).max(0.25), pos(d))
        }
        Prim::RoundedBox(a, b, c, r) => SdfNode::rounded_box(pos(a), pos(b), pos(c), pos(r) * 0.2),
    }
}

fn build(input: &Input) -> SdfNode {
    let mut node = prim(take(&input.root));
    let mut depth = 0;
    for op in input.ops.iter().take(MAX_DEPTH * 2) {
        if depth >= MAX_DEPTH {
            break;
        }
        depth += 1;
        node = match op {
            Op::Translate(x, y, z) => {
                node.translate(p(*x, -3.0, 3.0), p(*y, -3.0, 3.0), p(*z, -3.0, 3.0))
            }
            Op::Scale(s) => node.scale(p(s.abs(), 0.25, 3.0)),
            Op::RotateEuler(x, y, z) => {
                node.rotate_euler(p(*x, -3.2, 3.2), p(*y, -3.2, 3.2), p(*z, -3.2, 3.2))
            }
            Op::Twist(k) => node.twist(p(*k, -2.0, 2.0)),
            Op::Bend(k) => node.bend(p(*k, -1.0, 1.0)),
            Op::Round(r) => node.round(pos(*r) * 0.3),
            Op::Onion(t) => node.onion(pos(*t) * 0.3),
            Op::Elongate(x, y, z) => node.elongate(pos(*x), pos(*y), pos(*z)),
            Op::Mirror(x, y, z) => node.mirror(*x, *y, *z),
            Op::RepeatInfinite(x, y, z) => {
                node.repeat_infinite(pos(*x) + 0.5, pos(*y) + 0.5, pos(*z) + 0.5)
            }
            Op::RepeatFinite(a, b, c, s) => node.repeat_finite(
                [
                    u32::from(*a % 5) + 1,
                    u32::from(*b % 5) + 1,
                    u32::from(*c % 5) + 1,
                ],
                Vec3::splat(pos(*s) + 0.5),
            ),
            Op::PolarRepeat(n) => node.polar_repeat(u32::from(*n % 12) + 2),
            Op::Taper(f) => node.taper(p(*f, -0.9, 0.9)),
            Op::Shear(a, b, c) => node.shear(p(*a, -1.0, 1.0), p(*b, -1.0, 1.0), p(*c, -1.0, 1.0)),
            Op::Union(q) => node.union(prim(take(q))),
            Op::Intersection(q) => node.intersection(prim(take(q))),
            Op::Subtract(q) => node.subtract(prim(take(q))),
            Op::SmoothUnion(q, k) => node.smooth_union(prim(take(q)), pos(*k) * 0.5),
            Op::SmoothIntersection(q, k) => node.smooth_intersection(prim(take(q)), pos(*k) * 0.5),
            Op::SmoothSubtract(q, k) => node.smooth_subtract(prim(take(q)), pos(*k) * 0.5),
            Op::ChamferUnion(q, r) => node.chamfer_union(prim(take(q)), pos(*r) * 0.5),
            Op::StairsUnion(q, r, n) => {
                node.stairs_union(prim(take(q)), pos(*r) * 0.5, f32::from(*n % 6) + 1.0)
            }
            Op::Xor(q) => node.xor(prim(take(q))),
            Op::Morph(q, t) => node.morph(prim(take(q)), p(*t, 0.0, 1.0)),
            Op::ExpSmoothUnion(q, k) => node.exp_smooth_union(prim(take(q)), pos(*k) * 0.5 + 0.1),
        };
    }
    node
}

/// Cheap by-value copy of a `Prim` (the derive gives no `Clone`).
fn take(q: &Prim) -> Prim {
    match q {
        Prim::Sphere(r) => Prim::Sphere(*r),
        Prim::Box(a, b, c) => Prim::Box(*a, *b, *c),
        Prim::Cylinder(r, h) => Prim::Cylinder(*r, *h),
        Prim::Torus(a, b) => Prim::Torus(*a, *b),
        Prim::Cone(r, h) => Prim::Cone(*r, *h),
        Prim::Capsule(r, h) => Prim::Capsule(*r, *h),
        Prim::Ellipsoid(a, b, c) => Prim::Ellipsoid(*a, *b, *c),
        Prim::HexPrism(r, h) => Prim::HexPrism(*r, *h),
        Prim::Octahedron(s) => Prim::Octahedron(*s),
        Prim::Pyramid(h) => Prim::Pyramid(*h),
        Prim::Helix(a, b, c, d) => Prim::Helix(*a, *b, *c, *d),
        Prim::RoundedBox(a, b, c, r) => Prim::RoundedBox(*a, *b, *c, *r),
    }
}

/// Points that sit exactly on the boundaries every snapping law uses:
/// integers, halves, and quarter multiples in `[-3, 3]`, so `p / spacing`
/// hits `k + 0.5` for the spacings the builder produces.
fn tie_points(scale: u8) -> Vec<Vec3> {
    let step = match scale % 4 {
        0 => 0.5,
        1 => 0.25,
        2 => 1.0,
        _ => 0.75,
    };
    let mut out = Vec::new();
    let n = (6.0 / step) as i32;
    for i in -n / 2..=n / 2 {
        let v = i as f32 * step;
        out.push(Vec3::new(v, 0.0, 0.0));
        out.push(Vec3::new(0.0, v, 0.0));
        out.push(Vec3::new(v, v, -v));
    }
    out
}

fn same(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= TOL * a.abs().max(b.abs()).max(1.0)
}

fuzz_target!(|input: Input| {
    let node = build(&input);
    let Ok(compiled) = CompiledSdf::try_compile(&node) else {
        // unsupported combination: not a parity question
        return;
    };

    let mut pts: Vec<Vec3> = input
        .points
        .iter()
        .take(16)
        .map(|&(x, y, z)| Vec3::new(p(x, -4.0, 4.0), p(y, -4.0, 4.0), p(z, -4.0, 4.0)))
        .collect();
    pts.extend(tie_points(input.tie_scale));
    while pts.len() % 8 != 0 {
        pts.push(Vec3::ZERO);
    }

    let simd = eval_compiled_batch_simd(&compiled, &pts);
    for (i, &pt) in pts.iter().enumerate() {
        let tree = eval(&node, pt);
        let scalar = eval_compiled(&compiled, pt);
        assert!(
            same(tree, scalar),
            "tree {tree} vs compiled scalar {scalar} at {pt:?}\nnode: {node:?}"
        );
        assert!(
            same(tree, simd[i]),
            "tree {tree} vs compiled SIMD {} at {pt:?}\nnode: {node:?}",
            simd[i]
        );
    }
});

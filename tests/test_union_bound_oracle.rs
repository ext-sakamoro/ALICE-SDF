//! Union operand bounds (R0): a union may skip an operand whose bounding sphere is farther than the
//! nearest operand so far, and the value never changes
//!
//! oracle, independent of the implementation:
//! - (a) the bounded union is bit-equal to the plain union on hit points (found by bisection along
//!   rays) and on random points near every operand, where any operand could be the minimum
//! - (b) the bound is a lower bound of the operand's distance: |p − c| − r ≤ d(p) on far points
//! - (c) a point inside an operand's bounding sphere is never decided by the bound (the operand is
//!   evaluated): the bounded union equals the plain one there, including where that operand is the
//!   minimum
//! - (d) every test fails when it compared nothing
//! - (e) the bound does skip: on points far from most operands, fewer operands are evaluated than
//!   the plain union evaluates (a bounded union that never skips is not green)
//! - the shader with bounds gives the CPU value (GPU, `ALICE_SDF_REQUIRE_GPU=1` makes a missing
//!   adapter a failure)

use alice_sdf::bound::{bound, bounding_sphere, BoundTarget};
use alice_sdf::eval::{eval, eval_union_bounded, eval_union_bounded_stats};
use alice_sdf::prelude::*;
use glam::Quat;

#[path = "common/gpu_batches.rs"]
mod gpu_batches;

/// A deterministic point generator (xorshift, no dependency)
struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
    fn v(&mut self, lo: f32, hi: f32) -> Vec3 {
        Vec3::new(self.f(), self.f(), self.f()) * (hi - lo) + Vec3::splat(lo)
    }
}

/// Operands that are posed, placed, twisted, rounded and warped-free: a mixed set
fn operands() -> Vec<SdfNode> {
    vec![
        SdfNode::sphere(0.2).translate(-0.5, 0.0, 0.0),
        SdfNode::box3d(0.15, 0.3, 0.1)
            .rigid_pose(Quat::from_rotation_z(0.4), Vec3::new(0.3, 0.1, 0.0)),
        SdfNode::capsule(Vec3::new(0.0, -0.2, 0.0), Vec3::new(0.0, 0.2, 0.0), 0.05)
            .rotate(Quat::from_rotation_x(1.1))
            .translate(0.0, 0.5, 0.2),
        SdfNode::torus(0.2, 0.04).translate(0.1, -0.5, -0.1),
        SdfNode::box3d(0.1, 0.1, 0.1)
            .twist(2.0)
            .translate(-0.2, -0.2, 0.4),
        SdfNode::sphere(0.1).scale(1.5).translate(0.6, -0.4, 0.3),
    ]
}

fn union_of(ops: &[SdfNode]) -> SdfNode {
    let mut it = ops.iter().cloned();
    let first = it.next().expect("operands");
    it.fold(first, |a, b| a.union(b))
}

/// Points on the zero set of `node`: bisection between an inside and an outside sample along rays
fn hit_points(node: &SdfNode, centre: Vec3, n: usize, rng: &mut Rng) -> Vec<Vec3> {
    let mut out = Vec::new();
    for _ in 0..n * 8 {
        if out.len() == n {
            break;
        }
        let dir = (rng.v(-1.0, 1.0)).normalize_or_zero();
        if dir == Vec3::ZERO {
            continue;
        }
        let (mut a, mut b) = (centre, centre + dir * 2.0);
        if eval(node, a) >= 0.0 || eval(node, b) <= 0.0 {
            continue;
        }
        for _ in 0..40 {
            let m = (a + b) * 0.5;
            if eval(node, m) < 0.0 {
                a = m;
            } else {
                b = m;
            }
        }
        out.push(b);
    }
    out
}

#[test]
fn a_bounded_union_is_bit_equal_on_hit_points_and_near_every_operand() {
    let ops = operands();
    let node = union_of(&ops);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut pts = Vec::new();
    for op in &ops {
        let (c, r) = bounding_sphere(op).expect("every operand here is bounded");
        pts.extend(hit_points(op, c, 64, &mut rng));
        for _ in 0..256 {
            pts.push(c + rng.v(-1.0, 1.0) * (r * 1.5));
        }
    }
    let mut compared = 0;
    for p in &pts {
        let (got, want) = (eval_union_bounded(&node, *p), eval(&node, *p));
        assert_eq!(got.to_bits(), want.to_bits(), "p {p}: {got} vs {want}");
        compared += 1;
    }
    assert!(compared >= ops.len() * 256, "compared {compared}");
}

#[test]
fn the_bound_is_a_lower_bound_of_the_operand_distance() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut compared = 0;
    for op in operands() {
        let (c, r) = bounding_sphere(&op).expect("bounded");
        for _ in 0..512 {
            let p = c + rng.v(-1.0, 1.0).normalize_or_zero() * (r * (1.0 + 4.0 * rng.f()));
            assert!(
                (p - c).length() - r <= eval(&op, p) + 1e-5,
                "p {p}: bound {} > distance {}",
                (p - c).length() - r,
                eval(&op, p)
            );
            compared += 1;
        }
        // the surface lies inside the sphere
        for h in hit_points(&op, c, 64, &mut rng) {
            assert!(
                (h - c).length() <= r + 1e-5,
                "surface point {h} outside the bound"
            );
            compared += 1;
        }
    }
    assert!(compared >= 6 * 512, "compared {compared}");
}

#[test]
fn inside_a_bound_the_operand_is_evaluated() {
    // the far operand is the minimum at points inside its bound but outside every other operand's:
    // a bound that skipped there would give the near operand's larger value
    let near = SdfNode::sphere(0.1).translate(-0.6, 0.0, 0.0);
    let far = SdfNode::box3d(0.2, 0.2, 0.2).translate(0.5, 0.0, 0.0);
    let node = near.union(far.clone());
    let (c, r) = bounding_sphere(&far).expect("bounded");
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let mut compared = 0;
    for _ in 0..1024 {
        let p = c + rng.v(-1.0, 1.0) * r;
        if (p - c).length() > r {
            continue;
        }
        let want = eval(&node, p);
        assert_eq!(
            eval(&far, p).to_bits(),
            want.to_bits(),
            "the far operand is the minimum"
        );
        assert_eq!(
            eval_union_bounded(&node, p).to_bits(),
            want.to_bits(),
            "p {p}"
        );
        compared += 1;
    }
    assert!(compared >= 300, "compared {compared}");
}

#[test]
fn the_bound_actually_skips_far_operands() {
    // points near one operand and far from the others: the others' bounds are farther than the
    // near operand's distance, so they are skipped
    let ops = operands();
    let node = union_of(&ops);
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    let (mut evaluated, mut skipped, mut plain, mut compared) = (0usize, 0usize, 0usize, 0usize);
    for op in &ops {
        let (c, r) = bounding_sphere(op).expect("bounded");
        for h in hit_points(op, c, 32, &mut rng) {
            let (v, st) = eval_union_bounded_stats(&node, h);
            assert_eq!(v.to_bits(), eval(&node, h).to_bits(), "value at {h}");
            assert_eq!(
                st.evaluated + st.skipped,
                ops.len(),
                "every operand is counted once"
            );
            evaluated += st.evaluated;
            skipped += st.skipped;
            plain += ops.len();
            compared += 1;
        }
        let _ = r;
    }
    assert!(compared >= ops.len() * 16, "compared {compared}");
    assert!(skipped > 0, "the bounded union skipped no operand");
    assert!(
        evaluated * 2 < plain,
        "evaluated {evaluated} of {plain}: near one operand most of the others are skipped"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn the_bounded_shader_gives_the_cpu_value() {
    use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
    let ops = operands();
    let node = union_of(&ops);
    let sh = WgslShader::transpile_with_union_bounds(&node, TranspileMode::Hardcoded);
    let mut rng = Rng(0x0bad_f00d_1234_5678);
    let mut pts = Vec::new();
    for op in &ops {
        let (c, r) = bounding_sphere(op).expect("bounded");
        pts.extend(hit_points(op, c, 32, &mut rng));
        for _ in 0..128 {
            pts.push(c + rng.v(-1.0, 1.0) * (r * 3.0));
        }
    }
    let g = match GpuEvaluator::from_shader(&sh) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "no GPU: {e}"
            );
            return;
        }
    };
    let got = gpu_batches::eval_in_batches(&g, &pts).expect("gpu eval");
    let mut compared = 0;
    for (p, v) in pts.iter().zip(&got) {
        let want = eval(&node, *p);
        assert!((v - want).abs() < 2e-6, "p {p}: gpu {v} vs cpu {want}");
        compared += 1;
    }
    assert!(compared >= ops.len() * 128, "compared {compared}");
}

/// (f) the viewer-path shader carries the bounds and the GPU skips with them:
/// - the rest-point shader (the viewer's) has one skip guard per bounded operand
/// - on points near one operand, the GPU skip count summed over the points is > 0 and more than
///   half of the operand evaluations are skipped (the CPU counts, measured on the same points)
/// - the bounded shader's GPU values are bit-equal to the plain shader's on the GPU: skipping
///   values greater than the result does not change `min`
#[cfg(feature = "gpu")]
#[test]
fn the_viewer_shader_skips_on_the_gpu_and_keeps_the_plain_bits() {
    use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
    let ops = operands();
    let node = union_of(&ops);

    // the rest point does not follow a twist: the viewer scene is the operands without it
    let rest_ops: Vec<SdfNode> = ops
        .iter()
        .filter(|op| !matches!(op, SdfNode::Translate { child, .. } if matches!(child.as_ref(), SdfNode::Twist { .. })))
        .cloned()
        .collect();
    assert_eq!(
        rest_ops.len(),
        ops.len() - 1,
        "one twisted operand left out"
    );
    let rest =
        WgslShader::transpile_with_rest_point(&union_of(&rest_ops), TranspileMode::Hardcoded)
            .expect("the operands have rest points");
    let guards = rest.source.matches("n_skip = n_skip + 1.0").count();
    assert_eq!(
        guards,
        rest_ops.len(),
        "one skip guard per bounded operand in the viewer shader"
    );

    let mut rng = Rng(0x7e57_0f00_dead_0001);
    let mut pts = Vec::new();
    for op in &ops {
        let (c, _) = bounding_sphere(op).expect("bounded");
        pts.extend(hit_points(op, c, 32, &mut rng));
    }
    assert!(pts.len() >= ops.len() * 16, "points {}", pts.len());

    let count = WgslShader::transpile_union_skip_count(&node, TranspileMode::Hardcoded);
    let gpu = |sh: &WgslShader| match GpuEvaluator::from_shader(sh) {
        Ok(g) => Some(gpu_batches::eval_in_batches(&g, &pts).expect("gpu eval")),
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "no GPU: {e}"
            );
            None
        }
    };
    let Some(skips) = gpu(&count) else { return };
    let skipped: f32 = skips.iter().sum();
    let plain_evals = (pts.len() * ops.len()) as f32;
    assert!(skipped > 0.0, "the GPU skipped no operand");
    assert!(
        skipped * 2.0 > plain_evals,
        "the GPU skipped {skipped} of {plain_evals} operand evaluations"
    );

    let bounded = gpu(&WgslShader::transpile_with_union_bounds(
        &node,
        TranspileMode::Hardcoded,
    ))
    .expect("gpu");
    let plain = gpu(&WgslShader::transpile(&node, TranspileMode::Hardcoded)).expect("gpu");
    let mut compared = 0;
    for ((p, b), q) in pts.iter().zip(&bounded).zip(&plain) {
        assert_eq!(
            b.to_bits(),
            q.to_bits(),
            "p {p}: bounded {b} vs plain {q} on the GPU"
        );
        compared += 1;
    }
    assert!(compared >= ops.len() * 16, "compared {compared}");
}

/// A random tree of every bounded node kind, with off-axis and off-centre children (twist of a
/// translated child, rotation of a translated child, nested unions, round, subtraction,
/// intersection, pose), from e4's verification. `neg` also draws negative and zero scales
fn random_tree(g: &mut Rng, depth: u32, neg: bool) -> SdfNode {
    let leaf = |g: &mut Rng| match (g.f() * 4.0) as u32 {
        0 => SdfNode::sphere(0.02 + 0.38 * g.f()),
        1 => SdfNode::box3d(
            0.02 + 0.28 * g.f(),
            0.02 + 0.28 * g.f(),
            0.02 + 0.28 * g.f(),
        ),
        2 => SdfNode::capsule(g.v(-0.3, 0.3), g.v(-0.3, 0.3), 0.01 + 0.09 * g.f()),
        _ => SdfNode::torus(0.05 + 0.25 * g.f(), 0.01 + 0.07 * g.f()),
    };
    if depth == 0 {
        return leaf(g);
    }
    let c = random_tree(g, depth - 1, neg);
    let off = |g: &mut Rng, s: f32| g.v(-s, s);
    match (g.f() * 11.0) as u32 {
        0 => {
            let o = off(g, 0.8);
            c.translate(o.x, o.y, o.z)
        }
        1 => {
            let o = off(g, 0.6);
            let axis = g.v(-1.0, 1.0).normalize_or(Vec3::Y);
            c.translate(o.x, o.y, o.z)
                .rotate(Quat::from_axis_angle(axis, 6.0 * g.f() - 3.0))
        }
        2 => {
            let s = if neg && g.f() < 0.3 {
                if g.f() < 0.5 {
                    -(0.2 + 2.8 * g.f())
                } else {
                    0.0
                }
            } else {
                0.2 + 2.8 * g.f()
            };
            let o = off(g, 0.5);
            c.translate(o.x, o.y, o.z).scale(s)
        }
        3 => {
            let o = off(g, 0.6);
            c.translate(o.x, o.y, o.z).twist(8.0 * g.f() - 4.0)
        }
        4 => c.round(0.1 * g.f()),
        5 => {
            let o = off(g, 0.6);
            c.translate(o.x, o.y, o.z)
                .union(random_tree(g, depth - 1, neg))
        }
        6 => c.intersection(random_tree(g, depth - 1, neg)),
        7 => {
            let o = off(g, 0.6);
            c.translate(o.x, o.y, o.z)
                .subtract(random_tree(g, depth - 1, neg))
        }
        8 => {
            let axis = g.v(-1.0, 1.0).normalize_or(Vec3::Z);
            c.rigid_pose(Quat::from_axis_angle(axis, 6.0 * g.f() - 3.0), off(g, 0.5))
        }
        9 => {
            let o = off(g, 0.6);
            c.translate(o.x, o.y, o.z).with_material(3)
        }
        _ => leaf(g).translate(0.3, 0.0, 0.0).union(c),
    }
}

/// (g) random trees: the bound is a lower bound of the evaluated value at every point (inside and
/// outside the sphere), and the bounded union is bit-equal. Teeth: e4's mutants of `src/bound.rs`
/// (twist axis growth, round radius, subtraction's operand, scale radius, rotate centre, the
/// slack, the enclosing centre, a negative scale) each turn this red
#[test]
fn random_trees_keep_the_bound_and_the_bits() {
    let mut g = Rng(0x2545_f491_4f6c_dd1d);
    let (mut compared, mut bounded, mut viol, mut neq) = (0u64, 0u64, 0u64, 0u64);
    let mut first = None;
    for t in 0..3000 {
        let d = 1 + (t % 4) as u32;
        let n = random_tree(&mut g, d, false);
        let (Some(b), Some((c, r))) = (bound(&n, BoundTarget::Cpu), bounding_sphere(&n)) else {
            continue;
        };
        bounded += 1;
        let u = n
            .clone()
            .union(random_tree(&mut g, d, false))
            .union(SdfNode::sphere(0.1).translate(1.0, 0.0, 0.0));
        for _ in 0..64 {
            let p = c + g.v(-1.0, 1.0) * (r * 2.5 + 0.1);
            let v = eval(&n, p);
            if v.is_finite() && b.lower(p) > v {
                viol += 1;
                first.get_or_insert_with(|| (t, p, v, b.lower(p)));
            }
            if eval_union_bounded(&u, p).to_bits() != eval(&u, p).to_bits() {
                neq += 1;
            }
            compared += 1;
        }
    }
    assert!(
        compared > 50_000 && bounded > 1000,
        "compared {compared}, bounded trees {bounded}"
    );
    assert_eq!(viol, 0, "bound violations {viol}, first {first:?}");
    assert_eq!(neq, 0, "bounded union bit mismatches {neq}");
}

/// (h) a scale that is not positive and finite has no bound (the operand is always evaluated), at
/// the node and anywhere below it in a tree
#[test]
fn a_non_positive_scale_has_no_bound() {
    let mut compared = 0;
    for s in [
        -1.5_f32,
        -0.0,
        0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(
            bounding_sphere(&SdfNode::sphere(0.2).scale(s)).is_none(),
            "scale {s}"
        );
        compared += 1;
    }
    let mut g = Rng(0x0dd5_ca1e_0dd5_ca1e);
    let mut drawn = 0;
    for t in 0..2000 {
        let n = random_tree(&mut g, 1 + (t % 4) as u32, true);
        if !has_non_positive_scale(&n) {
            continue;
        }
        drawn += 1;
        // the bound may still exist above a subtraction's or intersection's other operand; where
        // it exists it must hold
        if let (Some(b), Some((c, r))) = (bound(&n, BoundTarget::Cpu), bounding_sphere(&n)) {
            for _ in 0..32 {
                let p = c + g.v(-1.0, 1.0) * (r * 2.5 + 0.1);
                let v = eval(&n, p);
                assert!(!(v.is_finite() && b.lower(p) > v), "tree {t} at {p}: {v}");
                compared += 1;
            }
        }
    }
    assert!(drawn >= 100, "trees with a non-positive scale {drawn}");
    assert!(compared >= 6, "compared {compared}");
}

fn has_non_positive_scale(n: &SdfNode) -> bool {
    match n {
        SdfNode::Scale { child, factor } => {
            *factor <= 0.0 || factor.is_nan() || has_non_positive_scale(child)
        }
        SdfNode::Translate { child, .. }
        | SdfNode::Rotate { child, .. }
        | SdfNode::Twist { child, .. }
        | SdfNode::Round { child, .. }
        | SdfNode::Pose { child }
        | SdfNode::WithMaterial { child, .. } => has_non_positive_scale(child),
        SdfNode::Union { a, b }
        | SdfNode::Intersection { a, b }
        | SdfNode::Subtraction { a, b } => has_non_positive_scale(a) || has_non_positive_scale(b),
        _ => false,
    }
}

/// (i) NaN operands: no bounded primitive gives NaN at a finite point, so a skipped operand is
/// never NaN; a NaN comes from an unbounded operand (here a zero scale: `p · (1/0)` then `· 0`),
/// which is always evaluated, and first. The union's `min` ignores a NaN operand, and a skip needs
/// `lower bound > minimum so far`, false against NaN: a NaN minimum skips nothing until a number
/// replaces it, the bounded union is bit-equal with a NaN operand present, and an all-NaN union
/// stays NaN
#[test]
fn a_nan_operand_keeps_the_bits() {
    let nan_op = SdfNode::sphere(0.2).scale(0.0);
    assert!(
        bounding_sphere(&nan_op).is_none(),
        "a zero scale has no bound"
    );
    let probe = eval(&nan_op, Vec3::new(1.0, 0.3, 0.0));
    assert!(
        probe.is_nan(),
        "the zero scale is a NaN operand here: {probe}"
    );
    let node = SdfNode::sphere(0.2)
        .translate(-0.6, 0.0, 0.0)
        .union(nan_op.clone())
        .union(SdfNode::box3d(0.1, 0.1, 0.1).translate(0.8, 0.0, 0.0));
    let all_nan = nan_op
        .clone()
        .union(nan_op.clone().translate(0.5, 0.0, 0.0));
    let mut rng = Rng(0x0a0a_5eed_0a0a_5eed);
    let (mut compared, mut nan_min, mut skipped) = (0, 0, 0);
    for _ in 0..2048 {
        let p = rng.v(-1.5, 1.5);
        let (v, st) = eval_union_bounded_stats(&node, p);
        assert_eq!(v.to_bits(), eval(&node, p).to_bits(), "p {p}: {v}");
        if eval(&nan_op, p).is_nan() {
            nan_min += 1;
        }
        skipped += st.skipped;
        let w = eval_union_bounded(&all_nan, p);
        assert!(w.is_nan(), "all-NaN union at {p}: {w}");
        assert_eq!(
            w.to_bits(),
            eval(&all_nan, p).to_bits(),
            "all-NaN union at {p}: {w}"
        );
        compared += 1;
    }
    assert!(
        compared == 2048 && nan_min > 1000,
        "compared {compared}, NaN operand {nan_min}"
    );
    // the NaN operand is evaluated first and starts a NaN minimum, which skips nothing; the next
    // operand replaces it (min ignores NaN), and from there the bounds skip as without it
    assert!(
        skipped > 0,
        "the bounds skipped nothing with a NaN operand present"
    );
}

/// (j) twist on the GPU: the shader's sine and cosine are the native ones (WGSL bounds their error
/// only in [−π, π]); the twist operand's guard is the axis-centred sphere taken at the union's
/// point (no trigonometry in the guard), and its slack covers the native error. Large strengths
/// and heights, an off-axis child, points on the guard sphere's shell and near the surface: the
/// bounded GPU values are bit-equal to the plain shader's
///
/// Known defect `g3-vulkan-nocontraction`, the arm of a fusing driver: a wgpu driver that fuses
/// `a * b + c` (Vulkan; Metal through wgpu, which compiles with fast math) does it per pipeline (the
/// emitted shader does not forbid it), so the same operand can round differently in the two
/// shaders. Measured on one Vulkan GPU at k 2, p [0.17136355, 2.0207052, -0.5316945]
/// (bounded 0.05508475, plain 0.05508481: 6e-8, one ulp of the twisted coordinates). When the
/// probe [`gpu_fuses_a_multiply_add`] finds the driver fusing, the values must agree within the
/// fusing's own budget ([`contraction_budget`], ~1e-7 here, the measured 6e-8 inside it) and the
/// known defect is printed; once the driver path forbids contraction (the MSL engine of the `msl`
/// feature does today) the probe finds no fusion and the bit-equal assert applies. The CPU is a
/// bit-equal assert always
#[cfg(feature = "gpu")]
#[test]
fn a_large_twist_keeps_the_plain_bits_on_the_gpu() {
    use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
    // KNOWN-DEFECT-ARM: known defect: g3-vulkan-nocontraction: a fusing wgpu driver (Vulkan, Metal through wgpu) is tolerated within contraction_budget until the driver path forbids contraction
    // the known defect's arm, see the doc: only while the driver fuses (Vulkan; Metal through wgpu)
    let known_defect = gpu_fuses_a_multiply_add() == Some(true);
    let mut differed = 0;
    let mut rng = Rng(0x7015_7015_7015_7015);
    let mut compared = 0;
    for k in [2.0_f32, 8.0, 20.0, 60.0] {
        let tw = SdfNode::box3d(0.1, 0.4, 0.05)
            .translate(0.25, 0.0, 0.1)
            .twist(k)
            .translate(0.0, 2.0, 0.0);
        let (c, r) = bounding_sphere(&tw).expect("bounded");
        // a competitor first in the union, so the twist operand's guard decides at every point: a
        // ring about the twist axis just outside the guard sphere. Near the ring the guard skips;
        // on the twisted surface (value 0) the ring is a little farther, so a guard that is too
        // tight there (a sphere about the child's untwisted centre) skips the minimum
        let ring_r = r + 0.02;
        let other = SdfNode::torus(ring_r, 0.01).translate(c.x, c.y, c.z);
        let node = other.union(tw.clone());
        let mut pts = Vec::new();
        for _ in 0..4096 {
            let th = rng.f() * std::f32::consts::TAU;
            let (sn, cs) = alice_det_math::sin_cos(th);
            let rad = r + 0.02 + 0.1 * rng.f();
            pts.push(c + Vec3::new(cs * rad, 0.05 * (rng.f() - 0.5), sn * rad));
        }
        pts.extend(hit_points(&tw, c, 2048, &mut rng));
        let run = |sh: &WgslShader| match GpuEvaluator::from_shader(sh) {
            Ok(g) => Some(gpu_batches::eval_in_batches(&g, &pts).expect("gpu eval")),
            Err(e) => {
                assert!(
                    std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                    "no GPU: {e}"
                );
                None
            }
        };
        let Some(b) = run(&WgslShader::transpile_with_union_bounds(
            &node,
            TranspileMode::Hardcoded,
        )) else {
            return;
        };
        let q = run(&WgslShader::transpile(&node, TranspileMode::Hardcoded)).expect("gpu");
        let skips = run(&WgslShader::transpile_union_skip_count(
            &node,
            TranspileMode::Hardcoded,
        ))
        .expect("gpu");
        let skipped: f32 = skips.iter().sum();
        assert!(
            skipped > 0.0,
            "k {k}: the twist operand's guard never skipped"
        );
        for ((p, x), y) in pts.iter().zip(&b).zip(&q) {
            if known_defect && x.to_bits() != y.to_bits() {
                let budget = contraction_budget(*p, k, ring_r, eval(&tw, *p));
                assert!(
                    x.is_finite() && y.is_finite() && (x - y).abs() <= budget,
                    "k {k} p {p}: bounded {x} vs plain {y}, |Δ| {:e} > {budget:e}",
                    (x - y).abs()
                );
                differed += 1;
            } else {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "k {k} p {p}: bounded {x} vs plain {y}"
                );
            }
            compared += 1;
        }
    }
    assert!(compared >= 4 * 4096, "compared {compared}");
    // the budget holds the measured gap and stays below a gap ten times larger
    let repro = Vec3::new(0.171_363_55, 2.020_705_2, -0.531_694_5);
    let tw2 = SdfNode::box3d(0.1, 0.4, 0.05)
        .translate(0.25, 0.0, 0.1)
        .twist(2.0)
        .translate(0.0, 2.0, 0.0);
    let at_repro = contraction_budget(repro, 2.0, 0.476_923_82 + 0.02, eval(&tw2, repro));
    assert!(
        (6.0e-8..6.0e-7).contains(&at_repro),
        "budget at the repro point {at_repro:e}"
    );
    if known_defect {
        eprintln!(
            "known defect: g3-vulkan-nocontraction: the driver fuses a multiply-add, {differed} of \
             {compared} values differ within the rounding budget"
        );
    }
}

/// The distance between one ulp-step and the next at `x`
#[cfg(feature = "gpu")]
fn ulp(x: f32) -> f32 {
    let a = x.abs();
    f32::from_bits(a.to_bits() + 1) - a
}

/// How far a driver's fusing of `a * b + c` can move the (j) scene's value at `p`, from the
/// emitted WGSL of its two operands (the ring first, then the twist): fusing one contractible op
/// changes it by at most ½ ulp of the product
/// plus one ulp of the result. The twist operand's contractible ops: the quadrant argument
/// `a · 2/π + 0.5` (moves the quadrant only within one ulp of an integer, then the reduced angle
/// moves by π/2 and the pair is the native one of another argument: 2⁻¹⁰ · |p_xz| per
/// component), the reduced angle's `− n · P2` (moves the angle, so each rotated component by
/// |p_xz| times that), and the rotation's two products and a sum per component; the box's
/// `length` (a dot product: two sums) after them. The box and the length are 1-Lipschitz, so the
/// point's move bounds the value's (`v_twist`, the twist operand's value, bounds the length's
/// argument). The ring's `length`s (one sum each). The union's `min` is
/// 1-Lipschitz in each operand: the larger of the two
#[cfg(feature = "gpu")]
fn contraction_budget(p: Vec3, k: f32, ring_r: f32, v_twist: f32) -> f32 {
    let l = p - Vec3::new(0.0, 2.0, 0.0);
    let rho = Vec3::new(l.x, 0.0, l.z).length();
    let a = k * l.y;
    let arg = a * 0.636_619_75 + 0.5;
    let n = arg.floor();
    let rot = 0.5 * ulp(rho) + ulp(std::f32::consts::SQRT_2 * rho);
    let ang = if (arg - arg.round()).abs() <= 2.0 * ulp(arg) {
        // a fused argument may pick the next quadrant: the native pair's own error, 2⁻¹⁰
        rho / 1024.0
    } else {
        rho * (0.5 * ulp(n * 0.000_483_826_8) + ulp(std::f32::consts::FRAC_PI_4 + 1e-3))
    };
    let dq = std::f32::consts::SQRT_2 * (rot + ang);
    // the length's argument is at most the box value outside (0 inside): two sums of squares
    let twist = dq + 2.0 * ulp(v_twist.abs() + dq);
    let ring = ulp(rho) + ulp((rho - ring_r).abs() + l.y.abs());
    twist.max(ring)
}

/// Whether the GPU behind `GpuEvaluator` rounds `a * b + c` once (fused): with
/// a = b = 1 + 2⁻¹² and c = −(1 + 2⁻¹¹) two roundings give 0 and one gives 2⁻²⁴ (the
/// discriminator of `tests/test_gpu_fma_contraction.rs`). `None` without a GPU
#[cfg(feature = "gpu")]
fn gpu_fuses_a_multiply_add() -> Option<bool> {
    use alice_sdf::compiled::GpuEvaluator;
    let wgsl = "struct InputPoint { x: f32, y: f32, z: f32, _pad: f32, }
struct OutputDistance { distance: f32, _pad1: f32, _pad2: f32, _pad3: f32, }
@group(0) @binding(0) var<storage, read> input_points: array<InputPoint>;
@group(0) @binding(1) var<storage, read_write> output_distances: array<OutputDistance>;
@group(0) @binding(2) var<uniform> point_count: u32;
fn sdf_eval(p: vec3<f32>) -> f32 { return p.x * p.y + p.z; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= point_count) { return; }
    let point = input_points[idx];
    output_distances[idx].distance = sdf_eval(vec3<f32>(point.x, point.y, point.z));
}
";
    let e = GpuEvaluator::from_wgsl(wgsl).ok()?.into_parts().0;
    let a = 1.0 + 1.0 / 4096.0_f32;
    let got = e
        .eval_batch(&[Vec3::new(a, a, -(1.0 + 1.0 / 2048.0))])
        .ok()?[0];
    Some(got.to_bits() != 0.0f32.to_bits())
}

/// How much the deep operand's twin is grown: more than one node's slack on the shell
/// 3.5 ≤ |p| ≤ 4 (2⁻²⁰ · 5.3 ≈ 5.05e-6), less than the deep operand's rounding there (measured
/// below the exact sphere: median 2.4e-6, 90th percentile 5.7e-6, max 1.4e-5)
const TWIN_GROWTH: f32 = 5.2e-6;

/// A deep operand: a sphere under 219 levels of rotations and reciprocal scale pairs about the
/// origin (the depth of the knee's fill stack), so its value is the sphere's up to the rounding
/// of 219 nodes: the bound's slack must grow with the depth
fn deep_operand() -> SdfNode {
    let mut n = SdfNode::sphere(0.3);
    let mut g = Rng(0xdee9_0000_0000_0219);
    for i in 0..219 {
        n = if i % 3 == 2 {
            let s = 0.7 + 0.6 * g.f();
            n.scale(s).scale(1.0 / s)
        } else {
            let axis = g.v(-1.0, 1.0).normalize_or(Vec3::Y);
            n.rotate(Quat::from_axis_angle(axis, 6.0 * g.f() - 3.0))
        };
    }
    n
}

/// (k) the deep operand against an exact twin (the plain sphere): where the deep operand's
/// rounding puts its value below the twin's, a slack that does not grow with the depth skips it
/// and the union gives the twin's larger value. With the depth's slack the bounded union is
/// bit-equal (CPU), the deep operand's bound holds, and the bound skips the deep operand next to
/// a near part
#[test]
fn a_deep_operand_keeps_the_bits() {
    // the tree is 365 nodes deep: the recursive evaluators need more than a test thread's stack
    on_big_stack(deep_cpu);
}

/// Run `f` on a thread with a 256 MiB stack (deep trees, debug builds)
fn on_big_stack(f: fn()) {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(f)
        .expect("spawn")
        .join()
        .expect("the test thread");
}

fn deep_cpu() {
    let deep = deep_operand();
    let b = bound(&deep, BoundTarget::Cpu).expect("the deep operand is bounded");
    assert!(b.depth() >= 219, "depth {}", b.depth());
    // measured: the deep operand's rounding goes up to 2.7e-6 · M below the exact sphere, more
    // than one node's budget (2⁻²⁰ ≈ 9.5e-7 · M): a slack that ignores the depth fails the bound.
    // The twin is the exact sphere grown by TWIN_GROWTH: where the deep operand is below it by
    // more than that, the deep operand is the minimum, and a slack under TWIN_GROWTH (none, or
    // one node's on the shell) skips it
    let twin = SdfNode::sphere(0.3 + TWIN_GROWTH);
    let tie = twin.clone().union(deep.clone());
    let near = SdfNode::sphere(0.05).translate(2.0, 0.0, 0.0);
    let skip = near.union(deep.clone());
    let mut rng = Rng(0x0219_0219_0219_0219);
    let (mut compared, mut below, mut skipped) = (0, 0, 0);
    for _ in 0..20_000 {
        let p = rng.v(-1.0, 1.0).normalize_or(Vec3::X) * (3.5 + 0.5 * rng.f());
        let v = eval(&deep, p);
        assert!(b.lower(p) <= v, "p {p}: bound {} > value {v}", b.lower(p));
        if v < eval(&twin, p) {
            below += 1;
        }
        assert_eq!(
            eval_union_bounded(&tie, p).to_bits(),
            eval(&tie, p).to_bits(),
            "p {p}"
        );
        compared += 1;
    }
    for _ in 0..2_000 {
        let p = Vec3::new(2.0, 0.0, 0.0) + rng.v(-0.2, 0.2);
        let (v, st) = eval_union_bounded_stats(&skip, p);
        assert_eq!(v.to_bits(), eval(&skip, p).to_bits(), "p {p}");
        skipped += st.skipped;
    }
    // the rounding of 219 nodes does go below the exact value: the tie is a real test
    assert!(
        below > 100,
        "the deep operand's value was below the twin's at {below} points"
    );
    assert!(compared == 20_000 && skipped > 1000, "skipped {skipped}");
}

/// (k) on the GPU: the same tie and skip scenes, the bounded shader's values bit-equal to the plain
/// shader's and the skip count > 0
#[cfg(feature = "gpu")]
#[test]
fn a_deep_operand_keeps_the_plain_bits_on_the_gpu() {
    on_big_stack(deep_gpu);
}

#[cfg(feature = "gpu")]
fn deep_gpu() {
    use alice_sdf::compiled::{GpuEvaluator, TranspileMode, WgslShader};
    let deep = deep_operand();
    let tie = SdfNode::sphere(0.3 + TWIN_GROWTH).union(deep.clone());
    let skip = SdfNode::sphere(0.05).translate(2.0, 0.0, 0.0).union(deep);
    let mut rng = Rng(0x0219_6e05_0219_6e05);
    let tie_pts: Vec<Vec3> = (0..20_000)
        .map(|_| rng.v(-1.0, 1.0).normalize_or(Vec3::X) * (3.5 + 0.5 * rng.f()))
        .collect();
    let skip_pts: Vec<Vec3> = (0..2_000)
        .map(|_| Vec3::new(2.0, 0.0, 0.0) + rng.v(-0.2, 0.2))
        .collect();
    let run = |sh: &WgslShader, pts: &[Vec3]| match GpuEvaluator::from_shader(sh) {
        Ok(g) => Some(gpu_batches::eval_in_batches(&g, pts).expect("gpu eval")),
        Err(e) => {
            assert!(
                std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(),
                "no GPU: {e}"
            );
            None
        }
    };
    let mut compared = 0;
    for (node, pts) in [(&tie, &tie_pts), (&skip, &skip_pts)] {
        let Some(b) = run(
            &WgslShader::transpile_with_union_bounds(node, TranspileMode::Hardcoded),
            pts,
        ) else {
            return;
        };
        let q = run(&WgslShader::transpile(node, TranspileMode::Hardcoded), pts).expect("gpu");
        for ((p, x), y) in pts.iter().zip(&b).zip(&q) {
            assert_eq!(x.to_bits(), y.to_bits(), "p {p}: bounded {x} vs plain {y}");
            compared += 1;
        }
    }
    let counts = run(
        &WgslShader::transpile_union_skip_count(&skip, TranspileMode::Hardcoded),
        &skip_pts,
    )
    .expect("gpu");
    let skipped: f32 = counts.iter().sum();
    assert!(
        skipped > 1000.0,
        "the GPU skipped the deep operand {skipped} times"
    );
    assert_eq!(compared, 22_000, "compared {compared}");
}

/// (l) the shader bound of a twist carries the native sine and cosine's slope: each twist on the
/// path lowers the bound by `δ = 2⁻¹⁰` of the distance against the CPU's (whose sine and cosine
/// keep the length). Far from the operand the two bounds differ by about `δ · |p − c|` per twist
#[test]
fn a_shader_twist_lowers_the_slope() {
    let op = SdfNode::box3d(0.1, 0.4, 0.05)
        .translate(0.25, 0.0, 0.1)
        .twist(2.0)
        .twist(-3.0);
    let cpu = bound(&op, BoundTarget::Cpu).expect("bounded");
    let gpu = bound(&op, BoundTarget::Shader).expect("bounded");
    let mut compared = 0;
    for d in [10.0_f32, 100.0, 1000.0] {
        let p = Vec3::new(d, 0.3, -0.2);
        let gap = cpu.lower(p) - gpu.lower(p);
        // two twists: 1 − (1 − δ)² ≈ 2δ of the distance
        let slope = 1.0 - 1.0 / 1024.0f32;
        let want = (1.0 - slope * slope) * d;
        assert!(
            (gap - want).abs() <= 0.01 * want,
            "at distance {d}: the shader bound is {gap} below the CPU's, want ≈ {want}"
        );
        compared += 1;
    }
    assert_eq!(compared, 3);
}

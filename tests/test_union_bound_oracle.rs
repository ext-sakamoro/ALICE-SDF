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

use alice_sdf::bound::bounding_sphere;
use alice_sdf::eval::{eval, eval_union_bounded, eval_union_bounded_stats};
use alice_sdf::prelude::*;
use glam::Quat;

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
        SdfNode::box3d(0.1, 0.1, 0.1).twist(2.0).translate(-0.2, -0.2, 0.4),
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
            assert!((h - c).length() <= r + 1e-5, "surface point {h} outside the bound");
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
    let node = near.clone().union(far.clone());
    let (c, r) = bounding_sphere(&far).expect("bounded");
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let mut compared = 0;
    for _ in 0..1024 {
        let p = c + rng.v(-1.0, 1.0) * r;
        if (p - c).length() > r {
            continue;
        }
        let want = eval(&node, p);
        assert_eq!(eval(&far, p).to_bits(), want.to_bits(), "the far operand is the minimum");
        assert_eq!(eval_union_bounded(&node, p).to_bits(), want.to_bits(), "p {p}");
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
            assert_eq!(st.evaluated + st.skipped, ops.len(), "every operand is counted once");
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
            assert!(std::env::var("ALICE_SDF_REQUIRE_GPU").is_err(), "no GPU: {e}");
            return;
        }
    };
    let got = g.eval_batch(&pts).expect("gpu eval");
    let mut compared = 0;
    for (p, v) in pts.iter().zip(&got) {
        let want = eval(&node, *p);
        assert!((v - want).abs() < 2e-6, "p {p}: gpu {v} vs cpu {want}");
        compared += 1;
    }
    assert!(compared >= ops.len() * 128, "compared {compared}");
}

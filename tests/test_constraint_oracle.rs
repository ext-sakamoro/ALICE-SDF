//! Constraint solver oracle: each system has a closed-form solution, and the
//! solved parameters are compared with it (and every constraint's residual is
//! recomputed here from the parameters, not taken from the solver).
//!
//! The incremental path is checked by bit equality: pushing solved values
//! into compiled bytecode through `ParamDependencyIndex` must evaluate
//! exactly like compiling the tree built with those values.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::OpCode;
use alice_sdf::prelude::*;

const TOL: f64 = 1e-9;

const fn p(i: u32) -> ParamId {
    ParamId::from_raw(i)
}

#[test]
fn linear_systems_reach_their_closed_form() {
    // a = 2, a + b = 5, c / b = 0.5, a·d = 12, |e − a| = 3 (e starts above a)
    let mut s = ConstraintSolver::new(vec![0.0, 1.0, 1.0, 1.0, 4.0]);
    s.fix(p(0), 2.0);
    s.sum(p(0), p(1), 5.0);
    s.ratio(p(2), p(1), 0.5);
    s.product(p(0), p(3), 12.0);
    s.distance(p(4), p(0), 3.0);
    assert_eq!(s.constraint_count(), 5);
    assert_eq!(s.param_count(), 5);

    let r = s.solve(100, 1e-12);
    let expected = [2.0, 3.0, 1.5, 6.0, 5.0];
    assert!(r.converged, "{r:?}");
    let mut compared = 0;
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (r.params[i] - e).abs() < 1e-7,
            "param {i}: {} vs {e}",
            r.params[i]
        );
        assert!((s.get(p(i as u32)) - e).abs() < 1e-7);
        compared += 1;
    }
    // residuals recomputed from the definitions
    let v = &r.params;
    let residuals = [
        v[0] - 2.0,
        v[0] + v[1] - 5.0,
        v[2] / v[1] - 0.5,
        v[0] * v[3] - 12.0,
        (v[4] - v[0]).abs() - 3.0,
    ];
    let sum_sq: f64 = residuals.iter().map(|x| x * x).sum();
    assert!(sum_sq.sqrt() < 1e-7);
    assert!(r.residual < 1e-7);
    assert!(r.iterations >= 1);
    assert_eq!(compared, 5);
}

#[test]
fn min_max_and_range() {
    let mut s = ConstraintSolver::new(vec![5.0, 1.0, 10.0]);
    s.fix(p(1), 1.0);
    // min(a, b) = 1 is already satisfied by b; max(a, c) = 7 must lower c
    s.add_constraint(Constraint {
        kind: ConstraintKind::Max {
            param_a: p(0),
            param_b: p(2),
        },
        target: 7.0,
    });
    s.add_constraint(Constraint {
        kind: ConstraintKind::Min {
            param_a: p(0),
            param_b: p(1),
        },
        target: 1.0,
    });
    // a must stay within 5 ± 0.5 (it starts inside, so it does not move)
    s.range(p(0), 5.0, 0.5);
    let r = s.solve(100, 1e-12);
    assert!(r.converged, "{r:?}");
    assert!((r.params[0] - 5.0).abs() < TOL);
    assert!((r.params[1] - 1.0).abs() < 1e-7);
    assert!((r.params[2] - 7.0).abs() < 1e-7);

    // Range from outside lands on the nearer bound: 9 → 5.5
    let mut s = ConstraintSolver::new(vec![9.0]);
    s.range(p(0), 5.0, 0.5);
    let r = s.solve(50, 1e-12);
    assert!((r.params[0] - 5.5).abs() < 1e-7, "{r:?}");
}

#[test]
fn set_get_and_ids() {
    let mut s = ConstraintSolver::new(vec![0.0; 3]);
    s.set(p(2), 4.25);
    assert_eq!(s.get(p(2)), 4.25);
    assert_eq!(p(7).as_u32(), 7);
    assert_eq!(p(7).as_index(), 7);
    // no constraints → converged at once, parameters untouched
    let r = s.solve(10, 1e-9);
    assert!(r.converged && r.iterations == 0 && r.params == vec![0.0, 0.0, 4.25]);
}

#[test]
fn incremental_update_is_bit_identical_to_recompiling() {
    // Union(Sphere(r), Translate(2,0,0)·Box(hx, hy, hz))
    let scene = |r: f32, hx: f32| {
        SdfNode::sphere(r).union(SdfNode::box3d(2.0 * hx, 1.0, 1.0).translate(2.0, 0.0, 0.0))
    };
    let mut compiled = CompiledSdf::compile(&scene(1.0, 0.5));
    let sphere = compiled
        .instructions()
        .iter()
        .position(|i| i.opcode == OpCode::Sphere)
        .unwrap();
    let cube = compiled
        .instructions()
        .iter()
        .position(|i| i.opcode == OpCode::Box3d)
        .unwrap();

    let mut index = ParamDependencyIndex::new();
    index.bind(p(0), &compiled, sphere, 0).unwrap();
    index.bind(p(1), &compiled, cube, 0).unwrap();
    assert_eq!(index.bindings_of(p(0)), &[InstructionSlot::new(sphere, 0)]);
    assert_eq!(index.bindings_of(p(1)), &[InstructionSlot::new(cube, 0)]);
    assert!(index.bindings_of(p(9)).is_empty());

    // Solve r + hx = 1.2 with hx fixed at 0.45 → r = 0.75 (exact in f32).
    let mut s = ConstraintSolver::new(vec![1.0, 0.5]);
    s.fix(p(1), 0.45);
    s.sum(p(0), p(1), 1.2);
    let r = s.solve(50, 1e-12);
    assert!(r.converged);
    index.apply_all(&mut compiled, &s).unwrap();

    let fresh = CompiledSdf::compile(&scene(s.get(p(0)) as f32, s.get(p(1)) as f32));
    let mut compared = 0;
    for i in 0..64 {
        let t = i as f32 * 0.37;
        let q = Vec3::new(t.sin() * 3.0, (t * 1.3).cos() * 2.0, (t * 0.7).sin());
        let a = eval_compiled(&compiled, q);
        let b = eval_compiled(&fresh, q);
        assert_eq!(a.to_bits(), b.to_bits(), "q={q}: {a} vs {b}");
        compared += 1;
    }
    // and the sphere radius really is 0.75: d(origin) = −0.75
    assert!((eval_compiled(&compiled, Vec3::ZERO) + 0.75).abs() < 1e-6);
    assert!(compared > 0);
}

//! Parametric design with `ConstraintSolver`.
//!
//! A bracket made of a post (sphere radius `r`) and a plate (box half-width
//! `w`, height `h`) is described by design rules instead of numbers:
//! the plate is twice as wide as the post radius, the plate area `2w·h` is
//! fixed, and the post sits within a tolerance band. The solver finds the
//! parameters, the dependency index pushes them into compiled bytecode, and
//! the result is checked against the closed-form solution.
//!
//! # Running
//! ```bash
//! cargo run --example constraint_solver
//! ```
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::OpCode;
use alice_sdf::prelude::*;

fn main() {
    println!("ALICE-SDF — constraint solver");
    println!("=============================");

    let (r, w, h, gap) = (
        ParamId::from_raw(0),
        ParamId::from_raw(1),
        ParamId::from_raw(2),
        ParamId::from_raw(3),
    );
    let mut solver = ConstraintSolver::new(vec![1.0, 1.0, 1.0, 0.0]);
    solver.fix(r, 0.5); //             r = 0.5
    solver.ratio(w, r, 2.0); //         w / r = 2   → w = 1
    solver.product(w, h, 0.25); //      w · h = 0.25 → h = 0.25
    solver.distance(gap, w, 0.5); //    |gap − w| = 0.5 (gap starts below w → 0.5)
    solver.sum(r, h, 0.75); //          redundant but consistent: r + h = 0.75
    solver.range(r, 0.5, 0.01); //      r within 0.5 ± 0.01
    println!(
        "{} parameters, {} constraints",
        solver.param_count(),
        solver.constraint_count()
    );

    let result = solver.solve(100, 1e-12);
    println!(
        "solve: converged={} after {} iterations, residual {:.3e}",
        result.converged, result.iterations, result.residual
    );
    for id in [r, w, h, gap] {
        println!("  p{} = {:.6}", id.as_u32(), solver.get(id));
    }
    assert!(result.converged);
    for (id, expected) in [(r, 0.5), (w, 1.0), (h, 0.25), (gap, 0.5)] {
        assert!((solver.get(id) - expected).abs() < 1e-7, "p{id}");
    }

    // Push r and w into compiled bytecode: Union(Sphere(r), Box(w, h, h) at x = 2).
    let scene = SdfNode::sphere(1.0).union(SdfNode::box3d(2.0, 0.5, 0.5).translate(2.0, 0.0, 0.0));
    let mut compiled = CompiledSdf::compile(&scene);
    let find = |op| {
        compiled
            .instructions()
            .iter()
            .position(|i| i.opcode == op)
            .expect("opcode in scene")
    };
    let (sphere_at, box_at) = (find(OpCode::Sphere), find(OpCode::Box3d));
    let mut index = ParamDependencyIndex::new();
    index.bind(r, &compiled, sphere_at, 0).expect("bind r");
    index.bind(w, &compiled, box_at, 0).expect("bind w");
    println!("bindings of r: {:?}", index.bindings_of(r));
    assert_eq!(index.bindings_of(r), &[InstructionSlot::new(sphere_at, 0)]);
    index.apply_all(&mut compiled, &solver).expect("apply");

    // d(origin) = −r; d(3.5, 0, 0) = 3.5 − (2 + w) = 0.5
    let d0 = eval_compiled(&compiled, Vec3::ZERO);
    let d1 = eval_compiled(&compiled, Vec3::new(3.5, 0.0, 0.0));
    println!("d(origin) = {d0}, d(3.5,0,0) = {d1}");
    assert!((d0 + 0.5).abs() < 1e-6);
    assert!((d1 - 0.5).abs() < 1e-6);

    // A one-off edit through the same index.
    solver.set(r, 0.25);
    index.apply_all(&mut compiled, &solver).expect("apply");
    assert!((eval_compiled(&compiled, Vec3::ZERO) + 0.25).abs() < 1e-6);
    println!("all checks passed");
}

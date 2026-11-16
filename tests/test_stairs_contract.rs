//! A staircase has at most `MAX_STAIR_STEPS` steps: refused at construction (`try_stairs`), at load (the stored
//! `n_steps` field), at compile and at every GPU evaluator construction; a hand-built literal over the maximum is
//! out of contract and bounded by the evaluators' loop cap.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::primitives::{StairsError, MAX_STAIR_STEPS};
use alice_sdf::SdfNode;
use glam::Vec3;

/// The out-of-contract tree under test: a `Stairs` literal (as a parser that builds the variant directly makes it).
/// The helpers cannot build it (`try_stairs` refuses the count), which is the point of the test, so the literal is
/// written here on purpose.
const fn literal(n_steps: f32) -> SdfNode {
    SdfNode::Stairs {
        step_width: 0.3,
        step_height: 0.2,
        n_steps,
        half_depth: 0.25,
    }
}

#[test]
fn try_stairs_accepts_the_maximum_and_refuses_one_more() {
    assert!(SdfNode::try_stairs(0.3, 0.2, MAX_STAIR_STEPS, 0.5).is_ok());
    let e = SdfNode::try_stairs(0.3, 0.2, MAX_STAIR_STEPS + 1, 0.5).unwrap_err();
    assert_eq!(
        e,
        StairsError::TooManySteps {
            n_steps: (MAX_STAIR_STEPS + 1) as f32
        }
    );
    assert!(e.to_string().contains("MAX_STAIR_STEPS"), "{e}");
}

#[test]
fn a_stored_count_over_the_maximum_is_refused_at_load() {
    // a tree serialised with a valid count, then the stored count raised past the maximum
    let ok = SdfNode::try_stairs(0.3, 0.2, 5, 0.5).unwrap();
    let text = serde_json::to_string(&ok).unwrap();
    assert!(text.contains("\"n_steps\":5.0"), "{text}");
    assert!(serde_json::from_str::<SdfNode>(&text).is_ok());
    let over = text.replace(
        "\"n_steps\":5.0",
        &format!("\"n_steps\":{}.0", MAX_STAIR_STEPS + 1),
    );
    let e = serde_json::from_str::<SdfNode>(&over)
        .unwrap_err()
        .to_string();
    assert!(e.contains("stairs:") && e.contains("over the most"), "{e}");
    let at_max = text.replace(
        "\"n_steps\":5.0",
        &format!("\"n_steps\":{MAX_STAIR_STEPS}.0"),
    );
    assert!(serde_json::from_str::<SdfNode>(&at_max).is_ok());
}

#[test]
fn a_literal_over_the_maximum_is_refused_at_compile() {
    assert!(CompiledSdf::try_compile(&literal(MAX_STAIR_STEPS as f32)).is_ok());
    let e = CompiledSdf::try_compile(&literal(MAX_STAIR_STEPS as f32 + 1.0)).unwrap_err();
    assert!(e.to_string().contains("stairs:"), "{e}");
    // also inside a larger tree (the walk reaches it)
    let deep = SdfNode::sphere(1.0).union(literal(f32::INFINITY).translate(1.0, 0.0, 0.0));
    let e = CompiledSdf::try_compile(&deep).unwrap_err();
    assert!(e.to_string().contains("not finite"), "{e}");
}

#[test]
fn a_literal_over_the_maximum_evaluated_directly_is_bounded_by_the_loop_cap() {
    // out of contract: the CPU evaluator stops at the cap, so the value is that of the first MAX_STAIR_STEPS steps
    // at the literal's centring (the same as sdf_stairs with the larger count), and it returns
    let n = MAX_STAIR_STEPS as f32 + 1000.0;
    let p = Vec3::new(0.0, 0.0, 0.0);
    let over = alice_sdf::eval(&literal(n), p);
    let direct = alice_sdf::primitives::sdf_stairs(p, 0.3, 0.2, n, 0.25);
    assert_eq!(over.to_bits(), direct.to_bits());
    assert!(over.is_finite());
}

#[cfg(feature = "gpu")]
#[test]
fn every_gpu_evaluator_construction_refuses_it_before_any_device_work() {
    use alice_sdf::compiled::{GpuError, GpuEvaluator};
    let bad = SdfNode::sphere(1.0).union(literal(MAX_STAIR_STEPS as f32 + 1.0));
    for (name, r) in [
        ("new", GpuEvaluator::new(&bad).err()),
        ("new_checked", GpuEvaluator::new_checked(&bad).err()),
        ("new_dynamic", GpuEvaluator::new_dynamic(&bad).err()),
    ] {
        match r {
            Some(GpuError::InvalidFormat(m)) => assert!(m.contains("stairs:"), "{name}: {m}"),
            other => {
                panic!("{name}: refused as InvalidFormat before any device work, got {other:?}")
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn a_polygon_over_the_gpu_maximum_is_refused_at_gpu_evaluator_construction() {
    use alice_sdf::compiled::{
        check_gpu_tree, GpuEvaluator, GpuTreeRefusal, MAX_GPU_POLYGON_VERTICES,
    };
    let ring = |n: usize| {
        let v: Vec<glam::Vec2> = (0..n)
            .map(|i| {
                // a convex arc of y = x^2 (no trigonometry)
                let x = -1.0 + 2.0 * i as f32 / n as f32;
                glam::Vec2::new(x, x * x)
            })
            .collect();
        SdfNode::polygon_2d(v, 0.2)
    };
    assert!(check_gpu_tree(&ring(MAX_GPU_POLYGON_VERTICES)).is_ok());
    let big = SdfNode::sphere(1.0).union(ring(MAX_GPU_POLYGON_VERTICES + 1));
    assert_eq!(
        check_gpu_tree(&big),
        Err(GpuTreeRefusal::PolygonVertices {
            vertices: MAX_GPU_POLYGON_VERTICES + 1
        })
    );
    match GpuEvaluator::new(&big) {
        Err(e) => assert!(e.to_string().contains("polygon:"), "{e}"),
        Ok(_) => panic!("refused before any device work"),
    }
}

//! `SdfSkinning` maps a posed point back to the rest pose.
//!
//! oracle: the forward linear blend skinning of one bone, posed = current · inv_bind · rest
//! (`BoneTransform`'s documented meanings: `inv_bind_pose` world → bone, `current_pose`
//! bone → world), built here in f64 with glam's `DMat4`, never through the crate. A point posed
//! that way must come back to its rest point, and the skinned distance at the posed point must be
//! the rest shape's distance at the rest point. Checked on the interpreter (`eval`), the compiled
//! evaluator (`eval_compiled`) and `sdf_skinning` itself.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::eval::eval;
use alice_sdf::transforms::skinning::{sdf_skinning, BoneTransform};
use alice_sdf::types::SdfNode;
use glam::{DMat4, DVec3, Vec3};

fn to_f32(m: DMat4) -> [f32; 16] {
    m.to_cols_array().map(|v| v as f32)
}

struct Case {
    name: &'static str,
    inv_bind: DMat4,
    current: DMat4,
    rest: DVec3,
}

fn cases() -> Vec<Case> {
    vec![
        // inv_bind = I: only the current pose has to be undone
        Case {
            name: "rot_z(0.5), inv_bind = I",
            inv_bind: DMat4::IDENTITY,
            current: DMat4::from_rotation_z(0.5),
            rest: DVec3::new(10.0, 0.0, 0.0),
        },
        // a bone bound away from the origin: the bind pose has to be undone too
        Case {
            name:
                "bound at (1, 0, 0) turned by 0.3 about y, posed at (2, 1, 0) turned by 0.5 about z",
            inv_bind: (DMat4::from_translation(DVec3::new(1.0, 0.0, 0.0))
                * DMat4::from_rotation_y(0.3))
            .inverse(),
            current: DMat4::from_translation(DVec3::new(2.0, 1.0, 0.0))
                * DMat4::from_rotation_z(0.5),
            rest: DVec3::new(1.5, 0.25, -0.5),
        },
    ]
}

fn bone(c: &Case) -> BoneTransform {
    BoneTransform {
        inv_bind_pose: to_f32(c.inv_bind),
        current_pose: to_f32(c.current),
        weight: 1.0,
    }
}

fn posed(c: &Case) -> Vec3 {
    (c.current * c.inv_bind).transform_point3(c.rest).as_vec3()
}

#[test]
fn the_fd_case_poses_to_the_stated_point() {
    // the brief's numbers: rest (10, 0, 0) under rot_z(0.5) is (8.776, 4.794)
    let p = posed(&cases()[0]);
    assert!(
        (p - Vec3::new(8.775_826, 4.794_255, 0.0)).length() < 1e-4,
        "{p}"
    );
}

#[test]
fn sdf_skinning_returns_the_rest_point() {
    for c in cases() {
        let (q, _) = sdf_skinning(posed(&c), &[bone(&c)]);
        let want = c.rest.as_vec3();
        assert!(
            (q - want).length() < 1e-4,
            "{}: posed {} came back to {q}, rest is {want}",
            c.name,
            posed(&c)
        );
    }
}

#[test]
fn the_skinned_distance_is_the_rest_distance_on_every_evaluator() {
    let centre = Vec3::new(1.0, 0.5, -0.25);
    let rest_shape = SdfNode::sphere(0.75).translate(centre.x, centre.y, centre.z);
    for c in cases() {
        let node = rest_shape.clone().sdf_skinning(vec![bone(&c)]);
        let compiled = CompiledSdf::compile(&node);
        let want = (c.rest.as_vec3() - centre).length() - 0.75;
        let p = posed(&c);
        for (path, got) in [
            ("eval", eval(&node, p)),
            ("eval_compiled", eval_compiled(&compiled, p)),
        ] {
            assert!(
                (got - want).abs() < 1e-4,
                "{} on {path}: distance {got} at the posed point, rest distance {want}",
                c.name
            );
        }
    }
}

/// The shader transpilers write the bits of `rest_from_posed` (the matrix the CPU evaluators use)
/// as literals, and no longer the raw `current_pose` / `inv_bind_pose` entries that only these
/// matrices have: every transpiler starts from the same 16 floats as the CPU.
#[cfg(any(feature = "gpu", feature = "glsl", feature = "hlsl"))]
#[test]
// each push is behind its own feature, so the list cannot be one `vec![]`
#[allow(clippy::vec_init_then_push)]
fn the_transpilers_write_the_cpu_rest_matrix() {
    use alice_sdf::compiled::transpiler_common::lit;
    use alice_sdf::transforms::skinning::rest_from_posed;
    let c = &cases()[1];
    let b = bone(c);
    let m = rest_from_posed(&b)
        .expect("an invertible pose")
        .to_cols_array();
    let node = SdfNode::sphere(0.75).sdf_skinning(vec![b.clone()]);
    let mut sources: Vec<(&str, String)> = Vec::new();
    #[cfg(feature = "gpu")]
    sources.push((
        "wgsl",
        alice_sdf::compiled::WgslShader::transpile(
            &node,
            alice_sdf::compiled::TranspileMode::Hardcoded,
        )
        .source,
    ));
    #[cfg(feature = "glsl")]
    sources.push((
        "glsl",
        alice_sdf::compiled::GlslShader::transpile(
            &node,
            alice_sdf::compiled::GlslTranspileMode::Hardcoded,
        )
        .source,
    ));
    #[cfg(feature = "hlsl")]
    sources.push((
        "hlsl",
        alice_sdf::compiled::HlslShader::transpile(
            &node,
            alice_sdf::compiled::HlslTranspileMode::Hardcoded,
        )
        .source,
    ));
    assert!(!sources.is_empty(), "no transpiler compared");
    // entries of the raw poses whose magnitude is no entry's of the rest matrix (a literal is
    // matched as text, and `-x` contains `x`): none may be written
    let raw: Vec<f32> = b
        .current_pose
        .iter()
        .chain(&b.inv_bind_pose)
        .copied()
        .filter(|v| !m.iter().any(|e| e.abs() == v.abs()) && v.fract() != 0.0)
        .collect();
    assert!(!raw.is_empty(), "the case has no raw entry to tell apart");
    for (name, src) in &sources {
        for v in m.iter().filter(|v| v.fract() != 0.0) {
            assert!(
                src.contains(&lit(*v)),
                "{name}: rest matrix entry {} not written",
                lit(*v)
            );
        }
        for v in &raw {
            assert!(
                !src.contains(&lit(*v)),
                "{name}: raw pose entry {} written",
                lit(*v)
            );
        }
    }
}

/// A bone whose pose cannot be inverted (a zero scale) has no rest pose to return to: it
/// contributes nothing, like a bone of zero weight. With only such bones the point is unchanged;
/// next to a valid bone only the valid bone counts. Same on every CPU evaluator, never NaN.
#[test]
fn a_bone_that_cannot_be_inverted_contributes_nothing() {
    let flat = BoneTransform {
        inv_bind_pose: DMat4::IDENTITY.to_cols_array().map(|v| v as f32),
        current_pose: to_f32(DMat4::from_scale(DVec3::new(1.0, 0.0, 1.0))),
        weight: 1.0,
    };
    let p = Vec3::new(0.5, -0.25, 2.0);
    let (q, _) = sdf_skinning(p, std::slice::from_ref(&flat));
    assert_eq!(q, p, "only a singular bone: the point is unchanged");
    let c = &cases()[1];
    let (q, _) = sdf_skinning(posed(c), &[flat.clone(), bone(c)]);
    assert!(
        (q - c.rest.as_vec3()).length() < 1e-4,
        "the singular bone counted: {q}"
    );
    let node = SdfNode::sphere(0.75).sdf_skinning(vec![flat]);
    let compiled = CompiledSdf::compile(&node);
    let want = p.length() - 0.75;
    for (path, got) in [
        ("eval", eval(&node, p)),
        ("eval_compiled", eval_compiled(&compiled, p)),
    ] {
        assert!(
            got.is_finite() && (got - want).abs() < 1e-6,
            "{path}: {got}, want {want}"
        );
    }
}

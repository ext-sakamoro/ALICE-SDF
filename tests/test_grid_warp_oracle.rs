//! Grid warp oracle (`SdfNode::GridWarp`, `transforms::GridWarp`)
//!
//! oracle: closed forms independent of the implementation
//! - a grid whose every node holds the rotation `R` is the rigid map `x = R p`, so a sphere of
//!   radius r at c warped by it is `|R p − c| − r = |p − Rᵀ c| − r` (sphere distance, closed form)
//! - the node is 1-Lipschitz for a 1-Lipschitz child (measured by difference quotients on a grid
//!   whose maps rotate with height, so the blend term is active), inside and outside the box
//! - outside the box the value is a lower bound of the true distance when the warped surface is
//!   inside the box (rigid grid: the true distance is the closed form above)
//! - the GPU (node data buffer, binding 4) gives the CPU value (`ALICE_SDF_REQUIRE_GPU` makes a
//!   missing adapter a failure)

use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};

fn grid(dims: [u32; 3], cell: f32, origin: [f32; 3], map: impl Fn(Vec3) -> [f32; 12]) -> GridWarp {
    let mut maps = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = Vec3::from(origin) + Vec3::new(i as f32, j as f32, k as f32) * cell;
                maps.extend_from_slice(&map(p));
            }
        }
    }
    GridWarp::new(GridWarpData {
        origin,
        cell,
        dims,
        maps,
    })
    .expect("valid grid")
}

fn rot_y(a: f32) -> [f32; 12] {
    let (s, c) = alice_det_math::sin_cos(a);
    [c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0]
}

/// points on a lattice through the box and 40 % beyond it on every side
fn points(lo: Vec3, hi: Vec3, n: usize) -> Vec<Vec3> {
    let ext = (hi - lo) * 0.4;
    let (a, b) = (lo - ext, hi + ext);
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let t = Vec3::new(i as f32 + 0.37, j as f32 + 0.61, k as f32 + 0.13) / n as f32;
                out.push(a + (b - a) * t);
            }
        }
    }
    out
}

const C: Vec3 = Vec3::new(0.03, 0.01, -0.02);
const R: f32 = 0.05;

fn sphere_at_c() -> SdfNode {
    SdfNode::sphere(R).translate(C.x, C.y, C.z)
}

#[test]
fn a_rotation_grid_is_the_rotated_sphere_inside_the_box() {
    let a = 0.7f32;
    let w = grid([5, 5, 5], 0.05, [-0.1; 3], |_| rot_y(a));
    let node = sphere_at_c().grid_warp(w);
    // closed form: x = R p with R the rotation about y by a, so the surface is the sphere at Rᵀ c
    let (s, c) = alice_det_math::sin_cos(a);
    let rt_c = Vec3::new(c * C.x - s * C.z, C.y, s * C.x + c * C.z);
    let mut n = 0;
    let mut worst = 0.0f32;
    for p in points(Vec3::splat(-0.1), Vec3::splat(0.1), 14) {
        if p.abs().max_element() > 0.1 {
            continue;
        }
        let want = (p - rt_c).length() - R;
        worst = worst.max((eval(&node, p) - want).abs());
        n += 1;
    }
    assert!(n > 500, "compared {n}");
    assert!(worst < 2e-6, "max |error| {worst}");
}

#[test]
fn outside_the_box_the_value_is_a_lower_bound() {
    let a = -0.4f32;
    let w = grid([5, 5, 5], 0.05, [-0.1; 3], |_| rot_y(a));
    let node = sphere_at_c().grid_warp(w);
    let (s, c) = alice_det_math::sin_cos(a);
    let rt_c = Vec3::new(c * C.x - s * C.z, C.y, s * C.x + c * C.z);
    let mut n = 0;
    for p in points(Vec3::splat(-0.1), Vec3::splat(0.1), 14) {
        if p.abs().max_element() <= 0.1 {
            continue;
        }
        let truth = (p - rt_c).length() - R;
        let v = eval(&node, p);
        assert!(
            v >= 0.0 && v <= truth + 1e-6,
            "p {p}: value {v}, distance {truth}"
        );
        n += 1;
    }
    assert!(n > 1000, "compared {n}");
}

#[test]
fn the_node_is_one_lipschitz_with_a_blended_grid() {
    // the maps rotate with height: neighbouring nodes differ, the bound must absorb the blend
    let w = grid([6, 9, 6], 0.04, [-0.1, -0.16, -0.1], |p| rot_y(p.y * 6.0));
    assert!(
        w.lipschitz() > 1.02,
        "the blend must raise the bound: {}",
        w.lipschitz()
    );
    let node = SdfNode::capsule(
        Vec3::new(0.0, -0.12, 0.02),
        Vec3::new(0.0, 0.12, 0.02),
        0.03,
    )
    .grid_warp(w);
    let dirs = [
        Vec3::new(1.0, 0.3, -0.2),
        Vec3::new(-0.2, 1.0, 0.5),
        Vec3::new(0.4, -0.6, 1.0),
    ]
    .map(Vec3::normalize);
    let mut n = 0;
    let mut worst = 0.0f32;
    for p in points(Vec3::new(-0.1, -0.16, -0.1), Vec3::new(0.1, 0.16, 0.1), 16) {
        for d in &dirs {
            let h = 1e-3;
            let q = (eval(&node, p + *d * h) - eval(&node, p)).abs() / h;
            worst = worst.max(q);
            n += 1;
        }
    }
    assert!(n > 10_000, "compared {n}");
    assert!(worst <= 1.0 + 1e-3, "difference quotient {worst} above 1");
    assert!(worst > 0.5, "the field must have a slope ({worst})");
}

#[test]
fn the_shader_text_declares_the_data_binding_only_with_node_data() {
    use alice_sdf::compiled::{TranspileMode, WgslShader};
    let w = grid([3, 3, 3], 0.1, [-0.1; 3], |_| rot_y(0.2));
    let node = sphere_at_c().grid_warp(w);
    let plain = WgslShader::transpile(&node, TranspileMode::Hardcoded);
    assert!(plain.node_data.is_empty());
    assert!(!plain.to_compute_shader().contains("@binding(4)"));
    assert!(
        plain.source.contains("alice_gw_"),
        "constants named after the node"
    );
    let buf = WgslShader::transpile_with_node_data(&node, TranspileMode::Hardcoded);
    assert_eq!(buf.node_data.len(), 27 * 12);
    assert!(buf
        .to_compute_shader()
        .contains("@group(0) @binding(4) var<storage, read> alice_node_data"));
    assert!(
        !buf.source.contains("array<f32, 324>"),
        "the maps must not be in the text"
    );
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_gives_the_cpu_value() {
    use alice_sdf::compiled::GpuEvaluator;
    let w = grid([6, 9, 6], 0.04, [-0.1, -0.16, -0.1], |p| rot_y(p.y * 6.0));
    let node = SdfNode::capsule(
        Vec3::new(0.0, -0.12, 0.02),
        Vec3::new(0.0, 0.12, 0.02),
        0.03,
    )
    .grid_warp(w)
    .union(SdfNode::sphere(0.02).translate(0.2, 0.0, 0.0));
    let gpu = match GpuEvaluator::new(&node) {
        Ok(g) => g,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU evaluator failed: {e}"
            );
            eprintln!("skipped: {e}");
            return;
        }
    };
    let pts = points(Vec3::new(-0.1, -0.16, -0.1), Vec3::new(0.1, 0.16, 0.1), 24);
    let got = gpu.eval_batch(&pts).expect("gpu eval");
    assert_eq!(got.len(), pts.len());
    let mut worst = 0.0f32;
    for (p, g) in pts.iter().zip(&got) {
        worst = worst.max((eval(&node, *p) - g).abs());
    }
    assert!(pts.len() > 10_000, "compared {}", pts.len());
    assert!(worst < 2e-6, "max |gpu − cpu| {worst}");
}

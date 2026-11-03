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

use alice_sdf::interval::eval_lipschitz;
use alice_sdf::prelude::*;
use alice_sdf::transforms::{GridWarp, GridWarpData};

fn grid(dims: [u32; 3], cell: f32, origin: [f32; 3], map: impl Fn(Vec3) -> [f32; 12]) -> GridWarp {
    grid_reach(dims, cell, origin, 4.0 * cell, map)
}

fn grid_reach(
    dims: [u32; 3],
    cell: f32,
    origin: [f32; 3],
    reach: f32,
    map: impl Fn(Vec3) -> [f32; 12],
) -> GridWarp {
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
        reach,
        floor: Vec::new(),
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
    let lmax = w.node_lambda().iter().copied().fold(0.0f32, f32::max);
    assert!(lmax > 1.02, "the blend must raise the node bounds: {lmax}");
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
    assert!(
        worst <= eval_lipschitz(&node),
        "difference quotient {worst} above the claim {}",
        eval_lipschitz(&node)
    );
    assert!(worst > 0.5, "the field must have a slope ({worst})");
}

#[cfg(feature = "gpu")]
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
    assert_eq!(
        buf.node_data.len(),
        27 * 12 + 27,
        "12 map floats and Λ per node"
    );
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

/// A grid that is the identity near a sphere and blends wild rotations far from it (beyond 0.12
/// from the centre): the far blend has a large slope, and must not shrink the value near the
/// surface. oracle: near the surface (|d| < 0.01) the identity region gives the sphere's closed-form
/// distance, so the value must be within 1 % of it.
fn sphere_with_wild_far_field() -> (SdfNode, f32) {
    // reach one cell: the node bounds look 3 cells around (cube), so cells within 5 cells
    // (index cube, 0.28 on the diagonal) of the surface band see only the identity
    let w = grid_reach([29, 29, 29], 0.025, [-0.35; 3], 0.025, |p| {
        let r = p.length();
        if r < 0.3 {
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        } else {
            rot_y(37.0 * p.x + 53.0 * p.z)
        }
    });
    (SdfNode::sphere(0.05).grid_warp(w), 0.05)
}

#[test]
fn a_far_blend_does_not_shrink_the_value_near_the_surface() {
    let (node, r) = sphere_with_wild_far_field();
    let mut n = 0;
    let mut worst = 0.0f32;
    for p in points(Vec3::splat(-0.06), Vec3::splat(0.06), 20) {
        let want = p.length() - r;
        if want.abs() > 0.01 || want.abs() < 1e-4 {
            continue;
        }
        worst = worst.max((eval(&node, p) / want - 1.0).abs());
        n += 1;
    }
    assert!(n > 300, "compared {n}");
    assert!(
        worst < 0.01,
        "value / distance off by {worst} near the surface"
    );
}

/// oracle (the sphere-tracing contract, independent of the bound's formula): a value v at p
/// promises no surface within |v|, so the sign is constant on the ball of radius |v| (sampled on
/// 26 directions × 4 radii)
#[test]
fn the_value_is_a_distance_lower_bound() {
    let (node, _) = sphere_with_wild_far_field();
    let w = grid([6, 9, 6], 0.04, [-0.1, -0.16, -0.1], |p| rot_y(p.y * 6.0));
    let bent = SdfNode::capsule(
        Vec3::new(0.0, -0.12, 0.02),
        Vec3::new(0.0, 0.12, 0.02),
        0.03,
    )
    .grid_warp(w);
    let mut dirs = Vec::new();
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                if (x, y, z) != (0, 0, 0) {
                    dirs.push(Vec3::new(x as f32, y as f32, z as f32).normalize());
                }
            }
        }
    }
    let mut n = 0;
    for (tree, lo, hi) in [
        (&node, Vec3::splat(-0.35), Vec3::splat(0.35)),
        (
            &bent,
            Vec3::new(-0.12, -0.2, -0.12),
            Vec3::new(0.12, 0.2, 0.12),
        ),
    ] {
        for p in points(lo, hi, 9) {
            let v = eval(tree, p);
            for d in &dirs {
                for f in [0.25f32, 0.5, 0.75, 0.98] {
                    let q = p + *d * (v.abs() * f);
                    let u = eval(tree, q);
                    assert!(
                        u == 0.0 || v == 0.0 || u.signum() == v.signum(),
                        "p {p} value {v}: sign flips at {q} ({u})"
                    );
                    n += 1;
                }
            }
        }
    }
    assert!(n > 100_000, "compared {n}");
}

/// A steep slab: translation-only maps, 0 below y0 = 0.05 and −0.15 from y0 + h up, so in the
/// one cell [y0, y0 + h] (h = 0.025) the warp compresses y by 1 − 0.15 / h = −5; the child is the
/// half space y < 0 (a large box with its top face at 0). oracle (closed form):
/// x_y = p_y below y0, 0.05 − 5 (p_y − y0) in the slab, p_y − 0.15 above it, so the surfaces are
/// y = 0, y = 0.06 and y = 0.15, and a point at 0.04 has the value 0.04 / Λ but lies 0.02 from
/// the surface at 0.06: only a bound that sees the slab from a neighbouring cell keeps it safe.
fn steep_slab(reach: f32) -> (SdfNode, f32) {
    let h = 0.025;
    let w = grid_reach([3, 13, 3], h, [-0.025, -0.1, -0.025], reach, |p| {
        let t = if p.y > 0.05 + 1e-4 { -0.15 } else { 0.0 };
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, t, 0.0, 0.0, 1.0, 0.0]
    });
    (
        SdfNode::box3d(2.0, 2.0, 2.0)
            .translate(0.0, -1.0, 0.0)
            .grid_warp(w),
        reach,
    )
}

fn slab_distance(y: f32) -> f32 {
    // the closed-form surfaces of `steep_slab` and the sign of x_y
    let x = if y < 0.05 {
        y
    } else if y < 0.075 {
        0.05 - 5.0 * (y - 0.05)
    } else {
        y - 0.15
    };
    let d = [y.abs(), (y - 0.06).abs(), (y - 0.15).abs()]
        .into_iter()
        .fold(f32::INFINITY, f32::min);
    if x < 0.0 {
        -d
    } else {
        d
    }
}

#[test]
fn a_steep_slab_never_gives_more_than_the_distance() {
    for reach in [0.01f32, 0.025, 0.08] {
        let (node, r) = steep_slab(reach);
        let mut n = 0;
        let mut clamped = 0;
        for j in 0..400 {
            let y = -0.09 + 0.285 * (j as f32 + 0.37) / 400.0; // inside the box (y ≤ 0.2): the clamp is the inside contract
            for (x, z) in [(0.0f32, 0.0f32), (0.011, -0.007), (-0.02, 0.019)] {
                let v = eval(&node, Vec3::new(x, y, z));
                let truth = slab_distance(y);
                assert!(
                    v.abs() <= truth.abs() + 1e-6,
                    "reach {reach}, y {y}: value {v}, distance {truth}"
                );
                assert!(
                    v == 0.0 || truth == 0.0 || v.signum() == truth.signum(),
                    "reach {reach}, y {y}: sign of {v} vs {truth}"
                );
                assert!(
                    v.abs() <= r + 1e-7,
                    "reach {reach}, y {y}: |value| {v} above the reach"
                );
                if v.abs() >= r - 1e-6 {
                    clamped += 1;
                }
                n += 1;
            }
        }
        assert!(n == 1200, "compared {n}");
        if reach < 0.05 {
            assert!(
                clamped > 50,
                "reach {reach}: the clamp must be active somewhere ({clamped})"
            );
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_gives_the_cpu_value_on_the_steep_slab() {
    use alice_sdf::compiled::GpuEvaluator;
    let (node, _) = steep_slab(0.01);
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
    let pts = points(
        Vec3::new(-0.025, -0.1, -0.025),
        Vec3::new(0.025, 0.2, 0.025),
        24,
    );
    let got = gpu.eval_batch(&pts).expect("gpu eval");
    let mut worst = 0.0f32;
    let mut clamped = 0;
    for (p, g) in pts.iter().zip(&got) {
        let c = eval(&node, *p);
        worst = worst.max((c - g).abs());
        if c.abs() >= 0.01 - 1e-6 {
            clamped += 1;
        }
    }
    assert!(
        pts.len() > 10_000 && clamped > 1000,
        "compared {}, clamped {clamped}",
        pts.len()
    );
    assert!(worst < 2e-6, "max |gpu − cpu| {worst}");
}

/// oracle: the claimed slope (`eval_lipschitz`) bounds the measured difference quotients where
/// `Λ` varies (the steep slab: `Λ` jumps from 1 to about 7 within the reach), and it is needed
/// there (some quotient is above 1)
#[test]
fn the_claimed_slope_holds_where_the_bound_varies() {
    let (node, _) = steep_slab(0.025);
    let claim = eval_lipschitz(&node);
    let h = 2e-4f32;
    let mut worst = 0.0f32;
    let mut n = 0;
    for j in 0..2000 {
        let y = -0.09 + 0.28 * (j as f32 + 0.5) / 2000.0;
        let p = Vec3::new(0.003, y, -0.004);
        let q = (eval(&node, p + Vec3::Y * h) - eval(&node, p)).abs() / h;
        worst = worst.max(q);
        n += 1;
    }
    assert!(n == 2000, "compared {n}");
    assert!(
        worst > 1.05,
        "the scene must need the claim's second term ({worst})"
    );
    assert!(
        worst <= claim * (1.0 + 1e-3),
        "difference quotient {worst} above the claim {claim}"
    );
}

/// A fold: identity for x < 0.075 and a translation by −0.15 along x from x = 0.1 on, so the
/// rest sphere (radius 0.05 at the origin) appears a second time around x = 0.15 although
/// nothing there is the image of it (the round trip x → forward(x) does not return). The floor
/// holds, at the nodes of the translated region, the closed-form distance to the true surface
/// (the sphere at the origin) minus the cell diagonal, and −reach elsewhere.
fn fold_scene(with_floor: bool) -> SdfNode {
    let (h, r, reach) = (0.025f32, 0.05f32, 0.05f32);
    let (dims, origin) = ([17u32, 9, 9], [-0.1f32, -0.1, -0.1]);
    let mut maps = Vec::new();
    let mut floor = Vec::new();
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let p = Vec3::from(origin) + Vec3::new(i as f32, j as f32, k as f32) * h;
                let t = if p.x > 0.09 { -0.15 } else { 0.0 };
                maps.extend_from_slice(&[1.0, 0.0, 0.0, t, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                floor.push(if p.x > 0.09 {
                    p.length() - r - h * 3f32.sqrt()
                } else {
                    -reach
                });
            }
        }
    }
    let w = GridWarp::new(GridWarpData {
        origin,
        cell: h,
        dims,
        maps,
        reach,
        floor: if with_floor { floor } else { Vec::new() },
    })
    .expect("valid grid");
    SdfNode::sphere(r).grid_warp(w)
}

#[test]
fn a_fold_shows_a_second_surface_without_the_floor_and_none_with_it() {
    let phantom = Vec3::new(0.15, 0.0, 0.0);
    assert!(
        eval(&fold_scene(false), phantom) < 0.0,
        "the scene must have the folded copy"
    );
    let node = fold_scene(true);
    let v = eval(&node, phantom);
    assert!(v > 0.0, "the floor must remove the folded copy: {v}");
    // oracle: the true surface is the sphere at the origin; |value| never exceeds its distance and
    // the value keeps its sign, inside and outside the box
    let mut n = 0;
    for p in points(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.3, 0.1, 0.1), 22) {
        let truth = p.length() - 0.05;
        let v = eval(&node, p);
        assert!(
            v.abs() <= truth.abs() + 1e-6,
            "p {p}: |value| {v} above the distance {truth}"
        );
        assert!(
            v == 0.0 || truth == 0.0 || v.signum() == truth.signum() || truth.abs() < 2e-3,
            "p {p}: sign of {v} vs {truth}"
        );
        n += 1;
    }
    assert!(n > 10_000, "compared {n}");
}

#[cfg(feature = "gpu")]
#[test]
fn the_gpu_gives_the_cpu_value_with_a_floor() {
    use alice_sdf::compiled::GpuEvaluator;
    let node = fold_scene(true);
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
    let pts = points(Vec3::new(-0.1, -0.1, -0.1), Vec3::new(0.3, 0.1, 0.1), 24);
    let got = gpu.eval_batch(&pts).expect("gpu eval");
    let mut worst = 0.0f32;
    let mut floored = 0;
    for (p, g) in pts.iter().zip(&got) {
        let c = eval(&node, *p);
        worst = worst.max((c - g).abs());
        if p.x > 0.11 && c > 0.0 {
            floored += 1;
        }
    }
    assert!(
        pts.len() > 10_000 && floored > 1000,
        "compared {}, floored {floored}",
        pts.len()
    );
    assert!(worst < 2e-6, "max |gpu − cpu| {worst}");
}

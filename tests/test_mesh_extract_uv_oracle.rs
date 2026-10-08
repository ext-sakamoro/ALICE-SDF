//! Mesh extraction presets, decimation and UV layout vs closed forms
//! (`mesh::sdf_to_mesh`, `mesh::dual_contouring`, `mesh::decimate`,
//! `mesh::lightmap`, `mesh::uv_unwrap`).
//!
//! The shape is the unit sphere, so every check has a closed form:
//! - extraction (`MarchingCubesConfig::aaa`, `AdaptiveConfig::aaa` through
//!   `adaptive_marching_cubes_compiled`, `DualContouringConfig::aaa` through
//!   `dual_contouring_compiled`): every vertex lies within one grid cell
//!   diagonal of `|p| = 1`, every face's geometric normal points outward
//!   (`n · p > 0`), the enclosed volume is `4π/3` within 5 %, the vertex
//!   normals are within 10° of `p / |p|`, the triplanar UV of each vertex is
//!   the position projected on the plane of the normal's dominant axis
//!   (`uv_scale = 1`), and the tangent is a unit vector orthogonal to the
//!   normal with handedness ±1. The compiled mesher emits as many triangles
//!   as the interpreted one (both see the same signs at the grid corners).
//! - decimation: the triangle count goes down monotonically from the input to
//!   `DecimateConfig::conservative()` (75 %) to `aggressive()` (25 %), never
//!   above the target; with `conservative()`'s quadric bound
//!   `Σ dist² <= 0.01` every collapsed vertex is within `0.1` of each of its
//!   original planes, so within `0.1 +` the input's own error of the sphere.
//! - lightmap UVs: every UV2 lies in `[0, 1]²`, no two triangles' UV2
//!   regions overlap (exact 2-D separating-axis test), and the fast variant is
//!   the closed form `0.5 + 0.5 ·` (position projected on the normal's
//!   dominant plane).
//! - UV density: for triangles of known UV area `a`, texels per face is
//!   `a · size²`, and the low-density count and warning follow the two
//!   documented thresholds.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![allow(clippy::float_cmp)]

use alice_sdf::compiled::CompiledSdf;
use alice_sdf::mesh::uv_unwrap::{compute_uv_density, UvDensityReport};
use alice_sdf::mesh::{
    adaptive_marching_cubes, adaptive_marching_cubes_compiled, decimate, dual_contouring,
    dual_contouring_compiled, generate_lightmap_uvs, generate_lightmap_uvs_fast, sdf_to_mesh,
    AdaptiveConfig, DecimateConfig, DualContouringConfig, MarchingCubesConfig, Mesh, Vertex,
};
use alice_sdf::SdfNode;
use glam::{Vec2, Vec3, Vec4};

const B: f32 = 1.5;

const fn sphere() -> SdfNode {
    SdfNode::sphere(1.0)
}

fn volume(m: &Mesh) -> f64 {
    m.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| m.vertices[t[k] as usize].position.as_dvec3());
            a.dot(b.cross(c)) / 6.0
        })
        .sum()
}

fn triplanar(p: Vec3, n: Vec3) -> Vec2 {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        Vec2::new(p.y, p.z)
    } else if a.y >= a.z {
        Vec2::new(p.x, p.z)
    } else {
        Vec2::new(p.x, p.y)
    }
}

/// returns the number of vertices checked
fn check_sphere_mesh(name: &str, m: &Mesh, cell: f32, attributes: bool) -> usize {
    assert!(
        m.triangle_count() > 100,
        "{name}: {} triangles",
        m.triangle_count()
    );
    let diag = cell * 3f32.sqrt();
    for v in &m.vertices {
        let r = v.position.length();
        assert!((r - 1.0).abs() <= diag, "{name}: |p| = {r}");
        if attributes {
            let radial = v.position / r;
            assert!(
                v.normal.dot(radial) >= 10f32.to_radians().cos(),
                "{name}: normal"
            );
            let want = triplanar(v.position, v.normal);
            assert!(
                (v.uv - want).length() <= 1e-5,
                "{name}: uv {:?} vs {want:?}",
                v.uv
            );
            let t = v.tangent.truncate();
            assert!((t.length() - 1.0).abs() <= 1e-3, "{name}: tangent length");
            assert!(
                t.dot(v.normal).abs() <= 1e-3,
                "{name}: tangent not orthogonal"
            );
            assert!(
                v.tangent.w.abs() == 1.0,
                "{name}: handedness {}",
                v.tangent.w
            );
            assert_eq!(v.material_id, 0);
        }
    }
    for t in m.indices.chunks_exact(3) {
        let [a, b, c] = [0, 1, 2].map(|k| m.vertices[t[k] as usize].position);
        let n = (b - a).cross(c - a);
        if n.length() > 1e-9 {
            assert!(n.dot(a + b + c) > 0.0, "{name}: inward face");
        }
    }
    let vol = volume(m);
    let want = 4.0 / 3.0 * std::f64::consts::PI;
    assert!(
        (vol - want).abs() <= 0.05 * want,
        "{name}: volume {vol} vs {want}"
    );
    m.vertices.len()
}

#[test]
fn aaa_presets_and_compiled_meshers_reproduce_the_sphere() {
    let node = sphere();
    let compiled = CompiledSdf::compile(&node);
    let (lo, hi) = (Vec3::splat(-B), Vec3::splat(B));
    let mut n = 0;

    let mc = MarchingCubesConfig::aaa(32);
    assert!(mc.compute_normals && mc.compute_uvs && mc.compute_tangents && mc.compute_materials);
    let m = sdf_to_mesh(&node, lo, hi, &mc);
    n += check_sphere_mesh("marching cubes aaa", &m, 2.0 * B / 32.0, true);

    let ad = AdaptiveConfig::aaa(5);
    assert_eq!((ad.max_depth, ad.min_depth), (5, 2));
    let a = adaptive_marching_cubes_compiled(&compiled, lo, hi, &ad);
    let ai = adaptive_marching_cubes(&node, lo, hi, &ad);
    assert_eq!(
        a.triangle_count(),
        ai.triangle_count(),
        "adaptive compiled vs interpreted"
    );
    n += check_sphere_mesh("adaptive compiled aaa", &a, 2.0 * B / 32.0, true);

    let dc = DualContouringConfig::aaa(32);
    assert!(dc.compute_uvs && dc.compute_tangents && dc.compute_materials && dc.clamp_to_cell);
    let d = dual_contouring_compiled(&compiled, lo, hi, &dc);
    let di = dual_contouring(&node, lo, hi, &dc);
    assert_eq!(
        d.triangle_count(),
        di.triangle_count(),
        "dc compiled vs interpreted"
    );
    n += check_sphere_mesh("dual contouring compiled aaa", &d, 2.0 * B / 32.0, true);
    assert!(n > 3000, "checked {n} vertices");
}

fn plain_sphere_mesh(res: usize) -> Mesh {
    sdf_to_mesh(
        &sphere(),
        Vec3::splat(-B),
        Vec3::splat(B),
        &MarchingCubesConfig {
            resolution: res,
            ..Default::default()
        },
    )
}

#[test]
fn decimation_presets_reduce_monotonically_within_their_bounds() {
    let input = plain_sphere_mesh(32);
    let t0 = input.triangle_count();
    let input_err = input
        .vertices
        .iter()
        .map(|v| (v.position.length() - 1.0).abs())
        .fold(0.0f32, f32::max);

    let mut cons = input.clone();
    let cfg = DecimateConfig::conservative();
    assert_eq!((cfg.target_ratio, cfg.max_error), (0.75, 0.01));
    decimate(&mut cons, &cfg);
    let t1 = cons.triangle_count();

    let mut aggr = input;
    let cfg = DecimateConfig::aggressive();
    assert_eq!(cfg.target_ratio, 0.25);
    decimate(&mut aggr, &cfg);
    let t2 = aggr.triangle_count();

    println!("triangles {t0} -> conservative {t1} -> aggressive {t2}");
    assert!(t1 < t0 && t2 < t1, "{t0} {t1} {t2}");
    assert!(
        t1 >= (t0 as f32 * 0.75) as usize - 2,
        "conservative overshoots: {t1}"
    );
    assert!(
        t2 <= (t0 as f32 * 0.25) as usize,
        "aggressive above target: {t2}"
    );
    for m in [&cons, &aggr] {
        for t in m.indices.chunks_exact(3) {
            assert!(t.iter().all(|&i| (i as usize) < m.vertices.len()));
            assert!(t[0] != t[1] && t[1] != t[2] && t[0] != t[2], "degenerate");
        }
    }
    let bound = 0.1 + input_err;
    for v in &cons.vertices {
        let e = (v.position.length() - 1.0).abs();
        assert!(
            e <= bound,
            "conservative vertex off the sphere by {e} (bound {bound})"
        );
    }
}

fn tri_overlap_2d(a: [Vec2; 3], b: [Vec2; 3]) -> bool {
    // separating axis test on the 6 edge normals; touching edges do not overlap
    for tri in [a, b] {
        for k in 0..3 {
            let e = tri[(k + 1) % 3] - tri[k];
            let axis = Vec2::new(-e.y, e.x);
            let proj = |t: [Vec2; 3]| {
                let d = t.map(|p| p.dot(axis));
                (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]))
            };
            let (a0, a1) = proj(a);
            let (b0, b1) = proj(b);
            let slack = 1e-7 * axis.length();
            if a1 <= b0 + slack || b1 <= a0 + slack {
                return false;
            }
        }
    }
    true
}

#[test]
fn lightmap_uvs_lie_in_the_unit_square_without_overlap() {
    let mut m = plain_sphere_mesh(12);
    let tris_before = m.triangle_count();
    generate_lightmap_uvs(&mut m, 512);
    assert_eq!(m.triangle_count(), tris_before);
    let tris: Vec<[Vec2; 3]> = m
        .indices
        .chunks_exact(3)
        .map(|t| [0, 1, 2].map(|k| m.vertices[t[k] as usize].uv2))
        .collect();
    for t in &tris {
        for p in t {
            assert!(
                (0.0..=1.0).contains(&p.x) && (0.0..=1.0).contains(&p.y),
                "{p:?}"
            );
        }
    }
    let mut pairs = 0usize;
    for i in 0..tris.len() {
        for j in i + 1..tris.len() {
            assert!(
                !tri_overlap_2d(tris[i], tris[j]),
                "triangles {i} and {j} overlap"
            );
            pairs += 1;
        }
    }
    assert!(pairs > 10_000, "compared {pairs} pairs");
}

#[test]
fn fast_lightmap_uvs_are_the_dominant_axis_projection() {
    let mut m = plain_sphere_mesh(10);
    generate_lightmap_uvs_fast(&mut m);
    for v in &m.vertices {
        let want = triplanar(v.position, v.normal) * 0.5 + Vec2::splat(0.5);
        assert!((v.uv2 - want).length() <= 1e-6);
    }
    assert!(!m.vertices.is_empty());
}

#[test]
fn uv_density_counts_texels_per_face_from_the_uv_area() {
    // four triangles with UV areas 1/2, 1/8, 1/2048, and 0 (collapsed)
    let v = |u: f32, w: f32| {
        Vertex::with_all(
            Vec3::new(u, w, 0.0),
            Vec3::Z,
            Vec2::new(u, w),
            Vec2::ZERO,
            Vec4::new(1.0, 0.0, 0.0, 1.0),
            [1.0; 4],
            7,
        )
    };
    let mut m = Mesh::new();
    let pts = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (1.0 / 64.0, 0.0),
        (0.0, 1.0 / 32.0),
        (0.25, 0.25),
    ];
    for (u, w) in pts {
        m.vertices.push(v(u, w));
    }
    assert_eq!(m.vertices[1].uv, Vec2::new(1.0, 0.0));
    assert_eq!(m.vertices[1].material_id, 7);
    m.indices = vec![0, 1, 2, 0, 3, 4, 0, 5, 6, 7, 7, 7];
    let size = 64u32;
    let r: UvDensityReport = compute_uv_density(&m, size);
    let px = f64::from(size * size);
    let texels = [0.5 * px, 0.125 * px, px / 4096.0];
    assert_eq!(r.texture_size, size);
    assert_eq!(r.total_face_count, 4);
    assert!((f64::from(r.max_texels_per_face) - texels[0]).abs() < 1e-3);
    assert!((f64::from(r.min_texels_per_face) - texels[2]).abs() < 1e-6);
    let avg = texels.iter().sum::<f64>() / 3.0;
    assert!((f64::from(r.avg_texels_per_face) - avg).abs() < 1e-3);
    // 1 texel (< 30) and the collapsed one are low density: 2 of 4
    assert_eq!(r.low_density_face_count, 2);
    assert_eq!(r.low_density_ratio, 0.5);
    assert!(r.low_density_ratio > UvDensityReport::WARN_LOW_DENSITY_RATIO);
    assert!(r.has_warning());
    assert!(texels[2] < f64::from(UvDensityReport::RECOMMENDED_MIN_TEXELS_PER_FACE));
    assert!(texels[1] > f64::from(UvDensityReport::RECOMMENDED_MIN_TEXELS_PER_FACE));

    // the same mesh on a 4096² texture: no face under 30 texels except the collapsed one
    let r = compute_uv_density(&m, 4096);
    assert_eq!(r.low_density_face_count, 1);
    assert_eq!(r.low_density_ratio, 0.25);
    assert!(r.has_warning());
    m.indices.truncate(9);
    let r = compute_uv_density(&m, 4096);
    assert_eq!(r.low_density_face_count, 0);
    assert!(!r.has_warning());
}

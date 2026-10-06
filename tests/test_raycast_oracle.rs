//! Ray casting oracle against the closed form of a sphere.
//!
//! For a sphere of radius R at the origin and a unit ray `o + t·d`:
//! `b = o·d`, `c = |o|² − R²`, the first hit is `t = −b − √(b² − c)` when
//! `b² − c ≥ 0` and `t ≥ 0`, otherwise the ray misses; the normal is the hit
//! point over R. Every marcher (tree, compiled, SIMD, JIT, batched,
//! detailed, each config preset) and every depth / normal renderer is
//! compared with that, pixel by pixel through the pinhole camera
//! `dir = normalize(f + u·tan(fov/2)·aspect·r + v·tan(fov/2)·up)` with
//! `u = 2x/w − 1`, `v = 1 − 2y/h`.
//!
//! Shadows and AO use scenes whose answer is closed form too:
//! - hard shadow: blocked iff the shadow ray hits the sphere
//! - AO in a spherical cavity of radius R, sampled up from the bottom point:
//!   the wall distance at height t is `R − |t − R|`
//!
//! Rays that graze the silhouette (`b² − c` within a margin of 0) are left
//! out: sphere tracing is allowed to stop either side of a tangent.
//!
//! Author: Moroya Sakamoto

use alice_sdf::compiled::Vec3x8;
use alice_sdf::prelude::*;
use alice_sdf::raycast::{
    ambient_occlusion_compiled, hard_shadow, hard_shadow_compiled, raymarch_batch,
    raymarch_compiled, raymarch_compiled_batch_parallel, raymarch_compiled_with_config,
    raymarch_detailed, raymarch_relaxed, raymarch_simd_8, render_depth, render_depth_compiled,
    render_depth_compiled_simd, render_normals, render_normals_compiled, soft_shadow_compiled,
};

const R: f32 = 1.0;
const MAX: f32 = 20.0;
const TOL: f32 = 2e-3;
/// `b² − c` margin around tangency
const GRAZE: f32 = 0.05;

/// Closed-form first hit, `None` on a miss, `Err(())` for a grazing ray.
fn sphere_hit(o: Vec3, d: Vec3) -> Result<Option<f32>, ()> {
    let d = d.normalize();
    let b = o.dot(d);
    let c = o.length_squared() - R * R;
    let disc = b * b - c;
    if disc.abs() < GRAZE {
        return Err(());
    }
    if disc < 0.0 {
        return Ok(None);
    }
    let t = -b - disc.sqrt();
    Ok((t >= 0.0).then_some(t))
}

fn rays() -> Vec<Ray> {
    let mut v = Vec::new();
    for i in 0..9 {
        for j in 0..9 {
            let o = Vec3::new(-3.0 + i as f32 * 0.25, -2.0 + j as f32 * 0.5, 5.0);
            let target = Vec3::new(0.3 * (j as f32 - 4.0) * 0.2, 0.1 * (i as f32 - 4.0), 0.0);
            v.push(Ray::new(o, (target - o).normalize()));
        }
    }
    v
}

fn check_hit(got: Option<f32>, o: Vec3, d: Vec3, what: &str) -> usize {
    match sphere_hit(o, d) {
        Err(()) => 0,
        Ok(None) => {
            assert!(
                got.is_none(),
                "{what}: o={o} d={d} expected a miss, got {got:?}"
            );
            1
        }
        Ok(Some(t)) => {
            let g = got.unwrap_or_else(|| panic!("{what}: o={o} d={d} expected t={t}, got a miss"));
            assert!((g - t).abs() < TOL, "{what}: o={o} d={d}: {g} vs {t}");
            1
        }
    }
}

#[test]
fn scalar_marchers_match_ray_sphere() {
    let node = SdfNode::sphere(R);
    let compiled = CompiledSdf::compile(&node);
    let configs = [
        ("default", RaymarchConfig::default()),
        ("fast", RaymarchConfig::fast()),
        ("high_quality", RaymarchConfig::high_quality()),
        ("relaxed", RaymarchConfig::relaxed(&node)),
    ];
    let mut n = 0;
    let mut hits = 0;
    for ray in rays() {
        let (o, d) = (ray.origin, ray.direction);
        n += check_hit(
            raymarch(&node, o, d, MAX).map(|h| h.distance),
            o,
            d,
            "raymarch",
        );
        n += check_hit(
            raycast(&node, ray, MAX).map(|h| h.distance),
            o,
            d,
            "raycast",
        );
        n += check_hit(
            raymarch_relaxed(&node, o, d, MAX).map(|h| h.distance),
            o,
            d,
            "relaxed",
        );
        n += check_hit(
            raymarch_compiled(&compiled, o, d, MAX).map(|h| h.distance),
            o,
            d,
            "compiled",
        );
        for (name, cfg) in &configs {
            n += check_hit(
                raymarch_with_config(&node, o, d, MAX, cfg).map(|h| h.distance),
                o,
                d,
                name,
            );
            n += check_hit(
                raymarch_compiled_with_config(&compiled, o, d, MAX, cfg).map(|h| h.distance),
                o,
                d,
                name,
            );
        }
        let det: RaymarchResult = raymarch_detailed(&node, o, d, MAX, &RaymarchConfig::default());
        n += check_hit(det.hit.then_some(det.distance), o, d, "detailed");
        if let Ok(Some(t)) = sphere_hit(o, d) {
            let p = o + d * t;
            assert!((det.point - p).length() < TOL);
            assert!(
                (det.normal - p / R).length() < 1e-2,
                "{} vs {}",
                det.normal,
                p / R
            );
            hits += 1;
        }
    }
    assert!(n > 100 && hits > 10, "compared {n}, hits {hits}");
}

#[test]
fn batched_marchers_match_ray_sphere() {
    let node = SdfNode::sphere(R);
    let compiled = CompiledSdf::compile(&node);
    let rays = rays();
    let outs = [
        ("batch", raymarch_batch(&node, &rays, MAX)),
        ("raycast_batch", raycast_batch(&node, &rays, MAX)),
        (
            "compiled_batch",
            raymarch_compiled_batch_parallel(&compiled, &rays, MAX),
        ),
    ];
    let mut n = 0;
    for (name, out) in &outs {
        assert_eq!(out.len(), rays.len());
        for (ray, hit) in rays.iter().zip(out) {
            n += check_hit(hit.map(|h| h.distance), ray.origin, ray.direction, name);
        }
    }
    // 8-wide packets
    for chunk in rays.chunks_exact(8) {
        let o: [Vec3; 8] = std::array::from_fn(|i| chunk[i].origin);
        let d: [Vec3; 8] = std::array::from_fn(|i| chunk[i].direction);
        let res = raymarch_simd_8(
            &compiled,
            Vec3x8::from_vecs(o),
            Vec3x8::from_vecs(d),
            MAX,
            &RaymarchConfig::default(),
        );
        for i in 0..8 {
            n += check_hit(res[i].map(|r| r.0), o[i], d[i], "simd_8");
        }
    }
    assert!(n > 100);
}

/// Pinhole camera at (0, 0, 4) looking down −Z.
fn camera_rays(w: usize, h: usize, fov: f32) -> Vec<(Vec3, Vec3)> {
    let pos = Vec3::new(0.0, 0.0, 4.0);
    let (f, up) = (Vec3::NEG_Z, Vec3::Y);
    let r = f.cross(up).normalize();
    let up = r.cross(f);
    let hh = (fov * 0.5).tan();
    let hw = hh * w as f32 / h as f32;
    let mut v = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let u = 2.0 * x as f32 / w as f32 - 1.0;
            let vv = 1.0 - 2.0 * y as f32 / h as f32;
            v.push((pos, (f + r * (u * hw) + up * (vv * hh)).normalize()));
        }
    }
    v
}

const W: usize = 20;
const H: usize = 12;
const FOV: f32 = 0.9;

fn check_depth(buf: &[f32], what: &str) -> usize {
    assert_eq!(buf.len(), W * H);
    let mut n = 0;
    for (px, (o, d)) in buf.iter().zip(camera_rays(W, H, FOV)) {
        let got = (*px != f32::MAX).then_some(*px);
        n += check_hit(got, o, d, what);
    }
    n
}

fn check_normals(buf: &[[u8; 3]], what: &str) -> usize {
    let mut n = 0;
    for (px, (o, d)) in buf.iter().zip(camera_rays(W, H, FOV)) {
        match sphere_hit(o, d) {
            Ok(Some(t)) => {
                let nrm = (o + d * t) / R;
                for k in 0..3 {
                    let expected = (nrm[k] * 0.5 + 0.5) * 255.0;
                    assert!(
                        (px[k] as f32 - expected).abs() <= 2.0,
                        "{what}: {px:?} vs {nrm}"
                    );
                }
                n += 1;
            }
            Ok(None) => {
                assert_eq!(*px, [0, 0, 0], "{what}: miss pixel");
                n += 1;
            }
            Err(()) => {}
        }
    }
    n
}

#[test]
fn depth_and_normal_renderers_match_the_pinhole_closed_form() {
    let node = SdfNode::sphere(R);
    let compiled = CompiledSdf::compile(&node);
    let pos = Vec3::new(0.0, 0.0, 4.0);
    let mut n = 0;
    n += check_depth(
        &render_depth(&node, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
        "render_depth",
    );
    n += check_depth(
        &render_depth_compiled(&compiled, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
        "render_depth_compiled",
    );
    n += check_depth(
        &render_depth_compiled_simd(&compiled, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
        "render_depth_compiled_simd",
    );
    n += check_normals(
        &render_normals(&node, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
        "render_normals",
    );
    n += check_normals(
        &render_normals_compiled(&compiled, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
        "render_normals_compiled",
    );
    assert!(n > 5 * 100, "compared {n}");
}

#[test]
fn hard_shadow_is_blocked_iff_the_ray_hits() {
    let node = SdfNode::sphere(R);
    let compiled = CompiledSdf::compile(&node);
    let mut n = 0;
    for ray in rays() {
        let (o, d) = (ray.origin, ray.direction);
        if let Ok(hit) = sphere_hit(o, d) {
            let expected = hit.is_some();
            assert_eq!(hard_shadow(&node, o, d, 0.0, MAX), expected, "o={o} d={d}");
            assert_eq!(hard_shadow_compiled(&compiled, o, d, 0.0, MAX), expected);
            n += 1;
        }
    }
    assert!(n > 20);
}

#[test]
fn soft_shadow_extremes() {
    let node = SdfNode::sphere(R);
    let compiled = CompiledSdf::compile(&node);
    // through the centre: fully shadowed
    let o = Vec3::new(0.0, -3.0, 0.0);
    assert_eq!(soft_shadow(&node, o, Vec3::Y, 0.01, MAX, 8.0), 0.0);
    assert_eq!(
        soft_shadow_compiled(&compiled, o, Vec3::Y, 0.01, MAX, 8.0),
        0.0
    );
    // pointing away: fully lit
    assert_eq!(soft_shadow(&node, o, Vec3::NEG_Y, 0.01, MAX, 8.0), 1.0);
    assert_eq!(
        soft_shadow_compiled(&compiled, o, Vec3::NEG_Y, 0.01, MAX, 8.0),
        1.0
    );
    // passing by at clearance c: the penumbra factor is at most k·c/t* < 1
    // and strictly between the two extremes
    let s = soft_shadow(&node, Vec3::new(1.05, -3.0, 0.0), Vec3::Y, 0.01, MAX, 8.0);
    assert!(s > 0.0 && s < 1.0, "{s}");
}

#[test]
fn ambient_occlusion_in_a_spherical_cavity() {
    // Solid everywhere except a cavity of radius R; sample up from its bottom.
    let cavity = SdfNode::box3d(20.0, 20.0, 20.0).subtract(SdfNode::sphere(R));
    let compiled = CompiledSdf::compile(&cavity);
    let p = Vec3::new(0.0, -R, 0.0);
    let mut n = 0;
    for &(samples, max_d) in &[(4_u32, 1.0_f32), (5, 1.5), (8, 1.9)] {
        let step = max_d / samples as f32;
        let occ: f32 = (1..=samples)
            .map(|i| {
                let t = i as f32 * step;
                let wall = R - (t - R).abs();
                (t - wall.max(0.0)) / t
            })
            .sum();
        let expected = (1.0 - occ / samples as f32).clamp(0.0, 1.0);
        let got = ambient_occlusion(&cavity, p, Vec3::Y, samples, max_d);
        let got_c = ambient_occlusion_compiled(&compiled, p, Vec3::Y, samples, max_d);
        assert!(
            (got - expected).abs() < 1e-5,
            "{samples}/{max_d}: {got} vs {expected}"
        );
        assert!(
            (got_c - expected).abs() < 1e-5,
            "{samples}/{max_d}: {got_c} vs {expected}"
        );
        n += 1;
    }
    // On an open plane the wall distance equals t: no occlusion at all.
    let ground = SdfNode::box3d(20.0, 2.0, 20.0).translate(0.0, -1.0, 0.0);
    assert!((ambient_occlusion(&ground, Vec3::ZERO, Vec3::Y, 6, 0.9) - 1.0).abs() < 1e-5);
    assert!(n > 0);
}

#[cfg(feature = "jit")]
mod jit {
    use super::*;
    use alice_sdf::compiled::jit::{JitCompiledSdf, JitSimdSdf};
    use alice_sdf::raycast::{
        raymarch_jit, raymarch_jit_batch_parallel, raymarch_jit_simd_8, raymarch_jit_with_config,
        render_depth_jit, render_depth_jit_simd,
    };

    #[test]
    fn jit_marchers_match_ray_sphere() {
        let node = SdfNode::sphere(R);
        let compiled = CompiledSdf::compile(&node);
        let jit = JitCompiledSdf::compile(&node).expect("jit");
        let simd = JitSimdSdf::compile(&compiled).expect("jit simd");
        let rays = rays();
        let mut n = 0;
        for ray in &rays {
            let (o, d) = (ray.origin, ray.direction);
            n += check_hit(
                raymarch_jit(&jit, o, d, MAX).map(|h| h.distance),
                o,
                d,
                "jit",
            );
            n += check_hit(
                raymarch_jit_with_config(&jit, o, d, MAX, &RaymarchConfig::fast())
                    .map(|h| h.distance),
                o,
                d,
                "jit fast",
            );
        }
        for (ray, hit) in rays
            .iter()
            .zip(raymarch_jit_batch_parallel(&jit, &rays, MAX))
        {
            n += check_hit(
                hit.map(|h| h.distance),
                ray.origin,
                ray.direction,
                "jit batch",
            );
        }
        for chunk in rays.chunks_exact(8) {
            let o: [Vec3; 8] = std::array::from_fn(|i| chunk[i].origin);
            let d: [Vec3; 8] = std::array::from_fn(|i| chunk[i].direction);
            let res = raymarch_jit_simd_8(
                &simd,
                &compiled,
                Vec3x8::from_vecs(o),
                Vec3x8::from_vecs(d),
                MAX,
                &RaymarchConfig::default(),
            );
            for i in 0..8 {
                n += check_hit(res[i].map(|r| r.0), o[i], d[i], "jit simd_8");
            }
        }
        let pos = Vec3::new(0.0, 0.0, 4.0);
        n += check_depth(
            &render_depth_jit(&jit, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
            "render_depth_jit",
        );
        n += check_depth(
            &render_depth_jit_simd(&simd, &compiled, pos, Vec3::NEG_Z, Vec3::Y, W, H, FOV, MAX),
            "render_depth_jit_simd",
        );
        assert!(n > 300, "compared {n}");
    }
}

//! `heightmap_displacement` projects a point onto the face of its dominant axis. Where two axes tie
//! (|x| = |y| and so on) the face changes, and with a hard choice the field jumps by up to the
//! amplitude times the difference of the two faces' samples. The law blends the faces within a band
//! of 1/16 of the dominant component: the field is continuous at the ties and equal, bit for bit, to
//! the single-face value wherever the dominant axis leads the others by more than the band.

use alice_sdf::modifiers::heightmap_displacement;
use glam::Vec3;

const W: u32 = 70;

/// A height map with large differences between neighbouring texels (the worst case for a jump)
fn map() -> Vec<f32> {
    (0..W * W)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0)
        .collect()
}

/// The single-face law: the face of the dominant axis (x, then y, then z on ties), bilinear sample
fn single_face(p: Vec3, hm: &[f32], amp: f32, scale: f32) -> f32 {
    let a = p.abs();
    let (u, v) = if a.x > a.y && a.x > a.z {
        (p.y * scale, p.z * scale)
    } else if a.y > a.z {
        (p.x * scale, p.z * scale)
    } else {
        (p.x * scale, p.y * scale)
    };
    let side = (W - 1) as f32;
    let u = ((u * 0.5 + 0.5) * side).clamp(0.0, side);
    let v = ((v * 0.5 + 0.5) * side).clamp(0.0, side);
    alice_sdf::modifiers::bilinear_sample(hm, W, W, u, v) * amp
}

/// The sample of one face (0 = YZ, 1 = XZ, 2 = XY) at `p`
fn face(p: Vec3, axis: usize, hm: &[f32], amp: f32, scale: f32) -> f32 {
    let (u, v) = match axis {
        0 => (p.y, p.z),
        1 => (p.x, p.z),
        _ => (p.x, p.y),
    };
    let side = (W - 1) as f32;
    let u = ((u * scale * 0.5 + 0.5) * side).clamp(0.0, side);
    let v = ((v * scale * 0.5 + 0.5) * side).clamp(0.0, side);
    alice_sdf::modifiers::bilinear_sample(hm, W, W, u, v) * amp
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

#[test]
fn the_field_is_continuous_across_the_face_ties() {
    let hm = map();
    let (amp, scale) = (0.05, 1.0);
    let mut r = lcg(3);
    let mut worst = 0.0f32;
    let mut checked = 0;
    // points on the three tie planes, each crossed along the direction that swaps the two axes
    for k in 0..600 {
        let (s, t) = (r() * 0.9, r() * 0.9);
        let c = 0.3 + 0.5 * r().abs();
        let sign = if r() > 0.0 { 1.0 } else { -1.0 };
        let (p, n) = match k % 3 {
            // |x| = |y| (x = y), the third axis smaller so the tie is between the two largest
            0 => (
                Vec3::new(c, sign * c, s * c * 0.8),
                Vec3::new(1.0, -sign, 0.0),
            ),
            1 => (
                Vec3::new(t * c * 0.8, c, sign * c),
                Vec3::new(0.0, 1.0, -sign),
            ),
            _ => (
                Vec3::new(c, s * c * 0.8, sign * c),
                Vec3::new(1.0, 0.0, -sign),
            ),
        };
        let n = n.normalize();
        let gap = 1e-5;
        let a = heightmap_displacement(p + n * gap, &hm, W, W, amp, scale);
        let b = heightmap_displacement(p - n * gap, &hm, W, W, amp, scale);
        worst = worst.max((a - b).abs());
        checked += 1;
    }
    eprintln!("largest step across a tie at a gap of 2e-5: {worst:e} over {checked} crossings");
    assert_eq!(checked, 600);
    // the hard choice jumps by up to amp × (largest sample difference) ≈ 0.05 here; a continuous
    // field moves by about its gradient times the gap (the blend's gradient is below amp × 16 / m)
    assert!(worst < 1e-3, "the field jumps at a face tie: {worst}");
}

#[test]
fn outside_the_band_the_value_is_the_single_face_value_bit_for_bit() {
    let hm = map();
    let (amp, scale) = (0.05, 1.3);
    let mut r = lcg(11);
    let (mut compared, mut skipped) = (0, 0);
    for _ in 0..20_000 {
        let p = Vec3::new(r(), r(), r()) * 1.2;
        let a = p.abs();
        let m = a.max_element();
        let band = m - m * 0.0625;
        // the other two axes below the band: only the dominant face has weight
        let others = [a.x, a.y, a.z].iter().filter(|c| **c > band).count();
        if others != 1 {
            skipped += 1;
            continue;
        }
        let got = heightmap_displacement(p, &hm, W, W, amp, scale);
        let want = single_face(p, &hm, amp, scale);
        assert_eq!(got.to_bits(), want.to_bits(), "{p:?}: {got} vs {want}");
        compared += 1;
    }
    eprintln!("{compared} points compared bit for bit, {skipped} in a band");
    assert!(compared > 15_000, "only {compared} points outside the band");
}

#[test]
fn inside_the_band_the_value_lies_between_the_faces_it_blends() {
    let hm = map();
    let (amp, scale) = (0.05, 1.0);
    let mut r = lcg(29);
    let mut checked = 0;
    for _ in 0..20_000 {
        let p = Vec3::new(r(), r(), r());
        let a = p.abs();
        let m = a.max_element();
        let band = m - m * 0.0625;
        let faces: Vec<usize> = (0..3).filter(|i| [a.x, a.y, a.z][*i] > band).collect();
        if faces.len() < 2 {
            continue;
        }
        let vals: Vec<f32> = faces.iter().map(|i| face(p, *i, &hm, amp, scale)).collect();
        let (lo, hi) = vals
            .iter()
            .fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
        let got = heightmap_displacement(p, &hm, W, W, amp, scale);
        assert!(
            got >= lo - 1e-7 && got <= hi + 1e-7,
            "{p:?}: {got} outside [{lo}, {hi}] of faces {faces:?}"
        );
        checked += 1;
    }
    eprintln!("{checked} points in a band checked");
    assert!(checked > 500, "only {checked} points in a band");
}

/// Tracing across the band: the blend adds a slope of about amp · Δh / (band · m) where the faces
/// differ. A displaced field is not 1-Lipschitz (this map's texels differ by up to 1, so |grad f|
/// is about 2.7 on a face, which is why `eval_lipschitz` reports no bound for the node), so the plain
/// sphere-tracing step steps past this surface with either law. The checks are that the blend does
/// not make it worse: its largest gradient in a band stays at the face's (measured 2.82 in a band,
/// 2.71 outside), the plain step steps past no more often than with the single-face law (which jumps
/// at ties), and a step scaled for a bound of 4 (distance / 4) never steps past it. The truth is the
/// first sign change of a fine fixed-step march
#[test]
fn tracing_across_the_band_does_not_step_past_the_surface() {
    let hm = map();
    let (amp, scale, radius) = (0.05f32, 1.0f32, 0.8f32);
    let blended = |p: Vec3| p.length() - radius - heightmap_displacement(p, &hm, W, W, amp, scale);
    let single = |p: Vec3| p.length() - radius - single_face(p, &hm, amp, scale);

    // the field's gradient near the ties (central differences, h = 1e-4), inside and outside a band
    let mut r = lcg(41);
    let (mut g_band, mut g_out) = (0.0f32, 0.0f32);
    for _ in 0..20_000 {
        let p = Vec3::new(r(), r(), r()).normalize() * (radius + 0.05 * r());
        let h = 1e-4;
        let g = Vec3::new(
            blended(p + Vec3::X * h) - blended(p - Vec3::X * h),
            blended(p + Vec3::Y * h) - blended(p - Vec3::Y * h),
            blended(p + Vec3::Z * h) - blended(p - Vec3::Z * h),
        ) / (2.0 * h);
        let a = p.abs();
        let m = a.max_element();
        let in_band = [a.x, a.y, a.z]
            .iter()
            .filter(|c| **c > m - m * 0.0625)
            .count()
            > 1;
        if in_band {
            g_band = g_band.max(g.length());
        } else {
            g_out = g_out.max(g.length());
        }
    }
    eprintln!("largest |grad f|: {g_band:.3} in a band, {g_out:.3} outside");

    // rays aimed at the tie diagonals, marched from outside
    let trace = |f: &dyn Fn(Vec3) -> f32, o: Vec3, d: Vec3, lip: f32| -> Option<f32> {
        let mut t = 0.0f32;
        for _ in 0..4096 {
            let s = f(o + d * t);
            if s < 1e-4 {
                return Some(t);
            }
            t += s / lip;
            if t > 4.0 {
                return None;
            }
        }
        None
    };
    let truth = |f: &dyn Fn(Vec3) -> f32, o: Vec3, d: Vec3| -> Option<f32> {
        let step = 1e-4f32;
        let mut t = 0.0f32;
        while t < 4.0 {
            if f(o + d * t) < 0.0 {
                return Some(t);
            }
            t += step;
        }
        None
    };
    let (mut over_blend, mut over_single, mut rays) = (0, 0, 0);
    let (mut over_blend_bound, mut over_single_bound) = (0, 0);
    for k in 0..400 {
        let s = (k as f32 / 400.0) * 2.0 - 1.0;
        // targets on the |x| = |y|, |y| = |z| and |x| = |z| diagonals of the sphere
        let target = match k % 3 {
            0 => Vec3::new(1.0, 1.0, s * 0.9),
            1 => Vec3::new(s * 0.9, 1.0, 1.0),
            _ => Vec3::new(1.0, s * 0.9, 1.0),
        }
        .normalize()
            * radius;
        let o = target * 2.5 + Vec3::new(0.013 * s, -0.011 * s, 0.007);
        let d = (target - o).normalize();
        let (Some(t_true), Some(t_hit)) = (truth(&blended, o, d), trace(&blended, o, d, 1.0))
        else {
            continue;
        };
        rays += 1;
        over_blend += usize::from(t_hit > t_true + 1e-3);
        if let Some(t_b) = trace(&blended, o, d, 4.0) {
            over_blend_bound += usize::from(t_b > t_true + 1e-3);
        }
        if let Some(t1) = truth(&single, o, d) {
            if let Some(t2) = trace(&single, o, d, 1.0) {
                over_single += usize::from(t2 > t1 + 1e-3);
            }
            if let Some(t2) = trace(&single, o, d, 4.0) {
                over_single_bound += usize::from(t2 > t1 + 1e-3);
            }
        }
    }
    eprintln!(
        "{rays} rays: plain step past the surface {over_blend} (blend) / {over_single} (single face); step / 4: {over_blend_bound} / {over_single_bound}"
    );
    assert!(rays > 300, "only {rays} rays hit");
    assert!(g_out > 1.0, "the face gradient was not measured ({g_out})");
    assert!(
        g_band <= g_out * 1.25,
        "the blend steepens the field: {g_band} in a band, {g_out} outside"
    );
    assert!(
        over_blend <= over_single,
        "the blend steps past more often: {over_blend} vs {over_single}"
    );
    assert_eq!(
        over_blend_bound, 0,
        "a step for a bound of 4 steps past the blended surface"
    );
}

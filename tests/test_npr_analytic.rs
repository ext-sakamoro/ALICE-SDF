//! Analytic oracles for the NPR colour laws: exact band levels, endpoints,
//! monotonicity, idempotence — properties a shading law must satisfy
//! regardless of implementation — plus the compiled pipeline against the
//! closed-form composition of those laws.
//!
//! Author: Moroya Sakamoto

use alice_sdf::npr::compiled_color::CompiledColorPipeline;
use alice_sdf::npr::composition::vignette;
use alice_sdf::npr::dsl::{NprColorContext, NprColorNode};
use alice_sdf::npr::outline::{distance_field_outline, distance_field_outline_soft};
use alice_sdf::npr::palette::palette_gradient;
use alice_sdf::npr::rim::fresnel_rim;
use alice_sdf::npr::toon::{posterize_color, soft_toon_ramp, toon_ramp};
use glam::{Vec2, Vec3};

fn samples() -> Vec<f32> {
    (0..=400).map(|i| i as f32 / 400.0).collect()
}

#[test]
fn toon_ramp_is_exact_band_quantisation() {
    for bands in 2..=8u32 {
        let b = bands as f32;
        let mut prev = -1.0f32;
        for x in samples() {
            let y = toon_ramp(x, bands);
            // one of the `bands` levels k / (bands - 1)
            let k = (y * (b - 1.0)).round();
            assert!(
                (y - k / (b - 1.0)).abs() < 1e-6 && (0.0..=b - 1.0).contains(&k),
                "bands {bands}: toon_ramp({x}) = {y} is not a level"
            );
            // the level is the band index of x: floor(x·bands), capped
            let expect = ((x * b).floor().min(b - 1.0)) / (b - 1.0);
            assert!(
                (y - expect).abs() < 1e-6,
                "bands {bands}: x={x} y={y} expect {expect}"
            );
            assert!(y >= prev, "bands {bands}: not monotone at x={x}");
            prev = y;
        }
        assert_eq!(toon_ramp(0.0, bands), 0.0);
        assert_eq!(toon_ramp(1.0, bands), 1.0);
        // out-of-range inputs clamp
        assert_eq!(toon_ramp(-3.0, bands), 0.0);
        assert_eq!(toon_ramp(7.0, bands), 1.0);
    }
}

#[test]
fn soft_toon_ramp_is_monotone_bounded_and_hits_the_endpoints() {
    for bands in 2..=6u32 {
        for smooth in [0.05f32, 0.2, 0.5] {
            let mut prev = -1.0f32;
            for x in samples() {
                let y = soft_toon_ramp(x, bands, smooth);
                assert!((0.0..=1.0).contains(&y), "bands {bands} s {smooth}: y={y}");
                assert!(
                    y + 1e-6 >= prev,
                    "bands {bands} s {smooth}: not monotone at {x}"
                );
                prev = y;
            }
            assert!(soft_toon_ramp(0.0, bands, smooth).abs() < 1e-6);
            assert!((soft_toon_ramp(1.0, bands, smooth) - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn posterize_is_idempotent_on_exact_levels() {
    for levels in 2..=6u32 {
        let l = levels as f32;
        for x in samples() {
            let c = posterize_color(Vec3::splat(x), levels);
            let k = (c.x * (l - 1.0)).round();
            assert!(
                (c.x - k / (l - 1.0)).abs() < 1e-6,
                "levels {levels}: {x} → {}",
                c.x
            );
            let again = posterize_color(c, levels);
            assert!(
                (again - c).length() < 1e-6,
                "levels {levels}: not idempotent at {x}"
            );
        }
        assert_eq!(posterize_color(Vec3::ZERO, levels), Vec3::ZERO);
        assert_eq!(posterize_color(Vec3::ONE, levels), Vec3::ONE);
    }
}

#[test]
fn fresnel_rim_endpoints_and_monotonicity() {
    for power in [0.5f32, 1.0, 2.0, 5.0] {
        assert_eq!(
            fresnel_rim(1.0, power, 1.0),
            0.0,
            "facing the viewer has no rim"
        );
        assert!(
            (fresnel_rim(0.0, power, 1.0) - 1.0).abs() < 1e-6,
            "grazing rim is the full intensity"
        );
        assert!((fresnel_rim(0.0, power, 0.3) - 0.3).abs() < 1e-6);
        let mut prev = f32::MAX;
        for x in samples() {
            let y = fresnel_rim(x, power, 1.0);
            assert!(
                y <= prev + 1e-6,
                "power {power}: rim not decreasing at n·v={x}"
            );
            prev = y;
        }
    }
    // power 1 is linear: rim(n·v) = 1 − n·v
    for x in samples() {
        assert!((fresnel_rim(x, 1.0, 1.0) - (1.0 - x)).abs() < 1e-6);
    }
}

#[test]
fn vignette_is_one_inside_zero_outside_and_radially_monotone() {
    let (radius, soft) = (0.3f32, 0.2f32);
    assert_eq!(vignette(0.5, 0.5, radius, soft), 1.0);
    assert_eq!(vignette(radius.mul_add(0.99, 0.5), 0.5, radius, soft), 1.0);
    assert_eq!(vignette(0.5 + radius + soft + 0.01, 0.5, radius, soft), 0.0);
    assert_eq!(vignette(0.0, 0.0, radius, soft), 0.0);
    let mut prev = 2.0f32;
    for i in 0..=200 {
        let r = 0.6 * i as f32 / 200.0;
        let v = vignette(0.5 + r, 0.5, radius, soft);
        assert!((0.0..=1.0).contains(&v));
        assert!(v <= prev + 1e-6, "vignette not decreasing at r={r}");
        prev = v;
        // radial symmetry
        let v2 = vignette(0.5, 0.5 - r, radius, soft);
        assert!((v - v2).abs() < 1e-6);
    }
    // continuous across inner / outer (smoothstep): values just inside the
    // transitions are within a small step of the plateaus
    assert!((vignette(0.5 + radius + 1e-4, 0.5, radius, soft) - 1.0).abs() < 1e-3);
    assert!(vignette(0.5 + radius + soft - 1e-4, 0.5, radius, soft) < 1e-3);
}

#[test]
fn outline_masks() {
    for w in [0.01f32, 0.05, 0.2] {
        assert_eq!(distance_field_outline(0.0, w), 1.0);
        assert_eq!(distance_field_outline(w * 0.99, w), 1.0);
        assert_eq!(distance_field_outline(-w * 0.99, w), 1.0);
        assert_eq!(distance_field_outline(w * 1.01, w), 0.0);
        assert_eq!(distance_field_outline(-w * 1.01, w), 0.0);
    }
    let (wi, wo) = (0.02f32, 0.1f32);
    assert!((distance_field_outline_soft(0.0, wi, wo) - 1.0).abs() < 1e-6);
    assert!((distance_field_outline_soft(wi * 0.5, wi, wo) - 1.0).abs() < 1e-6);
    assert!(distance_field_outline_soft(wo, wi, wo).abs() < 1e-6);
    assert!(distance_field_outline_soft(wo * 3.0, wi, wo).abs() < 1e-6);
    let mut prev = 2.0f32;
    for i in 0..=100 {
        let d = 0.15 * i as f32 / 100.0;
        let m = distance_field_outline_soft(d, wi, wo);
        assert!(m <= prev + 1e-6, "soft outline not decreasing at {d}");
        assert!(
            (m - distance_field_outline_soft(-d, wi, wo)).abs() < 1e-6,
            "not symmetric"
        );
        prev = m;
    }
}

#[test]
fn palette_gradient_endpoints_and_linearity() {
    let p = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    assert_eq!(palette_gradient(0.0, &p), p[0]);
    assert_eq!(palette_gradient(1.0, &p), p[2]);
    assert!((palette_gradient(0.5, &p) - p[1]).length() < 1e-6);
    assert!((palette_gradient(0.25, &p) - p[0].lerp(p[1], 0.5)).length() < 1e-6);
    assert_eq!(palette_gradient(-1.0, &p), p[0]);
    assert_eq!(palette_gradient(2.0, &p), p[2]);
    assert_eq!(palette_gradient(0.7, &[p[1]]), p[1]);
    assert_eq!(palette_gradient(0.7, &[]), Vec3::ZERO);
}

/// The bytecode pipeline must reproduce the closed-form composition of the
/// laws above: Toon → Fresnel-free multiply / add → posterize, evaluated
/// over a sweep of light directions.
#[test]
fn compiled_pipeline_matches_closed_form_composition() {
    let (shadow, light) = (Vec3::new(0.1, 0.1, 0.2), Vec3::new(0.9, 0.8, 0.7));
    let bands = 4;
    let node = NprColorNode::Multiply {
        a: Box::new(NprColorNode::Toon {
            shadow,
            light,
            bands,
        }),
        b: Box::new(NprColorNode::Constant(Vec3::new(0.5, 1.0, 2.0))),
    };
    let pipeline = CompiledColorPipeline::compile(&node);
    pipeline.validate().expect("balanced");
    for i in 0..=100 {
        let a = std::f32::consts::PI * i as f32 / 100.0;
        let ctx = NprColorContext {
            sdf: 0.0,
            normal: Vec3::Z,
            view: Vec3::Z,
            light: Vec3::new(a.sin(), 0.0, a.cos()).normalize(),
            uv: Vec2::splat(0.5),
            time: 0.0,
        };
        let n_dot_l = ctx.normal.dot(ctx.light);
        let expect = shadow.lerp(light, toon_ramp(n_dot_l, bands)) * Vec3::new(0.5, 1.0, 2.0);
        let got = pipeline.eval(&ctx);
        let direct = node.eval(&ctx);
        assert!(
            (got - expect).length() < 1e-5,
            "angle {a}: compiled {got:?} vs closed form {expect:?}"
        );
        assert!(
            (direct - expect).length() < 1e-5,
            "angle {a}: tree eval {direct:?} vs closed form {expect:?}"
        );
    }
}

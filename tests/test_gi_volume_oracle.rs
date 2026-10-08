//! Analytic oracles for `gi` (cone tracing global illumination) and `volume`
//! (3D volume texture bake / mip chain).
//!
//! Every expected value below comes from a closed form written out in the test
//! (or in the doc comment above it), never from calling the function under
//! test. The closed forms used:
//!
//! * `sky_color` is affine in `dir.y` — `ground + (sky - ground) * (y/2 + 1/2)`
//!   — so a weighted average of sky samples equals the sky evaluated at the
//!   weighted mean of `y`. That turns `trace_hemisphere` over an empty scene
//!   into a closed form that also pins the cone directions and their weights.
//! * `direct_lighting` is the clamped Lambert term `color * max(0, n·l)`.
//! * With no surface inside the cone footprint `cone_trace` accumulates no
//!   alpha, so the result is exactly `sky_color(dir)` with `occlusion == 1`
//!   ("1 = fully open", see `ConeTraceResult`). Inside a solid the first step
//!   saturates alpha, so `occlusion == 0` exactly.
//! * `SH1` is the orthonormal l ≤ 1 basis `Y₀ = √(1/4π)`,
//!   `Y₁ = √(3/4π)·{x,y,z}` (`0.282095` and `0.488603` are those to six
//!   decimals). `project` estimates the projection integrals `cᵢ = ∫ f Yᵢ dΩ`
//!   from one direction, so the sample carries the full `4π` of solid angle;
//!   `evaluate` is the plain reconstruction `Σ cᵢ Yᵢ`. Composing them on a
//!   single sample therefore gives the reproducing kernel
//!   `4π·v·Σ Yᵢ(d) Yᵢ(e) = v·(1 + 3 d·e)`, and — since the basis is orthonormal
//!   and spans the affine functions of direction — a Monte-Carlo projection of
//!   any `f(d) = a + b·d` followed by `normalize(N)` reconstructs `f(e)` itself,
//!   at its physical magnitude.
//! * Trilinear interpolation reproduces a 1st-degree polynomial exactly, both
//!   for the probe grid and for `Volume3D::sample_trilinear`.
//! * `bake_volume` samples grid *nodes* `world_min + i · size/(res-1)` (not
//!   cell centres), so a baked sphere must equal `|p| - r` there, and the
//!   trilinear reconstruction error between nodes must fall with `h²`.
//! * A mip level must be a lower bound of the base voxels inside its
//!   footprint — that is what "min-downsample preserves the SDF distance
//!   property" means, and it is what makes a mip usable to skip empty space.
//!
//! Author: Moroya Sakamoto
#![allow(
    clippy::disallowed_methods,
    reason = "test code: the platform libm and fused mul_add serve as independent references"
)]
#![allow(clippy::float_cmp)]

#[cfg(any(feature = "gi", feature = "volume"))]
use glam::Vec3;

// ---------------------------------------------------------------------------
// gi
// ---------------------------------------------------------------------------

#[cfg(feature = "gi")]
mod gi_oracle {
    use super::Vec3;
    use alice_sdf::gi::irradiance::SH1;
    use alice_sdf::gi::{
        bake_irradiance_grid, cone_trace, direct_lighting, sky_color, trace_hemisphere,
        BakeGiConfig, ConeTraceConfig, DirectionalLight, IrradianceGrid,
    };
    use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};
    use alice_sdf::types::SdfNode;

    /// oracle: `sky_color` doc contract — hemisphere blend between a ground and
    /// a sky colour, so the two endpoints are the endpoints and the map is
    /// affine in `dir.y`.
    const GROUND: Vec3 = Vec3::new(0.1, 0.08, 0.05);
    const SKY: Vec3 = Vec3::new(0.4, 0.6, 1.0);

    /// Closed form of the sky model, written independently of `sky_color`.
    fn sky_closed_form(y: f32) -> Vec3 {
        GROUND + (SKY - GROUND) * (0.5 * y + 0.5)
    }

    fn scene_svo(shape: &SdfNode, depth: u32) -> SparseVoxelOctree {
        SparseVoxelOctree::build(
            shape,
            &SvoBuildConfig {
                max_depth: depth,
                bounds_min: Vec3::splat(-2.0),
                bounds_max: Vec3::splat(2.0),
                ..Default::default()
            },
        )
    }

    #[test]
    fn sky_color_is_the_affine_hemisphere_blend() {
        assert_eq!(
            sky_color(Vec3::Y),
            SKY,
            "straight up must be the sky colour"
        );
        assert_eq!(
            sky_color(Vec3::NEG_Y),
            GROUND,
            "straight down must be the ground colour"
        );
        // The horizon is the midpoint. Not bit-equal to `(GROUND + SKY) * 0.5`:
        // `sky_color` reaches it as `ground + (sky - ground) * 0.5`, one
        // rounding apart from averaging the endpoints.
        assert!(
            (sky_color(Vec3::X) - (GROUND + SKY) * 0.5).length() < 1e-7,
            "the horizon must be the midpoint, got {}",
            sky_color(Vec3::X)
        );

        // Affine in y: f((a+b)/2) == (f(a)+f(b))/2 for every pair.
        for i in 0..=8 {
            for j in 0..=8 {
                let ya = i as f32 / 4.0 - 1.0;
                let yb = j as f32 / 4.0 - 1.0;
                let mid = sky_color(Vec3::new(0.0, f32::midpoint(ya, yb), 0.0));
                let avg = (sky_closed_form(ya) + sky_closed_form(yb)) * 0.5;
                assert!(
                    (mid - avg).length() < 1e-6,
                    "sky_color must be affine in y: y=({ya},{yb}) gave {mid} vs {avg}"
                );
            }
        }
    }

    /// oracle: Lambert `E = color · max(0, n·l)`, exact at 0°, 60°, 90°, 180°.
    #[test]
    fn direct_lighting_is_the_clamped_cosine() {
        let light = DirectionalLight {
            direction: Vec3::Y,
            color: Vec3::new(1.0, 0.5, 0.25),
        };

        assert_eq!(direct_lighting(Vec3::Y, &light), light.color, "n·l = 1");
        assert_eq!(
            direct_lighting(Vec3::X, &light),
            Vec3::ZERO,
            "n·l = 0 must be black, not a signed value"
        );
        assert_eq!(
            direct_lighting(Vec3::NEG_Y, &light),
            Vec3::ZERO,
            "back-facing must clamp to 0"
        );

        // 60°: cos = 1/2 exactly for n = (sin60, cos60, 0) · (0,1,0).
        let n = Vec3::new(3.0f32.sqrt() / 2.0, 0.5, 0.0);
        let got = direct_lighting(n, &light);
        let want = light.color * 0.5;
        assert!(
            (got - want).length() < 1e-6,
            "60° must give half the colour: {got} vs {want}"
        );
    }

    /// oracle: with no surface inside the cone footprint no alpha accumulates,
    /// so `cone_trace` returns the sky term verbatim and reports fully open.
    /// Swept over `max_distance` and `cone_angle` because neither may change a
    /// pure miss.
    #[test]
    fn cone_trace_miss_is_exactly_sky_and_fully_open() {
        let svo = scene_svo(&SdfNode::sphere(1.0), 4);
        // 10 units above the SVO bounds, marching further away.
        let origin = Vec3::new(0.0, 10.0, 0.0);

        for &max_distance in &[0.5f32, 1.0, 2.0, 4.0] {
            for &cone_angle in &[0.2f32, 0.5, 0.8] {
                let config = ConeTraceConfig {
                    max_distance,
                    cone_angle,
                    ..Default::default()
                };
                let r = cone_trace(&svo, origin, Vec3::Y, &config, None);
                assert_eq!(
                    r.color,
                    sky_color(Vec3::Y),
                    "a miss must return the sky term verbatim (d={max_distance}, a={cone_angle})"
                );
                assert_eq!(
                    r.color, SKY,
                    "and that sky term is the closed form for +Y (d={max_distance})"
                );
                assert_eq!(
                    r.occlusion, 1.0,
                    "a miss must report fully open (d={max_distance}, a={cone_angle})"
                );
            }
        }
    }

    /// oracle: a point deep inside a solid saturates the cone on the first
    /// step (`dist < 0 ≤ cone_radius` ⇒ `opacity = 1`), so occlusion is exactly
    /// 0 and the colour is the unlit surface estimate `albedo · ambient`
    /// = 0.5 · 0.05 with no sky contribution.
    #[test]
    fn cone_trace_inside_a_solid_is_fully_occluded() {
        // A sphere far larger than the bounds: every queried point is interior.
        let svo = scene_svo(&SdfNode::sphere(10.0), 3);
        let config = ConeTraceConfig {
            max_distance: 1.5,
            ..Default::default()
        };

        for dir in [Vec3::Y, Vec3::X, Vec3::NEG_Z] {
            let r = cone_trace(&svo, Vec3::ZERO, dir, &config, None);
            assert_eq!(
                r.occlusion, 0.0,
                "inside a solid must report fully occluded (dir={dir})"
            );
            let want = Vec3::splat(0.5 * 0.05);
            assert!(
                (r.color - want).length() < 1e-7,
                "unlit interior radiance must be albedo·ambient: {} vs {want}",
                r.color
            );
        }
    }

    /// oracle: alpha is non-decreasing along the march and the step sequence
    /// does not depend on `max_distance`, so `occlusion = 1 - alpha` can only
    /// fall as the cone is allowed to march further. A rise would mean the
    /// march re-orders or drops samples.
    #[test]
    fn cone_trace_openness_falls_monotonically_with_max_distance() {
        let svo = scene_svo(&SdfNode::sphere(1.0), 5);
        let origin = Vec3::new(0.0, 1.9, 0.0);

        let mut prev = f32::INFINITY;
        let mut saw_drop = false;
        for i in 1..=16 {
            let config = ConeTraceConfig {
                max_distance: i as f32 * 0.25,
                ..Default::default()
            };
            let r = cone_trace(&svo, origin, Vec3::NEG_Y, &config, None);
            assert!(
                r.occlusion <= prev + 1e-7,
                "openness rose from {prev} to {} at max_distance={}",
                r.occlusion,
                i as f32 * 0.25
            );
            if r.occlusion < prev - 1e-6 {
                saw_drop = true;
            }
            prev = r.occlusion;
        }
        assert!(
            saw_drop,
            "the sweep never occluded anything, so monotonicity was vacuous"
        );
    }

    /// oracle: over an empty scene every cone returns `sky_color(dir_i)` and
    /// `ao_factor` is 1 (openness 1), so
    /// `trace_hemisphere = Σ w_i sky(d_i) / Σ w_i`. `sky_color` is affine in y,
    /// so that equals `sky_closed_form(ȳ)` with `ȳ = Σ w_i y_i / Σ w_i`.
    /// The cone set is the documented one: the normal with weight 0.4 plus a
    /// ring of `n-1` cones at `cos = 0.5` with weight 0.5 each, which for a
    /// `+Y` normal means `y_i = 0.5` on the whole ring. Hence
    /// `ȳ = (0.4 + 0.5·0.5·(n-1)) / (0.4 + 0.5·(n-1))`.
    #[test]
    fn trace_hemisphere_over_empty_space_is_the_weighted_sky_mean() {
        let svo = scene_svo(&SdfNode::sphere(1.0), 4);
        let position = Vec3::new(0.0, 10.0, 0.0);

        for &num_cones in &[1u32, 2, 5, 9, 17] {
            let config = ConeTraceConfig {
                num_cones,
                max_distance: 2.0,
                ..Default::default()
            };
            let got = trace_hemisphere(&svo, position, Vec3::Y, &config, None);

            let y_mean = if num_cones == 1 {
                1.0
            } else {
                let ring = (num_cones - 1) as f32;
                (0.4f32 + 0.5 * 0.5 * ring) / (0.4 + 0.5 * ring)
            };
            let want = sky_closed_form(y_mean);
            assert!(
                (got - want).length() < 1e-5,
                "num_cones={num_cones}: {got} vs closed form {want}"
            );
        }
    }

    /// oracle: derived from the orthonormal basis, not from the code.
    /// `project` estimates `cᵢ = ∫ f Yᵢ dΩ` from one direction, so it carries the
    /// sphere's whole solid angle: `cᵢ = 4π·v·Yᵢ(d)`. `evaluate` reconstructs
    /// `Σ cᵢ Yᵢ(e)`. With `Y₀ = √(1/4π)` and `Y₁ = √(3/4π)·d`,
    ///
    /// ```text
    ///   4π·v·[ Y₀² + (3/4π)(d·e) ] = 4π·v·[ 1/4π + 3(d·e)/4π ] = v·(1 + 3 d·e)
    /// ```
    ///
    /// the l ≤ 1 reproducing kernel. At `d = e` that is exactly `4v`; at
    /// `d·e = -1/3` it is exactly 0.
    #[test]
    fn sh1_single_sample_is_the_l1_reproducing_kernel() {
        let dirs = [
            Vec3::Y,
            Vec3::X,
            Vec3::NEG_Z,
            Vec3::new(1.0, 2.0, -3.0).normalize(),
            Vec3::new(-0.4, 0.5, 0.6).normalize(),
        ];

        for &d in &dirs {
            for &e in &dirs {
                for &v in &[1.0f32, 0.25, 3.0] {
                    let got = SH1::project(d, v).evaluate(e);
                    let want = v * (1.0 + 3.0 * d.dot(e));
                    // The basis constants are 6-decimal roundings of √(1/4π) and
                    // √(3/4π), which bounds the relative error of the kernel at
                    // ~2e-6; the kernel itself is at most 4·v.
                    assert!(
                        (got - want).abs() <= 2e-5 * v.abs().max(1.0),
                        "project({d},{v}).evaluate({e}) = {got}, kernel says {want}"
                    );
                }
            }
        }

        // d = e ⇒ exactly 4v (the kernel's peak, 1 + 3·1).
        let self_kernel = SH1::project(Vec3::Y, 1.0).evaluate(Vec3::Y);
        assert!(
            (self_kernel - 4.0).abs() < 2e-5,
            "self projection must be 4, got {self_kernel}"
        );
        // d·e = -1/3 ⇒ exactly 0.
        let e = Vec3::new((8.0f32 / 9.0).sqrt(), -1.0 / 3.0, 0.0);
        let null = SH1::project(Vec3::Y, 1.0).evaluate(e);
        assert!(
            null.abs() < 2e-5,
            "the kernel must vanish at d·e = -1/3, got {null}"
        );
    }

    /// Fibonacci sphere written here in the test — deliberately *not* the
    /// crate's `generate_uniform_directions`, so the oracle does not inherit
    /// the sampler under test.
    fn uniform_dirs(n: usize) -> Vec<Vec3> {
        let ga = std::f32::consts::PI * (3.0 - 5.0f32.sqrt()); // golden angle
        (0..n)
            .map(|i| {
                let z = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
                let r = (1.0 - z * z).max(0.0).sqrt();
                let t = ga * i as f32;
                Vec3::new(r * t.cos(), r * t.sin(), z)
            })
            .collect()
    }

    /// oracle: projecting `f(d) = a + b·d` with the `4π` measure and averaging
    /// over `N` samples estimates `cᵢ = ∫ f Yᵢ dΩ`, and the reconstruction is `f`
    /// **itself** — no scale factor. Derivation from `∫ d dΩ = 0` and
    /// `∫ dⱼ dₖ dΩ = (4π/3)·δⱼₖ`:
    ///
    /// ```text
    ///   c₀  = ∫ (a + b·d) Y₀ dΩ  = 4π·a·√(1/4π)        = a·√(4π)
    ///   c₁ₖ = ∫ (a + b·d) Y₁ dₖ dΩ = (4π/3)·bₖ·√(3/4π)
    ///   Σ cᵢ Yᵢ(e) = a·√(4π)·√(1/4π) + Σ (4π/3)·bₖ·(3/4π)·eₖ = a + b·e
    /// ```
    ///
    /// The residual is the sampler's discrepancy and must fall as N grows.
    #[test]
    fn sh1_projection_reproduces_affine_fields() {
        let a = 0.7f32;
        let b = Vec3::new(0.2, -0.35, 0.15);
        let probe_dirs = [
            Vec3::Y,
            Vec3::NEG_X,
            Vec3::new(0.3, 0.4, -0.5).normalize(),
            Vec3::new(-0.6, -0.2, 0.7).normalize(),
        ];

        let mut errors = Vec::new();
        for &n in &[64usize, 256, 1024] {
            let mut sh = SH1::default();
            for &d in &uniform_dirs(n) {
                sh.add(&SH1::project(d, a + b.dot(d)));
            }
            sh.scale(1.0 / n as f32);

            let mut worst = 0.0f32;
            for &e in &probe_dirs {
                let want = a + b.dot(e);
                worst = worst.max((sh.evaluate(e) - want).abs());
            }
            errors.push(worst);
        }

        // Measured residuals fall as 2.96e-3 -> 3.21e-4 -> 1.60e-5 over
        // N = 64 / 256 / 1024, i.e. the Fibonacci lattice's discrepancy. The
        // bound keeps an order of magnitude of margin over the finest one, which
        // is 1.3e-5 of the field's own scale (|f| <= 1.2).
        assert!(
            errors[2] < 2e-4,
            "N=1024 residual {:?} is not a reproduction of the affine field",
            errors
        );
        assert!(
            errors[2] < errors[0],
            "refining the sample set must reduce the residual: {errors:?}"
        );
    }

    /// oracle: `IrradianceGrid::new` places probes at cell centres
    /// `bounds_min + (i + 1/2)·size/n`, checked against that formula.
    #[test]
    fn irradiance_grid_probes_sit_on_cell_centres() {
        for &n in &[[2u32, 2, 2], [3, 1, 4], [5, 5, 5]] {
            let lo = Vec3::new(-2.0, 0.0, 1.0);
            let hi = Vec3::new(2.0, 4.0, 3.0);
            let grid = IrradianceGrid::new(n, lo, hi);
            assert_eq!(
                grid.probe_count(),
                (n[0] * n[1] * n[2]) as usize,
                "probe count must be the product of the grid dims"
            );

            let step = (hi - lo) / Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32);
            for z in 0..n[2] {
                for y in 0..n[1] {
                    for x in 0..n[0] {
                        let want = lo
                            + Vec3::new(
                                (x as f32 + 0.5) * step.x,
                                (y as f32 + 0.5) * step.y,
                                (z as f32 + 0.5) * step.z,
                            );
                        let got = grid.get_probe(x, y, z).expect("probe in range").position;
                        assert!(
                            (got - want).length() < 1e-6,
                            "probe ({x},{y},{z}) at {got}, cell centre is {want}"
                        );
                    }
                }
            }
        }
    }

    /// oracle: two effects composed, both exact.
    ///
    /// 1. A probe fed two antipodal samples of a constant `v` carries `v` as a
    ///    pure DC term — the linear coefficients cancel between `d` and `-d` —
    ///    and with the `4π` measure in place its reconstruction is `4π·v·Y₀² = v`
    ///    in every direction.
    /// 2. Trilinear interpolation reproduces a 1st-degree polynomial exactly.
    ///
    /// So loading probe `i` with the constant `g(pᵢ)` for
    /// `g(p) = 1 + 0.3x + 0.2y - 0.1z` must make `sample` return `g(p)` itself at
    /// every point inside the hull of the probe centres (outside it the sampler
    /// clamps, which is a different contract and is not asserted here).
    #[test]
    fn irradiance_grid_sample_reproduces_a_linear_field() {
        fn g(p: Vec3) -> f32 {
            0.3f32.mul_add(p.x, 1.0) + 0.2 * p.y - 0.1 * p.z
        }

        for &n in &[[2u32, 2, 2], [4, 3, 2], [5, 5, 5]] {
            let lo = Vec3::splat(-2.0);
            let hi = Vec3::splat(2.0);
            let mut grid = IrradianceGrid::new(n, lo, hi);
            for probe in &mut grid.probes {
                // `g` stays positive over the probe hull, so `evaluate`'s clamp
                // at 0 never fires and the reproduction is the full field.
                let v = Vec3::splat(g(probe.position));
                probe.add_sample(Vec3::Y, v);
                probe.add_sample(Vec3::NEG_Y, v);
                probe.normalize(2);
            }

            // Hull of the probe centres.
            let step = (hi - lo) / Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32);
            let first = lo + step * 0.5;
            let last = hi - step * 0.5;

            let mut worst = (0.0f32, Vec3::ZERO, 0.0f32, 0.0f32);
            for i in 0..=7 {
                for j in 0..=7 {
                    for k in 0..=7 {
                        let f = Vec3::new(i as f32 / 7.0, j as f32 / 7.0, k as f32 / 7.0);
                        let p = first + (last - first) * f;
                        let want = g(p);
                        let got = grid.sample(p, Vec3::Y).x;
                        if (got - want).abs() > worst.0 {
                            worst = ((got - want).abs(), p, got, want);
                        }
                    }
                }
            }
            assert!(
                worst.0 < 2e-5,
                "grid {n:?} at {}: {} vs linear reproduction {} (err {})",
                worst.1,
                worst.2,
                worst.3,
                worst.0
            );
        }
    }

    /// oracle: bake a probe grid far outside the SVO so every cone misses. Each
    /// probe then projects `sky_color`, which is affine in `dir.y` and therefore
    /// lies inside the l ≤ 1 basis, so the baked probe reconstructs
    /// `sky_color(e)` **itself** (see `sh1_projection_reproduces_affine_fields`
    /// for the derivation) up to the sampler's discrepancy. Refining
    /// `samples_per_probe` must shrink the residual.
    #[test]
    fn bake_irradiance_grid_over_empty_space_recovers_the_sky() {
        let svo = scene_svo(&SdfNode::sphere(1.0), 3);
        let probe_dirs = [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::new(0.3, 0.5, -0.8).normalize(),
        ];

        let mut errors = Vec::new();
        for &samples in &[64u32, 256, 1024] {
            let config = BakeGiConfig {
                grid_size: [2, 2, 2],
                // Far away from the SVO bounds (±2), so nothing is hit.
                bounds_min: Vec3::splat(100.0),
                bounds_max: Vec3::splat(102.0),
                samples_per_probe: samples,
                cone_config: ConeTraceConfig {
                    max_distance: 2.0,
                    ..Default::default()
                },
                sun: None,
            };
            let grid = bake_irradiance_grid(&svo, &config);
            assert_eq!(grid.probe_count(), 8);

            let mut worst = 0.0f32;
            for probe in &grid.probes {
                for &e in &probe_dirs {
                    let want = sky_color(e);
                    worst = worst.max((probe.evaluate(e) - want).length());
                }
            }
            errors.push(worst);
        }

        // Measured residuals fall as 3.21e-3 -> 4.72e-4 -> 3.20e-5 over
        // samples_per_probe = 64 / 256 / 1024 (the crate's own Fibonacci
        // sampler). The bound keeps an order of magnitude of margin, which is
        // 2.7e-5 of the sky's own scale (|sky| <= 1.2 per channel).
        assert!(
            errors[2] < 5e-4,
            "N=1024 bake residual {errors:?} is not the sky field"
        );
        assert!(
            errors[2] < errors[0],
            "more probe samples must reduce the residual: {errors:?}"
        );
    }

    /// oracle: `num_cones = 0` is reachable through the public `ConeTraceConfig`
    /// field, and `trace_hemisphere` already documents the empty case by
    /// returning `Vec3::ZERO` when the cone weights sum to zero. Nothing about
    /// a cone count may make the call diverge.
    #[test]
    fn trace_hemisphere_with_zero_cones_returns_black() {
        let svo = scene_svo(&SdfNode::sphere(1.0), 3);
        let config = ConeTraceConfig {
            num_cones: 0,
            max_distance: 2.0,
            ..Default::default()
        };
        let got = trace_hemisphere(&svo, Vec3::new(0.0, 10.0, 0.0), Vec3::Y, &config, None);
        assert_eq!(got, Vec3::ZERO, "zero cones must integrate to black");
    }
}

// ---------------------------------------------------------------------------
// volume
// ---------------------------------------------------------------------------

#[cfg(feature = "volume")]
mod volume_oracle {
    use super::Vec3;
    use alice_sdf::types::SdfNode;
    use alice_sdf::volume::{bake_volume, generate_mip_chain, BakeConfig, Volume3D};

    const R: f32 = 1.0;
    const LO: f32 = -2.0;
    const HI: f32 = 2.0;

    /// The analytic field being baked: a sphere of radius `R` at the origin.
    fn sphere_sdf(p: Vec3) -> f32 {
        p.length() - R
    }

    /// Grid-node position, written from the documented sampling rule
    /// (`world_min + i · size/(res-1)`) rather than read back from the volume.
    fn node_pos(i: [u32; 3], res: [u32; 3], lo: Vec3, hi: Vec3) -> Vec3 {
        let size = hi - lo;
        let step = Vec3::new(
            size.x / (res[0] as f32 - 1.0).max(1.0),
            size.y / (res[1] as f32 - 1.0).max(1.0),
            size.z / (res[2] as f32 - 1.0).max(1.0),
        );
        lo + Vec3::new(i[0] as f32, i[1] as f32, i[2] as f32) * step
    }

    fn cfg(res: [u32; 3]) -> BakeConfig {
        BakeConfig {
            resolution: res,
            bounds_min: Vec3::splat(LO),
            bounds_max: Vec3::splat(HI),
            ..Default::default()
        }
    }

    /// oracle: every baked voxel must equal the analytic sphere distance at its
    /// grid node. Swept over resolutions and over a non-cubic grid, because a
    /// wrong step or a transposed index shows up as a large error, not a small
    /// one.
    #[test]
    fn bake_volume_equals_the_analytic_sphere_at_grid_nodes() {
        for &res in &[[8u32, 8, 8], [16, 16, 16], [5, 9, 3]] {
            let volume = bake_volume(&SdfNode::sphere(R), &cfg(res));
            assert_eq!(
                volume.voxel_count(),
                (res[0] * res[1] * res[2]) as usize,
                "voxel count must be the product of the resolution"
            );

            let mut worst = 0.0f32;
            for z in 0..res[2] {
                for y in 0..res[1] {
                    for x in 0..res[0] {
                        let p = node_pos([x, y, z], res, Vec3::splat(LO), Vec3::splat(HI));
                        let err = (volume.get(x, y, z) - sphere_sdf(p)).abs();
                        worst = worst.max(err);
                    }
                }
            }
            assert!(
                worst < 1e-6,
                "res {res:?}: worst node error {worst} — the bake is not the analytic field"
            );
        }
    }

    /// oracle: `voxel_to_world` must agree with the same grid-node rule, so the
    /// inverse of the bake sampling is consistent with it.
    #[test]
    fn voxel_to_world_is_the_grid_node_rule() {
        let res = [7u32, 4, 9];
        let lo = Vec3::new(-2.0, 1.0, 0.5);
        let hi = Vec3::new(2.0, 3.0, 4.5);
        let volume: Volume3D<f32> = Volume3D::new(res, lo, hi);
        for z in 0..res[2] {
            for y in 0..res[1] {
                for x in 0..res[0] {
                    let want = node_pos([x, y, z], res, lo, hi);
                    let got = volume.voxel_to_world(x, y, z);
                    assert!(
                        (got - want).length() < 1e-6,
                        "voxel ({x},{y},{z}) at {got}, grid node is {want}"
                    );
                }
            }
        }
        // The last node must land exactly on the far bound.
        let corner = volume.voxel_to_world(res[0] - 1, res[1] - 1, res[2] - 1);
        assert!(
            (corner - hi).length() < 1e-5,
            "the last grid node must be world_max: {corner} vs {hi}"
        );
    }

    /// oracle: the bake stores exactly the analytic value at nodes, so
    /// `sample_trilinear` at a node must return that value bit for bit (no
    /// smoothing, no half-voxel offset).
    #[test]
    fn sample_trilinear_at_a_node_returns_that_node() {
        let res = [9u32, 9, 9];
        let volume = bake_volume(&SdfNode::sphere(R), &cfg(res));
        for z in (0..res[2]).step_by(2) {
            for y in (0..res[1]).step_by(2) {
                for x in (0..res[0]).step_by(2) {
                    let p = node_pos([x, y, z], res, Vec3::splat(LO), Vec3::splat(HI));
                    let got = volume.sample_trilinear(p);
                    assert_eq!(
                        got,
                        volume.get(x, y, z),
                        "sample at node ({x},{y},{z}) = {p} drifted"
                    );
                }
            }
        }
    }

    /// oracle: trilinear reconstruction of a C² field is 2nd order, so halving
    /// the voxel size must cut the worst mid-cell error by about 4. Measured
    /// away from the sphere's centre, where the distance field is smooth.
    #[test]
    fn trilinear_reconstruction_is_second_order_in_voxel_size() {
        fn worst_mid_cell_error(res_n: u32) -> f32 {
            let res = [res_n, res_n, res_n];
            let volume = bake_volume(&SdfNode::sphere(R), &cfg(res));
            let mut worst = 0.0f32;
            // Cell centres of the interior cells, skipping the ball around the
            // origin where |p| - r is not differentiable.
            for z in 0..res_n - 1 {
                for y in 0..res_n - 1 {
                    for x in 0..res_n - 1 {
                        let a = node_pos([x, y, z], res, Vec3::splat(LO), Vec3::splat(HI));
                        let b =
                            node_pos([x + 1, y + 1, z + 1], res, Vec3::splat(LO), Vec3::splat(HI));
                        let p = (a + b) * 0.5;
                        if p.length() < 0.5 {
                            continue;
                        }
                        worst = worst.max((volume.sample_trilinear(p) - sphere_sdf(p)).abs());
                    }
                }
            }
            worst
        }

        let e16 = worst_mid_cell_error(17);
        let e32 = worst_mid_cell_error(33);
        assert!(
            e32 > 0.0,
            "the fine grid reproduced the field exactly — the error probe is vacuous"
        );
        let ratio = e16 / e32;
        assert!(
            (3.0..=5.5).contains(&ratio),
            "halving h must cut the error ~4x (2nd order): {e16} -> {e32} (ratio {ratio})"
        );
    }

    /// oracle: with `padding = 0` the baked bounds are the requested bounds
    /// exactly; with `padding = k` they grow by `k · size / res` on each side
    /// (the documented per-voxel padding), which is an exact closed form, not
    /// an inequality.
    #[test]
    fn bake_volume_world_bounds_are_exact() {
        let lo = Vec3::new(-1.0, -2.0, 0.0);
        let hi = Vec3::new(1.0, 2.0, 4.0);
        let res = [8u32, 16, 4];

        let base = BakeConfig {
            resolution: res,
            bounds_min: lo,
            bounds_max: hi,
            ..Default::default()
        };
        let v = bake_volume(&SdfNode::sphere(R), &base);
        assert_eq!(v.world_min, lo, "padding 0 must not move world_min");
        assert_eq!(v.world_max, hi, "padding 0 must not move world_max");
        assert_eq!(v.world_size(), hi - lo, "world_size must be the span");

        for &pad in &[1u32, 2, 3] {
            let v = bake_volume(
                &SdfNode::sphere(R),
                &BakeConfig {
                    padding: pad,
                    ..base.clone()
                },
            );
            let per_voxel = (hi - lo) / Vec3::new(res[0] as f32, res[1] as f32, res[2] as f32);
            let grow = per_voxel * pad as f32;
            assert_eq!(v.world_min, lo - grow, "padding {pad}: world_min");
            assert_eq!(v.world_max, hi + grow, "padding {pad}: world_max");
        }
    }

    /// Closed-form mip resolution chain: halve, floor, stop when it stops
    /// changing (min 1 per axis).
    fn mip_resolutions(base: [u32; 3]) -> Vec<[u32; 3]> {
        let mut chain = vec![base];
        loop {
            let p = *chain.last().unwrap();
            let n = [(p[0] / 2).max(1), (p[1] / 2).max(1), (p[2] / 2).max(1)];
            if n == p {
                return chain;
            }
            chain.push(n);
        }
    }

    #[test]
    fn mip_chain_resolution_sequence_is_the_halving_chain() {
        for &res in &[[8u32, 8, 8], [6, 6, 6], [5, 3, 16], [1, 4, 4]] {
            let mut volume: Volume3D<f32> = Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
            for (i, v) in volume.data.iter_mut().enumerate() {
                *v = i as f32;
            }
            let mips = generate_mip_chain(&volume);
            let want = mip_resolutions(res);
            assert_eq!(
                mips.len(),
                want.len() - 1,
                "res {res:?}: expected the chain {want:?}"
            );
            for (level, data) in mips.iter().enumerate() {
                let r = want[level + 1];
                assert_eq!(
                    data.len(),
                    (r[0] * r[1] * r[2]) as usize,
                    "res {res:?} level {}: expected {r:?}",
                    level + 1
                );
            }
        }
    }

    /// oracle: "min-downsample preserves the SDF distance property (closest
    /// surface wins)" means no base voxel may be left out of the chain — a mip
    /// that drops one reports a larger distance than the surface it stands for,
    /// and a hierarchy built on it skips past that surface.
    ///
    /// How the levels are partitioned is an implementation choice, so the oracle
    /// does not assume one. Instead it puts the unique minimum at **every base
    /// position in turn** and requires every level to still carry it: that holds
    /// exactly when the children of each level are partitioned among the parents
    /// with none dropped, whichever partition is used.
    #[test]
    fn no_base_voxel_is_dropped_from_any_mip_level() {
        let mut dropped: Vec<String> = Vec::new();
        for &res in &[[8u32, 8, 8], [6, 6, 6], [5, 5, 5], [12, 6, 3], [10, 10, 10]] {
            let total = (res[0] * res[1] * res[2]) as usize;
            for hole in 0..total {
                let mut volume: Volume3D<f32> =
                    Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
                volume.data.fill(10.0);
                volume.data[hole] = -1.0;

                for (level, data) in generate_mip_chain(&volume).iter().enumerate() {
                    let got = data.iter().copied().fold(f32::MAX, f32::min);
                    if got > -1.0 + 1e-6 {
                        dropped.push(format!(
                            "res {res:?}: the minimum at flat index {hole} is missing from level {} \
                             (that level's minimum is {got})",
                            level + 1
                        ));
                        break;
                    }
                }
            }
        }
        assert!(
            dropped.is_empty(),
            "{} base voxels are dropped from the mip chain:\n{}",
            dropped.len(),
            dropped
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// oracle: a *min* downsample selects a value, it does not blend, so on a
    /// volume whose voxels are all distinct every mip voxel must be one of the
    /// previous level's values (an average filter fails this immediately), and
    /// each level's minimum must equal the one above it (nothing is lost, and
    /// nothing below the true minimum is invented).
    #[test]
    fn mip_values_are_selected_from_the_previous_level_not_blended() {
        for &res in &[[8u32, 8, 8], [6, 6, 6], [5, 5, 5], [12, 6, 3]] {
            let mut volume: Volume3D<f32> = Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
            // All distinct, so "is one of the children" is a real constraint.
            for (i, v) in volume.data.iter_mut().enumerate() {
                *v = i as f32 * 0.5 - 3.0;
            }

            let mips = generate_mip_chain(&volume);
            let mut prev: &[f32] = &volume.data;
            for (level, data) in mips.iter().enumerate() {
                let prev_min = prev.iter().copied().fold(f32::MAX, f32::min);
                for (i, &got) in data.iter().enumerate() {
                    assert!(
                        prev.contains(&got),
                        "res {res:?} level {} voxel {i}: {got} is not any value of the level above",
                        level + 1
                    );
                }
                let got_min = data.iter().copied().fold(f32::MAX, f32::min);
                assert_eq!(
                    got_min,
                    prev_min,
                    "res {res:?} level {}: minimum moved from {prev_min} to {got_min}",
                    level + 1
                );
                prev = data;
            }
        }
    }

    /// oracle: the distance+gradient chain must keep the same two properties —
    /// no child is dropped (the unique minimum survives from every position),
    /// and the normal stored with a mip voxel is the normal *of* the child that
    /// supplied the distance, so the whole `(distance, n)` tuple must be one of
    /// the previous level's tuples rather than a blend.
    #[test]
    fn mip_chain_distgrad_keeps_the_min_child_tuple_and_drops_nothing() {
        use alice_sdf::volume::mipchain::generate_mip_chain_distgrad;
        use alice_sdf::volume::VoxelDistGrad;

        let tag = |i: usize, d: f32| VoxelDistGrad {
            distance: d,
            nx: i as f32,
            ny: -(i as f32),
            nz: 1.0,
        };

        for &res in &[[8u32, 8, 8], [6, 6, 6], [5, 5, 5], [12, 6, 3]] {
            // (a) no child is dropped: sweep the unique minimum over every base
            //     position and require every level to still carry it.
            let total = (res[0] * res[1] * res[2]) as usize;
            for hole in 0..total {
                let mut volume: Volume3D<VoxelDistGrad> =
                    Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
                for (i, v) in volume.data.iter_mut().enumerate() {
                    *v = tag(i, 10.0);
                }
                volume.data[hole] = tag(hole, -1.0);

                for (level, data) in generate_mip_chain_distgrad(&volume).iter().enumerate() {
                    let got = data
                        .iter()
                        .map(|v| v.distance)
                        .fold(f32::MAX, |a, b| a.min(b));
                    assert!(
                        got <= -1.0 + 1e-6,
                        "res {res:?}: the minimum at flat index {hole} is missing from distgrad \
                         level {} (that level's minimum is {got})",
                        level + 1
                    );
                }
            }

            // (b) the stored tuple is one of the previous level's tuples, and it
            //     is one whose distance is the minimum that survived.
            let mut volume: Volume3D<VoxelDistGrad> =
                Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
            for (i, v) in volume.data.iter_mut().enumerate() {
                *v = tag(i, i as f32 * 0.5 - 3.0);
            }
            let mips = generate_mip_chain_distgrad(&volume);
            let mut prev: &[VoxelDistGrad] = &volume.data;
            for (level, data) in mips.iter().enumerate() {
                for (i, got) in data.iter().enumerate() {
                    let found = prev.iter().any(|m| {
                        m.distance == got.distance
                            && m.nx == got.nx
                            && m.ny == got.ny
                            && m.nz == got.nz
                    });
                    assert!(
                        found,
                        "res {res:?} level {} voxel {i}: tuple (d={}, nx={}) is not any tuple of \
                         the level above — the normal was blended away from its child",
                        level + 1,
                        got.distance,
                        got.nx
                    );
                }
                prev = data;
            }
        }
    }

    /// oracle: the coarsest level of a min chain stands for the whole volume,
    /// so it must equal the global minimum of the base data exactly.
    #[test]
    fn coarsest_mip_equals_the_global_minimum() {
        let mut wrong: Vec<String> = Vec::new();
        for &res in &[
            [8u32, 8, 8],
            [6, 6, 6],
            [4, 4, 4],
            [12, 6, 3],
            [16, 16, 16],
            [10, 10, 10],
        ] {
            let mut volume: Volume3D<f32> = Volume3D::new(res, Vec3::splat(LO), Vec3::splat(HI));
            // Deterministic values with the unique minimum in the far corner.
            for (i, v) in volume.data.iter_mut().enumerate() {
                *v = 1.0 + (i % 7) as f32;
            }
            let last = volume.data.len() - 1;
            volume.data[last] = -3.5;
            let global_min = volume.data.iter().copied().fold(f32::MAX, f32::min);

            let mips = generate_mip_chain(&volume);
            let coarsest = mips.last().expect("a chain with at least one level");
            assert_eq!(
                coarsest.len(),
                1,
                "res {res:?}: coarsest level must be 1 voxel"
            );
            if coarsest[0] != global_min {
                wrong.push(format!(
                    "res {res:?}: coarsest mip {} but the global minimum is {global_min}",
                    coarsest[0]
                ));
            }
        }
        assert!(
            wrong.is_empty(),
            "the coarsest min-mip must be the global minimum, {} wrong:\n{}",
            wrong.len(),
            wrong.join("\n")
        );
    }
}

// ---------------------------------------------------------------------------
// volume, GPU device required (no silent skip: the gpu-parity CI job sets
// ALICE_SDF_REQUIRE_GPU=1 and then a missing adapter is a failure)
// ---------------------------------------------------------------------------

#[cfg(feature = "volume")]
mod volume_gpu_oracle {
    use super::Vec3;
    use alice_sdf::types::SdfNode;
    use alice_sdf::volume::{bake_volume, gpu_bake::gpu_bake_volume, BakeConfig};

    /// oracle: the GPU bake must be the same analytic sphere field the CPU bake
    /// is checked against in `bake_volume_equals_the_analytic_sphere_at_grid_nodes`,
    /// evaluated at the same grid nodes.
    #[test]
    fn gpu_bake_equals_the_analytic_sphere_at_grid_nodes() {
        let res = [16u32, 16, 16];
        let config = BakeConfig {
            resolution: res,
            bounds_min: Vec3::splat(-2.0),
            bounds_max: Vec3::splat(2.0),
            ..Default::default()
        };

        let gpu = match gpu_bake_volume(&SdfNode::sphere(1.0), &config) {
            Ok(v) => v,
            Err(e) => {
                assert!(
                    std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                    "ALICE_SDF_REQUIRE_GPU is set but the GPU volume bake failed: {e}"
                );
                eprintln!("skipping GPU volume bake oracle: {e}");
                return;
            }
        };

        let cpu = bake_volume(&SdfNode::sphere(1.0), &config);
        assert_eq!(gpu.resolution, cpu.resolution);
        assert_eq!(gpu.world_min, cpu.world_min);
        assert_eq!(gpu.world_max, cpu.world_max);

        let size = config.bounds_max - config.bounds_min;
        let step = size
            / Vec3::new(
                res[0] as f32 - 1.0,
                res[1] as f32 - 1.0,
                res[2] as f32 - 1.0,
            );
        let mut worst_analytic = 0.0f32;
        let mut worst_parity = 0.0f32;
        for z in 0..res[2] {
            for y in 0..res[1] {
                for x in 0..res[0] {
                    let p = config.bounds_min + Vec3::new(x as f32, y as f32, z as f32) * step;
                    let want = p.length() - 1.0;
                    worst_analytic = worst_analytic.max((gpu.get(x, y, z) - want).abs());
                    worst_parity = worst_parity.max((gpu.get(x, y, z) - cpu.get(x, y, z)).abs());
                }
            }
        }
        assert!(
            worst_analytic < 1e-5,
            "GPU bake worst analytic error {worst_analytic}"
        );
        assert!(
            worst_parity < 1e-5,
            "GPU vs CPU bake worst difference {worst_parity}"
        );
    }
}

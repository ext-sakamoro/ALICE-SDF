//! Oracles for the GI API that `examples/gi_cone_trace.rs` wires: probe access
//! on the irradiance grid, its size accounting, and cone tracing inside a
//! closed spherical cavity.
//!
//! | target | oracle |
//! |---|---|
//! | `IrradianceGrid::get_probe` / `get_probe_mut` | flat index `x + y·sx + z·sx·sy`, probe at the cell centre `min + (i + ½)·size/n`, `None` as soon as one coordinate is outside its axis |
//! | `IrradianceGrid::sample` | at a probe centre the trilinear weights are 0 / 1, so the sample is that probe's own `evaluate(normal)` (a probe edited through `get_probe_mut` shows up there and nowhere it should not) |
//! | `probe_count` / `memory_bytes` | `sx·sy·sz` and `count · size_of::<IrradianceProbe>()` (64-byte aligned) |
//! | `cone_trace` / `trace_hemisphere` in a cavity | every cone is stopped by the wall before `max_distance`, so its openness is `1 − α ≤ 0.05` (the loop's stopping rule), its colour is `ρ·ambient·α + sky·(1 − α)` with `ρ·ambient = 0.5·0.05` and no light, and the hemisphere mixes those with weights `1 − ao·α`: each channel lies in `[0.95·0.025·(1 − ao), (0.025 + 0.05·max sky)·(1 − 0.95·ao)]` |
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "gi")]

use alice_sdf::gi::irradiance::SH1;
use alice_sdf::gi::{
    cone_trace, trace_hemisphere, ConeTraceConfig, IrradianceGrid, IrradianceProbe,
};
use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};
use alice_sdf::types::SdfNode;
use glam::Vec3;

#[test]
fn probes_are_addressed_per_axis_and_sit_on_cell_centres() {
    let (lo, hi) = (Vec3::new(-2.0, 0.0, 1.0), Vec3::new(2.0, 3.0, 5.0));
    let grid = IrradianceGrid::new([4, 3, 2], lo, hi);
    assert_eq!(grid.probe_count(), 24);
    assert_eq!(std::mem::size_of::<IrradianceProbe>() % 64, 0);
    assert_eq!(
        grid.memory_bytes(),
        24 * std::mem::size_of::<IrradianceProbe>()
    );
    let step = (hi - lo) / Vec3::new(4.0, 3.0, 2.0);
    for z in 0..2 {
        for y in 0..3 {
            for x in 0..4 {
                let p = grid.get_probe(x, y, z).expect("in range");
                let want = lo + (Vec3::new(x as f32, y as f32, z as f32) + 0.5) * step;
                assert!((p.position - want).length() < 1e-6);
                assert_eq!(
                    p.position,
                    grid.probes[(x + 4 * y + 12 * z) as usize].position
                );
            }
        }
    }
    // one coordinate past its axis: no probe, even when the flat index would
    // land on another probe (x = 4 is the start of the next row)
    for (x, y, z) in [
        (4, 0, 0),
        (4, 1, 0),
        (0, 3, 0),
        (3, 3, 0),
        (0, 0, 2),
        (u32::MAX, 0, 0),
    ] {
        assert!(grid.get_probe(x, y, z).is_none(), "({x},{y},{z})");
    }
    let mut g = IrradianceGrid::new([4, 3, 2], lo, hi);
    assert!(g.get_probe_mut(4, 0, 0).is_none());
    assert!(g.get_probe_mut(1, 2, 1).is_some());
}

fn constant_probe(value: Vec3) -> (SH1, SH1, SH1) {
    // evaluate = c₀·Y₀ with Y₀ = √(1/4π)
    let y0 = (1.0 / (4.0 * std::f32::consts::PI)).sqrt();
    let sh = |v: f32| SH1 {
        coeffs: [v / y0, 0.0, 0.0, 0.0],
    };
    (sh(value.x), sh(value.y), sh(value.z))
}

#[test]
fn a_probe_edited_in_place_is_what_sample_returns_at_its_centre() {
    let mut grid = IrradianceGrid::new([3, 3, 3], Vec3::splat(-1.5), Vec3::splat(1.5));
    let value = Vec3::new(0.7, 0.2, 0.4);
    let (r, g, b) = constant_probe(value);
    {
        let p = grid.get_probe_mut(2, 1, 0).unwrap();
        p.sh_r = r;
        p.sh_g = g;
        p.sh_b = b;
    }
    let centre = grid.get_probe(2, 1, 0).unwrap().position;
    for n in [Vec3::X, Vec3::Y, Vec3::new(0.3, -0.5, 0.8).normalize()] {
        let s = grid.sample(centre, n);
        assert!(
            (s - value).abs().max_element() < 1e-6,
            "{s} at the edited probe"
        );
        assert_eq!(grid.get_probe(2, 1, 0).unwrap().evaluate(n), s);
        // the probe one row up (index 2 + 3 = 5, x wrapped) stays black
        let other = grid.get_probe(0, 2, 0).unwrap().position;
        assert_eq!(grid.sample(other, n), Vec3::ZERO);
    }
}

fn cavity_svo(radius: f32) -> SparseVoxelOctree {
    // solid block with a spherical hole at the origin
    let shape = SdfNode::box3d(8.0, 8.0, 8.0).subtract(SdfNode::sphere(radius));
    SparseVoxelOctree::build(
        &shape,
        &SvoBuildConfig {
            max_depth: 6,
            bounds_min: Vec3::splat(-4.0),
            bounds_max: Vec3::splat(4.0),
            use_compiled: false,
            ..Default::default()
        },
    )
}

#[test]
fn every_cone_from_inside_a_cavity_is_stopped_by_the_wall() {
    let svo = cavity_svo(1.5);
    let cfg = ConeTraceConfig::default();
    let sky_max = 1.0f32; // the brightest sky channel (blue, at the zenith)
    let mut n = 0;
    for i in 0..64 {
        let t = i as f32 * 0.618;
        let z = 1.0 - 2.0 * (i as f32 + 0.5) / 64.0;
        let rxy = (1.0 - z * z).sqrt();
        let dir = Vec3::new(
            rxy * (t * std::f32::consts::TAU).cos(),
            rxy * (t * std::f32::consts::TAU).sin(),
            z,
        );
        let r = cone_trace(&svo, Vec3::new(0.1, -0.2, 0.05), dir, &cfg, None);
        assert!(r.occlusion <= 0.05, "dir {dir}: openness {}", r.occlusion);
        let alpha = 1.0 - r.occlusion;
        for c in r.color.to_array() {
            assert!(
                c >= 0.025 * alpha - 1e-6 && c <= 0.025 * alpha + sky_max * (1.0 - alpha) + 1e-6
            );
        }
        n += 1;
    }
    assert_eq!(n, 64);
}

#[test]
fn hemisphere_inside_a_cavity_is_dark_within_the_closed_form_band() {
    let svo = cavity_svo(1.5);
    let empty = SparseVoxelOctree::build(
        &SdfNode::sphere(0.1).translate(100.0, 0.0, 0.0),
        &SvoBuildConfig {
            max_depth: 3,
            bounds_min: Vec3::splat(-4.0),
            bounds_max: Vec3::splat(4.0),
            use_compiled: false,
            ..Default::default()
        },
    );
    let cfg = ConeTraceConfig::default();
    let ao = cfg.ao_weight;
    let (lo, hi) = (
        0.95 * 0.025 * (1.0 - ao),
        (0.025 + 0.05) * (1.0 - 0.95 * ao),
    );
    for normal in [
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::X,
        Vec3::new(1.0, 1.0, -1.0).normalize(),
    ] {
        let inside = trace_hemisphere(&svo, Vec3::ZERO, normal, &cfg, None);
        for c in inside.to_array() {
            assert!(
                (lo..=hi).contains(&c),
                "normal {normal}: channel {c} outside [{lo}, {hi}]"
            );
        }
        // the same trace in open space sees the sky, an order of magnitude brighter
        let open = trace_hemisphere(&empty, Vec3::ZERO, normal, &cfg, None);
        assert!(open.max_element() > 4.0 * hi, "open {open}");
    }
}

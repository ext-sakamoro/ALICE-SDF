//! Cone-traced global illumination over a sparse voxel octree: hemisphere
//! traces on an open floor and inside a closed cavity, and an irradiance probe
//! grid baked from the octree and sampled at a point.
//!
//! Run: `cargo run --example gi_cone_trace --features gi`
//!
//! The cavity values are checked against the closed-form band derived in
//! `tests/test_gi_api_oracle.rs`; the open-sky values against the sky model.
//!
//! Author: Moroya Sakamoto

use alice_sdf::gi::irradiance::SH1;
use alice_sdf::gi::{
    bake_irradiance_grid, sky_color, trace_hemisphere, BakeGiConfig, ConeTraceConfig,
    DirectionalLight, IrradianceProbe,
};
use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig};
use alice_sdf::types::SdfNode;
use glam::Vec3;

fn octree(shape: &SdfNode) -> SparseVoxelOctree {
    SparseVoxelOctree::build(
        shape,
        &SvoBuildConfig {
            max_depth: 6,
            bounds_min: Vec3::splat(-4.0),
            bounds_max: Vec3::splat(4.0),
            use_compiled: false,
            ..Default::default()
        },
    )
}

fn main() {
    let cones = ConeTraceConfig::default();
    let sun = DirectionalLight::default();

    // open floor: the hemisphere above it sees mostly sky
    let floor = octree(&SdfNode::box3d(8.0, 1.0, 8.0).translate(0.0, -0.5, 0.0));
    let open = trace_hemisphere(
        &floor,
        Vec3::new(0.0, 0.05, 0.0),
        Vec3::Y,
        &cones,
        Some(&sun),
    );
    println!(
        "open floor, up-facing:   {open:.4} (zenith sky {:.4})",
        sky_color(Vec3::Y)
    );

    // closed cavity: every cone is stopped by the wall
    let cavity = octree(&SdfNode::box3d(8.0, 8.0, 8.0).subtract(SdfNode::sphere(1.5)));
    let inside = trace_hemisphere(&cavity, Vec3::ZERO, Vec3::Y, &cones, None);
    let ao = cones.ao_weight;
    let (lo, hi) = (
        0.95 * 0.025 * (1.0 - ao),
        (0.025 + 0.05) * (1.0 - 0.95 * ao),
    );
    println!("inside a cavity, no sun: {inside:.4} (closed-form band [{lo:.4}, {hi:.4}])");
    for c in inside.to_array() {
        assert!((lo..=hi).contains(&c));
    }
    assert!(open.max_element() > 4.0 * hi);

    // probe grid baked over the open floor
    let config = BakeGiConfig {
        grid_size: [4, 3, 4],
        samples_per_probe: 16,
        ..Default::default()
    };
    let mut grid = bake_irradiance_grid(&floor, &config);
    let p: IrradianceProbe = *grid.get_probe(1, 2, 1).expect("probe in range");
    let up = p.evaluate(Vec3::Y);
    println!(
        "probe grid: {} probes, {} bytes; probe (1,2,1) at {:.2} sees {up:.4} from above (SH DC red {:.4})",
        grid.probe_count(),
        grid.memory_bytes(),
        p.position,
        p.sh_r.evaluate(Vec3::ZERO)
    );
    assert_eq!(grid.probe_count(), 48);
    assert!(
        grid.get_probe(4, 0, 0).is_none(),
        "x = 4 is outside a 4-wide grid"
    );
    let s = grid.sample(p.position, Vec3::Y);
    assert!(
        (s - up).abs().max_element() < 1e-5,
        "sampling at a probe centre returns it"
    );

    // overwrite one probe with a constant field and read it back
    let y0 = (1.0 / (4.0 * std::f32::consts::PI)).sqrt();
    let flat = SH1 {
        coeffs: [0.5 / y0, 0.0, 0.0, 0.0],
    };
    {
        let probe = grid.get_probe_mut(0, 0, 0).unwrap();
        (probe.sh_r, probe.sh_g, probe.sh_b) = (flat, flat, flat);
    }
    let corner = grid.get_probe(0, 0, 0).unwrap().position;
    let v = grid.sample(corner, Vec3::X);
    println!("edited probe (0,0,0) samples {v:.4} (set to 0.5)");
    assert!((v - Vec3::splat(0.5)).abs().max_element() < 1e-5);
}

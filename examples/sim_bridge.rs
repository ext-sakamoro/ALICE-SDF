//! An SDF shape as an ALICE-Physics collider with simulation modifiers
//! (feature `physics`; the GPU mesh part also needs `gpu`).
//!
//! ```sh
//! cargo run --example sim_bridge --features physics
//! cargo run --example sim_bridge --features physics,gpu
//! ```

use std::sync::Arc;

use alice_physics::erosion::ErosionConfig;
use alice_physics::fracture::FractureConfig;
use alice_physics::phase_change::PhaseChangeConfig;
use alice_physics::pressure::PressureConfig;
use alice_physics::sdf_collider::SdfField;
use alice_physics::thermal::ThermalConfig;
use alice_sdf::compiled::{eval_compiled, CompiledSdf};
use alice_sdf::physics_bridge::{sdf_to_physics_field, CompiledSdfField};
use alice_sdf::prelude::*;
use alice_sdf::sim_bridge::{attach_physics, simulate_sdf, SimulatedSdf};

fn main() {
    let node = SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.4, 1.2, 0.4), 0.2);

    // A physics field, and a second one sharing the same compiled program.
    let field = sdf_to_physics_field(&node);
    let fine = CompiledSdfField::from_arc(field.arc()).with_epsilon(1e-4);
    let own = CompiledSdfField::new(CompiledSdf::compile(&node));
    println!(
        "field: {} instructions, distance at (2,0,0) = {}, normal = {:?}, eps {} / {}",
        fine.compiled().instruction_count(),
        own.distance(2.0, 0.0, 0.0),
        fine.normal(2.0, 0.0, 0.0),
        field.epsilon,
        fine.epsilon
    );

    // Simulation modifiers on top.
    let mut sim = attach_physics(&node, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
    let cold = ThermalConfig {
        ambient_temperature: -30.0,
        ..ThermalConfig::default()
    };
    let inflate = PressureConfig {
        internal_pressure: 5.0,
        ..PressureConfig::default()
    };
    let ids = [
        sim.add_thermal(cold, 8),
        sim.add_pressure(inflate, 8),
        sim.add_erosion(ErosionConfig::default(), 8),
        sim.add_fracture(FractureConfig::default(), 8),
        sim.add_phase_change(PhaseChangeConfig::default(), 8),
    ];
    for _ in 0..4 {
        sim.update(1.0 / 60.0);
    }
    let names: Vec<String> = ids
        .iter()
        .map(|&i| {
            sim.modifier_mut(i)
                .map(|m| m.name().to_string())
                .unwrap_or_default()
        })
        .collect();
    let base = eval_compiled(sim.compiled(), Vec3::new(2.0, 0.0, 0.0));
    println!(
        "{} modifiers {names:?} in {:?}: distance at (2,0,0) {base} -> {}, normal {:?}",
        sim.modifier_count(),
        sim.bounds(),
        sim.distance(2.0, 0.0, 0.0),
        sim.normal(2.0, 0.0, 0.0)
    );
    // Freeze growth 0.2 and expansion 0.05 both pull the surface outwards.
    assert!(sim.distance(2.0, 0.0, 0.0) < base);
    sim.clear_modifiers();
    assert_eq!(sim.distance(2.0, 0.0, 0.0), base);

    // The other constructors.
    let shared = SimulatedSdf::from_arc(Arc::new(CompiledSdf::compile(&node)));
    let plain = SimulatedSdf::new(CompiledSdf::compile(&node))
        .with_bounds((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0));
    println!(
        "simulate_sdf bounds {:?}, with_bounds {:?}, from_arc distance {}",
        simulate_sdf(&node).bounds(),
        plain.bounds(),
        shared.distance(2.0, 0.0, 0.0)
    );

    #[cfg(feature = "gpu")]
    {
        use alice_sdf::mesh::gpu_marching_cubes::GpuMarchingCubesConfig;
        use alice_sdf::sim_bridge::{gpu_mesh_with_physics, GpuPhysicsBundle};
        match gpu_mesh_with_physics(
            &node,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &GpuMarchingCubesConfig::default(),
        ) {
            Ok(GpuPhysicsBundle { mesh, sim }) => println!(
                "GPU mesh: {} vertices, physics bounds {:?}",
                mesh.vertices.len(),
                sim.bounds()
            ),
            Err(e) => println!("GPU mesh skipped: {e}"),
        }
    }
}

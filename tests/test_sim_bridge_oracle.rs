//! Oracle tests for `sim_bridge` and the constructors / accessors of
//! `physics_bridge::CompiledSdfField` (feature `physics`).
//!
//! The boundary determinism of `CompiledSdfField` as an `SdfField` (distance,
//! normal, contacts) is pinned by `test_physics_bridge_determinism.rs`; this
//! file does not repeat it. Here:
//!
//! * no modifiers: `SimulatedSdf::distance` is `eval_compiled` bit for bit, and
//!   `normal` is the central difference with step 1e-3 that
//!   `alice_physics::sim_modifier::ModifiedSdf` documents, recomputed here;
//! * each convenience adder installs the modifier its name says (checked
//!   through `PhysicsModifier::name`) at the returned index;
//! * closed-form offsets of alice-physics 1.4 modifiers on a uniform field:
//!   thermal freeze growth `min((T_freeze - T_ambient) * rate, 1)` when the
//!   ambient temperature is below freezing, pressure expansion
//!   `internal_pressure * expansion_rate`; erosion, fracture and phase change
//!   leave a fresh field unchanged;
//! * a modifier defined here (`d - t`, `t += dt` per update) for the generic
//!   `add_modifier` / `update` / `modifier_mut` path.

#![cfg(feature = "physics")]

use std::sync::Arc;

use alice_physics::erosion::ErosionConfig;
use alice_physics::fracture::FractureConfig;
use alice_physics::phase_change::PhaseChangeConfig;
use alice_physics::pressure::PressureConfig;
use alice_physics::sdf_collider::SdfField;
use alice_physics::sim_modifier::PhysicsModifier;
use alice_physics::thermal::ThermalConfig;
use alice_sdf::compiled::{eval_compiled, eval_compiled_normal, CompiledSdf};
use alice_sdf::physics_bridge::{sdf_to_physics_field, CompiledSdfField};
use alice_sdf::prelude::*;
use alice_sdf::sim_bridge::{attach_physics, simulate_sdf, SimulatedSdf};

fn scene() -> SdfNode {
    SdfNode::sphere(1.0).smooth_union(SdfNode::box3d(0.4, 1.2, 0.4), 0.2)
}

fn points() -> Vec<[f32; 3]> {
    let mut v = Vec::new();
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                v.push([
                    -1.6 + 1.05 * i as f32,
                    -1.55 + 1.1 * j as f32,
                    -1.5 + 0.95 * k as f32,
                ]);
            }
        }
    }
    v
}

/// Central-difference normal of a distance function, step `e`, as documented
/// by `ModifiedSdf::normal`.
fn central_normal(f: impl Fn(f32, f32, f32) -> f32, p: [f32; 3], e: f32) -> (f32, f32, f32) {
    let [x, y, z] = p;
    let dx = f(x + e, y, z) - f(x - e, y, z);
    let dy = f(x, y + e, z) - f(x, y - e, z);
    let dz = f(x, y, z + e) - f(x, y, z - e);
    let len = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();
    if len < 1e-10 {
        (0.0, 1.0, 0.0)
    } else {
        (dx / len, dy / len, dz / len)
    }
}

#[test]
fn compiled_field_constructors_share_one_compiled_sdf() {
    let node = scene();
    let field = sdf_to_physics_field(&node);
    assert_eq!(field.epsilon, 0.001);
    let arc = field.arc();
    assert!(Arc::ptr_eq(&arc, &field.arc()));
    let shared = CompiledSdfField::from_arc(Arc::clone(&arc)).with_epsilon(0.01);
    assert!(Arc::ptr_eq(&shared.arc(), &arc));
    assert_eq!(shared.epsilon, 0.01);
    assert!(std::ptr::eq(shared.compiled(), field.compiled()));

    let own = CompiledSdfField::new(CompiledSdf::compile(&node));
    let mut compared = 0;
    for p in points() {
        let v = Vec3::from(p);
        let want = eval_compiled(&arc, v);
        assert_eq!(own.distance(p[0], p[1], p[2]).to_bits(), want.to_bits());
        assert_eq!(shared.distance(p[0], p[1], p[2]).to_bits(), want.to_bits());
        // with_epsilon reaches the normal stencil.
        let n = eval_compiled_normal(&arc, v, 0.01);
        let (nx, ny, nz) = shared.normal(p[0], p[1], p[2]);
        assert_eq!(
            [nx, ny, nz].map(f32::to_bits),
            [n.x, n.y, n.z].map(f32::to_bits)
        );
        compared += 1;
    }
    assert_eq!(compared, 64);
}

#[test]
fn without_modifiers_the_simulated_field_is_the_compiled_field() {
    let node = scene();
    let compiled = CompiledSdf::compile(&node);
    let sims = [
        simulate_sdf(&node),
        SimulatedSdf::new(CompiledSdf::compile(&node)),
        SimulatedSdf::from_arc(Arc::new(CompiledSdf::compile(&node))),
        attach_physics(&node, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
    ];
    let mut compared = 0;
    for sim in &sims {
        assert_eq!(sim.modifier_count(), 0);
        assert_eq!(
            sim.compiled().instruction_count(),
            compiled.instruction_count()
        );
        for p in points() {
            let want = eval_compiled(&compiled, Vec3::from(p));
            assert_eq!(sim.distance(p[0], p[1], p[2]).to_bits(), want.to_bits());
            let n = central_normal(
                |x, y, z| eval_compiled(&compiled, Vec3::new(x, y, z)),
                p,
                1e-3,
            );
            let got = sim.normal(p[0], p[1], p[2]);
            assert_eq!(
                [got.0, got.1, got.2].map(f32::to_bits),
                [n.0, n.1, n.2].map(f32::to_bits)
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 4 * 64);
}

#[test]
fn bounds_are_the_default_or_the_given_ones() {
    let node = SdfNode::sphere(1.0);
    assert_eq!(
        simulate_sdf(&node).bounds(),
        ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
    );
    let b = ((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0));
    assert_eq!(attach_physics(&node, b.0, b.1).bounds(), b);
    assert_eq!(simulate_sdf(&node).with_bounds(b.0, b.1).bounds(), b);
}

#[test]
fn each_adder_installs_its_modifier_at_the_returned_index() {
    let mut sim = simulate_sdf(&scene());
    let added = [
        sim.add_thermal(ThermalConfig::default(), 4),
        sim.add_pressure(PressureConfig::default(), 4),
        sim.add_erosion(ErosionConfig::default(), 4),
        sim.add_fracture(FractureConfig::default(), 4),
        sim.add_phase_change(PhaseChangeConfig::default(), 4),
    ];
    assert_eq!(added, [0, 1, 2, 3, 4]);
    assert_eq!(sim.modifier_count(), 5);
    let names: Vec<String> = (0..5)
        .map(|i| sim.modifier_mut(i).expect("modifier").name().to_string())
        .collect();
    assert_eq!(
        names,
        ["thermal", "pressure", "erosion", "fracture", "phase_change"]
    );
    assert!(sim.modifier_mut(5).is_none());
    sim.clear_modifiers();
    assert_eq!(sim.modifier_count(), 0);
}

#[test]
fn modifiers_apply_their_closed_form_offsets_in_order() {
    let node = scene();
    let compiled = CompiledSdf::compile(&node);
    let thermal = ThermalConfig {
        ambient_temperature: -30.0,
        freeze_temperature: -10.0,
        freeze_rate: 0.01,
        ..ThermalConfig::default()
    };
    let pressure = PressureConfig {
        internal_pressure: 5.0,
        expansion_rate: 0.02,
        ..PressureConfig::default()
    };
    let freeze =
        ((thermal.freeze_temperature - thermal.ambient_temperature) * thermal.freeze_rate).min(1.0);
    let expand = pressure.internal_pressure * pressure.expansion_rate;

    let mut sim = attach_physics(&node, (-2.0, -2.0, -2.0), (2.0, 2.0, 2.0));
    sim.add_thermal(thermal, 6);
    sim.add_erosion(ErosionConfig::default(), 6);
    sim.add_fracture(FractureConfig::default(), 6);
    sim.add_phase_change(PhaseChangeConfig::default(), 6);
    sim.add_pressure(pressure, 6);

    let mut compared = 0;
    for p in points() {
        let base = eval_compiled(&compiled, Vec3::from(p));
        // thermal, then three identities, then pressure — f32 in that order.
        let want = (base - freeze) - expand;
        assert_eq!(
            sim.distance(p[0], p[1], p[2]).to_bits(),
            want.to_bits(),
            "{p:?}"
        );
        compared += 1;
    }
    assert_eq!(compared, 64);

    // Erosion and fracture keep a fresh field unchanged through updates.
    let mut quiet = simulate_sdf(&node);
    quiet.add_erosion(ErosionConfig::default(), 6);
    quiet.add_fracture(FractureConfig::default(), 6);
    for _ in 0..3 {
        quiet.update(0.1);
    }
    for p in points() {
        assert_eq!(
            quiet.distance(p[0], p[1], p[2]).to_bits(),
            eval_compiled(&compiled, Vec3::from(p)).to_bits()
        );
    }
}

/// `d - t` where `t` grows by `dt` on every update.
struct Grow {
    t: f32,
}

impl PhysicsModifier for Grow {
    fn modify_distance(&self, _x: f32, _y: f32, _z: f32, d: f32) -> f32 {
        d - self.t
    }
    fn update(&mut self, dt: f32) {
        self.t += dt;
    }
    fn name(&self) -> &'static str {
        "grow"
    }
}

#[test]
fn generic_modifiers_are_updated_and_applied() {
    let r = 1.0f32;
    let mut sim = simulate_sdf(&SdfNode::sphere(r));
    sim.add_modifier(Box::new(Grow { t: 0.0 }));
    assert_eq!(sim.modifier_count(), 1);
    assert_eq!(sim.modifier_mut(0).unwrap().name(), "grow");
    sim.update(0.25);
    sim.update(0.25);
    // Points on the axes: |p| is exact, so the sphere distance is exact.
    let mut compared = 0;
    for (p, len) in [
        ([2.0f32, 0.0, 0.0], 2.0f32),
        ([0.0, -3.0, 0.0], 3.0),
        ([0.0, 0.0, 0.5], 0.5),
    ] {
        assert_eq!(sim.distance(p[0], p[1], p[2]), (len - r) - 0.5);
        compared += 1;
    }
    assert_eq!(compared, 3);
    // A constant offset leaves the normal of the sphere radial.
    let (nx, ny, nz) = sim.normal(2.0, 0.0, 0.0);
    assert!((nx - 1.0).abs() < 1e-3 && ny.abs() < 1e-3 && nz.abs() < 1e-3);
    sim.clear_modifiers();
    assert_eq!(sim.distance(2.0, 0.0, 0.0), 1.0);
}

/// `gpu_mesh_with_physics` is `gpu_marching_cubes` plus `attach_physics` over
/// the same bounds. Skips without an adapter unless ALICE_SDF_REQUIRE_GPU=1.
#[cfg(feature = "gpu")]
#[test]
fn gpu_mesh_with_physics_is_gpu_mc_plus_attach_physics() {
    use alice_sdf::mesh::gpu_marching_cubes::{gpu_marching_cubes, GpuMarchingCubesConfig};
    use alice_sdf::sim_bridge::gpu_mesh_with_physics;

    let node = scene();
    let (lo, hi) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    let config = GpuMarchingCubesConfig::default();
    let bundle = match gpu_mesh_with_physics(&node, lo, hi, &config) {
        Ok(b) => b,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but GPU MC failed: {e}"
            );
            eprintln!("skip: no GPU adapter ({e})");
            return;
        }
    };
    let mesh = gpu_marching_cubes(&node, lo, hi, &config).expect("GPU MC");
    assert!(!mesh.vertices.is_empty());
    assert_eq!(bundle.mesh.indices, mesh.indices);
    assert_eq!(bundle.mesh.vertices.len(), mesh.vertices.len());
    for (a, b) in bundle.mesh.vertices.iter().zip(&mesh.vertices) {
        assert_eq!(a.position, b.position);
    }
    assert_eq!(bundle.sim.bounds(), ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)));
    assert_eq!(bundle.sim.modifier_count(), 0);
    let compiled = CompiledSdf::compile(&node);
    for p in points() {
        assert_eq!(
            bundle.sim.distance(p[0], p[1], p[2]).to_bits(),
            eval_compiled(&compiled, Vec3::from(p)).to_bits()
        );
    }
}

//! One shape for physics and rendering: a ball dropped on a slab carves a
//! crater, the world's collider sees it at once and the render mesh
//! re-meshes only the chunks around it.
//!
//! The slab is a `LiveSdf` registered as an SDF collider and as a world
//! participant. Contacts are read with `detect_sdf_contacts` (the per-step
//! contact record arrives with `alice-physics` 2.1) and handed to a
//! `FracturePolicy`, which carves craters with the impact rule of
//! `destruction_from_impact`.
//!
//! Run: `cargo run --example live_sdf --features physics`
//!
//! Author: Moroya Sakamoto

use alice_physics::fracture::{FractureConfig, FractureModifier};
use alice_physics::sdf_collider::{detect_sdf_contacts, SdfCollider};
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, Vec3Fix};
use alice_sdf::live_sdf::{
    wake_bodies_in, FracturePolicy, ImpactContact, LiveMesh, LiveMeshConfig, LiveSdf,
};
use alice_sdf::SdfNode;
use glam::Vec3;

fn main() {
    // A 5 x 2 x 5 slab with its top face at y = 0.
    let slab = LiveSdf::new(SdfNode::box3d(5.0, 2.0, 5.0).translate(0.0, -1.0, 0.0));

    let mut world = PhysicsWorld::new(PhysicsConfig::default());
    let radius = Fix128::from_f32(0.25);
    world.set_sdf_collision_radius(radius);
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(slab.clone()),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    world
        .add_participant(Box::new(slab.clone()))
        .expect("register the slab");
    let ball = world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(0.3, 3.0, -0.2),
        Fix128::ONE,
    ));

    let layout = LiveMeshConfig {
        origin: Vec3::new(-3.05, -2.05, -3.05),
        cell_size: 0.1,
        chunk_cells: 8,
        chunks: [8, 5, 8],
    };
    let total_chunks = 8 * 5 * 8;
    let mut mesh = LiveMesh::new(&slab, layout).expect("valid layout");
    let (lo, hi) = layout.domain();
    println!(
        "initial mesh: {} chunks over {lo} .. {hi}, {} triangles",
        total_chunks,
        mesh.merged().indices.len() / 3
    );

    // Hits faster than 5 m/s carve a crater of 0.1 m per m/s, 0.4 .. 0.9 m.
    let policy = FracturePolicy::new(Fix128::from_f32(5.0), 0.1, 0.4, 0.9).for_collider(0);
    let dt = Fix128::from_f32(1.0 / 60.0);
    let mut carved = 0;
    let mut remeshed = 0;
    // The world resolves a penetration inside its step, so the contact is
    // read after the step (bodies within 0.02 m of touching) with the
    // velocity the body had before it: the speed it hit with. A body still
    // falling through the band (the surface did not stop it) is not an
    // impact yet.
    let band = radius + Fix128::from_f32(0.02);
    for step in 0..360u32 {
        let before: Vec<Vec3Fix> = world.bodies.iter().map(|b| b.velocity).collect();
        world.step(dt);
        let contacts: Vec<ImpactContact> =
            detect_sdf_contacts(&world.bodies, &world.sdf_colliders, band)
                .into_iter()
                .map(|(body, c)| {
                    let hit = -before[body].dot(c.normal);
                    let still = -world.bodies[body].velocity.dot(c.normal);
                    let stopped = still.to_f32() < 0.5 * hit.to_f32();
                    ImpactContact {
                        body_index: body,
                        collider_index: 0,
                        point: c.point_b,
                        normal: c.normal,
                        depth: c.depth,
                        approach_speed: if stopped { hit } else { Fix128::ZERO },
                        substep: 0,
                    }
                })
                .collect();
        let generation = slab.generation();
        let n = slab.apply_impacts(&policy, &contacts);
        if n > 0 {
            carved += n;
            let changes = slab.changes_since(generation);
            let woken = wake_bodies_in(&mut world, &changes.regions, 0.25);
            let report = mesh.sync();
            remeshed += report.remeshed.len();
            println!(
                "step {step}: impact at {:.2} m/s carved {n} crater(s), woke {woken} body, re-meshed {} of {total_chunks} chunks (generation {})",
                contacts[0].approach_speed.to_f32(),
                report.remeshed.len(),
                report.generation
            );
        }
    }

    let y = world.bodies[ball].position.y.to_f32();
    let merged = mesh.merged();
    println!(
        "ball rests at y = {y:.3}; mesh {} triangles; craters {}",
        merged.indices.len() / 3,
        slab.crater_count()
    );

    assert_eq!(carved, 1, "one fast impact, then the ball rests");
    assert!(
        remeshed > 0 && remeshed < total_chunks,
        "only chunks near the crater"
    );
    assert!(
        y < 0.0,
        "the ball sits inside the crater, below the slab top"
    );
    for v in &merged.vertices {
        let d = slab.eval(v.position);
        assert!(
            d.abs() <= 0.1,
            "mesh vertex off the collider's surface: {d}"
        );
    }
    println!("ok: the collider and the mesh show the same crater");

    // The shape has no modifiers yet, so it is one SdfNode and the GPU
    // mesher can take it.
    #[cfg(feature = "gpu")]
    match slab.gpu_mesh(lo, hi, &alice_sdf::mesh::GpuMarchingCubesConfig::default()) {
        Ok(m) => println!(
            "GPU marching cubes of the same shape: {} vertices",
            m.vertices.len()
        ),
        Err(e) => println!("GPU marching cubes skipped: {e}"),
    }

    // A fracture modifier joins the shape. Stress at the impact point goes
    // through the handle; the world's steps grow the cracks (the shape is a
    // participant), and the collider and the mesh follow.
    let fracture = slab.add_modifier(FractureModifier::new(
        FractureConfig {
            fracture_toughness: 10.0,
            crack_width: 0.08,
            ..FractureConfig::default()
        },
        12,
        (-3.0, -3.0, -3.0),
        (3.0, 3.0, 3.0),
    ));
    let collider_handle = slab.clone();
    assert!(collider_handle.same_shape(&slab));
    slab.with_modifier_mut::<FractureModifier, _>(fracture, |m| {
        m.apply_stress_at(0.3, -0.6, -0.2, 500.0, 1.0);
    })
    .expect("modifier 0 is the fracture modifier");
    let generation = slab.generation();
    for _ in 0..4 {
        world.step(dt);
    }
    let changes = slab.changes_since(generation);
    let report = mesh.sync();
    let crater_chunk = [4, 2, 3];
    println!(
        "fracture: {} shape change(s) over 4 steps, re-meshed {} chunks, chunk {crater_chunk:?} has {} triangles",
        changes.regions.len(),
        report.remeshed.len(),
        mesh.chunk(crater_chunk).map_or(0, |m| m.indices.len() / 3)
    );
    assert!(!changes.regions.is_empty(), "the world's steps grew cracks");
    assert!(report.remeshed.len() < total_chunks);
    for v in &mesh.merged().vertices {
        assert!(collider_handle.eval(v.position).abs() <= 0.1);
    }

    // Without a world the same shape advances by `update` (an editor or a
    // replay viewer); the mesh follows the same way.
    let before = slab.generation();
    slab.update(1.0 / 60.0);
    let report = mesh.sync();
    let (cmin, cmax) = mesh.config().chunk_bounds(crater_chunk);
    println!(
        "offline update: {} modifier(s), generation {before} -> {}, re-meshed {} chunks; chunk {crater_chunk:?} spans {cmin} .. {cmax}",
        slab.modifier_count(),
        mesh.generation(),
        report.remeshed.len()
    );
    assert_eq!(mesh.generation(), slab.generation());
}

//! Oracles for `live_sdf`: one shape shared by the physics world (collider +
//! participant) and the chunked render mesh.
//!
//! Every expected value comes from a closed form or an independent
//! reference (`ModifiedSdf` of `alice-physics` updated by hand, a mesh built
//! from scratch), never from the code under test. Each test fails when it
//! compared nothing.
#![cfg(feature = "physics")]

use alice_physics::erosion::{ErosionConfig, ErosionModifier};
use alice_physics::fracture::{FractureConfig, FractureModifier};
use alice_physics::sdf_collider::{SdfCollider, SdfField};
use alice_physics::sim_modifier::ModifiedSdf;
use alice_physics::world_participant::Participant;
use alice_physics::{Fix128, PhysicsConfig, PhysicsWorld, QuatFix, RigidBody, Vec3Fix};
use alice_sdf::compiled::CompiledSdf;
use alice_sdf::live_sdf::{
    wake_bodies_in, DirtyRegion, FracturePolicy, ImpactContact, LiveMesh, LiveMeshConfig, LiveSdf,
    LiveSdfError,
};
use alice_sdf::mesh::Mesh;
use alice_sdf::physics_bridge::CompiledSdfField;
use alice_sdf::SdfNode;
use glam::Vec3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// A 5 × 2 × 5 slab whose top face is the plane y = 0.
fn slab() -> SdfNode {
    SdfNode::box3d(5.0, 2.0, 5.0).translate(0.0, -1.0, 0.0)
}

const CELL: f32 = 0.1;

/// Lattice of 64 × 40 × 64 cells in 8³-cell chunks. The lattice planes sit
/// half a cell off the slab's faces (y = 0, y = −2, x, z = ±2.5), so no
/// sample lands on the surface.
const fn layout() -> LiveMeshConfig {
    LiveMeshConfig {
        origin: Vec3::new(-3.05, -2.05, -3.05),
        cell_size: CELL,
        chunk_cells: 8,
        chunks: [8, 5, 8],
    }
}

const CRATER_C: Vec3 = Vec3::new(0.3, 0.0, -0.2);
const CRATER_R: f32 = 0.8;

fn world_with(live: &LiveSdf, as_participant: bool, substeps: usize) -> PhysicsWorld {
    let config = PhysicsConfig {
        substeps,
        ..PhysicsConfig::default()
    };
    let mut world = PhysicsWorld::new(config);
    world.add_sdf_collider(SdfCollider::new_static(
        Box::new(live.clone()),
        Vec3Fix::ZERO,
        QuatFix::IDENTITY,
    ));
    if as_participant {
        world
            .add_participant(Box::new(live.clone()))
            .expect("register the shape as a participant");
    }
    world
}

fn boundary_edges(mesh: &Mesh) -> usize {
    let mut count: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    count.values().filter(|&&n| n != 2).count()
}

fn signed_volume(mesh: &Mesh) -> f64 {
    let mut v = 0.0f64;
    for t in mesh.indices.chunks_exact(3) {
        let p = |i: u32| mesh.vertices[i as usize].position.as_dvec3();
        let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
        v += a.dot(b.cross(c)) / 6.0;
    }
    v
}

fn mesh_bits(m: &Mesh) -> (Vec<[u32; 6]>, Vec<u32>) {
    let v = m
        .vertices
        .iter()
        .map(|v| {
            [
                v.position.x.to_bits(),
                v.position.y.to_bits(),
                v.position.z.to_bits(),
                v.normal.x.to_bits(),
                v.normal.y.to_bits(),
                v.normal.z.to_bits(),
            ]
        })
        .collect();
    (v, m.indices.clone())
}

fn sample_points() -> Vec<Vec3> {
    let mut out = Vec::new();
    for i in 0..9 {
        for j in 0..5 {
            for k in 0..9 {
                out.push(Vec3::new(
                    -2.0 + 0.5 * i as f32,
                    -0.6 + 0.3 * j as f32,
                    -2.0 + 0.5 * k as f32,
                ));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// World sharing
// ---------------------------------------------------------------------------

/// oracle: a ball (collision radius ρ) resting on the plane y = 0 sits at
/// y = ρ; above the crater's centre the surface is `R + y = 0` (the sphere
/// term of max(slab, R − |p − c|)), so it rests R lower. The two balls are
/// compared to each other so the solver's resting penetration cancels.
#[test]
fn a_crater_carved_after_settling_drops_the_ball_by_its_radius() {
    let live = LiveSdf::new(slab());
    let mut world = world_with(&live, true, 8);
    let rho = 0.25;
    world.set_sdf_collision_radius(Fix128::from_f32(rho));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(CRATER_C.x, 1.0, CRATER_C.z),
        Fix128::ONE,
    ));
    world.add_body(RigidBody::new_dynamic(
        Vec3Fix::from_f32(2.0, 1.0, 1.5),
        Fix128::ONE,
    ));
    let dt = Fix128::from_f32(1.0 / 60.0);
    for _ in 0..240 {
        world.step(dt);
    }
    let flat_before = world.bodies[1].position.y.to_f32();
    let over_crater_before = world.bodies[0].position.y.to_f32();
    assert!(
        (over_crater_before - flat_before).abs() < 1e-3,
        "both balls rest on the plane: {over_crater_before} vs {flat_before}"
    );
    assert!(
        (flat_before - rho).abs() < 0.05,
        "resting height {flat_before} vs radius {rho}"
    );

    assert!(
        world.is_sleeping(0) && world.is_sleeping(1),
        "both balls are parked"
    );

    // Carve through the handle; the collider in the world is the same shape.
    let g = live.generation();
    live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    let p = Vec3::new(CRATER_C.x, -0.3, CRATER_C.z);
    let d = world.sdf_colliders[0].field.distance(p.x, p.y, p.z);
    assert!(
        (d - (CRATER_R + p.y)).abs() < 1e-6,
        "the world's collider answers with the carved shape at once: {d}"
    );
    // A parked body does not query the collider again until woken.
    for _ in 0..30 {
        world.step(dt);
    }
    assert_eq!(
        world.bodies[0].position.y.to_f32().to_bits(),
        over_crater_before.to_bits()
    );
    let woken = wake_bodies_in(&mut world, &live.changes_since(g).regions, rho);
    assert_eq!(woken, 1, "only the ball over the crater is in its box");
    assert!(!world.is_sleeping(0) && world.is_sleeping(1));
    for _ in 0..240 {
        world.step(dt);
    }
    let flat = world.bodies[1].position.y.to_f32();
    let in_crater = world.bodies[0].position.y.to_f32();
    assert_eq!(
        flat.to_bits(),
        flat_before.to_bits(),
        "the far ball does not move"
    );
    let drop = flat - in_crater;
    assert!(
        (drop - CRATER_R).abs() < 2e-3,
        "the ball sank {drop}, closed form {CRATER_R}"
    );
}

fn erosion() -> ErosionModifier {
    let config = ErosionConfig {
        rate: 1.0,
        hardness: 0.0,
        max_depth: 0.5,
        ..ErosionConfig::default()
    };
    let mut m = ErosionModifier::new(config, 12, (-3.0, -3.0, -3.0), (3.0, 3.0, 3.0));
    m.set_exposure_at(0.5, 0.0, 0.0, 1.0, 1.5);
    m
}

/// oracle: `ModifiedSdf` of `alice-physics` over the same compiled slab
/// with an identical erosion modifier, `update(h)` called by hand once per
/// world substep (substeps = 1, so h = dt). Without the participant the
/// collider must still be the bare slab.
#[test]
fn stepping_the_world_advances_the_modifiers_the_collider_sees() {
    let dt = Fix128::from_f32(1.0 / 30.0);
    let steps = 20;

    let wired = LiveSdf::new(slab());
    wired.add_modifier(erosion());
    let mut world = world_with(&wired, true, 1);

    let unwired = LiveSdf::new(slab());
    unwired.add_modifier(erosion());
    let mut world_unwired = world_with(&unwired, false, 1);

    let mut reference = ModifiedSdf::new(Box::new(CompiledSdfField::new(CompiledSdf::compile(
        &slab(),
    ))))
    .with_modifier(Box::new(erosion()));
    let bare = CompiledSdfField::new(CompiledSdf::compile(&slab()));

    let g0 = wired.generation();
    for _ in 0..steps {
        world.step(dt);
        world_unwired.step(dt);
        reference.update(dt.to_f32());
    }
    assert!(wired.generation() > g0, "the participant changed the shape");
    assert_eq!(
        unwired.generation(),
        g0,
        "an unregistered shape is not advanced"
    );

    let mut compared = 0;
    let mut moved = 0;
    for p in sample_points() {
        let got = world.sdf_colliders[0].field.distance(p.x, p.y, p.z);
        let want = reference.distance(p.x, p.y, p.z);
        assert_eq!(got.to_bits(), want.to_bits(), "at {p}: {got} vs {want}");
        let still = world_unwired.sdf_colliders[0].field.distance(p.x, p.y, p.z);
        let b = bare.distance(p.x, p.y, p.z);
        assert_eq!(still.to_bits(), b.to_bits(), "unwired at {p}");
        if got.to_bits() != b.to_bits() {
            moved += 1;
        }
        compared += 1;
    }
    assert!(compared > 0);
    assert!(moved > 0, "erosion must have moved the surface somewhere");
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

fn carved() -> LiveSdf {
    let live = LiveSdf::new(slab());
    live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    live
}

/// oracle: for a 1-Lipschitz field a marching-cubes vertex on an edge of
/// length h has |d| ≤ h (from |d(v) − d(endpoint)| ≤ distance along the
/// edge); on the crater wall d is the closed-form sphere term, on the flat
/// top the plane y = 0 (exact under linear interpolation).
#[test]
fn every_synced_vertex_lies_on_the_surface_the_collider_sees() {
    let live = LiveSdf::new(slab());
    let mut mesh = LiveMesh::new(&live, layout()).expect("mesh");
    live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    mesh.sync();
    let merged = mesh.merged();
    assert!(merged.vertices.len() > 1000);
    let (mut wall, mut top, mut rim) = (0, 0, 0);
    for v in &merged.vertices {
        let p = v.position;
        let d = live.eval(p);
        assert!(d.abs() <= CELL, "vertex {p} has distance {d}");
        let horizontal = Vec3::new(p.x - CRATER_C.x, 0.0, p.z - CRATER_C.z).length();
        if p.y < -1e-3 && p.y > -0.75 && horizontal < CRATER_R {
            let sphere = CRATER_R - (p - CRATER_C).length();
            assert!(
                sphere.abs() <= CELL,
                "wall vertex {p}: sphere term {sphere}"
            );
            wall += 1;
        }
        if p.y.abs() <= 1e-5 {
            assert!(
                horizontal >= CRATER_R - CELL,
                "top vertex {p} inside the crater mouth"
            );
            if horizontal < CRATER_R + CELL {
                rim += 1;
            }
            top += 1;
        }
    }
    assert!(
        wall > 50 && top > 100 && rim > 8,
        "wall {wall} top {top} rim {rim}"
    );
}

/// oracle: chunks are picked independently here from the crater box widened
/// by 3h (2h for the values a 1-Lipschitz field can change near a sign
/// change, h for the one-cell padding). Every chunk after the sync is
/// bit-identical to a mesh built from scratch, and the untouched chunks
/// are the very same allocations as before.
#[test]
fn sync_remeshes_only_the_chunks_the_crater_can_reach() {
    let live = LiveSdf::new(slab());
    let mut mesh = LiveMesh::new(&live, layout()).expect("mesh");
    let cfg = layout();
    let mut before = HashMap::new();
    let mut all = Vec::new();
    for z in 0..cfg.chunks[2] {
        for y in 0..cfg.chunks[1] {
            for x in 0..cfg.chunks[0] {
                let c = [x, y, z];
                before.insert(c, Arc::clone(mesh.chunk(c).expect("chunk")));
                all.push(c);
            }
        }
    }

    live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    let report = mesh.sync();
    assert_eq!(report.generation, live.generation());

    let lo = CRATER_C - Vec3::splat(CRATER_R + 3.0 * CELL);
    let hi = CRATER_C + Vec3::splat(CRATER_R + 3.0 * CELL);
    let len = 8.0 * CELL;
    let expected: HashSet<[usize; 3]> = all
        .iter()
        .copied()
        .filter(|c| {
            (0..3).all(|a| {
                let cmin = cfg.origin[a] + c[a] as f32 * len;
                let cmax = cmin + len;
                cmin <= hi[a] && cmax >= lo[a]
            })
        })
        .collect();
    let got: HashSet<[usize; 3]> = report.remeshed.iter().copied().collect();
    assert_eq!(got, expected);
    assert_eq!(got.len(), 4 * 3 * 3, "x 4, y 3, z 3 chunks");

    let fresh = LiveMesh::new(&carved(), layout()).expect("fresh");
    let mut reused = 0;
    for &c in &all {
        let now = mesh.chunk(c).expect("chunk");
        assert_eq!(
            mesh_bits(now),
            mesh_bits(fresh.chunk(c).expect("fresh chunk")),
            "chunk {c:?} differs from a mesh built from scratch"
        );
        if !got.contains(&c) {
            assert!(Arc::ptr_eq(now, &before[&c]), "chunk {c:?} was rebuilt");
            reused += 1;
        }
    }
    assert_eq!(reused, all.len() - got.len());
    assert!(reused > 0);

    let again = mesh.sync();
    assert!(
        again.remeshed.is_empty(),
        "nothing changed since the last sync"
    );
}

/// oracle: the slab with a half-ball crater is a closed solid, so the
/// joined chunks have no boundary edge; the removed volume is the half
/// ball (2/3)·π·R³.
#[test]
fn chunk_seams_close_and_the_crater_removes_a_half_ball() {
    let live = LiveSdf::new(slab());
    let mut mesh = LiveMesh::new(&live, layout()).expect("mesh");
    let v_before = signed_volume(&mesh.merged());
    live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    mesh.sync();
    let merged = mesh.merged();
    assert!(merged.indices.len() / 3 > 1000);
    assert_eq!(boundary_edges(&merged), 0, "seams must be closed");
    let removed = v_before - signed_volume(&merged);
    let cr = f64::from(CRATER_R);
    let half_ball = 2.0 / 3.0 * std::f64::consts::PI * (cr * cr * cr);
    assert!(
        ((removed - half_ball) / half_ball).abs() < 0.02,
        "removed {removed}, closed form {half_ball}"
    );
    assert!(
        (v_before - 50.0).abs() / 50.0 < 0.01,
        "slab volume {v_before}"
    );
}

/// Same edits twice: distances and meshes are bit-identical.
#[test]
fn the_same_edits_give_the_same_bits() {
    let build = || {
        let live = LiveSdf::new(slab());
        live.add_modifier(erosion());
        let mut mesh = LiveMesh::new(&live, layout()).expect("mesh");
        live.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
        live.subtract_sphere(Vec3::new(-1.2, 0.0, 1.0), 0.5)
            .expect("crater");
        for _ in 0..5 {
            live.update(1.0 / 30.0);
        }
        mesh.sync();
        (live, mesh)
    };
    let (a, ma) = build();
    let (b, mb) = build();
    let mut compared = 0;
    for p in sample_points() {
        assert_eq!(a.eval(p).to_bits(), b.eval(p).to_bits());
        compared += 1;
    }
    assert!(compared > 0);
    let (x, y) = (ma.merged(), mb.merged());
    assert!(!x.indices.is_empty());
    assert_eq!(mesh_bits(&x), mesh_bits(&y));
}

/// oracle: a fracture modifier reports the crack capsules' box; syncing on
/// that region alone still matches a from-scratch mesh, and the region
/// is a box, not the whole space.
#[test]
fn fracture_cracks_dirty_only_their_box_and_the_sync_still_matches() {
    let plain = || {
        let config = FractureConfig {
            fracture_toughness: 10.0,
            // Wider than the 3h margin of the sync, so a box that left out
            // the width would miss chunks.
            crack_width: 0.4,
            ..FractureConfig::default()
        };
        FractureModifier::new(config, 12, (-3.0, -3.0, -3.0), (3.0, 3.0, 3.0))
    };
    let make = || {
        let mut m = plain();
        m.apply_stress_at(0.0, -0.2, 0.0, 500.0, 1.0);
        m
    };
    let live = LiveSdf::new(slab());
    let index = live.add_modifier(plain());
    let mut mesh = LiveMesh::new(&live, layout()).expect("mesh");
    let g = live.generation();
    // Stress through the handle: it seeds cracks but is not shape yet.
    let applied = live.with_modifier_mut::<FractureModifier, _>(index, |m| {
        m.apply_stress_at(0.0, -0.2, 0.0, 500.0, 1.0);
    });
    assert_eq!(applied, Some(()));
    assert_eq!(
        live.generation(),
        g,
        "stress alone does not change the shape"
    );
    for _ in 0..4 {
        live.update(1.0 / 60.0);
    }
    let changes = live.changes_since(g);
    assert!(!changes.regions.is_empty(), "cracks must have appeared");
    assert!(
        changes
            .regions
            .iter()
            .all(|r| matches!(r, DirtyRegion::Aabb { .. })),
        "{:?}",
        changes.regions
    );
    let report = mesh.sync();
    assert!(!report.remeshed.is_empty());
    assert!(report.remeshed.len() < 8 * 5 * 8);

    let reference = LiveSdf::new(slab());
    reference.add_modifier(make());
    for _ in 0..4 {
        reference.update(1.0 / 60.0);
    }
    let fresh = LiveMesh::new(&reference, layout()).expect("fresh");
    assert_eq!(mesh_bits(&mesh.merged()), mesh_bits(&fresh.merged()));
}

// ---------------------------------------------------------------------------
// Impacts
// ---------------------------------------------------------------------------

fn contact(collider: usize, speed: f32, at: Vec3) -> ImpactContact {
    ImpactContact {
        body_index: 0,
        collider_index: collider,
        point: Vec3Fix::from_f32(at.x, at.y, at.z),
        normal: Vec3Fix::from_f32(0.0, 1.0, 0.0),
        depth: Fix128::from_f32(0.01),
        approach_speed: Fix128::from_f32(speed),
        substep: 0,
    }
}

/// oracle: radius = clamp(speed · scale, min, max), carved only above the
/// threshold (strictly) and only on the chosen collider.
#[test]
fn the_fracture_policy_carves_by_the_impact_rule() {
    let policy = FracturePolicy::new(Fix128::from_f32(2.0), 0.1, 0.3, 0.9).for_collider(0);
    let cases = [
        (3.0f32, 0usize, Some(0.3f32)), // 0.3 = clamp(0.3)
        (6.0, 0, Some(0.6)),
        (20.0, 0, Some(0.9)),
        (2.0, 0, None), // at the threshold: no
        (1.0, 0, None),
        (6.0, 1, None), // another collider
    ];
    let mut compared = 0;
    for (speed, collider, want) in cases {
        let got = policy.crater_for(&contact(collider, speed, Vec3::new(0.5, 0.0, -0.5)));
        match (got, want) {
            (None, None) => {}
            (Some((c, r)), Some(w)) => {
                assert!((r - w).abs() < 1e-6, "speed {speed}: radius {r} vs {w}");
                assert!((c - Vec3::new(0.5, 0.0, -0.5)).length() < 1e-6);
            }
            _ => panic!("speed {speed} collider {collider}: {got:?} vs {want:?}"),
        }
        compared += 1;
    }
    assert_eq!(compared, cases.len());

    let live = LiveSdf::new(slab());
    let contacts: Vec<_> = cases
        .iter()
        .map(|&(s, c, _)| contact(c, s, Vec3::new(0.0, 0.0, 0.0)))
        .collect();
    let g = live.generation();
    assert_eq!(live.apply_impacts(&policy, &contacts), 3);
    assert_eq!(live.crater_count(), 3);
    assert_eq!(live.generation(), g + 3);
    // The deepest crater (0.9) decides the floor under the impact point.
    let d = live.eval(Vec3::new(0.0, -0.5, 0.0));
    assert!((d - (0.9 - 0.5)).abs() < 1e-6, "{d}");
    assert_eq!(live.apply_impacts(&policy, &[]), 0);
}

// ---------------------------------------------------------------------------
// Snapshot, edges
// ---------------------------------------------------------------------------

#[test]
fn a_snapshot_restores_craters_and_modifiers_bit_for_bit() {
    let a = LiveSdf::new(slab());
    a.add_modifier(erosion());
    a.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    for _ in 0..6 {
        a.update(1.0 / 30.0);
    }
    let mut payload = Vec::new();
    a.write_state(&mut payload);

    let mut b = LiveSdf::new(slab());
    b.add_modifier(erosion());
    assert_eq!(b.check_state(&payload), Ok(()));
    let g = b.generation();
    b.read_state(&payload);
    assert_eq!(b.generation(), g + 1);
    assert_eq!(b.changes_since(g).regions, vec![DirtyRegion::Everywhere]);
    let mut compared = 0;
    for p in sample_points() {
        assert_eq!(a.eval(p).to_bits(), b.eval(p).to_bits(), "at {p}");
        compared += 1;
    }
    assert!(compared > 0);

    // Refusals leave the shape unchanged.
    assert!(b.check_state(&payload[..payload.len() - 1]).is_err());
    let mut longer = payload.clone();
    longer.push(0);
    assert!(b.check_state(&longer).is_err());
    let bare = LiveSdf::new(slab());
    assert!(
        bare.check_state(&payload).is_err(),
        "modifier count differs"
    );
}

#[test]
fn invalid_input_is_refused() {
    let live = LiveSdf::new(slab());
    assert_eq!(
        live.subtract_sphere(Vec3::ZERO, 0.0),
        Err(LiveSdfError::InvalidCrater)
    );
    assert_eq!(
        live.subtract_sphere(Vec3::ZERO, -1.0),
        Err(LiveSdfError::InvalidCrater)
    );
    assert_eq!(
        live.subtract_sphere(Vec3::new(f32::NAN, 0.0, 0.0), 1.0),
        Err(LiveSdfError::InvalidCrater)
    );
    assert_eq!(
        live.subtract_sphere(Vec3::ZERO, f32::INFINITY),
        Err(LiveSdfError::InvalidCrater)
    );
    assert_eq!(live.generation(), 0, "refused edits change nothing");
    assert!(live.changes_since(5).regions.is_empty());
    for bad in [
        LiveMeshConfig {
            cell_size: 0.0,
            ..layout()
        },
        LiveMeshConfig {
            chunk_cells: 0,
            ..layout()
        },
        LiveMeshConfig {
            chunks: [1, 0, 1],
            ..layout()
        },
        LiveMeshConfig {
            origin: Vec3::new(f32::NAN, 0.0, 0.0),
            ..layout()
        },
    ] {
        assert_eq!(
            LiveMesh::new(&live, bad).err(),
            Some(LiveSdfError::InvalidMeshConfig)
        );
    }
    assert!(live.to_sdf_node().is_some());
    live.add_modifier(erosion());
    assert!(live.to_sdf_node().is_none(), "modifiers have no node form");
    assert!(live
        .with_modifier_mut::<ErosionModifier, _>(7, |_| ())
        .is_none());
    assert!(
        live.with_modifier_mut::<FractureModifier, _>(0, |_| ())
            .is_none(),
        "modifier 0 is an erosion modifier"
    );
}

/// A clone is the same shape; an edit through one handle is seen by all.
#[test]
fn clones_share_one_shape() {
    let a = LiveSdf::new(slab());
    let b = a.clone();
    assert!(a.same_shape(&b));
    assert!(!a.same_shape(&LiveSdf::new(slab())));
    b.subtract_sphere(CRATER_C, CRATER_R).expect("crater");
    assert_eq!(a.crater_count(), 1);
    let p = Vec3::new(CRATER_C.x, -0.4, CRATER_C.z);
    assert!((a.eval(p) - (CRATER_R - 0.4)).abs() < 1e-6);
}

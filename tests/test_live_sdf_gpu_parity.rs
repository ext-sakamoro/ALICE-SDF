//! GPU parity of `LiveSdf::gpu_mesh`: the GPU marching cubes of the shape's
//! `SdfNode` form (base minus craters) against the CPU chunked mesh of the
//! same shape on the same lattice.
//!
//! oracle: `LiveMesh` (CPU, the distance the collider answers with). Every
//! GPU vertex must have a CPU vertex within `TOL` and the other way round;
//! the lattices agree up to the rounding of the GPU's `size / resolution`
//! cell, and both interpolate along the same edges.
//!
//! Needs a GPU adapter; with `ALICE_SDF_REQUIRE_GPU=1` (CI, preflight) a
//! missing adapter is a failure instead of a skip.
#![cfg(all(feature = "physics", feature = "gpu"))]

use alice_physics::erosion::{ErosionConfig, ErosionModifier};
use alice_sdf::live_sdf::{LiveMesh, LiveMeshConfig, LiveSdf, LiveSdfError};
use alice_sdf::mesh::{GpuMarchingCubesConfig, Mesh};
use alice_sdf::SdfNode;
use glam::Vec3;
use std::collections::HashMap;

const TOL: f32 = 1e-4;

const fn layout() -> LiveMeshConfig {
    LiveMeshConfig {
        origin: Vec3::new(-3.05, -2.05, -3.05),
        cell_size: 0.1,
        chunk_cells: 8,
        chunks: [8, 8, 8],
    }
}

fn shape() -> LiveSdf {
    let live = LiveSdf::new(SdfNode::box3d(5.0, 2.0, 5.0).translate(0.0, -1.0, 0.0));
    live.subtract_sphere(Vec3::new(0.3, 0.0, -0.2), 0.8)
        .expect("crater");
    live.subtract_sphere(Vec3::new(-1.2, 0.0, 1.0), 0.5)
        .expect("crater");
    live
}

fn bucket(p: Vec3) -> [i64; 3] {
    let q = p / 0.01;
    [q.x.floor() as i64, q.y.floor() as i64, q.z.floor() as i64]
}

/// Number of vertices of `a` with no vertex of `b` within `TOL`.
fn unmatched(a: &Mesh, b: &Mesh) -> usize {
    let mut grid: HashMap<[i64; 3], Vec<Vec3>> = HashMap::new();
    for v in &b.vertices {
        grid.entry(bucket(v.position)).or_default().push(v.position);
    }
    a.vertices
        .iter()
        .filter(|v| {
            let k = bucket(v.position);
            let mut found = false;
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if let Some(list) = grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]) {
                            found |= list.iter().any(|p| (*p - v.position).length() <= TOL);
                        }
                    }
                }
            }
            !found
        })
        .count()
}

#[test]
fn gpu_mesh_of_the_node_form_matches_the_cpu_chunks() {
    let live = shape();
    let cpu = LiveMesh::new(&live, layout()).expect("cpu mesh").merged();
    let (lo, hi) = layout().domain();
    let config = GpuMarchingCubesConfig {
        resolution: 64,
        compute_normals: false,
        ..GpuMarchingCubesConfig::default()
    };
    let gpu = match live.gpu_mesh(lo, hi, &config) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                std::env::var_os("ALICE_SDF_REQUIRE_GPU").is_none(),
                "ALICE_SDF_REQUIRE_GPU is set but the GPU mesher failed: {e}"
            );
            eprintln!("no GPU adapter: skipped ({e})");
            return;
        }
    };
    assert!(cpu.vertices.len() > 1000, "cpu {}", cpu.vertices.len());
    assert!(gpu.vertices.len() > 1000, "gpu {}", gpu.vertices.len());
    let gpu_only = unmatched(&gpu, &cpu);
    let cpu_only = unmatched(&cpu, &gpu);
    eprintln!(
        "cpu {} vertices, gpu {} vertices, unmatched gpu {gpu_only} cpu {cpu_only}",
        cpu.vertices.len(),
        gpu.vertices.len()
    );
    assert_eq!(gpu_only, 0, "GPU vertices with no CPU vertex within {TOL}");
    assert_eq!(cpu_only, 0, "CPU vertices with no GPU vertex within {TOL}");
}

#[test]
fn gpu_mesh_refuses_a_shape_with_modifiers() {
    let live = shape();
    live.add_modifier(ErosionModifier::new(
        ErosionConfig::default(),
        4,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
    ));
    let (lo, hi) = layout().domain();
    let r = live.gpu_mesh(lo, hi, &GpuMarchingCubesConfig::default());
    assert_eq!(r.err(), Some(LiveSdfError::HasModifiers));
}

//! Oracles for the `destruction` API that `examples/destruction_carve.rs` wires:
//! chunked remeshing, dirty tracking, `set_distance`, `carve_batch`, `explode`.
//!
//! Every expected value is written from a definition, not by calling the
//! function under test to produce it:
//!
//! | target | oracle |
//! |---|---|
//! | `remesh_chunk` over all chunks | the set of cells that carry triangles equals the set of *active* cells (corners of both signs) computed from `grid.distances`, and every cell is meshed by exactly one chunk, so the chunk meshes together equal the one-chunk mesh triangle for triangle |
//! | `set_distance` + dirty tracking | the dirty set is exactly the chunks owning a cell that reads the voxel: cells `i ∈ {v−1, v} ∩ [0, res−2]` on each axis, chunk `i / chunk_size` |
//! | `remesh_all_dirty` | after an edit, untouched chunk meshes plus the remeshed dirty ones equal a full remesh of the edited grid |
//! | `chunks_per_axis` | `ceil(res / chunk_size)` |
//! | `carve_batch` | where every shape's AABB covers the grid, the field is the CSG closed form `max(old, −d₁, −d₂, …)` |
//! | `explode` | documented geometry: the main sphere is emptied, signs change only within `0.8·r + 0.7·r = 1.5·r` of the centre; with no fragments it is a single `carve` |
//!
//! Author: Moroya Sakamoto
#![cfg(feature = "destruction")]

use std::collections::{BTreeMap, BTreeSet};

use alice_sdf::destruction::operations::explode;
use alice_sdf::destruction::{carve, carve_batch, CarveShape, MutableVoxelGrid};
use alice_sdf::mesh::Mesh;
use alice_sdf::types::SdfNode;
use glam::Vec3;

const BOUND: f32 = 2.0;

fn grid(res: u32, chunk: u32) -> MutableVoxelGrid {
    MutableVoxelGrid::from_sdf_with_chunk_size(
        &SdfNode::sphere(1.3),
        [res, res, res],
        Vec3::splat(-BOUND),
        Vec3::splat(BOUND),
        chunk,
    )
}

/// Triangles as bit patterns, so two meshes can be compared as multisets.
fn triangle_bits(mesh: &Mesh) -> BTreeMap<[u32; 9], usize> {
    let mut out = BTreeMap::new();
    for tri in mesh.indices.chunks_exact(3) {
        let mut key = [0u32; 9];
        for (k, &i) in tri.iter().enumerate() {
            let p = mesh.vertices[i as usize].position;
            key[3 * k] = p.x.to_bits();
            key[3 * k + 1] = p.y.to_bits();
            key[3 * k + 2] = p.z.to_bits();
        }
        *out.entry(key).or_insert(0) += 1;
    }
    out
}

fn all_chunk_meshes(g: &MutableVoxelGrid) -> Vec<Mesh> {
    let [cx, cy, cz] = g.chunks_per_axis();
    let mut meshes = Vec::new();
    for z in 0..cz {
        for y in 0..cy {
            for x in 0..cx {
                meshes.push(g.remesh_chunk(x, y, z));
            }
        }
    }
    meshes
}

fn merged_bits(meshes: &[Mesh]) -> BTreeMap<[u32; 9], usize> {
    let mut all = BTreeMap::new();
    for m in meshes {
        for (k, n) in triangle_bits(m) {
            *all.entry(k).or_insert(0) += n;
        }
    }
    all
}

/// Cells whose 8 corner voxels are not all of one sign (`< 0` = inside, the
/// convention `remesh_chunk` documents), from the raw field.
fn active_cells(g: &MutableVoxelGrid) -> BTreeSet<[u32; 3]> {
    let [rx, ry, rz] = g.resolution;
    let mut cells = BTreeSet::new();
    for z in 0..rz - 1 {
        for y in 0..ry - 1 {
            for x in 0..rx - 1 {
                let mut inside = 0;
                for c in 0..8u32 {
                    let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
                    let i = ((x + dx) + (y + dy) * rx + (z + dz) * rx * ry) as usize;
                    if g.distances[i] < 0.0 {
                        inside += 1;
                    }
                }
                if inside != 0 && inside != 8 {
                    cells.insert([x, y, z]);
                }
            }
        }
    }
    cells
}

/// The cell a triangle belongs to: its centroid lies inside the cell box whose
/// low corner is the voxel centre `grid_to_world(cell)` (MC on voxel centres).
fn cell_of_triangle(g: &MutableVoxelGrid, mesh: &Mesh, tri: &[u32]) -> [u32; 3] {
    let c = tri
        .iter()
        .map(|&i| mesh.vertices[i as usize].position)
        .sum::<Vec3>()
        / 3.0;
    let vs = (g.bounds_max - g.bounds_min)
        / Vec3::new(
            g.resolution[0] as f32,
            g.resolution[1] as f32,
            g.resolution[2] as f32,
        );
    let f = (c - g.bounds_min - vs * 0.5) / vs;
    [f.x.floor() as u32, f.y.floor() as u32, f.z.floor() as u32]
}

#[test]
fn chunk_meshes_cover_exactly_the_active_cells_of_the_whole_grid() {
    // 24 voxels in chunks of 5: seams at 5, 10, 15, 20 and a ragged last chunk
    let g = grid(24, 5);
    let want = active_cells(&g);
    assert!(
        want.len() > 100,
        "scene must cross many seams, got {}",
        want.len()
    );

    let mut got: BTreeSet<[u32; 3]> = BTreeSet::new();
    let mut owner: BTreeMap<[u32; 3], usize> = BTreeMap::new();
    for (k, m) in all_chunk_meshes(&g).iter().enumerate() {
        for tri in m.indices.chunks_exact(3) {
            let cell = cell_of_triangle(&g, m, tri);
            got.insert(cell);
            let prev = owner.insert(cell, k);
            assert!(
                prev.is_none() || prev == Some(k),
                "cell {cell:?} meshed by chunks {prev:?} and {k}"
            );
        }
    }
    let missing: Vec<_> = want.difference(&got).collect();
    let extra: Vec<_> = got.difference(&want).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{} active cells unmeshed (first {:?}), {} meshed cells not active (first {:?})",
        missing.len(),
        missing.first(),
        extra.len(),
        extra.first()
    );
}

#[test]
fn chunked_remesh_equals_the_single_chunk_remesh_triangle_for_triangle() {
    for chunk in [3u32, 4, 7, 8] {
        let chunked = merged_bits(&all_chunk_meshes(&grid(16, chunk)));
        let whole = grid(16, 16);
        assert_eq!(whole.chunks_per_axis(), [1, 1, 1]);
        let single = triangle_bits(&whole.remesh_chunk(0, 0, 0));
        assert!(!single.is_empty());
        assert_eq!(
            chunked.values().sum::<usize>(),
            single.values().sum::<usize>(),
            "triangle count, chunk size {chunk}"
        );
        assert_eq!(chunked, single, "triangle multiset, chunk size {chunk}");
    }
}

#[test]
fn chunk_geometry_matches_its_closed_form() {
    for (res, cs) in [(16u32, 16u32), (16, 5), (17, 4), (1, 16), (33, 8)] {
        let g = grid(res, cs);
        assert_eq!(g.chunk_size(), cs);
        let c = res.div_ceil(cs);
        assert_eq!(g.chunks_per_axis(), [c, c, c], "res {res} chunk {cs}");
    }
}

/// Chunks whose cells read voxel `v` on one axis: cells `i ∈ {v−1, v}`, valid
/// when `0 ≤ i ≤ res−2`, owned by chunk `i / cs`.
fn reading_chunks(v: u32, res: u32, cs: u32) -> BTreeSet<u32> {
    let mut s = BTreeSet::new();
    for i in [v.wrapping_sub(1), v] {
        if res >= 2 && i <= res - 2 {
            s.insert(i / cs);
        }
    }
    s
}

#[test]
fn set_distance_dirties_exactly_the_chunks_whose_cells_read_the_voxel() {
    let (res, cs) = (20u32, 5u32);
    let probes = [
        [0, 0, 0],
        [4, 7, 12],
        [5, 7, 12],   // x on a seam
        [5, 10, 15],  // all three on seams
        [19, 19, 19], // last voxel: no cell starts here
        [10, 0, 19],
    ];
    for v in probes {
        let mut g = grid(res, cs);
        assert!(g.dirty_chunks().is_empty(), "from_sdf leaves nothing dirty");
        let before = g.get_distance(v[0], v[1], v[2]);
        g.set_distance(v[0], v[1], v[2], 7.5);
        assert_eq!(g.get_distance(v[0], v[1], v[2]), 7.5);
        assert_ne!(before, 7.5);

        let mut want = BTreeSet::new();
        for cz in reading_chunks(v[2], res, cs) {
            for cy in reading_chunks(v[1], res, cs) {
                for cx in reading_chunks(v[0], res, cs) {
                    want.insert([cx, cy, cz]);
                }
            }
        }
        // the voxel's own chunk is always marked, even when no cell reads it
        // (marking is conservative there, never missing)
        want.insert([v[0] / cs, v[1] / cs, v[2] / cs]);
        let got: BTreeSet<[u32; 3]> = g.dirty_chunks().into_iter().collect();
        assert_eq!(got, want, "voxel {v:?}");
        for c in &want {
            assert!(g.is_chunk_dirty(c[0], c[1], c[2]));
        }
        // out of range chunk coordinates are never dirty
        assert!(!g.is_chunk_dirty(99, 0, 0));
        g.clear_dirty();
        assert!(g.dirty_chunks().is_empty());
        assert!(!g.is_chunk_dirty(v[0] / cs, v[1] / cs, v[2] / cs));
    }
}

#[test]
fn remeshing_only_dirty_chunks_reproduces_a_full_remesh_after_a_seam_edit() {
    let (res, cs) = (20u32, 5u32);
    let mut g = grid(res, cs);
    let before: Vec<Mesh> = all_chunk_meshes(&g);
    let [cpx, cpy, _] = g.chunks_per_axis();

    // dig a column of voxels lying on the x = 10 seam, through the surface
    for y in 6..14 {
        g.set_distance(10, y, 10, 0.4);
    }
    let dirty = g.remesh_all_dirty();
    assert!(
        g.dirty_chunks().is_empty(),
        "remesh_all_dirty clears the flags"
    );
    assert!(
        dirty.iter().any(|c| c.chunk[0] == 1),
        "the chunk below the seam"
    );
    assert!(
        dirty.iter().any(|c| c.chunk[0] == 2),
        "the chunk above the seam"
    );

    let mut patched = before;
    for cm in &dirty {
        let [x, y, z] = cm.chunk;
        patched[(x + y * cpx + z * cpx * cpy) as usize] = Mesh {
            vertices: cm.mesh.vertices.clone(),
            indices: cm.mesh.indices.clone(),
        };
    }
    let full = all_chunk_meshes(&g);
    assert_eq!(merged_bits(&patched), merged_bits(&full));
}

#[test]
fn materials_start_at_zero_and_degenerate_grids_mesh_to_nothing() {
    let g = grid(6, 4);
    for z in 0..6 {
        for y in 0..6 {
            for x in 0..6 {
                assert_eq!(g.get_material(x, y, z), 0);
            }
        }
    }
    // a 1-voxel grid has no MC cell; a 2-voxel grid has exactly one
    let one = grid(1, 16);
    let m = one.remesh_chunk(0, 0, 0);
    assert!(m.vertices.is_empty() && m.indices.is_empty());
    let mut two = MutableVoxelGrid::from_sdf_with_chunk_size(
        &SdfNode::sphere(1.0),
        [2, 2, 2],
        Vec3::splat(-1.0),
        Vec3::splat(1.0),
        1,
    );
    // corners at (±0.5)³, |p| = 0.866 < 1: all inside, no surface
    assert!(two.remesh_chunk(0, 0, 0).indices.is_empty());
    two.set_distance(1, 1, 1, 0.5);
    let tri = two.remesh_chunk(0, 0, 0);
    assert_eq!(tri.indices.len(), 3, "one outside corner cuts one triangle");
    // chunk (1, ·, ·) owns no cell (its low voxel is the grid's last)
    assert!(two.remesh_chunk(1, 1, 1).indices.is_empty());
}

#[test]
fn carve_batch_is_the_csg_closed_form_where_every_aabb_covers_the_grid() {
    // grid [-1, 1]³; both spheres' AABBs contain it, so every voxel is visited
    let shapes = [
        CarveShape::Sphere {
            center: Vec3::new(-0.5, 0.0, 0.0),
            radius: 1.6,
        },
        CarveShape::Sphere {
            center: Vec3::new(0.5, 0.3, 0.0),
            radius: 1.6,
        },
    ];
    let make = || {
        MutableVoxelGrid::from_sdf_with_chunk_size(
            &SdfNode::sphere(2.5),
            [12, 12, 12],
            Vec3::splat(-1.0),
            Vec3::splat(1.0),
            4,
        )
    };
    let original = make();
    let mut g = make();
    let r = carve_batch(&mut g, &shapes);
    let mut changed = 0u32;
    for z in 0..12 {
        for y in 0..12 {
            for x in 0..12 {
                let p = g.grid_to_world(x, y, z);
                let old = original.get_distance(x, y, z);
                let d1 = (p - Vec3::new(-0.5, 0.0, 0.0)).length() - 1.6;
                let d2 = (p - Vec3::new(0.5, 0.3, 0.0)).length() - 1.6;
                let want = old.max(-d1).max(-d2);
                assert_eq!(g.get_distance(x, y, z), want, "voxel ({x},{y},{z})");
                if want != old {
                    changed += 1;
                }
            }
        }
    }
    assert!(changed > 0);
    assert!(r.modified_voxels >= changed);
    // every chunk was touched, and the batch reports exactly the dirty set
    assert_eq!(r.dirty_chunks.len(), 27);
    assert_eq!(r.dirty_chunks, g.dirty_chunks());
}

#[test]
fn explode_empties_its_main_sphere_and_changes_signs_only_within_one_and_a_half_radii() {
    let make = || {
        MutableVoxelGrid::from_sdf_with_chunk_size(
            &SdfNode::box3d(3.6, 3.6, 3.6),
            [32, 32, 32],
            Vec3::splat(-BOUND),
            Vec3::splat(BOUND),
            8,
        )
    };
    let original = make();
    let center = Vec3::new(0.2, -0.1, 0.3);
    let radius = 0.5;
    for seed in [1u64, 7, 42, 0xDEAD_BEEF] {
        let mut g = make();
        let r = explode(&mut g, center, radius, 6, seed);
        assert!(r.modified_voxels > 0);
        let mut flipped = 0;
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let p = g.grid_to_world(x, y, z);
                    let dist = (p - center).length();
                    let (old, new) = (original.get_distance(x, y, z), g.get_distance(x, y, z));
                    if dist < radius {
                        // max(old, −(dist − r)) ≥ r − dist > 0
                        assert!(
                            new >= radius - dist,
                            "seed {seed}: ({x},{y},{z}) not emptied"
                        );
                    }
                    if (old < 0.0) != (new < 0.0) {
                        flipped += 1;
                        assert!(
                            dist <= 1.5 * radius + 1e-5,
                            "seed {seed}: sign changed {dist} from the centre (> 1.5 r)"
                        );
                    }
                }
            }
        }
        assert!(flipped > 0);
    }

    // no fragments: exactly one carve of the main sphere
    let mut a = make();
    let mut b = make();
    let ra = explode(&mut a, center, radius, 0, 99);
    let rb = carve(&mut b, &CarveShape::Sphere { center, radius });
    assert_eq!(a.distances, b.distances);
    assert_eq!(ra.modified_voxels, rb.modified_voxels);
    assert_eq!(ra.removed_volume, rb.removed_volume);
}

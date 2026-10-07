//! Real-time destruction on a voxel grid: carve, batch-carve, explode, edit
//! voxels by hand, remesh only the dirty chunks, and spawn debris.
//!
//! Run: `cargo run --example destruction_carve --features destruction`
//!
//! Each step is checked against its definition (pinned in
//! `tests/test_destruction_api_oracle.rs` and
//! `tests/test_terrain_destruction_oracle.rs`).
//!
//! Author: Moroya Sakamoto

use alice_sdf::destruction::operations::explode;
use alice_sdf::destruction::{
    carve, carve_batch, generate_debris, CarveShape, ChunkMesh, DebrisConfig, DebrisPiece,
    MutableVoxelGrid,
};
use alice_sdf::types::SdfNode;
use glam::{Quat, Vec3};

fn main() {
    let mut grid = MutableVoxelGrid::from_sdf_with_chunk_size(
        &SdfNode::box3d(3.0, 3.0, 3.0),
        [32, 32, 32],
        Vec3::splat(-2.0),
        Vec3::splat(2.0),
        8,
    );
    println!(
        "grid {:?}: {} voxels, chunks {:?} of {}, material at the centre {}",
        grid.resolution,
        grid.voxel_count(),
        grid.chunks_per_axis(),
        grid.chunk_size(),
        grid.get_material(16, 16, 16)
    );
    assert_eq!(grid.chunks_per_axis(), [4, 4, 4]);

    // a full remesh: one mesh per chunk, together the whole surface
    let mut full = 0;
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                full += grid.remesh_chunk(x, y, z).indices.len() / 3;
            }
        }
    }
    println!("  initial surface: {full} triangles");

    // single carve, then a batch of a sphere and a rotated box
    let hole = carve(
        &mut grid,
        &CarveShape::Sphere {
            center: Vec3::new(1.4, 0.0, 0.0),
            radius: 0.6,
        },
    );
    println!(
        "carve: {} voxels changed, {:.3} removed, {} dirty chunks",
        hole.modified_voxels,
        hole.removed_volume,
        hole.dirty_chunks.len()
    );
    let batch = carve_batch(
        &mut grid,
        &[
            CarveShape::Sphere {
                center: Vec3::new(-1.4, 1.4, 0.0),
                radius: 0.5,
            },
            CarveShape::Box {
                center: Vec3::new(0.0, -1.4, 0.5),
                half_extents: Vec3::new(0.4, 0.3, 0.6),
                rotation: Quat::from_rotation_y(0.5),
            },
        ],
    );
    println!(
        "batch: {} voxels changed, {:.3} removed",
        batch.modified_voxels, batch.removed_volume
    );
    assert!(hole.removed_volume > 0.0 && batch.removed_volume > 0.0);

    // a crater: the main sphere is emptied, fragments stay within 1.5 r
    let center = Vec3::new(0.0, 1.45, -0.8);
    let boom = explode(&mut grid, center, 0.5, 6, 2026);
    let inside = grid.world_to_grid(center).unwrap();
    println!(
        "explode: {} voxels changed, {:.3} removed",
        boom.modified_voxels, boom.removed_volume
    );
    assert!(grid.get_distance(inside[0], inside[1], inside[2]) > 0.0);
    assert!(boom.removed_volume < 4.0 / 3.0 * std::f32::consts::PI * 0.75f32.powi(3) * 1.2);

    // hand edit on a chunk seam: both chunks that read the voxel get dirty
    grid.clear_dirty();
    grid.set_distance(8, 3, 3, -0.2);
    let dirty = grid.dirty_chunks();
    println!("seam edit at x = 8: dirty chunks {dirty:?}");
    assert!(grid.is_chunk_dirty(0, 0, 0) && grid.is_chunk_dirty(1, 0, 0));
    let remeshed: Vec<ChunkMesh> = grid.remesh_all_dirty();
    assert_eq!(remeshed.len(), dirty.len());
    assert!(grid.dirty_chunks().is_empty());

    // debris for the crater
    let pieces: Vec<DebrisPiece> = generate_debris(
        center,
        0.5,
        &DebrisConfig {
            max_pieces: 6,
            ..Default::default()
        },
    );
    let total: f32 = pieces.iter().map(|p| p.volume).sum();
    println!("debris: {} pieces, {:.4} total volume", pieces.len(), total);
    for p in &pieces {
        assert!((p.center - center).length() <= 0.5 + 1e-5);
        assert_eq!(p.mesh.indices.len(), 24, "octahedron");
    }
}
